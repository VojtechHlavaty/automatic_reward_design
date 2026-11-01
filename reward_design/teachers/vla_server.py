"""Serve a LeRobot VLA policy (e.g. pi0.5 for the G1 with Dex1 grippers) to the IsaacLab process over a local socket.

Run with the LeRobot venv: ~/envs/lerobot/bin/python vla_server.py --policy <dir> --port 5591
Request (pickled dict): state float32 [N, 16], head / left_wrist / right_wrist uint8 [N, 480, 640, 3], task str.
Reply (pickled): action chunk float32 [N, chunk, 16] (raw joint targets and gripper values).

pi0.5's text tokenizer is PaliGemma's SentencePiece model; it is read from paligemma_tokenizer/tokenizer.model (Google's
public big_vision copy, gs://big_vision/paligemma_tokenizer.model, the one OpenPI uses) instead of the gated Hugging Face
repository that LeRobot would otherwise download.
"""

import argparse
import os
import pickle
import socket
import struct
import time

import numpy as np
import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


class SentencePieceTokenizer:
    """The part of a Hugging Face tokenizer that LeRobot's TokenizerProcessorStep calls (BOS first, no EOS)."""

    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.bos, self.pad = self.sp.bos_id(), max(self.sp.pad_id(), 0)

    def __call__(self, text, max_length, truncation=True, padding="max_length", padding_side="right",
                 return_tensors="pt"):
        texts = [text] if isinstance(text, str) else list(text)
        ids = torch.full((len(texts), max_length), self.pad, dtype=torch.long)
        mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for i, t in enumerate(texts):
            tokens = [self.bos] + self.sp.encode(t)
            tokens = tokens[:max_length] if truncation else tokens
            if padding_side == "left":
                ids[i, max_length - len(tokens):] = torch.tensor(tokens)
                mask[i, max_length - len(tokens):] = 1
            else:
                ids[i, :len(tokens)] = torch.tensor(tokens)
                mask[i, :len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def load(policy_dir, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(policy_dir)
    policy = get_policy_class(config.type).from_pretrained(policy_dir).to(device).eval()
    overrides = {"device_processor": {"device": device}}
    if config.type in ("pi0", "pi05"):
        overrides["tokenizer_processor"] = {
            "tokenizer": SentencePieceTokenizer(os.path.join(HERE, "paligemma_tokenizer", "tokenizer.model"))}
    pre, post = make_pre_post_processors(policy.config, pretrained_path=policy_dir, preprocessor_overrides=overrides)
    return policy, pre, post


def recv(conn):
    size = struct.unpack("!Q", conn.recv(8, socket.MSG_WAITALL))[0]
    return pickle.loads(conn.recv(size, socket.MSG_WAITALL))


def send(conn, obj):
    data = pickle.dumps(obj, protocol=5)
    conn.sendall(struct.pack("!Q", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=os.path.join(HERE, "pi05_unifolm"))
    parser.add_argument("--port", type=int, default=5591)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    policy, pre, post = load(args.policy, args.device)
    keys = list(policy.config.image_features)
    print("image inputs:", keys, "chunk", policy.config.chunk_size, flush=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    print(f"ready on port {args.port}", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    req = recv(conn)
                except (struct.error, ConnectionError, EOFError):
                    break
                t = time.time()
                n = len(req["state"])
                batch = {"observation.state": torch.as_tensor(req["state"]), "task": [req["task"]] * n}
                for key, name in zip(keys, ("head", "left_wrist", "right_wrist")):
                    batch[key] = torch.as_tensor(req[name]).permute(0, 3, 1, 2).float() / 255.0
                with torch.no_grad():
                    actions = post(policy.predict_action_chunk(pre(batch)))
                send(conn, actions.float().cpu().numpy())
                print(f"{n} envs, {time.time() - t:.2f} s", flush=True)


if __name__ == "__main__":
    main()
