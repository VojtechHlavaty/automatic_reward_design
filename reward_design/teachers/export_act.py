"""Export a LeRobot ACT checkpoint as one TorchScript module with its normalization built in, so that the IsaacLab
environment (which has no LeRobot) can run it: inputs the raw 16-dim state [N, 16] and three uint8 images
[N, 480, 640, 3] (head, left wrist, right wrist), output the raw action chunk [N, chunk, 16].

Run with the LeRobot venv: ~/envs/lerobot/bin/python export_act.py --repo RooibosT/Sim_act_dex1_bt16_s50k --out act.pt
"""

import argparse

import torch
from huggingface_hub import snapshot_download
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="RooibosT/Sim_act_dex1_bt16_s50k")
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

path = snapshot_download(args.repo)
policy = ACTPolicy.from_pretrained(path).to(args.device).eval()
pre = load_file(f"{path}/policy_preprocessor_step_3_normalizer_processor.safetensors")
post = load_file(f"{path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
print("normalizer keys:", sorted(pre), sorted(post))
image_keys = list(policy.config.image_features)
print("image order:", image_keys, "chunk", policy.config.chunk_size)


class Exported(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = policy.model
        self.register_buffer("s_mean", pre["observation.state.mean"])
        self.register_buffer("s_std", pre["observation.state.std"])
        self.register_buffer("a_mean", post["action.mean"])
        self.register_buffer("a_std", post["action.std"])
        for i, k in enumerate(image_keys):
            self.register_buffer(f"i{i}_mean", pre[f"{k}.mean"].view(1, 3, 1, 1))
            self.register_buffer(f"i{i}_std", pre[f"{k}.std"].view(1, 3, 1, 1))

    def forward(self, state, head, left_wrist, right_wrist):
        images = []
        for i, img in enumerate((head, left_wrist, right_wrist)):
            x = img.permute(0, 3, 1, 2).float() / 255.0
            images.append((x - getattr(self, f"i{i}_mean")) / (getattr(self, f"i{i}_std") + 1e-8))
        batch = {"observation.state": (state - self.s_mean) / (self.s_std + 1e-8), "observation.images": images}
        actions = self.model(batch)[0]
        return actions * self.a_std + self.a_mean


assert image_keys == ["observation.images.cam_left_high", "observation.images.cam_left_wrist",
                      "observation.images.cam_right_wrist"], image_keys
module = Exported().to(args.device).eval()
n = 2
example = (torch.zeros(n, 16, device=args.device), *(torch.zeros(n, 480, 640, 3, dtype=torch.uint8, device=args.device)
                                                     for _ in range(3)))
with torch.no_grad():
    traced = torch.jit.trace(module, example, check_trace=False)
    ref = module(*example)
    out = traced(*example)
print("trace error", (ref - out).abs().max().item(), "output", tuple(out.shape))
traced.save(args.out)
