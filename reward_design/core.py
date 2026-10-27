from omegaconf import OmegaConf
import numpy as np
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
import shutil
import time

from utils.misc import TRAINING_DONE_MARKER, filter_traceback, get_gpu_ids
from utils.file_utils import load_tensorboard_logs
from utils.extract_task_code import file_to_string

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = f"{ROOT_DIR}/isaaclab_backend/train.py"
DUMMY_FAILURE = -10000.


class LLMClient:
    """Samples chat completions from an OpenAI-compatible API.

    With ``LLMClient.usage_file`` set, every request appends a JSON line with the client's role and its token usage
    (prompt, cached prompt, completion tokens and the time taken), so token use can be measured per role."""

    usage_file = None
    _usage_lock = None
    # cache warm-up: requests that share a cached prefix and start together (e.g. the parallel stage writers of a wide
    # round) would all miss the cache; the first request with a new prefix goes ahead and the others wait for it
    _warm_lock = None
    _warm = {}  # prefix hash -> (start time, threading.Event set when the first request is done)
    WARM_SECONDS = 240  # a prefix older than this may have left the 5-minute cache

    def __init__(self, cfg):
        from openai import OpenAI

        # the API key: from the environment variable named by llm_api_key_env (default OPENAI_API_KEY), or from the
        # file llm_api_key_file (one line, readable only by the user); never from the config itself
        key_env = cfg.get("llm_api_key_env") or "OPENAI_API_KEY"
        api_key = os.getenv(key_env)
        key_file = cfg.get("llm_api_key_file")
        if api_key is None and key_file and os.path.isfile(os.path.expanduser(key_file)):
            with open(os.path.expanduser(key_file)) as f:
                api_key = f.read().strip() or None
        if api_key is None and cfg.llm_base_url is not None and cfg.get("llm_api_key_env") is None:
            api_key = "EMPTY"  # local OpenAI-compatible servers (e.g. vLLM) do not check the key
        if api_key is None:
            raise RuntimeError(f"Set {key_env}" + (f" or write the key to {key_file}" if key_file else "")
                               + f" to use {cfg.model} (or select a local model, e.g. llm=qwen3.8_27b_fp8).")
        self.client = OpenAI(api_key=api_key, base_url=cfg.llm_base_url, timeout=cfg.llm_timeout)
        self.role = "main"
        # prompt caching for APIs that need explicit cache breakpoints (Anthropic models, also through OpenRouter):
        # prompts mark the end of each reusable prefix with CACHE_MARK; without caching the marks are removed
        self.prompt_cache = bool(cfg.get("prompt_cache"))
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.chunk_size = cfg.chunk_size
        self.supports_n = True
        self.extra_kwargs = {}
        if cfg.top_p is not None:
            self.extra_kwargs["top_p"] = cfg.top_p
        if cfg.max_tokens is not None:
            self.extra_kwargs["max_tokens"] = cfg.max_tokens
        if cfg.get("reasoning_effort") is not None:
            self.extra_kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.llm_extra_body:
            self.extra_kwargs["extra_body"] = OmegaConf.to_container(cfg.llm_extra_body, resolve=True)

    CACHE_MARK = "<<<RD_CACHE>>>"

    def _prepare(self, messages):
        """Messages with the cache marks turned into cache breakpoints (prompt_cache) or removed."""
        prepared = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or self.CACHE_MARK not in content:
                prepared.append(message)
                continue
            parts = [part.strip("\n") for part in content.split(self.CACHE_MARK)]
            if not self.prompt_cache:
                prepared.append(dict(message, content="\n\n".join(p for p in parts if p)))
                continue
            blocks = []
            for i, part in enumerate(parts):
                if not part:
                    continue
                block = {"type": "text", "text": part}
                if i < len(parts) - 1:
                    block["cache_control"] = {"type": "ephemeral"}  # everything up to here is a reusable prefix
                blocks.append(block)
            prepared.append(dict(message, content=blocks))
        return prepared

    def _first_prefix(self, prepared):
        """Hash of the prompt up to its first cache breakpoint (the part shared by parallel requests), or None."""
        import hashlib

        parts = []
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                parts.append(str(content))
                continue
            for block in content:
                parts.append(block.get("text", ""))
                if block.get("cache_control"):
                    return hashlib.sha1("\x00".join([self.model] + parts).encode()).hexdigest()
        return None

    def _wait_for_warm_cache(self, prepared):
        """The event to set after the request when this request writes a new cached prefix; otherwise waits (at most
        3 minutes) until the request that writes this prefix is done, so that this one reads it from the cache."""
        import threading

        key = self._first_prefix(prepared) if self.prompt_cache else None
        if key is None:
            return None
        if LLMClient._warm_lock is None:
            LLMClient._warm_lock = threading.Lock()
        with LLMClient._warm_lock:
            entry = LLMClient._warm.get(key)
            if entry is None or time.time() - entry[0] > self.WARM_SECONDS:
                event = threading.Event()
                LLMClient._warm[key] = (time.time(), event)
                return event
        entry[1].wait(timeout=180)
        return None

    def _log_usage(self, response, seconds):
        if LLMClient.usage_file is None or getattr(response, "usage", None) is None:
            return
        import json
        import threading

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        line = json.dumps({"time": time.time(), "role": self.role, "prompt_tokens": usage.prompt_tokens,
                           "cached_prompt_tokens": cached, "completion_tokens": usage.completion_tokens,
                           "seconds": round(seconds, 2)})
        if LLMClient._usage_lock is None:
            LLMClient._usage_lock = threading.Lock()
        with LLMClient._usage_lock, open(LLMClient.usage_file, "a") as f:
            f.write(line + "\n")

    def with_options(self, thinking=None, max_tokens=None, role=None):
        """A copy of the client that shares the connection, with reasoning switched on or off and a token cap.

        ``thinking=False`` turns the model's thinking mode off (chat template ``enable_thinking`` for local
        thinking models such as Qwen3, ``reasoning_effort`` "low" for API reasoning models); ``max_tokens`` caps the
        answer (including any reasoning).
        """
        import copy

        clone = copy.copy(self)
        clone.extra_kwargs = copy.deepcopy(self.extra_kwargs)
        if thinking is not None:
            template = clone.extra_kwargs.get("extra_body", {}).get("chat_template_kwargs")
            if template is not None and "enable_thinking" in template:
                template["enable_thinking"] = bool(thinking)
            if "reasoning_effort" in clone.extra_kwargs and not thinking:
                clone.extra_kwargs["reasoning_effort"] = "low"
        if max_tokens is not None:
            clone.extra_kwargs["max_tokens"] = int(max_tokens)
        if role is not None:
            clone.role = role
        return clone

    def sample(self, messages, num_samples):
        """Return (list of response strings, prompt tokens, completion tokens, total tokens)."""
        responses = []
        prompt_tokens = completion_tokens = total_tokens = 0
        chunk_size = self.chunk_size
        while len(responses) < num_samples:
            n = min(chunk_size, num_samples - len(responses)) if self.supports_n else 1
            kwargs = {"model": self.model, "messages": self._prepare(messages), **self.extra_kwargs}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if n > 1:
                kwargs["n"] = n
            warm = self._wait_for_warm_cache(kwargs["messages"])
            try:
                for attempt in range(1000):
                    try:
                        started = time.time()
                        response = self.client.chat.completions.create(**kwargs)
                        self._log_usage(response, time.time() - started)
                        break
                    except Exception as e:
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        if "'n'" in str(e) or "parameter n" in str(e).lower():
                            self.supports_n = False
                            kwargs.pop("n", None)
                        elif attempt >= 10:
                            chunk_size = max(chunk_size // 2, 1)
                            kwargs["n"] = min(chunk_size, n)
                            if kwargs["n"] == 1:
                                kwargs.pop("n")
                            logging.info(f"Current chunk size: {chunk_size}")
                        if attempt >= 50:
                            raise RuntimeError("Too many failed LLM requests!") from e
                        time.sleep(min(2 ** attempt, 60))
            finally:
                if warm is not None:
                    warm.set()  # the prefix is in the cache now: waiting requests can go
            for choice in response.choices:
                if choice.finish_reason == "length":
                    logging.info("An LLM response was cut off at the token limit")
                responses.append(strip_thinking(choice.message.content or ""))
            if response.usage is not None:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                total_tokens += response.usage.total_tokens
        return responses[:num_samples], prompt_tokens, completion_tokens, total_tokens

    def complete_once(self, messages):
        """One request without retries; raises the API error (used to probe model capabilities)."""
        kwargs = {"model": self.model, "messages": self._prepare(messages), **self.extra_kwargs}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        started = time.time()
        response = self.client.chat.completions.create(**kwargs)
        self._log_usage(response, time.time() - started)
        return strip_thinking(response.choices[0].message.content or "")


def make_vlm_client(cfg, llm):
    """Client for the vision-language review. It is the reward LLM unless vlm_model or vlm_base_url is set."""
    if cfg.vlm_model is None and cfg.vlm_base_url is None:
        return llm
    overrides = {}
    if cfg.vlm_model is not None:
        overrides["model"] = cfg.vlm_model
    if cfg.vlm_base_url is not None:
        overrides["llm_base_url"] = cfg.vlm_base_url
    return LLMClient(OmegaConf.merge(cfg, overrides))


def strip_thinking(response):
    """Remove the reasoning block of thinking models (e.g. Qwen3) from a response."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in response:  # reasoning cut off before it ended
        response = response.split("<think>", 1)[0]
    if "</think>" in response:  # opening tag added by the chat template, not the model
        response = response.split("</think>", 1)[1]
    return response.strip()




class TrainingLauncher:
    """Runs IsaacLab training processes on a pool of GPU slots."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.python = cfg.isaaclab_python or sys.executable
        gpus = list(cfg.gpus) if cfg.gpus is not None else get_gpu_ids()
        self.slots = [gpu for _ in range(cfg.max_runs_per_gpu) for gpu in gpus]
        self.running = []  # list of (process, gpu, log file handle, start time)

    def launch(self, reward_file, log_file, run_name, max_iterations, seed=None, extra_args=None):
        cmd = [
            self.python, '-u', TRAIN_SCRIPT,
            '--task', self.cfg.env.task,
            '--reward_file', reward_file,
            '--isaaclab_root', self.cfg.isaaclab_root,
            '--run_name', run_name,
        ]
        if max_iterations is not None:
            cmd += ['--max_iterations', str(max_iterations)]
        if self.cfg.num_envs is not None:
            cmd += ['--num_envs', str(self.cfg.num_envs)]
        if seed is not None:
            cmd += ['--seed', str(seed)]
        if self.cfg.capture_video:
            cmd += ['--video']
        if self.cfg.use_wandb:
            cmd += ['--logger', 'wandb', '--log_project_name', self.cfg.wandb_project]
        cmd += [str(arg) for arg in self.cfg.extra_train_args]
        cmd += [str(arg) for arg in extra_args or []]
        return self.launch_command(cmd, log_file, run_name)

    def launch_command(self, cmd, log_file, run_name):
        """Run a command on the least busy GPU slot, waiting for a free slot first."""
        while len(self.running) >= len(self.slots):
            self._wait_any()
        busy = [gpu for _, gpu, _, _ in self.running]
        gpu = min(dict.fromkeys(self.slots), key=lambda g: busy.count(g) / self.slots.count(g))

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        f = open(log_file, 'w')
        # A separate session keeps Ctrl-C away from the training runs; terminate_all() stops them instead.
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.running.append((process, gpu, f, time.time()))
        logging.info(f"Launched {run_name} on GPU {gpu} (log: {log_file})")
        return process

    def _poll(self):
        still_running = []
        for process, gpu, f, start in self.running:
            timeout = self.cfg.train_timeout
            if process.poll() is None and timeout is not None and time.time() - start > timeout:
                logging.info(f"Training process {process.pid} exceeded {timeout} s, killing it")
                process.kill()
                process.wait()
                f.write(f"\nTraceback (most recent call last):\nTimeoutError: training exceeded {timeout} seconds\n")
            if process.poll() is None:
                still_running.append((process, gpu, f, start))
            else:
                f.close()
        self.running = still_running

    def _wait_any(self):
        num_running = len(self.running)
        while len(self.running) == num_running:
            self._poll()
            if len(self.running) == num_running:
                time.sleep(5)

    def wait_all(self):
        while self.running:
            self._poll()
            if self.running:
                time.sleep(5)

    def terminate_all(self):
        """Stop all running training processes."""
        if self.running:
            logging.info(f"Stopping {len(self.running)} running training processes")
        terminate_processes([process.pid for process, _, _, _ in self.running])
        for process, _, f, _ in self.running:
            process.wait()
            f.close()
        self.running = []


def terminate_processes(pids, timeout=30):
    """Terminate processes and their children, killing the ones that do not exit in time."""
    import psutil

    procs = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            procs += [proc] + proc.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def terminate_stale_runs(run_dir):
    """Stop training processes of an earlier pipeline process that still use reward files in run_dir."""
    import psutil

    prefix = os.path.join(os.path.abspath(run_dir), "")
    stale = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info["cmdline"] or []
        if TRAIN_SCRIPT in cmdline and "--reward_file" in cmdline:
            reward_file = cmdline[cmdline.index("--reward_file") + 1]
            if reward_file.startswith(prefix):
                stale.append(proc.info["pid"])
    if stale:
        logging.info(f"Stopping {len(stale)} training processes left over from the previous pipeline process")
        terminate_processes(stale)


def training_finished(log_file):
    """True when a training log shows a completed or crashed run, False when it was interrupted or never ran."""
    if not os.path.exists(log_file):
        return False
    stdout_str = file_to_string(log_file)
    if TRAINING_DONE_MARKER in stdout_str:
        return True
    traceback_msg = filter_traceback(stdout_str)
    return bool(traceback_msg) and "KeyboardInterrupt" not in traceback_msg


def reduce_metric(values, reduce):
    values = np.array(values, dtype=float)
    if reduce == "max":
        return float(values.max())
    if reduce == "min":
        return float(values.min())
    # mean over the last 10% of training
    return float(values[-max(len(values) // 10, 1):].mean())


def shorten_traceback(traceback_msg, reward_file):
    """Keep the frames of the reward file, the last frame and the exception of a traceback."""
    lines = traceback_msg.split('\n')
    frames = []  # (start, end) line ranges of "  File ..." frames
    for i, line in enumerate(lines):
        if line.startswith('  File "'):
            frames.append([i, len(lines)])
            if len(frames) > 1:
                frames[-2][1] = i
    if not frames:
        return traceback_msg
    # the last frame ends where the exception message starts
    last_start = frames[-1][0]
    exception_start = next(
        (i for i in range(last_start + 1, len(lines)) if lines[i] and not lines[i].startswith(' ')), len(lines)
    )
    frames[-1][1] = exception_start
    reward_name = os.path.basename(reward_file)
    keep = [f for f in frames if reward_name in lines[f[0]]]
    if frames[-1] not in keep:
        keep.append(frames[-1])
    out = [lines[0]]
    for start, end in keep:
        out += lines[start:end]
    out += lines[exception_start:]
    return '\n'.join(out).replace(os.path.dirname(os.path.abspath(reward_file)) + os.sep, '')


def parse_training_log(log_file, reward_file=None):
    """Return (tensorboard log dir or None, traceback message)."""
    stdout_str = file_to_string(log_file) if os.path.exists(log_file) else ""
    traceback_msg = filter_traceback(stdout_str)
    if traceback_msg and reward_file is not None:
        traceback_msg = shorten_traceback(traceback_msg, reward_file)
    log_dir = None
    for line in stdout_str.split('\n'):
        if line.startswith('Tensorboard Directory:'):
            log_dir = line.split(':', 1)[-1].strip()
    if not traceback_msg and TRAINING_DONE_MARKER not in stdout_str:
        tail = "\n".join(stdout_str.strip().split('\n')[-20:])
        traceback_msg = f"Training did not finish. Last lines of the training log:\n{tail}"
    return log_dir, traceback_msg


def evaluate_run(cfg, log_dir):
    """Return (task score, secondary score, correlation, epoch frequency, feedback content) for a finished training run.

    The secondary score (``cfg.env.secondary_metric``) breaks ties between samples with the same task score.
    """
    tensorboard_logs = load_tensorboard_logs(log_dir)
    success_metric = cfg.env.success_metric
    sign = 1.0 if cfg.env.success_higher_is_better else -1.0

    gpt_rewards = tensorboard_logs.get("Generated/gpt_reward", [])
    num_logged = len(gpt_rewards) if gpt_rewards else len(tensorboard_logs.get("Train/mean_reward", [1]))
    epoch_freq = max(int(num_logged // 10), 1)

    reward_correlation = DUMMY_FAILURE
    if "Generated/gt_reward" in tensorboard_logs and gpt_rewards:
        gt_reward = np.array(tensorboard_logs["Generated/gt_reward"])
        gpt_reward = np.array(gpt_rewards)
        n = min(len(gt_reward), len(gpt_reward))
        if n > 1 and gt_reward[:n].std() > 0 and gpt_reward[:n].std() > 0:
            reward_correlation = float(np.corrcoef(gt_reward[:n], gpt_reward[:n])[0, 1])

    if success_metric in tensorboard_logs:
        score = sign * reduce_metric(tensorboard_logs[success_metric], cfg.env.success_reduce)
    elif "Generated/gt_reward" in tensorboard_logs:
        score = reduce_metric(tensorboard_logs["Generated/gt_reward"], cfg.env.success_reduce)
    else:
        score = DUMMY_FAILURE

    secondary_metric = cfg.env.get("secondary_metric")
    if secondary_metric and secondary_metric in tensorboard_logs:
        secondary = reduce_metric(tensorboard_logs[secondary_metric], cfg.env.success_reduce)
    else:
        secondary = DUMMY_FAILURE

    def summary(name, values):
        values = list(values)
        # 3 significant digits: small per-step rewards (e.g. 0.0026) stay visible to the LLM
        metric_cur = ['{:.3g}'.format(x) for x in values[::epoch_freq]]
        return (
            f"{name}: {metric_cur}, Max: {max(values):.3g}, Mean: {sum(values) / len(values):.3g},"
            f" Min: {min(values):.3g} \n"
        )

    content = ''
    if success_metric in tensorboard_logs:
        content += summary("task_score", tensorboard_logs[success_metric])
    elif "Generated/gt_reward" in tensorboard_logs:
        content += summary("ground-truth score", tensorboard_logs["Generated/gt_reward"])
    if secondary_metric and secondary_metric in tensorboard_logs and success_metric in tensorboard_logs:
        # progress signal that stays informative while the task score is still zero
        name = "ground-truth score" if secondary_metric == "Generated/gt_reward" else secondary_metric
        content += summary(name, tensorboard_logs[secondary_metric])
    if "Train/mean_episode_length" in tensorboard_logs:
        content += summary("episode_length", tensorboard_logs["Train/mean_episode_length"])
    for metric in tensorboard_logs:
        if metric.startswith("Generated/") and metric not in ("Generated/gt_reward", "Generated/gpt_reward"):
            content += summary(metric[len("Generated/"):], tensorboard_logs[metric])
    return score, secondary, reward_correlation, epoch_freq, content


STATE_FILE = "run_state.json"
CONFIG_FILE = "run_config.yaml"
# Config groups and keys that cannot be changed when resuming a run
RESUME_FIXED_KEYS = ("resume", "env", "llm")


def resume_config(cfg):
    """Switch to the run directory named by cfg.resume and return that run's config.

    Command-line overrides (except config groups) are applied on top of the saved config.
    """
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path

    run_dir = Path(to_absolute_path(cfg.resume)).resolve()
    saved_cfg_file = run_dir / CONFIG_FILE
    if not saved_cfg_file.exists():
        # runs started before resume support: Hydra's copy of the config
        saved_cfg_file = run_dir / ".hydra" / "config.yaml"
    if not saved_cfg_file.exists():
        raise FileNotFoundError(f"No run to resume in {run_dir}")
    saved_cfg = OmegaConf.load(saved_cfg_file)

    overrides = []
    for override in HydraConfig.get().overrides.task:
        key = override.split("=", 1)[0].lstrip("+")
        if override.startswith("~") or key.split(".")[0] in RESUME_FIXED_KEYS or key.startswith("hydra"):
            if not key.startswith("resume"):
                logging.info(f"Ignoring override '{override}' when resuming")
            continue
        overrides.append(override.lstrip("+"))
    if overrides:
        logging.info(f"Overrides applied to the resumed run: {overrides}")
    # keys added after the run was started (e.g. new review settings) take their current defaults
    defaults = OmegaConf.to_container(cfg, resolve=False)
    defaults.pop("env", None)
    new_cfg = OmegaConf.merge(defaults, saved_cfg, OmegaConf.from_dotlist(overrides))

    # Continue logging into the resumed run and drop the new, empty Hydra output directory
    new_dir = Path.cwd()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            formatter = handler.formatter
            root_logger.removeHandler(handler)
            handler.close()
            file_handler = logging.FileHandler(run_dir / "run.log", mode="a")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    os.chdir(run_dir)
    if new_dir != run_dir and {p.name for p in new_dir.iterdir()} <= {".hydra", "run.log"}:
        shutil.rmtree(new_dir)
    return new_cfg
















