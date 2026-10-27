"""Video and failure review of trained policies by a vision-language model (VLM).

1. Video review: a candidate's policy is rolled out for a few episodes and recorded; the VLM watches frames of all
   episodes, scores how far each step of the curriculum was completed and explains the failure modes.
2. Failure review: the start states of failed episodes are replayed and recorded at the moment they failed; the VLM
   explains what went wrong.
Both reports go back to the LLM agents as feedback for the next curriculum.
"""

import base64
import glob
import json
import os
import re

from omegaconf import OmegaConf


def fill(template, **values):
    """Replace ``{name}`` placeholders; other braces (e.g. in JSON examples) are left untouched."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def parse_json_object(text):
    """Return the first JSON object in a model response (plain or in a ```json block), or None."""
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) + [text]
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        body = candidate[start:end + 1]
        for _ in range(5):
            try:
                return json.loads(body)
            except json.JSONDecodeError as e:
                # a stray closing bracket or brace (e.g. a string field closed with "]") is dropped and parsing retried
                if e.msg.startswith("Expecting ',' delimiter") and e.pos < len(body) and body[e.pos] in "]}":
                    body = body[:e.pos] + body[e.pos + 1:]
                    continue
                break
    return None


def image_part(b64_jpeg):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}}


def encode_image_file(path, width):
    """Base64 JPEG of an image file, resized to at most ``width`` pixels, or None."""
    import cv2

    image = cv2.imread(path)
    if image is None:
        return None
    height, image_width = image.shape[:2]
    if image_width > width:
        image = cv2.resize(image, (width, round(height * width / image_width)))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode() if ok else None


# ---------------------------------------------------------------------------
# Subtasks
# ---------------------------------------------------------------------------




def format_subtasks(subtasks):
    lines = []
    for k, subtask in enumerate(subtasks, start=1):
        line = f"{k}. {subtask['name']}: {subtask['description']}"
        if subtask.get("success_criterion"):
            line += f" (done when: {subtask['success_criterion']})"
        lines.append(line)
    return "\n".join(lines)






# ---------------------------------------------------------------------------
# Scene image and rollout videos
# ---------------------------------------------------------------------------


def viewer_overrides(cfg):
    """Hydra overrides that set the camera from the ``video_viewer`` entries of the env config."""
    viewer = cfg.env.get("video_viewer") or {}
    if OmegaConf.is_config(viewer):
        viewer = OmegaConf.to_container(viewer, resolve=True)
    overrides = []
    for key, value in viewer.items():
        if isinstance(value, (list, tuple)):
            value = "[" + ",".join(str(v) for v in value) + "]"
        overrides.append(f"env.viewer.{key}={value}")
    return overrides




def episode_length(cfg):
    """Control steps of one rollout episode in the videos."""
    return int(cfg.env.get("video_length") or cfg.video_length)


def build_video_command(cfg, python, checkpoint):
    """Command that records ``visual_rollouts`` consecutive episodes of a checkpoint with IsaacLab's play script.

    The play script writes the video to ``<checkpoint dir>/videos/play/``. The environment resets itself
    between episodes, so every episode starts from a new random initial state.
    """
    play_script = os.path.join(cfg.isaaclab_root, "scripts", "reinforcement_learning", "rsl_rl", "play.py")
    # the play wrapper applies the policy settings of training (output squashing, progress-flag observations)
    wrapper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaaclab_backend", "play_wrapper.py")
    cmd = [
        python, "-u", *([wrapper] if os.path.isfile(wrapper) else []), play_script,
        # the video can use a variant of the task with the same policy inputs (e.g. a colored scene for the VLM)
        "--task", cfg.env.get("video_task") or cfg.env.task,
        "--checkpoint", checkpoint,
        "--num_envs", "1",
        "--video",
        "--video_length", str(episode_length(cfg) * max(int(cfg.visual_rollouts), 1)),
        "--headless",
    ]
    cmd += viewer_overrides(cfg)
    cmd += [str(arg) for arg in cfg.video_extra_args]
    return cmd


def latest_checkpoint(log_dir):
    """Path of the RSL-RL checkpoint with the highest iteration number in a run directory, or None."""
    best_path, best_iteration = None, -1
    for path in glob.glob(os.path.join(log_dir, "model_*.pt")):
        match = re.search(r"model_(\d+)\.pt$", path)
        if match and int(match.group(1)) > best_iteration:
            best_path, best_iteration = path, int(match.group(1))
    return best_path


def find_new_video(log_dir, since):
    """Newest video written by the play script into a run directory after the time ``since``, or None."""
    videos = [
        path for path in glob.glob(os.path.join(log_dir, "videos", "play", "*.mp4"))
        if os.path.getmtime(path) >= since - 1
    ]
    return max(videos, key=os.path.getmtime) if videos else None


def extract_rollout_frames(video_path, num_rollouts, frames_per_rollout, width, crop=None):
    """Evenly spaced frames of each rollout of a video.

    The video holds ``num_rollouts`` consecutive episodes of equal planned length; it is split into that many
    equal segments. Returns a list of rollouts, each a list of (time in the rollout [s], base64 JPEG), or of
    (time, base64 JPEG, base64 JPEG of the close-up) with ``crop`` = [x0, y0, x1, y1] (fractions of the frame): a
    close-up of the work area at the video's resolution, in which small objects and their pose are visible.
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        capture.release()
        return []
    num_rollouts = max(int(num_rollouts), 1)
    count = max(int(frames_per_rollout), 1)
    segment = total / num_rollouts
    rollouts = []
    for r in range(num_rollouts):
        # skip the first frame of each segment: it is often black or shows the reset
        first = int(r * segment) + 1
        last = max(int((r + 1) * segment) - 1, first)
        indices = sorted({round(first + k * (last - first) / max(count - 1, 1)) for k in range(count)})
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, total - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            height, frame_width = frame.shape[:2]
            close = None
            if crop:
                x0, y0, x1, y1 = (round(f * n) for f, n in zip(crop, (frame_width, height, frame_width, height)))
                close = frame[y0:y1, x0:x1]
                if close.shape[1] > 2 * width:
                    close = cv2.resize(close, (2 * width, round(close.shape[0] * 2 * width / close.shape[1])))
            if frame_width > width:
                frame = cv2.resize(frame, (width, round(height * width / frame_width)))
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                item = ((index - int(r * segment)) / fps, base64.b64encode(buffer.tobytes()).decode())
                if close is not None:
                    ok_close, close_buffer = cv2.imencode(".jpg", close, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok_close:
                        item += (base64.b64encode(close_buffer.tobytes()).decode(),)
                frames.append(item)
        if frames:
            rollouts.append(frames)
    capture.release()
    return rollouts


# ---------------------------------------------------------------------------
# Visual evaluation
# ---------------------------------------------------------------------------


def probe_vision(client):
    """Send one small image to the model. Returns (True, "") when it accepts images, else (False, error)."""
    import cv2
    import numpy as np

    image = np.full((64, 64, 3), (40, 40, 200), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What is the main color of this image? Answer with one word."},
        image_part(base64.b64encode(buffer.tobytes()).decode()),
    ]}]
    try:
        client.complete_once(messages)
    except Exception as e:  # the API rejects image input
        return False, str(e)
    return True, ""


def normalize_report(obj):
    """Return a clean visual report dict from a parsed model response, or None."""
    if not isinstance(obj, dict):
        return None
    scores = []
    for item in obj.get("subtasks") or []:
        if not isinstance(item, dict):
            continue
        try:
            completion = float(item.get("completion", 0.0))
        except (TypeError, ValueError):
            completion = 0.0
        scores.append({
            "name": str(item.get("name", "")).strip(),
            "completion": min(max(completion, 0.0), 1.0),
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return {
        "subtasks": scores,
        "object_state": str(obj.get("object_state", "")).strip(),
        "summary": str(obj.get("summary", "")).strip(),
        "failure_modes": str(obj.get("failure_modes", "")).strip(),
        "suggestions": str(obj.get("suggestions", "")).strip(),
    }


def evaluate_rollouts(client, prompts, task_description, subtasks, rollouts, facts=""):
    """Ask the VLM to score each subtask from the frames of several rollouts.

    Returns a report dict (with "error" on failure).
    """
    if not rollouts:
        return {"error": "the video has no readable frames"}
    subtask_text = format_subtasks(subtasks) if subtasks else f"1. whole task: {task_description}"
    content = [{"type": "text", "text": fill(
        prompts["visual_eval_user"],
        task_description=task_description,
        subtasks=subtask_text,
        num_rollouts=len(rollouts),
        frames_per_rollout=len(rollouts[0]),
        duration=f"{rollouts[0][-1][0]:.1f}",
        facts=facts or "(none)",
    )}]
    for r, frames in enumerate(rollouts, start=1):
        for t, b64_jpeg, *close in frames:
            content.append({"type": "text", "text": f"Rollout {r}, t = {t:.2f} s:"})
            content.append(image_part(b64_jpeg))
            if close:
                content.append({"type": "text", "text": f"Rollout {r}, t = {t:.2f} s, close-up of the work area:"})
                content.append(image_part(close[0]))
    messages = [
        {"role": "system", "content": prompts["visual_eval_system"]},
        {"role": "user", "content": content},
    ]
    response = client.sample(messages, 1)[0][0]
    report = normalize_report(parse_json_object(response))
    if report is None:
        return {"error": "the VLM response could not be parsed", "raw_response": response}
    report["raw_response"] = response
    return report


def visual_score(report):
    """Average subtask completion of a report; -1 for a failed evaluation (ranks below any real score)."""
    if "error" in report:
        return -1.0
    scores = [s["completion"] for s in report.get("subtasks", [])]
    return sum(scores) / len(scores) if scores else 0.0


def format_report(report):
    """Plain-text version of a visual report for the LLM prompt."""
    if "error" in report:
        return f"  (visual evaluation failed: {report['error']})\n"
    lines = [
        f"  - {s['name']}: average completion {s['completion']:.2f}. {s['explanation']}"
        for s in report.get("subtasks", [])
    ]
    if report.get("failure_counts"):
        counts = ", ".join(f"{k} {v}" for k, v in sorted(report["failure_counts"].items(), key=lambda x: -x[1]))
        lines.append(f"  Failed episodes of {report.get('episodes', '?')} recorded, by the last success condition "
                     f"reached: {counts}")
    for key, label in (("object_state", "Object state"), ("summary", "What the robot does"),
                       ("failure_modes", "Failure modes"), ("suggestions", "Suggestions")):
        if report.get(key):
            lines.append(f"  {label}: {report[key]}")
    return "\n".join(lines) + "\n"


def load_prompts(prompt_dir):
    names = ["visual_eval_system", "visual_eval_user", "visual_feedback", "failure_eval_user"]
    prompts = {}
    for name in names:
        with open(os.path.join(prompt_dir, f"review_{name}.txt")) as f:
            prompts[name] = f.read()
    return prompts


# ---------------------------------------------------------------------------
# Visual evaluation of failed episodes
# ---------------------------------------------------------------------------


def build_failures_command(cfg, python, checkpoint, out_dir):
    """Command that records frames of failed episodes of a checkpoint (isaaclab_backend/record_failures.py)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaaclab_backend", "record_failures.py")
    cmd = [
        python, "-u", script,
        "--task", cfg.env.get("video_task") or cfg.env.task,
        "--checkpoint", checkpoint,
        "--adapter", cfg.env.adapter,
        "--success_flag", cfg.env.get("success_flag") or "succeeded",
        "--out_dir", out_dir,
        "--num_envs", str(int(cfg.get("visual_failure_envs", 32))),
        "--max_failures", str(int(cfg.get("visual_failures", 3))),
        "--dense_frames", str(int(cfg.get("visual_dense_frames", 6))),
        "--clip",
        "--headless",
    ]
    if cfg.env.get("visual_crop"):
        cmd += ["--crop", json.dumps(list(cfg.env.visual_crop))]
    return cmd + viewer_overrides(cfg)


def load_failure_frames(out_dir, width):
    """The recorded failures (failures.json) with their frames as base64 JPEGs, or None."""
    path = os.path.join(out_dir, "failures.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    for failure in data["failures"]:
        for frame in failure["frames"]:
            # close-ups keep up to twice the width: small objects and the gripper must stay visible
            frame["b64"] = encode_image_file(os.path.join(out_dir, frame["file"]),
                                             2 * width if frame["kind"] == "critical" else width)
            if frame.get("close_file"):
                frame["close_b64"] = encode_image_file(os.path.join(out_dir, frame["close_file"]), 2 * width)
    return data


def evaluate_failures(client, prompts, task_description, subtasks, data, facts=""):
    """Ask the VLM what goes wrong in the recorded failed episodes. Returns a report dict (with "error" on failure)."""
    failures = [f for f in data.get("failures", []) if any(fr.get("b64") for fr in f["frames"])]
    if not failures:
        return {"error": "no failure frames could be recorded"}
    subtask_text = format_subtasks(subtasks) if subtasks else f"1. whole task: {task_description}"
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(data.get("failure_counts", {}).items(), key=lambda x: -x[1]))
    content = [{"type": "text", "text": fill(
        prompts["failure_eval_user"],
        task_description=task_description,
        subtasks=subtask_text,
        episodes=data.get("episodes", 0),
        success_rate=f"{100 * data.get('success_rate', 0):.0f}%",
        failure_counts=counts or "(none)",
        num_failures=len(failures),
        note=data.get("note", ""),
        facts=facts or "(none)",
    )}]
    labels = {"start": "start", "context": "context", "critical": "critical moment (close-up of the work area)",
              "end": "end of the episode"}
    for r, failure in enumerate(failures, start=1):
        content.append({"type": "text", "text": (
            f"Failed episode {r} (it ended at {failure['end_s']:.2f} s; last success condition reached: "
            f"{failure['last_condition'] or 'none'}). Critical moment: {failure['critical_moment']}.\n"
            f"Timeline of the success conditions in this episode:\n{failure['timeline']}")})
        for frame in failure["frames"]:
            if not frame.get("b64"):
                continue
            content.append({"type": "text", "text": f"Episode {r}, t = {frame['t']:.2f} s, {labels[frame['kind']]}:"})
            content.append(image_part(frame["b64"]))
            if frame.get("close_b64"):
                content.append({"type": "text", "text": f"Episode {r}, t = {frame['t']:.2f} s, close-up:"})
                content.append(image_part(frame["close_b64"]))
    messages = [
        {"role": "system", "content": prompts["visual_eval_system"]},
        {"role": "user", "content": content},
    ]
    response = client.sample(messages, 1)[0][0]
    report = normalize_report(parse_json_object(response))
    if report is None:
        return {"error": "the VLM response could not be parsed", "raw_response": response}
    report["raw_response"] = response
    report["failure_counts"] = data.get("failure_counts", {})
    report["episodes"] = data.get("episodes", 0)
    return report
