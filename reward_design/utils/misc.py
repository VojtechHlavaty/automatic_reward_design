import re
import subprocess

TRACEBACK_BEGIN = "=== TRAINING TRACEBACK BEGIN ==="
TRACEBACK_END = "=== TRAINING TRACEBACK END ==="
TRAINING_DONE_MARKER = "Training time:"
EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b.*")


def get_gpu_ids():
    """Return the indices of all visible GPUs."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout
    return [int(line) for line in out.split()]


def filter_traceback(s):
    """Return the first Python traceback in a training log, or an empty string."""
    if TRACEBACK_BEGIN in s:
        s = s.split(TRACEBACK_BEGIN, 1)[1].split(TRACEBACK_END, 1)[0]
    lines = s.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            filtered_lines = []
            for j in range(i, len(lines)):
                filtered_lines.append(lines[j])
                if j > i and EXCEPTION_LINE.match(lines[j]):
                    break
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found
