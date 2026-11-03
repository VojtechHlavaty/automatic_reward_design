"""Run the whole STL check for the Franka lift-cube task: record the trajectory suite, score the rewards, analyze.

    python stl_check/run_suite.py --out_dir stl_check/suite --gpu 0 \\
        --rewards <stage YAMLs of LLM-generated lift rewards> [--outcomes "success 12%" ...] \\
        [--checkpoint <trained policy, adds a 'policy' success set>] [--skip_record]

Steps (each simulator step is its own process, as Isaac Sim starts once per process):
1. record_suite.py for every kind: success, random, hover, push, lift_drop (and policy with --checkpoint);
2. score_rewards.py on every suite file with all rewards (one replay per file);
3. analyze.py: spec validation and the results table (<out_dir>/results.md).
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KINDS = ["success", "random", "hover", "push", "lift_drop"]


def run(cmd, env, log):
    print("$ " + " ".join(cmd), flush=True)
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
    if code != 0:
        raise SystemExit(f"failed ({code}); see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(HERE, "suite"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_envs", type=int, default=16, help="episodes per kind")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rewards", nargs="*", default=[])
    parser.add_argument("--outcomes", nargs="*", default=[])
    parser.add_argument("--skip_record", action="store_true", help="reuse the recorded suite files")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMNI_KIT_ACCEPT_EULA="YES")
    for key in ("RD_SKILL", "RD_HARD_STARTS", "RD_OBS_FLAGS", "RD_EXPLORATION", "RD_REWARD_FILE"):
        env.pop(key, None)
    python = sys.executable
    kinds = KINDS + (["policy"] if args.checkpoint else [])
    if not args.skip_record:
        for i, kind in enumerate(kinds):
            cmd = [python, os.path.join(HERE, "record_suite.py"), "--kind", kind, "--num_envs", str(args.num_envs),
                   "--seed", str(args.seed + i), "--out", os.path.join(args.out_dir, f"{kind}.pt"), "--headless"]
            if kind == "policy":
                cmd += ["--checkpoint", args.checkpoint]
            run(cmd, env, os.path.join(args.out_dir, f"record_{kind}.log"))
    if args.rewards:
        for kind in kinds:
            cmd = [python, os.path.join(HERE, "score_rewards.py"), "--suite", os.path.join(args.out_dir, f"{kind}.pt"),
                   "--rewards", *[os.path.abspath(r) for r in args.rewards],
                   "--out", os.path.join(args.out_dir, f"{kind}_returns.json"), "--headless"]
            run(cmd, env, os.path.join(args.out_dir, f"score_{kind}.log"))
    cmd = [python, "-m", "stl_check.analyze", "--suite_dir", args.out_dir, "--out",
           os.path.join(args.out_dir, "results.md")]
    if args.rewards:
        cmd += ["--rewards", *[os.path.abspath(r) for r in args.rewards]]
    if args.outcomes:
        cmd += ["--outcomes", *args.outcomes]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
