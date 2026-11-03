"""Validate the spec on the trajectory suite and test the LLM rewards against it (CPU only, no simulator).

    python -m stl_check.analyze --suite_dir suite --rewards <stage0.yaml> ... --out suite/results.md

1. Robustness of every episode (specs/lift_cube.SPEC): every success must score rho > 0, every random and hack
   episode rho < 0. Episodes on the wrong side are listed with the clause that decided them.
2. For every reward (its returns from score_rewards.py, files <kind>_returns.json in the suite folder):
   Kendall tau between return and rho over all episodes, the number of hack episodes whose return beats the worst
   success, and LLM #2's verdict (the stage YAML's hacking_review, written by the pipeline's reward-hacking review).
Writes the results table (markdown) and results.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import yaml
from scipy.stats import kendalltau

from stl_check import predicates
from stl_check.robustness import episode_robustness, explain
from stl_check.specs import lift_cube

SUCCESS_KINDS = ("success", "policy")
HACK_KINDS = ("hover", "push", "lift_drop")


def load_suite(suite_dir):
    """{kind: suite dict} of the recorded files <kind>.pt."""
    suite = {}
    for path in sorted(glob.glob(os.path.join(suite_dir, "*.pt"))):
        data = torch.load(path)
        suite[data["kind"]] = data
    return suite


def rho_of(data):
    traj = {k: v.float() for k, v in data["traj"].items()}
    sig = predicates.signals(traj)
    return episode_robustness(lift_cube.SPEC, sig, data["dt"], data["valid"]), sig


def verdict(reward_path):
    """LLM #2's verdict on a reward: the stage YAML's hacking_review (severity and the first sentence)."""
    if not reward_path.endswith((".yaml", ".yml")):
        return "(not a stage YAML)"
    review = (yaml.safe_load(open(reward_path)) or {}).get("hacking_review") or {}
    if not review:
        return "(no review stored)"
    text = str(review.get("highest_reward_behavior") or review.get("summary") or "").split(". ")[0]
    return f"{review.get('severity', '?')}: {text[:160]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", required=True)
    parser.add_argument("--rewards", nargs="*", default=[])
    parser.add_argument("--outcomes", nargs="*", default=[], help="policy outcome per reward (e.g. 'success 12%%')")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suite = load_suite(args.suite_dir)
    lines = ["# STL check: Franka lift-cube", "", "## Spec validation", "",
             "| Episode type | Episodes | rho min | rho max | on the wrong side |", "|---|---|---|---|---|"]
    rhos, kinds = {}, {}
    wrong = []
    for kind, data in suite.items():
        rho, sig = rho_of(data)
        rhos[kind], kinds[kind] = rho, [kind] * len(rho)
        should_pass = kind in SUCCESS_KINDS
        bad = [(i, float(r)) for i, r in enumerate(rho) if (r > 0) != should_pass]
        lines.append(f"| {kind} | {len(rho)} | {float(rho.min()):+.3f} | {float(rho.max()):+.3f} | {len(bad)} |")
        for i, r in bad[:3]:
            wrong.append((kind, i, r, explain(lift_cube.SPEC, sig, data["dt"], data["valid"], episode=i)))
    if wrong:
        lines += ["", "Episodes on the wrong side (robustness of every part at step 0):", ""]
        for kind, i, r, tree in wrong:
            lines += [f"- {kind} episode {i}: rho {r:+.3f}", "```", *tree, "```"]

    results = {"rho": {k: v.tolist() for k, v in rhos.items()}, "rewards": {}}
    if args.rewards:
        lines += ["", "## Rewards", "", "| Reward | Kendall tau vs rho | Hacks ranked above a success | LLM #2 verdict | "
                  "Policy outcome |", "|---|---|---|---|---|"]
        returns = {}
        for path in glob.glob(os.path.join(args.suite_dir, "*_returns.json")):
            data = json.load(open(path))
            returns[data["kind"]] = data["returns"]
        for n, reward in enumerate(args.rewards):
            ret, rho, kind = [], [], []
            for k in rhos:
                if k in returns and reward in returns[k]:
                    ret += returns[k][reward]
                    rho += rhos[k].tolist()
                    kind += kinds[k]
            if not ret:
                lines.append(f"| reward_{n + 1} | (no returns: run score_rewards.py) | | {verdict(reward)} | |")
                continue
            tau, p = kendalltau(ret, rho)
            worst_success = min((r for r, k in zip(ret, kind) if k in SUCCESS_KINDS), default=None)
            above = [k for r, k in zip(ret, kind) if k in HACK_KINDS and worst_success is not None and r > worst_success]
            outcome = args.outcomes[n] if n < len(args.outcomes) else ""
            by_kind = {k: sum(1 for x in above if x == k) for k in HACK_KINDS}
            lines.append(f"| reward_{n + 1} | {tau:+.2f} (p = {p:.2g}) | {len(above)} "
                         f"({', '.join(f'{k} {v}' for k, v in by_kind.items() if v) or 'none'}) | {verdict(reward)} | "
                         f"{outcome} |")
            results["rewards"][reward] = {"tau": tau, "p": p, "hacks_above_a_success": len(above),
                                          "by_kind": by_kind, "verdict": verdict(reward)}
        lines += ["", "Rewards:", ""] + [f"- reward_{n + 1}: `{r}`" for n, r in enumerate(args.rewards)]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
