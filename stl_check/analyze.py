"""Validate the spec on the trajectory suite and test the LLM rewards against it (CPU only, no simulator).

    python -m stl_check.analyze --suite_dir suite --rewards <stage0.yaml> ... --out suite/results.md

1. Robustness of every episode (specs/lift_cube.SPEC): every success must score rho > 0, every random and hack
   episode rho < 0. Episodes on the wrong side are listed with the parts that decided them, and when. The stage
   specs (lift_cube.STAGE_SPECS) are listed too: how many episodes of each kind pass each.
2. For every reward (its returns from score_rewards.py, files <kind>_returns.json in the suite folder), against
   the spec of its own stage (from the stage YAML's stage name):
   Kendall tau between return and rho over all episodes, and the number of episodes failing that spec whose return
   beats the worst episode passing it. For reference also the number of hack episodes whose return beats the worst
   success under the whole-task spec. Then LLM #2's verdict (the stage YAML's hacking_review, written by the
   pipeline's reward-hacking review) and the policy outcome (read from the run's pipeline.log, three folders above
   iter<I>/cand<C>/stage<S>.yaml, unless --outcomes gives it).
Writes the results table (markdown) and results.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

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


def signals_of(data):
    traj = {k: v.float() for k, v in data["traj"].items()}
    return predicates.signals(traj)


def stage_yaml(reward_path):
    if not reward_path.endswith((".yaml", ".yml")):
        return {}
    return yaml.safe_load(open(reward_path)) or {}


def stage_of(reward_path):
    """(stage name, stage spec kind) of a reward; a reward that is not a stage YAML is judged on the whole task."""
    stage = stage_yaml(reward_path).get("stage") or {}
    name = str(stage.get("name", "?"))
    flag = (stage.get("transition") or {}).get("flag")
    return name, lift_cube.stage_kind(name, flag) if name != "?" else "transport"


def verdict(reward_path):
    """LLM #2's verdict on a reward: (severity, its finding in full) from the stage YAML's hacking_review.

    The review stores the behavior the reward pays most for under highest_reward_behavior. A reward rewritten after
    the review ("not reviewed again after the last fix") keeps the earlier review's finding under
    previous_findings."""
    if not reward_path.endswith((".yaml", ".yml")):
        return "(not a stage YAML)", ""
    review = stage_yaml(reward_path).get("hacking_review") or {}
    if not review:
        return "(no review stored)", ""
    severity = str(review.get("severity", "?"))
    text = review.get("highest_reward_behavior") or review.get("summary")
    if not text and review.get("previous_findings"):
        severity, text = "not reviewed after the last fix", f"earlier finding: {review['previous_findings']}"
    return severity, " ".join(str(text or "").split())


RESULT_LINE = re.compile(r"(\w+)_iter(\d+)_cand(\d+)_stage(\d+)(?:_ext\d+)?: task score .*?, (\w+) rate ([\d.]+) "
                         r"\((reached|not reached)\)")
EVAL_LINE = re.compile(r"Iteration (\d+): candidate (\d+) stage (\d+): Checkpoint evaluation [^:]*: success ([\d.]+)%;"
                       r" episodes reaching each success condition: (.*?)\. Share")
STAGE_PATH = re.compile(r"iter(\d+)/cand(\d+)/stage(\d+)\.ya?ml$")


def outcome(reward_path):
    """Policy outcome of the stage trained on this reward, from the run's pipeline.log: the stage's flag rate at the
    end of training (the last extension's, if any), and the checkpoint evaluation if this stage's policy was
    evaluated (the best stage of each candidate)."""
    m = STAGE_PATH.search(reward_path)
    log = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(reward_path))), "pipeline.log")
    if not m or not os.path.exists(log):
        return ""
    key = tuple(int(x) for x in m.groups())
    end, evaluation = "", ""
    for line in open(log, errors="replace"):
        r = RESULT_LINE.search(line)
        if r and tuple(int(x) for x in r.groups()[1:4]) == key:
            end = f"{r.group(5)} {float(r.group(6)):.0%} at the end of training ({r.group(7)})"
        e = EVAL_LINE.search(line)
        if e and tuple(int(x) for x in e.groups()[:3]) == key:
            evaluation = f"evaluated: success {float(e.group(4)):g}%, {e.group(5)}"
    return "; ".join(x for x in (end, evaluation) if x)


def ranked_above(ret, passed, labels, failing):
    """Labels of the episodes in ``failing`` whose return beats the worst return of an episode in ``passed``."""
    worst = min((r for r, ok in zip(ret, passed) if ok), default=None)
    if worst is None:
        return None
    return [k for r, k, bad in zip(ret, labels, failing) if bad and r > worst]


def count_by_kind(above, kinds):
    return {k: sum(1 for x in above if x == k) for k in kinds}


def fmt_counts(above, kinds):
    if above is None:
        return "(no episode passes)"
    by_kind = count_by_kind(above, kinds)
    return f"{len(above)} ({', '.join(f'{k} {v}' for k, v in by_kind.items() if v) or 'none'})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", required=True)
    parser.add_argument("--rewards", nargs="*", default=[])
    parser.add_argument("--outcomes", nargs="*", default=[], help="policy outcome per reward (e.g. 'success 12%%'); "
                        "default: read from the run's pipeline.log")
    parser.add_argument("--out", required=True)
    parser.add_argument("--undiscounted", action="store_true",
                        help="rank by the plain sum of the rewards instead of the discounted return PPO maximizes")
    args = parser.parse_args()

    suite = load_suite(args.suite_dir)
    sigs = {kind: signals_of(data) for kind, data in suite.items()}

    def rho(phi, kind):
        return episode_robustness(phi, sigs[kind], suite[kind]["dt"], suite[kind]["valid"])

    lines = ["# STL check: Franka lift-cube", "", "## Spec validation", "",
             "| Episode type | Episodes | rho min | rho max | on the wrong side |", "|---|---|---|---|---|"]
    rhos, kinds = {}, {}
    wrong = []
    for kind, data in suite.items():
        r = rho(lift_cube.SPEC, kind)
        rhos[kind], kinds[kind] = r, [kind] * len(r)
        should_pass = kind in SUCCESS_KINDS
        bad = [(i, float(x)) for i, x in enumerate(r) if (x > 0) != should_pass]
        lines.append(f"| {kind} | {len(r)} | {float(r.min()):+.3f} | {float(r.max()):+.3f} | {len(bad)} |")
        for i, x in bad[:3]:
            wrong.append((kind, i, x, explain(lift_cube.SPEC, sigs[kind], data["dt"], data["valid"], episode=i)))
    if wrong:
        lines += ["", "Episodes on the wrong side. Each part's robustness at the time that decided its parent (F: "
                  "the best step of its window, G: the worst):", ""]
        for kind, i, x, tree in wrong:
            lines += [f"- {kind} episode {i}: rho {x:+.3f}", "```", *tree, "```"]

    stage_rhos = {s: {k: rho(phi, k) for k in suite} for s, phi in lift_cube.STAGE_SPECS.items()}
    lines += ["", "### Stage specs", "",
              "Episodes passing the spec of each curriculum stage (rho > 0), of the episodes of each kind. A stage "
              "spec only asks for that stage's goal, so a hack of the whole task can pass an early stage's spec "
              "(lift_drop does grasp).", "",
              "| Stage spec | " + " | ".join(suite) + " |", "|---|" + "---|" * len(suite)]
    for s, by_kind in stage_rhos.items():
        lines.append(f"| {s} | " + " | ".join(f"{int((v > 0).sum())}/{len(v)}" for v in by_kind.values()) + " |")

    results = {"rho": {k: v.tolist() for k, v in rhos.items()},
               "stage_rho": {s: {k: v.tolist() for k, v in d.items()} for s, d in stage_rhos.items()},
               "rewards": {}}
    if args.rewards:
        lines += ["", "## Rewards", "",
                  ("Returns: the plain sum of the rewards over the episode. " if args.undiscounted else
                   "Returns: discounted with the PPO gamma of the stage (what PPO maximizes; potential terms then "
                   "add up to gamma^T phi_T - phi_0). ") +
                  "Against the spec of the reward's own stage: Kendall tau of return vs rho, and the episodes failing "
                  "that spec whose return beats the worst episode passing it. Last column but two: hack episodes "
                  "ranked above the worst success (a success or policy episode with whole-task rho > 0; only meaningful "
                  "for the transport stage).", "",
                  "| Reward | Stage (spec) | Kendall tau vs stage rho | Failing episodes above a passing one | "
                  "Hacks above a success (whole task) | LLM #2 verdict | Policy outcome |",
                  "|---|---|---|---|---|---|---|"]
        returns = {}
        # <kind>_returns.json and <kind>_returns_<group>.json (rewards scored in separate runs of score_rewards.py)
        for path in glob.glob(os.path.join(args.suite_dir, "*_returns*.json")):
            data = json.load(open(path))
            key = "returns" if args.undiscounted else "discounted_returns"
            returns.setdefault(data["kind"], {}).update(data[key])
        findings = []
        for n, reward in enumerate(args.rewards):
            name, spec_kind = stage_of(reward)
            severity, finding = verdict(reward)
            findings.append((n, severity, finding))
            out = args.outcomes[n] if n < len(args.outcomes) else outcome(reward)
            ret, task_rho, st_rho, kind = [], [], [], []
            for k in rhos:
                if k in returns and reward in returns[k]:
                    ret += returns[k][reward]
                    task_rho += rhos[k].tolist()
                    st_rho += stage_rhos[spec_kind][k].tolist()
                    kind += kinds[k]
            label = f"reward_{n + 1}"
            if not ret:
                lines.append(f"| {label} | {name} ({spec_kind}) | (no returns: run score_rewards.py) | | | "
                             f"{severity} | {out} |")
                continue
            tau, p = kendalltau(ret, st_rho)
            failing_above = ranked_above(ret, [x > 0 for x in st_rho], kind, [x <= 0 for x in st_rho])
            # a success: an episode of a success kind that the whole-task spec accepts (a policy episode can fail)
            real = [k in SUCCESS_KINDS and r > 0 for k, r in zip(kind, task_rho)]
            hacks_above = ranked_above(ret, real, kind, [k in HACK_KINDS for k in kind])
            lines.append(f"| {label} | {name} ({spec_kind}) | {tau:+.2f} (p = {p:.2g}) | "
                         f"{fmt_counts(failing_above, list(suite))} | {fmt_counts(hacks_above, HACK_KINDS)} | "
                         f"{severity} | {out} |")
            results["rewards"][reward] = {
                "stage": name, "spec": spec_kind, "tau": tau, "p": p,
                "failing_above_a_passing": None if failing_above is None else len(failing_above),
                "failing_by_kind": None if failing_above is None else count_by_kind(failing_above, list(suite)),
                "hacks_above_a_success": None if hacks_above is None else len(hacks_above),
                "hacks_by_kind": None if hacks_above is None else count_by_kind(hacks_above, HACK_KINDS),
                "severity": severity, "finding": finding, "outcome": out}
        lines += ["", "LLM #2 findings (the behavior the review found the reward pays most for):", ""]
        lines += [f"- reward_{n + 1} ({severity}): {finding or '(none)'}" for n, severity, finding in findings]
        lines += ["", "Rewards:", ""] + [f"- reward_{n + 1}: `{r}`" for n, r in enumerate(args.rewards)]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
