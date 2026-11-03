"""Cross-check our robustness against rtamt on recorded trajectories (CPU only; needs ``pip install rtamt``).

    python -m stl_check.check_rtamt --suite_dir suite --episodes 3

For each chosen episode, every predicate of the spec is turned into its scaled margin signal p_i (so that rtamt's
"p_i >= 0" scores exactly our margin / scale), the spec is written in rtamt's discrete-time syntax with the time
bounds in steps (robustness.rtamt_spec), and both robustness values at step 0 must agree.
"""

from __future__ import annotations

import argparse

import torch

from stl_check import predicates
from stl_check.analyze import load_suite
from stl_check.robustness import episode_robustness, robustness, rtamt_spec
from stl_check.specs import lift_cube


def rtamt_value(formula, preds, sig, valid, dt, episode):
    import rtamt

    spec = rtamt.StlDiscreteTimeSpecification()
    for name in preds.values():
        spec.declare_var(name, "float")
    spec.spec = formula
    spec.parse()
    steps = int(valid[episode].sum())  # only the steps of this episode
    data = {"time": list(range(steps))}
    for pred, name in preds.items():
        data[name] = robustness(pred, sig, dt)[episode, :steps].tolist()
    out = spec.evaluate(data)
    return float(out[0][1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    formula, preds = rtamt_spec(lift_cube.SPEC, lift_cube.DT)
    print("rtamt formula:", formula)
    checked, mismatches = 0, 0
    for kind, data in load_suite(args.suite_dir).items():
        traj = {k: v.float() for k, v in data["traj"].items()}
        sig = predicates.signals(traj)
        ours = episode_robustness(lift_cube.SPEC, sig, data["dt"], data["valid"])
        for e in range(min(args.episodes, len(ours))):
            theirs = rtamt_value(formula, preds, sig, data["valid"], data["dt"], e)
            ok = abs(theirs - float(ours[e])) <= args.tolerance
            checked += 1
            mismatches += not ok
            print(f"{kind} episode {e}: ours {float(ours[e]):+.6f}, rtamt {theirs:+.6f} {'ok' if ok else 'MISMATCH'}")
    print(f"{checked} episodes checked, {mismatches} mismatches")


if __name__ == "__main__":
    main()
