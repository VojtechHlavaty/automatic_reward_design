"""Tests of the STL robustness code and the lift-cube spec on synthetic trajectories (CPU only, no simulator).

    python -m pytest stl_check/tests -q        (from the repository root)
"""

import math

import torch

from stl_check import predicates
from stl_check.robustness import F, G, Not, Pred, all_of, any_of, episode_robustness, explain, robustness
from stl_check.specs import lift_cube

DT = 0.02
T = 250  # one 5 s episode


def sig(values):
    return {"x": torch.tensor([values], dtype=torch.float32)}


def test_predicate_margins():
    s = sig([0.06, 0.12])
    assert torch.allclose(robustness(Pred("x", ">", 0.10), s, 1.0), torch.tensor([[-0.04, 0.02]]), atol=1e-6)
    assert torch.allclose(robustness(Pred("x", "<", 0.10, scale=0.1), s, 1.0), torch.tensor([[0.4, -0.2]]), atol=1e-6)


def test_eventually_and_always_windows():
    s = sig([0.0, 1.0, 2.0, 3.0])
    x = Pred("x", ">", 0.0)
    assert robustness(F(0, 1, x), s, 1.0)[0].tolist() == [1.0, 2.0, 3.0, 3.0]
    assert robustness(G(1, 2, x), s, 1.0)[0].tolist()[:2] == [1.0, 2.0]
    # past the end: F cannot be satisfied, G is not violated
    assert robustness(F(1, 2, x), s, 1.0)[0, 3] == -math.inf
    assert robustness(G(1, 2, x), s, 1.0)[0, 3] == math.inf


def test_logic():
    s = {"x": torch.tensor([[1.0]]), "y": torch.tensor([[-2.0]])}
    px, py = Pred("x", ">", 0.0), Pred("y", ">", 0.0)
    assert float(robustness(all_of(px, py), s, 1.0)) == -2.0
    assert float(robustness(any_of(px, py), s, 1.0)) == 1.0
    assert float(robustness(Not(px), s, 1.0)) == -1.0


def test_valid_mask_ends_the_episode():
    s = sig([1.0, 1.0, -5.0, -5.0])  # the episode ended after step 1: the -5 values do not exist
    valid = torch.tensor([[True, True, False, False]])
    x = Pred("x", ">", 0.0)
    assert float(robustness(G(0, 3, x), s, 1.0, valid)[0, 0]) == 1.0
    assert float(robustness(F(2, 3, x), s, 1.0, valid)[0, 0]) == -math.inf


# --- synthetic lift-cube episodes ---------------------------------------------------------------------------------


def episode(kind: str) -> dict:
    """A scripted trajectory of one kind, the same layout as record_suite.py writes (one episode)."""
    t = torch.arange(T, dtype=torch.float32) * DT
    rest = 0.024
    cube = torch.zeros(T, 3)
    cube[:, 0], cube[:, 2] = 0.5, rest
    goal = torch.tensor([0.5, 0.1, 0.35]).expand(T, 3).clone()
    ee = torch.zeros(T, 3)
    ee[:, 0], ee[:, 2] = 0.5, 0.3
    fingers = torch.full((T, 2), 0.04)
    action = torch.ones(T)
    vel = torch.zeros(T, 3)
    force = torch.full((T,), 0.5)
    reach = (t >= 0.8)
    if kind in ("success", "lift_drop", "fling"):
        # reach at 0.8 s, close at 1.0 s, lift to the goal over 1.4-2.4 s and hold
        ee[reach, 2] = rest
        closed = t >= 1.0
        action[closed] = -1.0
        fingers[closed] = 0.024
        up = ((t - 1.4) / 1.0).clamp(0.0, 1.0)
        cube[:, :3] = torch.stack((0.5 + (goal[:, 0] - 0.5) * up, goal[:, 1] * up, rest + (goal[:, 2] - rest) * up), -1)
        ee[closed] = cube[closed]
        if kind == "lift_drop":  # opens the gripper at 2.6 s: the cube falls back to the table
            drop = t >= 2.6
            action[drop] = 1.0
            fingers[drop] = 0.04
            cube[drop, 2] = rest
        if kind == "fling":  # throws the cube through the goal at 3 m/s
            vel[(t > 2.3) & (t < 2.5)] = torch.tensor([3.0, 0.0, 0.0])
    elif kind == "hover":  # next to the cube, open gripper
        ee[reach] = torch.tensor([0.5, 0.06, rest])
    elif kind == "push":  # closed fingers sweep the cube along the table
        action[:] = -1.0
        fingers[:] = 0.0
        ee[reach] = torch.tensor([0.5, 0.0, rest])
        cube[t >= 1.0, 1] = ((t[t >= 1.0] - 1.0) * 0.1)
        vel[t >= 1.0] = torch.tensor([0.0, 0.1, 0.0])
        force[t >= 1.0] = 5.0
    traj = {"ee_pos": ee, "cube_pos": cube, "goal_pos": goal, "finger_pos": fingers, "gripper_action": action,
            "cube_lin_vel": vel, "cube_quat": torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(T, 4), "cube_force": force}
    return {k: v.unsqueeze(0) for k, v in traj.items()}


def rho(kind: str) -> float:
    return float(episode_robustness(lift_cube.SPEC, predicates.signals(episode(kind)), DT)[0])


def test_spec_accepts_success_and_rejects_hacks():
    assert rho("success") > 0
    for kind in ("hover", "push", "lift_drop", "fling"):
        assert rho(kind) < 0, kind


def test_explain_names_the_failed_part():
    lines = explain(lift_cube.SPEC, predicates.signals(episode("hover")), DT)
    assert lines[0].startswith("-") and any("grasp > 0" in line for line in lines)
