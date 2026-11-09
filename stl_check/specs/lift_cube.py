"""Hand-written STL spec of the Franka lift-cube task (``Isaac-Lift-Cube-Franka-v0``).

Episode: 5 s at a 0.02 s control step (250 steps, 0 .. 4.98 s). Every time window below ends by 4.9 s, so every
window a successful episode needs lies inside the recording.

    phi_lift =
         F[0,2.5]( grasp > 0  &  F[0,1.5]( G[0,0.5]( grasp > 0  &  cube_height > 0.10 ) ) )     (1)
      &  F[0,4.4]( G[0,0.5]( grasp > 0  &  cube_goal_dist < 0.05 ) )                              (2)
      &  G[0,4.9]( cube_height > -0.01  &  contact_force < 250 )                                  (3)

(1) grasp the cube within 2.5 s; within 1.5 s after that it is held at least 0.10 m high for 0.5 s straight.
(2) before 4.4 s the cube is held within 0.05 m of the goal for 0.5 s straight.
(3) the cube never leaves the table top downwards and is never pushed harder than 250 N.

The first spec also had ``cube_speed < 1.0`` in (3), to catch a thrown cube. It was dropped after the 16-episode
suite (8 Nov 2026): the trained policy (franka_progress, 98.8% success) lifts the held cube at up to 2.4 m/s, so all
16 of its episodes failed only on the speed limit, and one scripted success did too (1.9 m/s while held). The limit
separated no hack (every hack fails (1) by at least 0.7), and a thrown cube already fails (2), which needs the cube
held at the goal for 0.5 s.

The task's own success ("the cube center was above 0.04 m and within 0.05 m of the goal at any one step") is
weaker: a cube flung through the goal, or one balanced on open fingers, satisfies it for a step. Clause (2)
needs the cube held at the goal for 0.5 s.

Scales put the margins on comparable footing under min/max (robustness = margin / scale): distances are divided by
their threshold, so -1 means "off by one threshold"; ``grasp`` is already normalized (predicates.grasp_margin).
"""

from __future__ import annotations

from stl_check.robustness import F, G, Pred, all_of

DT = 0.02  # control step of the task [s]

GRASPED = Pred("grasp", ">", 0.0, scale=1.0,
               note="normalized grasp margin: fingertip center within 2 cm of the cube, gripper commanded closed, "
                    "fingers not shut (no finger contact sensor in the task)")
LIFTED = Pred("cube_height", ">", 0.10, scale=0.10,
              note="the cube center rests at 0.021 m (measured); 0.10 m is about 8 cm off the table. The task's own "
                   "lift test (0.04 m) is only about 2 cm off the table, which a tipped or bumped cube nearly "
                   "reaches")
AT_GOAL = Pred("cube_goal_dist", "<", 0.05, scale=0.05,
               note="the task's own success threshold")
ON_TABLE = Pred("cube_height", ">", -0.01, scale=0.10,
                note="below the table top means the cube fell off (the task ends the episode at -0.05)")
CALM = Pred("cube_speed", "<", 1.0, scale=1.0,
            note="not in the spec any more: the trained policy lifts the held cube at up to 2.4 m/s (see above)")
GENTLE = Pred("contact_force", "<", 250.0, scale=250.0,
              note="first guess 50 N was wrong: in the recorded suite a normal grasp peaks at 20-134 N when the "
                   "binary gripper snaps shut, and pushes reach only 59-83 N, so a force limit cannot tell a push "
                   "from a grasp (the grasp and height clauses do); 250 N only catches crushing or slamming (needs "
                   "the contact sensor that env_setup.py adds)")

GRASP_THEN_LIFT = F(0.0, 2.5, all_of(GRASPED, F(0.0, 1.5, G(0.0, 0.5, all_of(GRASPED, LIFTED)))))
HELD_AT_GOAL = F(0.0, 4.4, G(0.0, 0.5, all_of(GRASPED, AT_GOAL)))
SAFE = G(0.0, 4.9, all_of(ON_TABLE, GENTLE))

SPEC = all_of(GRASP_THEN_LIFT, HELD_AT_GOAL, SAFE)
CLAUSES = {"grasp_then_lift": GRASP_THEN_LIFT, "held_at_goal": HELD_AT_GOAL, "safe": SAFE}

# Stage specs. The pipeline trains a curriculum of stages (reach, grasp, lift, transport), and each stage's reward
# only pays for that stage's goal: under the whole-task spec a reach reward that pays for hovering is right, not
# hacked. Each stage reward is therefore also judged against the spec of its own stage: that stage's goal, held for
# 0.5 s within 2.5 s, with the safety clause. The transport stage ends the task, so its spec is the whole spec.
AT_CUBE = Pred("ee_cube_dist", "<", 0.03, scale=0.03,
               note="the pipeline's own at_cube flag (tasks/franka_lift/quantities.py, REACH_RADIUS)")
REACHED = F(0.0, 2.5, G(0.0, 0.5, AT_CUBE))
GRASP_HELD = F(0.0, 2.5, G(0.0, 0.5, GRASPED))
STAGE_SPECS = {
    "reach": all_of(REACHED, SAFE),
    "grasp": all_of(GRASP_HELD, SAFE),
    "lift": all_of(GRASP_THEN_LIFT, SAFE),
    "transport": SPEC,
}


FLAG_KINDS = {"at_cube": "reach", "gripper_closing": "grasp", "grasped": "grasp", "lifted": "lift",
              "held_lifted": "lift", "at_goal": "transport", "succeeded": "transport"}


def stage_kind(stage_name: str, flag: str | None = None) -> str:
    """The stage spec of a pipeline stage. By its name first (reach_dwell, grasp_secure, lift_held,
    transport_to_goal; the latest stage word wins, so grasp_lift_stabilize is a lift stage), since the pipeline
    changed the transition flag of some stages (a grasp stage with the at_cube flag) when a stage was extended. A
    name without a stage word (sustained_goal_hold, secure_upright_carry) goes by its transition flag."""
    for kind in ("transport", "lift", "grasp", "reach"):
        if kind in stage_name:
            return kind
    return FLAG_KINDS.get(flag or "", "transport")
