"""Hand-written STL spec of the Franka lift-cube task (``Isaac-Lift-Cube-Franka-v0``).

Episode: 5 s at a 0.02 s control step (250 steps, 0 .. 4.98 s). Every time window below ends by 4.9 s, so every
window a successful episode needs lies inside the recording.

    phi_lift =
         F[0,2.5]( grasp > 0  &  F[0,1.5]( G[0,0.5]( grasp > 0  &  cube_height > 0.10 ) ) )     (1)
      &  F[0,4.4]( G[0,0.5]( grasp > 0  &  cube_goal_dist < 0.05 ) )                              (2)
      &  G[0,4.9]( cube_height > -0.01  &  cube_speed < 1.0  &  contact_force < 250 )             (3)

(1) grasp the cube within 2.5 s; within 1.5 s after that it is held at least 0.10 m high for 0.5 s straight.
(2) before 4.4 s the cube is held within 0.05 m of the goal for 0.5 s straight.
(3) the cube never leaves the table top downwards, never moves faster than 1 m/s and is never pushed harder than
    250 N.

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
            note="a held cube moves with the hand at well below 1 m/s; above it, the cube was thrown or flicked")
GENTLE = Pred("contact_force", "<", 250.0, scale=250.0,
              note="first guess 50 N was wrong: in the recorded suite a normal grasp peaks at 20-134 N when the "
                   "binary gripper snaps shut, and pushes reach only 59-83 N, so a force limit cannot tell a push "
                   "from a grasp (the grasp and height clauses do); 250 N only catches crushing or slamming (needs "
                   "the contact sensor that env_setup.py adds)")

GRASP_THEN_LIFT = F(0.0, 2.5, all_of(GRASPED, F(0.0, 1.5, G(0.0, 0.5, all_of(GRASPED, LIFTED)))))
HELD_AT_GOAL = F(0.0, 4.4, G(0.0, 0.5, all_of(GRASPED, AT_GOAL)))
SAFE = G(0.0, 4.9, all_of(ON_TABLE, CALM, GENTLE))

SPEC = all_of(GRASP_THEN_LIFT, HELD_AT_GOAL, SAFE)
CLAUSES = {"grasp_then_lift": GRASP_THEN_LIFT, "held_at_goal": HELD_AT_GOAL, "safe": SAFE}
