"""Curriculum reward design with LLM agents on IsaacLab.

* ``quantities``: the quantity type and the loader of the task adapters (``tasks/<package>/quantities.py``).
* ``reward_registry``: the reward-term function registry and the compiler from a stage YAML to ``compute_reward(env)``.
* ``schema``: schema and semantic validation of stage YAMLs.
* ``vdb``: the retrieval database of earlier curricula, their outcomes and feedback.
* ``agents``: the LLM agents (query, selector, planner, stage generator, feedback).
"""
