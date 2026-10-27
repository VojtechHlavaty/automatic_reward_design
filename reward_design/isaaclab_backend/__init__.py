"""IsaacLab backend of the reward-design pipeline.

The backend injects the generated reward with a runtime
reward hook: an LLM-written ``compute_reward(env)`` function is loaded from a file and patched
into an unmodified IsaacLab environment, which is then trained with IsaacLab's RSL-RL script.
"""
