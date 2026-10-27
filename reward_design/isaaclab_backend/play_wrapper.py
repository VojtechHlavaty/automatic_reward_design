"""Run IsaacLab's RSL-RL play script with the pipeline's policy settings: the exploration settings (output squashing, as in
training) and the optional progress-flag observations. Both are installed when play.py imports rsl_rl, i.e. after
the simulator has started (installing them earlier breaks the simulator's startup).

    python -m isaaclab_backend.play_wrapper <path/to/play.py> <play.py arguments ...>
"""

import builtins
import os
import runpy
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    play = sys.argv.pop(1)
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.dirname(os.path.abspath(play)))
    sys.argv[0] = play
    original_import = builtins.__import__
    state = {"done": False}

    def hooked(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if not state["done"] and name.startswith("rsl_rl"):
            state["done"] = True
            if os.environ.get("RD_EXPLORATION"):
                from rd_lib import exploration  # noqa: F401
            if os.environ.get("RD_OBS_FLAGS"):
                from rd_lib import obs_extension  # noqa: F401
            from isaaclab_backend.tasks_import import register_tasks

            register_tasks()  # the tasks of this repository (tasks/)
        return module

    builtins.__import__ = hooked
    runpy.run_path(play, run_name="__main__")


if __name__ == "__main__":
    main()
