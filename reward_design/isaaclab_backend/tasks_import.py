"""Register the gym tasks of the task packages in this repository's ``tasks`` folder.

Each sub-folder of ``tasks`` with an ``__init__.py`` is a task package that registers its tasks with gymnasium when
imported (like IsaacLab's ``isaaclab_tasks``). Call :func:`register_tasks` after IsaacLab's simulator has started.
"""

import importlib
import os
import sys

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tasks")


def register_tasks():
    """Import every task package in ``tasks`` (registers their gym tasks); returns the package names."""
    if TASKS_DIR not in sys.path:
        sys.path.insert(0, TASKS_DIR)
    names = sorted(n for n in os.listdir(TASKS_DIR) if os.path.isfile(os.path.join(TASKS_DIR, n, "__init__.py")))
    for name in names:
        importlib.import_module(name)
    return names
