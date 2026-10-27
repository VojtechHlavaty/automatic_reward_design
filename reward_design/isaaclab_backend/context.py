"""Build the environment source code that the pipeline shows to the LLM for an IsaacLab task.

The context holds the task's environment and configuration sources with the human-engineered
reward removed:

* Direct environments: the environment class file without ``_get_rewards``, plus the config files.
* Manager-based environments: the config class files (the task config and its parents inside
  ``isaaclab_tasks``) without reward config classes and ``rewards`` assignments.

Usage (from the ``reward_design`` directory, with the IsaacLab Python)::

    python -m isaaclab_backend.context <gym-id> env_context/<env_name>_obs.py
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys


def _module_file(module_name: str) -> str:
    return importlib.util.find_spec(module_name).origin


def _strip_header(source: str) -> str:
    """Remove the leading license comment block."""
    lines = source.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


def _remove_lines(source: str, ranges: list[tuple[int, int]]) -> str:
    """Remove 1-indexed inclusive line ranges from the source."""
    drop = set()
    for start, end in ranges:
        drop.update(range(start, end + 1))
    return "\n".join(line for i, line in enumerate(source.split("\n"), start=1) if i not in drop)


def _node_range(node: ast.AST) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def strip_direct_rewards(source: str) -> str:
    """Remove ``_get_rewards`` methods from a direct environment file."""
    ranges = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_get_rewards":
                    ranges.append(_node_range(item))
    return _remove_lines(source, ranges)


def strip_manager_rewards(source: str) -> str:
    """Remove reward config classes and ``rewards`` settings from a manager-based config file."""
    tree = ast.parse(source)
    ranges = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and "reward" in node.name.lower():
            ranges.append(_node_range(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                target = getattr(item, "target", None)
                targets = getattr(item, "targets", [target])
                if any(isinstance(t, ast.Name) and t.id == "rewards" for t in targets):
                    ranges.append(_node_range(item))
        if isinstance(node, ast.FunctionDef):
            for stmt in ast.walk(node):
                if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr)):
                    segment = ast.get_source_segment(source, stmt) or ""
                    if "self.rewards" in segment:
                        ranges.append(_node_range(stmt))
    return _remove_lines(source, ranges)


def _cfg_class_files(cfg_cls: type) -> list[str]:
    """Files of the config class and its parents that live inside ``isaaclab_tasks``."""
    files = []
    for cls in cfg_cls.__mro__:
        if not cls.__module__.startswith("isaaclab_tasks") and not _in_tasks_dir(cls):
            continue
        path = inspect.getsourcefile(cls)
        if path not in files:
            files.append(path)
    return files


def _in_tasks_dir(cls) -> bool:
    """Whether a class is defined in one of this repository's task packages."""
    from isaaclab_backend.tasks_import import TASKS_DIR

    try:
        path = inspect.getsourcefile(cls) or ""
    except TypeError:  # built-in classes have no source file
        return False
    return os.path.abspath(path).startswith(TASKS_DIR + os.sep)


def _display_path(path: str) -> str:
    """A machine-independent name of a source file: relative to this repository, or to IsaacLab's ``source`` folder."""
    from isaaclab_backend.tasks_import import TASKS_DIR

    path = os.path.abspath(path)
    repo = os.path.dirname(TASKS_DIR)
    if path.startswith(repo + os.sep):
        return os.path.relpath(path, repo)
    marker = os.sep + "source" + os.sep
    return path.split(marker, 1)[1] if marker in path else os.path.basename(path)


def _load_attr(entry_point: str):
    module_name, attr_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), attr_name)


def build_task_context(task_id: str) -> str:
    """Return the reward-free environment source for an IsaacLab gym task."""
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_backend.tasks_import import register_tasks

    register_tasks()

    spec = gym.spec(task_id)
    cfg_cls = _load_attr(spec.kwargs["env_cfg_entry_point"])
    cfg_bases = {cls.__name__ for cls in cfg_cls.__mro__}
    parts = []
    if "DirectRLEnvCfg" in cfg_bases:
        env_file = _module_file(spec.entry_point.split(":")[0])
        env_source = strip_direct_rewards(_strip_header(open(env_file).read()))
        parts.append((env_file, env_source))
        for path in _cfg_class_files(cfg_cls):
            parts.append((path, _strip_header(open(path).read())))
    elif "ManagerBasedRLEnvCfg" in cfg_bases:
        for path in reversed(_cfg_class_files(cfg_cls)):
            parts.append((path, strip_manager_rewards(_strip_header(open(path).read()))))
    else:
        raise TypeError(f"Unsupported environment config type for '{task_id}': {cfg_cls.__name__}.")

    env_kind = "direct" if "DirectRLEnvCfg" in cfg_bases else "manager-based"
    header = f"# IsaacLab task: {task_id} ({env_kind} environment)\n"
    body = "\n\n".join(f"# ---- File: {_display_path(path)} ----\n{source.strip()}\n" for path, source in parts)
    return header + "\n" + body


if __name__ == "__main__":
    context = build_task_context(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(context)
    else:
        print(context)
