"""Task quantities that reward terms can use.

Each task adapter maps quantity names to a function of the IsaacLab environment. A quantity is either a
``scalar`` (a float tensor of shape (num_envs,), with a unit) or a ``flag`` (a bool tensor of shape (num_envs,)).
The LLM only composes reward terms from these quantities, so it never reads environment internals directly.

The adapters belong to the tasks: every ``tasks/<package>/quantities.py`` (``RD_TASKS_DIR`` overrides the tasks
folder) defines ``ADAPTERS`` ({name: function returning {quantity name: Quantity}}) and optionally ``FUNNELS``
({name: function returning the ordered success conditions [(condition name, fn(env) -> bool tensor)]}) and
``HOLDS`` ({name: function returning (timer fn, required time [s], [(condition name, fn)])}). Adapter names must
be unique over all tasks. This module holds no task knowledge.
"""

import importlib.util
import os
from dataclasses import dataclass
from typing import Callable


@dataclass
class Quantity:
    fn: Callable
    kind: str  # "scalar" or "flag"
    description: str
    latched: bool = False  # a flag that stays true for the rest of the episode once it became true


# flags that stay true once set (set on the adapter's quantities with mark_latched)
def mark_latched(quantities, *names):
    for name in names:
        if name in quantities:
            quantities[name].latched = True
    return quantities


def describe(quantities):
    """Plain-text table of the quantities for the LLM prompts."""
    lines = []
    for name, q in quantities.items():
        kind = "flag, latched: stays true for the rest of the episode once set" if q.latched else q.kind
        lines.append(f"- {name} ({kind}): {q.description}")
    return "\n".join(lines)


TASKS_DIR = os.environ.get("RD_TASKS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tasks")
_TABLES = None


def _tables():
    """(adapters, funnels, holds) merged over the quantities.py files of all task packages (loaded once)."""
    global _TABLES
    if _TABLES is None:
        tables = ({}, {}, {})
        names = sorted(os.listdir(TASKS_DIR)) if os.path.isdir(TASKS_DIR) else []
        for package in names:
            path = os.path.join(TASKS_DIR, package, "quantities.py")
            if not os.path.isfile(path):
                continue
            # loaded by path, so the task package's __init__ (gym registration, IsaacLab imports) does not run
            spec = importlib.util.spec_from_file_location(f"rd_task_quantities.{package}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for table, attr in zip(tables, ("ADAPTERS", "FUNNELS", "HOLDS")):
                for name, fn in getattr(module, attr, {}).items():
                    if name in table:
                        raise KeyError(f"Task adapter '{name}' is defined by more than one task package ({path})")
                    table[name] = fn
        _TABLES = tables
    return _TABLES


def get_adapter(name):
    adapters = _tables()[0]
    if name not in adapters:
        raise KeyError(f"Unknown task adapter '{name}'. Known adapters (from {TASKS_DIR}/*/quantities.py): "
                       f"{sorted(adapters)}")
    return adapters[name]()


def get_funnel(name):
    """[(condition name, fn(env) -> bool tensor)] of the task's success conditions in order, or []."""
    funnels = _tables()[1]
    return funnels[name]() if name in funnels else []


def get_hold(name):
    """(timer fn, required time, [(condition name, fn)]) of the task's success hold, or None."""
    holds = _tables()[2]
    return holds[name]() if name in holds else None
