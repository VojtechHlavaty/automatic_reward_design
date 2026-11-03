"""Signal Temporal Logic (STL) formulas and their quantitative robustness, batched over episodes in PyTorch.

A formula is a tree of the nodes below. ``robustness(phi, signals, dt)`` returns, for every episode and time step,
the robustness of ``phi`` evaluated from that step on, shape [episodes, steps]; the value at step 0 is the
robustness of the whole episode (> 0: the episode satisfies the formula, < 0: it does not, the size says by how
much).

Semantics (discrete time, bounded future, as in the usual STL robustness):

* ``Pred(signal, ">", c)`` scores ``x - c``, ``Pred(signal, "<", c)`` scores ``c - x``, divided by ``scale`` (so
  that predicates in different units can be compared under min and max);
* ``Not`` negates, ``And`` takes the minimum of its parts, ``Or`` the maximum;
* ``F(a, b, phi)`` ("eventually within [a, b] s") takes the maximum of ``phi`` over steps t + a .. t + b,
  ``G(a, b, phi)`` ("always within [a, b] s") the minimum.

The end of an episode: steps past the last valid step of an episode (``valid`` mask, e.g. after an early
termination) and steps past the end of the recording do not exist. ``F`` cannot be satisfied by them (they count as
-inf) and ``G`` is not violated by them (+inf), the convention of offline monitors for bounded-future operators
on finite traces. ``rtamt_spec(phi)`` writes the same formula in rtamt's syntax for cross-checking
(``check_rtamt.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Pred:
    signal: str  # name of a signal in the trajectory (see predicates.signals)
    op: str  # ">" or "<"
    c: float  # threshold, in the signal's units
    scale: float = 1.0  # robustness = margin / scale
    note: str = ""  # why this threshold (documentation only)


@dataclass(frozen=True)
class Not:
    phi: object


@dataclass(frozen=True)
class And:
    parts: tuple


@dataclass(frozen=True)
class Or:
    parts: tuple


@dataclass(frozen=True)
class F:
    a: float  # window start [s], relative to the time at which F is evaluated
    b: float  # window end [s]
    phi: object


@dataclass(frozen=True)
class G:
    a: float
    b: float
    phi: object


def all_of(*parts):
    return And(tuple(parts))


def any_of(*parts):
    return Or(tuple(parts))


def _steps(seconds: float, dt: float) -> int:
    return int(round(seconds / dt))


def _window(x: torch.Tensor, a: int, b: int, fill: float, reduce: str) -> torch.Tensor:
    """For every step t: reduce (max or min) of x over steps t + a .. t + b, missing steps filled with ``fill``."""
    _, steps = x.shape
    pad = torch.full((x.shape[0], b + 1), fill, dtype=x.dtype, device=x.device)
    padded = torch.cat((x, pad), dim=1)  # [E, steps + b + 1]
    windows = padded[:, a:].unfold(1, b - a + 1, 1)[:, :steps]  # [E, steps, b - a + 1]
    return windows.amax(dim=-1) if reduce == "max" else windows.amin(dim=-1)


def robustness(phi, signals: dict, dt: float, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Robustness trace of ``phi``: [episodes, steps]. ``signals``: {name: [episodes, steps] float tensor};
    ``valid``: [episodes, steps] bool, False after an episode ended (those steps do not exist)."""
    any_signal = next(iter(signals.values()))
    if valid is None:
        valid = torch.ones(any_signal.shape, dtype=torch.bool, device=any_signal.device)

    def rho(node) -> torch.Tensor:
        if isinstance(node, Pred):
            x = signals[node.signal].float()
            margin = x - node.c if node.op == ">" else node.c - x
            return margin / node.scale
        if isinstance(node, Not):
            return -rho(node.phi)
        if isinstance(node, And):
            return torch.stack([rho(p) for p in node.parts]).amin(dim=0)
        if isinstance(node, Or):
            return torch.stack([rho(p) for p in node.parts]).amax(dim=0)
        if isinstance(node, (F, G)):
            inner = rho(node.phi)
            fill = float("-inf") if isinstance(node, F) else float("inf")
            inner = torch.where(valid, inner, torch.full_like(inner, fill))  # steps that do not exist
            out = _window(inner, _steps(node.a, dt), _steps(node.b, dt), fill, "max" if isinstance(node, F) else "min")
            return out
        raise TypeError(f"not an STL node: {node!r}")

    return rho(phi)


def episode_robustness(phi, signals: dict, dt: float, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Robustness of each whole episode (the trace at step 0): [episodes]."""
    return robustness(phi, signals, dt, valid)[:, 0]


def explain(phi, signals: dict, dt: float, valid: torch.Tensor | None = None, episode: int = 0) -> list[str]:
    """Robustness of every sub-formula at step 0 of one episode, as indented lines (which part failed, by how much)."""
    lines = []

    def walk(node, depth):
        value = float(robustness(node, signals, dt, valid)[episode, 0])
        lines.append("  " * depth + f"{value:+.4f}  {describe(node, top=False)}")
        for child in children(node):
            walk(child, depth + 1)

    walk(phi, 0)
    return lines


def children(node) -> list:
    if isinstance(node, (And, Or)):
        return list(node.parts)
    if isinstance(node, (Not, F, G)):
        return [node.phi]
    return []


def predicates_of(phi) -> list[Pred]:
    """The predicates of a formula (each once)."""
    out = []

    def walk(node):
        if isinstance(node, Pred):
            if node not in out:
                out.append(node)
        for child in children(node):
            walk(child)

    walk(phi)
    return out


def describe(node, top: bool = True) -> str:
    """The formula in the usual notation (F[a,b], G[a,b], &, |, !)."""
    if isinstance(node, Pred):
        return f"{node.signal} {node.op} {node.c:g}"
    if isinstance(node, Not):
        return f"!({describe(node.phi, False)})"
    if isinstance(node, (And, Or)):
        joint = " & " if isinstance(node, And) else " | "
        return "(" + joint.join(describe(p, False) for p in node.parts) + ")"
    if isinstance(node, (F, G)):
        name = "F" if isinstance(node, F) else "G"
        return f"{name}[{node.a:g},{node.b:g}]({describe(node.phi, False)})"
    raise TypeError(node)


def rtamt_spec(node, dt: float) -> str:
    """The formula in rtamt's discrete-time syntax (time bounds in steps; predicate signals pre-scaled: each predicate
    is written over a variable ``<signal>__<index>`` that check_rtamt.py fills with the scaled margin's argument)."""
    preds = {}

    def walk(n):
        if isinstance(n, Pred):
            if n not in preds:
                preds[n] = f"p{len(preds)}"
            return f"({preds[n]} >= 0)"
        if isinstance(n, Not):
            return f"(not {walk(n.phi)})"
        if isinstance(n, (And, Or)):
            joint = " and " if isinstance(n, And) else " or "
            return "(" + joint.join(walk(p) for p in n.parts) + ")"
        if isinstance(n, (F, G)):
            name = "eventually" if isinstance(n, F) else "always"
            return f"({name}[{_steps(n.a, dt)}:{_steps(n.b, dt)}] {walk(n.phi)})"
        raise TypeError(n)

    return walk(node), preds
