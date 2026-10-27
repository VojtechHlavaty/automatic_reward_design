"""Reward-term function registry and the compiler from a stage YAML to ``compute_reward(env)``.

A reward is a list of terms. Each term has a ``kind``:

* ``shaping``: a bounded transform (in [0, 1]) of one quantity, with a positive weight.
* ``penalty``: a transform of one quantity with a negative weight.
* ``event``: a one-time bonus (positive weight) in the step in which a flag becomes true.
* ``potential``: potential-based shaping, weight * (gamma * phi(s_t+1) - phi(s_t)) with phi a bounded transform of
  one quantity (optionally gated): it pays for progress along phi (a full traversal of phi from 0 to 1 earns about
  ``weight`` in total) and takes back what is undone, so neither standing still nor oscillating pays, and it does
  not change which behavior is optimal (it is not scaled by the time step).

``shaping`` and ``penalty`` terms can be gated: ``require`` lists flags that must be true, ``forbid`` lists flags
that must be false. Continuous terms are multiplied by the control time step when ``scale_by_dt`` is true, so
their sum over an episode does not depend on the episode length; events are not scaled.

A stage may also define derived quantities (``derived``: short Python functions written by the LLM, see
:mod:`rd_lib.derived`); terms use them like registered quantities.

The compiled function also returns the rate of every flag of the task adapter as ``metric_<flag>`` components
and the mean of every derived quantity as ``derived_<name>`` components, so the training logs show how often each
task condition holds and what the derived quantities measure.
"""

import math
import os

import torch
import yaml

from .quantities import get_adapter

# name: (function, parameter names, output bounded in [0, 1] for inputs of the expected kind, input kind)
FUNCTIONS = {
    "identity": (lambda x: x, [], False, "scalar"),
    "square": (lambda x: torch.square(x), [], False, "scalar"),
    "exp_kernel": (lambda x, scale: torch.exp(-x / scale), ["scale"], True, "scalar"),
    "gaussian_kernel": (lambda x, scale: torch.exp(-torch.square(x / scale)), ["scale"], True, "scalar"),
    "tanh_kernel": (lambda x, scale: 1.0 - torch.tanh(x / scale), ["scale"], True, "scalar"),
    "linear_ramp": (lambda x, low, high: ((x - low) / (high - low)).clamp(0.0, 1.0), ["low", "high"], True, "scalar"),
    "inverse_ramp": (lambda x, low, high: 1.0 - ((x - low) / (high - low)).clamp(0.0, 1.0), ["low", "high"], True, "scalar"),
    "indicator": (lambda x: x.float(), [], True, "flag"),
}

FUNCTION_DOCS = {
    "identity": "x (unbounded; penalties only)",
    "square": "x^2 (unbounded; penalties only)",
    "exp_kernel": "exp(-x / scale), 1 at x = 0, for distances and errors (x >= 0)",
    "gaussian_kernel": "exp(-(x / scale)^2), 1 at x = 0",
    "tanh_kernel": "1 - tanh(x / scale), 1 at x = 0",
    "linear_ramp": "clamp((x - low) / (high - low), 0, 1): 0 below low, 1 above high (e.g. a height to reach)",
    "inverse_ramp": "1 - linear_ramp(x, low, high): 1 below low, 0 above high",
    "indicator": "1 when a flag is true, else 0",
}


def describe_functions():
    return "\n".join(f"- {name}({', '.join(['x'] + FUNCTIONS[name][1])}): {doc}" for name, doc in FUNCTION_DOCS.items())


def load_stage(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_reward_fn(stage_path):
    """Compile a validated stage YAML into ``compute_reward(env) -> (reward, components)``."""
    return build_reward_fn_from_spec(load_stage(stage_path))


def build_reward_fn_from_spec(spec):
    """Compile a validated stage (dict with an ``adapter`` entry) into ``compute_reward(env)``."""
    if "adapter" not in spec:
        raise KeyError("The stage YAML has no 'adapter' entry (a task adapter from tasks/<package>/quantities.py, "
                       "the 'adapter' of the task configuration). The pipeline adds it when it saves a stage; "
                       "add it to stage YAMLs written by hand.")
    quantities = get_adapter(spec["adapter"])
    derived = spec.get("derived") or []
    if derived:
        from . import helpers  # needs IsaacLab: imported only when a stage defines derived quantities
        from .derived import compile_derived
        from .quantities import Quantity

        for item in derived:
            quantities[item["name"]] = Quantity(compile_derived(item, helpers), item["kind"], item["description"])
    reward = spec["reward"]
    terms = reward["terms"]
    scale_by_dt = bool(reward.get("scale_by_dt", True))
    derived_names = {item["name"] for item in derived}
    previous_flags = {}
    previous_potentials = {}
    gamma = float(os.environ.get("RD_GAMMA", "0.99"))

    def compute_reward(env):
        cache = {}

        def value_of(name):
            if name not in cache:
                q = quantities[name]
                value = q.fn(env, value_of) if name in derived_names else q.fn(env)
                if name in derived_names and value.dim() == 2 and value.shape[1] == 1:
                    value = value.squeeze(1)
                cache[name] = value
            return cache[name]

        dt = env.step_dt if scale_by_dt else 1.0
        total = torch.zeros(env.num_envs, device=env.device)
        components = {}
        for term in terms:
            if term["kind"] == "event":
                flag = value_of(term["quantity"]).bool()
                previous = previous_flags.get(term["name"])
                if previous is None or previous.shape != flag.shape:
                    previous = torch.zeros_like(flag)
                previous_flags[term["name"]] = flag.clone()
                contribution = float(term["weight"]) * (flag & ~previous).float()
            elif term["kind"] == "potential":
                fn, param_names, _, input_kind = FUNCTIONS[term["fn"]]
                x = value_of(term["quantity"])
                x = x.bool() if input_kind == "flag" else x.float()
                params = {k: float(term.get("params", {})[k]) for k in param_names}
                phi = fn(x, **params).float()
                gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                for flag_name in term.get("require", []) or []:
                    gate = gate & value_of(flag_name).bool()
                for flag_name in term.get("forbid", []) or []:
                    gate = gate & ~value_of(flag_name).bool()
                phi = phi * gate.float()
                previous = previous_potentials.get(term["name"])
                # the first step of an episode has no previous state: no shaping in that step
                start = getattr(env, "episode_length_buf", None)
                if previous is None or previous.shape != phi.shape:
                    previous = phi.clone()
                if start is not None:
                    previous = torch.where(start <= 1, phi, previous)
                previous_potentials[term["name"]] = phi.clone()
                contribution = float(term["weight"]) * (gamma * phi - previous)
                total = total + contribution
                components[term["name"]] = contribution
                if float(term["weight"]) > 0:
                    # the potential's level (0..1) and where its gates are open, as for shaping terms
                    components["fill_" + term["name"]] = phi
                    components["gate_" + term["name"]] = gate.float()
                continue
            else:
                fn, param_names, _, input_kind = FUNCTIONS[term["fn"]]
                x = value_of(term["quantity"])
                x = x.bool() if input_kind == "flag" else x.float()
                params = {k: float(term.get("params", {})[k]) for k in param_names}
                contribution = float(term["weight"]) * fn(x, **params).float() * dt
            gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            for flag_name in term.get("require", []) or []:
                gate = gate & value_of(flag_name).bool()
            for flag_name in term.get("forbid", []) or []:
                gate = gate & ~value_of(flag_name).bool()
            contribution = contribution * gate.float()
            total = total + contribution
            components[term["name"]] = contribution
            if term["kind"] == "shaping" and float(term["weight"]) > 0:
                # share of the term's maximum that the policy earns (0..1) over all steps, and the share of steps in
                # which its gates are open: fill / gate is the term's level where it applies (phase-conditioned)
                components["fill_" + term["name"]] = contribution / (float(term["weight"]) * dt)
                components["gate_" + term["name"]] = gate.float()
        for name, quantity in quantities.items():
            if name in derived_names:
                components["derived_" + name] = value_of(name).float()
            elif quantity.kind == "flag":
                components["metric_" + name] = value_of(name).float()
        return total, components

    return compute_reward


def write_reward_file(stage_path, reward_path, root_dir):
    """Write the Python reward file that the IsaacLab backend loads; it compiles the stage YAML at import."""
    code = (
        '"""Reward compiled from a stage YAML (do not edit; edit the YAML)."""\n'
        "import sys\n"
        f"sys.path.insert(0, {root_dir!r})\n"
        "from rd_lib.reward_registry import build_reward_fn\n"
        "\n"
        f"compute_reward = build_reward_fn({stage_path!r})\n"
    )
    with open(reward_path, "w") as f:
        f.write(code)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
