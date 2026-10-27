"""Validation of stage YAMLs: structure (JSON schema) and semantics (references, types, math, budgets).

``validate_stage`` returns a list of human-readable errors; an empty list means the stage can be trained.
The errors are fed back to the stage generator, which retries.
"""

import math
import re

import jsonschema

from .derived import static_errors
from .quantities import Quantity
from .reward_registry import FUNCTIONS

NAME_PATTERN = r"^[a-z][a-z0-9_]{0,40}$"
MAX_TERMS = 24  # reward terms per stage (later stages keep paying the terms of the earlier stages)

STAGE_SCHEMA = {
    "type": "object",
    "required": ["stage", "reward", "training"],
    "additionalProperties": False,
    "properties": {
        "stage": {
            "type": "object",
            "required": ["name", "objective", "transition"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "pattern": NAME_PATTERN},
                "objective": {"type": "string", "minLength": 5},
                "transition": {
                    "type": "object",
                    "required": ["flag", "threshold"],
                    "additionalProperties": False,
                    "properties": {
                        "flag": {"type": "string"},
                        "threshold": {"type": "number"},
                    },
                },
            },
        },
        "reward": {
            "type": "object",
            "required": ["terms"],
            "additionalProperties": False,
            "properties": {
                "scale_by_dt": {"type": "boolean"},
                "terms": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_TERMS,
                    "items": {
                        "type": "object",
                        "required": ["name", "kind", "quantity", "weight"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string", "pattern": NAME_PATTERN},
                            "kind": {"enum": ["shaping", "penalty", "event", "potential"]},
                            "quantity": {"type": "string"},
                            "fn": {"type": "string"},
                            "params": {"type": "object", "additionalProperties": {"type": "number"}},
                            "weight": {"type": "number"},
                            "require": {"type": "array", "items": {"type": "string"}},
                            "forbid": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "training": {
            "type": "object",
            "required": ["max_iterations"],
            "additionalProperties": False,
            "properties": {
                "max_iterations": {"type": "integer"},
                "entropy_coef": {"type": "number"},
                "learning_rate": {"type": "number"},
                "gamma": {"type": "number"},
                "lam": {"type": "number"},
                "steps_per_env": {"enum": [24, 48, 96]},
                "reset_critic": {"type": "boolean"},
                "exploration": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "min_std": {"type": "number"},
                        "min_std_final": {"type": "number"},
                        "max_std": {"type": "number"},
                        "per_action": {"type": "object", "additionalProperties": {"type": "number"}},
                    },
                },
            },
        },
        "randomization": {"type": "object", "additionalProperties": {"type": "number"}},
        "derived": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["name", "kind", "description", "code"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "pattern": NAME_PATTERN},
                    "kind": {"enum": ["scalar", "flag"]},
                    "description": {"type": "string", "minLength": 5},
                    "range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "code": {"type": "string", "minLength": 10},
                },
            },
        },
    },
}


def with_derived(stage, quantities):
    """The registered quantities plus placeholders for the stage's derived quantities (for validation)."""
    combined = dict(quantities)
    for item in stage.get("derived") or []:
        if isinstance(item, dict) and item.get("name") and item.get("kind") in ("scalar", "flag"):
            combined.setdefault(item["name"], Quantity(None, item["kind"], item.get("description", "")))
    return combined


def derived_errors(stage, quantities):
    """Static errors of the stage's derived quantities (the live check runs separately)."""
    errors = []
    names = [item["name"] for item in stage.get("derived") or []]
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"derived quantity name '{name}' is used more than once")
    for item in stage.get("derived") or []:
        name = item["name"]
        if name in quantities:
            errors.append(f"derived quantity '{name}' has the name of a registered quantity; choose another name")
        if name.startswith(("metric_", "derived_")):
            errors.append(f"derived quantity names must not start with 'metric_' or 'derived_'")
        if "range" in item:
            if item["kind"] != "scalar":
                errors.append(f"derived quantity '{name}': only a scalar has a range")
            elif not (all(math.isfinite(v) for v in item["range"]) and item["range"][0] < item["range"][1]):
                errors.append(f"derived quantity '{name}': range must be [low, high] with low < high")
        errors += static_errors(item)
    return errors


def validate_stage(stage, quantities, limits, randomization_options):
    """Return a list of errors for a parsed stage dict.

    Args:
        stage: the parsed YAML.
        quantities: the task adapter's quantities ({name: Quantity}).
        limits: dict with the allowed ranges: min_iterations, max_iterations, entropy_coef (lo, hi),
            learning_rate (lo, hi), max_abs_weight.
        randomization_options: {key: {"min": .., "max": ..}} from the env config.
    """
    errors = []
    try:
        jsonschema.validate(stage, STAGE_SCHEMA)
    except jsonschema.ValidationError as e:
        location = "/".join(str(p) for p in e.absolute_path) or "(top level)"
        if e.validator == "maxItems" and isinstance(e.instance, list):
            return [f"schema error at {location}: {len(e.instance)} entries, at most {e.validator_value} are allowed; "
                    "merge or drop terms (keep the terms that pay earlier skills)"]
        message = e.message if len(e.message) < 400 else e.message[:200] + " ... " + e.message[-150:]
        return [f"schema error at {location}: {message}"]

    errors += derived_errors(stage, quantities)
    # stage transitions use registered flags only (they are logged for every stage); reward terms and their
    # gates may also use the stage's derived quantities
    flags = {n for n, q in quantities.items() if q.kind == "flag"}
    quantities = with_derived(stage, quantities)
    term_flags = {n for n, q in quantities.items() if q.kind == "flag"}

    # stage and transition
    transition = stage["stage"]["transition"]
    if transition["flag"] not in flags:
        errors.append(f"stage.transition.flag '{transition['flag']}' is not a flag quantity; use one of {sorted(flags)}")
    if not 0.0 < transition["threshold"] <= 1.0:
        errors.append("stage.transition.threshold must be in (0, 1] (a rate of episodes or steps)")

    # reward terms
    terms = stage["reward"]["terms"]
    names = [t["name"] for t in terms]
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"reward term name '{name}' is used more than once")
    if any(n.startswith(("metric_", "derived_")) for n in names):
        errors.append("reward term names must not start with 'metric_' or 'derived_' (reserved for logged values)")
    has_positive = False
    for t in terms:
        where = f"reward term '{t['name']}'"
        kind, quantity, weight = t["kind"], t["quantity"], t["weight"]
        if quantity not in quantities:
            errors.append(f"{where}: unknown quantity '{quantity}'; use one of {sorted(quantities)}")
            continue
        if not math.isfinite(weight) or weight == 0 or abs(weight) > limits["max_abs_weight"]:
            errors.append(f"{where}: weight must be finite, non-zero and at most {limits['max_abs_weight']} in magnitude")
        for gate_key in ("require", "forbid"):
            for flag in t.get(gate_key, []) or []:
                if flag not in term_flags:
                    errors.append(f"{where}: {gate_key} entry '{flag}' is not a flag quantity")
        if kind == "event":
            if quantities[quantity].kind != "flag":
                errors.append(f"{where}: an event needs a flag quantity, '{quantity}' is a scalar")
            if "fn" in t or "params" in t:
                errors.append(f"{where}: an event has no fn or params")
            if t.get("require") or t.get("forbid"):
                errors.append(f"{where}: an event cannot be gated")
            if weight <= 0:
                errors.append(f"{where}: an event bonus needs a positive weight")
            else:
                has_positive = True
            continue
        fn_name = t.get("fn")
        if fn_name not in FUNCTIONS:
            errors.append(f"{where}: unknown fn '{fn_name}'; use one of {sorted(FUNCTIONS)}")
            continue
        _, param_names, bounded, input_kind = FUNCTIONS[fn_name]
        if quantities[quantity].kind != input_kind:
            errors.append(f"{where}: fn '{fn_name}' needs a {input_kind} quantity, '{quantity}' is a {quantities[quantity].kind}")
        params = t.get("params", {}) or {}
        if set(params) != set(param_names):
            errors.append(f"{where}: fn '{fn_name}' needs exactly the params {param_names}, got {sorted(params)}")
        else:
            if any(not math.isfinite(v) for v in params.values()):
                errors.append(f"{where}: params must be finite numbers")
            if "scale" in params and params["scale"] <= 0:
                errors.append(f"{where}: scale must be positive")
            if "low" in params and "high" in params and params["high"] <= params["low"]:
                errors.append(f"{where}: high must be greater than low")
        if kind in ("shaping", "potential"):
            if not bounded:
                errors.append(f"{where}: a {kind} term needs a bounded fn (kernel, ramp or indicator), not '{fn_name}'")
            if weight <= 0:
                errors.append(f"{where}: a {kind} term needs a positive weight (use kind 'penalty' for negative weights)")
            else:
                has_positive = True
            if kind == "shaping" and weight > 0 and getattr(quantities[quantity], "latched", False):
                errors.append(
                    f"{where}: '{quantity}' is a latched flag (it stays true for the rest of the episode once set), so "
                    "per-second credit on it keeps paying after the step is over, wherever the policy goes next (e.g. it "
                    "passes the target once and then parks elsewhere). Pay reaching it once with an event term, and pay "
                    "per second only for states that hold now (it can still be used in require/forbid gates)")
        elif kind == "penalty" and weight >= 0:
            errors.append(f"{where}: a penalty needs a negative weight")
    if not has_positive:
        errors.append("the reward needs at least one shaping term or event with a positive weight")

    # training
    training = stage["training"]
    if not limits["min_iterations"] <= training["max_iterations"] <= limits["max_iterations"]:
        errors.append(
            f"training.max_iterations must be between {limits['min_iterations']} and {limits['max_iterations']} "
            "(the remaining curriculum budget)"
        )
    for key in ("entropy_coef", "learning_rate"):
        if key in training:
            lo, hi = limits[key]
            if not lo <= training[key] <= hi:
                errors.append(f"training.{key} must be between {lo} and {hi}")
    for key, (lo, hi) in (("gamma", (0.98, 0.998)), ("lam", (0.9, 0.99))):
        if key in training and not lo <= training[key] <= hi:
            errors.append(f"training.{key} must be between {lo} and {hi}")
    exploration = training.get("exploration") or {}
    for key in ("min_std", "min_std_final", "max_std"):
        if key in exploration and not 0.05 <= exploration[key] <= 1.0:
            errors.append(f"training.exploration.{key} must be between 0.05 and 1.0")
    action_names = list(limits.get("action_names") or [])
    for name, value in (exploration.get("per_action") or {}).items():
        if name not in action_names:
            errors.append(f"training.exploration.per_action: unknown action '{name}'; actions: {action_names}")
        elif not 0.05 <= value <= 1.0:
            errors.append(f"training.exploration.per_action.{name} must be between 0.05 and 1.0")

    # randomization
    for key, value in (stage.get("randomization") or {}).items():
        if key not in randomization_options:
            errors.append(f"randomization key '{key}' is not allowed; allowed: {sorted(randomization_options)}")
            continue
        lo, hi = randomization_options[key]["min"], randomization_options[key]["max"]
        if not lo <= value <= hi:
            errors.append(f"randomization.{key} must be between {lo} and {hi}")
    return errors


def valid_name(name):
    return re.match(NAME_PATTERN, name) is not None
