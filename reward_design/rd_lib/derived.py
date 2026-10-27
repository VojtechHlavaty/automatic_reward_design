"""Derived quantities: task quantities that the LLM defines in a stage YAML as a short Python function.

A stage YAML may contain::

    derived:
      - name: gripper_object_yaw_error
        kind: scalar            # or flag
        description: angle between the gripper closing line and the nearest face normal of a box-shaped object
        range: [0.0, 0.79]      # expected values (scalars only); checked on live environment states
        code: |
          def compute(env, q, h):
              ...
              return value      # (num_envs,) float tensor for a scalar, bool tensor for a flag

``q(name)`` returns a registered (or earlier derived) quantity, ``h`` is :mod:`rd_lib.helpers`. The code is
checked statically here (no imports, no file or attribute writes, no dunder access); the live check in the
simulator (:mod:`rd_lib.derived_checker`) runs it on real environment states.
"""

import ast
import math

import torch

FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "__import__", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys", "subprocess",
}
ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "range": range, "len": len, "float": float, "int": int, "bool": bool,
    "sum": sum, "zip": zip, "enumerate": enumerate, "list": list, "tuple": tuple, "dict": dict, "round": round,
    "isinstance": isinstance, "ValueError": ValueError, "KeyError": KeyError,
}


def static_errors(item):
    """Errors of one derived-quantity definition that can be found without running it."""
    name = item.get("name", "?")
    where = f"derived quantity '{name}'"
    code = item.get("code", "")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"{where}: syntax error in code, line {e.lineno}: {e.msg}"]
    errors = []
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute"]
    if len(functions) != 1:
        errors.append(f"{where}: code must define exactly one function 'def compute(env, q, h):'")
    elif [a.arg for a in functions[0].args.args] != ["env", "q", "h"]:
        errors.append(f"{where}: the function signature must be 'def compute(env, q, h):'")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Assign)) or (
            isinstance(node, ast.Assign) and not isinstance(node.value, ast.Constant)
        ):
            errors.append(f"{where}: only 'def compute' and constant assignments are allowed at the top level")
            break
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(f"{where}: imports are not allowed; torch, math and the helpers h are available")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"{where}: '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"{where}: dunder attributes ('{node.attr}') are not allowed")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, (ast.Attribute, ast.Subscript)):
                        errors.append(
                            f"{where}: assigning to attributes or items is not allowed (it could change the "
                            "environment); compute new tensors instead")
                        break
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.endswith("_"):
            errors.append(f"{where}: in-place tensor operations ('{node.func.attr}') are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef)):
            errors.append(f"{where}: {type(node).__name__} is not allowed")
    return sorted(set(errors))


def compile_derived(item, helpers_module):
    """The callable ``fn(env, q)`` of a statically valid derived-quantity definition."""
    namespace = {"__builtins__": ALLOWED_BUILTINS, "torch": torch, "math": math}
    exec(compile(item["code"], f"<derived {item['name']}>", "exec"), namespace)  # noqa: S102
    compute = namespace["compute"]

    def fn(env, q):
        return compute(env, q, helpers_module)

    return fn


def check_value(item, value, num_envs):
    """Error message for a wrongly typed or shaped value, or None."""
    where = f"derived quantity '{item['name']}'"
    if not isinstance(value, torch.Tensor):
        return f"{where}: compute must return a torch.Tensor, got {type(value).__name__}"
    if value.dim() == 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    if tuple(value.shape) != (num_envs,):
        return f"{where}: compute must return shape ({num_envs},) (one value per environment), got {tuple(value.shape)}"
    if item["kind"] == "flag" and value.dtype != torch.bool:
        return f"{where}: a flag must return a bool tensor (e.g. a comparison), got {value.dtype}"
    if item["kind"] == "scalar":
        if value.dtype == torch.bool or not value.is_floating_point():
            return f"{where}: a scalar must return a float tensor, got {value.dtype}"
        if not torch.isfinite(value).all():
            return f"{where}: the value contains NaN or Inf"
    return None


def describe_derived(items):
    """Text of the derived quantities of a stage (for later stages and the feedback agent)."""
    lines = []
    for item in items or []:
        unit = f" range {item['range']}" if item.get("range") else ""
        lines.append(f"- {item['name']} ({item['kind']}, derived{unit}): {item.get('description', '')}")
    return "\n".join(lines)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


HELPERS_DOC = """Helpers h.* (batched, first dimension = environments; quaternions (x, y, z, w) as in IsaacLab; positions
environment-local, i.e. world minus env.scene.env_origins):
- h.body_pos(env, body_name) -> (N, 3) position of a robot body; h.body_quat(env, body_name) -> (N, 4)
- h.root_pos(env, asset) / h.root_quat(env, asset) -> pose of another asset of the env (asset = its attribute name on env)
- h.rotate(quat, vec), h.rotate_inverse(quat, vec) -> (N, 3); h.axis(quat, "x"|"y"|"z") -> world direction of a local axis
- h.to_frame(points, frame_pos, frame_quat) -> points expressed in that frame
- h.yaw(quat) -> yaw [rad] about world z; h.tilt(quat) -> angle [rad] of the local z axis from vertical (0 = upright)
- h.heading(vec) -> yaw [rad] of the horizontal part of a vector; h.wrap_to_pi(angle)
- h.angle_diff_mod(a, b, period) -> smallest |a - b| modulo period, in [0, period/2] (period = pi/2 for the faces of a cube,
  pi for an undirected line such as a parallel gripper's closing direction)
- h.angle_between(u, v) -> [rad] in [0, pi]; h.distance(a, b) -> (N,); h.PI"""
