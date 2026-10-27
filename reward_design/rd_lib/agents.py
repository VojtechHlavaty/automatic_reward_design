"""The LLM agents of the pipeline.

* query agent: turns the task and the latest feedback into a retrieval query;
* selector agent: picks the most relevant earlier curriculum from the retrieved records (or none);
* planner agent: decides the curriculum stages (objectives, transition conditions, iteration budgets);
* stage generator: writes one stage YAML (reward terms, training settings, randomization), validated with retries;
* feedback agent: analyzes a trained curriculum from its training statistics, visual report and human feedback;
* reward-hacking reviewer: an independent check of every stage YAML before training (does the highest-reward
  behavior differ from the objective?); its major findings go back to the stage generator;
* reward-hacking auditor: after training, explains stages whose statistics show signs of hacking (reward up while
  task progress went down); its diagnosis goes to the feedback agent and the database.
"""

import json
import os
import re

import yaml

from review import fill, parse_json_object

from .derived import describe_derived
from .quantities import describe
from .reward_registry import describe_functions
from .schema import MAX_TERMS, validate_stage

PROMPT_NAMES = [
    "system", "query", "selector", "planner", "stage", "stage_retry", "stage_retry_patch", "feedback", "hacking_review",
    "hacking_audit",
]


def load_prompts(prompt_dir):
    prompts = {}
    for name in PROMPT_NAMES:
        with open(os.path.join(prompt_dir, f"{name}.txt")) as f:
            prompts[name] = f.read()
    return prompts


def ask_json(client, messages, retries=2):
    """Request a JSON answer; on a parse failure, ask again with a reminder. Returns (object or None, raw)."""
    response = ""
    for _ in range(retries + 1):
        response = client.sample(messages, 1)[0][0]
        obj = parse_json_object(response)
        if obj is not None:
            return obj, response
        messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Your answer did not contain a valid JSON object. Answer only with the JSON object."},
        ]
    return None, response


def record_summary(record, max_chars=1500):
    """Short text of a database record for the selector and planner prompts."""
    lines = [
        f"task: {record.get('task', '')[:300]}",
        f"curriculum: {' -> '.join(s.get('name', '?') for s in record.get('plan', {}).get('stages', []))}",
        f"outcome: {record.get('outcome_summary', '')}",
    ]
    if record.get("feedback"):
        lines.append(f"automatic feedback: {record['feedback']}")
    if record.get("human_feedback"):
        lines.append(f"human feedback: {record['human_feedback']}")
    return "\n".join(lines)[:max_chars]


def query_agent(client, prompts, task, history):
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(prompts["query"], task=task, history=history or "(none yet)")},
    ]
    obj, _ = ask_json(client, messages)
    if isinstance(obj, dict) and obj.get("query"):
        return str(obj["query"])
    return task


def selector_agent(client, prompts, task, hits):
    """Index (into ``hits``) of the most relevant record, or None."""
    if not hits:
        return None, "the database is empty"
    listing = "\n\n".join(
        f"[{k}] similarity {score:.3f}\n{record_summary(record)}" for k, (score, record) in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(prompts["selector"], task=task, records=listing)},
    ]
    obj, _ = ask_json(client, messages)
    if not isinstance(obj, dict):
        return None, "the selector answer could not be parsed"
    choice = obj.get("choice")
    reason = str(obj.get("reason", ""))
    if isinstance(choice, int) and 0 <= choice < len(hits):
        return choice, reason
    return None, reason


def planner_context(task, quantities, budget, limits, randomization_options, example_text, feedback_text):
    return dict(
        task=task,
        quantities=describe(quantities),
        budget=budget,
        min_iterations=limits["min_iterations"],
        randomization=randomization_text(randomization_options),
        example=example_text or "(no earlier curriculum was retrieved)",
        feedback=feedback_text or "(this is the first iteration)",
    )


def randomization_text(options):
    if not options:
        return "(none for this task)"
    return "\n".join(f"- {k}: between {v['min']} and {v['max']} ({v.get('description', '')})" for k, v in options.items())


def normalize_plan(obj, budget, min_iterations, flags, min_threshold=0.0):
    """Clean a planner answer: 1-5 stages, known transition flags, iterations summing to the budget. A plan with a
    transition threshold below ``min_threshold`` (the task's target rate) is not used (the planner is asked again)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("stages"), list):
        return None
    stages = []
    for item in obj["stages"][:5]:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = re.sub(r"[^a-z0-9_]", "_", str(item["name"]).lower())[:40].strip("_") or f"stage_{len(stages)}"
        if not name[0].isalpha():
            name = "s_" + name
        transition = item.get("transition") if isinstance(item.get("transition"), dict) else {}
        flag = transition.get("flag") if transition.get("flag") in flags else None
        try:
            threshold = float(transition.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        if threshold < min_threshold:
            return None
        try:
            iterations = int(item.get("iterations", 0))
        except (TypeError, ValueError):
            iterations = 0
        stages.append({
            "name": name,
            "objective": str(item.get("objective", "")),
            "transition": {"flag": flag or sorted(flags)[0], "threshold": min(max(threshold, 0.01), 1.0)},
            "iterations": max(iterations, min_iterations),
            "notes": str(item.get("notes", "")),
        })
    if not stages:
        return None
    # scale the iteration budgets so that they sum to the curriculum budget
    stages = stages[:max(budget // min_iterations, 1)]  # every stage needs at least min_iterations
    total = sum(s["iterations"] for s in stages)
    if total != budget:
        for s in stages:
            s["iterations"] = max(min_iterations, int(round(s["iterations"] * budget / total)))
        # move the rounding difference to the stages that can absorb it without going below the minimum
        difference = budget - sum(s["iterations"] for s in stages)
        for s in sorted(stages, key=lambda s: -s["iterations"]):
            if difference == 0:
                break
            change = max(difference, min_iterations - s["iterations"])
            s["iterations"] += change
            difference -= change
    return {"rationale": str(obj.get("rationale", "")), "stages": stages}


def planner_agent(client, prompts, context, num_plans, budget, min_iterations, flags, min_threshold=0.0):
    """``num_plans`` independently sampled curriculum plans (None for answers that cannot be used)."""
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(prompts["planner"], **{k: str(v) for k, v in context.items()})},
    ]
    responses = client.sample(messages, num_plans)[0]
    return [normalize_plan(parse_json_object(r), budget, min_iterations, flags, min_threshold) for r in responses], responses


def extract_yaml(response):
    blocks = re.findall(r"```(?:yaml|yml)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        text = blocks[0]
    else:
        # an unclosed block (or stray fences): drop the fence lines
        text = "\n".join(line for line in response.splitlines() if not line.strip().startswith("```"))
    try:
        return yaml.safe_load(text), text
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"


def _merge_named(base, items, remove=()):
    """Entries of ``base`` (dicts with a name) with ``items`` added or replacing the entry of the same name, and the
    names in ``remove`` dropped; the order of ``base`` is kept, new entries come last."""
    items = [i for i in (items or []) if isinstance(i, dict)]
    replaced = {i.get("name"): i for i in items}
    merged = [replaced.pop(b.get("name"), b) for b in (base or []) if b.get("name") not in set(remove or [])]
    return merged + [i for i in items if i.get("name") in replaced]


def expand_stage(stage, previous_stage):
    """A stage written with ``reward.inherit: true`` keeps the reward terms (and derived quantities) of the previous
    stage: its own terms are added, or replace a term of the same name, and ``reward.remove`` drops terms. Returns the
    complete stage (as trained, checked and shown to later stages)."""
    reward = stage.get("reward")
    if not isinstance(reward, dict) or not reward.get("inherit"):
        if isinstance(reward, dict):
            reward.pop("inherit", None)
            reward.pop("remove", None)
        return stage
    if not previous_stage:
        reward.pop("inherit", None)
        reward.pop("remove", None)
        return stage
    old = previous_stage.get("reward") or {}
    merged = {k: v for k, v in old.items() if k not in ("terms",)}
    merged.update({k: v for k, v in reward.items() if k not in ("inherit", "remove", "terms")})
    merged["terms"] = _merge_named(old.get("terms"), reward.get("terms"), reward.get("remove"))
    stage = dict(stage, reward=merged)
    if previous_stage.get("derived") or stage.get("derived"):
        derived = _merge_named(previous_stage.get("derived"), stage.get("derived"))
        own = {d.get("name") for d in stage.get("derived") or []}
        # inherited derived quantities are kept only while something still uses them (a term, a gate or another
        # derived quantity), so that the stage's derived list does not grow with every stage
        used = set()
        for term in merged["terms"]:
            used |= {term.get("quantity"), *(term.get("require") or []), *(term.get("forbid") or [])}
        changed = True
        while changed:
            changed = False
            for d in derived:
                if (d.get("name") in used or d.get("name") in own) and d.get("name") not in used | {None}:
                    used.add(d.get("name"))
                if d.get("name") in used:
                    for other in derived:
                        if other.get("name") not in used and f"'{other.get('name')}'" in str(d.get("code", "")) or \
                                other.get("name") not in used and f'"{other.get("name")}"' in str(d.get("code", "")):
                            used.add(other.get("name"))
                            changed = True
        stage["derived"] = [d for d in derived if d.get("name") in used or d.get("name") in own]
        if not stage["derived"]:
            stage.pop("derived")
    return stage


def apply_patch(current, patch):
    """A retry answer marked ``patch: true`` changes only what it lists: reward terms (added or replacing the term of
    the same name), ``reward.remove`` (terms dropped), derived quantities (by name) and any other top-level key
    (replaced as a whole). Returns the complete stage."""
    patch = dict(patch)
    patch.pop("patch", None)
    stage = {k: v for k, v in current.items()}
    reward_patch = patch.pop("reward", None)
    if isinstance(reward_patch, dict):
        reward = dict(stage.get("reward") or {})
        reward.update({k: v for k, v in reward_patch.items() if k not in ("terms", "remove", "inherit")})
        for key in [k for k, v in reward.items() if v is None]:
            reward.pop(key)  # a key set to null is removed (e.g. a setting written in the wrong place)
        reward["terms"] = _merge_named(reward.get("terms"), reward_patch.get("terms"), reward_patch.get("remove"))
        stage["reward"] = reward
    if "derived" in patch:
        stage["derived"] = _merge_named(stage.get("derived"), patch.pop("derived"))
    stage.update(patch)
    for key in [k for k, v in stage.items() if v is None]:
        stage.pop(key)  # a top-level key set to null is removed
    return stage


def measured_table_text(stats):
    """The checker's per-term reward per second at each task step (live simulator states), as compact text."""
    table = (stats or {}).get("terms_by_progress_level") or {}
    totals = (stats or {}).get("reward_by_progress_level") or {}
    if not table:
        return ""
    lines = []
    for level, row in table.items():
        total = (totals.get(level) or {}).get("reward_per_s")
        terms = ", ".join(f"{name} {value:+.3g}" for name, value in sorted(row.items(), key=lambda x: -abs(x[1])))
        lines.append(f"- {level}" + (f" (total {total:+.3g}/s)" if total is not None else "") + f": {terms or 'nothing'}")
    return "\n".join(lines)


def stage_changes(old, new):
    """What changed between two complete stage drafts: reward terms added, removed and changed (by name), and other
    changed top-level keys."""
    old_terms = {t.get("name"): t for t in (old.get("reward") or {}).get("terms") or []}
    new_terms = {t.get("name"): t for t in (new.get("reward") or {}).get("terms") or []}
    lines = []
    for name in new_terms.keys() - old_terms.keys():
        lines.append(f"- added: {json.dumps(new_terms[name])}")
    for name in old_terms.keys() - new_terms.keys():
        lines.append(f"- removed: {name}")
    for name in new_terms.keys() & old_terms.keys():
        if new_terms[name] != old_terms[name]:
            lines.append(f"- changed: {json.dumps(old_terms[name])} -> {json.dumps(new_terms[name])}")
    for key in sorted((old.keys() | new.keys()) - {"reward", "adapter", "derived_check", "hacking_review",
                                                    "ordering_warning"}):
        if old.get(key) != new.get(key):
            lines.append(f"- {key} changed: {json.dumps(new.get(key), default=str)}")
    return "\n".join(lines) or "- (nothing changed)"


def hacking_review_agent(client, prompts, context, plan, stage_index, previous_yamls, stage, quantities,
                         measured=None, previous_review=None, changes=None):
    """Independent review of a stage YAML for reward hacking (a fresh conversation, adversarial instructions).

    ``measured``: the checker's per-term reward per second at each task step (text), so the reviewer uses
    measurements instead of estimating them. ``previous_review`` and ``changes``: a re-review after a fix only checks
    whether the earlier findings are fixed and whether the changed terms add a new exploit.

    Returns {"severity": "none"|"minor"|"major", "highest_reward_behavior": str, "exploits": [...]}.
    """
    shown = {k: v for k, v in stage.items() if k not in ("adapter", "derived_check", "hacking_review")}
    # the stage's derived quantities go after the cache mark: the part before it is the same for every stage review
    quantity_text = describe(quantities)
    stage_derived = describe_derived(stage["derived"]) if stage.get("derived") else "(none)"
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(
            prompts["hacking_review"], task=context["task"], quantities=quantity_text, stage_derived=stage_derived,
            functions=describe_functions(), stage_index=stage_index + 1, num_stages=len(plan["stages"]),
            stage_plan=json.dumps(plan["stages"][stage_index], indent=2),
            previous=("\n\n".join(previous_yamls) if previous_yamls else "(this is the first stage)"),
            stage_yaml=yaml.safe_dump(shown, sort_keys=False),
        )},
    ]
    extra = []
    if measured:
        extra.append(
            "Measured on live simulator states of the current policy: this stage's reward per second at each task step "
            "the policy reaches (grouped by the furthest progress flag that holds), per term; shaping, penalty and "
            "potential terms, event bonuses excluded (they pay once, as given in the YAML). Use these numbers instead "
            "of computing per-second rewards yourself; estimate only the steps that are not listed (the policy does not "
            "reach them yet):\n" + measured)
    if previous_review and changes:
        findings = "; ".join(f"{e.get('description', '')} (terms: {', '.join(map(str, e.get('terms', [])))})"
                             for e in previous_review.get("exploits", [])) or "(none listed)"
        extra.append(
            "This is a re-review after a fix. The previous review of this stage found: "
            f"{previous_review.get('highest_reward_behavior', '')} Exploits: {findings}\n"
            f"Changes since then:\n{changes}\n"
            "Check only (1) whether each previous finding is fixed and (2) whether the changed terms create a new "
            "exploit; the unchanged terms were reviewed already, so look at them only as far as the changes need.")
    if extra:
        messages[-1]["content"] += "\n\n" + "\n\n".join(extra) + "\n\nAnswer only with the JSON object described above."
    obj, raw = ask_json(client, messages)
    if not isinstance(obj, dict):
        return {"severity": "none", "highest_reward_behavior": "", "exploits": [], "unparsed": raw[:2000]}
    severity = str(obj.get("severity", "none")).lower()
    exploits = obj.get("exploits") if isinstance(obj.get("exploits"), list) else []
    return {
        "severity": severity if severity in ("none", "minor", "major") else "minor",
        "highest_reward_behavior": str(obj.get("highest_reward_behavior", "")),
        "exploits": [e for e in exploits if isinstance(e, dict)],
    }


def hacking_audit_agent(client, prompts, task, stage_yaml, statistics, warnings):
    """Diagnosis of a trained stage whose statistics show signs of reward hacking."""
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(
            prompts["hacking_audit"], task=task, stage_yaml=stage_yaml, statistics=statistics,
            warnings=warnings or "(none)",
        )},
    ]
    obj, raw = ask_json(client, messages)
    if not isinstance(obj, dict):
        return {"hacking": None, "diagnosis": raw.strip()[:1500], "exploited_terms": [], "fix": ""}
    return {
        "hacking": bool(obj.get("hacking")),
        "diagnosis": str(obj.get("diagnosis", "")),
        "exploited_terms": [str(t) for t in obj.get("exploited_terms", []) if t] if isinstance(obj.get("exploited_terms"), list) else [],
        "fix": str(obj.get("fix", "")),
    }


FORMAT_ERRORS = ("YAML parse error", "the answer does not contain")  # schema errors go back to the stage writer


def stage_agent(client, prompts, context, plan, stage_index, previous_yamls, quantities, limits,
                randomization_options, adapter, max_retries=5, checker=None, hacking_review_rounds=0,
                review_client=None, light_client=None, progress_flags=None, review_measured=False,
                review_incremental=False):
    """Write and validate the YAML of one stage. Returns (stage dict or None, list of attempt logs).

    With a ``checker`` (:class:`rd_lib.derived_checker.DerivedChecker`), the stage's derived quantities are
    also run on live environment states; their errors and value statistics go back to the LLM. With
    ``hacking_review_rounds`` > 0, a valid YAML is also reviewed for reward hacking by an independent reviewer;
    major findings go back to the LLM (at most that many times; after that the YAML is accepted with the review).

    Token use: a retry sends only the original prompt, the last answer and its errors (not every earlier attempt);
    answers with only format errors are fixed by ``light_client`` (a cheaper model or no thinking), the review uses
    ``review_client`` (thinking), the stage itself ``client``.
    """
    review_client = review_client or client
    light_client = light_client or client
    reviews_left = hacking_review_rounds
    last_review = None
    stage_plan = plan["stages"][stage_index]
    stage_limits = dict(limits, min_iterations=stage_plan["iterations"], max_iterations=stage_plan["iterations"])
    user = fill(
        prompts["stage"],
        **{k: str(v) for k, v in context.items()},
        functions=describe_functions(),
        plan=json.dumps(plan, indent=2),
        stage_index=stage_index + 1,
        num_stages=len(plan["stages"]),
        stage_plan=json.dumps(stage_plan, indent=2),
        previous=("\n\n".join(previous_yamls) if previous_yamls else "(this is the first stage)"),
        iterations=stage_plan["iterations"],
        entropy_range=list(limits["entropy_coef"]),
        lr_range=list(limits["learning_rate"]),
        max_abs_weight=limits["max_abs_weight"],
        max_terms=MAX_TERMS,
    )
    base = [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]
    messages = base
    attempts = []
    writer = client
    previous_stage = None
    if previous_yamls:
        try:
            previous_stage = yaml.safe_load(previous_yamls[-1])
        except yaml.YAMLError:
            previous_stage = None
    current = None  # the last complete draft (a retry answer may be a patch of it)
    fallback = None  # the last draft whose only problems were reward-ordering findings (used if no attempt passes)
    reviewed_stage = None  # the draft of the last review (a re-review looks at the changes since)
    measured = ""
    for attempt in range(max_retries + 1):
        response = writer.sample(messages, 1)[0][0]
        stage, text = extract_yaml(response)
        if stage is None or not isinstance(stage, dict):
            errors = [text if text.startswith("YAML parse error") else "the answer does not contain a YAML mapping"]
        else:
            if stage.get("patch") and current is not None:
                stage = apply_patch(current, stage)
            stage.pop("patch", None)
            stage = expand_stage(stage, previous_stage)
            for key in ("adapter", "derived_check", "hacking_review", "ordering_warning"):
                stage.pop(key, None)
            current = stage
            errors = validate_stage(stage, quantities, stage_limits, randomization_options)
        stats = {}
        ordering_note = None
        if not errors and checker is not None:
            # derived quantities on live states and the ordering of the reward over the task's progress levels
            errors, stats = checker.check(stage, adapter, progress_flags, term_table=review_measured)
            measured = measured_table_text(stats) if review_measured else ""
            stats = {k: v for k, v in stats.items() if k != "terms_by_progress_level"}  # (for the reviewer only)
            # on the last attempt, a YAML whose only problem is the ordering is accepted with a note
            if errors and attempt == max_retries and all(e.startswith("reward ordering") for e in errors):
                ordering_note, errors = errors, []
        review = None
        if not errors and reviews_left > 0:
            reviews_left -= 1
            incremental = (review_incremental and last_review is not None and reviewed_stage is not None
                           and last_review.get("severity") == "major")
            review = hacking_review_agent(
                review_client, prompts, context, plan, stage_index, previous_yamls, stage, quantities,
                measured=measured if review_measured else None,
                previous_review=last_review if incremental else None,
                changes=stage_changes(reviewed_stage, stage) if incremental else None)
            last_review = review
            reviewed_stage = dict(stage)
            # on the last attempt a YAML that is otherwise valid is accepted with its review instead of failing
            if review["severity"] == "major" and attempt < max_retries:
                errors = [
                    "an independent reward-hacking review found that the highest-reward behavior is not the objective: "
                    + review["highest_reward_behavior"]
                ] + [
                    f"exploit ({', '.join(map(str, e.get('terms', [])))}): {e.get('description', '')} Suggested fix: "
                    f"{e.get('fix', '')}" for e in review["exploits"]
                ]
        attempts.append({"attempt": attempt, "errors": errors, "derived_stats": stats, "hacking_review": review})
        if errors and stage is not None and all(e.startswith("reward ordering") for e in errors):
            fallback = (dict(stage), list(errors), stats)
        if not errors:
            stage["adapter"] = adapter
            if stats:
                stage["derived_check"] = stats  # values on live states (also shown to later stages)
            if ordering_note:
                stage["ordering_warning"] = ordering_note
            if review is not None:
                stage["hacking_review"] = {"severity": review["severity"],
                                           "highest_reward_behavior": review["highest_reward_behavior"]}
            elif last_review is not None:
                stage["hacking_review"] = {"severity": "not reviewed again after the last fix",
                                           "previous_findings": last_review["highest_reward_behavior"]}
            return stage, attempts
        error_text = "\n".join(f"- {e}" for e in errors)
        if stats:
            error_text += "\nValues measured on live states: " + json.dumps(stats)
        # short retry: the original prompt, the last complete draft and its errors only; the answer can be a patch of
        # that draft (only the changed parts), unless the last answer could not be read
        repeated = len(attempts) >= 2 and attempts[-1]["errors"] == attempts[-2]["errors"]
        if current is not None and not repeated:
            draft = "```yaml\n" + yaml.safe_dump(current, sort_keys=False) + "```"
            retry = prompts.get("stage_retry_patch") or prompts["stage_retry"]
        elif current is not None:
            # the same errors twice in a row: a patch did not fix them, so ask for the complete corrected YAML
            draft = "```yaml\n" + yaml.safe_dump(current, sort_keys=False) + "```"
            retry = prompts["stage_retry"]
        else:
            draft, retry = response, prompts["stage_retry"]
        messages = base + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": fill(retry, errors=error_text)},
        ]
        writer = light_client if all(e.startswith(FORMAT_ERRORS) for e in errors) else client
    if fallback is not None:
        # no attempt passed, but an earlier draft failed only the reward-ordering findings: it is accepted with them,
        # as on the last attempt (a late patch that breaks the schema does not drop the whole stage)
        stage, ordering_note, stats = fallback
        stage["adapter"] = adapter
        if stats:
            stage["derived_check"] = stats
        stage["ordering_warning"] = ordering_note
        attempts.append({"attempt": "fallback", "errors": [], "derived_stats": stats, "hacking_review": None,
                         "note": "accepted the last draft whose only problems were reward-ordering findings"})
        return stage, attempts
    return None, attempts


def feedback_agent(client, prompts, task, curriculum_text, outcomes_text, visual_text, human_text):
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": fill(
            prompts["feedback"], task=task, curriculum=curriculum_text, outcomes=outcomes_text,
            visual=visual_text or "(no visual evaluation)", human=human_text or "(none)",
        )},
    ]
    obj, raw = ask_json(client, messages)
    if not isinstance(obj, dict):
        return {"analysis": raw.strip()[:3000], "recommendations": ""}
    return {"analysis": str(obj.get("analysis", "")), "recommendations": str(obj.get("recommendations", ""))}
