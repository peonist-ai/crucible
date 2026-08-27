"""BFCL multi-turn — stateful tool use across 4-6 turns.

The single cheapest thing here that touches error accumulation. ACBench found
4-bit compression costs 1-3% on single-call tool use but 10-15% on real
multi-step application accuracy; `bfcl_simple` measures the first band and
this measures the second, on the same dataset family and the same endpoint.

## How it is graded

State-based, which is why this needs an optional dependency. Each entry names
`involved_classes` (GorillaFileSystem, TradingBot, VehicleControlAPI, ...) and
an `initial_config` seeding them. We instantiate two independent sets:

  - the model's, mutated by whatever calls the model actually emits
  - ground truth's, mutated by executing the dataset's expected call sequence

After every turn we compare the public attributes of the two sets, and check
that ground truth's execution results all appear among the model's. A turn
passes when both hold; an entry passes when every turn does.

The API classes themselves come from `bfcl_eval` — 350KB of simulated file
systems, trading desks and vehicle controllers we would otherwise have to
vendor. Installed via the `bfcl` extra; missing, this raises rather than
scoring anything, on the same rule as IFEval's language checkers.

## Where this deviates, and which direction it errs

We drive generation ourselves rather than through `bfcl generate`, so the
result JSON carries `finish_reason` and per-turn divergence, which is the
whole reason to own the loop: "diverged at turn 4 of 6" is actionable in a way
that "62%" is not.

That also means these numbers are ours, not BFCL's. Do not put them beside a
leaderboard entry without reproducing through `bfcl evaluate` first. Model
calls are dispatched by name against the instances rather than `eval`'d, and
ground-truth call strings are parsed with `ast` instead of evaluated, so
neither path executes dataset text as code.
"""

from __future__ import annotations

import ast
import copy
import json
import urllib.request

from crucible_bench.testbench.api import truncate
from crucible_bench.testbench.tasks_tools import _bfcl_normalize_fn

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

CATEGORIES = ("base", "composite", "long_context", "miss_func", "miss_param")

# Verified against the dataset: every method named in an entry's `path` for a
# class resolves in that class's doc file. Derived rather than assumed, because
# the names do not match (TwitterAPI's tools live in posting_api.json).
CLASS_DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# A turn is done when the model stops calling tools. This caps a model that
# never stops -- without it one looping entry stalls the whole run.
MAX_STEPS_PER_TURN = 20


class MissingBFCLDependency(RuntimeError):
    """The simulated API classes are not installed.

    `uv pip install 'crucible-bench[bfcl]'`. Without them there is nothing to
    execute the calls against, so there is no state to compare and no score to
    report.
    """


def _load_api_classes(class_names):
    """Import and return {class_name: class}, from bfcl_eval."""
    try:
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING,
            STATELESS_CLASSES,
        )
    except ImportError as e:
        raise MissingBFCLDependency(
            "bfcl_eval is required to score BFCL multi-turn: the graded state "
            "lives in its simulated API classes. Install the extra: "
            "uv pip install 'crucible-bench[bfcl]'"
        ) from e

    import importlib

    classes = {}
    for name in class_names:
        if name not in CLASS_FILE_PATH_MAPPING:
            raise ValueError(
                f"bfcl_eval has no class {name!r}. The dataset and the "
                f"installed bfcl_eval disagree; do not score this run."
            )
        module = importlib.import_module(CLASS_FILE_PATH_MAPPING[name])
        classes[name] = (getattr(module, name), name in STATELESS_CLASSES)
    return classes


def _instantiate(class_names, initial_config, long_context=False):
    """Fresh, seeded instances. Two independent sets per entry, never shared."""
    classes = _load_api_classes(class_names)
    instances = {}
    for name, (cls, stateless) in classes.items():
        instance = cls()
        if not stateless:
            instance._load_scenario(
                copy.deepcopy(initial_config.get(name, {})), long_context=long_context
            )
        instances[name] = instance
    return instances


def _method_index(instances):
    """{method_name: instance}, so a call can be routed without eval."""
    index = {}
    for instance in instances.values():
        for attr in dir(instance):
            if attr.startswith("_"):
                continue
            if callable(getattr(instance, attr)):
                index.setdefault(attr, instance)
    return index


def _public_state(instances):
    """The comparable state: public attributes of every instance."""
    return {
        name: {
            k: v for k, v in vars(instance).items() if not k.startswith("_")
        }
        for name, instance in instances.items()
    }


def _parse_ground_truth_call(call: str):
    """`"mv(source='a', destination='b')"` -> ("mv", [...], {...}).

    Parsed, not evaluated. The reference implementation `eval`s these strings;
    there is no reason a benchmark harness needs to execute dataset text to
    read a function name and some literals out of it.
    """
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"ground-truth entry {call!r} is not a call expression")
    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return name, args, kwargs


def _invoke(index, name, args, kwargs):
    """Dispatch one call. Returns (result, error)."""
    target = index.get(name)
    if target is None:
        return None, f"no such function: {name}"
    try:
        return getattr(target, name)(*args, **kwargs), None
    except Exception as e:  # the simulated APIs raise on invalid arguments
        return None, f"{type(e).__name__}: {e}"


def _serialize(result):
    try:
        return json.dumps(result, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(result)


def _chat(url, messages, tools, max_tokens, temperature):
    """One assistant turn. Returns (message, meta)."""
    body = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    return choice["message"], {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
    }


def _run_model_turn(url, messages, tools, instances, index, max_tokens, temperature):
    """Drive one user turn to completion. Returns (calls, results, meta)."""
    calls, results = [], []
    meta = {"steps": 0, "finish_reason": None, "hit_step_cap": False}

    for _ in range(MAX_STEPS_PER_TURN):
        message, step_meta = _chat(url, messages, tools, max_tokens, temperature)
        meta["steps"] += 1
        meta["finish_reason"] = step_meta["finish_reason"]
        tool_calls = message.get("tool_calls") or []
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })
        if not tool_calls:
            return calls, results, meta

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments") or "{}"
            try:
                kwargs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                kwargs = {}
            result, error = _invoke(index, name, [], kwargs or {})
            calls.append({"name": name, "arguments": kwargs, "error": error})
            if error is None:
                results.append(_serialize(result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": error if error else _serialize(result),
            })
    else:
        # Never stopped calling tools. Recorded rather than scored as a plain
        # miss: a model stuck in a tool loop and a model that got the state
        # wrong are different failures with different fixes.
        meta["hit_step_cap"] = True
    return calls, results, meta


def _apply_ground_truth(index, turn_calls):
    """Execute one turn of expected calls. Returns the serialized results."""
    results = []
    for call in turn_calls:
        name, args, kwargs = _parse_ground_truth_call(call)
        result, error = _invoke(index, name, args, kwargs)
        if error is not None:
            raise ValueError(
                f"ground-truth call {call!r} failed against the installed "
                f"bfcl_eval: {error}. The dataset and the API classes are out "
                f"of step -- do not score this run."
            )
        results.append(_serialize(result))
    return results


def grade_entry(url, entry, ground_truth, tools, max_tokens, temperature):
    """Run one multi-turn entry and grade every turn. Returns a record.

    Two independent instance sets — the model mutates one, ground truth the
    other — so a model call can never accidentally satisfy the comparison by
    mutating the thing it is compared against.
    """
    involved = entry["involved_classes"]
    long_context = "long_context" in entry["id"]
    model_instances = _instantiate(involved, entry["initial_config"], long_context)
    gt_instances = _instantiate(involved, entry["initial_config"], long_context)
    model_index = _method_index(model_instances)
    gt_index = _method_index(gt_instances)

    messages: list[dict] = []
    turns = []
    model_results: list[str] = []
    first_divergence = None

    for turn_idx, user_turn in enumerate(entry["question"]):
        messages.extend(user_turn)
        calls, results, meta = _run_model_turn(
            url, messages, tools, model_instances, model_index,
            max_tokens, temperature,
        )
        model_results.extend(results)

        gt_turn = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
        gt_results = _apply_ground_truth(gt_index, gt_turn)

        state_ok = _public_state(model_instances) == _public_state(gt_instances)
        # Unordered subset: several calls in a turn may run in any order.
        missing = [r for r in gt_results if r not in model_results]
        response_ok = not missing
        passed = state_ok and response_ok
        if not passed and first_divergence is None:
            first_divergence = turn_idx

        turns.append({
            "turn": turn_idx,
            "passed": passed,
            "state_match": state_ok,
            "response_match": response_ok,
            "model_calls": [c["name"] for c in calls],
            "expected_calls": gt_turn,
            "call_errors": [c["error"] for c in calls if c["error"]],
            "steps": meta["steps"],
            "hit_step_cap": meta["hit_step_cap"],
            "finish_reason": meta["finish_reason"],
        })

    return {
        "id": entry["id"],
        "passed": all(t["passed"] for t in turns),
        "n_turns": len(turns),
        "turns_passed": sum(1 for t in turns if t["passed"]),
        # Where it fell apart, which is the number worth having: an entry that
        # dies at turn 1 and one that dies at turn 5 are different diagnoses,
        # and a marginal pass rate cannot tell them apart.
        "first_divergence": first_divergence,
        "involved_classes": involved,
        "turn_detail": turns,
        "finish_reason": turns[-1]["finish_reason"] if turns else None,
    }


def _load_category(category):
    from huggingface_hub import hf_hub_download

    q = hf_hub_download(
        BFCL_REPO, f"BFCL_v3_multi_turn_{category}.json", repo_type="dataset"
    )
    a = hf_hub_download(
        BFCL_REPO,
        f"possible_answer/BFCL_v3_multi_turn_{category}.json",
        repo_type="dataset",
    )
    entries = [json.loads(line) for line in open(q) if line.strip()]
    answers = {
        x["id"]: x["ground_truth"]
        for x in (json.loads(line) for line in open(a) if line.strip())
    }
    missing = [e["id"] for e in entries if e["id"] not in answers]
    if missing:
        raise RuntimeError(
            f"{len(missing)} multi-turn entries have no ground truth "
            f"(first few: {missing[:5]}). Each would score as a model failure "
            f"it did not earn."
        )
    return entries, answers


def _tools_for(class_names):
    """Assemble the tool schemas the model sees, from the involved classes."""
    from huggingface_hub import hf_hub_download

    tools = []
    for name in class_names:
        if name not in CLASS_DOC_FILE:
            raise ValueError(
                f"no tool-schema file mapped for class {name!r}; the dataset "
                f"has grown a class this task does not know how to describe."
            )
        path = hf_hub_download(
            BFCL_REPO,
            f"multi_turn_func_doc/{CLASS_DOC_FILE[name]}.json",
            repo_type="dataset",
        )
        for line in open(path):
            if line.strip():
                tools.append({
                    "type": "function",
                    "function": _bfcl_normalize_fn(json.loads(line)),
                })
    return tools


def _make_runner(category):
    def run(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
        entries, answers = _load_category(category)
        if limit:
            entries = entries[: min(limit, len(entries))]

        # Fail before generating if the API classes are missing, rather than
        # an hour in with nothing to show for it.
        _load_api_classes({c for e in entries for c in e["involved_classes"]})

        results = []
        for i, entry in enumerate(entries):
            print(f"  [{i+1}/{len(entries)}] {entry['id']}...", end=" ", flush=True)
            try:
                record = grade_entry(
                    url, entry, answers[entry["id"]],
                    _tools_for(entry["involved_classes"]),
                    max_tokens, temperature,
                )
            except MissingBFCLDependency:
                raise
            except Exception as e:
                print(f"ERROR ({e})")
                results.append({
                    "id": entry["id"], "passed": False,
                    "error": truncate(str(e), 300),
                })
                if checkpoint:
                    checkpoint(results)
                continue
            if record["passed"]:
                print("PASS")
            else:
                print(f"FAIL (turn {record['first_divergence']}"
                      f"/{record['n_turns']})")
            results.append(record)
            if checkpoint:
                checkpoint(results)

        _print_divergence_summary(results)
        return results

    run.__name__ = f"run_bfcl_multi_turn_{category}"
    run.__doc__ = (
        f"BFCL multi-turn / {category} — 200 entries, state-graded across "
        f"4-6 turns."
    )
    return run


def _print_divergence_summary(results):
    """Where entries die, not just how many.

    The distribution is the diagnosis. Compression that costs a couple of
    points of first-turn accuracy looks nothing like compression that holds
    turn 1 and falls apart by turn 4, and only one of those is error
    accumulation.
    """
    graded = [r for r in results if "first_divergence" in r]
    if not graded:
        return
    failed = [r for r in graded if not r["passed"]]
    print(f"\n  {len(graded) - len(failed)}/{len(graded)} entries fully correct")
    if not failed:
        return
    buckets: dict[int, int] = {}
    for r in failed:
        buckets[r["first_divergence"]] = buckets.get(r["first_divergence"], 0) + 1
    print("  First divergence by turn:")
    for turn in sorted(buckets):
        print(f"    turn {turn}: {buckets[turn]} entries")
    capped = sum(1 for r in graded
                 if any(t["hit_step_cap"] for t in r["turn_detail"]))
    if capped:
        print(f"  NOTE: {capped}/{len(graded)} entries hit the "
              f"{MAX_STEPS_PER_TURN}-step cap — a tool loop, not a wrong answer.")


run_bfcl_multi_turn_base = _make_runner("base")
run_bfcl_multi_turn_composite = _make_runner("composite")
run_bfcl_multi_turn_long_context = _make_runner("long_context")
run_bfcl_multi_turn_miss_func = _make_runner("miss_func")
run_bfcl_multi_turn_miss_param = _make_runner("miss_param")
