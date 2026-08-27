"""BFCL — structured function calling.

Tool calling is the capability we most need to survive compression, so this
is graded strictly: BFCL's own AST match, not a "did it mention the function"
heuristic. The parsing here is deliberately forgiving about *transport*
(native tool_calls, JSON in content, <tool_call> tags) and strict about
*content* (function name, argument values).
"""

from __future__ import annotations

import json
import re
import urllib.request

from crucible_bench.testbench.api import strip_thinking, truncate


def run_bfcl_simple(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    """BFCL-simple: single-function calling, 400 test cases.

    Each test gives the model one tool and a user message. Model must emit
    a function call with the right name and arguments. Grading uses BFCL's
    AST-match logic: function name exact-match, and each arg's value must
    be in the ground-truth accepted-values list (with type coercion).

    Loads directly from gorilla-llm/Berkeley-Function-Calling-Leaderboard
    without pulling the evalplus or BFCL packages as dependencies.
    """
    from huggingface_hub import hf_hub_download

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    q_path = hf_hub_download(repo, "BFCL_v3_simple.json", repo_type="dataset")
    a_path = hf_hub_download(
        repo, "possible_answer/BFCL_v3_simple.json", repo_type="dataset"
    )

    with open(q_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(a_path) as f:
        answers_by_id = {
            a["id"]: a["ground_truth"]
            for a in (json.loads(line) for line in f if line.strip())
        }

    if limit:
        questions = questions[: min(limit, len(questions))]

    _bfcl_preflight(questions, answers_by_id)

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        messages = q["question"][0]  # single-turn; outer list allows multi-turn
        # BFCL uses {"type": "dict"} in parameter schemas which isn't valid
        # JSON Schema — vLLM/OpenAI expect "object". Normalize recursively.
        tools = [
            {"type": "function", "function": _bfcl_normalize_fn(fn)}
            for fn in q["function"]
        ]
        fn_name = q["function"][0]["name"]

        print(f"  [{i+1}/{len(questions)}] {qid}...", end=" ", flush=True)

        try:
            call, raw, meta = _bfcl_chat_with_tools(
                url, messages, tools, max_tokens, temperature
            )
        except Exception as e:
            print(f"ERROR ({e})")
            results.append({
                "id": qid, "passed": False, "error": str(e)[:200],
                "expected_fn": fn_name,
            })
            if checkpoint:
                checkpoint(results)
            continue

        passed, reason = _bfcl_verdict(call, answers_by_id[qid], meta)

        print("PASS" if passed else f"FAIL ({reason})")
        results.append({
            "id": qid,
            "passed": passed,
            "expected_fn": fn_name,
            "actual_call": call,
            "raw_response": truncate(raw, 1000),
            "reason": reason if not passed else None,
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def _bfcl_verdict(call: dict | None, expected: list, meta: dict) -> tuple[bool, str]:
    """The whole grading decision for one case, in one place.

    Separate from `_bfcl_match` because the response shape matters as much as
    the call's contents: BFCL-simple hands the model one tool and expects one
    call, and grading `tool_calls[0]` alone made a scattershot answer -- the
    right call plus two invented ones -- score identically to a clean one.
    """
    num_calls = meta.get("num_tool_calls", 0)
    if num_calls > 1:
        return False, f"multiple_calls ({num_calls})"
    return _bfcl_match(call, expected)


def _bfcl_preflight(questions: list[dict], answers_by_id: dict) -> None:
    """Check every question has usable ground truth before generating anything.

    All three of these used to be silent: a missing answer made `expected` an
    empty list (nothing to match, scored a plain failure), and a malformed
    accepted-values list made `_bfcl_match` skip that parameter entirely. Both
    are data problems wearing a model problem's clothes, and both are free to
    catch here -- the answer file is already on disk.
    """
    missing = [q["id"] for q in questions if q["id"] not in answers_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} BFCL question(s) have no ground-truth answer "
            f"(first few: {missing[:5]}). Every one would be scored as a model "
            f"failure it did not earn."
        )
    for q in questions:
        for entry in answers_by_id[q["id"]]:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise RuntimeError(
                    f"BFCL ground truth for {q['id']} is shaped "
                    f"{entry!r}, expected a single-key {{fn: {{arg: [...]}}}}."
                )
            for arg_name, accepted in (next(iter(entry.values())) or {}).items():
                if not isinstance(accepted, list) or not accepted:
                    raise RuntimeError(
                        f"BFCL ground truth for {q['id']} parameter "
                        f"{arg_name!r} is {accepted!r}, not a non-empty list."
                    )


_BFCL_TYPE_REMAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _bfcl_normalize_fn(fn: dict) -> dict:
    """Recursively rewrite Python-style type names inside a BFCL parameter
    schema so it's valid JSON Schema. BFCL uses `dict`/`float`/`tuple`/`any`
    where standard JSON Schema expects `object`/`number`/`array`/`string`.
    vLLM tolerates the BFCL spellings; llama.cpp's `--jinja` tool-schema
    validator rejects them with HTTP 400 ("Unrecognized schema").
    """
    if isinstance(fn, dict):
        out = {}
        for k, v in fn.items():
            if k == "type" and isinstance(v, str) and v in _BFCL_TYPE_REMAP:
                out[k] = _BFCL_TYPE_REMAP[v]
            else:
                out[k] = _bfcl_normalize_fn(v)
        return out
    if isinstance(fn, list):
        return [_bfcl_normalize_fn(x) for x in fn]
    return fn


def _bfcl_chat_with_tools(
    url: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[dict | None, str, dict]:
    """Send chat completion with tools, return (parsed_call, raw_content, meta).

    parsed_call is {"name": ..., "arguments": {...}} or None if no call found.
    raw_content is the original content + repr of any tool_calls — useful for
    debugging failures.

    meta carries what the response says about *why* generation stopped. An
    unterminated tool-call block parses to no call and no content, so a
    "model emitted nothing" failure and a "model ran out of budget" failure look
    identical in raw_content — `finish_reason` is the only thing that separates
    them, and reasoning_content is invisible to the fallback parser. Recording
    all three cost one session to learn.

    Tries native `tool_calls` response first (OpenAI-compatible API). Falls back
    to parsing the content text for JSON-style function calls (for servers that
    don't produce tool_calls cleanly).
    """
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    choice = data["choices"][0]
    msg = choice["message"]
    raw_content = msg.get("content") or ""
    meta = {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
        # Servers running a reasoning parser put thinking here, where neither
        # raw_content nor strip_thinking() can see it.
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
    }
    tcs = msg.get("tool_calls") or []
    meta["num_tool_calls"] = len(tcs)
    if tcs:
        tc = tcs[0]
        name = tc.get("function", {}).get("name")
        args_raw = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        # Include tool_calls in the raw for debugging
        raw = f"[tool_calls]: {json.dumps(tcs)}\n[content]: {raw_content}"
        return {"name": name, "arguments": args}, raw, meta

    # Fallback: look for JSON function call in content
    stripped = strip_thinking(raw_content)
    return _bfcl_parse_content_call(stripped), raw_content, meta


def _bfcl_parse_content_call(text: str) -> dict | None:
    """Extract a function call from raw content when tool_calls isn't used.

    Handles several common shapes models emit:
      - ```json\\n{"name": "...", "arguments": {...}}\\n```
      - {"name": "...", "arguments": {...}}
      - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Strip tool-call tags
    text = re.sub(r"</?tool_call>", "", text)
    text = text.strip()

    # Try parsing whole thing as JSON
    for candidate in (text, _first_json_object(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "name" in obj and "arguments" in obj:
                args = obj["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return {"name": obj["name"], "arguments": args or {}}
            # Some models emit {fn_name: {arg: val, ...}}
            if len(obj) == 1:
                name = next(iter(obj))
                args = obj[name]
                if isinstance(args, dict):
                    return {"name": name, "arguments": args}
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict) and "name" in first:
                return {"name": first["name"], "arguments": first.get("arguments", {}) or {}}
    return None


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} JSON object from text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _bfcl_match(actual: dict | None, expected: list) -> tuple[bool, str]:
    """BFCL AST-match for the simple category.

    expected is a list of acceptable answers; each is
      {fn_name: {arg_name: [acceptable_value, ...], ...}}

    The model's call passes if ANY expected answer matches: same fn name,
    every required arg (from expected) is present with a value in its
    accepted list.
    """
    if actual is None:
        return False, "no_call"
    actual_name = actual.get("name")
    actual_args = actual.get("arguments") or {}

    for entry in expected:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        exp_name = next(iter(entry))
        if exp_name != actual_name:
            continue
        exp_args = entry[exp_name] or {}

        # A parameter the ground truth has never heard of is a hallucinated
        # argument, and upstream BFCL fails it. Checking only the expected
        # keys -- which is what this did -- scores an invented `units="metric"`
        # as a clean call.
        unexpected = sorted(set(actual_args) - set(exp_args))
        if unexpected:
            return False, f"unexpected_args={unexpected}"

        all_ok = True
        for arg_name, accepted in exp_args.items():
            if not isinstance(accepted, list) or not accepted:
                raise ValueError(
                    f"BFCL ground truth for {exp_name}.{arg_name} is "
                    f"{accepted!r}, not a non-empty list of accepted values. "
                    f"Skipping it would pass any value the model invented."
                )
            if arg_name not in actual_args:
                # Allowed to omit arg only if "" (empty) is an accepted value
                if "" in accepted or None in accepted:
                    continue
                all_ok = False
                break
            if not _bfcl_value_match(actual_args[arg_name], accepted):
                all_ok = False
                break
        if all_ok:
            return True, ""
    return False, f"name={actual_name} args={list(actual_args.keys())}"


def _bfcl_value_match(actual, accepted: list) -> bool:
    """Check if actual value is in the accepted list, with loose type match."""
    return any(_bfcl_one_value_match(actual, acc) for acc in accepted)


def _bfcl_one_value_match(actual, acc) -> bool:
    """Match one candidate. Recurses into lists and dicts.

    Ground truth nests: a `conditions` parameter's accepted value can be a
    list of dicts whose own leaves are accepted-value lists, e.g.

        [{"field": ["age"], "operation": [">"], "value": ["25"]}]

    Comparing that to the model's plain `[{"field": "age", ...}]` by equality
    fails every time. Measured 2026-08-22 against halogen-qwen3.8-27b: all 3
    of 100 cases shaped like this failed while the other 97 passed 87 — the
    model's calls were correct and the checker could not see it. Deflation is
    as wrong as inflation; it just flatters nobody.
    """
    if actual == acc:
        return True

    # Nested accepted structure: {param: [accepted, ...], ...}
    if isinstance(acc, dict) and isinstance(actual, dict):
        for key, sub_accepted in acc.items():
            if key not in actual:
                # Absent is fine only where the ground truth accepts empty.
                if isinstance(sub_accepted, list) and (
                    "" in sub_accepted or None in sub_accepted
                ):
                    continue
                return False
            candidates = sub_accepted if isinstance(sub_accepted, list) else [sub_accepted]
            if not _bfcl_value_match(actual[key], candidates):
                return False
        # Keys the ground truth never mentions are hallucinated, same rule as
        # the top-level argument check.
        return not (set(actual) - set(acc))

    # Element-wise for lists, so lists of dicts recurse into the branch above.
    if isinstance(actual, list) and isinstance(acc, list):
        return len(actual) == len(acc) and all(
            _bfcl_one_value_match(a, e) for a, e in zip(actual, acc)
        )

    # Numeric coercion: 10 == "10" == 10.0
    try:
        if float(actual) == float(acc):
            return True
    except (TypeError, ValueError):
        pass

    # String coercion (case-insensitive for string args)
    if isinstance(actual, str) and isinstance(acc, str):
        return actual.strip().lower() == acc.strip().lower()

    return False
