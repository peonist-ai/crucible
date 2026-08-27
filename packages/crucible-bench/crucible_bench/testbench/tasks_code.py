"""Code-generation benchmarks: HumanEval(+), MBPP(+), BigCodeBench, LiveCodeBench.

Every runner here has the same shape — ask for a function, extract the code,
run it against the dataset's tests in whatever isolation set_sandbox() chose,
and report pass/fail per problem.
"""

from __future__ import annotations

import json
import os

from crucible_bench.testbench.api import chat_completion, extract_code, truncate
from crucible_bench.testbench.sandbox import execute_code, require_modules


def run_humaneval(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    for i, problem in enumerate(ds):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        print(f"  [{i+1}/{len(ds)}] {task_id}...", end=" ", flush=True)

        message = (
            f"Complete this Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{prompt}"
        )
        response, meta = chat_completion(url, message, max_tokens,
                                         temperature, extra_body)
        code = extract_code(response, fallback_prefix="")

        full_code = _assemble_humaneval_code(code, prompt, entry_point) + \
            "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        passed, error = execute_code(full_code)

        print("PASS" if passed else "FAIL")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": truncate(error, 200),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def run_humaneval_plus(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    """HumanEval+ — same 164 problems, augmented test cases (~80x more tests).

    Dataset columns match vanilla HumanEval (task_id, prompt, canonical_solution,
    entry_point, test). Only the `test` field is expanded. Scores typically
    run 10-20pt lower than HumanEval — that's the point.
    """
    require_modules(["numpy"])

    from datasets import load_dataset

    ds = load_dataset("evalplus/humanevalplus", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    for i, problem in enumerate(ds):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        print(f"  [{i+1}/{len(ds)}] {task_id}...", end=" ", flush=True)

        message = (
            f"Complete this Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{prompt}"
        )
        response, meta = chat_completion(url, message, max_tokens,
                                         temperature, extra_body)
        code = extract_code(response, fallback_prefix="")

        full_code = _assemble_humaneval_code(code, prompt, entry_point) + \
            "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        # Augmented tests run many more cases; bump timeout.
        passed, error = execute_code(full_code, timeout=30)

        print("PASS" if passed else "FAIL")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": truncate(error, 200),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def run_mbpp(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    from datasets import load_dataset

    ds = load_dataset("mbpp", "full", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    for i, problem in enumerate(ds):
        task_id = problem["task_id"]
        prompt_text = problem["text"]
        test_list = problem["test_list"]

        print(f"  [{i+1}/{len(ds)}] task_{task_id}...", end=" ", flush=True)

        # Include first test assertion so model knows the expected function name
        hint = test_list[0] if test_list else ""
        message = (
            f"Write a Python function for this task. Return ONLY the code, "
            f"no explanation:\n\n{prompt_text}\n\n"
            f"The function must satisfy: {hint}"
        )
        response, meta = chat_completion(url, message, max_tokens,
                                         temperature, extra_body)
        code = extract_code(response)

        test_code = "\n".join(test_list)
        full_code = code + "\n\n" + test_code + "\n"
        passed, error = execute_code(full_code)

        print("PASS" if passed else "FAIL")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": truncate(error, 200),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def run_mbpp_plus(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    """MBPP+ — 378 high-quality problems (filtered from MBPP's 500),
    augmented with many more test cases per problem.

    Unlike vanilla MBPP we run the combined `test` field (includes both base
    + plus tests). `test_list` is still present for getting the function
    name hint via the first assertion.
    """
    require_modules(["numpy"])

    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    for i, problem in enumerate(ds):
        task_id = problem["task_id"]
        prompt_text = problem["prompt"]
        test_list = problem["test_list"]
        test_code = problem["test"]

        print(f"  [{i+1}/{len(ds)}] task_{task_id}...", end=" ", flush=True)

        hint = test_list[0] if test_list else ""
        message = (
            f"Write a Python function for this task. Return ONLY the code, "
            f"no explanation:\n\n{prompt_text}\n\n"
            f"The function must satisfy: {hint}"
        )
        response, meta = chat_completion(url, message, max_tokens,
                                         temperature, extra_body)
        code = extract_code(response)

        # The plus test harness defines a check() that takes the candidate;
        # it mirrors HumanEval+'s pattern but we need the function name.
        # Append check invocation using the first assertion's function name
        # pattern (test_list[0] like `assert foo(...) == ...`).
        # Simpler: concatenate code + test, then rely on test to invoke check.
        full_code = code + "\n\n" + test_code + "\n"
        passed, error = execute_code(full_code, timeout=30)

        print("PASS" if passed else "FAIL")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": truncate(error, 200),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def run_bigcodebench(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    """BigCodeBench (1140 problems) — real-library Python coding.

    Uses the latest split (v0.1.4). Each problem provides:
      - complete_prompt: full def with imports + signature + docstring
      - test:            unittest.TestCase block (NOT a check() function
                         like HumanEval) that exercises the solution
      - entry_point:     name of the function the model must implement

    Harness assembly: complete_prompt + model body + test block + a
    discovery shim that runs every TestCase subclass via TextTestRunner
    and exits non-zero on failure. The shim avoids relying on
    `unittest.main()`'s argv/__main__ machinery, so it works regardless
    of test naming.
    """
    require_modules(["numpy", "pandas", "flask", "sklearn", "matplotlib"])

    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Optional incremental NDJSON debug log — opt-in via env var so it doesn't
    # affect other tasks. Each line: {task_id, passed, error[:3000], code}.
    # Use this for long runs where end-of-run JSON is too late if the bench
    # is killed or crashes.
    debug_log_path = os.environ.get("BIGCODE_DEBUG_LOG")
    debug_log = open(debug_log_path, "a") if debug_log_path else None

    runner = (
        "\n\nif __name__ == '__main__':\n"
        "    import unittest, sys\n"
        "    _loader = unittest.TestLoader()\n"
        "    _suites = []\n"
        "    for _name, _obj in list(globals().items()):\n"
        "        if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase) "
        "and _obj is not unittest.TestCase:\n"
        "            _suites.append(_loader.loadTestsFromTestCase(_obj))\n"
        "    if not _suites:\n"
        "        sys.exit('no test cases found')\n"
        "    _suite = unittest.TestSuite(_suites)\n"
        "    _result = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(_suite)\n"
        "    sys.exit(0 if _result.wasSuccessful() else 1)\n"
    )

    results = []
    for i, problem in enumerate(ds):
        task_id = problem["task_id"]
        complete_prompt = problem["complete_prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        print(f"  [{i+1}/{len(ds)}] {task_id}...", end=" ", flush=True)

        message = (
            f"Complete this Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{complete_prompt}"
        )
        response, meta = chat_completion(url, message, max_tokens,
                                         temperature, extra_body)
        code = extract_code(response, fallback_prefix="")

        # Reuse HumanEval's assembly logic: same shape (signature + docstring,
        # body to fill in). If model returned the full def the assembler keeps
        # the model's version; if body-only, indent and append to prompt.
        full_code = (
            _assemble_humaneval_code(code, complete_prompt, entry_point)
            + "\n\n" + test_code + runner
        )
        # BigCodeBench tests can hit network mocks, file I/O, image ops; bump.
        passed, error = execute_code(full_code, timeout=60)

        print("PASS" if passed else "FAIL")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": truncate(error, 200),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

        if debug_log:
            debug_log.write(json.dumps({
                "task_id": task_id,
                "passed": passed,
                "error": (error or "")[:3000],
                "code": code,
            }) + "\n")
            debug_log.flush()

    if debug_log:
        debug_log.close()
    return results


def run_livecodebench(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    """Not implemented. Refuses to run rather than report a number.

    This used to score `passed = "def " in code`: no tests were executed, so
    any response containing the substring `def ` counted as solved and the
    task reported something close to 100% for every model it was pointed at.
    The package docstring advertised it as pass@1. That is the one failure
    mode a benchmark harness must never have, so the fake scorer is gone and
    nothing has replaced it yet.

    Doing this properly is real work, not a patch:
      - `code_generation_lite` is config-per-release (`release_v1`..`v6`), and
        picking a release is a contamination decision -- problems published
        before a model's cutoff are memorised, not solved.
      - Problems come in two flavours. `metadata.func_name` marks call-based
        problems, graded by invoking that function; the rest are stdin/stdout
        and need the solution run as a script with input piped in.
      - `private_test_cases` are base64(zlib(pickle)) and hold most of the
        signal; grading on `public_test_cases` alone overstates pass@1.
      - Each case needs a per-test timeout, not one per problem.

    Until that exists, use `bigcodebench` for execution-graded coding.
    """
    raise NotImplementedError(
        "livecodebench has no scorer. It previously reported "
        "'response contains def' as pass@1, which was not a measurement. "
        "Use --tasks bigcodebench for execution-graded coding, or implement "
        "the stdin/stdout and call-based runners described in this "
        "function's docstring."
    )


def _assemble_humaneval_code(model_code: str, prompt: str, entry_point: str) -> str:
    """Combine the model's completion with the HumanEval prompt.

    The HumanEval prompt contains critical context (imports like
    `from typing import List`) that the model often omits when it returns
    just the function. We always prepend the prompt:

      - If the model returned the full function (already contains `def
        entry_point`), appending it after the prompt produces a duplicate
        def that Python resolves by using the second one (the model's).
      - If the model returned just the body / a completion, we indent it
        and place it after the prompt's signature + docstring.

    Either way, imports from the prompt are in scope for the model's code.
    """
    if f"def {entry_point}" in model_code:
        return prompt + "\n\n" + model_code
    # Body-only completion: indent non-empty unindented lines, then append
    # to prompt (which ends with the docstring, ready for a body).
    lines = model_code.split("\n")
    indented: list[str] = []
    for line in lines:
        if line.strip() and not line.startswith((" ", "\t")):
            indented.append("    " + line)
        else:
            indented.append(line)
    return prompt + "\n".join(indented)
