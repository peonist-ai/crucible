"""IFEval — instruction-following fidelity.

Graded twice: prompt-level (every instruction in the prompt satisfied) and
instruction-level (how many individual constraints held). The pass/fail we
report is prompt-level; the instruction-level number is printed because it
degrades more gradually and shows compression damage earlier.

Both numbers are strict-mode only — see `crucible_bench.ifeval` for what that
means and where the port deviates from the reference implementation.
"""

from __future__ import annotations

from crucible_bench.testbench.api import chat_completion, truncate


def _preflight(ds) -> None:
    """Refuse to start a run we cannot score honestly.

    Two ways an IFEval run silently produces a flattering number: an
    instruction id with no checker, and a missing optional dependency whose
    ImportError gets swallowed into "instruction followed". Both are cheap to
    detect now and expensive to detect after an hour of generation, so they
    are errors before the first request rather than warnings after the last.
    """
    from crucible_bench.ifeval import (
        MissingIFEvalDependency,
        _detect_language,
        _word_tokenize,
        unsupported_instruction_ids,
    )

    all_ids = {iid for row in ds for iid in row["instruction_id_list"]}
    missing = unsupported_instruction_ids(all_ids)
    if missing:
        raise RuntimeError(
            f"IFEval dataset contains {len(missing)} instruction id(s) with no "
            f"checker: {missing}. Scoring them either way would be a guess -- "
            f"add checkers to crucible_bench/ifeval.py before running."
        )

    # Touch the two optional backends once, on a trivial input, so a missing
    # install fails here instead of quietly passing 95 of 834 instructions.
    try:
        _detect_language("this is a sentence in english")
        _word_tokenize("this is a sentence")
    except MissingIFEvalDependency as e:
        raise RuntimeError(f"IFEval cannot be scored: {e}") from e


def run_ifeval(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    from datasets import load_dataset

    from crucible_bench.ifeval import check_all_instructions

    ds = load_dataset("google/IFEval", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    _preflight(ds)

    results = []
    inst_total = 0
    inst_passed = 0

    for i, problem in enumerate(ds):
        prompt = problem["prompt"]
        instruction_ids = problem["instruction_id_list"]
        kwargs_list = problem["kwargs"]

        response, meta = chat_completion(url, prompt, max_tokens,
                                         temperature, extra_body)
        all_passed, per_inst = check_all_instructions(
            response, instruction_ids, kwargs_list,
        )

        inst_total += len(per_inst)
        inst_passed += sum(per_inst)

        # Show which instructions failed for debugging
        failed_ids = [
            iid for iid, passed in zip(instruction_ids, per_inst) if not passed
        ]
        status = "PASS" if all_passed else f"FAIL {failed_ids}"
        print(f"  [{i+1}/{len(ds)}] {status}", flush=True)

        results.append({
            "key": problem["key"],
            "passed": all_passed,
            "instruction_ids": instruction_ids,
            "per_instruction": per_inst,
            "instructions_passed": sum(per_inst),
            "instructions_total": len(per_inst),
            "prompt": truncate(prompt, 500),
            "response": truncate(response, 1000),
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    # Print summary stats
    prompt_acc = sum(1 for r in results if r["passed"]) / len(results)
    inst_acc = inst_passed / inst_total if inst_total > 0 else 0
    print(f"\n  Prompt-level accuracy (strict): {prompt_acc:.1%}")
    print(f"  Instruction-level accuracy (strict): {inst_acc:.1%} "
          f"({inst_passed}/{inst_total})")

    return results
