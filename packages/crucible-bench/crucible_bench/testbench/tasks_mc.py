"""Multiple-choice benchmarks: MMLU-Pro and GPQA Diamond.

Secondary metrics — a sanity check that compression didn't crater general
knowledge, not something to optimize. Both run with thinking enabled, so
answer extraction has to survive a long chain of reasoning before the letter.
"""

from __future__ import annotations

import random
import re

from crucible_bench.testbench.api import chat_completion, truncate

LETTERS = "ABCDEFGHIJ"


def run_mmlu_pro(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    from datasets import load_dataset

    ds_test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds_val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    # Build few-shot examples grouped by category (5 per category)
    val_by_cat = {}
    for ex in ds_val:
        cat = ex.get("category", "unknown")
        if cat not in val_by_cat:
            val_by_cat[cat] = []
        if len(val_by_cat[cat]) < 5:
            val_by_cat[cat].append(ex)

    if limit:
        # Sample randomly for representative coverage across categories
        indices = list(range(len(ds_test)))
        random.seed(seed)
        random.shuffle(indices)
        ds_test = ds_test.select(indices[:limit])

    results = []
    for i, problem in enumerate(ds_test):
        qid = problem["question_id"]
        question = problem["question"]
        options = problem["options"]
        correct = problem["answer"]
        category = problem.get("category", "unknown")

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(ds_test)}] ({category})...", flush=True)

        prompt = _format_mmlu_pro_prompt(question, options, category,
                                         val_by_cat.get(category, []))
        response, meta = chat_completion(url, prompt, max_tokens=max_tokens,
                                         temperature=0.0,
                                         extra_body=extra_body)
        predicted, tier = _extract_mmlu_pro_answer(response, len(options))

        passed = predicted == correct
        status = "PASS" if passed else f"FAIL (pred={predicted} correct={correct})"
        print(f"  [{i+1}/{len(ds_test)}] ({category}) {status}", flush=True)

        results.append({
            "question_id": qid,
            "passed": passed,
            "predicted": predicted,
            # Which pattern produced `predicted`; "none" means unparseable and
            # scored wrong. A score carried by a loose tier is a score to
            # re-read, so it goes in the record rather than the console.
            "extraction_tier": tier,
            "correct": correct,
            "category": category,
            "prompt": truncate(prompt, 500),
            "response": truncate(response, 1000),
            # An empty response with finish_reason "length" is the model
            # spending its whole budget reasoning, not failing the question.
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    # Print per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["correct"] += 1

    print("\n  Per-category results:")
    for cat in sorted(categories):
        c = categories[cat]
        pct = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"    {cat:<30} {pct:.1%} ({c['correct']}/{c['total']})")

    return results


def run_gpqa_diamond(url, max_tokens, temperature, limit, seed, extra_body=None, checkpoint=None):
    from datasets import load_dataset

    ds = load_dataset("nmayorga7/gpqa_diamond", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    for i, problem in enumerate(ds):
        question = problem["Question"]
        correct_answer = problem["Correct Answer"]
        options = [
            correct_answer,
            problem["Incorrect Answer 1"],
            problem["Incorrect Answer 2"],
            problem["Incorrect Answer 3"],
        ]
        # Shuffle options so correct isn't always first
        random.seed(seed + i)
        indices = list(range(4))
        random.shuffle(indices)
        shuffled = [options[j] for j in indices]
        correct_letter = chr(65 + indices.index(0))  # letter for correct answer

        domain = problem.get("High-level domain", "unknown")
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(ds)}] ({domain})...", flush=True)

        prompt = _format_gpqa_prompt(question, shuffled)
        response, meta = chat_completion(url, prompt, max_tokens=max_tokens,
                                         temperature=0.0,
                                         extra_body=extra_body)
        predicted, tier = _extract_gpqa_answer(response)

        passed = predicted == correct_letter
        status = "PASS" if passed else f"FAIL (pred={predicted} correct={correct_letter})"
        print(f"  [{i+1}/{len(ds)}] ({domain}) {status}", flush=True)

        results.append({
            "question_id": i,
            "passed": passed,
            "predicted": predicted,
            "extraction_tier": tier,
            "correct": correct_letter,
            "domain": domain,
            "prompt": truncate(prompt, 500),
            "response": truncate(response, 1000),
            # An empty response with finish_reason "length" is the model
            # spending its whole budget reasoning, not failing the question.
            **meta,
        })
        if checkpoint:
            checkpoint(results)

    return results


def _format_mmlu_pro_prompt(
    question: str, options: list[str], category: str, few_shot: list[dict],
) -> str:
    """Format MMLU-Pro prompt following TIGER-Lab's standard 5-shot CoT format."""
    header = (
        f"The following are multiple choice questions (with answers) about "
        f"{category}. Think step by step and then output the answer in the "
        f"format of \"The answer is (X)\" at the end.\n\n"
    )

    # Few-shot examples from validation set
    examples = ""
    for ex in few_shot:
        examples += "Question:\n"
        examples += ex["question"] + "\n"
        examples += "Options:\n"
        for j, opt in enumerate(ex["options"]):
            examples += f"{LETTERS[j]}. {opt}\n"
        # Use the CoT content from the dataset
        cot = ex.get("cot_content", "")
        if cot:
            cot = cot.replace(
                "A: Let's think step by step.",
                "Answer: Let's think step by step.",
            )
            examples += cot + "\n\n"

    # Test question
    test_q = "Question:\n"
    test_q += question + "\n"
    test_q += "Options:\n"
    for j, opt in enumerate(options):
        test_q += f"{LETTERS[j]}. {opt}\n"
    test_q += "Answer: Let's think step by step."

    return header + examples + test_q


def _extract_mmlu_pro_answer(response: str, num_options: int) -> tuple[str, str]:
    """Extract the answer letter from an MMLU-Pro CoT response.

    Returns (letter, tier), where tier names which pattern matched so the
    runner can report how much of a score rests on the looser one. An
    unparseable response returns ("?", "none") and is scored wrong.

    There used to be a third tier that took the last standalone letter A-J
    anywhere in the response. That is not an extraction, it is a 1-in-10
    guess made on the model's behalf, and it quietly propped up the score of
    exactly the runs most worth distrusting -- the ones that rambled or got
    truncated. TIGER-Lab's own harness random-guesses here; we would rather
    report the miss.
    """
    valid = set(LETTERS[:num_options])

    # Primary: "the answer is (X)" or "the answer is X"
    match = re.search(r"answer is \(?([A-J])\)?", response, re.IGNORECASE)
    if match and match.group(1) in valid:
        return match.group(1), "answer_is"

    # Fallback: "Answer: X"
    match = re.search(r"[aA]nswer:\s*([A-J])", response)
    if match and match.group(1) in valid:
        return match.group(1), "answer_colon"

    return "?", "none"


def _format_gpqa_prompt(question: str, options: list[str]) -> str:
    """Format GPQA Diamond prompt following the standard 0-shot CoT format."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"Answer the following multiple choice question. The last line of "
        f"your response should be of the following format: 'ANSWER: LETTER' "
        f"(without quotes) where LETTER is one of ABCD. Think step by step "
        f"before answering.\n\n"
        f"{question}\n\n{opts}"
    )


def _extract_gpqa_answer(response: str) -> tuple[str, str]:
    """Extract the answer letter from a GPQA response.

    Returns (letter, tier). The prompt asks for a final 'ANSWER: X' line and
    that is what we look for, plus the 'answer is X' phrasing models reach for
    anyway. Anything else is ("?", "none") and scores wrong, which is how the
    official grader treats it.

    The removed last-resort tier -- last standalone letter A-D in the response
    -- was actively harmful here. GPQA is physics and chemistry, where bare
    B (magnetic field), A (area) and C (capacitance) appear all through the
    reasoning, so it handed a free 25% to every response it could not parse.
    """
    match = re.search(r"ANSWER:\s*\(?([A-D])\)?", response)
    if match:
        return match.group(1), "answer_colon"

    match = re.search(r"answer is \(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1), "answer_is"

    return "?", "none"
