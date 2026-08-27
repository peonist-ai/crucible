"""Task registry and suites.

One entry per benchmark: its runner and its generation budget. Adding a task
means adding a TaskSpec here — the CLI, `--tasks` validation and the suites
all read off this dict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from crucible_bench.testbench.tasks_code import (
    run_bigcodebench,
    run_humaneval,
    run_humaneval_plus,
    run_livecodebench,
    run_mbpp,
    run_mbpp_plus,
)
from crucible_bench.testbench.tasks_instruct import run_ifeval
from crucible_bench.testbench.tasks_mc import run_gpqa_diamond, run_mmlu_pro
from crucible_bench.testbench.tasks_tools import run_bfcl_simple
from crucible_bench.testbench.tasks_tools_multiturn import (
    run_bfcl_multi_turn_base,
    run_bfcl_multi_turn_composite,
    run_bfcl_multi_turn_long_context,
    run_bfcl_multi_turn_miss_func,
    run_bfcl_multi_turn_miss_param,
)


class TaskRunner(Protocol):
    """Run one benchmark and return a per-problem record list.

    Every record carries at least `passed`; the rest is task-specific and
    lands verbatim in the result JSON's `details`.

    `checkpoint` is called with the records so far after each problem. Runners
    just call it; how often it actually writes is the caller's business. It
    exists because a task is not the right unit of loss — GPQA Diamond is one
    task and over an hour of generation, and losing it whole to an interrupted
    run happened twice on 2026-08-23 before this hook existed.
    """

    def __call__(
        self,
        url: str,
        max_tokens: int,
        temperature: float,
        limit: int | None,
        seed: int,
        extra_body: dict | None = None,
        checkpoint: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]: ...


@dataclass(frozen=True)
class TaskSpec:
    """A benchmark and the generation budget it needs.

    max_tokens covers reasoning + answer combined, since servers bill both
    against it.

    `reasoning` declares that the task *benefits* from the model thinking
    before answering — it does not say how to ask for that. How to ask is the
    server's dialect (`reasoning_effort` on newer seats, a token budget on
    older ones) and belongs on the command line, not baked into the registry.
    Hard-coding one family's spelling here is what made the harness quietly
    report a thinking budget no server ever honoured.
    """

    name: str
    run: TaskRunner
    max_tokens: int = 2048
    reasoning: bool = False


TASKS: dict[str, TaskSpec] = {
    t.name: t for t in (
        TaskSpec("humaneval", run_humaneval),
        TaskSpec("humaneval_plus", run_humaneval_plus),
        TaskSpec("mbpp", run_mbpp),
        TaskSpec("mbpp_plus", run_mbpp_plus),
        # BigCodeBench solutions often involve multiple imports + helpers.
        TaskSpec("bigcodebench", run_bigcodebench, max_tokens=4096),
        # Was pinned to 1024 on the assumption that a single tool call is short.
        # It isn't: measured 2026-08-17, calls on a served 35B needed up to 1461
        # tokens, and a tool-call block cut off by the cap parses to *nothing* --
        # scored as "no tool call" and silently blamed on whatever was under test.
        # Takes the 2048 default now; check `finish_reason` in the results if a
        # model ever looks like it stopped emitting calls.
        TaskSpec("bfcl_simple", run_bfcl_simple),
        # Multi-turn is one request per step, not per entry: 4-6 turns and
        # up to 20 tool steps each. The budget is per request, so 2048 is
        # right for the same reason it is for bfcl_simple -- a truncated
        # tool-call block parses to nothing and looks like a refusal.
        TaskSpec("bfcl_multi_turn", run_bfcl_multi_turn_base),
        TaskSpec("bfcl_multi_turn_composite", run_bfcl_multi_turn_composite),
        TaskSpec("bfcl_multi_turn_long_context", run_bfcl_multi_turn_long_context,
                 max_tokens=4096),
        TaskSpec("bfcl_multi_turn_miss_func", run_bfcl_multi_turn_miss_func),
        TaskSpec("bfcl_multi_turn_miss_param", run_bfcl_multi_turn_miss_param),
        TaskSpec("mmlu_pro", run_mmlu_pro, reasoning=True),
        TaskSpec("gpqa_diamond", run_gpqa_diamond, max_tokens=4096, reasoning=True),
        TaskSpec("ifeval", run_ifeval),
        # Registered so `--tasks livecodebench` fails with an explanation
        # rather than argparse's "invalid choice". It has no scorer; see
        # tasks_code.run_livecodebench.
        TaskSpec("livecodebench", run_livecodebench),
    )
}

SUPPORTED_TASKS = list(TASKS)

SUITES = {
    "quick": ["humaneval"],
    "coding": ["humaneval", "mbpp"],
    # EvalPlus-augmented coding — augmented tests catch "lucky pass" solutions
    # that trivially pass the tiny base test suites. Scores run 10-20pt lower
    # than base, and that gap is what we care about for regression tracking.
    "coding_plus": ["humaneval_plus", "mbpp_plus"],
    # Quality gate for compressed Qwen 3.6: cheap regression checks that
    # cover code correctness + structured tool calling together.
    "regression": ["humaneval_plus", "mbpp_plus", "bfcl_simple"],
    # Multi-turn stateful tool use. Separate from `regression` because it is
    # far more expensive -- every entry is 4-6 turns of several requests each --
    # and because it measures a different band: single-call tool use barely
    # moves under compression while multi-step application accuracy drops
    # several times as much.
    "agentic": ["bfcl_multi_turn", "bfcl_multi_turn_miss_func"],
    "instruct": ["ifeval"],
    "full": ["humaneval", "mbpp", "bigcodebench", "mmlu_pro", "gpqa_diamond", "ifeval"],
}
