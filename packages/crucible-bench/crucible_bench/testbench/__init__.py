"""Crucible test bench: benchmark models via OpenAI-compatible API.

Runs standardized benchmarks against a model served via llama-server,
vLLM, or any OpenAI-compatible endpoint. Handles Gemma 4's thinking
output format (channel tags) transparently.

Supported benchmarks (see suites.py for the full registry):
  humaneval(_plus) — code generation (164 problems, pass@1)
  mbpp(_plus)      — code generation (500 / 378 problems, pass@1)
  bigcodebench     — real-library Python coding (1140 problems, pass@1)
  bfcl_simple      — single-function tool calling (400 cases, AST match)
  bfcl_multi_turn* — stateful tool use over 4-6 turns (200 cases each,
                     state-graded; needs the `bfcl` extra)
  ifeval           — instruction following
  mmlu_pro         — multiple choice, 10 options (12K questions, accuracy)
  gpqa_diamond     — graduate-level science MC (198 questions, accuracy)
  livecodebench    — NOT IMPLEMENTED, raises. No scorer exists; see
                     tasks_code.run_livecodebench for what one would need.

Suites:
  quick    → humaneval                                (~15 min)
  coding   → humaneval, mbpp                          (~2 hrs)
  agentic  → bfcl_multi_turn, bfcl_multi_turn_miss_func
  full     → humaneval, mbpp, bigcodebench, mmlu_pro, gpqa_diamond, ifeval

Usage — normally via the CLI:

    crucible-bench run --url http://localhost:8091 --model baseline --tasks humaneval
    crucible-bench run --url http://localhost:8091 --model baseline --suite regression
    crucible-bench run --url http://localhost:8091 --model baseline --suite full --limit 100

The module is also runnable directly, which skips the CLI's sandbox setup:

    python -m crucible_bench.testbench --url http://localhost:8091 --model baseline
"""

from crucible_bench.testbench.api import (
    api_get,
    chat_completion,
    extract_code,
    strip_thinking,
    text_completion,
    truncate,
)
from crucible_bench.testbench.runner import main, run_bench
from crucible_bench.testbench.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    SANDBOX_MODES,
    execute_code,
    set_sandbox,
)
from crucible_bench.testbench.suites import SUITES, SUPPORTED_TASKS, TASKS, TaskSpec

__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "SANDBOX_MODES",
    "SUITES",
    "SUPPORTED_TASKS",
    "TASKS",
    "TaskSpec",
    "api_get",
    "chat_completion",
    "execute_code",
    "extract_code",
    "main",
    "run_bench",
    "set_sandbox",
    "strip_thinking",
    "text_completion",
    "truncate",
]
