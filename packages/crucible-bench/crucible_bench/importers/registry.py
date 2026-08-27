"""What `--tool` resolves to, and the shape every importer returns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImportedRun:
    """One external benchmark run, normalized.

    `score` is `passed / total` and `total` is the benchmark's own denominator,
    so the number matches what the tool reports rather than a recomputation
    that quietly disagrees with the leaderboard.

    `incomplete` counts trials that produced no verdict at all -- a container
    that failed to build, an agent that crashed, a patch that never applied.
    They are inside `total` and scored as failures, because that is what the
    benchmarks do, but they are counted separately so a run wrecked by
    infrastructure cannot be read as a model that got worse.
    """

    task: str
    score: float
    passed: int
    total: int
    details: list[dict]
    provenance: dict
    incomplete: int = 0
    elapsed: float | None = None
    timestamp: str | None = None
    warnings: list[str] = field(default_factory=list)


Importer = Callable[[Path, str | None], ImportedRun]


def _terminal_bench(path: Path, task: str | None) -> ImportedRun:
    from crucible_bench.importers.terminal_bench import import_run

    return import_run(path, task)


def _swebench(path: Path, task: str | None) -> ImportedRun:
    from crucible_bench.importers.swebench import import_run

    return import_run(path, task)


IMPORTERS: dict[str, Importer] = {
    "terminal-bench": _terminal_bench,
    "swebench": _swebench,
}

SUPPORTED_TOOLS = list(IMPORTERS)


def get_importer(tool: str) -> Importer:
    if tool not in IMPORTERS:
        raise ValueError(
            f"no importer for {tool!r}; known tools: {SUPPORTED_TOOLS}"
        )
    return IMPORTERS[tool]
