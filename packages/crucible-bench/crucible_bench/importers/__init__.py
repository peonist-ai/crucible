"""Importers for benchmarks crucible-bench does not run itself.

Some benchmarks we cannot honestly own. Terminal-Bench needs Harbor to build
and drive containers; SWE-bench needs its own evaluation harness to apply
patches and run repo test suites. Reimplementing either would mean claiming to
measure something we actually delegated, which is the failure mode the
`livecodebench` scorer used to be.

So we import instead. The external tool runs under its own harness, and this
turns its native output into the same result JSON `compare` reads — with a
`provenance` block recording what crucible-bench did *not* control: which tool,
which version, which dataset, which agent scaffold. An imported score without
that block is a number with no way to check it.

Adding a tool is one module plus one registry entry; nothing else changes.
"""

from __future__ import annotations

from crucible_bench.importers.registry import (
    IMPORTERS,
    SUPPORTED_TOOLS,
    ImportedRun,
    get_importer,
)

__all__ = ["IMPORTERS", "SUPPORTED_TOOLS", "ImportedRun", "get_importer"]
