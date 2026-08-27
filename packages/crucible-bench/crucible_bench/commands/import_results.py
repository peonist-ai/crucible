"""crucible-bench import — turn an external benchmark's output into a result file.

For benchmarks we deliberately do not run: see `crucible_bench.importers`.
The output is the same shape `crucible-bench run` writes, so `compare` treats
an imported Terminal-Bench score and a native HumanEval+ score identically —
with one difference that matters: `"imported": true` and a `provenance` block
naming every variable the external tool controlled and we did not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

NAME = "import"
HELP = "Import results from an external benchmark harness"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from crucible_bench.importers import SUPPORTED_TOOLS

    parser.add_argument("--tool", required=True, choices=SUPPORTED_TOOLS,
                        help="Which external harness produced the output")
    parser.add_argument("--from", dest="source", required=True,
                        help="The harness's run directory or report file")
    parser.add_argument("--model", required=True,
                        help="Label for this model variant, matching the label "
                             "used for its native benchmark runs so compare can "
                             "align them (e.g. baseline, reap-37pct)")
    parser.add_argument("--task", default=None,
                        help="Task name in the result file. Defaults to the "
                             "tool's own name; override to distinguish datasets "
                             "(e.g. swebench_verified_mini vs swebench_verified)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset and version the harness ran, recorded in "
                             "provenance (e.g. 'terminal-bench@2.0'). Neither "
                             "harness reliably reports this, and a score without "
                             "it cannot be compared to anyone else's.")
    parser.add_argument("--agent", default=None,
                        help="Agent scaffold and version, if the harness does "
                             "not report it (e.g. 'mini-swe-agent 1.2.0')")
    # dest is not "command": argparse's subparsers already own that name, and
    # a flag that shadows it silently sets the dispatch target to None.
    parser.add_argument("--command", dest="command_line", default=None,
                        help="The exact command line that produced this run. "
                             "The single most useful thing for reproducing it.")
    parser.add_argument("--endpoint", default=None,
                        help="URL the harness pointed at, for the record")
    parser.add_argument("--metadata", default=None,
                        help="compression_metadata.json from the crucible run "
                             "that produced these weights; embedded so the score "
                             "and the compression that caused it stay together")
    parser.add_argument("-o", "--output", default="results",
                        help="Directory to write the result JSON (default: results)")


def run(args) -> None:
    from crucible_bench.importers import get_importer

    importer = get_importer(args.tool)
    source = Path(args.source)
    if not source.exists():
        print(f"Error: {source} does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        imported = importer(source, args.task)
    except (ValueError, FileNotFoundError) as e:
        # These are all "the input is not what you said it was". Failing here
        # is the point: an importer that shrugs and writes a zero produces a
        # regression that never happened.
        print(f"Error importing {args.tool} run: {e}", file=sys.stderr)
        sys.exit(1)

    metadata = None
    if args.metadata:
        with open(args.metadata) as f:
            metadata = json.load(f)

    timestamp = imported.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"{args.model}_{args.tool}_{_stamp(timestamp)}"

    provenance = dict(imported.provenance)
    provenance.update({
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "agent": args.agent or provenance.get("agents"),
        "command": args.command_line,
        "endpoint": args.endpoint,
    })

    result = {
        "model": args.model,
        "run_id": run_id,
        "timestamp": timestamp,
        "url": args.endpoint,
        # Loud on purpose. Everything below the scores was measured by another
        # tool under conditions this file only records second-hand.
        "imported": True,
        "provenance": provenance,
        "config": {
            "max_tokens_override": None,
            "temperature": None,
            "seed": None,
            "limit": None,
        },
        "results": {
            imported.task: {
                "score": imported.score,
                "passed": imported.passed,
                "total": imported.total,
                "max_tokens": None,
                "thinking_budget": None,
                "truncated": None,
                "unparsed": None,
                # Trials that produced no verdict. Inside `total` and scored as
                # failures, counted here so an infrastructure-wrecked run is not
                # read as a worse model.
                "incomplete": imported.incomplete,
            }
        },
        "details": {imported.task: imported.details},
        "elapsed": {imported.task: imported.elapsed},
    }
    if metadata is not None:
        result["compression_metadata"] = metadata

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  {imported.task}: {imported.passed}/{imported.total} = "
          f"{imported.score:.1%}")
    for warning in imported.warnings:
        print(f"  WARNING: {warning}")
    if not args.dataset:
        print("  NOTE: no --dataset recorded. This score cannot be compared to "
              "a published number without knowing which dataset version ran.")
    print(f"\n  Imported to {out_file}")


def _stamp(timestamp: str) -> str:
    """A filename-safe stamp from the tool's own finish time when we have it.

    Importing the same run twice overwrites rather than accumulating near-
    duplicate files that differ only in when someone re-ran the import.
    """
    return "".join(c for c in timestamp if c.isdigit())[:14] or time.strftime(
        "%Y%m%d%H%M%S"
    )
