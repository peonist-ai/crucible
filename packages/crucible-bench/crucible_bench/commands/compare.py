"""crucible-bench compare — diff two benchmark result files."""

from __future__ import annotations

import argparse

NAME = "compare"
HELP = "Compare two benchmark results"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("baseline", help="Baseline results JSON")
    parser.add_argument("compressed", help="Compressed results JSON")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a markdown table instead of the terminal report, ready to "
             "paste into a model card or README.",
    )


def run(args) -> None:
    from crucible_bench.compare import (
        compare,
        load_results,
        markdown_comparison,
        print_comparison,
    )

    baseline = load_results(args.baseline)
    compressed = load_results(args.compressed)
    comparison = compare(baseline, compressed)
    if args.markdown:
        print(markdown_comparison(comparison))
    else:
        print_comparison(comparison)
