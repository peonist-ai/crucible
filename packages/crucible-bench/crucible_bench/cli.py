"""CLI entrypoint for crucible-bench.

Parser assembly and dispatch only. Each subcommand lives in
`crucible_bench/commands/`, declares its own flags, and is imported here
purely to be registered.
"""

from __future__ import annotations

import argparse
import sys

from crucible_bench.commands import bench, compare, evaluate, import_results

# Order here is the order subcommands appear in `crucible-bench --help`.
COMMANDS = (bench, import_results, compare, evaluate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible-bench",
        description="Benchmark a served model over an OpenAI-compatible API",
    )
    # dest is "subcommand", not "command": a subcommand that declares a
    # `--command` flag would otherwise overwrite the dispatch target with
    # its own default and the CLI would silently print help instead of
    # running. That is a very quiet way to lose a benchmark run.
    subparsers = parser.add_subparsers(dest="subcommand")

    for command in COMMANDS:
        sub = subparsers.add_parser(command.NAME, help=command.HELP)
        command.add_arguments(sub)
        sub.set_defaults(run=command.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    args.run(args)


if __name__ == "__main__":
    main()
