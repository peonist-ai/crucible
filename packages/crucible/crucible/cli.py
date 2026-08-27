"""CLI entrypoint for crucible.

Parser assembly and dispatch only. Each subcommand lives in
`crucible/commands/`, declares its own flags, and is imported here purely to
be registered — the command modules keep torch and transformers out of module
scope so building the parser stays instant.
"""

from __future__ import annotations

import argparse
import sys

from crucible.commands import (
    compress,
    inspect,
    observe,
    plan,
    quantize,
    quantize_mlx,
)

# Order here is the order subcommands appear in `crucible --help`.
# Benchmarking is not here on purpose: measuring a served model is a separate
# concern with a separate dependency tree, and lives in the `crucible-bench`
# package. crucible produces weights; crucible-bench measures an endpoint.
COMMANDS = (
    compress, quantize, quantize_mlx, plan, inspect, observe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="MoE expert compression toolkit — prune, merge, and forge specialist models",
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
