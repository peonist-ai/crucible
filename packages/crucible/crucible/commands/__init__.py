"""One module per `crucible` subcommand.

Each command module owns its own flags (`add_arguments`) and its own
implementation (`run`), so `cli.py` stays a parser and a dispatch table.
"""
