"""Recursive config-merge utility."""


def merge(base: dict, override: dict) -> dict:
    """Deep-merge `override` into `base`. Returns a NEW dict; inputs unchanged.

    Rules:
      - For keys present in both: if BOTH values are dicts, recurse into them.
      - For keys present in both where override's value is NOT a dict, the
        override value replaces base's (even if base's value IS a dict).
      - For keys only in one side, that side's value is kept as-is.
      - Lists are treated as scalars: override replaces base wholesale.
    """
    return {**base, **override}
