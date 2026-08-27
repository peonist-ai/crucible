"""Tests for merge.py.

Designed so the obvious first-attempt fix likely fails at least one case:
- shallow merge passes top-level but breaks nested
- naive recursion may break override-replaces-dict-with-scalar
- list handling must explicitly NOT recurse (lists are scalars per spec)
"""
from merge import merge


def test_top_level_disjoint():
    # Keys in only one side — both kept
    assert merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_top_level_override():
    # Override wins for scalars at top level
    assert merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    over = {"a": {"y": 2}}
    _ = merge(base, over)
    assert base == {"a": {"x": 1}}, f"base mutated: {base}"
    assert over == {"a": {"y": 2}}, f"override mutated: {over}"


def test_nested_dict_merge():
    # The headline case: nested dicts should be merged, not replaced
    base = {"db": {"host": "localhost", "port": 5432}, "log": "info"}
    over = {"db": {"port": 6000}}
    expected = {"db": {"host": "localhost", "port": 6000}, "log": "info"}
    assert merge(base, over) == expected


def test_three_levels_deep():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    over = {"a": {"b": {"c": 10}}}
    assert merge(base, over) == {"a": {"b": {"c": 10, "d": 2}}}


def test_override_dict_with_scalar():
    # Per spec: if override's value is NOT a dict, it replaces base wholesale
    assert merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}


def test_override_scalar_with_dict():
    # Per spec: same direction — override wins. Base scalar replaced by override dict.
    assert merge({"a": 1}, {"a": {"x": 2}}) == {"a": {"x": 2}}


def test_lists_are_scalars():
    # Per spec: lists are treated as scalars. Override replaces wholesale.
    assert merge({"items": [1, 2, 3]}, {"items": [4, 5]}) == {"items": [4, 5]}


def test_list_inside_dict_preserved():
    # Lists nested inside dicts also follow the scalar rule
    base = {"server": {"hosts": ["a", "b"], "port": 80}}
    over = {"server": {"hosts": ["c"]}}
    assert merge(base, over) == {"server": {"hosts": ["c"], "port": 80}}


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"]).returncode)
