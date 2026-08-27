"""Tests for ops.py."""
import pytest
from ops import add, sub, mul, div


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_sub():
    assert sub(10, 3) == 7
    assert sub(0, 5) == -5


def test_mul():
    assert mul(4, 5) == 20
    assert mul(0, 100) == 0


def test_div():
    assert div(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        div(1, 0)


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"]).returncode)
