"""Tests for buggy.py — at least one test fails until the bug is fixed."""
from buggy import Item, total_value_cents, low_stock


def test_total_value_cents():
    items = [Item("A", qty=2, price_cents=500), Item("B", qty=3, price_cents=100)]
    # 2 * 500 + 3 * 100 = 1300
    assert total_value_cents(items) == 1300, f"got {total_value_cents(items)}"


def test_low_stock():
    items = [Item("A", 10, 100), Item("B", 2, 100), Item("C", 4, 100), Item("D", 8, 100)]
    result = low_stock(items, threshold=5)
    assert [it.sku for it in result] == ["B", "C"]


if __name__ == "__main__":
    test_total_value_cents()
    test_low_stock()
    print("all tests passed")
