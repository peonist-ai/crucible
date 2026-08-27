"""A small inventory module."""
from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    qty: int
    price_cents: int


def total_value_cents(items: list[Item]) -> int:
    """Return total value of all items in cents."""
    return sum(it.qty + it.price_cents for it in items)


def low_stock(items: list[Item], threshold: int = 5) -> list[Item]:
    """Return items below threshold, sorted by qty ascending."""
    return sorted([it for it in items if it.qty < threshold], key=lambda x: x.qty)
