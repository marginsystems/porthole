import pytest
from inventory.store import Item, Store
from inventory.pricing import apply_discount, bulk_price


def make():
    s = Store()
    s.add(Item("A1", "bolt", 25, 100))
    s.add(Item("B2", "nut", 10, 3))
    return s


def test_total_value_counts_quantity():
    assert make().total_value_cents() == 25 * 100 + 10 * 3


def test_remove_all_deletes_item():
    s = make()
    s.remove("B2", 3)
    assert "B2" not in s.items


def test_remove_too_many():
    with pytest.raises(ValueError):
        make().remove("B2", 4)


def test_low_stock():
    assert make().low_stock() == ["B2"]


def test_discount():
    assert apply_discount(1000, 15) == 850
    assert apply_discount(999, 10) == 899


def test_bulk_boundaries():
    assert bulk_price(100, 9) == 900
    assert bulk_price(100, 10) == 900
    assert bulk_price(100, 50) == 4000
