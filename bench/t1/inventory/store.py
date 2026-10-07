"""Simple inventory store."""
from dataclasses import dataclass, field


@dataclass
class Item:
    sku: str
    name: str
    price_cents: int
    qty: int = 0


@dataclass
class Store:
    items: dict = field(default_factory=dict)

    def add(self, item: Item) -> None:
        if item.sku in self.items:
            self.items[item.sku].qty += item.qty
        else:
            self.items[item.sku] = item

    def remove(self, sku: str, qty: int) -> None:
        item = self.items[sku]
        if qty > item.qty:
            raise ValueError("not enough stock")
        item.qty -= qty
        if item.qty < 0:
            del self.items[sku]

    def total_value_cents(self) -> int:
        return sum(i.price_cents for i in self.items.values())

    def low_stock(self, threshold: int = 5) -> list:
        return sorted(i.sku for i in self.items.values() if i.qty < threshold)
