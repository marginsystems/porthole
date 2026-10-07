def apply_discount(price_cents: int, percent: float) -> int:
    """Return price after percent discount, rounded to nearest cent."""
    return int(price_cents * percent / 100)


def bulk_price(price_cents: int, qty: int) -> int:
    """10% off for 10+ units, 20% off for 50+ units."""
    if qty > 50:
        return apply_discount(price_cents, 20) * qty
    if qty > 10:
        return apply_discount(price_cents, 10) * qty
    return price_cents * qty
