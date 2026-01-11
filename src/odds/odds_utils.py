from __future__ import annotations

import math
import unicodedata


def normalize_name(name: str) -> str:
    """Normalize player names for joining across datasets."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    # Remove diacritics (e.g. Sebastián -> Sebastian)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    a = float(american)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def as_decimal_odds(value: object, odds_format: str | None = None) -> float:
    """Parse odds as decimal odds.

    Args:
        value: odds value (decimal like 1.85, or american like +150 / -120)
        odds_format: 'decimal' (default) or 'american'
    """
    if value is None:
        raise ValueError("Missing odds value")

    if isinstance(value, str):
        v = value.strip()
        if v == "":
            raise ValueError("Missing odds value")
        # tolerate leading '+'
        try:
            value_f = float(v.replace("+", ""))
        except ValueError as e:  # noqa: BLE001
            raise ValueError(f"Invalid odds value: {value!r}") from e
    else:
        value_f = float(value)

    fmt = (odds_format or "decimal").strip().lower()
    if fmt in {"dec", "decimal"}:
        if value_f <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0 (got {value_f})")
        return value_f

    if fmt in {"amer", "american", "us"}:
        return american_to_decimal(value_f)

    raise ValueError(f"Unknown odds_format: {odds_format!r}")


def implied_prob_from_decimal(decimal_odds: float) -> float:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / d


def normalize_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    denom = float(p_a) + float(p_b)
    if denom <= 0 or math.isnan(denom):
        raise ValueError("Invalid implied probabilities")
    return float(p_a) / denom, float(p_b) / denom
