from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.odds.odds_utils import (
    as_decimal_odds,
    implied_prob_from_decimal,
    normalize_name,
    normalize_two_way,
)
from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


@dataclass(frozen=True)
class OddsCompareConfig:
    normalize_market: bool = True


def _build_match_key(
    tournament: object,
    player_a: object,
    player_b: object,
) -> tuple[str, str, str]:
    t = normalize_name(tournament)
    a = normalize_name(player_a)
    b = normalize_name(player_b)
    return t, a, b


def _build_pair_key(player_a: object, player_b: object) -> tuple[str, str]:
    return normalize_name(player_a), normalize_name(player_b)


def compare_odds(
    predictions_path: Path | None = None,
    odds_path: Path | None = None,
    out_path: Path | None = None,
    config: OddsCompareConfig | None = None,
) -> Path:
    """Compare model predictions to market odds.

    Expected odds CSV columns (match-level):
        - date (optional)
        - tournament (optional)
        - player_a, player_b
        - odds_a, odds_b
        - odds_format (optional: 'decimal' or 'american'; default decimal)
        - book (optional)

    Writes a merged CSV with implied market probabilities, edges, EV, and Kelly.
    """
    paths = get_paths()
    config = config or OddsCompareConfig()

    predictions_path = predictions_path or (paths.outputs_dir / "predictions_latest.csv")
    odds_path = odds_path or (paths.raw_dir / "odds.csv")
    out_path = out_path or (paths.outputs_dir / "odds_comparison_latest.csv")

    preds = read_csv(predictions_path)
    odds = read_csv(odds_path)

    required_odds = {"player_a", "player_b", "odds_a", "odds_b"}
    missing = required_odds - set(odds.columns)
    if missing:
        raise ValueError(f"Odds CSV missing required columns: {sorted(missing)}")

    # Store odds offers keyed both strictly (tournament+players) and loosely (players only),
    # because many odds feeds don't provide tournament names.
    offers_by_match: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    offers_by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}

    for row in odds.to_dict(orient="records"):
        tournament = row.get("tournament", "")
        t, a, b = _build_match_key(tournament, row.get("player_a"), row.get("player_b"))

        fmt = row.get("odds_format", "decimal")
        da = as_decimal_odds(row.get("odds_a"), fmt)
        db = as_decimal_odds(row.get("odds_b"), fmt)

        base = dict(row)
        base["odds_a_decimal"] = da
        base["odds_b_decimal"] = db

        offers_by_match.setdefault((t, a, b), []).append(base)
        offers_by_pair.setdefault((a, b), []).append(base)

        # Also store reverse so we can match predictions regardless of order.
        rev = dict(base)
        rev["player_a"], rev["player_b"] = base.get("player_b"), base.get("player_a")
        rev["odds_a_decimal"], rev["odds_b_decimal"] = db, da
        offers_by_match.setdefault((t, b, a), []).append(rev)
        offers_by_pair.setdefault((b, a), []).append(rev)

    out_rows: list[dict[str, object]] = []

    for p in preds.to_dict(orient="records"):
        tournament = p.get("tournament", "")
        t, a, b = _build_match_key(tournament, p.get("player_a"), p.get("player_b"))
        offers = offers_by_match.get((t, a, b))
        if not offers:
            offers = offers_by_pair.get(_build_pair_key(p.get("player_a"), p.get("player_b")))
        if not offers:
            continue

        p_a = float(p.get("win_prob_a"))
        p_b = float(p.get("win_prob_b"))

        def _kelly(p_win: float, dec: float) -> float:
            b_ = dec - 1.0
            if b_ <= 0:
                return 0.0
            f = (p_win * dec - 1.0) / b_
            return max(0.0, float(f))

        for o in offers:
            odds_a = float(o["odds_a_decimal"])
            odds_b = float(o["odds_b_decimal"])

            imp_a = implied_prob_from_decimal(odds_a)
            imp_b = implied_prob_from_decimal(odds_b)
            if config.normalize_market:
                mkt_a, mkt_b = normalize_two_way(imp_a, imp_b)
            else:
                mkt_a, mkt_b = imp_a, imp_b

            ev_a = p_a * odds_a - 1.0
            ev_b = p_b * odds_b - 1.0
            k_a = _kelly(p_a, odds_a)
            k_b = _kelly(p_b, odds_b)

            pick = None
            if ev_a > 0 and ev_b > 0:
                pick = "A" if ev_a >= ev_b else "B"
            elif ev_a > 0:
                pick = "A"
            elif ev_b > 0:
                pick = "B"

            out = dict(p)
            out.update({
                "book": o.get("book", pd.NA),
                "odds_a_decimal": odds_a,
                "odds_b_decimal": odds_b,
                "market_prob_a": mkt_a,
                "market_prob_b": mkt_b,
                "edge_a": p_a - mkt_a,
                "edge_b": p_b - mkt_b,
                "ev_a": ev_a,
                "ev_b": ev_b,
                "kelly_a": k_a,
                "kelly_b": k_b,
                "recommended_side": pick,
            })
            out_rows.append(out)

    if not out_rows:
        raise RuntimeError(
            "No matches joined between predictions and odds. "
            "Check tournament/player name spelling and odds CSV format."
        )

    out_df = pd.DataFrame(out_rows)
    sort_cols = [c for c in ["date", "tournament", "round"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, na_position="last")

    write_csv(out_df.reset_index(drop=True), out_path)
    return out_path
