from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    base_rating: float = 1500.0
    k_best_of_3: float = 32.0
    k_best_of_5: float = 40.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def add_elo_features(matches: pd.DataFrame, config: EloConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Elo pre-match features and compute final ratings.

    Expects `matches` sorted by date (ascending) and containing:
    - player_a, player_b, winner, best_of

    Returns:
        (matches_with_elo, ratings_df)
    """
    config = config or EloConfig()

    required = {"player_a", "player_b", "winner", "best_of"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Matches missing required columns for Elo: {sorted(missing)}")

    ratings: dict[str, float] = {}

    elo_a_pre: list[float] = []
    elo_b_pre: list[float] = []
    elo_diff: list[float] = []

    for row in matches.itertuples(index=False):
        player_a = getattr(row, "player_a")
        player_b = getattr(row, "player_b")
        winner = getattr(row, "winner")
        best_of = int(getattr(row, "best_of"))

        ra = ratings.get(player_a, config.base_rating)
        rb = ratings.get(player_b, config.base_rating)

        elo_a_pre.append(ra)
        elo_b_pre.append(rb)
        elo_diff.append(ra - rb)

        expected_a = _expected_score(ra, rb)
        score_a = 1.0 if winner == "A" else 0.0

        k = config.k_best_of_5 if best_of == 5 else config.k_best_of_3

        ra_new = ra + k * (score_a - expected_a)
        rb_new = rb + k * ((1.0 - score_a) - (1.0 - expected_a))

        ratings[player_a] = ra_new
        ratings[player_b] = rb_new

    out = matches.copy()
    out["elo_a_pre"] = np.array(elo_a_pre, dtype=float)
    out["elo_b_pre"] = np.array(elo_b_pre, dtype=float)
    out["elo_diff"] = np.array(elo_diff, dtype=float)

    ratings_df = (
        pd.DataFrame({"player": list(ratings.keys()), "elo": list(ratings.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )

    return out, ratings_df
