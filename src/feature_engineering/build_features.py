from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering.elo import add_elo_features
from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


def _safe_div(num: float, den: float) -> float | None:
    try:
        if den is None or pd.isna(den) or float(den) == 0.0:
            return None
        if num is None or pd.isna(num):
            return None
        return float(num) / float(den)
    except Exception:  # noqa: BLE001
        return None


def _ema_update(old: float | None, value: float | None, alpha: float) -> float | None:
    if value is None:
        return old
    if old is None:
        return value
    return alpha * value + (1.0 - alpha) * old


def _standardize_surface(surface: str) -> str:
    if not isinstance(surface, str) or not surface.strip():
        return "Unknown"
    s = surface.strip().lower()
    mapping = {
        "hard": "Hard",
        "clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
    }
    return mapping.get(s, surface.strip().title())


def build_features(
    interim_matches_path: Path | None = None,
    interim_upcoming_path: Path | None = None,
) -> tuple[Path, Path]:
    """Create model-ready training and upcoming feature tables."""
    paths = get_paths()

    interim_matches_path = interim_matches_path or (paths.interim_dir / "matches.csv")
    interim_upcoming_path = interim_upcoming_path or (paths.interim_dir / "upcoming_matches.csv")

    matches = read_csv(interim_matches_path)
    upcoming = read_csv(interim_upcoming_path)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    if matches["date"].isna().any():
        raise ValueError("Some historical match dates could not be parsed")

    upcoming["date"] = pd.to_datetime(upcoming["date"], errors="coerce")
    if upcoming["date"].isna().any():
        raise ValueError("Some upcoming match dates could not be parsed")

    matches = matches.sort_values(["date"]).reset_index(drop=True)

    matches["surface"] = matches["surface"].apply(_standardize_surface)
    upcoming["surface"] = upcoming["surface"].apply(_standardize_surface)

    # Ensure optional Sackmann stats exist (keeps offline sample working)
    stat_cols = [
        "svpt_a",
        "svpt_b",
        "first_in_a",
        "first_in_b",
        "first_won_a",
        "first_won_b",
        "second_won_a",
        "second_won_b",
        "ace_a",
        "ace_b",
        "df_a",
        "df_b",
        "bp_saved_a",
        "bp_saved_b",
        "bp_faced_a",
        "bp_faced_b",
    ]
    for c in stat_cols:
        if c not in matches.columns:
            matches[c] = pd.NA

    # Rolling strength features (EMA) computed strictly pre-match (no leakage)
    alpha = 0.15
    default_rate = 0.5

    metrics = [
        "serve_pw",  # (first_won+second_won)/svpt
        "first_in",  # first_in/svpt
        "first_win",  # first_won/first_in
        "second_win",  # second_won/(svpt-first_in)
        "ace_rate",  # ace/svpt
        "df_rate",  # df/svpt
        "bp_saved",  # bp_saved/bp_faced
        "return_pw",  # 1 - opponent serve_pw
    ]

    # dicts keyed by player or (player,surface)
    overall: dict[str, dict[str, float | None]] = {}
    by_surface: dict[tuple[str, str], dict[str, float | None]] = {}

    def _get_state(player: str, surface: str) -> tuple[dict[str, float | None], dict[str, float | None]]:
        o = overall.setdefault(player, {m: None for m in metrics})
        s = by_surface.setdefault((player, surface), {m: None for m in metrics})
        return o, s

    def _rate_defaults(state: dict[str, float | None]) -> dict[str, float]:
        return {m: (default_rate if state[m] is None else float(state[m])) for m in metrics}

    a_overall_pre: list[dict[str, float]] = []
    b_overall_pre: list[dict[str, float]] = []
    a_surface_pre: list[dict[str, float]] = []
    b_surface_pre: list[dict[str, float]] = []

    # iterate row-wise in date order
    for row in matches.itertuples(index=False):
        surface = _standardize_surface(getattr(row, "surface"))
        pa = getattr(row, "player_a")
        pb = getattr(row, "player_b")

        o_a, s_a = _get_state(pa, surface)
        o_b, s_b = _get_state(pb, surface)

        a_overall_pre.append(_rate_defaults(o_a))
        b_overall_pre.append(_rate_defaults(o_b))
        a_surface_pre.append(_rate_defaults(s_a))
        b_surface_pre.append(_rate_defaults(s_b))

        # Compute match-level rates from stats (if present) to update EMAs
        svpt_a = getattr(row, "svpt_a")
        svpt_b = getattr(row, "svpt_b")
        first_in_a = getattr(row, "first_in_a")
        first_in_b = getattr(row, "first_in_b")
        first_won_a = getattr(row, "first_won_a")
        first_won_b = getattr(row, "first_won_b")
        second_won_a = getattr(row, "second_won_a")
        second_won_b = getattr(row, "second_won_b")
        ace_a = getattr(row, "ace_a")
        ace_b = getattr(row, "ace_b")
        df_a = getattr(row, "df_a")
        df_b = getattr(row, "df_b")
        bp_saved_a = getattr(row, "bp_saved_a")
        bp_saved_b = getattr(row, "bp_saved_b")
        bp_faced_a = getattr(row, "bp_faced_a")
        bp_faced_b = getattr(row, "bp_faced_b")

        def _num(x: object) -> float | None:
            return None if x is None or pd.isna(x) else float(x)

        svpt_a_n = _num(svpt_a)
        svpt_b_n = _num(svpt_b)
        first_in_a_n = _num(first_in_a)
        first_in_b_n = _num(first_in_b)
        first_won_a_n = _num(first_won_a)
        first_won_b_n = _num(first_won_b)
        second_won_a_n = _num(second_won_a)
        second_won_b_n = _num(second_won_b)
        ace_a_n = _num(ace_a)
        ace_b_n = _num(ace_b)
        df_a_n = _num(df_a)
        df_b_n = _num(df_b)
        bp_saved_a_n = _num(bp_saved_a)
        bp_saved_b_n = _num(bp_saved_b)
        bp_faced_a_n = _num(bp_faced_a)
        bp_faced_b_n = _num(bp_faced_b)

        serve_pw_a = _safe_div((first_won_a_n or 0.0) + (second_won_a_n or 0.0), svpt_a_n)
        serve_pw_b = _safe_div((first_won_b_n or 0.0) + (second_won_b_n or 0.0), svpt_b_n)

        # If we can't compute serve_pw, skip updating derived return_pw too.
        return_pw_a = (None if serve_pw_b is None else 1.0 - serve_pw_b)
        return_pw_b = (None if serve_pw_a is None else 1.0 - serve_pw_a)

        rates_a = {
            "serve_pw": serve_pw_a,
            "first_in": _safe_div(first_in_a_n, svpt_a_n),
            "first_win": _safe_div(first_won_a_n, first_in_a_n),
            "second_win": _safe_div(
                second_won_a_n,
                None
                if svpt_a_n is None
                else (svpt_a_n - (first_in_a_n or 0.0) if (svpt_a_n - (first_in_a_n or 0.0)) > 0.0 else None),
            ),
            "ace_rate": _safe_div(ace_a_n, svpt_a_n),
            "df_rate": _safe_div(df_a_n, svpt_a_n),
            "bp_saved": _safe_div(bp_saved_a_n, bp_faced_a_n),
            "return_pw": return_pw_a,
        }
        rates_b = {
            "serve_pw": serve_pw_b,
            "first_in": _safe_div(first_in_b_n, svpt_b_n),
            "first_win": _safe_div(first_won_b_n, first_in_b_n),
            "second_win": _safe_div(
                second_won_b_n,
                None
                if svpt_b_n is None
                else (svpt_b_n - (first_in_b_n or 0.0) if (svpt_b_n - (first_in_b_n or 0.0)) > 0.0 else None),
            ),
            "ace_rate": _safe_div(ace_b_n, svpt_b_n),
            "df_rate": _safe_div(df_b_n, svpt_b_n),
            "bp_saved": _safe_div(bp_saved_b_n, bp_faced_b_n),
            "return_pw": return_pw_b,
        }

        for m in metrics:
            o_a[m] = _ema_update(o_a[m], rates_a[m], alpha)
            s_a[m] = _ema_update(s_a[m], rates_a[m], alpha)
            o_b[m] = _ema_update(o_b[m], rates_b[m], alpha)
            s_b[m] = _ema_update(s_b[m], rates_b[m], alpha)

    matches_with_elo, ratings_df = add_elo_features(matches)

    # Attach EMA feature columns to matches_with_elo (same row order as matches)
    ema_cols = []
    for m in metrics:
        matches_with_elo[f"a_{m}_ema"] = [d[m] for d in a_overall_pre]
        matches_with_elo[f"b_{m}_ema"] = [d[m] for d in b_overall_pre]
        matches_with_elo[f"a_{m}_ema_surface"] = [d[m] for d in a_surface_pre]
        matches_with_elo[f"b_{m}_ema_surface"] = [d[m] for d in b_surface_pre]
        matches_with_elo[f"{m}_ema_diff"] = matches_with_elo[f"a_{m}_ema"] - matches_with_elo[f"b_{m}_ema"]
        matches_with_elo[f"{m}_ema_surface_diff"] = matches_with_elo[f"a_{m}_ema_surface"] - matches_with_elo[f"b_{m}_ema_surface"]
        ema_cols.extend(
            [
                f"a_{m}_ema",
                f"b_{m}_ema",
                f"a_{m}_ema_surface",
                f"b_{m}_ema_surface",
                f"{m}_ema_diff",
                f"{m}_ema_surface_diff",
            ]
        )

    # Persist latest Elo table for inspection/debugging.
    elo_path = paths.interim_dir / "elo_ratings.csv"
    write_csv(ratings_df, elo_path)

    # Build training set
    train = matches_with_elo.copy()
    train["y"] = (train["winner"] == "A").astype(int)

    for col in ["rank_a", "rank_b"]:
        if col not in train.columns:
            train[col] = pd.NA

    train["rank_a"] = pd.to_numeric(train["rank_a"], errors="coerce")
    train["rank_b"] = pd.to_numeric(train["rank_b"], errors="coerce")
    train["rank_diff"] = train["rank_b"] - train["rank_a"]

    # Minimal feature set (keep it easy to extend)
    feature_cols = [
        "date",
        "surface",
        "best_of",
        "player_a",
        "player_b",
        "elo_a_pre",
        "elo_b_pre",
        "elo_diff",
        "rank_a",
        "rank_b",
        "rank_diff",
        *ema_cols,
        "y",
    ]

    train_features = train[[c for c in feature_cols if c in train.columns]].copy()

    train_out = paths.features_dir / "training_match_features.csv"
    write_csv(train_features, train_out)

    # Upcoming features: attach latest Elo ratings (pre-match)
    elo_map = dict(zip(ratings_df["player"], ratings_df["elo"]))

    # Build latest EMA tables from the post-loop state
    overall_latest = (
        pd.DataFrame(
            [
                {"player": p, **{f"{m}_ema": (default_rate if st[m] is None else float(st[m])) for m in metrics}}
                for p, st in overall.items()
            ]
        )
        if overall
        else pd.DataFrame(columns=["player"] + [f"{m}_ema" for m in metrics])
    )
    surface_latest = (
        pd.DataFrame(
            [
                {
                    "player": p,
                    "surface": s,
                    **{f"{m}_ema_surface": (default_rate if st[m] is None else float(st[m])) for m in metrics},
                }
                for (p, s), st in by_surface.items()
            ]
        )
        if by_surface
        else pd.DataFrame(columns=["player", "surface"] + [f"{m}_ema_surface" for m in metrics])
    )

    def _elo_for(player: str) -> float:
        return float(elo_map.get(player, 1500.0))

    upcoming_feat = upcoming.copy()
    upcoming_feat["elo_a_pre"] = upcoming_feat["player_a"].apply(_elo_for)
    upcoming_feat["elo_b_pre"] = upcoming_feat["player_b"].apply(_elo_for)
    upcoming_feat["elo_diff"] = upcoming_feat["elo_a_pre"] - upcoming_feat["elo_b_pre"]

    # Join overall EMA features
    upcoming_feat = upcoming_feat.merge(
        overall_latest.add_prefix("a_").rename(columns={"a_player": "player_a"}),
        how="left",
        on="player_a",
    )
    upcoming_feat = upcoming_feat.merge(
        overall_latest.add_prefix("b_").rename(columns={"b_player": "player_b"}),
        how="left",
        on="player_b",
    )

    # Join surface EMA features
    upcoming_feat = upcoming_feat.merge(
        surface_latest.add_prefix("a_").rename(columns={"a_player": "player_a", "a_surface": "surface"}),
        how="left",
        on=["player_a", "surface"],
    )
    upcoming_feat = upcoming_feat.merge(
        surface_latest.add_prefix("b_").rename(columns={"b_player": "player_b", "b_surface": "surface"}),
        how="left",
        on=["player_b", "surface"],
    )

    # Fill missing EMA values with defaults (new/unknown players)
    for m in metrics:
        a_overall_col = f"a_{m}_ema"
        b_overall_col = f"b_{m}_ema"
        a_surface_col = f"a_{m}_ema_surface"
        b_surface_col = f"b_{m}_ema_surface"

        if a_overall_col not in upcoming_feat.columns:
            upcoming_feat[a_overall_col] = default_rate
        else:
            upcoming_feat[a_overall_col] = pd.to_numeric(upcoming_feat[a_overall_col], errors="coerce").fillna(default_rate)

        if b_overall_col not in upcoming_feat.columns:
            upcoming_feat[b_overall_col] = default_rate
        else:
            upcoming_feat[b_overall_col] = pd.to_numeric(upcoming_feat[b_overall_col], errors="coerce").fillna(default_rate)

        if a_surface_col not in upcoming_feat.columns:
            upcoming_feat[a_surface_col] = upcoming_feat[a_overall_col]
        else:
            upcoming_feat[a_surface_col] = pd.to_numeric(upcoming_feat[a_surface_col], errors="coerce").fillna(upcoming_feat[a_overall_col])

        if b_surface_col not in upcoming_feat.columns:
            upcoming_feat[b_surface_col] = upcoming_feat[b_overall_col]
        else:
            upcoming_feat[b_surface_col] = pd.to_numeric(upcoming_feat[b_surface_col], errors="coerce").fillna(upcoming_feat[b_overall_col])

        upcoming_feat[f"{m}_ema_diff"] = upcoming_feat[f"a_{m}_ema"] - upcoming_feat[f"b_{m}_ema"]
        upcoming_feat[f"{m}_ema_surface_diff"] = upcoming_feat[f"a_{m}_ema_surface"] - upcoming_feat[f"b_{m}_ema_surface"]

    for col in ["rank_a", "rank_b"]:
        if col not in upcoming_feat.columns:
            upcoming_feat[col] = pd.NA

    upcoming_feat["rank_a"] = pd.to_numeric(upcoming_feat["rank_a"], errors="coerce")
    upcoming_feat["rank_b"] = pd.to_numeric(upcoming_feat["rank_b"], errors="coerce")
    upcoming_feat["rank_diff"] = upcoming_feat["rank_b"] - upcoming_feat["rank_a"]

    upcoming_cols = [
        "date",
        "tournament",
        "round",
        "surface",
        "best_of",
        "player_a",
        "player_b",
        "elo_a_pre",
        "elo_b_pre",
        "elo_diff",
        "rank_a",
        "rank_b",
        "rank_diff",
        *ema_cols,
    ]

    upcoming_out = paths.features_dir / "upcoming_match_features.csv"
    write_csv(upcoming_feat[[c for c in upcoming_cols if c in upcoming_feat.columns]], upcoming_out)

    return train_out, upcoming_out
