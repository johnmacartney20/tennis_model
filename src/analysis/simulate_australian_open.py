from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


@dataclass(frozen=True)
class TournamentConfig:
    name: str = "Australian Open"
    surface: str = "Hard"
    best_of: int = 5
    field_size: int = 128
    sims: int = 5000
    random_seed: int = 7
    tour: str = "ATP"
    max_inactive_days: int = 365


def _implied_decimal_from_prob(p: float) -> float | None:
    if p <= 0:
        return None
    return 1.0 / p


def _load_model_artifact(model_path: Path) -> tuple[object, list[str]]:
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["feature_cols"]


def _build_player_table(paths: Path, surface: str, tour: str, max_inactive_days: int) -> pd.DataFrame:
    ratings = read_csv(paths / "elo_ratings.csv")
    overall = read_csv(paths / "player_strength_overall.csv")
    surf = read_csv(paths / "player_strength_surface.csv")
    matches = read_csv(paths / "matches.csv")

    ratings["tour"] = ratings["tour"].astype(str).str.upper()
    overall["tour"] = overall["tour"].astype(str).str.upper()
    surf["tour"] = surf["tour"].astype(str).str.upper()

    tour = tour.upper()

    ratings = ratings[ratings["tour"] == tour].copy()
    overall = overall[overall["tour"] == tour].copy()
    surf = surf[(surf["tour"] == tour) & (surf["surface"] == surface)].copy()

    # Filter to recently active players to avoid stale Elo from retired players.
    matches["tour"] = matches.get("tour", "UNK").astype(str).str.upper()
    matches = matches[matches["tour"] == tour].copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"])
    if not matches.empty:
        ref_date = matches["date"].max()
        cutoff = ref_date - pd.Timedelta(days=int(max_inactive_days))

        last_a = matches.groupby("player_a")["date"].max()
        last_b = matches.groupby("player_b")["date"].max()
        last = pd.concat([last_a, last_b]).groupby(level=0).max().rename("last_match_date")
        last = last.reset_index().rename(columns={"index": "player"})
    else:
        last = pd.DataFrame(columns=["player", "last_match_date"])
        cutoff = None

    df = ratings.merge(overall.drop(columns=["tour"], errors="ignore"), how="left", on="player")
    df = df.merge(surf.drop(columns=["tour", "surface"], errors="ignore"), how="left", on="player")
    df = df.merge(last, how="left", on="player")

    if cutoff is not None:
        df["last_match_date"] = pd.to_datetime(df["last_match_date"], errors="coerce")
        df = df[df["last_match_date"].notna() & (df["last_match_date"] >= cutoff)].copy()

    # Fill missing EMA values with neutral defaults.
    for c in df.columns:
        if c.endswith("_ema") or c.endswith("_ema_surface"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5)

    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    df = df.dropna(subset=["elo"]).sort_values("elo", ascending=False).reset_index(drop=True)

    return df


def _feature_row(
    feature_cols: list[str],
    surface: str,
    best_of: int,
    tour: str,
    player_a: pd.Series,
    player_b: pd.Series,
) -> pd.DataFrame:
    row: dict[str, object] = {}

    # Base
    if "tour" in feature_cols:
        row["tour"] = tour
    if "surface" in feature_cols:
        row["surface"] = surface
    if "best_of" in feature_cols:
        row["best_of"] = best_of

    # Diffs
    if "elo_diff" in feature_cols:
        row["elo_diff"] = float(player_a["elo"]) - float(player_b["elo"])

    # Some pipelines may not include ranks; keep neutral if present.
    if "rank_diff" in feature_cols:
        row["rank_diff"] = 0.0

    # Any other *_diff columns are assumed to be EMA diffs and are computed from the saved strength columns.
    for c in feature_cols:
        if not c.endswith("_diff"):
            continue
        if c in {"elo_diff", "rank_diff"}:
            continue

        if c.endswith("_ema_surface_diff"):
            metric = c.replace("_ema_surface_diff", "")
            col = f"{metric}_ema_surface"
            row[c] = float(player_a.get(col, 0.5)) - float(player_b.get(col, 0.5))
        elif c.endswith("_ema_diff"):
            metric = c.replace("_ema_diff", "")
            col = f"{metric}_ema"
            row[c] = float(player_a.get(col, 0.5)) - float(player_b.get(col, 0.5))
        else:
            # Unknown diff feature; keep neutral
            row[c] = 0.0

    # Ensure all feature cols exist
    for c in feature_cols:
        if c not in row:
            row[c] = np.nan

    return pd.DataFrame([row], columns=feature_cols)


def simulate_australian_open(
    config: TournamentConfig | None = None,
    model_path: Path | None = None,
) -> Path:
    paths = get_paths()
    config = config or TournamentConfig()

    # Require that build_features() has been run on up-to-date historical matches.
    interim = paths.interim_dir
    required = [
        interim / "elo_ratings.csv",
        interim / "player_strength_overall.csv",
        interim / "player_strength_surface.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run: python main.py features (or pipeline) after downloading matches."
            )

    model_path = model_path or (paths.models_dir / "win_model.joblib")
    model, feature_cols = _load_model_artifact(model_path)

    players = _build_player_table(
        interim, surface=config.surface, tour=config.tour, max_inactive_days=config.max_inactive_days
    )
    field = players.head(config.field_size).reset_index(drop=True)

    if len(field) < 8:
        raise RuntimeError("Not enough players in field to simulate tournament")

    rng = np.random.default_rng(config.random_seed)

    win_counts: dict[str, int] = {p: 0 for p in field["player"].tolist()}

    # Monte Carlo tournament simulation with random bracket (pre-draw).
    # If you provide the actual draw matchups, we can replace this with a bracket-accurate simulation.
    for _ in range(config.sims):
        bracket = field.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        current = bracket
        while len(current) > 1:
            next_round: list[pd.Series] = []
            for i in range(0, len(current), 2):
                a = current.iloc[i]
                b = current.iloc[i + 1]

                X = _feature_row(feature_cols, config.surface, config.best_of, config.tour, a, b)
                p_a = float(model.predict_proba(X)[0, 1])
                # Numerical safety
                p_a = min(max(p_a, 1e-6), 1.0 - 1e-6)

                winner = a if rng.random() < p_a else b
                next_round.append(winner)
            current = pd.DataFrame(next_round)

        champ = str(current.iloc[0]["player"])
        win_counts[champ] = win_counts.get(champ, 0) + 1

    out = pd.DataFrame(
        {
            "player": list(win_counts.keys()),
            "win_prob": [c / config.sims for c in win_counts.values()],
        }
    ).sort_values("win_prob", ascending=False)

    out["implied_decimal_odds"] = out["win_prob"].apply(_implied_decimal_from_prob)
    out["tournament"] = config.name
    out["tour"] = config.tour
    out["surface"] = config.surface
    out["best_of"] = config.best_of
    out["sims"] = config.sims

    tour_slug = config.tour.lower()
    out_path = paths.outputs_dir / f"australian_open_{tour_slug}_outrights.csv"
    write_csv(out.reset_index(drop=True), out_path)
    return out_path


if __name__ == "__main__":
    p = simulate_australian_open()
    print(f"Wrote {p}")
