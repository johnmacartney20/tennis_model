from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


def predict_upcoming(
    upcoming_features_path: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    paths = get_paths()
    upcoming_features_path = upcoming_features_path or (paths.features_dir / "upcoming_match_features.csv")
    model_path = model_path or (paths.models_dir / "win_model.joblib")

    df = read_csv(upcoming_features_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = pd.NA

    X = df[feature_cols]
    proba_a = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["win_prob_a"] = proba_a
    out["win_prob_b"] = 1.0 - out["win_prob_a"]

    keep = [
        "date",
        "tournament",
        "round",
        "surface",
        "best_of",
        "player_a",
        "player_b",
        "win_prob_a",
        "win_prob_b",
    ]
    keep = [c for c in keep if c in out.columns]

    out_path = paths.outputs_dir / "predictions_latest.csv"
    write_csv(out[keep].sort_values(["date", "tournament", "round"], na_position="last"), out_path)

    return out_path
