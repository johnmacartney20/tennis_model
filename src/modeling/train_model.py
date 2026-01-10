from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import get_paths
from src.utils.helpers import read_csv


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict[str, float]
    feature_cols: list[str]


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column for time split")

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    cut = max(1, int(np.floor(n * (1.0 - test_frac))))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_model(features_path: Path | None = None) -> TrainResult:
    paths = get_paths()
    features_path = features_path or (paths.features_dir / "training_match_features.csv")

    df = read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required = {"surface", "best_of", "elo_diff", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training features missing required columns: {sorted(missing)}")

    train_df, test_df = _time_split(df, test_frac=0.25)

    y_train = train_df["y"].astype(int)
    y_test = test_df["y"].astype(int)

    # Model features (keep player names out for now; can add player embeddings later)
    categorical = ["surface"]

    # Auto-include engineered numeric diffs
    numeric = ["best_of"]
    numeric += [c for c in df.columns if c.endswith("_diff")]
    numeric = [c for c in numeric if c in df.columns]

    feature_cols = categorical + numeric

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000)
    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {}
    if len(np.unique(y_test)) > 1:
        metrics["auc"] = float(roc_auc_score(y_test, proba_test))
    metrics["log_loss"] = float(log_loss(y_test, proba_test, labels=[0, 1]))
    metrics["n_train"] = float(len(train_df))
    metrics["n_test"] = float(len(test_df))

    model_path = paths.models_dir / "win_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols}
    joblib.dump(artifact, model_path)

    return TrainResult(model_path=model_path, metrics=metrics, feature_cols=feature_cols)
