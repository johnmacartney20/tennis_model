from __future__ import annotations

import os

from src.data_ingest.import_tennis_data import ingest
from src.feature_engineering.build_features import build_features
from src.modeling.predict_upcoming import predict_upcoming
from src.modeling.train_model import train_model


def _maybe_run_odds_step() -> None:
    # Keep the pipeline fully offline by default.
    # If a key is set, fetch latest odds and write a merged comparison CSV.
    if not os.getenv("THE_ODDS_API_KEY"):
        return

    try:
        from src.odds.compare_odds import compare_odds
        from src.odds.fetch_theoddsapi import TheOddsApiConfig, fetch_odds_theoddsapi

        cfg = TheOddsApiConfig(regions=os.getenv("THE_ODDS_API_REGIONS", "us"))
        fetch_odds_theoddsapi(config=cfg)
        compare_odds()
    except Exception as e:  # noqa: BLE001
        # Don't fail the core model pipeline if odds fetching breaks.
        print(f"[odds] Skipping odds fetch/compare due to error: {e}")


def run_pipeline() -> None:
    ingest()
    build_features()
    train_model()
    predict_upcoming()
    _maybe_run_odds_step()
