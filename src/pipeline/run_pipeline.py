from __future__ import annotations

from src.data_ingest.import_tennis_data import ingest
from src.feature_engineering.build_features import build_features
from src.modeling.predict_upcoming import predict_upcoming
from src.modeling.train_model import train_model


def run_pipeline() -> None:
    ingest()
    build_features()
    train_model()
    predict_upcoming()
