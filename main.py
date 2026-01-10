from __future__ import annotations

import argparse

from src.data_ingest.download_sackmann import download_sackmann_both, download_sackmann_matches
from src.pipeline.run_pipeline import run_pipeline
from src.data_ingest.import_tennis_data import ingest
from src.feature_engineering.build_features import build_features
from src.modeling.train_model import train_model
from src.modeling.predict_upcoming import predict_upcoming


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pro tennis prediction pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("predict")
    sub.add_parser("pipeline")

    dl = sub.add_parser("download", help="Download free historical match data")
    dl.add_argument("--tour", choices=["atp", "wta", "both"], default="atp")
    dl.add_argument("--start-year", type=int, default=2015)
    dl.add_argument("--end-year", type=int, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "features":
        build_features()
    elif args.cmd == "train":
        train_model()
    elif args.cmd == "predict":
        predict_upcoming()
    elif args.cmd == "pipeline":
        run_pipeline()
    elif args.cmd == "download":
        if args.tour == "both":
            download_sackmann_both(start_year=args.start_year, end_year=args.end_year)
        else:
            download_sackmann_matches(tour=args.tour, start_year=args.start_year, end_year=args.end_year)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
