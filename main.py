from __future__ import annotations

import argparse
import os

from src.data_ingest.download_sackmann import download_sackmann_both, download_sackmann_matches
from src.analysis.simulate_australian_open import TournamentConfig, simulate_australian_open
from src.odds.compare_odds import OddsCompareConfig, compare_odds
from src.odds.fetch_theoddsapi import fetch_odds_theoddsapi
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

    fetch = sub.add_parser("fetch-odds", help="Fetch current tennis odds into data/raw/odds.csv")
    fetch.add_argument(
        "--upcoming-csv",
        default=None,
        help="Upcoming matches CSV (default: data/raw/upcoming_matches.csv)",
    )
    fetch.add_argument(
        "--out-csv",
        default=None,
        help="Write odds CSV here (default: data/raw/odds.csv)",
    )
    fetch.add_argument(
        "--regions",
        default=os.getenv("THE_ODDS_API_REGIONS", "us"),
        help="The Odds API regions param (default: us). Supported: us, us2, uk, au, eu",
    )
    fetch.add_argument(
        "--sport-keys",
        default=os.getenv("THE_ODDS_API_SPORT_KEYS", ""),
        help="Override The Odds API sport keys (comma-delimited). Also supports env THE_ODDS_API_SPORT_KEYS.",
    )

    odds = sub.add_parser("compare-odds", help="Compare predictions to market odds (CSV)")
    odds.add_argument(
        "--odds-csv",
        default=None,
        help="Path to odds CSV (default: data/raw/odds.csv)",
    )
    odds.add_argument(
        "--predictions-csv",
        default=None,
        help="Path to predictions CSV (default: data/outputs/predictions_latest.csv)",
    )
    odds.add_argument(
        "--out-csv",
        default=None,
        help="Write merged comparison CSV here (default: data/outputs/odds_comparison_latest.csv)",
    )
    odds.add_argument(
        "--no-vig-normalize",
        action="store_true",
        help="Do not normalize implied probs to remove vig (uses raw 1/odds).",
    )

    dl = sub.add_parser("download", help="Download free historical match data")
    dl.add_argument("--tour", choices=["atp", "wta", "both"], default="atp")
    dl.add_argument("--start-year", type=int, default=2015)
    dl.add_argument("--end-year", type=int, default=None)

    sim = sub.add_parser("ao", help="Simulate Australian Open outrights (pre-draw)")
    sim.add_argument("--tour", choices=["atp", "wta"], default="atp")
    sim.add_argument("--sims", type=int, default=5000)

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
    elif args.cmd == "fetch-odds":
        from pathlib import Path

        # Allow overriding regions without exposing full config surface.
        from src.odds.fetch_theoddsapi import TheOddsApiConfig

        cfg = TheOddsApiConfig(regions=args.regions)
        if args.sport_keys:
            os.environ["THE_ODDS_API_SPORT_KEYS"] = str(args.sport_keys)
        out = fetch_odds_theoddsapi(
            upcoming_csv=(None if args.upcoming_csv is None else Path(args.upcoming_csv)),
            out_csv=(None if args.out_csv is None else Path(args.out_csv)),
            config=cfg,
        )
        print(f"Wrote {out}")
    elif args.cmd == "compare-odds":
        from pathlib import Path

        cfg = OddsCompareConfig(normalize_market=not args.no_vig_normalize)
        out = compare_odds(
            predictions_path=(None if args.predictions_csv is None else Path(args.predictions_csv)),
            odds_path=(None if args.odds_csv is None else Path(args.odds_csv)),
            out_path=(None if args.out_csv is None else Path(args.out_csv)),
            config=cfg,
        )
        print(f"Wrote {out}")
    elif args.cmd == "download":
        if args.tour == "both":
            download_sackmann_both(start_year=args.start_year, end_year=args.end_year)
        else:
            download_sackmann_matches(tour=args.tour, start_year=args.start_year, end_year=args.end_year)
    elif args.cmd == "ao":
        tour = args.tour.upper()
        best_of = 5 if tour == "ATP" else 3
        cfg = TournamentConfig(tour=tour, best_of=best_of, sims=args.sims)
        simulate_australian_open(cfg)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
