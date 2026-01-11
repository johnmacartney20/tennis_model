from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


REQUIRED_MATCH_COLS = {
    "date",
    "surface",
    "best_of",
    "player_a",
    "player_b",
    "winner",
}


def _validate_matches(df: pd.DataFrame) -> None:
    missing = REQUIRED_MATCH_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Historical matches missing required columns: {sorted(missing)}")

    if not set(df["winner"].dropna().unique()).issubset({"A", "B"}):
        bad = sorted(set(df["winner"].dropna().unique()) - {"A", "B"})
        raise ValueError(f"Invalid winner values (expected 'A'/'B'): {bad}")


def ingest(
    matches_path: Path | None = None,
    upcoming_path: Path | None = None,
) -> tuple[Path, Path]:
    """Standardize raw inputs into interim CSVs.

    If no raw files are provided, uses the bundled sample files so the
    pipeline runs fully offline.

    Returns:
        (interim_matches_path, interim_upcoming_path)
    """
    paths = get_paths()

    matches_path = matches_path or (paths.raw_dir / "matches.csv")
    upcoming_path = upcoming_path or (paths.raw_dir / "upcoming_matches.csv")

    if not matches_path.exists():
        matches_path = paths.raw_dir / "matches_sample.csv"

    if not upcoming_path.exists():
        upcoming_path = paths.raw_dir / "upcoming_matches_sample.csv"

    matches = read_csv(matches_path)
    upcoming = read_csv(upcoming_path)

    _validate_matches(matches)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce").dt.date
    if matches["date"].isna().any():
        raise ValueError("Some historical match dates could not be parsed")

    upcoming["date"] = pd.to_datetime(upcoming["date"], errors="coerce").dt.date
    if upcoming["date"].isna().any():
        raise ValueError("Some upcoming match dates could not be parsed")

    matches = matches.sort_values(["date"]).reset_index(drop=True)

    # Add tour (ATP/WTA) if missing.
    if "tour" not in matches.columns:
        if "match_id" in matches.columns:
            matches["tour"] = (
                matches["match_id"].astype(str).str.split(":", n=1).str[0].str.upper()
            )
            matches.loc[~matches["tour"].isin(["ATP", "WTA"]), "tour"] = "ATP"
        else:
            matches["tour"] = "ATP"

    if "tour" not in upcoming.columns:
        upcoming["tour"] = "ATP"
    upcoming["tour"] = upcoming["tour"].astype(str).str.upper().where(upcoming["tour"].notna(), "ATP")
    upcoming.loc[~upcoming["tour"].isin(["ATP", "WTA"]), "tour"] = "ATP"

    interim_matches_path = paths.interim_dir / "matches.csv"
    interim_upcoming_path = paths.interim_dir / "upcoming_matches.csv"

    write_csv(matches, interim_matches_path)
    write_csv(upcoming, interim_upcoming_path)

    return interim_matches_path, interim_upcoming_path
