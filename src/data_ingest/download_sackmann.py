from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import get_paths
from src.utils.helpers import write_csv


@dataclass(frozen=True)
class SackmannDownloadConfig:
    timeout_seconds: int = 60
    user_agent: str = "tennis_model/0.1 (research; contact: none)"


def _stable_swap_flag(match_key: str) -> bool:
    """Deterministically decide whether to swap A/B.

    We avoid Python's built-in hash because it is salted per-process.
    """
    digest = hashlib.md5(match_key.encode("utf-8"), usedforsecurity=False).digest()
    return bool(digest[0] & 1)


def _build_url(tour: str, year: int) -> str:
    tour = tour.lower()
    if tour == "atp":
        return f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    if tour == "wta":
        return f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
    raise ValueError("tour must be one of: atp, wta")


def download_sackmann_matches(
    tour: str = "atp",
    start_year: int = 2015,
    end_year: int | None = None,
    out_path: Path | None = None,
    config: SackmannDownloadConfig | None = None,
) -> Path:
    """Download free match results from Jeff Sackmann's tennis_atp/tennis_wta repos.

    Produces a standardized `matches.csv` compatible with this repo.

    Notes:
    - These datasets include match outcomes and many match-level stats.
    - Upcoming matches are not provided; you still supply `upcoming_matches.csv`.
    """
    config = config or SackmannDownloadConfig()
    paths = get_paths()

    if end_year is None:
        end_year = pd.Timestamp.utcnow().year

    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    out_path = out_path or (paths.raw_dir / "matches.csv")

    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent})

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for year in range(start_year, end_year + 1):
        url = _build_url(tour, year)
        try:
            resp = session.get(url, timeout=config.timeout_seconds)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            df["source_tour"] = tour.upper()
            df["source_year"] = year
            frames.append(df)
        except Exception as e:  # noqa: BLE001 - keep downloader robust
            errors.append(f"{year}: {type(e).__name__}: {e}")

    if not frames:
        extra = f" Errors: {errors[:3]}" if errors else ""
        raise RuntimeError(f"No data downloaded for {tour} {start_year}-{end_year}.{extra}")

    raw = pd.concat(frames, ignore_index=True)

    needed = {
        "tourney_id",
        "tourney_name",
        "surface",
        "tourney_date",
        "match_num",
        "best_of",
        "round",
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_ace",
        "w_df",
        "w_bpSaved",
        "w_bpFaced",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_ace",
        "l_df",
        "l_bpSaved",
        "l_bpFaced",
    }
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"Unexpected source schema; missing columns: {sorted(missing)}")

    # Standardize date
    raw["date"] = pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["date"] = raw["date"].dt.date

    # Create deterministic match keys
    raw["match_key"] = (
        raw["source_tour"].astype(str)
        + ":"
        + raw["tourney_id"].astype(str)
        + ":"
        + raw["match_num"].astype(str)
    )

    swap = raw["match_key"].apply(_stable_swap_flag)

    out = pd.DataFrame()
    out["match_id"] = raw["match_key"]
    out["date"] = raw["date"]
    out["tournament"] = raw["tourney_name"]
    out["surface"] = raw["surface"].fillna("Unknown")
    out["best_of"] = pd.to_numeric(raw["best_of"], errors="coerce").fillna(3).astype(int)
    out["round"] = raw["round"].fillna("")

    # Assign A/B with deterministic swapping to avoid label leakage.
    out["player_a"] = raw["winner_name"].where(~swap, raw["loser_name"])
    out["player_b"] = raw["loser_name"].where(~swap, raw["winner_name"])
    out["winner"] = pd.Series("A", index=raw.index).where(~swap, "B")

    out["rank_a"] = raw["winner_rank"].where(~swap, raw["loser_rank"])
    out["rank_b"] = raw["loser_rank"].where(~swap, raw["winner_rank"])

    # Clean ranks
    out["rank_a"] = pd.to_numeric(out["rank_a"], errors="coerce")
    out["rank_b"] = pd.to_numeric(out["rank_b"], errors="coerce")

    def _map_stat(w_col: str, l_col: str) -> tuple[pd.Series, pd.Series]:
        a = raw[w_col].where(~swap, raw[l_col])
        b = raw[l_col].where(~swap, raw[w_col])
        return pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")

    out["svpt_a"], out["svpt_b"] = _map_stat("w_svpt", "l_svpt")
    out["first_in_a"], out["first_in_b"] = _map_stat("w_1stIn", "l_1stIn")
    out["first_won_a"], out["first_won_b"] = _map_stat("w_1stWon", "l_1stWon")
    out["second_won_a"], out["second_won_b"] = _map_stat("w_2ndWon", "l_2ndWon")
    out["ace_a"], out["ace_b"] = _map_stat("w_ace", "l_ace")
    out["df_a"], out["df_b"] = _map_stat("w_df", "l_df")
    out["bp_saved_a"], out["bp_saved_b"] = _map_stat("w_bpSaved", "l_bpSaved")
    out["bp_faced_a"], out["bp_faced_b"] = _map_stat("w_bpFaced", "l_bpFaced")

    # Light de-dupe
    out = out.drop_duplicates(subset=["match_id"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    write_csv(out, out_path)

    return out_path


def download_sackmann_both(
    start_year: int = 2015,
    end_year: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """Download ATP + WTA and merge into one standardized matches.csv."""
    paths = get_paths()
    out_path = out_path or (paths.raw_dir / "matches.csv")

    atp_path = download_sackmann_matches("atp", start_year=start_year, end_year=end_year, out_path=paths.raw_dir / "matches_atp.csv")
    wta_path = download_sackmann_matches("wta", start_year=start_year, end_year=end_year, out_path=paths.raw_dir / "matches_wta.csv")

    atp = pd.read_csv(atp_path)
    wta = pd.read_csv(wta_path)

    merged = pd.concat([atp, wta], ignore_index=True).sort_values(["date", "match_id"]).reset_index(drop=True)
    write_csv(merged, out_path)

    return out_path
