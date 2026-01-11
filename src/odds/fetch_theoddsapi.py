from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.odds.odds_utils import normalize_name
from src.utils.config import get_paths
from src.utils.helpers import read_csv, write_csv


@dataclass(frozen=True)
class TheOddsApiConfig:
    api_key_env: str = "THE_ODDS_API_KEY"
    regions_env: str = "THE_ODDS_API_REGIONS"
    sport_keys_env: str = "THE_ODDS_API_SPORT_KEYS"
    regions: str = "us"
    markets: str = "h2h"
    odds_format: str = "decimal"  # request odds in decimal
    date_format: str = "%Y-%m-%d"


def _validate_regions(regions: str) -> str:
    # The Odds API v4 docs list: us, us2, uk, au, eu
    allowed = {"us", "us2", "uk", "au", "eu"}
    raw = [r.strip().lower() for r in str(regions).split(",") if r.strip()]
    if not raw:
        return "us"
    if "ca" in raw:
        raise ValueError("The Odds API does not support regions='ca'. Use 'us' or 'us2' instead.")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unsupported regions: {unknown}. Supported: {sorted(allowed)}")
    return ",".join(raw)


def _sport_key_for_tour(tour: str) -> str:
    # Kept for backward compatibility; the actual keys depend on the provider.
    t = str(tour).strip().upper()
    return "tennis_wta" if t == "WTA" else "tennis_atp"


def _list_tennis_sports(session: requests.Session, api_key: str) -> list[dict[str, Any]]:
    url = "https://api.the-odds-api.com/v4/sports"
    r = session.get(url, params={"apiKey": api_key, "all": "true"}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"The Odds API error ({r.status_code}) on /sports: {r.text[:500]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from The Odds API /sports (expected list)")
    return [s for s in data if str(s.get("group", "")).strip().lower() == "tennis"]


def _choose_sport_keys_for_upcoming(
    upcoming: pd.DataFrame,
    tennis_sports: list[dict[str, Any]],
) -> list[str]:
    # Try to match on tournament name if possible.
    tournaments = sorted({str(t).strip().lower() for t in upcoming.get("tournament", pd.Series(dtype=str)).dropna().unique() if str(t).strip()})
    if not tournaments:
        return []

    def _norm(s: str) -> str:
        return normalize_name(s)

    keys: list[str] = []
    for t in tournaments:
        t_norm = _norm(t)
        # match against title (e.g. "ATP Australian Open")
        for sp in tennis_sports:
            title = _norm(str(sp.get("title", "")))
            key = str(sp.get("key", "")).strip()
            if not key:
                continue
            if t_norm and t_norm in title:
                keys.append(key)

    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def fetch_odds_theoddsapi(
    upcoming_csv: Path | None = None,
    out_csv: Path | None = None,
    config: TheOddsApiConfig | None = None,
    api_key: str | None = None,
) -> Path:
    """Fetch current tennis moneyline odds from The Odds API.

    This fetches *all* upcoming odds for ATP/WTA and then filters down to the
    matchups listed in upcoming_matches.csv.

    Output format matches data/raw/odds_template.csv.

    Notes:
      - Many odds feeds don't include tournament names; we fill tournament/date from upcoming CSV.
      - Requires an API key in env var THE_ODDS_API_KEY (or pass api_key=...).
    """

    paths = get_paths()
    config = config or TheOddsApiConfig()

    upcoming_csv = upcoming_csv or (paths.raw_dir / "upcoming_matches.csv")
    out_csv = out_csv or (paths.raw_dir / "odds.csv")

    regions = _validate_regions(os.getenv(config.regions_env, config.regions))

    key = api_key or os.getenv(config.api_key_env)
    if not key:
        raise RuntimeError(
            f"Missing API key. Set {config.api_key_env} in your environment (The Odds API)."
        )

    upcoming = read_csv(upcoming_csv)
    if upcoming.empty:
        raise ValueError(f"Upcoming matches file is empty: {upcoming_csv}")

    # Minimal validation
    for c in ["player_a", "player_b"]:
        if c not in upcoming.columns:
            raise ValueError(f"Upcoming matches missing column: {c}")

    if "tour" not in upcoming.columns:
        upcoming["tour"] = "ATP"

    # Normalize and build lookup so we can fill tournament/date fields.
    def _norm_pair(a: Any, b: Any) -> tuple[str, str]:
        return normalize_name(a), normalize_name(b)

    upcoming_pairs: set[tuple[str, str]] = set()
    meta_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    for r in upcoming.to_dict(orient="records"):
        a = r.get("player_a")
        b = r.get("player_b")
        if a is None or b is None:
            continue
        k1 = _norm_pair(a, b)
        k2 = _norm_pair(b, a)
        upcoming_pairs.add(k1)
        upcoming_pairs.add(k2)

        # Persist metadata for both directions
        meta = {
            "date": r.get("date"),
            "tournament": r.get("tournament", ""),
            "tour": r.get("tour", "ATP"),
        }
        meta_by_pair[k1] = meta
        meta_by_pair[k2] = meta

    session = requests.Session()
    tennis_sports = _list_tennis_sports(session, key)

    # Allow explicit override of sport keys (comma-delimited).
    override = os.getenv(config.sport_keys_env, "").strip()
    if override:
        sport_keys = [s.strip() for s in override.split(",") if s.strip()]
    else:
        sport_keys = _choose_sport_keys_for_upcoming(upcoming, tennis_sports)

    if not sport_keys:
        available = sorted({str(s.get("title", "")) for s in tennis_sports if s.get("title")})
        raise RuntimeError(
            "Could not determine a tennis sport key for these upcoming matches. "
            "This usually means the odds provider doesn't cover that tournament (e.g. Auckland). "
            "\n\nOptions:" 
            f"\n- Provide odds manually in data/raw/odds.csv and run: python main.py compare-odds" 
            f"\n- Or set {config.sport_keys_env} to a supported tennis key (comma-delimited)." 
            f"\n\nTennis sports available from this API key include titles like: {available[:12]}"
        )
    rows: list[dict[str, Any]] = []

    for sport_key in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            "apiKey": key,
            "regions": regions,
            "markets": config.markets,
            "oddsFormat": config.odds_format,
        }

        resp = session.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"The Odds API error ({resp.status_code}): {resp.text[:500]}")

        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected response from The Odds API (expected list)")

        for event in data:
            home = event.get("home_team")
            away = event.get("away_team")
            if not home or not away:
                continue

            pair = _norm_pair(home, away)
            if pair not in upcoming_pairs:
                continue

            meta = meta_by_pair.get(pair, {})

            # For each bookmaker, pull h2h prices.
            for book in event.get("bookmakers", []) or []:
                book_title = book.get("title") or book.get("key") or ""
                for market in book.get("markets", []) or []:
                    if market.get("key") != "h2h":
                        continue

                    price_by_name: dict[str, Any] = {}
                    for outcome in market.get("outcomes", []) or []:
                        nm = outcome.get("name")
                        pr = outcome.get("price")
                        if nm is None or pr is None:
                            continue
                        price_by_name[normalize_name(nm)] = pr

                    a_norm, b_norm = pair
                    # Resolve odds for the ordered (home, away) in this event.
                    if a_norm not in price_by_name or b_norm not in price_by_name:
                        continue

                    rows.append(
                        {
                            "date": meta.get("date") or None,
                            "tournament": meta.get("tournament") or "",
                            "book": book_title,
                            "odds_format": "decimal",
                            "player_a": str(home),
                            "player_b": str(away),
                            "odds_a": float(price_by_name[a_norm]),
                            "odds_b": float(price_by_name[b_norm]),
                        }
                    )

    if not rows:
        raise RuntimeError(
            "Fetched odds but none matched the upcoming matchups. "
            "This is usually a player-name mismatch between your upcoming file and the book feed."
        )

    out = pd.DataFrame(rows)

    # Normalize date strings (keep consistent with template)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        # If upcoming provided date as string, some may be NaT; leave blank
        out["date"] = out["date"].apply(lambda d: d.isoformat() if isinstance(d, date) else "")

    # De-dup within (tournament, players, book) keeping latest row.
    out = out.drop_duplicates(subset=["tournament", "player_a", "player_b", "book"], keep="last")

    write_csv(out.reset_index(drop=True), out_csv)
    return out_csv
