# tennis_model

A lightweight, end-to-end **professional tennis match win probability** pipeline (data ingest → Elo + features → model training → upcoming match predictions).

This repo is intentionally structured similarly to the NFL project, but adapted to tennis.

## Quickstart (offline demo)

1) (Recommended) Create a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install deps:

```bash
python -m pip install -r requirements.txt
```

3) Run the full pipeline:

```bash
./run.sh
```

Outputs:
- `data/outputs/predictions_latest.csv`
- `models/trained/win_model.joblib`

## Free historical data (ATP/WTA)

This repo can download and standardize historical match results from Jeff Sackmann’s public tennis datasets.

Example (ATP, 2015–present):

```bash
python main.py download --tour atp --start-year 2015
./run.sh
```

Example (WTA):

```bash
python main.py download --tour wta --start-year 2015
./run.sh
```

Notes:
- These sources provide historical matches (and many match-level stats), but **not upcoming schedules**.
- For predictions you still provide `data/raw/upcoming_matches.csv` (or the sample file).

## Features (instead of “EPA”)

Tennis doesn’t have a direct equivalent to NFL EPA without point-by-point data.
With freely available match logs, this pipeline uses strong “player quality” proxies:

- Surface + best-of context
- Elo (pre-match)
- Rolling/EMA serve & return strength from match-level stats (service points won %, return points won %, ace/DF rates, break-point save rate)

## Data format

### Historical matches (required)
The pipeline expects a CSV like `data/raw/matches.csv` with at least:

- `date` (YYYY-MM-DD)
- `surface` (Hard/Clay/Grass/Carpet)
- `best_of` (3 or 5)
- `player_a`, `player_b`
- `winner` ("A" or "B")
- optional: `rank_a`, `rank_b`

### Upcoming matches (required for prediction)
`data/raw/upcoming_matches.csv` must include:

- `date`, `surface`, `best_of`, `player_a`, `player_b`
- optional: `rank_a`, `rank_b`

## Notes / next steps

- You can swap the baseline Logistic Regression for XGBoost/LightGBM later.
- If you want odds shopping + Kelly (like the NFL repo), we can add an odds ingest module and expected value tooling.
