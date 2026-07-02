"""
Fetch MLB closing moneyline odds from SportsBookReview (via sbr-odds-scraper).

Saves raw per-sportsbook rows to data/odds_moneyline.csv. Re-running merges new
dates and refreshes overlapping rows (same game_id + sportsbook).

Requires network access. For live/current odds via The Odds API, set ODDS_API_KEY
(optional; historical backtests use SBR).
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ODDS_CSV = DATA_DIR / "odds_moneyline.csv"
FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"

SBR_COLUMNS = [
    "game_id",
    "date",
    "start_time",
    "away_team",
    "away_team_short",
    "home_team",
    "home_team_short",
    "away_score",
    "home_score",
    "venue",
    "game_type",
    "status",
    "sportsbook",
    "odds_type",
    "opening_home_odds",
    "opening_away_odds",
    "current_home_odds",
    "current_away_odds",
]


def load_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def dates_missing_from_odds(
    game_dates: pd.Series, odds_df: pd.DataFrame | None
) -> list[str]:
    """Dates present in schedule but not yet in odds cache."""
    needed = sorted({d.date().isoformat() for d in pd.to_datetime(game_dates).dropna()})
    if odds_df is None or odds_df.empty or "date" not in odds_df.columns:
        return needed
    have = set(pd.to_datetime(odds_df["date"]).dt.date.astype(str))
    return [d for d in needed if d not in have]


def scrape_sbr(start_iso: str, end_iso: str, *, fast: bool = False) -> pd.DataFrame:
    import sbr_odds_scraper as sbr

    print(f"  Scraping SBR moneylines {start_iso} – {end_iso} ...")
    df = sbr.scrape(start_iso, end_iso, odds_types=["moneyline"], fast=fast)
    if df is None or df.empty:
        return pd.DataFrame(columns=SBR_COLUMNS)
    keep = [c for c in SBR_COLUMNS if c in df.columns]
    return df[keep].copy()


def fetch_odds_api_snapshot(out_path: Path) -> pd.DataFrame:
    """Optional: snapshot current MLB h2h odds from The Odds API (free tier = current only)."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY to use The Odds API.")

    import requests

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  The Odds API requests remaining: {remaining}")

    rows: list[dict] = []
    for event in resp.json():
        game_date = (event.get("commence_time") or "")[:10]
        home = away = None
        for team in event.get("home_team", ""), event.get("away_team", ""):
            pass
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market.get("outcomes") or []}
                rows.append(
                    {
                        "game_id": event.get("id"),
                        "date": game_date,
                        "away_team": away_name,
                        "home_team": home_name,
                        "sportsbook": book.get("key"),
                        "odds_type": "moneyline",
                        "current_home_odds": prices.get(home_name),
                        "current_away_odds": prices.get(away_name),
                        "status": "Scheduled",
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
        print(f"  Wrote {len(df)} rows to {out_path}")
    return df


def upsert_odds(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming.empty:
        return existing if existing is not None else pd.DataFrame(columns=SBR_COLUMNS)
    if existing is None or existing.empty:
        return incoming.sort_values(["date", "game_id", "sportsbook"]).reset_index(drop=True)

    combined = pd.concat([existing, incoming], ignore_index=True)
    subset = ["game_id", "sportsbook", "odds_type"]
    subset = [c for c in subset if c in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    combined = combined.sort_values(["date", "game_id", "sportsbook"]).reset_index(drop=True)
    return combined


def fetch_season_odds(
    season: int,
    out_path: Path = DEFAULT_ODDS_CSV,
    *,
    missing_only: bool = True,
    fast: bool = False,
) -> Path:
    """Fetch SBR moneylines for all game dates in `season`; return output path."""
    if not FEATURED_CSV.exists():
        raise FileNotFoundError(f"{FEATURED_CSV} not found. Run split_n_preprocess.py first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)

    sched = pd.read_csv(FEATURED_CSV)
    sched["game_date"] = pd.to_datetime(sched["game_date"])
    game_dates = sched.loc[sched["game_date"].dt.year == season, "game_date"]
    if game_dates.empty:
        raise RuntimeError(f"No games for season {season} in {FEATURED_CSV}")

    if missing_only:
        dates = dates_missing_from_odds(game_dates, existing)
        if not dates:
            print(f"All game dates for {season} already in {out_path}")
            return out_path
        start_iso, end_iso = dates[0], dates[-1]
    else:
        start_iso = game_dates.min().date().isoformat()
        end_iso = game_dates.max().date().isoformat()

    print(f"Fetching odds → {out_path}")
    incoming = scrape_sbr(start_iso, end_iso, fast=fast)
    if incoming.empty:
        print("No odds returned.")
        return out_path

    merged = upsert_odds(existing, incoming)
    merged.to_csv(out_path, index=False)
    print(f"  Saved {len(merged):,} rows ({incoming['date'].nunique()} dates in this fetch)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch MLB moneyline odds into data/odds_moneyline.csv")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--season",
        type=int,
        default=None,
        help="If set, fetch all game dates for this season from featured CSV (overrides --start/--end).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_ODDS_CSV),
        help="Output CSV path",
    )
    p.add_argument(
        "--missing-only",
        action="store_true",
        help="Only fetch dates not already present in the output file.",
    )
    p.add_argument(
        "--odds-api",
        action="store_true",
        help="Fetch current snapshot from The Odds API (requires ODDS_API_KEY).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="SBR fast mode (may hit rate limits).",
    )
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)

    if args.odds_api:
        fetch_odds_api_snapshot(out_path)
        return

    if args.season is not None:
        fetch_season_odds(
            args.season,
            out_path,
            missing_only=args.missing_only,
            fast=args.fast,
        )
        return

    end_iso = args.end or date.today().isoformat()
    start_iso = args.start or end_iso

    print(f"Fetching odds → {out_path}")
    incoming = scrape_sbr(start_iso, end_iso, fast=args.fast)
    if incoming.empty:
        print("No odds returned.")
        return

    merged = upsert_odds(existing, incoming)
    merged.to_csv(out_path, index=False)
    print(f"  Saved {len(merged):,} rows ({incoming['date'].nunique()} dates in this fetch)")


if __name__ == "__main__":
    main()
