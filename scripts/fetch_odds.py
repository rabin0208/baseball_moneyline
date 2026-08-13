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
from zoneinfo import ZoneInfo

import pandas as pd

# MLB slate dates align with the US calendar day; Odds API commence_time is UTC,
# so evening West Coast games often fall on the next UTC date.
_ODDS_SLATE_TZ = ZoneInfo("America/New_York")


def _slate_date_from_commence(commence_time: str) -> str:
    """Calendar date for an Odds API event in US/Eastern (YYYY-MM-DD)."""
    if not commence_time:
        return ""
    dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).astimezone(
        _ODDS_SLATE_TZ
    )
    return dt.date().isoformat()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ODDS_CSV = DATA_DIR / "odds_moneyline.csv"
FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def _load_dotenv() -> None:
    """Load PROJECT_ROOT/.env into os.environ (does not override existing vars)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def ensure_odds_api_key() -> str | None:
    """
    Resolve ODDS_API_KEY from (in order): existing env, project .env,
    Streamlit secrets (Community Cloud / local secrets.toml).
    """
    _load_dotenv()
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        secret = st.secrets.get("ODDS_API_KEY")  # type: ignore[attr-defined]
        if secret:
            os.environ["ODDS_API_KEY"] = str(secret)
            return str(secret)
    except Exception:
        pass
    return None

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
    try:
        df = sbr.scrape(start_iso, end_iso, odds_types=["moneyline"], fast=fast)
    except Exception as exc:  # noqa: BLE001 — network/HTML scraper failures are common
        print(f"  SBR scrape failed: {exc}")
        return pd.DataFrame(columns=SBR_COLUMNS)
    if df is None or df.empty:
        return pd.DataFrame(columns=SBR_COLUMNS)
    keep = [c for c in SBR_COLUMNS if c in df.columns]
    return df[keep].copy()


def fetch_odds_api(*, target_date: date | None = None) -> pd.DataFrame:
    """
    Current MLB h2h moneylines from The Odds API (free tier = live/upcoming only).

    Requires ODDS_API_KEY. Optionally filter to a single US/Eastern calendar date
    (not the raw UTC prefix of commence_time — late West Coast games are next-day UTC).
    """
    ensure_odds_api_key()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY (env, .env, or Streamlit secrets) to use The Odds API.")

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

    want = target_date.isoformat() if target_date is not None else None
    rows: list[dict] = []
    for event in resp.json():
        commence = event.get("commence_time") or ""
        game_date = _slate_date_from_commence(commence)
        if want is not None and game_date != want:
            continue
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
    if not rows:
        return pd.DataFrame(columns=SBR_COLUMNS)
    return pd.DataFrame(rows)


def fetch_odds_api_snapshot(out_path: Path) -> pd.DataFrame:
    """Optional: snapshot current MLB h2h odds from The Odds API and write CSV."""
    df = fetch_odds_api()
    if not df.empty:
        df.to_csv(out_path, index=False)
        print(f"  Wrote {len(df)} rows to {out_path}")
    return df


def odds_from_recommendation_csvs() -> pd.DataFrame:
    """
    Rebuild SBR-shaped moneyline rows from saved daily recommendation CSVs.

    Used when SportsBookReview returns 503 so the ROI tab can still score
    completed games that were priced live at recommendation time.
    """
    paths = sorted(TABLES_DIR.glob("bet_recommendations_*.csv"))
    rows: list[dict] = []
    for path in paths:
        df = pd.read_csv(path)
        needed = {"home_odds", "away_odds", "home_name", "away_name"}
        if not needed.issubset(df.columns):
            continue
        stem_date = path.stem.replace("bet_recommendations_", "")
        for _, rec in df.iterrows():
            if pd.isna(rec.get("home_odds")) or pd.isna(rec.get("away_odds")):
                continue
            raw_date = rec.get("game_date")
            if pd.notna(raw_date):
                date_str = pd.to_datetime(raw_date).date().isoformat()
            else:
                date_str = stem_date
            rows.append(
                {
                    "game_id": rec.get("game_id"),
                    "date": date_str,
                    "start_time": "",
                    "away_team": rec["away_name"],
                    "away_team_short": "",
                    "home_team": rec["home_name"],
                    "home_team_short": "",
                    "away_score": rec.get("away_score", ""),
                    "home_score": rec.get("home_score", ""),
                    "venue": "",
                    "game_type": "",
                    "status": rec.get("status", ""),
                    "sportsbook": "live_snapshot",
                    "odds_type": "moneyline",
                    "opening_home_odds": rec["home_odds"],
                    "opening_away_odds": rec["away_odds"],
                    "current_home_odds": rec["home_odds"],
                    "current_away_odds": rec["away_odds"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=SBR_COLUMNS)
    return pd.DataFrame(rows)


def fill_missing_odds_from_recommendations(
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add live-snapshot odds only for games not already in the odds file."""
    snapshots = odds_from_recommendation_csvs()
    if snapshots.empty:
        return existing if existing is not None else pd.DataFrame(columns=SBR_COLUMNS)
    if existing is None or existing.empty:
        return snapshots
    have_ids = set(
        pd.to_numeric(existing["game_id"], errors="coerce").dropna().astype(int)
    )
    snap_ids = pd.to_numeric(snapshots["game_id"], errors="coerce")
    new_rows = snapshots.loc[snap_ids.notna() & ~snap_ids.astype(int).isin(have_ids)].copy()
    if new_rows.empty:
        return existing
    print(f"  Backfilled {new_rows['game_id'].nunique()} games from recommendation CSVs")
    return upsert_odds(existing, new_rows)


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
    merged = existing
    if incoming.empty:
        print("No SBR odds returned.")
    else:
        merged = upsert_odds(existing, incoming)
        print(f"  SBR saved {incoming['date'].nunique()} dates")

    merged = fill_missing_odds_from_recommendations(merged)
    if merged is None or merged.empty:
        print("No odds available.")
        return out_path

    merged.to_csv(out_path, index=False)
    print(f"  Saved {len(merged):,} rows to {out_path}")
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
