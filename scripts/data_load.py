"""
Download MLB schedule data from the MLB Stats API for the last 8 seasons.
Saves to the project data folder as CSV (or JSON if the dataset is very large).
Uses the API directly (no statsapi parsing) to avoid KeyError on varying response shapes.
"""
from pathlib import Path

import pandas as pd
import requests

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Last 8 full seasons (2018–2025)
SEASON_START_MONTH_DAY = "03/01"  # March 1
SEASON_END_MONTH_DAY = "11/30"   # November 30 (postseason)
NUM_SEASONS = 8
# Rows above this: save as JSON to avoid huge CSV (optional)
JSON_IF_ROWS_OVER = 100_000

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def get_season_dates():
    """Yield (start_date_iso, end_date_iso) for each of the last NUM_SEASONS seasons."""
    from datetime import datetime

    current_year = datetime.now().year
    # e.g. 2026 -> last 8 full seasons: 2018..2025
    start_year = current_year - NUM_SEASONS
    for year in range(start_year, start_year + NUM_SEASONS):
        start_iso = f"{year}-03-01"
        end_iso = f"{year}-11-30"
        yield start_iso, end_iso


def _game_to_row(g):
    """Convert one game dict from the API into a flat row compatible with eda.py."""
    teams = g.get("teams") or {}
    away = teams.get("away") or {}
    home = teams.get("home") or {}
    away_team = away.get("team") or {}
    home_team = home.get("team") or {}
    status = (g.get("status") or {}).get("detailedState") or (g.get("status") or {}).get("abstractGameState") or ""
    venue = g.get("venue") or {}
    decisions = g.get("decisions") or {}
    winner = (decisions.get("winner") or {}).get("fullName", "")
    loser = (decisions.get("loser") or {}).get("fullName", "")
    home_probable = (home.get("probablePitcher") or {}).get("fullName", "")
    away_probable = (away.get("probablePitcher") or {}).get("fullName", "")
    return {
        "game_id": g.get("gamePk"),
        "game_date": g.get("officialDate") or (g.get("gameDate") or "")[:10],
        "game_datetime": g.get("gameDate"),
        "status": status,
        "away_id": away_team.get("id"),
        "away_name": away_team.get("name", ""),
        "away_score": away.get("score"),
        "home_id": home_team.get("id"),
        "home_name": home_team.get("name", ""),
        "home_score": home.get("score"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name", ""),
        "game_type": g.get("gameType"),
        "doubleheader": g.get("doubleHeader", "N"),
        "winning_pitcher": winner if isinstance(winner, str) else "",
        "losing_pitcher": loser if isinstance(loser, str) else "",
        "home_probable_pitcher": home_probable if isinstance(home_probable, str) else "",
        "away_probable_pitcher": away_probable if isinstance(away_probable, str) else "",
    }


def fetch_all_seasons():
    """Fetch schedule from MLB Stats API for all seasons; return one DataFrame."""
    # Hydrate decisions + probablePitcher so we get winning/losing/probable pitchers in one request
    hydrate = "decisions,probablePitcher(note),linescore"
    all_rows = []
    for start_iso, end_iso in get_season_dates():
        print(f"  Fetching {start_iso} – {end_iso} ...")
        params = {
            "sportId": 1,
            "startDate": start_iso,
            "endDate": end_iso,
            "hydrate": hydrate,
        }
        resp = requests.get(SCHEDULE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for date_block in data.get("dates") or []:
            for g in date_block.get("games") or []:
                all_rows.append(_game_to_row(g))
    return pd.DataFrame(all_rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MLB schedule data (last 8 seasons) from MLB Stats API...")
    df = fetch_all_seasons()
    n = len(df)
    if n == 0:
        print("No games returned. Check API or date range.")
        return

    print(f"  Total games: {n}")

    base_name = "schedule_8_seasons"
    csv_path = DATA_DIR / f"{base_name}.csv"
    json_path = DATA_DIR / f"{base_name}.json"

    if n > JSON_IF_ROWS_OVER:
        # Save as JSON (lines format) for very large data
        json_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(json_path, orient="records", lines=True, date_format="iso")
        print(f"  Saved to {json_path} (JSON, {n} rows)")
    else:
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path} (CSV, {n} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
