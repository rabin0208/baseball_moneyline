"""
Download MLB schedule data from the MLB Stats API for the last 8 seasons,
plus the current calendar year when it falls after that window (in-season updates).
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
INCREMENTAL_LOOKBACK_DAYS = 3


def get_season_dates():
    """Yield (start_date_iso, end_date_iso) for each of the last NUM_SEASONS seasons.

    Also fetches the current and next calendar year when they are not already in that
    block (so new-season games exist for evaluation even if the rolling window ended
    at the prior year, or the machine clock is a year behind).
    """
    from datetime import datetime

    current_year = datetime.now().year
    start_year = current_year - NUM_SEASONS
    fetched_years: set[int] = set()
    for year in range(start_year, start_year + NUM_SEASONS):
        fetched_years.add(year)
        yield f"{year}-03-01", f"{year}-11-30"
    for extra in (current_year, current_year + 1):
        if extra not in fetched_years:
            fetched_years.add(extra)
            yield f"{extra}-03-01", f"{extra}-11-30"


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


def fetch_date_range(start_iso: str, end_iso: str) -> pd.DataFrame:
    """Fetch schedule from MLB Stats API for one date range; return one DataFrame."""
    hydrate = "decisions,probablePitcher(note),linescore"
    all_rows = []
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


def load_existing_schedule(base_name: str) -> tuple[pd.DataFrame | None, Path | None]:
    """Load existing schedule data if present (CSV first, then JSON lines)."""
    csv_path = DATA_DIR / f"{base_name}.csv"
    json_path = DATA_DIR / f"{base_name}.json"

    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path
    if json_path.exists():
        return pd.read_json(json_path, orient="records", lines=True), json_path
    return None, None


def upsert_games(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge new games into existing data by game_id, keeping the newest fetched row."""
    if incoming.empty:
        return existing
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    if "game_date" in combined.columns:
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined = combined.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        combined["game_date"] = combined["game_date"].dt.strftime("%Y-%m-%d")
    return combined


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_name = "schedule_8_seasons"
    csv_path = DATA_DIR / f"{base_name}.csv"
    json_path = DATA_DIR / f"{base_name}.json"
    existing_df, existing_path = load_existing_schedule(base_name)

    if existing_df is None:
        print("No local schedule file found. Downloading full history window from MLB Stats API...")
        df = fetch_all_seasons()
    else:
        print(f"Found existing schedule file: {existing_path}")
        if "game_date" not in existing_df.columns or existing_df.empty:
            print("Existing file has no usable game_date rows. Downloading full history window...")
            df = fetch_all_seasons()
        else:
            existing_df["game_date"] = pd.to_datetime(existing_df["game_date"], errors="coerce")
            max_date = existing_df["game_date"].max()
            if pd.isna(max_date):
                print("Existing file has invalid game_date values. Downloading full history window...")
                df = fetch_all_seasons()
            else:
                # Use the latest *completed* game, not the furthest scheduled date.
                # Otherwise a full-season schedule fetch skips re-scoring games that
                # have finished since the last update (e.g. stuck at early April).
                final_dates = existing_df.loc[
                    existing_df["status"] == "Final", "game_date"
                ]
                last_known = (
                    final_dates.max().normalize()
                    if not final_dates.empty and pd.notna(final_dates.max())
                    else max_date.normalize()
                )
                start_ts = last_known - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
                # Fetch through the furthest season end the script currently targets.
                season_ends = [pd.Timestamp(end_iso) for _, end_iso in get_season_dates()]
                end_ts = max(season_ends)
                if start_ts > end_ts:
                    print("Local data is already beyond configured fetch window. Keeping existing file as-is.")
                    df = existing_df.copy()
                else:
                    print(
                        "Incremental update enabled: "
                        f"fetching from {start_ts.date().isoformat()} to {end_ts.date().isoformat()} "
                        f"(last Final game: {last_known.date().isoformat()}, "
                        f"{INCREMENTAL_LOOKBACK_DAYS}-day lookback)."
                    )
                    new_df = fetch_date_range(start_ts.date().isoformat(), end_ts.date().isoformat())
                    if new_df.empty:
                        print("No new/updated games returned by API. Keeping existing file as-is.")
                        df = existing_df.copy()
                    else:
                        before_n = len(existing_df)
                        df = upsert_games(existing_df, new_df)
                        print(
                            f"  Existing rows: {before_n:,}, fetched rows: {len(new_df):,}, "
                            f"merged rows: {len(df):,}"
                        )

    n = len(df)
    if n == 0:
        print("No games returned. Check API or date range.")
        return

    print(f"  Total games after update: {n}")

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
