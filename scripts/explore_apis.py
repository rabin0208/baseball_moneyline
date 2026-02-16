"""
Explore data from pybaseball and MLB Stats API.
Fetches the same week of games from both sources for comparison.
"""
from pathlib import Path

import pandas as pd

# Output directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Same week for both APIs (June 1–7, 2024)
WEEK_START = "06/01/2024"
WEEK_END = "06/07/2024"
WEEK_START_ISO = "2024-06-01"
WEEK_END_ISO = "2024-06-07"

# All 30 MLB teams (Baseball Reference abbreviations)
MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]


def fetch_mlb_stats_api():
    """Fetch schedule/game data from MLB Stats API for the target week."""
    import statsapi

    games = statsapi.schedule(start_date=WEEK_START, end_date=WEEK_END)
    return pd.DataFrame(games)


def _parse_pybaseball_date(date_ser: pd.Series, year: int = 2024) -> pd.Series:
    """Parse pybaseball 'Date' (e.g. 'Saturday, Jun 1' or 'Sunday, Apr 21 (1)') to date."""
    s = date_ser.astype(str).str.replace(r"\s*\(\d\)$", "", regex=True)
    return pd.to_datetime(s + f", {year}", format="%A, %b %d, %Y", errors="coerce")


def fetch_pybaseball_same_week():
    """Fetch all teams' 2024 game logs from pybaseball, then filter to the target week."""
    from pybaseball import schedule_and_record

    frames = []
    with pd.option_context("mode.chained_assignment", None):
        for team in MLB_TEAMS:
            df = schedule_and_record(2024, team)
            df["game_date"] = _parse_pybaseball_date(df["Date"], 2024)
            df = df.dropna(subset=["game_date"])
            week = df[
                (df["game_date"] >= WEEK_START_ISO)
                & (df["game_date"] <= WEEK_END_ISO)
            ]
            if not week.empty:
                frames.append(week)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["game_date", "Tm"])


def main():
    print(f"Using same week for both: {WEEK_START} – {WEEK_END}\n")

    print("Fetching from MLB Stats API...")
    try:
        df_mlb = fetch_mlb_stats_api()
        out_mlb = DATA_DIR / "mlb_stats_api_schedule_sample.csv"
        df_mlb.to_csv(out_mlb, index=False)
        print(f"  Saved {len(df_mlb)} rows to {out_mlb}")
        print(f"  Columns: {list(df_mlb.columns)}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nFetching from pybaseball (all teams, then filter by date)...")
    try:
        df_pb = fetch_pybaseball_same_week()
        out_pb = DATA_DIR / "pybaseball_schedule_and_record_sample.csv"
        df_pb.to_csv(out_pb, index=False)
        print(f"  Saved {len(df_pb)} rows to {out_pb}")
        print(f"  Columns: {list(df_pb.columns)}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nDone. Compare the two CSVs in 'data/' for the same week.")


if __name__ == "__main__":
    main()
