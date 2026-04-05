"""
Feature engineering for MLB schedule data.
Reads the final EDA CSV, adds lagged features (last K games per stat), and saves processed data.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

from model_utils import (
    LAG_WINDOW,
    h2h_lag_column_names,
    lag_vector,
    lag_vector_pitcher_centered,
    pitcher_lag_column_names,
    team_lag_column_names,
)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

FINAL_CSV = DATA_DIR / "schedule_8_seasons_final.csv"


def load_final_data() -> pd.DataFrame:
    """Load final (complete-case) schedule data and sort by date."""
    df = pd.read_csv(FINAL_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return df


def add_lagged_team_features(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    """
    For home and away: last `window` games of win (0/1), runs, runs allowed, run diff.
    Lag 1 = most recent prior game; missing lags padded with 0.
    """
    team_wins: dict[int, list[int]] = defaultdict(list)
    team_runs: dict[int, list[float]] = defaultdict(list)
    team_runs_allowed: dict[int, list[float]] = defaultdict(list)
    team_run_diff: dict[int, list[float]] = defaultdict(list)

    cnames = team_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    def append_side(team_id: int, prefix: str) -> None:
        tw = lag_vector(team_wins[team_id], window, 0.0)
        tr = lag_vector(team_runs[team_id], window, 0.0)
        tra = lag_vector(team_runs_allowed[team_id], window, 0.0)
        trd = lag_vector(team_run_diff[team_id], window, 0.0)
        for k in range(window):
            cols[f"{prefix}_win_lag_{k + 1}"].append(tw[k])
            cols[f"{prefix}_runs_lag_{k + 1}"].append(tr[k])
            cols[f"{prefix}_runs_allowed_lag_{k + 1}"].append(tra[k])
            cols[f"{prefix}_run_diff_lag_{k + 1}"].append(trd[k])

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        home_won = int(row["home_win"])
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])

        append_side(home_id, "home")
        append_side(away_id, "away")

        team_wins[home_id].append(home_won)
        team_wins[away_id].append(1 - home_won)
        team_runs[home_id].append(home_score)
        team_runs[away_id].append(away_score)
        team_runs_allowed[home_id].append(away_score)
        team_runs_allowed[away_id].append(home_score)
        team_run_diff[home_id].append(home_score - away_score)
        team_run_diff[away_id].append(away_score - home_score)

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_lagged_h2h(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    """
    Per meeting: 1 if current home team won that past H2H game, else 0.
    Lag 1 = most recent H2H meeting; missing lags → 0.
    """
    h2h: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)
    cnames = h2h_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_won = int(row["home_win"])

        key = (min(home_id, away_id), max(home_id, away_id))
        past = h2h[key]
        last_n = [p for p in past if p[0] < game_date][-window:]
        wins_chrono: list[int] = []
        for _past_date, past_home_id, past_home_won in last_n:
            if past_home_id == home_id:
                wins_chrono.append(past_home_won)
            else:
                wins_chrono.append(1 - past_home_won)

        lags = lag_vector(wins_chrono, window, 0.0)
        for k in range(window):
            cols[f"home_h2h_win_lag_{k + 1}"].append(lags[k])

        h2h[key].append((game_date, home_id, home_won))

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add days since last game for home and away team.
    Uses only games before the current game; first game of a team gets NaN (no prior game).
    """
    team_last_game_date: dict[int, pd.Timestamp] = {}
    home_rest_days = []
    away_rest_days = []

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]

        if home_id in team_last_game_date:
            home_rest_days.append((game_date - team_last_game_date[home_id]).days)
        else:
            home_rest_days.append(pd.NA)

        if away_id in team_last_game_date:
            away_rest_days.append((game_date - team_last_game_date[away_id]).days)
        else:
            away_rest_days.append(pd.NA)

        team_last_game_date[home_id] = game_date
        team_last_game_date[away_id] = game_date

    df = df.copy()
    df["home_rest_days"] = home_rest_days
    df["away_rest_days"] = away_rest_days
    return df


def add_pitcher_lag_centered(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    """
    Last `window` starts for each probable pitcher: win encoded as (win - 0.5); missing → 0.
    """
    pitcher_starts: dict[str, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    cnames = pitcher_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    for _, row in df.iterrows():
        game_date = row["game_date"]
        home_won = int(row["home_win"])
        home_pitcher = str(row.get("home_probable_pitcher", "") or "").strip()
        away_pitcher = str(row.get("away_probable_pitcher", "") or "").strip()

        def fill(pitcher_name: str, prefix: str) -> None:
            if not pitcher_name:
                for k in range(window):
                    cols[f"{prefix}_pitcher_win_lag_{k + 1}"].append(0.0)
                return
            past = [(d, w) for d, w in pitcher_starts[pitcher_name] if d < game_date]
            win_vals = [w for _, w in past[-window:]]
            lags = lag_vector_pitcher_centered(win_vals, window)
            for k in range(window):
                cols[f"{prefix}_pitcher_win_lag_{k + 1}"].append(lags[k])

        fill(home_pitcher, "home")
        fill(away_pitcher, "away")

        if home_pitcher:
            pitcher_starts[home_pitcher].append((game_date, home_won))
        if away_pitcher:
            pitcher_starts[away_pitcher].append((game_date, 1 - home_won))

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def main():
    print(f"Loading {FINAL_CSV}...")
    df = load_final_data()
    print(f"  Loaded {len(df)} games")

    print(
        f"Adding lagged team features (last {LAG_WINDOW} games): win, runs, runs allowed, run diff..."
    )
    df = add_lagged_team_features(df, window=LAG_WINDOW)
    print("  Done.")

    print(f"Adding lagged head-to-head wins (last {LAG_WINDOW} meetings)...")
    df = add_lagged_h2h(df, window=LAG_WINDOW)
    print("  Done.")

    print("Adding rest days (days since last game) for home and away...")
    df = add_rest_days(df)
    print("  Done.")

    print(f"Adding pitcher lagged wins (centered, last {LAG_WINDOW} starts; not found → 0)...")
    df = add_pitcher_lag_centered(df, window=LAG_WINDOW)
    print("  Done.")

    out_path = DATA_DIR / "schedule_8_seasons_featured.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
