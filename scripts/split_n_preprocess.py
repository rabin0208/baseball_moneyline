"""
Feature engineering and train/val/test split for MLB schedule data.
Reads the final EDA CSV, adds features (e.g. rolling win averages), and saves processed data.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

FINAL_CSV = DATA_DIR / "schedule_8_seasons_final.csv"
ROLLING_WINDOW = 10


def load_final_data() -> pd.DataFrame:
    """Load final (complete-case) schedule data and sort by date."""
    df = pd.read_csv(FINAL_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return df


def add_rolling_avg_wins_and_runs(
    df: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Add rolling average of wins (win rate), runs scored, runs allowed, and run differential
    in the last `window` games for home and away team.
    Run differential = runs scored - runs allowed per game.
    Uses only games before the current game; early games use available history (< window).
    """
    team_wins: dict[int, list[int]] = defaultdict(list)
    team_runs: dict[int, list[float]] = defaultdict(list)
    team_runs_allowed: dict[int, list[float]] = defaultdict(list)
    team_run_diff: dict[int, list[float]] = defaultdict(list)

    home_wins_rolling = []
    away_wins_rolling = []
    home_runs_rolling = []
    away_runs_rolling = []
    home_runs_allowed_rolling = []
    away_runs_allowed_rolling = []
    home_run_diff_rolling = []
    away_run_diff_rolling = []

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        home_won = int(row["home_win"])
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])

        # Last `window` games (before this game)
        home_wins_last = team_wins[home_id][-window:] if team_wins[home_id] else []
        away_wins_last = team_wins[away_id][-window:] if team_wins[away_id] else []
        home_runs_last = team_runs[home_id][-window:] if team_runs[home_id] else []
        away_runs_last = team_runs[away_id][-window:] if team_runs[away_id] else []
        home_ra_last = team_runs_allowed[home_id][-window:] if team_runs_allowed[home_id] else []
        away_ra_last = team_runs_allowed[away_id][-window:] if team_runs_allowed[away_id] else []
        home_rd_last = team_run_diff[home_id][-window:] if team_run_diff[home_id] else []
        away_rd_last = team_run_diff[away_id][-window:] if team_run_diff[away_id] else []

        home_wins_rolling.append(
            sum(home_wins_last) / len(home_wins_last) if home_wins_last else 0.0
        )
        away_wins_rolling.append(
            sum(away_wins_last) / len(away_wins_last) if away_wins_last else 0.0
        )
        home_runs_rolling.append(
            sum(home_runs_last) / len(home_runs_last) if home_runs_last else 0.0
        )
        away_runs_rolling.append(
            sum(away_runs_last) / len(away_runs_last) if away_runs_last else 0.0
        )
        home_runs_allowed_rolling.append(
            sum(home_ra_last) / len(home_ra_last) if home_ra_last else 0.0
        )
        away_runs_allowed_rolling.append(
            sum(away_ra_last) / len(away_ra_last) if away_ra_last else 0.0
        )
        home_run_diff_rolling.append(
            sum(home_rd_last) / len(home_rd_last) if home_rd_last else 0.0
        )
        away_run_diff_rolling.append(
            sum(away_rd_last) / len(away_rd_last) if away_rd_last else 0.0
        )

        # Append this game to both teams' histories (run diff = runs scored - runs allowed)
        team_wins[home_id].append(home_won)
        team_wins[away_id].append(1 - home_won)
        team_runs[home_id].append(home_score)
        team_runs[away_id].append(away_score)
        team_runs_allowed[home_id].append(away_score)
        team_runs_allowed[away_id].append(home_score)
        team_run_diff[home_id].append(home_score - away_score)
        team_run_diff[away_id].append(away_score - home_score)

    df = df.copy()
    df["home_rolling_avg_wins_10"] = home_wins_rolling
    df["away_rolling_avg_wins_10"] = away_wins_rolling
    df["home_rolling_avg_runs_10"] = home_runs_rolling
    df["away_rolling_avg_runs_10"] = away_runs_rolling
    df["home_rolling_avg_runs_allowed_10"] = home_runs_allowed_rolling
    df["away_rolling_avg_runs_allowed_10"] = away_runs_allowed_rolling
    df["home_rolling_avg_run_diff_10"] = home_run_diff_rolling
    df["away_rolling_avg_run_diff_10"] = away_run_diff_rolling
    return df


def add_rolling_avg_h2h(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Add rolling average of wins for the home team in head-to-head games:
    win rate of the current home team in the last `window` meetings between these two teams.
    Uses only games before the current game.
    """
    # Key (min_id, max_id) -> list of (game_date, home_id, home_won) in chronological order
    h2h: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)

    home_h2h_rolling = []

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_won = int(row["home_win"])

        key = (min(home_id, away_id), max(home_id, away_id))
        past = h2h[key]  # all prior H2H games (date < this game)
        last_n = [p for p in past if p[0] < game_date][-window:]

        # For each past H2H game: did the *current* home team win that meeting?
        current_home_wins = []
        for past_date, past_home_id, past_home_won in last_n:
            if past_home_id == home_id:
                current_home_wins.append(past_home_won)
            else:
                current_home_wins.append(1 - past_home_won)

        home_avg = (
            sum(current_home_wins) / len(current_home_wins) if current_home_wins else 0.0
        )
        home_h2h_rolling.append(home_avg)

        h2h[key].append((game_date, home_id, home_won))

    df = df.copy()
    df["home_rolling_avg_h2h_wins_10"] = home_h2h_rolling
    return df


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


def add_pitcher_rolling_wins_centered(
    df: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Add rolling win rate (centered) for probable pitchers in their last `window` starts.
    Value = win_rate - 0.5: positive if won more than lost, negative if lost more, 0 if .500 or pitcher not found.
    Pitcher not found (missing/empty) → 0.
    """
    # pitcher name -> list of (game_date, got_win 0/1) in chronological order
    pitcher_starts: dict[str, list[tuple[pd.Timestamp, int]]] = defaultdict(list)

    home_vals = []
    away_vals = []

    for _, row in df.iterrows():
        game_date = row["game_date"]
        home_won = int(row["home_win"])
        home_pitcher = str(row.get("home_probable_pitcher", "") or "").strip()
        away_pitcher = str(row.get("away_probable_pitcher", "") or "").strip()

        def centered_value(pitcher_name: str) -> float:
            if not pitcher_name:
                return 0.0
            past = [
                (d, w)
                for d, w in pitcher_starts[pitcher_name]
                if d < game_date
            ][-window:]
            if not past:
                return 0.0
            win_rate = sum(w for _, w in past) / len(past)
            return win_rate - 0.5

        home_vals.append(centered_value(home_pitcher))
        away_vals.append(centered_value(away_pitcher))

        if home_pitcher:
            pitcher_starts[home_pitcher].append((game_date, home_won))
        if away_pitcher:
            pitcher_starts[away_pitcher].append((game_date, 1 - home_won))

    df = df.copy()
    df["home_pitcher_rolling_wins_centered_10"] = home_vals
    df["away_pitcher_rolling_wins_centered_10"] = away_vals
    return df


def main():
    print(f"Loading {FINAL_CSV}...")
    df = load_final_data()
    print(f"  Loaded {len(df)} games")

    print(
        f"Adding rolling averages (last {ROLLING_WINDOW} games): wins, runs, runs allowed, and run diff for home and away..."
    )
    df = add_rolling_avg_wins_and_runs(df, window=ROLLING_WINDOW)
    print("  Done.")

    print(f"Adding rolling average of home team wins in last {ROLLING_WINDOW} head-to-head games...")
    df = add_rolling_avg_h2h(df, window=ROLLING_WINDOW)
    print("  Done.")

    print("Adding rest days (days since last game) for home and away...")
    df = add_rest_days(df)
    print("  Done.")

    print(f"Adding pitcher rolling win rate (centered, last {ROLLING_WINDOW} starts; not found → 0)...")
    df = add_pitcher_rolling_wins_centered(df, window=ROLLING_WINDOW)
    print("  Done.")

    out_path = DATA_DIR / "schedule_8_seasons_featured.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
