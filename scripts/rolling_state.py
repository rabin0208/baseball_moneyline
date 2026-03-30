"""
Rolling feature state for live prediction — mirrors split_n_preprocess.py logic.
Seed with completed games, then for each new game: read features, then update state if Final.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

ROLLING_WINDOW = 10


class RollingFeatureState:
    """Holds team / H2H / rest / pitcher history to match split_n_preprocess feature order."""

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self.window = window
        self.team_wins: dict[int, list[int]] = defaultdict(list)
        self.team_runs: dict[int, list[float]] = defaultdict(list)
        self.team_runs_allowed: dict[int, list[float]] = defaultdict(list)
        self.team_run_diff: dict[int, list[float]] = defaultdict(list)
        self.h2h: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)
        self.team_last_game_date: dict[int, pd.Timestamp] = {}
        self.pitcher_starts: dict[str, list[tuple[pd.Timestamp, int]]] = defaultdict(list)

    def seed_from_completed(self, df: pd.DataFrame) -> None:
        """Walk chronological completed games and apply outcomes (same as training pipeline)."""
        for _, row in df.iterrows():
            self.update_after_final_game(row)

    def features_for_game(self, row: pd.Series) -> dict[str, float]:
        """Feature vector before this game is played (uses history only)."""
        w = self.window
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_pitcher = str(row.get("home_probable_pitcher", "") or "").strip()
        away_pitcher = str(row.get("away_probable_pitcher", "") or "").strip()

        tw, tr, tra, trd = self.team_wins, self.team_runs, self.team_runs_allowed, self.team_run_diff

        home_wins_last = tw[home_id][-w:] if tw[home_id] else []
        away_wins_last = tw[away_id][-w:] if tw[away_id] else []
        home_runs_last = tr[home_id][-w:] if tr[home_id] else []
        away_runs_last = tr[away_id][-w:] if tr[away_id] else []
        home_ra_last = tra[home_id][-w:] if tra[home_id] else []
        away_ra_last = tra[away_id][-w:] if tra[away_id] else []
        home_rd_last = trd[home_id][-w:] if trd[home_id] else []
        away_rd_last = trd[away_id][-w:] if trd[away_id] else []

        home_rolling_avg_wins_10 = (
            sum(home_wins_last) / len(home_wins_last) if home_wins_last else 0.0
        )
        away_rolling_avg_wins_10 = (
            sum(away_wins_last) / len(away_wins_last) if away_wins_last else 0.0
        )
        home_rolling_avg_runs_10 = (
            sum(home_runs_last) / len(home_runs_last) if home_runs_last else 0.0
        )
        away_rolling_avg_runs_10 = (
            sum(away_runs_last) / len(away_runs_last) if away_runs_last else 0.0
        )
        home_rolling_avg_runs_allowed_10 = (
            sum(home_ra_last) / len(home_ra_last) if home_ra_last else 0.0
        )
        away_rolling_avg_runs_allowed_10 = (
            sum(away_ra_last) / len(away_ra_last) if away_ra_last else 0.0
        )
        home_rolling_avg_run_diff_10 = (
            sum(home_rd_last) / len(home_rd_last) if home_rd_last else 0.0
        )
        away_rolling_avg_run_diff_10 = (
            sum(away_rd_last) / len(away_rd_last) if away_rd_last else 0.0
        )

        key = (min(home_id, away_id), max(home_id, away_id))
        past_h2h = self.h2h[key]
        last_n = [p for p in past_h2h if p[0] < game_date][-w:]
        current_home_wins = []
        for past_date, past_home_id, past_home_won in last_n:
            if past_home_id == home_id:
                current_home_wins.append(past_home_won)
            else:
                current_home_wins.append(1 - past_home_won)
        home_rolling_avg_h2h_wins_10 = (
            sum(current_home_wins) / len(current_home_wins) if current_home_wins else 0.0
        )

        if home_id in self.team_last_game_date:
            home_rest_days = float((game_date - self.team_last_game_date[home_id]).days)
        else:
            home_rest_days = 0.0
        if away_id in self.team_last_game_date:
            away_rest_days = float((game_date - self.team_last_game_date[away_id]).days)
        else:
            away_rest_days = 0.0

        def centered_pitcher(pitcher_name: str) -> float:
            if not pitcher_name:
                return 0.0
            past = [(d, x) for d, x in self.pitcher_starts[pitcher_name] if d < game_date][-w:]
            if not past:
                return 0.0
            win_rate = sum(x for _, x in past) / len(past)
            return win_rate - 0.5

        home_pitcher_rolling_wins_centered_10 = centered_pitcher(home_pitcher)
        away_pitcher_rolling_wins_centered_10 = centered_pitcher(away_pitcher)

        return {
            "home_rolling_avg_wins_10": home_rolling_avg_wins_10,
            "away_rolling_avg_wins_10": away_rolling_avg_wins_10,
            "home_rolling_avg_runs_10": home_rolling_avg_runs_10,
            "away_rolling_avg_runs_10": away_rolling_avg_runs_10,
            "home_rolling_avg_runs_allowed_10": home_rolling_avg_runs_allowed_10,
            "away_rolling_avg_runs_allowed_10": away_rolling_avg_runs_allowed_10,
            "home_rolling_avg_run_diff_10": home_rolling_avg_run_diff_10,
            "away_rolling_avg_run_diff_10": away_rolling_avg_run_diff_10,
            "home_rolling_avg_h2h_wins_10": home_rolling_avg_h2h_wins_10,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "home_pitcher_rolling_wins_centered_10": home_pitcher_rolling_wins_centered_10,
            "away_pitcher_rolling_wins_centered_10": away_pitcher_rolling_wins_centered_10,
        }

    def update_after_final_game(self, row: pd.Series) -> None:
        """Apply a completed game's result (must match split_n_preprocess update order)."""
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_won = int(row["home_win"])
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        home_pitcher = str(row.get("home_probable_pitcher", "") or "").strip()
        away_pitcher = str(row.get("away_probable_pitcher", "") or "").strip()

        tw, tr, tra, trd = self.team_wins, self.team_runs, self.team_runs_allowed, self.team_run_diff
        tw[home_id].append(home_won)
        tw[away_id].append(1 - home_won)
        tr[home_id].append(home_score)
        tr[away_id].append(away_score)
        tra[home_id].append(away_score)
        tra[away_id].append(home_score)
        trd[home_id].append(home_score - away_score)
        trd[away_id].append(away_score - home_score)

        key = (min(home_id, away_id), max(home_id, away_id))
        self.h2h[key].append((game_date, home_id, home_won))

        self.team_last_game_date[home_id] = game_date
        self.team_last_game_date[away_id] = game_date

        if home_pitcher:
            self.pitcher_starts[home_pitcher].append((game_date, home_won))
        if away_pitcher:
            self.pitcher_starts[away_pitcher].append((game_date, 1 - home_won))
