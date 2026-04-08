"""
Rolling feature state for live prediction — mirrors split_n_preprocess.py logic.
Seed with completed games, then for each new game: read features, then update state if Final.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from model_utils import LAG_WINDOW, lag_vector, lag_vector_pitcher_centered


class RollingFeatureState:
    """Holds team / H2H / rest / pitcher history to match split_n_preprocess feature order."""

    def __init__(self, window: int = LAG_WINDOW) -> None:
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

        feats: dict[str, float] = {}

        def side_lags(team_id: int, prefix: str) -> None:
            tw_l = lag_vector(tw[team_id], w, 0.0)
            tr_l = lag_vector(tr[team_id], w, 0.0)
            tra_l = lag_vector(tra[team_id], w, 0.0)
            trd_l = lag_vector(trd[team_id], w, 0.0)
            for k in range(w):
                feats[f"{prefix}_win_lag_{k + 1}"] = tw_l[k]
                feats[f"{prefix}_runs_lag_{k + 1}"] = tr_l[k]
                feats[f"{prefix}_runs_allowed_lag_{k + 1}"] = tra_l[k]
                feats[f"{prefix}_run_diff_lag_{k + 1}"] = trd_l[k]

        side_lags(home_id, "home")
        side_lags(away_id, "away")

        key = (min(home_id, away_id), max(home_id, away_id))
        past_h2h = self.h2h[key]
        last_n = [p for p in past_h2h if p[0] < game_date][-w:]
        wins_chrono: list[int] = []
        for _past_date, past_home_id, past_home_won in last_n:
            if past_home_id == home_id:
                wins_chrono.append(past_home_won)
            else:
                wins_chrono.append(1 - past_home_won)
        h2h_l = lag_vector(wins_chrono, w, 0.0)
        for k in range(w):
            feats[f"home_h2h_win_lag_{k + 1}"] = h2h_l[k]

        if home_id in self.team_last_game_date:
            feats["home_rest_days"] = float((game_date - self.team_last_game_date[home_id]).days)
        else:
            feats["home_rest_days"] = 0.0
        if away_id in self.team_last_game_date:
            feats["away_rest_days"] = float((game_date - self.team_last_game_date[away_id]).days)
        else:
            feats["away_rest_days"] = 0.0

        ts = pd.to_datetime(game_date)
        feats["season"] = float(ts.year)
        feats["week_of_year"] = float(int(ts.strftime("%V")))

        def pitcher_lags(pitcher_name: str, prefix: str) -> None:
            if not pitcher_name:
                for k in range(w):
                    feats[f"{prefix}_pitcher_win_lag_{k + 1}"] = 0.0
                return
            past = [(d, x) for d, x in self.pitcher_starts[pitcher_name] if d < game_date]
            win_vals = [x for _, x in past[-w:]]
            pl = lag_vector_pitcher_centered(win_vals, w)
            for k in range(w):
                feats[f"{prefix}_pitcher_win_lag_{k + 1}"] = pl[k]

        pitcher_lags(home_pitcher, "home")
        pitcher_lags(away_pitcher, "away")

        return feats

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
