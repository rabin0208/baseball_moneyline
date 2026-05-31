"""
Utilities for moneyline odds: implied probability, vig removal, ROI, team matching.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalize_team(name: str) -> str:
    """Lowercase team name for joins across MLB Stats API and odds sources."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def american_to_implied_prob(odds: float) -> float:
    """Convert American moneyline to implied win probability (includes vig)."""
    o = float(odds)
    if o == 0:
        return float("nan")
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Normalize two implied probabilities so they sum to 1."""
    total = p_home + p_away
    if total <= 0:
        return float("nan"), float("nan")
    return p_home / total, p_away / total


def flat_bet_profit(american_odds: float, won: bool) -> float:
    """Profit on a 1-unit flat bet at American odds."""
    if not won:
        return -1.0
    o = float(american_odds)
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def consensus_moneyline(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Median closing moneyline per game across sportsbooks.
    Expects SBR-style columns: date, home_team, away_team, current_home_odds, current_away_odds.
    """
    required = {"date", "home_team", "away_team", "current_home_odds", "current_away_odds"}
    missing = required - set(odds_df.columns)
    if missing:
        raise ValueError(f"odds_df missing columns: {sorted(missing)}")

    df = odds_df.copy()
    df["join_home"] = df["home_team"].map(normalize_team)
    df["join_away"] = df["away_team"].map(normalize_team)
    df["current_home_odds"] = pd.to_numeric(df["current_home_odds"], errors="coerce")
    df["current_away_odds"] = pd.to_numeric(df["current_away_odds"], errors="coerce")
    df = df.dropna(subset=["current_home_odds", "current_away_odds"])

    cons = (
        df.groupby(["date", "join_home", "join_away"], as_index=False)
        .agg(
            home_odds=("current_home_odds", "median"),
            away_odds=("current_away_odds", "median"),
            n_books=("sportsbook", "nunique") if "sportsbook" in df.columns else ("join_home", "count"),
        )
        .rename(columns={"join_home": "home_team_key", "join_away": "away_team_key"})
    )
    return cons


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw and vig-free market probabilities from home_odds / away_odds columns."""
    out = df.copy()
    out["p_home_mkt_raw"] = out["home_odds"].map(american_to_implied_prob)
    out["p_away_mkt_raw"] = out["away_odds"].map(american_to_implied_prob)
    fair = out.apply(
        lambda r: remove_vig(r["p_home_mkt_raw"], r["p_away_mkt_raw"]),
        axis=1,
        result_type="expand",
    )
    out["p_home_mkt"] = fair[0]
    out["p_away_mkt"] = fair[1]
    return out


def pick_bets(
    df: pd.DataFrame,
    *,
    edge_threshold: float = 0.03,
    prob_col: str = "p_home_win",
) -> pd.DataFrame:
    """
    For each game, optionally bet home or away when model edge exceeds threshold.
    Edge = model prob minus fair market prob for that side.
    """
    out = df.copy()
    out["edge_home"] = out[prob_col] - out["p_home_mkt"]
    out["edge_away"] = (1.0 - out[prob_col]) - out["p_away_mkt"]

    bet_side: list[str | None] = []
    bet_odds: list[float | None] = []
    bet_edge: list[float | None] = []

    for _, row in out.iterrows():
        eh, ea = row["edge_home"], row["edge_away"]
        side: str | None = None
        if eh >= edge_threshold and eh >= ea:
            side = "home"
        elif ea >= edge_threshold:
            side = "away"
        if side == "home":
            bet_side.append("home")
            bet_odds.append(float(row["home_odds"]))
            bet_edge.append(float(eh))
        elif side == "away":
            bet_side.append("away")
            bet_odds.append(float(row["away_odds"]))
            bet_edge.append(float(ea))
        else:
            bet_side.append(None)
            bet_odds.append(None)
            bet_edge.append(None)

    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_edge"] = bet_edge
    out["bet_won"] = [
        (s == "home" and bool(w)) or (s == "away" and not bool(w))
        if s is not None
        else np.nan
        for s, w in zip(out["bet_side"], out["home_win"])
    ]
    out["bet_profit"] = [
        flat_bet_profit(o, bool(w)) if s is not None and pd.notna(o) else np.nan
        for s, o, w in zip(out["bet_side"], out["bet_odds"], out["bet_won"])
    ]
    return out


def log_loss(y_true: pd.Series, p: pd.Series, eps: float = 1e-15) -> float:
    p_clip = p.clip(eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))


def brier_score(y_true: pd.Series, p: pd.Series) -> float:
    return float(np.mean((p - y_true) ** 2))
