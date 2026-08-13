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
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    # SBR often duplicates the nickname after the A's relocated ("Athletics Athletics").
    return _TEAM_ALIASES.get(key, key)


# Map alternate book / API spellings onto a single join key.
_TEAM_ALIASES = {
    "athletics athletics": "athletics",
    "oakland athletics": "athletics",
}

def is_valid_american_odds(odds: float) -> bool:
    """American moneylines are <= -100 or >= +100."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return False
    o = float(odds)
    return o <= -100.0 or o >= 100.0


def american_to_implied_prob(odds: float) -> float:
    """Convert American moneyline to implied win probability (includes vig)."""
    o = float(odds)
    if o == 0:
        return float("nan")
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def american_to_decimal(odds: float) -> float:
    """Convert American moneyline to European decimal odds."""
    o = float(odds)
    if not is_valid_american_odds(o):
        return float("nan")
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / abs(o)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert European decimal odds back to American moneyline."""
    d = float(decimal_odds)
    if np.isnan(d) or d <= 1.0:
        return float("nan")
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


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


# Live sizing: believe only half the model's disagreement with the market,
# bet a quarter of full Kelly, and cap exposure.
MARKET_SHRINK = 0.5
KELLY_MULTIPLIER = 0.25
MAX_BET_FRAC = 0.05
SLATE_CAP_FRAC = 0.10


def kelly_fraction(p: float, american_odds: float) -> float:
    """
    Full-Kelly fraction of bankroll for a win/lose moneyline.

    f* = (p * d - 1) / (d - 1), where d is the decimal price actually paid.
    Negative means the posted odds are -EV at probability p.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return float("nan")
    d = american_to_decimal(american_odds)
    if np.isnan(d) or d <= 1.0:
        return float("nan")
    return (float(p) * d - 1.0) / (d - 1.0)


def recommended_bankroll_frac(
    p_model: float,
    p_market: float,
    american_odds: float,
    *,
    shrink: float = MARKET_SHRINK,
    multiplier: float = KELLY_MULTIPLIER,
    max_frac: float = MAX_BET_FRAC,
) -> float:
    """
    Fraction of bankroll to bet on one moneyline.

    Uses a shrink toward the market probability, then fractional Kelly, then a
    per-bet cap. Returns 0 when the blended probability is not +EV at the price.
    """
    try:
        p_model_f = float(p_model)
        p_market_f = float(p_market)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(p_model_f) or np.isnan(p_market_f):
        return float("nan")
    p_hat = (1.0 - shrink) * p_model_f + shrink * p_market_f
    full = kelly_fraction(p_hat, american_odds)
    if np.isnan(full) or full <= 0:
        return 0.0
    return float(min(multiplier * full, max_frac))


def cap_slate_stakes(
    stake_frac: pd.Series,
    *,
    cap: float = SLATE_CAP_FRAC,
) -> pd.Series:
    """Scale a day's recommended stakes so they do not exceed `cap` of bankroll."""
    s = pd.to_numeric(stake_frac, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total <= 0 or total <= cap:
        return s
    return s * (cap / total)


def normalize_stakes_mean_one(stakes: pd.Series) -> pd.Series:
    """Rescale stakes so the mean is 1 (same total wagered as 1u flats)."""
    s = pd.to_numeric(stakes, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    n = len(s)
    if total <= 0 or n == 0:
        return s
    return s * (n / total)


def roi_on_wagered(profit: pd.Series, stake: pd.Series) -> float:
    """Profit / amount wagered. Independent of a constant stake scale."""
    wagered = float(pd.to_numeric(stake, errors="coerce").fillna(0.0).sum())
    if wagered <= 0:
        return float("nan")
    return float(pd.to_numeric(profit, errors="coerce").fillna(0.0).sum()) / wagered


def consensus_moneyline(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Median closing moneyline per game across sportsbooks.

    Medians are taken in decimal-odds space so pick'em books that straddle
    +100/-100 do not average to an invalid American price like -1. Results are
    converted back to American for display and betting.

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
    df = df[
        df["current_home_odds"].map(is_valid_american_odds)
        & df["current_away_odds"].map(is_valid_american_odds)
    ]

    df["home_decimal"] = df["current_home_odds"].map(american_to_decimal)
    df["away_decimal"] = df["current_away_odds"].map(american_to_decimal)
    df = df.dropna(subset=["home_decimal", "away_decimal"])

    cons = (
        df.groupby(["date", "join_home", "join_away"], as_index=False)
        .agg(
            home_decimal=("home_decimal", "median"),
            away_decimal=("away_decimal", "median"),
            n_books=("sportsbook", "nunique") if "sportsbook" in df.columns else ("join_home", "count"),
        )
        .rename(columns={"join_home": "home_team_key", "join_away": "away_team_key"})
    )
    cons["home_odds"] = cons["home_decimal"].map(decimal_to_american)
    cons["away_odds"] = cons["away_decimal"].map(decimal_to_american)
    return cons.drop(columns=["home_decimal", "away_decimal"])


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
    bet_prob: list[float | None] = []
    bet_mkt: list[float | None] = []

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
            bet_prob.append(float(row[prob_col]))
            mkt = row.get("p_home_mkt")
            bet_mkt.append(float(mkt) if pd.notna(mkt) else None)
        elif side == "away":
            bet_side.append("away")
            bet_odds.append(float(row["away_odds"]))
            bet_edge.append(float(ea))
            bet_prob.append(1.0 - float(row[prob_col]))
            mkt = row.get("p_away_mkt")
            bet_mkt.append(float(mkt) if pd.notna(mkt) else None)
        else:
            bet_side.append(None)
            bet_odds.append(None)
            bet_edge.append(None)
            bet_prob.append(None)
            bet_mkt.append(None)

    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_edge"] = bet_edge
    out["bet_prob"] = bet_prob
    out["bet_mkt_prob"] = bet_mkt
    out["kelly_f"] = [
        kelly_fraction(p, o) if p is not None and o is not None else np.nan
        for p, o in zip(bet_prob, bet_odds)
    ]
    out["kelly_stake"] = out["kelly_f"].clip(lower=0.0, upper=1.0)
    out["p_shrunk"] = [
        (1.0 - MARKET_SHRINK) * p + MARKET_SHRINK * m
        if p is not None and m is not None
        else np.nan
        for p, m in zip(bet_prob, bet_mkt)
    ]
    out["kelly_f_shrunk"] = [
        kelly_fraction(p, o) if pd.notna(p) and o is not None else np.nan
        for p, o in zip(out["p_shrunk"], bet_odds)
    ]
    out["stake_frac"] = [
        recommended_bankroll_frac(p, m, o)
        if p is not None and m is not None and o is not None
        else np.nan
        for p, m, o in zip(bet_prob, bet_mkt, bet_odds)
    ]
    if "home_win" in out.columns:
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
        out["kelly_profit"] = [
            float(s) * flat_bet_profit(o, bool(w))
            if pd.notna(s) and s is not None and pd.notna(o) and pd.notna(w)
            else np.nan
            for s, o, w in zip(out["kelly_stake"], out["bet_odds"], out["bet_won"])
        ]
    return out


def format_american_odds(odds: float) -> str:
    o = int(round(float(odds)))
    return f"+{o}" if o > 0 else str(o)


def format_decimal_odds(american_odds: float) -> str:
    """Display American moneylines as European decimal odds (e.g. 1.91)."""
    d = american_to_decimal(american_odds)
    if np.isnan(d):
        return "—"
    return f"{d:.2f}"


def log_loss(y_true: pd.Series, p: pd.Series, eps: float = 1e-15) -> float:
    p_clip = p.clip(eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))


def brier_score(y_true: pd.Series, p: pd.Series) -> float:
    return float(np.mean((p - y_true) ** 2))
