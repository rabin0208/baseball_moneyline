"""
Compare model predictions to closing moneylines and simulate flat-bet ROI.

Joins:
  - Model probabilities (from saved logistic model + featured CSV, or --predictions CSV)
  - Closing moneylines (data/odds_moneyline.csv from fetch_odds.py)
  - Actual outcomes (home_win)

Prints accuracy / log-loss / Brier for model vs market, plus ROI at edge thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from model_utils import FEATURE_COLS, TEST_SEASONS
from odds_utils import (
    add_market_probs,
    brier_score,
    consensus_moneyline,
    log_loss,
    normalize_team,
    pick_bets,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"
DEFAULT_ODDS_CSV = DATA_DIR / "odds_moneyline.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def load_games_with_model_probs(season: int) -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.loc[df["game_date"].dt.year == season].copy()
    df = df.dropna(subset=FEATURE_COLS + ["home_win"])

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = df[FEATURE_COLS].astype(float)
    Xs = scaler.transform(X)
    df["p_home_win"] = model.predict_proba(Xs)[:, 1]
    df["pred_home_win"] = (df["p_home_win"] >= 0.5).astype(int)
    return df


def load_predictions(path: Path, season: int | None) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["game_date"] = pd.to_datetime(pred["game_date"])
    if season is not None:
        pred = pred.loc[pred["game_date"].dt.year == season]
    actual = pd.read_csv(FEATURED_CSV)[["game_id", "home_win", "home_name", "away_name"]]
    merged = pred.merge(actual, on="game_id", how="left", suffixes=("", "_act"))
    if "home_name_act" in merged.columns:
        merged["home_name"] = merged["home_name"].fillna(merged["home_name_act"])
        merged["away_name"] = merged["away_name"].fillna(merged["away_name_act"])
    return merged.dropna(subset=["home_win", "p_home_win"])


def join_games_odds(games: pd.DataFrame, odds_raw: pd.DataFrame) -> pd.DataFrame:
    cons = consensus_moneyline(odds_raw)
    g = games.copy()
    g["date"] = g["game_date"].dt.date.astype(str)
    g["home_team_key"] = g["home_name"].map(normalize_team)
    g["away_team_key"] = g["away_name"].map(normalize_team)

    merged = g.merge(
        cons,
        on=["date", "home_team_key", "away_team_key"],
        how="inner",
    )
    return add_market_probs(merged)


def summarize_binary(name: str, y_true: pd.Series, p_home: pd.Series, pred_home: pd.Series) -> None:
    acc = accuracy_score(y_true, pred_home)
    try:
        auc = roc_auc_score(y_true, p_home)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y_true, p_home)
    brier = brier_score(y_true, p_home)
    auc_str = f"{auc:.4f}" if auc == auc else "n/a"
    print(f"  {name:16s}  accuracy={acc:.4f}  roc_auc={auc_str}  log_loss={ll:.4f}  brier={brier:.4f}")


def summarize_roi(df: pd.DataFrame, edge_threshold: float) -> dict:
    bets = pick_bets(df, edge_threshold=edge_threshold)
    placed = bets.dropna(subset=["bet_side"])
    if placed.empty:
        return {"edge_threshold": edge_threshold, "n_bets": 0, "roi": float("nan"), "hit_rate": float("nan")}

    profit = placed["bet_profit"].sum()
    n = len(placed)
    roi = profit / n
    hit = placed["bet_won"].mean()
    return {
        "edge_threshold": edge_threshold,
        "n_bets": n,
        "n_wins": int(placed["bet_won"].sum()),
        "units_wagered": n,
        "units_profit": float(profit),
        "roi": float(roi),
        "hit_rate": float(hit),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model vs closing moneylines.")
    p.add_argument("--season", type=int, default=TEST_SEASONS[0], help="Season to evaluate")
    p.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional predictions CSV with game_id, p_home_win (else score from saved model).",
    )
    p.add_argument(
        "--odds",
        type=str,
        default=str(DEFAULT_ODDS_CSV),
        help="Moneyline CSV from fetch_odds.py",
    )
    p.add_argument(
        "--fetch-odds",
        action="store_true",
        help="Run fetch_odds.py for missing dates before evaluating.",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05",
        help="Comma-separated edge thresholds for ROI (model prob minus market prob).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional per-game CSV output path.",
    )
    args = p.parse_args()

    odds_path = Path(args.odds)
    if args.fetch_odds or not odds_path.exists():
        from fetch_odds import fetch_season_odds

        print("Fetching missing odds...")
        fetch_season_odds(args.season, odds_path, missing_only=True)

    if not odds_path.exists():
        raise FileNotFoundError(
            f"{odds_path} not found. Run: python scripts/fetch_odds.py --season {args.season}"
        )

    if args.predictions:
        games = load_predictions(Path(args.predictions), args.season)
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run fit_logistic_model.py first.")
        games = load_games_with_model_probs(args.season)

    odds_raw = pd.read_csv(odds_path)
    odds_raw = odds_raw.loc[odds_raw.get("odds_type", "moneyline") == "moneyline"].copy()

    merged = join_games_odds(games, odds_raw)
    n_games = len(games)
    n_matched = len(merged)
    print(f"\nSeason {args.season}: {n_games:,} completed games with features")
    print(f"  Matched to closing lines: {n_matched:,} ({100 * n_matched / max(n_games, 1):.1f}%)")

    if merged.empty:
        print("No games matched. Check odds coverage or run fetch_odds.py.")
        return

    merged["pred_home_mkt"] = (merged["p_home_mkt"] >= 0.5).astype(int)
    merged["pred_home_favorite"] = (merged["home_odds"] <= merged["away_odds"]).astype(int)

    print("\nHead-to-head (matched games only):")
    summarize_binary("Model", merged["home_win"], merged["p_home_win"], merged["pred_home_win"])
    summarize_binary("Market (fair)", merged["home_win"], merged["p_home_mkt"], merged["pred_home_mkt"])
    summarize_binary("Favorite", merged["home_win"], merged["p_home_mkt"], merged["pred_home_favorite"])

    mean_vig = (merged["p_home_mkt_raw"] + merged["p_away_mkt_raw"] - 1).mean()
    print(f"\n  Mean overround (vig): {mean_vig:.4f}")

    thresholds = [float(x.strip()) for x in args.edge.split(",") if x.strip()]
    print("\nFlat 1-unit ROI (bet when model edge ≥ threshold):")
    roi_rows = []
    for t in thresholds:
        s = summarize_roi(merged, t)
        roi_rows.append(s)
        if s["n_bets"] == 0:
            print(f"  edge ≥ {t:.0%}: no bets")
        else:
            print(
                f"  edge ≥ {t:.0%}: {s['n_bets']} bets, hit rate {s['hit_rate']:.1%}, "
                f"profit {s['units_profit']:+.2f}u, ROI {s['roi']:+.1%}"
            )

    roi_path = TABLES_DIR / f"market_roi_{args.season}.csv"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(roi_rows).to_csv(roi_path, index=False)
    print(f"\nWrote ROI summary to {roi_path}")

    if args.output:
        out = pick_bets(merged, edge_threshold=min(t for t in thresholds if t > 0) if any(t > 0 for t in thresholds) else 0.03)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "game_id",
            "game_date",
            "home_name",
            "away_name",
            "home_win",
            "p_home_win",
            "p_home_mkt",
            "home_odds",
            "away_odds",
            "edge_home",
            "edge_away",
            "bet_side",
            "bet_odds",
            "bet_edge",
            "bet_won",
            "bet_profit",
            "n_books",
        ]
        out[[c for c in cols if c in out.columns]].to_csv(out_path, index=False)
        print(f"Wrote per-game details to {out_path}")


if __name__ == "__main__":
    main()
