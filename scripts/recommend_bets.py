"""
Daily betting recommendations: model vs market moneyline edge.

Fetches today's (or a chosen date's) scheduled MLB games, scores them with the
saved logistic model, pulls current moneylines, and lists flat-bet opportunities
where model probability beats the fair market line by at least --edge (default 5%).

Example:
    python scripts/recommend_bets.py
    python scripts/recommend_bets.py --tomorrow --edge 0.05
    python scripts/recommend_bets.py --date 2026-06-01 --edge 0.03
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from fetch_odds import scrape_sbr
from model_utils import FEATURE_COLS, LAG_WINDOW
from odds_utils import (
    add_market_probs,
    consensus_moneyline,
    format_american_odds,
    normalize_team,
    pick_bets,
)
from predict_2026 import HISTORY_CSV, MODEL_PATH, SCALER_PATH, fetch_schedule, load_history
from rolling_state import RollingFeatureState

SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR.parent / "results" / "tables"


def predict_slate(target: date, season: int) -> pd.DataFrame:
    """
    Model probabilities for games on `target`.

    - Past dates: score that day's games (typically Final) using rolling state
      built only from earlier results (no look-ahead).
    - Today / future: advance state through Finals through today, then score
      not-yet-final games on `target` (live betting slate).
    """
    if not HISTORY_CSV.is_file():
        raise FileNotFoundError(f"Missing {HISTORY_CSV}; run eda.py first.")
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Missing model; run fit_logistic_model.py first.")

    today = date.today()
    start_iso = f"{season}-03-01"
    end_iso = min(max(target, today), date(season, 11, 30)).isoformat()

    history = load_history()
    # Prior seasons only — current-season Finals are applied from the API walk so
    # we do not double-count games already present in the history CSV.
    history_seed = history.loc[history["game_date"].dt.year < season]
    state = RollingFeatureState(window=LAG_WINDOW)
    state.seed_from_completed(history_seed)

    sched = fetch_schedule(start_iso, end_iso)
    if sched.empty:
        return pd.DataFrame()

    sched["game_date"] = pd.to_datetime(sched["game_date"])
    sched = sched[sched["game_date"].dt.year == season].sort_values(["game_date", "game_id"])

    def _apply_final(row: pd.Series) -> None:
        hs = pd.to_numeric(row.get("home_score"), errors="coerce")
        aws = pd.to_numeric(row.get("away_score"), errors="coerce")
        if pd.notna(hs) and pd.notna(aws):
            u = row.copy()
            u["home_score"] = float(hs)
            u["away_score"] = float(aws)
            u["home_win"] = int(hs > aws)
            state.update_after_final_game(u)

    if target < today:
        prior = sched[
            (sched["game_date"].dt.date < target) & (sched["status"] == "Final")
        ]
        for _, row in prior.iterrows():
            _apply_final(row)
        to_predict = sched[sched["game_date"].dt.date == target].copy()
    else:
        advance = sched[
            (sched["game_date"].dt.date <= today) & (sched["status"] == "Final")
        ]
        for _, row in advance.iterrows():
            _apply_final(row)
        to_predict = sched[
            (sched["game_date"].dt.date == target) & (sched["status"] != "Final")
        ].copy()

    if to_predict.empty:
        return pd.DataFrame()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    rows: list[dict] = []
    for _, row in to_predict.iterrows():
        feats = state.features_for_game(row)
        X_df = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
        p_home = float(model.predict_proba(scaler.transform(X_df))[0, 1])
        hs = pd.to_numeric(row.get("home_score"), errors="coerce")
        aws = pd.to_numeric(row.get("away_score"), errors="coerce")
        rows.append(
            {
                "game_id": row["game_id"],
                "game_date": row["game_date"].date().isoformat(),
                "status": row["status"],
                "away_name": row["away_name"],
                "home_name": row["home_name"],
                "away_probable_pitcher": row.get("away_probable_pitcher", ""),
                "home_probable_pitcher": row.get("home_probable_pitcher", ""),
                "away_score": float(aws) if pd.notna(aws) else None,
                "home_score": float(hs) if pd.notna(hs) else None,
                "p_home_win": p_home,
            }
        )
    return pd.DataFrame(rows)


def fetch_live_odds(target: date) -> pd.DataFrame:
    """Current moneylines for one calendar day (SBR consensus)."""
    iso = target.isoformat()
    raw = scrape_sbr(iso, iso)
    if raw.empty:
        return raw
    return raw.loc[raw.get("odds_type", "moneyline") == "moneyline"].copy()


def join_predictions_odds(preds: pd.DataFrame, odds_raw: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or odds_raw.empty:
        return pd.DataFrame()

    cons = consensus_moneyline(odds_raw)
    p = preds.copy()
    p["date"] = p["game_date"]
    p["home_team_key"] = p["home_name"].map(normalize_team)
    p["away_team_key"] = p["away_name"].map(normalize_team)
    merged = p.merge(cons, on=["date", "home_team_key", "away_team_key"], how="left")
    return add_market_probs(merged)


def print_recommendations(recs: pd.DataFrame, *, edge: float, target: date) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Bet recommendations for {target.isoformat()}  (edge ≥ {edge:.0%})")
    print(f"{'=' * 72}")

    if recs.empty:
        print("  No qualifying bets today.")
        if not with_odds.empty:
            best = with_odds.loc[with_odds[["edge_home", "edge_away"]].max(axis=1).idxmax()]
            best_edge = max(best["edge_home"], best["edge_away"])
            best_side = "home" if best["edge_home"] >= best["edge_away"] else "away"
            team = best["home_name"] if best_side == "home" else best["away_name"]
            print(
                f"  Closest: {team} at {best_edge:.1%} edge "
                f"(below {edge:.0%} threshold)."
            )
        return

    for _, r in recs.sort_values("bet_edge", ascending=False).iterrows():
        side = r["bet_side"]
        team = r["home_name"] if side == "home" else r["away_name"]
        p_model = r["p_home_win"] if side == "home" else (1.0 - r["p_home_win"])
        p_mkt = r["p_home_mkt"] if side == "home" else r["p_away_mkt"]
        odds_str = format_american_odds(r["bet_odds"])
        print(
            f"\n  BET {team}  {odds_str}\n"
            f"    {r['away_name']} @ {r['home_name']}\n"
            f"    model {p_model:.1%}  vs  market {p_mkt:.1%}  →  edge {r['bet_edge']:.1%}\n"
            f"    pitchers: {r.get('away_probable_pitcher', '')} vs {r.get('home_probable_pitcher', '')}\n"
            f"    status: {r.get('status', '')}  ({int(r.get('n_books', 0))} books)"
        )

    print(f"\n  Total: {len(recs)} bet(s)")
    print(f"{'=' * 72}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="List moneyline bets where model edge exceeds market (flat 1u strategy)."
    )
    p.add_argument("--season", type=int, default=date.today().year)
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to find bets for YYYY-MM-DD (default: today).",
    )
    p.add_argument(
        "--tomorrow",
        action="store_true",
        help="Use tomorrow's slate instead of today.",
    )
    p.add_argument(
        "--edge",
        type=float,
        default=0.05,
        help="Minimum model edge vs fair market prob (default 0.05 = 5%%).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional CSV path (default: results/tables/bet_recommendations_<date>.csv).",
    )
    args = p.parse_args()

    if args.tomorrow and args.date:
        raise SystemExit("Use either --tomorrow or --date, not both.")

    today = date.today()
    if args.date:
        target = date.fromisoformat(args.date)
    elif args.tomorrow:
        target = today + timedelta(days=1)
    else:
        target = today

    if target < today:
        raise SystemExit("--date must be today or a future date for live recommendations.")

    print(f"Scoring model for {target.isoformat()}...")
    preds = predict_slate(target, args.season)
    if preds.empty:
        print("No upcoming games on that date (off day or all games final).")
        if target == date.today():
            print("  Tip: use --tomorrow for the next day's slate, or run earlier before games start.")
        return
    print(f"  {len(preds)} not-yet-final game(s) with model probabilities.")

    print("Fetching current moneylines...")
    odds_raw = fetch_live_odds(target)
    if odds_raw.empty:
        print("No odds returned. Try again closer to game time.")
        return
    n_odds_games = odds_raw["game_id"].nunique()
    print(f"  {n_odds_games} game(s) with odds from SBR.")
    if n_odds_games > len(preds):
        n_done = n_odds_games - len(preds)
        print(
            f"  Note: {n_done} game(s) on today's slate are already Final — "
            "only not-yet-final games are scored for betting."
        )
        if target == date.today() and len(preds) <= 3:
            print("  Tip: run earlier in the day, or use --tomorrow for the full upcoming slate.")

    merged = join_predictions_odds(preds, odds_raw)
    missing_odds = merged["home_odds"].isna().sum()
    if missing_odds:
        print(f"  Warning: {missing_odds} game(s) could not be matched to odds.")

    with_odds = merged.dropna(subset=["home_odds", "away_odds"]).copy()
    picks = pick_bets(with_odds, edge_threshold=args.edge)
    recs = picks.dropna(subset=["bet_side"]).copy()
    recs["bet_team"] = [
        h if s == "home" else a
        for s, h, a in zip(recs["bet_side"], recs["home_name"], recs["away_name"])
    ]
    recs["bet_odds_fmt"] = recs["bet_odds"].map(format_american_odds)
    recs["p_model_bet"] = [
        ph if s == "home" else 1.0 - ph
        for s, ph in zip(recs["bet_side"], recs["p_home_win"])
    ]
    recs["p_market_bet"] = [
        ph if s == "home" else pa
        for s, ph, pa in zip(recs["bet_side"], recs["p_home_mkt"], recs["p_away_mkt"])
    ]

    print_recommendations(recs, edge=args.edge, target=target)

    out_cols = [
        "game_id",
        "game_date",
        "status",
        "away_name",
        "home_name",
        "bet_side",
        "bet_team",
        "bet_odds",
        "bet_odds_fmt",
        "p_home_win",
        "p_model_bet",
        "p_market_bet",
        "bet_edge",
        "edge_home",
        "edge_away",
        "home_odds",
        "away_odds",
        "n_books",
        "away_probable_pitcher",
        "home_probable_pitcher",
    ]
    out_path = Path(args.output) if args.output else TABLES_DIR / f"bet_recommendations_{target.isoformat()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = picks.copy()
    export["bet_team"] = [
        (h if s == "home" else a) if pd.notna(s) else ""
        for s, h, a in zip(export["bet_side"], export["home_name"], export["away_name"])
    ]
    export["bet_odds_fmt"] = export["bet_odds"].map(
        lambda x: format_american_odds(x) if pd.notna(x) else ""
    )
    export[[c for c in out_cols if c in export.columns]].to_csv(out_path, index=False)
    n_qual = len(recs)
    print(f"Wrote {len(export)} game(s) to {out_path} ({n_qual} with edge ≥ {args.edge:.0%})")


if __name__ == "__main__":
    main()
