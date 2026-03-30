"""
Fetch a season's schedule from the MLB Stats API and score each game with the trained
logistic regression model. Seeds rolling features from data/schedule_8_seasons_final.csv
(2018–2025), then walks games in date order: predict, then update state for Final games.

By default, only completed games (status Final) are scored; the fetch range ends today
(or use --forecast for the full season including future games).

Use --next-day to predict the scheduled slate for tomorrow (state is advanced through all
Final games through today first).
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests

from data_load import SCHEDULE_URL, _game_to_row
from fit_logistic_model import FEATURE_COLS
from rolling_state import RollingFeatureState, ROLLING_WINDOW

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

HISTORY_CSV = DATA_DIR / "schedule_8_seasons_final.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def fetch_schedule(start_iso: str, end_iso: str) -> pd.DataFrame:
    hydrate = "decisions,probablePitcher(note),linescore"
    params = {
        "sportId": 1,
        "startDate": start_iso,
        "endDate": end_iso,
        "hydrate": hydrate,
    }
    resp = requests.get(SCHEDULE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for date_block in data.get("dates") or []:
        for g in date_block.get("games") or []:
            rows.append(_game_to_row(g))
    return pd.DataFrame(rows)


def load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict home-win probability for MLB games.")
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season year to fetch (default: 2026).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: season March 1).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD. Default: today (completed games only) or Nov 30 with --forecast.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Include scheduled/unplayed games through --end (default end: end of season). "
        "Without this flag, only Final games are scored and --end defaults to today.",
    )
    parser.add_argument(
        "--next-day",
        action="store_true",
        help="Predict the next calendar day's scheduled games only. Rolling state is updated "
        "using every Final game from the season start through today, then probabilities "
        "are produced for that day's matchups (cannot combine with --forecast).",
    )
    parser.add_argument(
        "--predict-date",
        type=str,
        default=None,
        help="With --next-day, date YYYY-MM-DD to predict (default: tomorrow).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/tables/predictions_<season>.csv).",
    )
    args = parser.parse_args()

    if args.next_day and args.forecast:
        raise SystemExit("--next-day cannot be used with --forecast.")

    season = args.season
    start_iso = args.start or f"{season}-03-01"
    season_end = date(season, 11, 30)
    today = date.today()
    season_start = date(season, 3, 1)

    if args.next_day:
        if args.predict_date:
            predict_d = date.fromisoformat(args.predict_date)
        else:
            predict_d = today + timedelta(days=1)
        if predict_d < today:
            raise SystemExit("--predict-date must be today or a future date.")
        end_fetch = min(max(predict_d, today), season_end)
        end_iso = end_fetch.isoformat()
    elif args.end is not None:
        end_iso = args.end
    elif args.forecast:
        end_iso = season_end.isoformat()
    else:
        cap = min(today, season_end)
        end_d = max(season_start, cap)
        end_iso = end_d.isoformat()

    if pd.to_datetime(start_iso).date() > pd.to_datetime(end_iso).date():
        raise ValueError(f"start {start_iso} is after end {end_iso}")

    if not HISTORY_CSV.is_file():
        raise FileNotFoundError(f"Missing {HISTORY_CSV}; run eda.py first.")
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Missing model under {MODELS_DIR}; run fit_logistic_model.py first.")

    print(f"Loading history from {HISTORY_CSV}...")
    history = load_history()
    print(f"  {len(history)} completed games for seeding.")

    state = RollingFeatureState(window=ROLLING_WINDOW)
    state.seed_from_completed(history)

    if args.next_day:
        mode = f"next-day predictions for {predict_d.isoformat()}"
    else:
        mode = "full-season forecast" if args.forecast else "completed games only"
    print(f"Fetching {start_iso} – {end_iso} from MLB Stats API ({mode})...")
    sched = fetch_schedule(start_iso, end_iso)
    if sched.empty:
        print("No games in range.")
        return

    sched["game_date"] = pd.to_datetime(sched["game_date"])
    sched = sched[sched["game_date"].dt.year == season].sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    out_rows: list[dict] = []

    if args.next_day:
        advance = sched[
            (sched["game_date"].dt.date <= today) & (sched["status"] == "Final")
        ].copy()
        print(f"  Advancing state on {len(advance)} Final games through {today.isoformat()}...")
        for _, row in advance.iterrows():
            hs = pd.to_numeric(row.get("home_score"), errors="coerce")
            aws = pd.to_numeric(row.get("away_score"), errors="coerce")
            if pd.notna(hs) and pd.notna(aws):
                u = row.copy()
                u["home_score"] = float(hs)
                u["away_score"] = float(aws)
                u["home_win"] = int(hs > aws)
                state.update_after_final_game(u)

        to_predict = sched[
            (sched["game_date"].dt.date == predict_d) & (sched["status"] != "Final")
        ].copy()
        print(f"  Predicting {len(to_predict)} not-yet-final games on {predict_d.isoformat()}...")
        if to_predict.empty:
            print("No upcoming games on that date (off day or all games final).")
            return
        for _, row in to_predict.iterrows():
            feats = state.features_for_game(row)
            X_df = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            Xs = scaler.transform(X_df)
            p_home = float(model.predict_proba(Xs)[0, 1])
            pred_home = int(p_home >= 0.5)
            out_rows.append(
                {
                    "game_id": row["game_id"],
                    "game_date": row["game_date"].date().isoformat(),
                    "status": row["status"],
                    "away_name": row["away_name"],
                    "home_name": row["home_name"],
                    "away_probable_pitcher": row.get("away_probable_pitcher", ""),
                    "home_probable_pitcher": row.get("home_probable_pitcher", ""),
                    "p_home_win": p_home,
                    "pred_home_win": pred_home,
                }
            )
    else:
        if not args.forecast:
            n_before = len(sched)
            sched = sched[sched["status"] == "Final"].copy()
            print(f"  Kept {len(sched)} Final games (dropped {n_before - len(sched)} not yet played).")
            if sched.empty:
                print("No completed games in range.")
                return

        for _, row in sched.iterrows():
            feats = state.features_for_game(row)
            # Match training: scaler was fit on a DataFrame so it expects column names.
            X_df = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            Xs = scaler.transform(X_df)
            p_home = float(model.predict_proba(Xs)[0, 1])
            pred_home = int(p_home >= 0.5)

            out_rows.append(
                {
                    "game_id": row["game_id"],
                    "game_date": row["game_date"].date().isoformat(),
                    "status": row["status"],
                    "away_name": row["away_name"],
                    "home_name": row["home_name"],
                    "away_probable_pitcher": row.get("away_probable_pitcher", ""),
                    "home_probable_pitcher": row.get("home_probable_pitcher", ""),
                    "p_home_win": p_home,
                    "pred_home_win": pred_home,
                }
            )

            if row["status"] == "Final":
                hs = pd.to_numeric(row.get("home_score"), errors="coerce")
                aws = pd.to_numeric(row.get("away_score"), errors="coerce")
                if pd.notna(hs) and pd.notna(aws):
                    u = row.copy()
                    u["home_score"] = float(hs)
                    u["away_score"] = float(aws)
                    u["home_win"] = int(hs > aws)
                    state.update_after_final_game(u)

    out_df = pd.DataFrame(out_rows)
    if args.next_day:
        default_name = f"predictions_next_day_{predict_d.isoformat()}.csv"
        out_path = Path(args.output) if args.output else TABLES_DIR / default_name
    else:
        out_path = Path(args.output) if args.output else TABLES_DIR / f"predictions_{season}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
