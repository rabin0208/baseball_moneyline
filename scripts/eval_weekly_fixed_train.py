"""
Fit logistic regression once on all games through end of `--train-through-year`, then
evaluate on each calendar period (default 7 days = weekly) in `--test-year`.

Unlike walk-forward, the model is not refit between periods—only metrics are computed
per week. Uses the same features as fit_logistic_model.py (no saved model).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from model_utils import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"
TARGET_COL = "home_win"


def load_featured() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    for c in ["home_rest_days", "away_rest_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def _safe_roc_auc(y_true: pd.Series, y_prob) -> float:
    if len(y_true) == 0:
        return float("nan")
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def iter_year_periods(
    season: int, period_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping [start, end) slices covering [season-01-01, season+1-01-01)."""
    year_start = pd.Timestamp(f"{season}-01-01")
    year_end = pd.Timestamp(f"{season + 1}-01-01")
    step = pd.Timedelta(days=period_days)
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    s = year_start
    while s < year_end:
        e = min(s + step, year_end)
        periods.append((s, e))
        s = e
    return periods


def main() -> None:
    p = argparse.ArgumentParser(
        description="One model trained through end of train-through year; weekly (or custom) test slices in test year."
    )
    p.add_argument(
        "--train-through-year",
        type=int,
        default=2025,
        help="Use all games with game_date before the next Jan 1 (default 2025 → train through 2025-12-31).",
    )
    p.add_argument(
        "--test-year",
        type=int,
        default=2026,
        help="Evaluate on games in this calendar year only (default 2026). Must be after train-through-year.",
    )
    p.add_argument(
        "--period-days",
        type=int,
        default=7,
        metavar="D",
        help="Length of each test window in days (default 7 = weekly).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for per-period metrics.",
    )
    args = p.parse_args()

    train_y = args.train_through_year
    test_y = args.test_year
    if test_y <= train_y:
        raise SystemExit(
            f"--test-year ({test_y}) must be greater than --train-through-year ({train_y}) "
            "so the test set is not included in training."
        )

    train_cutoff = pd.Timestamp(f"{train_y + 1}-01-01")

    print(f"Loading {FEATURED_CSV}...")
    df = load_featured()
    print(f"  Rows after dropna(features): {len(df):,}, features: {len(FEATURE_COLS)}")

    train_df = df.loc[df["game_date"] < train_cutoff].copy()
    test_pool = df.loc[df["game_date"].dt.year == test_y].copy()

    if train_df.empty:
        raise SystemExit("Training set is empty; check featured data and --train-through-year.")
    if test_pool.empty:
        raise SystemExit(
            f"No games in {test_y} in featured data. Rebuild data or choose another --test-year."
        )

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    print(
        f"\nFitted one model on n={len(y_train):,} games (game_date < {train_cutoff.date().isoformat()})."
    )
    print(
        f"Evaluating on {test_y} in windows of {args.period_days} day(s) (Jan 1 … Dec 31).\n"
    )

    periods = iter_year_periods(test_y, args.period_days)
    rows_out: list[dict[str, Any]] = []
    accs: list[float] = []
    aucs: list[float] = []

    for i, (s, e) in enumerate(periods, start=1):
        test_week = test_pool.loc[
            (test_pool["game_date"] >= s) & (test_pool["game_date"] < e)
        ]
        label = f"Period {i:2d}  test [{s.date().isoformat()}, {e.date().isoformat()})"

        if test_week.empty:
            print(f"{label}")
            print("  (no games — skipped)\n")
            continue

        X_test = test_week[FEATURE_COLS].astype(float)
        y_test = test_week[TARGET_COL].astype(int)
        X_test_s = scaler.transform(X_test)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        acc = float(accuracy_score(y_test, y_pred))
        auc = _safe_roc_auc(y_test, pd.Series(y_prob))
        accs.append(acc)
        if auc == auc:
            aucs.append(auc)

        auc_str = f"{auc:.4f}" if auc == auc else "n/a"
        print(f"{label}")
        print(f"  test n={len(y_test):,}  accuracy={acc:.4f}  roc_auc={auc_str}\n")

        rows_out.append(
            {
                "period": i,
                "test_start": s.date().isoformat(),
                "test_end": e.date().isoformat(),
                "n_test": len(y_test),
                "accuracy": acc,
                "roc_auc": auc,
            }
        )

    if rows_out:
        print(
            f"Summary over {len(rows_out)} period(s) with games: mean accuracy = {mean(accs):.4f}"
        )
        if aucs:
            print(
                f"mean ROC-AUC     = {mean(aucs):.4f}  (over {len(aucs)} period(s) with both classes in test)"
            )
        else:
            print("mean ROC-AUC     = n/a")

        out_path = (
            Path(args.output)
            if args.output
            else TABLES_DIR / f"weekly_fixed_train_{train_y}_test_{test_y}.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(out_path, index=False)
        print(f"\nWrote per-period metrics to {out_path}")
    else:
        print("No test periods had games.")

    print("Done.")


if __name__ == "__main__":
    main()
