"""
Biweekly walk-forward evaluation for a full calendar season (default 2025).

For each consecutive period of `period_days` (default 14) from Jan 1 through Dec 31:
  - Train: all games with game_date < period_start
  - Test:  games in [period_start, period_end)

Skips periods with no test games. Uses the same logistic pipeline as fit_logistic_model.py
(no saved model). Optionally writes a metrics table to results/tables/.
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


def fit_eval_logistic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    X_train = train_df[FEATURE_COLS].astype(float)
    X_test = test_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = float(accuracy_score(y_test, y_pred)) if len(y_test) else float("nan")
    auc = _safe_roc_auc(y_test, pd.Series(y_prob))

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": acc,
        "roc_auc": auc,
    }


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
        description="Biweekly (or custom length) walk-forward eval for a full calendar year."
    )
    p.add_argument("--season", type=int, default=2025, help="Calendar year (default 2025).")
    p.add_argument(
        "--period-days",
        type=int,
        default=14,
        metavar="D",
        help="Length of each test period in days (default 14 = biweekly).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"Optional CSV path for per-period metrics (default: {TABLES_DIR}/walkforward_biweekly_<season>.csv).",
    )
    args = p.parse_args()

    season = args.season
    period_days = args.period_days

    print(f"Loading {FEATURED_CSV}...")
    df = load_featured()
    print(f"  Rows after dropna(features): {len(df):,}, features: {len(FEATURE_COLS)}")

    periods = iter_year_periods(season, period_days)
    print(
        f"\nWalk-forward: {len(periods)} period(s) of up to {period_days} day(s) "
        f"covering {season}-01-01 … {season}-12-31"
    )
    print("For each period: train on all games strictly before the period start; test inside the period.\n")

    rows_out: list[dict[str, Any]] = []
    accs: list[float] = []
    aucs: list[float] = []

    for i, (s, e) in enumerate(periods, start=1):
        train_df = df.loc[df["game_date"] < s]
        test_df = df.loc[(df["game_date"] >= s) & (df["game_date"] < e)]

        label = (
            f"Period {i:2d}  test [{s.date().isoformat()}, {e.date().isoformat()})"
        )

        if test_df.empty:
            print(f"{label}")
            print(f"  (no games in test window — skipped)\n")
            continue
        if train_df.empty:
            print(f"{label}")
            print(f"  (no training data before window — skipped)\n")
            continue

        m = fit_eval_logistic(train_df, test_df)
        acc = m["accuracy"]
        auc = m["roc_auc"]
        accs.append(acc)
        if auc == auc:  # not NaN
            aucs.append(auc)

        auc_str = f"{auc:.4f}" if auc == auc else "n/a"
        print(f"{label}")
        print(
            f"  train n={m['n_train']:,}  test n={m['n_test']:,}  "
            f"accuracy={acc:.4f}  roc_auc={auc_str}\n"
        )

        rows_out.append(
            {
                "period": i,
                "test_start": s.date().isoformat(),
                "test_end": e.date().isoformat(),
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "accuracy": acc,
                "roc_auc": auc,
            }
        )

    if rows_out:
        print(
            f"Summary over {len(rows_out)} period(s) with games: "
            f"mean accuracy = {mean(accs):.4f}"
        )
        if aucs:
            print(f"mean ROC-AUC     = {mean(aucs):.4f}  (over {len(aucs)} period(s) with both classes in test)")
        else:
            print("mean ROC-AUC     = n/a")

        out_path = (
            Path(args.output)
            if args.output
            else TABLES_DIR / f"walkforward_biweekly_{season}.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(out_path, index=False)
        print(f"\nWrote per-period metrics to {out_path}")
    else:
        print("No periods produced metrics (no overlapping games with train/test splits).")

    print("Done.")


if __name__ == "__main__":
    main()
