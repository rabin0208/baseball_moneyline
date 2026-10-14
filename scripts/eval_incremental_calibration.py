"""
Compare calibration of the current frozen logistic model vs a biweekly incremental refit.

Frozen (production): fit once on all games with game_date < Jan 1 of --test-year
  (same split as fit_logistic_model.py: train 2018–2025, score 2026).

Incremental: for each 14-day window in the test year, refit on all games strictly
  before the window start (prior seasons plus completed test-year games so far),
  then score games inside the window.

Writes per-period and pooled calibration metrics, a reliability table, and figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from model_utils import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"
TARGET_COL = "home_win"
N_BINS = 10


def load_featured() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    for c in ["home_rest_days", "away_rest_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)


def iter_year_periods(
    season: int, period_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
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


def fit_logistic(train_df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler]:
    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(int)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    return model, scaler


def predict_proba(
    model: LogisticRegression, scaler: StandardScaler, test_df: pd.DataFrame
) -> np.ndarray:
    X_test = test_df[FEATURE_COLS].astype(float)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Weighted |accuracy − confidence| over equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """Unregularized logistic fit of outcome on logit(p). Slope 1 / intercept 0 is ideal."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    logit = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, prefix: str) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_roc_auc": _safe_roc_auc(y_true, y_prob),
        f"{prefix}_brier": float(brier_score_loss(y_true, y_prob)),
        f"{prefix}_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        f"{prefix}_ece": expected_calibration_error(y_true, y_prob),
        f"{prefix}_mean_p": float(y_prob.mean()),
        f"{prefix}_calib_slope": slope,
        f"{prefix}_calib_intercept": intercept,
    }


def reliability_rows(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, n_bins: int = N_BINS
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "bin_index": i,
                "bin_left": float(lo),
                "bin_right": float(hi),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
                "n": k,
            }
        )
    return rows


def save_figures(
    period_df: pd.DataFrame,
    y_true: np.ndarray,
    p_frozen: np.ndarray,
    p_incr: np.ndarray,
    test_year: int,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    labels = [
        pd.Timestamp(s).strftime("%m/%d") for s in period_df["test_start"]
    ]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, probs, color in (
        ("Frozen (2018–2025)", p_frozen, "steelblue"),
        ("Incremental (refit every 14d)", p_incr, "darkorange"),
    ):
        frac, mean_p = calibration_curve(y_true, probs, n_bins=N_BINS, strategy="uniform")
        ax.plot(mean_p, frac, marker="o", color=color, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0.35, 0.75)
    ax.set_ylim(0.35, 0.75)
    ax.set_xlabel("Mean predicted home-win probability")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title(f"Reliability diagram · {test_year} completed games")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / f"calibration_reliability_incremental_vs_frozen_{test_year}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(labels, period_df["frozen_brier"], marker="o", color="steelblue", label="Frozen")
    ax.plot(labels, period_df["incremental_brier"], marker="o", color="darkorange", label="Incremental")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(f"Brier score by 14-day window · {test_year}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"calibration_brier_by_period_{test_year}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(labels, period_df["frozen_ece"], marker="o", color="steelblue", label="Frozen")
    ax.plot(labels, period_df["incremental_ece"], marker="o", color="darkorange", label="Incremental")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Expected calibration error (lower is better)")
    ax.set_title(f"ECE (10 equal-width bins) by 14-day window · {test_year}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"calibration_ece_by_period_{test_year}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p_frozen, bins=20, range=(0.3, 0.8), alpha=0.55, color="steelblue", label="Frozen")
    ax.hist(p_incr, bins=20, range=(0.3, 0.8), alpha=0.55, color="darkorange", label="Incremental")
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Games")
    ax.set_title(f"Predicted-probability sharpness · {test_year}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"calibration_prob_hist_incremental_vs_frozen_{test_year}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def _fmt(x: float, digits: int = 4) -> str:
    if x != x:
        return "n/a"
    return f"{x:.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Biweekly incremental logistic vs frozen model: calibration comparison."
    )
    p.add_argument(
        "--train-through-year",
        type=int,
        default=2025,
        help="Frozen model trains on all games before the next Jan 1 (default 2025).",
    )
    p.add_argument(
        "--test-year",
        type=int,
        default=2026,
        help="Score completed games in this calendar year (default 2026).",
    )
    p.add_argument(
        "--period-days",
        type=int,
        default=14,
        metavar="D",
        help="Refit / test window length in days (default 14 = biweekly).",
    )
    args = p.parse_args()

    train_y = args.train_through_year
    test_y = args.test_year
    if test_y <= train_y:
        raise SystemExit("--test-year must be greater than --train-through-year.")

    frozen_cutoff = pd.Timestamp(f"{train_y + 1}-01-01")

    print(f"Loading {FEATURED_CSV}...")
    df = load_featured()
    print(f"  Rows after dropna(features): {len(df):,}, features: {len(FEATURE_COLS)}")

    frozen_train = df.loc[df["game_date"] < frozen_cutoff]
    test_pool = df.loc[df["game_date"].dt.year == test_y]
    if frozen_train.empty:
        raise SystemExit("Frozen training set is empty.")
    if test_pool.empty:
        raise SystemExit(f"No {test_y} games in featured data.")

    print(
        f"\nFitting frozen logistic on n={len(frozen_train):,} "
        f"(game_date < {frozen_cutoff.date().isoformat()})..."
    )
    frozen_model, frozen_scaler = fit_logistic(frozen_train)

    periods = iter_year_periods(test_y, args.period_days)
    print(
        f"Incremental: refit every {args.period_days} day(s) on all games before each window; "
        f"score {test_y} games in the window.\n"
    )

    period_rows: list[dict[str, Any]] = []
    y_all: list[np.ndarray] = []
    p_frozen_all: list[np.ndarray] = []
    p_incr_all: list[np.ndarray] = []

    for i, (s, e) in enumerate(periods, start=1):
        test_df = test_pool.loc[(test_pool["game_date"] >= s) & (test_pool["game_date"] < e)]
        label = f"Period {i:2d}  test [{s.date().isoformat()}, {e.date().isoformat()})"
        if test_df.empty:
            continue

        incr_train = df.loc[df["game_date"] < s]
        if incr_train.empty:
            print(f"{label}\n  (no training data — skipped)\n")
            continue

        incr_model, incr_scaler = fit_logistic(incr_train)
        y = test_df[TARGET_COL].astype(int).to_numpy()
        p_frozen = predict_proba(frozen_model, frozen_scaler, test_df)
        p_incr = predict_proba(incr_model, incr_scaler, test_df)

        row = {
            "period": i,
            "test_start": s.date().isoformat(),
            "test_end": e.date().isoformat(),
            "n_train_frozen": len(frozen_train),
            "n_train_incremental": len(incr_train),
            "n_test": len(test_df),
            "actual_home_win_rate": float(y.mean()),
        }
        row.update(metrics_dict(y, p_frozen, "frozen"))
        row.update(metrics_dict(y, p_incr, "incremental"))
        period_rows.append(row)

        y_all.append(y)
        p_frozen_all.append(p_frozen)
        p_incr_all.append(p_incr)

        print(label)
        print(
            f"  n_test={len(y):,}  incr train n={len(incr_train):,}  "
            f"home win rate={y.mean():.3f}"
        )
        print(
            f"  frozen       Brier={_fmt(row['frozen_brier'])}  ECE={_fmt(row['frozen_ece'])}  "
            f"logloss={_fmt(row['frozen_log_loss'])}  AUC={_fmt(row['frozen_roc_auc'])}"
        )
        print(
            f"  incremental  Brier={_fmt(row['incremental_brier'])}  ECE={_fmt(row['incremental_ece'])}  "
            f"logloss={_fmt(row['incremental_log_loss'])}  AUC={_fmt(row['incremental_roc_auc'])}\n"
        )

    if not period_rows:
        print("No test periods had games.")
        return

    y_true = np.concatenate(y_all)
    p_frozen = np.concatenate(p_frozen_all)
    p_incr = np.concatenate(p_incr_all)

    pooled = {
        "split": "pooled_out_of_sample",
        "test_year": test_y,
        "n_test": int(len(y_true)),
        "n_train_frozen": int(len(frozen_train)),
        "actual_home_win_rate": float(y_true.mean()),
    }
    pooled.update(metrics_dict(y_true, p_frozen, "frozen"))
    pooled.update(metrics_dict(y_true, p_incr, "incremental"))

    print("=" * 72)
    print(
        f"Pooled {test_y} out-of-sample  n={len(y_true):,}  "
        f"actual home win rate={y_true.mean():.4f}"
    )
    print(
        f"  frozen       Brier={_fmt(pooled['frozen_brier'])}  ECE={_fmt(pooled['frozen_ece'])}  "
        f"logloss={_fmt(pooled['frozen_log_loss'])}  AUC={_fmt(pooled['frozen_roc_auc'])}  "
        f"acc={_fmt(pooled['frozen_accuracy'])}"
    )
    print(
        f"  incremental  Brier={_fmt(pooled['incremental_brier'])}  ECE={_fmt(pooled['incremental_ece'])}  "
        f"logloss={_fmt(pooled['incremental_log_loss'])}  AUC={_fmt(pooled['incremental_roc_auc'])}  "
        f"acc={_fmt(pooled['incremental_accuracy'])}"
    )
    print(
        f"  frozen       calib slope={_fmt(pooled['frozen_calib_slope'])}  "
        f"intercept={_fmt(pooled['frozen_calib_intercept'])}  mean p={_fmt(pooled['frozen_mean_p'])}"
    )
    print(
        f"  incremental  calib slope={_fmt(pooled['incremental_calib_slope'])}  "
        f"intercept={_fmt(pooled['incremental_calib_intercept'])}  mean p={_fmt(pooled['incremental_mean_p'])}"
    )
    d_brier = pooled["incremental_brier"] - pooled["frozen_brier"]
    d_ece = pooled["incremental_ece"] - pooled["frozen_ece"]
    winner = "incremental" if d_brier < 0 else "frozen"
    print(
        f"\n  Δ Brier (incr − frozen) = {d_brier:+.5f}   "
        f"Δ ECE = {d_ece:+.5f}   better Brier: {winner}"
    )
    print("=" * 72)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    period_df = pd.DataFrame(period_rows)
    period_path = TABLES_DIR / f"incremental_vs_frozen_calibration_{test_y}.csv"
    period_df.to_csv(period_path, index=False)
    print(f"\nWrote per-period metrics to {period_path}")

    summary_path = TABLES_DIR / f"incremental_vs_frozen_calibration_summary_{test_y}.csv"
    pd.DataFrame([pooled]).to_csv(summary_path, index=False)
    print(f"Wrote pooled summary to {summary_path}")

    rel_rows = reliability_rows(y_true, p_frozen, "frozen") + reliability_rows(
        y_true, p_incr, "incremental"
    )
    rel_path = TABLES_DIR / f"incremental_vs_frozen_reliability_{test_y}.csv"
    pd.DataFrame(rel_rows).to_csv(rel_path, index=False)
    print(f"Wrote reliability bins to {rel_path}")

    save_figures(period_df, y_true, p_frozen, p_incr, test_y)
    print("Done.")


if __name__ == "__main__":
    main()
