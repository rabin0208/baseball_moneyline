"""
Fit a logistic regression model to predict home team win (home_win).
Uses a season-based train/test split: train on earlier seasons, test on holdout season(s).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

FEATURED_CSV = DATA_DIR / "schedule_8_seasons_featured.csv"
TEST_SEASONS = [2025]

FEATURE_COLS = [
    "home_rolling_avg_wins_10",
    "away_rolling_avg_wins_10",
    "home_rolling_avg_runs_10",
    "away_rolling_avg_runs_10",
    "home_rolling_avg_runs_allowed_10",
    "away_rolling_avg_runs_allowed_10",
    "home_rolling_avg_run_diff_10",
    "away_rolling_avg_run_diff_10",
    "home_rolling_avg_h2h_wins_10",
    "home_rest_days",
    "away_rest_days",
    "home_pitcher_rolling_wins_centered_10",
    "away_pitcher_rolling_wins_centered_10",
]
TARGET_COL = "home_win"


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load featured CSV, drop rows with missing features, sort by date. Return X, y, season."""
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year
    df = df.sort_values("game_date").reset_index(drop=True)

    for c in ["home_rest_days", "away_rest_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)
    season = df["season"]
    return X, y, season


def season_split(
    X: pd.DataFrame, y: pd.Series, season: pd.Series, test_seasons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by season: train on seasons not in test_seasons, test on test_seasons."""
    train_mask = ~season.isin(test_seasons)
    test_mask = season.isin(test_seasons)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def main():
    print(f"Loading {FEATURED_CSV}...")
    X, y, season = load_and_prepare()
    print(f"  Samples: {len(X)}, features: {len(FEATURE_COLS)}")

    X_train, X_test, y_train, y_test = season_split(X, y, season, TEST_SEASONS)
    print(f"  Train: {len(y_train)} (seasons not in {TEST_SEASONS}), test: {len(y_test)} (seasons {TEST_SEASONS})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nFitting logistic regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"\nSaved model and scaler to {MODELS_DIR}/")

    coef = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_path = PROJECT_ROOT / "results" / "tables" / "logistic_regression_coefficients.csv"
    coef_path.parent.mkdir(parents=True, exist_ok=True)
    coef.to_csv(coef_path, index=False)
    print(f"Saved coefficients to {coef_path}")


if __name__ == "__main__":
    main()
