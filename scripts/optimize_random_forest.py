"""
Hyperparameter optimization for random forest using RandomizedSearchCV.
Uses the same season-based split: train on earlier seasons, tune via CV on training data,
evaluate best model on held-out test season(s).
"""
from pathlib import Path

import joblib

from model_utils import print_feature_importance
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

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

PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
N_ITER = 50
CV_SPLITS = 5


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

    print(f"\nRunning randomized search (n_iter={N_ITER}, cv={CV_SPLITS} TimeSeriesSplit)...")
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  Test accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    print_feature_importance(FEATURE_COLS, best_model.feature_importances_.tolist())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "random_forest_optimized.pkl")
    print(f"\nSaved best model to {MODELS_DIR}/random_forest_optimized.pkl")

    # Save search results and best params
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    results_df.to_csv(TABLES_DIR / "random_forest_search_results.csv", index=False)
    pd.DataFrame([search.best_params_]).to_csv(
        TABLES_DIR / "random_forest_best_params.csv", index=False
    )
    print(f"Saved search results and best params to {TABLES_DIR}/")

    imp = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(TABLES_DIR / "random_forest_optimized_importances.csv", index=False)
    print(f"Saved feature importances to {TABLES_DIR}/random_forest_optimized_importances.csv")


if __name__ == "__main__":
    main()
