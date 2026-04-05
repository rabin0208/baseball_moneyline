"""
Shared utilities for model fitting and evaluation.
"""

# Holdout season for train/test evaluation in fit_* and optimize_* scripts.
# Training uses all other seasons present in schedule_8_seasons_featured.csv
# (typically 2018 through year before TEST_SEASONS).
TEST_SEASONS: list[int] = [2026]


def verify_test_set(y_test, test_seasons: list[int]) -> None:
    """Raise with a clear message if the holdout season has no rows in the featured CSV."""
    if len(y_test) == 0:
        raise RuntimeError(
            f"No rows for test seasons {test_seasons}. "
            "Re-run data_load.py (fetches the current season when needed), then eda.py and split_n_preprocess.py."
        )


def print_feature_importance(feature_names: list[str], values: list[float], title: str = "Feature importance") -> None:
    """Print feature names and their importance/coefficient values, sorted by absolute value (descending)."""
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {title} (sorted by |value|):")
    for name, val in pairs:
        print(f"    {name}: {val:.4f}")
