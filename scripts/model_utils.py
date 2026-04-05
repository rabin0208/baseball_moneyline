"""
Shared utilities for model fitting and evaluation.
"""
from __future__ import annotations

LAG_WINDOW = 10


def team_lag_column_names() -> list[str]:
    """Per-game lags (lag_1 = most recent); win/runs/runs_allowed/run_diff × home/away."""
    names: list[str] = []
    for side in ("home", "away"):
        for stat in ("win", "runs", "runs_allowed", "run_diff"):
            for k in range(1, LAG_WINDOW + 1):
                names.append(f"{side}_{stat}_lag_{k}")
    return names


def h2h_lag_column_names() -> list[str]:
    """Did current home team win in each of the last K head-to-head meetings (0 if none)."""
    return [f"home_h2h_win_lag_{k}" for k in range(1, LAG_WINDOW + 1)]


def pitcher_lag_column_names() -> list[str]:
    """Per-start win encoded as win - 0.5; 0 if no start at that lag."""
    names: list[str] = []
    for side in ("home", "away"):
        for k in range(1, LAG_WINDOW + 1):
            names.append(f"{side}_pitcher_win_lag_{k}")
    return names


def feature_column_names() -> list[str]:
    """Column order matches split_n_preprocess.py and rolling_state.RollingFeatureState."""
    return (
        team_lag_column_names()
        + h2h_lag_column_names()
        + ["home_rest_days", "away_rest_days"]
        + pitcher_lag_column_names()
    )


FEATURE_COLS: list[str] = feature_column_names()


def lag_vector(values: list, window: int, pad: float = 0.0) -> list[float]:
    """Lag k = k-th most recent game; `values` is chronological (oldest first)."""
    out: list[float] = []
    for k in range(1, window + 1):
        if len(values) >= k:
            out.append(float(values[-k]))
        else:
            out.append(pad)
    return out


def lag_vector_pitcher_centered(win_vals: list[int], window: int) -> list[float]:
    """Per-start (win - 0.5); missing lags → 0.0."""
    out: list[float] = []
    for k in range(1, window + 1):
        if len(win_vals) >= k:
            out.append(float(win_vals[-k]) - 0.5)
        else:
            out.append(0.0)
    return out


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


def print_feature_importance(
    feature_names: list[str],
    values: list[float],
    title: str = "Feature importance",
    *,
    limit: int | None = 40,
) -> None:
    """Print feature names and values sorted by |value|. If many features, show top `limit` only."""
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {title} (sorted by |value|):")
    if limit is not None and len(pairs) > limit:
        pairs = pairs[:limit]
        print(f"    (showing top {limit} of {len(feature_names)})")
    for name, val in pairs:
        print(f"    {name}: {val:.4f}")
