"""
Shared utilities for model fitting and evaluation.
"""


def print_feature_importance(feature_names: list[str], values: list[float], title: str = "Feature importance") -> None:
    """Print feature names and their importance/coefficient values, sorted by absolute value (descending)."""
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {title} (sorted by |value|):")
    for name, val in pairs:
        print(f"    {name}: {val:.4f}")
