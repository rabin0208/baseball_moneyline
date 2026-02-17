"""
Exploratory data analysis for MLB Stats API schedule data.
Saves tables to results/tables and figures to results/figures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Date range for EDA (enough completed games for distributions)
EDA_START = "04/01/2024"
EDA_END = "07/31/2024"


def fetch_schedule():
    """Fetch schedule from MLB Stats API and return as DataFrame."""
    import statsapi

    games = statsapi.schedule(start_date=EDA_START, end_date=EDA_END)
    return pd.DataFrame(games)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed games and add derived columns."""
    completed = df[df["status"] == "Final"].copy()
    if completed.empty:
        return completed

    # Numeric scores (API may return empty string for postponed)
    completed["away_score"] = pd.to_numeric(completed["away_score"], errors="coerce")
    completed["home_score"] = pd.to_numeric(completed["home_score"], errors="coerce")
    completed = completed.dropna(subset=["away_score", "home_score"])

    completed["game_date"] = pd.to_datetime(completed["game_date"])
    completed["home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    completed["total_runs"] = completed["home_score"] + completed["away_score"]
    completed["run_diff"] = completed["home_score"] - completed["away_score"]
    completed["day_of_week"] = completed["game_date"].dt.day_name()

    return completed


def save_tables(df: pd.DataFrame) -> None:
    """Compute and save summary tables to results/tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Score summary (mean, std, min, max)
    score_summary = df[["away_score", "home_score", "total_runs"]].agg(
        ["mean", "std", "min", "max", "count"]
    ).round(2)
    score_summary.to_csv(TABLES_DIR / "score_summary.csv")
    print(f"  Saved {TABLES_DIR / 'score_summary.csv'}")

    # 2. Home win rate (overall and by day of week)
    home_win_overall = pd.DataFrame(
        {"home_win_rate": [df["home_win"].mean()], "n_games": [len(df)]}
    )
    home_win_overall.to_csv(TABLES_DIR / "home_win_rate_overall.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'home_win_rate_overall.csv'}")

    by_dow = (
        df.groupby("day_of_week", observed=True)
        .agg(home_win_rate=("home_win", "mean"), n_games=("game_id", "count"))
        .reset_index()
    )
    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=day_order, ordered=True)
    by_dow = by_dow.sort_values("day_of_week")
    by_dow.to_csv(TABLES_DIR / "home_win_rate_by_day.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'home_win_rate_by_day.csv'}")

    # 3. Games per venue (top 20)
    venue_counts = df["venue_name"].value_counts().reset_index()
    venue_counts.columns = ["venue_name", "n_games"]
    venue_counts.head(20).to_csv(TABLES_DIR / "games_by_venue_top20.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'games_by_venue_top20.csv'}")

    # 4. Games per date (for time series)
    games_per_date = df.groupby("game_date").size().reset_index(name="n_games")
    games_per_date.to_csv(TABLES_DIR / "games_per_date.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'games_per_date.csv'}")

    # 5. Pitcher decision: probable starter got the W vs bullpen
    df_p = df.copy()
    df_p["home_probable"] = df_p["home_probable_pitcher"].fillna("").astype(str).str.strip()
    df_p["away_probable"] = df_p["away_probable_pitcher"].fillna("").astype(str).str.strip()
    df_p["winning_pitcher"] = df_p["winning_pitcher"].fillna("").astype(str).str.strip()
    home_probable_got_w = (df_p["winning_pitcher"] == df_p["home_probable"]) & (df_p["home_probable"] != "")
    away_probable_got_w = (df_p["winning_pitcher"] == df_p["away_probable"]) & (df_p["away_probable"] != "")
    decision = []
    if home_probable_got_w.any():
        decision.append({"decision": "home_probable_got_win", "n_games": home_probable_got_w.sum()})
    if away_probable_got_w.any():
        decision.append({"decision": "away_probable_got_win", "n_games": away_probable_got_w.sum()})
    other = (~home_probable_got_w & ~away_probable_got_w).sum()
    if other > 0:
        decision.append({"decision": "other_bullpen_or_unknown", "n_games": other})
    decision_df = pd.DataFrame(decision)
    decision_df["pct"] = (decision_df["n_games"] / len(df)).round(2)
    decision_df.to_csv(TABLES_DIR / "pitcher_decision_probable_vs_actual.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'pitcher_decision_probable_vs_actual.csv'}")

    # 6. Top winning and losing pitchers (by count in sample)
    win_counts = df["winning_pitcher"].value_counts().reset_index()
    win_counts.columns = ["pitcher", "wins"]
    win_counts.head(20).to_csv(TABLES_DIR / "top_winning_pitchers.csv", index=False)
    loss_counts = df["losing_pitcher"].value_counts().reset_index()
    loss_counts.columns = ["pitcher", "losses"]
    loss_counts.head(20).to_csv(TABLES_DIR / "top_losing_pitchers.csv", index=False)
    print(f"  Saved {TABLES_DIR / 'top_winning_pitchers.csv'}, top_losing_pitchers.csv")


def save_figures(df: pd.DataFrame) -> None:
    """Generate and save EDA figures to results/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Distribution of home and away scores
    fig, ax = plt.subplots(figsize=(8, 4))
    df[["away_score", "home_score"]].plot(kind="hist", bins=range(0, 25), alpha=0.6, ax=ax, legend=True)
    ax.set_xlabel("Runs")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of away vs home scores")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'score_distribution.png'}")

    # 2. Total runs per game
    fig, ax = plt.subplots(figsize=(6, 4))
    df["total_runs"].hist(bins=range(0, 30), ax=ax, edgecolor="white")
    ax.set_xlabel("Total runs (away + home)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of total runs per game")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "total_runs_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'total_runs_distribution.png'}")

    # 3. Home win rate by day of week
    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    by_dow = df.groupby("day_of_week", observed=True)["home_win"].mean().reindex(day_order)
    fig, ax = plt.subplots(figsize=(7, 4))
    by_dow.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.axhline(y=df["home_win"].mean(), color="gray", linestyle="--", label="Overall")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Home win rate")
    ax.set_title("Home win rate by day of week")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "home_win_rate_by_day.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'home_win_rate_by_day.png'}")

    # 4. Run differential (home - away) distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["run_diff"].hist(bins=range(-20, 22), ax=ax, edgecolor="white")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Run differential (home − away)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of run differential")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "run_differential_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'run_differential_distribution.png'}")

    # 5. Pitcher: who got the win (probable starter vs other)
    df_p = df.copy()
    df_p["home_probable"] = df_p["home_probable_pitcher"].fillna("").astype(str).str.strip()
    df_p["away_probable"] = df_p["away_probable_pitcher"].fillna("").astype(str).str.strip()
    df_p["winning_pitcher"] = df_p["winning_pitcher"].fillna("").astype(str).str.strip()
    home_probable_got_w = (df_p["winning_pitcher"] == df_p["home_probable"]) & (df_p["home_probable"] != "")
    away_probable_got_w = (df_p["winning_pitcher"] == df_p["away_probable"]) & (df_p["away_probable"] != "")
    other = (~home_probable_got_w & ~away_probable_got_w).sum()
    labels = ["Home probable got W", "Away probable got W", "Other (bullpen/unknown)"]
    counts = [home_probable_got_w.sum(), away_probable_got_w.sum(), other]
    colors = ["#2ecc71", "#3498db", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Winning pitcher: probable starter vs other")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pitcher_decision_probable_vs_actual.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'pitcher_decision_probable_vs_actual.png'}")

    # 6. Top 12 winning and top 12 losing pitchers (by count in sample)
    top_n = 12
    win_counts = df["winning_pitcher"].value_counts().head(top_n)
    loss_counts = df["losing_pitcher"].value_counts().head(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    win_counts.plot(kind="barh", ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Wins (in sample)")
    ax1.set_ylabel("")
    ax1.set_title(f"Top {top_n} winning pitchers")
    ax1.invert_yaxis()
    loss_counts.plot(kind="barh", ax=ax2, color="coral", edgecolor="white")
    ax2.set_xlabel("Losses (in sample)")
    ax2.set_ylabel("")
    ax2.set_title(f"Top {top_n} losing pitchers")
    ax2.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pitcher_wins_and_losses_top12.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'pitcher_wins_and_losses_top12.png'}")


def main():
    print("Fetching schedule from MLB Stats API...")
    df_raw = fetch_schedule()
    print(f"  Retrieved {len(df_raw)} games")

    df = prepare_data(df_raw)
    if df.empty:
        print("No completed games in range. Adjust EDA_START / EDA_END or check API.")
        return
    print(f"  {len(df)} completed games after filtering\n")

    print("Saving tables to results/tables/")
    save_tables(df)

    print("\nSaving figures to results/figures/")
    save_figures(df)

    print("\nEDA done.")


if __name__ == "__main__":
    main()
