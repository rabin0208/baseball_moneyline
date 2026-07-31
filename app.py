"""Streamlit dashboard for comparing model and sportsbook win probabilities."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ODDS_CSV = PROJECT_ROOT / "data" / "odds_moneyline.csv"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_vs_market import join_games_odds, load_games_with_model_probs  # noqa: E402
from odds_utils import format_american_odds, is_valid_american_odds, pick_bets  # noqa: E402
from recommend_bets import (  # noqa: E402
    fetch_live_odds,
    join_predictions_odds,
    predict_slate,
)


st.set_page_config(
    page_title="Moneyline Edge",
    page_icon="⚾",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --bm-bg: #07101f;
      --bm-panel: #0b1424;
      --bm-panel-2: #101b2e;
      --bm-line: #203049;
      --bm-text: #eef4ff;
      --bm-muted: #9aa9bd;
      --bm-accent: #00d7f7;
      --bm-green: #35d07f;
      --bm-orange: #ffad5c;
    }
    .stApp {
      background:
        radial-gradient(circle at 74% 5%, rgba(0, 215, 247, 0.12), transparent 26rem),
        linear-gradient(180deg, #111a2c 0, var(--bm-bg) 24rem);
      color: var(--bm-text);
    }
    [data-testid="stSidebar"] {
      background: rgba(11, 20, 36, 0.98);
      border-right: 1px solid var(--bm-line);
    }
    [data-testid="stMetric"] {
      background: rgba(11, 20, 36, 0.78);
      border: 1px solid var(--bm-line);
      border-radius: 0.75rem;
      padding: 0.85rem 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(11, 20, 36, 0.76);
      border-color: var(--bm-line);
      border-radius: 0.8rem;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
    }
    .bm-hero {
      margin: 0 0 1.5rem;
      padding: clamp(1.25rem, 3vw, 2.25rem);
      background:
        linear-gradient(90deg, rgba(0, 215, 247, 0.18), transparent 62%),
        linear-gradient(135deg, #0069ff 0%, #087cff 52%, #00bfe8 100%);
      border: 1px solid rgba(126, 238, 255, 0.38);
      border-radius: 0.8rem;
      box-shadow: 0 26px 60px rgba(0, 0, 0, 0.28);
    }
    .bm-hero h1 {
      margin: 0;
      color: white;
      font-size: clamp(2.2rem, 5vw, 4rem);
      line-height: 1;
    }
    .bm-hero p {
      margin: 0.65rem 0 0;
      color: rgba(255, 255, 255, 0.88);
      font-size: 1.05rem;
    }
    .bm-game-title {
      color: var(--bm-text);
      font-size: 1.2rem;
      font-weight: 750;
    }
    .bm-meta {
      color: var(--bm-muted);
      font-size: 0.83rem;
    }
    .bm-pick {
      display: inline-block;
      margin: 0.15rem 0 0.8rem;
      padding: 0.28rem 0.6rem;
      color: #03140c;
      background: var(--bm-green);
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .bm-no-pick {
      color: var(--bm-muted);
      font-size: 0.8rem;
    }
    h1, h2, h3, label, [data-testid="stMetricLabel"] {
      color: var(--bm-text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_comparison(target: date, season: int) -> pd.DataFrame:
    """Score the slate; attach live sportsbook odds only for today/future dates."""
    predictions = predict_slate(target, season)
    if predictions.empty:
        return predictions

    predictions = predictions.copy()
    predictions["p_away_win"] = 1.0 - predictions["p_home_win"]

    if target < date.today():
        return predictions

    odds = fetch_live_odds(target)
    if odds.empty:
        raise RuntimeError("No sportsbook odds are available for this date yet.")

    comparison = join_predictions_odds(predictions, odds)
    if comparison.empty:
        raise RuntimeError("Games were found, but none could be joined to sportsbook odds.")

    comparison["p_away_win"] = 1.0 - comparison["p_home_win"]
    return comparison


@st.cache_data(ttl=3600, show_spinner=False)
def load_roi_games(season: int) -> pd.DataFrame:
    """Completed games with model probs joined to closing moneylines."""
    if not ODDS_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {ODDS_CSV}. Run: python scripts/fetch_odds.py --season {season}"
        )
    games = load_games_with_model_probs(season)
    odds_raw = pd.read_csv(ODDS_CSV)
    if "odds_type" in odds_raw.columns:
        odds_raw = odds_raw.loc[odds_raw["odds_type"] == "moneyline"].copy()
    merged = join_games_odds(games, odds_raw)
    if merged.empty:
        raise RuntimeError("No completed games could be matched to closing moneylines.")
    return merged


def monthly_roi_table(games: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Flat 1u ROI by calendar month at the given edge threshold."""
    scored = pick_bets(games, edge_threshold=edge_threshold)
    bets = scored.dropna(subset=["bet_side", "bet_odds", "bet_profit"]).copy()
    valid = bets["bet_odds"].map(is_valid_american_odds)
    bets = bets.loc[valid]
    if bets.empty:
        return pd.DataFrame(
            columns=["Month", "Bets", "Wins", "Hit rate", "Profit (u)", "ROI"]
        )

    bets["month"] = pd.to_datetime(bets["game_date"]).dt.to_period("M")
    monthly = (
        bets.groupby("month", as_index=False)
        .agg(
            Bets=("bet_profit", "count"),
            Wins=("bet_won", "sum"),
            profit=("bet_profit", "sum"),
        )
        .sort_values("month")
    )
    monthly["Wins"] = monthly["Wins"].astype(int)
    monthly["Hit rate"] = monthly["Wins"] / monthly["Bets"]
    monthly["ROI"] = monthly["profit"] / monthly["Bets"]
    monthly["Month"] = monthly["month"].astype(str)
    monthly["Profit (u)"] = monthly["profit"]
    return monthly[["Month", "Bets", "Wins", "Hit rate", "Profit (u)", "ROI"]]


def percent(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def moneyline(value: object) -> str:
    return "—" if pd.isna(value) else format_american_odds(float(value))


def count(value: object) -> int:
    return 0 if pd.isna(value) else int(value)


def render_team(
    *,
    team: str,
    pitcher: object,
    model_prob: object,
    market_prob: object,
    edge: object,
    odds: object,
    show_market: bool = True,
) -> None:
    st.subheader(team)
    pitcher_name = "TBD" if pd.isna(pitcher) or not str(pitcher).strip() else str(pitcher)
    st.caption(f"Probable pitcher: {pitcher_name}")
    st.metric(
        "Model win probability",
        percent(model_prob),
        delta=(
            f"{float(edge):+.1%} vs market"
            if show_market and pd.notna(edge)
            else None
        ),
        delta_color="normal",
    )
    if pd.notna(model_prob):
        st.progress(float(model_prob), text="Logistic regression")
    if show_market:
        st.metric("Sportsbook fair probability", percent(market_prob))
        if pd.notna(market_prob):
            st.progress(float(market_prob), text=f"Consensus moneyline {moneyline(odds)}")


def final_score_caption(game: pd.Series) -> str:
    away_score = game.get("away_score")
    home_score = game.get("home_score")
    if pd.isna(away_score) or pd.isna(home_score):
        return str(game.get("status", "") or "")
    return f"Final {int(away_score)}-{int(home_score)}"


def render_daily_slate(
    *,
    target_date: date,
    is_past: bool,
    recommendations_only: bool,
    edge_threshold: float,
) -> None:
    spinner_msg = (
        f"Scoring the {target_date:%B %-d} slate…"
        if is_past
        else f"Scoring the {target_date:%B %-d} slate and fetching current odds…"
    )
    with st.spinner(spinner_msg):
        try:
            games = load_comparison(target_date, target_date.year)
        except Exception as exc:
            st.error(str(exc))
            st.info(
                "Confirm the model and scaler exist, the history CSV is current, and the "
                "machine has network access. Odds may not be posted until closer to game time."
            )
            return

    if games.empty:
        st.info(
            "No MLB games were found for this date."
            if is_past
            else "No not-yet-final MLB games were found for this date."
        )
        return

    if is_past:
        games = games.sort_values("game_id").reset_index(drop=True)
        recommendations = games.iloc[0:0]
        matched = games.iloc[0:0]
        strongest = float("nan")
    else:
        games = pick_bets(games, edge_threshold=edge_threshold).sort_values(
            ["bet_edge", "game_id"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        recommendations = games.dropna(subset=["bet_side"])
        matched = games.dropna(subset=["home_odds", "away_odds"])
        strongest = games[["edge_home", "edge_away"]].max(axis=1).max()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Games", len(games))
    if is_past:
        kpi_cols[1].metric("Market matches", "—")
        kpi_cols[2].metric("Recommended bets", "—")
        kpi_cols[3].metric("Strongest model edge", "—")
    else:
        kpi_cols[1].metric("Market matches", f"{len(matched)}/{len(games)}")
        kpi_cols[2].metric("Recommended bets", len(recommendations))
        kpi_cols[3].metric("Strongest model edge", percent(strongest))

    visible_games = recommendations if (recommendations_only and not is_past) else games
    if visible_games.empty:
        st.info(f"No bets clear the {edge_threshold:.0%} edge threshold.")
        return

    st.subheader(f"{target_date:%A, %B %-d} matchups")
    if is_past:
        st.caption(f"Showing {len(visible_games)} games · model probabilities only")
    else:
        st.caption(
            f"Showing {len(visible_games)} of {len(games)} games · "
            f"recommendation threshold {edge_threshold:.0%}"
        )

    for _, game in visible_games.iterrows():
        with st.container(border=True):
            header, status = st.columns([4, 1])
            header.markdown(
                f"<div class='bm-game-title'>{game['away_name']} @ {game['home_name']}</div>",
                unsafe_allow_html=True,
            )
            if is_past:
                status.markdown(
                    f"<div class='bm-meta'>{final_score_caption(game)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                status.markdown(
                    f"<div class='bm-meta'>{game.get('status', '')}<br>"
                    f"{count(game.get('n_books', 0))} books</div>",
                    unsafe_allow_html=True,
                )

            if is_past:
                st.markdown(
                    "<span class='bm-no-pick'>Historical slate — model probabilities only.</span>",
                    unsafe_allow_html=True,
                )
            elif pd.notna(game.get("bet_side")):
                side = str(game["bet_side"])
                team = game["home_name"] if side == "home" else game["away_name"]
                st.markdown(
                    f"<span class='bm-pick'>Model edge: {team} "
                    f"{moneyline(game['bet_odds'])} · {percent(game['bet_edge'])}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='bm-no-pick'>No side clears the selected edge threshold.</span>",
                    unsafe_allow_html=True,
                )

            away_col, home_col = st.columns(2, gap="large")
            with away_col:
                render_team(
                    team=game["away_name"],
                    pitcher=game.get("away_probable_pitcher", ""),
                    model_prob=game["p_away_win"],
                    market_prob=game.get("p_away_mkt"),
                    edge=game.get("edge_away"),
                    odds=game.get("away_odds"),
                    show_market=not is_past,
                )
            with home_col:
                render_team(
                    team=game["home_name"],
                    pitcher=game.get("home_probable_pitcher", ""),
                    model_prob=game["p_home_win"],
                    market_prob=game.get("p_home_mkt"),
                    edge=game.get("edge_home"),
                    odds=game.get("home_odds"),
                    show_market=not is_past,
                )


def render_roi_tab(*, season: int, edge_threshold: float) -> None:
    st.subheader("Flat-bet ROI by month")
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=int(round(edge_threshold * 100)),
        step=1,
        format="%d%%",
        key="roi_edge_percent",
        help="Model probability minus fair closing-line probability.",
    )
    edge_threshold = edge_percent / 100.0
    st.caption(
        f"1-unit bets when model edge ≥ {edge_threshold:.0%} vs vig-free closing line. "
        "Uses completed games matched to SportsBookReview closing moneylines."
    )

    with st.spinner(f"Loading {season} model vs market results…"):
        try:
            games = load_roi_games(season)
        except Exception as exc:
            st.error(str(exc))
            st.info(
                "Run `python scripts/fetch_odds.py --season "
                f"{season}` and ensure the logistic model is fitted."
            )
            return

    monthly = monthly_roi_table(games, edge_threshold)
    season_bets = pick_bets(games, edge_threshold=edge_threshold).dropna(
        subset=["bet_side", "bet_odds", "bet_profit"]
    )
    season_bets = season_bets.loc[season_bets["bet_odds"].map(is_valid_american_odds)]

    kpi = st.columns(4)
    if season_bets.empty:
        kpi[0].metric("Season bets", 0)
        kpi[1].metric("Hit rate", "—")
        kpi[2].metric("Profit", "—")
        kpi[3].metric("ROI", "—")
        st.info(f"No bets clear the {edge_threshold:.0%} edge threshold in {season}.")
        return

    profit = float(season_bets["bet_profit"].sum())
    n_bets = len(season_bets)
    hit = float(season_bets["bet_won"].mean())
    kpi[0].metric("Season bets", n_bets)
    kpi[1].metric("Hit rate", f"{hit:.1%}")
    kpi[2].metric("Profit", f"{profit:+.1f}u")
    kpi[3].metric("ROI", f"{profit / n_bets:+.1%}")

    display = monthly.copy()
    display["Hit rate"] = display["Hit rate"].map(lambda x: f"{x:.1%}")
    display["Profit (u)"] = display["Profit (u)"].map(lambda x: f"{x:+.2f}")
    display["ROI"] = display["ROI"].map(lambda x: f"{x:+.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_data = monthly.copy()
    chart_data["Result"] = chart_data["ROI"].map(
        lambda r: "Positive ROI" if r >= 0 else "Negative ROI"
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Month:N",
                title="Month",
                sort=list(chart_data["Month"]),
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(
                "ROI:Q",
                title="Return on investment (ROI)",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Result:N",
                title="Result",
                scale=alt.Scale(
                    domain=["Positive ROI", "Negative ROI"],
                    range=["#35d07f", "#ff6b6b"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip("ROI:Q", title="ROI", format=".1%"),
                alt.Tooltip("Bets:Q", title="Bets"),
                alt.Tooltip("Profit (u):Q", title="Profit (u)", format="+.2f"),
                alt.Tooltip("Result:N", title="Result"),
            ],
        )
        .properties(
            title=f"Monthly flat-bet ROI (edge ≥ {edge_threshold:.0%})",
            height=340,
        )
        .configure_title(fontSize=16, anchor="start", color="#eef4ff")
        .configure_axis(labelColor="#9aa9bd", titleColor="#eef4ff")
        .configure_legend(labelColor="#eef4ff", titleColor="#eef4ff")
    )
    st.altair_chart(chart, use_container_width=True)

    date_min = pd.to_datetime(games["game_date"]).min().date()
    date_max = pd.to_datetime(games["game_date"]).max().date()
    st.caption(
        f"Matched games: {len(games):,} · closing lines {date_min} → {date_max}. "
        f"Refresh odds coverage with fetch_odds.py / eval_vs_market.py as needed."
    )


st.markdown(
    """
    <section class="bm-hero">
      <h1>Moneyline Edge</h1>
      <p>Logistic regression win probabilities compared with the sportsbook consensus.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

today = date.today()
season_start = date(today.year, 3, 1)
if "target_date" not in st.session_state:
    st.session_state["target_date"] = today


def choose_tomorrow() -> None:
    st.session_state["target_date"] = today + timedelta(days=1)


with st.sidebar:
    st.header("Controls")
    target_date = st.date_input(
        "Game date",
        key="target_date",
        min_value=season_start,
        max_value=date(today.year, 11, 30),
    )
    is_past = target_date < today
    recommendations_only = st.toggle(
        "Recommendations only",
        value=False,
        disabled=is_past,
        help="Daily slate only. Requires live sportsbook odds (today and future dates).",
    )
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
        disabled=is_past or not recommendations_only,
        help="Enabled when Recommendations only is on. "
        "A recommendation appears when model probability exceeds fair market probability by this amount.",
    )
    edge_threshold = edge_percent / 100.0
    if st.button("Refresh data", type="primary", use_container_width=True):
        load_comparison.clear()
        load_roi_games.clear()
        st.rerun()
    if target_date == today:
        st.button(
            "View tomorrow",
            use_container_width=True,
            on_click=choose_tomorrow,
        )
    st.divider()
    if is_past:
        st.caption(
            "Past dates on the Daily slate show model win probabilities only."
        )
    else:
        st.caption(
            "Sportsbook probabilities use the median moneyline across available books "
            "(in decimal-odds space), then remove the vig so both teams sum to 100%."
        )

slate_tab, roi_tab = st.tabs(["Daily slate", "ROI by month"])

with slate_tab:
    render_daily_slate(
        target_date=target_date,
        is_past=is_past,
        recommendations_only=recommendations_only,
        edge_threshold=edge_threshold,
    )

with roi_tab:
    render_roi_tab(season=today.year, edge_threshold=edge_threshold)
