"""Streamlit dashboard for comparing model and sportsbook win probabilities."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from odds_utils import format_american_odds, pick_bets  # noqa: E402
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
    """Run the existing recommendation pipeline and retain every game."""
    predictions = predict_slate(target, season)
    if predictions.empty:
        return predictions

    odds = fetch_live_odds(target)
    if odds.empty:
        raise RuntimeError("No sportsbook odds are available for this date yet.")

    comparison = join_predictions_odds(predictions, odds)
    if comparison.empty:
        raise RuntimeError("Games were found, but none could be joined to sportsbook odds.")

    comparison["p_away_win"] = 1.0 - comparison["p_home_win"]
    return comparison


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
) -> None:
    st.subheader(team)
    pitcher_name = "TBD" if pd.isna(pitcher) or not str(pitcher).strip() else str(pitcher)
    st.caption(f"Probable pitcher: {pitcher_name}")
    st.metric(
        "Model win probability",
        percent(model_prob),
        delta=f"{float(edge):+.1%} vs market" if pd.notna(edge) else None,
        delta_color="normal",
    )
    if pd.notna(model_prob):
        st.progress(float(model_prob), text="Logistic regression")
    st.metric("Sportsbook fair probability", percent(market_prob))
    if pd.notna(market_prob):
        st.progress(float(market_prob), text=f"Consensus moneyline {moneyline(odds)}")


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
if "target_date" not in st.session_state:
    st.session_state["target_date"] = today


def choose_tomorrow() -> None:
    st.session_state["target_date"] = today + timedelta(days=1)


with st.sidebar:
    st.header("Slate controls")
    target_date = st.date_input(
        "Game date",
        key="target_date",
        min_value=today,
        max_value=date(today.year, 11, 30),
    )
    edge_threshold = st.slider(
        "Minimum edge",
        min_value=0.0,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.0f%%",
        help="A recommendation appears when model probability exceeds fair market probability by this amount.",
    )
    recommendations_only = st.toggle("Recommendations only", value=False)
    if st.button("Refresh model and odds", type="primary", use_container_width=True):
        load_comparison.clear()
        st.rerun()
    if target_date == today:
        st.button(
            "View tomorrow",
            use_container_width=True,
            on_click=choose_tomorrow,
        )
    st.divider()
    st.caption(
        "Sportsbook probabilities use the median moneyline across available books, "
        "then remove the vig so both teams sum to 100%."
    )

with st.spinner(f"Scoring the {target_date:%B %-d} slate and fetching current odds…"):
    try:
        games = load_comparison(target_date, target_date.year)
    except Exception as exc:  # The UI should turn upstream failures into actionable feedback.
        st.error(str(exc))
        st.info(
            "Confirm the model and scaler exist, the history CSV is current, and the "
            "machine has network access. Odds may not be posted until closer to game time."
        )
        st.stop()

if games.empty:
    st.info("No not-yet-final MLB games were found for this date.")
    st.stop()

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
kpi_cols[1].metric("Market matches", f"{len(matched)}/{len(games)}")
kpi_cols[2].metric("Recommended bets", len(recommendations))
kpi_cols[3].metric("Strongest model edge", percent(strongest))

visible_games = recommendations if recommendations_only else games
if visible_games.empty:
    st.info(f"No bets clear the {edge_threshold:.0%} edge threshold.")
    st.stop()

st.subheader(f"{target_date:%A, %B %-d} matchups")
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
        status.markdown(
            f"<div class='bm-meta'>{game.get('status', '')}<br>"
            f"{count(game.get('n_books', 0))} books</div>",
            unsafe_allow_html=True,
        )

        if pd.notna(game.get("bet_side")):
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
            )
        with home_col:
            render_team(
                team=game["home_name"],
                pitcher=game.get("home_probable_pitcher", ""),
                model_prob=game["p_home_win"],
                market_prob=game.get("p_home_mkt"),
                edge=game.get("edge_home"),
                odds=game.get("home_odds"),
            )
