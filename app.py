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
from fetch_odds import ensure_odds_api_key  # noqa: E402
from odds_utils import (  # noqa: E402
    SLATE_CAP_FRAC,
    american_to_decimal,
    cap_slate_stakes,
    format_decimal_odds,
    is_valid_american_odds,
    normalize_stakes_mean_one,
    pick_bets,
    roi_on_wagered,
)
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
      margin: 0.15rem 0.5rem 0.8rem 0;
      padding: 0.28rem 0.6rem;
      color: #03140c;
      background: var(--bm-green);
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .bm-stake {
      display: inline-block;
      margin: 0.15rem 0 0.8rem;
      padding: 0.28rem 0.6rem;
      color: #1a1206;
      background: var(--bm-orange);
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .bm-stake-skip {
      background: #3a4456;
      color: #eef4ff;
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


@st.cache_data(ttl=12 * 3600, show_spinner=False)
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
        raise RuntimeError(
            "No sportsbook odds are available for this date yet. "
            "SBR often returns HTTP 503 to scrapers — retry later, or set ODDS_API_KEY "
            "so recommend_bets/fetch_live_odds can fall back to The Odds API."
        )

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


def prepare_roi_bets(games: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Threshold bets with flat 1u profit and Kelly stakes rescaled to mean 1u."""
    scored = pick_bets(games, edge_threshold=edge_threshold)
    bets = scored.dropna(subset=["bet_side", "bet_odds", "bet_profit"]).copy()
    bets = bets.loc[bets["bet_odds"].map(is_valid_american_odds)]
    if bets.empty:
        return bets
    bets["kelly_stake_u"] = normalize_stakes_mean_one(bets["kelly_stake"])
    bets["kelly_profit_u"] = bets["kelly_stake_u"] * bets["bet_profit"]
    return bets


def monthly_roi_table(bets: pd.DataFrame) -> pd.DataFrame:
    """Flat vs Kelly-weighted ROI by calendar month. Kelly mean stake is 1u."""
    empty_cols = [
        "Month",
        "Bets",
        "Wins",
        "Hit rate",
        "Flat profit (u)",
        "Flat ROI",
        "Kelly wagered (u)",
        "Kelly profit (u)",
        "Kelly ROI",
    ]
    if bets.empty:
        return pd.DataFrame(columns=empty_cols)

    work = bets.copy()
    work["month"] = pd.to_datetime(work["game_date"]).dt.to_period("M")
    monthly = (
        work.groupby("month", as_index=False)
        .agg(
            Bets=("bet_profit", "count"),
            Wins=("bet_won", "sum"),
            flat_profit=("bet_profit", "sum"),
            kelly_wagered=("kelly_stake_u", "sum"),
            kelly_profit=("kelly_profit_u", "sum"),
        )
        .sort_values("month")
    )
    monthly["Wins"] = monthly["Wins"].astype(int)
    monthly["Hit rate"] = monthly["Wins"] / monthly["Bets"]
    monthly["Flat ROI"] = monthly["flat_profit"] / monthly["Bets"]
    monthly["Kelly ROI"] = monthly["kelly_profit"] / monthly["kelly_wagered"].mask(
        monthly["kelly_wagered"] <= 0
    )
    monthly["Month"] = monthly["month"].astype(str)
    monthly["Flat profit (u)"] = monthly["flat_profit"]
    monthly["Kelly wagered (u)"] = monthly["kelly_wagered"]
    monthly["Kelly profit (u)"] = monthly["kelly_profit"]
    return monthly[empty_cols]


def percent(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def moneyline(value: object) -> str:
    return "—" if pd.isna(value) else format_decimal_odds(float(value))


def count(value: object) -> int:
    return 0 if pd.isna(value) else int(value)


def dollars(amount: object) -> str:
    return "—" if pd.isna(amount) else f"${float(amount):,.0f}"


def slate_profit_if_all_win(recs: pd.DataFrame, bankroll: float) -> float:
    """Net profit if every recommended bet with a positive stake wins."""
    profit = 0.0
    for _, rec in recs.iterrows():
        frac = rec.get("stake_frac")
        odds = rec.get("bet_odds")
        if pd.isna(frac) or float(frac) <= 0 or pd.isna(odds):
            continue
        decimal = american_to_decimal(odds)
        if pd.isna(decimal) or decimal <= 1.0:
            continue
        stake = float(frac) * bankroll if bankroll > 0 else float(frac)
        profit += stake * (decimal - 1.0)
    return profit


def stake_badge_html(frac: object, bankroll: float) -> str:
    if pd.isna(frac) or float(frac) <= 0:
        return (
            "<span class='bm-stake bm-stake-skip'>Stake: skip — not +EV after "
            "shrink toward market</span>"
        )
    f = float(frac)
    label = f"Stake: {f:.1%} of bankroll"
    if bankroll > 0:
        label += f" · {dollars(f * bankroll)}"
    return f"<span class='bm-stake'>{label}</span>"


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
            st.progress(float(market_prob), text=f"Consensus {moneyline(odds)}")


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
    bankroll: float,
) -> None:
    spinner_msg = (
        f"Scoring the {target_date:%B %-d} slate…"
        if is_past
        else f"Scoring the {target_date:%B %-d} slate and fetching current odds…"
    )
    if not is_past and ensure_odds_api_key():
        st.warning(
            "Using **The Odds API** for live moneylines. Results are cached for **12 hours** — "
            "frequent refreshes or date changes burn free-tier credits (~500/month)."
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
        slate_stake = 0.0
    else:
        games = pick_bets(games, edge_threshold=edge_threshold).sort_values(
            ["bet_edge", "game_id"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        rec_mask = games["bet_side"].notna()
        if rec_mask.any():
            games.loc[rec_mask, "stake_frac"] = cap_slate_stakes(
                games.loc[rec_mask, "stake_frac"]
            )
        recommendations = games.loc[rec_mask]
        matched = games.dropna(subset=["home_odds", "away_odds"])
        slate_stake = float(games.loc[rec_mask, "stake_frac"].fillna(0).sum())

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Games", len(games))
    if is_past:
        kpi_cols[1].metric("Market matches", "—")
        kpi_cols[2].metric("Recommended bets", "—")
        kpi_cols[3].metric("Slate stake", "—")
        kpi_cols[4].metric("If all win", "—")
    else:
        kpi_cols[1].metric("Market matches", f"{len(matched)}/{len(games)}")
        kpi_cols[2].metric("Recommended bets", len(recommendations))
        if bankroll > 0:
            kpi_cols[3].metric(
                "Slate stake",
                dollars(slate_stake * bankroll),
                delta=percent(slate_stake),
                delta_color="off",
            )
            all_win = slate_profit_if_all_win(recommendations, bankroll)
            kpi_cols[4].metric(
                "If all win",
                dollars(all_win),
                delta=percent(all_win / bankroll) if bankroll else None,
                delta_color="off",
                help="Net profit if every recommended bet with a stake wins.",
            )
        else:
            kpi_cols[3].metric("Slate stake", percent(slate_stake))
            all_win = slate_profit_if_all_win(recommendations, 0.0)
            kpi_cols[4].metric(
                "If all win",
                percent(all_win),
                help="Net profit as a fraction of bankroll if every recommended bet with a stake wins.",
            )

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
            f"recommendation threshold {edge_threshold:.0%} · "
            f"¼ Kelly on a 50/50 model+market blend, max 5%/bet, "
            f"{SLATE_CAP_FRAC:.0%} slate cap"
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
                    f"{moneyline(game['bet_odds'])} · {percent(game['bet_edge'])}</span>"
                    f"{stake_badge_html(game.get('stake_frac'), bankroll)}",
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
    st.subheader("Flat vs Kelly-weighted ROI")
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
        f"Same bets when model edge ≥ {edge_threshold:.0%} vs the vig-free closing line. "
        "Flat stakes 1u each. Kelly stakes are proportional to full Kelly at the **posted** "
        "moneyline, then rescaled so the mean stake is also 1u (same total amount wagered, "
        "no compounding). Fractional Kelly would not change this ROI."
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

    season_bets = prepare_roi_bets(games, edge_threshold)
    monthly = monthly_roi_table(season_bets)

    kpi = st.columns(4)
    if season_bets.empty:
        kpi[0].metric("Season bets", 0)
        kpi[1].metric("Hit rate", "—")
        kpi[2].metric("Flat ROI", "—")
        kpi[3].metric("Kelly ROI", "—")
        st.info(f"No bets clear the {edge_threshold:.0%} edge threshold in {season}.")
        return

    n_bets = len(season_bets)
    hit = float(season_bets["bet_won"].mean())
    flat_profit = float(season_bets["bet_profit"].sum())
    flat_roi = flat_profit / n_bets
    kelly_profit = float(season_bets["kelly_profit_u"].sum())
    kelly_roi = roi_on_wagered(season_bets["kelly_profit_u"], season_bets["kelly_stake_u"])
    n_kelly = int((season_bets["kelly_stake"] > 0).sum())
    delta = kelly_roi - flat_roi if kelly_roi == kelly_roi else float("nan")

    kpi[0].metric("Season bets", n_bets)
    kpi[1].metric("Hit rate", f"{hit:.1%}")
    kpi[2].metric("Flat ROI", f"{flat_roi:+.1%}")
    if delta != delta:
        kpi[3].metric("Kelly ROI", "—")
    else:
        kpi[3].metric("Kelly ROI", f"{kelly_roi:+.1%}", delta=f"{delta:+.1%} vs flat")

    if delta == delta:
        if delta > 0.005:
            st.success(
                "Kelly-weighted ROI is higher: larger model edges earned more per dollar than "
                "smaller ones. Sizing up with edge would have beaten flat 1u on this sample."
            )
        elif delta < -0.005:
            st.warning(
                "Kelly-weighted ROI is lower: the model's biggest edges underperformed. That "
                "is what overconfidence looks like — do not size up until calibration improves."
            )
        else:
            st.info(
                "Kelly and flat ROI are nearly the same. Edge size did not add much information "
                "beyond the yes/no threshold."
            )

    st.caption(
        f"Flat profit {flat_profit:+.1f}u on {n_bets}u wagered. "
        f"Kelly profit {kelly_profit:+.1f}u on "
        f"{float(season_bets['kelly_stake_u'].sum()):.1f}u wagered "
        f"({n_kelly} bets with Kelly f* > 0)."
    )

    display = monthly.copy()
    display["Hit rate"] = display["Hit rate"].map(lambda x: f"{x:.1%}")
    display["Flat profit (u)"] = display["Flat profit (u)"].map(lambda x: f"{x:+.2f}")
    display["Flat ROI"] = display["Flat ROI"].map(lambda x: f"{x:+.1%}")
    display["Kelly wagered (u)"] = display["Kelly wagered (u)"].map(lambda x: f"{x:.2f}")
    display["Kelly profit (u)"] = display["Kelly profit (u)"].map(lambda x: f"{x:+.2f}")
    display["Kelly ROI"] = display["Kelly ROI"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.1%}"
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_data = monthly.melt(
        id_vars=["Month", "Bets"],
        value_vars=["Flat ROI", "Kelly ROI"],
        var_name="Strategy",
        value_name="ROI",
    ).dropna(subset=["ROI"])
    chart_data["Strategy"] = chart_data["Strategy"].map(
        {"Flat ROI": "Flat 1u", "Kelly ROI": "Kelly-weighted"}
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Month:N",
                title="Month",
                sort=list(monthly["Month"]),
                axis=alt.Axis(labelAngle=-35),
            ),
            xOffset="Strategy:N",
            y=alt.Y(
                "ROI:Q",
                title="Return on investment (ROI)",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Strategy:N",
                title="Strategy",
                scale=alt.Scale(
                    domain=["Flat 1u", "Kelly-weighted"],
                    range=["#00d7f7", "#35d07f"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip("Strategy:N", title="Strategy"),
                alt.Tooltip("ROI:Q", title="ROI", format=".1%"),
                alt.Tooltip("Bets:Q", title="Bets"),
            ],
        )
        .properties(
            title=f"Monthly ROI: flat 1u vs Kelly-weighted (edge ≥ {edge_threshold:.0%})",
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
        f"Matched games: {len(games):,} · lines {date_min} → {date_max}. "
        "SBR closing lines when available; recent dates use the live consensus "
        "saved with that day's recommendations (SBR often returns HTTP 503). "
        "Kelly uses posted (vigged) odds, not the vig-free probability. "
        "This is a sizing diagnostic, not a compounding bankroll path."
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
        value=10,
        step=1,
        format="%d%%",
        disabled=is_past or not recommendations_only,
        help="Enabled when Recommendations only is on. "
        "A recommendation appears when model probability exceeds fair market probability by this amount.",
    )
    edge_threshold = edge_percent / 100.0
    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=0,
        value=1000,
        step=500,
        disabled=is_past,
        help=(
            "Converts each recommended stake into dollars. Sizing is ¼ Kelly on a "
            "50/50 blend of model and market probability, capped at 5% per bet and "
            f"{SLATE_CAP_FRAC:.0%} for the whole slate. Set to 0 to show percents only."
        ),
    )
    if st.button(
        "Refresh data",
        type="primary",
        use_container_width=True,
        help=(
            "Clears the cache and reloads. For live dates this calls The Odds API "
            "again and spends credits."
            if ensure_odds_api_key() and not is_past
            else "Clears the cache and reloads model / odds data."
        ),
    ):
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
    if ensure_odds_api_key() and not is_past:
        st.caption(
            "⚠ Live odds via The Odds API · cached 12 h · "
            "Refresh / date changes spend credits (~500/month free)."
        )
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
        bankroll=float(bankroll),
    )

with roi_tab:
    render_roi_tab(season=today.year, edge_threshold=edge_threshold)
