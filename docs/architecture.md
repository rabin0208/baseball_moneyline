# Baseball Moneyline architecture

## Scope

This document describes the current repository architecture at Project Systems onboarding time. It is a map of existing behavior, not a redesign proposal.

The authoritative implementation remains the code and validated outputs in this repository. `README.md` owns the current user-facing command sequence.

## End-to-end flow

The current pipeline is approximately:

`MLB schedule/history → cleaning/EDA → lagged pre-game feature engineering → model training/evaluation → current-game prediction → sportsbook moneyline normalization/comparison → recommendation/staking logic → Streamlit dashboard`

## Data acquisition

`scripts/data_load.py` uses the MLB Stats API to build historical/current schedule data.

Market-line tooling uses SportsBookReview scraping and can fall back to The Odds API when configured. Odds-provider behavior is an external dependency and should fail visibly rather than silently changing interpretation.

## Feature engineering

`scripts/split_n_preprocess.py` and shared helpers build pre-game lagged features. Current documented feature families include:

- recent wins;
- runs scored/allowed;
- run differential;
- lagged head-to-head results;
- rest days;
- calendar season and ISO week;
- lagged pitcher outcomes.

The project depends on preserving leakage-safe chronology: a game's prediction features must not use information that would only become known after that game begins/completes.

## Models

The repository includes training paths for:

- logistic regression;
- random forest;
- gradient boosting.

The current documented live prediction path relies on the fitted logistic-regression model plus scaler.

Model selection should be justified by current validation/market evidence rather than by in-sample fit alone.

## Evaluation

Current evaluation surfaces include:

- season-based chronological holdout;
- biweekly walk-forward evaluation;
- fixed-train weekly/period evaluation;
- accuracy and ROC-AUC;
- market comparison metrics including log loss, Brier score, and flat-bet ROI.

Changes to split chronology, holdout definitions, market-probability conversion, ROI calculations, or recommendation thresholds are methodology changes and need explicit issue-level acceptance criteria.

## Prediction and recommendation layer

`scripts/predict_2026.py` supports completed-season scoring, full-season forecasting, and next-day/today prediction modes using rolling state intended to mirror training-time feature construction.

`scripts/recommend_bets.py` combines model probabilities with current moneylines, evaluates model edge against vig-free/fair market probability, and applies capped fractional-Kelly staking logic.

This repository may analyze betting opportunities, but Project Systems onboarding does not authorize sportsbook-account access or automated wager placement.

## Application surface

`app.py` provides a Streamlit dashboard for daily slate probabilities, sportsbook comparison, edge filtering, bankroll-aware recommended stake display, and cache refresh behavior.

Production/deployment changes are human-gated. The onboarding contract does not alter the existing public Streamlit deployment.

## Key architecture invariants

Preserve these unless an approved issue explicitly changes them:

1. pre-game features remain leakage-safe;
2. training/test/evaluation chronology is explicit and reproducible;
3. prediction-time rolling state is compatible with training feature semantics;
4. model and market probabilities are distinguishable and auditable;
5. recommendation/staking calculations are deterministic from documented inputs;
6. provider failures do not silently become fabricated odds/data;
7. evaluation outputs identify the exact data window/model/configuration used;
8. dashboard behavior should reuse shared prediction/recommendation logic rather than fork hidden business logic where practical.

## Deferred architecture decisions

Project Systems onboarding does not decide:

- a new model family;
- a new sportsbook/odds-provider contract;
- automated betting execution;
- a new hosting/deployment platform;
- Project HQ lifecycle automation;
- a dedicated Grok or other execution-worker route;
- Baseball-to-Hockey asset reuse rights.

Those require separate decision/specification work in the appropriate governance/product surface.
