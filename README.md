# Baseball Moneyline

ML project to predict which team wins a baseball game (home vs away) using the MLB Stats API and scikit-learn.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate baseball_moneyline
```

Run all commands from the project root.

## Commands (pipeline order)

### 1. Download schedule data
```bash
python scripts/data_load.py
```
Fetches 8 seasons (2018–2025) of schedule data from the MLB Stats API. Outputs `data/schedule_8_seasons.csv`.

### 2. Exploratory data analysis
```bash
python scripts/eda.py
```
Filters to completed ("Final") games, drops rows with missing values, adds derived columns (e.g. `home_win`, `day_of_week`). Saves summary tables to `results/tables/`, figures to `results/figures/`, and the cleaned dataset to `data/schedule_8_seasons_final.csv`.

### 3. Feature engineering
```bash
python scripts/split_n_preprocess.py
```
Adds rolling features (wins, runs, runs allowed, run differential over last 10 games), H2H win rate, rest days, and pitcher rolling win rate (centered). Outputs `data/schedule_8_seasons_featured.csv`.

### 4. Fit models
```bash
python scripts/fit_logistic_model.py
python scripts/fit_random_forest.py
python scripts/fit_gradient_boosting.py
```
Each script loads the featured CSV, splits by season (**train on 2018–2024, test on 2025**), fits the model, prints accuracy and ROC-AUC, and saves the model to `results/models/` (e.g. `logistic_regression.pkl` and `scaler.pkl` for the logistic pipeline).

### 5. Predictions (deployment on a new season)

Requires `eda.py` output (`schedule_8_seasons_final.csv`) and a fitted `fit_logistic_model.py` (model + scaler). Predictions use `scripts/rolling_state.py` to match training-time rolling features: history is seeded from the final CSV, then the API schedule is walked in date order.

**Completed games only (default)** — scores every **Final** game from **season start through today** (fetch window ends today; no future games). Good for backtesting or calibration on the season so far.

```bash
python scripts/predict_2026.py
```

Writes `results/tables/predictions_<season>.csv` (default `--season` is `2026`).

**Full-season forecast** — includes scheduled games not yet played through the end of the season:

```bash
python scripts/predict_2026.py --forecast
```

**Next calendar day** — advances state through all **Final** games through **today**, then predicts only games on the **target day** that are **not** Final yet (tomorrow’s slate by default):

```bash
python scripts/predict_2026.py --next-day
```

Optional `--predict-date YYYY-MM-DD` (today or a future date). Output: `results/tables/predictions_next_day_<date>.csv`. Do not combine `--next-day` with `--forecast`.

Use `--start`, `--end`, `-o` / `--output`, and `--season` as needed for other seasons or paths.

### 6. Hyperparameter tuning (optional)
```bash
python scripts/optimize_random_forest.py
python scripts/optimize_gradient_boosting.py
```
Uses `RandomizedSearchCV` with `TimeSeriesSplit` to tune hyperparameters. Saves the best model and search results to `results/models/` and `results/tables/`.

## Train / test / deploy

| Phase | Seasons (with current pipeline) |
|--------|-----------------------------------|
| Training | 2018–2024 |
| Test (holdout) | 2025 |
| Live predictions | e.g. 2026 via `predict_2026.py` |

`data_load.py` pulls the **last 8 full calendar years** of completed seasons (e.g. in 2026 that is **2018–2025**), which feeds EDA and feature engineering.

## Environment

- **Python** 3.11
- **Data:** pandas, numpy, pybaseball, MLB-StatsAPI
- **Modeling:** scikit-learn
- **Exploration:** jupyter, matplotlib, seaborn
