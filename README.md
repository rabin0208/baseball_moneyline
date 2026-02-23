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
Adds rolling features (wins, runs, runs allowed, run differential over last 10 games), H2H win rate, rest days, pitcher rolling win rate, and game_type/doubleheader dummies. Outputs `data/schedule_8_seasons_featured.csv`.

### 4. Fit models
```bash
python scripts/fit_logistic_model.py
python scripts/fit_random_forest.py
python scripts/fit_gradient_boosting.py
```
Each script loads the featured CSV, splits by season (train on 2018–2024, test on 2025), fits the model, prints accuracy and ROC-AUC, and saves the model to `results/models/`.

### 5. Hyperparameter tuning (optional)
```bash
python scripts/optimize_random_forest.py
python scripts/optimize_gradient_boosting.py
```
Uses `RandomizedSearchCV` with `TimeSeriesSplit` to tune hyperparameters. Saves the best model and search results to `results/models/` and `results/tables/`.

## Environment

- **Python** 3.11
- **Data:** pandas, numpy, pybaseball, MLB-StatsAPI
- **Modeling:** scikit-learn
- **Exploration:** jupyter, matplotlib, seaborn
