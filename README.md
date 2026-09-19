# NBA Game Outcome Predictor

End-to-end machine-learning pipeline for predicting NBA game winners from historical game logs.

## Reported experiment

The original experiment achieved **60.93% out-of-sample accuracy across 4,182 game records**, compared with a roughly 50/50 naive baseline. That result came from an earlier local training run; it is documented here for portfolio context and is not automatically reproduced by every run.

## Method

- Download historical NBA game logs through `nba_api`.
- Build one pregame row per matchup.
- Generate every feature chronologically to prevent future-data leakage.
- Engineer expanding win percentage, recent form, rest days, back-to-backs, win streak, game number, home court, and opponent-difference features.
- Split train/test data by date, not randomly.
- Train and evaluate an XGBoost binary classifier.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --start-season 2014-15 --test-season 2023-24
```

The NBA Stats service may rate-limit automated requests. If data collection is interrupted, retry later or supply a previously exported CSV with `--input-csv`.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Data collection, chronological features, training, and evaluation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes datasets, environments, and model artifacts |

## Reproducibility notes

Results vary with season range, NBA Stats revisions, package versions, and hyperparameters. The saved model and original dataset are intentionally not published; the code rebuilds the pipeline from source data.
