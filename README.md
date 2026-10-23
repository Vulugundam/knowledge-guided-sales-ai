# Knowledge-Guided AI for Sales Forecasting

Compact, reproducible code for the short communication **Knowledge-Guided AI
for Sales Data Analysis and Marketing Performance: A Hybrid Forecasting Model**.

The method combines an explicit store-weekday sales norm and learned promotion
uplift with a histogram gradient-boosting model through a knowledge feature. Business constraints
set closed-store forecasts to zero and prevent negative sales. Evaluation uses
the final 42 days as a chronological holdout and reports MAE, WAPE, and RMSPE.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

The script downloads the public Rossmann benchmark automatically. To use a
local copy instead:

```bash
python run_experiment.py --data /path/to/train.csv
```

Two manuscript-ready CSV tables are written to `results/`. The dataset is not
redistributed in this repository.

## Method in one line

`forecast = AI(data features + marketing knowledge prior) + business rules`

## Reproducibility

- Fixed random seed: 42
- Leakage-resistant chronological split: last 42 days
- No customer-count feature (unavailable at forecast time)
- Source data: Rossmann Store Sales public Kaggle benchmark
- License: MIT for the code; source-data terms remain with the data provider
