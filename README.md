# Knowledge-Guided AI for Sales Forecasting

Compact, reproducible code for the short communication **Knowledge-Guided AI
for Sales Data Analysis and Marketing Performance: A Hybrid Forecasting Model**.

The method combines an explicit store-weekday sales norm and bounded promotion
uplift with histogram gradient boosting. Business rules set closed-store
forecasts to zero and prevent negative sales. Evaluation uses three rolling
42-day tests, matched baselines, component ablations, marketing-segment
diagnostics, and a paired day-level statistical test.

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

The script creates exactly two manuscript tables and two publication figures
inside `results/`. The dataset is not redistributed in this repository.

## Method in one line

`forecast = AI(data features + marketing knowledge prior) + business rules`

## Reproducibility

- Fixed random seed: 42
- Leakage-resistant evaluation: three rolling 42-day windows
- No customer-count feature (unavailable at forecast time)
- Source data: Rossmann Store Sales public Kaggle benchmark
- License: MIT for the code; source-data terms remain with the data provider
