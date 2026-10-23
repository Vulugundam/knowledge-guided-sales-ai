#!/usr/bin/env python3
"""Reproduce the knowledge-guided retail sales forecasting experiment.

The script downloads the public Rossmann training file when --data is omitted,
uses a chronological 42-day holdout, and prints manuscript-ready CSV tables.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_URL = (
    "https://raw.githubusercontent.com/Muosvr/rossmann-store-sales/"
    "master/data/train.csv"
)
SEED = 42


def load_data(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path, low_memory=False)
    print(f"Downloading public benchmark from {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=180) as response:
        return pd.read_csv(io.BytesIO(response.read()), low_memory=False)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Store", "DayOfWeek", "Date", "Sales", "Open", "Promo",
        "StateHoliday", "SchoolHoliday",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.loc[:, sorted(required)].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    out["StateHoliday"] = out["StateHoliday"].astype(str).ne("0").astype("int8")
    out["month"] = out["Date"].dt.month.astype("int8")
    out["day"] = out["Date"].dt.day.astype("int8")
    out["week"] = out["Date"].dt.isocalendar().week.astype("int16")
    out["trend"] = (out["Date"] - out["Date"].min()).dt.days.astype("int16")
    return out.sort_values(["Date", "Store"]).reset_index(drop=True)


def rmspe(y: np.ndarray, pred: np.ndarray) -> float:
    mask = y > 0
    return float(np.sqrt(np.mean(np.square((y[mask] - pred[mask]) / y[mask]))))


def wmape(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).sum() / np.abs(y).sum())


def build_prior(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Explicit knowledge: store/weekday norm adjusted by learned promo uplift."""
    open_train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    base = (
        open_train[open_train["Promo"] == 0]
        .groupby(["Store", "DayOfWeek"])["Sales"]
        .median()
    )
    fallback = open_train.groupby("Store")["Sales"].median()
    global_median = float(open_train["Sales"].median())

    promo_medians = open_train.groupby(["Store", "Promo"])["Sales"].median().unstack()
    uplift = (promo_medians.get(1) / promo_medians.get(0)).replace([np.inf, -np.inf], np.nan)
    uplift = uplift.clip(0.8, 2.0).fillna(1.0)

    key = pd.MultiIndex.from_frame(frame[["Store", "DayOfWeek"]])
    prior = base.reindex(key).to_numpy(dtype=float)
    store_fallback = fallback.reindex(frame["Store"]).to_numpy(dtype=float)
    prior = np.where(np.isnan(prior), store_fallback, prior)
    prior = np.where(np.isnan(prior), global_median, prior)
    promo_uplift = uplift.reindex(frame["Store"]).fillna(1.0).to_numpy(dtype=float)
    prior *= np.where(frame["Promo"].to_numpy() == 1, promo_uplift, 1.0)
    return np.where(frame["Open"].to_numpy() == 0, 0.0, prior)


def features(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "Store", "DayOfWeek", "Open", "Promo", "StateHoliday",
        "SchoolHoliday", "month", "day", "week", "trend", "store_level",
    ]
    return df[cols].to_numpy(dtype=np.float32)


def constrained(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    pred = np.maximum(pred, 0.0)
    return np.where(frame["Open"].to_numpy() == 0, 0.0, pred)


def score(name: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float | str]:
    return {
        "Model": name,
        "MAE": mean_absolute_error(y, pred),
        "WAPE": wmape(y, pred),
        "RMSPE": rmspe(y, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to Rossmann train.csv; downloads if omitted")
    parser.add_argument("--output", default="results", help="Directory for two CSV tables")
    args = parser.parse_args()

    df = prepare(load_data(args.data))
    cutoff = df["Date"].max() - pd.Timedelta(days=41)
    train, test = df[df["Date"] < cutoff], df[df["Date"] >= cutoff]
    train, test = train.copy(), test.copy()
    # A target-derived store level gives the data-only comparator a fair way to
    # represent 1,115 store identities without a high-dimensional one-hot matrix.
    store_level = train.groupby("Store")["Sales"].median()
    global_level = float(train["Sales"].median())
    train["store_level"] = train["Store"].map(store_level).fillna(global_level)
    test["store_level"] = test["Store"].map(store_level).fillna(global_level)
    x_train, x_test = features(train), features(test)
    y_train, y_test = train["Sales"].to_numpy(float), test["Sales"].to_numpy(float)

    prior_train = build_prior(train, train)
    prior_test = build_prior(train, test)

    common = dict(
        learning_rate=0.08,
        max_iter=240,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )
    data_only = HistGradientBoostingRegressor(**common)
    data_only.fit(x_train, np.log1p(y_train))
    pred_data = constrained(np.expm1(data_only.predict(x_test)), test)

    # Hybrid: expose the explicit marketing prior to AI as a knowledge feature.
    hybrid = HistGradientBoostingRegressor(**common)
    hybrid.fit(
        np.column_stack([x_train, np.log1p(prior_train)]),
        np.log1p(y_train),
    )
    pred_hybrid = np.expm1(
        hybrid.predict(np.column_stack([x_test, np.log1p(prior_test)]))
    )
    pred_hybrid = constrained(pred_hybrid, test)

    metrics = pd.DataFrame([
        score("Knowledge prior", y_test, prior_test),
        score("Data-only AI", y_test, pred_data),
        score("Knowledge-guided hybrid AI", y_test, pred_hybrid),
    ])
    metrics[["MAE", "WAPE", "RMSPE"]] = metrics[["MAE", "WAPE", "RMSPE"]].round(4)

    segments = []
    for label, mask in [
        ("Promotion days", test["Promo"].eq(1)),
        ("Non-promotion days", test["Promo"].eq(0)),
        ("School holidays", test["SchoolHoliday"].eq(1)),
        ("All open days", test["Open"].eq(1)),
    ]:
        yt, ph, pd_only = y_test[mask], pred_hybrid[mask], pred_data[mask]
        segments.append({
            "Segment": label,
            "Observations": int(mask.sum()),
            "Hybrid_WAPE": wmape(yt, ph),
            "Data_only_WAPE": wmape(yt, pd_only),
            "Relative_gain_pct": 100 * (wmape(yt, pd_only) - wmape(yt, ph)) / wmape(yt, pd_only),
        })
    segment_table = pd.DataFrame(segments).round(4)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "table1_model_performance.csv", index=False)
    segment_table.to_csv(output / "table2_marketing_segments.csv", index=False)

    print(f"Train: {train['Date'].min().date()} to {train['Date'].max().date()} ({len(train):,} rows)")
    print(f"Test:  {test['Date'].min().date()} to {test['Date'].max().date()} ({len(test):,} rows)")
    print("\nTable 1\n", metrics.to_string(index=False))
    print("\nTable 2\n", segment_table.to_string(index=False))


if __name__ == "__main__":
    main()
