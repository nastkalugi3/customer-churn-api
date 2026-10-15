"""Generate a small synthetic CSV with the same columns as data/churn.csv.

The fixture is used by tests and CI, where the real dataset is not available.
It is random data with a weak churn signal, not a sample of the real dataset.

Usage:
    python scripts/make_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 200
RANDOM_STATE = 42
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "churn_sample.csv"


def make_fixture(rows: int = ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    products = rng.choice([1, 2, 3, 4], rows, p=[0.5, 0.45, 0.04, 0.01])
    active = rng.integers(0, 2, rows)
    geography = rng.choice(["France", "Spain", "Germany"], rows, p=[0.5, 0.25, 0.25])
    balance = np.where(rng.random(rows) < 0.3, 0.0, rng.normal(80000, 30000, rows).clip(0)).round(2)

    # Weak, made-up signal so both classes are present and the model can learn something.
    logit = -1.5 + 0.04 * (age - 40) - 0.9 * active + 0.6 * (geography == "Germany") + 0.5 * (products >= 3)
    churn = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "CustomerId": 15600000 + np.arange(rows),
        "Surname": [f"Customer{i}" for i in range(rows)],
        "CreditScore": rng.integers(350, 851, rows),
        "Geography": geography,
        "Gender": rng.choice(["Male", "Female"], rows),
        "Age": age,
        "Tenure": rng.integers(0, 11, rows),
        "Balance": balance,
        "Num Of Products": products,
        "Has Credit Card": rng.integers(0, 2, rows),
        "Is Active Member": active,
        "Estimated Salary": rng.uniform(1000, 200000, rows).round(2),
        "Churn": churn,
    })


if __name__ == "__main__":
    df = make_fixture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT} (churn rate {df['Churn'].mean():.3f})")
