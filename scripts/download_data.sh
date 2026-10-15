#!/usr/bin/env bash
# Download the Churn Modelling dataset from Kaggle into data/churn.csv.
# Needs the Kaggle CLI and credentials (https://www.kaggle.com/docs/api).
set -euo pipefail

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

kaggle datasets download shrutimechlearn/churn-modelling -p "$tmp" --unzip

# Drop RowNumber and rename columns to the names train.py expects.
python3 - "$tmp/Churn_Modelling.csv" data/churn.csv <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1]).drop(columns=["RowNumber"])
df = df.rename(columns={
    "NumOfProducts": "Num Of Products", "HasCrCard": "Has Credit Card",
    "IsActiveMember": "Is Active Member", "EstimatedSalary": "Estimated Salary",
    "Exited": "Churn",
})
df.to_csv(sys.argv[2], index=False)
print(f"Wrote {len(df)} rows to {sys.argv[2]}")
PY
