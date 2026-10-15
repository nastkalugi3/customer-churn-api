"""
predict.py - offline predictions from the command line, no API needed.

Run train.py first to create the model. For --csv you also need the dataset (scripts/download_data.sh).

Examples:
  python predict.py --json '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":42,"Tenure":3,"Balance":75000,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":100000}'
  python predict.py --csv data/churn.csv --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from pydantic import ValidationError

from app.model import churn_model
from app.schemas import Customer

RENAME = {
    "Num Of Products": "NumOfProducts", "Has Credit Card": "HasCrCard",
    "Is Active Member": "IsActiveMember", "Estimated Salary": "EstimatedSalary",
}


def main() -> int:
    """Parse arguments, load the model and print one JSON prediction per line. Returns the exit code."""
    p = argparse.ArgumentParser(description="Offline churn prediction")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", help="JSON for a single customer")
    g.add_argument("--csv", help="CSV file with customers")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    try:
        churn_model.load()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        try:
            # Same validation as the API: ranges, allowed values, no extra or missing fields.
            customer = Customer.model_validate_json(args.json)
        except ValidationError as exc:
            print(f"Invalid customer JSON:\n{exc}", file=sys.stderr)
            return 2
        records = [customer.model_dump(mode="json")]
    else:
        df = pd.read_csv(args.csv).rename(columns=RENAME).head(args.limit)
        records = df[churn_model.features].to_dict(orient="records")

    for out in churn_model.predict(records):
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
