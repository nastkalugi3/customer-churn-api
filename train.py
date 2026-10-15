"""Train the churn model.

Builds a self-contained sklearn Pipeline (preprocessing + classifier), evaluates it
on a stratified holdout split and with 5-fold cross-validation, and saves the
artifact to models/churn_model.joblib. Preprocessing lives inside the pipeline,
so the API does not duplicate any data preparation logic.

Usage:
    python train.py
    python train.py --data data/churn.csv --out models/churn_model.joblib
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

RANDOM_STATE = 42

# Features the model accepts (identifiers Surname / CustomerId are dropped).
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
TARGET = "Churn"

# Map raw dataset column names to the feature names used by the API.
RENAME = {
    "Num Of Products": "NumOfProducts",
    "Has Credit Card": "HasCrCard",
    "Is Active Member": "IsActiveMember",
    "Estimated Salary": "EstimatedSalary",
}


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV, rename raw columns to API feature names and check required columns.

    Raises ValueError if any feature or the target column is missing.
    """
    df = pd.read_csv(path).rename(columns=RENAME)
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def build_pipeline() -> Pipeline:
    """Return an unfitted Pipeline: passthrough numeric + one-hot categorical -> gradient boosting."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def main(data_path: str, out_path: str, run_cv: bool = True) -> None:
    """Train on a stratified 80/20 split, report holdout ROC-AUC (and 5-fold CV if run_cv).

    Saves the pipeline with its metadata to out_path and the metrics to metrics.json
    next to it. The model is fitted on the training part of the split only.
    """
    df = load_data(Path(data_path))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe = build_pipeline()
    logger.info("Training on %d rows...", len(X_tr))
    pipe.fit(X_tr, y_tr)

    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    logger.info("Holdout ROC-AUC = %.4f", auc)

    metrics = {"holdout_roc_auc": round(float(auc), 4)}
    if run_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="roc_auc")
        logger.info("5-fold CV ROC-AUC = %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())
        metrics["cv5_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv5_roc_auc_std"] = round(float(cv_scores.std()), 4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipe,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
    }
    joblib.dump(artifact, out)
    logger.info("Model saved to %s", out)
    (out.parent / "metrics.json").write_text(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the customer churn model")
    p.add_argument("--data", default="data/churn.csv")
    p.add_argument("--out", default="models/churn_model.joblib")
    args = p.parse_args()
    main(args.data, args.out)
