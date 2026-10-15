"""Model artifact loading and inference."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

logger = logging.getLogger("app.model")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.joblib"))


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested before the model is loaded."""


class ChurnModel:
    """Thin wrapper around the saved pipeline: load + predict."""

    def __init__(self) -> None:
        self._pipeline = None
        self._metrics: Dict[str, float] = {}
        self.features: List[str] = []

    def load(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Run train.py first."
            )
        artifact = joblib.load(path)
        self._pipeline = artifact["pipeline"]
        self._metrics = artifact.get("metrics", {})
        self.features = artifact["numeric_features"] + artifact["categorical_features"]
        logger.info("Model loaded from %s (metrics: %s)", path, self._metrics)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def roc_auc(self) -> float | None:
        return self._metrics.get("holdout_roc_auc")

    @staticmethod
    def _risk_level(p: float) -> str:
        if p < 0.30:
            return "low"
        if p < 0.60:
            return "medium"
        return "high"

    def predict(self, records: List[dict]) -> List[dict]:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded.")
        df = pd.DataFrame(records)[self.features]  # select features in training order
        probs = self._pipeline.predict_proba(df)[:, 1]
        return [
            {
                "churn_probability": round(float(p), 4),
                "churn": bool(p >= 0.5),
                "risk_level": self._risk_level(float(p)),
            }
            for p in probs
        ]


# One shared instance per process.
churn_model = ChurnModel()
