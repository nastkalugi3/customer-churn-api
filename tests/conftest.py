"""Make sure a model artifact exists, so `pytest` works on a clean clone.

If models/churn_model.joblib is missing, train one. The real dataset (data/churn.csv) is not
stored in the repository, so without it the model is trained on the small synthetic fixture.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = Path(os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "churn_model.joblib")))
FULL_DATA = ROOT / "data" / "churn.csv"
FIXTURE_DATA = ROOT / "tests" / "fixtures" / "churn_sample.csv"

if not MODEL_PATH.exists():
    import train  # noqa: E402

    data = FULL_DATA if FULL_DATA.exists() else FIXTURE_DATA
    train.main(str(data), str(MODEL_PATH), run_cv=False)
