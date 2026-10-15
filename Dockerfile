FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/churn_model.joblib

WORKDIR /app

# Dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and data, then train inside the image so it is self-contained.
# The full dataset is not in the repository (see data/README.md): if data/churn.csv is absent,
# the model is trained on the small synthetic fixture and is only good for a smoke test.
COPY app/ ./app/
COPY train.py ./
COPY data/ ./data/
COPY tests/fixtures/ ./tests/fixtures/
RUN if [ -f data/churn.csv ]; then python train.py; \
    else python train.py --data tests/fixtures/churn_sample.csv; fi

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','8000')).status==200 else 1)"

# Render / Railway provide $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
