"""
FastAPI application for bank customer churn prediction.

Endpoints:
  GET  /health         - service and model status
  POST /predict        - prediction for one customer
  POST /predict/batch  - predictions for a list of customers
Swagger UI is at /docs, the OpenAPI schema at /openapi.json.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app import __version__
from app.model import ModelNotLoadedError, churn_model
from app.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    Customer,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    try:
        churn_model.load()
    except FileNotFoundError as exc:
        # The service still starts; /predict returns 503 and /health shows "degraded".
        logger.error("Could not load the model at startup: %s", exc)
    yield


app = FastAPI(
    title="Customer Churn API",
    description="Predicts the probability that a bank customer will churn.",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its processing time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Model is unavailable. Check /health and train the model (train.py)."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Report service status: "ok" if the model is loaded, "degraded" otherwise."""
    return HealthResponse(
        status="ok" if churn_model.is_loaded else "degraded",
        model_loaded=churn_model.is_loaded,
        model_roc_auc=churn_model.roc_auc,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: Customer) -> PredictionResponse:
    """Return churn probability, predicted class and risk level for one customer.

    422 on invalid input, 503 if the model is not loaded.
    """
    result = churn_model.predict([customer.model_dump()])[0]
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchRequest) -> BatchPredictionResponse:
    """Predict for 1 to 1000 customers; results come back in the same order as the input."""
    records = [c.model_dump() for c in payload.customers]
    results = churn_model.predict(records)
    return BatchPredictionResponse(predictions=[PredictionResponse(**r) for r in results])


@app.get("/", tags=["service"])
def root():
    """Return the service name, version and the docs path."""
    return {"service": "Customer Churn API", "version": __version__, "docs": "/docs"}
