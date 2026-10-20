"""
FastAPI prediction service for Breast Cancer Triage.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Configuration via Environment Variables
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REDIS_HOST = os.getenv("REDIS_HOST", None)
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")

# Logging setup (stdout only)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("triage-service")

# Global model state
loaded_model = None
model_card_data: Dict[str, Any] = {}
redis_client = None


# ---------------------------------------------------------
# Redis Helper
# ---------------------------------------------------------
def init_redis():
    global redis_client
    if REDIS_HOST:
        try:
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            redis_client = client
            logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            logger.warning("Could not connect to Redis: %s. Service will continue without caching/counting.", e)
            redis_client = None


def increment_request_count():
    if redis_client:
        try:
            redis_client.incr("prediction_count")
        except Exception as e:
            logger.warning("Failed to increment Redis counter: %s", e)


def get_request_count() -> int:
    if redis_client:
        try:
            val = redis_client.get("prediction_count")
            return int(val) if val is not None else 0
        except Exception as e:
            logger.warning("Failed to read Redis counter: %s", e)
            return 0
    return 0


# ---------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model, model_card_data
    logger.info("Starting Breast Cancer Triage Service...")
    logger.info("MODEL_PATH: %s | THRESHOLD: %s | LOG_LEVEL: %s", MODEL_PATH, DECISION_THRESHOLD, LOG_LEVEL)

    # Fail fast if model artifact is missing
    model_file = Path(MODEL_PATH)
    if not model_file.is_file():
        logger.error("FATAL: Model artifact not found at %s. Exiting immediately.", MODEL_PATH)
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")

    try:
        loaded_model = joblib.load(model_file)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        logger.error("FATAL: Failed to deserialize model at %s: %s", MODEL_PATH, e)
        raise RuntimeError(f"Failed to load model: {e}")

    # Load model card if available
    card_file = model_file.parent / "model_card.json"
    if card_file.is_file():
        try:
            with open(card_file, "r", encoding="utf-8") as f:
                model_card_data = json.load(f)
        except Exception as e:
            logger.warning("Could not load model card: %s", e)

    init_redis()
    yield
    logger.info("Shutting down Breast Cancer Triage Service...")


app = FastAPI(
    title="Breast Cancer Triage Service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Pydantic Schemas (Strict validation: 30 features, >= 0)
# ---------------------------------------------------------
class PredictionInput(BaseModel):
    mean_radius: float = Field(..., ge=0.0)
    mean_texture: float = Field(..., ge=0.0)
    mean_perimeter: float = Field(..., ge=0.0)
    mean_area: float = Field(..., ge=0.0)
    mean_smoothness: float = Field(..., ge=0.0)
    mean_compactness: float = Field(..., ge=0.0)
    mean_concavity: float = Field(..., ge=0.0)
    mean_concave_points: float = Field(..., ge=0.0)
    mean_symmetry: float = Field(..., ge=0.0)
    mean_fractal_dimension: float = Field(..., ge=0.0)
    radius_error: float = Field(..., ge=0.0)
    texture_error: float = Field(..., ge=0.0)
    perimeter_error: float = Field(..., ge=0.0)
    area_error: float = Field(..., ge=0.0)
    smoothness_error: float = Field(..., ge=0.0)
    compactness_error: float = Field(..., ge=0.0)
    concavity_error: float = Field(..., ge=0.0)
    concave_points_error: float = Field(..., ge=0.0)
    symmetry_error: float = Field(..., ge=0.0)
    fractal_dimension_error: float = Field(..., ge=0.0)
    worst_radius: float = Field(..., ge=0.0)
    worst_texture: float = Field(..., ge=0.0)
    worst_perimeter: float = Field(..., ge=0.0)
    worst_area: float = Field(..., ge=0.0)
    worst_smoothness: float = Field(..., ge=0.0)
    worst_compactness: float = Field(..., ge=0.0)
    worst_concavity: float = Field(..., ge=0.0)
    worst_concave_points: float = Field(..., ge=0.0)
    worst_symmetry: float = Field(..., ge=0.0)
    worst_fractal_dimension: float = Field(..., ge=0.0)


class PredictionResponse(BaseModel):
    label: str
    malignant_probability: float
    threshold: float
    model_version: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint. Must not depend on Redis."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    """Serve a prediction for a single specimen record."""
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    data_dict = payload.model_dump()
    features = model_card_data.get("feature_names", list(data_dict.keys()))
    row_values = np.array([[data_dict[f] for f in features]], dtype=float)

    try:
        # Probability for malignant (positive class 1)
        probabilities = loaded_model.predict_proba(row_values)[0]
        malignant_prob = float(probabilities[1])
    except Exception as e:
        logger.error("Inference failure: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Determine label based on decision threshold
    label = "malignant" if malignant_prob >= DECISION_THRESHOLD else "benign"

    # Increment Redis counter if available (non-blocking)
    increment_request_count()

    return PredictionResponse(
        label=label,
        malignant_probability=round(malignant_prob, 4),
        threshold=DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info")
def model_info():
    """Returns model metadata and test set training metrics."""
    feature_names = model_card_data.get("feature_names", [])
    test_metrics = (
        model_card_data.get("metrics", {}).get("test")
        or model_card_data.get("test_metrics")
        or {}
    )

    return {
        "model_version": MODEL_VERSION,
        "model_family": model_card_data.get("model_family", "LogisticRegression"),
        "scikit_learn_version": model_card_data.get("sklearn_version", model_card_data.get("scikit_learn_version", "unknown")),
        "git_commit": GIT_SHA,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(feature_names) if feature_names else 30,
        "test_metrics": test_metrics,
    }


@app.get("/stats")
def stats():
    """Returns the count of served predictions from Redis."""
    return {"predictions_served": get_request_count()}
