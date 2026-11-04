"""Anomaly detection API: serves the trained model and exposes Prometheus metrics."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

PREDICTIONS = Counter("anomaly_predictions_total", "Predictions made", ["server", "result"])
LATENCY = Histogram("anomaly_prediction_seconds", "Time spent computing one prediction")
LAST_SCORE = Gauge("anomaly_score", "Last anomaly score per server", ["server"])
LAST_FLAG = Gauge("anomaly_flag", "1 if the last prediction was an anomaly, else 0", ["server"])
MODEL_LOADED = Gauge("anomaly_model_loaded", "1 if a model is loaded")

state = {"model": None, "meta": {}}


def load_model() -> bool:
    """Load model.joblib + metadata.json from MODEL_DIR. Returns True if loaded."""
    model_dir = Path(os.environ.get("MODEL_DIR", "models/current"))
    model_file, meta_file = model_dir / "model.joblib", model_dir / "metadata.json"
    if not model_file.exists() or not meta_file.exists():
        state["model"], state["meta"] = None, {}
        MODEL_LOADED.set(0)
        return False
    state["model"] = joblib.load(model_file)
    state["meta"] = json.loads(meta_file.read_text())
    MODEL_LOADED.set(1)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Infra Anomaly API", version="0.1.0", lifespan=lifespan)


class ServerMetrics(BaseModel):
    server: str = Field(examples=["web01"])
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    disk_pct: float = Field(ge=0, le=100)
    load1: float = Field(ge=0)
    net_rx_kbps: float = Field(ge=0)
    net_tx_kbps: float = Field(ge=0)


class Prediction(BaseModel):
    server: str
    anomaly: bool
    score: float
    model_version: Optional[str] = None


@app.get("/health")
def health():
    meta = state["meta"]
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_version": meta.get("model_version"),
        "trained_at": meta.get("trained_at"),
    }


@app.post("/predict", response_model=Prediction)
def predict(m: ServerMetrics):
    model, meta = state["model"], state["meta"]
    if model is None:
        raise HTTPException(503, "No model loaded. Run `python src/train.py`, then POST /reload.")
    if m.server not in meta["servers"]:
        raise HTTPException(422, f"Unknown server '{m.server}'. Known servers: {meta['servers']}")

    start = time.perf_counter()
    row = m.model_dump()
    for s in meta["servers"]:
        row[f"is_{s}"] = int(m.server == s)
    X = pd.DataFrame([row])[meta["features"]]

    anomaly = bool(model.predict(X)[0] == -1)
    score = float(-model.score_samples(X)[0])  # higher = more abnormal

    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.labels(m.server, "anomaly" if anomaly else "normal").inc()
    LAST_SCORE.labels(m.server).set(score)
    LAST_FLAG.labels(m.server).set(1 if anomaly else 0)
    return Prediction(server=m.server, anomaly=anomaly, score=round(score, 4),
                      model_version=meta.get("model_version"))


@app.post("/reload")
def reload_model():
    """Load the newest model from disk (after retraining) without restarting."""
    if not load_model():
        raise HTTPException(404, "No model found in MODEL_DIR.")
    return {"reloaded": True, "model_version": state["meta"].get("model_version")}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
