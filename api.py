from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException
from fastapi import Header, Request
from pydantic import BaseModel

from scripts.predict import predict


class PredictRequest(BaseModel):
    text: str


app = FastAPI(title="Atha Text Classifier API", version="1.0.0")

API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30
REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)


def _enforce_rate_limit(client_ip: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    events = REQUEST_LOG[client_ip]

    while events and events[0] < window_start:
        events.popleft()

    if len(events) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    events.append(now)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(
    payload: PredictRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> dict[str, float | str]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text tidak boleh kosong")

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    client_ip = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_ip)

    try:
        label, confidence = predict(text=text, model_dir="artifacts/model")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(status_code=500, detail="model inference failed")

    return {"label": label, "confidence": round(confidence, 4)}
