from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException
from fastapi import Header, Request
from pydantic import BaseModel

from scripts.predict import predict


class PredictRequest(BaseModel):
    text: str


app = FastAPI(title="Atha Text Classifier API", version="1.0.0")

RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
USAGE_DB_PATH = Path(os.getenv("USAGE_DB_PATH", "artifacts/usage/usage.db"))

REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)


def _load_api_key_config() -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}

    raw_json = os.getenv("API_KEYS_JSON", "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("API_KEYS_JSON harus berupa JSON object")
        for key, value in parsed.items():
            if not isinstance(value, dict):
                raise ValueError("Setiap value API key harus object")
            owner = str(value.get("owner", "anonymous"))
            rpm = int(value.get("rpm", DEFAULT_RATE_LIMIT_PER_MIN))
            config[str(key)] = {"owner": owner, "rpm": rpm}

    # Backward compatibility: single API_KEY lama tetap didukung.
    legacy_key = os.getenv("API_KEY", "").strip()
    if legacy_key and legacy_key not in config:
        config[legacy_key] = {"owner": "legacy", "rpm": DEFAULT_RATE_LIMIT_PER_MIN}

    return config


API_KEY_CONFIG = _load_api_key_config()


def _init_usage_db() -> None:
    USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(USAGE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                owner TEXT NOT NULL,
                api_key_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                text_len INTEGER NOT NULL,
                label TEXT,
                confidence REAL,
                client_ip TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_logs(ts)"
        )


_init_usage_db()


def _mask_key(key: str) -> str:
    if len(key) <= 6:
        return "***"
    return f"{key[:3]}...{key[-3:]}"


def _resolve_client(api_key: str | None) -> tuple[str, str, int]:
    # Jika API keys dikonfigurasi, key wajib valid.
    if API_KEY_CONFIG:
        if not api_key or api_key not in API_KEY_CONFIG:
            raise HTTPException(status_code=401, detail="invalid api key")
        cfg = API_KEY_CONFIG[api_key]
        return str(cfg["owner"]), _mask_key(api_key), int(cfg["rpm"])

    # Local development mode tanpa key
    return "public", "public", DEFAULT_RATE_LIMIT_PER_MIN


def _log_usage(
    *,
    request_id: str,
    owner: str,
    api_key_id: str,
    endpoint: str,
    status_code: int,
    latency_ms: int,
    text_len: int,
    label: str | None,
    confidence: float | None,
    client_ip: str,
    error: str | None,
) -> None:
    with sqlite3.connect(USAGE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO usage_logs (
                request_id, ts, owner, api_key_id, endpoint, status_code,
                latency_ms, text_len, label, confidence, client_ip, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                int(time.time()),
                owner,
                api_key_id,
                endpoint,
                status_code,
                latency_ms,
                text_len,
                label,
                confidence,
                client_ip,
                error,
            ),
        )


def _enforce_rate_limit(scope: str, max_requests: int) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    events = REQUEST_LOG[scope]

    while events and events[0] < window_start:
        events.popleft()

    if len(events) >= max_requests:
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
    request_id = str(uuid4())
    started_at = time.time()

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text tidak boleh kosong")

    owner, api_key_id, rpm = _resolve_client(x_api_key)

    client_ip = request.client.host if request.client else "unknown"
    rate_limit_scope = f"{owner}:{api_key_id}"
    _enforce_rate_limit(rate_limit_scope, rpm)

    try:
        label, confidence = predict(text=text, model_dir="artifacts/model")
        status_code = 200
        error = None
    except FileNotFoundError as exc:
        status_code = 503
        error = str(exc)
        _log_usage(
            request_id=request_id,
            owner=owner,
            api_key_id=api_key_id,
            endpoint="/predict",
            status_code=status_code,
            latency_ms=int((time.time() - started_at) * 1000),
            text_len=len(text),
            label=None,
            confidence=None,
            client_ip=client_ip,
            error=error,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError):
        status_code = 500
        error = "model inference failed"
        _log_usage(
            request_id=request_id,
            owner=owner,
            api_key_id=api_key_id,
            endpoint="/predict",
            status_code=status_code,
            latency_ms=int((time.time() - started_at) * 1000),
            text_len=len(text),
            label=None,
            confidence=None,
            client_ip=client_ip,
            error=error,
        )
        raise HTTPException(status_code=500, detail="model inference failed")

    latency_ms = int((time.time() - started_at) * 1000)
    rounded_confidence = round(confidence, 4)

    _log_usage(
        request_id=request_id,
        owner=owner,
        api_key_id=api_key_id,
        endpoint="/predict",
        status_code=status_code,
        latency_ms=latency_ms,
        text_len=len(text),
        label=label,
        confidence=rounded_confidence,
        client_ip=client_ip,
        error=error,
    )

    return {
        "request_id": request_id,
        "label": label,
        "confidence": rounded_confidence,
    }


@app.get("/usage/summary")
def usage_summary(
    window_seconds: int = 86400,
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="invalid admin key")

    since_ts = int(time.time()) - window_seconds
    with sqlite3.connect(USAGE_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT owner,
                   COUNT(*) AS total_requests,
                   SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) AS success_requests,
                   SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS failed_requests
            FROM usage_logs
            WHERE ts >= ?
            GROUP BY owner
            ORDER BY total_requests DESC
            """,
            (since_ts,),
        ).fetchall()

    summary = [
        {
            "owner": row[0],
            "total_requests": int(row[1] or 0),
            "success_requests": int(row[2] or 0),
            "failed_requests": int(row[3] or 0),
        }
        for row in rows
    ]

    return {"window_seconds": window_seconds, "items": summary}
