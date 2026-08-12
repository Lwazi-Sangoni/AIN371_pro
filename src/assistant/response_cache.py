"""Simple file cache for Gemini responses to reduce API usage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import OUTPUTS_DIR

CACHE_PATH = OUTPUTS_DIR / "gemini_cache.json"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def make_cache_key(student_id: str, features: dict, question: str) -> str:
    payload = json.dumps(
        {"student_id": student_id, "features": features, "question": question.strip().lower()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_cached(key: str) -> str | None:
    return _load_cache().get(key)


def set_cached(key: str, explanation: str) -> None:
    cache = _load_cache()
    cache[key] = explanation
    _save_cache(cache)
