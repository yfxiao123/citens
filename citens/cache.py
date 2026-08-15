"""On-disk cache for search results and LLM responses.

Keyed by a stable hash of the logical inputs (namespace + serialized payload),
so identical queries/completions short-circuit on re-runs. Disable via
``CACHE_ENABLED=false``. Cheap by design: one JSON file per entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from citens.config import settings


def _key(namespace: str, payload: Any) -> str:
    raw = json.dumps({"ns": namespace, "p": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _path(key: str) -> str:
    d = os.path.join(settings.cache_dir, "kv")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")


def get(namespace: str, payload: Any) -> Any | None:
    if not settings.cache_enabled:
        return None
    p = _path(_key(namespace, payload))
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def put(namespace: str, payload: Any, value: Any) -> None:
    if not settings.cache_enabled:
        return
    p = _path(_key(namespace, payload))
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except OSError:
        pass
