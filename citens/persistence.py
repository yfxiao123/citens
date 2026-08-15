"""Run-directory persistence.

Each run writes to ``runs/<slug>-<timestamp>/`` with a stable layout, replacing
the old flat dump of timestamped files:

    runs/<slug>-<ts>/
    ├── meta.json            # RunMeta
    ├── review.md            # final review
    ├── references.bib       # BibTeX export (Phase 2)
    ├── provenance.json      # claim → citation map (Phase 2)
    └── steps/<NN_name>.json # intermediate artifacts per stage
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from citens.config import settings


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return (s or "run")[:max_len]


def new_run_dir(topic: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(settings.output_dir, f"{_slugify(topic)}-{ts}")
    os.makedirs(os.path.join(path, "steps"), exist_ok=True)
    return path


def _json_default(obj):
    """json.dump default: serialize nested pydantic models / sets / datetimes."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_step(run_dir: str, name: str, data: Any) -> str:
    """Persist an intermediate artifact. Handles pydantic models, lists, raw,
    and dicts with nested pydantic objects."""
    path = os.path.join(run_dir, "steps", f"{name}.json")
    if isinstance(data, list):
        payload: Any = [d.model_dump() if hasattr(d, "model_dump") else d for d in data]
    elif hasattr(data, "model_dump"):
        payload = data.model_dump()
    else:
        payload = data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def save_text(run_dir: str, filename: str, text: str) -> str:
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def save_json(run_dir: str, filename: str, data: Any) -> str:
    path = os.path.join(run_dir, filename)
    payload = data.model_dump() if hasattr(data, "model_dump") else data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return path
