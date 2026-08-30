"""Live quota / rate-limit probe for every search source.

Answers "为什么又 429 了" with receipts: one polite request per source,
then the provider's own rate-limit headers and error bodies — the same
numbers their dashboards would show. Provider limits are SERVER-SIDE
accounting (per key/IP/day); proxies change latency, never quotas.
"""

from __future__ import annotations

import asyncio

import httpx

from citens.config import settings

_RATE_KEYS = (
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-rate-limit-limit",
    "x-rate-limit-interval",
    "x-rate-limit-reset",
    "retry-after",
)


def _headers_of(resp_headers: httpx.Headers) -> dict[str, str]:
    low = {k.lower(): v for k, v in resp_headers.items()}
    return {k: low[k] for k in _RATE_KEYS if k in low}


async def probe_openalex(hc: httpx.AsyncClient) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("polite-mailto", "yes" if settings.openalex_email else "NO"),
        ("premium-api-key", "yes" if settings.openalex_api_key else "NO"),
    ]
    r = await hc.get(
        "https://api.openalex.org/works",
        params={"per-page": 1, "search": "test"},
    )
    body_note = ""
    if r.status_code == 429:
        try:
            j = r.json().get("error") or {}
            body_note = (
                f"{j.get('status')}: cap={j.get('daily_cap')} "
                f"used={j.get('used_records_in_grid_day')} "
                f"reset={j.get('next_reset_utc')}"
            )
        except Exception:  # noqa: BLE001
            body_note = (r.text or "")[:120]
    rows.append(("http", str(r.status_code)))
    for k, v in _headers_of(r.headers).items():
        rows.append((k, v))
    if body_note:
        rows.append(("body", body_note))
    return rows


async def probe_s2(hc: httpx.AsyncClient) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("api-key", "yes" if settings.semantic_scholar_api_key else "NO")
    ]
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    r = await hc.get(
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2005.11401",
        params={"fields": "title"},
        headers=headers,
    )
    rows.append(("http", str(r.status_code)))
    for k, v in _headers_of(r.headers).items():
        rows.append((k, v))
    if r.status_code == 429:
        rows.append(("body", (r.text or "")[:120]))
    return rows


async def probe_crossref(hc: httpx.AsyncClient) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    r = await hc.get("https://api.crossref.org/works/10.1038/nature12373")
    rows.append(("http", str(r.status_code)))
    for k, v in _headers_of(r.headers).items():
        rows.append((k, v))
    return rows


async def probe_arxiv() -> list[tuple[str, str]]:
    """arXiv has no headers to read; latency is the only signal."""
    import time

    async with httpx.AsyncClient(timeout=20) as hc:
        t0 = time.monotonic()
        r = await hc.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": "all:test", "max_results": 1},
        )
    return [
        ("http", str(r.status_code)),
        ("latency_ms", str(int((time.monotonic() - t0) * 1000))),
    ]


async def probe_all() -> dict[str, list[tuple[str, str]]]:
    """One polite request per source; rate-limit headers as plain pairs."""
    results: dict[str, list[tuple[str, str]]] = {}

    async def _run(name: str, coro) -> None:
        try:
            results[name] = await coro
        except Exception as e:  # noqa: BLE001 — report, never raise
            results[name] = [("error", f"{type(e).__name__}: {e}"[:100])]

    async with httpx.AsyncClient(timeout=25) as hc:
        await asyncio.gather(
            _run("openalex", probe_openalex(hc)),
            _run("semantic_scholar", probe_s2(hc)),
            _run("crossref", probe_crossref(hc)),
        )
    await asyncio.gather(_run("arxiv", probe_arxiv()))
    return results
