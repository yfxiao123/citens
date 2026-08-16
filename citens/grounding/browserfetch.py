"""Browser-grade PDF fetch for publisher endpoints behind bot challenges.

Publishers front their PDF endpoints with a JavaScript client challenge
(Fingerprint-style: the interstitial's URL paths start with ``/_fs-ch-``).
No amount of cookies makes a plain HTTP client pass it — the challenge
wants a real browser engine. This module replays the harvested SSO cookie
jar into a Chromium context (Playwright, optional ``[login]`` extra) and
fetches the PDF with it. Returns None on any failure — callers fall back
to whatever worked before (OA links, abstract).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from citens.config import settings

_CHALLENGE_MARKERS = (b"Client Challenge", b"_fs-ch-", b"challenge-platform")


def _jar_to_playwright_cookies() -> list[dict]:
    """data/cookies.json (host -> header) -> playwright cookie records."""
    import json

    p = Path(settings.cookie_jar_path)
    if not p.is_file():
        return []
    try:
        jar = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for host, header in jar.items():
        for pair in header.split("; "):
            if "=" not in pair:
                continue
            name, value = pair.split("=", 1)
            if not name or not value:
                continue
            out.append(
                {
                    "name": name,
                    "value": value,
                    "domain": host.lstrip(".").lower(),
                    "path": "/",
                }
            )
    return out


def looks_like_challenge(body: bytes) -> bool:
    """True if the response body is a bot-challenge interstitial."""
    if not body or body[:5] == b"%PDF-":
        return False
    head = body[:4000]
    return any(m in head for m in _CHALLENGE_MARKERS)


def fetch_pdf_via_browser(url: str, timeout_ms: int = 45000) -> bytes | None:
    """Fetch `url`'s PDF bytes with Chromium + the SSO cookie jar.

    Two-stage: the context's request API first (fast, no page overhead);
    a full page navigation second (executes the challenge JS, catches
    attachment-style downloads). None if playwright is unavailable or the
    fetch yields no PDF.

    Runs HEADED by default: IEEE (and several publishers) fingerprint
    headless Chromium and serve 418/block pages — a visible window is the
    price of entitlement. Set CITENS_HEADLESS=1 to force headless.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    import os
    from typing import Any

    cookies: list[Any] = _jar_to_playwright_cookies()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=os.environ.get("CITENS_HEADLESS", "") == "1"
            )
            try:
                ctx = browser.new_context(accept_downloads=True)
                if cookies:
                    ctx.add_cookies(cookies)

                # stage 1: API request through the browser's network stack
                try:
                    api_resp = ctx.request.get(url, timeout=timeout_ms)
                    body = api_resp.body()
                    if body[:5] == b"%PDF-":
                        return body
                except Exception:  # noqa: BLE001
                    pass

                # stage 2: real navigation — executes the challenge JS;
                # PDFs arrive either as a response body or a download
                page = ctx.new_page()
                downloaded: list[bytes] = []

                def _save_download(dl):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fh:
                            dl.save_as(fh.name)
                            downloaded.append(Path(fh.name).read_bytes())
                    except Exception:  # noqa: BLE001
                        pass

                page.on("download", _save_download)
                try:
                    nav_resp = page.goto(url, timeout=timeout_ms, wait_until="commit")
                    if nav_resp:
                        body = nav_resp.body()
                        if body[:5] == b"%PDF-":
                            return body
                except Exception:  # noqa: BLE001 — timeouts may still deliver
                    pass
                if downloaded:
                    return downloaded[0]
                return None
            finally:
                browser.close()
    except Exception:  # noqa: BLE001
        return None
