"""Network access layer — honors the user's declared access.

Users often have institutional access the public sources don't (campus EZproxy,
a local proxy, or a VPN). This module routes HTTP fetches (PDFs, landing pages,
enrichment APIs) through a configured proxy and, optionally, only for domains
the user says they can reach. Keeps the httpx version difference (proxy= vs
proxies=) in one place.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from litreview.config import settings


def domain_allowed(url: str | None) -> bool:
    """True if `url`'s host is in the user's accessible_domains (empty = all)."""
    if not url:
        return True
    allowed = [d.strip().lower() for d in settings.accessible_domains.split(",") if d.strip()]
    if not allowed:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == a or host.endswith("." + a) for a in allowed)


def proxy_url_for(url: str | None) -> str | None:
    """The proxy to use for `url`, or None. Respects accessible_domains."""
    if url and not domain_allowed(url):
        return None
    if not (settings.http_proxy or settings.https_proxy):
        return None
    scheme = urlparse(url).scheme if url else "https"
    if scheme == "http":
        return settings.http_proxy or settings.https_proxy or None
    return settings.https_proxy or settings.http_proxy or None


def sync_client(url: str | None = None, **kwargs) -> httpx.Client:
    """A sync httpx.Client configured with timeout/redirects + proxy (if any)."""
    kwargs.setdefault("timeout", 60)
    kwargs.setdefault("follow_redirects", True)
    proxy = proxy_url_for(url)
    if proxy:
        try:  # httpx >= 0.28
            return httpx.Client(proxy=proxy, **kwargs)
        except TypeError:  # httpx < 0.28
            return httpx.Client(proxies=proxy, **kwargs)
    return httpx.Client(**kwargs)
