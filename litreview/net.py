"""Network access layer — honors the user's declared access.

Users often have institutional access the public sources don't (campus EZproxy,
a local proxy, or a VPN). This module routes HTTP fetches (PDFs, landing pages,
enrichment APIs) through a configured proxy and, optionally, only for domains
the user says they can reach. It also supports EZproxy URL-rewriting, the way
most campus libraries proxy publisher content:

    https://lib.univ.edu.cn/login?url=https%3A%2F%2Fwww.sciencedirect.com%2F...

Keeps the httpx version difference (proxy= vs proxies=) in one place.
"""

from __future__ import annotations

from urllib.parse import quote, urlparse

import httpx

from litreview.config import settings

# Hosts that serve open metadata/PDFs — never worth routing through a proxy
# or an EZproxy rewrite (would only add latency and breakage).
_FREE_HOSTS = {
    "arxiv.org",
    "export.arxiv.org",
    "api.unpaywall.org",
    "api.openalex.org",
    "api.crossref.org",
    "api.semanticscholar.org",
}


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


def rewrite_url(url: str) -> str:
    """Rewrite `url` through the campus EZproxy prefix when appropriate.

    Rules:
      * no prefix configured -> unchanged;
      * free/open hosts (arXiv, OpenAlex, ...) -> unchanged;
      * ACCESSIBLE_DOMAINS set -> rewrite only those hosts;
      * ACCESSIBLE_DOMAINS empty -> rewrite every non-free host (setting a
        prefix means "this is how I reach the web").
    """
    prefix = settings.ezproxy_prefix.strip()
    if not prefix:
        return url
    host = (urlparse(url).hostname or "").lower()
    if not host or host in _FREE_HOSTS:
        return url
    if not domain_allowed(url):
        return url
    if not prefix.endswith("="):
        prefix += "?url="  # tolerate a prefix given without the query param
    return f"{prefix}{quote(url, safe='')}"


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
