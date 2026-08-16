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

from citens.config import settings

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


def _ezproxy_headers(url: str) -> dict[str, str]:
    """Cookie for the EZproxy host when the user lent us their SSO session.

    Off-campus EZproxy authenticates by session cookie; the rewritten URL
    alone just bounces to an SSO login page (HTML, rejected downstream).
    """
    prefix = settings.ezproxy_prefix.strip()
    cookie = settings.ezproxy_cookie.strip()
    if not prefix or not cookie:
        return {}
    prefix_host = (urlparse(prefix).hostname or "").lower()
    host = (urlparse(url).hostname or "").lower()
    if prefix_host and host == prefix_host:
        return {"Cookie": cookie}
    return {}


def load_cookie_jar() -> dict[str, str]:
    """Host -> raw Cookie header, written by ``citens login``."""
    import json
    from pathlib import Path

    p = Path(settings.cookie_jar_path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except Exception:  # noqa: BLE001 — a corrupt jar must not break fetching
        return {}


def save_cookie_jar(jar: dict[str, str]) -> None:
    import json
    from pathlib import Path

    p = Path(settings.cookie_jar_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jar, indent=2), encoding="utf-8")


def _jar_cookies() -> httpx.Cookies:
    """The harvested SSO jar as a real cookie jar (per-cookie domains).

    Publisher entitlement flows CROSS DOMAINS by redirect (link.springer.com
    -> springernature.com idp -> back with a token); a static Cookie header
    for one host dies at the first hop. A populated httpx cookie jar sends
    the right cookies at every hop automatically.
    """
    cookies = httpx.Cookies()
    for host, header in load_cookie_jar().items():
        for pair in header.split("; "):
            if "=" not in pair:
                continue
            name, value = pair.split("=", 1)
            # leading dot => domain cookie (covers subdomains like
            # www.sciencedirect.com); without it cookielib does exact-host only
            dom = host.lstrip(".").lower()
            try:
                cookies.set(name, value, domain=f".{dom}")
            except Exception:  # noqa: BLE001 — a malformed entry skips silently
                continue
    return cookies


def sync_client(url: str | None = None, **kwargs) -> httpx.Client:
    """A sync httpx.Client configured with timeout/redirects + proxy (if any).

    Loads the SSO cookie jar (when present) as native cookies so redirect
    chains through publisher/IdP domains keep their sessions.
    """
    kwargs.setdefault("timeout", 60)
    kwargs.setdefault("follow_redirects", True)
    jar = _jar_cookies()
    if len(jar.jar):
        existing = kwargs.pop("cookies", None)
        merged = httpx.Cookies(existing) if existing else httpx.Cookies()
        merged.update(jar)
        kwargs["cookies"] = merged
    if url:
        headers = {
            **_ezproxy_headers(url),
            **(kwargs.pop("headers", {}) or {}),
        }
        if headers:
            kwargs["headers"] = headers
    proxy = proxy_url_for(url)
    if proxy:
        try:  # httpx >= 0.28
            return httpx.Client(proxy=proxy, **kwargs)
        except TypeError:  # httpx < 0.28
            return httpx.Client(proxies=proxy, **kwargs)  # type: ignore[call-arg]
    return httpx.Client(**kwargs)
