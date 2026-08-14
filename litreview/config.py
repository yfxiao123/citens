"""Application configuration, loaded from environment / .env via pydantic-settings.

All settings can be overridden by environment variables (case-insensitive) or a
local ``.env`` file. Run-time knobs (``--max-papers`` etc.) live on the CLI and
the :class:`RunOptions` dataclass, not here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings. Stable defaults let the app run with a minimal ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM backend ---------------------------------------------------------
    # "openai"  -> OpenAI-compatible Chat Completions (OpenAI / DeepSeek / Ollama /
    #              OpenRouter / vLLM / Groq ...). Uses LLM_API_BASE when set.
    # "litellm" -> native multi-provider via LiteLLM (requires the [multi] extra).
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_api_base: str = ""  # e.g. https://api.deepseek.com/v1
    llm_timeout: float = 120.0
    # Reasoning models (e.g. DeepSeek-V4-flash) consume tokens for "thinking";
    # leave headroom so JSON outputs are not truncated to empty.
    llm_max_tokens_default: int = 4096
    # Stronger model for intelligence-heavy stages (writer / synth / verifier).
    # Empty = use the same LLM_MODEL everywhere (cheap stages — planner/filter/
    # extract — always use LLM_MODEL).
    llm_model_strong: str = ""
    # Thread-pool size for parallel LLM calls (extract / verify / write
    # sections). 1 = sequential.
    llm_concurrency: int = 6

    # --- Search sources ------------------------------------------------------
    # Comma-separated subset, or "all". Order does not imply priority.
    search_sources: str = "arxiv,semantic_scholar,openalex,crossref"
    semantic_scholar_api_key: str = ""  # optional, raises rate limits
    openalex_email: str = ""  # optional, polite-pool

    # --- Access layer (declare what YOU can reach; empty = public web only) ---
    # HTTP(S) proxy used when fetching paywalled PDFs / landing pages (e.g. a
    # campus EZproxy or local SOCKS/HTTP proxy). Lets you reach content the
    # public sources can't.
    http_proxy: str = ""
    https_proxy: str = ""
    # Comma-separated domains you have institutional access to (routed via the
    # proxy). Empty = apply proxy to every fetch when a proxy is set.
    accessible_domains: str = ""
    # Campus EZproxy URL prefix. When set, paywalled fetch URLs are rewritten as
    #   {EZPROXY_PREFIX}url=<urlencoded target>
    # so the request rides YOUR library session (e.g. after SSO login in the
    # same network). Example: https://lib.univ.edu.cn/login?url=
    # Leave empty to disable rewriting.
    ezproxy_prefix: str = ""
    # Drop folder for manually downloaded PDFs (see fetch_list.md emitted by a
    # run). Full-text lookup checks here BEFORE hitting the network, so files
    # you drop are picked up by the next run.
    papers_dir: str = "papers"

    # --- Metadata / enrichment API keys (optional) --------------------------
    # Used to fill in missing abstracts and find full text by DOI.
    crossref_email: str = ""  # Crossref polite pool (free, no key needed)
    springer_api_key: str = ""  # api.springernature.com
    elsevier_api_key: str = ""  # dev.elsevier.com
    ieee_api_key: str = ""  # IEEE Xplore

    # --- Defaults for a run (overridable via CLI / RunOptions) ---------------
    default_max_results: int = 60  # candidate pool target before filtering
    default_max_papers: int = 15  # final papers kept after filtering (0 = no cap)
    enrich_abstracts: bool = True  # cross-source DOI enrichment for missing abstracts
    output_dir: str = "runs"

    # --- Venue-aware ranking -------------------------------------------------
    # Composite retrieval score = w_rel*(relevance/5) + w_cit*citation_factor
    #                        + w_ven*venue_factor(SJR quartile). See ranking.py.
    rank_weight_relevance: float = 0.6
    rank_weight_citations: float = 0.2
    rank_weight_venue: float = 0.2
    # SCImago journal-rank CSV (semicolon-delimited). Downloaded once via
    # `litreview sjr` — CC BY-NC licensed, so it is fetched, not shipped.
    sjr_csv_path: str = "data/sjr/sjr.csv"

    # --- Reliability ---------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: str = ".cache"


settings = Settings()
