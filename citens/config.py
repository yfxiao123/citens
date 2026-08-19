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
    # Deliberation (thinking/reasoning) for judge-side calls — verify batches,
    # defense, rewriter, spot-check, canary, reflect/absence audit. Hybrid
    # reasoning models share the completion budget between thinking and body;
    # full deliberation made each verify batch 3-12x slower. "low" keeps a
    # short deliberation: golden-set validated (agreement with human verdicts
    # preserved); "none" was measurably MORE LENIENT (0.20 vs 0.49 agreement);
    # true restores full thinking; false = "none".
    judge_thinking: bool | str = "low"
    # Supplementary retrieval rounds after the first compose (deep_review used
    # to run 2). Each round is a full recompose incl. re-verification — round 2
    # alone cost ~37 min in the 08-19 run for 3 added papers. 1 keeps the
    # highest-value supplement at a third of the price.
    reflect_max_rounds: int = 1

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
    papers_dir: str = "papers"

    # --- API server (citens.api) ----------------------------------------------
    # Bearer token required on /run, /clarify, /runs, /result when set.
    # Empty = no auth (localhost dev only). SET THIS before exposing the
    # server beyond localhost — /run spends your LLM credits.
    api_token: str = ""
    # Comma-separated allowed CORS origins (empty = no CORS middleware).
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # --- Metadata / enrichment API keys (optional) --------------------------
    # Used to fill in missing abstracts and find full text by DOI.
    crossref_email: str = ""  # Crossref polite pool (free, no key needed)
    springer_api_key: str = ""  # api.springernature.com
    elsevier_api_key: str = ""  # dev.elsevier.com
    ieee_api_key: str = ""  # IEEE Xplore
    core_api_key: str = ""  # api.core.ac.uk (free key; OA fulltext aggregator)
    # Chunk retrieval for grounding: "bm25" (default) | "keyword" | "embedding".
    # "embedding" additionally needs EMBEDDING_MODEL and degrades to bm25 on failure.
    retriever: str = "bm25"
    embedding_model: str = ""  # e.g. text-embedding-3-small (OpenAI-compatible)

    # --- Defaults for a run (overridable via CLI / RunOptions) ---------------
    default_max_results: int = 60  # candidate pool target before filtering
    default_max_papers: int = 20  # final papers kept after filtering (0 = no cap)
    # Supporting-reference layer: filtered-relevant papers beyond the core cap
    # that join the bibliography as abstract-only citations (background /
    # comparison cites). This is what separates "papers deep-read" from
    # "papers cited" — a real survey cites far more than it dissects.
    default_support_papers: int = 15
    enrich_abstracts: bool = True  # cross-source DOI enrichment for missing abstracts
    output_dir: str = "runs"
    # Persistent literature pools built by `citens collect` (JSONL per topic).
    litdb_dir: str = "data/litdb"
    # Domain profile for collect/run (see citens profiles for the list).
    # "" = generic. Example: "finance".
    profile: str = ""
    # Output language of the review prose AND section headings: "en" or "zh".
    # Default Chinese — the primary reader writes Chinese reviews; anything
    # not recognized as Chinese falls back to English.
    review_language: str = "zh"

    # --- Venue-aware ranking -------------------------------------------------
    # Composite retrieval score = w_rel*(relevance/5) + w_cit*citation_factor
    #                        + w_ven*venue_factor(SJR quartile). See ranking.py.
    rank_weight_relevance: float = 0.6
    rank_weight_citations: float = 0.2
    rank_weight_venue: float = 0.2
    # First-author engagement (works/h-index from `citens collect`); excluded
    # from the composite (weights renormalized) when the metadata is unknown.
    rank_weight_author: float = 0.15
    # SCImago journal-rank CSV (semicolon-delimited). Downloaded once via
    # `citens sjr` — CC BY-NC licensed, so it is fetched, not shipped.
    sjr_csv_path: str = "data/sjr/sjr.csv"

    # --- Reliability ---------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    # Max age of cache entries, in days (0 = never expire). Search/enrich
    # results go stale as the indexes move; LLM responses keyed on fixed
    # prompts never do, so namespaces expire at different rates in cache.py.
    cache_ttl_days: int = 30
    # Auto-delete entries older than ttl on put(), throttled to one sweep
    # per N days (marker file). 0 disables sweeping.
    cache_sweep_interval_days: int = 1


settings = Settings()
