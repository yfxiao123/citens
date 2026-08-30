"""Web bridge channel: SERP parsing, engine fallback, title anchoring."""

DDG_HTML = """
<div class="result results_links">
 <div class="links_main links_deep result__body">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
    href="//duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F2307.07176&amp;rut=x">SafeDreamer: Safe RL</a></h2>
  <div class="result__extras"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ficlr.cc%2Fx">iclr</a></div>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F2307.07176">The deployment of <b>RL</b> in real-world applications</a>
 </div>
</div>
"""

BING_HTML = """
<li class="b_algo">
  <h2><a href="https://arxiv.org/abs/2305.01812">Post-Abstention: Towards Reliably Re-Attempting</a></h2>
  <div class="b_caption"><p>We introduce the Post-Abstention task.</p></div>
</li>
<li class="b_algo">
  <h2><a href="/ck/a?!wrong">no http prefix</a></h2>
  <p>skipped</p>
</li>
"""


def test_parse_ddg_results_decodes_and_strips():
    from citens.search.websearch import parse_ddg_results

    res = parse_ddg_results(DDG_HTML)
    assert len(res) == 2  # title anchor + the extras-block anchor
    assert res[0]["title"] == "SafeDreamer: Safe RL"
    assert res[0]["url"] == "https://arxiv.org/abs/2307.07176"
    assert "RL" in res[0]["snippet"] and "<b>" not in res[0]["snippet"]


def test_parse_bing_results_skips_non_http():
    from citens.search.websearch import parse_bing_results

    res = parse_bing_results(BING_HTML)
    assert len(res) == 1
    assert res[0]["url"].startswith("https://arxiv.org/")
    assert "Post-Abstention" in res[0]["title"]


def test_title_candidate_strips_site_suffix():
    from citens.search.websearch import _clean_title_candidate

    assert _clean_title_candidate("MISC: A Mixed Strategy-Aware Model - 知乎") == \
        "MISC: A Mixed Strategy-Aware Model"
    assert _clean_title_candidate("Some Paper | CSDN博客") == "Some Paper"


def test_has_cjk():
    from citens.search.websearch import _has_cjk

    assert _has_cjk("世界模型 规划")
    assert not _has_cjk("world model planning")


def test_engine_chain_serper_primary(monkeypatch):
    import asyncio

    from citens.search import websearch as ws
    from citens.config import settings

    async def fake_serper(client, kw, limit):
        return [{"title": "G", "snippet": "", "url": "https://arxiv.org/abs/1"}]

    monkeypatch.setattr(settings, "serper_api_key", "k")
    monkeypatch.setattr(ws, "_serper_results", fake_serper)
    src = ws.WebSearchSource()

    async def noop(*a, **k):  # keyless engines must not be called
        raise AssertionError("fallback engine used despite serper success")

    monkeypatch.setattr(ws, "parse_bing_results", noop)
    papers = asyncio.run(src.search(["q"], max_results=6))
    assert len(papers) == 0  # 'G' is a junk title candidate — no anchor, no crash


def test_web_query_list_bilingual():
    from citens.search import websearch as ws

    monkey = {
        "en": ["a b c", ""], "zh": ["中文 查询 一"], "bad": "x",
    }
    import citens.llm as llm

    orig = llm.chat_json
    llm.chat_json = lambda *a, **k: monkey
    try:
        out = ws.generate_web_query_list("q")
    finally:
        llm.chat_json = orig
    assert out == ["a b c", "中文 查询 一"]


def test_source_registered():
    from citens.search.base import REGISTRY

    assert "web" in REGISTRY
