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


def test_arxiv_href_regex_includes_alphaxiv_and_html():
    from citens.search.websearch import _ARXIV_HREF_RE

    m = _ARXIV_HREF_RE.search("https://alphaxiv.org/abs/2402.15610")
    assert m and m.group(1) == "2402.15610"
    m = _ARXIV_HREF_RE.search("https://arxiv.org/html/2309.15840v2")
    assert m and m.group(1) == "2309.15840"
    assert not _ARXIV_HREF_RE.search("https://arxiv.org/list/cs.LG/recent")


def test_s2_href_regex():
    from citens.search.websearch import _S2_HREF_RE

    url = ("https://www.semanticscholar.org/paper/SafeDreamer%3A-Safe-RL"
           "-with-World-Models/abc123abc123abc123abc123abc123abc123abc1")
    m = _S2_HREF_RE.search(url)
    assert m and m.group(1) == "abc123abc123abc123abc123abc123abc123abc1"
    assert not _S2_HREF_RE.search("https://semanticscholar.org/my-library")


class _FakeResp:
    def __init__(self, text="", payload=None, status_code=200):
        self.text = text
        self._payload = payload or {}
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    async def get(self, url, **kw):
        self.calls.append(url)
        return self._resp


def test_title_anchor_arxiv_parses_and_gates():
    import asyncio

    from citens.search.websearch import _title_anchor_arxiv

    feed = """<feed><entry><title>SafeDreamer: Safe Reinforcement Learning
      with World Models</title><id>http://arxiv.org/abs/2408.08787v2</id>
      </entry><entry><title>Unrelated Paper on Cooking</title>
      <id>http://arxiv.org/abs/1900.00001v1</id></entry></feed>"""
    p = asyncio.run(_title_anchor_arxiv(
        _FakeClient(_FakeResp(text=feed)),
        "Safe Reinforcement Learning with World Models",
    ))
    assert p is not None
    assert p.title.startswith("SafeDreamer")
    assert p.doi == "10.48550/arxiv.2408.08787v2"

    # subtitle-level similarity (0.88 measured) still passes the 0.75 gate
    feed2 = feed.replace("SafeDreamer: Safe", "SafeDreamer%3A Safe")
    p2 = asyncio.run(_title_anchor_arxiv(
        _FakeClient(_FakeResp(text=feed2)),
        "Safe Reinforcement Learning with World Models",
    ))
    assert p2 is not None

    # no similar title -> no anchor
    p3 = asyncio.run(_title_anchor_arxiv(
        _FakeClient(_FakeResp(text=feed)),
        "Completely Different Query About Quantum Chemistry",
    ))
    assert p3 is None


def test_title_anchor_crossref_gate():
    import asyncio

    from citens.search.websearch import _title_anchor_crossref

    miss = _FakeResp(payload={"message": {"items": [
        {"title": ["Some Other Journal Article Name"], "DOI": "10.1/x"},
    ]}})
    p = asyncio.run(_title_anchor_crossref(
        _FakeClient(miss), "Completely Different Query About Something",
    ))
    assert p is None

    hit = _FakeResp(payload={"message": {"items": [
        {"title": ["A Paper About Exactly This Topic With A Long Title"],
         "DOI": "10.1234/abc", "is-referenced-by-count": 7},
    ]}})
    p2 = asyncio.run(_title_anchor_crossref(
        _FakeClient(hit), "A Paper About Exactly This Topic With A Long Title",
    ))
    assert p2 is not None and p2.doi == "10.1234/abc"


def test_query_prompt_requires_verbatim_phrase():
    from citens.search.websearch import QUERY_SYSTEM

    assert "VERBATIM" in QUERY_SYSTEM


def test_doi_href_regex_catches_publisher_paths():
    from urllib.parse import unquote

    from citens.search.websearch import _DOI_HREF_RE

    cases = [
        "https://doi.org/10.18653/v1/2021.naacl-main.79",
        "https://dl.acm.org/doi/10.1145/3542927",
        "https://journals.sagepub.com/doi/10.3233/FAIA251029",
        "https://link.springer.com/article/10.1007/s10489-026-07175-4",
        "https://ieeexplore.ieee.org/document/12345",  # no DOI → no match
    ]
    got = [_DOI_HREF_RE.search(u) for u in cases]
    assert got[0] and got[0].group(1) == "10.18653/v1/2021.naacl-main.79"
    assert got[1] and got[1].group(1) == "10.1145/3542927"
    assert got[2] and got[2].group(1) == "10.3233/FAIA251029"
    assert got[3] and got[3].group(1) == "10.1007/s10489-026-07175-4"
    assert got[4] is None


def test_acl_href_regex_strips_extension():
    from citens.search.websearch import _ACL_HREF_RE

    m = _ACL_HREF_RE.search("https://aclanthology.org/2021.naacl-main.79.pdf")
    assert m and m.group(1) == "2021.naacl-main.79"
    m = _ACL_HREF_RE.search("https://aclanthology.org/2021.naacl-main.79/")
    assert m and m.group(1) == "2021.naacl-main.79"
    m = _ACL_HREF_RE.search("https://aclanthology.org/P19-1234")
    assert m and m.group(1) == "P19-1234"
