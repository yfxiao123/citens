"""Survey-seed recall: seed picking and reference-harvest orchestration."""

from citens.models import Paper


def _p(title, cites=0, doi=""):
    return Paper(
        title=title, authors=[], year=2023, abstract="",
        source="t", citation_count=cites, doi=doi,
    )


def test_review_shaped():
    from citens.search.survey_seed import review_shaped

    assert review_shaped("A Survey of Large Language Models")
    assert review_shaped("World Models: A Brief Review")
    assert review_shaped("安全强化学习综述")
    assert review_shaped("Deep Learning in NLP: An Overview")
    assert not review_shaped("SafeDreamer: Safe Reinforcement Learning")
    assert not review_shaped("Attention Is All You Need")


def test_pick_pool_seeds_most_cited_first():
    from citens.search.survey_seed import pick_pool_seeds

    pool = [
        _p("SafeDreamer: Safe RL", 0, "10.1/a"),
        _p("A Survey of World Models", 500, "10.1/b"),
        _p("Safe RL: A Review", 40, "10.1/c"),
        _p("Some Paper", 9999),  # huge citations but not review-shaped
    ]
    seeds = pick_pool_seeds(pool, k=2)
    assert [s.doi for s in seeds] == ["10.1/b", "10.1/c"]


def test_survey_seed_papers_harvest_and_order(monkeypatch):
    import asyncio

    import citens.search.snowball as sb
    from citens.search.survey_seed import survey_seed_papers

    async def fake_neighbors(doi, field, limit):
        assert field == "references"
        if doi == "10.1/b":
            return [
                {"title": "Milestone Paper", "citationCount": 900,
                 "externalIds": {"DOI": "10.2/x"}},
                {"title": "Niche Paper", "citationCount": 0,
                 "externalIds": {"DOI": "10.2/y"}},
                {"title": "Duplicate via other key", "citationCount": 5},
            ]
        return []

    def fake_to_paper(item, direction, from_doi):
        ext = item.get("externalIds") or {}
        return _p(
            item.get("title") or "", item.get("citationCount", 0),
            ext.get("DOI") or "",
        )

    monkeypatch.setattr(sb, "_s2_neighbors", fake_neighbors)
    monkeypatch.setattr(sb, "_s2_to_paper", fake_to_paper)

    pool = [
        _p("A Survey of World Models for Planning", 500, "10.1/b"),
        _p("Random Research Paper", 3, "10.1/z"),
    ]
    papers, seeds = asyncio.run(survey_seed_papers(
        "q", pool, planner=["world model planning"], quota=2,
    ))
    assert seeds == ["A Survey of World Models for Planning"]
    titles = [p.title for p in papers]
    assert titles[0] == "Milestone Paper"  # citation-ordered canon head
    assert len(titles) == 2


def test_s2_search_surveys_filters_review_titles(monkeypatch):
    import asyncio

    import citens.search.semantic_scholar as s2mod
    from citens.search import survey_seed as ss

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [
                {"title": "A Survey of Safe RL", "citationCount": 30,
                 "paperId": "a" * 40, "externalIds": {"DOI": "10.3/s"}},
                {"title": "SafeDreamer: Safe RL", "citationCount": 10,
                 "paperId": "b" * 40, "externalIds": {"DOI": "10.3/d"}},
            ]}

    async def fake_get(client, url, *, params, headers=None):
        return _Resp()

    monkeypatch.setattr(s2mod, "_throttled_get", fake_get)
    out = asyncio.run(ss._s2_search_surveys("safe reinforcement learning"))
    assert [it["title"] for it in out] == ["A Survey of Safe RL"]  # dicts now


def test_survey_seed_topic_guard_blocks_wrong_subfield(monkeypatch):
    import asyncio

    import citens.search.snowball as sb
    from citens.search import survey_seed as ss

    async def fake_search(term, limit=10):
        # astronomy literal-"Survey" paper + right-subfield review
        return [
            {"title": "Probing Density Fluctuations using the FIRST Radio Survey",
             "citationCount": 900, "paperId": "c" * 40, "externalIds": {}},
            {"title": "Safe Reinforcement Learning: A Survey",
             "citationCount": 100, "paperId": "d" * 40, "externalIds": {}},
        ]

    async def fake_neighbors(ref, field, limit):
        assert ref == "d" * 40  # only the topic-guarded seed is harvested
        return [{"title": "Milestone", "citationCount": 5, "externalIds": {}}]

    def fake_to_paper(item, direction, from_doi):
        return _p(item.get("title") or "", item.get("citationCount", 0))

    monkeypatch.setattr(ss, "_s2_search_surveys", fake_search)
    monkeypatch.setattr(sb, "_s2_neighbors", fake_neighbors)
    monkeypatch.setattr(sb, "_s2_to_paper", fake_to_paper)

    papers, seeds = asyncio.run(ss.survey_seed_papers(
        "q", [], planner=["safe reinforcement learning world model"],
    ))
    assert seeds == ["Safe Reinforcement Learning: A Survey"]
    assert [p.title for p in papers] == ["Milestone"]


def test_pairwise_refine_swaps_only_tied_adjacents(monkeypatch):
    from citens.agents import rerank as rr
    from citens.models import Paper

    mk = lambda t: Paper(title=t, authors=[], year=2024, abstract="",
                         source="t")
    papers = [mk(p) for p in ["A0", "B0", "C0", "D0", "E0", "F0", "G0", "H0",
                              "tail1", "tail2"]]
    scores = [10, 9, 9, 9, 8, 8, 7, 7]  # ties: (1,2),(2,3),(4,5),(6,7)

    calls = []

    def fake_pick(question, a, b):
        calls.append((a.title, b.title))
        return "B"  # always prefer the second

    monkeypatch.setattr(rr, "_pairwise_pick", fake_pick)
    out = rr._pairwise_refine("q", papers, scores)
    titles = [p.title for p in out]
    assert titles[0] == "A0"  # no tie with neighbor -> untouched
    assert titles[-2:] == ["tail1", "tail2"]  # tail untouched
    # always-B rotates each tied run: the run's original head moves last
    # ((B0,C0,D0) -> C0,D0,B0; (E0,F0) -> F0,E0)
    assert titles[1:4] == ["C0", "D0", "B0"]
    assert titles[4:6] == ["F0", "E0"]
    # only adjacent tied pairs were compared
    assert len(calls) == 4
