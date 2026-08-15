"""Scoping filters: clarify answers must actually shape retrieval."""

from __future__ import annotations

from datetime import date

from citens.agents.scoping import filters_block, min_year_from_filters


def test_filters_block_renders_answers():
    block = filters_block({"focus": "deep learning methods", "timeframe": "近5年"})
    assert "deep learning methods" in block
    assert "近5年" in block
    assert "focus" in block


def test_filters_block_empty_for_no_filters():
    assert filters_block(None) == ""
    assert filters_block({}) == ""
    assert filters_block({"focus": "  ", "tf": ""}) == ""


def test_min_year_recent_n_years():
    y = min_year_from_filters({"timeframe": "近5年"})
    assert y == date.today().year - 5 + 1
    assert min_year_from_filters({"tf": "last 3 years"}) == date.today().year - 3 + 1
    assert min_year_from_filters({"tf": "最近10年"}) == date.today().year - 10 + 1


def test_min_year_since():
    assert min_year_from_filters({"tf": "since 2020"}) == 2020
    assert min_year_from_filters({"tf": "2020年以后"}) == 2020
    assert min_year_from_filters({"tf": "from 2018"}) == 2018


def test_min_year_none_when_absent():
    assert min_year_from_filters(None) is None
    assert min_year_from_filters({"focus": "queueing models"}) is None


def test_min_year_prefers_recent_over_bare_year():
    # "近3年" beats a stray year mention elsewhere in the answers
    f = {"timeframe": "近3年", "note": "inspired by 2019 surveys"}
    assert min_year_from_filters(f) == date.today().year - 3 + 1
