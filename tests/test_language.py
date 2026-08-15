"""Output-language control: settings.review_language drives prompt injection
and heading localization (no more accidental zh-intro/en-body mix)."""

import pytest

from litreview.agents import writer
from litreview.config import settings


@pytest.fixture()
def lang(request):
    old = settings.review_language
    settings.review_language = request.param
    yield request.param
    settings.review_language = old


@pytest.mark.parametrize("lang", ["en", "zh"], indirect=True)
def test_lang_instruction_matches_language(lang):
    line = writer.lang_instruction()
    if lang == "zh":
        assert "中文" in line
    else:
        assert "English" in line


@pytest.mark.parametrize(
    "lang,heading",
    [("en", "Introduction"), ("zh", "引言")],
    indirect=["lang"],
)
def test_headings_localized(lang, heading):
    assert writer.localized_heading("intro") == heading


@pytest.mark.parametrize("lang", ["en", "zh"], indirect=True)
def test_all_heading_keys_present(lang):
    for key in ("intro", "crit", "conclusion", "refs"):
        assert writer.localized_heading(key).strip()


@pytest.mark.parametrize("lang", ["zh", "chinese", "中文", "CN"], indirect=True)
def test_zh_aliases_normalized(lang):
    assert writer.localized_heading("refs") == "参考文献"


def test_unknown_language_falls_back_to_english():
    old = settings.review_language
    settings.review_language = "fr"
    try:
        assert writer.localized_heading("intro") == "Introduction"
        assert "English" in writer.lang_instruction()
    finally:
        settings.review_language = old
