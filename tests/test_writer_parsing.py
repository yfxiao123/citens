"""Writer post-processing helpers: the regex layer that cleans LLM output.

These run on every section of every review — a regression here corrupts all
output (duplicate headings, model-hallucinated reference lists, mid-sentence
truncations), yet none of it was covered before.
"""

from litreview.agents.writer import (
    _complete,
    _strip_leading_headings,
    _strip_tail_references,
)


class TestStripLeadingHeadings:
    def test_removes_hash_heading(self):
        assert _strip_leading_headings("## Introduction\n\nBody text.") == "Body text."

    def test_removes_bold_only_title(self):
        assert _strip_leading_headings("**Critical Synthesis**\nBody.") == "Body."

    def test_removes_stacked_headings(self):
        assert (
            _strip_leading_headings("## Theme\n**Theme**\n\nActual prose.")
            == "Actual prose."
        )

    def test_keeps_normal_prose_untouched(self):
        text = "Regular paragraph mentioning # hashtags inline."
        assert _strip_leading_headings(text) == text

    def test_does_not_eat_body_lines_starting_with_asterisks(self):
        # only a WHOLE line that is bold counts as a title
        text = "Prose start. **Bold at end of sentence** is fine."
        assert _strip_leading_headings(text) == text

    def test_empty(self):
        assert _strip_leading_headings("") == ""


class TestStripTailReferences:
    def test_removes_english_references_block(self):
        text = "Final sentence.\n\nReferences\n[0] Some Paper.\n[1] Another."
        assert _strip_tail_references(text) == "Final sentence."

    def test_removes_chinese_references_block(self):
        text = "结论。\n\n参考文献\n[0] 某论文。"
        assert _strip_tail_references(text) == "结论。"

    def test_removes_bold_variant(self):
        text = "End.\n\n**References**\n[0] X."
        assert _strip_tail_references(text) == "End."

    def test_keeps_text_without_references(self):
        assert _strip_tail_references("Just prose. No list.") == "Just prose. No list."

    def test_keeps_inline_mention_of_the_word_references(self):
        # only a standalone header line triggers the strip
        text = "The references cited here are real."
        assert _strip_tail_references(text) == text


class TestComplete:
    def test_terminal_punctuation_variants(self):
        for end in "。．.!?！？；;”\"')]}】》…":
            assert _complete(f"ends with {end}")

    def test_mid_sentence_is_incomplete(self):
        assert not _complete("ends without punctuation")
        assert not _complete("ends with a comma,")

    def test_empty(self):
        assert not _complete("")
        assert not _complete("   ")
