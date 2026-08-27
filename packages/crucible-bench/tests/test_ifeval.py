"""IFEval checkers, against the reference implementation's semantics.

Most cases here are regression locks: each one passed under the previous,
looser implementation and must now fail. A benchmark that drifts generous is
worse than no benchmark, because it reports a number either way.

Reference: github.com/google-research/google-research/tree/master/instruction_following_eval
"""

import sys

import pytest

from crucible_bench.ifeval import (
    INSTRUCTION_CHECKERS,
    UnknownInstruction,
    check_all_instructions,
    check_instruction,
    unsupported_instruction_ids,
)

# The 25 ids the reference registry enables, which is exactly the set that
# appears in google/IFEval (541 prompts, 834 instructions).
REFERENCE_IDS = {
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "startend:quotation",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
}


class TestCoverage:
    def test_registry_matches_the_reference_exactly(self):
        # Neither direction is harmless. A missing id used to reach the
        # unknown-instruction branch and score as followed; an extra id is a
        # checker written against a name the dataset never uses, which is how
        # detectable_format:constrained_response went unnoticed while filed
        # under combination:.
        assert set(INSTRUCTION_CHECKERS) == REFERENCE_IDS

    def test_unknown_instruction_raises_rather_than_passing(self):
        with pytest.raises(UnknownInstruction):
            check_instruction("detectable_format:invented", "anything", {})

    def test_unsupported_ids_are_reported_for_preflight(self):
        assert unsupported_instruction_ids(REFERENCE_IDS) == []
        assert unsupported_instruction_ids(
            {"keywords:existence", "nope:one", "nope:two"}
        ) == ["nope:one", "nope:two"]

    def test_all_instructions_is_conjunctive(self):
        ids = ["punctuation:no_comma", "startend:quotation"]
        passed, per = check_all_instructions('"no commas here"', ids, [{}, {}])
        assert passed and per == [True, True]
        passed, per = check_all_instructions('"has, a comma"', ids, [{}, {}])
        assert not passed and per == [False, True]


class TestBulletLists:
    ID = "detectable_format:number_bullet_lists"

    def test_exact_count_required(self):
        three = "* one\n* two\n* three"
        assert check_instruction(self.ID, three, {"num_bullets": 3})

    def test_over_delivering_is_not_compliance(self):
        # Regression lock: `>= num_bullets` scored this as followed.
        four = "* one\n* two\n* three\n* four"
        assert not check_instruction(self.ID, four, {"num_bullets": 3})

    def test_bold_markers_are_not_bullets(self):
        assert not check_instruction(self.ID, "**bold**\n**also**", {"num_bullets": 2})

    def test_dashes_count(self):
        assert check_instruction(self.ID, "- one\n- two", {"num_bullets": 2})


class TestHighlightedSections:
    ID = "detectable_format:number_highlighted_sections"

    def test_emphasis_counts(self):
        assert check_instruction(self.ID, "*one* and *two*", {"num_highlights": 2})

    def test_headers_do_not_count(self):
        # Regression lock: markdown headers were counted as highlights, which
        # handed free credit to any model that structured its answer -- on the
        # most common instruction id in the dataset.
        text = "# Heading\n## Another\n### Third"
        assert not check_instruction(self.ID, text, {"num_highlights": 2})

    def test_empty_emphasis_does_not_count(self):
        assert not check_instruction(self.ID, "** **", {"num_highlights": 1})


class TestJsonFormat:
    ID = "detectable_format:json_format"

    def test_whole_response_must_parse(self):
        assert check_instruction(self.ID, '{"a": 1}', {})

    def test_fenced_json_is_unwrapped(self):
        assert check_instruction(self.ID, '```json\n{"a": 1}\n```', {})

    def test_prose_containing_an_object_fails(self):
        # Regression lock: the old fallback searched for any embedded {...},
        # so an answer that merely quoted an object passed.
        assert not check_instruction(
            self.ID, 'Sure! Here you go: {"a": 1} — let me know.', {}
        )


class TestTwoResponses:
    ID = "combination:two_responses"

    def test_exactly_two_distinct(self):
        assert check_instruction(self.ID, "first\n******\nsecond", {})

    def test_three_responses_fail(self):
        # Regression lock: `len(parts) >= 2` accepted any number.
        assert not check_instruction(self.ID, "a\n******\nb\n******\nc", {})

    def test_identical_responses_fail(self):
        assert not check_instruction(self.ID, "same\n******\nsame", {})


class TestRepeatPrompt:
    ID = "combination:repeat_prompt"

    def test_must_start_with_the_prompt(self):
        kw = {"prompt_to_repeat": "Write a poem."}
        assert check_instruction(self.ID, "Write a poem. Roses are red...", kw)

    def test_prompt_buried_mid_response_fails(self):
        # Regression lock: a substring test made this a much easier
        # instruction than the one the prompt actually gives.
        kw = {"prompt_to_repeat": "Write a poem."}
        assert not check_instruction(self.ID, "Certainly! Write a poem. Roses...", kw)


class TestConstrainedResponse:
    ID = "detectable_format:constrained_response"

    def test_one_of_three_fixed_options(self):
        assert check_instruction(self.ID, "My answer is maybe.", {})

    def test_anything_else_fails(self):
        # Under the old registry this id was filed under `combination:`, so it
        # never matched and every instance passed through the unknown-id
        # branch as "followed".
        assert not check_instruction(self.ID, "Yes, definitely.", {})


class TestLetterFrequency:
    ID = "keywords:letter_frequency"

    def test_at_least(self):
        kw = {"letter": "a", "let_frequency": 3, "let_relation": "at least"}
        assert check_instruction(self.ID, "aardvark banana", kw)

    def test_less_than_is_honoured(self):
        # Regression lock: the relation was ignored and `>=` hardcoded, so
        # every "less than" instance was graded backwards.
        kw = {"letter": "z", "let_frequency": 2, "let_relation": "less than"}
        assert check_instruction(self.ID, "no such letter here", kw)
        kw = {"letter": "e", "let_frequency": 2, "let_relation": "less than"}
        assert not check_instruction(self.ID, "eee", kw)

    def test_unknown_relation_raises(self):
        kw = {"letter": "a", "let_frequency": 1, "let_relation": "roughly"}
        with pytest.raises(ValueError):
            check_instruction(self.ID, "aaa", kw)


class TestForbiddenWords:
    ID = "keywords:forbidden_words"

    def test_word_boundaries(self):
        # "cat" must not fire on "category" -- the old substring test made
        # this checker stricter than the benchmark.
        assert check_instruction(self.ID, "a category of things", {"forbidden_words": ["cat"]})

    def test_actual_word_fails(self):
        assert not check_instruction(self.ID, "the cat sat", {"forbidden_words": ["cat"]})


class TestParagraphs:
    def test_number_paragraphs_splits_on_markdown_divider(self):
        # The prompts ask for paragraphs separated by ***, and the reference
        # splits on that. The old blank-line split measured something else
        # entirely, then filtered out the *** dividers as "pure formatting".
        text = "one\n***\ntwo\n***\nthree"
        assert check_instruction("length_constraints:number_paragraphs", text,
                                 {"num_paragraphs": 3})
        assert not check_instruction("length_constraints:number_paragraphs",
                                     "one\n\ntwo\n\nthree", {"num_paragraphs": 3})

    def test_nth_paragraph_checks_the_count_too(self):
        # Regression lock: num_paragraphs was ignored, so only the first word
        # mattered and a response with the wrong structure still passed.
        kw = {"nth_paragraph": 2, "first_word": "second", "num_paragraphs": 2}
        assert check_instruction("length_constraints:nth_paragraph_first_word",
                                 "first para\n\nsecond para", kw)
        assert not check_instruction("length_constraints:nth_paragraph_first_word",
                                     "first para\n\nsecond para\n\nthird para", kw)

    def test_nth_paragraph_first_word_is_case_insensitive(self):
        kw = {"nth_paragraph": 1, "first_word": "Hello", "num_paragraphs": 1}
        assert check_instruction("length_constraints:nth_paragraph_first_word",
                                 "hello there", kw)

    def test_nth_paragraph_strips_punctuation_from_first_word(self):
        kw = {"nth_paragraph": 1, "first_word": "hello", "num_paragraphs": 1}
        assert check_instruction("length_constraints:nth_paragraph_first_word",
                                 '"Hello," she said.', kw)


class TestWordsAndPlaceholders:
    def test_word_count_matches_the_regexp_tokenizer(self):
        # `\w+`, which splits "don't" into two tokens -- the reference's
        # count, and not what str.split() returns.
        kw = {"num_words": 3, "relation": "at least"}
        assert check_instruction("length_constraints:number_words", "don't stop", kw)

    def test_word_count_less_than(self):
        kw = {"num_words": 3, "relation": "less than"}
        assert check_instruction("length_constraints:number_words", "two words", kw)
        assert not check_instruction("length_constraints:number_words", "one two three", kw)

    def test_placeholders(self):
        kw = {"num_placeholders": 2}
        assert check_instruction("detectable_content:number_placeholders",
                                 "[name] went to [place]", kw)
        assert not check_instruction("detectable_content:number_placeholders",
                                     "[name] only", kw)


class TestPostscript:
    ID = "detectable_content:postscript"

    def test_ps_marker_is_matched_case_insensitively(self):
        assert check_instruction(self.ID, "body\n\np.s. one more thing",
                                 {"postscript_marker": "P.S."})

    def test_pps_marker_tolerates_spacing(self):
        assert check_instruction(self.ID, "body\n\nP. P. S. and another",
                                 {"postscript_marker": "P.P.S"})

    def test_missing_postscript_fails(self):
        assert not check_instruction(self.ID, "body only", {"postscript_marker": "P.S."})


class TestTitleAndQuotation:
    def test_title_needs_content(self):
        assert check_instruction("detectable_format:title", "<<A Real Title>>", {})
        assert not check_instruction("detectable_format:title", "<<>>", {})

    def test_quotation_needs_both_ends(self):
        assert check_instruction("startend:quotation", '"wrapped"', {})
        assert not check_instruction("startend:quotation", '"unclosed', {})
        # A lone quote character is not a wrapped response.
        assert not check_instruction("startend:quotation", '"', {})

    def test_end_checker_ignores_case_and_wrapping_quotes(self):
        kw = {"end_phrase": "Any other questions?"}
        assert check_instruction("startend:end_checker",
                                 '"...and that is all. any other questions?"', kw)


class TestSections:
    ID = "detectable_format:multiple_sections"

    def test_counts_splitter_occurrences(self):
        text = "intro\nSection 1\nbody\nSection 2\nmore"
        assert check_instruction(self.ID, text,
                                 {"num_sections": 2, "section_spliter": "Section"})

    def test_case_sensitive(self):
        # The reference interpolates the splitter into the pattern without
        # IGNORECASE; matching case-insensitively counted headings the
        # instruction did not ask for.
        text = "intro\nsection 1\nbody\nsection 2\n"
        assert not check_instruction(self.ID, text,
                                     {"num_sections": 2, "section_spliter": "Section"})


class TestDependencyBackedCheckers:
    """These need langdetect / nltk, and must fail loudly without them.

    The previous implementation caught the ImportError and returned True,
    which auto-passed 95 of the dataset's 834 instructions on any machine
    where the packages were not installed -- silently, and in the model's
    favour. `sys.modules[name] = None` makes `import name` raise ImportError,
    so this holds regardless of what the test machine happens to have.
    """

    MISSING_IMPORT_IDS = [
        ("language:response_language", {"language": "en"}, "langdetect"),
        ("change_case:english_capital", {}, "langdetect"),
        ("change_case:english_lowercase", {}, "langdetect"),
        ("length_constraints:number_sentences",
         {"num_sentences": 2, "relation": "at least"}, "nltk"),
        ("change_case:capital_word_frequency",
         {"capital_frequency": 1, "capital_relation": "at least"}, "nltk"),
    ]

    @pytest.mark.parametrize("iid,kwargs,module", MISSING_IMPORT_IDS)
    def test_missing_backend_raises_instead_of_passing(
        self, iid, kwargs, module, monkeypatch
    ):
        from crucible_bench.ifeval import MissingIFEvalDependency

        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(MissingIFEvalDependency):
            check_instruction(iid, "SOME RESPONSE. ANOTHER ONE.", kwargs)

    def test_language_checker_verdicts_when_langdetect_is_present(self):
        pytest.importorskip("langdetect")
        assert check_instruction(
            "language:response_language",
            "This is an English sentence written plainly.",
            {"language": "en"},
        )
        assert not check_instruction(
            "language:response_language",
            "This is an English sentence written plainly.",
            {"language": "de"},
        )

    def test_sentence_count_verdicts_when_punkt_is_present(self):
        pytest.importorskip("nltk")
        from crucible_bench.ifeval import MissingIFEvalDependency

        kw = {"num_sentences": 2, "relation": "at least"}
        try:
            assert check_instruction("length_constraints:number_sentences",
                                     "One thing. Two things.", kw)
        except MissingIFEvalDependency:
            pytest.skip("nltk punkt data not downloaded on this machine")
