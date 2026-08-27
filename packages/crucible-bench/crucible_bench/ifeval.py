"""IFEval verification functions.

A port of the reference implementation at
github.com/google-research/google-research/tree/master/instruction_following_eval
(`instructions.py`, `instructions_util.py`, `instructions_registry.py`),
checker by checker. The 25 instruction ids here are exactly the 25 the
reference registry enables — `multi-turn:constrained_start` and
`keywords:key_sentences` are commented out upstream and absent from the
dataset, so they are absent here too.

Deliberate deviations from the reference, all of them in the direction of
noticing problems rather than absorbing them:

  - An unrecognized instruction id raises `UnknownInstruction` instead of
    being counted as followed. The reference would KeyError; an earlier
    version of this file returned True, which silently handed the model a
    free pass on every id we had misspelled.
  - A `relation` kwarg outside {"less than", "at least"} raises rather than
    falling through to a falsy None.
  - langdetect failing to *import* is a hard error. The reference only
    swallows `LangDetectException` (text too short or ambiguous to
    classify), and so do we; a missing dependency is not a detection
    failure and must not be scored as one.

We report strict accuracy only. The reference additionally reports a "loose"
variant that retries each check against markdown-stripped and
first/last-line-dropped versions of the response; strict is the harder and
less flattering of the two.
"""

from __future__ import annotations

import collections
import json
import re

# The reference's _COMPARISON_RELATION, positionally.
_LESS_THAN = "less than"
_AT_LEAST = "at least"
_RELATIONS = (_LESS_THAN, _AT_LEAST)

_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


class UnknownInstruction(KeyError):
    """The dataset asked for an instruction id this module cannot check.

    Raised rather than skipped: an instruction we cannot verify is a hole in
    the measurement, and a hole that scores as "followed" inflates the result.
    """


class MissingIFEvalDependency(RuntimeError):
    """A checker needs a package that is not installed.

    IFEval cannot be scored honestly without them -- three instruction types
    need language detection and two need NLTK's tokenizers. Install with
    `uv pip install -e '.[ifeval]'` (plus `python -m nltk.downloader punkt
    punkt_tab` for the sentence tokenizer).
    """


# ---------------------------------------------------------------------------
# Tokenization helpers (instructions_util)
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    """Reference `count_words`: nltk RegexpTokenizer(r"\\w+"), which for this
    pattern is exactly `re.findall`, so we skip the dependency."""
    return len(re.findall(r"\w+", text))


def _nltk():
    try:
        import nltk
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise MissingIFEvalDependency(
            "nltk is required for length_constraints:number_sentences and "
            "change_case:capital_word_frequency. Install the ifeval extra: "
            "uv pip install -e '.[ifeval]'"
        ) from e
    return nltk


def _count_sentences(text: str) -> int:
    """Reference `count_sentences`: NLTK's punkt sentence tokenizer."""
    nltk = _nltk()
    try:
        tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    except LookupError as e:
        raise MissingIFEvalDependency(
            "NLTK's punkt tokenizer data is not downloaded. Run: "
            "python -m nltk.downloader punkt punkt_tab"
        ) from e
    return len(tokenizer.tokenize(text))


def _word_tokenize(text: str) -> list[str]:
    nltk = _nltk()
    try:
        return nltk.word_tokenize(text)
    except LookupError as e:
        raise MissingIFEvalDependency(
            "NLTK's punkt tokenizer data is not downloaded. Run: "
            "python -m nltk.downloader punkt punkt_tab"
        ) from e


def _detect_language(text: str) -> str | None:
    """Reference behaviour: detect, and treat a LangDetectException as
    "instruction followed" (returns None here, which each caller maps to True).

    A missing langdetect install is *not* a detection failure and raises.
    """
    try:
        import langdetect
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise MissingIFEvalDependency(
            "langdetect is required for language:response_language, "
            "change_case:english_capital and change_case:english_lowercase. "
            "Install the ifeval extra: uv pip install -e '.[ifeval]'"
        ) from e
    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        return None


def _relation(kwargs: dict, key: str = "relation") -> str:
    rel = kwargs.get(key)
    if rel not in _RELATIONS:
        raise ValueError(
            f"IFEval kwarg {key!r} was {rel!r}; expected one of {_RELATIONS}. "
            f"The dataset changed shape -- do not score this run."
        )
    return rel


def _compare(count: int, threshold: int, relation: str) -> bool:
    return count < threshold if relation == _LESS_THAN else count >= threshold


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def check_instruction(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Check whether a response follows one instruction.

    Raises:
        UnknownInstruction: the id has no checker. Never silently passes.
    """
    checker = INSTRUCTION_CHECKERS.get(instruction_id)
    if checker is None:
        raise UnknownInstruction(
            f"no checker for instruction id {instruction_id!r}; "
            f"scoring it either way would be a guess"
        )
    return checker(response, kwargs)


def check_all_instructions(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
) -> tuple[bool, list[bool]]:
    """Check every instruction attached to one prompt.

    Returns:
        (all_passed, per_instruction_results)
    """
    results = [
        check_instruction(iid, response, kw)
        for iid, kw in zip(instruction_ids, kwargs_list)
    ]
    return all(results), results


def unsupported_instruction_ids(instruction_ids) -> list[str]:
    """Which of these ids have no checker. Empty means full coverage.

    Callers use this to refuse a run up front rather than discover the gap an
    hour in -- or, worse, not discover it at all.
    """
    return sorted({i for i in instruction_ids if i not in INSTRUCTION_CHECKERS})


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


def _check_number_words(response: str, kwargs: dict) -> bool:
    return _compare(
        _count_words(response), kwargs["num_words"], _relation(kwargs)
    )


def _check_number_sentences(response: str, kwargs: dict) -> bool:
    return _compare(
        _count_sentences(response), kwargs["num_sentences"], _relation(kwargs)
    )


def _check_number_paragraphs(response: str, kwargs: dict) -> bool:
    """Paragraphs are separated by the markdown divider `***`, not by blank
    lines -- that is what the prompts ask for and what the reference splits on.
    An empty chunk is tolerated only at the very start or end.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    num_paragraphs = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                num_paragraphs -= 1
            else:
                return False
    return num_paragraphs == kwargs["num_paragraphs"]


def _check_paragraph_first_word(response: str, kwargs: dict) -> bool:
    """This one splits on blank lines (unlike number_paragraphs) and checks
    the paragraph *count* as well as the nth paragraph's first word.
    """
    paragraphs = re.split(r"\n\n", response)
    num_paragraphs = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            num_paragraphs -= 1

    nth = kwargs["nth_paragraph"]
    if nth > num_paragraphs:
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False

    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first_word = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        first_word += letter.lower()

    # The reference lowercases the expected word when it builds the
    # instruction, so the comparison is case-insensitive on both sides. Every
    # value in the current dataset is already lowercase; this keeps a
    # differently-cased one from failing all 12 instances silently.
    expected_first = kwargs["first_word"].lower()
    return num_paragraphs == kwargs["num_paragraphs"] and first_word == expected_first


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------


def _check_keyword_existence(response: str, kwargs: dict) -> bool:
    # Keywords are treated as regexes by the reference, not literals.
    return all(
        re.search(kw, response, flags=re.IGNORECASE) for kw in kwargs["keywords"]
    )


def _check_keyword_frequency(response: str, kwargs: dict) -> bool:
    occurrences = len(re.findall(kwargs["keyword"], response, flags=re.IGNORECASE))
    return _compare(occurrences, kwargs["frequency"], _relation(kwargs))


def _check_forbidden_words(response: str, kwargs: dict) -> bool:
    # Word-boundary matching: "cat" must not fail a response containing
    # "category". A plain substring test made this checker stricter than the
    # benchmark and cost the model points it had earned.
    return not any(
        re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE)
        for word in kwargs["forbidden_words"]
    )


def _check_letter_frequency(response: str, kwargs: dict) -> bool:
    letters = collections.Counter(response.lower())
    return _compare(
        letters[kwargs["letter"].lower()],
        kwargs["let_frequency"],
        _relation(kwargs, "let_relation"),
    )


# ---------------------------------------------------------------------------
# Language / case
# ---------------------------------------------------------------------------


def _check_language(response: str, kwargs: dict) -> bool:
    # kwargs["language"] is an ISO 639-1 code ("kn", "pa", "vi"), already in
    # langdetect's vocabulary -- no name-to-code mapping needed.
    detected = _detect_language(response)
    return True if detected is None else detected == kwargs["language"]


def _check_lowercase(response: str, kwargs: dict) -> bool:
    detected = _detect_language(response)
    return True if detected is None else (response.islower() and detected == "en")


def _check_uppercase(response: str, kwargs: dict) -> bool:
    detected = _detect_language(response)
    return True if detected is None else (response.isupper() and detected == "en")


def _check_capital_word_frequency(response: str, kwargs: dict) -> bool:
    capital_words = sum(1 for w in _word_tokenize(response) if w.isupper())
    return _compare(
        capital_words, kwargs["capital_frequency"], _relation(kwargs, "capital_relation")
    )


# ---------------------------------------------------------------------------
# Punctuation
# ---------------------------------------------------------------------------


def _check_no_comma(response: str, kwargs: dict) -> bool:
    return not re.search(r"\,", response)


# ---------------------------------------------------------------------------
# Detectable format
# ---------------------------------------------------------------------------


def _check_json_format(response: str, kwargs: dict) -> bool:
    """The *whole* response must parse, once a wrapping code fence is removed.

    Searching the response for any embedded {...} — which this used to do —
    passes a model that answered in prose and happened to quote an object.
    """
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _check_number_bullet_lists(response: str, kwargs: dict) -> bool:
    # Exactly, not at least: the prompts ask for a specific number of bullets
    # and over-delivering is not compliance.
    bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets) + len(dashes) == kwargs["num_bullets"]


def _check_number_highlighted_sections(response: str, kwargs: dict) -> bool:
    """Highlights are *emphasis* and **strong**, and nothing else.

    Markdown headers used to count here, which handed free credit to any
    model that structured its answer with `#` -- on the most common
    instruction id in the set.
    """
    num_highlights = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", response):
        if highlight.strip("*").strip():
            num_highlights += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if highlight.removeprefix("**").removesuffix("**").strip():
            num_highlights += 1
    return num_highlights >= kwargs["num_highlights"]


def _check_multiple_sections(response: str, kwargs: dict) -> bool:
    # `section_spliter` is the dataset's spelling, and the reference
    # interpolates it into the pattern unescaped. Case-sensitive.
    pattern = r"\s?" + kwargs["section_spliter"] + r"\s?\d+\s?"
    return len(re.split(pattern, response)) - 1 >= kwargs["num_sections"]


def _check_title(response: str, kwargs: dict) -> bool:
    for title in re.findall(r"<<[^\n]+>>", response):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def _check_constrained_response(response: str, kwargs: dict) -> bool:
    """One of three fixed sentences.

    Registered under `detectable_format:` -- the id this was filed under
    (`combination:`) does not exist, so all 10 instances were reaching the
    unknown-id branch and passing for free.
    """
    value = response.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSE_OPTIONS)


# ---------------------------------------------------------------------------
# Detectable content
# ---------------------------------------------------------------------------


def _check_postscript(response: str, kwargs: dict) -> bool:
    marker = kwargs["postscript_marker"]
    value = response.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _check_number_placeholders(response: str, kwargs: dict) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= kwargs["num_placeholders"]


# ---------------------------------------------------------------------------
# Start / end
# ---------------------------------------------------------------------------


def _check_quotation(response: str, kwargs: dict) -> bool:
    value = response.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


def _check_end_checker(response: str, kwargs: dict) -> bool:
    value = response.strip().strip('"').lower()
    return value.endswith(kwargs["end_phrase"].strip().lower())


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def _check_repeat_prompt(response: str, kwargs: dict) -> bool:
    # Must *start with* the prompt. Finding it anywhere in the response, which
    # is what this used to accept, is a different and much easier instruction.
    return response.strip().lower().startswith(kwargs["prompt_to_repeat"].strip().lower())


def _check_two_responses(response: str, kwargs: dict) -> bool:
    # Exactly two, and they must differ.
    valid: list[str] = []
    responses = response.split("******")
    for index, part in enumerate(responses):
        if not part.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INSTRUCTION_CHECKERS = {
    # Keywords
    "keywords:existence": _check_keyword_existence,
    "keywords:frequency": _check_keyword_frequency,
    "keywords:forbidden_words": _check_forbidden_words,
    "keywords:letter_frequency": _check_letter_frequency,
    # Language
    "language:response_language": _check_language,
    # Length constraints
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:nth_paragraph_first_word": _check_paragraph_first_word,
    # Detectable content
    "detectable_content:number_placeholders": _check_number_placeholders,
    "detectable_content:postscript": _check_postscript,
    # Detectable format
    "detectable_format:number_bullet_lists": _check_number_bullet_lists,
    "detectable_format:constrained_response": _check_constrained_response,
    "detectable_format:number_highlighted_sections": _check_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "detectable_format:json_format": _check_json_format,
    "detectable_format:title": _check_title,
    # Combination
    "combination:two_responses": _check_two_responses,
    "combination:repeat_prompt": _check_repeat_prompt,
    # Start / end
    "startend:end_checker": _check_end_checker,
    "startend:quotation": _check_quotation,
    # Case
    "change_case:capital_word_frequency": _check_capital_word_frequency,
    "change_case:english_capital": _check_uppercase,
    "change_case:english_lowercase": _check_lowercase,
    # Punctuation
    "punctuation:no_comma": _check_no_comma,
}
