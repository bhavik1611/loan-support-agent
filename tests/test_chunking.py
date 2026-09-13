"""Both chunkers are pure functions and must behave predictably at the edges."""

import pytest

import config
from rag import chunking


def test_fixed_respects_size_and_advances_by_size_minus_overlap():
    text = "x" * 1000
    chunks = chunking.chunk_fixed(text, size=400, overlap=80)
    assert all(len(c) <= 400 for c in chunks)
    # step 320: windows start at 0, 320 and 640, and the third reaches the end
    assert len(chunks) == 3


def test_fixed_chunks_overlap_by_the_declared_amount():
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = chunking.chunk_fixed(text, size=400, overlap=80)
    assert chunks[0][-80:] == chunks[1][:80]


def test_fixed_returns_one_chunk_for_short_text():
    assert chunking.chunk_fixed("A short policy sentence.", size=400, overlap=80) == [
        "A short policy sentence."
    ]


def test_fixed_rejects_overlap_at_or_above_size():
    with pytest.raises(ValueError):
        chunking.chunk_fixed("abc", size=100, overlap=100)


def test_fixed_and_sentences_ignore_empty_input():
    assert chunking.chunk_fixed("   ") == []
    assert chunking.chunk_sentences("   ") == []


def test_sentences_splits_on_terminal_punctuation():
    text = "First sentence here. Second one follows! And a third? Yes."
    assert chunking.chunk_sentences(text) == [
        "First sentence here.",
        "Second one follows!",
        "And a third?",
        "Yes.",
    ]


def test_sentences_does_not_split_inside_a_rupee_abbreviation():
    text = "The fee is Rs. 5,000 per year. It is waived above a stated spend."
    chunks = chunking.chunk_sentences(text)
    assert len(chunks) == 2
    assert chunks[0] == "The fee is Rs. 5,000 per year."


def test_sentences_does_not_split_on_common_abbreviations():
    for text in [
        "Contact Mr. Rao for details. He handles disputes.",
        "Submit PAN, passport, etc. before the deadline. A copy suffices.",
        "Use a utility bill, i.e. electricity or water. Both are accepted.",
    ]:
        assert len(chunking.chunk_sentences(text)) == 2, text


def test_sentences_does_not_split_on_a_decimal_number():
    text = "The rate is 10.5 percent per annum. It resets quarterly."
    assert len(chunking.chunk_sentences(text)) == 2


def test_sentences_handles_a_newline_between_sentences():
    text = "One sentence.\nTwo sentence.\nThree sentence."
    assert chunking.chunk_sentences(text) == [
        "One sentence.",
        "Two sentence.",
        "Three sentence.",
    ]


def test_dispatch_matches_the_direct_calls():
    text = "One sentence. Two sentence. " * 40
    assert chunking.chunk(text, config.STRATEGY_FIXED) == chunking.chunk_fixed(text)
    assert chunking.chunk(text, config.STRATEGY_SENTENCES) == chunking.chunk_sentences(text)
    with pytest.raises(ValueError):
        chunking.chunk(text, "semantic")
