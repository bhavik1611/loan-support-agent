"""The product gate, tested in both directions.

Acceptance criteria 24a and 28 pull against each other on purpose. One says an
out-of-catalogue product must be refused before retrieval; the other says a
genuine question must never be. A curated list can drift into violating the
second while still satisfying the first, and only testing both catches that.

Neither test is tautological, because the item strings in eval/queries.py and
the phrases in rag/scope.KNOWN_ADJACENT are separate hand-authored data. They
are never generated from one another, and doing so later would hollow out both
tests without failing either.
"""

import pytest

import config
from eval import queries
from rag import generate, kb, scope


def test_a_known_adjacent_product_is_refused_before_retrieval():
    """Criterion 24a: every outside_boundary item is refused at the gate."""
    items = queries.items_of_kind(queries.KIND_OUTSIDE_BOUNDARY)
    assert len(items) == 13
    for item in items:
        verdict = scope.classify(item.text)
        assert verdict.known_adjacent, f"{item.item_id} passed the gate: {item.text}"
        assert not verdict.in_catalogue
        assert verdict.product, f"{item.item_id} was refused without naming a product"


def test_no_answerable_item_is_refused_by_the_gate():
    """Criterion 28, and the one that protects users rather than the bank."""
    for item in queries.items_of_kind(queries.KIND_ANSWERABLE):
        verdict = scope.classify(item.text)
        assert not verdict.known_adjacent, (
            f"{item.item_id} would be refused before retrieval: {item.text}"
        )


def test_the_declared_product_is_the_one_the_gate_reads():
    """The dataset's product column and the gate must not drift apart."""
    for item in queries.GOLDEN_DATASET:
        assert scope.classify(item.text).product == item.product, item.item_id


def test_a_query_naming_no_product_falls_through_unfiltered():
    verdict = scope.classify("How is the EMI on a loan calculated?")
    assert verdict.product == ""
    assert not verdict.in_catalogue
    assert not verdict.known_adjacent


def test_matching_is_case_folded_and_takes_the_longest_phrase():
    assert scope.classify("FIXED DEPOSIT rates please").product == "fixed deposit"
    assert scope.classify("what is a home loan").product == "Home Loan"
    # "stock market" is longer than "shares", so it is the phrase reported.
    assert scope.classify("Which shares trade on the stock market?").product == "stock market"


def test_a_plural_matches_but_a_shorter_word_does_not():
    """The "s" is added to the canonical phrase, never made optional on it.

    An optional trailing "s" on "shares" would also match the verb "share", and
    "Can I share my account with my wife?" is not a question about equities.
    """
    assert scope.classify("Which mutual funds do you offer?").product == "mutual fund"
    assert scope.classify("Can I share my account with my wife?").product == ""


def test_the_catalogue_wins_when_a_query_names_both():
    """A Meridian product in the query outranks an adjacent one of equal length."""
    verdict = scope.classify("Is interest on an NRE account free of income tax?")
    assert verdict.product == "NRE account"
    assert verdict.in_catalogue


def test_known_adjacent_never_collides_with_a_catalogue_product():
    """If Meridian sells it, it is in scope, and the gate must not refuse it."""
    catalogue = {p.casefold() for p in kb.catalogue_products()}
    adjacent = {p.casefold() for p in scope.KNOWN_ADJACENT}
    assert not catalogue & adjacent


def test_the_filter_narrows_to_the_documents_the_catalogue_tags():
    """D-53: the filter is a doc_id $in clause built from catalogue.json."""
    tagged = kb.documents_for_product("joint account")
    assert "kb-11-joint-account-rules" in tagged
    assert "kb-13-personal-loan-eligibility" not in tagged
    assert tagged == sorted(tagged)


def test_an_unknown_product_is_an_error_not_an_empty_filter():
    with pytest.raises(ValueError, match="fixed deposit"):
        kb.documents_for_product("fixed deposit")


def test_a_gate_refusal_carries_the_product_and_retrieves_nothing(built_index):
    """The structured refusal Part 2 reads. Per D-54 the prose is Part 2's job."""
    result = generate.answer(
        "What is the interest rate on a fixed deposit for 5 years?",
        config.STRATEGY_SENTENCES,
    )
    assert result.outcome == generate.OUTCOME_REFUSED_GATE
    assert result.product == "fixed deposit"
    assert not result.supported
    assert result.text == generate.FALLBACK_TEXT
    assert result.citations == ()
    assert result.hits == ()
    assert result.top1_similarity == 0.0


def test_a_threshold_refusal_is_a_different_outcome(built_index):
    result = generate.answer(
        "What is the best recipe for a chocolate sponge cake?", config.STRATEGY_SENTENCES
    )
    assert result.outcome == generate.OUTCOME_REFUSED_THRESHOLD
    assert result.product == ""
    assert result.hits, "a threshold refusal retrieved first, so it has hits"


def test_an_answer_reports_the_answered_outcome_and_its_product(built_index):
    result = generate.answer(
        "What minimum balance must I keep in my savings account?",
        config.STRATEGY_SENTENCES,
    )
    assert result.outcome == generate.OUTCOME_ANSWERED
    assert result.product == "savings account"
    assert "kb-09-minimum-balance" in result.citations


def test_the_filter_keeps_a_product_query_on_its_own_documents(built_index):
    """EQ-02 names savings account, and kb-06 is the document that answers it."""
    result = generate.answer(
        "What is the process to close my savings account?", config.STRATEGY_SENTENCES
    )
    assert result.product == "savings account"
    tagged = set(kb.documents_for_product("savings account"))
    assert {h.doc_id for h in result.hits} <= tagged
    assert "kb-06-account-closure" in result.citations
