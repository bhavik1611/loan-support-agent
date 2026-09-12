"""Grounded generation answers in scope and refuses out of scope."""

import config
from rag import generate


def test_an_in_scope_query_is_answered_and_cited(built_index):
    result = generate.answer("How is the EMI on a loan calculated?", config.STRATEGY_SENTENCES)
    assert result.supported
    assert result.text != generate.FALLBACK_TEXT
    assert "kb-02-emi-calculation" in result.citations
    assert result.top1_similarity >= config.SIMILARITY_THRESHOLD


def test_an_out_of_scope_query_triggers_the_fallback(built_index):
    result = generate.answer(
        "What is the best recipe for a chocolate sponge cake?", config.STRATEGY_SENTENCES
    )
    assert not result.supported
    assert result.text == generate.FALLBACK_TEXT
    assert result.citations == ()


def test_the_fallback_never_cites_anything(built_index):
    for query in [
        "Which team won the football World Cup in 2018?",
        "How do I replace the timing belt on a diesel engine?",
    ]:
        result = generate.answer(query, config.STRATEGY_FIXED)
        assert result.text == generate.FALLBACK_TEXT
        assert result.citations == ()


def test_every_citation_names_a_retrieved_document(built_index):
    result = generate.answer("What annual fee does the credit card carry?", config.STRATEGY_FIXED)
    retrieved = {h.doc_id for h in result.hits}
    assert set(result.citations) <= retrieved


def test_both_strategies_answer_every_evaluation_query(built_index):
    """All twelve labelled queries clear both conditions in both collections."""
    from eval.queries import EVAL_QUERIES

    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        for query in EVAL_QUERIES:
            result = generate.answer(query.text, strategy)
            assert result.supported, f"{strategy}/{query.item_id}"
            assert result.strategy == strategy


def test_a_query_spread_across_three_parents_is_refused(built_index):
    """The support rule's false-refusal mode, recorded rather than hidden.

    "What documents are needed for KYC?" is in scope and scores 0.5124 top-1,
    far above the threshold. Under sentence chunking the top three chunks land
    on kb-04, kb-12 and kb-16, three different parents, so no two agree and the
    rule refuses. That is D-07 working as specified: the knowledge base is asked
    to agree with itself before the system speaks, and here it does not. The
    cost is a refusal on a broad question that spans documents, which is the
    price of the six deliberately confusable neighbours. Section 13 of the spec
    lists the two-signal fallback as the V2 upgrade that removes it.
    """
    result = generate.answer(
        "What documents are needed for KYC?", config.STRATEGY_SENTENCES
    )
    assert result.top1_similarity > config.SIMILARITY_THRESHOLD
    assert len({h.doc_id for h in result.hits}) == 3
    assert not result.supported
    assert result.text == generate.FALLBACK_TEXT


def test_the_prompt_matches_the_frozen_contract(built_index):
    from rag import retrieve
    import llm

    hits = retrieve.retrieve("How do I close my account?", config.STRATEGY_SENTENCES)
    system, user = generate.build_prompt("How do I close my account?", hits)
    assert user.startswith("QUESTION: How do I close my account?")
    assert "\nCONTEXT:\n" in user
    question, sources = llm.parse_prompt(user)
    assert question == "How do I close my account?"
    assert len(sources) == len(hits)
    assert "\n" not in "".join(text for _, text in sources)


def test_generation_is_deterministic(built_index):
    first = generate.answer("What is the minimum balance requirement?", config.STRATEGY_SENTENCES)
    second = generate.answer("What is the minimum balance requirement?", config.STRATEGY_SENTENCES)
    assert first.text == second.text
    assert first.citations == second.citations
