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
    assert len(items) == 10
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


# Criterion 28 guards the twelve answerable items and nothing else, and not one
# of them names a tax topic. That is exactly why four bad phrases got through it:
# a curated dataset can only catch the mistakes its authors already imagined.
# These six are drawn from the corpus instead, each one a question kb-01, kb-03,
# kb-08, kb-09, kb-11, kb-12, kb-13, kb-15 or kb-18 answers, and each one was
# gate-refused at some point during fix round 1.
CORPUS_QUESTIONS_THE_GATE_MUST_NOT_REFUSE = [
    # GST: kb-03, kb-08 and kb-09 each state Meridian's treatment of it.
    "Does GST apply to the late payment fee on my credit card?",
    # The spelling the documents themselves use. The pair has to agree.
    "Is goods and services tax charged on the prepayment penalty?",
    # kb-01 and kb-04 both accept income-tax returns as proof of income. Both
    # spellings, because one spelling passing is what hid the defect for a round.
    "Do I need my income-tax returns to prove income for a loan?",
    "Do I need my income tax returns to prove income for a loan?",
    # "shares" as a verb, not as equity. kb-11 answers this at 0.3619.
    "My wife shares the account with me, can she operate it?",
    # kb-12 and kb-18 both answer this one directly.
    "Is interest on an NRE account exempt from income tax?",
]


def test_no_corpus_question_is_refused_by_the_gate():
    """Criterion 28 widened from curated items to questions the corpus answers.

    This is the test that generalises. Every phrase removed in fix round 1 was
    removed because it failed here while passing criterion 28, so a future
    addition to KNOWN_ADJACENT has to clear this before it counts as safe.
    """
    for question in CORPUS_QUESTIONS_THE_GATE_MUST_NOT_REFUSE:
        verdict = scope.classify(question)
        assert not verdict.known_adjacent, (
            f"gate refuses a question the corpus answers, on {verdict.product!r}: {question}"
        )


def test_the_two_spellings_of_a_tax_question_land_on_the_same_side():
    """One question, two spellings, one verdict. The inconsistency test.

    "GST" was in the vocabulary and "goods and services tax", the spelling the
    documents actually use, was not, so the same question was refused or answered
    depending on how the customer typed it. Same for the hyphen in income-tax.
    """
    pairs = [
        (
            "Does GST apply to the late payment fee on my credit card?",
            "Is goods and services tax charged on the prepayment penalty?",
        ),
        (
            "Do I need my income-tax returns to prove income for a loan?",
            "Do I need my income tax returns to prove income for a loan?",
        ),
    ]
    for first, second in pairs:
        assert scope.classify(first).known_adjacent == scope.classify(second).known_adjacent


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
    # "mutual fund" is longer than "gold", so it is the phrase reported.
    assert scope.classify("Which mutual fund invests in gold?").product == "mutual fund"


def test_a_plural_matches_but_a_shorter_word_does_not():
    """The "s" is added to the canonical phrase, never made optional on it.

    The plural of the phrase is the phrase; a shorter word inside it is not.
    "Can I deposit a cheque today?" is not a question about fixed deposits, and
    an optional "s" spelling would have matched the removed entry "shares"
    against the verb "share" the same way.
    """
    assert scope.classify("Which mutual funds do you offer?").product == "mutual fund"
    assert scope.classify("Do you offer fixed deposits?").product == "fixed deposit"
    assert scope.classify("Can I deposit a cheque today?").product == ""
    # The guarantee itself, asserted rather than left to the phrase that needed
    # it. "shares" was removed for colliding with the third-person verb, but the
    # pattern it needed must survive its removal: an optional trailing "s" on a
    # phrase already ending in "s" matches the shorter word too.
    assert not scope._pattern("shares").search("my wife will share the account")


# Every name in both lists, against the plural a customer would actually type.
# The table is exhaustive by assertion below, so adding a phrase to either list
# without deciding its plural fails here rather than at a customer.
#
# "golds", "ELSSs", "demats", "life insurances" and "term insurances" are forms
# nobody writes; they are recorded as what the rule produces, and a form no
# customer types matches no query, so they cost nothing. The ones that carry
# the test are the deposits, the funds, the accounts, the cards and
# "insurance policies", which is the one the "-s" rule got wrong.
NATURAL_PLURALS = {
    "fixed deposit": "fixed deposits",
    "recurring deposit": "recurring deposits",
    "mutual fund": "mutual funds",
    "SIP": "SIPs",
    "ELSS": "ELSSs",
    "demat": "demats",
    "stock market": "stock markets",
    "insurance policy": "insurance policies",
    "life insurance": "life insurances",
    "insurance cover": "insurance covers",
    "term insurance": "term insurances",
    "gold": "golds",
    "cryptocurrency": "cryptocurrencies",
    "Personal Loan": "Personal Loans",
    "Auto Loan": "Auto Loans",
    "Education Loan": "Education Loans",
    "Business Loan": "Business Loans",
    "Home Loan": "Home Loans",
    "Meridian Rewards Card": "Meridian Rewards Cards",
    "savings account": "savings accounts",
    "salary account": "salary accounts",
    "joint account": "joint accounts",
    "NRE account": "NRE accounts",
    "NRO account": "NRO accounts",
}


def test_every_name_in_both_lists_matches_its_natural_plural():
    """The "-y" to "-ies" hole, closed and held over all 24 names.

    `_pattern` built `\\b(?:insurance policy|insurance policys)\\b`, so the
    plainest form of the product question walked through the gate: "What
    insurance policies does Meridian offer?" was answered on kb_sentences at
    0.4838 out of kb-13 and kb-01, with an answer about the fixed-obligation-
    to-income ratio for a Personal Loan. English writes "policies".

    Two of the 24 names end in a consonant plus "y" and both are affected, and
    "salary account" is the reason the rule reads the end of the phrase rather
    than looking for a "y" anywhere in it.
    """
    names = list(kb.catalogue_products()) + list(scope.KNOWN_ADJACENT)
    assert sorted(NATURAL_PLURALS) == sorted(names), "the plural table has drifted"

    for name, plural in NATURAL_PLURALS.items():
        pattern = scope._pattern(name)
        assert pattern.search(name.casefold()), name
        assert pattern.search(plural.casefold()), f"{name!r} does not match {plural!r}"

    assert scope._plural("insurance policy") == "insurance policies"
    assert scope._plural("cryptocurrency") == "cryptocurrencies"
    assert scope._plural("salary account") == "salary accounts"


def test_the_plural_of_a_product_phrase_is_gated_like_the_singular():
    """The measured failure, as a verdict rather than as a regex assertion."""
    for refused in [
        "What insurance policies does Meridian offer?",
        "Does Meridian Bank sell insurance policies?",
        "Do you accept cryptocurrencies as collateral?",
    ]:
        assert scope.classify(refused).known_adjacent, refused
    assert scope.classify("What insurance policies does Meridian offer?").product == (
        "insurance policy"
    )


def test_a_query_naming_both_sides_is_decided_by_the_corpus_not_by_the_phrases():
    """The pairing rule, in both directions, with the licence doing the deciding.

    Two earlier rules decided this shape by a property unrelated to whether the
    documents cover the pair, and each was wrong on a different half.
    Longest-match-first refused "Can I open a fixed deposit in my NRE account?"
    because "fixed deposit" runs two characters longer than "NRE account".
    Catalogue-wins-unconditionally answered "Is my joint account covered by life
    insurance?" at 0.5615 out of kb-11-joint-account-rules, a survivorship
    document with nothing to say about insurance.

    kb-18 line 10 licenses the first pair and no document licenses the second,
    so that is what the gate reads now.
    """
    licensed = scope.classify("Can I open a fixed deposit in my NRE account?")
    assert licensed.product == "NRE account"
    assert licensed.in_catalogue
    assert not licensed.known_adjacent
    # The adjacent phrase is the longer one, and the licence beats the lengths.
    assert len("fixed deposit") > len("NRE account")
    assert scope.classify("Can I open a fixed deposit in my NRO account?").in_catalogue

    for unlicensed in [
        "Is my joint account covered by life insurance?",
        "Can I get a home loan against my cryptocurrency holdings?",
        "Is my Meridian Rewards Card covered by an insurance policy?",
        "Can I buy a mutual fund through my savings account?",
        "Can I open a fixed deposit as a savings account?",
    ]:
        verdict = scope.classify(unlicensed)
        assert verdict.known_adjacent, unlicensed
        assert not verdict.in_catalogue, unlicensed

    # Naming the licensed pair does not license the rest of the query: the
    # unlicensed phrase is still standing, so the refusal names it.
    both = scope.classify("Can I open a fixed deposit or buy gold in my NRE account?")
    assert both.product == "gold"
    assert both.known_adjacent


def test_every_licensed_pair_is_a_sentence_the_corpus_actually_carries():
    """The licence cannot drift from the corpus the way the tags could.

    catalogue.json's `licensed_pairs` is the one thing that lets an adjacent
    product phrase stand beside a Meridian product without refusing, so it is
    the widest the gate ever opens. Each entry quotes the sentence that licenses
    it, and this reads the document to check that the sentence is really there
    and really names both terms. Delete the line from kb-18 and this fails.
    """
    bodies = {document.doc_id: document.body for document in kb.load_documents()}
    catalogue = kb.catalogue_products()
    pairs = kb.licensed_pairs()
    assert pairs, "the licence is not empty today; kb-18 line 10 carries one pairing"

    for pair in pairs:
        assert pair.product in catalogue, pair
        assert pair.adjacent in scope.KNOWN_ADJACENT, pair
        body = bodies[pair.document]
        assert pair.sentence in body, (
            f"{pair.document} does not carry the sentence licensing "
            f"{pair.product!r} with {pair.adjacent!r}"
        )
        folded = pair.sentence.casefold()
        assert scope._pattern(pair.product).search(folded), pair
        assert scope._pattern(pair.corpus_phrase).search(folded), pair
        # The document that licenses the pairing has to be reachable under the
        # filter the licensed verdict then applies, or the licence buys nothing.
        assert pair.document in kb.documents_for_product(pair.product), pair


def test_a_query_naming_only_an_adjacent_product_is_still_refused():
    """The other half of the tie-break: no catalogue product, no reprieve.

    OB-01 names "fixed deposit" and nothing Meridian sells, so it refuses. That
    is why "fixed deposit" stays in the list rather than being deleted once
    kb-18 turned out to answer the NRE question: without it, OB-01 is answered
    at 0.5694 out of kb-07-interest-rate-slabs, a document that carries loan
    rates and says nothing about deposit rates.
    """
    verdict = scope.classify("What is the interest rate on a fixed deposit for 5 years?")
    assert verdict.product == "fixed deposit"
    assert verdict.known_adjacent


def test_no_outside_boundary_item_names_a_catalogue_product():
    """The dataset's blind spot, recorded so nobody measures against it again.

    Not one golden item names a catalogue product and an adjacent one in the
    same sentence, which is why a tie-break change between those two lists was
    confirmed green against the whole dataset while turning
    "Is my joint account covered by life insurance?" into a cited answer out of
    kb-11. The probes in
    test_a_query_naming_both_sides_is_decided_by_the_corpus_not_by_the_phrases
    cover the shape instead; this test states the gap rather than filling it.
    """
    catalogue = kb.catalogue_products()
    for item in queries.items_of_kind(queries.KIND_OUTSIDE_BOUNDARY):
        folded = item.text.casefold()
        named = [p for p in catalogue if scope._pattern(p).search(folded)]
        assert not named, f"{item.item_id} names {named}, so the gate will pass it"


def test_the_industry_noun_passes_and_the_product_phrases_refuse():
    """Fix round 2: "insurance" alone names an industry, not a Meridian product.

    In a banking question the bare noun usually names a merchant, a salary
    deduction or a document. "An insurance company debited my card twice"
    answers from kb-05 at 0.4221 with its chunks agreeing, and the gate refused
    it along with seven other probes the corpus answers. Deleting the noun
    outright was measured and rejected too: with no insurance phrase at all,
    OB-08 is answered at 0.3685 out of kb-06-account-closure.
    """
    for passes in [
        "An insurance company debited my card twice without my authorisation, "
        "how do I dispute it?",
        "My employer deducts an insurance premium from my salary, does that count as income?",
        "Which insurance documents does the bank accept as address proof?",
    ]:
        assert not scope.classify(passes).known_adjacent, passes

    for refuses in [
        "Does the bank sell term life insurance cover?",
        "What does a car insurance policy cost here?",
        "Do you sell life insurance?",
        "Is term insurance available through the app?",
    ]:
        assert scope.classify(refuses).known_adjacent, refuses


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


def test_the_filter_never_hides_the_document_that_answers_a_kyc_question(built_index):
    """D-53's filter is only safe while catalogue.json tags cover the document.

    kb-16 was tagged with the five loans, the card and "savings account" only,
    although its own line 10 says Meridian applies the same re-verification
    cycle across savings, loan and card accounts, and kb-12 line 12 puts every
    non-resident account under the same KYC standards. So "What happens if KYC
    is overdue on my joint account?" filtered to documents that do not answer
    it and was **answered** at 0.4853 citing kb-04 and kb-11 - the worst failure
    mode of the pair, because the support rule cannot catch it: the two wrong
    documents agree with each other. Unfiltered the same query reads 0.6272 with
    kb-16 in all three slots.
    """
    tagged = kb.documents_for_product("joint account")
    assert "kb-16-kyc-reverification" in tagged

    result = generate.answer(
        "What happens if KYC is overdue on my joint account?", config.STRATEGY_SENTENCES
    )
    assert result.product == "joint account"
    assert result.citations == ("kb-16-kyc-reverification",)


def test_a_document_is_tagged_with_every_catalogue_product_its_body_names():
    """The corpus is the authority over catalogue.json, not the other way round.

    A document that names a Meridian product in its own prose covers it, so the
    filter must not hide the document from a question about it. This catches
    the under-tagging shape mechanically; a claim written in generic terms
    ("savings accounts, loan accounts and card accounts") still needs a reader,
    which is how kb-16 escaped for a round.
    """
    catalogue = kb.catalogue_products()
    for document in kb.load_documents():
        folded = document.body.casefold()
        for product in catalogue:
            if scope._pattern(product).search(folded):
                assert document.doc_id in kb.documents_for_product(product), (
                    f"{document.doc_id} names {product!r} and is not tagged with it"
                )


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
