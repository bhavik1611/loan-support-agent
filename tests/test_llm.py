"""The mock provider must be deterministic and strictly grounded."""

import pytest

import llm

PROMPT = """QUESTION: What annual fee does the credit card carry?

CONTEXT:
[kb-03-credit-card-fees] The Meridian Rewards Card carries an annual fee of 999 rupees. The fee is waived when annual spend crosses two lakh rupees.
[kb-03-credit-card-fees] Goods and services tax applies on top of every fee that the card levies.
[kb-15-card-late-payment-charges] A late-payment fee applies when the minimum amount due is not paid by the due date.
"""


def test_parse_prompt_recovers_the_question_and_the_sources():
    question, sources = llm.parse_prompt(PROMPT)
    assert question == "What annual fee does the credit card carry?"
    assert len(sources) == 3
    assert sources[0][0] == "kb-03-credit-card-fees"
    assert "999 rupees" in sources[0][1]


@pytest.mark.parametrize(
    "question,shape",
    [
        ("Which documents do I need to submit for KYC?", "documents"),
        ("What annual fee does the credit card carry?", "amount_or_rate"),
        ("Am I eligible for a business loan?", "eligibility"),
        ("How do I close my savings account?", "process"),
        ("How long does a refund take?", "duration"),
        ("What is an NRE account?", "definition"),
    ],
)
def test_shape_classification_is_stable(question, shape):
    assert llm.classify_shape(question) == shape


def test_the_mock_is_deterministic():
    assert llm.generate("system", PROMPT) == llm.generate("system", PROMPT)


def test_every_sentence_in_the_answer_came_from_the_context():
    """The whole groundedness guarantee under MOCK_LLM: strip the template's own
    prefix, and every remaining sentence must be a substring of the context."""
    question, sources = llm.parse_prompt(PROMPT)
    answer = llm.generate("system", PROMPT)
    body = answer.split("Sources:")[0]
    prefix = llm._TEMPLATES[llm.classify_shape(question)].split("{body}")[0]
    assert body.startswith(prefix)
    quoted = body[len(prefix):].strip()
    context = " ".join(text for _, text in sources)
    for sentence in llm._SENTENCE.split(quoted):
        sentence = sentence.strip()
        if sentence:
            assert sentence in context, f"ungrounded sentence: {sentence!r}"


def test_the_answer_cites_only_documents_it_used():
    answer = llm.generate("system", PROMPT)
    assert "Sources:" in answer
    cited = answer.split("Sources:")[1]
    assert "[kb-03-credit-card-fees]" in cited
    for doc_id in cited.replace("[", " ").replace("]", " ").split():
        assert doc_id.startswith("kb-")


def test_an_empty_context_yields_no_answer():
    assert llm.generate("system", "QUESTION: anything\n\nCONTEXT:\n") == ""


def test_an_unwired_provider_raises_rather_than_pretending(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(NotImplementedError, match="LLM_PROVIDER"):
        llm.generate("system", PROMPT)
