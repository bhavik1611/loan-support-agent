"""Task 11. The FastAPI deployment."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_36_ask_answers_a_policy_question_through_the_api():
    """Spec section 16 test 36, Part 3 criterion 1."""
    response = client.post("/ask", json={"query": "How is the EMI on a loan calculated?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "policy"
    assert body["answer"]
    assert body["policy"]["citations"]


def test_ask_routes_a_lookup_question_to_the_record_tool():
    response = client.post("/ask", json={"query": "What is the status of LN-1042?"})
    assert response.status_code == 200
    assert response.json()["route"] == "lookup"


def test_ask_defaults_the_thread_id():
    response = client.post("/ask", json={"query": "How is the EMI calculated?"})
    assert response.json()["thread_id"] == "default"


def test_ask_rejects_an_empty_query_with_422():
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_the_response_matches_the_committed_schema():
    """The envelope is AgentResponse unchanged, per D-66. Nothing is widened."""
    import json

    import config

    schema = json.loads(config.RESPONSE_SCHEMA_PATH.read_text())
    body = client.post("/ask", json={"query": "How is the EMI calculated?"}).json()
    import jsonschema

    jsonschema.validate(body, schema)


def _clear_uploads():
    import shutil

    import config

    if config.UPLOAD_DIR.exists():
        shutil.rmtree(config.UPLOAD_DIR)


def test_add_document_makes_a_previously_refused_question_answerable():
    """The loop the endpoint exists to demonstrate: refuse, add, answer.

    The query names no catalogue product on purpose. D-53 narrows retrieval
    to catalogue.json's doc ids whenever a query names a catalogue product,
    and an upload is never in the catalogue (D-72 forbids writing there), so
    a product-naming query can never reach an upload - see the boundary test
    below and the comment on add_document_endpoint. "How long does a home
    loan take to disburse after approval?" names Home Loan and is refused at
    top-1 0.4534 today, but adding this same document never moves that
    number, unfiltered or not. This query is the one candidate that is both
    unfiltered and genuinely refused beforehand: refused_threshold at top-1
    0.3820, measured 2026-09-13.
    """
    _clear_uploads()
    query = "How long does disbursement take after approval?"

    before = client.post("/ask", json={"query": query, "thread_id": "up-a"}).json()
    assert before["policy"]["outcome"].startswith("refused")
    assert before["policy"]["top1_similarity"] == 0.382

    added = client.post("/add-document", json={
        "title": "Home loan disbursal timeline",
        "body": (
            "A sanctioned home loan is disbursed within seven working days of "
            "formal approval. Disbursement for a ready property is completed "
            "within seven working days of approval, released in one tranche. "
            "Disbursement for a property under construction is completed "
            "within seven working days of each approval milestone, released "
            "in stages. The seven working day disbursement clock starts on "
            "the date of formal loan approval."
        ),
        "products": ["Home Loan"],
    })
    assert added.status_code == 200
    assert added.json()["doc_id"].startswith("kb-up-")
    assert added.json()["chunks_added"] >= 2

    after = client.post("/ask", json={"query": query, "thread_id": "up-b"}).json()
    assert after["policy"]["outcome"] == "answered"
    assert after["policy"]["top1_similarity"] == 0.7569
    assert any(c.startswith("kb-up-") for c in after["policy"]["citations"])


def test_add_document_is_invisible_to_a_query_naming_its_catalogue_product():
    """The known boundary D-72 and D-53 leave behind, pinned rather than left as prose.

    D-53 narrows retrieval to the doc ids catalogue.json tags with a named
    product; an upload is never in the catalogue, so it is unreachable from a
    query naming one, even though the identical question with no product
    named reaches it. Fixing this means either writing into knowledge_base/,
    which D-72 forbids, or teaching the product filter about uploads, which
    is a V2 change to rag/kb.py and rag/retrieve.py, out of this task's
    scope. This test fails loudly if the filter ever changes to close the gap.
    """
    _clear_uploads()
    added = client.post("/add-document", json={
        "title": "Home loan disbursal timeline",
        "body": (
            "A sanctioned home loan is disbursed within seven working days of "
            "formal approval. Disbursement for a ready property is completed "
            "within seven working days of approval, released in one tranche. "
            "Disbursement for a property under construction is completed "
            "within seven working days of each approval milestone, released "
            "in stages. The seven working day disbursement clock starts on "
            "the date of formal loan approval."
        ),
        "products": ["Home Loan"],
    })
    assert added.status_code == 200

    named_product = client.post("/ask", json={
        "query": "How long does a home loan take to disburse after approval?",
        "thread_id": "boundary-named",
    }).json()
    assert not any(c.startswith("kb-up-") for c in named_product["policy"]["citations"])

    no_product_named = client.post("/ask", json={
        "query": "How long does disbursement take after approval?",
        "thread_id": "boundary-unnamed",
    }).json()
    assert any(c.startswith("kb-up-") for c in no_product_named["policy"]["citations"])


def test_add_document_rejects_a_product_outside_the_catalogue():
    """A document nothing can reach is worse than no document.

    rag/scope.py refuses any question naming an out-of-catalogue product
    before retrieval runs, so the document would be permanently unreachable.
    """
    response = client.post("/add-document", json={
        "title": "Fixed deposit early closure",
        "body": "A fixed deposit closed early earns a reduced rate. " * 3,
        "products": ["Fixed Deposit"],
    })
    assert response.status_code == 422


def test_40_add_document_never_writes_into_the_knowledge_base():
    """Spec section 16 test 40, and the whole of D-72."""
    import hashlib

    import config

    def fingerprint() -> str:
        digest = hashlib.sha256()
        for path in sorted(config.KB_DIR.iterdir()):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    before = fingerprint()
    client.post("/add-document", json={
        "title": "Auto loan top up",
        "body": "An existing auto loan may be topped up after twelve paid EMIs. " * 3,
        "products": ["Auto Loan"],
    })
    assert fingerprint() == before


def test_a_rebuild_drops_every_upload():
    """The rollback D-72 relies on, which is behaviour build_index already has."""
    import config
    from rag import index

    client.post("/add-document", json={
        "title": "Education loan moratorium",
        "body": "An education loan carries a moratorium until course completion. " * 3,
        "products": ["Education Loan"],
    })
    index.build_index(rebuild=True)
    collection = index.get_collection(config.UPLOAD_STRATEGY)
    stored = collection.get(include=["metadatas"])
    assert not [
        m for m in stored["metadatas"]
        if m["doc_id"].startswith(config.UPLOAD_DOC_PREFIX)
    ]
    _clear_uploads()
