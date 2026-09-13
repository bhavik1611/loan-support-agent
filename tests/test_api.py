"""Task 11. The FastAPI deployment."""

import pytest
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


@pytest.fixture
def clean_uploads():
    """Every test that writes an upload gets the index back exactly as it found it.

    A yield fixture, not test ordering: no test may rely on some *other* test's
    cleanup to leave the shared, persistent kb_sentences collection at its
    215-chunk baseline. Before this existed, only test_a_rebuild_drops_every_upload
    cleaned up, and three other tests wrote real chunks and relied on file
    collection order to reach it last - `pytest tests/test_api.py -k add_document`
    deselects that test (its name has no "add_document" substring) and left the
    index at 222 chunks with kb-up-01/02 residue on disk, measured 2026-09-13.
    """
    _clear_uploads()
    yield
    from rag import index

    index.build_index(rebuild=True)
    _clear_uploads()


def test_add_document_makes_a_previously_refused_question_answerable(clean_uploads):
    """The loop the endpoint exists to demonstrate: refuse, add, answer.

    The query names its catalogue product on purpose, and succeeds anyway:
    rag/retrieve.py's product narrowing (D-53) is now the catalogue's doc ids
    union whatever POST /add-document has written to data/uploads/, so an
    upload is reachable from exactly the query a user is most likely to ask -
    the one naming the product the upload is about. Before that change this
    demo had to use a product-free query to reach the upload at all; that
    boundary is gone (see test_add_document_is_visible_to_a_query_naming_its_catalogue_product)
    and this is the better demo because of it.

    "How long does a home loan take to disburse after approval?" names Home
    Loan and is refused today at top-1 0.4534; adding this document raises it
    to top-1 0.8702 and answers, measured 2026-09-13. D-13 leaves
    Precision@3/Recall@3-adjacent numbers like these unpinned on purpose -
    they move legitimately when chunk parameters are tuned - so the
    assertions below are relational (refused before, answered after, on a
    strictly higher score) and the measured values live in this docstring as
    recorded evidence rather than as a test that would fight retuning.
    """
    query = "How long does a home loan take to disburse after approval?"

    before = client.post("/ask", json={"query": query, "thread_id": "up-a"}).json()
    assert before["policy"]["outcome"].startswith("refused")
    before_similarity = before["policy"]["top1_similarity"]

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
    assert after["policy"]["top1_similarity"] > before_similarity
    assert any(c.startswith("kb-up-") for c in after["policy"]["citations"])


def test_add_document_is_visible_to_a_query_naming_its_catalogue_product(clean_uploads):
    """The boundary this used to pin is reversed on purpose, not a regression.

    Before, rag/retrieve.py's product narrowing (D-53) built its `doc_id` `$in`
    clause from catalogue.json alone, so a query naming a catalogue product
    could never reach an upload - only the identical question with no product
    named could. Bhavik chose to close that gap rather than document it as a
    V1 limit: the narrowing set is now the catalogue's doc ids union every
    doc id POST /add-document has written to data/uploads/, read straight off
    the directory (rag/retrieve.py::_uploaded_doc_ids), so rag/ still has no
    dependency on api/ and knowledge_base/ and catalogue.json stay untouched -
    D-72 is unchanged. This test now pins the closed gap: a query naming the
    upload's product must reach it, exactly like a query naming no product.
    """
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
    assert any(c.startswith("kb-up-") for c in named_product["policy"]["citations"])

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


def test_add_document_rejects_a_body_that_chunks_below_the_minimum():
    """api/main.py:227-237's other 422 branch, otherwise untested.

    A single-sentence body produces one chunk under STRATEGY_SENTENCES, below
    config.MIN_CHUNKS_PER_DOCUMENT (2): a single chunk can never satisfy the
    same-parent support rule (D-07), so it would be permanently unanswerable.
    Rejected before anything is written, so no upload cleanup is needed here.
    """
    response = client.post("/add-document", json={
        "title": "Too short",
        "body": "A personal loan is unsecured.",
        "products": ["Personal Loan"],
    })
    assert response.status_code == 422
    assert "chunk" in response.json()["detail"]


def test_40_add_document_never_writes_into_the_knowledge_base(clean_uploads):
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


def test_a_rebuild_drops_every_upload(clean_uploads):
    """The rollback D-72 relies on, which is behaviour build_index already has.

    This is the one test that used to double as cleanup for the others, by
    file-order accident: it rebuilds the index itself, so the other tests
    relied on it running last. It no longer needs to be that - clean_uploads
    puts every upload-writing test back at its own 215-chunk baseline
    regardless of which subset of tests runs - but it still asserts the real
    property: a rebuild drops every upload.
    """
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
