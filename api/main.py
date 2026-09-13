"""Part 3 Task 11. The FastAPI deployment, and Task 12's log line.

Two endpoints over one call. Spec section 15 contracted exactly one entry
point from agent/, `ask`, and this module imports exactly that.

Both endpoints are declared `def` rather than `async def` deliberately. Part 4
makes `ask` wrap asyncio.run, per D-78, and asyncio.run raises inside a
running event loop. A `def` endpoint runs in FastAPI's threadpool where there
is no loop, so it works identically before and after Part 4 lands.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from agent.graph import ask
from rag import chunking, index, kb

app = FastAPI(
    title="Meridian Bank loan support agent",
    description="Part 3 Task 11. Every response is the Part 2 envelope, unchanged.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    """One turn of a conversation."""
    query: str = Field(min_length=1, description="The member's question.")
    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation key. The same id carries memory across turns.",
    )


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    """Answer one question, through the full nine-node graph.

    `ask` wraps asyncio.run (D-78). A slow tool call ties up this threadpool
    worker for its full duration even after the graph's own timeout has
    fired, because asyncio.run's shutdown waits for every worker thread it
    handed work to and CPython cannot kill a running thread. The graph's
    timeouts bound the graph, not this endpoint's wall clock. This is a known,
    documented cost, not something fixed here.
    """
    return ask(request.query, thread_id=request.thread_id)


class AddDocumentRequest(BaseModel):
    """A document to add to the live collection for the rest of this process."""
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    products: list[str] = Field(
        min_length=1,
        description="Products this document covers. Each must already be in "
                    "knowledge_base/catalogue.json.",
    )


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    collection: str


def _next_upload_id() -> str:
    """Stable within a run, and visibly an upload in any citation."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    used = {
        int(path.stem.removeprefix(config.UPLOAD_DOC_PREFIX))
        for path in config.UPLOAD_DIR.glob(f"{config.UPLOAD_DOC_PREFIX}*.txt")
    }
    return f"{config.UPLOAD_DOC_PREFIX}{max(used, default=0) + 1:02d}"


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(request: AddDocumentRequest) -> AddDocumentResponse:
    """Add a document to the live kb_sentences collection.

    knowledge_base/ is never written, per D-72: a document there would move
    Precision@3, Recall@3 and every chunk count in the Part 1 transcripts.
    The next rag.index.build_index drops everything added here, because it
    rebuilds any collection whose count does not match the corpus on disk.

    Known boundary: an uploaded document is invisible to any query that names
    a catalogue product. rag/generate.py narrows retrieval to the doc ids
    knowledge_base/catalogue.json tags with that product whenever a query
    names one (D-53), and an upload is never in the catalogue, so it can
    never appear in that narrowed set. Only a query naming no catalogue
    product reaches an upload. Lifting this means either writing the
    upload's product into catalogue.json, which D-72 forbids, or teaching the
    product filter in rag/kb.py and rag/retrieve.py about uploads, which is a
    V2 change, not this task's. tests/test_api.py pins both halves of this as
    known behaviour.
    """
    from fastapi import HTTPException

    catalogue = set(kb.catalogue_products())
    unknown = [p for p in request.products if p not in catalogue]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{unknown} is not in the catalogue. rag/scope.py refuses any "
                f"question naming an out-of-catalogue product before retrieval "
                f"runs, so this document would be permanently unreachable."
            ),
        )

    pieces = chunking.chunk(request.body, config.UPLOAD_STRATEGY)
    if len(pieces) < config.MIN_CHUNKS_PER_DOCUMENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body produced {len(pieces)} chunk(s), needs at least "
                f"{config.MIN_CHUNKS_PER_DOCUMENT}. A single-chunk document can "
                f"never satisfy the same-parent support rule, so it would be "
                f"permanently unanswerable."
            ),
        )

    doc_id = _next_upload_id()
    (config.UPLOAD_DIR / f"{doc_id}.txt").write_text(
        f"{request.title}\n\n{request.body}\n", encoding="utf-8"
    )

    collection = index.get_collection(config.UPLOAD_STRATEGY)
    collection.add(
        ids=[f"{doc_id}::{config.UPLOAD_STRATEGY}::{i:03d}" for i in range(len(pieces))],
        documents=pieces,
        embeddings=index.embed(pieces),
        metadatas=[{
            "doc_id": doc_id,
            "title": request.title,
            "topic": "uploaded",
            "required": False,
            "chunk_index": i,
            "strategy": config.UPLOAD_STRATEGY,
        } for i in range(len(pieces))],
    )
    return AddDocumentResponse(
        doc_id=doc_id,
        chunks_added=len(pieces),
        collection=index.collection_name(config.UPLOAD_STRATEGY),
    )
