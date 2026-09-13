"""Part 3 Task 11. The FastAPI deployment, and Task 12's log line.

Two endpoints over one call. Spec section 15 contracted exactly one entry
point from agent/, `ask`, and this module imports exactly that.

Both endpoints are declared `def` rather than `async def` deliberately. Part 4
makes `ask` wrap asyncio.run, per D-78, and asyncio.run raises inside a
running event loop. A `def` endpoint runs in FastAPI's threadpool where there
is no loop, so it works identically before and after Part 4 lands.
"""

import hashlib
import json
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.requests import Request

import config
import obs
from agent import guardrails
from agent.graph import ask
from agent.schema import AgentResponse as AskResponse
from rag import chunking, index, kb

obs.configure()

app = FastAPI(
    title="Meridian Bank loan support agent",
    description="Part 3 Task 11. Every response is the Part 2 envelope, unchanged.",
    version="1.0.0",
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Task 12. One JSON line per request, per D-75, on every path including a 500.

    `BaseHTTPMiddleware` (what `@app.middleware("http")` builds on) re-raises
    whatever `call_next` raises rather than returning a response for it, so an
    unhandled exception from the endpoint below - a `NodeTimeoutError` escaping
    `ask`, for instance - would previously skip the whole logging block that
    used to sit after `await call_next`, and the request went completely
    unlogged. Criterion 2 is "log every request", so the line now lives in a
    `finally`: it runs whether `call_next` returns or raises, and the status it
    records is 200-ish on the happy path or 500 on the path where Starlette's
    `ServerErrorMiddleware` (outside this one) converts the escaped exception
    into a response this middleware never sees. The `raise` is implicit -
    `finally` runs and then the original exception keeps propagating - so the
    client-visible behaviour (and ServerErrorMiddleware's own handling) is
    unchanged.

    The middleware is async because Starlette's middleware contract is; the
    endpoints below it stay sync for the reason in the module docstring.
    """
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive

    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = round(
            (time.perf_counter() - started) * 1000, config.LOG_DURATION_PLACES
        )

        payload = {}
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}

        raw_query = payload.get("query", "")
        masked_query, _ = guardrails.mask_pii(raw_query) if raw_query else ("", ())

        if request.url.path == "/ask" and raw_query:
            # The response's own trace id, so the log line joins to the transcript
            # that produced it, per D-38.
            #
            # Deliberately no `+ 1` here, unlike agent/graph.py:111's
            # `len(memory.load(thread_id).turns) + 1`: that call runs *before* the
            # graph executes, so it has to predict the turn number a not-yet-persisted
            # turn will get. This line runs after `await call_next` above, i.e. after
            # the endpoint (and the memory.record_turn call inside it) has already
            # completed on the success path, so `len(memory.load(thread_id).turns)`
            # here already counts the turn this same request just persisted - it
            # already equals the `turn` D-38's trace_id was built from. Adding
            # another `+ 1` double counts and was measured to produce a trace_id
            # one turn ahead of the response's own; do not "fix" this back to
            # `+ 1` without re-measuring. On the exception path no turn was
            # persisted, so this reads the same turn count the request started
            # with - the id still deterministically identifies the attempt.
            from agent import memory, schema

            thread_id = payload.get("thread_id", "default")
            turn = len(memory.load(thread_id).turns)
            trace_id = schema.trace_id(thread_id, turn, masked_query)
        else:
            # No AgentResponse exists for this request, so the id comes from the
            # body. Canonical JSON, so key order in the request cannot change it.
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            trace_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

        obs.event(
            "http_request",
            trace_id=trace_id,
            path=request.url.path,
            status=status_code,
            duration_ms=duration_ms,
            query_masked=masked_query,
        )


class HealthResponse(BaseModel):
    """What the deployment is running, for a client that wants to say so."""
    status: str = Field(description='Always "ok" when the process answers.')
    provider: str = Field(description="The live LLM_PROVIDER, resolved per call.")
    model: str | None = Field(
        default=None, description="The model name, when the provider takes one."
    )


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    """Report the live provider.

    Read through config.resolve_provider() rather than config.LLM_PROVIDER,
    because the frozen constant is whatever .env held at first import and this
    endpoint's whole job is to be right about now. That distinction already
    cost this repository a non-deterministic suite once; the CLAUDE.md rule
    naming it applies here too.

    Reporting, not setting. There is deliberately no way to change the
    provider over HTTP: the variable llm.generate reads is process-global and
    /ask runs in a threadpool, so a request that wrote it would flip the
    provider under every concurrent request. Switching providers is a restart.
    """
    provider = config.resolve_provider()
    return HealthResponse(
        status="ok",
        provider=provider,
        model=config.GROQ_MODEL if provider == config.PROVIDER_GROQ else None,
    )


class AskRequest(BaseModel):
    """One turn of a conversation."""
    query: str = Field(min_length=1, description="The member's question.")
    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation key. The same id carries memory across turns.",
    )


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer one question, through the full nine-node graph.

    `ask` wraps asyncio.run (D-78). A slow tool call ties up this threadpool
    worker for its full duration even after the graph's own timeout has
    fired, because asyncio.run's shutdown waits for every worker thread it
    handed work to and CPython cannot kill a running thread. The graph's
    timeouts bound the graph, not this endpoint's wall clock. This is a known,
    documented cost, not something fixed here.

    `AskResponse` is `agent.schema.AgentResponse` itself, not a second model
    describing the same shape by hand: `ask()` already returns exactly that
    envelope, validated against the committed JSON Schema before it ever
    reaches here (agent/nodes.py::compose), so redeclaring the fields here
    would only be a second place for the two to drift apart.
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

    A query naming this document's product still reaches it. rag/generate.py
    narrows retrieval to a set of doc ids whenever a query names a catalogue
    product (D-53); that set is the catalogue's doc ids union every doc id
    under data/uploads/ (rag/retrieve.py::_uploaded_doc_ids), read straight
    off that directory rather than off catalogue.json, so D-72 stays
    unmodified and rag/ still has no dependency on api/. tests/test_api.py
    pins this for both a product-naming and a product-free query.

    `request.products` is validated against the catalogue above and then
    discarded (D-86): every upload joins the narrowing set of every product,
    not only the ones it declared, so nothing downstream reads the field
    back to decide what an upload is "about".
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
