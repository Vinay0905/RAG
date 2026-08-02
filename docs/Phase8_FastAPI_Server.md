# Phase 8 — The FastAPI Async Server

> **Prerequisite:** Phases 1–6. This phase exposes Phase 6's `RAGService` over HTTP and adds no
> retrieval or reasoning logic of its own.
>
> **Budget: ~600 lines of Python across 7 files, on budget.** §2 lists what was cut.
>
> **API verified 2026-08-02:** FastAPI's `lifespan` context manager, `sse-starlette`'s
> `EventSourceResponse`, and `app.dependency_overrides` for tests.
>
> **This file replaces the previous `Phase8_FastAPI_Server.md`.**

---

## 1. What Makes This Phase Hard

The routes are thin. Four things around them are not.

**Heavy resources must be built once.** An ONNX embedding model, a cross-encoder, a Qdrant client, a
Groq client, and a Redis pool. Build them per request and the first query takes eight seconds and the
tenth exhausts your file descriptors.

**Errors have to become status codes without anyone parsing prose.** Phase 1 gave every exception a
`status_code` and a `to_dict()`; Phase 5 gave every answer an `AnswerStatus`. This phase is where
that groundwork either pays off or is quietly ignored in favour of `except Exception: return 500`.

**Streaming conflicts with grading, and the decision is already made.** Phase 5 §11: you cannot
un-stream an ungrounded answer. Stream the *trace* — the progress of the graph — and deliver the
verified answer in one piece. This phase implements that; it does not relitigate it.

**A slow endpoint blocks everything if you get one `def` wrong.** FastAPI runs `async def` routes on
the event loop and plain `def` routes in a threadpool. An `async def` route that calls something
blocking stalls every other request in the process.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| 1 | `src/api/schemas.py` | 90 | Request and response models |
| 2 | `src/api/dependencies.py` | 70 | Lifespan resources → routes |
| 3 | `src/api/errors.py` | 70 | `RAGException` → JSON envelope |
| 4 | `src/api/routes/query.py` | 130 | `/query` and `/query/stream` |
| 5 | `src/api/routes/health.py` | 70 | Liveness and readiness |
| 6 | `src/api/routes/admin.py` | 60 | Cache clear, stats |
| 7 | `src/api/app.py` | 110 | Lifespan, middleware, wiring |

```text
src/api/
├── __init__.py
├── app.py
├── schemas.py
├── dependencies.py
├── errors.py
└── routes/
    ├── __init__.py
    ├── query.py
    ├── health.py
    └── admin.py
```

---

## 2. What Was Cut

**Authentication and authorisation.** Phase 11 owns this, properly, with tenant filters that reach
into retrieval. A bearer-token check here would be security theatre that Phase 11 then rips out.

**Rate limiting.** One `slowapi` decorator is tempting. It protects a single-process deployment
against nobody, since the thing to protect is the Groq quota, and Phase 6's provider already handles
429s with backoff.

**WebSockets.** SSE covers server-to-client streaming, which is all this needs. WebSockets add a
second protocol, a second set of reconnection semantics, and a second thing to test.

**An ingestion endpoint that runs ingestion.** Phase 2 is a multiprocessing job that runs for hours;
triggering it from a web worker means a request that cannot complete and a pool that cannot fork
safely. `scripts/index_corpus.py` is the interface. Admin exposes *status*, not a trigger.

**Prometheus metrics, OpenTelemetry, structured request IDs beyond Phase 1's.** Phase 1 already has
correlation IDs and telemetry spans; wiring an exporter is deployment work, not learning.

That is ~350 lines declined.

---

## 3. File 1 — `src/api/schemas.py`

### Design Decision

**Separate API models from domain models.** `RAGAnswer` has `sources: list[ScoredChunk]`, and a
`ScoredChunk` contains the full text of a 1,600-token parent chunk. Returning five of those is a
40 KB response where 2 KB was wanted, and it couples your public JSON to an internal model — every
future change to `Chunk` becomes a breaking API change.

```python
from pydantic import BaseModel, Field

from src.core.models import AnswerStatus, RAGAnswer


class QueryRequest(BaseModel):
    """A question. Validated here so bad input never reaches the pipeline."""

    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    #: Only the fields Phase 3 indexed. An unsupported key raises inside the store,
    #: and letting it get that far turns a 400 into a 503.
    year: int | None = Field(default=None, ge=1900, le=2100)
    doc_id: str | None = None

    def to_filters(self) -> dict | None:
        filters = {k: v for k, v in
                   {"year": self.year, "doc_id": self.doc_id}.items() if v is not None}
        return filters or None


class SourceOut(BaseModel):
    """A citation-sized view of a source. Deliberately not the whole chunk."""

    chunk_id: str
    contract_name: str
    section_title: str
    year: int | None = None
    #: A snippet, not the passage. Phase 9's UI expands on demand via /source/{id}.
    excerpt: str
    score: float


class GradingOut(BaseModel):
    verified: bool
    grounded: bool
    relevant: bool
    confidence: float


class QueryResponse(BaseModel):
    """What a client gets. Stable, small, and independent of internal models."""

    question: str
    answer: str
    status: AnswerStatus
    sources: list[SourceOut] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    grading: GradingOut | None = None
    warnings: list[str] = Field(default_factory=list)
    retry_count: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0

    @classmethod
    def from_answer(cls, answer: RAGAnswer, excerpt_chars: int = 400) -> "QueryResponse":
        return cls(
            question=answer.query,
            answer=answer.answer,
            status=answer.status,
            sources=[
                SourceOut(
                    chunk_id=s.chunk.chunk_id,
                    contract_name=s.chunk.contract_name,
                    section_title=s.chunk.section_title,
                    year=s.chunk.year,
                    excerpt=s.chunk.text[:excerpt_chars],
                    score=round(s.effective_score, 4),
                )
                for s in answer.sources
            ],
            citations=[c.source_index for c in answer.citations],
            grading=GradingOut(
                verified=answer.grading.verified,
                grounded=answer.grading.is_grounded,
                relevant=answer.grading.is_relevant,
                confidence=answer.grading.confidence,
            ) if answer.grading else None,
            warnings=answer.warnings,
            retry_count=answer.retry_count,
            cache_hit=answer.cache_hit,
            latency_ms=round(answer.total_latency_ms, 1),
        )
```

`status` is on the response for the reason Phase 5 added it: a client needs to render `UNVERIFIED`
differently from `ANSWERED` without reading English.

---

## 4. Files 2 and 3 — dependencies and errors

### `src/api/dependencies.py`

```python
from typing import Annotated

from fastapi import Depends, Request

from config.settings import Settings, get_settings
from src.app import RAGService


def get_service(request: Request) -> RAGService:
    """The service built once in the lifespan.

    A dependency rather than `request.app.state.service` at every call site: routes
    stay unaware of where it lives, and `app.dependency_overrides[get_service]` lets
    Phase 10 substitute a fake with no monkeypatching.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        # Only reachable if a route is called outside the lifespan — which is what
        # happens when a test uses TestClient WITHOUT the context-manager form.
        raise RuntimeError(
            "Service unavailable: the app lifespan did not run. In tests, use "
            "`with TestClient(app) as client:` rather than a bare TestClient."
        )
    return service


ServiceDep = Annotated[RAGService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
```

`get_settings` is Phase 1's `@lru_cache`d accessor, used as a dependency exactly as Phase 1 §4
promised — that is what makes it overridable in tests.

### `src/api/errors.py`

### Design Decision

**One handler for `RAGException`, driven by the data already on it.** Phase 1 put `status_code`,
`retryable`, and `to_dict()` on the hierarchy specifically so this file could be short and no route
would ever build an error response by hand.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import RAGException
from src.core.logging import get_request_id, logger


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RAGException)
    async def handle_rag_exception(request: Request, exc: RAGException) -> JSONResponse:
        """Every deliberate failure, mapped by its own declared status.

        No `isinstance` ladder. `InvalidQueryError` carries 400, `PromptInjectionError`
        400, `RetrievalError` 503, `RateLimitError` 429, `CollectionNotFoundError` 404
        — the exception knows, so the handler does not have to.
        """
        payload = exc.to_dict()
        payload["request_id"] = get_request_id()

        # 5xx is ours; 4xx is theirs. Logging every rejected query at ERROR trains
        # people to ignore the error log.
        log = logger.error if exc.status_code >= 500 else logger.info
        log(
            "Request failed",
            extra={"error": type(exc).__name__, "status": exc.status_code,
                   "path": request.url.path},
        )

        headers = {"Retry-After": "5"} if exc.retryable else None
        return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Anything not ours is a bug. Log it fully, tell the client nothing.

        The message is deliberately generic: an internal traceback in an HTTP
        response is an information leak, and on this corpus the leak could be
        contract text.
        """
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": "An unexpected error occurred.",
                "request_id": get_request_id(),
            },
        )
```

---

## 5. File 4 — `src/api/routes/query.py`

### The Problem

Two endpoints. `/query` is a normal request/response. `/query/stream` has to solve the problem Phase
5 §11 identified: streaming answer tokens means the user reads an answer before it has been graded,
and you cannot take it back.

### Design Decision

**Stream the trace, not the tokens.** Events describe what the agent is doing — `retrieving`,
`generating`, `verifying` — and the final event carries the complete, graded answer. The user sees
continuous progress and never sees unverified content. This is the option Phase 5 §11 recommended and
it is why `AgentState.trace` exists.

**Poll the graph's state rather than instrument the nodes.** LangGraph's `astream` yields state after
each super-step, which is exactly a progress feed. No callbacks threaded through seven nodes.

**Check `request.is_disconnected()` between events.** A browser that closes the tab leaves a
generator running through four LLM calls that nobody will read.

```python
import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import ServiceDep
from src.api.schemas import QueryRequest, QueryResponse
from src.core.exceptions import RAGException
from src.core.logging import logger, new_request_id

router = APIRouter(tags=["query"])

#: Human-readable labels for the trace entries Phase 5's nodes emit.
_STAGES = {
    "router": "Understanding the question",
    "retrieval": "Searching contracts",
    "generation": "Drafting an answer",
    "grading": "Verifying against sources",
    "rewrite": "Refining the search",
    "no_context": "No matching passages",
}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, service: ServiceDep) -> QueryResponse:
    """Ask a question. Returns a complete, graded answer.

    `async def`, because everything inside it awaits: the embedder offloads to a
    thread, the store and the LLM are network calls. A plain `def` here would move
    the whole request to the threadpool and lose the concurrency the last five
    phases were built for.

    Raises nothing directly — `RAGException` subclasses propagate to the handler in
    `errors.py`, which maps them by their own `status_code`.
    """
    new_request_id()
    answer = await service.answer(
        request.question, top_k=request.top_k, filters=request.to_filters()
    )
    return QueryResponse.from_answer(answer)


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest, http_request: Request, service: ServiceDep
) -> EventSourceResponse:
    """Ask a question, streaming PROGRESS and then the finished answer.

    Deliberately not token streaming. Phase 5 §11: grading happens after generation,
    so streaming tokens shows the user an answer before it has been checked and
    there is no way to retract it. Progress events give the same perceived
    responsiveness without ever displaying unverified content.
    """
    new_request_id()
    return EventSourceResponse(_events(request, http_request, service))


async def _events(
    request: QueryRequest, http_request: Request, service: ServiceDep
) -> AsyncIterator[dict]:
    """Yield SSE events: several `progress`, then one `answer` or one `error`.

    Wrapped in try/finally so a client disconnect or an exception still closes the
    task cleanly — an abandoned generator here holds an LLM call open.
    """
    task = asyncio.create_task(
        service.answer(request.question, top_k=request.top_k, filters=request.to_filters())
    )
    seen = 0

    try:
        while not task.done():
            if await http_request.is_disconnected():
                # The user closed the tab. Cancelling matters: without it this run
                # continues through four LLM calls that nobody will read.
                task.cancel()
                logger.info("Client disconnected; cancelled the run")
                return

            trace = getattr(service.agent, "last_trace", [])
            for entry in trace[seen:]:
                stage = entry.split(":")[0]
                yield {
                    "event": "progress",
                    "data": json.dumps({"stage": stage,
                                        "label": _STAGES.get(stage, stage)}),
                }
            seen = len(trace)
            await asyncio.sleep(0.2)

        answer = await task
        yield {
            "event": "answer",
            "data": QueryResponse.from_answer(answer).model_dump_json(),
        }

    except asyncio.CancelledError:
        raise
    except RAGException as exc:
        # An error inside a stream cannot be an HTTP status — the 200 was sent with
        # the first byte. It has to be an event the client knows how to render.
        yield {"event": "error", "data": json.dumps(exc.to_dict())}
    except Exception:
        logger.exception("Streaming query failed")
        yield {
            "event": "error",
            "data": json.dumps({"error": "InternalError",
                                "message": "An unexpected error occurred."}),
        }
    finally:
        if not task.done():
            task.cancel()
```

### The one thing this needs from Phase 5

`service.agent.last_trace` — the running trace of the current run. Add to `RAGAgent`:

```python
        self.last_trace: list[str] = []
```

and inside `answer()`, replace the single `ainvoke` with a streaming consume:

```python
        final: AgentState = {}
        self.last_trace = []
        async for state in self.graph.astream(state, config, stream_mode="values"):
            final = state
            self.last_trace = list(state.get("trace", []))
```

`stream_mode="values"` yields the full state after each super-step, so the last one is the final
state and `ainvoke`'s result is unchanged. This is a genuine cross-phase edit and the only one this
phase needs.

**Its honest limitation:** `last_trace` is per-agent, not per-request, so two concurrent streaming
requests share it and each sees a mix of both traces. For a single-user system that is invisible; for
a multi-user one it is wrong. The correct fix is a per-request state object threaded through, which
is Phase 11's territory since that phase introduces the request principal anyway. Noted rather than
hidden.

---

## 6. Files 5 and 6 — health and admin

### `src/api/routes/health.py`

### Design Decision

**Liveness and readiness are different questions and need different endpoints.** `/health` answers
"is this process alive" — no dependencies, no I/O, always fast. `/ready` answers "can it serve a
request", which means checking Qdrant. Conflating them means a Qdrant blip causes your orchestrator
to *restart the process*, which cannot possibly help.

```python
from fastapi import APIRouter, Response, status

from config.settings import settings
from src.api.dependencies import ServiceDep
from src.core.logging import logger

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness. No dependencies — if this process can run code, it is alive."""
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@router.get("/ready")
async def ready(service: ServiceDep, response: Response) -> dict:
    """Readiness. Touches the vector store, so it can legitimately fail.

    Returns 503 with a reason rather than raising: a readiness probe is polled every
    few seconds, and an exception traceback in the log on every poll during a
    restart is noise that hides the actual problem.
    """
    try:
        points = await service.store.count()
    except Exception as exc:
        logger.warning("Readiness check failed", extra={"error": type(exc).__name__})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "reason": f"vector store unreachable ({type(exc).__name__})"}

    if points == 0:
        # Reachable but empty: every query would return NO_MATCH. Not an error, but
        # not ready either, and the distinction saves someone an hour.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "reason": "collection is empty; run ingestion"}

    return {"ready": True, "points": points, "collection": settings.QDRANT_COLLECTION}
```

### `src/api/routes/admin.py`

```python
from fastapi import APIRouter

from config.settings import settings
from src.api.dependencies import ServiceDep

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/cache/clear")
async def clear_cache(service: ServiceDep) -> dict:
    """Drop every cached answer.

    This is an operational necessity, not a convenience: the cache key cannot
    observe that the index changed underneath it, so **re-ingestion must be
    followed by a cache clear** or stale answers are served against new evidence
    until the TTL expires. Phase 6 says so; this is the button.
    """
    await service.cache.clear()
    return {"cleared": True}


@router.get("/stats")
async def stats(service: ServiceDep) -> dict:
    """Enough to answer 'why is it behaving like that' without shell access."""
    return {
        "collection": settings.QDRANT_COLLECTION,
        "points": await service.store.count(),
        "cache": {
            "hits": service.cache.exact.hits,
            "misses": service.cache.exact.misses,
            "semantic_index": len(service.cache._index),
        },
        "models": {
            "generation": settings.GENERATION_MODEL,
            "grader": settings.GRADER_MODEL,
            "embedding": settings.DENSE_MODEL_NAME,
        },
        "flags": {
            "multi_query": settings.ENABLE_MULTI_QUERY,
            "hyde": settings.ENABLE_HYDE,
            "reranking": settings.ENABLE_RERANKING,
            "self_correction": settings.ENABLE_SELF_CORRECTION,
            "semantic_cache": settings.ENABLE_SEMANTIC_CACHE,
        },
    }
```

**No ingestion trigger.** Phase 2 is a multiprocessing job that runs for hours; forking a worker pool
from inside a web process is a reliable way to produce orphaned processes, and an HTTP request cannot
wait for it. `scripts/index_corpus.py` is the interface.

`/admin` has no authentication, which is only acceptable because §7 binds the server to `127.0.0.1`.
Phase 11 adds the real thing.

---

## 7. File 7 — `src/api/app.py`

### Design Decision

**One `lifespan` context manager owns every resource.** Everything before `yield` is startup and
everything after is shutdown, and Starlette guarantees the shutdown block runs. The deprecated
`@app.on_event` decorators are not used.

**Build `RAGService` once, store it on `app.state`.** Phase 6's `build_service()` already handles
partial-failure cleanup, so a startup crash releases what it built.

**Bind to `127.0.0.1`.** There is no authentication until Phase 11, and a service that answers
questions about confidential contracts must not be listening on `0.0.0.0` in the meantime.

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.errors import register_error_handlers
from src.api.routes import admin, health, query
from src.app import build_service
from src.core.logging import logger, new_request_id


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every heavy resource once; release them all on shutdown.

    The models loaded here are the reason this exists: an ONNX embedder and a
    cross-encoder are ~200 MB and several hundred milliseconds each. Per-request
    construction would make the first query slow and the hundredth fatal.
    """
    logger.info("Starting", extra={"app": settings.APP_NAME, "env": settings.ENVIRONMENT})
    for warning in settings.validate_runtime():
        logger.warning("Configuration", extra={"warning": warning})

    app.state.service = await build_service()
    logger.info("Ready")

    yield

    logger.info("Shutting down")
    service = app.state.service
    # Each close is independent: one failing must not skip the others.
    for closer in (service.llm.close, service.store.close,
                   service.embedder.close, service.cache.exact.close):
        try:
            await closer()
        except Exception as exc:
            logger.warning("Shutdown step failed", extra={"error": type(exc).__name__})


def create_app() -> FastAPI:
    """Application factory. A factory, not a module-level `app`, so tests can build
    a fresh instance with overridden dependencies."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        # The interactive docs are the fastest way to exercise this by hand, and
        # they are generated from the Pydantic schemas in `schemas.py` for free —
        # the payoff Phase 1 §7.1 promised for typing everything.
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        # Phase 9's dashboard is served from a file, so the origin is null/localhost.
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "null"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):
        """One request ID per request, echoed in the response.

        Phase 1 built `new_request_id()` and contextvar propagation for this: every
        log line inside the request carries the same ID, and the client gets it in a
        header, so a user reporting "it failed" can hand you one string that finds
        every line.
        """
        request_id = new_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(admin.router)
    return app


app = create_app()
```

Run it:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Add to `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `sse-starlette`, and `httpx` for tests.

### The Theory: `async def` versus `def`, and the one that bites

FastAPI treats the two differently, and the difference is not stylistic.

An **`async def`** route runs *on the event loop*. Thousands can be in flight because each yields at
every `await`. If it calls something blocking — `time.sleep`, `requests.get`, a synchronous ONNX
inference — the loop stops, and **every other request in the process waits**, including health checks.

A **`def`** route runs in a threadpool. It may block safely, but the pool is bounded (40 by default),
so it caps concurrency and adds a context switch.

The rule: `async def` when the body only awaits; `def` when it must block. Every route here is
`async def`, which is only correct because the blocking work was pushed down phases ago — Phase 3's
embedder wraps ONNX in `asyncio.to_thread`, Phase 4's reranker does the same. If `/query` feels like
it serialises requests, something below it is blocking, and the fix belongs there rather than here.

---

## 8. Verification (deferred)

Save as `scripts/verify_phase8.py`. No Qdrant, no API key — the service is overridden with a fake,
which is the point of the dependency.

```python
"""Phase 8 verification. Runs entirely against a fake service."""
import sys

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_service
from src.core.exceptions import InvalidQueryError, RetrievalError
from src.core.models import AnswerStatus, Chunk, ChunkLevel, RAGAnswer, ScoredChunk
from src.core.utils import make_chunk_id, make_doc_id, make_section_id

DOC = make_doc_id("data/contracts/2003/api.txt")
SEC = make_section_id(DOC, "Termination", 0)


def source() -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=make_chunk_id(DOC, SEC, 0), doc_id=DOC, section_id=SEC,
            text="Either party may terminate on ninety (90) days notice. " * 20,
            chunk_index=0, token_count=200, contract_name="Acme Agreement",
            section_title="Termination", year=2003, chunk_level=ChunkLevel.PARENT,
        ),
        score=0.9, rank=0,
    )


class FakeService:
    def __init__(self, answer=None, error=None) -> None:
        self._answer = answer
        self._error = error
        self.calls = 0
        self.store = self
        self.cache = type("C", (), {"exact": type("E", (), {"hits": 3, "misses": 7})(),
                                    "_index": []})()

    async def answer(self, question, top_k=None, filters=None):
        self.calls += 1
        if self._error:
            raise self._error
        return self._answer

    async def count(self) -> int:
        return 1234


def client_for(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    # Context-manager form so the lifespan runs. Without it, `app.state.service` is
    # never set — though the override means we never reach it.
    return TestClient(app)


def check_query_shape() -> None:
    answer = RAGAnswer(
        query="what notice is required?", answer="Ninety days [SOURCE 1].",
        sources=[source()], status=AnswerStatus.ANSWERED, total_latency_ms=250.0,
    )
    with client_for(FakeService(answer=answer)) as client:
        response = client.post("/query", json={"question": "what notice is required?"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["contract_name"] == "Acme Agreement"
    assert len(body["sources"][0]["excerpt"]) <= 400, (
        "the response must return an excerpt, not the whole 1,600-token parent"
    )
    assert "text" not in body["sources"][0], "internal Chunk fields must not leak"
    assert response.headers.get("X-Request-ID"), "every response needs a correlation ID"
    print("✓ query: response is small, typed, and correlated")


def check_status_mapping() -> None:
    """Non-answers are 200 with a status; failures are their exception's code."""
    unsupported = RAGAnswer(
        query="how many contracts?", answer="Aggregation is not supported.",
        status=AnswerStatus.UNSUPPORTED, failure_reason="aggregation_not_supported",
    )
    with client_for(FakeService(answer=unsupported)) as client:
        response = client.post("/query", json={"question": "how many contracts?"})
    assert response.status_code == 200, "an unsupported question is not an HTTP error"
    assert response.json()["status"] == "unsupported"

    with client_for(FakeService(error=RetrievalError("down", retryable=True))) as client:
        response = client.post("/query", json={"question": "anything at all"})
    assert response.status_code == 503, f"expected 503, got {response.status_code}"
    assert response.json()["retryable"] is True
    assert response.headers.get("Retry-After"), "a retryable error should say so"

    with client_for(FakeService(error=InvalidQueryError("blank"))) as client:
        response = client.post("/query", json={"question": "valid length question"})
    assert response.status_code == 400, "a bad request must not be reported as an outage"

    print("✓ errors: mapped by the exception's own status_code, no isinstance ladder")


def check_validation() -> None:
    service = FakeService(answer=RAGAnswer(query="q", answer="a"))
    with client_for(service) as client:
        assert client.post("/query", json={"question": "hi"}).status_code == 422
        assert client.post("/query", json={"question": "x" * 2000}).status_code == 422
        assert client.post("/query", json={"question": "valid question here",
                                           "top_k": 99}).status_code == 422
    assert service.calls == 0, "invalid input must be rejected before the service runs"
    print("✓ validation: FastAPI rejects bad input before any work happens")


class EmptyStoreService(FakeService):
    """Reachable but empty — the most common state of a fresh deployment."""

    async def count(self) -> int:
        return 0


def check_health() -> None:
    with client_for(FakeService(answer=None)) as client:
        assert client.get("/health").status_code == 200, "liveness must never depend on I/O"
        assert client.get("/ready").status_code == 200

    with client_for(EmptyStoreService(answer=None)) as client:
        response = client.get("/ready")
    assert response.status_code == 503, "an empty collection is not ready to serve"
    assert "ingestion" in response.json()["reason"], (
        "the reason must name the fix; 'not ready' alone costs someone an hour"
    )
    print("✓ health: liveness is dependency-free, readiness checks the store")


def check_stream() -> None:
    answer = RAGAnswer(query="q", answer="Ninety days [SOURCE 1].",
                       sources=[source()], status=AnswerStatus.ANSWERED)
    service = FakeService(answer=answer)
    service.agent = type("A", (), {"last_trace": ["router:vague", "retrieval:3"]})()

    with client_for(service) as client:
        with client.stream("POST", "/query/stream",
                           json={"question": "what notice is required?"}) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "event: answer" in body, "the stream must end with the finished answer"
    assert "Ninety days" in body
    print("✓ stream: progress events then one complete answer")


def main() -> int:
    try:
        check_query_shape()
        check_status_mapping()
        check_validation()
        check_health()
        check_stream()
    except AssertionError as exc:
        print(f"\n✗ FAILED: {exc}")
        return 1
    print("\nPhase 8 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### What this does not cover

- **The real lifespan.** Every test overrides `get_service`, so `build_service()` is never exercised
  here. Phase 10 starts the app for real against a test collection.
- **Concurrency under load.** The `last_trace` limitation in §5 is exactly the kind of thing a
  two-client test would expose, and it is not tested.

---

## 9. What Phase 8 Bought You

**An HTTP surface over six phases of machinery**, with request/response models that do not leak
internal types and an OpenAPI schema generated from them for free.

**Errors that carry their own status codes.** `register_error_handlers` is twenty lines because Phase
1 put `status_code`, `retryable`, and `to_dict()` on the exception hierarchy. That decision is
finally cashed in here.

**Streaming that does not show unverified content.** Progress events, then one graded answer — the
resolution Phase 5 §11 argued for, implemented rather than restated.

**Health endpoints that distinguish "restart me" from "wait for me".** And a readiness check that
tells you the collection is empty, which is the single most common reason a fresh deployment answers
nothing.

### What is deliberately not here

Authentication (Phase 11), rate limiting, WebSockets, an ingestion trigger, and metrics exporters.
§2 has the reasoning for each. The only one I would revisit early is auth, and only because Phase 11
is where the whole tenant model arrives — bolting a token check on now would be replaced wholesale.

---

## Next

**Phase 9 — the CLI and web dashboard.** Both are clients of this API: a Rich-based terminal
interface for interactive use, and a single-page dashboard that consumes `/query/stream` and renders
the progress events this phase emits.
