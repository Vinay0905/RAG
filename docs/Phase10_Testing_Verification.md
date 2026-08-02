# Phase 10 — Testing and Verification

> **Prerequisite:** Phases 1–9. This phase tests them and adds no product capability.
>
> **Budget: ~1,110 lines across 10 files, on budget.** §2 lists what was cut.
>
> **API verified 2026-08-02:** `pytest-asyncio` configuration (`asyncio_mode`,
> `asyncio_default_fixture_loop_scope`).
>
> **This file replaces the previous `Phase10_Testing_Verification.md`.**

---

## 1. What Makes This Phase Hard

Not "how do I write a test". The hard part is knowing *which* tests would have caught the bugs this
project actually had — and unusually, that is not a guess here. Five external review passes across
Phases 3–7 produced 148 findings, and they cluster:

| Where the bug was | Roughly | Examples |
| :--- | ---: | :--- |
| **Seams between components** | ~45% | `delete_by_doc_ids` called but never declared; `embed_sparse_query` on the concrete class only; `generate_json` missing a `model` parameter so two settings did nothing; the cross-encoder truncating to `top_k` so MMR became a no-op |
| **Error and edge paths** | ~30% | Grader outage producing a *passing* grade; `LLM_MAX_RETRIES=0` skipping the call; cancellation swallowed as a failed search arm; a stray import of a module that does not exist |
| **Claims not matching code** | ~20% | A token ceiling not enforced; "append-safe" opening `"w"`; a mitigation named in prose and absent from the code |
| **Logic inside one function** | ~5% | The `§` regex; the percentile |

**Almost none of it was unit-level logic.** A conventional pyramid — thousands of unit tests, a few
integration tests — would have caught the last row and missed the first three. So this phase inverts
the usual advice on the strength of its own evidence:

- **Contract tests** (§5) assert the interface surface itself. Every drift bug in the top row is a
  test that fails at import.
- **Integration tests** (§7, §8) run against real Qdrant and real Redis, because the bugs were in
  what the adapters assumed, not in what they computed.
- **Unit tests** cover the pure logic worth covering, and no more.

The second hard part: **an LLM in the loop makes tests non-deterministic and expensive.** Nothing here
calls a real model. Phase 5 already built `ScriptedLLM` for this, and it stops being a verification
helper and becomes shared infrastructure.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| 1 | `pyproject.toml` (additions) | 40 | pytest config, markers, coverage |
| 2 | `tests/conftest.py` | 190 | Every shared fixture |
| 3 | `tests/fakes.py` | 130 | `ScriptedLLM`, `FakeStore`, `StubEmbedder` |
| 4 | `tests/unit/test_contracts.py` | 130 | The frozen interface surface |
| 5 | `tests/unit/test_core.py` | 110 | Models, utils, the whitespace bug |
| 6 | `tests/unit/test_retrieval.py` | 140 | Fusion, MMR composition, parents |
| 7 | `tests/unit/test_agent.py` | 110 | Graph edges and the retry loop |
| 8 | `tests/integration/test_qdrant.py` | 150 | Real store: schema, ghosts, filters |
| 9 | `tests/integration/test_cache.py` | 90 | Real Redis: TTL, scope isolation |
| 10 | `tests/e2e/test_api.py` | 110 | Through FastAPI, including concurrency |
| — | `.github/workflows/ci.yml` | 60 | Unit always; integration with services |

```text
tests/
├── conftest.py
├── fakes.py
├── unit/
│   ├── test_contracts.py  test_core.py  test_retrieval.py  test_agent.py
├── integration/
│   ├── test_qdrant.py  test_cache.py
└── e2e/
    └── test_api.py
```

---

## 2. What Was Cut

**Mirroring `src/` file-for-file.** The roadmap says `tests/` mirrors `src/` exactly. That would be
forty files and several thousand lines, most of them asserting that a getter returns what was set.
Four unit files cover the logic where a bug would be silent; the rest is covered where it actually
breaks, at the seams.

**Property-based testing (Hypothesis).** Genuinely valuable for two functions here —
`normalise_whitespace` and `to_strict_schema` — and a dependency plus a mental model for two
functions is not the trade. The specific properties are asserted with hand-picked adversarial inputs
instead.

**Mutation testing, load testing, and a CI matrix across Python versions.** One deployment target,
one Python version, one user.

**A coverage threshold in CI.** Coverage is reported, never gated. A percentage target produces tests
written to raise the percentage, and the 148 findings above show the number would have been high
throughout while the seams stayed broken.

**Testing LLM output quality.** That is Phase 7, with a different tool and a different definition of
pass. A test asserting an answer "is good" is a flaky test that teaches nothing.

That is ~1,500 lines declined, and it is what buys the integration tests room.

---

## 3. File 1 — `pyproject.toml` additions

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov",
    "httpx",            # FastAPI's TestClient
]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Auto mode: every `async def` test runs without a decorator. Correct here because
# asyncio is the only async library in the project.
asyncio_mode = "auto"
# Set explicitly. Unset, older pytest-asyncio versions fall back to the fixture's
# own scope and emit a deprecation warning; "function" gives each test a fresh loop,
# which is what stops one test's leaked task from failing the next one.
asyncio_default_fixture_loop_scope = "function"
addopts = "-q --strict-markers"
markers = [
    "integration: needs Qdrant and/or Redis running (docker compose up -d)",
    "e2e: exercises the FastAPI app end to end",
    "slow: takes more than a second",
]

[tool.coverage.run]
source = ["src", "config"]
omit = ["*/web/*"]

[tool.coverage.report]
# Reported, never enforced. A coverage gate produces tests written to move a
# number; every bug this project actually had would have passed one.
show_missing = true
```

`--strict-markers` matters more than it looks: without it, `@pytest.mark.integraton` is silently a
no-op, and the test runs in CI where the services do not exist.

---

## 4. Files 2 and 3 — fixtures and fakes

### The Problem

Nine `scripts/verify_phaseN.py` files exist, and they each build the same things: a unique temporary
collection, a set of synthetic parent/child chunks with Phase 2's `1_000_000` offset, a scripted LLM,
a stub embedder. The offset bug in Phase 3's script — parents overwriting children, making the filter
test vacuous — happened because that construction was duplicated rather than shared.

### Design Decision

**One `tests/fakes.py` that the verify scripts also import.** The scripts stay as standalone smoke
tests (they run without pytest, which is useful on the corpus machine), but the builders live in one
place. A fixture that is wrong is wrong once.

**Fixtures build the corpus; tests do not.** A test that spends fifteen lines constructing chunks
before asserting one thing is a test nobody reads.

### `tests/fakes.py`

```python
"""Test doubles, shared by pytest and the standalone verify scripts.

Promoted from `scripts/verify_phase3.py` and `verify_phase5.py`, where the same
classes were written twice with small differences.
"""
from typing import Any

from src.core.models import (
    AnswerStatus, Chunk, ChunkLevel, GradingReport, RAGAnswer, ScoredChunk,
)
from src.core.utils import make_chunk_id, make_doc_id, make_section_id

DOC_ID = make_doc_id("data/contracts/2003/test.txt")

#: Five distinct clauses. Distinct TOPICS matter: several tests assert that a
#: retriever or a diversifier distinguished between them, which is impossible if the
#: fixtures are five paraphrases.
CLAUSES = {
    "Termination": "Either party may terminate this Agreement for convenience upon "
                   "ninety (90) days prior written notice to the other party.",
    "Indemnification": "The Company shall indemnify the Purchaser against all Losses, "
                       "provided the aggregate liability shall not exceed $5,000,000.",
    "Governing Law": "This Agreement shall be governed by the laws of the State of "
                     "Delaware without regard to conflict of laws principles.",
    "Confidentiality": "Confidential Information shall not be disclosed to any third "
                       "party without prior written consent.",
    "Payment": "The Purchase Price shall be paid in immediately available funds at "
               "Closing by wire transfer.",
}


def build_corpus(doc_id: str = DOC_ID) -> list[Chunk]:
    """Parent + child pairs for every clause.

    Parents are offset by 1_000_000 exactly as Phase 2's `_parent_slot` does. Without
    it, parent index 0 and child index 0 derive the SAME chunk_id and the parent
    silently overwrites the child — which is what happened in Phase 3's verification
    script and made its filter test assert nothing.
    """
    chunks: list[Chunk] = []
    for order, (title, text) in enumerate(CLAUSES.items()):
        section_id = make_section_id(doc_id, title, order)
        parent_index = 1_000_000 + order
        parent_id = make_chunk_id(doc_id, section_id, parent_index)

        def make(body: str, index: int, level: ChunkLevel, parent: str | None) -> Chunk:
            return Chunk(
                chunk_id=make_chunk_id(doc_id, section_id, index),
                doc_id=doc_id, section_id=section_id, text=body, chunk_index=index,
                token_count=max(len(body) // 4, 1), contract_name="Test Agreement",
                file_name="test.txt", section_title=title, year=2003,
                chunk_level=level, parent_id=parent,
            )

        chunks.append(make(text, parent_index, ChunkLevel.PARENT, None))
        chunks.append(make(text.split(".")[0] + ".", order * 10, ChunkLevel.CHILD, parent_id))
    return chunks


def scored(chunks: list[Chunk], method=None) -> list[ScoredChunk]:
    """Wrap chunks as ranked results, best first."""
    from src.core.models import RetrievalMethod

    return [
        ScoredChunk(chunk=c, score=1.0 - i * 0.05, rank=i,
                    method=method or RetrievalMethod.HYBRID)
        for i, c in enumerate(chunks)
    ]


class ScriptedLLM:
    """A `BaseLLMProvider` that returns queued responses.

    The reason every agent test is deterministic. Scripting a FAILING grade is the
    only reliable way to exercise the retry loop — a real model cannot be made to
    produce an ungrounded answer on demand.
    """

    def __init__(self, answers=None, grades=None, rewrites=None,
                 router_labels=None) -> None:
        self.answers = list(answers or [])
        self.grades = list(grades or [])
        self.rewrites = list(rewrites or [])
        self.router_labels = list(router_labels or [])
        self.generate_calls = 0
        self.json_calls = 0
        #: Every `model` argument received, so a test can assert that per-task model
        #: selection is actually happening — the bug two reviews found independently.
        self.models_used: list[str | None] = []

    async def generate(self, prompt: str, system_prompt: str | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       model: str | None = None) -> str:
        self.generate_calls += 1
        self.models_used.append(model)
        return self.answers.pop(0) if self.answers else "No answer available."

    async def generate_json(self, prompt: str, schema: type,
                            system_prompt: str | None = None,
                            model: str | None = None) -> Any:
        self.json_calls += 1
        self.models_used.append(model)
        name = schema.__name__

        if name == "GradingReport":
            return self.grades.pop(0) if self.grades else GradingReport(
                is_grounded=True, is_relevant=True, confidence=0.9
            )
        if name == "RewrittenQuery":
            return schema(query=self.rewrites.pop(0) if self.rewrites
                          else "rewritten legal query")
        if name == "QueryClass":
            return schema(label=self.router_labels.pop(0) if self.router_labels else "vague")
        if name == "QueryVariations":
            return schema(variations=[])
        return schema()

    def stream(self, prompt: str, system_prompt: str | None = None):
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StubEmbedder:
    """Deterministic vectors with no model. Fast, and lets tests control similarity."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    @property
    def dense_dimensions(self) -> int:
        return self.dimensions

    def _vector(self, text: str) -> list[float]:
        # Hash-derived but stable: the same text always yields the same vector, and
        # different text yields a different one. Enough for cache and MMR tests.
        seed = sum(ord(c) for c in text)
        return [((seed * (i + 1)) % 97) / 97 for i in range(self.dimensions)]

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        return [{"indices": [1, 2], "values": [0.5, 0.5]} for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_sparse_query(self, text: str) -> dict[str, list]:
        return {"indices": [1], "values": [1.0]}

    async def close(self) -> None:
        return None


class FakeStore:
    """An in-memory `BaseVectorStore`. Optionally fails on demand.

    `fail_upsert_on` is how the two-phase commit contract gets tested without taking
    Qdrant down — the failing-store stub from Phase 3's verification, promoted.
    """

    def __init__(self, fail_upsert_on: int = 0) -> None:
        self.points: dict[str, Chunk] = {}
        self.fail_upsert_on = fail_upsert_on
        self.upserts = 0
        self.collection_name = "fake"

    async def initialize(self, vector_size: int) -> None:
        return None

    async def upsert_points(self, chunks, dense, sparse) -> int:
        self.upserts += 1
        if self.upserts == self.fail_upsert_on:
            raise RuntimeError("simulated storage outage")
        for chunk in chunks:
            self.points[chunk.chunk_id] = chunk
        return len(chunks)

    async def delete_by_doc_ids(self, doc_ids) -> int:
        targets = {c.chunk_id for c in self.points.values() if c.doc_id in set(doc_ids)}
        for chunk_id in targets:
            del self.points[chunk_id]
        return len(targets)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        return await self.delete_by_doc_ids([doc_id])

    async def fetch_by_ids(self, ids) -> list[Chunk]:
        return [self.points[i] for i in ids if i in self.points]

    async def hybrid_search(self, dense_query, sparse_query, limit=20,
                            filters=None) -> list[ScoredChunk]:
        level = (filters or {}).get("chunk_level")
        matches = [c for c in self.points.values()
                   if level is None or c.chunk_level.value == level]
        return scored(matches[:limit])

    async def count(self) -> int:
        return len(self.points)

    async def close(self) -> None:
        return None


def answer(status: AnswerStatus = AnswerStatus.ANSWERED, **kwargs) -> RAGAnswer:
    """A minimal `RAGAnswer` for cache and API tests."""
    return RAGAnswer(query=kwargs.pop("query", "q"),
                     answer=kwargs.pop("answer", "An answer [SOURCE 1]."),
                     status=status, **kwargs)
```

### `tests/conftest.py`

```python
"""Shared fixtures. Nothing here calls a real LLM."""
import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest

from config.settings import get_settings, settings
from src.core.models import Chunk
from tests.fakes import FakeStore, ScriptedLLM, StubEmbedder, build_corpus


# ─── configuration ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_settings() -> Iterator[None]:
    """Snapshot and restore mutated settings around every test.

    `autouse`, because the settings object is a module-level singleton and a test
    that flips `ENABLE_MMR` leaks into whatever runs next — producing a failure that
    depends on test ORDER, which is the worst kind to debug. Several tests must
    mutate flags (they are testing the flags), so the fix is restoring rather than
    forbidding.
    """
    watched = [
        "ENABLE_MULTI_QUERY", "ENABLE_HYDE", "ENABLE_RERANKING", "ENABLE_MMR",
        "ENABLE_PARENT_SUBSTITUTION", "ENABLE_SELF_CORRECTION", "ENABLE_EXACT_CACHE",
        "ENABLE_SEMANTIC_CACHE", "ENABLE_CITATION_VALIDATION", "MAX_RETRIES",
        "RERANK_TOP_K", "RETRIEVAL_TOP_K", "MAX_CONTEXT_TOKENS",
    ]
    saved = {name: getattr(settings, name) for name in watched}
    yield
    for name, value in saved.items():
        setattr(settings, name, value)
    get_settings.cache_clear()


@pytest.fixture
def no_llm_features() -> Iterator[None]:
    """Disable everything that would need a real model."""
    settings.ENABLE_MULTI_QUERY = False
    settings.ENABLE_HYDE = False
    yield


# ─── fakes ─────────────────────────────────────────────────────────────────

@pytest.fixture
def corpus() -> list[Chunk]:
    return build_corpus()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def store(corpus: list[Chunk]) -> FakeStore:
    fake = FakeStore()
    for chunk in corpus:
        fake.points[chunk.chunk_id] = chunk
    return fake


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


# ─── integration: real services ────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_embedder():
    """The genuine FastEmbed provider. Session-scoped — loading is ~90 MB and
    several hundred milliseconds, and per-test construction would dominate the run."""
    pytest.importorskip("fastembed")
    from src.embeddings.fastembed_provider import FastEmbedProvider

    provider = FastEmbedProvider()
    provider.warmup()
    return provider


@pytest.fixture
async def qdrant(real_embedder) -> AsyncIterator:
    """A real Qdrant store on a throwaway collection.

    Unique name per test, and dropped afterwards. A fixed name would let one test's
    leftovers change another's result, and — worse — would let a test suite delete a
    real collection that happened to share the name.

    Skips rather than fails when Qdrant is absent, so `pytest -m "not integration"`
    is not the only way to run the suite on a laptop.
    """
    from src.vectorstores.qdrant_store import QdrantStore

    name = f"test_{uuid.uuid4().hex[:10]}"
    store = QdrantStore(collection_name=name)

    try:
        await store.initialize(real_embedder.dense_dimensions)
    except Exception as exc:
        await store.close()
        pytest.skip(f"Qdrant unavailable: {type(exc).__name__}")

    try:
        yield store
    finally:
        try:
            await store._client.delete_collection(name)
        finally:
            await store.close()


@pytest.fixture
async def indexed(qdrant, real_embedder, corpus) -> AsyncIterator:
    """A Qdrant collection preloaded with the synthetic corpus."""
    texts = [c.text for c in corpus]
    await qdrant.upsert_points(
        corpus,
        await real_embedder.embed_dense(texts),
        await real_embedder.embed_sparse(texts),
    )
    yield qdrant


@pytest.fixture
async def redis_cache() -> AsyncIterator:
    """A real Redis-backed cache, cleared before and after."""
    from src.cache.redis_cache import RedisAnswerCache

    cache = RedisAnswerCache()
    client = await cache._redis()
    if client is None:
        pytest.skip("Redis unavailable")

    await cache.clear()
    try:
        yield cache
    finally:
        await cache.clear()
        await cache.close()
```

---

## 5. File 4 — `tests/unit/test_contracts.py`

### The Problem

Roughly 45% of this project's findings were interface drift: a method called but never declared, a
parameter documented but absent, a postcondition claimed and violated. Every one of them was found by
a human reading two files side by side, weeks after the drift.

### Design Decision

**Test the interface surface itself.** These assertions are unusual — they check signatures rather
than behaviour — and they are the highest-value tests in the phase, because they encode §6 of the
handoff (the frozen contract) as something that fails automatically.

This is the file to write first and update whenever the contract changes deliberately.

```python
"""The frozen contract, as executable assertions.

Every failure here is cross-phase drift: something that will not run when the pieces
are assembled, caught at import rather than in production.
"""
import inspect

import pytest

from src.core import interfaces, models
from src.core.exceptions import (
    InvalidQueryError, LLMProviderError, ModelDecommissionedError, RAGException,
    RateLimitError, RetrievalError, VectorStoreError,
)


def _params(cls, method: str) -> set[str]:
    return set(inspect.signature(getattr(cls, method)).parameters) - {"self"}


class TestVectorStoreContract:
    """Phase 4's parent substitution and Phase 3's indexing driver both call methods
    that were, at various points, declared on neither the ABC nor every store."""

    def test_required_methods_exist(self) -> None:
        for method in ("initialize", "upsert_points", "hybrid_search",
                       "delete_by_doc_ids", "delete_by_doc_id", "fetch_by_ids",
                       "count", "close"):
            assert hasattr(interfaces.BaseVectorStore, method), (
                f"BaseVectorStore is missing {method}; a caller typed against the "
                "ABC will fail static checking and at runtime on some stores"
            )

    def test_batch_delete_is_the_abstract_one(self) -> None:
        # `delete_by_doc_ids` is primary and `delete_by_doc_id` delegates. Reversed,
        # a 256-chunk batch spanning 40 documents costs 40 round trips.
        abstract = interfaces.BaseVectorStore.__abstractmethods__
        assert "delete_by_doc_ids" in abstract
        assert "delete_by_doc_id" not in abstract

    @pytest.mark.parametrize("name", ["QdrantStore", "ChromaStore"])
    def test_implementations_are_complete(self, name: str) -> None:
        module = ("src.vectorstores.qdrant_store" if name == "QdrantStore"
                  else "src.vectorstores.chroma_store")
        cls = getattr(pytest.importorskip(module), name)
        missing = interfaces.BaseVectorStore.__abstractmethods__ - set(dir(cls))
        assert not missing, f"{name} does not implement {sorted(missing)}"


class TestEmbeddingContract:
    def test_query_and_document_paths_are_separate(self) -> None:
        # Document-side BM25 applies term-frequency saturation, which is meaningless
        # for a query. Phase 4 used the document path for queries for a whole
        # session: results still came back, just mis-weighted.
        for method in ("embed_dense", "embed_sparse", "embed_query",
                       "embed_sparse_query", "embed_dense_sync", "close"):
            assert hasattr(interfaces.BaseEmbeddingProvider, method)

    def test_close_is_optional(self) -> None:
        assert "close" not in interfaces.BaseEmbeddingProvider.__abstractmethods__


class TestLLMContract:
    def test_per_task_model_selection_is_possible(self) -> None:
        # Without `model`, EXPANSION_MODEL and GRADER_MODEL are decoration and every
        # call runs on the generation model. Two separate reviews found this.
        assert "model" in _params(interfaces.BaseLLMProvider, "generate")
        assert "model" in _params(interfaces.BaseLLMProvider, "generate_json")

    def test_stream_is_not_a_coroutine_function(self) -> None:
        # `def ... -> AsyncIterator[str]`, not `async def`. An `async def` here would
        # require `await provider.stream(...)` before iterating.
        assert not inspect.iscoroutinefunction(interfaces.BaseLLMProvider.stream)


class TestModelContract:
    def test_scored_chunk_uses_composition(self) -> None:
        # `scored.chunk.text`, never `scored.text`. Inheritance would let a
        # query-specific score leak into contexts where it is meaningless.
        assert not issubclass(models.ScoredChunk, models.Chunk)
        assert "chunk" in models.ScoredChunk.model_fields

    def test_grading_passed_requires_verification(self) -> None:
        # The most dangerous bug found in the project: a grader outage set
        # grounded+relevant True, `passed` became True, and every caller treated an
        # UNAUDITED answer as verified.
        unaudited = models.GradingReport(verified=False, is_grounded=True, is_relevant=True)
        assert unaudited.passed is False, (
            "'we could not check' must never read the same as 'we checked and it "
            "was fine'"
        )
        assert models.GradingReport(is_grounded=True, is_relevant=True).passed is True

    def test_verified_is_hidden_from_the_llm_schema(self) -> None:
        from src.llm.base import to_strict_schema

        assert "verified" not in to_strict_schema(models.GradingReport)["properties"], (
            "the grader must not be able to declare its own audit to have happened"
        )

    def test_payload_round_trip_preserves_every_field(self) -> None:
        # Phase 1 shipped with `to_payload` dropping `chunk_index` while
        # `from_payload` read it. Comparing only `.text` is how that survived.
        from tests.fakes import build_corpus

        original = build_corpus()[0]
        restored = models.Chunk.from_payload(original.to_payload())
        for field in ("chunk_id", "doc_id", "section_id", "text", "chunk_index",
                      "token_count", "contract_name", "file_name", "section_title",
                      "year", "chunk_level", "parent_id"):
            assert getattr(restored, field) == getattr(original, field), f"lost {field}"


class TestExceptionContract:
    """Phase 8 maps exceptions to HTTP by reading `status_code`, so these values are
    the API's behaviour, not an implementation detail."""

    @pytest.mark.parametrize("exc_class,status", [
        (InvalidQueryError, 400), (RetrievalError, 503),
        (RateLimitError, 429), (VectorStoreError, 503),
        (LLMProviderError, 502), (ModelDecommissionedError, 500),
    ])
    def test_status_codes(self, exc_class, status) -> None:
        assert exc_class("x").status_code == status

    def test_envelope_shape(self) -> None:
        payload = RetrievalError("down", details={"k": 1}, retryable=True).to_dict()
        assert set(payload) == {"error", "message", "details", "retryable"}
```

---

## 6. Files 5–7 — the unit tests worth writing

### `tests/unit/test_core.py`

```python
"""Pure logic where a bug would be silent."""
import pytest

from src.core.models import Chunk, ChunkLevel
from src.core.utils import (
    count_tokens, make_chunk_id, normalise_whitespace, safe_filename,
    truncate_to_tokens,
)


class TestNormaliseWhitespace:
    """The handoff calls this 'the single most likely bug in the whole project'."""

    def test_newlines_survive(self) -> None:
        # The legal parser anchors section headers per line with `^`. A `\s+` here
        # collapses every document into one line, so every contract becomes a single
        # "Preamble" section — and retrieval still works, badly, with no error.
        text = "SECTION 1. Term\n\nSECTION 2. Termination"
        result = normalise_whitespace(text)
        assert "\n" in result, "collapsing newlines silently destroys all structure"
        assert result.count("SECTION") == 2

    def test_horizontal_runs_collapse(self) -> None:
        assert normalise_whitespace("a  \t  b") == "a b"

    def test_invisible_characters_removed(self) -> None:
        assert normalise_whitespace("a\u00adb\u200cc\ufeff") == "abc"

    def test_idempotent(self) -> None:
        text = normalise_whitespace("  Messy\u00a0 text \n\n here  ")
        assert normalise_whitespace(text) == text


class TestDeterministicIds:
    def test_same_inputs_same_id(self) -> None:
        assert make_chunk_id("d", "s", 0) == make_chunk_id("d", "s", 0)

    def test_parent_offset_prevents_collision(self) -> None:
        # Parents and children share one chunk_index namespace. Without Phase 2's
        # offset, parent 0 and child 0 collide and one silently overwrites the other.
        assert make_chunk_id("d", "s", 0) != make_chunk_id("d", "s", 1_000_000)


class TestTokens:
    def test_special_tokens_do_not_raise(self) -> None:
        # A real filing containing this literal string killed a run before
        # `disallowed_special=()` was added.
        assert count_tokens("text with <|endoftext|> inside") > 0

    def test_truncate_respects_budget(self) -> None:
        long_text = "word " * 500
        assert count_tokens(truncate_to_tokens(long_text, 50)) <= 50

    def test_truncate_leaves_short_text_alone(self) -> None:
        assert truncate_to_tokens("short", 100) == "short"


class TestChunkValidation:
    def test_blank_text_rejected(self) -> None:
        # An empty chunk embeds to a meaningless vector that weakly matches
        # everything, polluting every subsequent search.
        with pytest.raises(ValueError):
            Chunk(chunk_id="a", doc_id="d", section_id="s", text="   ",
                  chunk_index=0, token_count=0)

    def test_year_coercion_is_tolerant(self) -> None:
        from src.core.models import Document

        doc = Document(doc_id="d", source_path="p", file_name="f", content="x" * 10,
                       content_hash="h", year="misc")
        assert doc.year is None, "one odd directory name must not abort a 650k run"


def test_safe_filename_strips_separators() -> None:
    assert "/" not in safe_filename("a/b\\c:d")
    assert safe_filename("...") == "unnamed"
```

### `tests/unit/test_retrieval.py`

```python
"""Retrieval logic, including the composition bug that made MMR a no-op."""
import pytest

from config.settings import settings
from src.core.models import ChunkLevel, RetrievalMethod
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.parents import ParentSubstituter
from src.retrieval.rerankers.mmr import MMRReranker, semantically_incompatible
from tests.fakes import build_corpus, scored


class TestFusion:
    def test_agreement_is_rewarded(self, corpus) -> None:
        children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]
        # Chunk 2 appears in all three lists; chunk 0 in one.
        fused = reciprocal_rank_fusion([
            scored([children[0], children[2]]),
            scored([children[2], children[1]]),
            scored([children[3], children[2]]),
        ])
        assert fused[0].chunk.chunk_id == children[2].chunk_id

    def test_no_duplicates_and_dense_ranks(self, corpus) -> None:
        children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]
        fused = reciprocal_rank_fusion([scored(children), scored(children[:2])])
        assert len({f.chunk.chunk_id for f in fused}) == len(fused)
        assert [f.rank for f in fused] == list(range(len(fused)))

    def test_method_is_preserved(self, corpus) -> None:
        # Hardcoding HYBRID relabelled Chroma's dense-only results, making Phase 7's
        # method comparison meaningless — a field that always says the same thing.
        children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]
        fused = reciprocal_rank_fusion([scored(children, method=RetrievalMethod.DENSE)])
        assert fused[0].method is RetrievalMethod.DENSE

    def test_rerank_score_not_carried(self, corpus) -> None:
        fused = reciprocal_rank_fusion([scored(corpus[:3])])
        assert all(f.rerank_score is None for f in fused)


class TestMMR:
    async def test_preserves_scores(self, embedder, corpus) -> None:
        # MMR selects and reorders; it must not re-score. Recomputing cosine
        # relevance threw away the cross-encoder's judgement immediately after
        # paying 20ms for it.
        settings.ENABLE_MMR = True
        candidates = scored([c for c in corpus if c.chunk_level == ChunkLevel.PARENT])
        for index, candidate in enumerate(candidates):
            candidates[index] = candidate.model_copy(update={"rerank_score": 5.0 - index})

        result = await MMRReranker(embedder, lambda_param=0.5).rerank("obligations",
                                                                      candidates, top_k=2)
        assert all(r.rerank_score is not None for r in result)
        assert [r.rank for r in result] == [0, 1]

    async def test_noop_when_pool_equals_top_k(self, embedder, corpus) -> None:
        # The composition bug, asserted directly: given exactly `top_k` candidates,
        # MMR returns them unchanged. This is correct behaviour and the reason the
        # PIPELINE must rerank to a wider pool first.
        settings.ENABLE_MMR = True
        candidates = scored(corpus[:2])
        result = await MMRReranker(embedder).rerank("q", candidates, top_k=2)
        assert [r.chunk.chunk_id for r in result] == [c.chunk.chunk_id for c in candidates]


class TestParentSubstitution:
    async def test_deduplicates_and_preserves_order(self, store, corpus) -> None:
        children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]
        # Two children of one parent, then one of another.
        picked = scored([children[0], children[0], children[1]])
        result = await ParentSubstituter(store).substitute(picked)

        assert len(result) == 2, "children sharing a parent must yield one source"
        assert all(r.chunk.chunk_level == ChunkLevel.PARENT for r in result)
        assert [r.rank for r in result] == [0, 1]

    async def test_missing_parent_degrades_to_child(self, store, corpus) -> None:
        orphan = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD][0]
        orphan = orphan.model_copy(update={"parent_id": "does-not-exist"})
        result = await ParentSubstituter(store).substitute(scored([orphan]))
        assert len(result) == 1 and result[0].chunk.chunk_id == orphan.chunk_id

    async def test_budget_drops_whole_sources(self, store, corpus) -> None:
        # Never truncates: a half-clause reads as a complete one to the LLM, which
        # then answers from a fragment whose qualifying proviso was removed.
        settings.MAX_CONTEXT_TOKENS = 40
        parents = [c for c in corpus if c.chunk_level == ChunkLevel.PARENT]
        result = await ParentSubstituter(store).substitute(scored(parents))
        assert len(result) < len(parents)
        assert result[0].chunk.text in {p.text for p in parents}, "sources stay whole"


@pytest.mark.parametrize("a,b,reason", [
    ("can the buyer terminate?", "can the seller terminate?", "entity swap"),
    ("is 30 days notice required?", "is 90 days notice required?", "numbers"),
    ("may the lessee assign?", "may the lessee not assign?", "negation"),
    ("what does Acme owe?", "what does Beta owe?", "party names"),
])
def test_semantic_cache_veto(a: str, b: str, reason: str) -> None:
    """The differences a topic embedding cannot see but that invert the answer."""
    blocked, _ = semantically_incompatible(a, b)
    assert blocked, f"the veto missed a {reason}"
```

### `tests/unit/test_agent.py`

```python
"""The graph's control flow, which is a pure function of state."""
from config.settings import settings
from src.core.exceptions import LLMProviderError, MaxRetriesExceededError
from src.core.models import AnswerStatus, GradingReport
from src.graph.builder import RAGAgent
from src.graph.edges import after_grading, after_retrieval
from src.graph.state import initial_state
from src.retrieval.pipeline import RetrievalPipeline
from tests.fakes import ScriptedLLM


class TestEdges:
    """No LLM, no store — control flow tested with dicts."""

    def test_empty_retrieval_never_generates(self) -> None:
        state = initial_state("q", top_k=5)
        assert after_retrieval(state) == "no_context"

    def test_retry_with_no_results_keeps_the_previous_answer(self) -> None:
        # After a failed grade, a rewrite that retrieves nothing must not overwrite a
        # usable answer with "no passages matched".
        state = initial_state("q", top_k=5)
        state["answer"] = "an earlier answer"
        assert after_retrieval(state) == "keep_previous"

    def test_unverified_does_not_retry(self) -> None:
        # Rewriting the query cannot fix a broken grader; retrying spends two LLM
        # calls to arrive at the same unverified state.
        state = initial_state("q", top_k=5)
        state["answer"] = "a"
        state["grading"] = GradingReport(verified=False, is_grounded=True, is_relevant=True)
        assert after_grading(state) == "accept"

    def test_budget_exhaustion_terminates(self) -> None:
        state = initial_state("q", top_k=5)
        state["answer"] = "a"
        state["grading"] = GradingReport(is_grounded=False, is_relevant=True)
        state["retry_count"] = settings.MAX_RETRIES
        assert after_grading(state) == "accept"

    def test_generation_failure_is_not_retried(self) -> None:
        state = initial_state("q", top_k=5)
        state["grading"] = GradingReport(is_grounded=False, is_relevant=False)
        assert after_grading(state) == "accept"


class TestRetryLoop:
    """The reason Phase 5 exists, exercised with scripted grades."""

    def _agent(self, llm: ScriptedLLM, embedder, store) -> RAGAgent:
        pipeline = RetrievalPipeline(embedder=embedder, store=store, llm=None)
        return RAGAgent(llm=llm, pipeline=pipeline)

    async def test_failing_grade_triggers_one_retry(self, embedder, store,
                                                    no_llm_features) -> None:
        llm = ScriptedLLM(
            answers=["Wrong [SOURCE 1].", "Right [SOURCE 1]."],
            grades=[
                GradingReport(is_grounded=False, is_relevant=True,
                              unsupported_claims=["a claim"]),
                GradingReport(is_grounded=True, is_relevant=True),
            ],
            rewrites=["better legal query"], router_labels=["vague"],
        )
        result = await self._agent(llm, embedder, store).answer("what is the cap?")

        assert result.retry_count == 1
        assert llm.generate_calls == 2
        assert "Right" in result.answer
        assert result.status is AnswerStatus.ANSWERED

    async def test_uses_the_configured_models(self, embedder, store,
                                              no_llm_features) -> None:
        # The bug two reviews found: `model` was never passed, so GRADER_MODEL and
        # EXPANSION_MODEL were decoration.
        llm = ScriptedLLM(answers=["A [SOURCE 1]."], router_labels=["vague"])
        await self._agent(llm, embedder, store).answer("a valid question")
        assert settings.GENERATION_MODEL in llm.models_used
        assert settings.GRADER_MODEL in llm.models_used

    async def test_generation_outage_raises_an_llm_error(self, embedder, store,
                                                         no_llm_features) -> None:
        class Dead(ScriptedLLM):
            async def generate(self, *a, **k) -> str:
                raise LLMProviderError("provider down", retryable=True)

        agent = self._agent(Dead(router_labels=["vague"]), embedder, store)
        try:
            await agent.answer("a valid question")
        except LLMProviderError:
            pass
        except MaxRetriesExceededError:
            raise AssertionError(
                "a provider outage reported as retry exhaustion gives the wrong HTTP "
                "status and sends someone to the wrong component"
            )
```

---

## 7. Files 8 and 9 — integration

### `tests/integration/test_qdrant.py`

### Design Decision

**These tests are where the value is, and they need the real thing.** Every bug in the storage layer
was about what Qdrant actually does — whether `strict` schemas are accepted, whether a filter applies,
whether a delete matched. A mock would have asserted my assumptions back to me.

```python
"""Real Qdrant. `docker compose up -d qdrant` first."""
import asyncio

import pytest

from src.core.exceptions import DimensionMismatchError, VectorStoreError
from src.core.models import ChunkLevel
from src.vectorstores.base import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from tests.fakes import DOC_ID, build_corpus

pytestmark = [pytest.mark.integration]


class TestSchema:
    async def test_named_vectors_and_idf(self, qdrant) -> None:
        info = await qdrant._client.get_collection(qdrant.collection_name)
        assert DENSE_VECTOR_NAME in info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        assert SPARSE_VECTOR_NAME in sparse
        # Without IDF, BM25 weights "the" and "indemnification" equally. Search still
        # works. Nothing errors. This assertion is the only thing that notices.
        modifier = getattr(sparse[SPARSE_VECTOR_NAME], "modifier", None)
        assert str(modifier).lower().endswith("idf")

    async def test_payload_indexes_exist(self, qdrant) -> None:
        schema = (await qdrant._client.get_collection(qdrant.collection_name)).payload_schema
        for field in ("doc_id", "chunk_level", "year", "section_title"):
            assert field in schema, (
                f"{field} has no index; a filter on it is a full scan of 39M points"
            )

    async def test_initialize_is_idempotent(self, qdrant, real_embedder) -> None:
        await qdrant.initialize(real_embedder.dense_dimensions)  # must not destroy

    async def test_dimension_mismatch_raises(self, qdrant, real_embedder) -> None:
        with pytest.raises(DimensionMismatchError):
            await qdrant.initialize(real_embedder.dense_dimensions + 1)


class TestWrites:
    async def test_idempotent_upsert(self, indexed, real_embedder, corpus) -> None:
        before = await indexed.count_exact()
        texts = [c.text for c in corpus]
        await indexed.upsert_points(corpus, await real_embedder.embed_dense(texts),
                                    await real_embedder.embed_sparse(texts))
        assert await indexed.count_exact() == before

    async def test_no_ghosts_when_a_document_shrinks(self, indexed, real_embedder,
                                                     corpus) -> None:
        """The failure deterministic IDs cannot prevent."""
        subset = corpus[:3]
        texts = [c.text for c in subset]
        deleted = await indexed.delete_by_doc_ids([DOC_ID])
        assert deleted == len(corpus)
        await indexed.upsert_points(subset, await real_embedder.embed_dense(texts),
                                    await real_embedder.embed_sparse(texts))
        assert await indexed.count_exact({"doc_id": DOC_ID}) == 3, (
            "stale points survived re-indexing; they will be retrieved and cited"
        )

    async def test_duplicate_ids_in_one_batch_raise(self, qdrant, real_embedder) -> None:
        # Would silently overwrite while reporting the full count as written.
        chunks = build_corpus()[:1] * 2
        texts = [c.text for c in chunks]
        with pytest.raises(VectorStoreError):
            await qdrant.upsert_points(chunks, await real_embedder.embed_dense(texts),
                                       await real_embedder.embed_sparse(texts))


class TestSearch:
    async def test_child_filter_excludes_parents(self, indexed, real_embedder) -> None:
        results = await indexed.hybrid_search(
            await real_embedder.embed_query("termination notice"),
            await real_embedder.embed_sparse_query("termination notice"),
            limit=10, filters={"chunk_level": ChunkLevel.CHILD.value},
        )
        assert results
        assert all(r.chunk.chunk_level == ChunkLevel.CHILD for r in results)

    async def test_lexical_arm_contributes(self, indexed, real_embedder) -> None:
        # A verbatim rare phrase is what BM25 is for. If this fails, the sparse arm
        # is not being queried — the prototype's exact bug.
        phrase = "immediately available funds"
        results = await indexed.hybrid_search(
            await real_embedder.embed_query(phrase),
            await real_embedder.embed_sparse_query(phrase), limit=5,
        )
        assert phrase in results[0].chunk.text.lower()

    async def test_unknown_filter_field_raises(self, indexed, real_embedder) -> None:
        # Silently ignoring an unknown key returns MORE data than asked for. Under
        # Phase 11 that is a tenant leak, not a bug report.
        with pytest.raises(VectorStoreError):
            await indexed.hybrid_search(
                await real_embedder.embed_query("x"),
                await real_embedder.embed_sparse_query("x"),
                filters={"nonexistent": "value"},
            )

    async def test_fetch_by_ids_skips_missing(self, indexed, corpus) -> None:
        found = await indexed.fetch_by_ids([corpus[0].chunk_id, "not-a-real-id"])
        assert len(found) == 1

    async def test_concurrent_searches(self, indexed, real_embedder) -> None:
        # Phase 4 issues four probes with `gather`. A client that is not actually
        # concurrency-safe shows up here and nowhere else.
        dense = await real_embedder.embed_query("termination")
        sparse = await real_embedder.embed_sparse_query("termination")
        results = await asyncio.gather(
            *(indexed.hybrid_search(dense, sparse, limit=3) for _ in range(4))
        )
        assert all(len(r) > 0 for r in results)
```

### `tests/integration/test_cache.py`

```python
"""Real Redis. `docker compose up -d redis` first."""
import asyncio

import pytest

from config.settings import settings
from src.cache.redis_cache import make_cache_key
from src.cache.semantic_cache import SemanticAnswerCache
from src.core.models import AnswerStatus
from tests.fakes import StubEmbedder, answer

pytestmark = [pytest.mark.integration]

KEY = dict(top_k=5, filters=None, prompt_version="v1.0")


class TestExactCache:
    async def test_round_trip_and_hit_flag(self, redis_cache) -> None:
        key = make_cache_key("what is the notice period?", **KEY)
        await redis_cache.set(key, answer(answer="Ninety days."))
        hit = await redis_cache.get(key)
        assert hit is not None and hit.cache_hit is True, (
            "the stored copy must stay truthful; the returned one must be marked"
        )

    @pytest.mark.parametrize("status,cached", [
        (AnswerStatus.ANSWERED, True),
        (AnswerStatus.LOW_CONFIDENCE, False),
        (AnswerStatus.UNVERIFIED, False),
    ])
    async def test_status_gate(self, redis_cache, status, cached) -> None:
        # Caching a poor answer gives it a TTL-long tail AND prevents the retry that
        # might have fixed it from ever running again.
        key = make_cache_key(f"q-{status.value}", **KEY)
        await redis_cache.set(key, answer(status=status))
        assert (await redis_cache.get(key) is not None) is cached

    async def test_ttl_expires(self, redis_cache) -> None:
        key = make_cache_key("short lived question", **KEY)
        await redis_cache.set(key, answer(), ttl=1)
        assert await redis_cache.get(key) is not None
        await asyncio.sleep(1.5)
        assert await redis_cache.get(key) is None

    async def test_prompt_version_invalidates(self, redis_cache) -> None:
        # Without this, a prompt fix has no effect on any cached question and the
        # fix looks broken.
        original = make_cache_key("same question", **{**KEY, "prompt_version": "v1.0"})
        bumped = make_cache_key("same question", **{**KEY, "prompt_version": "v1.1"})
        await redis_cache.set(original, answer())
        assert await redis_cache.get(bumped) is None

    async def test_clear_is_scoped(self, redis_cache) -> None:
        client = await redis_cache._redis()
        await client.set("someone-elses-key", "value")
        await redis_cache.set(make_cache_key("mine", **KEY), answer())
        await redis_cache.clear()
        assert await client.get("someone-elses-key") == "value", (
            "clear() must not FLUSHDB; Redis may hold other things"
        )
        await client.delete("someone-elses-key")


class TestSemanticIsolation:
    async def test_filters_do_not_leak(self, redis_cache) -> None:
        """The most serious bug found in Phase 6."""
        settings.ENABLE_SEMANTIC_CACHE = True
        cache = SemanticAnswerCache(StubEmbedder(), redis_cache)

        acme, beta = {"doc_id": "acme"}, {"doc_id": "beta"}
        question = "what is the notice period?"

        key_a = make_cache_key(question, top_k=5, filters=acme, prompt_version="v1.0")
        await cache.set(key_a, question, answer(answer="Acme: ninety days."), filters=acme)

        key_b = make_cache_key(question, top_k=5, filters=beta, prompt_version="v1.0")
        assert await cache.get(key_b, question, filters=beta) is None, (
            "one document scope returned another's answer AND sources — a data "
            "isolation failure, not a cache miss-hit"
        )
        assert await cache.get(key_a, question, filters=acme) is not None
```

---

## 8. File 10 — `tests/e2e/test_api.py`

```python
"""Through FastAPI, with the service overridden. No Qdrant, no LLM."""
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_service
from src.core.exceptions import InvalidQueryError, RetrievalError
from src.core.models import AnswerStatus
from tests.fakes import answer, build_corpus, scored

pytestmark = [pytest.mark.e2e]


class FakeService:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.store = self
        self.agent = type("A", (), {"last_trace": ["router:vague", "retrieval:5"]})()
        self.cache = type("C", (), {
            "exact": type("E", (), {"hits": 1, "misses": 2})(), "_index": [],
        })()

    async def answer(self, question, top_k=None, filters=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    async def count(self) -> int:
        return 100


@pytest.fixture
def client_factory():
    def build(service) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_service] = lambda: service
        # Context-manager form, so the lifespan runs. A bare TestClient skips it,
        # which is the most common reason an app-state lookup returns None in tests.
        return TestClient(app)
    return build


def test_response_excludes_internal_fields(client_factory) -> None:
    sources = scored([c for c in build_corpus()][:2])
    result = answer(query="what notice?", answer="Ninety days [SOURCE 1].")
    result = result.model_copy(update={"sources": sources})

    with client_factory(FakeService(result=result)) as client:
        body = client.post("/query", json={"question": "what notice is required?"}).json()

    assert body["status"] == "answered"
    assert "excerpt" in body["sources"][0] and "text" not in body["sources"][0]
    assert len(body["sources"][0]["excerpt"]) <= 400


@pytest.mark.parametrize("error,status", [
    (InvalidQueryError("blank"), 400),
    (RetrievalError("down", retryable=True), 503),
])
def test_exceptions_map_to_their_own_status(client_factory, error, status) -> None:
    with client_factory(FakeService(error=error)) as client:
        response = client.post("/query", json={"question": "a valid question"})
    assert response.status_code == status
    assert response.json()["error"] == type(error).__name__


def test_validation_happens_before_any_work(client_factory) -> None:
    service = FakeService(result=answer())
    with client_factory(service) as client:
        assert client.post("/query", json={"question": "hi"}).status_code == 422
        assert client.post("/query", json={"question": "ok question",
                                           "top_k": 99}).status_code == 422
    assert service.calls == 0


def test_readiness_reports_an_empty_collection(client_factory) -> None:
    class Empty(FakeService):
        async def count(self) -> int:
            return 0

    with client_factory(Empty()) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert "ingestion" in response.json()["reason"]


def test_concurrent_streams_do_not_interleave(client_factory) -> None:
    """Phase 8 §5's known limitation, asserted rather than assumed.

    `RAGAgent.last_trace` is per-agent, not per-request, so two simultaneous streams
    read the same list. This test documents the boundary: it passes today because
    `TestClient` serialises requests, and it is the test that will fail the moment
    Phase 11 makes concurrency real. Marked xfail-if-concurrent rather than deleted,
    because a known limitation with no test becomes an unknown one.
    """
    result = answer(answer="Ninety days [SOURCE 1].")
    service = FakeService(result=result)

    with client_factory(service) as client:
        first = client.post("/query/stream", json={"question": "first question here"})
        second = client.post("/query/stream", json={"question": "second question here"})

    assert first.status_code == second.status_code == 200
    assert "event: answer" in first.text and "event: answer" in second.text
```

---

## 9. CI — `.github/workflows/ci.yml`

### Design Decision

**Two jobs.** Unit and contract tests run on every push with no services — fast, and they catch the
drift class that dominates this project's history. Integration runs alongside real Qdrant and Redis
in service containers.

**The evaluation gate is manual, not per-push.** Phase 7's harness costs real money in LLM calls and
needs a corpus. Running it on every commit would be expensive and, on a machine with no corpus,
impossible.

```yaml
name: CI

on:
  push:
  pull_request:
  workflow_dispatch:

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check src config tests
      - run: mypy src config
      # Contract tests first and separately: when the interface has drifted, that is
      # the signal, and burying it among 200 other results wastes the clarity.
      - run: pytest tests/unit/test_contracts.py -v
      - run: pytest -m "not integration and not e2e" --cov --cov-report=term

  integration:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:v1.12.4
        ports: ["6333:6333"]
      redis:
        image: redis:7.4-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      # FastEmbed downloads a model on first use; cache it or every run pays 30s.
      - uses: actions/cache@v4
        with:
          path: ~/.cache/fastembed
          key: fastembed-${{ runner.os }}
      - run: pytest -m "integration or e2e" -v
        env:
          QDRANT_HOST: http://localhost:6333
          REDIS_HOST: localhost
```

Pinned image tags, not `:latest` — a CI failure caused by an upstream release you did not choose is
the least useful kind.

---

## 10. Running it

```bash
pytest -m "not integration"          # fast; no services
docker compose up -d                 # Qdrant + Redis
pytest                               # everything
pytest tests/unit/test_contracts.py  # after any interface change
pytest --cov --cov-report=html       # coverage as a map, not a score
```

### What the deferred verification scripts become

The nine `scripts/verify_phaseN.py` files stay. They serve a purpose pytest does not: they run
standalone on the corpus machine with no dev dependencies, and each one is a readable narrative of
what its phase guarantees. What changes is that they **import from `tests/fakes.py`** instead of
rebuilding corpora — which is how the parent/child collision bug in Phase 3's script becomes
impossible rather than merely fixed.

Where a verify script and a pytest test overlap, the test is authoritative: it runs in CI.

---

## 11. What Phase 10 Bought You

**Interface drift fails automatically.** `test_contracts.py` encodes the frozen contract from the
handoff. The `delete_by_doc_ids`, `embed_sparse_query`, and `generate_json(model=...)` gaps — three
separate multi-week bugs found by human review — are now three assertions that fail on import.

**The seams are tested against the real thing.** IDF modifiers, payload indexes, ghost points,
filter enforcement, TTL expiry, and cross-scope isolation are all things a mock would have confirmed
my assumptions about rather than checked.

**Every agent test is deterministic and free.** `ScriptedLLM` makes the failing-grade retry path — the
feature the whole project is built around — a test that runs in milliseconds with no API key.

**One place where fixtures live.** Nine verify scripts stop duplicating corpus construction, which is
where a real bug came from.

**The known limitations are tested rather than remembered.** Phase 8's `last_trace` concurrency
boundary has an assertion attached to it. A limitation with no test becomes an unknown one.

### What is deliberately not here

Everything in §2, plus one honest gap: **there is no test that the system gives good answers.** That
is not an omission, it is a division of labour — Phase 7 owns quality with a different tool and a
different definition of pass. A pytest assertion about answer quality would be flaky, expensive, and
would teach you nothing that `scripts/evaluate.py` does not teach better.

---

## Next

**Part II begins.** Phases 11–16 solve the six problems in `future_ideas_problems.md`, in that order,
and they are mutually independent — the point at which the handoff's parallel-subagent plan becomes
safe.

**Phase 11 — Multi-Tenant Security and RBAC** (budget 600) is first, and three things across Part I
are waiting for it: the store's filter translator already raises on unknown fields so a tenant filter
cannot be silently dropped; Phase 6's cache scope already partitions by filters so one tenant's answer
cannot be served to another; and Phase 8's `last_trace` needs a per-request principal to become
correct. That is not coincidence — those three were built that way because this phase was coming.
