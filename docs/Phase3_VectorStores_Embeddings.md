# Phase 3 — Embeddings and Vector Stores

> **Prerequisite:** Phases 1 and 2 complete. This phase imports `Chunk`, `ScoredChunk`,
> `RetrievalMethod`, `ChunkLevel`, the `Base*` interfaces, `settings`, `logger`, `telemetry`, and the
> exception hierarchy. It consumes `IngestionPipeline.run()` from Phase 2.
>
> **Budget:** ~1,450 lines of Python across 9 files. (Budgeted at 800 in the roadmap. The overrun is
> the ninth file — the indexing driver — plus the retry, validation, and schema-verification code that
> a store adapter needs to be trustworthy. See the roadmap's calibration note.)
>
> **Depends on:** Phase 1 for models and interfaces, Phase 2 for the `ChunkBatch` stream. Phase 4
> depends on this one for `hybrid_search`; Phase 15 depends on it for `delete_by_doc_id`.
>
> **This file replaces the previous `Phase3_VectorStores_Embeddings.md` entirely.** The old draft
> exposed synchronous, `dict`-based interfaces, used the deprecated `recreate_collection`, and had no
> delete-before-upsert. Do not consult it.

---

## 1. What Makes This Phase Hard

Phase 2 ends with `Chunk` objects in memory. This phase turns them into vectors, puts them in a
database, and gets them back out. Four things make that harder than it sounds, and every design
decision below traces to one of them.

**Two vectors per chunk, not one.** Dense embeddings understand meaning but are blind to exact
tokens; a search for "Section 7.02" against a dense index returns clauses that are *about* the same
subject as Section 7.02, which is not what was asked. Sparse BM25 vectors match tokens exactly but
understand nothing. Legal retrieval needs both, in the same query, fused. That means named vectors,
a fusion query, and two embedding models kept in lockstep.

**Alignment is invisible when it breaks.** `upsert_points(chunks, dense, sparse)` zips three lists.
If the embedder ever returns 63 vectors for 64 texts, every chunk from that point on gets the
*previous chunk's* vector. Nothing raises. The index is quietly, permanently wrong, and the only
symptom is that retrieval feels slightly stupid. Most of the validation in this phase exists to make
that impossible rather than merely unlikely.

**Re-ingestion leaves ghosts.** Phase 1 established this and Phase 2's handoff makes it mandatory:
deterministic UUIDv5 IDs make an *unchanged* re-ingest idempotent, but they do not garbage-collect. A
document that shrinks from 400 chunks to 300 leaves 100 stale points that still get retrieved and
cited. Every write in this phase is therefore delete-then-upsert.

**The consumer owns the commit.** Phase 2 deliberately refuses to mark documents as ingested, because
ingestion does not know whether storage succeeded. This phase is the party that knows. If the
indexing driver gets the transaction boundary wrong, a Qdrant outage permanently marks thousands of
documents as indexed and nothing ever tells you.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| | **Part A — text → vectors** | | |
| 1 | `src/embeddings/base.py` | 130 | Validation, batching, the sync/async bridge |
| 2 | `src/embeddings/fastembed_provider.py` | 230 | Local ONNX dense + BM25 sparse |
| 3 | `src/embeddings/openai_provider.py` | 170 | Network dense, local sparse |
| 4 | `src/embeddings/factory.py` | 70 | Provider selection |
| | **Part B — vectors → database** | | |
| 5 | `src/vectorstores/base.py` | 140 | Shared names, alignment checks, payload → `ScoredChunk` |
| 6 | `src/vectorstores/qdrant_store.py` | 400 | The real store: named vectors, RRF, delete-by-filter |
| 7 | `src/vectorstores/chroma_store.py` | 160 | Dense-only fallback that proves the abstraction |
| 8 | `src/vectorstores/factory.py` | 60 | Store selection, deliberately uncached |
| | **Part C — the two halves joined** | | |
| 9 | `src/pipelines/indexing_pipeline.py` | 230 | Consumes `ChunkBatch`, drives commit/rollback |

### Directory to create

```text
src/embeddings/
├── __init__.py
├── base.py
├── fastembed_provider.py
├── openai_provider.py
└── factory.py

src/vectorstores/
├── __init__.py
├── base.py
├── qdrant_store.py
├── chroma_store.py
└── factory.py

src/pipelines/
├── __init__.py
└── indexing_pipeline.py
```

**`src/pipelines/` is not in the roadmap's tree — I am adding it, and here is why.** The indexing
driver imports from `src/ingestion/`, `src/embeddings/`, and `src/vectorstores/`. Putting it inside
any one of those three makes that package import its siblings, which violates the roadmap's one
structural rule (dependencies point inward and downward, never sideways). A composition layer that
sits *above* all three and is imported by none of them keeps the rule intact. Phase 8's API and Phase
9's CLI will both call into it. Update the roadmap tree when you next touch it.

### Data flow

```text
IngestionPipeline.run()  ──yields──►  ChunkBatch(chunks, document_count)
                                            │
                                            ▼
                      embedder.embed_dense(texts)   ─┐
                      embedder.embed_sparse(texts)   ├─ concurrently
                                            │        ─┘
                                            ▼
                       store.delete_by_doc_ids({doc_ids in batch})
                                            │
                                            ▼
                       store.upsert_points(chunks, dense, sparse)
                                            │
                        ┌───────────────────┴───────────────────┐
                    succeeded                                failed
                        │                                       │
              ingestion.commit()                      ingestion.rollback()
         (documents now durable-done)          (documents retried next run)
```

Then, later, Pipeline 2 reads from the other side of the same collection:

```text
query ──► embedder.embed_query      ──► dense query vector  ─┐
     └──► embedder.embed_sparse     ──► sparse query vector  ├─► store.hybrid_search
                                                             ─┘        │
                                       filtered to chunk_level=child   │
                                                                       ▼
                                                            list[ScoredChunk]
```

---

## 2. Before Any Code — Settings You Must Add

Per the arrangement, I do not edit `config/settings.py`. These are the additions this phase needs;
add them to the `Settings` class yourself, in the sections indicated.

Add to the **Qdrant** block:

```python
    VECTOR_STORE: str = Field(default="qdrant")
    UPSERT_BATCH_SIZE: int = Field(default=256, ge=1, le=1000)
    PREFETCH_MULTIPLIER: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Candidates each retrieval arm fetches, as a multiple of the final limit.",
    )
    CHROMA_PATH: str = Field(default="./data/chroma")
```

Add to the **Models** block:

```python
    EMBEDDING_PROVIDER: str = Field(default="fastembed")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, ge=1, le=512)
```

Add these two validators alongside the existing ones:

```python
    @field_validator("EMBEDDING_PROVIDER")
    @classmethod
    def _validate_embedding_provider(cls, v: str) -> str:
        allowed = {"fastembed", "openai"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {sorted(allowed)}, got {v!r}")
        return lower

    @field_validator("VECTOR_STORE")
    @classmethod
    def _validate_vector_store(cls, v: str) -> str:
        allowed = {"qdrant", "chroma"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"VECTOR_STORE must be one of {sorted(allowed)}, got {v!r}")
        return lower
```

Add to `validate_runtime()`, inside the existing warnings list:

```python
        if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            warnings.append("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset.")

        if self.VECTOR_STORE == "chroma":
            warnings.append(
                "VECTOR_STORE=chroma has no sparse vector support — retrieval will be "
                "dense-only, not hybrid."
            )
```

And to `.env.example`:

```bash
# ─── Phase 3: embeddings and storage ───────────────────────────────────────
EMBEDDING_PROVIDER=fastembed        # fastembed | openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=64
VECTOR_STORE=qdrant                 # qdrant | chroma
UPSERT_BATCH_SIZE=256
PREFETCH_MULTIPLIER=4
CHROMA_PATH=./data/chroma
```

Dependencies, if they are not already in `pyproject.toml`: `fastembed`, `qdrant-client>=1.10`,
`openai`, and `chromadb` (the last one only if you want the fallback store to import).

---

## 3. File 1 — `src/embeddings/base.py`

### The Problem

`BaseEmbeddingProvider` in Phase 1 declares five methods, and three of them have identical
obligations that have nothing to do with any particular model:

- The output list must be the same length as the input list, in the same order.
- Empty or whitespace-only text must not reach a model.
- Long text must be truncated before it silently blows a model's context window.
- Large inputs must be split into batches the model can actually process.

The alignment obligation is the serious one. Every caller in this system zips the returned vectors
against the input chunks. If one provider returns a shorter list — because it skipped a blank string,
or a batch partially failed and the code appended only what came back — then chunk *n* gets chunk
*n−1*'s vector, silently, from that point on. Retrieval keeps working. It just returns the wrong
passages, and no test that checks "did we get results" will notice.

There is also a second, structural problem. Phase 1 froze `BaseEmbeddingProvider` as **async** with a
non-abstract sync escape hatch `embed_dense_sync`, because the OpenAI provider is network I/O and
would otherwise be forced to block the event loop. Local ONNX providers therefore need to implement
the sync version and have the async version delegate to it through `asyncio.to_thread`. Writing that
delegation once per local provider is three copies of the same four lines.

### Design Decision

**Two layers between the ABC and the implementations.**

`EmbeddingProviderBase` holds the validation and batching that every provider needs, regardless of
where the computation happens. It stays abstract.

`LocalEmbeddingProvider` extends it with the sync→async bridge: it implements `embed_dense`,
`embed_sparse`, and `embed_query` as `asyncio.to_thread` calls into abstract *sync* hooks. A local
provider then writes only synchronous code and gets a correct async surface for free. This is the
one place in the project where the awkward `embed_dense_sync` from the frozen interface is
explained and contained.

**Blank text raises instead of being skipped.** This is deliberate and it is the second lesson from
Phase 2 applied: when you write a filter, ask what happens to the data it rejects. Skipping a blank
string would break alignment; substituting a zero vector would put a meaningless point into the
index that matches everything weakly. Raising is the only option whose failure mode is loud. Phase
2's `Chunk` validator already rejects blank text, so if this ever fires, something upstream is
broken and you want to know.

```python
import asyncio
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from typing import TypeVar

from src.core.exceptions import EmbeddingError
from src.core.interfaces import BaseEmbeddingProvider
from src.core.logging import logger
from src.core.utils import count_tokens, truncate_to_tokens

T = TypeVar("T")

#: Inputs per model call. Larger batches amortise fixed overhead; too large and a
#: single 64-document batch of long contracts exceeds available RAM during ONNX
#: inference. 64 is a safe default for 400-token chunks on a laptop.
DEFAULT_BATCH_SIZE = 64

#: Hard ceiling per input. bge-small-en-v1.5 has a 512-token window and silently
#: truncates beyond it. We truncate explicitly so the loss is logged rather than
#: invisible. Chunks from Phase 2 are ~400 tokens, so this should never fire —
#: which is exactly why it must be loud when it does.
MAX_INPUT_TOKENS = 512


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield fixed-size slices of a sequence. The last slice may be short."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class EmbeddingProviderBase(BaseEmbeddingProvider):
    """Behaviour shared by every embedding provider, local or remote.

    Subclasses get input validation, truncation, batching, and an alignment
    assertion. They do not get to skip any of it, because `_prepare` is the only
    supported way to turn caller input into model input.
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.batch_size = batch_size

    # ─── input handling ────────────────────────────────────────────────────

    def _prepare(self, texts: Sequence[str]) -> list[str]:
        """Validate and normalise a batch of inputs.

        Returns a list of exactly `len(texts)` strings — never fewer. Dropping an
        input here would misalign every downstream vector against its chunk, a
        corruption with no visible symptom.

        Raises:
            EmbeddingError: an input is empty or whitespace-only.
        """
        if not texts:
            return []

        prepared: list[str] = []
        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingError(
                    "Refusing to embed empty text",
                    details={"index": index, "batch_size": len(texts)},
                )
            if count_tokens(text) > MAX_INPUT_TOKENS:
                logger.warning(
                    "Truncating oversized embedding input",
                    extra={"index": index, "max_tokens": MAX_INPUT_TOKENS},
                )
                text = truncate_to_tokens(text, MAX_INPUT_TOKENS)
            prepared.append(text)
        return prepared

    def _check_alignment(self, texts: Sequence[str], vectors: Sequence[object]) -> None:
        """Assert one vector came back per input.

        This is the single most important check in the file. A provider that
        returns a short list produces an index where every subsequent chunk holds
        its neighbour's vector — retrieval degrades subtly and permanently, with
        no error anywhere.
        """
        if len(vectors) != len(texts):
            raise EmbeddingError(
                "Embedding count does not match input count",
                details={"inputs": len(texts), "vectors": len(vectors)},
            )


class LocalEmbeddingProvider(EmbeddingProviderBase):
    """A provider whose computation is CPU-bound and in-process.

    Implements the async interface by offloading to a worker thread, so callers
    get a uniform `await`-able API while the implementation stays plainly
    synchronous. Subclasses implement only the three `*_sync` hooks.

    Offloading is not theatre: ONNX Runtime releases the GIL during inference, so
    the event loop genuinely stays responsive while a batch is being embedded, and
    two concurrent `to_thread` calls do overlap to a useful degree. What it does
    NOT give you is linear speed-up across cores — that needs separate processes,
    which is why Phase 2's ingestion is multiprocessing.
    """

    @abstractmethod
    def embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        """Blocking dense embedding. Overrides the interface's default raiser."""

    @abstractmethod
    def embed_sparse_sync(self, texts: list[str]) -> list[dict[str, list]]:
        """Blocking sparse embedding."""

    @abstractmethod
    def embed_query_sync(self, text: str) -> list[float]:
        """Blocking single-query embedding, with any query-side prefix applied."""

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_dense_sync, texts)

    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        return await asyncio.to_thread(self.embed_sparse_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query_sync, text)
```

### The Theory: why the interface is async when the work is not

Phase 1's `interfaces.py` went back and forth on this and landed on async. The reasoning is worth
restating, because it is the general rule for designing an interface with heterogeneous
implementations.

**An interface must be shaped by its most demanding implementation, not its simplest one.** The
FastEmbed provider is pure CPU. If it were the only implementation, a synchronous interface would be
honest and simpler. But the OpenAI provider is an HTTP round trip of 100–500ms, and a synchronous
interface would force every caller — including Phase 8's API server, handling concurrent requests —
to block the entire event loop for the duration. One implementation cannot be made async by its
caller; the other can be made to *look* async by wrapping a thread. So the async shape is the one
that accommodates both, and the CPU-bound implementation pays a small, contained cost.

The cost is real but tiny: `asyncio.to_thread` adds roughly 50–100 microseconds of scheduling
overhead per call, against an embedding batch that takes 20–200 milliseconds. Three orders of
magnitude. Compare that to the alternative, where a synchronous interface forces every network
provider to block, and where every caller must remember to wrap it — and the ones that forget
produce a server that stops responding under load, diagnosed only by profiling.

### Failure Modes

**`EmbeddingError: Refusing to embed empty text`.** Something upstream produced a blank chunk. Since
`Chunk` already validates non-blank text, the likely cause is code building a text list by hand
rather than from `chunk.text`. Fix the producer; do not relax this check.

**`EmbeddingError: Embedding count does not match input count`.** A provider returned a short list.
For FastEmbed this usually means you forgot `list(...)` around a generator and iterated it twice —
the second iteration yields nothing. For OpenAI it means a partial batch failure was swallowed.

**A subclass forgets `embed_query_sync` and gets a confusing error.** Because these hooks are
`@abstractmethod` on `LocalEmbeddingProvider`, instantiation fails immediately with
`TypeError: Can't instantiate abstract class`. That is the intended behaviour and the reason they are
abstract here even though the interface declares only `embed_dense_sync`.

**Truncation warnings during ingestion.** Your chunker is producing chunks larger than the embedding
model's window. Phase 2's `MAX_TOKENS_PER_CHUNK` defaults to 400 against a 512-token model, so this
should be silent. If it is not, someone raised that setting without checking the model.

---

## 4. File 2 — `src/embeddings/fastembed_provider.py`

### The Problem

This is the provider that will actually run over 650,000 documents, so three practical concerns
dominate.

**Model loading is expensive and must happen once per process.** `TextEmbedding("BAAI/bge-small-en-v1.5")`
downloads (first time) and initialises an ONNX session — hundreds of milliseconds and ~90 MB of
resident memory. Construct it per call and you have a pipeline that spends all its time loading
models.

**Import must not load anything.** If importing this module loaded the model, then Phase 2's
multiprocessing workers — which import half the codebase on spawn — would each pay the cost and the
memory, for a model they never use. Loading has to be lazy.

**Queries and documents are not embedded the same way.** `bge` models are trained asymmetrically:
the query side expects an instruction prefix ("Represent this sentence for searching relevant
passages: ") and the document side expects none. Embed both with the same call and you lose several
points of retrieval accuracy for no visible reason. FastEmbed exposes `query_embed` and
`passage_embed` precisely to handle this per model.

### Design Decision

**Module-level `@lru_cache` factories for the models.** This mirrors `get_tokenizer` from Phase 1 and
gives exactly what is needed: lazy (nothing happens at import), per-process (the cache lives in the
process's module globals, so each multiprocessing worker builds its own), and shared (two provider
instances in the same process reuse one ONNX session).

**Dimensions determined empirically, not from a registry lookup.** FastEmbed exposes
`TextEmbedding.list_supported_models()`, and the obvious implementation reads `dim` out of it. I am
not doing that, because the shape of those entries has changed between FastEmbed versions — dicts in
some releases, description objects in others — and I cannot execute anything here to check which one
you will have. Embedding the single word `"dimension probe"` once and taking `len(vector)` is
authoritative, costs one forward pass at startup, and cannot drift with the library. It is then
cross-checked against `DENSE_VECTOR_SIZE` so a model swap without a config change fails loudly.

**`passage_embed` for documents, `query_embed` for queries, consistently.** The absolute rule is
symmetry: whatever transformation was applied at index time must be matched at query time. Mixing
`embed()` at index time with `query_embed()` at search time puts the query in a slightly different
region of the space than the documents, and the failure presents as "hybrid search is mediocre",
which is nearly impossible to attribute.

```python
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from config.settings import settings
from src.core.exceptions import DimensionMismatchError, EmbeddingError
from src.core.logging import logger

from .base import DEFAULT_BATCH_SIZE, LocalEmbeddingProvider, batched


@lru_cache(maxsize=2)
def get_dense_model(model_name: str) -> TextEmbedding:
    """Process-wide dense model singleton.

    Lazy and cached for the same reasons as `get_tokenizer`: loading costs
    hundreds of milliseconds and ~90 MB, the cache is per-process so
    multiprocessing workers each build their own, and importing this module
    costs nothing.
    """
    logger.info("Loading dense embedding model", extra={"model": model_name})
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=2)
def get_sparse_model(model_name: str) -> SparseTextEmbedding:
    """Process-wide sparse model singleton.

    `Qdrant/bm25` is not a neural model — it is stemming plus token hashing, about
    10 MB — but it still holds state worth loading once.
    """
    logger.info("Loading sparse embedding model", extra={"model": model_name})
    return SparseTextEmbedding(model_name=model_name)


def _to_float_list(vector: object) -> list[float]:
    """Convert a numpy array to a plain Python list of floats.

    FastEmbed returns `numpy.ndarray` of `float32`. Qdrant's client will often
    accept that, but JSON serialisation of `float32` is not universally safe and
    the failure surfaces far from here. Converting at the boundary keeps every
    layer above this file in plain Python types.
    """
    tolist = getattr(vector, "tolist", None)
    return [float(x) for x in (tolist() if tolist else vector)]  # type: ignore[operator]


class FastEmbedProvider(LocalEmbeddingProvider):
    """Local ONNX embeddings: dense `bge` vectors plus BM25 sparse vectors.

    The default provider for this project. Runs on CPU with no CUDA dependency,
    which is what makes embedding 650,000 documents on a laptop merely slow
    rather than impossible.
    """

    def __init__(
        self,
        dense_model_name: str | None = None,
        sparse_model_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.dense_model_name = dense_model_name or settings.DENSE_MODEL_NAME
        self.sparse_model_name = sparse_model_name or settings.SPARSE_MODEL_NAME
        self._dimensions: int | None = None

    # ─── model access ──────────────────────────────────────────────────────

    @property
    def _dense(self) -> TextEmbedding:
        return get_dense_model(self.dense_model_name)

    @property
    def _sparse(self) -> SparseTextEmbedding:
        return get_sparse_model(self.sparse_model_name)

    @property
    def dense_dimensions(self) -> int:
        """Vector width, measured rather than looked up.

        NOT free on first call: it loads the model and runs one forward pass. That
        is deliberate — the alternative is reading a version-dependent model
        registry, and being wrong about this creates a collection whose schema
        silently rejects every subsequent upsert.

        Raises:
            DimensionMismatchError: the model disagrees with DENSE_VECTOR_SIZE.
        """
        if self._dimensions is None:
            probe = list(self._dense.passage_embed(["dimension probe"]))
            measured = len(_to_float_list(probe[0]))

            if measured != settings.DENSE_VECTOR_SIZE:
                raise DimensionMismatchError(
                    "Embedding model dimension does not match configuration",
                    details={
                        "model": self.dense_model_name,
                        "measured": measured,
                        "configured": settings.DENSE_VECTOR_SIZE,
                    },
                )
            self._dimensions = measured
            logger.info(
                "Dense model ready",
                extra={"model": self.dense_model_name, "dimensions": measured},
            )
        return self._dimensions

    def warmup(self) -> None:
        """Load both models and validate dimensions before real work starts.

        Call this at application startup. Without it the first user query pays a
        one-second model load, which looks like a slow database.
        """
        _ = self.dense_dimensions
        _ = list(self._sparse.embed(["warmup"]))

    # ─── synchronous implementations ───────────────────────────────────────

    def embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        """Dense document embeddings, batched.

        `passage_embed`, not `embed`: for asymmetric models like bge the document
        side and the query side get different prefixes, and `embed` applies
        neither. What matters is that index time and query time agree.
        """
        prepared = self._prepare(texts)
        if not prepared:
            return []

        vectors: list[list[float]] = []
        try:
            for group in batched(prepared, self.batch_size):
                vectors.extend(_to_float_list(v) for v in self._dense.passage_embed(group))
        except Exception as exc:
            raise EmbeddingError(
                "Dense embedding failed",
                details={"model": self.dense_model_name, "inputs": len(prepared)},
            ) from exc

        self._check_alignment(prepared, vectors)
        return vectors

    def embed_sparse_sync(self, texts: list[str]) -> list[dict[str, list]]:
        """BM25 sparse vectors as {"indices": [...], "values": [...]}.

        The dict shape is the project's neutral representation, chosen so that
        `BaseEmbeddingProvider` does not leak a Qdrant type into every signature.
        The store converts it to `models.SparseVector` at the last moment.
        """
        prepared = self._prepare(texts)
        if not prepared:
            return []

        vectors: list[dict[str, list]] = []
        try:
            for group in batched(prepared, self.batch_size):
                for embedding in self._sparse.embed(group):
                    vectors.append(
                        {
                            "indices": [int(i) for i in embedding.indices],
                            "values": [float(v) for v in embedding.values],
                        }
                    )
        except Exception as exc:
            raise EmbeddingError(
                "Sparse embedding failed",
                details={"model": self.sparse_model_name, "inputs": len(prepared)},
            ) from exc

        self._check_alignment(prepared, vectors)

        # A chunk of pure stopwords stems away to nothing. Keep it — dense
        # retrieval still works for it — but say so, because an empty sparse
        # vector is invisible in the index and confusing when you meet it later.
        empty = sum(1 for v in vectors if not v["indices"])
        if empty:
            logger.warning("Sparse vectors with no terms", extra={"count": empty})

        return vectors

    def embed_query_sync(self, text: str) -> list[float]:
        """Dense query embedding, with the model's query-side prefix applied."""
        prepared = self._prepare([text])
        try:
            vectors = list(self._dense.query_embed(prepared))
        except Exception as exc:
            raise EmbeddingError(
                "Query embedding failed", details={"model": self.dense_model_name}
            ) from exc

        if not vectors:
            raise EmbeddingError("Query embedding returned nothing")
        return _to_float_list(vectors[0])

    def embed_sparse_query_sync(self, text: str) -> dict[str, list]:
        """BM25 sparse vector for a query.

        Not part of `BaseEmbeddingProvider` — Phase 4 calls it directly on the
        concrete provider. `query_embed` differs from `embed` on the sparse side
        too: the query representation carries no term-frequency saturation,
        because a query has no meaningful term frequencies.
        """
        prepared = self._prepare([text])
        embeddings = list(self._sparse.query_embed(prepared[0]))
        if not embeddings:
            return {"indices": [], "values": []}
        return {
            "indices": [int(i) for i in embeddings[0].indices],
            "values": [float(v) for v in embeddings[0].values],
        }
```

### The Theory: BM25 as a sparse vector, and where the IDF went

Classic BM25 scores a document against a query with a formula that has three moving parts: term
frequency in the document (with saturation, so the tenth occurrence of "indemnify" adds far less
than the second), document length normalisation, and **inverse document frequency** — a weight that
makes rare terms count more than common ones.

The first two are properties of a single document. The third is not: IDF depends on how many
documents in the *whole corpus* contain the term. That is the crux.

FastEmbed embeds one document at a time and has no idea what your corpus looks like. So
`Qdrant/bm25` deliberately computes only the document-local part — it stems each token, hashes it to
an integer index with MurmurHash3, and stores the saturated, length-normalised term weight as the
value. The IDF component is **intentionally absent**.

Qdrant supplies it. When a sparse vector field is created with `modifier=models.Modifier.IDF`, Qdrant
maintains term document-frequencies at the collection level and applies the IDF weight during
scoring. This is the right split: the statistic that requires global knowledge is computed by the
component that has global knowledge.

The consequence is a configuration trap with no error message. **Create the sparse field without the
IDF modifier and your BM25 becomes term-frequency-only scoring.** Every query still returns results.
They are just weighted as though "the" and "indemnification" were equally informative. Nothing
raises, nothing logs, and the only symptom is that lexical search is oddly bad. This is why the
verification script in §12 explicitly asserts the modifier is set, and why `initialize()` checks for
it on an existing collection.

An aside worth knowing: because the indices are hashes rather than vocabulary positions, BM25 sparse
vectors have no fixed dimensionality and there is no out-of-vocabulary problem. A defined term
invented by one contract — "Permitted Encumbrance" — gets an index like any other token. That is a
genuine advantage over neural sparse models such as SPLADE, which are bounded by a 30k WordPiece
vocabulary and will split unusual legal terms into fragments.

### Failure Modes

**The first call takes 30 seconds.** FastEmbed is downloading the model to its cache
(`~/.cache/fastembed` or `%LOCALAPPDATA%`). Subsequent runs load from disk. Call `warmup()` at
startup rather than discovering this on a user's first query.

**`DimensionMismatchError` at startup.** You changed `DENSE_MODEL_NAME` without changing
`DENSE_VECTOR_SIZE`. Fix the config — and note that if a collection already exists at the old size,
you need Phase 15's migration, not a config edit.

**Memory grows by ~90 MB per process.** Expected: each multiprocessing worker holds its own ONNX
session. With eight workers that is ~700 MB. This is the price of escaping the GIL, and it is why
Phase 2 chose *not* to embed inside its workers — chunks are cheap to ship across the process
boundary, and embedding in the parent keeps the memory to one copy.

**`list(...)` forgotten around a FastEmbed call.** All three FastEmbed methods return generators. Use
the result twice — once to count, once to iterate — and the second use yields nothing, producing an
alignment error rather than a crash. Always materialise immediately.

**Sparse search returns nothing for a short query.** BM25 stems and drops stopwords, so "what is it"
can reduce to an empty vector. With RRF fusion the dense arm still contributes, so the query works —
but if you ever query the sparse arm alone, guard for an empty `indices` list.

---

## 5. File 3 — `src/embeddings/openai_provider.py`

### The Problem

The second implementation exists for two reasons that are not "we might switch someday". First, it is
what proves the interface is an abstraction rather than a FastEmbed wrapper — a claim you cannot make
with one implementation. Second, `text-embedding-3-large` is genuinely better than `bge-small` on
retrieval benchmarks, and Phase 7's evaluation harness needs a stronger baseline to measure against.

It brings a problem FastEmbed does not have: **OpenAI has no sparse embedding endpoint.** The
interface requires `embed_sparse`. Three options, and only one is defensible.

### Design Decision

**Dense from OpenAI, sparse from local BM25.** Raising `NotImplementedError` from `embed_sparse`
would mean selecting this provider silently disables hybrid search — the exact class of silent
degradation this project keeps avoiding. Returning empty sparse vectors is worse, because it looks
like it works.

Delegating to the local BM25 model is not a hack once you see what BM25 is: a lexical statistic
computed from stemmed tokens, with no learned parameters and no relationship to the dense model. It
is not an "OpenAI sparse embedding", it is *the same BM25 everyone uses*, and pairing it with a
better dense model is exactly what a hybrid system should do. Mixing a remote dense model with a
local lexical one is a normal production topology.

**`embed_dense_sync` stays unimplemented.** It inherits the interface's `NotImplementedError`, on
purpose. Running an HTTP client inside a forked multiprocessing worker is a reliable way to produce
hung connections and duplicated request state.

**Explicit retry with exponential backoff.** Rate limits are the normal operating condition of a bulk
embedding job, not an exceptional one. Phase 1 already gave `RateLimitError` a `retryable` flag and a
`retry_after`; this is the first place that machinery earns its keep.

```python
import asyncio

from openai import APIError, AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError

from config.settings import settings
from src.core.exceptions import DimensionMismatchError, EmbeddingError, RateLimitError
from src.core.logging import logger

from .base import EmbeddingProviderBase, batched
from .fastembed_provider import get_sparse_model

#: Published dimensions per model. text-embedding-3-* support Matryoshka
#: truncation, so a requested `dimensions` below the native size is valid and the
#: vector remains usable — see the note below on why that is not free.
_NATIVE_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

#: The API accepts far more per request, but large batches mean a large blast
#: radius on a retry and a long tail latency. 128 is a reasonable compromise.
_API_BATCH_SIZE = 128

_MAX_ATTEMPTS = 4


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """Dense embeddings from the OpenAI API; sparse BM25 computed locally.

    The sparse half is not a compromise. BM25 is a lexical statistic with no
    learned parameters — pairing a strong remote dense model with local BM25 is a
    normal hybrid topology, not a workaround for a missing endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = _API_BATCH_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.model = model or settings.OPENAI_EMBEDDING_MODEL

        if not settings.OPENAI_API_KEY:
            raise EmbeddingError(
                "OPENAI_API_KEY is required for the OpenAI embedding provider",
                details={"model": self.model},
            )

        native = _NATIVE_DIMENSIONS.get(self.model)
        if native is None:
            logger.warning(
                "Unknown OpenAI embedding model; trusting DENSE_VECTOR_SIZE",
                extra={"model": self.model},
            )
        self._dimensions = dimensions or native or settings.DENSE_VECTOR_SIZE

        if native is not None and self._dimensions > native:
            raise DimensionMismatchError(
                "Requested more dimensions than the model produces",
                details={"model": self.model, "requested": self._dimensions, "native": native},
            )

        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._sparse_model_name = settings.SPARSE_MODEL_NAME

    @property
    def dense_dimensions(self) -> int:
        return self._dimensions

    # ─── dense: network ────────────────────────────────────────────────────

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        prepared = self._prepare(texts)
        if not prepared:
            return []

        vectors: list[list[float]] = []
        for group in batched(prepared, self.batch_size):
            vectors.extend(await self._embed_batch(group))

        self._check_alignment(prepared, vectors)
        return vectors

    async def _embed_batch(self, group: list[str]) -> list[list[float]]:
        """One API call, retried with exponential backoff on transient failures."""
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=group,
                    dimensions=self._dimensions,
                )
                # The API documents that `data` is ordered by `index`, but sorting
                # costs microseconds and removes the need to trust that. An
                # out-of-order response would misalign vectors against chunks with
                # no visible symptom, which is the one failure worth paranoia.
                ordered = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in ordered]

            except OpenAIRateLimitError as exc:
                last_error = exc
                logger.warning(
                    "OpenAI rate limit; backing off",
                    extra={"attempt": attempt, "sleep_s": delay},
                )
                await asyncio.sleep(delay)
                delay *= 2

            except APIError as exc:
                last_error = exc
                # 5xx is worth retrying; 4xx is a request we should stop repeating.
                status = getattr(exc, "status_code", None)
                if status is not None and status < 500:
                    raise EmbeddingError(
                        "OpenAI rejected the embedding request",
                        details={"model": self.model, "status": status},
                    ) from exc
                await asyncio.sleep(delay)
                delay *= 2

        raise RateLimitError(
            "OpenAI embedding failed after retries",
            details={"model": self.model, "attempts": _MAX_ATTEMPTS},
        ) from last_error

    async def embed_query(self, text: str) -> list[float]:
        """Single query embedding.

        No query/document asymmetry here: OpenAI's embedding models are trained
        symmetrically, unlike bge. So this is `embed_dense` of one item, and that
        is the whole difference between the two providers' query paths.
        """
        vectors = await self.embed_dense([text])
        return vectors[0]

    # ─── sparse: local ─────────────────────────────────────────────────────

    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        prepared = self._prepare(texts)
        if not prepared:
            return []
        vectors = await asyncio.to_thread(self._embed_sparse_sync, prepared)
        self._check_alignment(prepared, vectors)
        return vectors

    def _embed_sparse_sync(self, texts: list[str]) -> list[dict[str, list]]:
        model = get_sparse_model(self._sparse_model_name)
        return [
            {
                "indices": [int(i) for i in embedding.indices],
                "values": [float(v) for v in embedding.values],
            }
            for embedding in model.embed(texts)
        ]

    async def close(self) -> None:
        """Release the HTTP connection pool. Called from Phase 8's lifespan hook."""
        await self._client.close()
```

### The Theory: Matryoshka embeddings, and the cost of shortening a vector

The `dimensions` parameter on `text-embedding-3-*` is unusual enough to be worth explaining, because
it looks like free lunch and is not.

Ordinarily an embedding is atomic: 1536 floats where no subset means anything on its own. Matryoshka
Representation Learning changes the training objective so that useful information is *front-loaded* —
the first 256 dimensions are trained to be a usable embedding by themselves, the first 512 a better
one, and so on. Truncating the vector therefore degrades quality gradually instead of destroying it.

That buys real things. Storage is linear in dimension, so 650k documents at ~60 chunks each is
39 million points; at 1536 float32 dimensions that is roughly 240 GB of vectors, and at 384 it is
60 GB. Search latency in HNSW is dominated by distance computations, which are also linear in
dimension.

The cost is that quality does drop — a few points of retrieval accuracy between 1536 and 256 on
standard benchmarks — and, more importantly for this project, **the dimension is baked into the
collection schema.** Choosing 384 to match an existing bge collection and later wanting 1536 means a
full re-index, which is Phase 15's entire subject. Decide before you index 39 million points, not
after.

### Failure Modes

**`EmbeddingError: OPENAI_API_KEY is required`** at construction. Deliberate: failing when the
provider is built is far better than failing 40,000 documents into a run.

**Costs more than expected.** At `text-embedding-3-small` pricing, the full EDGAR corpus is a real
number, not a rounding error. Run with `SAMPLE_MODE=true` first and multiply. This is the practical
reason FastEmbed is the default.

**Retries make a bad batch slow instead of failing.** Four attempts with doubling backoff is up to
~15 seconds per batch. If every batch is retrying, stop the run — you are rate-limited at the account
level and waiting will not help.

**Dimension mismatch against an existing collection.** The store's `initialize()` will raise
`DimensionMismatchError` rather than write 1536-dimension vectors into a 384-dimension schema. That
check exists because Qdrant's own error for this is less obvious than you would like.

---

## 6. File 4 — `src/embeddings/factory.py`

### The Problem

Nine call sites will need an embedding provider. If each one writes
`FastEmbedProvider() if settings.EMBEDDING_PROVIDER == "fastembed" else OpenAIEmbeddingProvider()`,
then adding a third provider means finding all nine, and every one of them loads a separate copy of
a 90 MB ONNX model.

### Design Decision

**A registry plus a cached accessor.** The registry maps a settings string to a constructor. The
`@lru_cache` makes the provider a per-process singleton, which is the behaviour you want: the model
is loaded once and shared.

**Caching is safe here specifically because embedding providers hold no connection state.** Contrast
this with the vector store factory in §10, which is deliberately *not* cached. The distinction is
worth internalising: cache things that are expensive to build and stateless; never cache things that
own a network connection bound to an event loop.

```python
from collections.abc import Callable
from functools import lru_cache

from config.settings import settings
from src.core.exceptions import EmbeddingError
from src.core.interfaces import BaseEmbeddingProvider
from src.core.logging import logger


def _build_fastembed() -> BaseEmbeddingProvider:
    # Imported inside the builder so that selecting `openai` does not import
    # fastembed, and vice versa. Keeps startup light and makes an uninstalled
    # optional dependency fail only when it is actually chosen.
    from .fastembed_provider import FastEmbedProvider

    return FastEmbedProvider(batch_size=settings.EMBEDDING_BATCH_SIZE)


def _build_openai() -> BaseEmbeddingProvider:
    from .openai_provider import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider()


_BUILDERS: dict[str, Callable[[], BaseEmbeddingProvider]] = {
    "fastembed": _build_fastembed,
    "openai": _build_openai,
}


@lru_cache(maxsize=4)
def get_embedding_provider(name: str | None = None) -> BaseEmbeddingProvider:
    """Return the configured embedding provider, one instance per process.

    Cached because building a provider loads a model. This is safe only because
    providers hold no connection state — see `get_vector_store`, which is
    deliberately not cached for exactly that reason.

    Tests that need a fresh instance call `get_embedding_provider.cache_clear()`.

    Raises:
        EmbeddingError: the configured name has no registered builder.
    """
    key = (name or settings.EMBEDDING_PROVIDER).lower()

    builder = _BUILDERS.get(key)
    if builder is None:
        raise EmbeddingError(
            "Unknown embedding provider",
            details={"requested": key, "available": sorted(_BUILDERS)},
        )

    logger.info("Building embedding provider", extra={"provider": key})
    return builder()
```

### Failure Modes

**Changing `EMBEDDING_PROVIDER` has no effect.** Two caches to defeat: `get_settings`'s `lru_cache`
and this one. Restart the process.

**`ModuleNotFoundError: fastembed` only when you select it.** Intended — the imports are inside the
builders. If you never use the OpenAI provider you never need the `openai` package installed.

**Two providers alive at once during Phase 15's migration.** Supported: pass explicit names, and
`maxsize=4` keeps both cached. This is the one scenario the `name` argument exists for.

---

## 7. File 5 — `src/vectorstores/base.py`

### The Problem

Two store implementations will need the same three things, and one of them is a correctness check
that must not be optional.

The vector field names (`dense-bge`, `sparse-bm25`) are written by the store at index time and read
by Phase 4 at query time. If those two ever disagree, Qdrant raises a not-especially-clear error
about an unknown vector name. They belong in one place that both import.

Converting a stored payload back into a `ScoredChunk` is identical everywhere and must go through
`Chunk.from_payload`, since Phase 1 established that only `to_payload`/`from_payload` may touch
payload dicts.

And the alignment check has to be repeated on the store side. The embedder already asserts it
produced one vector per text — but the store receives three lists assembled by a *caller*, and it is
the caller who might zip the wrong things together.

### Design Decision

**A thin base class with no vendor imports.** `base.py` must not import `qdrant_client`, otherwise
the Chroma store transitively depends on Qdrant. Payload index specifications are therefore declared
as neutral `(field, kind)` string pairs and each store maps them to its own types.

**Validation raises `DimensionMismatchError`, which Phase 1 marked non-retryable.** That is the right
classification: a wrong vector width means the embedding model changed without a re-index, and
retrying will never fix it. The exception type carries that fact so Phase 8's error handler does not
have to know it.

```python
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from src.core.exceptions import DimensionMismatchError, VectorStoreError
from src.core.interfaces import BaseVectorStore
from src.core.models import Chunk, RetrievalMethod, ScoredChunk

T = TypeVar("T")

#: Named vector fields inside the collection. Written at index time, referenced by
#: `using=` at query time — the two MUST agree, which is why they live here rather
#: than as literals in two files.
#:
#: The names embed the model family, which is a small lie of convenience: swapping
#: to a different dense model changes the vector width and therefore needs a new
#: collection anyway (Phase 15), so the label never goes stale in practice.
DENSE_VECTOR_NAME = "dense-bge"
SPARSE_VECTOR_NAME = "sparse-bm25"

#: Payload fields that get an index, and the kind of index they need.
#:
#: `doc_id`   — delete-before-upsert filters on it for every batch. Unindexed,
#:              that filter is a full collection scan; at 39M points that is the
#:              difference between milliseconds and minutes.
#: `chunk_level` — Phase 4 filters every search to children.
#: `year`, `section_title` — user-facing metadata filters.
PAYLOAD_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    ("doc_id", "keyword"),
    ("chunk_level", "keyword"),
    ("year", "integer"),
    ("section_title", "keyword"),
)


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield fixed-size slices. Duplicated from `embeddings.base` on purpose —
    `vectorstores` must not import `embeddings`; they are peer packages."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class VectorStoreBase(BaseVectorStore):
    """Store-agnostic helpers. Holds no vendor imports."""

    def _validate_batch(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[dict[str, list]],
        expected_dim: int | None = None,
    ) -> None:
        """Refuse to write a batch whose three lists do not correspond.

        The caller assembles these separately, so a mismatch here means chunk N is
        about to be stored with chunk N-1's vector. Nothing downstream can detect
        that, so it must be impossible to write.

        Raises:
            VectorStoreError: list lengths disagree.
            DimensionMismatchError: a dense vector is the wrong width.
        """
        if not (len(chunks) == len(dense) == len(sparse)):
            raise VectorStoreError(
                "Chunk, dense, and sparse counts must match",
                details={"chunks": len(chunks), "dense": len(dense), "sparse": len(sparse)},
            )

        if expected_dim is not None:
            for index, vector in enumerate(dense):
                if len(vector) != expected_dim:
                    raise DimensionMismatchError(
                        "Dense vector width does not match the collection schema",
                        details={
                            "index": index,
                            "got": len(vector),
                            "expected": expected_dim,
                            "chunk_id": chunks[index].chunk_id,
                        },
                    )

    def _to_scored_chunks(
        self,
        rows: Sequence[tuple[dict[str, Any], float]],
        method: RetrievalMethod = RetrievalMethod.HYBRID,
    ) -> list[ScoredChunk]:
        """Turn (payload, score) pairs into ranked `ScoredChunk`s.

        Rank is assigned from the order the store returned, which is already
        sorted by score. `Chunk.from_payload` is the only sanctioned way to read a
        payload — no store may pick fields out of the dict by hand.

        Raises:
            VectorStoreError: a payload is not one this project wrote.
        """
        results: list[ScoredChunk] = []
        for rank, (payload, score) in enumerate(rows):
            if not payload or "text" not in payload:
                raise VectorStoreError(
                    "Stored point has no usable payload",
                    details={"rank": rank, "keys": sorted(payload or {})},
                )
            results.append(
                ScoredChunk(
                    chunk=Chunk.from_payload(payload),
                    score=float(score),
                    rank=rank,
                    method=method,
                )
            )
        return results
```

### Failure Modes

**`VectorStoreError: Stored point has no usable payload`.** Something wrote to this collection
without going through `Chunk.to_payload()` — usually an experiment in a notebook. Use a separate
collection for experiments.

**`DimensionMismatchError` on the first upsert after changing models.** Working as intended. You need
a new collection or Phase 15's migration.

**Tempted to add a `chunk_level` default to `_to_scored_chunks`.** Do not. Filtering is the caller's
decision (Phase 4's), and a default here would silently apply retrieval policy inside the storage
layer.

---

## 8. File 6 — `src/vectorstores/qdrant_store.py`

### The Problem

This is the file the whole phase exists for, and it has to get five things right that the prototype
in `Pipelines/` gets wrong.

The prototype indexes sparse vectors and then never queries them — so despite the code containing the
word "hybrid", retrieval is dense-only. It uses `recreate_collection`, which **deletes the collection
and rebuilds it**, so a second run destroys hours of indexing. It has no payload indexes, so every
filtered query is a full scan. It never deletes before upserting, so changed documents leave ghosts.
And it uses the synchronous client, so Phase 8's API server would block on every query.

### Design Decision

**`AsyncQdrantClient` throughout.** Every operation is a network round trip. Phase 4 issues three
query variations concurrently with `asyncio.gather`; that only works if the client is async.

**`collection_exists()` + `create_collection()`, never `recreate_collection`.** `initialize()` must be
idempotent and non-destructive. Calling it at every application startup — which Phase 8 does — must
never risk the index. This is deprecated in modern clients anyway, but the reason to avoid it is not
deprecation, it is that its name understates what it does.

**One collection for parents and children, separated by `chunk_level`.** This is Phase 2's decision
and this file honours it. The two rejected alternatives are worth recording: duplicating parent text
into each child payload costs roughly 4× storage on 39 million points, and a Redis docstore for
parents would put them in a cache that Phase 1 configured with `allkeys-lru` eviction — parents would
quietly disappear.

**The Universal Query API for hybrid search.** `query_points` with two `Prefetch` arms and a
`FusionQuery` performs both searches and fuses them **server-side, in one round trip**. The
alternative — two queries and client-side fusion — doubles the latency and moves 2× the candidate
payloads over the network. `client.search(...)` is the old single-vector API and cannot express this.

**Bulk mode.** Qdrant builds its HNSW graph incrementally as points arrive, which is the right
behaviour for a live collection and the wrong behaviour for a 39-million-point bulk load, where it
competes with the write path for CPU. Setting `indexing_threshold=0` defers graph construction until
you turn it back on. This is the single highest-leverage performance knob in the phase.

```python
import asyncio
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from config.settings import settings
from src.core.exceptions import (
    CollectionNotFoundError,
    DimensionMismatchError,
    VectorStoreError,
)
from src.core.logging import logger
from src.core.models import RetrievalMethod, ScoredChunk
from src.core.telemetry import telemetry

from .base import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
    VectorStoreBase,
    batched,
)

#: Normal operating value. Qdrant starts building HNSW once a segment holds this
#: many vectors. Set to 0 during bulk load to defer graph construction entirely.
_INDEXING_THRESHOLD = 20_000

_MAX_ATTEMPTS = 3

_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
}


class QdrantStore(VectorStoreBase):
    """Qdrant adapter: named dense + sparse vectors, server-side RRF fusion.

    One collection holds child chunks, parent chunks, and (from Phase 16) summary
    nodes, separated by the indexed `chunk_level` payload field.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        timeout: int | None = None,
        upsert_batch_size: int | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.upsert_batch_size = upsert_batch_size or settings.UPSERT_BATCH_SIZE
        self._vector_size: int | None = None

        # Constructing the client does not connect — no I/O happens until the
        # first call, so this is safe outside a running event loop.
        self._client = AsyncQdrantClient(
            url=url or settings.QDRANT_HOST,
            timeout=timeout or settings.QDRANT_TIMEOUT,
        )

    # ─── connection helpers ────────────────────────────────────────────────

    async def _retry(self, description: str, operation: Any) -> Any:
        """Run an async Qdrant call, retrying transient failures with backoff.

        `operation` is a zero-argument callable returning a coroutine, not a
        coroutine — a coroutine can only be awaited once, so a retry would raise
        `RuntimeError: cannot reuse already awaited coroutine`.
        """
        delay = 0.5
        last: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await operation()
            except Exception as exc:
                last = exc
                if attempt == _MAX_ATTEMPTS:
                    break
                logger.warning(
                    "Qdrant call failed; retrying",
                    extra={
                        "operation": description,
                        "attempt": attempt,
                        "error": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay)
                delay *= 2

        raise VectorStoreError(
            f"Qdrant operation failed: {description}",
            details={"collection": self.collection_name, "attempts": _MAX_ATTEMPTS},
            retryable=True,
        ) from last

    # ─── schema ────────────────────────────────────────────────────────────

    async def initialize(self, vector_size: int) -> None:
        """Create the collection and payload indexes if they do not exist.

        Idempotent and non-destructive: on an existing collection this verifies
        the schema and ensures indexes, and writes nothing. Safe to call at every
        application startup.

        Raises:
            DimensionMismatchError: the existing collection has a different
                vector width — you need Phase 15's migration, not a restart.
        """
        self._vector_size = vector_size

        exists = await self._retry(
            "collection_exists",
            lambda: self._client.collection_exists(self.collection_name),
        )

        if exists:
            await self._verify_schema(vector_size)
        else:
            logger.info(
                "Creating collection",
                extra={"collection": self.collection_name, "vector_size": vector_size},
            )
            await self._retry(
                "create_collection",
                lambda: self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        # IDF is not optional. FastEmbed's BM25 deliberately omits
                        # the inverse-document-frequency term because it cannot
                        # know the corpus; Qdrant supplies it here. Without this
                        # modifier, lexical scoring silently treats "the" and
                        # "indemnification" as equally informative.
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    },
                    # Payload on disk: text is the bulk of the data and is only
                    # read for results that are actually returned, not for every
                    # candidate scored.
                    on_disk_payload=True,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=_INDEXING_THRESHOLD,
                    ),
                ),
            )

        await self._ensure_payload_indexes()

    async def _verify_schema(self, vector_size: int) -> None:
        """Best-effort check that an existing collection matches expectations.

        Deliberately tolerant of introspection failures. The shape of
        `CollectionInfo` has changed across client releases, and I cannot execute
        anything here to confirm which shape you have. A schema check that crashes
        on a nested attribute would be worse than one that warns — the
        authoritative check remains the upsert itself, which Qdrant rejects on a
        genuine mismatch. What we do NOT do is let a real dimension mismatch
        through when we can see it.
        """
        info = await self._retry(
            "get_collection", lambda: self._client.get_collection(self.collection_name)
        )

        try:
            vectors = info.config.params.vectors
            existing = vectors[DENSE_VECTOR_NAME].size  # type: ignore[index]
        except Exception:
            logger.warning(
                "Could not introspect collection vector config",
                extra={"collection": self.collection_name},
            )
            return

        if existing != vector_size:
            raise DimensionMismatchError(
                "Existing collection has a different vector size",
                details={
                    "collection": self.collection_name,
                    "existing": existing,
                    "requested": vector_size,
                },
            )

        try:
            sparse = info.config.params.sparse_vectors or {}
            modifier = getattr(sparse.get(SPARSE_VECTOR_NAME), "modifier", None)
            if modifier != models.Modifier.IDF:
                logger.warning(
                    "Sparse vector field has no IDF modifier — BM25 scoring will "
                    "ignore term rarity. Recreate the collection to fix.",
                    extra={"collection": self.collection_name, "modifier": str(modifier)},
                )
        except Exception:
            logger.debug("Could not introspect sparse vector config")

    async def _ensure_payload_indexes(self) -> None:
        """Create the payload indexes. Existing indexes are left alone.

        Qdrant returns an error when the index already exists, and there is no
        `create_if_missing`. Swallowing the error is the accepted pattern — but
        note it is swallowed only here, where "already exists" is the expected
        outcome on every startup after the first.
        """
        for field, kind in PAYLOAD_INDEX_FIELDS:
            try:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=_SCHEMA_MAP[kind],
                )
                logger.info("Created payload index", extra={"field": field, "kind": kind})
            except Exception as exc:
                logger.debug(
                    "Payload index not created (probably already present)",
                    extra={"field": field, "error": type(exc).__name__},
                )

    async def set_bulk_mode(self, enabled: bool) -> None:
        """Defer or resume HNSW graph construction.

        During a bulk load, incremental index building competes with writes for
        CPU and is largely wasted work, since the graph is rebuilt as segments
        merge. `indexing_threshold=0` postpones it; restoring the threshold
        afterwards triggers a single optimisation pass.

        Turning bulk mode ON is a performance choice. Turning it back OFF is
        mandatory — leave it on and searches fall back to brute-force scans over
        every point, which at 39 million points means seconds per query.
        """
        threshold = 0 if enabled else _INDEXING_THRESHOLD
        await self._retry(
            "update_collection",
            lambda: self._client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
            ),
        )
        logger.info("Bulk indexing mode", extra={"enabled": enabled})

    # ─── writes ────────────────────────────────────────────────────────────

    @telemetry.measure_async("qdrant.upsert")
    async def upsert_points(
        self,
        chunks: Sequence[Any],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[dict[str, list]],
    ) -> int:
        """Write chunks with both vectors. Returns the number of points written.

        Point IDs are the chunks' deterministic UUIDv5s, so re-writing an
        unchanged chunk updates in place. That is NOT sufficient on its own —
        see `delete_by_doc_ids`.
        """
        self._validate_batch(chunks, dense, sparse, expected_dim=self._vector_size)
        if not chunks:
            return 0

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE_VECTOR_NAME: list(dense_vector),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                },
                payload=chunk.to_payload(),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]

        for group in batched(points, self.upsert_batch_size):
            await self._retry(
                "upsert",
                # wait=True is load-bearing, not caution. With wait=False the call
                # returns before the write is durable, and the indexing pipeline
                # would then call `commit()` on documents that may never land —
                # exactly the permanent silent data loss Phase 2's two-phase
                # commit exists to prevent.
                lambda group=group: self._client.upsert(
                    collection_name=self.collection_name, points=group, wait=True
                ),
            )

        return len(points)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove every chunk of one document. Returns the count deleted."""
        return await self.delete_by_doc_ids([doc_id])

    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        """Batch form of `delete_by_doc_id` — one round trip for many documents.

        The interface only requires the single-document version, but a batch of
        256 chunks typically spans dozens of documents, and issuing one delete per
        document would double the round trips of the whole ingestion run.

        Returns the number of points that matched, obtained with a separate exact
        count because Qdrant's delete does not report one. That count is only fast
        because `doc_id` has a payload index; on an unindexed field it is a full
        collection scan.
        """
        unique = sorted(set(doc_ids))
        if not unique:
            return 0

        selector = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=unique))]
        )

        existing = await self._retry(
            "count_before_delete",
            lambda: self._client.count(
                collection_name=self.collection_name, count_filter=selector, exact=True
            ),
        )
        if not existing.count:
            return 0

        await self._retry(
            "delete",
            lambda: self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=selector),
                wait=True,
            ),
        )

        logger.info(
            "Deleted existing points before re-index",
            extra={"documents": len(unique), "points": existing.count},
        )
        return existing.count

    # ─── reads ─────────────────────────────────────────────────────────────

    @telemetry.span("qdrant.hybrid_search", warn_over_ms=500)
    async def hybrid_search(
        self,
        dense_query: list[float],
        sparse_query: dict[str, list],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Dense + sparse search, fused server-side with Reciprocal Rank Fusion.

        One round trip. Both arms run inside Qdrant and only the fused top-`limit`
        payloads cross the network.

        Raises:
            CollectionNotFoundError: the collection does not exist — ingestion has
                not run, or QDRANT_COLLECTION is wrong.
            VectorStoreError: the query failed.
        """
        prefetch_limit = max(limit * settings.PREFETCH_MULTIPLIER, limit)
        query_filter = self._build_filter(filters)

        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=query_filter,
            )
        ]

        # A query of only stopwords stems to an empty sparse vector. Sending it
        # would ask Qdrant to match nothing; omitting the arm degrades cleanly to
        # dense-only for that one query, which is the correct behaviour, but say
        # so rather than letting it look like hybrid search happened.
        if sparse_query.get("indices"):
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"], values=sparse_query["values"]
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                )
            )
        else:
            logger.warning("Empty sparse query — falling back to dense-only retrieval")

        try:
            response = await self._client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            if "not found" in str(exc).lower() or "doesn't exist" in str(exc).lower():
                raise CollectionNotFoundError(
                    "Collection does not exist — has ingestion run?",
                    details={"collection": self.collection_name},
                ) from exc
            raise VectorStoreError(
                "Hybrid query failed",
                details={"collection": self.collection_name, "limit": limit},
                retryable=True,
            ) from exc

        method = (
            RetrievalMethod.HYBRID if len(prefetch) == 2 else RetrievalMethod.DENSE
        )
        return self._to_scored_chunks(
            [(point.payload or {}, point.score) for point in response.points], method=method
        )

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """Translate a plain dict into a Qdrant filter.

        Supported value shapes:
            "child"            → exact match
            ["child", "parent"] → match any
            {"gte": 2010, "lte": 2015} → range

        Unknown keys RAISE. This is the important decision in this method. A
        typo'd filter key that is silently ignored returns unfiltered results,
        which looks like success — and in Phase 11, where filters carry access
        control, "silently returned everything" is a security failure, not a bug.

        Raises:
            VectorStoreError: an unindexed or unknown field was requested.
        """
        if not filters:
            return None

        allowed = {field for field, _ in PAYLOAD_INDEX_FIELDS}
        conditions: list[models.FieldCondition] = []

        for key, value in filters.items():
            if key not in allowed:
                raise VectorStoreError(
                    "Unsupported filter field",
                    details={"field": key, "supported": sorted(allowed)},
                )

            if isinstance(value, dict):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(gte=value.get("gte"), lte=value.get("lte")),
                    )
                )
            elif isinstance(value, (list, tuple, set)):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
                )
            else:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )

        return models.Filter(must=conditions)

    async def count(self) -> int:
        """Approximate number of points. Used by health checks and progress logs.

        `exact=False` returns the segment-level estimate, which is effectively
        free. An exact count scans, and at 39 million points that is not something
        a health endpoint should do. Treat this number as approximate — the
        verification script asks for exact counts explicitly.
        """
        result = await self._retry(
            "count",
            lambda: self._client.count(collection_name=self.collection_name, exact=False),
        )
        return int(result.count)

    async def count_exact(self, filters: dict[str, Any] | None = None) -> int:
        """Exact count, optionally filtered. For verification and Phase 15 audits."""
        result = await self._retry(
            "count_exact",
            lambda: self._client.count(
                collection_name=self.collection_name,
                count_filter=self._build_filter(filters),
                exact=True,
            ),
        )
        return int(result.count)

    async def close(self) -> None:
        """Release connections. Called from Phase 8's lifespan shutdown."""
        await self._client.close()
```

### The Theory: Reciprocal Rank Fusion, and why it beats score averaging

You have two ranked lists — one from cosine similarity over dense vectors, one from BM25 — and you
need one list. The obvious approach is to combine the scores: `0.5 * dense + 0.5 * sparse`.

That approach is broken, and understanding why is the point of this section.

**The scores are not on the same scale and never will be.** Cosine similarity is bounded in
[-1, 1] and, for a good embedding model, real results cluster tightly in something like [0.6, 0.9].
BM25 is unbounded above; its values depend on corpus size, document lengths, and term rarity, and a
score of 18.4 is perfectly ordinary. Adding those numbers means whichever scale happens to be larger
dominates the result, and the weight you chose is not the weight you got.

You could normalise each list to [0, 1] first — that is Distribution-Based Score Fusion, and Qdrant
supports it as `models.Fusion.DBSF`. It works, but it inherits a nasty property: normalisation
depends on the *maximum score in this particular result set*, so the same document scores differently
depending on what else was retrieved alongside it. That makes results unstable in ways that are hard
to reason about, and makes evaluation noisier.

**RRF throws the scores away and keeps only the ranks.** Each document gets

\[ \text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)} \]

with `k` conventionally 60. A document ranked 1st by dense and 3rd by sparse scores
`1/61 + 1/63 ≈ 0.0323`. A document ranked 1st by dense only scores `1/61 ≈ 0.0164`.

Three properties fall out of this, and all three matter:

**Scale-invariance.** Ranks are ordinal. It is irrelevant that BM25 says 18.4 and cosine says 0.83;
only "which came first" survives. This is why RRF works with any pair of retrievers without tuning.

**Agreement is rewarded.** A document both retrievers like beats a document either one loves alone.
For legal retrieval that is exactly the desired bias: a clause that is both semantically on-topic and
contains the literal defined term is almost always the right answer.

**`k` damps the top.** The constant flattens the difference between rank 1 and rank 2 (1/61 vs 1/62)
relative to the difference between rank 1 and rank 50 (1/61 vs 1/110). This encodes a real property
of retrieval: the distinction between the 1st and 2nd result is mostly noise, while the distinction
between the 1st and 50th is signal.

The cost is genuine and you should be able to state it in an interview: **RRF discards magnitude.**
If dense retrieval is *certain* about its top hit — 0.95 against 0.62 for everything else — RRF
cannot express that confidence. It only knows that hit came first. In practice, across most corpora,
this loses less than the score-scale problems it avoids, which is why it is the default in Qdrant,
Elasticsearch, and Weaviate alike. Phase 7 will measure both on your corpus, and DBSF is one line
away if the numbers say otherwise.

### Why prefetch limits must exceed the final limit

A subtle failure worth internalising. The prefetch arms fetch candidates; the fusion step then picks
`limit` from the union. If each arm fetches 20 and you ask for 20 fused results, then the only
documents that can appear are those already in one arm's top 20 — you have paid for two retrievers
and let the fusion do almost nothing.

Worse, if a prefetch limit is *smaller* than the main limit plus offset, Qdrant can return fewer
results than requested, or none at all when paginating. `PREFETCH_MULTIPLIER=4` means each arm
fetches 80 candidates for a final 20, so fusion has real material to work with. The cost is bounded:
those 80 candidates never leave the server, only the fused 20 carry payloads over the network.

### Failure Modes

**`Not existing vector name error: dense-bge`.** The collection was created with different vector
names — almost always by an older script or a notebook. The names live in `base.py` for exactly this
reason. Recreate the collection.

**Hybrid search returns nothing on a collection you know has data.** Check the filter first. A
`chunk_level="child"` filter against a collection ingested before parent-child chunking existed will
match zero points, because those chunks are all `standalone`.

**Everything is slow after a bulk load.** You left bulk mode on. With `indexing_threshold=0` there is
no HNSW graph, so every search is an exact brute-force scan. `await store.set_bulk_mode(False)` and
wait for the optimiser; `get_collection` reports status `yellow` while it works and `green` when done.

**Deletes take minutes.** The `doc_id` payload index is missing. `_ensure_payload_indexes` swallows
creation errors by design, so an index that failed to create for a real reason is only visible in the
debug log. Check with `get_collection(...).payload_schema`.

**`RuntimeError: cannot reuse already awaited coroutine` inside `_retry`.** You passed
`self._client.count(...)` instead of `lambda: self._client.count(...)`. A coroutine object is
single-use; the retry needs a factory.

**A closure captures the wrong loop variable.** Note `lambda group=group:` in the upsert loop. Python
closures capture variables, not values, so a plain `lambda: ...group...` inside a `for` would see the
final group on every retry. The default-argument trick binds the value at definition time.

---

## 9. File 7 — `src/vectorstores/chroma_store.py`

### The Problem

One implementation of an interface is not an abstraction, it is indirection with extra steps. Until a
second store exists, there is no evidence that `BaseVectorStore` describes vector storage rather than
describing Qdrant.

Chroma also serves a practical purpose: it runs in-process with no Docker, so Phase 10's integration
tests and anyone cloning the repo can exercise the full pipeline without a running server.

### Design Decision

**Implement it honestly, including the part it cannot do.** Chroma has no sparse vector support.
`hybrid_search` therefore performs a dense-only search, tags the results
`RetrievalMethod.DENSE` so downstream code and evaluation can see what actually happened, and warns
once. It does not pretend.

Tagging matters more than warning. Phase 7 evaluates retrieval quality; if a dense-only run reported
itself as hybrid, the comparison between stores would be meaningless and nobody would notice.

**Wrap the synchronous client in `asyncio.to_thread`.** Chroma's async client API has moved around
between releases, and I cannot verify it by running it. The synchronous persistent-client API
(`get_or_create_collection`, `add`, `query`, `delete`, `count`) has been stable for a long time.
Wrapping stable sync calls in a thread is the lower-risk choice, and the pattern is already
established by `LocalEmbeddingProvider`.

```python
import asyncio
from collections.abc import Sequence
from typing import Any

import chromadb

from config.settings import settings
from src.core.exceptions import VectorStoreError
from src.core.logging import logger
from src.core.models import RetrievalMethod, ScoredChunk

from .base import VectorStoreBase


class ChromaStore(VectorStoreBase):
    """Local, embedded, dense-only fallback store.

    Exists to prove `BaseVectorStore` is a real abstraction and to let tests run
    without Docker. NOT a production option for this project: no sparse vectors
    means no hybrid retrieval, which is half the point of the system.
    """

    def __init__(self, collection_name: str | None = None, path: str | None = None) -> None:
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._path = path or settings.CHROMA_PATH
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection: Any = None
        self._warned_sparse = False

    async def initialize(self, vector_size: int) -> None:
        """Create or open the collection.

        `vector_size` is accepted and ignored: Chroma infers dimensionality from
        the first vector added and has no schema to validate against. That is a
        genuine weakness — a dimension mismatch surfaces as an error on `add`
        rather than at startup.
        """
        logger.warning(
            "Using ChromaStore — retrieval will be dense-only, not hybrid",
            extra={"collection": self.collection_name, "path": self._path},
        )
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert_points(
        self,
        chunks: Sequence[Any],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[dict[str, list]],
    ) -> int:
        """Write dense vectors and payloads. `sparse` is accepted and discarded."""
        self._validate_batch(chunks, dense, sparse)
        if not chunks:
            return 0
        if not self._warned_sparse:
            logger.warning("ChromaStore discards sparse vectors")
            self._warned_sparse = True

        payloads = [chunk.to_payload() for chunk in chunks]

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(v) for v in dense],
            documents=[chunk.text for chunk in chunks],
            # Chroma metadata values must be scalars, and `year` may be None.
            # Dropping None keys is safe here because `Chunk.from_payload`
            # defaults every optional field.
            metadatas=[{k: v for k, v in p.items() if v is not None} for p in payloads],
        )
        return len(chunks)

    async def hybrid_search(
        self,
        dense_query: list[float],
        sparse_query: dict[str, list],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Dense-only search. The name is the interface's, not a description."""
        where = {k: v for k, v in (filters or {}).items() if not isinstance(v, (dict, list))}

        try:
            response = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[dense_query],
                n_results=limit,
                where=where or None,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                "Chroma query failed", details={"collection": self.collection_name}
            ) from exc

        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        # Chroma returns cosine DISTANCE (0 = identical); the project's contract is
        # a similarity score where higher is better. These scores are not
        # comparable to Qdrant's RRF scores, which is fine — ranks are what
        # downstream code uses.
        rows = [(meta, 1.0 - float(dist)) for meta, dist in zip(metadatas, distances, strict=True)]
        return self._to_scored_chunks(rows, method=RetrievalMethod.DENSE)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        existing = await asyncio.to_thread(self._collection.get, where={"doc_id": doc_id})
        count = len(existing.get("ids") or [])
        if count:
            await asyncio.to_thread(self._collection.delete, where={"doc_id": doc_id})
        return count

    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        total = 0
        for doc_id in sorted(set(doc_ids)):
            total += await self.delete_by_doc_id(doc_id)
        return total

    async def count(self) -> int:
        return int(await asyncio.to_thread(self._collection.count))

    async def close(self) -> None:
        """No-op. Chroma's persistent client holds a local file handle, not a
        connection pool, and has no explicit close in the stable API."""
        return None
```

### Failure Modes

**Retrieval quality is noticeably worse than with Qdrant.** Expected. No BM25 means exact term
matches — section numbers, defined terms, party names — carry no extra weight.

**`ValueError: Expected metadata value to be a str, int, float or bool`.** A payload field holds
`None` or a list. The filter comprehension in `upsert_points` handles `None`; if you add a list-valued
payload field later, handle it there.

**Memory grows during a large run.** Chroma's persistent client keeps a good deal in memory. It is
not built for 39 million points. Use it for tests and samples.

---

## 10. File 8 — `src/vectorstores/factory.py`

### Design Decision

Same registry shape as the embedding factory, with **one deliberate difference: no caching.**

A vector store owns an HTTP connection pool. An `AsyncQdrantClient` created inside one event loop and
reused inside another produces `RuntimeError: Event loop is closed` or, worse, hangs — a class of bug
that is miserable to diagnose because it depends on which test ran first. Phase 8's FastAPI lifespan
will create exactly one store at startup and close it at shutdown; Phase 10's tests create one per
test. Both patterns require that the *caller* owns the lifetime.

So: expensive-and-stateless gets cached (embeddings), connection-owning gets constructed on demand
(stores). Whoever calls this function is responsible for calling `close()`.

```python
from collections.abc import Callable

from config.settings import settings
from src.core.exceptions import VectorStoreError
from src.core.interfaces import BaseVectorStore
from src.core.logging import logger


def _build_qdrant() -> BaseVectorStore:
    from .qdrant_store import QdrantStore

    return QdrantStore()


def _build_chroma() -> BaseVectorStore:
    from .chroma_store import ChromaStore

    return ChromaStore()


_BUILDERS: dict[str, Callable[[], BaseVectorStore]] = {
    "qdrant": _build_qdrant,
    "chroma": _build_chroma,
}


def get_vector_store(name: str | None = None) -> BaseVectorStore:
    """Construct the configured vector store. NOT cached, on purpose.

    A store owns a connection pool bound to the event loop that uses it. Caching
    one across loops produces "Event loop is closed" errors and intermittent
    hangs. The caller owns the instance and must `await store.close()`.

    Raises:
        VectorStoreError: the configured name has no registered builder.
    """
    key = (name or settings.VECTOR_STORE).lower()

    builder = _BUILDERS.get(key)
    if builder is None:
        raise VectorStoreError(
            "Unknown vector store",
            details={"requested": key, "available": sorted(_BUILDERS)},
        )

    logger.info("Building vector store", extra={"store": key})
    return builder()
```

---

## 11. File 9 — `src/pipelines/indexing_pipeline.py`

### The Problem

Phase 2 produces a **synchronous generator** of `ChunkBatch` objects, driven by a `multiprocessing`
pool. Phase 3's store is **async**. Something must join them, and that something also carries the
transaction responsibility that Phase 2 explicitly refused:

> Ingestion only *stages* checkpoint records; it must never mark a document done, because it does not
> know whether storage succeeded. Phase 3's ingest driver must implement this loop.

Three distinct hazards live in this one file.

**Blocking the event loop.** `for batch in pipeline.run()` inside an `async def` blocks the loop for
however long the worker pool takes to fill a batch — often seconds. Nothing else on that loop runs.

**`StopIteration` crossing an async boundary.** The natural fix, `await asyncio.to_thread(next, it)`,
is a trap. When the generator is exhausted, `next` raises `StopIteration`, and Python explicitly
converts a `StopIteration` that escapes into a coroutine or generator frame into
`RuntimeError: coroutine raised StopIteration` (PEP 479). The error message points nowhere useful.

**Non-atomic delete-then-upsert.** The mandatory contract is delete, then upsert. Those are two
separate operations with no transaction around them. A crash in between leaves the document deleted
and not re-written.

### Design Decision

**Pull batches with `next(iterator, None)` inside `asyncio.to_thread`.** The two-argument form of
`next` returns the sentinel instead of raising, so nothing ever throws `StopIteration` across the
boundary. This is a small detail with a large diagnostic cost if you get it wrong.

**Close the generator explicitly in a `finally`.** Abandoning a half-consumed generator leaves
Phase 2's `with mp.Pool(...)` and `with IngestionCheckpoint(...)` blocks un-exited until garbage
collection runs — which on Windows means orphaned worker processes. `generator.close()` raises
`GeneratorExit` inside `run()` at its current yield point, unwinding both context managers properly.

**Accept that delete-then-upsert is not atomic, and make the non-atomicity safe.** This is the key
insight of the file. If the process dies between the delete and the upsert, the document is missing
from the index — but it is also **un-committed in the checkpoint**, because `commit()` only runs
after a successful upsert. The next run reprocesses it from scratch. The window is real; recovery is
automatic. Two-phase commit is what makes an ordinary, non-transactional pair of operations safe, and
this is the concrete payoff for the design Phase 2 was rewritten to support.

**Embedding failures abort the run; they do not go to the dead-letter queue.** A per-document parse
failure is data being bad and belongs in the DLQ. An embedding failure is the *system* being broken —
the model is gone, or memory is exhausted — and it will fail identically for the next 600,000
documents. Failing fast is correct.

```python
import asyncio
from dataclasses import dataclass

from config.settings import settings
from src.core.exceptions import RAGException
from src.core.interfaces import BaseEmbeddingProvider, BaseVectorStore
from src.core.logging import logger, new_request_id
from src.core.telemetry import telemetry
from src.ingestion.pipeline import ChunkBatch, IngestionPipeline


@dataclass
class IndexingStats:
    """Storage-side counters. Ingestion keeps its own in `pipeline.stats`."""

    batches: int = 0
    documents: int = 0
    chunks_indexed: int = 0
    points_deleted: int = 0
    batches_rolled_back: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "batches": self.batches,
            "documents": self.documents,
            "chunks_indexed": self.chunks_indexed,
            "points_deleted": self.points_deleted,
            "rolled_back": self.batches_rolled_back,
        }


class IndexingPipeline:
    """Joins Pipeline 1 to the vector store and owns the transaction boundary.

    Dependencies are injected rather than constructed here, so Phase 10 can pass
    a fake store and Phase 15 can point the same driver at a shadow collection.
    """

    def __init__(
        self,
        ingestion: IngestionPipeline,
        embedder: BaseEmbeddingProvider,
        store: BaseVectorStore,
        bulk_mode: bool = True,
    ) -> None:
        self.ingestion = ingestion
        self.embedder = embedder
        self.store = store
        self.bulk_mode = bulk_mode
        self.stats = IndexingStats()

    @telemetry.measure_async("indexing.run")
    async def run(
        self,
        limit: int | None = None,
        years: set[str] | None = None,
        resume: bool = True,
    ) -> IndexingStats:
        """Ingest, embed, and store the corpus. Returns storage-side statistics."""
        new_request_id()

        await self.store.initialize(self.embedder.dense_dimensions)
        await self._set_bulk_mode(True)

        batches = self.ingestion.run(limit=limit, years=years, resume=resume)
        try:
            while True:
                # The two-argument `next` returns the sentinel at exhaustion
                # instead of raising StopIteration. A StopIteration escaping into
                # a coroutine frame becomes "RuntimeError: coroutine raised
                # StopIteration" (PEP 479), which points nowhere useful.
                batch = await asyncio.to_thread(next, batches, None)
                if batch is None:
                    break
                await self._index_batch(batch)
        finally:
            # Unwinds Phase 2's `with mp.Pool(...)` and checkpoint context
            # managers at their current yield point. Without this, an early exit
            # leaves worker processes alive until garbage collection.
            batches.close()
            await self._set_bulk_mode(False)

        logger.info(
            "Indexing complete",
            extra={**self.stats.summary(), **self.ingestion.stats.summary()},
        )
        return self.stats

    async def _index_batch(self, batch: ChunkBatch) -> None:
        """Embed and store one batch, then commit or roll back the checkpoint."""
        texts = [chunk.text for chunk in batch.chunks]

        # Dense and sparse are independent. Both offload to threads internally,
        # and ONNX Runtime releases the GIL during inference, so these overlap
        # usefully — though not linearly. The larger point is that neither blocks
        # the event loop.
        dense, sparse = await asyncio.gather(
            self.embedder.embed_dense(texts),
            self.embedder.embed_sparse(texts),
        )

        doc_ids = {chunk.doc_id for chunk in batch.chunks}

        try:
            # MANDATORY: delete before upsert. Deterministic IDs make an unchanged
            # re-ingest idempotent, but they cannot remove chunks that no longer
            # exist. A document that shrinks from 400 chunks to 300 would leave
            # 100 stale points that still get retrieved and cited.
            #
            # These two calls are not atomic, and that is acceptable: a crash
            # between them leaves the document absent AND un-committed, so the
            # next run reprocesses it. Two-phase commit is what makes a
            # non-transactional pair of operations safe.
            deleted = await self.store.delete_by_doc_ids(doc_ids)
            written = await self.store.upsert_points(batch.chunks, dense, sparse)

        except Exception as exc:
            self.stats.batches_rolled_back += 1
            self.ingestion.rollback()
            logger.error(
                "Storage failed; batch rolled back and will be retried next run",
                extra={"documents": len(doc_ids), "error": type(exc).__name__},
            )
            raise

        # Only now are these documents genuinely done. `commit()` persists the
        # checkpoint records that ingestion merely staged.
        committed = self.ingestion.commit()

        self.stats.batches += 1
        self.stats.documents += batch.document_count
        self.stats.chunks_indexed += written
        self.stats.points_deleted += deleted

        logger.info(
            "Batch indexed",
            extra={
                "chunks": written,
                "deleted": deleted,
                "documents_committed": committed,
                "total_chunks": self.stats.chunks_indexed,
            },
        )

    async def _set_bulk_mode(self, enabled: bool) -> None:
        """Toggle deferred HNSW construction, if the store supports it.

        Duck-typed rather than added to `BaseVectorStore`: it is a Qdrant-specific
        optimisation, and putting a vendor performance knob in the shared
        interface would force every future store to implement a no-op.
        """
        if not self.bulk_mode:
            return
        setter = getattr(self.store, "set_bulk_mode", None)
        if setter is None:
            return
        try:
            await setter(enabled)
        except RAGException as exc:
            # Never let a performance optimisation abort a nine-hour run.
            logger.warning("Could not toggle bulk mode", extra={"error": str(exc)})
```

### The entry point

Save this as `scripts/index_corpus.py`. The `if __name__ == "__main__":` guard is **mandatory on
Windows** — without it, `multiprocessing` spawn re-imports the module, which starts the pool again,
recursively.

```python
"""Run the full ingest → embed → store pipeline.

    python scripts/index_corpus.py --limit 100 --years 2003 2004
"""
import argparse
import asyncio

from src.core.logging import logger
from src.embeddings.factory import get_embedding_provider
from src.ingestion.pipeline import IngestionPipeline
from src.pipelines.indexing_pipeline import IndexingPipeline
from src.vectorstores.factory import get_vector_store


async def main(args: argparse.Namespace) -> None:
    embedder = get_embedding_provider()
    store = get_vector_store()

    pipeline = IndexingPipeline(
        ingestion=IngestionPipeline(workers=args.workers, batch_size=args.batch_size),
        embedder=embedder,
        store=store,
    )
    try:
        stats = await pipeline.run(
            limit=args.limit,
            years=set(args.years) if args.years else None,
            resume=not args.no_resume,
        )
        logger.info("Done", extra=stats.summary())
    finally:
        # The factory does not cache stores, so this instance is ours to close.
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-resume", action="store_true")
    asyncio.run(main(parser.parse_args()))
```

### Failure Modes

**`RuntimeError: coroutine raised StopIteration`.** You used `await asyncio.to_thread(next, batches)`
without the sentinel. Add the `None`.

**Worker processes survive after Ctrl+C.** The generator was not closed. The `finally` block handles
the normal case; if you restructure this loop, keep it.

**Every batch rolls back.** Read the logged error type. `CollectionNotFoundError` means
`initialize()` never ran or `QDRANT_COLLECTION` is wrong. `DimensionMismatchError` means the model
and the collection disagree. Neither is retryable; the run correctly aborts on the first one.

**Resume reprocesses documents that are already in Qdrant.** They were upserted but never committed —
the process died between the upsert and the checkpoint write. Harmless: deterministic IDs mean the
re-write updates in place rather than duplicating. This is the deliberate direction of the trade;
the opposite error (committed but not stored) would be permanent.

**Progress stalls with no log output.** Batches are 256 chunks, which at ~60 chunks per document is
about four documents — but the *ingestion* side must fill that batch first, and with a cold worker
pool the first batch can take a minute. Phase 2 logs progress every 1,000 documents.

**Memory climbs steadily.** Something is holding batches. `run()` releases each one after indexing;
check that you are not accumulating `IndexingStats` history or logging whole chunk lists.

---

## 12. Verification (deferred)

Save as `scripts/verify_phase3.py`. Run on the machine that has Qdrant, after Phase 2's script
passes. It uses a throwaway collection and deletes it at the end, so it never touches your real
index.

```python
"""Phase 3 verification. Run from the project root, with Qdrant running:

    python scripts/verify_phase3.py

Creates and destroys a temporary collection. Never writes to QDRANT_COLLECTION.
"""
import asyncio
import sys

from src.core.models import Chunk, ChunkLevel
from src.core.utils import make_chunk_id, make_doc_id, make_section_id
from src.embeddings.fastembed_provider import FastEmbedProvider
from src.vectorstores.base import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.vectorstores.qdrant_store import QdrantStore

TEMP_COLLECTION = "verify_phase3_tmp"

DOC_ID = make_doc_id("data/contracts/2003/verify.txt")
SECTION_ID = make_section_id(DOC_ID, "Termination", 0)

TEXTS = [
    "Either party may terminate this Agreement upon ninety (90) days written notice.",
    "The Company shall indemnify and hold harmless the Purchaser against all Losses.",
    "This Agreement shall be governed by the laws of the State of Delaware.",
    "Confidential Information shall not be disclosed to any third party.",
    "The Purchase Price shall be paid in immediately available funds at Closing.",
]


def make_chunks(count: int, level: ChunkLevel = ChunkLevel.CHILD) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=make_chunk_id(DOC_ID, SECTION_ID, i),
            doc_id=DOC_ID,
            section_id=SECTION_ID,
            text=TEXTS[i % len(TEXTS)],
            chunk_index=i,
            token_count=20,
            contract_name="Verification Agreement",
            file_name="verify.txt",
            section_title="Termination",
            year=2003,
            chunk_level=level,
        )
        for i in range(count)
    ]


async def embed(provider: FastEmbedProvider, chunks: list[Chunk]):
    texts = [c.text for c in chunks]
    dense = await provider.embed_dense(texts)
    sparse = await provider.embed_sparse(texts)
    return dense, sparse


def check_embeddings(provider: FastEmbedProvider) -> None:
    dims = provider.dense_dimensions
    assert dims > 0, "dense_dimensions must be positive"

    dense = provider.embed_dense_sync(TEXTS)
    assert len(dense) == len(TEXTS), f"alignment: {len(dense)} vectors for {len(TEXTS)} texts"
    assert all(len(v) == dims for v in dense), "inconsistent vector widths"

    sparse = provider.embed_sparse_sync(TEXTS)
    assert len(sparse) == len(TEXTS), "sparse alignment"
    assert all(len(s["indices"]) == len(s["values"]) for s in sparse), "indices/values mismatch"
    assert any(s["indices"] for s in sparse), "every sparse vector is empty — wrong model?"

    query = provider.embed_query_sync("termination notice period")
    assert len(query) == dims, "query and document vectors must share a width"

    # Asymmetric models produce a DIFFERENT vector for the same text via
    # query_embed than via passage_embed. If these are identical, either the
    # model is symmetric or query_embed is not applying its prefix — worth
    # knowing which, because Phase 4 depends on the query path.
    same_text_as_passage = provider.embed_dense_sync([TEXTS[0]])[0]
    same_text_as_query = provider.embed_query_sync(TEXTS[0])
    identical = all(abs(a - b) < 1e-9 for a, b in zip(same_text_as_passage, same_text_as_query))
    print(f"  query/passage embeddings identical: {identical} "
          f"(False expected for bge; True means a symmetric model)")

    print(f"✓ embeddings: {dims} dense dims, sparse terms per text = "
          f"{[len(s['indices']) for s in sparse]}")


async def check_schema(store: QdrantStore) -> None:
    info = await store._client.get_collection(TEMP_COLLECTION)

    vectors = info.config.params.vectors
    assert DENSE_VECTOR_NAME in vectors, f"missing named vector {DENSE_VECTOR_NAME}"

    sparse_config = info.config.params.sparse_vectors or {}
    assert SPARSE_VECTOR_NAME in sparse_config, f"missing sparse vector {SPARSE_VECTOR_NAME}"

    modifier = getattr(sparse_config[SPARSE_VECTOR_NAME], "modifier", None)
    assert str(modifier).lower().endswith("idf"), (
        "sparse field has no IDF modifier — BM25 will ignore term rarity and "
        "nothing will ever report an error"
    )

    schema = info.payload_schema or {}
    for field in ("doc_id", "chunk_level", "year", "section_title"):
        assert field in schema, f"missing payload index on {field}"

    print("✓ schema: named vectors, IDF modifier, and all four payload indexes present")


async def check_roundtrip(store: QdrantStore, provider: FastEmbedProvider) -> None:
    """Every persisted field must survive to_payload → Qdrant → from_payload.

    Phase 1's original verification only compared `.text`, which is exactly why it
    missed `chunk_index` being dropped from `to_payload`. Compare everything.
    """
    chunks = make_chunks(3)
    dense, sparse = await embed(provider, chunks)
    await store.upsert_points(chunks, dense, sparse)

    query_dense = await provider.embed_query("termination ninety days written notice")
    query_sparse = provider.embed_sparse_query_sync("termination ninety days written notice")
    results = await store.hybrid_search(query_dense, query_sparse, limit=3)

    assert results, "hybrid search returned nothing"

    top = results[0]
    assert top.chunk.text, "ScoredChunk.chunk.text is empty"
    assert top.rank == 0 and results[-1].rank == len(results) - 1, "ranks must be 0-based"
    assert top.effective_score == top.score, "rerank_score should be unset before Phase 4"

    original = {c.chunk_id: c for c in chunks}[top.chunk.chunk_id]
    for field in (
        "doc_id", "section_id", "text", "chunk_index", "token_count",
        "contract_name", "file_name", "section_title", "year",
        "chunk_level", "parent_id",
    ):
        assert getattr(top.chunk, field) == getattr(original, field), (
            f"round-trip lost {field}: {getattr(top.chunk, field)!r} "
            f"!= {getattr(original, field)!r}"
        )

    print(f"✓ round-trip: all persisted fields survived, top score={top.score:.4f}")


async def check_idempotence(store: QdrantStore, provider: FastEmbedProvider) -> None:
    chunks = make_chunks(3)
    dense, sparse = await embed(provider, chunks)

    await store.upsert_points(chunks, dense, sparse)
    first = await store.count_exact()
    await store.upsert_points(chunks, dense, sparse)
    second = await store.count_exact()

    assert first == second, f"re-upsert duplicated points: {first} → {second}"
    print(f"✓ idempotence: re-upserting {len(chunks)} chunks left the count at {second}")


async def check_ghosts(store: QdrantStore, provider: FastEmbedProvider) -> None:
    """The failure deterministic IDs cannot prevent.

    Index 5 chunks, then re-index the same document as 3 chunks. Without
    delete-before-upsert, chunks 3 and 4 linger forever holding stale text.
    """
    five = make_chunks(5)
    dense, sparse = await embed(provider, five)
    await store.delete_by_doc_ids([DOC_ID])
    await store.upsert_points(five, dense, sparse)
    assert await store.count_exact({"doc_id": DOC_ID}) == 5

    three = make_chunks(3)
    dense, sparse = await embed(provider, three)
    deleted = await store.delete_by_doc_ids([DOC_ID])
    await store.upsert_points(three, dense, sparse)

    remaining = await store.count_exact({"doc_id": DOC_ID})
    assert remaining == 3, f"{remaining - 3} ghost points survived re-indexing"
    assert deleted == 5, f"delete reported {deleted}, expected 5"

    print("✓ ghosts: shrinking a document from 5 chunks to 3 left exactly 3 points")


async def check_filters(store: QdrantStore, provider: FastEmbedProvider) -> None:
    children = make_chunks(3, ChunkLevel.CHILD)
    parents = make_chunks(2, ChunkLevel.PARENT)
    for group in (children, parents):
        dense, sparse = await embed(provider, group)
        await store.upsert_points(group, dense, sparse)

    query_dense = await provider.embed_query("indemnify hold harmless")
    query_sparse = provider.embed_sparse_query_sync("indemnify hold harmless")

    filtered = await store.hybrid_search(
        query_dense, query_sparse, limit=10, filters={"chunk_level": "child"}
    )
    assert filtered, "child filter returned nothing"
    assert all(r.chunk.chunk_level == ChunkLevel.CHILD for r in filtered), (
        "chunk_level filter leaked parents — Phase 4 depends on this"
    )

    ranged = await store.hybrid_search(
        query_dense, query_sparse, limit=10, filters={"year": {"gte": 2000, "lte": 2010}}
    )
    assert ranged, "year range filter returned nothing"

    try:
        await store.hybrid_search(query_dense, query_sparse, filters={"nonexistent": "x"})
    except Exception as exc:
        assert "Unsupported filter field" in str(exc), f"wrong error for a bad filter: {exc}"
    else:
        raise AssertionError(
            "an unknown filter key was silently ignored — in Phase 11 that is a "
            "security failure, not a bug"
        )

    print("✓ filters: chunk_level isolates children, ranges work, unknown keys raise")


async def check_lexical_advantage(store: QdrantStore, provider: FastEmbedProvider) -> None:
    """Evidence that the sparse arm is actually contributing.

    A rare, exact token is what BM25 is for. If hybrid search cannot find a
    verbatim phrase, the sparse arm is not being queried — which is precisely the
    prototype's bug.
    """
    chunks = make_chunks(5)
    dense, sparse = await embed(provider, chunks)
    await store.delete_by_doc_ids([DOC_ID])
    await store.upsert_points(chunks, dense, sparse)

    phrase = "immediately available funds at Closing"
    query_dense = await provider.embed_query(phrase)
    query_sparse = provider.embed_sparse_query_sync(phrase)
    assert query_sparse["indices"], "sparse query is empty — BM25 stemmed everything away"

    results = await store.hybrid_search(query_dense, query_sparse, limit=5)
    assert phrase.lower() in results[0].chunk.text.lower(), (
        f"verbatim phrase did not rank first; got {results[0].chunk.text[:80]!r}"
    )
    print("✓ lexical: a verbatim phrase ranks first under RRF fusion")


async def main() -> int:
    provider = FastEmbedProvider()
    provider.warmup()
    check_embeddings(provider)

    store = QdrantStore(collection_name=TEMP_COLLECTION)
    try:
        await store.initialize(provider.dense_dimensions)
        await check_schema(store)
        await check_roundtrip(store, provider)
        await check_idempotence(store, provider)
        await check_ghosts(store, provider)
        await check_filters(store, provider)
        await check_lexical_advantage(store, provider)
    except AssertionError as exc:
        print(f"\n✗ FAILED: {exc}")
        return 1
    finally:
        try:
            await store._client.delete_collection(TEMP_COLLECTION)
            print(f"  (cleaned up {TEMP_COLLECTION})")
        finally:
            await store.close()

    print("\nPhase 3 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

### What to look at, beyond the assertions

**The `query/passage embeddings identical` line.** For `bge-small-en-v1.5` this should print `False`.
If it prints `True`, FastEmbed is not applying a query prefix for your model — which is not
necessarily wrong (OpenAI models are symmetric), but you should know which regime you are in before
Phase 4 tunes retrieval.

**Sparse terms per text.** The printed list should be roughly two-thirds of the word count of each
sentence — stemming merges some tokens and stopwords are dropped. If every entry is 0, the sparse
model is not what you think it is. If they equal the raw word counts, stopword removal is not running.

**The top hybrid score.** RRF scores are small (around 0.016–0.033 for two arms with `k=60`) and are
*not* similarities. If you see values near 0.8, you are looking at a cosine score, which means fusion
did not happen and only one arm ran.

**After a real ingestion run,** check the collection status with `get_collection`. `green` means the
HNSW index is built; `yellow` means the optimiser is still working, and searches until then are
slower but correct.

---

## 13. What Phase 3 Bought You

The system can now store and retrieve. Concretely:

**A vector database with a schema that supports the whole roadmap.** Named dense and sparse vectors
in one collection; parents, children, and future RAPTOR summaries separated by an indexed
`chunk_level`; payload indexes on the four fields anything will ever filter by. Phase 4 filters to
children, Phase 11 will add tenant filters, Phase 16 will add summary nodes — none of them needs a
migration.

**Hybrid search in one round trip.** Server-side RRF over both arms. This is the single largest
retrieval-quality win in the project, and it is the specific thing the prototype in `Pipelines/`
claimed to do and did not.

**A closed loop from disk to index.** `scripts/index_corpus.py` walks 37 GB, chunks it across every
core, embeds it, stores it, and can be killed at any moment and resumed without duplication or loss.
The two-phase commit that Phase 2 was rewritten to support now has the consumer it was designed for.

**Two implementations of each interface.** Not decoration: the second implementation is the only
proof that `BaseEmbeddingProvider` and `BaseVectorStore` are abstractions rather than wrappers. Phase
15's shadow indexing depends on running two of each side by side.

**The delete-before-upsert contract, enforced in code.** The stale-ghost failure — the one that
deterministic IDs cannot prevent and that produces confidently cited text that no longer exists in
the source document — now has a test that fails if anyone removes the delete.

### What is deliberately not here

**Query expansion, HyDE, reranking, and parent substitution.** All Phase 4. This phase provides
`hybrid_search(dense, sparse, limit, filters)` and nothing above it. `hybrid_search` does not know
that Phase 4 will filter to `chunk_level="child"` and swap in parents before generation — that is
retrieval policy, and putting it in the storage layer would make the store untestable and Phase 4
unmodifiable.

**Quantisation and multi-node sharding.** Scalar quantisation would cut vector memory by 4× and is
one config block away, but it trades recall for RAM and there is no measurement yet to justify the
trade. Phase 7 provides the measurement. Adding it now would be tuning without data.

**Retry policy above the store.** `QdrantStore._retry` handles transient connection failures for a
single call. It does not resurrect a batch whose upsert failed three times — that surfaces to the
indexing pipeline, which rolls back and lets the next run retry. Recovery belongs at the transaction
boundary, not inside the adapter.

---

## Next

**Phase 4 — Hybrid Retrieval and Reranking.** It consumes `hybrid_search`, adds multi-query expansion
and HyDE, filters to `chunk_level == "child"` and substitutes parents before generation, and puts a
cross-encoder reranker in front of the LLM. The two contracts it inherits from this phase: results
are `ScoredChunk` objects accessed as `scored.chunk.text`, and `rerank_score` is `None` until the
reranker sets it.
