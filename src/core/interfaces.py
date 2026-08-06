from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.models import Chunk, Document, ScoredChunk, Section



class BaseDocumentLoader(ABC):
    """Bytes on disk → a `Document`. One implementation per file format."""

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Whether this loader can handle the given path. Used by the factory to dispatch."""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Read and decode one file.

        Raises:
            DocumentLoadError: unreadable, undecodable, or empty.
        """


class BaseParser(ABC):
    """A `Document` → its structural `Section`s."""

    @abstractmethod
    def parse(self, document: Document) -> list[Section]:
        """Segment a document along structural boundaries.

        Raises:
            ParsingError: the document structure could not be interpreted.
        """

class BaseChunker(ABC):
    """A `Section` → retrievable `Chunk`s."""

    @abstractmethod
    def chunk(self, section: Section, document: Document) -> list[Chunk]:
        """Split a section into token-bounded chunks.

        `document` is passed so the chunker can denormalise contract metadata
        onto each chunk (see the `Chunk` model's rationale).
        """

class BaseEmbeddingProvider(ABC):
    """Text → vectors. Dense and sparse.

    Async, even though the local FastEmbed implementation is CPU-bound. The reason
    is that the *other* planned implementation — OpenAI — is a network call, and a
    synchronous interface would force it to block the event loop on every request.
    The interface must accommodate both, so it is async, and the CPU-bound
    implementation offloads to a thread internally (roadmap §7.2).

    Phase 2's multiprocessing ingestion path needs a synchronous entry point and so
    calls `embed_dense_sync` directly, bypassing the event loop entirely.
    """

    @property
    @abstractmethod
    def dense_dimensions(self) -> int:
        """Vector size, needed to create the collection schema in Phase 3."""

    @abstractmethod
    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense semantic embeddings."""

    @abstractmethod
    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        """Generate sparse BM25-style vectors as {"indices": [...], "values": [...]}."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from `embed_dense` because some models require an asymmetric
        instruction prefix for queries versus documents — `bge` is one of them.
        """

    @abstractmethod
    async def embed_sparse_query(self, text: str) -> dict[str, list]:
        """Sparse vector for a single query.

        Separate from `embed_sparse` for the same reason `embed_query` is separate
        from `embed_dense`, and it is not a nicety: BM25's document
        representation applies term-frequency saturation and length
        normalisation, neither of which is meaningful for a query. Using the
        document-side embedding for a query produces a subtly mis-weighted vector
        that still returns results, so the mistake is invisible.
        """

    async def close(self) -> None:
        """Release any resources the provider holds. Default is a no-op.

        Not abstract because local providers hold only an ONNX session, which the
        interpreter reclaims. Network-backed providers own an HTTP connection pool
        and must override this — see the lifecycle note on the Phase 3 factory.
        """
        return None

    def embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        """Blocking dense embedding, for use inside multiprocessing workers.

        Not abstract: local providers override it with the real implementation and
        have `embed_dense` delegate here via `asyncio.to_thread`. Network-backed
        providers leave it unimplemented, because running an HTTP client inside a
        forked worker process is a bad idea regardless.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support synchronous embedding; "
            "use the async interface."
        )


class BaseVectorStore(ABC):
    """Vector database operations. Async, because every call is network I/O."""

    @abstractmethod
    async def initialize(self, vector_size: int) -> None:
        """Create the collection and payload indexes if absent. Must be idempotent —
        calling it twice on an existing collection must not destroy data."""

    @abstractmethod
    async def upsert_points(self, chunks: list[Chunk], dense: list[list[float]],
                            sparse: list[dict[str, list]]) -> int:
        """Write chunks with both vector types. Returns the number of points written."""

    @abstractmethod
    async def hybrid_search(self, dense_query: list[float], sparse_query: dict[str, list],
                            limit: int = 20,
                            filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        """Server-side fused dense + sparse search.

        Raises:
            CollectionNotFoundError: the collection does not exist.
            VectorStoreError: the query failed.
        """

    @abstractmethod
    async def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Retrieve specific points by ID, with no vector search.

        Needed because `chunk_id` is not a filterable field — the payload indexes
        cover `doc_id`, `chunk_level`, `year`, and `section_title`. Phase 4's
        parent substitution looks parents up by ID, and Phases 13 and 16 walk
        chunk references the same way.

        Returns only the chunks that exist. A missing ID is not an error: Phase 4
        treats an absent parent as "use the child", which is a correct degradation
        rather than a failure.
        """

    @abstractmethod
    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        """Remove every chunk of every listed document. Returns the count deleted.

        Called immediately before re-upserting. Deterministic chunk IDs make an
        *unchanged* re-ingest idempotent, but they cannot remove chunks that no
        longer exist — if a document's chunk count shrinks, or its section titles
        change, the old points would otherwise linger as stale ghosts. See the
        discussion in `utils.py` below.

        The batch form is the abstract one because that is what the caller
        actually needs: a 256-chunk batch spans dozens of documents, and one
        request per document would double the round trips of an entire ingestion
        run. A store that can only delete singly implements this as a loop.
        """

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Single-document convenience wrapper. Not abstract — it delegates."""
        return await self.delete_by_doc_ids([doc_id])

    @abstractmethod
    async def count(self) -> int:
        """Number of points currently indexed. Used by health checks."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections. Called from the FastAPI lifespan hook in Phase 8."""


class BaseReranker(ABC):
    """Reorders retrieved candidates by deeper relevance scoring."""

    @abstractmethod
    async def rerank(self, query: str, candidates: list[ScoredChunk],
                     top_k: int = 5) -> list[ScoredChunk]:
        """Re-score and truncate. Returns at most `top_k`, re-ranked from 0.

        Async despite being CPU-bound: implementations offload to
        `asyncio.to_thread` internally, so callers get a uniform await-able API.

        **`rerank_score` contract:** populate it when scoring actually happened;
        leave it `None` when reranking was skipped or failed and the input order
        was passed through. This is deliberately not "always populate" — writing
        the retrieval score into `rerank_score` on a fallback would make a skipped
        rerank indistinguishable from a completed one, and Phase 7 has to be able
        to tell those apart to measure whether reranking earns its latency.
        Callers therefore treat `rerank_score` as "reranked, if present" and sort
        by `ScoredChunk.effective_score`, which handles both cases.
        """


class BaseLLMProvider(ABC):
    """Text generation. Async — every call is network I/O."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       model: str | None = None) -> str:
        """Single completion.

        Raises:
            RateLimitError: provider throttled the request (retryable).
            ModelDecommissionedError: the model ID no longer exists.
            LLMProviderError: any other API failure.
        """

    @abstractmethod
    async def generate_json(self, prompt: str, schema: type,
                            system_prompt: str | None = None,
                            model: str | None = None) -> Any:
        """Completion constrained to JSON, validated against a Pydantic model.

        Returns an INSTANCE of `schema`, not a dict. Callers do
        `report.is_grounded`, never `report["is_grounded"]`.

        `model` overrides the provider's default, and it is not optional
        decoration: Phase 4 expands queries with `EXPANSION_MODEL` and Phase 5
        grades with `GRADER_MODEL` while generating with `GENERATION_MODEL`. One
        provider instance serves all of them, so per-call model selection is the
        only way that cost/quality split can exist. `generate` takes the same
        argument for the same reason.

        **Implementations MUST wrap every failure in an `LLMProviderError`
        subclass — including Pydantic's `ValidationError` when the model returns
        malformed JSON.** Phase 4's expander and Phase 5's grader, router, and
        rewriter all degrade gracefully on `RAGException`; a bare `ValidationError`
        escaping this method turns each of those documented fallbacks into an
        unhandled crash.
        """

    @abstractmethod
    def stream(self, prompt: str, system_prompt: str | None = None) -> AsyncIterator[str]:
        """Yield tokens as they arrive. Powers SSE streaming in Phase 8.

        Declared `def`, not `async def`, returning `AsyncIterator[str]`. This is the
        correct signature for an async generator: the function itself is not a
        coroutine — calling it returns the iterator immediately, and you consume it
        with `async for`. Implementations use `async def` with `yield`, which Python
        types as returning `AsyncIterator`. Writing `async def stream(...)` in the ABC
        would demand `await provider.stream(...)` before iterating, which is wrong.
        """

    async def close(self) -> None:
        """Release the connection pool. Default is a no-op.

        Same lifecycle pattern as `BaseEmbeddingProvider.close()`: every LLM
        provider is network-backed, so in practice all of them override this — but
        it is non-abstract so that a test double or a local provider need not.
        Callers close unconditionally and never branch on provider type.
        """
        return None


class BaseCache(ABC):
    """Response caching, exact-match or semantic."""

    @abstractmethod
    async def get(self, query: str) -> Any | None:
        """Return a cached answer, or None on miss."""

    @abstractmethod
    async def set(self, query: str, value: Any, ttl: int | None = None) -> None:
        """Store a response."""

    @abstractmethod
    async def clear(self) -> None:
        """Evict everything."""


class BaseGuardrail(ABC):
    """A safety check applied to input or output."""

    @abstractmethod
    def check(self, text: str) -> tuple[bool, str | None]:
        """Return `(passed, reason)`. `reason` is None when passed."""


class BaseMetric(ABC):
    """One evaluation metric. Phase 7."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def score(self, query: str, answer: str, contexts: list[str]) -> float:
        """Return a score in [0.0, 1.0]."""
        