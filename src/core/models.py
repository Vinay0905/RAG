from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp.

    Note `datetime.now(timezone.utc)`, not the deprecated `datetime.utcnow()` —
    the latter returns a naive datetime, which compares incorrectly against aware ones.
    """
    return datetime.now(timezone.utc)


class DocumentType(str,Enum):
    """Source file format.

    Inheriting from `str` as well as `Enum` means the member serialises directly
    to JSON as "txt" rather than needing a custom encoder.
    """
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


class ChunkLevel(str, Enum):
    """Granularity of a chunk in the parent-child hierarchy (Phase 2).

    CHILD chunks are small and precise — they are what we embed and search against,
    because a focused passage produces a sharper vector. PARENT chunks are the larger
    surrounding context we actually feed the LLM, because a 400-token fragment is
    often too little to answer from. STANDALONE means no hierarchy was applied.

    Retrieval filters to CHILD, then swaps in the parents. Phase 16 adds SUMMARY
    nodes above PARENT to form the RAPTOR tree.
    """
    STANDALONE = "standalone"
    CHILD = "child"
    PARENT = "parent"
    SUMMARY = "summary"


class AnswerStatus(str, Enum):
    """How a run terminated. The machine-readable counterpart to `RAGAnswer.answer`.

    Without this, a caller has to parse English prose to distinguish "no contract
    matched" from "the search service is down" — the first is a 200 with an
    explanation, the second is a 503 that should be retried. Phase 8 maps these to
    HTTP status codes and Phase 9 renders them differently.
    """
    ANSWERED = "answered"              # generated and passed its audit
    UNVERIFIED = "unverified"          # generated, but the audit did not run
    LOW_CONFIDENCE = "low_confidence"  # generated, audited, and failed
    NO_MATCH = "no_match"              # retrieval found nothing to answer from
    UNSUPPORTED = "unsupported"        # a question top-k retrieval cannot answer


class RetrievalMethod(str, Enum):
    """Which retrieval strategy surfaced a given chunk. Used for debugging and evaluation."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"          # Phase 13
    SUMMARY_TREE = "summary" # Phase 16


# ─── Ingestion stage models ────────────────────────────────────────────────

class Document(BaseModel):
    """One source file, loaded but not yet parsed. Output of `src/ingestion/loaders/`."""

    # Shallow immutability: attribute assignment raises, but `metadata` is still a
    # mutable dict — `doc.metadata["k"] = v` succeeds. Pydantic cannot deep-freeze a
    # dict field. Treat `metadata` as append-only by convention; the guarantee here
    # is against *reassigning* fields, not against mutating a container inside one.
    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Deterministic UUIDv5 derived from the source path")
    source_path: str
    file_name: str
    content: str = Field(repr=False)         # excluded from repr — could be megabytes
    content_hash: str = Field(description="SHA-256 of content, for differential re-ingestion")
    doc_type: DocumentType = DocumentType.TXT

    contract_name: str = Field(default="Unknown Contract", max_length=300)
    year: int | None = Field(default=None, ge=1900, le=2100)

    loaded_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> int | None:
        """Directory names give us the string "2003"; we want the integer 2003.

        `mode="before"` runs prior to type coercion so we can handle junk directory
        names (e.g. "misc") by returning None instead of raising.
        """
        if v is None or v == "":
            return None
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return None

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)


class Section(BaseModel):
    """A structural division of a contract — `ARTICLE IV`, `SECTION 7.02`, `EXHIBIT B`.

    Output of `src/ingestion/parsers/`. Chunking happens *within* a section so that
    clause boundaries are never crossed.
    """

    section_id: str
    doc_id: str
    title: str = Field(default="Preamble", max_length=200)
    text: str = Field(repr=False)
    order: int = Field(ge=0, description="Zero-based position within the document")

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """The atomic unit of retrieval. One embedded, indexed passage.

    Carries denormalised document metadata so a vector search result is
    self-sufficient for citation without a second lookup.
    """

    chunk_id: str = Field(description="Deterministic UUIDv5 — stable across re-ingestion")
    doc_id: str
    section_id: str
    text: str

    chunk_index: int = Field(ge=0, description="Position within the parent section")
    token_count: int = Field(ge=0)

    # ── denormalised from Document and Section, for citation without a join ──
    contract_name: str = "Unknown Contract"
    file_name: str = ""
    section_title: str = "Preamble"
    year: int | None = None

    # ── parent-child chunking, Phase 2 ──
    chunk_level: ChunkLevel = Field(
        default=ChunkLevel.STANDALONE,
        description="Search filters to CHILD; generation substitutes the PARENT.",
    )
    parent_id: str | None = Field(
        default=None,
        description="The enclosing PARENT chunk's id. None for parents and standalone chunks.",
    )

    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        return v

    def to_payload(self) -> dict[str, Any]:
        """Flatten to a Qdrant payload dict.

        Qdrant stores plain JSON, so this is the one place we deliberately leave the
        typed world. Keeping the conversion here means no other module hand-builds
        payload dicts, and Phase 3's store adapter stays free of field knowledge.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "contract_name": self.contract_name,
            "file_name": self.file_name,
            "section_title": self.section_title,
            "year": self.year,
            "token_count": self.token_count,
            "chunk_level": self.chunk_level.value,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        """Rebuild a Chunk from a Qdrant payload. The inverse of `to_payload`."""
        return cls(
            chunk_id=payload["chunk_id"],
            doc_id=payload.get("doc_id", ""),
            section_id=payload.get("section_id", ""),
            text=payload["text"],
            chunk_index=payload.get("chunk_index", 0),
            token_count=payload.get("token_count", 0),
            contract_name=payload.get("contract_name", "Unknown Contract"),
            file_name=payload.get("file_name", ""),
            section_title=payload.get("section_title", "Preamble"),
            year=payload.get("year"),
            chunk_level=ChunkLevel(payload.get("chunk_level", "standalone")),
            parent_id=payload.get("parent_id"),
        )


# ─── Retrieval stage models ────────────────────────────────────────────────

class ScoredChunk(BaseModel):
    """A chunk plus its relevance score. Composition, not inheritance.

    A scored chunk is not a *kind of* chunk — it is a chunk *with* a score attached
    in a particular retrieval context. Subclassing `Chunk` here would let a scored
    result be passed anywhere a plain chunk is expected, hiding the fact that its
    score is only meaningful relative to one query.
    """

    chunk: Chunk
    score: float
    rank: int = Field(ge=0)
    method: RetrievalMethod = RetrievalMethod.HYBRID

    #: Populated only after cross-encoder reranking, so pre/post scores stay comparable.
    rerank_score: float | None = None

    @computed_field
    @property
    def effective_score(self) -> float:
        """The score to sort by — reranked if available, else the retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.score


class RetrievalResult(BaseModel):
    """Everything one retrieval round produced, including diagnostics.

    The query variations and timing are not decoration — Phase 7's evaluation
    harness reads them, and they are what you inspect when recall is poor.
    """

    original_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    chunks: list[ScoredChunk] = Field(default_factory=list)
    total_candidates: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    #: Search arms that failed. Phase 4 runs several probes concurrently and
    #: tolerates partial failure — but a result assembled from two arms instead of
    #: four is a DEGRADED result, and Phase 7 cannot attribute a poor answer without
    #: knowing that. Logging it is not enough: logs are not joined to evaluations.
    failed_arms: int = Field(default=0, ge=0)

    @computed_field
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ─── Generation and grading models ─────────────────────────────────────────

class Citation(BaseModel):
    """One source reference in a generated answer."""

    source_index: int = Field(ge=1, description="The N in [SOURCE N]")
    chunk_id: str
    contract_name: str
    section_title: str
    year: int | None = None


class GradingReport(BaseModel):
    """The self-correction verdict. Parsed from the grader LLM's JSON output.

    This model is the reason grading is reliable: the LLM is asked for JSON, and
    Pydantic rejects anything that does not conform, so a malformed grade fails
    loudly instead of being read as a silent pass.
    """

    is_grounded: bool = Field(description="Every claim traceable to retrieved context")
    is_relevant: bool = Field(description="The answer addresses the question asked")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=2000)
    unsupported_claims: list[str] = Field(default_factory=list)

    #: False when the audit did not actually run — the grader was disabled, errored,
    #: or returned malformed output. This exists because the alternative is worse in
    #: a specific and dangerous way: a fail-open fallback that sets
    #: is_grounded=is_relevant=True makes `passed` True, and every caller that
    #: checks `.passed` then treats an UNAUDITED answer as a verified one. The flag
    #: is excluded from the model the LLM fills — the grader never sets it, only
    #: our own code does.
    verified: bool = Field(default=True, exclude=True)

    @computed_field
    @property
    def passed(self) -> bool:
        """Audited, grounded, and relevant. Drives the retry edge in Phase 5.

        `verified` is part of the conjunction on purpose: "we could not check this"
        must never read the same as "we checked and it was fine". Phase 5
        distinguishes the two when choosing whether to retry — an unverified answer
        should not trigger a query rewrite, because rewriting the query does not
        fix a broken grader.
        """
        return self.verified and self.is_grounded and self.is_relevant


class RAGAnswer(BaseModel):
    """The final response object. Serialised directly by Phase 8's API."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[ScoredChunk] = Field(default_factory=list)

    grading: GradingReport | None = None
    retry_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=_utc_now)

    #: How this run ended, without parsing the prose. See `AnswerStatus`.
    status: AnswerStatus = AnswerStatus.ANSWERED
    #: Machine-readable detail behind a non-ANSWERED status, e.g.
    #: "aggregation_not_supported". None when the run succeeded normally.
    failure_reason: str | None = None
    #: The graph thread this run used. Returned so a caller can resume or inspect
    #: it later — an auto-generated ID the caller never sees is not resumable.
    thread_id: str | None = None
    #: Citation markers the model emitted that pointed at sources it was not given.
    #: Non-zero means the answer's provenance is partly fabricated, which is worth
    #: surfacing even though Phase 6 owns strict validation.
    invalid_citations: int = Field(default=0, ge=0)
    #: Non-fatal findings about this answer, in plain language — uncited figures, a
    #: missing citation, a degraded retrieval arm. Phase 8 renders these next to the
    #: answer and Phase 7 counts them. They live here rather than in logs because a
    #: warning nobody can join to the answer it describes is not a warning.
    warnings: list[str] = Field(default_factory=list)