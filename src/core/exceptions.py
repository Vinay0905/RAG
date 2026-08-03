from typing import Any

class RAGException(Exception):
    """Root of the application's exception hierarchy.

    Catching `RAGException` catches every deliberate failure our own code raises,
    while letting genuine programming errors (TypeError, AttributeError) propagate.
    """
    status_code:int =500
    def __init__(
            self,
            message:str,
            details:dict[str,Any]|None=None,
            retryable: bool = False,

    )->None:
        super().__init__(message)
        self.message=message
        self.details=details or{}
        self.retryable=retryable

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API error envelopes and structured logs."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
        return f"{self.message} ({rendered})"
# ─── Ingestion, Pipeline 1 ─────────────────────────────────────────────────

class IngestionError(RAGException):
    """Base for any failure in the ingestion pipeline."""
    status_code = 500


class DocumentLoadError(IngestionError):
    """A file could not be read or decoded. Usually quarantined, not fatal."""


class ParsingError(IngestionError):
    """Section or metadata extraction failed on an otherwise readable document."""


class ChunkingError(IngestionError):
    """Text could not be segmented into chunks."""

class VectorStoreError(RAGException):
    """Vector database connection, indexing, or query failure."""
    status_code = 503


class CollectionNotFoundError(VectorStoreError):
    """The target collection does not exist. Ingestion has not run."""
    status_code = 404


class DimensionMismatchError(VectorStoreError):
    """Vector size does not match the collection schema. Never retryable —
    it means the embedding model changed without a re-index. See Phase 15."""
    status_code = 500


class EmbeddingError(RAGException):
    """Dense or sparse vector generation failed."""
    status_code = 500



class RetrievalError(RAGException):
    """Hybrid search, fusion, or reranking failure."""
    status_code = 503


class RerankerError(RetrievalError):
    """Cross-encoder scoring failed."""
class LLMProviderError(RAGException):
    """An LLM API call failed."""
    status_code = 502


class RateLimitError(LLMProviderError):
    """Provider rate limit hit. Always retryable with backoff."""
    status_code = 429

    def __init__(self, message: str, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class ModelDecommissionedError(LLMProviderError):
    """The configured model ID no longer exists. Requires a config change, not a retry."""
    status_code = 500


class GraphExecutionError(RAGException):
    """A LangGraph node or state transition failed."""
    status_code = 500


class MaxRetriesExceededError(GraphExecutionError):
    """The agent loop terminated without producing any answer at all.

    Deliberately NOT raised when the retry budget is exhausted with a *failing*
    grade — that case returns the best answer with `status=LOW_CONFIDENCE`, its
    `GradingReport` attached, and a caveat appended, because a partially-supported
    answer plus an honest warning serves the user better than an HTTP 500. Phase 5
    §2 argues this at length.

    So this means "there is nothing to return": generation never succeeded, or a
    routing bug hit the graph's recursion limit.
    """


class InvalidQueryError(RAGException):
    """The request itself is malformed — blank query, non-positive top_k.

    Separate from `RetrievalError` so a bad request cannot be reported as a
    vector-store outage. Never retryable: sending the same empty string again
    will fail identically.
    """
    status_code = 400


# ─── Guardrails, evaluation, security ──────────────────────────────────────

class GuardrailError(RAGException):
    """A safety check blocked the request or response."""
    status_code = 400


class PromptInjectionError(GuardrailError):
    """Input matched a prompt-injection signature."""


class CitationValidationError(GuardrailError):
    """The generated answer cited a source that was not in the retrieved context."""


class EvaluationError(RAGException):
    """Metric scoring or synthetic dataset generation failed."""
    status_code = 500


class AuthorizationError(RAGException):
    """The requesting principal lacks access to the requested resource. Phase 11."""
    status_code = 403