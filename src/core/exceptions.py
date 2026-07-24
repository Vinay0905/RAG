class RAGException(Exception):
    """Base Exception for all errors with the RAG application"""

    def __init__(self,message:str,details:dict|None=None):
        super().__init__(message)
        self.message=message
        self.details=details or {}

class IngestionError(RAGException):
    """
    Raised When document loading , seection parsing  or chunking fails
    """

    pass 
class VectorStoreError(RAGException):
    """Raised when connection, index creation, or query execution on vector database fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when dense or sparse vector generation fails."""
    pass


class RetrievalError(RAGException):
    """Raised when hybrid search, RRF fusion, or reranking fails."""
    pass


class GraphExecutionError(RAGException):
    """Raised when a node or state transition in the LangGraph state machine fails."""
    pass


class LLMProviderError(RAGException):
    """Raised when an API call to Groq or OpenAI fails or hits rate limits."""
    pass


class GuardrailError(RAGException):
    """Raised when prompt injection detection or PII masking rules trigger a failure."""
    pass


class EvaluationError(RAGException):
    """Raised when automated RAG triad scoring or synthetic QA generation fails."""
    pass
