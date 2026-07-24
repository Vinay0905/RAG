from abc import ABC , abstractmethod
from typing import Any,Dict,List

class BassDocumentLoader(ABC):
    """Interface for loading raw documents from disk"""
    @abstractmethod
    def load(self,file_path:str)->Dict[str,Any]:
        """Loads a document and returns a raw dictionaru representation."""
        pass
class BaseVectorStore(ABC):
    """Interface for vector database operations (Qdrant, Chroma, PGVector)."""

    @abstractmethod
    #1
    async def initialize(self) -> None:
        """Prepares database collection and indexes."""
        pass

    @abstractmethod
    #2
    async def upsert_points(self, points: List[Dict[str, Any]]) -> bool:
        """Upserts dense-sparse payload points into the collection."""
        pass

    @abstractmethod
    #3
    async def search_dense(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Executes a dense vector similarity search."""
        pass



class BaseEmbeddingProvider(ABC):
    """Interface for embedding generation providers."""

    @abstractmethod
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of text strings."""
        pass

class BaseReranker(ABC):
    """Interface for candidate text rerankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Reranks candidate chunks based on semantic similarity to query."""
        pass


class BaseLLMProvider(ABC):
    """Interface for LLM interaction adapters."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0) -> str:
        """Generates text from an LLM provider given a prompt."""
        pass