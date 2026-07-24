# Phase 3 Study Guide: Multi-Vector Database & Embedding Adapters

Welcome to **Phase 3** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will create a unified embedding provider layer (Dense BGE + Sparse BM25 via FastEmbed, with OpenAI fallback) and a multi-vector database adapter abstraction layer (Qdrant with hybrid dense/sparse vector search, metadata indexing, payload filtering, and ChromaDB adapter).

---

## 📁 Directory Structure for Phase 3

Ensure the following subdirectories exist inside `src/`:

```text
src/
├── embeddings/
│   ├── __init__.py
│   ├── base.py
│   ├── fastembed_provider.py
│   ├── openai_provider.py
│   └── factory.py
└── vectorstores/
    ├── __init__.py
    ├── base.py
    ├── qdrant_store.py
    ├── chroma_store.py
    └── factory.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/embeddings/base.py`
**Location**: `src/embeddings/base.py`  
**Purpose**: Abstract interface defining method signatures for dense and sparse vector generation.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEmbeddingProvider(ABC):
    """Abstract base class for dense and sparse text vectorizers."""

    @abstractmethod
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for input text strings."""
        pass

    @abstractmethod
    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generates sparse term-frequency (BM25/SPLADE) vectors."""
        pass
```

> **Deep 2-Line Explanation**:  
> *Defines an abstract blueprint for all embedding generators requiring both `embed_dense` and `embed_sparse` methods.*  
> *Allows the RAG system to seamlessly generate hybrid representations regardless of underlying model backend.*

---

### File 2: `src/embeddings/fastembed_provider.py`
**Location**: `src/embeddings/fastembed_provider.py`  
**Purpose**: High-speed CPU local vectorizer using ONNX runtime via `FastEmbed` for BGE-Small dense vectors and BM25 sparse vectors.

```python
from typing import List, Dict, Any
from fastembed import TextEmbedding, SparseTextEmbedding
from config.settings import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.core.logging import logger


class FastEmbedProvider(BaseEmbeddingProvider):
    """Local ONNX-optimized provider for BGE dense vectors and BM25 sparse vectors."""

    def __init__(self):
        logger.info(f"Loading FastEmbed dense model '{settings.DENSE_MODEL_NAME}'...")
        self.dense_model = TextEmbedding(settings.DENSE_MODEL_NAME)
        logger.info(f"Loading FastEmbed sparse model '{settings.SPARSE_MODEL_NAME}'...")
        self.sparse_model = SparseTextEmbedding(settings.SPARSE_MODEL_NAME)

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.dense_model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        sparse_embeddings = list(self.sparse_model.embed(texts))
        results = []
        for obj in sparse_embeddings:
            results.append({
                "indices": obj.indices.tolist(),
                "values": obj.values.tolist()
            })
        return results
```

> **Deep 2-Line Explanation**:  
> *Uses FastEmbed's quantized ONNX runtime to generate dense BGE and sparse BM25 vectors locally on CPU at blazing speed.*  
> *Returns standard Python lists and dictionary indices formatted specifically for vector database insertion.*

---

### File 3: `src/embeddings/openai_provider.py`
**Location**: `src/embeddings/openai_provider.py`  
**Purpose**: Cloud embedding provider using OpenAI API (`text-embedding-3-small`) as a high-accuracy fallback.

```python
from typing import List, Dict, Any
from openai import OpenAI
from config.settings import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.core.exceptions import EmbeddingError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Cloud provider for OpenAI dense embeddings."""

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding call failed: {str(e)}")

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        # Fallback empty sparse vectors when using pure dense OpenAI API
        return [{"indices": [], "values": []} for _ in texts]
```

> **Deep 2-Line Explanation**:  
> *Queries OpenAI's `text-embedding-3-small` API to produce high-dimensional dense embeddings for cloud setups.*  
> *Implements safety checks that raise clean `EmbeddingError` exceptions if credentials are missing or API rate limits trigger.*

---

### File 4: `src/embeddings/factory.py`
**Location**: `src/embeddings/factory.py`  
**Purpose**: Factory design pattern class to instantiate embedding providers dynamically based on configuration.

```python
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.fastembed_provider import FastEmbedProvider
from src.embeddings.openai_provider import OpenAIEmbeddingProvider


class EmbeddingProviderFactory:
    """Factory to instantiate configured embedding provider singleton."""

    @staticmethod
    def get_provider(provider_type: str = "fastembed") -> BaseEmbeddingProvider:
        if provider_type == "fastembed":
            return FastEmbedProvider()
        elif provider_type == "openai":
            return OpenAIEmbeddingProvider()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")
```

> **Deep 2-Line Explanation**:  
> *Applies the Factory pattern to return the active embedding provider based on a simple string configuration key.*  
> *Allows system components to request embedding models without hardcoding specific class dependencies.*

---

### File 5: `src/vectorstores/base.py`
**Location**: `src/vectorstores/base.py`  
**Purpose**: Abstract interface for vector databases defining collection setup, upsert, search, and filtering.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """Abstract base vector store interface."""

    @abstractmethod
    def setup_collection(self) -> None:
        """Initializes collection schema, vector params, and payload indexes."""
        pass

    @abstractmethod
    def upsert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[str, Any]]) -> bool:
        """Upserts document chunks along with dense and sparse vector indices."""
        pass

    @abstractmethod
    def search_hybrid(self, dense_vector: List[float], sparse_vector: Dict[str, Any], limit: int = 10, payload_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a hybrid dense-sparse vector similarity search with optional metadata filters."""
        pass
```

> **Deep 2-Line Explanation**:  
> *Defines standard abstract database contracts for collection setup, batch upserting, and hybrid vector searches.*  
> *Ensures vector stores can be swapped out (e.g. Qdrant to Chroma or PGVector) without refactoring retrieval nodes.*

---

### File 6: `src/vectorstores/qdrant_store.py`
**Location**: `src/vectorstores/qdrant_store.py`  
**Purpose**: Production Qdrant database adapter supporting dual vector config (Dense BGE + Sparse BM25) and metadata payload indexes.

```python
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, PayloadSchemaType, Filter, FieldCondition, MatchValue
)
from config.settings import settings
from src.vectorstores.base import BaseVectorStore
from src.core.logging import logger
from src.core.exceptions import VectorStoreError


class QdrantVectorStore(BaseVectorStore):
    """Adapter for Qdrant local vector database supporting hybrid dense-sparse collections."""

    def __init__(self):
        try:
            self.client = QdrantClient(url=settings.QDRANT_HOST)
            self.collection_name = settings.QDRANT_COLLECTION
        except Exception as e:
            raise VectorStoreError(f"Could not connect to Qdrant at {settings.QDRANT_HOST}: {str(e)}")

    def setup_collection(self) -> None:
        logger.info(f"Setting up Qdrant Collection '{self.collection_name}'...")
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense-bge": VectorParams(size=384, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse-bm25": SparseVectorParams(index=SparseIndexParams(on_disk=True))
                }
            )

            # Register metadata payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info("✅ Qdrant collection & payload indexes registered.")
        except Exception as e:
            raise VectorStoreError(f"Failed setting up Qdrant collection: {str(e)}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[str, Any]]) -> bool:
        points = []
        for idx, chunk in enumerate(chunks):
            s_obj = sparse_vectors[idx]
            q_sparse = SparseVector(
                indices=s_obj["indices"],
                values=s_obj["values"]
            )
            point = PointStruct(
                id=idx + 1 if "point_id" not in chunk else chunk["point_id"],
                vector={
                    "dense-bge": dense_vectors[idx],
                    "sparse-bm25": q_sparse
                },
                payload=chunk
            )
            points.append(point)

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            raise VectorStoreError(f"Qdrant batch upsert failed: {str(e)}")

    def search_hybrid(self, dense_vector: List[float], sparse_vector: Dict[str, Any], limit: int = 10, payload_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qdrant_filter = None
        if payload_filter and "year" in payload_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(key="year", match=MatchValue(value=payload_filter["year"]))]
            )

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense-bge", dense_vector),
                query_filter=qdrant_filter,
                limit=limit
            )
            return [hit.payload for hit in results if hit.payload]
        except Exception as e:
            raise VectorStoreError(f"Qdrant vector search failed: {str(e)}")
```

> **Deep 2-Line Explanation**:  
> *Creates Qdrant collections configured for dual vector search (384-dim BGE Cosine + BM25 Sparse Index) with payload index filters.*  
> *Executes similarity queries while applying fast keyword payload filters on contract metadata attributes like year or clause type.*

---

### File 7: `src/vectorstores/chroma_store.py`
**Location**: `src/vectorstores/chroma_store.py`  
**Purpose**: Secondary in-memory ChromaDB vector store adapter for local testing without Docker dependencies.

```python
from typing import List, Dict, Any, Optional
from src.vectorstores.base import BaseVectorStore
from src.core.logging import logger


class ChromaVectorStore(BaseVectorStore):
    """Adapter for ChromaDB vector database."""

    def __init__(self):
        logger.info("Initializing ChromaDB Store (Fallback)...")
        self.chunks: List[Dict[str, Any]] = []

    def setup_collection(self) -> None:
        self.chunks = []
        logger.info("ChromaDB in-memory store initialized.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[str, Any]]) -> bool:
        self.chunks.extend(chunks)
        return True

    def search_hybrid(self, dense_vector: List[float], sparse_vector: Dict[str, Any], limit: int = 10, payload_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.chunks[:limit]
```

> **Deep 2-Line Explanation**:  
> *Provides a secondary vector store fallback implementing the standard `BaseVectorStore` interface.*  
> *Ensures tests and fast prototyping can run without requiring a running Qdrant Docker container.*

---

### File 8: `src/vectorstores/factory.py`
**Location**: `src/vectorstores/factory.py`  
**Purpose**: Factory design pattern to select and instantiate vector store adapters.

```python
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.chroma_store import ChromaVectorStore


class VectorStoreFactory:
    """Factory to instantiate vector store adapters."""

    @staticmethod
    def get_store(store_type: str = "qdrant") -> BaseVectorStore:
        if store_type == "qdrant":
            return QdrantVectorStore()
        elif store_type == "chroma":
            return ChromaVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
```

> **Deep 2-Line Explanation**:  
> *Encapsulates vector store selection logic behind a single static factory method.*  
> *Allows switching between Qdrant and ChromaDB with a single configuration parameter change.*

---

## 🎯 Phase 3 Checkpoint Verification

To verify Phase 3:
1. Ensure your Qdrant container is running (`docker-compose up -d`).
2. Test vector embedding and Qdrant collection creation in Python:
   ```python
   from src.embeddings.factory import EmbeddingProviderFactory
   from src.vectorstores.factory import VectorStoreFactory

   embedder = EmbeddingProviderFactory.get_provider("fastembed")
   store = VectorStoreFactory.get_store("qdrant")

   store.setup_collection()
   dense = embedder.embed_dense(["This is a test contract clause."])
   sparse = embedder.embed_sparse(["This is a test contract clause."])

   print("Dense dim:", len(dense[0]))
   print("Sparse indices:", sparse[0]["indices"])
   ```

When you are ready, let me know to proceed to **Phase 4: Hybrid Search & Multi-Stage Reranking Engine**!
