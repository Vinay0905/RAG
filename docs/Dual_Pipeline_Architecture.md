# 🏗️ Dual-Pipeline System Architecture: Ingestion & Retrieval Pipelines

This document outlines the **Dual-Pipeline Architecture** powering our Production-Grade Self-Corrective RAG System, following the standard enterprise RAG pattern (Data Ingestion Pipeline on the left, Retrieval & Generation Pipeline on the right).

---

## 📐 System Flow Diagram

```text
┌────────────────────────────────────────────────────────┐     ┌────────────────────────────────────────────────────────┐
│             PIPELINE 1: DATA INGESTION                 │     │         PIPELINE 2: RETRIEVAL & GENERATION             │
│                                                        │     │                                                        │
│   ┌──────────────────┐                                 │     │                 ┌──────────────────┐                   │
│   │   DATA INGEST    │ (PDF, TXT, DOCX, HTML, DB)        │     │                 │    USER QUERY    │                   │
│   └────────┬─────────┘                                 │     │                 └────────┬─────────┘                   │
│            │ Read Data -> Document                     │     │                          │ Query                       │
│            ▼                                           │     │                          ▼                             │
│   ┌──────────────────┐                                 │     │                 ┌──────────────────┐                   │
│   │   DATA PARSING   │ (Section Parsing & Chunking)    │     │                 │    RETRIEVER     │                   │
│   └────────┬─────────┘                                 │     │                 └────────┬─────────┘                   │
│            │ Text Chunks                               │     │                          │ Reads Context               │
│            ▼                                           │     │                          ▼                             │
│   ┌──────────────────┐                                 │     │   ┌────────────────────────────────────────────────┐   │
│   │    EMBEDDING     │ (Text -> Vectors: Dense+Sparse) │     │   │              VECTORSTORE (Qdrant)             │   │
│   └────────┬─────────┘                                 │     │   └──────────────────────┬─────────────────────────┘   │
│            │ Vector Points + Payloads                  │     │                          │ Context Chunks              │
│            └───────────────────────────┬───────────────┼────►│                          ▼                             │
│                                        │ Upsert        │     │                 ┌──────────────────┐                   │
│                                        ▼               │     │                 │   LLM GENERATOR  │                   │
│                       ┌────────────────────────────────┐     │                 └────────┬─────────┘                   │
│                       │      VECTORSTORE (Qdrant)      │     │                          │ Answer + Citations          │
│                       └────────────────────────────────┘     │                          ▼                             │
│                                                              │                 ┌──────────────────┐                   │
│                                                              │                 │   FINAL OUTPUT   │                   │
│                                                              │                 └──────────────────┘                   │
└────────────────────────────────────────────────────────┘     └────────────────────────────────────────────────────────┘
```

---

## 🔀 Breakdown of the Two Pipelines

### 1️⃣ PIPELINE 1: Data Ingestion Pipeline (`src/ingestion/pipeline.py`)
- **Stage A: Data Ingestion (`loaders/`)**: Reads raw contract documents from filesystem or database sources (`.txt`, `.pdf`, `.docx`, `.html`). Computes SHA-256 checksums for differential ingestion tracking.
- **Stage B: Data Parsing & Chunking (`parsers/`, `chunkers/`)**: Segments raw text into structured legal sections (`SECTION`, `ARTICLE`) and creates parent-child sentence-aligned token chunks.
- **Stage C: Text-to-Vector Embedding (`embeddings/`)**: Converts text chunks into dual dense embeddings (BAAI/bge-small-en-v1.5) and sparse BM25 term frequency vectors.
- **Stage D: VectorStore Storage (`vectorstores/`)**: Writes points, vectors, and metadata payload indexes directly into Qdrant vector database collections.

---

### 2️⃣ PIPELINE 2: Retrieval & Generation Pipeline (`src/retrieval/` & `src/graph/`)
- **Stage A: User Query Entry**: Receives user query input via REST API, WebSockets, or CLI interface.
- **Stage B: Multi-Query & HyDE Expansion (`retrieval/multi_query.py`, `hyde.py`)**: Expands input query into 3 semantic variations and hypothetical document clauses.
- **Stage C: Hybrid Retriever & VectorStore Lookup (`retrieval/pipeline.py`)**: Executes parallel dense-sparse vector queries on Qdrant, applies Reciprocal Rank Fusion (RRF), and context re-scoring via Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Stage D: LLM Generation & Self-Correction (`graph/nodes/`)**: Passes top context chunks to Groq Llama-3.1 to generate answers with inline source citations (`[SOURCE 1]`). Evaluates answer faithfulness and relevance via self-correction grader nodes.
- **Stage E: Final Output**: Delivers grounded response and source citation metadata back to user.

---

## 🛠️ Code Implementations for Dual Pipelines

### Pipeline 1 Execution Script: `src/ingestion/pipeline.py`
```python
from src.ingestion.loaders.txt_loader import TextDocumentLoader
from src.ingestion.chunkers.parent_child_chunker import ParentChildChunker
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.core.logging import logger

class DataIngestionPipeline:
    """PIPELINE 1: Data Ingestion -> Data Parsing -> Embedding -> VectorStore Storage."""
    def __init__(self):
        self.loader = TextDocumentLoader()
        self.chunker = ParentChildChunker()
        self.embedder = EmbeddingProviderFactory.get_provider("fastembed")
        self.store = VectorStoreFactory.get_store("qdrant")

    def run(self, file_paths: list):
        logger.info("⚡ [PIPELINE 1 START] Data Ingestion & Vector Storage...")
        self.store.setup_collection()
        all_chunks = []
        for path in file_paths:
            doc = self.loader.load(path)
            chunks = self.chunker.process_document(doc)
            all_chunks.extend(chunks)

        texts = [c["chunk_text"] for c in all_chunks]
        dense_vecs = self.embedder.embed_dense(texts)
        sparse_vecs = self.embedder.embed_sparse(texts)
        self.store.upsert_chunks(all_chunks, dense_vecs, sparse_vecs)
        logger.info(f"✅ [PIPELINE 1 COMPLETE] Indexed {len(all_chunks)} chunks in VectorStore.")
```

### Pipeline 2 Execution Script: `src/retrieval/pipeline.py`
```python
from src.retrieval.multi_query import MultiQueryExpander
from src.retrieval.rerankers.cross_encoder import CrossEncoderReranker
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.graph.builder import RAGAgentGraphBuilder
from src.core.logging import logger

class RetrievalAndGenerationPipeline:
    """PIPELINE 2: User Query -> Retriever/VectorStore -> LLM Generation -> Output."""
    def __init__(self):
        builder = RAGAgentGraphBuilder()
        self.graph_app = builder.build()

    def run(self, query: str) -> dict:
        logger.info(f"⚡ [PIPELINE 2 START] Retrieval & Generation for query: '{query}'")
        initial_state = {
            "original_query": query,
            "current_query": query,
            "expanded_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "response": "",
            "retry_count": 0,
            "max_retries": 2,
            "grading_report": {},
            "intent": "",
            "is_grounded": False,
            "is_relevant": False
        }
        final_state = self.graph_app.invoke(initial_state)
        logger.info("✅ [PIPELINE 2 COMPLETE] Answer generated.")
        return {
            "query": query,
            "response": final_state.get("response", ""),
            "sources": final_state.get("reranked_chunks", []),
            "retries": final_state.get("retry_count", 0)
        }
```

---

## 🎯 Verification Checklist for Dual Pipeline Architecture
1. **Pipeline 1 Independent Run**: Test running `python -m src.cli.main ingest --limit 10` (populates Qdrant VectorStore).
2. **Pipeline 2 Independent Run**: Test running `python -m src.cli.main query "What is the termination clause?"` (reads Qdrant VectorStore & generates LLM response).
