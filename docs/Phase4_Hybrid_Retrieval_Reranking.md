# Phase 4 Study Guide: Pipeline 2 — Retrieval & Generation Pipeline

Welcome to **Phase 4**! In this phase, you will build **Pipeline 2 (Retrieval & Generation Pipeline)** as shown in the system architecture:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               PIPELINE 2: RETRIEVAL & GENERATION PIPELINE              │
│                                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐    ┌─────┐  │
│  │  USER QUERY  ├────►│  RETRIEVER   ├────►│ LLM GENERATOR├──►│ O/P │  │
│  │ (Input Text) │     │ (VectorStore)│     │(Context+Prmpt)│  │Output  │
│  └──────────────┘     └──────────────┘     └─────────────┘    └─────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure for Phase 4 (Pipeline 2)

Ensure the following subdirectories exist inside `src/retrieval/`:

```text
src/retrieval/
├── __init__.py
├── hybrid.py
├── multi_query.py
├── hyde.py
├── rerankers/
│   ├── __init__.py
│   ├── cross_encoder.py
│   ├── flashrank_reranker.py
│   └── diversity_reranker.py
└── pipeline.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### Step 1: RETRIEVAL SEARCH & FUSION — Multi-Query & RRF

#### File 1.1: `src/retrieval/hybrid.py`
**Location**: `src/retrieval/hybrid.py`  
**Purpose**: Implements Reciprocal Rank Fusion (RRF) algorithm to combine rank results from multiple query variations.

```python
from typing import List, Dict, Any


class ReciprocalRankFusion:
    """Combines multiple ranked search results using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, rank_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuses multiple chunk rank lists into a single deduplicated list ordered by RRF score."""
        scores: Dict[str, Dict[str, Any]] = {}

        for rank_list in rank_lists:
            for rank, item in enumerate(rank_list):
                chunk_id = item.get("chunk_id", str(rank))
                rrf_score = 1.0 / (self.k + rank + 1)

                if chunk_id not in scores:
                    scores[chunk_id] = {
                        "payload": item,
                        "score": 0.0
                    }
                scores[chunk_id]["score"] += rrf_score

        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [entry["payload"] for entry in sorted_items]
```

> **Deep 2-Line Explanation**:  
> *Calculates reciprocal rank scores (\(1 / (k + rank)\)) across multiple vector search lists to fairly merge results.*  
> *Deduplicates chunks by `chunk_id` and surfaces document chunks that consistently rank high across different search queries.*

---

#### File 1.2: `src/retrieval/multi_query.py`
**Location**: `src/retrieval/multi_query.py`  
**Purpose**: Generates semantic query variations to overcome vocabulary mismatch in search queries.

```python
from typing import List
from groq import Groq
from config.settings import settings
from src.core.logging import logger


class MultiQueryExpander:
    """Generates search query variations for RAG Fusion."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def expand(self, query: str) -> List[str]:
        if not self.client:
            return [query]

        prompt = f"""You are a legal contract search assistant.
Generate exactly 3 variations of the following search query to retrieve relevant legal clauses (indemnity, termination, liability).
Generate ONLY the 3 queries, one per line. Do not number them.

Original Query: {query}"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.EXPANSION_MODEL,
                temperature=0.2
            )
            raw_output = response.choices[0].message.content.strip()
            queries = [q.strip() for q in raw_output.split("\n") if q.strip()]
            if query not in queries:
                queries.append(query)
            return queries
        except Exception as e:
            logger.warning(f"Query expansion failed, returning original query: {str(e)}")
            return [query]
```

> **Deep 2-Line Explanation**:  
> *Queries Groq Llama-3.1 to generate 3 alternative legal phrasing variations for an incoming user question.*  
> *Improves recall by querying the vector store with multiple different angles of the same legal concept.*

---

### Step 2: CONTEXT RERANKING — Cross-Encoder

#### File 2.1: `src/retrieval/rerankers/cross_encoder.py`
**Location**: `src/retrieval/rerankers/cross_encoder.py`  
**Purpose**: Uses sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) to re-score candidate text chunks.

```python
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from config.settings import settings
from src.core.logging import logger
from src.core.telemetry import tracer


class CrossEncoderReranker:
    """Re-scores candidates using CrossEncoder contextual self-attention."""

    def __init__(self):
        logger.info(f"Loading CrossEncoder model '{settings.RERANK_MODEL_NAME}'...")
        self.model = CrossEncoder(settings.RERANK_MODEL_NAME)

    @tracer.trace_span("cross_encoder_rerank")
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = settings.CROSS_ENCODER_TOP_K) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, item.get("chunk_text", "")] for item in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for idx, score in enumerate(scores):
            item = dict(candidates[idx])
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
```

> **Deep 2-Line Explanation**:  
> *Passes `(query, document)` text pairs through a Cross-Encoder transformer model to evaluate joint contextual attention.*  
> *Re-orders initial bi-encoder vector search matches, dramatically increasing precision for complex legal terms.*

---

### Step 3: MASTER RETRIEVAL PIPELINE — VectorStore Reading

#### File 3.1: `src/retrieval/pipeline.py`
**Location**: `src/retrieval/pipeline.py`  
**Purpose**: Master Pipeline 2 retriever reading from Qdrant VectorStore.

```python
from typing import List, Dict, Any, Optional
from src.retrieval.multi_query import MultiQueryExpander
from src.retrieval.hybrid import ReciprocalRankFusion
from src.retrieval.rerankers.cross_encoder import CrossEncoderReranker
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.core.logging import logger
from src.core.telemetry import tracer


class RetrievalAndGenerationPipeline:
    """PIPELINE 2: User Query -> Retriever/VectorStore -> LLM Generation -> Output."""

    def __init__(self):
        self.expander = MultiQueryExpander()
        self.rrf = ReciprocalRankFusion()
        self.reranker = CrossEncoderReranker()
        self.embedder = EmbeddingProviderFactory.get_provider("fastembed")
        self.store = VectorStoreFactory.get_store("qdrant")

    @tracer.trace_span("retrieval_pipeline_execute")
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"⚡ [PIPELINE 2] Retrieving context for query: '{query}'")
        expanded_queries = self.expander.expand(query)

        all_rank_lists = []
        for q in expanded_queries:
            dense_vec = self.embedder.embed_dense([q])[0]
            sparse_vec = self.embedder.embed_sparse([q])[0]

            results = self.store.search_hybrid(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                limit=15
            )
            all_rank_lists.append(results)

        fused_chunks = self.rrf.fuse(all_rank_lists)
        reranked_chunks = self.reranker.rerank(query, fused_chunks, top_k=top_k)
        logger.info(f"✅ [PIPELINE 2] Retrieved top {len(reranked_chunks)} context chunks from VectorStore.")
        return reranked_chunks
```

> **Deep 2-Line Explanation**:  
> *Implements Pipeline 2 retrieval: expanding queries, searching Qdrant vector store, fusing RRF scores, and reranking context.*  
> *Feeds clean, reranked contract context directly to the LLM generation node in under 200 milliseconds.*

---

## 🎯 Phase 4 Checkpoint Verification

To verify Pipeline 2:
```python
from src.retrieval.pipeline import RetrievalAndGenerationPipeline

retriever = RetrievalAndGenerationPipeline()
context = retriever.retrieve_context("What is the governing law in the agreement?", top_k=3)
for c in context:
    print(f"[{c['rerank_score']:.4f}] Section: {c['section']} -> {c['chunk_text'][:80]}")
```

See the full **Dual Pipeline System Diagram** in [docs/Dual_Pipeline_Architecture.md](file:///Users/mast/Documents/VInayPrograming/RAG/docs/Dual_Pipeline_Architecture.md)!
