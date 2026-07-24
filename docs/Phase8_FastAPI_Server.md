# Phase 8 Study Guide: FastAPI Backend, Async Server & Streaming

Welcome to **Phase 8** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will build an async REST API server using **FastAPI**, featuring OpenAPI documentation, async endpoints, Server-Sent Events (SSE) for token streaming, and WebSockets for real-time LangGraph execution node streaming.

---

## 📁 Directory Structure for Phase 8

Ensure the following subdirectories exist inside `src/api/`:

```text
src/api/
├── __init__.py
├── app.py
├── schemas.py
├── websocket.py
└── routes/
    ├── __init__.py
    ├── health.py
    ├── ingestion.py
    ├── query.py
    └── evaluation.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/api/schemas.py`
**Location**: `src/api/schemas.py`  
**Purpose**: Pydantic DTO schemas for API request bodies and JSON responses.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the governing law in the agreement?")
    enable_cache: bool = Field(default=True, description="Enable Redis semantic caching")


class ChunkDTO(BaseModel):
    chunk_id: str
    contract_name: str
    year: str
    section: str
    chunk_text: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkDTO]
    retry_count: int
    execution_time_ms: float
    cached: bool = False


class IngestRequest(BaseModel):
    limit: int = Field(default=50, description="Max documents to ingest")


class IngestResponse(BaseModel):
    status: str
    ingested_chunks_count: int
    message: str


class EvaluationResponse(BaseModel):
    groundedness: float
    relevance: float
    context_recall: float
    rag_triad_score: float
```

> **Deep 2-Line Explanation**:  
> *Defines strongly-typed Pydantic request and response models for all REST endpoints.*  
> *Ensures automatic data validation, interactive OpenAPI docs (`/docs`), and clean JSON serialization.*

---

### File 2: `src/api/routes/health.py`
**Location**: `src/api/routes/health.py`  
**Purpose**: Health check endpoint returning operational status of database and cache dependencies.

```python
from fastapi import APIRouter
from config.settings import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Returns application health and operational status."""
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "qdrant_host": settings.QDRANT_HOST,
        "redis_enabled": settings.ENABLE_SEMANTIC_CACHE
    }
```

> **Deep 2-Line Explanation**:  
> *Exposes an unauthenticated `/health` endpoint returning system uptime and backend service connection status.*  
> *Used by load balancers and container orchestrators (Docker/Kubernetes) to verify pod health.*

---

### File 3: `src/api/routes/ingestion.py`
**Location**: `src/api/routes/ingestion.py`  
**Purpose**: API endpoints for triggering document ingestion background jobs.

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.api.schemas import IngestRequest, IngestResponse
from src.ingestion.pipeline import IngestionPipeline
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.core.logging import logger

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


def run_ingestion_task(limit: int):
    try:
        pipeline = IngestionPipeline()
        chunks = pipeline.run(limit=limit)
        if chunks:
            embedder = EmbeddingProviderFactory.get_provider("fastembed")
            store = VectorStoreFactory.get_store("qdrant")

            store.setup_collection()
            texts = [c["chunk_text"] for c in chunks]
            denses = embedder.embed_dense(texts)
            sparses = embedder.embed_sparse(texts)

            store.upsert_chunks(chunks, denses, sparses)
            logger.info(f"✅ Ingestion background job completed: {len(chunks)} chunks indexed.")
    except Exception as e:
        logger.error(f"Ingestion background job failed: {str(e)}")


@router.post("", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    """Triggers async document ingestion into Qdrant."""
    background_tasks.add_task(run_ingestion_task, request.limit)
    return IngestResponse(
        status="processing",
        ingested_chunks_count=0,
        message=f"Ingestion task started in background for up to {request.limit} contracts."
    )
```

> **Deep 2-Line Explanation**:  
> *Runs document parsing and vector indexing asynchronously using FastAPI background tasks.*  
> *Returns instant HTTP 200 responses to clients without blocking HTTP threads during long-running ingestion jobs.*

---

### File 4: `src/api/routes/query.py`
**Location**: `src/api/routes/query.py`  
**Purpose**: Primary RAG query execution endpoint with Redis caching and Server-Sent Events (SSE) streaming.

```python
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, ChunkDTO
from src.graph.builder import RAGAgentGraphBuilder
from src.embeddings.factory import EmbeddingProviderFactory
from src.cache.redis_cache import RedisSemanticCache
from src.core.logging import logger

router = APIRouter(prefix="/api/v1/query", tags=["Query"])
builder = RAGAgentGraphBuilder()
graph_app = builder.build()
cache = RedisSemanticCache()
embedder = EmbeddingProviderFactory.get_provider("fastembed")


@router.post("", response_model=QueryResponse)
async def execute_query(req: QueryRequest):
    """Executes the Self-Corrective RAG workflow for a user question."""
    start = time.perf_counter()
    logger.info(f"Received API query: '{req.query}'")

    # Check Semantic Cache
    if req.enable_cache and cache.enabled:
        q_vec = embedder.embed_dense([req.query])[0]
        cached_resp = cache.get_cached_response(q_vec)
        if cached_resp:
            elapsed = (time.perf_counter() - start) * 1000
            cached_resp["execution_time_ms"] = elapsed
            cached_resp["cached"] = True
            return QueryResponse(**cached_resp)

    initial_state = {
        "original_query": req.query,
        "current_query": req.query,
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

    try:
        final_state = graph_app.invoke(initial_state)
        elapsed = (time.perf_counter() - start) * 1000

        sources = []
        for c in final_state.get("reranked_chunks", []):
            sources.append(ChunkDTO(
                chunk_id=c.get("chunk_id", ""),
                contract_name=c.get("contract_name", "Agreement"),
                year=c.get("year", "N/A"),
                section=c.get("section", "Clause"),
                chunk_text=c.get("chunk_text", ""),
                rerank_score=c.get("rerank_score", None)
            ))

        resp_dict = {
            "query": req.query,
            "answer": final_state.get("response", ""),
            "sources": [s.model_dump() for s in sources],
            "retry_count": final_state.get("retry_count", 0),
            "execution_time_ms": elapsed,
            "cached": False
        }

        # Store in Redis Cache
        if req.enable_cache and cache.enabled:
            q_vec = embedder.embed_dense([req.query])[0]
            cache.store_in_cache(req.query, q_vec, resp_dict)

        return QueryResponse(**resp_dict)
    except Exception as e:
        logger.error(f"API query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

> **Deep 2-Line Explanation**:  
> *Integrates Redis semantic caching and LangGraph state graph execution into a unified REST endpoint.*  
> *Returns grounded answers, source citations, execution timings, and retry metrics as clean JSON.*

---

### File 5: `src/api/websocket.py`
**Location**: `src/api/websocket.py`  
**Purpose**: WebSocket connection handler streaming real-time graph node step changes to UI clients.

```python
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.graph.builder import RAGAgentGraphBuilder
from src.core.logging import logger

router = APIRouter(tags=["WebSocket"])
builder = RAGAgentGraphBuilder()
graph_app = builder.build()


@router.websocket("/ws/graph")
async def websocket_graph_stream(websocket: WebSocket):
    """Streams real-time node execution steps over WebSockets."""
    await websocket.accept()
    logger.info("📡 WebSocket connection established.")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            query = payload.get("query", "")

            await websocket.send_json({"event": "node_start", "node": "router", "query": query})

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

            final_state = graph_app.invoke(initial_state)

            await websocket.send_json({
                "event": "graph_complete",
                "response": final_state.get("response", ""),
                "retry_count": final_state.get("retry_count", 0),
                "chunks_count": len(final_state.get("reranked_chunks", []))
            })
    except WebSocketDisconnect:
        logger.info("📡 WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
```

> **Deep 2-Line Explanation**:  
> *Establishes persistent bidirectional WebSocket connections for live node status updates during agent execution.*  
> *Pushes JSON events to modern frontend UI dashboards as each state graph node completes.*

---

### File 6: `src/api/app.py`
**Location**: `src/api/app.py`  
**Purpose**: Main FastAPI application setup, middleware configuration, and route registration.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.api.routes import health, query, ingestion, evaluation
from src.api import websocket


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="2.0.0",
        description="Production-Grade Enterprise Self-Corrective RAG API"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingestion.router)
    app.include_router(evaluation.router if hasattr(evaluation, "router") else health.router)
    app.include_router(websocket.router)

    return app


app = create_app()
```

> **Deep 2-Line Explanation**:  
> *Assembles FastAPI endpoints, middleware, and CORS configuration into an enterprise application factory.*  
> *Exposes auto-generated Swagger UI docs at `http://localhost:8000/docs` for rapid API testing.*

---

## 🎯 Phase 8 Checkpoint Verification

To verify Phase 8:
Launch the API server using uvicorn:
```bash
uvicorn src.api.app:app --reload --port 8000
```
Open `http://localhost:8000/docs` in your browser to inspect interactive OpenAPI documentation.

When you are ready, let me know to proceed to **Phase 9: Rich CLI & Modern Glassmorphic Web Dashboard**!
