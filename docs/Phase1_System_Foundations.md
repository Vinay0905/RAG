# Phase 1 Study Guide: System Foundations & Enterprise Configuration

Welcome to **Phase 1** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will lay the structural foundations of the application: configuring dependencies, environment variables, centralized settings using Pydantic V2, custom exception hierarchies, abstract interface contracts, structured logging, and telemetry tracing.

---

## 📁 Directory Structure to Create

Before writing any code, set up the following directory layout in your project root:

```text
RAG/
├── config/
│   ├── __init__.py
│   └── settings.py
├── docs/
├── src/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── exceptions.py
│       ├── interfaces.py
│       ├── logging.py
│       └── telemetry.py
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `pyproject.toml`
**Location**: `pyproject.toml` (Project Root)  
**Purpose**: Defines project metadata, Python runtime requirements, and third-party library dependencies.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "cuad-production-rag"
version = "2.0.0"
description = "Enterprise Production-Grade Self-Corrective RAG for Legal Contracts"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "qdrant-client>=1.7.0",
    "fastembed>=0.2.0",
    "sentence-transformers>=2.3.0",
    "langgraph>=0.0.25",
    "groq>=0.4.0",
    "openai>=1.12.0",
    "redis>=5.0.0",
    "tiktoken>=0.6.0",
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",
    "typer>=0.9.0",
    "rich>=13.7.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "httpx>=0.26.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0"
]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --import-mode=importlib"
testpaths = ["tests"]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
```

> **Deep 2-Line Explanation**:  
> *This configuration standardizes our Python build environment, declaring every production dependency needed for FastEmbed, Qdrant, LangGraph, and FastAPI.*  
> *It also configures testing (pytest) and code formatting (black/ruff) so our code adheres to strict enterprise quality standards.*

---

### File 2: `docker-compose.yml`
**Location**: `docker-compose.yml` (Project Root)  
**Purpose**: Orchestrates local containerized services (Qdrant vector database and Redis cache).

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: cuad_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: always

  redis:
    image: redis:7-alpine
    container_name: cuad_redis
    ports:
      - "6379:6379"
    restart: always

volumes:
  qdrant_storage:
```

> **Deep 2-Line Explanation**:  
> *Spawns local instances of Qdrant (for high-speed hybrid dense/sparse vector search) and Redis (for semantic query caching) with persistent volumes.*  
> *This ensures your RAG pipeline runs entirely isolated in Docker containers without polluting host machine ports.*

---

### File 3: `.env.example`
**Location**: `.env.example` (Project Root)  
**Purpose**: Template file declaring required environment variables.

```env
# System Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data Paths
DATASET_PATH=/Users/mast/Documents/VInayPrograming/RAG/dataset

# Qdrant Vector DB
QDRANT_HOST=http://localhost:6333
QDRANT_COLLECTION=cuad_advanced_prod

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_SEMANTIC_CACHE=true

# API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Models
DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5
SPARSE_MODEL_NAME=Qdrant/bm25
RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
EXPANSION_MODEL=llama-3.1-8b-instant
GENERATION_MODEL=llama-3.1-70b-versatile
GRADER_MODEL=llama-3.1-8b-instant

# Hyperparameters
MAX_TOKENS_PER_CHUNK=400
SENTENCE_OVERLAP=2
MAX_RETRIES=2
CROSS_ENCODER_TOP_K=5
```

> **Deep 2-Line Explanation**:  
> *Serves as a blueprint for all system credentials, vector DB host URLs, embedding model choices, and RAG execution hyperparameters.*  
> *Developers copy this to `.env` to customize their environment securely without hardcoding secret keys into source code.*

---

### File 4: `config/settings.py`
**Location**: `config/settings.py`  
**Purpose**: Type-safe settings manager powered by Pydantic V2 `BaseSettings`.

```python
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "CUAD Production RAG Engine"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity level")

    # Data Paths
    DATASET_PATH: str = Field(
        default="/Users/mast/Documents/VInayPrograming/RAG/dataset",
        description="Path to CUAD legal dataset directory"
    )

    # Qdrant Database Settings
    QDRANT_HOST: str = Field(default="http://localhost:6333", description="Qdrant service URL")
    QDRANT_COLLECTION: str = Field(default="cuad_advanced_prod", description="Collection name")

    # Redis Cache Settings
    REDIS_HOST: str = Field(default="localhost", description="Redis hostname")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    ENABLE_SEMANTIC_CACHE: bool = Field(default=True, description="Enable vector semantic caching")
    CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92, description="Cosine similarity threshold for cache hits")

    # API Keys
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (optional fallback)")

    # Model Specifications
    DENSE_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5", description="FastEmbed dense model")
    SPARSE_MODEL_NAME: str = Field(default="Qdrant/bm25", description="FastEmbed sparse BM25 model")
    RERANK_MODEL_NAME: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder reranker model")

    # LLM Models
    EXPANSION_MODEL: str = Field(default="llama-3.1-8b-instant", description="Model for query expansion")
    GENERATION_MODEL: str = Field(default="llama-3.1-70b-versatile", description="Model for grounded answer generation")
    GRADER_MODEL: str = Field(default="llama-3.1-8b-instant", description="Model for hallucination/relevance grading")

    # RAG Hyperparameters
    MAX_TOKENS_PER_CHUNK: int = Field(default=400, description="Target token size for document chunks")
    SENTENCE_OVERLAP: int = Field(default=2, description="Sentence overlap count between adjacent chunks")
    MAX_RETRIES: int = Field(default=2, description="Maximum self-correction rewrite attempts in LangGraph")
    CROSS_ENCODER_TOP_K: int = Field(default=5, description="Top K candidate chunks retained post-reranking")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_setup(self) -> None:
        """Validates critical settings during startup."""
        if not self.GROQ_API_KEY:
            print("⚠️ WARNING: GROQ_API_KEY is not set in environment or .env file.")


settings = Settings()
```

> **Deep 2-Line Explanation**:  
> *Uses Pydantic V2 to automatically read `.env` variables, validate data types, and expose a global `settings` object across the app.*  
> *Includes custom validation (`validate_setup`) to immediately alert developers if critical API keys are missing on startup.*

---

### File 5: `src/core/exceptions.py`
**Location**: `src/core/exceptions.py`  
**Purpose**: Centralized custom exception hierarchy for explicit error handling across modules.

```python
class RAGException(Exception):
    """Base exception for all errors within the RAG application."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(RAGException):
    """Raised when document loading, section parsing, or chunking fails."""
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
```

> **Deep 2-Line Explanation**:  
> *Establishes a clean, structured exception inheritance hierarchy rooted at `RAGException` for fine-grained error catching.*  
> *Prevents generic uncaught crashes by allowing specific handlers (e.g. `VectorStoreError`, `GuardrailError`) to execute graceful fallbacks.*

---

### File 6: `src/core/interfaces.py`
**Location**: `src/core/interfaces.py`  
**Purpose**: Defines Abstract Base Classes (ABCs) to enforce standard API contracts across system components.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDocumentLoader(ABC):
    """Interface for loading raw documents from disk or stream."""

    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Loads a document and returns a raw dictionary representation."""
        pass


class BaseVectorStore(ABC):
    """Interface for vector database operations (Qdrant, Chroma, PGVector)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Prepares database collection and indexes."""
        pass

    @abstractmethod
    async def upsert_points(self, points: List[Dict[str, Any]]) -> bool:
        """Upserts dense-sparse payload points into the collection."""
        pass

    @abstractmethod
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
```

> **Deep 2-Line Explanation**:  
> *Defines abstract blueprints using Python's `ABC` class so that all Vector Stores, Embeddings, Rerankers, and LLMs share strict, predictable method signatures.*  
> *This decouples our higher-level RAG graph logic from concrete third-party SDKs, allowing seamless swapping of Qdrant with Chroma or Groq with OpenAI.*

---

### File 7: `src/core/logging.py`
**Location**: `src/core/logging.py`  
**Purpose**: Structured JSON logger using standard Python `logging` for enterprise log aggregation.

```python
import json
import logging
import sys
from datetime import datetime
from config.settings import settings


class StructuredJSONFormatter(logging.Formatter):
    """Formats log record outputs into structured JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line_no": record.lineno,
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data["extra"] = record.extra

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(name: str = "cuad_rag") -> logging.Logger:
    """Configures and returns a structured JSON logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()
```

> **Deep 2-Line Explanation**:  
> *Formats all operational output into structured JSON strings containing UTC timestamps, line numbers, log levels, and extra context.*  
> *Allows log collection agents (like Datadog, ELK stack, or CloudWatch) to easily parse and index RAG execution events.*

---

### File 8: `src/core/telemetry.py`
**Location**: `src/core/telemetry.py`  
**Purpose**: Provides OpenTelemetry tracing hooks to track execution latency across RAG pipeline nodes.

```python
import time
from functools import wraps
from typing import Callable, Any
from src.core.logging import logger


class TelemetryTracer:
    """Lightweight tracer for recording step latency and telemetry."""

    @staticmethod
    def trace_span(span_name: str) -> Callable:
        """Decorator to trace function execution duration and log metrics."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                logger.info(f"⚡ [SPAN START] {span_name}")
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"✅ [SPAN END] {span_name} completed in {elapsed_ms:.2f}ms")
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.error(f"❌ [SPAN FAILED] {span_name} failed after {elapsed_ms:.2f}ms: {str(e)}")
                    raise e
            return wrapper
        return decorator


tracer = TelemetryTracer()
```

> **Deep 2-Line Explanation**:  
> *Implements a Python decorator (`@tracer.trace_span`) that measures millisecond-precise execution speeds for any RAG graph node or database call.*  
> *Gives full visibility into pipeline bottlenecks, automatically logging performance metrics and exceptions.*

---

## 🎯 Phase 1 Checkpoint Verification

After creating these 8 files:
1. Copy `.env.example` to `.env`.
2. Start the Docker containers by running:
   ```bash
   docker-compose up -d
   ```
3. Test your configuration loading in Python:
   ```python
   from config.settings import settings
   from src.core.logging import logger

   logger.info(f"Loaded {settings.APP_NAME} in {settings.ENVIRONMENT} mode.")
   ```

When you are ready, let me know to proceed to **Phase 2: Ingestion & Document Processing Engine**!
