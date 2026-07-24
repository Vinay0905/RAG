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