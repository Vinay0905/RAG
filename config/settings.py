import difflib
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _looks_like_typo_of(candidate: str, target: str, threshold: float = 0.75) -> bool:
    """Whether `candidate` is close enough to `target` to be a probable misspelling.

    Uses stdlib `difflib` rather than a Levenshtein dependency. "locolhost" scores
    ~0.89 against "localhost" and is caught; "qdrant", "127.0.0.1", and
    "host.docker.internal" score far below the threshold and pass through untouched.
    """
    if candidate == target:
        return False
    return difflib.SequenceMatcher(None, candidate, target).ratio() >= threshold


class Settings(BaseSettings):
    """Type-safe application configuration, loaded from environment and `.env`.

    Every field is validated on construction. Invalid configuration raises
    `ValidationError` at import time rather than failing deep inside a pipeline run.
    """
     # ─── System ────────────────────────────────────────────────────────────
    APP_NAME: str = "Legal Contract RAG Engine"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # ─── Data ──────────────────────────────────────────────────────────────
    DATASET_PATH: str = Field(default="./data")
    SAMPLE_MODE: bool = Field(
        default=True,
        description="Process only SAMPLE_LIMIT documents. Keep true until the pipeline is proven.",
    )
    SAMPLE_LIMIT: int = Field(default=50, ge=1)

    # ─── Qdrant ────────────────────────────────────────────────────────────
    QDRANT_HOST: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="legal_contracts_v1")
    QDRANT_TIMEOUT: int = Field(default=60, ge=1)

    # ─── Redis ─────────────────────────────────────────────────────────────
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    ENABLE_SEMANTIC_CACHE: bool = Field(default=True)
    CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92, ge=0.0, le=1.0)
    CACHE_TTL_SECONDS: int = Field(default=86_400, ge=0)

    # ─── API keys ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")

    # ─── Models ────────────────────────────────────────────────────────────
    DENSE_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5")
    DENSE_VECTOR_SIZE: int = Field(default=384, ge=1)
    SPARSE_MODEL_NAME: str = Field(default="Qdrant/bm25")
    # FastEmbed's ONNX registry name for the ms-marco MiniLM cross-encoder. NOT
    # "cross-encoder/ms-marco-MiniLM-L-6-v2" — same weights, but that is the
    # sentence-transformers name and FastEmbed rejects it. Phase 4 uses FastEmbed
    # to avoid pulling PyTorch in for a 90 MB CPU model.
    RERANK_MODEL_NAME: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")

    EXPANSION_MODEL: str = Field(default="openai/gpt-oss-20b")
    GENERATION_MODEL: str = Field(default="openai/gpt-oss-120b")
    GRADER_MODEL: str = Field(default="openai/gpt-oss-20b")

    # ─── RAG hyperparameters ───────────────────────────────────────────────
    MAX_TOKENS_PER_CHUNK: int = Field(default=400, ge=50, le=8192)
    SENTENCE_OVERLAP: int = Field(default=2, ge=0)
    RETRIEVAL_TOP_K: int = Field(default=20, ge=1)
    RERANK_TOP_K: int = Field(default=5, ge=1)
    MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    NUM_QUERY_VARIATIONS: int = Field(default=3, ge=1, le=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # tolerate unknown keys in .env instead of crashing
    )

    # ─── Validators ────────────────────────────────────────────────────────

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @field_validator("ENVIRONMENT")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {sorted(allowed)}, got {v!r}")
        return lower

    @field_validator("QDRANT_HOST")
    @classmethod
    def _validate_qdrant_host(cls, v: str) -> str:
        """Reject a missing scheme, and warn on a likely 'localhost' misspelling.

        The prototype shipped `http://locolhost:6333`, which surfaces only as an
        opaque connection timeout minutes later. A scheme check alone would not
        catch that, so we also compare the hostname against known-good values.
        """
        from urllib.parse import urlparse

        if not v.startswith(("http://", "https://")):
            raise ValueError(f"QDRANT_HOST must include a scheme, got {v!r}")

        host = (urlparse(v).hostname or "").lower()
        if not host:
            raise ValueError(f"QDRANT_HOST has no hostname, got {v!r}")

        # A hostname one or two edits away from "localhost" is almost certainly a
        # typo. Anything genuinely different (a container name, an IP, a remote
        # host) is left alone.
        if host != "localhost" and _looks_like_typo_of(host, "localhost"):
            raise ValueError(
                f"QDRANT_HOST hostname {host!r} looks like a misspelling of 'localhost'"
            )
        return v.rstrip("/")

    @field_validator("RERANK_TOP_K")
    @classmethod
    def _validate_rerank_window(cls, v: int) -> int:
        # A hard cap. The cross-check against RETRIEVAL_TOP_K lives in
        # validate_runtime(), because field validators cannot see sibling fields.
        if v > 100:
            raise ValueError(f"RERANK_TOP_K above 100 defeats the purpose of reranking, got {v}")
        return v

    # ─── Computed properties ───────────────────────────────────────────────

    @computed_field
    @property
    def contracts_dir(self) -> Path:
        """Absolute path to the year-partitioned contracts directory."""
        return Path(self.DATASET_PATH).expanduser().resolve() / "contracts"

    @computed_field
    @property
    def checkpoint_dir(self) -> Path:
        """Where Phase 2 writes resumable ingestion state."""
        return Path(self.DATASET_PATH).expanduser().resolve() / "checkpoints"

    @computed_field
    @property
    def dead_letter_dir(self) -> Path:
        """Where Phase 2 quarantines documents that failed to parse."""
        return Path(self.DATASET_PATH).expanduser().resolve() / "dead_letter"

    @computed_field
    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    # ─── Cross-field runtime validation ────────────────────────────────────

    def validate_runtime(self) -> list[str]:
        """Check conditions that are warnings, not fatal errors.

        Returns a list of human-readable warnings. Called explicitly at
        application startup rather than at import, so that importing this module
        stays cheap and side-effect-light.
        """
        warnings: list[str] = []

        if not self.GROQ_API_KEY or self.GROQ_API_KEY.startswith("gsk_replace"):
            warnings.append("GROQ_API_KEY is unset or still a placeholder — LLM calls will fail.")

        if self.RERANK_TOP_K > self.RETRIEVAL_TOP_K:
            warnings.append(
                f"RERANK_TOP_K ({self.RERANK_TOP_K}) exceeds RETRIEVAL_TOP_K "
                f"({self.RETRIEVAL_TOP_K}); reranking cannot return more than it receives."
            )

        if self.is_production and self.SAMPLE_MODE:
            warnings.append("SAMPLE_MODE is enabled in production — only a subset will be indexed.")

        if self.is_production and self.LOG_LEVEL == "DEBUG":
            warnings.append("LOG_LEVEL=DEBUG in production risks leaking document content to logs.")

        return warnings


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Phase 8 uses this as a FastAPI dependency so tests can override it."""
    return Settings()


settings = get_settings()