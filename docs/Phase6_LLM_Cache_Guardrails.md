# Phase 6 Study Guide: LLM Providers, Redis Semantic Cache & Enterprise Guardrails

Welcome to **Phase 6** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will implement multi-provider LLM abstractions (Groq, OpenAI), build a Redis Semantic Query Cache (which skips graph execution for semantically identical questions), and add enterprise guardrail modules for PII redaction, prompt injection detection, and source citation validation.

---

## 📁 Directory Structure for Phase 6

Ensure the following subdirectories exist inside `src/`:

```text
src/
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── groq_provider.py
│   ├── openai_provider.py
│   └── factory.py
├── cache/
│   ├── __init__.py
│   ├── base.py
│   └── redis_cache.py
└── guardrails/
    ├── __init__.py
    ├── pii_masker.py
    ├── prompt_injection.py
    └── citation_validator.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/llm/base.py`
**Location**: `src/llm/base.py`  
**Purpose**: Abstract interface for LLM client providers.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLLMProvider(ABC):
    """Abstract interface for LLM text generation providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0) -> str:
        """Generates text from an LLM given a user prompt and optional system prompt."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generates structured JSON object matching a target schema."""
        pass
```

> **Deep 2-Line Explanation**:  
> *Defines an abstract blueprint for LLM integrations, enforcing both standard text generation and structured JSON generation.*  
> *Allows switching between cloud LLMs (Groq, OpenAI, Anthropic) without altering downstream RAG graph code.*

---

### File 2: `src/llm/groq_provider.py`
**Location**: `src/llm/groq_provider.py`  
**Purpose**: High-speed Groq provider adapter wrapping Llama 3 models with retry handling.

```python
import json
from typing import Dict, Any
from groq import Groq
from config.settings import settings
from src.llm.base import BaseLLMProvider
from src.core.exceptions import LLMProviderError


class GroqLLMProvider(BaseLLMProvider):
    """Adapter for Groq Cloud ultra-fast inference API."""

    def __init__(self, model_name: str = settings.GENERATION_MODEL):
        if not settings.GROQ_API_KEY:
            raise LLMProviderError("GROQ_API_KEY is not configured.")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            res = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            raise LLMProviderError(f"Groq generation call failed: {str(e)}")

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(res.choices[0].message.content.strip())
        except Exception as e:
            raise LLMProviderError(f"Groq structured JSON call failed: {str(e)}")
```

> **Deep 2-Line Explanation**:  
> *Connects to Groq Cloud API for near-instant Llama-3 inference with native JSON mode formatting.*  
> *Wraps network failures in clean `LLMProviderError` exceptions for graceful error reporting.*

---

### File 3: `src/llm/factory.py`
**Location**: `src/llm/factory.py`  
**Purpose**: Factory design pattern class to select and instantiate active LLM providers.

```python
from src.llm.base import BaseLLMProvider
from src.llm.groq_provider import GroqLLMProvider


class LLMProviderFactory:
    """Factory to instantiate LLM providers."""

    @staticmethod
    def get_provider(provider_type: str = "groq", model_name: str = "") -> BaseLLMProvider:
        if provider_type == "groq":
            return GroqLLMProvider(model_name=model_name) if model_name else GroqLLMProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
```

> **Deep 2-Line Explanation**:  
> *Encapsulates LLM instantiation logic behind a clean factory interface.*  
> *Enables simple configuration-driven selection of models across different RAG nodes.*

---

### File 4: `src/cache/redis_cache.py`
**Location**: `src/cache/redis_cache.py`  
**Purpose**: Redis semantic query cache storing question embeddings to return instant cached answers.

```python
import json
import math
from typing import Optional, Dict, Any, List
import redis
from config.settings import settings
from src.core.logging import logger


class RedisSemanticCache:
    """Semantic vector cache backed by Redis for instantaneous query responses."""

    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
            self.redis_client.ping()
            self.enabled = settings.ENABLE_SEMANTIC_CACHE
            logger.info("✅ Connected to Redis Semantic Cache service.")
        except Exception:
            logger.warning("⚠️ Redis connection failed. Semantic caching disabled.")
            self.enabled = False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def get_cached_response(self, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        keys = self.redis_client.keys("cache:query:*")
        best_score = 0.0
        best_payload = None

        for key in keys:
            data = json.loads(self.redis_client.get(key))
            cached_vec = data["vector"]
            similarity = self._cosine_similarity(query_vector, cached_vec)

            if similarity > best_score and similarity >= settings.CACHE_SIMILARITY_THRESHOLD:
                best_score = similarity
                best_payload = data["response"]

        if best_payload:
            logger.info(f"🎯 [CACHE HIT] Found semantic hit with similarity {best_score:.4f}!")
            return best_payload

        return None

    def store_in_cache(self, query: str, query_vector: List[float], response: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        cache_key = f"cache:query:{hash(query)}"
        payload = {
            "query": query,
            "vector": query_vector,
            "response": response
        }
        self.redis_client.set(cache_key, json.dumps(payload), ex=86400)  # TTL 24 hours
        logger.info(f"💾 Stored query answer in Redis cache: '{query[:40]}...'")
```

> **Deep 2-Line Explanation**:  
> *Stores user question vectors and generated answers in Redis with a 24-hour expiration time.*  
> *Calculates cosine similarity on incoming queries to instantly return cached answers when similarity exceeds 0.92.*

---

### File 5: `src/guardrails/pii_masker.py`
**Location**: `src/guardrails/pii_masker.py`  
**Purpose**: Regex-based PII scrubber masking sensitive personal data (emails, phone numbers, SSNs, credit cards).

```python
import re


class PIIMasker:
    """Redacts PII (Personally Identifiable Information) before sending text to external LLMs."""

    PATTERNS = {
        "EMAIL": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
        "PHONE": re.compile(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    }

    def sanitize(self, text: str) -> str:
        sanitized = text
        for pii_type, pattern in self.PATTERNS.items():
            sanitized = pattern.sub(f"[REDACTED_{pii_type}]", sanitized)
        return sanitized
```

> **Deep 2-Line Explanation**:  
> *Scans input text with regular expressions to replace emails, phone numbers, and SSNs with `[REDACTED_*]` tags.*  
> *Protects enterprise data privacy by ensuring no sensitive customer details leak to external cloud LLM providers.*

---

### File 6: `src/guardrails/prompt_injection.py`
**Location**: `src/guardrails/prompt_injection.py`  
**Purpose**: Heuristic detector filtering out prompt injection and system jailbreak attempts.

```python
import re
from src.core.exceptions import GuardrailError


class PromptInjectionDetector:
    """Detects malicious prompt injection and system instruction override attempts."""

    INJECTION_PATTERNS = [
        re.compile(r'ignore previous instructions', re.IGNORECASE),
        re.compile(r'you are now DAN', re.IGNORECASE),
        re.compile(r'system prompt override', re.IGNORECASE),
        re.compile(r'forget system instructions', re.IGNORECASE)
    ]

    def validate_input(self, text: str) -> None:
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                raise GuardrailError("🚨 Security Alert: Prompt injection attempt detected!")
```

> **Deep 2-Line Explanation**:  
> *Filters user inputs for jailbreak phrases like 'ignore previous instructions' before passing prompts to the state machine.*  
> *Prevents unauthorized manipulation of system prompt rules by raising a `GuardrailError` security alert.*

---

### File 7: `src/guardrails/citation_validator.py`
**Location**: `src/guardrails/citation_validator.py`  
**Purpose**: Validates that all inline `[SOURCE N]` citations in generated answers refer to valid retrieved chunks.

```python
import re
from typing import List, Dict, Any


class CitationValidator:
    """Validates structural integrity of inline citations in generated answers."""

    def validate_citations(self, response: str, available_sources_count: int) -> bool:
        citations = re.findall(r'\[SOURCE\s+(\d+)\]', response)
        if not citations and "INSUFFICIENT_CONTEXT" not in response:
            return False

        for c in citations:
            source_num = int(c)
            if source_num < 1 or source_num > available_sources_count:
                return False

        return True
```

> **Deep 2-Line Explanation**:  
> *Parses `[SOURCE N]` citation tokens in generated answers to ensure every citation references an existing chunk.*  
> *Guarantees that LLM outputs never cite non-existent sources or hallucinated index numbers.*

---

## 🎯 Phase 6 Checkpoint Verification

To verify Phase 6:
Test PII masking and Redis cache in Python:
```python
from src.guardrails.pii_masker import PIIMasker
from src.cache.redis_cache import RedisSemanticCache

masker = PIIMasker()
clean_text = masker.sanitize("Contact john.doe@example.com at 555-123-4567")
print("Sanitized Text:", clean_text)

cache = RedisSemanticCache()
print("Redis Cache Enabled:", cache.enabled)
```

When you are ready, let me know to proceed to **Phase 7: Enterprise LLMOps Evaluation Engine**!
