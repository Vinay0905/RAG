# Phase 1 — System Foundations

> **Prerequisite:** read `00_MASTER_ROADMAP.md` first, especially §4 (directory structure) and §7
> (coding conventions). This phase implements those conventions; it will not make sense otherwise.
>
> **Budget:** ~1,100 lines of Python across 7 modules, plus ~180 lines of config
> (TOML/YAML/env). Originally estimated at 550 — see the calibration note in the roadmap's §5.
> **Depends on:** nothing. This is the bottom of the stack.
> **Everything depends on this.** Phases 2–16 all import from `src/core/`.

---

## What This Phase Is Really For

Phase 1 produces no features. You cannot query anything at the end of it. That makes it feel like
throwaway scaffolding, and it is the phase most people skim. Skimming it is the single most expensive
mistake you can make in this build.

Here is why. Every later phase communicates through the types and contracts defined here. When Phase
4 hands retrieved chunks to Phase 5's generation node, the thing being handed over is a
`list[ScoredChunk]` defined in this phase. When Phase 11 needs to enforce tenant permissions, it adds
a field to the `Chunk` model defined in this phase and every layer inherits it for free. When Phase
15 migrates embedding models without downtime, it relies on the deterministic chunk IDs defined in
this phase.

Get these ten files right and the next fifteen phases are mostly mechanical. Get them wrong and you
will be rewriting them in Phase 5, with four phases of dependent code to fix.

### The ten files

| # | File | Lines | What it is |
| :-- | :--- | ---: | :--- |
| 1 | `pyproject.toml` | 95 | Dependency and tooling declaration |
| 2 | `docker-compose.yml` | 45 | Qdrant + Redis, localhost-bound, pinned tags |
| 3 | `.env.example` | 45 | Config template (plus a `.gitignore` fix) |
| 4 | `config/settings.py` | 200 | Validated, type-safe settings |
| 5 | `src/core/exceptions.py` | 120 | Error hierarchy |
| 6 | **`src/core/models.py`** | **250** | **The domain models — the spine of the system** |
| 7 | `src/core/interfaces.py` | 180 | Abstract contracts |
| 8 | `src/core/logging.py` | 140 | Structured JSON logs + correlation IDs |
| 9 | `src/core/telemetry.py` | 125 | Sync + async span tracing |
| 10 | `src/core/utils.py` | 130 | Tokenizing, hashing, deterministic IDs |

Files 6 and 10 are new relative to the old draft, and they are the two that matter most. `models.py`
is where Pydantic stops being a config library and becomes the architecture. `utils.py` contains a
three-line function that fixes a real data-corruption bug in the current prototype.

### Directory to create

```text
RAG/
├── config/
│   ├── __init__.py            (empty file)
│   └── settings.py
├── src/
│   ├── __init__.py            (empty file)
│   └── core/
│       ├── __init__.py        (empty file)
│       ├── exceptions.py
│       ├── models.py
│       ├── interfaces.py
│       ├── logging.py
│       ├── telemetry.py
│       └── utils.py
├── data/                      (empty dirs: contracts/ checkpoints/ dead_letter/)
├── .env.example
├── docker-compose.yml
└── pyproject.toml
```

The `__init__.py` files must exist, even empty. Without them Python treats the directories as
namespace packages, and while that mostly works, it breaks `pytest` discovery in Phase 10 and
confuses editors about import roots. Create them and forget about them.

---

## File 1 — `pyproject.toml`

**Location:** project root

### The Problem

You will install about thirty packages over sixteen phases. Six months from now you will pull this
repo onto a different machine, `pip install` will resolve different versions than you have today, and
something will break in a way that takes an afternoon to diagnose. A dependency manifest is how you
make "it works on my machine" reproducible.

### Design Decision

**`pyproject.toml` over `requirements.txt`.** A `requirements.txt` is a flat list of packages with no
notion of grouping, no project metadata, and no home for tool configuration. `pyproject.toml` is the
current Python standard (PEP 621): it declares your project as an installable package, supports
*optional dependency groups*, and holds `pytest`/`black`/`ruff` config in the same file. That last
point matters more than it sounds — one file to configure instead of four dotfiles.

**Optional groups over one flat list.** Phases 13–16 need heavy dependencies: Neo4j drivers,
`pdfplumber`, `networkx`, OCR libraries. Someone building only Part I should not have to install
them. Declaring `[project.optional-dependencies]` means `pip install -e .` gives you the baseline and
`pip install -e ".[graph,pdf]"` adds Part II when you get there.

**Floor pins (`>=`) over exact pins (`==`).** For a library you would pin exactly. For an application
under active development, exact pins mean you fight the resolver every time you add a package. Floors
express "I need at least this API" without over-constraining. When the project stabilises, generate a
lockfile — that is the right tool for exact reproducibility, not the manifest.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "legal-rag-engine"
version = "2.0.0"
description = "Production-grade self-corrective RAG for commercial legal contracts"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    # Configuration & validation — the backbone of every data structure we define
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",

    # Vector database & embeddings
    "qdrant-client>=1.12.0",          # 1.10+ required for the query_points Universal API
    "fastembed>=0.4.0",               # ONNX-quantised embeddings, CPU-only, no torch needed

    # Reranking (this one does pull in torch — it is the heaviest install)
    "sentence-transformers>=3.0.0",

    # Agent orchestration
    "langgraph>=0.2.0",
    "langgraph-checkpoint-sqlite>=2.0.0",   # persistent graph state, Phase 5

    # LLM providers
    "groq>=0.11.0",
    "openai>=1.40.0",                 # fallback provider, Phase 6

    # Caching
    "redis>=5.0.0",

    # Utilities
    "tiktoken>=0.7.0",                # token counting — must match the LLM's tokenizer
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",

    # API layer, Phase 8
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "httpx>=0.27.0",
    "sse-starlette>=2.1.0",           # server-sent events for token streaming

    # CLI layer, Phase 9
    "typer>=0.12.0",
    "rich>=13.7.0",
]

[project.optional-dependencies]
# Phase 14 — layout-aware PDF parsing and OCR
pdf = [
    "pdfplumber>=0.11.0",
    "pypdf>=4.2.0",
    "python-docx>=1.1.0",
    "pytesseract>=0.3.10",
    "Pillow>=10.3.0",
]

# Phase 13 — GraphRAG
graph = [
    "neo4j>=5.20.0",
    "networkx>=3.3",
]

# Phase 10 — testing
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "black>=24.4.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[project.scripts]
legalrag = "src.cli.main:app"        # gives you a `legalrag` command after install, Phase 9

# Explicit package discovery. Required because this project has TWO top-level
# packages (`src` and `config`) rather than the single package setuptools expects.
# Without this, `pip install -e .` fails with "Multiple top-level packages
# discovered in a flat-layout" and refuses to guess.
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "config*"]

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
asyncio_mode = "auto"                 # lets you write `async def test_...` with no decorator
markers = [
    "integration: requires running Qdrant/Redis containers",
    "slow: takes more than five seconds",
]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "ASYNC"]
ignore = ["E501"]                     # black already enforces line length

[tool.mypy]
python_version = "3.10"
warn_return_any = true
disallow_untyped_defs = true
ignore_missing_imports = true
```

### Notes on specific choices

`asyncio_mode = "auto"` is a small quality-of-life decision with an outsized payoff. Our codebase is
async-first, so nearly every test is async. Without this you decorate every single test with
`@pytest.mark.asyncio`. With it, `pytest-asyncio` detects coroutine tests automatically.

The `ruff` rule selection is deliberate: `UP` (pyupgrade) will flag `List[str]` and tell you to write
`list[str]`, mechanically enforcing roadmap §7.3. `ASYNC` catches blocking calls inside `async def`,
which is exactly the class of bug described in §7.2 — it will find them for you.

Note `[tool.ruff.lint] select`, not `[tool.ruff] select`. Ruff moved lint configuration under a
`lint` subtable; the old form now emits a deprecation warning on every run.

`sentence-transformers` is the only genuinely heavy dependency, because it pulls PyTorch — roughly
2 GB. It is needed solely for the cross-encoder reranker in Phase 4. If disk is tight, you can defer
installing it until then.

The `[tool.setuptools.packages.find]` block deserves a word, because it exists to solve a problem you
would otherwise hit immediately. Setuptools recognises two project shapes: *flat layout*, where the
package sits at the repo root, and *src layout*, where `src/` is a container holding the real package
(`src/mypackage/`). This project is neither — `src` is itself the importable package, and `config` is
a second one beside it. Auto-discovery sees two top-level candidates, cannot decide, and aborts.
Listing them explicitly resolves the ambiguity. An alternative would be to rename `src/` to
`legalrag/` and follow convention properly, but the roadmap's directory map uses `src/` throughout,
so we declare instead of rename.

### Failure Modes

**`ERROR: Package 'legal-rag-engine' requires a different Python`** — you are on 3.9 or older. The
codebase uses `X | Y` unions and `list[str]` generics, which are 3.10+ syntax and will raise
`TypeError` at import time on 3.9. Upgrade Python; there is no workaround.

**Resolver hangs for several minutes on `pip install -e .`** — normal on first install. `pip`'s
backtracking resolver is exploring the `torch`/`transformers` version space. Let it finish.

**Import errors on `src.*` despite installing** — you missed the `-e` flag, or an `__init__.py` is
absent. `pip install -e .` installs in editable mode, which puts the project root on `sys.path`; a
plain `pip install .` copies a snapshot into `site-packages` and your edits stop taking effect.

---

## File 2 — `docker-compose.yml`

**Location:** project root

### The Problem

The system needs Qdrant (vector database) and Redis (semantic cache). Installing either natively
means platform-specific package managers, version drift between your machine and any other, and no
clean uninstall. Docker gives one command that produces identical services anywhere.

### Design Decision

**Named volumes over bind mounts.** `qdrant_storage:/qdrant/storage` stores data in a Docker-managed
volume rather than a host directory. On Windows especially, bind-mounting a host folder into a
container that does heavy random I/O — which a vector database does constantly — is slow, because
writes cross the WSL2 filesystem boundary. Named volumes live inside the Linux VM and are
dramatically faster.

**A real healthcheck where one is possible, and none where it is not.** A container reporting
"running" means the process started, not that it is ready to serve. Redis ships `redis-cli`, so
`redis-cli ping` is a genuine readiness probe. Qdrant's image is distroless — no shell, no `curl`, no
`wget` — so there is no way to issue an HTTP request from inside it. Rather than write a probe that
merely proves the binary is executable (which tells you nothing about whether the server accepts
requests, and would report healthy while Qdrant is unavailable), we omit it and do readiness checking
from the application side in Phase 8. **A healthcheck that cannot fail is worse than no healthcheck,
because it manufactures false confidence.**

**Pinned image tags, not `latest`.** `latest` means your teammate, your CI, and your laptop in three
months all run different Qdrant versions. Pin explicitly and bump deliberately.

**No `version:` key.** The Compose Spec deprecated it. Including it produces a warning on every
command.

```yaml
services:
  qdrant:
    # Pin this. Replace with the exact tag you first run successfully, and record it
    # here — check https://hub.docker.com/r/qdrant/qdrant/tags for available versions.
    # Must be >= v1.10 for the query_points Universal API used in Phases 3 and 4.
    image: qdrant/qdrant:v1.12.4
    container_name: legalrag_qdrant
    ports:
      # Bound to localhost only. Qdrant has no authentication by default, so
      # exposing 0.0.0.0 would publish an unauthenticated database to your network.
      - "127.0.0.1:6333:6333"     # REST API — this is what the Python client uses
      - "127.0.0.1:6334:6334"     # gRPC API — faster for bulk upserts, optional
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__MAX_REQUEST_SIZE_MB: 64      # default 32 MB; our upsert batches exceed it
      QDRANT__LOG_LEVEL: INFO
    # No healthcheck: the image is distroless and cannot make an HTTP request to
    # itself. Readiness is verified by the application in Phase 8's lifespan hook.
    restart: unless-stopped

  redis:
    image: redis:7.4-alpine
    container_name: legalrag_redis
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis_storage:/data
    # Cache eviction: when memory fills, drop least-recently-used keys instead of erroring.
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

volumes:
  qdrant_storage:
  redis_storage:
```

### Why those Redis flags

`--maxmemory-policy allkeys-lru` is the important one, and it is a genuine architectural decision.
Redis's default policy is `noeviction`: when memory fills, writes start returning errors. For a cache
that is precisely wrong — a full cache should quietly forget its oldest entries, not break the
application. `allkeys-lru` evicts the least recently used key on pressure, which is the correct
behaviour for the semantic cache in Phase 6.

`--appendonly yes` enables the append-only file so the cache survives a container restart. For a pure
cache this is arguably unnecessary; we keep it because a cold semantic cache means re-paying for
LLM calls you already made, and during development you restart containers often.

### Failure Modes

**`Bind for 0.0.0.0:6333 failed: port is already allocated`** — something already holds the port,
usually a Qdrant container from the earlier prototype. Find it with `docker ps -a` and remove it, or
remap to `"6335:6333"` and update `QDRANT_HOST` in `.env` to match.

**`manifest for qdrant/qdrant:v1.12.4 not found`** — that tag does not exist. Pick a real one from
the tag list, confirm it is v1.10 or newer, and write it into the file. Do not fall back to `latest`
to make the error go away; you are trading a five-second lookup for an unreproducible environment.

**Qdrant shows no health status in `docker compose ps`** — expected. We removed the healthcheck
deliberately, for the reason given above. Verify readiness from the host instead:
`curl http://localhost:6333/healthz`, or in PowerShell `Invoke-WebRequest http://localhost:6333/healthz`.

**A note on the security posture.** Both services bind to `127.0.0.1` and neither requires
authentication. That is acceptable *only* because this is a local development stack, which is what the
roadmap commits to. If you ever deploy this beyond your own machine, Qdrant needs an API key
(`QDRANT__SERVICE__API_KEY`), Redis needs `requirepass`, both need TLS, and neither should be
published to a public interface. This compose file is not a deployment artefact.

**Data survives a change you expected to reset** — named volumes persist across `docker compose down`.
To wipe them you need `docker compose down -v`. The `-v` is easy to forget and will make you think
your ingestion code is broken when it is actually working perfectly against stale data.

---

## File 3 — `.env.example`

**Location:** project root

### The Problem

Secrets must never enter git. A leaked API key in a public repo is scraped and abused within minutes.
But configuration also needs to be discoverable — a new developer must be able to tell what values
the system requires without reading every source file.

### Design Decision

The standard two-file split. `.env.example` is committed, contains every key with safe placeholder
values, and serves as documentation. `.env` is gitignored, holds real values, and never leaves the
machine. Anyone cloning the repo runs `cp .env.example .env` and fills in the blanks.

### ⚠️ Fix `.gitignore` before you create anything

Your `.gitignore` currently contains exactly two lines: `.env` and `dataset/`. The secret is covered,
but **the corpus is not** — this phase tells you to create `data/contracts/`, and nothing is ignoring
`data/`. Commit 37 GB of contracts once and it is in your git history permanently; `git rm` will not
remove it, and the repository becomes unusable.

Replace `.gitignore` with this before creating the `data/` directory:

```gitignore
# Secrets — never commit
.env
.env.*
!.env.example

# Corpus and derived artefacts — large, regenerable, must stay out of git
data/
dataset/

# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
build/
dist/

# Tooling caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# Editors and OS
.DS_Store
.idea/
*.swp
```

Two subtleties worth noticing. `.env.*` with `!.env.example` ignores every environment variant while
keeping the committed template — order matters, since the negation must follow the pattern it
overrides. And `data/` is ignored wholesale rather than `data/contracts/`, because Phase 2 also writes
checkpoints and dead-letter files there, and those are equally regenerable.

If you have already committed something large, stop and deal with it now rather than later —
rewriting history gets harder with every commit built on top.

```env
# ─── System ────────────────────────────────────────────────────────────────
ENVIRONMENT=development
LOG_LEVEL=INFO

# ─── Data ──────────────────────────────────────────────────────────────────
# EDGAR corpus root. Must contain contracts/ partitioned by year: contracts/2000/, 2001/, ...
# Windows: use forward slashes or escaped backslashes — C:/data/edgar  or  C:\\data\\edgar
DATASET_PATH=./data
SAMPLE_MODE=true
SAMPLE_LIMIT=50

# ─── Qdrant ────────────────────────────────────────────────────────────────
QDRANT_HOST=http://localhost:6333
QDRANT_COLLECTION=legal_contracts_v1
QDRANT_TIMEOUT=60

# ─── Redis ─────────────────────────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_SEMANTIC_CACHE=true
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_SECONDS=86400

# ─── API keys — REAL VALUES GO IN .env, NEVER HERE ─────────────────────────
GROQ_API_KEY=gsk_replace_me
OPENAI_API_KEY=sk_replace_me_optional

# ─── Embedding & reranking models ──────────────────────────────────────────
DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5
DENSE_VECTOR_SIZE=384
SPARSE_MODEL_NAME=Qdrant/bm25
RERANK_MODEL_NAME=Xenova/ms-marco-MiniLM-L-6-v2

# ─── LLM models ────────────────────────────────────────────────────────────
# Groq retires model IDs on a few months' notice. Verify against
# https://console.groq.com/docs/models before a fresh run.
# Llama 3.1 8B Instant and Llama 3.3 70B Versatile are scheduled for shutdown
# on 2026-08-16; the gpt-oss pair below are Groq's named replacements.
EXPANSION_MODEL=openai/gpt-oss-20b
GENERATION_MODEL=openai/gpt-oss-120b
GRADER_MODEL=openai/gpt-oss-20b

# ─── RAG hyperparameters ───────────────────────────────────────────────────
MAX_TOKENS_PER_CHUNK=400
SENTENCE_OVERLAP=2
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
MAX_RETRIES=2
NUM_QUERY_VARIATIONS=3
```

### Why three different LLM models

This is a cost-and-latency decision worth internalising, because it recurs in every agentic system
you will ever build.

The three LLM jobs in this system have wildly different difficulty. **Query expansion** rewrites one
sentence into three — a trivial task where a small fast model is indistinguishable from a large one.
**Grading** answers a yes/no question with a short justification — also easy, and it runs on *every*
query, sometimes several times per query in the correction loop. **Generation** must synthesise an
accurate legal answer from five source passages with correct citations — genuinely hard, and the one
place quality is visible to the user.

So we spend on generation (`gpt-oss-120b`) and economise on the other two (`gpt-oss-20b`). Since
grading can run three times per query, using the large model there would roughly triple your cost for
no measurable benefit. Routing each task to the cheapest model that can do it well is the core
economic skill of building on LLMs.

### Failure Modes

**`400 model_decommissioned`** — Groq retired the ID you are using. This is not hypothetical; it is
the most common recurring breakage in this project. Fix by checking the live list at
`https://console.groq.com/docs/models` and updating the three model variables. Phase 6 builds a
resolver that maps dead IDs to replacements automatically so this fails loudly and early rather than
mid-run.

**Settings silently keep their defaults on Windows** — the `.env` file was saved as UTF-16 (PowerShell
`>` redirection does this) or as `.env.txt` with a hidden extension. Save as UTF-8 and verify the
filename has no extension.

**`DATASET_PATH` with single backslashes** — `C:\data\edgar` contains `\d` and `\e`, which some
parsers treat as escape sequences. Use forward slashes; Python handles them fine on Windows.

---

## File 4 — `config/settings.py`

**Location:** `config/settings.py`

### The Problem

`os.getenv("MAX_TOKENS_PER_CHUNK")` returns the *string* `"400"`, not the integer `400`. Do
arithmetic on it and you get `"400400400"` instead of `1200`, or a `TypeError` somewhere far from the
cause. Worse, `os.getenv("ENABLE_SEMANTIC_CACHE")` returns the string `"false"`, which is *truthy* in
Python — so setting a flag to false turns the feature **on**. This is a real bug class that has
shipped to production in many codebases.

Environment variables are all strings. Something must convert and validate them at the boundary.

### Design Decision

**`pydantic-settings` over manual `os.getenv` calls.** It reads `.env` and the process environment,
coerces each value to its annotated type, and raises a clear error at startup if a value cannot
convert. `"false"` becomes `False`, `"400"` becomes `400`, and a malformed value fails immediately
with the field name — not three hours into an ingestion run.

**Fail fast on invalid values.** A field validator that rejects `CACHE_SIMILARITY_THRESHOLD=1.5`
at import time is worth far more than the same value silently producing zero cache hits forever.
Errors should surface at the boundary where bad data enters — roadmap §7.4.

**A module-level singleton, with eyes open.** The last line is `settings = Settings()`, which means
config loads once at import and everything shares that instance. This is convenient and it is what
most Python projects do, but be honest about the tradeoff: it is an **import-time side effect**.
Importing this module reads the filesystem, and if `.env` is malformed your program dies during
`import` rather than in `main()`. It also makes tests awkward, since you cannot easily construct a
`Settings` with different values. Phase 10 addresses this by wrapping construction in a
`@lru_cache`'d `get_settings()` function that tests can override via FastAPI's dependency injection.
For now, the singleton is the right simplicity/purity trade — just know it is a trade.

```python
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
```

### The Theory: why validators run where they do

Pydantic gives you two validation scopes and you need to know which to reach for.

A **`field_validator`** sees exactly one field. It runs during construction, and raising inside it
aborts the whole object. Use it for rules that depend on nothing else: "log level must be one of
these five strings", "port must be in range".

Cross-field rules are the interesting case. `RERANK_TOP_K > RETRIEVAL_TOP_K` is nonsense — you cannot
rerank twenty candidates down to fifty — but a `field_validator` on `RERANK_TOP_K` cannot see
`RETRIEVAL_TOP_K`, because field validators run per-field and ordering is not guaranteed. Pydantic's
answer is `@model_validator(mode="after")`, which runs once with the fully built object.

We deliberately did **not** use it here. A misconfigured top-k should not prevent the program from
starting — it is a bad-but-recoverable setting, and crashing at import over it is hostile. So it lives
in `validate_runtime()`, which returns warnings that startup code logs. The rule of thumb:
**`field_validator` for impossible values, `validate_runtime()` for unwise ones.**

`@computed_field` is the third tool: a derived value that is not read from the environment.
`contracts_dir` is always `DATASET_PATH / "contracts"`, so deriving it once here means no other file
ever hardcodes that join. Notice it also calls `.resolve()`, turning the relative `./data` into an
absolute path — which matters in Phase 2, where multiprocessing workers may not share a working
directory.

### Failure Modes

**`ValidationError: 1 validation error for Settings`** — this is the system working. Read the field
name and message; it tells you exactly which `.env` line is wrong.

**A `.env` change has no effect** — the `@lru_cache` on `get_settings()` means the object is built
once per process. Restart. In a Jupyter notebook, restart the kernel; re-running the cell will not
help.

**`extra="ignore"` hides your typos.** This is a genuine tradeoff. Writing `QDRANT_HOSTT=...` in
`.env` will be silently ignored and you will get the default. The stricter `extra="forbid"` catches
that but then crashes when unrelated environment variables are present, which is common in CI and
Docker. We chose tolerance; when a setting mysteriously does not apply, suspect a typo first.

---

## File 5 — `src/core/exceptions.py`

**Location:** `src/core/exceptions.py`

### The Problem

When retrieval fails you need to know *why*, because the correct response differs completely by
cause. Qdrant unreachable means retry with backoff. A malformed query means reject the request and
tell the user. A Groq rate limit means back off and retry later. An embedding dimension mismatch means
stop entirely — the collection is misconfigured and retrying will never help.

If every failure is a bare `Exception`, the calling code cannot distinguish these and must treat them
all identically. Typed exceptions let each caller handle what it can and re-raise what it cannot.

### Design Decision

**A hierarchy with one root.** Everything inherits from `RAGException`. This gives callers a choice
of granularity: `except VectorStoreError` for one specific failure, or `except RAGException` at the
API boundary to catch anything our code raised while letting genuine bugs like `TypeError` propagate
to the error tracker where they belong. A flat set of unrelated exception classes cannot do that.

**Structured `details`, not string formatting.** Message strings are for humans and cannot be
queried. Carrying a `details` dict means Phase 8 can serialise the error to JSON, and log aggregators
can filter on `details.collection` — impossible if the collection name is embedded in prose.

**A `to_dict()` method.** Phase 8 needs a consistent JSON error envelope for every API failure.
Putting that serialisation on the exception itself means the API layer never builds error responses
by hand.

```python
from typing import Any


class RAGException(Exception):
    """Root of the application's exception hierarchy.

    Catching `RAGException` catches every deliberate failure our own code raises,
    while letting genuine programming errors (TypeError, AttributeError) propagate.
    """

    #: Overridden by subclasses; used by the API layer to pick an HTTP status.
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API error envelopes and structured logs."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
        return f"{self.message} ({rendered})"


# ─── Ingestion, Pipeline 1 ─────────────────────────────────────────────────

class IngestionError(RAGException):
    """Base for any failure in the ingestion pipeline."""
    status_code = 500


class DocumentLoadError(IngestionError):
    """A file could not be read or decoded. Usually quarantined, not fatal."""


class ParsingError(IngestionError):
    """Section or metadata extraction failed on an otherwise readable document."""


class ChunkingError(IngestionError):
    """Text could not be segmented into chunks."""


# ─── Storage and vectors ───────────────────────────────────────────────────

class VectorStoreError(RAGException):
    """Vector database connection, indexing, or query failure."""
    status_code = 503


class CollectionNotFoundError(VectorStoreError):
    """The target collection does not exist. Ingestion has not run."""
    status_code = 404


class DimensionMismatchError(VectorStoreError):
    """Vector size does not match the collection schema. Never retryable —
    it means the embedding model changed without a re-index. See Phase 15."""
    status_code = 500


class EmbeddingError(RAGException):
    """Dense or sparse vector generation failed."""
    status_code = 500


# ─── Retrieval, Pipeline 2 ─────────────────────────────────────────────────

class RetrievalError(RAGException):
    """Hybrid search, fusion, or reranking failure."""
    status_code = 503


class RerankerError(RetrievalError):
    """Cross-encoder scoring failed."""


# ─── LLM and agent ─────────────────────────────────────────────────────────

class LLMProviderError(RAGException):
    """An LLM API call failed."""
    status_code = 502


class RateLimitError(LLMProviderError):
    """Provider rate limit hit. Always retryable with backoff."""
    status_code = 429

    def __init__(self, message: str, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class ModelDecommissionedError(LLMProviderError):
    """The configured model ID no longer exists. Requires a config change, not a retry."""
    status_code = 500


class GraphExecutionError(RAGException):
    """A LangGraph node or state transition failed."""
    status_code = 500


class MaxRetriesExceededError(GraphExecutionError):
    """The self-correction loop exhausted its retry budget without a passing grade."""


# ─── Guardrails, evaluation, security ──────────────────────────────────────

class GuardrailError(RAGException):
    """A safety check blocked the request or response."""
    status_code = 400


class PromptInjectionError(GuardrailError):
    """Input matched a prompt-injection signature."""


class CitationValidationError(GuardrailError):
    """The generated answer cited a source that was not in the retrieved context."""


class EvaluationError(RAGException):
    """Metric scoring or synthetic dataset generation failed."""
    status_code = 500


class AuthorizationError(RAGException):
    """The requesting principal lacks access to the requested resource. Phase 11."""
    status_code = 403
```

### The Theory: exception chaining and why `from exc` matters

When you catch a low-level error and raise your own, you have three options, and only one is right.

```python
# ❌ Original traceback destroyed. You know retrieval failed, not why.
except Exception:
    raise VectorStoreError("query failed")

# ❌ Python shows "During handling of the above exception, another occurred" —
#    technically it keeps both, but signals them as unrelated, which is misleading.
except Exception as exc:
    raise VectorStoreError("query failed")

# ✅ Explicit causal chain: "VectorStoreError ... caused by ConnectionRefusedError"
except Exception as exc:
    raise VectorStoreError(
        "Hybrid query failed",
        details={"collection": name},
    ) from exc
```

`from exc` sets `__cause__`, and Python renders it as *"The above exception was the direct cause of
the following exception."* You get your semantic error type for the caller to branch on **and** the
original network-level traceback for debugging. Never discard the cause.

The `retryable` flag exists so callers do not have to maintain their own list of which exception
types are transient:

```python
try:
    return await store.hybrid_search(...)
except RAGException as exc:
    if exc.retryable:
        await asyncio.sleep(backoff)
        continue
    raise
```

That is only possible because retryability is data on the exception rather than knowledge scattered
across call sites.

### Failure Modes

**Catching `RAGException` too early.** Deep in a loader, `except RAGException: continue` will swallow
a `DimensionMismatchError` that should have stopped the entire run. Catch narrowly at the bottom,
broadly at the top.

**Over-engineering the hierarchy.** Twenty-two classes may look excessive now. The test is whether
any caller ever branches on the distinction. `RateLimitError` earns its place because Phase 6 retries
it specifically. If a class never gets caught by name after Phase 10, it was noise.

---

## File 6 — `src/core/models.py` ★ the important one

**Location:** `src/core/models.py`

### The Problem

This is the file that fixes the deepest flaw in the existing prototype. Look at what
`Pipelines/ingestion.py` builds and hands onward:

```python
chunk = {
    "chunk_id": str(uuid.uuid4()),
    "chunk_text": c_text,
    "section": section_name,
}
chunk.update(meta)          # silently merges four more keys from somewhere else
```

Now `Pipelines/agent.py` consumes it:

```python
contexts += f"\nSOURCE: {c['contract_name']} ({c['year']}) Section: {c['section']}\n"
```

Four separate problems, all invisible until runtime.

Nothing documents the shape. To learn what keys a chunk has you must read `parse_document`, find the
`meta` dict, and mentally merge them. There is no single place that answers "what is a chunk?"

Typos are runtime crashes. `c['contract_nmae']` raises `KeyError` during generation — after
embedding, storage, retrieval, and reranking have all succeeded and cost you real time and money.

Types are unenforced. `year` arrives from a directory name, so it is the string `"2003"`. Any
comparison like `year > 2000` raises `TypeError`. Nothing prevents it.

No editor support. Your IDE cannot autocomplete a dict key or warn about a bad one.

### Design Decision

**Pydantic `BaseModel` for every structure that crosses a module boundary.** Fields are declared
once, validated at construction, and autocompleted everywhere. Bad data fails at the boundary where
it enters instead of five layers downstream.

**Why not `@dataclass`?** Dataclasses give typed fields and autocomplete but perform **no runtime
validation** — `Chunk(token_count="four hundred")` constructs happily. Pydantic coerces and rejects.
Since our data originates in files, environment variables, and LLM output — all untrusted, all
stringly-typed — runtime validation is the whole point. Dataclasses are right for internal
structures you fully control; these are not that.

**Why not `TypedDict`?** It is checked statically but is a plain dict at runtime, so no validation and
no methods. We do use `TypedDict` for exactly one thing — LangGraph state in Phase 5 — because
LangGraph merges partial state updates key by key, which needs real dict semantics.

**Separate `Document`, `Section`, and `Chunk`.** These represent genuinely different stages, and
collapsing them is how the prototype ended up with one dict carrying everything. A `Document` is a
whole file. A `Section` is `ARTICLE IV` within it. A `Chunk` is ~400 tokens within that section. The
ingestion pipeline is literally the transformation `Document → list[Section] → list[Chunk]`, and
naming each stage makes the pipeline self-documenting.

**`Chunk` carries denormalised metadata.** `contract_name` and `year` live on `Document`, yet we copy
them onto every `Chunk`. In a relational database this would be bad design. Here it is required: a
vector search returns chunk payloads, and the generation node needs the contract name to write a
citation. Making a second lookup per chunk to fetch it would add a round trip to every query. This is
the standard denormalisation tradeoff — storage is cheap, query latency is not.

```python
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp.

    Note `datetime.now(timezone.utc)`, not the deprecated `datetime.utcnow()` —
    the latter returns a naive datetime, which compares incorrectly against aware ones.
    """
    return datetime.now(timezone.utc)


class DocumentType(str, Enum):
    """Source file format.

    Inheriting from `str` as well as `Enum` means the member serialises directly
    to JSON as "txt" rather than needing a custom encoder.
    """
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


class ChunkLevel(str, Enum):
    """Granularity of a chunk in the parent-child hierarchy (Phase 2).

    CHILD chunks are small and precise — they are what we embed and search against,
    because a focused passage produces a sharper vector. PARENT chunks are the larger
    surrounding context we actually feed the LLM, because a 400-token fragment is
    often too little to answer from. STANDALONE means no hierarchy was applied.

    Retrieval filters to CHILD, then swaps in the parents. Phase 16 adds SUMMARY
    nodes above PARENT to form the RAPTOR tree.
    """
    STANDALONE = "standalone"
    CHILD = "child"
    PARENT = "parent"
    SUMMARY = "summary"


class RetrievalMethod(str, Enum):
    """Which retrieval strategy surfaced a given chunk. Used for debugging and evaluation."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"          # Phase 13
    SUMMARY_TREE = "summary" # Phase 16


# ─── Ingestion stage models ────────────────────────────────────────────────

class Document(BaseModel):
    """One source file, loaded but not yet parsed. Output of `src/ingestion/loaders/`."""

    # Shallow immutability: attribute assignment raises, but `metadata` is still a
    # mutable dict — `doc.metadata["k"] = v` succeeds. Pydantic cannot deep-freeze a
    # dict field. Treat `metadata` as append-only by convention; the guarantee here
    # is against *reassigning* fields, not against mutating a container inside one.
    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Deterministic UUIDv5 derived from the source path")
    source_path: str
    file_name: str
    content: str = Field(repr=False)         # excluded from repr — could be megabytes
    content_hash: str = Field(description="SHA-256 of content, for differential re-ingestion")
    doc_type: DocumentType = DocumentType.TXT

    contract_name: str = Field(default="Unknown Contract", max_length=300)
    year: int | None = Field(default=None, ge=1900, le=2100)

    loaded_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> int | None:
        """Directory names give us the string "2003"; we want the integer 2003.

        `mode="before"` runs prior to type coercion so we can handle junk directory
        names (e.g. "misc") by returning None instead of raising.
        """
        if v is None or v == "":
            return None
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return None

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)


class Section(BaseModel):
    """A structural division of a contract — `ARTICLE IV`, `SECTION 7.02`, `EXHIBIT B`.

    Output of `src/ingestion/parsers/`. Chunking happens *within* a section so that
    clause boundaries are never crossed.
    """

    section_id: str
    doc_id: str
    title: str = Field(default="Preamble", max_length=200)
    text: str = Field(repr=False)
    order: int = Field(ge=0, description="Zero-based position within the document")

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """The atomic unit of retrieval. One embedded, indexed passage.

    Carries denormalised document metadata so a vector search result is
    self-sufficient for citation without a second lookup.
    """

    chunk_id: str = Field(description="Deterministic UUIDv5 — stable across re-ingestion")
    doc_id: str
    section_id: str
    text: str

    chunk_index: int = Field(ge=0, description="Position within the parent section")
    token_count: int = Field(ge=0)

    # ── denormalised from Document and Section, for citation without a join ──
    contract_name: str = "Unknown Contract"
    file_name: str = ""
    section_title: str = "Preamble"
    year: int | None = None

    # ── parent-child chunking, Phase 2 ──
    chunk_level: ChunkLevel = Field(
        default=ChunkLevel.STANDALONE,
        description="Search filters to CHILD; generation substitutes the PARENT.",
    )
    parent_id: str | None = Field(
        default=None,
        description="The enclosing PARENT chunk's id. None for parents and standalone chunks.",
    )

    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        return v

    def to_payload(self) -> dict[str, Any]:
        """Flatten to a Qdrant payload dict.

        Qdrant stores plain JSON, so this is the one place we deliberately leave the
        typed world. Keeping the conversion here means no other module hand-builds
        payload dicts, and Phase 3's store adapter stays free of field knowledge.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "contract_name": self.contract_name,
            "file_name": self.file_name,
            "section_title": self.section_title,
            "year": self.year,
            "token_count": self.token_count,
            "chunk_level": self.chunk_level.value,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        """Rebuild a Chunk from a Qdrant payload. The inverse of `to_payload`."""
        return cls(
            chunk_id=payload["chunk_id"],
            doc_id=payload.get("doc_id", ""),
            section_id=payload.get("section_id", ""),
            text=payload["text"],
            chunk_index=payload.get("chunk_index", 0),
            token_count=payload.get("token_count", 0),
            contract_name=payload.get("contract_name", "Unknown Contract"),
            file_name=payload.get("file_name", ""),
            section_title=payload.get("section_title", "Preamble"),
            year=payload.get("year"),
            chunk_level=ChunkLevel(payload.get("chunk_level", "standalone")),
            parent_id=payload.get("parent_id"),
        )


# ─── Retrieval stage models ────────────────────────────────────────────────

class ScoredChunk(BaseModel):
    """A chunk plus its relevance score. Composition, not inheritance.

    A scored chunk is not a *kind of* chunk — it is a chunk *with* a score attached
    in a particular retrieval context. Subclassing `Chunk` here would let a scored
    result be passed anywhere a plain chunk is expected, hiding the fact that its
    score is only meaningful relative to one query.
    """

    chunk: Chunk
    score: float
    rank: int = Field(ge=0)
    method: RetrievalMethod = RetrievalMethod.HYBRID

    #: Populated only after cross-encoder reranking, so pre/post scores stay comparable.
    rerank_score: float | None = None

    @computed_field
    @property
    def effective_score(self) -> float:
        """The score to sort by — reranked if available, else the retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.score


class RetrievalResult(BaseModel):
    """Everything one retrieval round produced, including diagnostics.

    The query variations and timing are not decoration — Phase 7's evaluation
    harness reads them, and they are what you inspect when recall is poor.
    """

    original_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    chunks: list[ScoredChunk] = Field(default_factory=list)
    total_candidates: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ─── Generation and grading models ─────────────────────────────────────────

class Citation(BaseModel):
    """One source reference in a generated answer."""

    source_index: int = Field(ge=1, description="The N in [SOURCE N]")
    chunk_id: str
    contract_name: str
    section_title: str
    year: int | None = None


class GradingReport(BaseModel):
    """The self-correction verdict. Parsed from the grader LLM's JSON output.

    This model is the reason grading is reliable: the LLM is asked for JSON, and
    Pydantic rejects anything that does not conform, so a malformed grade fails
    loudly instead of being read as a silent pass.
    """

    is_grounded: bool = Field(description="Every claim traceable to retrieved context")
    is_relevant: bool = Field(description="The answer addresses the question asked")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=2000)
    unsupported_claims: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def passed(self) -> bool:
        """Both gates must clear. This drives the retry edge in Phase 5's graph."""
        return self.is_grounded and self.is_relevant


class RAGAnswer(BaseModel):
    """The final response object. Serialised directly by Phase 8's API."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[ScoredChunk] = Field(default_factory=list)

    grading: GradingReport | None = None
    retry_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=_utc_now)
```

### The Theory: composition versus inheritance, and why `ScoredChunk` is built the way it is

`ScoredChunk` embeds a `Chunk` as a field rather than subclassing it. This is the most consequential
design choice in the file and it is worth understanding properly.

Inheritance asserts *is-a*. If `ScoredChunk(Chunk)`, then by the Liskov substitution principle a
`ScoredChunk` is usable anywhere a `Chunk` is expected. But a score is not an intrinsic property of a
passage — it is a property of *this passage relative to that query*. A chunk scoring 0.94 for
"termination notice period" scores 0.02 for "governing law". Baking the score into the chunk type
lets a query-specific value leak into contexts where it is meaningless.

Composition asserts *has-a*, which is what is actually true: this result *has* a chunk and *has* a
score. It also keeps `Chunk` clean — the storage layer never sees score fields it must ignore, and
`to_payload()` has no query-specific data to strip.

The cost is one extra attribute hop: `result.chunk.text` instead of `result.text`. That verbosity is
a feature. It tells the reader, at every use site, that they are looking at a retrieval result rather
than a stored document.

The `rerank_score` field follows the same reasoning. It would be simpler to overwrite `score` during
reranking, but then you lose the ability to compare the two — and comparing them is exactly how you
prove your reranker earns its 150ms. Phase 4's checkpoint depends on both being present, and it is
one of the project's success metrics in the PRD.

### Failure Modes

**`ValidationError: Chunk text cannot be empty`** during ingestion. The validator is doing its job:
your parser produced an empty section, usually from consecutive section headers with no body between
them. Fix the parser to skip empty sections — do not relax the validator. An empty chunk embeds to a
meaningless vector that pollutes every subsequent search.

**`frozen=True` on `Document` raises on assignment.** Deliberate. If you need a modified document,
use `doc.model_copy(update={"contract_name": "..."})`, which returns a new instance.

Be precise about what this guarantees, though: `frozen=True` is **shallow**. It blocks
`doc.contract_name = "x"`, but `doc.metadata["source"] = "x"` still succeeds, because the field holds
an ordinary mutable dict and Pydantic has no way to freeze its contents. So the protection covers
field *reassignment*, not mutation of a container inside a field. If you need the stronger guarantee,
the options are a `frozen=True` nested model instead of a dict, or converting to a `MappingProxyType`
after construction — both add friction that is not worth it here. Treat `metadata` as append-only by
convention and know that the convention is not enforced.

**`content` missing from log output.** Also deliberate: `Field(repr=False)`. Printing a `Document`
would otherwise dump the entire contract to your terminal, and roadmap §7.5 forbids logging document
bodies. Access `doc.content` explicitly when you actually need it.

**`year` is `None` when you expected a number.** The `_coerce_year` validator returns `None` for
unparseable directory names rather than raising, because one oddly-named folder should not abort a
650,000-file ingestion. Always treat `year` as `int | None` downstream.

---

## File 7 — `src/core/interfaces.py`

**Location:** `src/core/interfaces.py`

### The Problem

Phase 3 builds a Qdrant adapter. Phase 15 needs to run two vector stores side by side during an
embedding migration. Phase 6 uses Groq but must fall back to OpenAI on an outage. Phase 14 adds three
new document loaders.

If `src/graph/nodes/retrieval_node.py` imports `QdrantStore` directly and calls
`qdrant_client.query_points(...)`, then every one of those changes requires editing the graph nodes.
The high-level agent logic becomes welded to a specific vendor SDK.

### Design Decision

**Abstract base classes to invert the dependency.** Instead of high-level code depending on
low-level implementations, both depend on an abstraction defined here. `retrieval_node.py` accepts a
`BaseVectorStore`; whether that is Qdrant, Chroma, or a test double is decided at wiring time by a
factory in Phase 3.

**Why ABCs rather than duck typing?** Python does not need interfaces — pass any object with the
right methods and it works. But duck typing fails *late* and *quietly*. Forget to implement
`upsert_points` on a new store and you find out when ingestion reaches the upsert call, possibly
after twenty minutes of embedding. `abc.ABC` with `@abstractmethod` makes the class impossible to
instantiate without every method present, so you get `TypeError: Can't instantiate abstract class` at
construction. Failing at the earliest possible moment is worth the boilerplate.

They are also documentation. `interfaces.py` is the one file a new reader can open to understand the
system's seams without reading any implementation.

**Typed against `models.py`, not `dict[str, Any]`.** The old draft declared
`upsert_points(points: List[Dict[str, Any]]) -> bool`, which communicates nothing. Ours says
`upsert_points(chunks: list[Chunk], ...) -> int`, which says exactly what goes in and what comes out.

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.models import Chunk, Document, ScoredChunk, Section


class BaseDocumentLoader(ABC):
    """Bytes on disk → a `Document`. One implementation per file format."""

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Whether this loader can handle the given path. Used by the factory to dispatch."""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Read and decode one file.

        Raises:
            DocumentLoadError: unreadable, undecodable, or empty.
        """


class BaseParser(ABC):
    """A `Document` → its structural `Section`s."""

    @abstractmethod
    def parse(self, document: Document) -> list[Section]:
        """Segment a document along structural boundaries.

        Raises:
            ParsingError: the document structure could not be interpreted.
        """


class BaseChunker(ABC):
    """A `Section` → retrievable `Chunk`s."""

    @abstractmethod
    def chunk(self, section: Section, document: Document) -> list[Chunk]:
        """Split a section into token-bounded chunks.

        `document` is passed so the chunker can denormalise contract metadata
        onto each chunk (see the `Chunk` model's rationale).
        """


class BaseEmbeddingProvider(ABC):
    """Text → vectors. Dense and sparse.

    Async, even though the local FastEmbed implementation is CPU-bound. The reason
    is that the *other* planned implementation — OpenAI — is a network call, and a
    synchronous interface would force it to block the event loop on every request.
    The interface must accommodate both, so it is async, and the CPU-bound
    implementation offloads to a thread internally (roadmap §7.2).

    Phase 2's multiprocessing ingestion path needs a synchronous entry point and so
    calls `embed_dense_sync` directly, bypassing the event loop entirely.
    """

    @property
    @abstractmethod
    def dense_dimensions(self) -> int:
        """Vector size, needed to create the collection schema in Phase 3."""

    @abstractmethod
    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense semantic embeddings."""

    @abstractmethod
    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        """Generate sparse BM25-style vectors as {"indices": [...], "values": [...]}."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from `embed_dense` because some models require an asymmetric
        instruction prefix for queries versus documents — `bge` is one of them.
        """

    @abstractmethod
    async def embed_sparse_query(self, text: str) -> dict[str, list]:
        """Sparse vector for a single query.

        Separate from `embed_sparse` for the same reason `embed_query` is separate
        from `embed_dense`, and it is not a nicety: BM25's document
        representation applies term-frequency saturation and length
        normalisation, neither of which is meaningful for a query. Using the
        document-side embedding for a query produces a subtly mis-weighted vector
        that still returns results, so the mistake is invisible.
        """

    async def close(self) -> None:
        """Release any resources the provider holds. Default is a no-op.

        Not abstract because local providers hold only an ONNX session, which the
        interpreter reclaims. Network-backed providers own an HTTP connection pool
        and must override this — see the lifecycle note on the Phase 3 factory.
        """
        return None

    def embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        """Blocking dense embedding, for use inside multiprocessing workers.

        Not abstract: local providers override it with the real implementation and
        have `embed_dense` delegate here via `asyncio.to_thread`. Network-backed
        providers leave it unimplemented, because running an HTTP client inside a
        forked worker process is a bad idea regardless.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support synchronous embedding; "
            "use the async interface."
        )


class BaseVectorStore(ABC):
    """Vector database operations. Async, because every call is network I/O."""

    @abstractmethod
    async def initialize(self, vector_size: int) -> None:
        """Create the collection and payload indexes if absent. Must be idempotent —
        calling it twice on an existing collection must not destroy data."""

    @abstractmethod
    async def upsert_points(self, chunks: list[Chunk], dense: list[list[float]],
                            sparse: list[dict[str, list]]) -> int:
        """Write chunks with both vector types. Returns the number of points written."""

    @abstractmethod
    async def hybrid_search(self, dense_query: list[float], sparse_query: dict[str, list],
                            limit: int = 20,
                            filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        """Server-side fused dense + sparse search.

        Raises:
            CollectionNotFoundError: the collection does not exist.
            VectorStoreError: the query failed.
        """

    @abstractmethod
    async def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Retrieve specific points by ID, with no vector search.

        Needed because `chunk_id` is not a filterable field — the payload indexes
        cover `doc_id`, `chunk_level`, `year`, and `section_title`. Phase 4's
        parent substitution looks parents up by ID, and Phases 13 and 16 walk
        chunk references the same way.

        Returns only the chunks that exist. A missing ID is not an error: Phase 4
        treats an absent parent as "use the child", which is a correct degradation
        rather than a failure.
        """

    @abstractmethod
    async def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        """Remove every chunk of every listed document. Returns the count deleted.

        Called immediately before re-upserting. Deterministic chunk IDs make an
        *unchanged* re-ingest idempotent, but they cannot remove chunks that no
        longer exist — if a document's chunk count shrinks, or its section titles
        change, the old points would otherwise linger as stale ghosts. See the
        discussion in `utils.py` below.

        The batch form is the abstract one because that is what the caller
        actually needs: a 256-chunk batch spans dozens of documents, and one
        request per document would double the round trips of an entire ingestion
        run. A store that can only delete singly implements this as a loop.
        """

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Single-document convenience wrapper. Not abstract — it delegates."""
        return await self.delete_by_doc_ids([doc_id])

    @abstractmethod
    async def count(self) -> int:
        """Number of points currently indexed. Used by health checks."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections. Called from the FastAPI lifespan hook in Phase 8."""


class BaseReranker(ABC):
    """Reorders retrieved candidates by deeper relevance scoring."""

    @abstractmethod
    async def rerank(self, query: str, candidates: list[ScoredChunk],
                     top_k: int = 5) -> list[ScoredChunk]:
        """Re-score and truncate.

        Async despite being CPU-bound: implementations offload to
        `asyncio.to_thread` internally, so callers get a uniform await-able API.
        Returned chunks must have `rerank_score` populated.
        """


class BaseLLMProvider(ABC):
    """Text generation. Async — every call is network I/O."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       model: str | None = None) -> str:
        """Single completion.

        Raises:
            RateLimitError: provider throttled the request (retryable).
            ModelDecommissionedError: the model ID no longer exists.
            LLMProviderError: any other API failure.
        """

    @abstractmethod
    async def generate_json(self, prompt: str, schema: type,
                            system_prompt: str | None = None) -> Any:
        """Completion constrained to JSON, validated against a Pydantic model.

        This is how `GradingReport` is produced reliably — the provider is asked
        for JSON mode and the result is parsed through Pydantic, so a malformed
        grade raises rather than silently reading as a pass.
        """

    @abstractmethod
    def stream(self, prompt: str, system_prompt: str | None = None) -> AsyncIterator[str]:
        """Yield tokens as they arrive. Powers SSE streaming in Phase 8.

        Declared `def`, not `async def`, returning `AsyncIterator[str]`. This is the
        correct signature for an async generator: the function itself is not a
        coroutine — calling it returns the iterator immediately, and you consume it
        with `async for`. Implementations use `async def` with `yield`, which Python
        types as returning `AsyncIterator`. Writing `async def stream(...)` in the ABC
        would demand `await provider.stream(...)` before iterating, which is wrong.
        """


class BaseCache(ABC):
    """Response caching, exact-match or semantic."""

    @abstractmethod
    async def get(self, query: str) -> Any | None:
        """Return a cached answer, or None on miss."""

    @abstractmethod
    async def set(self, query: str, value: Any, ttl: int | None = None) -> None:
        """Store a response."""

    @abstractmethod
    async def clear(self) -> None:
        """Evict everything."""


class BaseGuardrail(ABC):
    """A safety check applied to input or output."""

    @abstractmethod
    def check(self, text: str) -> tuple[bool, str | None]:
        """Return `(passed, reason)`. `reason` is None when passed."""


class BaseMetric(ABC):
    """One evaluation metric. Phase 7."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def score(self, query: str, answer: str, contexts: list[str]) -> float:
        """Return a score in [0.0, 1.0]."""
```

### Why some methods are async and others are not

This distinction is deliberate and reflects roadmap §7.2, so make sure it lands.

`BaseEmbeddingProvider` methods are **async**, and this is the one that needs justifying, because the
default implementation is CPU-bound. The rule is that **an interface is shaped by its most demanding
implementation, not its simplest one.** FastEmbed runs an ONNX model on your CPU with nothing to
await; if it were the only implementation, sync would be the honest signature. But the planned OpenAI
embedding provider is a 100–500ms HTTP round trip, and a synchronous interface would force it to
block the event loop — Phase 8's server would stop serving every other request for the duration. A
caller cannot make a sync network call concurrent; an implementation *can* present a CPU-bound
computation behind an async surface by offloading internally. So the interface is async, and the
local provider delegates through `asyncio.to_thread` in one place (Phase 3's
`LocalEmbeddingProvider`). The non-abstract `embed_dense_sync` is the escape hatch for callers that
have no event loop at all, such as a multiprocessing worker.

`BaseVectorStore` and `BaseLLMProvider` methods are **async**. Every call crosses a network boundary
and spends nearly all its wall-clock time waiting. This is precisely what `async` is for: Phase 4
issues three query variations concurrently with `asyncio.gather`, turning three sequential 200ms
round trips into one 200ms wait.

`BaseReranker.rerank` is **async for the same reason**, minus the network justification: the reranker
sits between two async stages in the graph, and forcing every caller to remember the `to_thread`
wrapper invites the exact bug we are avoiding. The implementation does it internally, once,
correctly. The interface presents a uniform async surface; the messy detail is encapsulated.

`BaseGuardrail.check` is **sync**, and that is the contrast that makes the rule visible. It is a
regex match over a string — microseconds, no I/O, no plausible alternative implementation that
blocks. Marking it async would add an await with nothing behind it. The test is not "is this work
slow", it is "could any reasonable implementation of this interface need to wait".

### Failure Modes

**`TypeError: Can't instantiate abstract class QdrantStore with abstract method close`** — the
mechanism working as intended. You missed a method. The message names it.

**Circular import between `interfaces.py` and `models.py`.** Only occurs if you later import
`interfaces` from `models`. Never do that — `models.py` must import nothing from `src/`. Dependencies
point one way (roadmap §4).

**An abstract method with a body someone forgot to override.** `@abstractmethod` prevents
instantiation of the abstract class, but if a subclass overrides nothing and the base has a real
implementation, you silently get base behaviour. Keep abstract method bodies as docstrings only, as
above — never a partial implementation.

---

## File 8 — `src/core/logging.py`

**Location:** `src/core/logging.py`

### The Problem

`print()` is invisible in production. It has no severity level, no timestamp, no source location, and
cannot be filtered or routed. When something fails at 3 a.m. inside one of forty concurrent requests,
you need to reconstruct that single request's path through the system.

That last point is the hard one. With concurrent async requests interleaved in one output stream, log
lines from different queries are shuffled together. Without a way to tag every line with which
request produced it, the log is unreadable exactly when you need it most.

### Design Decision

**JSON output rather than formatted text.** Human-readable lines are pleasant locally and useless at
scale. `{"level":"ERROR","chunk_id":"abc"}` can be queried by any log aggregator; `ERROR: failed on
chunk abc` requires regex archaeology.

**`contextvars` for correlation IDs.** This is the piece the old draft lacked and the reason this
file gets rewritten. A `ContextVar` is like a thread-local that also works correctly with `asyncio` —
each task gets its own value, automatically propagated through every `await` in that task. Set a
request ID once when a request arrives, and every log line emitted anywhere downstream carries it,
with no need to thread the ID through forty function signatures.

**`datetime.now(timezone.utc)`, not `datetime.utcnow()`.** The latter is deprecated in Python 3.12
and — worse — returns a *naive* datetime with no timezone attached, which compares incorrectly
against aware datetimes and silently produces wrong durations.

```python
import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from config.settings import settings

# Propagates automatically across `await` boundaries within one asyncio task.
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Generate and bind a fresh correlation ID for the current context.

    Called once per inbound request (Phase 8 middleware) or per CLI invocation.
    Every subsequent log line in that context inherits it.
    """
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    """Bind an externally supplied ID, e.g. from an X-Request-ID header."""
    _request_id.set(rid)


def get_request_id() -> str | None:
    return _request_id.get()


#: Attributes present on every LogRecord. Anything not in here was passed via
#: `extra=` and should be surfaced as structured context.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class StructuredJSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        rid = _request_id.get()
        if rid:
            payload["request_id"] = rid

        # Anything passed as logger.info("...", extra={"chunk_id": x}) lands on the
        # record as a plain attribute. Promote those into a nested context object.
        context = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if context:
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # default=str so datetimes, Paths, and Enums serialise instead of raising.
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Compact coloured output for local development.

    JSON is correct for machines and miserable to read while coding. We select
    between the two on ENVIRONMENT rather than making anyone choose.
    """

    _COLOURS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        rid = _request_id.get()
        prefix = f"[{rid}] " if rid else ""
        line = f"{colour}{stamp} {record.levelname:<8}{self._RESET} {prefix}{record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logger(name: str = "legalrag") -> logging.Logger:
    """Configure and return the application logger. Idempotent."""
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    # Guard against duplicate handlers — this module may be imported many times,
    # and each extra handler duplicates every line of output.
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            StructuredJSONFormatter() if settings.is_production else HumanFormatter()
        )
        log.addHandler(handler)

    # Do not forward to the root logger, which would print everything twice
    # once uvicorn installs its own handlers in Phase 8.
    log.propagate = False
    return log


logger = setup_logger()
```

### The Theory: what a `ContextVar` actually does

Consider forty concurrent requests in one process. Thread-locals are no help — `asyncio` runs all of
them on a single thread, so they would share one value and overwrite each other.

A `ContextVar` is scoped to the *asyncio task*, not the thread. When a task awaits, its context is
saved; when it resumes, the context is restored. So:

```python
# Request A                          # Request B (concurrent)
set_request_id("aaa")                set_request_id("bbb")
await store.search(...)              await store.search(...)
logger.info("done")   # → aaa        logger.info("done")   # → bbb
```

Both tasks call the same `logger.info` in the same function, and each line carries the correct ID,
because the formatter reads the ContextVar from whichever task's context is active. No parameter
threading, no globals, no mutexes.

This is what makes distributed tracing possible. In Phase 8 one middleware sets the ID; every
downstream log line in that request — through retrieval, reranking, generation, grading — inherits
it. Filtering your log aggregator on `request_id=aaa` then replays exactly one user's query end to
end.

### Failure Modes

**Every log line appears twice or three times.** Duplicate handlers. Either `setup_logger` ran
without the `if not log.handlers` guard, or you are also seeing root-logger output — hence
`propagate = False`.

**`TypeError: Object of type PosixPath is not JSON serializable`** — this is why `default=str` is
passed to `json.dumps`. Without it, logging a `Path` or `datetime` in `extra=` crashes the log call,
which is a spectacular way to lose the error you were trying to record.

**`request_id` absent from lines inside a `multiprocessing` worker.** This is the real boundary, and
it is worth being precise about where it lies, because the two cases behave differently.

`asyncio.to_thread` **does** propagate context — it copies the current `contextvars.Context` into the
worker thread specifically so that variables like this remain visible. Anything logged from inside a
`to_thread` call keeps its `request_id`. Same for `loop.run_in_executor` when you pass a context
explicitly, and for `asyncio.create_task`, which copies the context at creation time.

What does **not** propagate is a separate *process*. Phase 2's multiprocessing ingestion workers get a
fresh interpreter with empty context, so log lines from a worker will have no `request_id` unless you
pass one through the task payload and call `set_request_id` at the top of the worker function. That is
why `set_request_id` exists alongside `new_request_id` — it is the seam for re-establishing context
across a process boundary.

**Colour escape codes appear as literal `\033[32m` garbage** in a file or an older Windows terminal.
Cosmetic. `HumanFormatter` is only used outside production; if it bothers you, gate the colours on
`sys.stdout.isatty()`.

---

## File 9 — `src/core/telemetry.py`

**Location:** `src/core/telemetry.py`

### The Problem

A query takes 4 seconds; the PRD's target is under 3. Which stage is responsible? Query expansion,
retrieval, reranking, generation, grading — each is a plausible suspect and guessing wastes hours. You
need per-stage timings without pasting `time.perf_counter()` into thirty functions.

### Design Decision

**A decorator, so instrumentation is one line per function** and the timing logic exists once.

**It must handle both sync and async functions.** This is why the old draft's version was
insufficient — it wrapped only sync callables. Decorating an `async def` with a sync wrapper does not
time the coroutine; it times how long it took to *create* the coroutine object, which is roughly zero
nanoseconds and utterly meaningless. Since most of our code is async, that decorator would have
reported near-zero latency for everything. We detect with `inspect.iscoroutinefunction` and return
the appropriate wrapper.

**Timings emitted as structured log context, not a metrics backend.** A real deployment would export
to Prometheus or OpenTelemetry. Our spans go out via `extra={"duration_ms": ...}`, which the JSON
formatter turns into a queryable field. That is enough to answer the question and adds no
infrastructure. Phase 7 aggregates these into the benchmark harness.

```python
import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from src.core.logging import logger

F = TypeVar("F", bound=Callable[..., Any])


class Telemetry:
    """Latency instrumentation for pipeline stages."""

    @staticmethod
    def span(name: str, warn_over_ms: float | None = None) -> Callable[[F], F]:
        """Decorator recording execution duration. Works on sync and async functions.

        Args:
            name: Span label, e.g. "retrieval.hybrid_search".
            warn_over_ms: Log at WARNING instead of INFO above this threshold.
                          Use the PRD's per-stage budgets here.
        """

        def decorator(func: F) -> F:
            def _emit(elapsed: float, failed: bool, exc: BaseException | None = None) -> None:
                ctx = {"span": name, "duration_ms": round(elapsed, 2)}
                if failed:
                    logger.error(f"{name} failed after {elapsed:.1f}ms", extra=ctx, exc_info=exc)
                elif warn_over_ms is not None and elapsed > warn_over_ms:
                    logger.warning(
                        f"{name} took {elapsed:.1f}ms (budget {warn_over_ms:.0f}ms)", extra=ctx
                    )
                else:
                    logger.debug(f"{name} completed in {elapsed:.1f}ms", extra=ctx)

            # Order matters. A generator function is not a coroutine function, and
            # timing one with the plain sync wrapper measures only how long it took to
            # CREATE the generator object — effectively zero — not how long iterating
            # it takes. Any pipeline stage written as a generator would silently report
            # ~0ms. Check the generator forms first.
            if inspect.isasyncgenfunction(func):
                @functools.wraps(func)
                async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter()
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                    except BaseException as exc:
                        _emit((time.perf_counter() - start) * 1000, True, exc)
                        raise
                    _emit((time.perf_counter() - start) * 1000, False)

                return async_gen_wrapper  # type: ignore[return-value]

            if inspect.isgeneratorfunction(func):
                @functools.wraps(func)
                def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter()
                    try:
                        yield from func(*args, **kwargs)
                    except BaseException as exc:
                        _emit((time.perf_counter() - start) * 1000, True, exc)
                        raise
                    _emit((time.perf_counter() - start) * 1000, False)

                return gen_wrapper  # type: ignore[return-value]

            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        _emit((time.perf_counter() - start) * 1000, True, exc)
                        raise
                    _emit((time.perf_counter() - start) * 1000, False)
                    return result

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    _emit((time.perf_counter() - start) * 1000, True, exc)
                    raise
                _emit((time.perf_counter() - start) * 1000, False)
                return result

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    @staticmethod
    @contextmanager
    def measure(name: str) -> Iterator[dict[str, float]]:
        """Time an arbitrary block rather than a whole function.

            with telemetry.measure("embed_batch") as m:
                vectors = embedder.embed_dense(texts)
            print(m["duration_ms"])

        The yielded dict is mutated on exit, so the caller can read the timing —
        useful for populating `RetrievalResult.latency_ms`.
        """
        holder: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield holder
        finally:
            holder["duration_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(
                f"{name} completed in {holder['duration_ms']:.1f}ms",
                extra={"span": name, "duration_ms": round(holder["duration_ms"], 2)},
            )

    @staticmethod
    @asynccontextmanager
    async def measure_async(name: str) -> AsyncIterator[dict[str, float]]:
        """Async counterpart to `measure`, for timing blocks containing awaits."""
        holder: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield holder
        finally:
            holder["duration_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(
                f"{name} completed in {holder['duration_ms']:.1f}ms",
                extra={"span": name, "duration_ms": round(holder["duration_ms"], 2)},
            )


telemetry = Telemetry()
```

### Why `functools.wraps` is not optional

Without `@functools.wraps(func)`, the wrapper replaces the original function's identity. `func.__name__`
becomes `"async_wrapper"`, the docstring vanishes, and — the part that actually bites — `pytest` can
no longer discover decorated test functions, and FastAPI in Phase 8 cannot introspect route handler
signatures to build its OpenAPI schema. `wraps` copies `__name__`, `__doc__`, `__module__`, and
`__wrapped__` across so the wrapper impersonates the original faithfully. Never write a decorator
without it.

Note also `except BaseException` rather than `except Exception`. We want the timing recorded even
when a task is cancelled — `asyncio.CancelledError` inherits from `BaseException`, not `Exception`,
so a bare `except Exception` would miss cancellations entirely. Since we re-raise immediately, the
broad catch changes no semantics; it only guarantees the measurement is emitted.

### Failure Modes

**Async functions report ~0.01ms.** You applied a sync-only decorator to a coroutine function. That
is precisely the bug this file exists to prevent — if you see suspiciously instant async spans, check
that `inspect.iscoroutinefunction` branch is present.

**Generator functions report ~0ms.** The same failure in a different disguise, and the reason the
`isgeneratorfunction` branch exists. Calling a generator function does not execute its body — it
returns a generator object, and the body runs only as the caller iterates. So a sync wrapper times
object *creation*. Phase 2's `IngestionPipeline.run()` is a generator that runs for hours, and without
this branch it would report a fraction of a millisecond. Note also that the generator branches emit
their span when iteration **completes**, so a caller that abandons the generator early never gets a
span — that is unavoidable and worth knowing.

**No timing output at all.** Successful spans log at `DEBUG`, and `LOG_LEVEL` defaults to `INFO`. Set
`LOG_LEVEL=DEBUG` in `.env`. They are intentionally at DEBUG because a span per stage per request is
enormously noisy in production.

**Decorator ordering with `@staticmethod` or `@property`.** `@telemetry.span` must be applied
closest to the function — i.e. listed *below* `@staticmethod`. Reversed, you are decorating a
descriptor object rather than a callable.

---

## File 10 — `src/core/utils.py`

**Location:** `src/core/utils.py`

### The Problem

Three small utilities, one of which fixes a genuine data-corruption bug.

**Token counting.** Chunk sizes are specified in tokens, not characters, because that is the unit LLM
context windows are measured in. `tiktoken.get_encoding(...)` takes about 100ms and loads a vocabulary
file. Calling it inside a chunking loop over 650,000 documents means 650,000 needless loads — hours of
pure waste.

A caveat to state up front, because it is easy to get wrong: **`tiktoken` cannot give you exact counts
for the models we actually use.** `cl100k_base` is GPT-3.5/GPT-4's encoding; the GPT-OSS models
configured in `.env` use `o200k_harmony`. So these counts are a close approximation, which is entirely
adequate for deciding chunk boundaries and completely unsuitable for cost accounting. We default to
the nearest available encoding and are explicit that it is an estimate.

**File hashing.** Re-running ingestion should skip unchanged files. That needs a stable content
fingerprint, computed without loading a 40 MB file entirely into memory.

**Deterministic IDs — the real bug.** Look at what the prototype does:

```python
chunk = {"chunk_id": str(uuid.uuid4()), ...}
```

`uuid4()` is random. Ingest a file today and its chunks get one set of IDs; re-ingest the same
unchanged file tomorrow and they get entirely different IDs. Qdrant sees new points, so instead of
updating in place it **inserts duplicates**. Every re-run silently multiplies your index. Retrieval
then returns the same passage three times, which crowds out other relevant results and wastes context
window on repetition.

### Design Decision

**A module-level tokenizer singleton behind `@lru_cache`.** Load once per process, reuse forever.

**SHA-256 streamed in fixed-size blocks.** Constant memory regardless of file size. Not MD5 — not for
security reasons here, but because SHA-256 is the modern default and there is no performance reason
to prefer MD5 at this scale.

**UUIDv5 for all IDs.** UUIDv5 is a *hash* of a namespace plus a name, so identical inputs always
produce the identical UUID. `chunk_id = uuid5(NAMESPACE, f"{doc_id}:{section}:{index}")` means
re-ingesting an unchanged document produces byte-identical IDs, so Qdrant's upsert genuinely
*updates* rather than duplicating. **Ingestion becomes idempotent** — you can re-run it safely any
number of times. This single change is also what makes Phase 15's shadow re-indexing and Phase 2's
resumable checkpointing possible at all.

```python
import hashlib
import re
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path

import tiktoken

#: Fixed namespace for all UUIDv5 generation in this project. Must never change —
#: altering it changes every ID in the system and orphans the entire existing index.
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

_HASH_BLOCK_SIZE = 65_536


#: Encoding used for chunk sizing. The GPT-OSS models we generate with use
#: `o200k_harmony`; `o200k_base` is its closest widely-available relative and is
#: what older tiktoken builds ship. Counts are an estimate either way — see
#: `count_tokens`.
DEFAULT_ENCODING = "o200k_harmony"
FALLBACK_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def get_tokenizer(encoding: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Process-wide tokenizer singleton.

    Loading an encoding costs ~100ms. `lru_cache` makes every call after the first
    effectively free, which matters inside a 650k-document chunking loop.

    Falls back gracefully if the installed `tiktoken` does not know the requested
    encoding, since encoding availability varies by version and a hard failure here
    would block the entire pipeline over a naming detail.
    """
    try:
        return tiktoken.get_encoding(encoding)
    except (KeyError, ValueError):
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def count_tokens(text: str) -> int:
    """Approximate token count, used for chunk sizing.

    Deliberately approximate. `tiktoken` implements OpenAI's encodings, and while
    GPT-OSS uses `o200k_harmony`, exact parity with any given model's tokenizer is
    not guaranteed — expect a few percent drift. That is fine for deciding where to
    cut a chunk, and it is NOT suitable for billing estimates.

    Still far better than a character heuristic (`len(text) // 4`), which drifts
    badly on legal text — dense with capitalised defined terms, section numbers, and
    punctuation that all tokenize unusually.
    """
    if not text:
        return 0
    return len(get_tokenizer().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate to a token budget. Last-resort guard before an LLM call."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def hash_file(file_path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in blocks so memory stays constant."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while block := handle.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of a string. Used for semantic-cache keys in Phase 6."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(source_path: str | Path) -> str:
    """Stable document ID derived from the source path.

    Normalised to POSIX separators so the same file yields the same ID whether
    ingested on Windows or Linux.
    """
    normalised = Path(source_path).as_posix()
    return str(uuid.uuid5(NAMESPACE, normalised))


def make_section_id(doc_id: str, section_title: str, order: int) -> str:
    """Stable section ID. `order` disambiguates repeated titles within one document."""
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{order}:{section_title}"))


def make_chunk_id(doc_id: str, section_id: str, chunk_index: int) -> str:
    """Stable chunk ID — the fix for duplicate-on-re-ingest.

    Because this is a pure function of its inputs, re-ingesting an unchanged
    document regenerates identical IDs, so the vector store updates in place
    instead of inserting duplicates. Ingestion becomes idempotent.
    """
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{section_id}:{chunk_index}"))


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip zero-width characters.

    Contract text scraped from filings is full of non-breaking spaces, soft
    hyphens, and zero-width joiners. They inflate token counts, break sentence
    splitting, and make otherwise-identical text hash differently.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")                    # soft hyphen
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)    # zero-width & BOM
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def safe_filename(name: str, max_length: int = 120) -> str:
    """Sanitise a string for use as a filename. Phase 2's dead-letter queue needs this."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_. ")
    return (cleaned[:max_length] or "unnamed")
```

### The Theory: UUIDv5 versus UUIDv4, concretely

The difference is worth seeing rather than describing.

```python
uuid.uuid4()                      # random every call
# 3f2a...  then  9c81...  then  b47e...

uuid.uuid5(NAMESPACE, "doc:sec:0")   # a SHA-1 hash of namespace + name
# 8b2f...  then  8b2f...  then  8b2f...   — always identical
```

UUIDv4 draws 122 random bits. UUIDv5 hashes the namespace and name with SHA-1 and formats the digest
as a UUID. Same inputs, same output, forever, on any machine.

The consequence for the pipeline is decisive. Qdrant's `upsert` is *upsert*: given a point ID that
already exists, it overwrites; given a new one, it inserts. With random IDs every ingestion run
generates fresh IDs, so every run inserts. Run ingestion three times and every passage exists three
times. Retrieval then burns three of its five context slots on the same text.

With deterministic IDs, three runs produce one copy. That property — idempotence — is what lets
Phase 2 crash at hour six and resume without deduplication logic, and what lets Phase 15 rebuild a
collection under a new embedding model while keeping IDs aligned between old and new.

### The limit of this, and why Phase 3 needs delete-before-upsert

Be precise about what deterministic IDs do and do not buy, because it is easy to over-trust them.

They guarantee that **re-ingesting identical content produces identical IDs**, so unchanged documents
update in place instead of duplicating. That is the whole win, and it is real.

They do **not** garbage-collect. `make_chunk_id` is a function of `(doc_id, section_id, chunk_index)`,
so anything that changes those inputs — or changes how many chunks a document produces — leaves the old
points stranded in the collection with no way to ever match them again:

| Change | What happens |
| :--- | :--- |
| You tune `MAX_TOKENS_PER_CHUNK` from 400 to 600 | The doc now yields 300 chunks instead of 400. Chunks 300–399 keep their old IDs and stay in Qdrant forever, holding stale text. |
| You improve the section parser | Section titles change → `section_id` changes → every chunk of that document gets a new ID, and the entire old set is orphaned. |
| A file moves from `2003/` to `2004/` | `doc_id` derives from the path, so it changes. The old document's chunks all persist as ghosts. |
| You pass an absolute path where you previously passed a relative one | Same problem — different `doc_id` for the same file. |

Those orphans are worse than plain duplicates, because they contain text that no longer reflects the
source and there is no ID collision to reveal them. They will be retrieved and cited.

So the correct ingestion contract is **delete-by-`doc_id`, then upsert**:

```python
# Phase 3's QdrantStore will implement this. Sketched here so the reasoning
# lives next to the ID scheme it compensates for.
await store.delete_by_doc_id(doc.doc_id)      # remove every existing chunk of this doc
await store.upsert_points(chunks, dense, sparse)
```

This makes re-ingesting a document a genuine *replace* rather than a merge, which is why `doc_id` gets
a payload index in Phase 3 — deleting by a filter on an unindexed field means a full collection scan.

The two mechanisms are complementary rather than redundant. Deterministic IDs make a *partial* re-run
safe, which is what checkpoint-resume in Phase 2 depends on: re-processing the last 500 documents after
a crash must not duplicate them. Delete-before-upsert makes a *changed* document safe. You need both.

The one rule about the namespace: **`NAMESPACE` must never change.** Change that constant and every ID
in the system changes, orphaning your entire index. It is effectively part of your data format, not a
tuneable.

### Failure Modes

**`ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'`** — real
documents occasionally contain that literal string. Hence `disallowed_special=()` on every `encode`
call, which tells `tiktoken` to treat such sequences as ordinary text. Omit it and one weird contract
kills the run.

**Token counts differ from what Groq bills.** Expected, and stated in the docstring. `tiktoken`
implements OpenAI's encodings and the served model may tokenize slightly differently. Counts land
within a few percent — fine for chunk sizing, useless for cost accounting. If you need real numbers,
read `usage.total_tokens` off the API response, which Phase 6 records.

**`KeyError: o200k_harmony`** — your installed `tiktoken` predates that encoding. The fallback in
`get_tokenizer` handles it silently; if you want to confirm which one you actually got, check
`get_tokenizer().name`.

**Duplicate points after switching from `uuid4` to `uuid5`.** Expected. The old random-ID points are
still in the collection and nothing will ever match their IDs again. Recreate the collection once
when you adopt Phase 3's store.

**`hash_file` is slow across the full corpus.** Hashing 37 GB reads 37 GB. Phase 2 caches hashes in
the checkpoint file so only new or modified files are re-hashed.

---

## Verification (deferred)

You are not running anything on this machine, so treat this as a script to execute later on the
machine with the corpus. Save it as `scripts/verify_phase1.py` when you get there.

```python
"""Phase 1 verification. Run from the project root: python scripts/verify_phase1.py"""

import asyncio

from config.settings import settings
from src.core.exceptions import RAGException, VectorStoreError
from src.core.logging import get_request_id, logger, new_request_id
from src.core.models import Chunk, Document, GradingReport, ScoredChunk
from src.core.telemetry import telemetry
from src.core.utils import count_tokens, hash_text, make_chunk_id, make_doc_id


def check_settings() -> None:
    logger.info(f"Loaded {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}, log level: {settings.LOG_LEVEL}")
    logger.info(f"Contracts dir resolves to: {settings.contracts_dir}")
    logger.info(f"Redis URL: {settings.redis_url}")
    for warning in settings.validate_runtime():
        logger.warning(warning)


def check_models() -> None:
    doc = Document(
        doc_id=make_doc_id("data/contracts/2003/example.txt"),
        source_path="data/contracts/2003/example.txt",
        file_name="example.txt",
        content="SECTION 1.01. Termination. Either party may terminate on 30 days notice.",
        content_hash=hash_text("x"),
        contract_name="Example Supply Agreement",
        year="2003",                     # a string — the validator coerces it
    )
    assert doc.year == 2003 and isinstance(doc.year, int), "year coercion failed"
    logger.info(f"Document OK — {doc.char_count} chars, year={doc.year}")

    chunk = Chunk(
        chunk_id=make_chunk_id(doc.doc_id, "sec-1", 3),
        doc_id=doc.doc_id,
        section_id="sec-1",
        text=doc.content,
        chunk_index=3,               # deliberately non-zero: catches a dropped field
        token_count=count_tokens(doc.content),
        contract_name=doc.contract_name,
        file_name=doc.file_name,
        section_title="SECTION 1.01",
        year=doc.year,
    )
    logger.info(f"Chunk OK — {chunk.token_count} tokens")

    # Determinism: the same inputs must produce the same ID, and different inputs
    # must not collide.
    assert chunk.chunk_id == make_chunk_id(doc.doc_id, "sec-1", 3), "IDs are not deterministic"
    assert make_chunk_id(doc.doc_id, "sec-1", 4) != chunk.chunk_id, "IDs collide across indexes"
    logger.info("Deterministic IDs confirmed")

    # Round-trip through the Qdrant payload representation. Compare EVERY field that
    # `from_payload` reads — an earlier version of this check only compared `.text`
    # and therefore missed `to_payload` silently dropping `chunk_index`.
    restored = Chunk.from_payload(chunk.to_payload())
    for field in (
        "chunk_id", "doc_id", "section_id", "text", "chunk_index",
        "token_count", "contract_name", "file_name", "section_title",
        "year", "chunk_level", "parent_id",
    ):
        assert getattr(restored, field) == getattr(chunk, field), (
            f"payload round-trip lost {field!r}: "
            f"{getattr(chunk, field)!r} -> {getattr(restored, field)!r}"
        )
    logger.info("Payload round-trip OK across all persisted fields")

    scored = ScoredChunk(chunk=chunk, score=0.87, rank=0)
    assert scored.effective_score == 0.87
    scored = scored.model_copy(update={"rerank_score": 0.95})
    assert scored.effective_score == 0.95, "effective_score ignored the rerank score"
    logger.info("ScoredChunk scoring precedence OK")

    assert GradingReport(is_grounded=True, is_relevant=True).passed
    assert not GradingReport(is_grounded=True, is_relevant=False).passed
    logger.info("GradingReport gating OK")


def check_validation_rejects_bad_data() -> None:
    from pydantic import ValidationError
    try:
        Chunk(chunk_id="x", doc_id="d", section_id="s", text="   ",
              chunk_index=0, token_count=0)
    except ValidationError:
        logger.info("Blank chunk text correctly rejected")
    else:
        raise AssertionError("blank chunk text was accepted — validator missing")


def check_exceptions() -> None:
    try:
        try:
            raise ConnectionRefusedError("qdrant unreachable")
        except ConnectionRefusedError as exc:
            raise VectorStoreError(
                "Hybrid query failed", details={"collection": "legal_contracts_v1"}
            ) from exc
    except RAGException as exc:
        assert exc.__cause__ is not None, "exception chaining lost the cause"
        logger.info(f"Exception OK — {exc}")
        logger.info(f"Serialised: {exc.to_dict()}")


@telemetry.span("verify.sync_span")
def a_sync_function() -> int:
    return sum(range(100_000))


@telemetry.span("verify.async_span", warn_over_ms=10)
async def an_async_function() -> str:
    await asyncio.sleep(0.05)
    return "done"


async def check_telemetry() -> None:
    a_sync_function()
    await an_async_function()          # must report ~50ms, NOT ~0ms
    with telemetry.measure("verify.block") as m:
        _ = [i**2 for i in range(50_000)]
    assert m["duration_ms"] > 0, "context manager recorded no duration"
    logger.info(f"Telemetry OK — block took {m['duration_ms']:.1f}ms")


async def main() -> None:
    rid = new_request_id()
    logger.info(f"Starting Phase 1 verification (request_id={rid})")
    assert get_request_id() == rid

    check_settings()
    check_models()
    check_validation_rejects_bad_data()
    check_exceptions()
    await check_telemetry()

    logger.info("PHASE 1 VERIFIED — foundations are sound.")


if __name__ == "__main__":
    asyncio.run(main())
```

### What to confirm when you do run it

The assertions cover most of it, but three results deserve your attention because they are the
substance of this phase rather than mechanics.

Set `LOG_LEVEL=DEBUG` and check that `verify.async_span` reports roughly **50ms, not 0ms**. A
near-zero reading means the sync/async detection in `telemetry.py` is wrong and every async timing in
the project will be meaningless.

Confirm `request_id` appears on every single log line. That is the correlation plumbing Phase 8
depends on.

Confirm the deterministic-ID assertion passes. It is three characters of difference from the
prototype and it is what makes ingestion idempotent.

---

## What Phase 1 Bought You

Concretely, going into Phase 2 you now have: a single validated source of configuration truth; a
vocabulary of typed domain objects (`Document → Section → Chunk`) that makes the ingestion pipeline's
signature self-explanatory; an error taxonomy that lets callers distinguish retry-worthy from fatal;
request-correlated structured logging; latency instrumentation that works on async code; and
idempotent ID generation.

The thing to carry forward is the pattern rather than the files. Every phase from here defines its
models first, declares its interface second, and implements last. That order is why the system stays
navigable at 10,000 lines.

**Next:** `Phase2_Ingestion_Engine.md` — Pipeline 1, at 650,000-document scale. Generators,
multiprocessing to escape the GIL, resumable checkpointing, and a dead-letter queue for the documents
that will inevitably fail to parse.
