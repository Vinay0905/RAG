# Master Roadmap — Production-Grade Self-Corrective RAG

**This is the top-level document. Read it once, fully, before typing a single line of code in any phase.**

Everything else in `docs/` is a leaf of this tree. This file tells you what we are building, why the
architecture looks the way it does, what order to build it in, and what conventions every file must
follow. If you ever open a phase guide and think "wait, why is this here?" — the answer is in this
document.

---

## 1. What We Are Building

A **self-corrective, agentic Retrieval-Augmented Generation system for commercial legal contracts**,
built on the EDGAR/SEC material-contracts corpus.

The one-sentence version: *a lawyer asks a question in plain English, and the system returns an
answer that is provably grounded in specific clauses of specific contracts, with citations, and
which the system itself has fact-checked before showing you.*

The words that matter in that sentence:

**"Provably grounded"** — the system does not merely *hope* the LLM didn't hallucinate. A separate
grader model reads the generated answer alongside the retrieved source text and returns a structured
verdict. If the verdict is negative, the system rewrites its own search query and tries again. This
is the *self-corrective* loop, and it is the single most important idea in the project.

**"Agentic"** — the query path is not a straight line. It is a state machine with branches and
cycles, expressed in LangGraph. A straight-line pipeline can only fail. A state machine can notice
it failed and do something about it.

**"Specific clauses"** — legal text has structure (`ARTICLE IV`, `SECTION 7.02`, `EXHIBIT B`). A
chunker that slices every 400 characters destroys that structure and will happily return you half of
an indemnity clause. We parse structure first, then chunk within it.

---

## 2. Why This Project Exists (and how to use these docs)

This is a **learning build**. The goal is not to ship a product as fast as possible — if it were,
the correct answer would be about 300 lines using LangChain's off-the-shelf components. The goal is
to understand every layer well enough to debug it, replace it, or defend it in an interview.

So the rule is: **you type every line yourself.** Keep the phase guide open on one side of the
screen and your editor on the other. Do not copy-paste. Typing is slow, and that slowness is the
entire point — it forces you to read each line as it goes by.

### The format of every phase guide

Each file you build is presented in the same six-part structure. When you see these headings,
here is what they mean and how to read them:

| Section | What it does | Can you skip it? |
| :--- | :--- | :--- |
| **The Problem** | The concrete failure this file prevents | No — this is the "why" |
| **Design Decision** | The 2–3 options considered, and why we chose one | No — this is the interview answer |
| **The Theory** | Math or CS concepts involved (only when there are any) | Skip on first pass, return before the checkpoint |
| **The Code** | Typed chunks with inline commentary | Type it |
| **Failure Modes** | What breaks, the exact error text, and the fix | Read it *before* you run the code |
| **Checkpoint** | A runnable command that proves the file works | No — never move on without a green checkpoint |

### The rule about checkpoints

Each phase ends with a verification script. In a layered system a broken foundation does not announce
itself — it produces a confusing error four phases later, in a file that is completely correct. So the
ideal is: **do not start phase N+1 until phase N's verification passes.**

That ideal assumes you can run the code. This project is being written on a machine without the
corpus and without the Docker services, so verification is **deferred**: each phase's script is
written out precisely and saved under `scripts/`, to be run later on the machine that has the data.
The phase guides label these sections *Verification (deferred)* rather than *Checkpoint* to make the
distinction explicit.

The practical consequence is that you will type several phases before running any of them, and errors
will therefore surface in batches rather than one at a time. Two habits make that survivable. Run the
verification scripts in phase order when you do get to a capable machine, since a Phase 1 failure will
manifest as nonsense in Phase 4. And read each *Failure Modes* section **before** typing the file
rather than after, because it is the only early-warning system available to you in this mode.

---

## 3. The Architecture: Two Pipelines, One Store

Almost every production RAG system in the world reduces to the same shape. Two independent
pipelines that never call each other, connected only through the vector database.

```text
        PIPELINE 1 — INGESTION (offline, batch, slow)
        ═══════════════════════════════════════════════
        Raw files  ──► Load ──► Parse ──► Chunk ──► Embed ──┐
        (650k .txt)                                          │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │     QDRANT      │
                                                    │  dense + sparse │
                                                    │   + payloads    │
                                                    └─────────────────┘
                                                             ▲
        PIPELINE 2 — RETRIEVAL (online, per-request, fast)   │
        ═══════════════════════════════════════════════      │
        Query ──► Expand ──► Retrieve ───────────────────────┘
                     ▲            │
                     │            ▼
                     │         Rerank
                     │            │
                     │            ▼
                     │        Generate
                     │            │
                     │            ▼
                     └──── Grade ─┴──► Answer + citations
                        (fail: rewrite & loop)
```

### Why the separation matters

These two pipelines have **opposite performance characteristics**, and that is the entire
justification for keeping them apart.

Ingestion is throughput-bound. It runs for hours, processes gigabytes, is CPU-heavy (embedding
650,000 documents), and nobody is waiting on it. If it crashes at hour six, the correct response is
to resume from a checkpoint — not to start over.

Retrieval is latency-bound. It runs in under two seconds, processes a few kilobytes, is I/O-heavy
(network calls to Qdrant and Groq), and a human is staring at a spinner. If it fails, the correct
response is to degrade gracefully and return *something*.

Code optimised for one is actively wrong for the other. Ingestion wants multiprocessing and large
batches; retrieval wants async I/O and tight timeouts. This is also why **Phase 2 is partly
synchronous while everything else is async** — see §7.

The vector store is the only contract between them. That is a feature: you can rebuild the entire
retrieval side without re-ingesting a single document.

---

## 4. Complete Target Directory Structure

This is the end state after Phase 16. You will not create all of this at once — each phase creates
its own slice — but keep this map in mind so you always know where a new file belongs.

```text
RAG/
├── config/
│   ├── __init__.py
│   └── settings.py                       # Pydantic Settings, single source of config truth
│
├── data/                                 # gitignored — never commit the corpus
│   ├── contracts/                        # EDGAR corpus, partitioned by year: 2000/ 2001/ ...
│   ├── checkpoints/                      # ingestion resume state (Phase 2)
│   └── dead_letter/                      # files that failed to parse (Phase 2)
│
├── docs/                                 # ← you are here
├── notebooks/                            # exploration only, never imported by src/
├── scripts/
│   ├── download_corpus.py                # Phase 2 — fetch EDGAR data
│   └── benchmark_retrieval.py            # Phase 7 — latency + recall harness
│
├── src/
│   ├── core/                             # ── PHASE 1 ── depends on nothing
│   │   ├── exceptions.py                 # RAGException hierarchy
│   │   ├── interfaces.py                 # ABCs — the plug sockets
│   │   ├── models.py                     # Pydantic domain models (Document, Chunk, ...)
│   │   ├── logging.py                    # structured JSON logs + correlation IDs
│   │   ├── telemetry.py                  # async-aware span tracing
│   │   └── utils.py                      # token counting, hashing, id generation
│   │
│   ├── ingestion/                        # ── PHASE 2 ── PIPELINE 1
│   │   ├── loaders/                      # bytes on disk  →  Document
│   │   │   ├── base.py  txt_loader.py  pdf_loader.py  docx_loader.py  factory.py
│   │   ├── parsers/                      # Document  →  list[Section]
│   │   │   ├── base.py  legal_parser.py  metadata_extractor.py
│   │   ├── chunkers/                     # Section  →  list[Chunk]
│   │   │   ├── base.py  sentence_chunker.py  parent_child_chunker.py  semantic_chunker.py
│   │   ├── checkpoint.py                 # resumable ingestion state
│   │   ├── dead_letter.py                # quarantine for unparseable files
│   │   └── pipeline.py                   # the orchestrator
│   │
│   ├── embeddings/                       # ── PHASE 3 ── text → vectors
│   │   └── base.py  fastembed_provider.py  openai_provider.py  factory.py
│   │
│   ├── vectorstores/                     # ── PHASE 3 ── vectors → database
│   │   └── base.py  qdrant_store.py  chroma_store.py  factory.py
│   │
│   ├── retrieval/                        # ── PHASE 4 ── PIPELINE 2, search half
│   │   ├── multi_query.py                # 1 query → N query variations
│   │   ├── hyde.py                       # hypothetical document embeddings
│   │   ├── fusion.py                     # Reciprocal Rank Fusion
│   │   ├── hybrid.py                     # dense + sparse, server-side
│   │   ├── rerankers/
│   │   │   └── base.py  cross_encoder.py  mmr.py
│   │   └── pipeline.py
│   │
│   ├── graph/                            # ── PHASE 5 ── PIPELINE 2, agent half
│   │   ├── state.py                      # the TypedDict that flows through the machine
│   │   ├── nodes/                        # one file per node — each is a pure function
│   │   │   ├── router_node.py  expansion_node.py  retrieval_node.py
│   │   │   ├── reranking_node.py  generation_node.py  grading_node.py
│   │   │   ├── rewriting_node.py
│   │   │   ├── mapreduce_node.py         # (Phase 12)
│   │   │   └── graph_traversal_node.py   # (Phase 13)
│   │   ├── prompts.py                    # every prompt string, versioned, in one place
│   │   ├── edges.py                      # conditional routing functions
│   │   └── builder.py                    # wires nodes + edges into a compiled graph
│   │
│   ├── llm/                              # ── PHASE 6 ──
│   │   └── base.py  groq_provider.py  openai_provider.py  factory.py
│   ├── cache/                            # ── PHASE 6 ──
│   │   └── redis_cache.py  semantic_cache.py
│   ├── guardrails/                       # ── PHASE 6 ──
│   │   └── pii_masker.py  prompt_injection.py  citation_validator.py
│   │
│   ├── evaluation/                       # ── PHASE 7 ──
│   │   ├── metrics/  groundedness.py  relevance.py  context_recall.py
│   │   ├── evaluator.py  synthetic_generator.py  regression.py
│   │
│   ├── api/                              # ── PHASE 8 ──
│   │   ├── schemas.py  dependencies.py  websocket.py  app.py
│   │   └── routes/  health.py  ingestion.py  query.py  admin.py
│   │
│   ├── cli/                              # ── PHASE 9 ──
│   │   └── ui.py  main.py
│   ├── web/                              # ── PHASE 9 ──
│   │   └── index.html  styles.css  app.js
│   │
│   ├── security/                         # ── PHASE 11 ── multi-tenant ACL
│   │   └── principal.py  acl.py  filters.py  middleware.py
│   ├── knowledge_graph/                  # ── PHASE 13 ── GraphRAG
│   │   └── extractor.py  neo4j_store.py  traversal.py
│   ├── indexing/                         # ── PHASE 15 ── embedding migration
│   │   └── shadow.py  migration.py  alias_manager.py
│   └── summarization/                    # ── PHASE 16 ── RAPTOR
│       └── tree_builder.py  raptor.py
│
└── tests/                                # ── PHASE 10 ── mirrors src/ exactly
    ├── conftest.py
    ├── unit/  integration/  e2e/
```

### The one structural rule

**Dependencies point inward and downward, never upward.**

`src/core/` imports nothing from the rest of `src/`. `src/ingestion/` may import `core`. `src/graph/`
may import `core`, `retrieval`, `llm`. Nothing ever imports from `src/api/` or `src/cli/` — those are
the outermost shell.

If you ever find yourself needing an upward import, you have put a piece of logic in the wrong
layer. Move the logic, do not add the import. This one rule is what keeps a 10,000-line codebase
navigable.

---

## 5. Phase Plan and Line Budget

Two parts. **Part I** (Phases 1–10) builds a complete, production-quality baseline system.
**Part II** (Phases 11–16) solves the six hard industrial problems catalogued in
`future_ideas_problems.md` — this is the part that makes the project unusual.

The line estimates are for Python you type, excluding blank lines and the docs themselves.

### Part I — The Baseline System

| # | Phase | What you learn | Est. LOC | Depends on |
| :-- | :--- | :--- | ---: | :--- |
| 1 | System Foundations | Pydantic models, ABCs, exception design, structured logging | 1,000 | — |
| 2 | Ingestion at Scale | Generators, multiprocessing, checkpointing, dead-letter queues | 1,000 | 1 |
| 3 | Embeddings & Vector Stores | Dense vs sparse vectors, adapter + factory patterns, Qdrant | 800 | 1 |
| 4 | Hybrid Retrieval & Reranking | BM25, RRF, HyDE, cross-encoders, MMR diversity | 700 | 1, 3 |
| 5 | LangGraph Agent | State machines, conditional edges, cycles, self-correction | 900 | 1, 4, 6 |
| 6 | LLM, Cache & Guardrails | Provider abstraction, semantic caching, PII, prompt injection | 700 | 1 |
| 7 | LLMOps & Evaluation | RAG triad metrics, synthetic QA generation, regression gates | 600 | 1, 5 |
| 8 | FastAPI Async Server | Dependency injection, SSE streaming, WebSockets, lifespan | 600 | 1, 5 |
| 9 | CLI & Web Dashboard | Typer, Rich, vanilla JS streaming client | 650 | 8 |
| 10 | Testing & Verification | Fixtures, mocking, async tests, integration + e2e layers | 1,100 | all |
| | **Part I subtotal** | | **8,050** | |

### Part II — The Hard Problems

Each of these maps directly to a numbered failure in `future_ideas_problems.md`.

| # | Phase | Problem it solves | Est. LOC | Depends on |
| :-- | :--- | :--- | ---: | :--- |
| 11 | Multi-Tenant Security & RBAC | #1 Permission leakage across tenants | 600 | 3, 8 |
| 12 | Map-Reduce Aggregation | #2 "Scan all 200 contracts and list…" | 550 | 5 |
| 13 | GraphRAG Multi-Hop | #3 Liability inherited across a 3-hop chain | 850 | 5, 6 |
| 14 | Layout-Aware PDF & Vision OCR | #4 Multi-column, scanned, and tabular PDFs | 700 | 2 |
| 15 | Embedding Drift & Shadow Indexing | #5 Migrating models without downtime | 550 | 3 |
| 16 | RAPTOR Summary Trees | #6 Whole-document summarisation | 650 | 2, 3 |
| | **Part II subtotal** | | **3,900** | |
| | **PROJECT TOTAL** | | **≈ 11,950** | |

The budget overshoots 10,000 deliberately. Estimates always run optimistic, and you should have
room to cut Phase 9's web UI or trim Chroma support without falling short of the target.

**A calibration note, recorded after Phase 1 was written.** Phase 1 was originally budgeted at 550
lines and came in at roughly 1,000 — the table above has been corrected. The cause is that
production-density code is dominated not by logic but by validators, docstrings, and type
annotations, which estimates habitually omit. Expect the remaining phases to run **1.5× to 2× their
stated numbers** for the same reason, putting the realistic project total somewhere around 13,000 to
15,000 lines. That is comfortably past the 10,000 target, which means **the goal is now scope
control, not scope growth.** If a phase starts ballooning well past its budget, that is a signal to
cut features rather than a licence to keep going.

### The critical path

You cannot reorder these freely. The hard dependency chain is:

```text
1 ──► 3 ──► 4 ──┐
│               ├──► 5 ──► 8 ──► 9
├──► 6 ─────────┘         │
├──► 2                    └──► 11
└──► 10 (write tests continuously, not at the end)
```

Phases 11–16 are genuinely independent of each other. Once Part I is done, build them in whatever
order interests you most. If you want the single most impressive one for a portfolio, it is
**Phase 13 (GraphRAG)**. If you want the one most likely to be asked about in an enterprise
interview, it is **Phase 11 (multi-tenant ACL)**.

---

## 6. Technology Choices and Why

Every one of these is a decision you should be able to defend. Memorising the stack is useless;
knowing the tradeoff is the point.

**Qdrant** as the vector store, over Pinecone, Weaviate, or pgvector. It runs locally in Docker with
zero cost, and critically it supports **named vectors** — a single point can hold both a dense
embedding and a sparse BM25 vector, fused server-side in one round trip. Pinecone is managed-only
and metered. pgvector has no native sparse support. Phase 3 covers this properly.

**FastEmbed** for embeddings, over `sentence-transformers` directly. FastEmbed runs ONNX-quantised
models on CPU, roughly 3× faster than PyTorch for inference, with no CUDA dependency. For embedding
650,000 documents on a laptop, that difference is measured in days.

**`BAAI/bge-small-en-v1.5`** as the dense model. 384 dimensions, 133 MB. Its bigger sibling
`bge-large` scores about 3% better on MTEB but is 1.3 GB and 4× slower. At our corpus size the
storage and time cost dominates the accuracy gain — and Phase 15 exists precisely so you can change
your mind about this later without a rebuild.

**Groq** for LLM inference, over OpenAI or Anthropic. Roughly 10× faster tokens-per-second on
open-weight models, with a usable free tier. Speed matters disproportionately here because the
self-corrective loop can invoke the LLM five or more times for a single user question. Phase 6
abstracts this behind an interface so swapping to OpenAI is a one-line config change.

**LangGraph**, not plain LangChain chains. A LangChain chain is a directed *acyclic* graph — it
physically cannot loop back. Our self-correction requires a cycle: grade → fail → rewrite → retrieve
again. LangGraph gives us cycles, per-node state updates, and checkpointing. This is the whole
reason the project uses it.

**Pydantic v2 everywhere**, not raw dicts. This is the biggest single upgrade over the prototype in
`Pipelines/`, and §7 explains why it earns its keep.

**Redis** for semantic caching. Not just key-value caching of exact strings — Phase 6 builds a
*vector* cache, where a query 92% cosine-similar to a previous one returns the cached answer.

---

## 7. Coding Conventions (non-negotiable)

These apply to every file in every phase. Read them now; the phase guides assume them.

### 7.1 Pydantic models at every boundary — never bare dicts

The prototype in `Pipelines/` passes dictionaries everywhere: `chunk["chunk_text"]`,
`payload["contract_name"]`. This works right up until you typo a key, at which point you get a
`KeyError` at runtime, three layers deep, with no indication of which layer produced the malformed
dict.

We define real models in `src/core/models.py` and pass those instead:

```python
# ❌ The prototype's way — nothing is checked, nothing is discoverable
chunk = {"chunk_id": "...", "chunk_text": "...", "section": "..."}
print(chunk["chunk_txt"])          # KeyError at runtime, in production

# ✅ Our way — validated at construction, autocompleted in the editor
chunk = Chunk(
    chunk_id=make_chunk_id(doc_id, section_id, 0),
    doc_id=doc_id,
    section_id=section_id,
    text="Either party may terminate on 30 days notice.",
    chunk_index=0,
    token_count=count_tokens(text),
    section_title="SECTION 1.01",
)
print(chunk.txt)                   # caught by the type checker, before you run
```

The exact field list is defined in Phase 1's `src/core/models.py` — that file is the authority, and
this snippet must match it.

The payoff is threefold: errors surface at the boundary where bad data enters rather than deep
inside consuming code; your editor can autocomplete every field; and FastAPI in Phase 8 generates
its entire OpenAPI schema from these same models for free.

Rule: **any data structure that crosses a module boundary is a Pydantic model or a TypedDict.**
Dicts are permitted only for genuinely dynamic key-value data.

### 7.2 Async-first, with a deliberate exception

Every I/O-bound function is `async def`. That means all vector store calls, all LLM calls, all cache
calls, and every graph node.

The complication is **CPU-bound work**, which gains nothing from `async` and will block the event
loop if you call it directly inside a coroutine. Cross-encoder reranking and local embedding
generation are both CPU-bound.

The resolution is important, and it is a rule about *where* the thread offload lives rather than
about which signatures are async. **All interfaces stay async. CPU-bound implementations offload
internally.**

```python
# ❌ Interface is sync, so a network-backed implementation cannot exist without blocking
def rerank(self, query: str, docs: list[ScoredChunk]) -> list[ScoredChunk]:
    return self.model.predict(pairs)

# ✅ Interface is async. A local model offloads to a thread; a hosted
#    reranker would simply await its HTTP call. Both satisfy the same contract.
async def rerank(self, query: str, docs: list[ScoredChunk],
                 top_k: int = 5) -> list[ScoredChunk]:
    scores = await asyncio.to_thread(self.model.predict, pairs)
    ...
```

Why this matters concretely: `BaseEmbeddingProvider` has two planned implementations —
FastEmbed, which is local CPU work, and OpenAI, which is a network call. A synchronous interface
would force the OpenAI provider to block the event loop on every request. So the interface is async
and FastEmbed wraps its own `to_thread`. The caller never needs to know which kind it holds. That is
the entire point of the abstraction.

Phase 2's ingestion pipeline is the one place we use `multiprocessing` rather than `asyncio`,
because embedding 650,000 documents is pure CPU work and needs to escape the GIL entirely. §3
explains why.

### 7.3 Full type hints, modern syntax

Every function signature is annotated. We target Python 3.10+, so use built-in generics and the
union operator — not the legacy `typing` equivalents:

```python
# ❌ Legacy
from typing import List, Dict, Optional
def f(x: List[str]) -> Optional[Dict[str, int]]: ...

# ✅ Modern
def f(x: list[str]) -> dict[str, int] | None: ...
```

### 7.4 Fail loudly at the boundary, degrade gracefully at the top

Low-level code raises specific exceptions from the `RAGException` hierarchy. It never swallows an
error and returns `None`. Only the outermost layer — an API route or a CLI command — catches broadly
and converts to a user-facing message.

```python
# ❌ The error vanishes; the caller gets an empty list and no idea why
try:
    return await self.client.query_points(...)
except Exception:
    return []

# ✅ Context is preserved and the failure is attributable
try:
    return await self.client.query_points(...)
except Exception as exc:
    raise VectorStoreError(
        f"Hybrid query failed on '{self.collection}'",
        details={"collection": self.collection, "limit": limit},
    ) from exc
```

The `from exc` matters — it chains the original traceback instead of hiding it.

### 7.5 Never log secrets, never log document bodies

Log identifiers, counts, and durations. Not API keys, not chunk text, not user queries in plaintext
if the deployment is multi-tenant. `chunk_id=abc123 tokens=412` is useful; dumping 400 tokens of a
confidential contract into stdout is a compliance incident.

### 7.6 Naming

Modules and functions are `snake_case`. Classes are `PascalCase`. Constants are `UPPER_SNAKE`.
Anything private to a module gets a leading underscore. Abstract base classes are prefixed `Base`
(`BaseVectorStore`); concrete implementations are named for their technology (`QdrantStore`).

---

## 8. Known Corrections to the Older Docs

The four numbered guides (`01_ingestion_chunking_guide.md` through
`04_pipeline_production_guide.md`) and the original Phase 1–10 drafts were written against an
earlier prototype. They remain in `docs/` for reference, but where they conflict with a rewritten
phase guide, **the phase guide wins**. Specifically, these are now wrong:

| Old | Current | Why it changed |
| :--- | :--- | :--- |
| `client.recreate_collection(...)` | `collection_exists()` + `create_collection()` | Deprecated; also destroys data silently |
| `client.search(...)` | `client.query_points(...)` | Deprecated; `query_points` is the only path to server-side hybrid fusion |
| Python-side dense/sparse merge | `prefetch=[...]` with `FusionQuery` | Qdrant fuses natively — one round trip instead of two |
| `llama-3.1-70b-versatile` | current Groq production model | Decommissioned by Groq |
| `datetime.utcnow()` | `datetime.now(timezone.utc)` | Deprecated in Python 3.12+ |
| `[tool.ruff] select` | `[tool.ruff.lint] select` | Ruff config schema moved |
| `StateGraph(dict)` | `StateGraph(AgentState)` (TypedDict) | Untyped state defeats the purpose |
| Sync `QdrantClient` | `AsyncQdrantClient` | See §7.2 |

The working prototype in `Pipelines/` also has two live bugs, kept here as a record: `main.py`
imports `IngestionPipeline` but the class is named `IngestPipeline`, and `config.py` has
`http://locolhost:6333`. Both are superseded by Phase 3.

---

## 9. Progress Tracker

Tick a phase only when its checkpoint command runs green.

```text
PART I — BASELINE
[ ]  Phase  1  System Foundations
[ ]  Phase  2  Ingestion at Scale
[ ]  Phase  3  Embeddings & Vector Stores
[ ]  Phase  4  Hybrid Retrieval & Reranking
[ ]  Phase  5  LangGraph Self-Corrective Agent
[ ]  Phase  6  LLM Providers, Cache & Guardrails
[ ]  Phase  7  LLMOps & Evaluation
[ ]  Phase  8  FastAPI Async Server
[ ]  Phase  9  CLI & Web Dashboard
[ ]  Phase 10  Testing & Verification

PART II — HARD PROBLEMS
[ ]  Phase 11  Multi-Tenant Security & RBAC
[ ]  Phase 12  Map-Reduce Aggregation
[ ]  Phase 13  GraphRAG Multi-Hop Traversal
[ ]  Phase 14  Layout-Aware PDF & Vision OCR
[ ]  Phase 15  Embedding Drift & Shadow Indexing
[ ]  Phase 16  RAPTOR Hierarchical Summary Trees
```

---

## 10. Before You Start Phase 1

Three things must be true:

**Python 3.10 or newer.** Run `python --version`. The codebase uses `X | Y` union syntax and
built-in generics throughout, which do not parse on 3.9.

**Docker Desktop running.** Phase 1 brings up Qdrant and Redis containers. Confirm with
`docker --version`.

**A Groq API key.** Free at `console.groq.com`. You do not need it until Phase 5, but Phase 1 sets
up the config slot for it.

You do **not** need the EDGAR corpus yet. Phase 2 includes a `scripts/download_corpus.py` step and
works against a small sample directory so you can build and test the ingestion pipeline before
committing 37 GB of disk.

Go to `Phase1_System_Foundations.md`.
