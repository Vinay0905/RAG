# SESSION HANDOFF — read this first, in full, before doing anything

> **You are continuing a multi-session project. You have no memory of the earlier sessions.**
> This file is the complete transfer. It is written to be self-sufficient: read it top to bottom and
> you will know the arrangement, the decisions, what exists, what is next, and the exact contracts
> your next document must honour.
>
> Internal working file — not one of the study guides. Keep it updated at the end of every session.
> Last updated: end of session 7 (2026-08-02).

---

## 1. THE ARRANGEMENT — the most important section

The user is **Vinay**. He is building a production-grade RAG system to *understand* it, not to ship
it fast, and to have something defensible in an interview.

**Your role is TEACHER, not coding agent.**

- You write **`.md` files inside `docs/` ONLY**.
- **Never** create or edit files in `src/`, `Pipelines/`, `config/`, `tests/`, `scripts/`, or any
  root config file. Not `pyproject.toml`, not `.gitignore`, not `.env`. If a root file needs
  changing, **write the instruction into a doc** and tell him.
- He types every line of Python himself, with the guide open beside his editor. The docs are the
  product. The code is his.

**Nothing is executed on this machine.** No tests, no runs, no Docker. He confirmed this explicitly.
Consequences:

- Every phase ends with a **"Verification (deferred)"** section — a precise script to save under
  `scripts/` and run later on the machine that has the corpus. Never frame it as a blocking gate.
- **You cannot verify any library API by running it.** So verify externally (web search) before
  writing any phase that leans on a fast-moving library. Already done for Groq and Qdrant (§7).
  **Do this for LangGraph before Phase 5 and FastAPI/SSE before Phase 8.**

**Goal:** ~10,000 lines of Python. Realistically landing 13,000–15,000 (§4). Learning priorities he
named: **RAG, LangGraph, Python, Pydantic.**

---

## 2. LOCKED DECISIONS

| Question | Decision |
| :--- | :--- |
| The 10 old phase drafts | **Full rewrite** at production density. The old drafts pass bare `dict`s everywhere, so Pydantic never actually gets taught. Phases 11–16 build on 3/4/5, so a weak foundation forces a later rewrite. |
| Teaching depth | **Full six-part format per file:** The Problem → Design Decision → The Theory → The Code → Failure Modes → Verification. This is non-negotiable; it is what he asked for. |
| Dataset | **EDGAR/SEC contracts. ~650,833 `.txt` files, ~37 GB, partitioned by year (2000–2020).** NOT CUAD (CUAD is 510 files — the PRD is wrong about this and he confirmed EDGAR). **The corpus is on a different laptop and is not present here.** |
| Code style | **Async-first.** All interfaces async. CPU-bound implementations offload internally via `asyncio.to_thread`. One sanctioned exception: Phase 2's ingestion is sync + `multiprocessing` to escape the GIL. |
| Delivery | Roadmap first, then **2–3 phases per batch**. He codes later, not in real time — keep shipping at a steady clip. |
| Parallel subagents | He asked about 10 simultaneous agents. Agreed plan: Phase 1 solo (it freezes the shared signatures), then dependency-safe batches, then a wide **6-way fan-out for Phases 11–16** (they are mutually independent by design). Rationale he accepted: independent agents invent incompatible signatures, and doc-generation speed is not his bottleneck since he hand-types everything. **If you fan out, brief each subagent with §6 verbatim.** |

### Are we following HIS ideas?
He asked this directly and the answer is yes — say so if asked again. Phases 1–10 are his phases,
rewritten deeper. The dual-pipeline split is his (`Dual_Pipeline_Architecture.md`). Phases 11–16 map
**one-to-one, in his order**, onto the six problems in `future_ideas_problems.md`. Nothing of his was
dropped. Additions made: `models.py` + `utils.py` (Phase 1), and the scale machinery in Phase 2 that
the EDGAR corpus size forces.

---

## 3. STATUS

### Written and current
| File | Lines | State |
| :--- | ---: | :--- |
| `docs/00_MASTER_ROADMAP.md` | ~540 | ✅ current (corrected in session 2) |
| `docs/Phase1_System_Foundations.md` | ~2,680 | ✅ current (interface additions s4–s5) |
| `docs/Phase2_Ingestion_Engine.md` | ~2,950 | ✅ current (rewritten session 2) |
| `docs/Phase3_VectorStores_Embeddings.md` | ~2,050 | ✅ current (rewritten s3; **16-finding correction pass s5**) |
| `docs/Phase4_Hybrid_Retrieval_Reranking.md` | ~2,200 | ✅ current (rewritten s4; **20-finding correction pass s6**) |
| `docs/Phase5_LangGraph_Agent.md` | ~2,350 | ✅ current (rewritten s5; **22-finding correction pass s6**) |
| `docs/Phase6_LLM_Cache_Guardrails.md` | ~2,400 | ✅ current (rewritten s6; **31-finding correction pass s7**) |
| `docs/Phase7_LLMOps_Evaluation.md` | ~1,300 | ✅ current (rewritten s7; **39-finding correction pass s8**) |
| `docs/Phase8_FastAPI_Server.md` | ~1,150 | ✅ current (rewritten s8, **on budget**) |
| `docs/Phase9_CLI_Web_UI.md` | ~1,200 | ✅ current (rewritten s8, **on budget**) |
| `docs/Phase10_Testing_Verification.md` | ~1,500 | ✅ current (rewritten s9, **on budget**) |

| `docs/Phase11_MultiTenant_Security.md` | ~1,050 | ✅ current (written s9, **on budget**) |

### ✅ PART I COMPLETE. Part II: 11 done; **12, 13, 14, 15, 16 do not exist yet.**

### External review files at repo root
`IssuesinPhase3.md` (16), `IssuesinPhase4.md` (20), `Issuesinphase5.md` (22),
`Issuesinphase6.md` (31), `issuesinphase7.md` (39) — **all addressed** (s5–s8; see §8). Leave the files; they are the record.
**If a review file appears for a later phase, do the correction pass BEFORE writing new phases** — the
Phase 4 sparse-query bug had already propagated from Phase 3 by the time it was caught, and the Phase
5 review found a Phase 6 interface mismatch that would have wasted a whole phase.

### Reference only — superseded, do not delete, do not follow
`docs/01_ingestion_chunking_guide.md` … `docs/04_pipeline_production_guide.md`, `docs/answers.md`,
`docs/Dual_Pipeline_Architecture.md`, `docs/product_requirements_document.md`. All prototype-era.
Where they conflict with a rewritten phase, **the phase wins**.

### ▶ IMMEDIATE NEXT ACTION — the agreed staging for Part II
He asked (session 9) whether 11–16 could all be written in one parallel run. **Agreed plan: staged,
not a six-way fan-out.** Reasoning he accepted: 45% of this project's 148 review findings were
interface drift produced *sequentially* by one writer with full context; six blind parallel writers
is the same failure with the rails off. And his bottleneck is typing ~4,000 lines, not doc generation.

**Round 1 — DONE:** Phase 11 solo, because it *changes* shared surfaces rather than extending them.

**Round 2 — NEXT: Phases 14 and 15 IN PARALLEL.** These two are genuinely isolated:
- **Phase 14** (budget 700) — layout-aware PDF + vision OCR. Touches `src/ingestion/loaders/` ONLY.
  Adds `pdf_loader.py` / `docx_loader.py` alongside Phase 2's `txt_loader.py`, behind the existing
  `BaseDocumentLoader` (`supports(path)`, `load(path) -> Document`) and `LoaderFactory`. Must produce
  the same `Document` shape; must not touch chunking, retrieval, or the graph.
- **Phase 15** (budget 550) — embedding drift + shadow indexing. Touches `src/vectorstores/` and
  `src/indexing/` ONLY. Uses `delete_by_doc_ids`, `fetch_by_ids`, `count_exact`, `set_bulk_mode`,
  and Phase 3's `IndexingPipeline`. Qdrant **collection aliases** are the mechanism. Must not touch
  the graph or the API beyond a stats field.
- Brief each with §6 verbatim + the Part I surface boxes. **Both need `allowed_roles` preserved**
  (Phase 11) — a loader that drops it or a shadow index that omits the payload index re-opens the ACL.

**Round 3 — Phases 12, 13, 16.** All three add nodes to Phase 5's graph (map-reduce, graph traversal,
RAPTOR summary retrieval), so they share surface and should be written with one consistent view of
it. Decide sequential-vs-parallel after seeing how much Part I moved in rounds 1–2.

Every remaining phase ships a **§2 What Was Cut**. That is the mechanism that held 7–11 to budget.

---

## 4. LINE BUDGET

| | Phase | Est. LOC |
| :-- | :--- | ---: |
| 1 | System Foundations | 1,000 ✅ |
| 2 | Ingestion at Scale | 1,600 ✅ |
| 3 | Embeddings & Vector Stores | 1,450 ✅ |
| 4 | Hybrid Retrieval & Reranking | 1,200 ✅ |
| 5 | LangGraph Agent | 1,250 ✅ |
| 6 | LLM, Cache & Guardrails | 1,350 ✅ |
| 7 | LLMOps & Evaluation | 760 ✅ |
| 8 | FastAPI Async Server | 600 ✅ |
| 9 | CLI & Web Dashboard | 640 ✅ |
| 10 | Testing & Verification | 1,110 ✅ |
| 11 | Multi-Tenant Security & RBAC | 610 ✅ |
| 12 | Map-Reduce Aggregation | 550 |
| 13 | GraphRAG Multi-Hop | 850 |
| 14 | Layout-Aware PDF & Vision OCR | 700 |
| 15 | Embedding Drift & Shadow Indexing | 550 |
| 16 | RAPTOR Summary Trees | 650 |

**Calibration:** **Part I is done at ~10,960 lines.** Remaining budgeted work (11–16) is **3,900**,
landing at **~14,860 if the rest hit budget** — inside the realistic 13,000–15,000 band, and the
10,000-line goal is already met by Part I alone. Phases 7–10 all came in at or near budget after the
rule below was adopted; **it works, keep applying it.**

> ### ⇒ THE RULE FROM PHASE 7 ONWARD: HIT THE BUDGET, DO NOT EXCEED IT.
> The 10,000-line target passed long ago; overrun is now the only risk. **Phase 7 came in at 620
> against 620 and is the model to follow.** Its §2 is a written list of what was cut and why (RAGAS,
> a dashboard, three extra LLM judges, experiment tracking — roughly 700 lines) and that section is
> the most valuable page in the phase. **Every remaining phase should have one.** When a phase starts
> to balloon, cut features; do not continue and apologise in the handoff.

Phases 11–16 map to `future_ideas_problems.md` #1–#6 in his exact order: ACL, map-reduce, GraphRAG,
PDF layout, embedding drift, RAPTOR.

---

## 5. DEPENDENCY ORDER

```text
1 ──► 3 ──► 4 ──┐
│               ├──► 5 ──► 8 ──► 9
├──► 6 ─────────┘         │
├──► 2 ──► (3 consumes)   └──► 11
└──► 10 (continuous)
```
Phases 11–16 are mutually independent — safe to parallelise once Part I is done.

---

## 6. ⚠️ FROZEN CONTRACT — every later phase and subagent must match EXACTLY

Phases 1 and 2 are written, so these names are fixed. Any doc that contradicts them produces code
that will not run. **Brief subagents with this section verbatim.**

### `src/core/models.py`
`DocumentType`, `ChunkLevel`, `RetrievalMethod`, `Document`, `Section`, `Chunk`, `ScoredChunk`,
`RetrievalResult`, `Citation`, `GradingReport`, `RAGAnswer`.

- `ChunkLevel` = `STANDALONE | CHILD | PARENT | SUMMARY` (SUMMARY reserved for Phase 16 RAPTOR).
- `Document` is **frozen** (shallow — `metadata` dict is still mutable). `.year` is `int | None`.
  Has `.char_count`.
- `Section`: `section_id`, `doc_id`, `title`, `text`, `order`, `.char_count`.
- `Chunk`: `chunk_id`, `doc_id`, `section_id`, `text`, `chunk_index`, `token_count`,
  `contract_name`, `file_name`, `section_title`, `year`, **`chunk_level`**, `parent_id`,
  `created_at`. Methods `.to_payload()` / `.from_payload()` — **only these touch Qdrant payload
  dicts**, and they must stay exact inverses (a bug here was already caught once).
- `ScoredChunk` uses **COMPOSITION, not inheritance**: `.chunk`, `.score`, `.rank`, `.method`,
  `.rerank_score`, computed `.effective_score`. So it is **`scored.chunk.text`, never
  `scored.text`**. Rationale: a score is a property of a chunk *relative to one query*, not of the
  chunk itself.
- `GradingReport`: `is_grounded`, `is_relevant`, `confidence`, `reasoning`, `unsupported_claims`,
  computed **`.passed`** (= grounded AND relevant) — this drives Phase 5's retry edge.
- `RetrievalResult`: `original_query`, `expanded_queries`, `chunks`, `total_candidates`,
  `latency_ms`, `.chunk_count`.

### `src/core/interfaces.py`
- `BaseDocumentLoader` — `supports(path)`, `load(path) -> Document`
- `BaseParser` — `parse(document) -> list[Section]`
- `BaseChunker` — `chunk(section, document) -> list[Chunk]`
- `BaseEmbeddingProvider` — `dense_dimensions` property; **async** `embed_dense`, `embed_sparse`,
  `embed_query`; plus **non-abstract sync `embed_dense_sync`** for Phase 2's multiprocessing workers
  (raises `NotImplementedError` by default; local providers override it and have the async version
  delegate through `asyncio.to_thread`).
- `BaseVectorStore` — **async** `initialize(vector_size)`, `upsert_points(chunks, dense, sparse)`,
  `hybrid_search(dense_query, sparse_query, limit, filters)`, **`delete_by_doc_id(doc_id) -> int`**,
  `count()`, `close()`.
- `BaseReranker` — **async** `rerank(query, candidates, top_k) -> list[ScoredChunk]`; must populate
  `rerank_score`; wraps `to_thread` internally.
- `BaseLLMProvider` — **async** `generate`, `generate_json(prompt, schema)`; **`stream` is
  `def ... -> AsyncIterator[str]`, NOT `async def`** (it is an async generator — calling it returns
  the iterator, you consume with `async for`; he questioned this and accepted the explanation).
- `BaseCache` — async `get`, `set`, `clear`. `BaseGuardrail` — sync `check(text) -> (bool, str|None)`.
  `BaseMetric` — `name` property, async `score(query, answer, contexts) -> float`.

### `src/core/utils.py`
`get_tokenizer`, `count_tokens`, `truncate_to_tokens`, `hash_file`, `hash_text`, `make_doc_id`,
`make_section_id(doc_id, title, order)`, `make_chunk_id(doc_id, section_id, chunk_index)`,
`normalise_whitespace`, `safe_filename`. Constants `NAMESPACE`, `DEFAULT_ENCODING`
(`o200k_harmony`), `FALLBACK_ENCODING` (`o200k_base`).
- Token counts are **approximate** — never call them exact.
- **`normalise_whitespace` MUST preserve `\n`** (final regex `[ \t\r\f\v]+`, never `\s+`). The legal
  parser anchors headers per line with `^`. Using `\s+` silently makes every document a single
  Preamble section. **This is the single most likely bug in the whole project.**

### Other Phase 1 surface
`from config.settings import settings` (+ `get_settings()`); computed props `contracts_dir`,
`checkpoint_dir`, `dead_letter_dir`, `redis_url`, `is_production`; method `validate_runtime()`.
`from src.core.logging import logger, new_request_id, set_request_id, get_request_id`.
`telemetry.span(name, warn_over_ms=None)`, `telemetry.measure(name)`,
`telemetry.measure_async(name)`. Exceptions carry `.details`, `.retryable`, `.status_code`,
`.to_dict()`; always chain with `raise ... from exc`.

### ⚠️ MANDATORY INGESTION CONTRACT (Phases 3, 15)
Deterministic UUIDv5 IDs make an **unchanged** re-ingest idempotent but do **NOT** garbage-collect.
If a document's chunk count shrinks, its section titles change, or the file moves, old points linger
as **stale ghosts that still get retrieved and cited** — worse than duplicates, because there is no
ID collision to reveal them. Therefore every re-ingest is:

```python
await store.delete_by_doc_id(doc.doc_id)      # then
await store.upsert_points(chunks, dense, sparse)
```

`doc_id` therefore **requires a Qdrant payload index** (deleting by filter on an unindexed field is
a full scan).

### Phase 2 decisions Phases 3 and 4 must honour
- **Parents live in the SAME Qdrant collection as children**, separated by `chunk_level`. Rejected:
  duplicating parent text into child payloads (~4× bloat); a Redis docstore (Phase 1 configured
  `allkeys-lru`, so parents would be evicted). ⇒ **Phase 3 needs payload indexes on `doc_id`,
  `chunk_level`, `year`, `section_title`.** ⇒ **Phase 4 must filter searches to
  `chunk_level == "child"`, then substitute parents before generation.**
- Parent `chunk_index` is offset by `1_000_000` so parents/children never collide in `make_chunk_id`.
- `IngestionPipeline.run(limit, years, resume)` is a **sync generator yielding `ChunkBatch`** (fields:
  `chunks: list[Chunk]`, `document_count: int`) and stores nothing.
- **Two-phase commit is mandatory.** After storing a batch the consumer calls `pipeline.commit()`, or
  `pipeline.rollback()` if the write failed. Ingestion only *stages* checkpoint records; it must never
  mark a document done, because it does not know whether storage succeeded. Phase 3's ingest driver must
  implement this loop.
- `IngestionCheckpoint` API: `should_skip_path(path)` (cheap stat-based, pre-dispatch),
  `should_skip_content(doc_id, hash)` (authoritative), `stage(...)`, `commit()`, `discard_staged()`,
  `reset()` (truncates in place; safe inside the context manager). Records carry a
  `pipeline_fingerprint()` header — changing `MAX_TOKENS_PER_CHUNK`, `SENTENCE_OVERLAP`, either model
  name, or the manual `"v1"` marker invalidates the whole checkpoint.
- `DeadLetterQueue.record(path, *, error_type: str, message: str, stage: str, details, traceback_text)`
  takes **strings, not an exception**, so worker-side error types survive the process boundary. It
  counts each failure exactly once; there is no `merge_counts`.
- Ingestion is sync + `multiprocessing` on purpose (GIL). Workers do CPU only and return picklable
  `Chunk` lists; the parent does all I/O. Windows **requires** `if __name__ == "__main__":`.

### ⚠️ INTERFACE + MODEL CHANGES made in session 6 (from the Phase 4 and 5 reviews)
All in Phase 1, all additive except where noted. Phase 1's guide is updated.
- **`BaseLLMProvider.generate_json(..., model: str | None = None)`** — the parameter did not exist, so
  `EXPANSION_MODEL` and `GRADER_MODEL` were **unused settings** and every call ran on the generation
  model. Both reviews found this independently. Its docstring now also *requires* implementations to
  wrap `ValidationError` in `LLMProviderError`.
- **`GradingReport.verified: bool = Field(default=True, exclude=True)`** and `passed` now means
  `verified and is_grounded and is_relevant`. Fixes the worst bug found: the grader's fail-open path
  set grounded+relevant True, so `passed` was True and **a grader outage silently marked every answer
  verified**. `exclude=True` keeps it out of the LLM's schema so a model cannot declare its own audit.
- **`AnswerStatus` enum** (`ANSWERED | UNVERIFIED | LOW_CONFIDENCE | NO_MATCH | UNSUPPORTED`) and
  `RAGAnswer.status`, `.failure_reason`, `.thread_id`, `.invalid_citations`. Callers were being made to
  parse English prose to tell "no contract matched" from "the vector store is down".
- **`RetrievalResult.failed_arms: int = 0`** — partial search failure existed only in logs.
- **`InvalidQueryError(RAGException)`, status 400** — a blank query was reaching the store and being
  reported as a *retryable* retrieval outage.
- **`MaxRetriesExceededError` docstring rewritten** to match Phase 5's actual behaviour (raised only
  when there is no answer at all), and **`BaseReranker.rerank`'s postcondition weakened deliberately**:
  `rerank_score` is populated when scoring happened and left `None` on a fallback, so "skipped" stays
  distinguishable from "reranked". The contract now matches the better behaviour instead of the reverse.

### ⚠️ INTERFACE CHANGES made in session 5 (from the Phase 3 review)
Three edits to `src/core/interfaces.py`. Phase 1's guide is already updated; if he typed it earlier he
must patch it.
- **`BaseVectorStore.delete_by_doc_ids(doc_ids)` is now the abstract method** and
  `delete_by_doc_id(doc_id)` is a concrete wrapper delegating to it. Previously only the singular was
  declared while `IndexingPipeline` called the plural — a type error and an unimplementable contract
  for any new store.
- **`BaseEmbeddingProvider.embed_sparse_query(text) -> dict[str, list]`** added as abstract. Was a
  concrete-only `FastEmbedProvider` method, which meant the OpenAI provider could not serve Phase 4 at
  all. **Phase 4's `embed_probe` was using document-side `embed_sparse` for queries** — mis-weighted
  BM25 that still returns results, so invisible. Fixed.
- **`BaseEmbeddingProvider.close()`** added as a non-abstract no-op. The OpenAI provider owns an
  `AsyncOpenAI` pool; callers close unconditionally without branching on provider type.

### ⚠️ INTERFACE ADDITION made in session 4 — `BaseVectorStore.fetch_by_ids`
`async fetch_by_ids(ids: Sequence[str]) -> list[Chunk]`. Additive only; nothing existing changed.
Needed because `chunk_id` is **not** a filterable field, so Phase 4's parent substitution (and
Phases 13/16's reference walking) cannot use `hybrid_search`. Missing IDs are **skipped, not raised**
— an absent parent degrades to the child. Already written into Phase 1's `interfaces.py` and both
Phase 3 stores (`QdrantStore` uses `client.retrieve(..., with_vectors=False)`; `ChromaStore` uses
`collection.get(ids=...)`). Any future store must implement it.

### Phase 3 surface — now also frozen (written session 3)
- `src/embeddings/base.py`: `DEFAULT_BATCH_SIZE`, `MAX_INPUT_TOKENS`, `batched()`,
  `EmbeddingProviderBase` (`_prepare`, `_check_alignment`), `LocalEmbeddingProvider` (abstract
  `embed_dense_sync` / `embed_sparse_sync` / `embed_query_sync`; the async trio delegates via
  `asyncio.to_thread`).
- `FastEmbedProvider` — module-level `get_dense_model()` / `get_sparse_model()` (`@lru_cache`, lazy,
  per-process), `warmup()`, and the extra **`embed_sparse_query_sync(text) -> dict`** that Phase 4
  needs. `dense_dimensions` is measured by a probe embedding, not a registry lookup, and raises
  `DimensionMismatchError` against `DENSE_VECTOR_SIZE`.
- `OpenAIEmbeddingProvider` — dense from the API, **sparse from local BM25** (no OpenAI sparse
  endpoint; raising would silently kill hybrid search). `embed_dense_sync` stays unimplemented.
- `get_embedding_provider()` is **`@lru_cache`d**; `get_vector_store()` is deliberately **not**
  (a store owns a connection pool bound to an event loop; the caller owns the lifetime and closes it).
- `src/vectorstores/base.py`: `DENSE_VECTOR_NAME = "dense-bge"`, `SPARSE_VECTOR_NAME = "sparse-bm25"`,
  `PAYLOAD_INDEX_FIELDS`, `VectorStoreBase._validate_batch` / `_to_scored_chunks`.
- `QdrantStore` extras beyond the interface: **`delete_by_doc_ids(seq)`** (batch; `delete_by_doc_id`
  delegates to it), **`count_exact(filters)`**, **`set_bulk_mode(bool)`** (`indexing_threshold` 0 ↔
  20 000 — leaving it on makes every search a brute-force scan), `_build_filter` (**unknown filter
  keys raise** — silently ignoring one returns unfiltered results, which is a Phase 11 security hole).
- Sparse field **must** be created with `models.Modifier.IDF`. FastEmbed's BM25 intentionally omits
  the IDF term; without the modifier, lexical scoring silently ignores term rarity and nothing errors.
- `upsert` uses `wait=True` — with `wait=False` the indexing pipeline would `commit()` writes that
  are not durable.
- **New package `src/pipelines/`** (not in the roadmap tree — update it). `IndexingPipeline` owns the
  transaction: pull batch → `embed_dense` + `embed_sparse` concurrently → `delete_by_doc_ids` →
  `upsert_points` → `commit()`, or `rollback()` + re-raise on failure. It pulls from Phase 2's sync
  generator with `await asyncio.to_thread(next, batches, None)` (the sentinel avoids PEP 479's
  "coroutine raised StopIteration") and calls `batches.close()` in a `finally`.
- **Settings additions Phase 3 requires** (instruction is written into the doc; he must type them):
  `VECTOR_STORE`, `UPSERT_BATCH_SIZE`, `PREFETCH_MULTIPLIER`, `CHROMA_PATH`, `EMBEDDING_PROVIDER`,
  `OPENAI_EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE`, plus two validators and two `validate_runtime()`
  warnings.

---

## 7. EXTERNALLY VERIFIED FACTS (checked 2026-08-01 — re-verify if months have passed)

### Groq models — TIME-SENSITIVE
`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are **scheduled for shutdown 2026-08-16**.
`llama-3.1-70b-versatile` was decommissioned long ago (it is still in the old drafts — remove on
sight). Use throughout:
- `GENERATION_MODEL=openai/gpt-oss-120b`
- `EXPANSION_MODEL=openai/gpt-oss-20b`
- `GRADER_MODEL=openai/gpt-oss-20b`

Live list: `https://console.groq.com/docs/models`. Phase 6 should build a resolver that maps dead IDs
to replacements so this fails loudly and early rather than mid-run.

### Qdrant hybrid search — the modern API
```python
from qdrant_client import AsyncQdrantClient, models

await client.query_points(
    collection_name=...,
    prefetch=[
        models.Prefetch(query=<dense list[float]>, using="dense-bge", limit=N),
        models.Prefetch(query=models.SparseVector(indices=[...], values=[...]),
                        using="sparse-bm25", limit=N),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),   # or Fusion.DBSF
    query_filter=models.Filter(...),
    limit=K,
)
```
Requires Qdrant **≥ 1.10** (Universal Query API). Prefetch limits must be ≥ main `limit + offset` or
you get empty results. Prefetches can nest. **Deprecated — never use:** `client.search(...)`,
`client.recreate_collection(...)`. Use `collection_exists()` + `create_collection()`.

### Phase 4 surface — frozen (written session 4)
- `src/retrieval/`: `multi_query.py` (`QueryExpander`, `QueryVariations`), `hyde.py`
  (`HyDEGenerator`), `hybrid.py` (`HybridRetriever`, `SearchOutcome`), `fusion.py`
  (`reciprocal_rank_fusion`), `parents.py` (`ParentSubstituter`),
  `rerankers/{base.py: RerankerBase, cross_encoder.py: CrossEncoderReranker, mmr.py: MMRReranker}`,
  `pipeline.py` (`RetrievalPipeline.retrieve -> RetrievalResult`).
- **`parents.py` is a tree addition** (roadmap lists neither it nor `src/pipelines/` from Phase 3 —
  update the tree when next touching the roadmap).
- **Phase 4 does NOT depend on Phase 6.** `QueryExpander` / `HyDEGenerator` take
  `llm: BaseLLMProvider | None`; with `None` they no-op and log it. This is what allowed 4 before 6,
  and Phase 10 tests retrieval with no API key. **This trick does not extend to Phase 5's generation
  node.**
- **Stage order is a decision, not a style:** expand+HyDE (concurrent) → search all probes
  (concurrent, forced to `chunk_level=child`) → client-side RRF across probes → cross-encoder rerank
  → MMR → parent substitution. **Rerank children, never parents** (parents blow the 512-token
  cross-encoder window and get scored on their first quarter). **Substitute parents last.**
- Two levels of RRF now exist: Phase 3 fuses dense+sparse *within* one query server-side; Phase 4's
  `fusion.py` fuses *across* queries client-side. Qdrant cannot do the second — it never knows the N
  calls were related.
- `HybridRetriever._child_filter` **overrides** any caller-supplied `chunk_level` on purpose. Phase 16
  must call the store directly to search summary nodes.
- `search_many` uses `return_exceptions=True`: partial arm failure degrades and is counted in
  `SearchOutcome.failed_queries`; **all** arms failing raises `RetrievalError(retryable=True)`.
- Rerankers never raise — they fall back to the retrieval order with `rerank_score` left **None**, so
  "skipped rerank" stays distinguishable from "reranked". Cross-encoder scores are **raw logits**
  (unbounded, often negative); never show them as confidence or threshold them with a constant.
- `ParentSubstituter` deduplicates parents (5 children → typically 2–3 parents), carries the best
  child's `score`/`rerank_score` onto the parent, and enforces `MAX_CONTEXT_TOKENS` by **dropping
  whole sources**, never truncating one.
- **New settings**: `ENABLE_MULTI_QUERY`, `ENABLE_HYDE` (default false), `ENABLE_RERANKING`,
  `ENABLE_MMR` (default false), `ENABLE_PARENT_SUBSTITUTION`, `RRF_K=60`, `MMR_LAMBDA=0.7`,
  `MAX_CONTEXT_TOKENS=12000`, plus two `validate_runtime()` warnings.
- **Corrected in Phase 1:** `RERANK_MODEL_NAME` is now `Xenova/ms-marco-MiniLM-L-6-v2`. The old
  `cross-encoder/ms-marco-MiniLM-L-6-v2` is the sentence-transformers name and FastEmbed rejects it.
  Fixed in both the settings block and `.env.example`.

### Phase 5 surface — frozen (written session 5)
- `src/graph/`: `state.py` (`AgentState` TypedDict, `initial_state()`), `prompts.py`
  (`PROMPT_VERSION`, all prompt constants), `edges.py` (`after_router`, `after_retrieval`,
  `after_grading`, `after_rewrite` — all **pure functions of state**), `builder.py`
  (`build_graph`, `RAGAgent.answer -> RAGAnswer`), `nodes/{router,retrieval,generation,grading,
  rewriting}_node.py`.
- **`expansion_node.py` and `reranking_node.py` from the roadmap tree are deliberately NOT created** —
  Phase 4's `RetrievalPipeline` owns that orchestration. The graph keeps only the *decision*
  (`router_node` sets `expand`). Update the roadmap tree.
- `AgentState` uses `Annotated[list, operator.add]` reducers for `trace` and `attempts`. **Nodes
  return partial dicts and must return only their own contribution to a reduced key** — returning the
  accumulated list re-appends it.
- **Two loop budgets, both required:** `settings.MAX_RETRIES` (semantic, in state, drives the grading
  edge) and LangGraph's `recursion_limit` (structural, catches a bug in our own edges). `RAGAgent`
  sets `recursion_limit = 4 * (MAX_RETRIES + 1) + 6` so an edge bug fails in a second, not after 1000
  steps.
- **`MaxRetriesExceededError` is raised ONLY when no answer exists at all.** A poor answer is returned
  with its `GradingReport` attached and `LOW_CONFIDENCE_CAVEAT` appended — decided explicitly; see the
  argument in Phase 5 §2.
- Generation answers `state["query"]` (the original), while retrieval searches
  `state["current_query"]` (possibly rewritten). Getting that backwards answers a question nobody
  asked.
- `rewriting_node` is the **sole owner** of `retry_count`, and sets it to `MAX_RETRIES` to stop the
  loop when a rewrite makes no progress.
- Empty retrieval routes to a `no_context` node and **never reaches generation**. Aggregate queries
  (`router` → `unsupported`) are refused with an explanation — Phase 12 fills that branch.
- Phase 4 gained `RetrievalPipeline.retrieve(..., expand: bool = True)` for the router to drive.
- `scripts/verify_phase5.py` needs **no API key**: a `ScriptedLLM` supplies canned answers/grades, so
  the failing-grade retry path is deterministically testable. Reuse this pattern in Phase 10.

### Phase 6 surface — frozen (written session 6)
- `src/llm/`: `model_resolver.py` (`GROQ_DEPRECATIONS`, `resolve_model`, `check_configured_models`,
  `verify_models_live`), `base.py` (`to_strict_schema`, `LLMProviderBase`), `groq_provider.py`
  (`GroqProvider`), `openai_provider.py`, `factory.py` (`get_llm_provider` — **not** cached).
- `src/cache/`: `redis_cache.py` (`make_cache_key`, `RedisAnswerCache`), `semantic_cache.py`
  (`SemanticAnswerCache`, `semantically_incompatible`).
- `src/guardrails/`: `citation_validator.py` (`CitationValidator`, `CitationReport`,
  `strip_fabricated`), `prompt_injection.py` (`PromptInjectionGuard`, `neutralise_document`),
  `pii_masker.py` (`mask_pii`, `safe_for_log`).
- **`src/app.py` → `RAGService`** is the composition root: guard → cache → `RAGAgent.answer` →
  citation validator → cache write. Phase 8's routes depend on this one object.
- **`to_strict_schema` is mandatory, not a nicety.** Groq's `strict: true` requires EVERY property in
  `required` and `additionalProperties: false`; `model_json_schema()` provides neither (Pydantic omits
  defaulted fields from `required`). Passing the raw schema 400s, and the natural wrong conclusion is
  "strict mode does not work". It also drops `Field(exclude=True)` fields, which is what keeps
  `GradingReport.verified` out of the model's reach.
- `make_cache_key` includes **`PROMPT_VERSION`** — without it, editing a prompt has no effect on any
  cached question and the fix looks broken. Only `ANSWERED` answers are cached (`CACHE_MIN_STATUS`).
- **The semantic cache raises its own threshold floor to 0.97**, overriding Phase 1's 0.92 default,
  and applies a deterministic `semantically_incompatible` veto (entity swaps, differing numbers,
  negation). "buyer may terminate" vs "seller may terminate" is ~0.97 cosine with opposite answers.
- **Prompt injection is checked in BOTH directions.** Queries are rejected; retrieved document text is
  **neutralised, never dropped** — dropping would let an adversary hide a clause by making it look
  like an attack. Indirect injection via corpus text is the real threat, not user-side injection.
- **PII masking deliberately does NOT touch retrieved sources.** In a contract system party names and
  amounts are the answer; masking them is a compliance win and a product failure. It covers queries
  and log lines only.

### Groq (verified 2026-08-01 — TIME-CRITICAL)
- Production models: `openai/gpt-oss-120b` ($0.15/$0.60 per 1M), `openai/gpt-oss-20b` ($0.075/$0.30),
  `whisper-large-v3`, `whisper-large-v3-turbo`. The `.env` values are correct.
- **`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` shut down 2026-08-16** → `openai/gpt-oss-20b`
  and `openai/gpt-oss-120b` (or `qwen/qwen3.6-27b`). Enterprise committed-spend accounts exempt.
- Structured outputs: `response_format={"type":"json_schema","json_schema":{"name":..., "strict":True,
  "schema":...}}`. `strict:True` gives 100% compliance via constrained decoding on gpt-oss models.
  **Streaming and tool use are NOT supported with structured outputs.** `{"type":"json_object"}` is the
  best-effort mode.
- `AsyncGroq(api_key=..., timeout=...)`. Live model list: `GET https://api.groq.com/openai/v1/models`
  (also `client.models.list()`).
- Preview models worth knowing about but NOT for production: `openai/gpt-oss-safeguard-20b`,
  `meta-llama/llama-prompt-guard-2-86m` (a purpose-built injection classifier — a possible upgrade
  over Phase 6's regexes if it ever reaches production status).

### LangGraph (verified 2026-08-01 — re-verify if months have passed)
`from langgraph.graph import END, START, StateGraph`; `from langgraph.errors import
GraphRecursionError`. `StateGraph(AgentState)` with a TypedDict, **never** `StateGraph(dict)`.
`add_node(name, fn)` accepts `async def`. `add_conditional_edges(source, path_fn, path_map)` where
`path_map: dict[Hashable, str]` is optional but preferred (declares destinations at build time).
`compile(checkpointer=...)`. Run with `await graph.ainvoke(state, config)`; stream with `astream`.
- **`recursion_limit` is a TOP-LEVEL config key, NOT inside `configurable`.** Wrong placement is
  silently ignored. Default is 1000 (since 1.0.6).
- `thread_id` goes in `config["configurable"]` and is **required** when a checkpointer is set.
- `AsyncSqliteSaver.from_conn_string(path)` from `langgraph.checkpoint.sqlite.aio` is an **async**
  context manager; the sync `SqliteSaver` would block the loop on every super-step. `InMemorySaver`
  lives in `langgraph.checkpoint.memory`.
- **Checkpoints are written at super-step boundaries, not inside nodes** — a resumed run re-runs the
  interrupted node from the start of its function, so side effects (paid LLM calls) repeat.
- Reducers: `Annotated[list[str], operator.add]`. Also available: `langgraph.types.Send` (Phase 12
  map-reduce), `Command`, `interrupt`, `langgraph.managed.RemainingSteps`.

### FastEmbed rerankers (verified 2026-08-01)
`from fastembed.rerank.cross_encoder import TextCrossEncoder`;
`TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")`; `list(encoder.rerank(query, docs))`
→ `list[float]` of **logits** (e.g. `[-11.48, 5.47]`), one per document, order preserved. Also
supports `rerank_pairs(pairs)`. Other registry entries: `Xenova/ms-marco-MiniLM-L-12-v2` (0.12 GB),
`BAAI/bge-reranker-base` (1.04 GB), `jinaai/jina-reranker-v1-tiny-en`. Chosen over
`sentence-transformers` `CrossEncoder` to avoid a PyTorch dependency for a 90 MB CPU model.

### FastEmbed (verified 2026-08-01)
`TextEmbedding` (dense) and `SparseTextEmbedding` (sparse) expose `embed`, `query_embed`,
`passage_embed`. Use `passage_embed` for documents and `query_embed` for queries — `bge` is
asymmetric, and index/query must use the matching transformation. `Qdrant/bm25` is stemming +
MurmurHash3 token hashing (~10 MB, no neural net) and **requires the IDF modifier on the Qdrant
side**. All three methods return **generators** — materialise with `list(...)` immediately or the
second use yields nothing. Dense results are `numpy` `float32` arrays; sparse results are
`SparseEmbedding` with `.indices` / `.values`. Convert both to plain Python at the boundary.
Deliberately avoided: `TextEmbedding.list_supported_models()` for dimensions — its entry shape
(dict vs description object) has drifted between releases. Phase 3 probes with one embedding instead.

### Still to verify before writing
- **FastAPI / sse-starlette** (before Phase 8): current SSE + lifespan patterns. Also note Phase 5
  §11's streaming-versus-grading tension — you cannot un-stream an ungrounded answer, and the
  recommendation there is to stream `AgentState.trace` rather than answer tokens.
- **Groq** — re-verify the deprecation table in `src/llm/model_resolver.py` if the date is anywhere
  near or past 2026-08-16. A stale deprecation table is worse than none, because it is trusted.

---

## 8. SESSION 2 CORRECTION PASS — and the lesson

An external review of the roadmap + Phase 1 raised 13 issues; **~9 were valid** and are now fixed.
Recorded so the same mistakes do not recur.

**Real errors, corrected:**
- `Chunk.to_payload()` omitted `chunk_index` while `from_payload()` read it → not the inverse I
  claimed. The verification script now compares **every** persisted field (the old one only checked
  `.text`, which is why it missed this).
- Claimed context does **not** propagate through `asyncio.to_thread`. **False** — Python explicitly
  copies the context into the worker thread. The real boundary is a separate *process*.
- `BaseEmbeddingProvider` was sync "because embedding is CPU-bound" — but the planned OpenAI
  embedding provider is network I/O and would have been forced to block. Now async. I got this right
  for `BaseReranker` and wrong for embeddings **in the same file**.
- `QDRANT_HOST` validator's comment claimed it caught `http://locolhost:6333`; it only checked the
  scheme. Added a `difflib`-based `_looks_like_typo_of`.
- Called `cl100k_base` counts "exact" and "same tokenizer family as the LLM", then later contradicted
  it. GPT-OSS uses `o200k_harmony`. Now honest.
- `frozen=True` described as making a document uncorruptible; it is **shallow**.
- `stream`, `measure`, `measure_async` lacked return annotations despite my own
  `disallow_untyped_defs = true`.
- **Deterministic IDs were overclaimed** as preventing duplicates → led to the mandatory
  delete-before-upsert contract in §6.

**Also fixed:** explicit `[tool.setuptools.packages.find]` (two top-level packages `src` + `config`
otherwise breaks flat-layout discovery); pinned Docker image tags; both services bound to
`127.0.0.1`; **removed the fake Qdrant healthcheck** (distroless image cannot HTTP itself — a probe
that cannot fail is worse than none); full `.gitignore` replacement instruction, because it contains
only `.env` + `dataset/` while Phase 1 creates `data/` (**37 GB into git history would be
unrecoverable — this is still un-actioned by him; re-check**).

**Roadmap fixes:** §7.1 `Chunk(...)` example used a nonexistent `section=` field; §7.2 reranker
example returned `list[Chunk]` instead of `list[ScoredChunk]`; mandatory-checkpoint rule reconciled
with deferred verification; Phase 1 budget 550 → 1,000 with the calibration note.

> ### ⚠️ THE LESSON — apply this to every remaining phase
> The failure pattern was **confident prose overclaiming what the code actually does.** Before
> writing "this is exact", "nothing can mutate this", "X is the inverse of Y", or "this prevents
> duplicates" — re-read the code and weaken the claim to what it genuinely guarantees. Also note that
> the signature drift I warned him about with parallel subagents, I then produced myself, in sequence,
> across two files. Cross-check every doc against §6 before finishing.

### Phase 2 correction pass (2026-08-01, external review, scored 6/10 before fixes)

All 17 findings were valid or partially valid. Fixed in place:

- **Checkpoint recorded before storage** → split into `stage()` / `commit()` / `discard_staged()`, with
  commit owned by the consumer. This was the worst bug: a Qdrant outage would permanently mark
  documents as indexed.
- **`reset()` closed the file handle inside the context manager** → now truncates and reopens, so
  `resume=False` works.
- **Checkpoint blind to config changes** → `pipeline_fingerprint()` header; mismatch discards all records.
- **DLQ double-counted failures** → deleted `merge_counts`; `record()` is the single tally point.
- **DLQ lost error types** → `record()` takes `error_type: str`, so worker-origin types survive; also
  captures `traceback_text` for unexpected failures and `details` for `RAGException`s.
- **`re.split` deleted closing quotes/brackets** → boundaries now located with `finditer` and sliced.
- **`_TRAILING_NUMBER` merged `$2,000,000.` forward** → removed; replaced by a narrow
  `_ENUMERATOR_ONLY` check requiring the whole short fragment to be an enumerator.
- **Resume still parsed every completed document** → `should_skip_path()` runs pre-dispatch on
  size+mtime; the false claim that `hash_file` was used as a prefilter is gone.
- **`limit` applied during discovery** → now counts dispatched documents, post-skip.
- **Short sections and chunks silently discarded** → `_absorb_short_sections` and `_merge_undersized`
  merge into neighbours instead (the merge respects `max_tokens`).
- **`max_tokens` was advisory** → chunker re-tokenises the joined text and sheds sentences; the
  verification's 10% tolerance is gone.
- **Discovery ignored root-level files when subdirs existed** → `_files_in(root)` walked first; the
  "memory-flat" claim is now qualified as per-directory.
- **Verification logged 200 chars of a contract** (violates roadmap §7.5) → logs a token distribution.
- Missing `Chunk` import in the verification script fixed. *(Note: the reviewer said this raises
  `NameError`; it would not — PEP 526 does not evaluate function-local annotations. Still wrong, still
  fixed.)*
- **Phase 1 telemetry could not time generators** → added `isgeneratorfunction` /
  `isasyncgenfunction` branches to `telemetry.span`. Without them `ingestion.run` reported ~0ms.
- Added `check_resume()` and a DLQ-vs-stats consistency assertion to the verification script.

> **THE SECOND LESSON:** nearly every one of these was *a filter or shortcut whose failure mode was
> silent data loss*. When writing any filter, threshold, skip, or fast path, ask what happens to the
> data it rejects. If the answer is "it disappears", make it loud or make it merge.

### Session 3 notes (Phase 3)

- **Found and fixed a leftover Phase 1 contradiction.** `interfaces.py`'s "Why some methods are async"
  prose still claimed `BaseEmbeddingProvider` was sync — the exact error §8 recorded as corrected,
  fixed in the code block but not in the paragraph 200 lines below it. Now rewritten to justify the
  async choice, with `BaseGuardrail.check` as the sync contrast. **Lesson: when correcting a design
  decision, grep the whole file for prose that restates it.** A corrected code block plus stale prose
  is worse than either alone, because the reader cannot tell which one is current.
- Phase 3 came in at ~1,450 lines against a budget of 800 — consistent with the 1.5–2× calibration.
  The overrun is one unplanned file (`indexing_pipeline.py`, which the two-phase commit contract
  forced) plus retry and schema-verification code. No features were added beyond the brief.
- `_verify_schema` in `QdrantStore` is deliberately tolerant of introspection failure and says so in
  its docstring. `CollectionInfo`'s nested shape varies by client version and nothing can be executed
  here to confirm it; a schema check that crashes on an attribute access would be worse than one that
  warns. A genuine dimension mismatch still raises when it is visible.
- Scope kept out on purpose, and stated in the doc: quantisation (no measurement yet to justify the
  recall trade — Phase 7 supplies it), and any retrieval policy (`chunk_level` filtering, parent
  substitution) which belongs to Phase 4.

### Session 4 notes (Phase 4)

- **Added one method to the frozen interface** rather than working around it. Parent substitution
  needs a lookup by `chunk_id`, which is not a filterable field, so the honest options were
  `fetch_by_ids` on `BaseVectorStore` or a duck-typed Qdrant-only call in Phase 4. Added the method
  and patched Phase 1 + both Phase 3 stores in the same session, so nothing drifted. **Contrast with
  Phase 2's `_parent_slot` offset**, where the cleaner fix (changing `make_chunk_id`'s signature)
  was rejected because it rippled across four phases. The test is blast radius, not elegance: one
  additive method touching three docs is cheap; a changed signature touching four phases is not.
- Phase 4 came in at ~1,200 lines against a budget of 700 — again the 1.5–2× calibration. Two files
  carry the excess (`parents.py` and `mmr.py`) and both are load-bearing. Nothing was added beyond
  the brief.
- **Two features ship OFF by default** (`ENABLE_HYDE`, `ENABLE_MMR`), and the doc argues both sides
  rather than recommending them. On this corpus multi-query already closes most of the register gap,
  and MMR hurts precise single-fact queries. Phase 7 decides. Resisting "add the technique because
  it is impressive" is the scope control §4 demands.
- **MMR re-embeds its candidates** (~30ms for 20) because `hybrid_search` returns payloads without
  vectors. Documented as a deliberate cost with the two rejected alternatives (`with_vectors=True`
  on every query; threading vectors through `RetrievalResult`, which would change a Phase 1 model).
  Do not "fix" this later without re-reading that reasoning.
- The §11 latency table is the most useful thing in the phase for later work: **the two LLM calls are
  ~80% of retrieval latency and both happen before any search starts.** Everything Phases 3 and 4
  built lives inside a ~200ms envelope. That is the justification for Phase 5's router node and
  Phase 6's semantic cache, and it is the number to quote when asked what to optimise.

### Session 5 — the Phase 3 correction pass (external review, `IssuesinPhase3.md`, 16 findings)

**All 16 were valid or partially valid.** Fixed in place. The ones worth remembering:

- **Interface incompleteness (2 findings).** `delete_by_doc_ids` and `embed_sparse_query` existed only
  on concrete classes while callers typed against the ABC. Both promoted — see the interface-changes
  box above. The second one had already propagated into Phase 4 as a real bug.
- **"Stateless providers" was false** (finding 3). The factory prose claimed caching was safe because
  providers hold no connection state, two sentences from a class that owns an `AsyncOpenAI` pool.
  Rewritten to state the actual trade, plus `close()` on the base and in `index_corpus.py`.
- **`_retry` retried permanent failures** (4). Now classifies: 4xx raises immediately, non-retryable,
  with a per-status hint; only transport failures and 5xx retry. Marking a bad API key `retryable=True`
  would have told Phase 8 to keep trying it.
- **Two swallowed-exception bugs, both my own second lesson violated** (5, 7). Payload-index creation
  suppressed *all* errors as "probably already exists" — now it reads `payload_schema`, creates only
  what is missing, and raises. And `_set_bulk_mode` swallowed failures in **both** directions; leaving
  bulk mode on makes every search a brute-force scan, so leaving it is now a retried, raising
  operation (`_restore_bulk_mode`) while *entering* it stays best-effort. **Asymmetric operations need
  asymmetric error handling.**
- **Schema verification conflated two different things** (6). "I could not read the schema" (tolerate,
  warn — client shapes drift) versus "I read it and it is wrong" (raise). The first draft warned for
  both. Missing named vector, wrong dimension, wrong distance metric, missing sparse field now all
  raise.
- **Chroma silently dropped range/list filters** (8) — the same failure the Qdrant store raises on, and
  a Phase 11 security hole. Now raises. Chroma *could* support `$gte`/`$in`; deliberately not
  implemented, because unused translation code is untested translation code.
- **Cancellation race in the sync-generator bridge** (9). `asyncio.to_thread` **cannot be cancelled** —
  the worker thread keeps running inside `next(batches)` while the `finally` calls `close()`, giving
  `ValueError: generator already executing` and orphaned worker processes. Fixed with a
  `threading.Lock` shared by `_pull` and `_close`, with the close itself run in a thread.
- **Duplicate `chunk_id`s inside one batch** (12) would silently overwrite while `upsert_points`
  reported the full count. Now raises, alongside a sparse indices/values length check.
- **The verification script had a parent/child ID collision** (13) that made `check_filters` pass while
  testing nothing — parents at index 0 overwrote children at index 0. Now uses Phase 2's 1,000,000
  offset. Also: unique temp collection name (14, it deleted a fixed name unconditionally), a missing
  return annotation (15), a query-vs-document sparse assertion, and a new `check_transaction` that
  tests commit/rollback with stubs and no Qdrant (16).

> **THE THIRD LESSON:** every one of findings 1, 2, 5, 7, 8, and 12 is the **same mistake in six
> places** — an error path that continues as though nothing happened. I wrote the lesson about this at
> the end of session 2 ("if the answer is 'it disappears', make it loud"), then violated it six times
> in one file. Writing a lesson down is not the same as applying it. Before finishing any file, grep it
> for `except` and for every filter, and ask what the caller learns when that path is taken.

### Session 5 notes (Phase 5)

- **Wrote the phase against `BaseLLMProvider` with a `ScriptedLLM` in the verification.** This is
  better than waiting for Phase 6, and not only for sequencing: scripting a *failing grade* is the only
  way to test the retry path deterministically. The unhappy path works because it is the easy one to
  test this way.
- **Two roadmap nodes deliberately not built** (`expansion_node`, `reranking_node`) because Phase 4
  already owns that orchestration. Recorded in the frozen-surface box.
- **Argued against raising on a poor answer.** `MaxRetriesExceededError` fires only when there is no
  answer at all; a partially-supported answer is returned with its grading and a caveat. If this is
  revisited later, the reasoning is in Phase 5 §2 — read it before changing it.
- The §8 discussion of self-grading bias is the most interview-relevant passage in the phase: what
  actually mitigates it (extractive task, demand named unsupported claims, different model, confidence
  floor) versus what is theatre (1–10 scores, chain-of-thought before a boolean, majority of two
  correlated samples).

### Session 9 (cont.) — Phase 11, and the Part II staging decision

- **Phase 11 came in at 610 against 600**, and cheaply, because three Part I decisions were made for
  it: `_build_filter` raises on unknown fields (a tenant filter cannot be silently dropped),
  `cache_scope()` partitions both cache layers (one line adds the principal), and payload indexes are
  declared in one tuple. Worth telling him: that is what the earlier "why is this raising instead of
  ignoring?" decisions were buying.
- **`SecureStore` is the design to remember** — a wrapper implementing `BaseVectorStore` with no
  unfiltered search method, so the five existing call sites and every future one inherit the ACL
  without knowing the phase exists. Same pattern as `HybridRetriever` owning the `chunk_level` filter.
- **Two non-obvious holes closed:** `fetch_by_ids` bypasses every filter (parent substitution fetches
  by ID), and the answer cache is keyed on the question, so without the principal in `cache_scope()`
  a perfect store filter still leaks one user's answer *and sources* to another.
- **`assert_permitted` raises rather than quietly dropping** forbidden chunks. A post-filter that
  cleans up silently lets a broken pre-filter run in production indefinitely.
- **Phase 8 §5's `last_trace` is now fixed properly** via a per-request `contextvar`, which makes
  `tests/e2e/test_api.py::test_concurrent_streams_do_not_interleave` a real test.
- **The one argued weak point:** `DEFAULT_ROLES = ["public"]` — unclassified documents are visible to
  all authenticated staff. Fail-closed would silently remove a large fraction of the corpus from every
  search, presenting as "retrieval got worse". It is one constant and a policy call; §4 argues both
  sides. If he pushes back on this, the counter-argument is legitimate.

### Session 9 — Phase 10, and Part I complete

- **Phase 10's design was derived from this project's own history rather than from testing dogma.**
  Five review passes produced 148 findings; classified, ~45% were **seams** (interface drift), ~30%
  **error paths**, ~20% **claims not matching code**, and only ~5% unit-level logic. A conventional
  pyramid would have caught the 5%. So the phase inverts it: contract tests first, integration
  against real services second, unit tests only where a bug would be silent. **That table is the most
  useful thing in the phase** and it is worth quoting in an interview.
- **`tests/unit/test_contracts.py` is the payoff.** It asserts the interface *surface* — that
  `delete_by_doc_ids` is the abstract method, that `generate_json` takes `model`, that
  `GradingReport(verified=False).passed` is False, that `stream` is not a coroutine function. Three
  separate multi-week bugs found by human review are now assertions that fail on import.
- **Known limitations now have tests attached.** Phase 8's `last_trace` concurrency boundary is
  asserted rather than remembered — a limitation with no test becomes an unknown one.
- Cut ~1,500 lines: mirroring `src/` file-for-file, Hypothesis, mutation and load testing, a CI
  matrix, and **a coverage gate** (every bug this project had would have passed one).
- Two small bugs fixed while writing it: Phase 9's CLI caught `httpx.ConnectError` without importing
  `httpx`, and Phase 8's health verification assigned an async function to an instance attribute
  instead of subclassing.

### Session 8 — the Phase 7 correction pass (39 findings), plus Phases 8 and 9

**Phase 7's two worst findings invalidated its own output**, and both are the same shape — a harness
that measured something other than what it claimed:

- **Evaluation ran through the production answer cache.** A second run answered previously evaluated
  questions from Redis, skipping retrieval, generation, grading, and self-correction entirely. Every
  ablation in §9 would have compared last week's cached answers against themselves. `evaluate.py` now
  disables both caches before building the service, and `validate_for_baseline` refuses to promote a
  run with any cache hits.
- **Reported latency included the judges.** The timer stopped after two extra LLM calls, so
  "does reranking earn 20ms?" was measuring the evaluator. Pipeline and judge time are now separate
  fields.

Also fixed: the regression gate checked only overall means (a collapsed `vague` category, a run where
a third of cases crashed, or a grader outage flipping everything to `unverified` all passed); metrics
returned `-1.0` on failure, violating `BaseMetric`'s documented `[0.0, 1.0]` (now raises
`MetricUnavailable`, which also isolates per-metric failures so one bad metric cannot lose a whole
run); both judges used `GRADER_MODEL`, so the evaluator measured agreement with the system's own
grader (new `JUDGE_MODEL`); runs recorded no provenance, so a baseline could not prove what produced
it; a missing baseline was auto-created, so a broken first run became the standard; `save_cases`
opened `"w"` while the prose called it append-safe; contract text went into the generation prompt
unsanitised, letting corpus content manipulate the benchmark; and `multi_hop` was promised in the
design and never generated.

> **THE FIFTH LESSON — for measurement code specifically.** Ordinary code is wrong loudly; a broken
> harness is wrong *quietly and authoritatively*, and its output gets used to justify decisions. The
> question to ask of every metric is not "is this computed correctly" but **"what else could produce
> this number?"** A cache, a judge's latency, a shrinking sample, and a correlated judge all produce
> plausible numbers that mean nothing.

### Session 8 notes (Phases 8 and 9)

- **Both on budget** (600 and 640), using Phase 7's method: write §2 "What Was Cut" first. Phase 8
  declined auth, rate limiting, WebSockets, an ingestion trigger, and metrics exporters (~350 lines).
  Phase 9 declined a framework build, chat history, filtering/export, and eval charts (~500 lines).
- **Phase 8 needs one cross-phase edit:** `RAGAgent.answer` switches from `ainvoke` to
  `astream(..., stream_mode="values")` and maintains `self.last_trace`, so the SSE endpoint can report
  progress. Documented in Phase 8 §5, including its honest limitation — `last_trace` is per-agent, not
  per-request, so two concurrent streams interleave. Phase 11 fixes it properly when requests gain a
  principal.
- **Streaming decision implemented, not relitigated:** progress events, then one graded answer. Phase
  5 §11 already argued it.
- **Phase 9's `escapeHtml` is the real security note.** Source excerpts are unreviewed EDGAR text
  reaching `innerHTML` — the same untrusted-corpus threat as Phase 6's indirect prompt injection, one
  layer out. The corpus now crosses three interpreter boundaries: the prompt, the API response, and
  the DOM.

### Session 7 — the Phase 6 correction pass (31 findings) and Phase 7

**All 31 valid or partially valid; all addressed.** The ones that generalise:

- **A stray import of a module I had decided not to create** (`from .keys import make_cache_key`, in
  the file that defines `make_cache_key`). It would have failed on import, before anything ran. I
  moved that helper during writing and left the import behind. **Grep every new file for imports of
  modules the file tree does not contain.**
- **The semantic cache leaked across filter scopes** — the exact key included filters and models, the
  semantic index stored only `(query, vector, key)` and matched globally. Same question under two
  `doc_id` filters returned the other's answer *and sources*; under Phase 11 that is a tenant breach.
  Fixed with a shared `cache_scope()` used by both layers and a hard scope partition before any
  cosine is computed. **When two lookups must agree on identity, they must share one definition of
  it.**
- **`neutralise_document` detected without neutralising** — it replaced colons with hyphens, so a
  matched sentence containing no colon came back unchanged with `modified=True`. And nothing on the
  request path called it at all. **Detection reported as mitigation is worse than no mitigation.**
  Now it wraps matched spans as labelled quotations, and Phase 5's `format_context` applies it at the
  trust boundary.
- **Fabricated citations were stripped but the status stayed `ANSWERED`**, so the cache stored an
  answer with invented provenance as a good one. A validation failure must affect *eligibility*, not
  just formatting.
- **`strip_fabricated` was case-sensitive while detection was not** — `[source 7]` was counted and
  left in the answer. Two regexes for one format is how that happens.
- **`ENABLE_PII_MASKING` was read by nothing.** A configuration control with no runtime effect is
  worse than an absent one.
- Provider bugs: `LLM_MAX_RETRIES=0` skipped the call entirely (`max_retries` now means retries, so
  the loop runs n+1); a sticky `rate_limited` flag reported a final 401 as a rate limit; a
  `ValidationError` was marked retryable and never retried (parsing now happens inside the loop);
  unknown exceptions including `TypeError` were retried as transport failures; a 404 lost
  `ModelDecommissionedError`; `to_strict_schema` transformed only the top level, so any nested model
  would 400; `verify_models_live` was written and never called; and `resolve_model` ignored
  `shutdown_date` entirely, so a model dying in 2099 refused to start today.
- **`ENABLE_SEMANTIC_CACHE` now defaults to False.** The lexical veto is a blocklist of known-fatal
  differences, not a proof of equivalence, and no threshold makes topic embeddings safe as an answer
  cache. It is opt-in until Phase 7 measures it.
- **`CitationValidator` no longer claims to be a `BaseGuardrail`.** `check(text)` cannot detect
  fabrication without the sources; the conforming implementation satisfied the signature and none of
  the meaning. **An interface you cannot honestly implement is one you should not declare.**

### Session 7 notes (Phase 7)

- **Came in at 620 against a 620 budget** — the first phase to do so. The mechanism was writing §2
  ("What Was Cut, and Why") *before* the files: RAGAS, a dashboard, three extra LLM judges, and
  experiment tracking, ~700 lines declined. Every remaining phase should open with that section.
- The synthetic-set biases are stated plainly in §4 (answerable by construction, vocabulary-
  contaminated, single-source) with the conclusion that **these numbers are for comparison, never for
  reporting.** If someone quotes an absolute score from this harness as a quality claim, they have
  misread it.
- `-1.0` is the sentinel for "the judge failed", excluded from means, so a flaky grader lowers the
  sample size rather than the score. Averaging a failure in makes a broken judge look like a quality
  regression and sends you debugging the wrong component.
- §9 lists the seven experiments to run, with **two predictions recorded in advance** (reranking pays
  for itself; HyDE does not). Writing the prediction down first is the only protection against reading
  a result as confirmation.
- Needs one new store method: `QdrantStore.scroll_sample(limit, filters)`. Written into the Phase 7
  doc; `ChromaStore` may raise `NotImplementedError`.

### Session 6 — the Phase 4 and 5 correction passes (42 findings across two reviews)

Both reviews were substantially correct. Everything material is fixed. The findings worth carrying
forward as lessons:

**The composition bug (Phase 4 #1) is the most instructive of the whole project.** The cross-encoder
truncated to `final_k`; MMR short-circuits when `len(candidates) <= top_k`; so **enabling MMR did
absolutely nothing.** Two individually correct components composed into a no-op, and a direct unit test
of MMR passed happily while the pipeline ignored it. Fixed with `MMR_POOL_MULTIPLIER` (rerank to
`top_k × 3`, then diversify to `top_k`) plus a `check_mmr_composition` test that exercises the *real*
pipeline. **Generalisation: a stage that both scores and truncates has two responsibilities, and the
truncation is the one that breaks composition. Only the last selector in a chain may cut to the final
size.**

**The unverified-grade bug (Phase 5 #3) was the most dangerous.** The grader's fail-open path set
`is_grounded=is_relevant=True`, making `passed` True, so every caller checking `.passed` treated an
*unaudited* answer as verified — a grader outage silently upgraded the entire system. Fixed with
`GradingReport.verified`, a separate `UNVERIFIED` status, a distinct caveat, and an edge branch that
does not retry (rewriting the query cannot fix a broken grader).

**Unused settings that read as implemented features.** `EXPANSION_MODEL` and `GRADER_MODEL` were never
passed, because `generate_json` had no `model` parameter — so the documented cost/quality split was
prose. Both reviews found it. **Lesson: when a doc claims a setting has an effect, trace the value to
its use site before shipping the claim.**

**Diagnostics that lived only in logs.** `SearchOutcome.failed_queries`, the HyDE arm's candidate
contribution, the invalid-citation count, and per-attempt grades were all logged and then dropped.
Phase 7 reads results, not logs. All four are now returned data. **Lesson: if a later phase needs a
number, logging it is the same as discarding it.**

**Errors classified by convenience rather than cause.** A generation outage raised
`MaxRetriesExceededError`; a retrieval outage was returned as friendly prose; a blank query surfaced as
a retryable vector-store failure. All three produce the wrong HTTP status and send someone to the wrong
component. Now: `InvalidQueryError` (400), `LLMProviderError`, `RetrievalError` (503, retryable), and
`AnswerStatus` for the non-exceptional cases.

**Smaller but real:** `\b§` never matches (word boundary before a non-word char), so the advertised
`§ 3.1(b)` route was dead; `"count" in query` matched "account" and "discount"; a retry reused the
router's original *skip expansion* decision even though the failed grade was evidence that broader
retrieval was needed; `_is_progress` documented a four-word minimum and enforced two; the cross-encoder
reserved a fixed 96 tokens for the query instead of measuring it; `_apply_budget` claimed a token
ceiling it did not enforce; the cross-encoder loaded even when reranking was disabled; only the
*concrete* reranker caught its own failures, so a different `BaseReranker` could abort retrieval;
`isinstance(item, BaseException)` swallowed `CancelledError` as a failed search arm; and RRF hardcoded
`method=HYBRID`, relabelling Chroma's dense-only results.

> ### ⚠️ THE FOURTH LESSON
> Three reviews, ~54 findings, and the recurring shape is now clear: **the code was usually right and
> the CLAIMS around it were wrong.** A docstring promising a ceiling the method did not enforce; a
> comment describing MMR diversifying reranked results while the code recomputed cosine relevance; a
> settings block implying per-task models; an interface postcondition the implementation deliberately
> violated for good reasons. Session 2's lesson was "weaken the claim to what the code guarantees" —
> that is still the right lesson, and it needs a mechanical form: **for every claim in a docstring or
> comment, name the line that makes it true. If you cannot, the claim is a wish.**

### Session 6 notes (Phase 6)

- **Did the correction passes before writing Phase 6, and it paid for itself immediately.** The Phase 5
  review's #1 finding was that the *old* Phase 6 draft defined a second, synchronous
  `BaseLLMProvider` — so "Phase 6 substitutes Groq and changes nothing else" would have failed
  structurally. Writing Phase 6 first would have wasted the phase.
- **`to_strict_schema` is the highest-value 30 lines in the phase** and appears in no plan. Groq's
  strict mode requires every property in `required` and `additionalProperties: false`; Pydantic
  provides neither. The naive implementation 400s, and the natural conclusion is "strict mode doesn't
  work, use json_object" — which quietly removes the guarantee that eight fail-open paths across
  Phases 4 and 5 depend on staying dormant.
- **Two files not in the roadmap tree:** `model_resolver.py` and `src/app.py` (`RAGService`). The
  resolver is justified by a shutdown date fifteen days out; the composition root is justified by
  Phase 8 needing one object rather than seven.
- **PII masking scope was cut deliberately** and the reasoning is in the doc: masking retrieved
  contract text would produce "[PERSON_1] shall indemnify [ORG_2]", which satisfies a checkbox and
  destroys the product. It covers queries and logs only. This is the kind of feature reduction §4 asks
  for.
- The semantic cache section is the most interview-ready material in the phase: **a cache that can
  return the wrong answer is not a cache, it is a bug with a latency benefit.** Cosine similarity
  encodes topic, not logical content, so one swapped entity moves the vector ~0.03 and inverts the
  answer.

---

## 9. REPO FACTS

- **Windows / PowerShell.** `ls -la` fails — use `Get-ChildItem`. Shell calls take ~20s, so batch
  them. Avoid shell where a dedicated tool exists.
- Notebooks folder is misspelled **`notesbooks/`**; target tree says `notebooks/`. Phase 2 mentions
  the rename once — don't nag.
- `.gitignore` currently contains ONLY `.env` and `dataset/`. Phase 1 has the full replacement.
- Working prototype in `Pipelines/` (flat, sync, runnable) — superseded but kept. Live bugs already
  documented in roadmap §8: `main.py` imports `IngestionPipeline` but the class is `IngestPipeline`;
  `config.py` has `http://locolhost:6333`; `agent.py` indexes sparse vectors but never queries them,
  so retrieval is not actually hybrid; `src/core/interfaces.py` has a `BassDocumentLoader` typo.
- `src/core/` on disk still holds the **old** `exceptions.py`, `interfaces.py`, `logging.py`,
  `telemetry.py`. `models.py` and `utils.py` **do not exist yet**. `config/settings.py` is the old
  version. He has not started typing the rewritten Phase 1 — that is expected and fine.
- Git repo at `C:/ProjectsCursor/RAG`. Do not commit unless asked.

---

## 10. TONE

He wants a teacher who gives a **recommendation with reasoning**, not a menu. Honest numbers land
well — the "2,522 fenced lines = only 22% of target" audit is what set the whole plan, and the
calibration note was received well. Keep flagging real problems proactively; the EDGAR-vs-CUAD catch
and the decommissioned Groq models were both appreciated. He asks short direct questions ("yes or
no?", "is that good?") and wants short direct answers before any elaboration. He does not want
defensiveness when reviewed — concede what is wrong, push back with reasons where warranted (the
"not production-safe Docker" point was partly scope disagreement and saying so was fine).
