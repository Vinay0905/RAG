# SESSION HANDOFF — read this first, in full, before doing anything

> **You are continuing a multi-session project. You have no memory of the earlier sessions.**
> This file is the complete transfer. It is written to be self-sufficient: read it top to bottom and
> you will know the arrangement, the decisions, what exists, what is next, and the exact contracts
> your next document must honour.
>
> Internal working file — not one of the study guides. Keep it updated at the end of every session.
> Last updated: end of session 2 (2026-08-01).

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
| `docs/Phase1_System_Foundations.md` | ~2,575 | ✅ current (rewritten + corrected + `ChunkLevel` added) |
| `docs/Phase2_Ingestion_Engine.md` | ~2,400 | ✅ current (rewritten session 2) |

### Still the OLD drafts — must be rewritten
`Phase3_VectorStores_Embeddings.md`, `Phase4_Hybrid_Retrieval_Reranking.md`,
`Phase5_LangGraph_Agent.md`, `Phase6_LLM_Cache_Guardrails.md`, `Phase7_LLMOps_Evaluation.md`,
`Phase8_FastAPI_Server.md`, `Phase9_CLI_Web_UI.md`, `Phase10_Testing_Verification.md`.
**Phases 11–16 do not exist yet.**

### Reference only — superseded, do not delete, do not follow
`docs/01_ingestion_chunking_guide.md` … `docs/04_pipeline_production_guide.md`, `docs/answers.md`,
`docs/Dual_Pipeline_Architecture.md`, `docs/product_requirements_document.md`. All prototype-era.
Where they conflict with a rewritten phase, **the phase wins**.

### ▶ IMMEDIATE NEXT ACTION
Rewrite **`Phase3_VectorStores_Embeddings.md`**, then **`Phase4_Hybrid_Retrieval_Reranking.md`**.
Phase 3 must cover, at minimum:
- `src/embeddings/` — `base.py`, `fastembed_provider.py`, `openai_provider.py`, `factory.py`.
  Async interface; FastEmbed implements `embed_dense_sync` for Phase 2's workers and has
  `embed_dense` delegate to it via `asyncio.to_thread`.
- `src/vectorstores/` — `base.py`, `qdrant_store.py`, `chroma_store.py`, `factory.py`.
- `AsyncQdrantClient`, named vectors (`dense-bge` + `sparse-bm25`), `collection_exists()` +
  `create_collection()` (**never** `recreate_collection`), payload indexes on **`doc_id`**,
  **`chunk_level`**, `year`, `section_title`, batched upserts, and **`delete_by_doc_id`**.
- Consume Phase 2's `IngestionPipeline.run()` generator of `ChunkBatch` objects, and **drive the
  `commit()` / `rollback()` contract** — this is now a hard requirement, not a nicety.
- The `Phase3_VectorStores_Embeddings.md` currently in `docs/` predates all of this: it still exposes
  synchronous, dict-based interfaces and no delete-before-upsert. **Rewrite it, do not patch it.**

---

## 4. LINE BUDGET

| | Phase | Est. LOC |
| :-- | :--- | ---: |
| 1 | System Foundations | 1,000 ✅ |
| 2 | Ingestion at Scale | 1,600 ✅ |
| 3 | Embeddings & Vector Stores | 800 |
| 4 | Hybrid Retrieval & Reranking | 700 |
| 5 | LangGraph Agent | 900 |
| 6 | LLM, Cache & Guardrails | 700 |
| 7 | LLMOps & Evaluation | 600 |
| 8 | FastAPI Async Server | 600 |
| 9 | CLI & Web Dashboard | 650 |
| 10 | Testing & Verification | 1,100 |
| 11 | Multi-Tenant Security & RBAC | 600 |
| 12 | Map-Reduce Aggregation | 550 |
| 13 | GraphRAG Multi-Hop | 850 |
| 14 | Layout-Aware PDF & Vision OCR | 700 |
| 15 | Embedding Drift & Shadow Indexing | 550 |
| 16 | RAPTOR Summary Trees | 650 |

**Calibration:** Phase 1 was budgeted 550 and came in ~1,000. Phase 2 budgeted 1,000, came in ~1,600.
Production-density code is dominated by validators, docstrings, and annotations, which estimates omit.
Expect **1.5–2× the stated numbers**, so realistically **13,000–15,000 total**. The 10,000 target is
already safe. **⇒ The job is now scope CONTROL, not scope growth.** If a phase balloons far past
budget, cut features rather than continuing.

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

### Still to verify before writing
- **LangGraph** (before Phase 5): current `StateGraph` construction, `add_conditional_edges`
  signature, checkpointer/`langgraph-checkpoint-sqlite` usage, recursion-limit handling.
- **FastAPI / sse-starlette** (before Phase 8): current SSE + lifespan patterns.

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
