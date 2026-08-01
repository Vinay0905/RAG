# Session Handoff — Working Notes

> Internal scratch file for continuity between working sessions. Not part of the study guides.
> Last updated: end of session 1.

---

## The arrangement

I act as a **teacher, not a coding agent**. I write **only `.md` files inside `docs/`**. Vinay types
all the actual Python himself, with the guide open beside the editor. Never create or edit files in
`src/`, `Pipelines/`, `config/`, or `tests/`.

Goal: a genuinely production-grade RAG system, ~10,000 lines, built to be *understood* rather than
shipped fast. Learning priorities named explicitly: **RAG, LangGraph, Python, Pydantic**.

---

## Decisions locked in session 1

| Question | Decision |
| :--- | :--- |
| Existing Phases 1–10 | **Full rewrite** at production density. Rationale: old drafts pass bare `dict`s everywhere, so Pydantic never gets taught. Phases 11–16 build on 3/4/5, so a weak foundation would force a rewrite later. |
| Teaching depth | **Full** — Problem / Design Decision / Theory / Code / Failure Modes / Checkpoint, six parts per file. |
| Dataset | **EDGAR/SEC, ~650k `.txt`, ~37 GB, year-partitioned.** Not CUAD. Corpus lives on a *different laptop* — not present here. So Phase 2 must teach large-scale ingestion (multiprocessing, checkpointing, dead-letter queue) and must work against a small sample dir first. |
| Code style | **Async-first.** Exception: CPU-bound work (embedding, cross-encoder) uses `asyncio.to_thread`; Phase 2 ingestion uses `multiprocessing` to escape the GIL. |
| Delivery | Master roadmap first, then **2–3 phases per batch**. Vinay codes later, not in real time — so keep shipping docs at a steady clip. |
| Scope worry | He flagged 10k lines as a big task. Agreed to generate in batches. Reassure with the per-phase budget rather than re-litigating scope. |
| **No local execution** (session 2) | Nothing gets run or tested on this laptop. Checkpoint sections are written as **"Verification (deferred)"** — precise scripts to run later on the corpus machine, never as blocking gates. |
| **Subagent parallelism** (session 2) | Asked about 10 parallel agents. Agreed plan: **Phase 1 written solo** (it freezes the shared model/interface signatures), then fan out in dependency-safe batches, then a wide 6-way fan-out for Phases 11–16. Rationale he accepted: independent agents invent incompatible signatures, and doc-generation speed is not his bottleneck since he hand-types everything. |

### Consequence of "no local execution"
I cannot verify any library API by running it, so **verify externally before writing**. Already done
for Groq model IDs and the Qdrant hybrid API. Do the same for LangGraph before Phase 5 and
FastAPI/SSE before Phase 8.

---

## Status

**Done**
- Full audit of the repo, the PRD, all 10 phase drafts, the 4 older numbered guides, and
  `future_ideas_problems.md`.
- Measured the real gap: existing Phases 1–10 contain **2,522 fenced lines total**, of which maybe
  2,000–2,200 is Python. That is ~22% of the 10k target. This number drove the whole plan.
- Wrote **`docs/00_MASTER_ROADMAP.md`** (~515 lines). Contains: architecture, complete target
  directory tree, 16-phase table with per-phase LOC budget totalling ≈11,500, dependency graph,
  tech-choice rationale, §7 coding conventions, §8 deprecation corrections table, progress tracker.

- **Session 2:** rewrote `Phase1_System_Foundations.md` in full (~1,450 lines of md, 10 files,
  ~550 LOC). Added the two new files: `src/core/models.py` (the Pydantic spine) and
  `src/core/utils.py` (deterministic UUIDv5 IDs). Verified Groq model IDs and the Qdrant hybrid API
  externally first.

- **Session 2 (cont.):** ran the correction pass on the roadmap + Phase 1 (see below), then rewrote
  `Phase2_Ingestion_Engine.md` in full — 11 files, ~1,600 LOC.

**Next up**
1. `Phase3_VectorStores_Embeddings.md` (budget 800 LOC, expect ~1,200).
2. `Phase4_Hybrid_Retrieval_Reranking.md` (budget 700 LOC).
3. `Phase5_LangGraph_Agent.md` — **verify the current LangGraph API externally first.**

### Phase 2 decisions that later phases must honour
- `ChunkLevel` enum was **added to Phase 1's `models.py`** during Phase 2 (STANDALONE / CHILD /
  PARENT / SUMMARY) plus a `chunk_level` field on `Chunk`, in `to_payload`/`from_payload` too.
  Parent-child needed it and hacking around it was worse. Phase 16's RAPTOR reuses `SUMMARY`.
- **Parents live in the same Qdrant collection as children**, separated by `chunk_level`. Rejected:
  duplicating parent text into child payloads (~4× bloat), and a Redis docstore (Phase 1 set
  `allkeys-lru`, so parents would be evicted). ⇒ Phase 3 needs payload indexes on **`doc_id`** and
  **`chunk_level`**; Phase 4 must filter searches to `chunk_level == "child"` and then substitute
  parents before generation.
- Parent `chunk_index` is offset by `1_000_000` so parents and children never collide in
  `make_chunk_id`. Cleaner would be adding `level` to the ID derivation, but that changes a Phase 1
  signature already referenced by Phases 3 and 15.
- `normalise_whitespace` **must preserve `\n`** (final regex `[ \t\r\f\v]+`, not `\s+`) — the legal
  parser anchors headers with `^` per line. Using `\s+` silently makes every document one Preamble
  section. This is the single most likely Phase 2 bug.
- Pipeline yields **batches of chunks** and stores nothing; Phase 3 consumes the generator.
- Workers do CPU only (load/parse/chunk) and return picklable `Chunk` lists; parent does all I/O.
- `IngestionPipeline.run()` is sync + `multiprocessing`, deliberately not async — GIL, see the doc's
  §12. This is the one sanctioned exception to async-first.
- `embed_dense_sync` on `BaseEmbeddingProvider` exists for these workers.

---

## FROZEN CONTRACT — every later phase and every subagent must match this exactly

Phase 1 is written, so these names are now fixed. Any doc that contradicts them produces code that
will not run. Brief subagents with this table verbatim.

**Models** (`src/core/models.py`): `DocumentType`, `RetrievalMethod`, `Document` (frozen),
`Section`, `Chunk`, `ScoredChunk`, `RetrievalResult`, `Citation`, `GradingReport`, `RAGAnswer`.
- `ScoredChunk` uses **composition** — `.chunk`, `.score`, `.rank`, `.method`, `.rerank_score`,
  and the computed `.effective_score`. It does **not** subclass `Chunk`. So it is
  `scored.chunk.text`, never `scored.text`.
- `Chunk` has `.to_payload()` / `.from_payload()`. Only these touch Qdrant payload dicts.
- `GradingReport.passed` is the computed gate driving Phase 5's retry edge.
- `Document.year` is `int | None` (validator coerces the string dir name; junk → `None`).

**Interfaces** (`src/core/interfaces.py`): `BaseDocumentLoader` (`supports`, `load`),
`BaseParser` (`parse`), `BaseChunker` (`chunk`), `BaseEmbeddingProvider`
(`dense_dimensions` property, `embed_dense`, `embed_sparse`, `embed_query` — **all async**, plus
non-abstract **sync** `embed_dense_sync` for Phase 2's multiprocessing workers),
`BaseVectorStore` (`initialize`, `upsert_points`, `hybrid_search`, **`delete_by_doc_id`**, `count`,
`close` — **async**), `BaseReranker` (`rerank` — async, wraps `to_thread` internally),
`BaseLLMProvider` (`generate`, `generate_json` — async; **`stream` is `def ... -> AsyncIterator[str]`**,
not `async def`, because it is an async generator), `BaseCache`, `BaseGuardrail`, `BaseMetric`.

**Utils** (`src/core/utils.py`): `get_tokenizer`, `count_tokens`, `truncate_to_tokens`,
`hash_file`, `hash_text`, `make_doc_id`, `make_section_id`, `make_chunk_id`,
`normalise_whitespace`, `safe_filename`, plus `NAMESPACE`, `DEFAULT_ENCODING`
(`o200k_harmony`), `FALLBACK_ENCODING` (`o200k_base`). Token counts are explicitly
**approximate** — never call them exact.

**MANDATORY INGESTION CONTRACT (Phases 2, 3, 15):** deterministic UUIDv5 IDs make an *unchanged*
re-ingest idempotent but do **not** garbage-collect. If chunk count shrinks, section titles change,
or a file moves, old points linger as stale ghosts that still get retrieved and cited. So every
document re-ingest must be `await store.delete_by_doc_id(doc.doc_id)` **then**
`await store.upsert_points(...)`. `doc_id` therefore needs a Qdrant payload index in Phase 3.

**Other:** `settings` + `get_settings()` from `config.settings`; computed props
`contracts_dir`, `checkpoint_dir`, `dead_letter_dir`, `redis_url`, `is_production`, plus
`validate_runtime()`. `logger` / `new_request_id` / `get_request_id` from `src.core.logging`.
`telemetry.span(name, warn_over_ms=)`, `telemetry.measure`, `telemetry.measure_async`.
Exceptions carry `.details`, `.retryable`, `.status_code`, `.to_dict()`.

**Model IDs — verified 2026-08-01.** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are
**shut down 2026-08-16**. Use `openai/gpt-oss-120b` (generation) and `openai/gpt-oss-20b`
(expansion + grading) everywhere.

**Qdrant hybrid — verified 2026-08-01.** `AsyncQdrantClient.query_points(collection_name=...,
prefetch=[models.Prefetch(query=<dense list>, using="dense", limit=N),
models.Prefetch(query=models.SparseVector(indices=..., values=...), using="sparse", limit=N)],
query=models.FusionQuery(fusion=models.Fusion.RRF), limit=K)`. Prefetch limits must be
≥ main `limit + offset`. `Fusion.DBSF` is the alternative to RRF.

---

## Phase 1 rewrite plan — ✅ DONE, kept as a record

Old draft had 8 files. New version has **10**, and the two additions are the important ones.

| File | Note |
| :--- | :--- |
| `pyproject.toml` | Fix `[tool.ruff] select` → `[tool.ruff.lint] select`. Add `neo4j`, `pdfplumber`, `networkx` as optional extras for Part II. |
| `docker-compose.yml` | Drop the obsolete `version:` key. Add healthchecks for both services. |
| `.env.example` | Windows-friendly `DATASET_PATH`. Replace the decommissioned Groq model. |
| `config/settings.py` | Add field validators; add `@computed_field`; split model config into a nested group. |
| `src/core/exceptions.py` | Keep the hierarchy, add `to_dict()` for API error envelopes in Phase 8. |
| **`src/core/models.py`** | **NEW — the spine.** `Document`, `Section`, `Chunk`, `ScoredChunk`, `RetrievalResult`, `GradingReport`. This is where Pydantic actually gets taught. |
| `src/core/interfaces.py` | Rewrite against the models above, not `Dict[str, Any]`. Fix the `BassDocumentLoader` typo that is live in his repo right now. |
| `src/core/logging.py` | `datetime.now(timezone.utc)`. Add `contextvars` correlation IDs — needed by Phase 8. |
| `src/core/telemetry.py` | Must handle **both** sync and async functions; the old decorator only wrapped sync. |
| **`src/core/utils.py`** | **NEW** — tokenizer singleton, SHA-256 file hashing, deterministic UUIDv5 chunk IDs. |

Teaching beats to hit in Phase 1: why ABCs over duck typing; why `Chunk` is a model and not a dict
(tie back to roadmap §7.1); exception chaining with `from exc`; why `settings` is a module-level
singleton and the import-time-side-effect tradeoff that comes with it.

---

## Correction pass — session 2, after an external review

An external review of the roadmap + Phase 1 flagged 13 issues; ~9 were valid and are now fixed in
both docs. Recorded so the same mistakes do not recur in later phases.

**Genuine errors I made, now corrected:**
- `Chunk.to_payload()` omitted `chunk_index` while `from_payload()` read it → not the inverse I
  claimed. Fixed; the verification script now compares **every** persisted field, not just `.text`.
- Claimed context does **not** propagate through `asyncio.to_thread`. **False** — Python explicitly
  copies the context into the worker thread. The real boundary is a separate *process*
  (Phase 2 multiprocessing), which is why `set_request_id` exists.
- `BaseEmbeddingProvider` was sync "because embedding is CPU-bound" — but the planned OpenAI
  embedding provider is network I/O and would have been forced to block. Now async. **Rule: all
  interfaces are async; CPU-bound implementations offload internally.** I got this right for
  `BaseReranker` and wrong for embeddings in the same file.
- `QDRANT_HOST` validator's comment claimed it caught `http://locolhost:6333`; it only checked the
  scheme, so the typo passed. Added a `difflib`-based `_looks_like_typo_of` check.
- Called `cl100k_base` counts "exact" and "same tokenizer family as the LLM", then later admitted
  they differ. GPT-OSS uses `o200k_harmony`. Now honest about being approximate.
- `frozen=True` described as making the doc uncorruptible; it is **shallow** — `metadata` dict stays
  mutable.
- `stream`, `measure`, `measure_async` had no return annotations despite my own
  `disallow_untyped_defs = true`.

**Fixed in the roadmap:** §7.1 `Chunk(...)` example used a nonexistent `section=` field; §7.2 reranker
example returned `list[Chunk]` instead of `list[ScoredChunk]`; the mandatory-checkpoint rule now
reconciles with deferred verification; Phase 1 budget corrected 550 → 1,000 with a calibration note
(expect all phases 1.5–2× estimate, realistic total 13–15k, so **scope control is now the job**).

**Also fixed:** explicit `[tool.setuptools.packages.find]` (two top-level packages `src` + `config`
otherwise breaks flat-layout discovery); pinned Docker tags; both services bound to `127.0.0.1`;
removed the fake Qdrant healthcheck (distroless image cannot HTTP itself — a probe that cannot fail
is worse than none); full `.gitignore` replacement because it only had `.env` + `dataset/` and Phase 1
creates `data/`.

**Lesson for later phases:** the failure pattern was *confident prose overclaiming what the code
does*. Before asserting "this is exact", "nothing can mutate this", "X is the inverse of Y", or "this
prevents duplicates" — re-read the code and weaken the claim to what it actually guarantees.

---

## Things to remember about the repo

- Windows / PowerShell. `ls -la` fails — use `Get-ChildItem`. Shell calls are slow (~20s each), so
  batch them.
- Notebooks folder is misspelled `notesbooks/`. The roadmap's target tree says `notebooks/`; mention
  the rename once when Phase 2 touches it, don't nag.
- Live bugs in the prototype, already documented in roadmap §8 — no need to re-explain:
  `main.py` imports `IngestionPipeline` but the class is `IngestPipeline`; `config.py` has
  `http://locolhost:6333`; `agent.py` never queries the sparse vector so retrieval is not actually
  hybrid; `src/core/interfaces.py` has `BassDocumentLoader`.
- `answers.md` and `Dual_Pipeline_Architecture.md` are prototype-era. Superseded, kept for
  reference. Don't delete.
- Deprecations to keep enforcing: `recreate_collection` / `client.search` → `query_points` with
  `prefetch`; sync `QdrantClient` → `AsyncQdrantClient`; `StateGraph(dict)` → typed state.

---

## Tone notes

He is building this to understand it and to have something defensible in an interview. He responds
well to being told the honest numbers (the 2,522-line audit landed well) and to being given a
recommendation with reasoning rather than a menu. Keep flagging real problems — the EDGAR-vs-CUAD
catch and the decommissioned Groq model were both appreciated.
