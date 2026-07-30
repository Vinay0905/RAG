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

**Next up — start here tomorrow**
1. Rewrite `Phase1_System_Foundations.md` (budget 550 LOC). It is currently open in his editor at
   502 lines, still the *old* draft.
2. Then `Phase2_Ingestion_Engine.md` (budget 1,000 LOC) — the biggest single rewrite, because the
   EDGAR scale decision changes it completely.
3. Then `Phase3_VectorStores_Embeddings.md` (budget 800 LOC).

---

## Phase 1 rewrite plan (worked out, ready to execute)

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
