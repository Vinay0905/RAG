# Phase 3 — Structural and Code Integrity Issues

Scope: code correctness and structural integrity only. Library versions and API compatibility were not assessed.

## Critical issues

1. **Batch deletion is missing from the interface**
   - `IndexingPipeline` accepts a `BaseVectorStore` but calls `delete_by_doc_ids()`.
   - `BaseVectorStore` only declares `delete_by_doc_id()`, so static type checking fails and alternative implementations are not required to support the batch operation.
   - References: `docs/Phase3_VectorStores_Embeddings.md:2180-2183, 2253`; `docs/Phase1_System_Foundations.md:1613-1621`.

2. **Sparse query embedding is absent from the abstraction**
   - FastEmbed defines `embed_sparse_query_sync()`, but `BaseEmbeddingProvider` does not expose a sparse-query method.
   - The OpenAI provider implements local sparse document embedding but no sparse query equivalent.
   - Phase 4 would therefore need a concrete provider or might incorrectly use document-side BM25 embedding for queries.
   - References: `docs/Phase3_VectorStores_Embeddings.md:659-674, 910-928`; `docs/Phase1_System_Foundations.md:1540-1585`.

3. **The cached OpenAI provider is stateful**
   - The factory describes embedding providers as stateless, but `OpenAIEmbeddingProvider` owns an `AsyncOpenAI` connection pool and defines `close()`.
   - Caching it across event loops can cause lifecycle errors, and `scripts/index_corpus.py` never closes it.
   - References: `docs/Phase3_VectorStores_Embeddings.md:834-835, 930-932, 986-993, 1026-1049, 2324-2342`.

4. **Qdrant retries permanent failures**
   - `_retry()` catches every exception, including authentication failures, invalid requests, and schema errors.
   - The final exception is always marked retryable, delaying deterministic failures and misclassifying them.
   - Reference: `docs/Phase3_VectorStores_Embeddings.md:1328-1360`.

5. **Mandatory payload-index failures are suppressed**
   - `_ensure_payload_indexes()` catches all exceptions and assumes the index probably already exists.
   - Authentication, network, and invalid-schema failures can therefore let initialization succeed without required indexes.
   - Reference: `docs/Phase3_VectorStores_Embeddings.md:1468-1488`.

6. **Schema verification is too permissive**
   - Missing named vectors or unexpected collection structures are converted into warnings.
   - Distance type, sparse-vector existence, and IDF configuration are not strictly enforced.
   - Reference: `docs/Phase3_VectorStores_Embeddings.md:1421-1466`.

7. **Bulk mode can remain enabled silently**
   - Disabling bulk mode is described as mandatory, but `_set_bulk_mode(False)` suppresses `RAGException`.
   - A run can report success while the collection remains in brute-force search mode.
   - References: `docs/Phase3_VectorStores_Embeddings.md:1490-1510, 2284-2300`.

8. **Chroma silently drops filters**
   - Range and list filters are removed instead of being supported or rejected.
   - This violates the shared vector-store contract and can return unfiltered documents, which becomes a security risk when filters enforce tenant authorization.
   - Reference: `docs/Phase3_VectorStores_Embeddings.md:1955-1971`.

9. **The synchronous-generator/async bridge has a cancellation race**
   - Cancelling while `asyncio.to_thread(next, batches, None)` is running does not stop the worker thread.
   - The `finally` block can call `batches.close()` concurrently, causing `generator already executing` and potentially leaving multiprocessing workers alive.
   - Closing the generator synchronously may also block the event loop.
   - Reference: `docs/Phase3_VectorStores_Embeddings.md:2204-2220`.

## Additional integrity problems

10. **Model initialization can block the event loop**
    - FastEmbed model loading and dimension probing occur synchronously inside the async indexing run.
    - References: `docs/Phase3_VectorStores_Embeddings.md:540-570, 2201`.

11. **OpenAI retry handling is inaccurate**
    - The implementation sleeps after the final failed attempt.
    - Exhausted 5xx failures are reported as `RateLimitError`, even when no rate limit occurred.
    - Reference: `docs/Phase3_VectorStores_Embeddings.md:860-898`.

12. **Store-side validation is incomplete**
    - It verifies list lengths and dense dimensions but not duplicate chunk IDs, sparse index/value alignment, or invalid numeric values.
    - Duplicate IDs can silently overwrite points while the method reports that every point was written.
    - Reference: `docs/Phase3_VectorStores_Embeddings.md:1142-1176`.

13. **The verification creates parent/child ID collisions**
    - `make_chunks()` starts both parent and child indexes at zero.
    - Parent upserts overwrite child points, weakening the filter test and hiding possible errors.
    - References: `docs/Phase3_VectorStores_Embeddings.md:2417-2433, 2567-2583`.

14. **The verification collection name is unsafe**
    - It uses the fixed name `verify_phase3_tmp` and deletes that collection unconditionally.
    - Existing data using the same name would be destroyed; the test should use a unique generated name.
    - References: `docs/Phase3_VectorStores_Embeddings.md:2403, 2644-2649`.

15. **A verification function lacks a return annotation**
    - `embed()` has no return annotation, violating the project’s strict typing convention.
    - Reference: `docs/Phase3_VectorStores_Embeddings.md:2436-2440`.

16. **Important contracts are not verified**
    - No verification covers Chroma behavior, checkpoint commit/rollback, cancellation, or partial storage failure.
    - These are among the phase’s most important structural guarantees.

## Sound design decisions

- The Phase 2 staged-checkpoint handoff correctly leaves commit ownership with the storage consumer.
- Delete-before-upsert recovers safely from changed documents and stale points.
- Named dense and sparse vectors, alignment checks, dependency injection, and server-side fusion are structurally sound.
- The remaining work should focus on interface completeness, provider lifecycle management, consistent filter semantics, strict schema initialization, and cancellation-safe bridging.
