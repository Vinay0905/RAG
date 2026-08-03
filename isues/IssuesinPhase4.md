# Phase 4 — Structural and Code Integrity Issues

Scope: code correctness and structural integrity only. Library versions and external API compatibility were not assessed.

## Critical issues

1. **MMR is effectively disabled by the pipeline ordering**
   - The cross-encoder first truncates the fused candidates to `final_k`.
   - MMR then receives exactly `final_k` candidates and short-circuits whenever `len(candidates) <= top_k`.
   - With the normal pipeline, enabling MMR therefore does nothing.
   - The cross-encoder should retain a wider candidate pool, and MMR should select the final `k`.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1388-1393, 1870-1879`.

2. **MMR discards the cross-encoder’s relevance scores**
   - The document says MMR runs after the cross-encoder so it diversifies within accurately reranked results.
   - The implementation instead recomputes dense cosine relevance and ignores existing `rerank_score` values.
   - `_finalise()` then replaces the cross-encoder scores with cosine scores, destroying the evidence that reranking produced.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1395-1419, 1421-1464, 1496-1500`.

3. **Sparse queries use the document-side embedding path**
   - `HybridRetriever.embed_probe()` calls `embed_sparse([text])` for ordinary user queries.
   - Phase 3 explicitly distinguishes sparse query embedding from document embedding because their term-frequency treatment differs.
   - `BaseEmbeddingProvider` still lacks a sparse-query method, forcing Phase 4 to violate its own embedding contract.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:736-751`; `docs/Phase3_VectorStores_Embeddings.md:659-674`.

4. **The configured expansion model is never used**
   - `QueryExpander` stores `self.model`, but its `generate_json()` call does not pass that model.
   - Changing `EXPANSION_MODEL` therefore has no guaranteed effect on query expansion.
   - Either `generate_json()` must accept a model argument or the unused setting must be removed.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:327-335, 356-364`.

5. **Reranker fallback violates the base interface**
   - Phase 1’s `BaseReranker` contract says returned chunks must have `rerank_score` populated.
   - `CrossEncoderReranker` deliberately returns chunks with `rerank_score=None` when disabled or when scoring fails.
   - This violates substitutability: callers typed against `BaseReranker` cannot rely on its documented postcondition.
   - References: `docs/Phase1_System_Foundations.md:1650-1661`; `docs/Phase4_Hybrid_Retrieval_Reranking.md:1170-1178, 1182-1189, 1214-1228`.

6. **Cancellation can be swallowed as a partial retrieval failure**
   - `asyncio.gather(..., return_exceptions=True)` results are tested with `isinstance(item, BaseException)`.
   - This treats cancellation and other non-operational `BaseException` subclasses as failed search arms instead of propagating them.
   - Cancellation should be re-raised; only ordinary retrieval exceptions should be degraded.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:689-708`.

7. **Chroma can still bypass retrieval filters**
   - Phase 4 relies on every search enforcing `chunk_level=child` and future authorization filters.
   - The Phase 3 Chroma implementation silently drops range and list filters, so the shared retrieval contract is not behaviorally consistent across stores.
   - This becomes a security issue when Phase 11 filters carry tenant permissions.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:604-613, 761-776`; `docs/Phase3_VectorStores_Embeddings.md:1955-1971`.

## Pipeline and diagnostic problems

8. **HyDE retrieval is not concurrent with the other probe searches**
   - Expansion and HyDE generation run concurrently, but the pipeline completes all expanded-query searches before embedding and searching the HyDE probe.
   - This contradicts the “all probes concurrently” pipeline description and adds avoidable latency.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:97-107, 1846-1867`.

9. **Candidate diagnostics exclude HyDE**
   - HyDE results are appended to `result_lists`, but `RetrievalResult.total_candidates` and the completion log use only `outcome.total_candidates`.
   - Candidate counts are therefore wrong whenever HyDE succeeds.
   - HyDE failures are also absent from `failed_arms`.
   - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1854-1867, 1888-1904`.

10. **RRF mislabels dense-only results as hybrid**
    - `reciprocal_rank_fusion()` always sets `method=RetrievalMethod.HYBRID`.
    - This overwrites `DENSE` results produced by Chroma or by a dense-only fallback, making evaluation and diagnostics inaccurate.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:844-847, 903-916`.

11. **Partial failures are missing from the result model**
    - `SearchOutcome` tracks failed queries, but `RetrievalResult` has no degraded/failed-arm field.
    - The information survives only in logs, despite the document saying Phase 7 must know how many retrieval arms succeeded.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:638-652, 692-718, 1888-1904`.

12. **Blank or invalid queries are not rejected at the boundary**
    - An empty query reaches embedding/search, and the resulting failure may be reported as a retryable retrieval outage.
    - Input validation should distinguish an invalid request from an unavailable vector store.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:672-708, 1831-1844`.

## Reranking and context-budget issues

13. **The cross-encoder pair budget does not use the actual query length**
    - Passage truncation always reserves 96 tokens for the query.
    - Queries longer than that can still push the pair beyond the 512-token ceiling, causing additional hidden truncation.
    - The query should be measured and, if necessary, explicitly truncated before calculating the passage budget.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:997-1026`.

14. **The context budget knowingly permits an oversized first source**
    - `_apply_budget()` always keeps the first source, even when it alone exceeds `MAX_CONTEXT_TOKENS`.
    - The method therefore does not actually guarantee its stated token ceiling.
    - This should be represented explicitly as an overflow condition or handled by a defined truncation/error policy.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1636-1668`.

15. **The cross-encoder loads even when reranking is disabled**
    - `RetrievalPipeline` always constructs a `CrossEncoderReranker`.
    - `warmup()` loads it without checking `ENABLE_RERANKING`, wasting startup time and memory when the feature is disabled.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1824-1829, 1909-1913`.

16. **Optional-stage degradation is implementation-specific**
    - The pipeline accepts any `BaseReranker`, but only the concrete cross-encoder catches its own failures.
    - A different valid reranker can raise and abort retrieval, contradicting the phase-level promise that reranking always degrades gracefully.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1816-1828, 1872-1879`.

17. **Expansion’s “never raises” guarantee is incomplete**
    - `QueryExpander.expand()` and `HyDEGenerator.generate()` catch only `RAGException`.
    - Validation errors or unexpected provider exceptions can still escape unless every future LLM provider wraps every failure perfectly.
    - The failure-mode text specifically claims validation errors are caught, but that is not enforced by these functions.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:338-370, 443-445, 517-539`.

## Verification problems

18. **The default verification fails to exercise MMR**
    - `ENABLE_MMR` defaults to `false`.
    - `check_mmr()` constructs `MMRReranker` but does not override the global feature flag, so the method returns the first two near-duplicate candidates and the assertion fails.
    - The script acknowledges this after the code instead of making the test self-contained.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:158-162, 1388-1393, 2161-2186, 2308-2309`.

19. **The verification uses a destructive fixed collection name**
    - It always uses `verify_phase4_tmp` and unconditionally deletes it in `finally`.
    - Existing data with that collection name would be destroyed. A generated unique name should be used.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:2017-2018, 2293-2298`.

20. **Core optional paths are not verified**
    - Query expansion and HyDE are tested only while disabled.
    - There are no tests for partial search failure, cancellation, sparse query embedding, context-budget overflow, or the actual cross-encoder→MMR pipeline composition.
    - The direct MMR unit check does not reveal that MMR is a no-op in the real pipeline.
    - References: `docs/Phase4_Hybrid_Retrieval_Reranking.md:1990-2002, 2161-2186, 2238-2264`.

## Sound design decisions

- Searching child chunks and substituting parents only after reranking is the correct high-level order.
- RRF across query probes is appropriately separate from Qdrant’s dense/sparse fusion.
- Parent deduplication and missing-parent fallback are sensible degradation behavior.
- Preserving retrieval and reranking scores separately is useful for evaluation.
- Dependency injection and keeping generation outside retrieval preserve a clean evaluation boundary.

The highest-priority fixes are the cross-encoder/MMR composition, sparse-query interface, RRF method preservation, degraded-result diagnostics, and self-contained verification.
