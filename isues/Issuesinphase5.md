# Phase 5 — Structural and Code Integrity Issues

Scope: code correctness, state-machine integrity, cross-phase contracts, and verification quality only. Library versions and external API compatibility were not assessed.

## Critical issues

1. **The current Phase 6 provider cannot be injected into this graph**
   - Phase 5 imports Phase 1's asynchronous `BaseLLMProvider`, awaits `generate()` and `generate_json()`, passes Pydantic model classes, and expects structured model instances in return.
   - The current Phase 6 guide creates a second `BaseLLMProvider` in `src/llm/base.py` with synchronous methods, different signatures, no `stream()` or `close()`, and dictionary-shaped JSON results.
   - Consequently, the promised “Phase 6 substitutes Groq and changes nothing else” handoff does not work structurally.
   - Phase 6 must implement the existing Phase 1 interface instead of redefining it.
   - References: `docs/Phase5_LangGraph_Agent.md:8-12, 144-147, 998-1027, 1144-1189`; `docs/Phase1_System_Foundations.md:1694-1728`; `docs/Phase6_LLM_Cache_Guardrails.md:38-59, 67-116`.

2. **The configured task-specific models are never selected**
   - The document says generation uses the large model and grading uses the smaller `GRADER_MODEL`.
   - `RAGAgent` injects the same provider into routing, generation, grading, and rewriting.
   - `generate_answer()` does not pass `GENERATION_MODEL`, while `generate_json()` has no model argument at all.
   - Router, grader, and rewriter therefore use whichever model the provider was constructed with; the stated cost and quality separation is not implemented.
   - References: `docs/Phase5_LangGraph_Agent.md:1123-1126, 1584-1587, 1641-1645`; `docs/Phase1_System_Foundations.md:500-515, 1697-1716`.

3. **A grader outage creates a logically passing grade**
   - On a grading exception, the fallback report sets both `is_grounded=True` and `is_relevant=True`, so `GradingReport.passed` is `True`.
   - Its `confidence=0.0` bypasses the confidence-floor logic because the function returns before that check.
   - `after_grading()` accepts the answer, and `RAGAgent.answer()` appends a warning only when `passed` is false.
   - Code that checks only `grading.passed` will therefore treat an unaudited answer as verified.
   - The result needs an explicit `unverified` status, or the fallback must fail the quality gate without triggering pointless query rewriting.
   - References: `docs/Phase5_LangGraph_Agent.md:1127-1130, 1190-1208, 1471-1506, 1774-1781`; `docs/Phase1_System_Foundations.md:1387-1391`.

4. **The documented validation-error fallback is not enforced**
   - Router, grading, and rewriting catch only `RAGException`.
   - A Pydantic `ValidationError` raised while validating malformed structured output is not inherently a `RAGException`.
   - The guide explicitly says malformed grader output fails open, but such an error can escape the graph unless every provider wraps it correctly.
   - The provider contract must guarantee wrapping, or nodes must catch validation errors explicitly.
   - References: `docs/Phase5_LangGraph_Agent.md:793-807, 1114-1116, 1182-1203, 1293-1295, 1352-1366`.

5. **Generation outages are misclassified as retry-budget exhaustion**
   - `generate_answer()` catches `LLMProviderError`, clears the answer, and stores a string `failure_reason`.
   - `after_grading()` deliberately does not retry generation failures.
   - The facade then raises `MaxRetriesExceededError` solely because the final answer is empty, even though no retry budget was exceeded.
   - This destroys the original retryability/status semantics and will produce the wrong API response.
   - A generation failure should preserve and surface an LLM-specific failure type or structured terminal status.
   - References: `docs/Phase5_LangGraph_Agent.md:1022-1035, 1164-1175, 1489-1493, 1764-1772`.

6. **Retrieval outages are returned as ordinary answers without a machine-readable status**
   - A `RetrievalError` is converted to the text “The search service is currently unavailable.”
   - `failure_reason` remains only in internal graph state and is omitted from `RAGAnswer`.
   - No-match, unsupported aggregation, and service outage results all have no sources, citations, or grading; callers must parse English answer text to distinguish them.
   - This contradicts the code comments describing an outage as retryable and makes correct HTTP mapping impossible.
   - References: `docs/Phase5_LangGraph_Agent.md:861-871, 1681-1694, 1796-1805`; `docs/Phase1_System_Foundations.md:1394-1406`.

7. **Retry retrieval reuses the original expansion decision**
   - The router runs once and writes `expand`.
   - A rewritten query changes `current_query` but never recalculates `expand`.
   - If a specific original query skipped expansion and then failed grading, its corrective rewrite also skips multi-query expansion and HyDE—even though the failure is evidence that broader retrieval may be needed.
   - Rewrites should reroute, or retries should use an explicit expansion policy.
   - References: `docs/Phase5_LangGraph_Agent.md:724-760, 842-860, 1342-1391, 1509-1513`.

8. **Checkpoint threads are neither resumed nor safely restarted**
   - `answer()` always calls `ainvoke()` with a fresh `initial_state`, even when the caller supplies an existing `thread_id`.
   - Correct continuation of an interrupted checkpoint requires an explicit resume path rather than supplying a new initial input.
   - Reusing a thread ID for a new question can also merge with prior checkpoint state; reduced fields such as `trace` and `attempts` can retain old entries.
   - The automatically generated thread ID is not returned in `RAGAnswer`, so callers cannot later request a resume.
   - This does not satisfy the stated crash-resume design.
   - References: `docs/Phase5_LangGraph_Agent.md:42-44, 365-370, 1537-1542, 1719-1752`.

9. **An empty retrieval during correction destroys the previous usable answer**
   - On the first attempt, routing empty retrieval to `no_context` is correct.
   - After a generated answer fails grading, however, the rewrite may produce an empty result or retrieval error.
   - `after_retrieval()` still routes to `_no_context_node()`, which overwrites the prior answer and citations instead of returning that answer with its failed grade and caveat.
   - This directly contradicts the stated policy of preserving a partially supported answer when correction cannot improve it.
   - The retry path needs a distinct terminal branch that preserves the best prior attempt.
   - References: `docs/Phase5_LangGraph_Agent.md:154-166, 842-871, 1457-1468, 1681-1694, 1777-1781`.

10. **`MaxRetriesExceededError` has conflicting cross-phase semantics**
   - Phase 1 defines it as exhaustion without a passing grade.
   - Phase 5 deliberately returns a caveated answer when the retry budget is exhausted with a failing grade, and reserves the exception for runs with no answer.
   - Callers written to the Phase 1 contract will therefore expect an exception in a case Phase 5 returns normally.
   - The shared exception contract must be changed or Phase 5 needs a different error/status model.
   - References: `docs/Phase1_System_Foundations.md:963-970`; `docs/Phase5_LangGraph_Agent.md:154-166, 1727-1733, 1764-1781`.

## Routing and state integrity problems

11. **Aggregate-marker matching uses unsafe substring tests**
   - `_looks_aggregate()` checks `marker in lowered`.
   - The marker `"count"` also matches unrelated words such as `"account"`, `"discount"`, and `"counterparty account"`.
   - The broad `"how many"` marker also rejects ordinary passage questions such as “how many days notice is required,” a failure the guide acknowledges.
   - Markers need token/word-boundary matching and ambiguous cases should go to classification rather than immediate rejection.
   - References: `docs/Phase5_LangGraph_Agent.md:710-715, 732-739, 763-765, 816-820`.

12. **The section-reference regex does not reliably match the advertised `§` form**
    - `_SECTION_REF` places `\b` before an alternation containing `§`.
    - `§` is a non-word character, so `\b` normally does not match before it when it follows whitespace or starts the query.
    - Queries such as `§ 3.1(b)` can therefore miss the cheap “specific” route despite being listed as a supported example.
    - References: `docs/Phase5_LangGraph_Agent.md:701-708, 768-780`.

13. **`attempts` does not contain generate-and-grade attempt records**
    - The state comment promises one record per “generate-and-grade attempt.”
    - Records are appended during generation before grading and contain no grade, confidence, pass/fail result, or unsupported claims.
    - Only the final grade remains in `state["grading"]`, so Phase 7 cannot reconstruct why earlier attempts failed.
    - References: `docs/Phase5_LangGraph_Agent.md:365-370, 1051-1064, 1182-1237`.

14. **Successful citation extraction is not a citation-integrity gate**
    - Out-of-range source markers are dropped, but the answer remains unchanged and may still be graded as passing.
    - Answers with no valid citations can also be accepted.
    - The guide says the invalid-marker count is available to Phase 7, but it exists only as a log field and is not stored in state or `RAGAnswer`.
    - If strict validation is intentionally deferred to Phase 6, Phase 5 should at least preserve citation errors as structured diagnostics.
    - References: `docs/Phase5_LangGraph_Agent.md:912-914, 954-995, 1037-1064, 2302-2304`.

15. **Public inputs are not validated**
    - `answer()` accepts blank questions and arbitrary `top_k` values.
    - `top_k or settings.RERANK_TOP_K` silently converts `0` to the default while allowing negative values through.
    - Invalid input can consequently be reported as retrieval or agent failure rather than a request-validation error.
    - References: `docs/Phase5_LangGraph_Agent.md:1719-1742`.

16. **The rewrite documentation and implementation disagree**
    - `_is_progress()` says it enforces a four-word minimum.
    - The implementation rejects only queries shorter than two words.
    - Two-word degenerate rewrites therefore pass a condition the documentation claims should reject them.
    - References: `docs/Phase5_LangGraph_Agent.md:1394-1403`.

17. **The grading prompt has no independent token budget**
    - Phase 4 budgets retrieved source text for generation.
    - Grading sends the same full context plus the original question and the complete generated answer.
    - A generation request that fits near the model limit can therefore overflow during grading and take the exception fallback—which currently marks the result as passed.
    - The grader input needs its own measured budget.
    - References: `docs/Phase5_LangGraph_Agent.md:536-545, 1177-1189`; `docs/Phase4_Hybrid_Retrieval_Reranking.md:1636-1668`.

## Verification problems

18. **The scripted end-to-end tests can pass with incorrect retrieval**
    - `ScriptedLLM` returns queued answers and grades without inspecting the prompts or retrieved context.
    - The happy and retry tests assert that an answer and citation exist, but do not verify that `[SOURCE 1]` is the clause supporting the answer.
    - A pipeline retrieving the wrong section can therefore pass the complete verification.
    - References: `docs/Phase5_LangGraph_Agent.md:1946-1990, 2097-2142`.

19. **The central checkpoint/resume claim is completely untested**
    - Verification constructs `RAGAgent` without a checkpointer in every test.
    - It does not test persistence, repeated thread IDs, interruption, resumption, reducer behavior after resume, or generated thread-ID visibility.
    - Checkpointing is one of the main reasons given for using LangGraph, so omitting it leaves the highest-risk state behavior unverified.
    - References: `docs/Phase5_LangGraph_Agent.md:1535-1542, 1598-1599, 2097-2209`.

20. **Router-to-expansion integration is not actually exercised**
    - The verification pipeline is created with `llm=None`.
    - Although the graph router may set `expand=True`, Phase 4's expander and HyDE generator cannot perform LLM expansion.
    - Tests therefore do not prove that routing controls both expansion probes as intended.
    - References: `docs/Phase5_LangGraph_Agent.md:218-240, 2097-2104, 2216-2231`.

21. **Important failure branches have no verification**
    - There is no test for grader exceptions, malformed JSON, low-confidence passing grades, no-progress rewrites, retrieval outages, invalid inputs, or reused checkpoint threads.
    - In particular, a grader exception test would expose the `passed=True, confidence=0.0` inconsistency.
   - `check_aggregate_routing()` is satisfied by the `"how many"` heuristic, so the LLM-returned aggregate branch is also untested.
    - References: `docs/Phase5_LangGraph_Agent.md:2059-2209`.

22. **Verification does not close the embedding provider**
    - The vector store is closed in `finally`, but `FastEmbedProvider` is not.
    - This violates the provider lifecycle contract and can leave resources alive after failed verification.
    - References: `docs/Phase5_LangGraph_Agent.md:2216-2245`; `docs/Phase1_System_Foundations.md:1690-1691`.

## Sound design decisions

- Keeping retrieval intelligence inside Phase 4 avoids duplicated orchestration.
- Numbering prompt sources and constructing the citation mapping in the same loop prevents provenance drift.
- Generating only when sources exist is a strong safety boundary.
- Answering the original question while using rewritten queries only for retrieval is correct.
- Pure conditional-edge functions make the retry policy easy to unit-test.
- A semantic retry budget plus an independent graph recursion limit provides useful defense in depth.
- Returning a caveated low-quality answer can be preferable to discarding it, provided “failed,” “unverified,” and infrastructure-error states remain distinct.

The highest-priority fixes are the Phase 6 interface mismatch, real task-specific model selection, explicit unverified grading status, correct terminal error propagation, checkpoint/resume semantics, and retry expansion policy.
