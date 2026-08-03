# Phase 10 — Testing and Verification Integrity Issues

Scope: test correctness, fixture isolation, async behavior, integration realism, CI reliability, cross-phase contracts, false positives, and coverage gaps only. Library versions were not assessed.

## Critical issues

1. **The “fake service” E2E tests still start the real application service**
   - `TestClient` is entered as a context manager, so the real application lifespan runs.
   - Lifespan calls `build_service()` before route dependency overrides are used.
   - Tests described as requiring no Qdrant or LLM therefore initialize the real embedder, vector store, Redis cache, and LLM provider.
   - A fake dependency does not replace lifespan construction; tests need an injected/no-op lifespan.
   - References: `docs/Phase10_Testing_Verification.md:1178-1223`; `docs/Phase8_FastAPI_Server.md:591-617`.

2. **The CI dependency installation does not match the declared dependency group**
   - Development dependencies are declared under `[dependency-groups] dev`.
   - CI installs `pip install -e ".[dev]"`, which selects a project optional extra, not a dependency group.
   - Unless a separate `[project.optional-dependencies].dev` also exists, pytest and related tools are not installed.
   - References: `docs/Phase10_Testing_Verification.md:97-106, 1313-1327, 1338-1350`.

3. **CI invokes tools that are not declared in the shown development dependencies**
   - The dev group lists pytest, pytest-asyncio, pytest-cov, and httpx.
   - CI runs `ruff` and `mypy`.
   - A clean runner can fail with missing commands before any test runs.
   - References: `docs/Phase10_Testing_Verification.md:99-106, 1321-1323`.

4. **Integration regressions are converted into skipped tests**
   - The Qdrant fixture catches every initialization exception and calls `pytest.skip()`.
   - That includes real code bugs, schema errors, incompatible configuration, and connection races—not only an intentionally absent local service.
   - The Redis fixture similarly skips when the cache’s fail-open connection helper returns `None`.
   - CI can therefore report success while all real integration checks were skipped.
   - References: `docs/Phase10_Testing_Verification.md:460-488, 503-518`.

5. **CI services have no readiness checks**
   - Qdrant and Redis service containers are declared without health checks or an explicit wait.
   - Tests can start while services are still booting.
   - Combined with skip-on-connection-error fixtures, startup races turn the entire integration job into a green run with no integration coverage.
   - References: `docs/Phase10_Testing_Verification.md:1329-1353`.

6. **The semantic-cache veto test imports from the wrong module**
   - `semantically_incompatible` belongs to the semantic cache.
   - `test_retrieval.py` imports it from `src.retrieval.rerankers.mmr`.
   - The test module fails at import unless an unrelated duplicate function exists.
   - References: `docs/Phase10_Testing_Verification.md:755-766, 850-859`; `docs/Phase6_LLM_Cache_Guardrails.md:1789-1800`.

7. **The advertised concurrency E2E test is sequential and asserts no isolation**
   - It issues one blocking `client.post()` and only then issues the second.
   - It asserts only that both returned an answer event.
   - It neither overlaps requests nor inspects progress traces, so shared `last_trace` corruption passes.
   - The prose even admits the test passes because requests are serialized, contradicting its name and purpose.
   - References: `docs/Phase10_Testing_Verification.md:1270-1287`.

8. **The manual evaluation gate is not wired into CI**
   - The workflow exposes `workflow_dispatch`, but every trigger runs only unit/integration pytest jobs.
   - No step invokes Phase 7’s evaluation or regression comparison.
   - Calling the evaluation gate “manual” does not implement it.
   - References: `docs/Phase10_Testing_Verification.md:1292-1354`.

## False-positive and incomplete tests

9. **The implementation-completeness contract test does not detect abstract subclasses**
   - It subtracts `set(dir(cls))` from the ABC’s abstract method names.
   - Inherited abstract methods still appear in `dir(cls)`, so an incomplete abstract implementation reports no missing names.
   - The test must check `inspect.isabstract(cls)`, inspect method overrides, or instantiate the class.
   - References: `docs/Phase10_Testing_Verification.md:560-586`.

10. **Signature contract tests check only selected parameter names**
    - They do not compare concrete implementation signatures, parameter order, defaults, keyword-only behavior, annotations, or return types.
    - A provider can expose `model` while remaining incompatible with callers.
    - The “frozen interface surface” is therefore only partially executable.
    - References: `docs/Phase10_Testing_Verification.md:539-612`.

11. **The MMR score-preservation test never compares scores**
    - Candidate rerank scores are populated before the call.
    - The assertion checks only that returned scores are non-`None`.
    - MMR can overwrite every cross-encoder score with different values and the test still passes.
    - References: `docs/Phase10_Testing_Verification.md:798-811`.

12. **The original cross-encoder→MMR composition bug is not tested**
    - A unit test confirms MMR is a no-op when candidate count equals `top_k`.
    - No test runs `RetrievalPipeline` and verifies that reranking supplies a wider pool before MMR selects the final results.
    - The exact pipeline bug motivating the test suite can therefore return unchanged.
    - References: `docs/Phase10_Testing_Verification.md:813-820`.

13. **Parent deduplication is tested with a duplicate object, not sibling children**
    - The fixture creates only one child per parent.
    - The test passes the same child twice, then a child from another parent.
    - It proves exact duplicate removal but not the intended behavior where two distinct sibling children map to one parent.
    - References: `docs/Phase10_Testing_Verification.md:191-216, 823-832`.

14. **Generation-outage test passes when no exception is raised**
    - The test handles `LLMProviderError` and rejects `MaxRetriesExceededError`.
    - It has no `else` branch that fails if `agent.answer()` returns normally.
    - A swallowed provider outage therefore makes the test pass.
    - References: `docs/Phase10_Testing_Verification.md:945-960`.

15. **The lexical-arm test does not prove sparse retrieval contributed**
    - It asks for a phrase present verbatim in one fixture and asserts that result ranks first.
    - Dense retrieval can produce the same ranking without a sparse query.
    - The test must isolate/disable one arm or compare dense-only against hybrid behavior.
    - References: `docs/Phase10_Testing_Verification.md:1055-1063`.

16. **Initialize-idempotency test does not prove data preservation**
    - It calls `initialize()` twice on an empty temporary collection and asserts only that no exception occurs.
    - A destructive implementation that recreates the collection would pass.
    - The test should index data, initialize again, and assert exact count/content remains.
    - References: `docs/Phase10_Testing_Verification.md:1007-1012`.

17. **Agent tests may load the real reranker**
    - `no_llm_features` disables multi-query and HyDE only.
    - It does not disable cross-encoder reranking, while `RetrievalPipeline` constructs the production reranker by default.
    - Tests described as millisecond, fully fake runs can download/load and execute a real model.
    - References: `docs/Phase10_Testing_Verification.md:413-418, 911-943`.

18. **The deterministic embedder has severe accidental collisions**
    - Vectors depend only on the sum of character code points.
    - Anagrams and many unrelated strings produce identical vectors, while related strings are not intentionally controlled.
    - Cache/MMR tests can pass for properties of the hash collision rather than the algorithm.
    - References: `docs/Phase10_Testing_Verification.md:284-310`.

## Fixture isolation and lifecycle problems

19. **The Redis fixture can delete real application cache entries**
    - It connects using the normal configured Redis URL.
    - Setup and teardown call `cache.clear()`, which removes every project answer key under the production-style prefix.
    - Running tests against a developer or shared Redis instance can destroy legitimate cached answers.
    - Tests need a dedicated database, random namespace, or disposable container.
    - References: `docs/Phase10_Testing_Verification.md:503-518`.

20. **The real embedding provider is never closed**
    - `real_embedder` is session-scoped and returned directly.
    - No fixture finalizer calls `close()`.
    - Resource lifecycle contracts are not exercised and long-lived test processes retain model resources.
    - References: `docs/Phase10_Testing_Verification.md:448-457`.

21. **Settings isolation watches only a handpicked subset**
    - The snapshot omits settings such as cache status/threshold, provider selection, LLM retry values, prompt-injection flags, and model IDs.
    - Tests mutating an unwatched setting can still contaminate later tests.
    - Clearing `get_settings()` does not replace modules that imported the separate singleton `settings`.
    - References: `docs/Phase10_Testing_Verification.md:388-410`.

22. **Foreign Redis test data may survive a failed assertion**
    - `test_clear_is_scoped` creates `someone-elses-key`.
    - Cleanup occurs after assertions rather than in `finally`.
    - A failure leaves the key behind in the selected Redis database.
    - References: `docs/Phase10_Testing_Verification.md:1145-1153`.

23. **TTL verification is timing-sensitive**
    - The test sleeps 1.5 seconds for a one-second TTL.
    - CI scheduling, Redis expiry timing, and clock boundaries can make the test flaky or unnecessarily slow.
    - Polling with a bounded deadline or inspecting TTL behavior is more robust.
    - References: `docs/Phase10_Testing_Verification.md:1130-1135`.

24. **Standalone verification scripts now depend on the test tree**
    - The guide says verify scripts run on corpus machines without development dependencies.
    - It also requires them to import `tests/fakes.py`.
    - Test packages are often excluded from production installs, and the shown `tests/` tree has no explicit `__init__.py`, making import resolution environment-dependent.
    - Shared builders belong in an installable support module if standalone scripts require them.
    - References: `docs/Phase10_Testing_Verification.md:57-67, 150-152, 1371-1379`.

## Missing behavioral coverage

25. **Chroma behavior is not integration-tested**
    - Contract tests only check method presence.
    - All real vector-store tests run against Qdrant.
    - Filter dropping, deletion semantics, missing-ID behavior, and dense-only method labeling can regress in Chroma undetected.
    - References: `docs/Phase10_Testing_Verification.md:580-586, 965-1088`.

26. **Provider retry and error mapping are not tested**
    - There are no tests for zero retries, retry counts, cancellation propagation, rate-limit timing, final-error classification, malformed structured output, or OpenAI/Groq exception wrapping.
    - These paths produced several of the earlier structural bugs but are absent from the suite.
    - References: `docs/Phase10_Testing_Verification.md:41-55, 523-668`.

27. **Strict-schema tests omit nested models**
    - The only schema assertion uses `GradingReport`, whose fields are top-level.
    - It does not verify recursive strictness for nested `$defs`, arrays of models, unions, or excluded nested fields.
    - The incomplete nested-schema transformation can still reach production.
    - References: `docs/Phase10_Testing_Verification.md:633-638`.

28. **API streaming failure and cancellation paths are untested**
    - E2E checks do not assert progress ordering, error events, request IDs, disconnect cancellation, missing terminal events, or task cleanup.
    - The known shared-trace defect is documented rather than tested.
    - References: `docs/Phase10_Testing_Verification.md:1178-1287`.

29. **Request-ID consistency is not tested**
    - The API creates IDs in middleware, routes, and the agent.
    - Tests do not compare `X-Request-ID` with the ID visible inside logs/error payloads.
    - A header-presence assertion would not catch correlation drift.
    - References: `docs/Phase10_Testing_Verification.md:1226-1247`; `docs/Phase8_FastAPI_Server.md:332-363, 640-652`.

30. **Security boundaries have almost no tests**
    - There are no tests for CORS `"null"` access, unauthenticated admin operations, document-side prompt-injection neutralization, HTML/terminal escaping, PII policy, or cache storage of source text.
    - These are the highest-impact failures in Phases 6–9.
    - References: `docs/Phase10_Testing_Verification.md:41-55, 965-1287`.

31. **Phase 7 regression integrity is not tested or invoked**
    - No tests cover category regressions, case-error thresholds, judge-failure sample counts, baseline promotion safety, cache-disabled evaluation, or production-vs-judge latency.
    - CI does not execute the regression harness even on manual dispatch.
    - References: `docs/Phase10_Testing_Verification.md:1292-1354`.

32. **No assertion prevents all integration tests from being skipped**
    - CI does not fail on unexpected skips or publish a required count of executed integration tests.
    - A broken service container or fixture can yield a green job with zero real Qdrant/Redis assertions.
    - References: `docs/Phase10_Testing_Verification.md:460-518, 1329-1353`.

33. **Static checking excludes the tests**
    - CI runs `mypy src config`, not `mypy tests`.
    - Fake providers/stores, fixture signatures, optional command arguments, and misuse of ABCs are not type-checked.
    - Many of the suite’s own contract errors can therefore survive the static-checking stage.
    - References: `docs/Phase10_Testing_Verification.md:1321-1327`.

## Additional confirmed issues

34. **The context-budget test never reaches the budget logic**
    - `test_budget_drops_whole_sources` passes parent chunks directly to `ParentSubstituter`.
    - The substituter returns early when there are no child `parent_id` values, before applying its context budget.
    - The expected `len(result) < len(parents)` assertion therefore fails instead of testing whole-source dropping.
    - It should pass child results whose parents are fetched, then apply a deliberately tight budget.
    - References: `docs/Phase10_Testing_Verification.md:840-847`; `docs/Phase4_Hybrid_Retrieval_Reranking.md:1715-1726`.

35. **The documented fast command still selects E2E tests**
    - `pytest -m "not integration"` excludes only integration-marked tests.
    - E2E tests carry a separate `e2e` marker, so the command still runs them.
    - Because the E2E lifespan currently builds the real service, the advertised no-service path can require Qdrant, Redis, models, and API credentials.
    - References: `docs/Phase10_Testing_Verification.md:1181-1192, 1361-1368`.

36. **The shared test doubles do not implement the frozen contracts**
    - `StubEmbedder` lacks the documented `embed_dense_sync` surface.
    - `FakeStore` omits operations such as exact filtered counting/bulk behavior used by real adapters.
    - None of the doubles subclasses its ABC, so contract drift is discovered only when a particular test happens to call the missing member.
    - This weakens the claim that one shared fixture location makes construction errors impossible.
    - References: `docs/Phase10_Testing_Verification.md:230-363, 589-600`.

37. **The context-budget test also uses an unsuitable corpus shape**
    - Each synthetic parent has only one child, so parent expansion and sibling consolidation are not realistically exercised.
    - Even after changing the test to pass children, the fixture should include multiple children under at least one parent to validate expansion, deduplication, ordering, and budget behavior together.
    - References: `docs/Phase10_Testing_Verification.md:191-216, 823-847`.

## Sound design decisions

- Prioritizing seam and integration tests over low-value getter tests matches the project’s actual bug history.
- Unique temporary Qdrant collection names avoid destructive collisions.
- Scripted LLM responses are appropriate for deterministic graph-control tests.
- Real Qdrant and Redis tests are valuable when service unavailability fails CI rather than skips it.
- Contract tests for required methods and per-call model selection are directionally useful.
- Keeping answer-quality evaluation in Phase 7 avoids flaky semantic assertions in pytest.
- Scoped cache clearing is safer than flushing the entire Redis database, though tests still need their own namespace.

The highest-priority fixes are a fake/injectable API lifespan, correct dependency installation, fail-not-skip integration fixtures in CI, a real concurrent streaming test, repaired false-positive assertions, and provider/security coverage.
