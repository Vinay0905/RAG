# Phase 7 — Structural and Evaluation-Integrity Issues

Scope: code correctness, metric validity, reproducibility, data leakage, cross-phase contracts, and verification quality only. Library versions were not assessed.

## Critical evaluation-integrity issues

1. **Evaluation runs through the production answer cache**
   - `scripts/evaluate.py` builds the normal `RAGService`, whose exact cache is enabled by default.
   - Previously evaluated questions can therefore skip retrieval, generation, grading, and self-correction.
   - Repeated runs may measure old cached outputs rather than the code or configuration under test, and latency becomes cache latency instead of pipeline latency.
   - Evaluation should explicitly disable/clear all answer caches or use a cache-free service.
   - References: `docs/Phase7_LLMOps_Evaluation.md:818-835`; `docs/Phase6_LLM_Cache_Guardrails.md:199-203, 1560-1607, 2526-2538`.

2. **Reported latency includes the evaluation judges**
   - `_one()` starts its timer before `service.answer()`.
   - It records elapsed time only after groundedness and relevance have each made another LLM call.
   - `latency_p50_ms` and `latency_p95_ms` therefore measure production latency plus evaluation overhead.
   - This makes the proposed latency comparisons—such as whether reranking earns 20 ms—invalid.
   - References: `docs/Phase7_LLMOps_Evaluation.md:633-664, 680-681, 881-889`.

3. **The regression gate ignores per-category regressions**
   - The evaluator correctly computes category breakdowns.
   - `compare()` checks only `summary["overall"]`.
   - A serious drop for vague queries can be hidden by stable specific-query performance, despite the document saying category breakdown is where the real finding lives.
   - References: `docs/Phase7_LLMOps_Evaluation.md:33-35, 601-602, 667-689, 754-770, 1062-1064`.

4. **Case errors cannot fail regression**
   - Aggregation records an error count but excludes errored cases from all means.
   - `compare()` checks neither `errors` nor an error-rate threshold.
   - A run where many difficult cases crash can retain high scores on the surviving easy cases and pass CI.
   - References: `docs/Phase7_LLMOps_Evaluation.md:633-643, 667-683, 754-770`.

5. **Judge failures silently change the evaluated sample**
   - Failed answer metrics return `-1.0`, and aggregation excludes those values.
   - The summary records no per-metric valid count or failure rate.
   - If the judge fails preferentially on difficult cases, the remaining mean is biased upward while appearing directly comparable to the baseline.
   - If every judgement fails, the metric becomes `0.0`, conflating “not measured” with a genuine zero score.
   - References: `docs/Phase7_LLMOps_Evaluation.md:553-558, 586-588, 667-698`.

6. **Answer evaluation uses the same grading model as the system being evaluated**
   - Phase 5 grades answers with `GRADER_MODEL`.
   - Both Phase 7 metrics also pass `settings.GRADER_MODEL`.
   - The “ideally a different model” mitigation is not implemented, so correlated model errors and preferences can make the evaluator agree with the system rather than independently assess it.
   - References: `docs/Phase7_LLMOps_Evaluation.md:29-31, 479-485, 543-552, 579-585`; `docs/Phase5_LangGraph_Agent.md:1246-1251`.

7. **Runs do not record enough configuration to be reproducible**
   - Saved summaries contain metrics, categories, latency, errors, and a timestamp.
   - They omit the git commit, model IDs, prompt version, feature flags, retrieval settings, dataset hash, corpus/index version, seed, and judge configuration.
   - A baseline cannot prove it used the same dataset or identify what system configuration produced its values.
   - `git log` cannot reconstruct environment variables that were never saved.
   - References: `docs/Phase7_LLMOps_Evaluation.md:81-82, 667-727, 813-815`.

8. **Missing baselines are created automatically**
   - `save_baseline()` says promotion is deliberate and never automatic.
   - The run script automatically saves the current summary when no baseline exists.
   - A broken, empty, or partially failed first run can silently become the standard against which future runs are judged.
   - References: `docs/Phase7_LLMOps_Evaluation.md:773-785, 845-852`.

9. **`--promote` can promote invalid runs**
   - Promotion does not reject empty datasets, case errors, failed judges, missing metrics, or non-`ANSWERED` results.
   - Any completed summary can overwrite the baseline.
   - A baseline update needs explicit quality and completeness checks before it becomes authoritative.
   - References: `docs/Phase7_LLMOps_Evaluation.md:667-698, 845-847`.

## Dataset-generation problems

10. **The promised multi-hop category is never generated**
    - The design claims three categories: `specific`, `vague`, and `multi_hop`.
    - `_STYLES` defines only specific and vague, and generation alternates only those two.
    - Every case records exactly one source chunk ID.
    - Recall for genuine multi-hop retrieval is therefore never evaluated.
    - References: `docs/Phase7_LLMOps_Evaluation.md:130-132, 193-195, 227-233, 276-292`.

11. **“Append-safe” dataset persistence overwrites the file**
    - The guide describes JSONL as append-only and says sets may be built over weeks.
    - `save_cases()` opens the path with mode `"w"`, replacing all existing cases.
    - Running `--generate` again destroys the prior evaluation set.
    - References: `docs/Phase7_LLMOps_Evaluation.md:100-102, 150-155, 821-825`.

12. **The fixed seed does not produce the same evaluation set**
    - The seed controls only shuffling of the sampled window.
    - Questions are regenerated by an LLM and `case_id` uses random `uuid.uuid4()`.
    - Re-running generation can therefore produce different questions and IDs despite the claim that the same corpus produces the same set.
    - References: `docs/Phase7_LLMOps_Evaluation.md:202-203, 256-263, 276-286, 319-332`.

13. **Generation may silently return fewer cases than requested**
    - Only twice the requested number of sampled chunks are attempted.
    - Rejected or failed questions are skipped, and the method returns whatever remains without enforcing `len(cases) == count`.
    - Small or category-skewed sets can then be evaluated and promoted without warning.
    - References: `docs/Phase7_LLMOps_Evaluation.md:265-299`.

14. **Duplicate questions are not detected**
    - Multiple chunks can produce the same generic question, especially governing-law or notice-period prompts.
    - No normalization or duplicate check exists.
    - Duplicate cases overweight common clauses and reduce the effective sample size while the summary still reports the raw case count.
    - References: `docs/Phase7_LLMOps_Evaluation.md:269-299, 319-337`.

15. **Sampling is a deterministic ID-order window, not a representative corpus sample**
    - `scroll_sample()` reads only the first `limit` points returned in collection order.
    - Shuffling that window cannot include contracts outside it.
    - Scores can therefore reflect one narrow collection region while being used to tune global retrieval behavior.
    - References: `docs/Phase7_LLMOps_Evaluation.md:301-317, 340-365`.

16. **Synthetic generation bypasses document-injection defenses**
    - Raw contract text is interpolated directly into the question-generation prompt.
    - An adversarial clause can instruct the generator to create biased or malformed evaluation cases.
    - This enables corpus content to manipulate the benchmark used to approve retrieval changes.
    - References: `docs/Phase7_LLMOps_Evaluation.md:319-326`; `docs/Phase6_LLM_Cache_Guardrails.md:1794-1819`.

17. **Evaluation data persists confidential contract text**
    - Every `EvalCase` stores the full source passage, and the guide describes the JSONL as greppable and diffable in git.
    - Evaluation runs also persist questions and generated answers.
    - This creates another uncontrolled copy of potentially confidential contract data without a redaction, access-control, or retention policy.
    - References: `docs/Phase7_LLMOps_Evaluation.md:95-102, 120-133, 709-725`.

18. **Dataset validation is too permissive**
    - `category` is an unrestricted string, `source_chunk_ids` may be empty, questions have no minimum validation, and duplicate `case_id` values are allowed.
    - Malformed lines are skipped, but a completely empty or heavily corrupted dataset is still accepted.
    - Such cases produce zeros or missing categories that look like system quality failures.
    - References: `docs/Phase7_LLMOps_Evaluation.md:112-147, 158-175`.

## Metric correctness and contract problems

19. **Metric failure sentinels violate `BaseMetric`**
    - Phase 1 requires `score()` to return a value in `[0.0, 1.0]`.
    - Both answer metrics return `-1.0` on judge failure.
    - Callers typed against `BaseMetric` cannot safely assume the documented range.
    - Failure should be represented separately from a valid score.
    - References: `docs/Phase1_System_Foundations.md:1863-1873`; `docs/Phase7_LLMOps_Evaluation.md:539-558, 576-589`.

20. **Parent/child equivalence works only in one direction**
    - `_ids()` adds a retrieved chunk’s own ID and its `parent_id`.
    - It can score a retrieved child against an expected parent.
    - It cannot score a returned parent against an expected child, despite the docstring claiming that child-origin cases still work.
    - The verification test exercises only the working direction.
    - References: `docs/Phase7_LLMOps_Evaluation.md:451-470, 947-955`.

21. **Groundedness can award a perfect score to a judge failure**
    - `total_claims == 0` always returns `1.0`.
    - The metric does not verify that the answer is actually a refusal or contains no factual claims.
    - A judge that fails to identify claims in a substantive answer produces perfect groundedness.
    - References: `docs/Phase7_LLMOps_Evaluation.md:513-516, 539-563`.

22. **Groundedness accepts logically inconsistent verdicts**
    - The schema independently validates non-negative `total_claims` and `supported_claims`.
    - It does not enforce `supported_claims <= total_claims`.
    - The scorer hides invalid verdicts with `min(..., 1.0)` instead of rejecting them, allowing malformed judgements to look perfect.
    - References: `docs/Phase7_LLMOps_Evaluation.md:513-516, 560-563`.

23. **The relevance judge cannot apply its own refusal rule**
    - The relevance prompt says an honest refusal scores 1.0 only if the sources genuinely lack the answer.
    - `RelevanceMetric` sends only the question and answer, not the sources.
    - The judge has no evidence with which to determine the stated condition.
    - References: `docs/Phase7_LLMOps_Evaluation.md:502-510, 576-585`.

24. **The harness cannot identify which retrieval stage lost recall**
    - Retrieval metrics inspect only `RAGAnswer.sources`, the final post-fusion, post-reranking, post-MMR, post-parent-substitution, context-budgeted list.
    - No pre-rerank candidates, fusion output, failed arms, or stage-specific ranks are recorded in `EvalResult`.
    - This cannot answer the phase’s stated question “which stage is losing recall.”
    - References: `docs/Phase7_LLMOps_Evaluation.md:33-35, 136-147, 645-663`.

25. **Per-case results omit the expected sources**
    - `EvalResult` stores retrieved IDs but not `source_chunk_ids` or source provenance.
    - Once the cases file changes, a saved run cannot independently explain whether a miss was correct.
    - This contradicts the claim that per-case results make regressions explainable.
    - References: `docs/Phase7_LLMOps_Evaluation.md:136-147, 658-664, 709-725`.

26. **A custom metric exception aborts the whole run**
    - Only `service.answer()` is inside `_one()`’s `try/except`.
    - Metric calls happen afterward without isolation.
    - Any `BaseMetric` implementation that raises causes `asyncio.gather()` to fail and loses all results, contradicting “failures are recorded, not raised.”
    - References: `docs/Phase7_LLMOps_Evaluation.md:604-605, 629-656`.

27. **Concurrency is not validated**
    - `concurrency=0` creates a semaphore that never releases a permit, hanging the evaluation.
    - Negative values are also invalid.
    - Constructor validation should require at least one worker.
    - References: `docs/Phase7_LLMOps_Evaluation.md:621-630`.

28. **Percentile calculation is incorrect for small and even-sized samples**
    - `_percentile()` selects `int(n * fraction)` with no interpolation or nearest-rank adjustment.
    - For two values, p50 returns the larger value rather than their median.
    - Reported latency summaries can therefore be misleading on small category/run samples.
    - References: `docs/Phase7_LLMOps_Evaluation.md:701-706`.

## Regression and verification gaps

29. **One loose tolerance is applied to deterministic and stochastic metrics**
    - Retrieval metrics are deterministic, but their regressions may drop by up to 0.03 without failing.
    - The rationale explicitly chooses simplicity over sensitivity.
    - A meaningful deterministic retrieval regression can therefore pass because answer-judge noise dictated the tolerance.
    - References: `docs/Phase7_LLMOps_Evaluation.md:736-751, 754-770`.

30. **Status regressions are not compared**
    - `EvalResult` records statuses such as answered, unverified, low-confidence, and no-match.
    - Aggregation and regression do not report or compare their distribution.
    - A grader outage that changes many answers to `UNVERIFIED` can pass if independent answer judges still score the text highly.
    - References: `docs/Phase7_LLMOps_Evaluation.md:136-147, 658-689, 754-770`.

31. **Evaluation shutdown leaves cache resources open**
    - The script closes the LLM, vector store, and embedder.
    - It does not close the exact Redis cache owned by the service.
    - Repeated or failed runs can leak the cache connection pool.
    - References: `docs/Phase7_LLMOps_Evaluation.md:818-867`; `docs/Phase6_LLM_Cache_Guardrails.md:1647-1650`.

32. **Saved run filenames can collide**
    - Filenames contain timestamps only to the second.
    - Two runs saved within one second overwrite the same file.
    - A UUID, higher-resolution timestamp, or collision check is needed for append-safe experiment history.
    - References: `docs/Phase7_LLMOps_Evaluation.md:709-725`.

33. **Verification omits the parts most likely to invalidate conclusions**
    - It does not test answer metrics, judge failures, async evaluator behavior, cache disabling, measured latency boundaries, category regression, error-rate regression, baseline promotion safety, multi-hop generation, duplicate cases, or sampling behavior.
    - Its parent/child test exercises the opposite direction from the failure described in the metric docstring.
    - References: `docs/Phase7_LLMOps_Evaluation.md:899-1043`.

34. **Cross-phase observability contracts are not consumed**
    - Earlier phases expose or promise `PROMPT_VERSION`, `RetrievalResult.failed_arms`, retrieval latency, `ScoredChunk.rerank_score`, grading confidence/verification, answer warnings, and cache-hit state specifically for evaluation.
    - `EvalResult` and the run summary record none of them.
    - The harness therefore cannot measure degraded retrieval arms, reranker score changes, grader availability, citation warnings, or cache contamination despite those fields existing for Phase 7.
    - References: `docs/Phase7_LLMOps_Evaluation.md:136-147, 645-689, 709-727`.

35. **Feature ablations are confounded by self-correction**
    - Retrieval metrics are computed from the final `RAGAnswer.sources`.
    - With self-correction enabled, a failed first answer can trigger query rewriting and a second retrieval, so an experiment intended to isolate reranking, HyDE, or MMR also measures the correction loop’s response.
    - Retrieval-stage experiments need a retrieval-only path or must explicitly disable correction.
    - References: `docs/Phase7_LLMOps_Evaluation.md:645-663, 881-889`.

36. **The cache behavior introduced in Phase 6 is not evaluated**
    - The introduction names semantic-cache threshold selection as a guess Phase 7 should replace with evidence.
    - The experiment matrix contains no cache hit-rate, false-hit, cross-scope, threshold, or cache-on/off experiment.
    - Consequently, the riskiest Phase 6 feature receives no shipping criterion.
    - References: `docs/Phase7_LLMOps_Evaluation.md:15-20, 876-889`.

37. **The context-dependence filter is incomplete relative to its own prompt**
    - The generation prompt forbids “this agreement.”
    - The deterministic filter contains only `"this agreement's"` and misses the ordinary phrase `"this agreement"`.
    - Other common references such as “the contract” or “herein” also pass, creating questions that cannot stand alone.
    - References: `docs/Phase7_LLMOps_Evaluation.md:213-242, 334-337`.

38. **Empty context conflates retrieval failure with answer groundedness**
    - `GroundednessMetric` returns `0.0` whenever contexts are empty.
    - This score says nothing about whether a returned no-match explanation is factually grounded; it primarily records that retrieval returned nothing.
    - Overall groundedness therefore mixes retrieval availability with generation faithfulness.
    - References: `docs/Phase7_LLMOps_Evaluation.md:539-563`.

39. **Concurrent evaluation can create cache cross-hits within one run**
    - Five cases execute concurrently through the shared service.
    - If semantic caching is enabled, completed cases are inserted into the shared in-process index while other cases are still running.
    - Near-duplicate synthetic questions can then answer one another from cache, making results order- and scheduling-dependent.
    - References: `docs/Phase7_LLMOps_Evaluation.md:598-630`; `docs/Phase6_LLM_Cache_Guardrails.md:1878-1888`.

## Sound design decisions

- Separating deterministic retrieval metrics from LLM-judged answer metrics is correct.
- Keeping per-case results alongside aggregates is the right direction, once expected-source provenance is included.
- Bounded case concurrency is appropriate for rate-limited providers.
- Explicitly documenting synthetic-data bias prevents absolute scores from being overclaimed.
- Prompt-versioned, reproducible evaluation datasets are the right goal.
- Refusing to average failed judge calls as valid negative quality scores is sensible, but failure counts must remain visible.
- Deliberate baseline promotion is the correct policy once automatic creation and promotion validation are fixed.

The highest-priority fixes are disabling caches during evaluation, separating production latency from judge latency, gating errors and category regressions, recording full run provenance, generating real multi-hop cases, and representing metric failures outside the valid score range.
