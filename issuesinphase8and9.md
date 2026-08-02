# Phases 8 and 9 — Structural, API, Streaming, and UI Issues

Scope: code correctness, security boundaries, request isolation, async lifecycle, API/client contracts, UI behavior, and verification quality only. Library versions were not assessed.

## Phase 8 — Critical server issues

1. **Streaming traces are shared across every request**
   - `RAGAgent.last_trace` is mutable singleton state on the lifespan-scoped service.
   - Every normal or streaming query resets and updates the same list.
   - Concurrent users can receive each other’s progress events, and one request can erase another’s trace.
   - This is request-data leakage and control-flow corruption, not an authentication feature that can safely wait for Phase 11.
   - References: `docs/Phase8_FastAPI_Server.md:374-399, 423-449, 571-575`.

2. **The correlation ID returned to the client does not match request logs**
   - Middleware creates one request ID and later writes it to `X-Request-ID`.
   - Both query routes call `new_request_id()` again, replacing the context variable used by downstream logs and exception handlers.
   - A client reporting the response header therefore supplies an ID that does not identify the service logs for its query.
   - References: `docs/Phase8_FastAPI_Server.md:240-262, 332-348, 351-363, 640-652`.

3. **The lifespan cleanup is not protected by `finally`**
   - The guide claims resources are always released.
   - Cleanup is written after a bare `yield` rather than inside `try: yield ... finally:`.
   - If the lifespan body is exited by an exception, code after the yield is not a guaranteed cleanup path.
   - References: `docs/Phase8_FastAPI_Server.md:567-617`.

4. **Disconnected stream tasks are cancelled but never awaited**
   - On disconnect and in `finally`, `_events()` calls `task.cancel()` and returns.
   - It does not await the task and suppress `CancelledError`.
   - Provider, graph, and checkpointer cleanup may still be running, and unobserved task exceptions can surface later.
   - References: `docs/Phase8_FastAPI_Server.md:374-421`.

5. **The source-detail endpoint promised by the API schema does not exist**
   - `SourceOut` says Phase 9 expands excerpts through `/source/{id}`.
   - No source route is defined or registered.
   - Clients receive only the first 400 characters of a parent chunk, which may not contain the cited claim.
   - The stated “one-keystroke provenance” workflow is therefore impossible.
   - References: `docs/Phase8_FastAPI_Server.md:117-125, 285-421, 453-552, 654-657`; `docs/Phase9_CLI_Web_UI.md:23-25, 245-274, 668-682`.

6. **Unauthenticated localhost access is exposed to arbitrary local files**
   - CORS explicitly trusts origin `"null"`, which is used by pages opened from the local filesystem.
   - Any downloaded or malicious local HTML file can call query and admin endpoints.
   - `/admin/cache/clear` is a state-changing unauthenticated operation, while query responses expose confidential contract information.
   - Binding documentation to localhost does not make `"null"` a trusted origin.
   - References: `docs/Phase8_FastAPI_Server.md:65-70, 515-558, 574-575, 632-638`.

7. **Server binding is not an application-level security boundary**
   - Security relies on the example Uvicorn command using `127.0.0.1`.
   - The application itself has no authentication and can be launched on `0.0.0.0` by a different command or deployment configuration.
   - Admin and confidential query routes then become remotely accessible without warning or startup refusal.
   - References: `docs/Phase8_FastAPI_Server.md:65-70, 558-575, 664-668`.

8. **The readiness endpoint does not establish service readiness**
   - `/ready` checks only vector-store count.
   - It does not verify the required LLM provider, embedding provider, graph, or model availability.
   - It can report ready while every query fails during generation.
   - References: `docs/Phase8_FastAPI_Server.md:459-501`.

9. **The verification script still starts the real lifespan service**
   - Tests override the `get_service` dependency but use `TestClient` as a context manager.
   - Entering the context runs the app lifespan, which calls real `build_service()` before route dependency overrides matter.
   - The script’s claim that it needs no Qdrant or API key is therefore false.
   - A test lifespan override or application factory with injected lifespan is required.
   - References: `docs/Phase8_FastAPI_Server.md:591-606, 690-747, 858-863`.

## Phase 8 — API and operational problems

10. **Progress events discard the information the UI is supposed to show**
    - Trace entries include values such as `grading:fail`, `grading:pass`, `rewrite:1`, and `retrieval:5`.
    - The server sends only `entry.split(":")[0]`.
    - Failure/pass state, retry attempt, and result count are lost before either Phase 9 client receives them.
    - References: `docs/Phase8_FastAPI_Server.md:321-329, 388-395`; `docs/Phase9_CLI_Web_UI.md:19-21`.

11. **Polling `last_trace` can miss all progress**
    - If the task completes before the polling loop observes an intermediate state, the loop exits and only the final answer is sent.
    - Polling every 200 ms also delays or batches fast graph steps.
    - Progress should come from a request-scoped graph stream rather than a side-channel list.
    - References: `docs/Phase8_FastAPI_Server.md:300-304, 374-403`.

12. **Retry headers ignore provider retry timing**
    - Every retryable exception receives `Retry-After: 5`.
    - `RateLimitError` already carries the provider’s actual `retry_after`.
    - Replacing it with five seconds can cause clients to retry too early or wait unnecessarily.
    - References: `docs/Phase8_FastAPI_Server.md:240-262`.

13. **Deliberate 5xx errors may expose internal details**
    - The generic unexpected-error handler hides internals.
    - `RAGException` responses return `exc.to_dict()` unchanged even for 5xx failures.
    - Exception details can include model names, collection information, causes, or internal configuration.
    - The external error envelope needs a safe detail policy by status/environment.
    - References: `docs/Phase8_FastAPI_Server.md:240-280`.

14. **Validation errors use a different response contract**
    - FastAPI request validation failures are not handled by `register_error_handlers()`.
    - They use the framework’s default 422 body instead of the documented `{error, message, request_id, retryable}` envelope.
    - Phase 9’s error parser consequently displays an empty or generic message for malformed requests.
    - References: `docs/Phase8_FastAPI_Server.md:224-280, 794-802`; `docs/Phase9_CLI_Web_UI.md:160-173`.

15. **Streaming error events omit correlation metadata**
    - Once streaming starts, errors are sent as SSE data.
    - These events do not include the request ID that clients need to locate logs.
    - Unexpected stream errors also omit retryability and any stable status classification.
    - References: `docs/Phase8_FastAPI_Server.md:405-417, 640-652`.

16. **Admin statistics depend on private cache implementation details**
    - `/admin/stats` directly reads `service.cache._index` and `service.cache.exact`.
    - Replacing the cache through its abstraction can break the route even if caching still works.
    - The endpoint also exposes model IDs, feature flags, collection names, and corpus size without authentication.
    - References: `docs/Phase8_FastAPI_Server.md:528-551`.

17. **Cache clearing and statistics have no concurrency policy**
    - Cache clear can run while queries read and append semantic-cache entries.
    - No lock, generation/version swap, or consistency behavior is defined.
    - A clear may report success while concurrent requests immediately repopulate stale answers.
    - References: `docs/Phase8_FastAPI_Server.md:515-551`.

18. **Rate limiting is dismissed for the wrong reason**
    - Provider backoff handles upstream 429 responses after quota pressure occurs.
    - It does not prevent abusive clients from consuming LLM budget, occupying worker capacity, or repeatedly triggering expensive retries.
    - Even a localhost service exposed through CORS needs request concurrency and cost controls.
    - References: `docs/Phase8_FastAPI_Server.md:68-70`.

19. **The public response drops useful terminal diagnostics**
    - `QueryResponse` omits `failure_reason`, `thread_id`, invalid-citation count, and full citation metadata.
    - Status helps, but clients cannot distinguish reasons within a status or inspect/resume the graph thread.
    - References: `docs/Phase8_FastAPI_Server.md:136-178`.

20. **Request-ID middleware may omit the header on middleware-level failure**
    - The header is added only after `await call_next(request)` returns.
    - If middleware or response construction raises outside registered handlers, no response with the correlation header is produced.
    - A `try/finally`/error-response strategy is needed if the header is a universal contract.
    - References: `docs/Phase8_FastAPI_Server.md:640-652`.

## Phase 9 — CLI issues

21. **`main.py` catches `httpx.ConnectError` without importing `httpx`**
    - The module imports Typer and Rich but not `httpx`.
    - When a connection error occurs, evaluating the exception handler raises `NameError`, masking the intended friendly message and exit code.
    - References: `docs/Phase9_CLI_Web_UI.md:307-317, 362-369`.

22. **Most network failures escape the CLI’s error handling**
    - One-shot mode handles only `RAGAPIError` and `ConnectError`.
    - Timeouts, protocol errors, malformed SSE JSON, truncated responses, and decoding failures propagate as tracebacks.
    - The REPL catches only `RAGAPIError`, so a transient timeout can terminate the whole session.
    - References: `docs/Phase9_CLI_Web_UI.md:349-371, 374-397, 419-440`.

23. **SSE error status and retryability are discarded**
    - Stream errors are reconstructed as `RAGAPIError(500, ...)` regardless of their server type.
    - Retryable retrieval or rate-limit failures become generic non-retryable 500s.
    - References: `docs/Phase9_CLI_Web_UI.md:374-395`.

24. **The hand-written SSE parser does not implement event framing**
    - The CLI yields immediately for each `data:` line rather than accumulating all data lines until a blank frame boundary.
    - Multi-line SSE payloads are parsed as separate incomplete JSON documents.
    - Comments describe frame semantics that the implementation does not follow.
    - References: `docs/Phase9_CLI_Web_UI.md:114-138`.

25. **Health and stats clients ignore HTTP failure status**
    - `health()` and `stats()` call `.json()` without checking `response.status_code`.
    - A proxy HTML error raises JSON decoding errors, while an API error envelope is treated as successful data and later causes key errors.
    - References: `docs/Phase9_CLI_Web_UI.md:140-144, 444-458`.

26. **CLI source rendering accepts terminal markup from contracts**
    - Contract name, section title, and excerpt are passed to Rich as raw strings.
    - Rich interprets markup in strings, so untrusted document text can alter terminal formatting or inject terminal links/control-like output.
    - Use `Text` objects or disable markup for corpus-derived values.
    - References: `docs/Phase9_CLI_Web_UI.md:245-274`.

27. **Streaming can finish silently without an answer**
    - If the connection closes without an `answer` or `error` event, `_stream_with_progress()` returns `None`.
    - One-shot and chat modes then display nothing and treat the request as successful.
    - References: `docs/Phase9_CLI_Web_UI.md:374-397`.

28. **CLI type declarations disagree with their defaults**
    - Typer options `top_k: int` and `year: int` use `None` defaults.
    - The internal functions correctly type them as optional, but the command boundary does not.
    - This weakens static checking and can produce framework-specific option behavior.
    - References: `docs/Phase9_CLI_Web_UI.md:324-334`.

29. **The trace cannot show grading failure despite claiming it does**
    - Phase 8 sends only the stage name, so the CLI receives `"grading"` rather than `"grading:fail"`.
    - `render_trace()` can display suffixes only if they exist, but `_stream_with_progress()` appends the stripped server stage.
    - The main feature described as “the trace is the product” loses its most important event.
    - References: `docs/Phase9_CLI_Web_UI.md:19-21, 277-287, 374-397`.

## Phase 9 — Web dashboard issues

30. **Serving the dashboard from “any static server” conflicts with CORS**
    - The API allows only localhost/127.0.0.1 on port 8000 plus `"null"`.
    - A normal static server on port 8080, 3000, or another host origin is blocked.
    - The documented deployment options and actual server policy do not match.
    - References: `docs/Phase8_FastAPI_Server.md:632-638`; `docs/Phase9_CLI_Web_UI.md:483-486, 613-615`.

31. **The web dashboard also cannot display failed-grade trace steps**
    - `addStep()` checks `stage.includes("fail")`.
    - The server has already stripped `:fail` from the trace entry.
    - `.step-fail` styling is therefore unreachable.
    - References: `docs/Phase8_FastAPI_Server.md:388-395`; `docs/Phase9_CLI_Web_UI.md:585-600, 634-643`.

32. **The dashboard does not make sources inspectable**
    - Sources are static 400-character excerpts with no links or expansion.
    - The missing `/source/{id}` endpoint prevents users from verifying claims outside the excerpt.
    - This contradicts the phase’s primary grounding objective.
    - References: `docs/Phase9_CLI_Web_UI.md:23-25, 528-531, 668-682, 890-891`.

33. **The browser silently accepts a stream with no terminal event**
    - End-of-stream simply exits the read loop.
    - If no answer/error frame was received, the button is re-enabled without explaining that the request was incomplete.
    - The final partial frame is also discarded when the stream closes without a trailing blank line.
    - References: `docs/Phase9_CLI_Web_UI.md:698-741`.

34. **HTTP errors are misreported as server-unreachable errors**
    - Any non-2xx response becomes `Error("HTTP N")`.
    - The catch block appends “Is the API running?”, even for validation, prompt-injection, or rate-limit responses from a healthy API.
    - The dashboard does not parse the structured error envelope.
    - References: `docs/Phase9_CLI_Web_UI.md:703-737`.

35. **The web UI omits the actual grading verdict**
    - It shows only confidence in the metadata.
    - `grounded`, `relevant`, and `verified` are not rendered, despite being present in the API and central to the phase’s stated purpose.
    - Status is useful but does not explain which grading dimension failed.
    - References: `docs/Phase9_CLI_Web_UI.md:646-666`.

36. **Basic accessibility semantics are missing**
    - The question input has no associated label.
    - Dynamic answer, error, readiness, and progress regions have no `aria-live` status behavior.
    - Status is conveyed primarily through color and short text without announced updates for assistive technology.
    - References: `docs/Phase9_CLI_Web_UI.md:493-537, 622-696`.

## Verification gaps

37. **Phase 8 verification does not test the real components it can break**
    - It does not test lifespan construction/cleanup, concurrent stream isolation, disconnect cancellation, request-ID consistency, admin routes, CORS, validation-envelope consistency, or source expansion.
    - The test’s lifespan setup unintentionally requires the real service while claiming the opposite.
    - References: `docs/Phase8_FastAPI_Server.md:690-863`.

38. **The stream verification does not verify progress correctness**
    - It asserts only that the final answer event exists.
    - It does not assert progress events, ordering, grading failure/pass suffixes, retry traces, errors, or cancellation.
    - The shared-trace and stripped-suffix bugs therefore pass.
    - References: `docs/Phase8_FastAPI_Server.md:823-837`.

39. **Phase 9 verification never imports or exercises `main.py`**
    - The missing `httpx` import and command error handling are not tested.
    - No HTTP client request, SSE parser, timeout, truncated stream, exit code, or REPL continuation behavior is exercised.
    - References: `docs/Phase9_CLI_Web_UI.md:766-869`.

40. **The dashboard’s security and behavior are entirely manual**
    - There is no automated test for HTML escaping, structured API errors, incomplete streams, trace suffixes, CORS-compatible hosting, or DOM rendering.
    - The manual checklist checks only four happy or visible scenarios.
    - References: `docs/Phase9_CLI_Web_UI.md:766-880`.

41. **Cache hits can emit stale trace data from a previous request**
    - Phase 8 polls `service.agent.last_trace`, but Phase 6 performs guard and cache lookup before entering `RAGAgent.answer()`.
    - The trace is reset only inside the agent.
    - A cache hit never enters the agent at all, so a new streaming request can report progress left behind by an earlier run.
    - References: `docs/Phase8_FastAPI_Server.md:374-399, 423-448`.

42. **Inbound correlation IDs are ignored**
    - Middleware always generates a new ID and never accepts a trusted `X-Request-ID` supplied by a gateway or client.
    - This prevents tracing one request consistently across proxy and application boundaries.
    - If external IDs are accepted later, they must be validated before entering logs.
    - References: `docs/Phase8_FastAPI_Server.md:640-652`.

43. **Interactive API documentation is always exposed**
    - `/docs` is enabled in every environment.
    - If the application is accidentally bound publicly, it advertises unauthenticated admin and query operations.
    - Documentation and OpenAPI exposure should be environment-controlled until authentication exists.
    - References: `docs/Phase8_FastAPI_Server.md:619-630`.

44. **Whitespace-only questions pass the API schema**
    - `min_length=3` validates the untrimmed string.
    - A question containing only spaces can pass request validation and fail later inside the service with a different status/envelope.
    - Normalize and validate stripped content at the API boundary.
    - References: `docs/Phase8_FastAPI_Server.md:101-114`.

45. **`LOW_CONFIDENCE` is labelled with the word “unverified”**
    - The CLI correctly uses a different colour from `UNVERIFIED`, but labels low confidence as “unverified claims.”
    - This linguistically collapses “the audit ran and failed” with “the audit did not run,” precisely the distinction the phase says users must understand.
    - References: `docs/Phase9_CLI_Web_UI.md:185-206`.

46. **The default REPL ignores configurable API URLs**
    - Subcommands accept `--url`.
    - The no-subcommand callback always starts chat against hardcoded `http://127.0.0.1:8000`.
    - Users cannot configure the most common invocation path without explicitly selecting the `chat` subcommand.
    - References: `docs/Phase9_CLI_Web_UI.md:319-346, 461-465`.

47. **Non-answer statuses are not visually distinct**
    - `NO_MATCH` and `UNSUPPORTED` use the same CLI colour and the same web CSS style.
    - The narrative says answer statuses look different, but these two materially different outcomes are visually collapsed.
    - References: `docs/Phase9_CLI_Web_UI.md:197-206, 585-591, 893-895`.

## Sound design decisions

- Keeping API schemas separate from domain models prevents full chunk payloads from leaking by default.
- Streaming progress instead of ungraded answer tokens is the correct legal-domain trade-off.
- Separating liveness from readiness is operationally sound.
- Lifespan-scoped heavy resources are the right ownership model once cleanup is placed in `finally`.
- UI rendering uses `textContent` or escaped HTML for answer and source text, preventing straightforward stored XSS.
- Status-driven rendering preserves the distinction between low-confidence and unverified answers.
- The CLI uses the HTTP API instead of loading a second copy of the RAG stack.

The highest-priority fixes are request-scoped streaming state, correlation-ID consistency, guaranteed task/resource cleanup, removal of unauthenticated `"null"` CORS access, a real source-detail route, and robust CLI/SSE error handling.
