# Phase 5 — The Self-Correcting LangGraph Agent

> **Prerequisite:** Phases 1, 3, and 4 complete. This phase consumes
> `RetrievalPipeline.retrieve()` and nothing below it.
>
> **Budget:** ~1,250 lines of Python across 9 files. (Budgeted at 900; the calibration note holds.)
>
> **This phase needs a real LLM, and Phase 6 is not written yet.** Phase 4's optional-LLM trick does
> not extend here — a generation node with no model has nothing to do. The resolution is in §2, and it
> is not "wait for Phase 6": every node is defined against Phase 1's `BaseLLMProvider`, and the
> verification script ships a scripted fake provider so the whole graph is testable **now**. Phase 6
> substitutes Groq for the fake and changes nothing else.
>
> **API verified 2026-08-01** against the current LangGraph docs. This library churns more than
> anything else in the stack, so §3 lists exactly what was checked.
>
> **This file replaces the previous `Phase5_LangGraph_Agent.md` entirely.** The old draft used
> `StateGraph(dict)`, which discards the only reason to use a typed state machine.

---

## 1. What Makes This Phase Hard

Phases 3 and 4 built retrieval. Feeding the top 5 passages to an LLM with "answer using this context"
is about fifteen lines, and it works most of the time. This phase is about the rest of the time.

**The LLM will answer confidently from context that does not support the answer.** This is the
failure that matters in a legal setting. Ask about a liability cap, retrieve a clause that mentions
one without stating it, and the model will produce a number — plausible, specific, and invented.
Nothing in the retrieval stack detects this, because retrieval did its job: the passage *is* about
liability caps.

**Retrieval failure and generation failure look identical from the outside.** A wrong answer might
come from bad retrieval (the right clause was never fetched) or bad generation (it was fetched and
misread). A pipeline that returns a string cannot distinguish them, so it cannot fix either.

**"Just retry" needs somewhere to put the retry.** If the answer is ungrounded, the sensible response
is to rewrite the query and search again. That makes control flow a **loop with a conditional exit**,
and a loop that can run forever if the exit condition never fires. Loops need budgets, and budgets
need state.

**State has to survive the process.** Phase 8 will stream this over HTTP and Phase 9 will show a
progress trace. A local variable in a function cannot be inspected mid-flight, resumed after a
crash, or streamed to a browser.

So the shape of this phase is a **state machine with a cycle**: retrieve → generate → grade → and
either finish or rewrite and go round again, with a bounded budget and inspectable state at every
step.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| 1 | `src/graph/state.py` | 130 | The typed state that flows through the machine |
| 2 | `src/graph/prompts.py` | 190 | Every prompt string, versioned, in one place |
| 3 | `src/graph/nodes/router_node.py` | 120 | Decide how much machinery this query needs |
| 4 | `src/graph/nodes/retrieval_node.py` | 90 | Call Phase 4, record diagnostics |
| 5 | `src/graph/nodes/generation_node.py` | 170 | Context block, answer, citations |
| 6 | `src/graph/nodes/grading_node.py` | 140 | Grounded? Relevant? → `GradingReport` |
| 7 | `src/graph/nodes/rewriting_node.py` | 110 | Turn a failed grade into a better query |
| 8 | `src/graph/edges.py` | 130 | The routing functions — where control flow lives |
| 9 | `src/graph/builder.py` | 170 | Wire it, compile it, expose `RAGAgent` |

### Directory to create

```text
src/graph/
├── __init__.py
├── state.py
├── prompts.py
├── edges.py
├── builder.py
└── nodes/
    ├── __init__.py
    ├── router_node.py
    ├── retrieval_node.py
    ├── generation_node.py
    ├── grading_node.py
    └── rewriting_node.py
```

**Two nodes from the roadmap's tree are deliberately absent: `expansion_node.py` and
`reranking_node.py`.** The roadmap was written before Phase 4, which put query expansion and
reranking inside `RetrievalPipeline` — where they belong, because they are retrieval concerns and
Phase 7 needs to evaluate them without a graph. Re-exposing them as graph nodes would duplicate that
orchestration in two places, and every extra node is another checkpoint write. What the graph keeps
is the *decision*: `router_node` chooses whether expansion happens, and passes that down. Update the
roadmap tree.

### The machine

```text
                          START
                            │
                            ▼
                     ┌─────────────┐
                     │   router    │  cheap classification: does this query
                     └──────┬──────┘  need expansion? is it even answerable
                            │         by top-k retrieval?
              ┌─────────────┼──────────────┐
         "retrieve"    "retrieve"      "unsupported"
        (expand=False) (expand=True)        │
              └─────────────┬───────────────┘         │
                            ▼                         │
                     ┌─────────────┐                  │
                     │  retrieval  │  Phase 4         │
                     └──────┬──────┘                  │
                            │                         │
                  ┌─────────┴─────────┐               │
              no sources           sources            │
                    │                 │               │
                    │                 ▼               │
                    │          ┌─────────────┐        │
                    │          │ generation  │        │
                    │          └──────┬──────┘        │
                    │                 ▼               │
                    │          ┌─────────────┐        │
                    │          │   grading   │        │
                    │          └──────┬──────┘        │
                    │                 │               │
                    │      ┌──────────┼──────────┐    │
                    │   passed    failed +      failed │
                    │      │      budget left   + no   │
                    │      │          │         budget │
                    │      │          ▼           │    │
                    │      │   ┌─────────────┐    │    │
                    │      │   │  rewriting  │    │    │
                    │      │   └──────┬──────┘    │    │
                    │      │          │           │    │
                    │      │          └───────────┼────┼──► back to retrieval
                    ▼      ▼                      ▼    ▼
                            END
```

The cycle is `retrieval → generation → grading → rewriting → retrieval`. That single back-edge is
the whole point of the phase, and it is why this is a graph rather than a sequence.

---

## 2. Before Any Code — Three Decisions and Some Settings

### Decision 1: how this phase exists before Phase 6

Every node takes a `BaseLLMProvider` — Phase 1's interface, which exists — and never imports a
concrete provider. The graph is assembled by `build_agent(llm=..., pipeline=...)`. Phase 6 will pass
`GroqProvider()`; the verification script in §13 passes a `ScriptedLLM` that returns canned
responses on cue.

That is not a workaround, it is how the whole system is wired, and it buys something specific: the
grader's retry edge can be tested by *scripting a failing grade*, which is nearly impossible to
trigger reliably with a real model. Testing the unhappy path is the only reason the unhappy path
works.

### Decision 2: the graph must not raise when it produces a bad answer

The instinct is to raise `MaxRetriesExceededError` when the retry budget runs out with a failing
grade. I am not doing that, and this is worth arguing because Phase 1 defined the exception.

A user who asked a question and got a partially-supported answer is better served by **the answer,
plus the grading report, plus an explicit caveat** than by an HTTP 500. The grading is already
structured data; handing it to the caller lets Phase 8 return a 200 with a `grading` block and lets
Phase 9 render a warning banner. Throwing away a usable answer to signal a quality problem is the
wrong trade.

`MaxRetriesExceededError` is therefore reserved for the case where **no answer exists at all** — the
loop terminated without generation ever succeeding. That is a genuine failure with nothing to return.

### Decision 3: two independent loop budgets, and why both

- **`MAX_RETRIES` (ours, semantic).** "Stop after 2 self-correction attempts." Lives in the state,
  drives the grading edge, and is the number a product owner would tune.
- **`recursion_limit` (LangGraph's, structural).** A hard ceiling on total super-steps. Default 1000.
  This is not redundant: it catches a *bug in our own edges* — a routing function that returns the
  wrong node name and creates a cycle our counter never sees. A semantic budget cannot protect
  against a mistake in the thing that enforces it.

We set `recursion_limit` explicitly to a small number derived from `MAX_RETRIES`, so an edge bug
fails in a second instead of after a thousand LLM calls. §11 shows the arithmetic.

### New settings

Add to the **RAG hyperparameters** block:

```python
    ENABLE_SELF_CORRECTION: bool = Field(
        default=True,
        description="Grade answers and retry with a rewritten query on failure.",
    )
    ENABLE_QUERY_ROUTING: bool = Field(default=True)
    GRADER_MIN_CONFIDENCE: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this, a passing grade is treated as a non-verdict and not trusted.",
    )
    CHECKPOINT_DB_PATH: str = Field(default="./data/graph_checkpoints.sqlite")
```

Add to `validate_runtime()`:

```python
        if not self.ENABLE_SELF_CORRECTION:
            warnings.append(
                "Self-correction is disabled — ungrounded answers will be returned unflagged."
            )
```

And `.env.example`:

```bash
# ─── Phase 5: agent ────────────────────────────────────────────────────────
ENABLE_SELF_CORRECTION=true
ENABLE_QUERY_ROUTING=true
GRADER_MIN_CONFIDENCE=0.5
CHECKPOINT_DB_PATH=./data/graph_checkpoints.sqlite
```

### One addition to Phase 4

`router_node` decides whether to expand, so `RetrievalPipeline.retrieve` needs to accept that
decision. Add the parameter (I have updated the Phase 4 guide to match):

```python
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        expand: bool = True,          # ← new
    ) -> RetrievalResult:
        ...
        queries, hypothetical = await asyncio.gather(
            self.expander.expand(query) if expand else _just(query),
            self.hyde.generate(query) if expand else _none(),
        )
```

where `_just` / `_none` are trivial coroutines returning `[query]` and `None`. The reason to pass a
flag rather than have the caller skip the expander is that skipping happens in *two* places
(expansion and HyDE) and a caller that remembers one and forgets the other gets a confusing latency
profile.

### Dependencies

```bash
langgraph
langgraph-checkpoint-sqlite    # SqliteSaver / AsyncSqliteSaver
```

---

## 3. What Was Verified, and What Changed in LangGraph

Checked against the current documentation on 2026-08-01. Re-verify if months have passed — this is
the fastest-moving dependency in the project.

| Item | Current form |
| :--- | :--- |
| Imports | `from langgraph.graph import END, START, StateGraph` |
| State | `StateGraph(AgentState)` where `AgentState` is a `TypedDict` |
| Nodes | `builder.add_node("name", fn)`; `fn` may be `async def` |
| Edges | `builder.add_edge(START, "router")`, `builder.add_edge("x", END)` |
| Conditional | `builder.add_conditional_edges(source, path_fn, path_map)` — `path_map` optional, maps the function's return value to a node name |
| Compile | `builder.compile(checkpointer=...)` → `CompiledStateGraph` |
| Run | `await graph.ainvoke(state, config)`; stream with `graph.astream(...)` |
| Checkpointer | `AsyncSqliteSaver.from_conn_string(path)` — an **async context manager** |
| Thread | `config={"configurable": {"thread_id": "..."}}` when a checkpointer is set |
| Recursion | `config={"recursion_limit": N}` — a **top-level** config key, *not* inside `configurable`. Default 1000. Raises `GraphRecursionError` from `langgraph.errors` |
| Reducers | `Annotated[list[str], operator.add]` in the state class |

Three things worth flagging because they are easy to get wrong:

**`recursion_limit` is not inside `configurable`.** Every other user-supplied config key is. Put it in
the wrong place and it is silently ignored, leaving you on the default of 1000.

**Checkpoints are written at super-step boundaries, not inside a node.** If execution stops and
resumes, the interrupted node **re-runs from the start of its function**. Side effects before the
pause happen twice. For us that means a resumed run can pay for the same generation call twice — worth
knowing before you build a UI that resumes threads.

**A node returns a partial state update, not the whole state.** Returning `{"answer": "..."}` merges
that one key. Returning the entire state dict works but is noise, and for keys with reducers it is
actively wrong — a reducer *appends*, so returning the accumulated list re-appends all of it.

---

## 4. File 1 — `src/graph/state.py`

### The Problem

Seven nodes need to share data. Node 3 writes `sources`; node 5 reads it. Node 6 writes `grading`;
an edge function reads it to decide where to go.

The old draft used `StateGraph(dict)`. That compiles and runs, and it throws away everything a state
machine is for: no autocomplete, no type checking, no documentation of what exists, and a typo like
`state["retrival"]` becomes a `KeyError` at runtime inside a node — or worse, a silent `None` from a
`.get()`.

There is also a subtler problem. Most state keys should be **overwritten** by whichever node writes
them last: `answer` is just the latest answer. But some should **accumulate**: the trace of which
nodes ran is a list that every node appends to. LangGraph needs to be told which is which.

### Design Decision

**A `TypedDict` with `Annotated` reducers where accumulation is wanted.**

By default LangGraph overwrites a key with whatever a node returned. Annotating a key with a reducer
function changes that: `Annotated[list[str], operator.add]` means returned lists are concatenated
onto the existing value instead of replacing it. That is exactly right for a trace, and exactly wrong
for `answer`.

**Every key is declared, and `initial_state()` populates all of them.** LangGraph tolerates missing
keys, but a node that does `state["retry_count"] + 1` on a key nothing has written yet raises
`KeyError` from inside the graph, where the traceback is least helpful. Building a complete initial
state costs nine lines and removes an entire class of failure.

**Domain objects live in the state, not dicts.** `retrieval: RetrievalResult | None`,
`grading: GradingReport | None`, `sources: list[ScoredChunk]`. The state is a transport, not an excuse
to go back to untyped data.

```python
import operator
from typing import Annotated, Any, TypedDict

from src.core.models import Citation, GradingReport, RetrievalResult, ScoredChunk


class AgentState(TypedDict):
    """State passed between every node in the graph.

    Keys without a reducer are OVERWRITTEN by whichever node returns them.
    Keys annotated with a reducer are combined — see `trace` and `attempts`.

    Nodes return a PARTIAL dict containing only the keys they changed. Returning
    the whole state is noise for plain keys and a bug for reduced keys, because a
    reducer appends: returning the accumulated trace would append it a second time.
    """

    # ── input, fixed for the run ───────────────────────────────────────────
    query: str
    filters: dict[str, Any] | None
    top_k: int

    # ── router output ──────────────────────────────────────────────────────
    route: str
    expand: bool

    # ── the query actually being searched; rewriting replaces it ───────────
    current_query: str

    # ── retrieval output ───────────────────────────────────────────────────
    retrieval: RetrievalResult | None
    sources: list[ScoredChunk]

    # ── generation output ──────────────────────────────────────────────────
    answer: str
    citations: list[Citation]
    #: Markers the model emitted for sources it was never given. Carried out to
    #: `RAGAnswer` rather than only logged — partly fabricated provenance is a
    #: property of the answer, not a debugging detail.
    invalid_citations: int

    # ── grading output ─────────────────────────────────────────────────────
    grading: GradingReport | None

    # ── loop control ───────────────────────────────────────────────────────
    retry_count: int

    # ── accumulated across the whole run ───────────────────────────────────
    #: Node-by-node breadcrumbs. `operator.add` concatenates, so each node
    #: returns only its own entry and the list grows.
    trace: Annotated[list[str], operator.add]
    #: One record per generate-and-grade attempt, for the API's debug view.
    attempts: Annotated[list[dict[str, Any]], operator.add]

    # ── terminal state ─────────────────────────────────────────────────────
    #: Set when the run ends without a usable answer. Distinct from a low grade:
    #: this means we could not answer at all, not that we answered poorly.
    failure_reason: str | None


def initial_state(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> AgentState:
    """Build a complete initial state.

    Every key is present, so no node has to defend against a missing one. The
    graph's entry conditions are visible in one place rather than distributed
    across seven `.get()` calls with seven different defaults.
    """
    return AgentState(
        query=query,
        filters=filters,
        top_k=top_k,
        route="retrieve",
        expand=True,
        current_query=query,
        retrieval=None,
        sources=[],
        answer="",
        citations=[],
        invalid_citations=0,
        grading=None,
        retry_count=0,
        trace=[],
        attempts=[],
        failure_reason=None,
    )
```

### The Theory: reducers, and the bug they exist to prevent

Without reducers, two nodes writing the same key means the second wins. That is usually what you
want, and it is why LangGraph made it the default.

Now consider the trace. Every node wants to record that it ran. Without a reducer:

```python
# router returns   {"trace": ["router"]}
# retrieval returns {"trace": ["retrieval"]}
# final state:      {"trace": ["retrieval"]}     ← the router's entry is gone
```

The obvious fix is for each node to read the trace, append, and return the whole list:

```python
return {"trace": state["trace"] + ["retrieval"]}
```

That works, and it is fragile in a specific way: it only works while nodes run **strictly in
sequence**. LangGraph runs nodes in parallel when several are reachable in one super-step, and two
parallel nodes that both read-modify-write the same list produce a lost update — classic
read-modify-write contention, in a framework that gives you no lock.

A reducer moves the combination into the framework, which applies it to every returned update
regardless of concurrency. That is why `Annotated[list, operator.add]` is not a convenience: it is the
only correct way to accumulate.

The corresponding trap, which follows directly: **with a reducer, never return the accumulated
value.** `{"trace": state["trace"] + ["x"]}` on a reduced key appends the whole history again, and the
trace grows quadratically. Return only your own contribution.

### Failure Modes

**`KeyError: 'retry_count'` inside a node.** State was built by hand instead of with
`initial_state()`. This is the failure that function exists to prevent.

**The trace contains repeated entries with growing length.** You returned the accumulated list on a
reduced key. Return only the new element.

**The answer from attempt 1 survives into the final state after attempt 2.** Only if generation
returned early without writing `answer`. Plain keys are overwritten, so a node that skips writing
leaves the previous value in place — which is occasionally what you want and is worth being deliberate
about.

**A `RetrievalResult` in the state breaks the SQLite checkpointer.** It should not: Pydantic models
serialise through LangGraph's default serialiser. If you add something exotic (an open client, a
lambda), checkpointing fails at the super-step boundary, not at assignment. Keep state to data.

---

## 5. File 2 — `src/graph/prompts.py`

### The Problem

Prompts are the highest-churn, highest-impact strings in the system, and they are invisible if they
live inline. Scattered across three node files, you cannot diff them, cannot tell which version
produced last week's evaluation numbers, and cannot review a change to the grading criteria without
reading generation code.

### Design Decision

**One module, module-level constants, an explicit version string.** Phase 7 records
`PROMPT_VERSION` alongside every evaluation result, so a change in scores can be attributed to a
prompt change rather than guessed at.

**The citation protocol is a contract, stated in the prompt and parsed in code.** The generator is
told to write `[SOURCE 1]`; the node parses exactly that. Both halves live one file apart and the
format appears in the prompt, the parser regex, and the verification — three places that must agree,
so the prompt is written to make the format hard to deviate from.

**The grounding instruction is extractive, not evaluative.** "Say only what these sources say" is a
constraint a model can follow. "Be accurate" is not.

```python
#: Bump on any change to a prompt below. Phase 7 records this with every
#: evaluation run so a score change can be attributed rather than guessed at.
PROMPT_VERSION = "v1.0"


GENERATION_SYSTEM = """You answer questions about legal contracts using ONLY the sources provided.

Rules, in order of importance:

1. Use only the numbered sources. Do not use general legal knowledge, do not
   infer what a contract "probably" says, and do not fill gaps.
2. Cite every factual claim inline as [SOURCE 1], [SOURCE 2]. A sentence stating
   a fact with no citation is a defect.
3. If the sources do not contain the answer, say exactly what is missing. A clear
   "the provided sources do not state the notice period" is a correct and useful
   answer. A guess is not.
4. Quote exact figures, dates, and defined terms verbatim. Never round, never
   paraphrase a number, never normalise a defined term's capitalisation.
5. Be concise. Two or three sentences unless the question needs more.

You are not giving legal advice. You are reporting what these documents say."""


GENERATION_USER = """Sources:

{context}

Question: {question}

Answer using only the sources above, citing each claim as [SOURCE N]."""


GRADING_SYSTEM = """You audit answers for factual grounding. You are strict and mechanical.

You receive a question, an answer, and the sources the answer was supposedly
based on. Decide two things independently:

is_grounded — Is EVERY factual claim in the answer supported by the sources?
  A claim is supported only if a source states it. A claim that is true in general
  but absent from the sources is NOT grounded. Numbers, dates, and party names
  must match exactly. List every unsupported claim you find.

is_relevant — Does the answer address the question that was asked?
  An answer that is well-grounded but about a different provision is not relevant.
  An honest "the sources do not state this" IS relevant, and IS grounded — it
  claims nothing.

confidence — How certain are you of this verdict, 0.0 to 1.0? Use a low value if
  the sources are ambiguous or the answer is hard to check.

Judge only grounding and relevance. Not style, not completeness, not tone."""


GRADING_USER = """Question: {question}

Answer to audit:
{answer}

Sources the answer must be grounded in:

{context}

Return your verdict as JSON."""


REWRITE_SYSTEM = """You rewrite failed search queries for a legal contract database.

A previous search retrieved passages that could not support a grounded answer.
Your job is to write a query that will retrieve BETTER passages.

Use the audit feedback to diagnose the failure:
- Unsupported claims name what was missing. Target those specifics.
- If the retrieved sources were about the wrong provision, use the drafted legal
  term rather than the user's phrasing: "termination for convenience", not
  "cancel early".
- If they were about the right provision but lacked detail, get more specific:
  name the sub-clause, the defined term, the section reference.

Rules:
- Output a search query, not a question to a person.
- Do not restate the previous query. If you cannot improve on it, make it more
  specific rather than returning it unchanged.
- Do not invent parties, dates, or amounts.
- Return only the JSON object."""


REWRITE_USER = """Original question: {question}

Query that was searched: {previous_query}

Why the answer failed the audit:
{reasoning}

Claims that were not supported by the retrieved sources:
{unsupported}

Write an improved search query."""


ROUTER_SYSTEM = """You classify questions about a legal contract database.

Return exactly one label:

"specific"  — Names a section, a defined term, or an exact phrase. The user's own
              words are already good search terms.
              e.g. "what does Section 7.02 say", "define Permitted Encumbrance"

"vague"     — Asks in everyday language about a concept the contract would express
              in formal legal terms. Needs rephrasing to retrieve well.
              e.g. "can I get out of this early", "who pays if we get sued"

"aggregate" — Requires counting, comparing, or summarising ACROSS many documents
              rather than finding a passage.
              e.g. "how many contracts are governed by Delaware law"

Return only the JSON object."""


ROUTER_USER = """Question: {question}

Classify it."""


#: Appended when the answer was audited and FAILED — the retry budget ran out and
#: the grader still found unsupported claims. The user gets the answer AND the
#: warning; see the argument in §2.
LOW_CONFIDENCE_CAVEAT = (
    "\n\n---\n**Note:** parts of this answer could not be verified against the "
    "retrieved sources. Treat the specifics as unconfirmed and check the cited "
    "documents directly."
)

#: Appended when the audit did not RUN — grader outage, malformed output, or
#: self-correction disabled. Deliberately different wording: "we checked and found
#: problems" and "we could not check" are different things to tell someone who is
#: about to rely on a contract term, and collapsing them into one message would hide
#: a broken grader behind a message users learn to ignore.
UNVERIFIED_CAVEAT = (
    "\n\n---\n**Note:** the automatic verification step did not run for this "
    "answer, so its grounding in the cited sources has not been checked."
)

#: Returned when retrieval found nothing at all. Not an error — an honest answer.
NO_SOURCES_ANSWER = (
    "No passages in the indexed contracts matched this question closely enough to "
    "answer it. The question may concern a provision that is not present, or may "
    "need rephrasing using the contract's own terminology."
)
```

### The Theory: why "cite your sources" actually reduces hallucination

It is easy to read the citation requirement as a UI feature — nice provenance links for the user. It
is doing something more mechanical than that.

Requiring an inline citation for every claim changes the generation task. Without it, the model
produces fluent text about the topic, and the retrieved context is one influence among many competing
with its parametric knowledge. With it, every sentence must be *attached* to a specific span of
provided text, which makes an unsupported claim structurally awkward — there is no source number to
write after it.

This is not a guarantee, and overstating it would be exactly the failure the project's first lesson
warns about. Models will happily write `[SOURCE 2]` after a claim source 2 does not make. What the
requirement buys is **checkability**: an unsupported claim now carries a specific, verifiable
assertion about where it came from. That is what makes the grading node in §8 possible and what makes
Phase 6's citation validator possible. The citation does not prevent the hallucination; it makes it
detectable.

The second, less obvious effect is on the "I don't know" path. Rule 3 gives the model an explicit,
sanctioned, low-effort action when the context is insufficient. Without a stated alternative, "answer
anyway" is the only behaviour the training distribution offers.

### Failure Modes

**Citations are missing or wrong-format.** Model too small, or the context is so long the rules
scrolled out of attention. Both push toward fewer, larger sources — which is what Phase 4's parent
deduplication already gives you.

**The grader passes everything.** Prompt is too soft, or the same model generated and graded (see
§8's discussion of self-grading bias). The stated criteria are deliberately mechanical for this
reason.

**The rewriter returns the original query.** Explicitly forbidden in the prompt, and §9 detects it in
code anyway. Trusting a prompt to enforce a constraint you can check is a mistake.

---

## 6. File 3 — `src/graph/nodes/router_node.py`

### The Problem

Phase 4's §11 latency table established that the two LLM calls before retrieval are ~80% of retrieval
latency. For a query like "what does Section 7.02 say", that spend buys nothing: the user's own words
are already an excellent lexical probe, and rephrasing them into synonyms can only add noise.

There is a second, larger problem the router is the right place to catch. "How many of our contracts
are governed by Delaware law" **cannot be answered by top-k retrieval at all.** The answer is a count
over the corpus; retrieving five passages about Delaware and asking an LLM to count produces a number
that is confidently wrong. No amount of reranking fixes this, because it is a category error, not a
quality problem.

### Design Decision

**Heuristics first, LLM only when the heuristics are unsure.** A section reference, a quoted phrase,
or a Title-Case defined term is detectable with a regex in microseconds. Spending 300ms of LLM
latency to classify a query the regex already answered would make the router cost more than it saves.

**The LLM leg is optional and fails open to "vague".** If routing is disabled or the call fails, the
default is the fuller pipeline. That direction matters: failing open to *more* machinery degrades
latency, and failing open to *less* would silently degrade answer quality. When a fallback is
unavoidable, pick the direction whose failure you can see.

**Aggregate queries route to a terminal explanation, not to a bad answer.** Phase 12 implements
map-reduce for these. Until then, saying "this needs corpus-wide aggregation, which is not yet
supported" is more useful and more honest than a fabricated count.

```python
import re

from pydantic import BaseModel, Field

from config.settings import settings
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger

from ..prompts import ROUTER_SYSTEM, ROUTER_USER
from ..state import AgentState

#: A section or article reference: "Section 7.02", "ARTICLE IV", "§ 3.1(b)".
#:
#: The `§` is a separate alternative WITHOUT a leading `\b`. A word boundary requires
#: a word character on one side, and `§` is not one — so `\b(section|...|§)` never
#: matches a `§` that follows a space or starts the string, which is every real
#: usage. The first draft advertised "§ 3.1(b)" as a supported example and could not
#: match it.
_SECTION_REF = re.compile(
    r"(?:\b(?:section|article|clause|exhibit|schedule)\b|§)\s*[\dIVXLC]+(?:\.\d+)*",
    re.IGNORECASE,
)
#: A quoted phrase — the user is asking for something verbatim.
_QUOTED = re.compile(r'"[^"]{4,}"|\u201c[^\u201d]{4,}\u201d')
#: Two or more consecutive Capitalised Words: a contract's defined term.
_DEFINED_TERM = re.compile(r"\b([A-Z][a-z]+\s+){1,}[A-Z][a-z]+\b")

#: Phrases that make a corpus-wide question near-certain. Matched on WORD
#: BOUNDARIES: substring matching put "count" inside "account", "discount", and
#: "counterparty account", so "what is in the escrow account" was refused as an
#: aggregation query.
_AGGREGATE_CERTAIN = re.compile(
    r"\b(?:across all|every contract|all contracts|list all|which contracts|"
    r"how many contracts|most common|what percentage)\b",
    re.IGNORECASE,
)

#: Phrases that MIGHT be aggregate. "how many days notice is required" is an
#: ordinary passage lookup; "how many agreements expire in 2025" is not. These no
#: longer short-circuit — they go to the LLM classifier, which can read the object of
#: the question. A cheap heuristic should only short-circuit when it is nearly
#: certain; when it is merely suspicious, its job is to escalate.
_AGGREGATE_AMBIGUOUS = re.compile(
    r"\b(?:how many|how much total|count|compare|average|total number)\b",
    re.IGNORECASE,
)


class QueryClass(BaseModel):
    """Schema for the routing LLM call."""

    label: str = Field(default="vague")


async def route_query(state: AgentState, llm: BaseLLMProvider | None = None) -> dict:
    """Classify the query and decide how much machinery it needs.

    Returns a partial state update: `route` (which node runs next) and `expand`
    (whether Phase 4 spends LLM calls rephrasing).
    """
    query = state["query"]

    if _AGGREGATE_CERTAIN.search(query):
        logger.info("Routed as aggregate by heuristic", extra={"chars": len(query)})
        return {
            "route": "unsupported",
            "expand": False,
            "trace": ["router:aggregate"],
            "failure_reason": "aggregation_not_supported",
        }

    ambiguous = bool(_AGGREGATE_AMBIGUOUS.search(query))

    if not settings.ENABLE_QUERY_ROUTING:
        return {"route": "retrieve", "expand": True, "trace": ["router:disabled"]}

    # A section reference beats an ambiguous aggregate marker: "how many days notice
    # does Section 7.02 require" is a passage lookup with a counting word in it.
    if _looks_specific(query) and not ambiguous:
        # Cheap and certain. No LLM call.
        logger.info("Routed as specific by heuristic", extra={"expand": False})
        return {"route": "retrieve", "expand": False, "trace": ["router:specific"]}

    label = await _classify(query, llm)

    if label == "aggregate":
        return {
            "route": "unsupported",
            "expand": False,
            "trace": ["router:aggregate"],
            "failure_reason": "aggregation_not_supported",
        }

    expand = label != "specific"
    return {"route": "retrieve", "expand": expand, "trace": [f"router:{label}"]}


def _looks_specific(query: str) -> bool:
    """Whether the query already contains good search terms.

    Deliberately conservative: a false positive here skips expansion for a query
    that needed it, which costs recall. A false negative just costs one LLM call.
    Bias the cheap check toward the expensive-but-correct path.
    """
    if _SECTION_REF.search(query) or _QUOTED.search(query):
        return True
    # A defined term alone is weaker evidence, so require a short query too —
    # "Permitted Encumbrance" is specific; a long sentence that happens to
    # contain two capitalised words is not.
    return bool(_DEFINED_TERM.search(query)) and len(query.split()) <= 8


async def _classify(query: str, llm: BaseLLMProvider | None) -> str:
    """LLM classification. Falls back to "vague" — the fuller pipeline.

    Failing toward MORE machinery is deliberate: the failure shows up as latency,
    which is visible. Failing toward less would show up as worse answers, which is
    not.
    """
    if llm is None:
        return "vague"

    try:
        result = await llm.generate_json(
            prompt=ROUTER_USER.format(question=query),
            schema=QueryClass,
            system_prompt=ROUTER_SYSTEM,
            # The small model. Classification is the cheapest task in the system and
            # must not run on the generation model.
            model=settings.EXPANSION_MODEL,
        )
    except Exception as exc:
        # Exception, not RAGException — a malformed-JSON ValidationError must land in
        # the documented fallback rather than escaping the graph.
        logger.warning("Routing failed; defaulting to full pipeline",
                       extra={"error": type(exc).__name__})
        return "vague"

    label = (result.label or "vague").strip().lower()
    if label not in {"specific", "vague", "aggregate"}:
        logger.warning("Router returned an unknown label", extra={"label": label})
        return "vague"
    return label
```

### Failure Modes

**Everything routes to "vague".** Either routing is disabled, no LLM was injected, or the heuristics
are not matching. Log the label distribution over a batch of real queries before tuning anything.

**A real question is rejected as aggregate.** The marker list is blunt: "how many days notice is
required" contains "how many" and is a perfectly ordinary passage lookup. This is the router's
weakest point, and it is why `_AGGREGATE_MARKERS` uses phrases rather than single words — but it will
still misfire. If it does often, gate the markers behind the LLM classification instead of
short-circuiting on them.

**Specific queries get worse answers after routing was enabled.** Skipping expansion cost recall.
`_looks_specific` is intentionally narrow; narrow it further before disabling the router.

---

## 7. Files 4 and 5 — retrieval and generation nodes

### `src/graph/nodes/retrieval_node.py`

The thinnest node in the graph, and deliberately so: all retrieval intelligence is Phase 4's. This
node adapts, records, and detects emptiness.

```python
from src.core.exceptions import RetrievalError
from src.core.logging import logger
from src.retrieval.pipeline import RetrievalPipeline

from ..state import AgentState


async def retrieve_context(state: AgentState, pipeline: RetrievalPipeline) -> dict:
    """Fetch context for `current_query`.

    Reads `current_query`, not `query`: on a retry the rewriting node has replaced
    it, and searching the original again would make the loop pointless.

    An empty result is NOT an error. It is a fact the graph needs, and the edge
    function in §10 routes on it — generating from nothing would produce a fluent
    answer with no sources, which is the worst output this system can emit.
    """
    query = state["current_query"]

    try:
        result = await pipeline.retrieve(
            query=query,
            top_k=state["top_k"],
            filters=state["filters"],
            expand=state["expand"],
        )
    except RetrievalError as exc:
        # Every search arm failed — an outage, not an empty corpus. Distinguished
        # from "no matches" because the response differs: one is retryable, the
        # other is answerable.
        logger.error("Retrieval failed", extra={"error": str(exc)})
        return {
            "sources": [],
            "retrieval": None,
            "trace": ["retrieval:error"],
            "failure_reason": "retrieval_unavailable",
        }

    logger.info(
        "Retrieved context",
        extra={
            "attempt": state["retry_count"] + 1,
            "sources": result.chunk_count,
            "candidates": result.total_candidates,
            "latency_ms": round(result.latency_ms, 1),
        },
    )

    return {
        "retrieval": result,
        "sources": result.chunks,
        "trace": [f"retrieval:{result.chunk_count}"],
    }
```

### `src/graph/nodes/generation_node.py`

### The Problem

Turn five `ScoredChunk`s into a prompt, get an answer, and extract which sources the answer actually
used.

The context-formatting step looks trivial and is not. The source numbering in the prompt must map
back to chunks exactly, or every citation in the answer points at the wrong document — a failure that
produces plausible, well-formatted, wrong provenance. The parsing step has the same property in
reverse: a model that writes `[SOURCE 7]` when five sources were supplied must not produce a citation
to a chunk that does not exist.

### Design Decision

**Number sources from 1, and build the mapping in the same loop as the text.** A single loop
guarantees the prompt and the mapping cannot drift. Splitting them into two passes over the same list
is how off-by-one provenance bugs happen.

**Include metadata in each source block.** The model is told the contract name and section title, so
it can say "the Termination clause of the Acme agreement provides…" instead of "source 2 says…".

**Out-of-range citations are dropped and counted.** A `[SOURCE 9]` against five sources is a model
error. Dropping it silently would hide the error; raising would discard an otherwise good answer. So
it is dropped and logged at WARNING, and the count is available to Phase 7.

```python
import re

from config.settings import settings
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger
from src.core.models import Citation, ScoredChunk
from src.core.telemetry import telemetry

from ..prompts import GENERATION_SYSTEM, GENERATION_USER, NO_SOURCES_ANSWER
from ..state import AgentState

#: Matches "[SOURCE 3]", "[source 3]", "[Source 3]".
_CITATION = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)


def format_context(sources: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]:
    """Build the numbered context block and the index → chunk mapping.

    Both are produced in ONE loop. Two passes over the same list is how citations
    end up pointing at the wrong document — a failure that looks like working
    provenance.
    """
    blocks: list[str] = []
    mapping: dict[int, ScoredChunk] = {}

    for position, scored in enumerate(sources, start=1):
        chunk = scored.chunk
        mapping[position] = scored
        year = f", {chunk.year}" if chunk.year else ""
        blocks.append(
            f"[SOURCE {position}] {chunk.contract_name} — {chunk.section_title}{year}\n"
            f"{chunk.text}"
        )

    return "\n\n".join(blocks), mapping


def extract_citations(
    answer: str, mapping: dict[int, ScoredChunk]
) -> tuple[list[Citation], int]:
    """Parse [SOURCE N] markers into `Citation` objects.

    Returns `(citations, invalid_marker_count)`. The count is RETURNED rather than
    only logged: a non-zero value means the answer's provenance is partly
    fabricated, and that belongs in the response and in Phase 7's data, not in a log
    line nobody joins to an evaluation.

    Out-of-range markers are dropped. Dropping silently would hide a real model
    error; raising would throw away an otherwise usable answer over a formatting
    mistake. Phase 6's citation validator is the strict gate — this is extraction,
    not enforcement.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    invalid = 0

    for match in _CITATION.finditer(answer):
        index = int(match.group(1))

        if index not in mapping:
            invalid += 1
            continue
        if index in seen:
            continue

        seen.add(index)
        chunk = mapping[index].chunk
        citations.append(
            Citation(
                source_index=index,
                # The PARENT's chunk_id, because Phase 4 substituted parents and
                # the parent is what the model actually read.
                chunk_id=chunk.chunk_id,
                contract_name=chunk.contract_name,
                section_title=chunk.section_title,
                year=chunk.year,
            )
        )

    if invalid:
        logger.warning(
            "Answer cited sources that were not provided",
            extra={"invalid_markers": invalid, "sources_given": len(mapping)},
        )

    return sorted(citations, key=lambda c: c.source_index), invalid


@telemetry.measure_async("graph.generate")
async def generate_answer(state: AgentState, llm: BaseLLMProvider) -> dict:
    """Produce an answer from the retrieved sources.

    Answers the ORIGINAL query, not the rewritten one. The rewrite exists to
    improve retrieval; the user asked the original question and that is what the
    answer must address. Getting this backwards produces answers to questions
    nobody asked, and it is a genuinely easy mistake — `current_query` is right
    there in the state.
    """
    sources = state["sources"]

    if not sources:
        # Should be unreachable: the edge in §10 routes empty retrieval away from
        # this node. Defended anyway, because "generate from no context" is the
        # single worst thing this system can do and one wrong edge would enable it.
        return {
            "answer": NO_SOURCES_ANSWER,
            "citations": [],
            "trace": ["generation:no-sources"],
        }

    context, mapping = format_context(sources)

    try:
        answer = await llm.generate(
            prompt=GENERATION_USER.format(context=context, question=state["query"]),
            system_prompt=GENERATION_SYSTEM,
            temperature=0.0,
            # The large model, explicitly. Without this the generator runs on
            # whatever the provider was constructed with, and the documented
            # cost/quality split between GENERATION_MODEL and GRADER_MODEL does not
            # exist.
            model=settings.GENERATION_MODEL,
        )
    except Exception as exc:
        logger.error("Generation failed", extra={"error": type(exc).__name__})
        return {
            "answer": "",
            "citations": [],
            "trace": ["generation:error"],
            # The exception TYPE is preserved in the reason so the facade can raise
            # something accurate. A generation outage reported as retry-budget
            # exhaustion produces the wrong HTTP status and sends whoever is on call
            # to look at the wrong component.
            "failure_reason": f"generation_failed:{type(exc).__name__}",
        }

    answer = answer.strip()
    citations, invalid_citations = extract_citations(answer, mapping)

    # Never log the answer or any source text — roadmap §7.5.
    logger.info(
        "Generated answer",
        extra={
            "attempt": state["retry_count"] + 1,
            "answer_chars": len(answer),
            "citations": len(citations),
            "sources_offered": len(sources),
        },
    )

    return {
        "answer": answer,
        "citations": citations,
        "invalid_citations": invalid_citations,
        "trace": ["generation"],
        "attempts": [
            {
                "attempt": state["retry_count"] + 1,
                "graded": False,          # the grading node appends its own record
                "query": state["current_query"],
                "sources": len(sources),
                "citations": len(citations),
                "invalid_citations": invalid_citations,
                "answer_chars": len(answer),
            }
        ],
    }
```

### Why `temperature=0.0`

Extraction wants determinism. There is exactly one correct notice period in a contract, and sampling
diversity can only move away from it. Temperature exists to make text less predictable, which is
valuable for prose and actively harmful when the task is "report what this document says".

It also makes the system debuggable: the same context and question produce the same answer, so a bad
answer is reproducible. With `temperature=0.7` you would be unable to tell whether a change you made
fixed anything. (Note that greedy decoding is not *guaranteed* deterministic across a provider's
infrastructure — batching and hardware can shift ties — so treat this as "as reproducible as it
gets", not as a promise.)

### Failure Modes

**Answers cite `[SOURCE 6]` when five were given.** Logged, dropped. If frequent, the model is
confusing source numbers with section numbers — usually a sign the context block's formatting is not
distinct enough.

**No citations at all.** Model too small, or the context is long enough that rule 2 lost attention.
Reduce `RERANK_TOP_K` before rewriting the prompt.

**The answer addresses the rewritten query, not the original.** You passed `current_query` to
generation. It should be `state["query"]`.

**Every answer says the sources are insufficient.** Retrieval is failing, not generation. Check
`retrieval.chunk_count` and the section titles being returned before touching the prompt.

---

## 8. File 6 — `src/graph/nodes/grading_node.py`

### The Problem

This node is the difference between a RAG demo and a RAG system, so it is worth being precise about
what it can and cannot do.

The generation node has produced an answer that *looks* right. It is fluent, it cites sources, and it
answers the question. None of that is evidence that the cited sources support it. Somebody has to
check, and there is no deterministic way to check "is this claim supported by this passage" — that is
a natural-language inference problem.

### Design Decision

**A second LLM call, with a narrow, mechanical, structured task.** The grader gets the question, the
answer, and the same sources, and returns a `GradingReport`. Three properties make this work better
than it sounds:

**`generate_json` with a Pydantic schema.** Phase 1 built `GradingReport` for this. A malformed grade
raises a validation error rather than parsing as a pass — which matters enormously, because the
default on a parse failure would otherwise be "looks fine, ship it".

**Two independent booleans, not one score.** Grounded and relevant fail for different reasons and
have different fixes. An ungrounded answer needs different context; an irrelevant answer needs a
different query. Collapsing them into "quality: 0.6" destroys the information the rewriting node
needs. `GradingReport.passed` (grounded AND relevant) is what the edge reads.

**A smaller, cheaper model than the generator.** `GRADER_MODEL` is `gpt-oss-20b` against the
generator's `120b`. Verification is easier than generation — checking whether a passage contains a
claim is closer to reading comprehension than to writing — and the grader runs on every request.

**A failed grading call fails OPEN, with low confidence recorded.** If the grader is down, returning
the answer marked unverified beats returning nothing. But `confidence=0.0` makes it visible in the
response and in Phase 7's metrics, so "the grader has been broken for a week" is discoverable rather
than invisible.

```python
from config.settings import settings
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger
from src.core.models import GradingReport
from src.core.telemetry import telemetry

from ..prompts import GRADING_SYSTEM, GRADING_USER
from ..state import AgentState
from .generation_node import format_context


@telemetry.measure_async("graph.grade")
async def grade_answer(state: AgentState, llm: BaseLLMProvider) -> dict:
    """Audit the answer for grounding and relevance.

    Returns a `GradingReport` in the state. The edge function reads `.passed` to
    decide whether to finish or retry.
    """
    if not settings.ENABLE_SELF_CORRECTION:
        return {
            "grading": GradingReport(
                verified=False,
                is_grounded=True,
                is_relevant=True,
                confidence=0.0,
                reasoning="Self-correction disabled; answer was not audited.",
            ),
            "trace": ["grading:disabled"],
        }

    if not state["answer"]:
        # Generation failed. There is nothing to audit, and grading an empty
        # string would waste a call to learn what we already know.
        return {
            "grading": GradingReport(
                is_grounded=False,
                is_relevant=False,
                confidence=1.0,
                reasoning="No answer was generated.",
            ),
            "trace": ["grading:no-answer"],
        }

    # The SAME context the generator saw — auditing a different context would
    # produce verdicts that cannot be reproduced. But budgeted independently: the
    # grading prompt carries the context PLUS the question PLUS the whole generated
    # answer, so a generation request that fitted comfortably can overflow here. An
    # overflow would take the exception path, and before `verified` existed that
    # path produced a passing grade — a context-length problem silently becoming a
    # verified answer.
    context = _grading_context(state)

    try:
        report = await llm.generate_json(
            prompt=GRADING_USER.format(
                question=state["query"], answer=state["answer"], context=context
            ),
            schema=GradingReport,
            system_prompt=GRADING_SYSTEM,
            model=settings.GRADER_MODEL,
        )
    except Exception as exc:
        # Catches Exception, not only RAGException: the provider contract requires
        # that a Pydantic ValidationError on malformed JSON is wrapped, but a
        # documented fallback that depends on every provider honouring that
        # perfectly is not a fallback.
        #
        # Fails OPEN — the answer is returned — but `verified=False`, which makes
        # `GradingReport.passed` False. The first draft set grounded=relevant=True
        # here, which made `passed` True, so every caller checking `.passed` treated
        # an UNAUDITED answer as a verified one. That was the most dangerous bug in
        # the phase: a grader outage silently upgraded every answer.
        logger.warning("Grading failed; returning answer unverified",
                       extra={"error": type(exc).__name__})
        return {
            "grading": GradingReport(
                verified=False,
                is_grounded=True,
                is_relevant=True,
                confidence=0.0,
                reasoning=f"Grading unavailable ({type(exc).__name__}); not verified.",
            ),
            "trace": ["grading:error"],
        }

    # A confident-sounding pass from an uncertain grader is not a pass. Without
    # this, "confidence: 0.1" would silently carry the same weight as "1.0" and
    # the field would be decoration.
    if report.passed and report.confidence < settings.GRADER_MIN_CONFIDENCE:
        logger.info(
            "Grade passed but below the confidence floor; treating as a non-verdict",
            extra={"confidence": report.confidence,
                   "floor": settings.GRADER_MIN_CONFIDENCE},
        )
        report = report.model_copy(
            update={
                "is_grounded": False,
                "reasoning": (
                    f"Grader confidence {report.confidence:.2f} is below the "
                    f"{settings.GRADER_MIN_CONFIDENCE:.2f} floor. "
                    f"Original reasoning: {report.reasoning}"
                ),
            }
        )

    logger.info(
        "Graded answer",
        extra={
            "attempt": state["retry_count"] + 1,
            "grounded": report.is_grounded,
            "relevant": report.is_relevant,
            "passed": report.passed,
            "confidence": round(report.confidence, 2),
            "unsupported_claims": len(report.unsupported_claims),
        },
    )

    return {
        "grading": report,
        "trace": [f"grading:{'pass' if report.passed else 'fail'}"],
        # Attach the verdict to THIS attempt's record. Generation wrote the attempt
        # before the grade existed, so without this the state promises "one record
        # per generate-and-grade attempt" and delivers records with no grade —
        # leaving Phase 7 unable to reconstruct why an earlier attempt failed, since
        # only the final grade survives in `state["grading"]`.
        "attempts": [
            {
                "attempt": state["retry_count"] + 1,
                "graded": True,
                "verified": report.verified,
                "grounded": report.is_grounded,
                "relevant": report.is_relevant,
                "passed": report.passed,
                "confidence": round(report.confidence, 3),
                "unsupported_claims": list(report.unsupported_claims),
            }
        ],
    }


def _grading_context(state: AgentState, reserve_tokens: int = 1500) -> str:
    """The generator's context, trimmed to leave room for the answer and question.

    The grader's input is strictly larger than the generator's, so it needs its own
    budget. Sources are dropped from the END — lowest-ranked first — so the audit
    keeps the material the answer most likely drew on. Dropping sources weakens the
    audit (a claim supported only by a dropped source now reads as unsupported), and
    that direction is the safe one: it produces a false failure, which costs a retry,
    rather than a false pass, which ships a wrong answer.
    """
    from config.settings import settings as _settings

    budget = max(_settings.MAX_CONTEXT_TOKENS - reserve_tokens, 1000)
    kept: list = []
    total = 0

    for scored in state["sources"]:
        tokens = scored.chunk.token_count
        if kept and total + tokens > budget:
            logger.warning(
                "Trimming sources for the grading prompt",
                extra={"kept": len(kept), "dropped": len(state["sources"]) - len(kept)},
            )
            break
        kept.append(scored)
        total += tokens

    context, _ = format_context(kept)
    return context
```

### The Theory: self-grading bias, and what actually mitigates it

An LLM asked to evaluate LLM output is a lenient judge. This is well documented, and the mechanisms
are worth knowing because they tell you which mitigations are real and which are theatre.

**Fluency reads as correctness.** Models are trained on human preference data, and humans prefer
confident, well-structured text. A polished ungrounded answer looks better than a hedged accurate one.

**The same model shares the same errors.** If the generator misreads a clause, a grader with the same
weights tends to misread it identically. It is not an independent check; it is correlated.

**"Is this good?" has no falsifiable answer.** Asked to evaluate quality, a model produces a vibe.

What genuinely helps, in order of effect:

**Make the task extractive rather than evaluative.** "Is every claim in this answer stated by these
sources" is close to a string-matching problem with synonyms. It has a checkable answer. "Is this a
good answer" does not. This is the single biggest lever and it is entirely in the prompt.

**Demand the evidence.** `unsupported_claims` forces the grader to *name* what failed. A model that
must produce specifics cannot vaguely approve — and the list is exactly what the rewriting node needs,
so the anti-bias measure and the feature are the same mechanism.

**Use a different model than the generator.** `gpt-oss-20b` grading `gpt-oss-120b` is not
independent — same family, similar training — but it is less correlated than self-grading. A genuinely
different family (Claude grading Llama) would be better and is a deployment choice, not a code change:
`generate_json` takes the provider, and Phase 6's factory can hand this node a different one.

**Keep the confidence floor.** A grader that is unsure is not a grader that approved.

What does **not** help: asking for a 1–10 score (models cluster at 7–8 regardless), asking it to
"think step by step" before a boolean (adds latency and rationalisation more than accuracy), or
grading twice and taking the majority of two correlated samples.

Be honest about the ceiling: this catches a large fraction of ungrounded answers, not all of them.
Phase 7 measures the fraction on your corpus, and Phase 6's citation validator adds a *deterministic*
check underneath — verifying that cited chunk IDs were actually in the context is string comparison,
not inference, and it cannot be talked out of its verdict.

### Failure Modes

**Everything passes.** Check `confidence` first. If confidences are high and everything passes,
either retrieval is genuinely good — check by hand — or the grader is rubber-stamping. Feed it a
deliberately wrong answer; if that passes, the prompt or the model is the problem.

**Everything fails and the loop always exhausts its budget.** Usually an over-strict reading of
grounding, where the answer paraphrases correctly and the grader wants verbatim text. The prompt says
numbers and defined terms must match exactly, not that every sentence must be quoted.

**Latency doubled after enabling grading.** Expected: grading is a second full LLM call with the
entire context. It is the price of knowing whether your answer is real, and it is why
`GRADER_MODEL` is the small one.

**`ValidationError` from `generate_json`.** The model did not return conforming JSON. Fails open with
`verified=False` and `confidence=0.0`, so the answer is returned but `passed` is False, the status is
`UNVERIFIED`, and no retry is attempted. Phase 6 makes JSON mode reliable; if it is frequent, the
grading model is too small for structured output.

**Answers are marked `UNVERIFIED` in bulk.** The grader is erroring on every request and the fail-open
path is now the normal path. This is exactly what `verified` exists to make visible — before it, this
condition looked like a system where every answer passed.

---

## 9. File 7 — `src/graph/nodes/rewriting_node.py`

### The Problem

Grading failed. Retrying the identical query would retrieve the identical passages and produce the
identical answer — an infinite loop that burns its budget accomplishing nothing.

The retry has to change something, and the only lever that affects retrieval is the query.

### Design Decision

**Rewrite using the grading feedback, not just the query.** The naive version asks for "a different
phrasing", which is what Phase 4's multi-query already did — and it already failed. The grader
produced `reasoning` and `unsupported_claims`, which say *what was missing*. A rewrite that targets
the specific gap is a genuinely different search, not a paraphrase.

**Detect and reject a no-op rewrite in code.** The prompt forbids returning the original. Prompts are
not enforcement. If the rewrite is unchanged after normalisation, the loop cannot make progress, so
the node says so and the edge stops the loop — burning two more LLM calls to reach the same answer is
worse than stopping.

**Increment `retry_count` here.** One node owns the counter. If both the grader and the rewriter
touched it, the budget would be double-counted, and a budget that is wrong in the direction of "more
retries" is expensive.

```python
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger
from src.core.utils import normalise_whitespace

from ..prompts import REWRITE_SYSTEM, REWRITE_USER
from ..state import AgentState


class RewrittenQuery(BaseModel):
    query: str = Field(default="")
    reasoning: str = Field(default="", max_length=500)


async def rewrite_query(state: AgentState, llm: BaseLLMProvider) -> dict:
    """Produce a better search query from the grader's diagnosis.

    Increments `retry_count` — this node is the sole owner of that counter.
    """
    grading = state["grading"]
    attempt = state["retry_count"] + 1

    unsupported = "\n".join(f"- {c}" for c in (grading.unsupported_claims if grading else []))

    try:
        result = await llm.generate_json(
            prompt=REWRITE_USER.format(
                question=state["query"],
                previous_query=state["current_query"],
                reasoning=grading.reasoning if grading else "unknown",
                unsupported=unsupported or "- (none listed)",
            ),
            schema=RewrittenQuery,
            system_prompt=REWRITE_SYSTEM,
            model=settings.EXPANSION_MODEL,
        )
        rewritten = (result.query or "").strip()
    except Exception as exc:
        logger.warning("Rewrite failed", extra={"error": type(exc).__name__})
        rewritten = ""

    if not _is_progress(rewritten, state["current_query"]):
        # No new query means the next retrieval is identical, which means the next
        # answer is identical. Stop, rather than spending two more LLM calls to
        # arrive back here.
        logger.warning("Rewrite produced no usable change; ending the retry loop")
        return {
            "retry_count": settings.MAX_RETRIES,   # exhaust the budget deliberately
            "trace": [f"rewrite:{attempt}:no-progress"],
        }

    logger.info(
        "Rewrote query for retry",
        extra={"attempt": attempt, "chars": len(rewritten)},
    )

    return {
        "current_query": rewritten,
        "retry_count": attempt,
        # Reset the grade so the next pass through grading starts clean. Leaving a
        # stale failing grade in the state would let the edge read a verdict from
        # the previous attempt if a later node returned early.
        "grading": None,
        # Force expansion ON for the retry, whatever the router decided originally.
        # The router's "this query is already specific, skip expansion" call was a
        # prediction, and a failed grade is evidence it was wrong: the narrow search
        # did not surface enough to ground an answer. Retrying with the same narrow
        # strategy and a slightly different query is the weakest possible retry.
        "expand": True,
        "trace": [f"rewrite:{attempt}"],
    }


def _is_progress(rewritten: str, previous: str) -> bool:
    """Whether the rewrite is a genuinely different query.

    Compared after whitespace normalisation and case folding, so trailing
    punctuation or capitalisation changes do not count as progress. A two-word
    minimum rejects degenerate outputs like "termination" or an empty string —
    two words is genuinely enough for a legal search ("liability cap",
    "permitted encumbrance"), so a higher floor would reject good rewrites.
    """
    if len(rewritten.split()) < 2:
        return False
    return normalise_whitespace(rewritten).lower() != normalise_whitespace(previous).lower()
```

### Failure Modes

**The rewrite is always a paraphrase and never helps.** The grader's `unsupported_claims` is probably
empty, so the rewriter has no diagnosis to work from. Fix the grading prompt first — the rewriter can
only be as good as the feedback it receives.

**`retry_count` jumps straight to `MAX_RETRIES`.** Intended, on a no-progress rewrite. Look for
`rewrite:N:no-progress` in the trace.

**The loop retries but the answer never changes.** Retrieval is returning the same passages for
genuinely different queries, which means the corpus does not contain the answer. That is a correct
finding, and the honest output is the caveat rather than a third attempt.

---

## 10. File 8 — `src/graph/edges.py`

### The Problem

The nodes do work. Something has to decide **what runs next**, and that decision is the actual
control flow of the application. Buried inside nodes as `if` statements, it is invisible; here, it is
four small pure functions you can read in one screen and unit-test without an LLM.

### Design Decision

**Routing functions are pure and take only the state.** No LLM, no I/O, no side effects. Given a
state, the next node is a deterministic function of it — which makes the entire control flow testable
with dicts.

**They return string literals, mapped explicitly in `builder.py`.** LangGraph allows returning node
names directly. Going through an explicit `path_map` means the *set of possible destinations* is
declared at wiring time, so a typo produces a graph-construction error instead of a runtime
`KeyError` deep in a request.

**Every function has a terminal branch.** An edge function with no path to `END` is an infinite loop
waiting for input that makes it stop.

```python
from typing import Literal

from config.settings import settings
from src.core.logging import logger

from .state import AgentState


def after_router(state: AgentState) -> Literal["retrieve", "unsupported"]:
    """Enter retrieval, or terminate for a query top-k cannot answer."""
    return "unsupported" if state["route"] == "unsupported" else "retrieve"


def after_retrieval(state: AgentState) -> Literal["generate", "no_context", "keep_previous"]:
    """Generate only if there is something to generate from.

    This is the most important edge in the graph. Generating from an empty context
    produces a fluent, confident, entirely unsourced answer — the single worst
    output this system can emit. It is cheaper and far safer to say nothing was
    found.

    The three-way split exists because "no sources" means different things on the
    first attempt and on a retry. On the first attempt there is nothing to return,
    so `no_context` is right. On a retry we already HAVE an answer that merely
    failed its audit, and a rewritten query that retrieves nothing is not a reason
    to throw it away — routing to `no_context` there would overwrite a usable
    answer with "no passages matched", which directly contradicts this phase's
    policy of preserving a partially-supported answer.
    """
    if state["sources"]:
        return "generate"

    if state["answer"]:
        logger.info(
            "Retry retrieved nothing; keeping the previous answer with its failed grade"
        )
        return "keep_previous"

    logger.info("No sources retrieved; skipping generation")
    return "no_context"


def after_grading(state: AgentState) -> Literal["accept", "retry"]:
    """The self-correction decision — the reason this is a graph.

    Retry only when ALL of these hold:
      - self-correction is enabled,
      - the grade failed,
      - the retry budget has room,
      - and an answer exists to have failed (a generation error is not something
        a query rewrite can fix).
    """
    grading = state["grading"]

    if not settings.ENABLE_SELF_CORRECTION or grading is None:
        return "accept"

    if grading.passed:
        return "accept"

    if not grading.verified:
        # The audit did not run — grader outage, malformed output, or disabled.
        # `passed` is False, so the answer will carry a caveat, but retrying is
        # pointless: rewriting the query does not fix a broken grader, and the
        # rewrite would consume two more LLM calls to reach the same unverified
        # state. This branch is why `verified` is a separate field rather than
        # folded into `is_grounded`.
        logger.info("Answer is unverified rather than failed; not retrying")
        return "accept"

    if not state["answer"]:
        # Generation itself failed. Rewriting the query does not address a 502
        # from the LLM provider, and retrying would just fail again more slowly.
        logger.info("No answer to correct; accepting the failure")
        return "accept"

    if state["retry_count"] >= settings.MAX_RETRIES:
        logger.info(
            "Retry budget exhausted; returning the best answer with a caveat",
            extra={"retries": state["retry_count"], "budget": settings.MAX_RETRIES},
        )
        return "accept"

    logger.info(
        "Grade failed; retrying with a rewritten query",
        extra={"attempt": state["retry_count"] + 1, "budget": settings.MAX_RETRIES},
    )
    return "retry"


def after_rewrite(state: AgentState) -> Literal["retrieve", "accept"]:
    """Search again, unless the rewriter exhausted the budget signalling no progress."""
    if state["retry_count"] >= settings.MAX_RETRIES:
        return "accept"
    return "retrieve"
```

### The Theory: why this is a graph and not a `while` loop

Worth confronting directly, because the honest answer is not "graphs are better".

The same logic as a loop is about twenty-five lines:

```python
for attempt in range(MAX_RETRIES + 1):
    context = await retrieve(query)
    answer = await generate(context, question)
    grade = await grade_answer(answer, context)
    if grade.passed:
        break
    query = await rewrite(query, grade)
```

That is readable, obvious, and easier to debug than a state machine. If this were the whole system,
the loop would be the right answer, and anyone who tells you otherwise is selling a framework.

Four things justify the graph, and they are all about what comes *after* this phase:

**Persistence between steps.** The checkpointer saves state at every super-step, so a run can be
inspected mid-flight, resumed after a crash, or paused for human approval. The loop's state lives in
stack frames that vanish. Phase 8 needs this to stream progress; Phase 9 needs it to show a trace.

**Streaming as a first-class output.** `astream` yields state after each node. Getting the equivalent
from the loop means threading a callback through every function.

**The topology is data.** Phase 12 adds a map-reduce node and Phase 13 adds graph traversal. In the
graph, each is a new node plus an edge in the router's `path_map`. In the loop, each is a new branch
inside a function that is already the most complex in the system — and the loop grows a new nesting
level every time.

**The control flow is separately testable.** `after_grading` is a pure function of a dict. Testing
"does it retry when grounding fails but the budget is spent" needs no LLM, no Qdrant, and no mocking
— which is why the retry logic in this phase is the part I am most confident is correct.

The cost is real: an indirection layer, a framework dependency that churns, and stack traces that go
through library code. That is the trade. Take it because of Phases 8, 12, and 13 — not because state
machines are elegant.

### Failure Modes

**`GraphRecursionError`.** Structural safety net firing. Almost always an edge returning a value not
in its `path_map`, or `after_rewrite` never reaching `accept` because `retry_count` is not
incrementing. Check the trace: repeated `rewrite:1` entries mean the counter is stuck.

**The loop never retries.** `ENABLE_SELF_CORRECTION` is off, grading is passing everything, or
`MAX_RETRIES` is 0. The trace shows which.

**It retries even when the grade passed.** `passed` is a computed field on `GradingReport` (grounded
AND relevant). If you overrode one of those after construction, recompute rather than patching the
edge.

---

## 11. File 9 — `src/graph/builder.py`

### The Problem

Assemble nine pieces into a runnable object, inject dependencies into nodes that LangGraph will call
with only the state, and expose something the API can hold.

The injection problem is the interesting one. LangGraph calls a node as `fn(state)`. Our nodes need an
LLM and a pipeline. Something has to bridge that.

### Design Decision

**`functools.partial` to bind dependencies.** `partial(generate_answer, llm=llm)` produces a callable
LangGraph can invoke with just the state. The alternative — module-level globals for the LLM and
pipeline — makes testing require monkeypatching and makes two differently-configured agents in one
process impossible (which Phase 7 needs for A/B evaluation).

**`RAGAgent` as a thin facade returning `RAGAnswer`.** Callers get Phase 1's domain model, not a raw
state dict. Phase 8 serialises `RAGAnswer` directly; nothing downstream learns that LangGraph exists.

**`recursion_limit` computed from `MAX_RETRIES`, not left at 1000.** The graph's longest legitimate
path is `router → (retrieve → generate → grade → rewrite) × (MAX_RETRIES + 1)`, so roughly
`4 × (MAX_RETRIES + 1) + 3` super-steps. Setting the limit near that means an edge bug fails in about
a second; at the default of 1000 it fails after hundreds of LLM calls and a real bill. Generous
headroom, still two orders of magnitude below the default.

**The checkpointer is optional and off by default.** It costs a SQLite write per super-step and is
only useful when something resumes or inspects threads. Phase 8 turns it on.

```python
import time
from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from src.core.exceptions import (
    InvalidQueryError,
    LLMProviderError,
    MaxRetriesExceededError,
    RetrievalError,
)
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger, new_request_id
from src.core.models import AnswerStatus, RAGAnswer
from src.core.telemetry import telemetry
from src.retrieval.pipeline import RetrievalPipeline

from .edges import after_grading, after_retrieval, after_rewrite, after_router
from .nodes.generation_node import generate_answer
from .nodes.grading_node import grade_answer
from .nodes.retrieval_node import retrieve_context
from .nodes.rewriting_node import rewrite_query
from .nodes.router_node import route_query
from .prompts import (
    LOW_CONFIDENCE_CAVEAT,
    NO_SOURCES_ANSWER,
    PROMPT_VERSION,
    UNVERIFIED_CAVEAT,
)
from .state import AgentState, initial_state


def build_graph(
    llm: BaseLLMProvider,
    pipeline: RetrievalPipeline,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Wire the nodes and edges into a compiled graph.

    Dependencies are bound with `partial` because LangGraph calls each node with
    the state alone. Module-level globals would work and would make two
    differently-configured agents in one process impossible — which Phase 7's A/B
    evaluation needs.
    """
    builder = StateGraph(AgentState)

    builder.add_node("router", partial(route_query, llm=llm))
    builder.add_node("retrieval", partial(retrieve_context, pipeline=pipeline))
    builder.add_node("generation", partial(generate_answer, llm=llm))
    builder.add_node("grading", partial(grade_answer, llm=llm))
    builder.add_node("rewriting", partial(rewrite_query, llm=llm))

    # A terminal node for questions retrieval cannot serve. A node rather than an
    # edge to END, because it must write the explanation into the state.
    builder.add_node("no_context", _no_context_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        after_router,
        {"retrieve": "retrieval", "unsupported": "no_context"},
    )
    builder.add_conditional_edges(
        "retrieval",
        after_retrieval,
        # `keep_previous` goes straight to END without touching `answer` or
        # `citations`, so the previous attempt's answer survives with its failing
        # grade and picks up the caveat in the facade.
        {"generate": "generation", "no_context": "no_context", "keep_previous": END},
    )
    builder.add_edge("generation", "grading")

    # THE cycle. Everything else in this file is scaffolding around this edge.
    builder.add_conditional_edges(
        "grading",
        after_grading,
        {"accept": END, "retry": "rewriting"},
    )
    builder.add_conditional_edges(
        "rewriting",
        after_rewrite,
        {"retrieve": "retrieval", "accept": END},
    )
    builder.add_edge("no_context", END)

    return builder.compile(checkpointer=checkpointer)


def _no_context_node(state: AgentState) -> dict:
    """Terminal explanation for a question this pipeline cannot answer.

    Note what this node does NOT do: invent a friendly message for a retrieval
    outage. An outage is not an answer, and returning prose for it means the caller
    must string-match English to discover that the vector store is down. The facade
    raises `RetrievalError` for that case instead — `failure_reason` carries it
    through.
    """
    if state["failure_reason"] == "aggregation_not_supported":
        answer = (
            "This question requires counting or comparing across many contracts, "
            "which passage retrieval cannot do. Corpus-wide aggregation is planned "
            "(Phase 12). Asking about a specific contract or provision will work."
        )
    else:
        answer = NO_SOURCES_ANSWER

    return {"answer": answer, "citations": [], "trace": ["no_context"]}


class RAGAgent:
    """The facade. Question in, `RAGAnswer` out.

    Nothing above this class knows LangGraph is involved, which is the point:
    Phase 8 serialises `RAGAnswer`, and swapping the orchestrator would not
    change a line of the API.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        pipeline: RetrievalPipeline,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.graph = build_graph(llm, pipeline, checkpointer)
        self._checkpointed = checkpointer is not None

        # Longest legitimate path: router, then (retrieve, generate, grade,
        # rewrite) per attempt, plus a terminal node. Sized close to that so a
        # buggy edge fails in a second instead of after hundreds of LLM calls.
        self.recursion_limit = 4 * (settings.MAX_RETRIES + 1) + 6

    @telemetry.span("agent.answer", warn_over_ms=15_000)
    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        thread_id: str | None = None,
    ) -> RAGAnswer:
        """Run the graph and return a structured answer.

        Args:
            thread_id: Checkpoint thread for THIS question. Supply one only when you
                intend the run to share checkpoint state with a previous one — see
                the warning in `resume()`. Defaults to a fresh ID per question, which
                is what you almost always want.

        Raises:
            InvalidQueryError: blank question, or top_k below 1.
            LLMProviderError: generation failed. Distinct from exhaustion.
            RetrievalError: the vector store is unavailable. Retryable.
            MaxRetriesExceededError: the graph terminated with no answer for a reason
                that is not one of the above. NOT raised for a poor answer — see §2.
        """
        # Validate before spending anything. `top_k or DEFAULT` silently turned 0
        # into the default while letting -5 through to the store, where it surfaced
        # as a retrieval failure — a request bug reported as an infrastructure one.
        if not question or not question.strip():
            raise InvalidQueryError("Question cannot be empty")
        if top_k is not None and top_k < 1:
            raise InvalidQueryError("top_k must be at least 1", details={"top_k": top_k})

        request_id = new_request_id()
        started = time.perf_counter()

        state = initial_state(
            query=question.strip(),
            top_k=top_k if top_k is not None else settings.RERANK_TOP_K,
            filters=filters,
        )

        # `recursion_limit` is a TOP-LEVEL config key. Putting it inside
        # `configurable` — where every other user key goes — silently leaves you
        # on the default of 1000.
        active_thread = thread_id or request_id
        config: dict = {"recursion_limit": self.recursion_limit}
        if self._checkpointed:
            config["configurable"] = {"thread_id": active_thread}

        try:
            final: AgentState = await self.graph.ainvoke(state, config)
        except GraphRecursionError as exc:
            # A bug in our edges, not a user problem. The semantic budget should
            # have stopped the loop long before this.
            raise MaxRetriesExceededError(
                "The agent graph exceeded its step limit — a routing edge is "
                "probably not terminating",
                details={"recursion_limit": self.recursion_limit},
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        return self._to_answer(question, final, elapsed_ms, active_thread)

    async def resume(self, thread_id: str) -> RAGAnswer:
        """Continue an interrupted run on an existing checkpoint thread.

        Resuming is `ainvoke(None, config)` — passing `None` as the input tells
        LangGraph to continue from the saved checkpoint rather than start over.

        The first draft had no resume path at all: `answer()` always supplied a fresh
        `initial_state`, so passing an existing `thread_id` did not continue that
        thread, it started a new run on top of it. And because `trace` and `attempts`
        carry `operator.add` reducers, the new run's entries were APPENDED to the old
        run's — producing a single trace containing two unrelated questions. So:
        one thread per question, and continuation goes through here.

        Raises:
            RuntimeError: no checkpointer, so there is nothing to resume from.
        """
        if not self._checkpointed:
            raise RuntimeError("resume() requires a checkpointer")

        started = time.perf_counter()
        config = {
            "recursion_limit": self.recursion_limit,
            "configurable": {"thread_id": thread_id},
        }
        final: AgentState = await self.graph.ainvoke(None, config)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return self._to_answer(final["query"], final, elapsed_ms, thread_id)

    def _to_answer(
        self, question: str, final: AgentState, elapsed_ms: float, thread: str
    ) -> RAGAnswer:
        """Map terminal graph state onto a `RAGAnswer`, or onto the right exception.

        The mapping matters as much as the answer does. Every no-answer case used to
        raise `MaxRetriesExceededError`, which told the caller "the retry budget ran
        out" when the truth was a 502 from the LLM or a dead vector store — the wrong
        HTTP status, and the wrong component to send someone to look at.
        """
        reason = final["failure_reason"]

        if reason == "retrieval_unavailable":
            # An outage is not an answer. Raising a retryable error lets Phase 8
            # return 503 and lets any retry logic act on `.retryable`.
            raise RetrievalError(
                "Retrieval is unavailable; no answer could be produced",
                details={"trace": final["trace"]},
                retryable=True,
            )

        if not final["answer"]:
            if reason and reason.startswith("generation_failed:"):
                raise LLMProviderError(
                    "Generation failed; no answer was produced",
                    details={"cause": reason.split(":", 1)[1], "trace": final["trace"]},
                    retryable=True,
                )
            raise MaxRetriesExceededError(
                "The agent produced no answer",
                details={
                    "reason": reason or "unknown",
                    "retries": final["retry_count"],
                    "trace": final["trace"],
                },
            )

        answer_text = final["answer"]
        grading = final["grading"]

        # Status is DERIVED, so no caller has to infer it from English prose.
        if reason == "aggregation_not_supported":
            status = AnswerStatus.UNSUPPORTED
        elif not final["sources"]:
            status = AnswerStatus.NO_MATCH
        elif grading is None or grading.passed:
            status = AnswerStatus.ANSWERED
        elif not grading.verified:
            status = AnswerStatus.UNVERIFIED
        else:
            status = AnswerStatus.LOW_CONFIDENCE

        # An answer that failed or skipped its audit is still returned — with the
        # warning attached. The two cases get different wording, because "we checked
        # and it looks wrong" and "we could not check" are different things to tell
        # someone about a contract.
        if status is AnswerStatus.LOW_CONFIDENCE:
            answer_text += LOW_CONFIDENCE_CAVEAT
        elif status is AnswerStatus.UNVERIFIED:
            answer_text += UNVERIFIED_CAVEAT

        logger.info(
            "Agent complete",
            extra={
                "status": status.value,
                "retries": final["retry_count"],
                "sources": len(final["sources"]),
                "citations": len(final["citations"]),
                "invalid_citations": final["invalid_citations"],
                "passed": grading.passed if grading else None,
                "latency_ms": round(elapsed_ms, 1),
                "prompt_version": PROMPT_VERSION,
                "trace": " → ".join(final["trace"]),
            },
        )

        return RAGAnswer(
            query=question,
            answer=answer_text,
            citations=final["citations"],
            sources=final["sources"],
            grading=grading,
            retry_count=final["retry_count"],
            cache_hit=False,          # Phase 6 owns the cache
            total_latency_ms=elapsed_ms,
            status=status,
            failure_reason=reason,
            thread_id=thread if self._checkpointed else None,
            invalid_citations=final["invalid_citations"],
        )
```

### Wiring it up

```python
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config.settings import settings
from src.embeddings.factory import get_embedding_provider
from src.graph.builder import RAGAgent
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstores.factory import get_vector_store


async def main() -> None:
    embedder = get_embedding_provider()
    store = get_vector_store()
    llm = ...   # Phase 6: GroqProvider()

    pipeline = RetrievalPipeline(embedder=embedder, store=store, llm=llm)
    pipeline.warmup()

    # AsyncSqliteSaver is an ASYNC context manager, and the async variant is
    # required for a graph run with `ainvoke`. The sync SqliteSaver would block
    # the event loop on every super-step.
    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as saver:
        agent = RAGAgent(llm=llm, pipeline=pipeline, checkpointer=saver)
        result = await agent.answer("what notice is required to terminate early?")

        print(result.answer)
        for citation in result.citations:
            print(f"  [{citation.source_index}] {citation.contract_name} — "
                  f"{citation.section_title}")
        if result.grading:
            print(f"  grounded={result.grading.is_grounded} "
                  f"relevant={result.grading.is_relevant} "
                  f"retries={result.retry_count}")

    await store.close()
    await embedder.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### The Theory: streaming versus grading, a tension worth knowing now

Phase 8 will stream answers token by token over SSE. Phase 5 grades answers after they are complete.
These two requirements are in direct conflict, and it is better to see it now than to discover it
while writing the API.

If you stream tokens to the user as they are generated, then by the time grading runs, the user has
already read the answer. **You cannot un-stream an ungrounded answer.** The three options, with their
real costs:

**Stream, then append the verdict.** The user reads the answer immediately and gets a warning banner a
second later if it failed. Best perceived latency; the user may have already acted on a bad answer.

**Grade first, then stream.** The answer is verified before a token is shown, but time-to-first-token
becomes the full generate-plus-grade latency — several seconds of nothing. Safest, and it throws away
the main benefit of streaming.

**Stream the *trace*, not the answer.** Send progress events ("retrieving", "generating", "verifying")
while working, then deliver the verified answer in one piece. The user sees continuous activity
without seeing unverified content.

For a legal system I would take the third, and it is why `AgentState.trace` exists and why every node
returns an entry for it. That decision belongs to Phase 8; the state is shaped to support it.

### Failure Modes

**`MaxRetriesExceededError: exceeded its step limit`.** An edge bug. The `details` carry the limit,
and the last state's trace shows the cycle.

**Checkpointer errors about a missing `thread_id`.** LangGraph requires one when a checkpointer is
set. `answer()` supplies the request ID by default; if you pass a checkpointer some other way, pass a
thread too.

**Every answer carries the caveat.** Grading is failing everything. Debug the grader (§8) rather than
removing the caveat.

**The trace shows `router → retrieval → generation → grading` and stops on a failed grade.** Look for
`grading:disabled`, a `retry_count` already at the budget, or an empty `answer`. All three are
`accept` branches in `after_grading`.

---

## 12. Verification (deferred)

Save as `scripts/verify_phase5.py`. It needs Qdrant but **no LLM API key** — a scripted provider
supplies canned responses, which is the only way to test the retry path deterministically.

```python
"""Phase 5 verification. Run from the project root, with Qdrant running:

    python scripts/verify_phase5.py

No API key required. A scripted LLM returns canned responses, which is what makes
the FAILING-grade retry path testable at all: you cannot reliably make a real model
produce an ungrounded answer on demand.
"""
import asyncio
import sys
import uuid
from typing import Any

from src.core.exceptions import MaxRetriesExceededError
from src.core.models import AnswerStatus, Chunk, ChunkLevel, GradingReport
from src.core.utils import make_chunk_id, make_doc_id, make_section_id
from src.embeddings.fastembed_provider import FastEmbedProvider
from src.graph.builder import RAGAgent
from src.graph.edges import after_grading, after_retrieval, after_router
from src.graph.nodes.generation_node import extract_citations, format_context
from src.graph.state import initial_state
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstores.qdrant_store import QdrantStore

TEMP_COLLECTION = f"verify_phase5_{uuid.uuid4().hex[:8]}"
DOC_ID = make_doc_id("data/contracts/2003/verify5.txt")

CLAUSES = {
    "Termination": (
        "Either party may terminate this Agreement for convenience upon ninety (90) "
        "days prior written notice. Upon termination the Company shall pay all "
        "amounts accrued through the effective date."
    ),
    "Indemnification": (
        "The Company shall indemnify the Purchaser against all Losses arising from "
        "a breach, provided the aggregate liability shall not exceed $5,000,000."
    ),
    "Governing Law": (
        "This Agreement shall be governed by the laws of the State of Delaware "
        "without regard to conflict of laws principles."
    ),
}


class ScriptedLLM:
    """A `BaseLLMProvider` that returns queued responses.

    Satisfies the interface structurally. Each method pops from its own queue, so
    a test can specify exactly which grade the second attempt receives.
    """

    def __init__(
        self,
        answers: list[str] | None = None,
        grades: list[GradingReport] | None = None,
        rewrites: list[str] | None = None,
        router_labels: list[str] | None = None,
    ) -> None:
        self.answers = list(answers or [])
        self.grades = list(grades or [])
        self.rewrites = list(rewrites or [])
        self.router_labels = list(router_labels or [])
        self.generate_calls = 0
        self.json_calls = 0
        #: Every `model` argument received, so a test can assert that the configured
        #: per-task models are actually being passed.
        self.models_used: list[str | None] = []

    async def generate(self, prompt: str, system_prompt: str | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       model: str | None = None) -> str:
        self.generate_calls += 1
        self.models_used.append(model)
        return self.answers.pop(0) if self.answers else "No answer available."

    async def generate_json(self, prompt: str, schema: type,
                            system_prompt: str | None = None,
                            model: str | None = None) -> Any:
        self.json_calls += 1
        self.models_used.append(model)
        name = schema.__name__

        if name == "GradingReport":
            return self.grades.pop(0) if self.grades else GradingReport(
                is_grounded=True, is_relevant=True, confidence=0.9
            )
        if name == "RewrittenQuery":
            text = self.rewrites.pop(0) if self.rewrites else "termination notice period days"
            return schema(query=text)
        if name == "QueryClass":
            label = self.router_labels.pop(0) if self.router_labels else "vague"
            return schema(label=label)
        if name == "QueryVariations":
            return schema(variations=[])
        return schema()

    def stream(self, prompt: str, system_prompt: str | None = None):
        raise NotImplementedError

    async def close(self) -> None:
        return None


def build_corpus() -> list[Chunk]:
    """Parent/child pairs, with Phase 2's parent index offset."""
    chunks: list[Chunk] = []
    for order, (title, text) in enumerate(CLAUSES.items()):
        section_id = make_section_id(DOC_ID, title, order)
        parent_index = 1_000_000 + order
        parent_id = make_chunk_id(DOC_ID, section_id, parent_index)

        def make(body: str, index: int, level: ChunkLevel, parent: str | None) -> Chunk:
            return Chunk(
                chunk_id=make_chunk_id(DOC_ID, section_id, index),
                doc_id=DOC_ID, section_id=section_id, text=body,
                chunk_index=index, token_count=len(body) // 4,
                contract_name="Phase 5 Verification Agreement",
                file_name="verify5.txt", section_title=title, year=2003,
                chunk_level=level, parent_id=parent,
            )

        chunks.append(make(text, parent_index, ChunkLevel.PARENT, None))
        chunks.append(make(text.split(".")[0] + ".", order * 10, ChunkLevel.CHILD, parent_id))
    return chunks


def check_context_and_citations() -> None:
    """Source numbering must map back exactly, and bad markers must be dropped."""
    from src.core.models import RetrievalMethod, ScoredChunk

    chunks = [c for c in build_corpus() if c.chunk_level == ChunkLevel.PARENT]
    sources = [
        ScoredChunk(chunk=c, score=0.9 - i * 0.1, rank=i, method=RetrievalMethod.HYBRID)
        for i, c in enumerate(chunks)
    ]

    context, mapping = format_context(sources)

    assert "[SOURCE 1]" in context and "[SOURCE 3]" in context, "sources must be numbered from 1"
    assert "[SOURCE 0]" not in context, "numbering must start at 1, not 0"
    assert set(mapping) == {1, 2, 3}, f"mapping keys should be 1..3, got {sorted(mapping)}"
    for index, scored in mapping.items():
        assert scored.chunk.text in context, "a mapped chunk is missing from the context"
        # The mapping must agree with the ORDER in the context block.
        assert context.index(f"[SOURCE {index}]") == context.index(
            f"[SOURCE {index}] {scored.chunk.contract_name}"
        )

    citations, invalid = extract_citations(
        "The notice period is ninety days [SOURCE 1]. The cap is $5,000,000 "
        "[SOURCE 2]. Unrelated [SOURCE 9]. Repeat [SOURCE 1].",
        mapping,
    )
    assert [c.source_index for c in citations] == [1, 2], (
        f"expected citations to 1 and 2 only, got {[c.source_index for c in citations]} — "
        "out-of-range markers must be dropped and duplicates collapsed"
    )
    assert invalid == 1, (
        f"expected 1 invalid marker to be COUNTED, got {invalid} — a fabricated "
        "citation must survive as structured data, not only as a log line"
    )
    assert citations[0].chunk_id == mapping[1].chunk.chunk_id, "citation points at the wrong chunk"
    assert citations[0].section_title == mapping[1].chunk.section_title

    print("✓ context/citations: numbering maps back exactly, bad markers dropped")


def check_edges() -> None:
    """Control flow is a pure function of state — test it with dicts."""
    from config.settings import settings

    state = initial_state("q", top_k=5)

    state["route"] = "unsupported"
    assert after_router(state) == "unsupported"
    state["route"] = "retrieve"
    assert after_router(state) == "retrieve"

    state["sources"] = []
    assert after_retrieval(state) == "no_context", (
        "empty retrieval MUST skip generation — generating from no context is the "
        "worst output this system can produce"
    )

    state["answer"] = "an answer"
    state["grading"] = GradingReport(is_grounded=True, is_relevant=True, confidence=0.9)
    assert after_grading(state) == "accept", "a passing grade must accept"

    state["grading"] = GradingReport(is_grounded=False, is_relevant=True, confidence=0.9)
    state["retry_count"] = 0
    assert after_grading(state) == "retry", "a failing grade with budget left must retry"

    state["retry_count"] = settings.MAX_RETRIES
    assert after_grading(state) == "accept", "an exhausted budget must stop, not loop"

    state["retry_count"] = 0
    state["answer"] = ""
    assert after_grading(state) == "accept", (
        "a generation failure must not be retried — rewriting the query does not "
        "fix an LLM outage"
    )

    print("✓ edges: every branch reachable, exhaustion and generation-failure both terminate")


async def check_happy_path(pipeline: RetrievalPipeline) -> None:
    llm = ScriptedLLM(
        answers=["The notice period is ninety (90) days [SOURCE 1]."],
        grades=[GradingReport(is_grounded=True, is_relevant=True, confidence=0.95)],
        router_labels=["vague"],
    )
    agent = RAGAgent(llm=llm, pipeline=pipeline)
    result = await agent.answer("how much notice to terminate?")

    assert result.retry_count == 0, f"no retry expected, got {result.retry_count}"
    assert result.citations, "the answer cited [SOURCE 1] but no Citation was produced"
    assert result.grading is not None and result.grading.passed
    assert result.status is AnswerStatus.ANSWERED, f"status was {result.status}"
    assert "Note:" not in result.answer, "a passing answer must not carry the caveat"
    assert llm.generate_calls == 1, f"expected one generation, got {llm.generate_calls}"
    assert result.total_latency_ms > 0
    assert result.sources, "sources must be attached to the answer"
    assert result.invalid_citations == 0

    # RETRIEVAL CORRECTNESS, not just "an answer came back". A ScriptedLLM returns
    # its canned answer regardless of what was retrieved, so without this the whole
    # end-to-end test passes while the pipeline fetches the wrong clause entirely.
    assert result.citations[0].section_title == "Termination", (
        f"the cited source was {result.citations[0].section_title!r}, not the "
        "Termination clause — retrieval is wrong even though the answer looks right"
    )
    assert "ninety (90) days" in result.sources[0].chunk.text, (
        "the top source does not contain the notice period the question asked about"
    )

    print(f"✓ happy path: 1 generation, {len(result.citations)} citation(s), no retry, "
          f"cited the Termination clause")


async def check_retry_then_pass(pipeline: RetrievalPipeline) -> None:
    """The reason this phase exists: a failing grade must trigger a real retry."""
    llm = ScriptedLLM(
        answers=[
            "The cap is $10,000,000 [SOURCE 1].",       # wrong — will fail grading
            "The cap is $5,000,000 [SOURCE 1].",        # corrected
        ],
        grades=[
            GradingReport(is_grounded=False, is_relevant=True, confidence=0.9,
                          reasoning="The stated cap does not appear in the sources.",
                          unsupported_claims=["aggregate liability cap of $10,000,000"]),
            GradingReport(is_grounded=True, is_relevant=True, confidence=0.95),
        ],
        rewrites=["aggregate liability cap indemnification amount"],
        router_labels=["vague"],
    )
    agent = RAGAgent(llm=llm, pipeline=pipeline)
    result = await agent.answer("what is the liability cap?")

    assert result.retry_count == 1, f"expected exactly 1 retry, got {result.retry_count}"
    assert llm.generate_calls == 2, f"expected 2 generations, got {llm.generate_calls}"
    assert "5,000,000" in result.answer, "the second, corrected answer should be returned"
    assert result.grading is not None and result.grading.passed
    assert "Note:" not in result.answer, "the final grade passed, so no caveat"

    print("✓ retry: failing grade → rewrite → second attempt → pass")


async def check_budget_exhaustion(pipeline: RetrievalPipeline) -> None:
    """Persistent failure must terminate and return the answer WITH a caveat."""
    from config.settings import settings

    failing = GradingReport(
        is_grounded=False, is_relevant=True, confidence=0.9,
        reasoning="Unsupported.", unsupported_claims=["a claim"],
    )
    llm = ScriptedLLM(
        answers=[f"Attempt {i} [SOURCE 1]." for i in range(1, 6)],
        grades=[failing] * 6,
        rewrites=[f"rewritten query number {i}" for i in range(1, 6)],
        router_labels=["vague"],
    )
    agent = RAGAgent(llm=llm, pipeline=pipeline)
    result = await agent.answer("what is the termination fee?")

    assert result.retry_count == settings.MAX_RETRIES, (
        f"should stop at MAX_RETRIES={settings.MAX_RETRIES}, got {result.retry_count}"
    )
    assert llm.generate_calls == settings.MAX_RETRIES + 1, (
        f"expected {settings.MAX_RETRIES + 1} generations, got {llm.generate_calls}"
    )
    assert "Note:" in result.answer, (
        "an answer that never passed must carry the caveat — returning it silently "
        "is worse than returning nothing"
    )
    assert result.grading is not None and not result.grading.passed, (
        "the failing grade must be attached so callers can see WHY"
    )

    print(f"✓ exhaustion: stopped after {settings.MAX_RETRIES} retries, caveat attached, "
          "grading preserved")


async def check_grader_outage(pipeline: RetrievalPipeline) -> None:
    """A grader that FAILS must not produce a passing grade.

    This is the check that catches the phase's most dangerous bug: a fail-open
    fallback setting is_grounded=is_relevant=True makes `passed` True, so every
    caller checking `.passed` treats an unaudited answer as verified. `verified=False`
    is what makes that impossible.
    """
    class BrokenGraderLLM(ScriptedLLM):
        async def generate_json(self, prompt: str, schema: type,
                                system_prompt: str | None = None,
                                model: str | None = None):
            if schema.__name__ == "GradingReport":
                # Not a RAGException — exactly the case that must still be caught.
                raise ValueError("simulated malformed grader output")
            return await super().generate_json(prompt, schema, system_prompt, model)

    llm = BrokenGraderLLM(
        answers=["The notice period is ninety (90) days [SOURCE 1]."],
        router_labels=["vague"],
    )
    agent = RAGAgent(llm=llm, pipeline=pipeline)
    result = await agent.answer("how much notice to terminate?")

    assert result.grading is not None, "a grading report must still be attached"
    assert result.grading.verified is False, "a failed audit must be marked unverified"
    assert result.grading.passed is False, (
        "an UNAUDITED answer must not report passed=True — this is the bug that makes "
        "a grader outage silently upgrade every answer to 'verified'"
    )
    assert result.status is AnswerStatus.UNVERIFIED, f"status was {result.status}"
    assert "verification step did not run" in result.answer, (
        "an unverified answer needs the unverified caveat, not the failed-audit one"
    )
    assert result.retry_count == 0, (
        "an unverified answer must NOT trigger a rewrite — rewriting the query does "
        "not fix a broken grader"
    )
    assert llm.generate_calls == 1, f"expected 1 generation, got {llm.generate_calls}"

    print("✓ grader outage: unverified, not passed, not retried, distinct caveat")


async def check_checkpoint_resume(pipeline: RetrievalPipeline) -> None:
    """Checkpointing is a headline reason for using LangGraph, so it gets tested.

    Uses `InMemorySaver` so the test needs no filesystem. Verifies that a thread ID
    comes back to the caller (an auto-generated ID nobody sees is not resumable) and
    that separate questions do not contaminate each other's reduced state.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    llm = ScriptedLLM(
        answers=["First answer [SOURCE 1].", "Second answer [SOURCE 1]."],
        grades=[
            GradingReport(is_grounded=True, is_relevant=True, confidence=0.9),
            GradingReport(is_grounded=True, is_relevant=True, confidence=0.9),
        ],
        router_labels=["vague", "vague"],
    )
    agent = RAGAgent(llm=llm, pipeline=pipeline, checkpointer=saver)

    first = await agent.answer("what notice is required to terminate?")
    assert first.thread_id, "a checkpointed run must return its thread_id"

    second = await agent.answer("what is the governing law?")
    assert second.thread_id and second.thread_id != first.thread_id, (
        "each question must get its own thread — sharing one appends the new run's "
        "trace and attempts onto the old run's, because those keys have reducers"
    )

    # The saved state must be readable and belong to the right question.
    snapshot = await agent.graph.aget_state(
        {"configurable": {"thread_id": first.thread_id}}
    )
    assert snapshot is not None, "no checkpoint was written"
    assert snapshot.values["query"].startswith("what notice"), (
        "the checkpoint holds the wrong question"
    )
    assert len(snapshot.values["trace"]) >= 4, (
        f"trace should record every node, got {snapshot.values['trace']}"
    )
    assert snapshot.values["trace"].count("generation") == 1, (
        "the trace contains entries from more than one run — thread contamination"
    )

    print(f"✓ checkpointing: distinct threads, state readable, "
          f"trace={' → '.join(snapshot.values['trace'])}")


async def check_input_validation(pipeline: RetrievalPipeline) -> None:
    """A malformed request must not be reported as an agent or retrieval failure."""
    from src.core.exceptions import InvalidQueryError

    llm = ScriptedLLM()
    agent = RAGAgent(llm=llm, pipeline=pipeline)

    for bad in ("", "   "):
        try:
            await agent.answer(bad)
        except InvalidQueryError:
            pass
        else:
            raise AssertionError(f"a blank question ({bad!r}) must raise InvalidQueryError")

    for bad_k in (0, -3):
        try:
            await agent.answer("valid question", top_k=bad_k)
        except InvalidQueryError:
            pass
        else:
            raise AssertionError(f"top_k={bad_k} must raise rather than be coerced")

    assert llm.generate_calls == 0, "invalid input must be rejected before any LLM call"
    print("✓ validation: blank questions and non-positive top_k rejected early")


async def check_no_answer_raises(pipeline: RetrievalPipeline) -> None:
    """No answer at all is the one case that raises."""
    llm = ScriptedLLM(answers=[""], grades=[
        GradingReport(is_grounded=False, is_relevant=False, confidence=1.0)
    ], router_labels=["vague"])
    agent = RAGAgent(llm=llm, pipeline=pipeline)

    try:
        await agent.answer("anything")
    except MaxRetriesExceededError as exc:
        assert "trace" in exc.details, "the failure must carry the trace for debugging"
        print("✓ no-answer: raises MaxRetriesExceededError with a trace")
    else:
        raise AssertionError("an empty answer must raise, not return an empty RAGAnswer")


async def check_generation_outage(pipeline: RetrievalPipeline) -> None:
    """A generation failure must raise an LLM error, not retry-budget exhaustion.

    Reporting a 502 from the provider as MaxRetriesExceededError gives the wrong HTTP
    status and points whoever is on call at the wrong component.
    """
    from src.core.exceptions import LLMProviderError

    class DeadGeneratorLLM(ScriptedLLM):
        async def generate(self, prompt: str, system_prompt: str | None = None,
                           temperature: float = 0.0, max_tokens: int | None = None,
                           model: str | None = None) -> str:
            raise LLMProviderError("simulated provider outage", retryable=True)

    agent = RAGAgent(llm=DeadGeneratorLLM(router_labels=["vague"]), pipeline=pipeline)

    try:
        await agent.answer("what notice is required?")
    except LLMProviderError as exc:
        assert exc.retryable, "a provider outage should be retryable"
        assert "cause" in exc.details, "the original exception type must be preserved"
        print("✓ generation outage: raises LLMProviderError, not MaxRetriesExceeded")
    except MaxRetriesExceededError:
        raise AssertionError(
            "a generation outage was reported as retry-budget exhaustion — the "
            "terminal-status mapping is wrong"
        )


async def check_aggregate_routing(pipeline: RetrievalPipeline) -> None:
    """A question top-k cannot answer must be refused, not answered badly."""
    llm = ScriptedLLM(answers=["A fabricated count [SOURCE 1]."], router_labels=["aggregate"])
    agent = RAGAgent(llm=llm, pipeline=pipeline)
    result = await agent.answer("how many contracts are governed by Delaware law?")

    assert llm.generate_calls == 0, (
        "an aggregate query must never reach generation — a fabricated count is "
        "worse than a refusal"
    )
    assert "aggregation" in result.answer.lower() or "Phase 12" in result.answer
    assert result.citations == []

    print("✓ routing: aggregate query refused without generating")


async def main() -> int:
    check_context_and_citations()
    check_edges()

    provider = FastEmbedProvider()
    provider.warmup()
    store = QdrantStore(collection_name=TEMP_COLLECTION)

    try:
        await store.initialize(provider.dense_dimensions)
        corpus = build_corpus()
        texts = [c.text for c in corpus]
        await store.upsert_points(
            corpus,
            await provider.embed_dense(texts),
            await provider.embed_sparse(texts),
        )

        pipeline = RetrievalPipeline(embedder=provider, store=store, llm=None)
        pipeline.warmup()

        await check_happy_path(pipeline)
        await check_retry_then_pass(pipeline)
        await check_budget_exhaustion(pipeline)
        await check_grader_outage(pipeline)
        await check_generation_outage(pipeline)
        await check_no_answer_raises(pipeline)
        await check_aggregate_routing(pipeline)
        await check_input_validation(pipeline)
        await check_checkpoint_resume(pipeline)
    except AssertionError as exc:
        print(f"\n✗ FAILED: {exc}")
        return 1
    finally:
        try:
            await store._client.delete_collection(TEMP_COLLECTION)
        finally:
            await store.close()
            # The provider lifecycle contract applies to test code too. `close()` is
            # a no-op on FastEmbed, which is exactly why calling it unconditionally
            # costs nothing and keeps the habit intact.
            await provider.close()

    print("\nPhase 5 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

`check_budget_exhaustion` reads `settings.MAX_RETRIES` rather than hardcoding 2, so it stays correct
when you tune the budget. If it fails with a generation count one higher than expected, the retry
counter is being incremented in two places.

### What to look at, beyond the assertions

**The trace on each run.** Log it and read it once by hand:
`router:vague → retrieval:3 → generation → grading:fail → rewrite:1 → retrieval:3 → generation →
grading:pass`. That line is the phase working, and it is what you show someone who asks what
"self-correcting" means.

**The generation count in the retry test.** Exactly `retries + 1`. Anything else means the loop is
running nodes it should not.

**Once Phase 6 supplies a real LLM,** re-run the happy path and check that `grading.confidence` is not
pinned at some constant. A grader returning 0.9 for everything is not grading — it is echoing the
example in the prompt.

---

## 13. What Phase 5 Bought You

**An agent that checks its own work.** Every answer is audited for grounding and relevance against
the sources it was given, and a failure triggers a genuine retry with a query rewritten from the
diagnosis. This is the feature that distinguishes this project from a retrieval demo, and it is
verifiable end to end without an API key.

**Honest failure at every boundary.** Empty retrieval never reaches generation. Aggregate questions
are refused with an explanation instead of answered with a fabricated count. An answer that never
passed its audit is returned *with* its grading and a caveat, not silently.

**Control flow you can test.** The four edge functions are pure functions of a dict. "Does it stop
when the budget is spent" is an assertion, not a hope.

**A shape the next four phases plug into.** Phase 6 replaces the LLM and wraps `answer()` in a cache.
Phase 8 streams `trace` over SSE. Phase 12 adds a map-reduce node and one router branch — which is
exactly the `unsupported` branch this phase already routes aggregate queries into. Phase 13 adds
graph traversal the same way.

**Every prompt in one versioned file.** `PROMPT_VERSION` travels with Phase 7's evaluation results, so
a score change can be attributed to a prompt edit instead of guessed at.

### What is deliberately not here

**The LLM provider itself.** Phase 6: Groq, the model-ID resolver, retries, the semantic cache, and
the guardrails. Nodes depend on `BaseLLMProvider` and nothing else.

**The deterministic citation validator.** Also Phase 6. The grader is an LLM checking claims by
inference; the validator is code checking that cited chunk IDs were actually in the context. They
catch different failures and the second one cannot be argued with.

**Streaming.** Phase 8, and §11 explains why the decision is not obvious: you cannot un-stream an
ungrounded answer. The `trace` field exists so Phase 8 can stream progress instead of unverified
content.

**Map-reduce for aggregate queries.** Phase 12. This phase's contribution is *recognising* them and
refusing, which is what makes the eventual feature an addition rather than a fix.

---

## Next

**Phase 6 — LLM providers, cache, and guardrails.** It supplies the `BaseLLMProvider` every node in
this phase is written against: Groq with a model-ID resolver (the handoff records that
`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are scheduled for shutdown, so this must fail
loudly at startup rather than mid-run), reliable JSON mode for `generate_json`, an OpenAI fallback, a
semantic cache in front of `RAGAgent.answer`, and the deterministic citation validator that sits
underneath the grader.
