# Phase 4 — Hybrid Retrieval and Reranking

> **Prerequisite:** Phases 1 and 3 complete. Phase 2 matters only in that it produced the parent and
> child chunks this phase exploits.
>
> **Budget:** ~1,200 lines of Python across 9 files. (Budgeted at 700 in the roadmap; the 1.5–2×
> calibration holds. Two files — `parents.py` and `mmr.py` — carry most of the excess, and both are
> load-bearing rather than optional.)
>
> **Does NOT depend on Phase 6.** Query expansion and HyDE need an LLM, and Phase 6 is where LLM
> providers get built. This phase depends only on Phase 1's `BaseLLMProvider` *interface* and takes a
> provider as an **optional injected dependency**. With no LLM it degrades to single-query hybrid
> retrieval and says so in the logs. §3 explains why that is the right call rather than a compromise.
>
> **This file replaces the previous `Phase4_Hybrid_Retrieval_Reranking.md` entirely.** The old draft
> merged dense and sparse results in Python (Phase 3 does it server-side), passed `dict`s, and
> reranked before it had anything worth reranking.

---

## 1. What Makes This Phase Hard

Phase 3 ended with a working `hybrid_search`. Point it at your corpus and you get results that are
*mostly* fine, which is the worst possible starting condition — good enough to look finished, bad
enough to fail the queries that matter. This phase is about the gap between "returns relevant-looking
passages" and "returns the right clause".

Four specific failures stand between those two, and each file below exists to close one.

**The user's words are not the document's words.** Someone asks "can I get out of this contract
early?" The contract says "Termination for Convenience". No lexical overlap, and only partial
semantic overlap — the embedding of a casual question sits in a different neighbourhood than the
embedding of formal legal prose. One query gets one shot at that gap. Several differently-phrased
queries get several.

**Retrieval and generation want different chunk sizes, and Phase 2 already exploited that.** The
index holds ~400-token children (precise to search) and ~1,600-token parents (complete enough to
answer from). Everything in this phase searches children. Something has to swap in the parents before
generation, and doing it wrong quadruples your prompt or silently duplicates context.

**Bi-encoders cannot tell "relevant" from "on the same topic".** The dense side of hybrid search
compares two vectors that were computed *without ever seeing each other*. That is what makes it fast
enough to search 39 million points, and it is also its ceiling. A cross-encoder that reads the query
and the passage together is dramatically more accurate and far too slow to search with. The
resolution — retrieve wide and cheap, then re-score narrow and expensive — is the single biggest
quality win available here.

**More results is not better context.** Twenty retrieved chunks about the same clause crowd out the
one chunk about the *other* clause the question also needed. The top-5 by score can be five
paraphrases of each other. Diversity has to be selected for deliberately.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| | **Search** | | |
| 1 | `src/retrieval/multi_query.py` | 130 | One question → N phrasings |
| 2 | `src/retrieval/hyde.py` | 110 | Query → hypothetical answer → embedding |
| 3 | `src/retrieval/hybrid.py` | 180 | Embed queries, search children, concurrently |
| 4 | `src/retrieval/fusion.py` | 130 | RRF *across queries* + deduplication |
| | **Refine** | | |
| 5 | `src/retrieval/rerankers/base.py` | 70 | Pair building, truncation, rank rewriting |
| 6 | `src/retrieval/rerankers/cross_encoder.py` | 170 | Joint query-passage scoring |
| 7 | `src/retrieval/rerankers/mmr.py` | 140 | Relevance-vs-diversity selection |
| | **Assemble** | | |
| 8 | `src/retrieval/parents.py` | 100 | Child → parent substitution, deduplicated |
| 9 | `src/retrieval/pipeline.py` | 200 | The orchestrator, returns `RetrievalResult` |

### Directory to create

```text
src/retrieval/
├── __init__.py
├── multi_query.py
├── hyde.py
├── hybrid.py
├── fusion.py
├── parents.py
└── rerankers/
    ├── __init__.py
    ├── base.py
    ├── cross_encoder.py
    └── mmr.py
```

`parents.py` is not in the roadmap's tree; I am adding it. Parent substitution is neither search nor
reranking, it is ~100 lines with real token-budget logic, and Phase 16 extends the same mechanism to
RAPTOR summary levels. Folding it into `pipeline.py` would bury it. Update the tree.

### The pipeline, end to end

```text
        "can I get out of this contract early?"
                        │
      ┌─────────────────┴─────────────────┐
      ▼                                   ▼
 multi_query.py                        hyde.py            (both optional,
 3 phrasings                    1 hypothetical clause      both need an LLM)
      │                                   │
      └─────────────────┬─────────────────┘
                        ▼
                   hybrid.py          N queries → N × store.hybrid_search
                        │             concurrently, filtered to chunk_level=child
                        │             (each search fuses dense+sparse server-side)
                        ▼
                   fusion.py          RRF across the N result lists,
                        │             deduplicated by chunk_id  →  ~20 candidates
                        ▼
              rerankers/cross_encoder  joint scoring, sets rerank_score  →  top 5
                        │
                        ▼
                 rerankers/mmr         optional: swap a redundant hit for a
                        │              different one
                        ▼
                   parents.py          each child → its 1,600-token parent,
                        │              deduplicated, token-capped
                        ▼
              RetrievalResult(chunks, expanded_queries, total_candidates, latency_ms)
```

Note the ordering of the last two stages, because it is a decision and not an accident: **rerank
children, then substitute parents.** Reversed, the cross-encoder would receive 1,600-token parents
against a 512-token model window — three quarters of each passage silently truncated away, and the
scores would reflect whatever happened to be in the first quarter. Reranking small and substituting
large is the only order that works.

---

## 2. Before Any Code — Settings and One Correction

### A correction to Phase 1

The Phase 1 guide originally specified:

```python
    RERANK_MODEL_NAME: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

That is the **sentence-transformers** name. We are using FastEmbed's ONNX cross-encoder, whose
registry knows the same weights as `Xenova/ms-marco-MiniLM-L-6-v2`; the old value fails at model load
with an unsupported-model error. **I have already corrected the Phase 1 guide and its `.env.example`
block** — but if you typed `config/settings.py` before now, change the default and your `.env` by
hand:

```python
    RERANK_MODEL_NAME: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
```

Why FastEmbed rather than `sentence-transformers`, given the model is identical: `sentence-transformers`
pulls in PyTorch — a multi-hundred-megabyte dependency with CUDA wheels — to run a 90 MB model on
CPU. FastEmbed is already a dependency and runs the ONNX export. Same weights, same scores, no torch.

### New settings

Add to the **RAG hyperparameters** block:

```python
    ENABLE_MULTI_QUERY: bool = Field(default=True)
    ENABLE_HYDE: bool = Field(default=False)
    ENABLE_RERANKING: bool = Field(default=True)
    ENABLE_MMR: bool = Field(default=False)
    ENABLE_PARENT_SUBSTITUTION: bool = Field(default=True)

    RRF_K: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant. 60 is the value from the original paper.",
    )
    MMR_LAMBDA: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="1.0 = pure relevance, 0.0 = pure diversity.",
    )
    MAX_CONTEXT_TOKENS: int = Field(
        default=12_000,
        ge=1000,
        description="Token ceiling for the assembled context after parent substitution.",
    )
```

Add to `validate_runtime()`:

```python
        if self.ENABLE_HYDE and self.ENABLE_MULTI_QUERY:
            warnings.append(
                "HyDE and multi-query are both on: every question costs two LLM calls "
                "before retrieval even starts."
            )

        if not self.ENABLE_RERANKING:
            warnings.append(
                "Reranking is disabled — retrieval precision will be materially worse."
            )
```

And `.env.example`:

```bash
# ─── Phase 4: retrieval ────────────────────────────────────────────────────
ENABLE_MULTI_QUERY=true
ENABLE_HYDE=false
ENABLE_RERANKING=true
ENABLE_MMR=false
ENABLE_PARENT_SUBSTITUTION=true
RRF_K=60
MMR_LAMBDA=0.7
MAX_CONTEXT_TOKENS=12000
```

Every one of these defaults to the setting I would ship, and two of them default to `false`. HyDE and
MMR are real techniques with real costs that this corpus may not need; §5 and §9 make the case for and
against each. Do not turn on a retrieval feature you have not measured — Phase 7 exists to measure
them.

### One addition to the frozen interface

Parent substitution needs to fetch specific points by ID, and `BaseVectorStore` has no method for it.
Filtering cannot substitute: the indexed filter fields are `doc_id`, `chunk_level`, `year`, and
`section_title` — none of them is `chunk_id`.

So `BaseVectorStore` gains one method. This is an addition, not a change; nothing that already exists
behaves differently. Add to `src/core/interfaces.py`:

```python
    @abstractmethod
    async def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Retrieve specific points by ID, without a vector search.

        Returns only the chunks that exist. A missing ID is not an error —
        Phase 4's parent substitution treats a missing parent as "use the child",
        which is a correct degradation rather than a failure.
        """
```

(`Sequence` comes from `collections.abc`.) I have added the corresponding implementations to the
Phase 3 guide for both `QdrantStore` and `ChromaStore`; if you have already typed those files, the
two methods are reproduced in §10 below.

---

## 3. File 1 — `src/retrieval/multi_query.py`

### The Problem

A single query vector is a single point in embedding space, and retrieval returns its neighbours. If
the question is phrased unlike the document, that point lands in the wrong neighbourhood and no amount
of tuning `top_k` recovers the right passage — it was never a neighbour to begin with.

Concretely, on this corpus:

| The user asks | The contract says |
| :--- | :--- |
| "can I get out of this early?" | "Termination for Convenience" |
| "what if they leak our data?" | "Breach of Confidentiality Obligations" |
| "who pays if we get sued?" | "Indemnification of Losses" |

This is **vocabulary mismatch**, and it is the oldest problem in information retrieval. Dense
embeddings reduce it substantially compared to keyword search — that is their entire selling point —
but they do not eliminate it, especially across a register gap as wide as casual English to drafted
legal English.

### Design Decision

**Ask an LLM for several phrasings, search with all of them, fuse the results.** Three variations plus
the original gives four shots at the neighbourhood. Recall improves markedly; the cost is one LLM call
and 4× the search work.

**The LLM is injected and optional.** `QueryExpander(llm=None)` returns `[original]` and logs it once.
This matters for three reasons that are worth stating plainly: Phase 6 (LLM providers) has not been
written yet and this phase must not block on it; Phase 10 needs to test retrieval without an API key;
and an LLM outage should degrade a RAG system to plain hybrid search, not take it down. A hard
dependency on a network service in the *query rewriting* stage would be a poor trade.

**Output is validated by Pydantic via `generate_json`.** Phase 1 put `generate_json(prompt, schema)`
on `BaseLLMProvider` for exactly this. Asking for prose and splitting on newlines works right up until
the model returns "Sure! Here are three variations:" as line one, and then you search for that.

**The original query always survives, first.** It is the only phrasing you know the user meant. An
expansion that replaces it can only lose information.

```python
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.exceptions import RAGException
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger
from src.core.telemetry import telemetry

_SYSTEM_PROMPT = """You rewrite search queries for a legal contract database.

The database contains commercial agreements: purchase agreements, employment
contracts, NDAs, licensing deals. Passages are written in formal legal English.

Rewrite the user's question into alternative search queries that would match the
CONTRACT'S wording rather than the user's. Prefer the drafted legal term over the
colloquial one: "termination for convenience" over "cancel early", "indemnify"
over "pay if sued", "confidential information" over "secrets".

Rules:
- Each variation is a standalone search query, not a question to a person.
- Vary the terminology, not just the word order. Two variations that differ only
  in punctuation are worth nothing.
- Do not invent facts, parties, dates, or amounts that are not in the question.
- Return only the JSON object."""


class QueryVariations(BaseModel):
    """Schema the LLM must fill. Validation is the point of this class.

    `max_length` on the list is a guard against a model that decides to return
    forty variations — which would multiply the search cost by forty.
    """

    variations: list[str] = Field(default_factory=list, max_length=10)


class QueryExpander:
    """One question → several phrasings that might match the corpus.

    The LLM is optional. Without it this class is an identity function that says
    so, which keeps retrieval working during an LLM outage and lets Phase 10 test
    the search path without an API key.
    """

    def __init__(
        self,
        llm: BaseLLMProvider | None = None,
        num_variations: int | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.num_variations = num_variations or settings.NUM_QUERY_VARIATIONS
        self.model = model or settings.EXPANSION_MODEL
        self._warned = False

    @telemetry.measure_async("retrieval.expand")
    async def expand(self, query: str) -> list[str]:
        """Return the original query followed by distinct variations.

        Never raises and never returns an empty list. Expansion is an enhancement:
        if it fails, searching with the original query is a perfectly good outcome,
        and turning a rewriting failure into a request failure would be the wrong
        trade. The failure is logged at WARNING so it is visible.
        """
        if not settings.ENABLE_MULTI_QUERY or self.llm is None:
            if not self._warned:
                logger.info(
                    "Query expansion disabled",
                    extra={"reason": "no LLM" if self.llm is None else "config"},
                )
                self._warned = True
            return [query]

        try:
            result = await self.llm.generate_json(
                prompt=(
                    f"Produce {self.num_variations} alternative search queries.\n\n"
                    f"Question: {query}"
                ),
                schema=QueryVariations,
                system_prompt=_SYSTEM_PROMPT,
            )
        except RAGException as exc:
            logger.warning(
                "Query expansion failed; falling back to the original query",
                extra={"error": type(exc).__name__},
            )
            return [query]

        return self._merge(query, result.variations)

    def _merge(self, query: str, variations: list[str]) -> list[str]:
        """Original first, then distinct non-empty variations, capped.

        Deduplication is case-insensitive and whitespace-normalised, because a
        variation differing only in capitalisation embeds to nearly the same vector
        and would buy a second search for nothing.
        """
        queries = [query]
        seen = {query.strip().lower()}

        for variation in variations:
            candidate = (variation or "").strip()
            key = candidate.lower()
            if not candidate or key in seen:
                continue
            seen.add(key)
            queries.append(candidate)
            if len(queries) > self.num_variations:
                break

        logger.info(
            "Query expanded",
            extra={"variations": len(queries) - 1, "requested": self.num_variations},
        )
        return queries
```

### The Theory: why several queries beat one better query

The instinct is that expansion is a workaround for a weak embedding model, and that a better model
would make it unnecessary. That is not quite right, and the reason is worth understanding.

An embedding maps text to one point. For an *ambiguous* query, there is no correct single point.
"Termination" in a commercial agreement might mean termination for cause, termination for
convenience, termination of employment, or the expiry of a term — four genuinely different regions of
the space. The true answer set is **multi-modal**, and no single vector can be near all four modes at
once. A better model places each mode more accurately; it does not let one point occupy four places.

Multi-query is therefore not error correction, it is **coverage**. Each variation probes a different
mode, and the fusion step in §6 rewards passages that several probes agree on.

The honest costs, all three of them:

**Latency in the critical path.** One expansion call at 200–600ms happens before any search starts.
For an interactive system that is the difference between fast and adequate. Phase 6's cache makes
repeat questions free, which helps more than it sounds like — real query distributions have a heavy
head.

**Search work multiplies.** Four queries is four `query_points` calls, each with two prefetch arms.
They run concurrently (§5), so wall-clock cost is roughly one search, but the *server* does 4× the
work. That matters when you have concurrent users, not when you are testing alone.

**Drift.** A model asked for variations will occasionally produce one that is about something else.
This is why the original query is always kept and always first, and why fusion uses rank rather than
score: a bad variation contributes its own ranks, and any passage only it liked ends up low in the
fused list.

### Failure Modes

**Variations are near-identical.** The model is paraphrasing rather than re-terminologising. The
system prompt targets this specifically ("vary the terminology, not just the word order"). If it
persists, the expansion model is too small — this is one of the few places where `gpt-oss-20b` is
genuinely worse than the 120b.

**Retrieval gets worse with expansion on.** Real, and measurable in Phase 7. It happens on queries
that were already precise — a query containing an exact defined term or a section number does not need
rephrasing, and variations only add noise. A router that skips expansion for such queries is Phase 5's
job (`router_node.py`), which is why that node exists.

**`ValidationError` from `generate_json`.** The model returned malformed JSON. Caught as a
`RAGException` and degraded to the original query. Phase 6 owns making JSON mode reliable; this layer
just refuses to break because of it.

---

## 4. File 2 — `src/retrieval/hyde.py`

### The Problem

Multi-query still compares a *question* to an *answer*. Those are different kinds of text, and
embedding models trained on symmetric similarity are not great at bridging them. "What is the notice
period for termination?" and "Either party may terminate this Agreement upon ninety (90) days prior
written notice" have almost no lexical overlap and different syntactic shapes — one is interrogative,
one is declarative.

### Design Decision

**HyDE: Hypothetical Document Embeddings.** Ask the LLM to *write the clause it expects to exist*,
then search with the embedding of that fabricated clause. You are now comparing an answer to an
answer, which is the symmetric comparison the embedding model is actually good at.

The counter-intuitive part, and the thing to be able to say in an interview: **the hypothetical
document does not need to be factually correct.** It is thrown away after embedding. Its job is to
land in the right *neighbourhood* of the vector space by using the right register, structure, and
terminology. A fabricated clause that says "ninety (90) days" when the real contract says thirty is
still an excellent search probe, because the vector is dominated by "termination", "written notice",
"either party" — the shape of the thing — not by the number.

**Off by default.** It costs a second LLM call in the critical path, and on a corpus this
domain-specific its benefit is smaller than on general text: multi-query already recovers most of the
register gap, because "termination for convenience" *is* the document's language. Phase 7 measures
whether it earns its latency here.

```python
from config.settings import settings
from src.core.exceptions import RAGException
from src.core.interfaces import BaseLLMProvider
from src.core.logging import logger
from src.core.telemetry import telemetry

_SYSTEM_PROMPT = """You draft contract language.

Given a question about a commercial agreement, write the clause that would answer
it, as it would appear in the contract itself. Use formal legal register, defined
terms in Title Case, and standard drafting conventions.

Write 2-4 sentences. No preamble, no explanation, no markdown — output only the
clause text. Invented specifics (dates, amounts, notice periods) are acceptable:
this text is used to locate real clauses, never shown to anyone."""


class HyDEGenerator:
    """Question → hypothetical clause, for use as a search probe.

    The generated text is never returned to the user and never enters a prompt.
    It exists only to be embedded, which is why factual accuracy is irrelevant and
    register accuracy is everything.
    """

    def __init__(
        self,
        llm: BaseLLMProvider | None = None,
        model: str | None = None,
        max_tokens: int = 200,
    ) -> None:
        self.llm = llm
        self.model = model or settings.EXPANSION_MODEL
        self.max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return settings.ENABLE_HYDE and self.llm is not None

    @telemetry.measure_async("retrieval.hyde")
    async def generate(self, query: str) -> str | None:
        """Return a hypothetical clause, or None if unavailable or failed.

        Returns None rather than raising or falling back to the query text.
        Returning the query would be worse than returning nothing: the caller
        would add a duplicate probe and dilute the fusion weights, believing it had
        added a new one.
        """
        if not self.available:
            return None

        try:
            clause = await self.llm.generate(  # type: ignore[union-attr]
                prompt=f"Question: {query}\n\nWrite the clause that answers it.",
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=self.max_tokens,
                model=self.model,
            )
        except RAGException as exc:
            logger.warning("HyDE generation failed", extra={"error": type(exc).__name__})
            return None

        clause = clause.strip()
        if len(clause) < 40:
            # A one-line answer is not a clause. It is usually the model refusing,
            # apologising, or echoing the question — all useless as probes, and all
            # actively harmful, because they embed near the question rather than
            # near the answer.
            logger.warning("HyDE output too short to be a useful probe")
            return None

        # NEVER log the generated clause. Roadmap §7.5 forbids logging document
        # bodies, and a plausible-looking fabricated contract clause in your logs
        # is a genuinely dangerous artefact — it is indistinguishable from a real
        # extract to anyone reading the log later.
        logger.info("HyDE probe generated", extra={"chars": len(clause)})
        return clause
```

### The Theory: why a fabrication is a good probe

Picture the embedding space as having two loose clusters for any given topic: questions about
termination, and termination clauses. They are near each other — the model knows they are related —
but not on top of each other. A question vector sits in the question cluster, and its nearest
neighbours include *other questions* and only then the clauses.

HyDE moves the probe from the question cluster into the clause cluster before searching. The
neighbours it finds there are clauses, which is what you wanted retrieved.

Two limits, both real:

**It fails when the model has no idea what the clause looks like.** For a niche or novel provision the
fabrication is generic boilerplate, and generic boilerplate is near *everything* in the clause
cluster — the probe stops discriminating. HyDE helps most where the model has strong priors, which for
standard commercial terms it certainly does.

**It can confidently probe the wrong region.** Ask about "termination" in an employment agreement and
get a fabricated *commercial* termination clause; the probe now sits among purchase agreements. This
is why HyDE is an *additional* probe in the fusion set, never a replacement for the original query.
Adding a probe can only cost you if you let it vote alone.

### Failure Modes

**HyDE makes retrieval worse.** Most likely on short, keyword-like queries ("Section 7.02", "Delaware
governing law"), where the original query was already an excellent lexical probe and the fabrication
adds an unfocused dense one. Route around it; do not tune it.

**The hypothetical clause leaks into an answer.** It must not. `generate()` returns it to the
retriever, which embeds it and drops it. If it ever reaches a prompt, you have built a system that
answers from fabricated contract text — the single worst failure mode this project could have.

**Latency doubles.** Expansion and HyDE are two independent calls and §11 issues them concurrently
with `asyncio.gather`, so the cost is one call's latency, not two. If you see additive latency, you
awaited them in sequence.

---

## 5. File 3 — `src/retrieval/hybrid.py`

### The Problem

Given N queries, produce N ranked candidate lists. Each query needs a dense vector and a sparse
vector, then a filtered `hybrid_search`. Done naively — a `for` loop with `await` inside — four
queries at 80ms each take 320ms for work that could take 90ms.

There is also a contract to enforce that nothing else will enforce: **every search must be filtered to
`chunk_level == "child"`.** Phase 2 put parents in the same collection. Forget the filter and roughly a
fifth of your results are 1,600-token parents competing with their own children, which looks like
duplicated results and wastes the candidate budget.

### Design Decision

**One class that owns the child filter.** Making it a default rather than a caller's responsibility is
the whole point — a policy that every call site must remember is a policy that will be forgotten in
Phase 5, Phase 12, and Phase 13 independently.

**`asyncio.gather` over the queries.** This is where Phase 1's insistence on an async vector store
finally pays off. Four queries become one round trip's worth of wall-clock time.

**`return_exceptions=True`, and count the failures.** If one of four searches fails, returning three
lists is a better outcome than failing the request — but silently returning three while reporting
success is exactly the "filter whose failure mode is invisible" pattern that Phase 2's second lesson
warns about. So partial failure is allowed, logged at WARNING, and reflected in the returned metadata.
If *every* search fails, that is not degradation and it raises.

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings
from src.core.exceptions import RetrievalError
from src.core.interfaces import BaseEmbeddingProvider, BaseVectorStore
from src.core.logging import logger
from src.core.models import ChunkLevel, ScoredChunk
from src.core.telemetry import telemetry


@dataclass
class SearchOutcome:
    """N result lists plus what went wrong getting them.

    The failure count is carried rather than logged-and-forgotten so the pipeline
    can report degraded retrieval in `RetrievalResult`. Phase 7 needs to know that
    a poor answer came from two arms instead of four.
    """

    results: list[list[ScoredChunk]] = field(default_factory=list)
    failed_queries: int = 0

    @property
    def total_candidates(self) -> int:
        return sum(len(r) for r in self.results)


class HybridRetriever:
    """Embeds queries and searches the child chunks, concurrently.

    Owns the `chunk_level == CHILD` filter so that no caller can forget it.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        store: BaseVectorStore,
        top_k: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k or settings.RETRIEVAL_TOP_K

    @telemetry.span("retrieval.hybrid_search", warn_over_ms=1000)
    async def search_many(
        self,
        queries: Sequence[str],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchOutcome:
        """Search with every query concurrently.

        Raises:
            RetrievalError: every query failed, or no queries were given.
        """
        if not queries:
            raise RetrievalError("search_many called with no queries")

        limit = top_k or self.top_k
        query_filter = self._child_filter(filters)

        tasks = [self._search_one(q, query_filter, limit) for q in queries]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        outcome = SearchOutcome()
        for query, item in zip(queries, raw, strict=True):
            if isinstance(item, BaseException):
                outcome.failed_queries += 1
                logger.warning(
                    "One retrieval arm failed",
                    extra={"error": type(item).__name__, "query_chars": len(query)},
                )
                continue
            outcome.results.append(item)

        if not outcome.results:
            raise RetrievalError(
                "Every retrieval arm failed",
                details={"queries": len(queries)},
                retryable=True,
            )

        logger.info(
            "Hybrid search complete",
            extra={
                "queries": len(queries),
                "failed": outcome.failed_queries,
                "candidates": outcome.total_candidates,
            },
        )
        return outcome

    async def search_vector(
        self,
        dense_query: list[float],
        sparse_query: dict[str, list],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Search with pre-computed vectors. Used for the HyDE probe, whose text
        must be embedded as a document rather than as a query."""
        return await self.store.hybrid_search(
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=top_k or self.top_k,
            filters=self._child_filter(filters),
        )

    async def embed_probe(
        self, text: str, as_document: bool = False
    ) -> tuple[list[float], dict[str, list]]:
        """Produce the (dense, sparse) pair for one probe.

        `as_document=True` embeds with the passage-side transformation instead of
        the query-side one. That is correct for a HyDE clause: it is pretending to
        BE a document, so it must be embedded the way documents were embedded, or
        it lands in the wrong region for an asymmetric model like bge.
        """
        if as_document:
            dense = (await self.embedder.embed_dense([text]))[0]
            # A HyDE clause is pretending to be a document on BOTH sides, so its
            # sparse vector uses the document-side BM25 weighting too.
            sparse = (await self.embedder.embed_sparse([text]))[0]
        else:
            dense = await self.embedder.embed_query(text)
            # `embed_sparse_query`, NOT `embed_sparse([text])[0]`. The document-side
            # BM25 representation applies term-frequency saturation and length
            # normalisation, neither of which means anything for a query. Using it
            # here returns results — just mis-weighted ones — so the bug is
            # invisible. This was wrong in the first draft of this file.
            sparse = await self.embedder.embed_sparse_query(text)

        return dense, sparse

    async def _search_one(
        self, query: str, query_filter: dict[str, Any], limit: int
    ) -> list[ScoredChunk]:
        dense, sparse = await self.embed_probe(query)
        return await self.store.hybrid_search(
            dense_query=dense, sparse_query=sparse, limit=limit, filters=query_filter
        )

    @staticmethod
    def _child_filter(filters: dict[str, Any] | None) -> dict[str, Any]:
        """Force `chunk_level == child` onto every search.

        Phase 2 stores parents in the same collection, separated only by this
        field. Without the filter, parents compete with their own children for
        candidate slots and results look duplicated.

        A caller-supplied `chunk_level` is deliberately overridden rather than
        respected. Phase 16 will need to search summary nodes and will call the
        store directly for that; letting an arbitrary caller widen this filter
        would make the duplication bug reachable again.
        """
        merged = dict(filters or {})
        merged["chunk_level"] = ChunkLevel.CHILD.value
        return merged
```

### The Theory: what `gather` does and does not buy

`asyncio.gather` starts every coroutine and waits for all of them. For four searches that each spend
75ms waiting on Qdrant and 5ms in Python, total wall-clock is roughly 80ms rather than 320ms, because
the waiting overlaps.

Be precise about the mechanism, because it is the concept the whole async design rests on. Nothing is
running in parallel. There is one thread. When `_search_one` hits `await` on the network call, it
yields control, and the event loop starts the next one. The four requests are in flight
simultaneously; the four *Python functions* take turns on a single core. That is why this works
beautifully for I/O and does nothing for the reranker in §8 — a cross-encoder never yields, it
computes.

Two consequences worth internalising:

**The embedding calls inside `gather` are not free.** `embed_probe` is CPU work (local ONNX), wrapped
in `to_thread` by Phase 3. Four concurrent `to_thread` calls do overlap, because ONNX releases the
GIL, but they contend for cores with everything else. At four probes it is irrelevant; if you ever
expand to twenty, embed them in one batched call instead.

**`return_exceptions=True` changes the type of what you get back.** Without it, one failure cancels
the gather and propagates. With it, you get exception *objects* mixed into your results list, and
forgetting the `isinstance` check means an exception object flows downstream as if it were a result
list. The `isinstance(item, BaseException)` line is not defensive padding; it is required by the
choice on the line above it.

### Failure Modes

**Results contain 1,600-token chunks.** The child filter did not apply. Either something called
`store.hybrid_search` directly, or the corpus was ingested before parent-child chunking existed, in
which case every chunk is `standalone` and this filter matches nothing at all — an empty result set,
not a wrong one.

**Every query returns identical results.** Your expansion produced near-identical variations, or the
embedder is returning the same vector for different text (a sign `embed_query` is being fed an empty
string after normalisation).

**`RetrievalError: Every retrieval arm failed`, retryable.** Qdrant is down or the collection name is
wrong. Note `retryable=True` — Phase 5's graph and Phase 8's error handler both read that flag rather
than maintaining their own list of transient failures.

---

## 6. File 4 — `src/retrieval/fusion.py`

### The Problem

Four queries returned four ranked lists of twenty. You need one ranked list of twenty. The same chunk
appears in three of the four lists, at ranks 2, 5, and 11.

Phase 3 already does RRF — but *inside* a single query, fusing that query's dense and sparse arms
server-side. Qdrant cannot fuse across four separate `query_points` calls, because it has no idea they
were related. This fusion is a different fusion, at a different level, and it has to happen here.

### Design Decision

**Client-side RRF, keyed by `chunk_id`.** The same algorithm and the same reasoning as Phase 3's
server-side fusion, applied one level up. It is the right choice here for an additional reason: the
scores coming back from four `query_points` calls are RRF scores computed against *different* query
vectors, and they are not comparable to each other in any meaningful way. Ranks are.

**Deduplicate by `chunk_id`, and keep every rank.** A chunk found by three probes should score higher
than one found by a single probe — that is the whole point of expanding the query. So each occurrence
contributes `1/(k + rank)` and they sum.

**`method` becomes `HYBRID`, and the surviving `ScoredChunk` keeps its best-ranked incarnation.** The
`Chunk` inside is identical across occurrences (same ID, same payload), so which one you keep does not
matter for content — but keeping the best-ranked one means the `score` field remains interpretable as
"how the best probe scored it".

```python
from collections.abc import Sequence

from config.settings import settings
from src.core.logging import logger
from src.core.models import RetrievalMethod, ScoredChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one, using ranks rather than scores.

        RRF(chunk) = sum over lists of 1 / (k + rank_in_that_list)

    Scores from different query vectors are not comparable — one probe's 0.031 and
    another's 0.028 say nothing about each other, because they were computed
    against different queries. Ranks are ordinal and therefore comparable, which is
    what makes this work without any normalisation or tuning.

    Args:
        result_lists: One ranked list per query probe, best first.
        k: The RRF constant. Larger flattens the contribution of top ranks.
        top_k: Truncate the fused list. None returns everything.

    Returns:
        A new list of `ScoredChunk`, re-ranked from 0, with `score` replaced by the
        fused RRF value. The originals are not mutated.
    """
    constant = k if k is not None else settings.RRF_K

    fused_scores: dict[str, float] = {}
    best_seen: dict[str, ScoredChunk] = {}
    best_rank: dict[str, int] = {}

    for results in result_lists:
        for rank, scored in enumerate(results):
            chunk_id = scored.chunk.chunk_id

            # Use the enumerated position, not `scored.rank`. They are usually the
            # same, but a caller that sliced a list would leave stale ranks behind,
            # and RRF is meaningless if the ranks are not this list's ranks.
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (constant + rank + 1)

            if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                best_rank[chunk_id] = rank
                best_seen[chunk_id] = scored

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    fused: list[ScoredChunk] = []
    for new_rank, (chunk_id, score) in enumerate(ordered):
        original = best_seen[chunk_id]
        fused.append(
            ScoredChunk(
                chunk=original.chunk,
                score=score,
                rank=new_rank,
                method=RetrievalMethod.HYBRID,
                # Deliberately NOT carried over. Nothing has been reranked yet, and
                # a stale rerank_score would make `effective_score` sort by a
                # previous query's cross-encoder output.
                rerank_score=None,
            )
        )

    unique = len(fused_scores)
    total = sum(len(r) for r in result_lists)
    logger.info(
        "Fused retrieval results",
        extra={
            "lists": len(result_lists),
            "total_hits": total,
            "unique_chunks": unique,
            "returned": len(fused),
            "overlap_ratio": round(1 - unique / total, 3) if total else 0.0,
        },
    )
    return fused
```

### The Theory: reading the overlap ratio

That last logged field is the most useful diagnostic in the phase, and it costs one division.

`overlap_ratio = 1 - unique/total`. Four probes returning twenty each is 80 hits; if 60 chunks are
unique, overlap is 0.25.

**Near 0.0** means the probes found completely different things. Either your expansion produced
variations about different topics — check them — or the corpus has so many marginally-relevant
passages that any probe finds twenty. On a 39-million-point index the second is common for vague
queries, and it is a signal that the answer is probably not in the top 5 no matter what you do.

**Near 0.75** (with four probes) means every probe found the same twenty chunks. Expansion bought you
nothing but latency. That is not a bug — it happens on precise queries — but if it happens on *every*
query, turn multi-query off and save the LLM call. This is exactly the measurement Phase 7 automates
and Phase 5's router acts on.

**Around 0.2–0.4** is the healthy range: the probes agree on a core and each contributes something,
which is the condition under which RRF's agreement-reward actually does work.

### Failure Modes

**The fused order looks worse than a single query's order.** Usually one bad variation. RRF is
robust to a bad probe but not immune — a probe whose top hits are all irrelevant contributes ranks 0
and 1 to irrelevant chunks. The reranker in §8 is the backstop for this, which is another reason it is
not optional.

**A chunk appears twice in the output.** Impossible via this function — `fused_scores` is keyed by
`chunk_id`. If you see it, two *different* chunk IDs hold the same text, which means the same document
was ingested under two paths (`doc_id` derives from the path). Phase 3's delete-before-upsert cannot
help across two different `doc_id`s.

**Every fused score is tiny and nearly equal.** Normal. With `k=60`, scores live in the 0.016–0.065
range and differences are small by construction. Do not show these numbers to users as confidence.

---

## 7. File 5 — `src/retrieval/rerankers/base.py`

### The Problem

Two rerankers with nothing in common mechanically — one runs a transformer, one does greedy
selection over cosine distances — still share three obligations: the returned list must be re-ranked
from 0, `rerank_score` must be populated (Phase 1's contract, and Phase 4's checkpoint asserts it),
and `top_k` must be respected without exceeding what was supplied.

There is also a truncation problem specific to cross-encoders that belongs here because both
implementations benefit from stating it once.

### Design Decision

**A small base that owns the output invariants.** The interesting logic lives in the subclasses; the
bookkeeping that Phase 5 and Phase 6 depend on lives here, once.

```python
from collections.abc import Sequence

from config.settings import settings
from src.core.interfaces import BaseReranker
from src.core.logging import logger
from src.core.models import ScoredChunk
from src.core.utils import count_tokens, truncate_to_tokens

#: Cross-encoders based on MiniLM take 512 tokens for the query and the passage
#: TOGETHER. Reserve room for the query and the separator tokens, and truncate the
#: passage to what is left. Phase 2's children are ~400 tokens, so this normally
#: does nothing — which is precisely why it must be measured rather than assumed.
MAX_PAIR_TOKENS = 512
QUERY_TOKEN_BUDGET = 96


class RerankerBase(BaseReranker):
    """Output invariants shared by every reranker.

    Subclasses implement `_score`. This class guarantees that what comes out is
    sorted, re-ranked from 0, truncated to `top_k`, and carries `rerank_score`.
    """

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.RERANK_TOP_K

    def _prepare_passage(self, text: str) -> str:
        """Fit one passage into the pair budget.

        Truncation is silent in the model but not here. A passage cut in half is
        scored on half its content, and if that happens routinely your reranking is
        being decided by whatever is in the first paragraph.
        """
        budget = MAX_PAIR_TOKENS - QUERY_TOKEN_BUDGET
        if count_tokens(text) <= budget:
            return text
        logger.warning("Truncating passage for reranking", extra={"budget": budget})
        return truncate_to_tokens(text, budget)

    def _finalise(
        self, candidates: Sequence[ScoredChunk], scores: Sequence[float], top_k: int
    ) -> list[ScoredChunk]:
        """Attach scores, sort, truncate, re-rank.

        Returns new `ScoredChunk` objects rather than mutating the inputs. The
        caller may still hold the pre-rerank list — Phase 7 compares the two orders
        to prove the reranker earns its latency — and mutating in place would
        destroy that comparison.
        """
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker produced {len(scores)} scores for {len(candidates)} candidates"
            )

        rescored = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=candidate.score,          # the retrieval score, preserved
                rank=candidate.rank,            # overwritten below; kept for clarity
                method=candidate.method,
                rerank_score=float(score),
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]

        # `effective_score` returns rerank_score when set, so this sorts by the new
        # scores while leaving the originals inspectable.
        rescored.sort(key=lambda item: item.effective_score, reverse=True)

        selected = rescored[: min(top_k, len(rescored))]
        return [
            ScoredChunk(
                chunk=item.chunk,
                score=item.score,
                rank=new_rank,
                method=item.method,
                rerank_score=item.rerank_score,
            )
            for new_rank, item in enumerate(selected)
        ]
```

### Failure Modes

**`ValueError: reranker produced N scores for M candidates`.** The same alignment failure class as
Phase 3's embedding check, in a new place. Almost always a generator consumed twice.

**Truncation warnings on every candidate.** You are reranking parents instead of children — the
pipeline order in §11 is wrong.

**`rerank_score` is set but ordering did not change.** Check that you sorted by `effective_score` and
not `score`. This is an easy and completely silent mistake: everything looks reranked, and nothing is.

---

## 8. File 6 — `src/retrieval/rerankers/cross_encoder.py`

### The Problem

Dense retrieval computed your query vector and your document vectors **independently, at different
times**. The document was embedded during ingestion, months before the query existed. The comparison
is one cosine similarity between two vectors that never met.

That independence is exactly what makes search possible: the 39 million document vectors are computed
once and indexed, and a query touches only a few thousand of them via HNSW. It is also a hard ceiling
on accuracy. A single vector must summarise a whole passage for *every possible query*, so it
necessarily encodes topic rather than answerhood. "This passage is about termination" and "this
passage answers what the notice period is" are different claims, and a bi-encoder can only make the
first.

### Design Decision

**A cross-encoder over the top ~20 candidates.** A cross-encoder concatenates query and passage into
one sequence and runs a transformer over both together, so every query token can attend to every
passage token. It directly predicts relevance rather than approximating it with a distance. On
standard benchmarks this is worth 10–20 points of NDCG over the bi-encoder ranking — it is not a
marginal refinement.

**Why it cannot replace retrieval, stated concretely:** it needs one forward pass *per candidate*, and
the passages cannot be precomputed because the query is half of every input. Twenty candidates is
~20ms on CPU. Thirty-nine million candidates is about nine months. The retrieve-then-rerank cascade is
not a compromise; it is the only structure in which a cross-encoder is usable at all.

**FastEmbed's `TextCrossEncoder`, not `sentence-transformers`.** Same MiniLM weights, ONNX runtime, no
PyTorch. FastEmbed is already a dependency for embeddings.

**Failures degrade to the retrieval order.** If the reranker breaks, the top 5 by RRF is a perfectly
serviceable answer set. Turning a refinement failure into a request failure is the wrong trade — the
same reasoning as query expansion in §3.

```python
import asyncio
from collections.abc import Sequence
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from config.settings import settings
from src.core.logging import logger
from src.core.models import ScoredChunk
from src.core.telemetry import telemetry

from .base import RerankerBase


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str) -> TextCrossEncoder:
    """Process-wide cross-encoder singleton.

    Same pattern as Phase 3's embedding models and Phase 1's tokenizer: lazy so
    importing this module is free, cached so the ~90 MB ONNX session loads once.
    """
    logger.info("Loading cross-encoder", extra={"model": model_name})
    return TextCrossEncoder(model_name=model_name)


class CrossEncoderReranker(RerankerBase):
    """Re-scores candidates by reading query and passage together.

    Roughly 1ms per candidate on CPU. At 20 candidates that is a latency cost you
    can afford once per request and could never afford once per document.
    """

    def __init__(self, model_name: str | None = None, top_k: int | None = None) -> None:
        super().__init__(top_k=top_k)
        self.model_name = model_name or settings.RERANK_MODEL_NAME

    def warmup(self) -> None:
        """Load the model and run one pair. Call at startup, not on first query."""
        get_cross_encoder(self.model_name).rerank("warmup", ["warmup passage"])

    @telemetry.span("retrieval.rerank", warn_over_ms=400)
    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        """Re-score and truncate.

        Async with the offload inside, per Phase 1's interface note: the reranker
        sits between two async stages and the caller should not have to remember a
        `to_thread` wrapper.

        Never raises. On failure it returns the retrieval order truncated to
        `top_k`, with `rerank_score` left as None — so `effective_score` falls back
        to the retrieval score and downstream code cannot tell the difference in
        shape, only in quality.
        """
        if not candidates:
            return []
        if not settings.ENABLE_RERANKING:
            return self._passthrough(candidates, top_k)

        passages = [self._prepare_passage(c.chunk.text) for c in candidates]

        try:
            scores = await asyncio.to_thread(self._score, query, passages)
        except Exception as exc:
            logger.warning(
                "Reranking failed; falling back to retrieval order",
                extra={"error": type(exc).__name__, "candidates": len(candidates)},
            )
            return self._passthrough(candidates, top_k)

        result = self._finalise(candidates, scores, top_k)

        # The one diagnostic worth logging here: did reranking actually change the
        # order? If the top result was already first, the reranker cost you 20ms
        # and bought nothing on this query. Over many queries this ratio tells you
        # whether it is earning its place.
        moved = result[0].chunk.chunk_id != candidates[0].chunk.chunk_id
        logger.info(
            "Reranked",
            extra={
                "candidates": len(candidates),
                "returned": len(result),
                "top_changed": moved,
                "top_score": round(result[0].rerank_score or 0.0, 4),
            },
        )
        return result

    def _score(self, query: str, passages: list[str]) -> list[float]:
        """Blocking scoring. `rerank` returns a generator — materialise it."""
        encoder = get_cross_encoder(self.model_name)
        return [float(s) for s in encoder.rerank(query, passages)]

    @staticmethod
    def _passthrough(candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Retrieval order, re-ranked from 0, with no rerank_score.

        Leaving `rerank_score` as None is deliberate and load-bearing: Phase 7
        must be able to distinguish "reranked and this was the order" from "not
        reranked". Writing the retrieval score into `rerank_score` would make a
        skipped rerank indistinguishable from a completed one.
        """
        return [
            ScoredChunk(
                chunk=item.chunk, score=item.score, rank=rank, method=item.method
            )
            for rank, item in enumerate(candidates[:top_k])
        ]
```

### The Theory: bi-encoder versus cross-encoder, precisely

```text
BI-ENCODER  (retrieval — Phase 3)
   query ──► [encoder] ──► vector_q  ─┐
                                      ├── cosine ──► score
   passage ─► [encoder] ──► vector_p  ─┘
                    ▲
        precomputed at ingestion time

   Cost per query: 1 encode + an approximate-nearest-neighbour lookup.
   Passages never see the query.


CROSS-ENCODER  (reranking — this file)
   [query [SEP] passage] ──► [transformer] ──► relevance score
                                   ▲
                       every query token attends to
                       every passage token

   Cost per query: one forward pass PER CANDIDATE. Nothing is precomputable.
```

The architectural difference is where the interaction happens. A bi-encoder compresses each side to a
vector *before* comparison, so all interaction is mediated by one dot product. A cross-encoder lets
the two texts interact through every attention layer, so it can represent things a dot product cannot
— for instance that the passage contains "ninety (90) days" and the query asked "how many days",
which is a *correspondence* between specific spans rather than a topical overlap.

**The scores are raw logits, not probabilities, and this trips people up.** FastEmbed's MiniLM
cross-encoder returns values like `-11.48` and `5.47`. They are unbounded, frequently negative, and
comparable *only within one query's candidate set*. Never show them to a user as a confidence, never
threshold them with a constant lifted from a blog post, and never compare them across queries. If you
need something bounded, `1/(1+e^-x)` gives a monotone [0,1] mapping — but it is calibrated to nothing,
so it is prettier without being more meaningful.

This is exactly why Phase 1 gave `ScoredChunk` a separate `rerank_score` field instead of overwriting
`score`. The two numbers are in different units, from different models, on different scales. Keeping
both is what makes `top_changed` above measurable, and it is what Phase 7 uses to prove the reranker
is worth its 20ms.

### Failure Modes

**`Model not supported: cross-encoder/ms-marco-MiniLM-L-6-v2`.** The Phase 1 default. Change it to
`Xenova/ms-marco-MiniLM-L-6-v2` (§2).

**Reranking takes 400ms and trips the telemetry warning.** Either `RETRIEVAL_TOP_K` is far above 20,
or you are on `BAAI/bge-reranker-base` (1 GB, much slower). Latency is linear in candidates: measure
at your actual `top_k`.

**`top_changed` is false on almost every query.** Retrieval was already good — genuinely possible on
an easy corpus — or the candidate set is too small for reranking to have anything to do. Reranking 5
candidates to pick 5 is a no-op by construction. Keep `RETRIEVAL_TOP_K` at least 3–4× `RERANK_TOP_K`.

**Scores are all negative.** Normal. They are logits. Ordering is what matters.

---

## 9. File 7 — `src/retrieval/rerankers/mmr.py`

### The Problem

The top 5 by relevance can be five near-duplicates. On this corpus that is not hypothetical: parent
chunks overlap, boilerplate recurs across thousands of contracts, and a query about indemnification
can easily return five paraphrases of the same standard clause from five different agreements.

Five slots spent on one idea means the *second* idea the question needed never reaches the LLM. And
generation cannot recover from missing context; it can only apologise or invent.

### Design Decision

**Maximal Marginal Relevance.** Select results one at a time, and at each step pick the candidate
that maximises

\[ \lambda \cdot \text{relevance}(c, q) - (1 - \lambda) \cdot \max_{s \in \text{selected}} \text{similarity}(c, s) \]

A candidate is chosen for being relevant *and* unlike what you have already chosen. `λ = 1.0` is pure
relevance (identical to no MMR); `λ = 0.0` picks the most mutually different set regardless of
relevance. `0.7` leans toward relevance, which is the right lean for a legal-answering system where a
diverse set of wrong clauses is worse than a redundant set of right ones.

**Implemented as a `BaseReranker`, which is the interesting part.** It shares an interface with the
cross-encoder while doing something completely unrelated internally — one runs a transformer, the
other does greedy selection over cosine distances. Two implementations this different satisfying one
interface is the strongest evidence available that the abstraction from Phase 1 is real.

**It re-embeds the candidate texts, and that is a genuine cost I am choosing to pay.** MMR needs
vectors to compute candidate-to-candidate similarity, and `hybrid_search` returns payloads without
vectors. Re-embedding 20 passages costs ~30ms locally. The alternatives were worse: requesting
`with_vectors=True` moves 20 × 384 floats over the network on every query and adds a Qdrant-specific
concept to a reranker, and threading vectors through `RetrievalResult` would change a Phase 1 model
that four later phases already reference. Paying 30ms of local CPU to keep the frozen interface intact
is the right trade, and it is the kind of trade worth being explicit about rather than hiding.

**Off by default.** Diversity helps some query shapes and hurts others. "List every termination
provision" wants diversity; "what is the notice period in section 7.02" wants the single best hit and
nothing else. Phase 12's map-reduce is the real answer for the first shape.

```python
import asyncio
import math
from collections.abc import Sequence

from config.settings import settings
from src.core.interfaces import BaseEmbeddingProvider
from src.core.logging import logger
from src.core.models import ScoredChunk
from src.core.telemetry import telemetry

from .base import RerankerBase


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, without numpy.

    Twenty candidates means at most 190 pairs of 384 floats — a few hundred
    microseconds in pure Python. Not worth a numpy dependency in this module, and
    the explicit arithmetic makes the formula readable.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MMRReranker(RerankerBase):
    """Selects a relevant AND internally diverse subset.

    Deliberately shares `BaseReranker` with the cross-encoder despite having
    nothing mechanically in common with it. Both answer the same question — "which
    of these candidates should the LLM see?" — by different means, which is what an
    interface is for.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        lambda_param: float | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__(top_k=top_k)
        self.embedder = embedder
        self.lambda_param = (
            lambda_param if lambda_param is not None else settings.MMR_LAMBDA
        )

    @telemetry.span("retrieval.mmr", warn_over_ms=300)
    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        """Greedy MMR selection.

        Never raises: on failure it returns the input order truncated, like the
        cross-encoder. Diversity is a refinement, not a requirement.
        """
        if not candidates:
            return []
        if not settings.ENABLE_MMR or len(candidates) <= top_k:
            # With fewer candidates than slots there is nothing to choose between,
            # and MMR would only reorder what is all going to the LLM anyway.
            return candidates[:top_k]

        try:
            query_vector, candidate_vectors = await self._embed(query, candidates)
        except Exception as exc:
            logger.warning("MMR embedding failed", extra={"error": type(exc).__name__})
            return candidates[:top_k]

        selected = self._select(query_vector, candidate_vectors, top_k)

        chosen = [candidates[i] for i in selected]
        scores = [
            cosine_similarity(query_vector, candidate_vectors[i]) for i in selected
        ]

        logger.info(
            "MMR selection",
            extra={
                "candidates": len(candidates),
                "selected": len(chosen),
                "lambda": self.lambda_param,
                # How far down the relevance list MMR reached. A high number means
                # the top of the list was redundant and diversity did real work.
                "deepest_index": max(selected) if selected else 0,
            },
        )
        return self._finalise(chosen, scores, top_k)

    async def _embed(
        self, query: str, candidates: Sequence[ScoredChunk]
    ) -> tuple[list[float], list[list[float]]]:
        """Embed the query and every candidate passage.

        The candidates were already embedded during ingestion; we are paying again
        because the store returns payloads, not vectors. ~30ms for 20 passages.
        """
        query_task = self.embedder.embed_query(query)
        docs_task = self.embedder.embed_dense([c.chunk.text for c in candidates])
        query_vector, candidate_vectors = await asyncio.gather(query_task, docs_task)
        return query_vector, candidate_vectors

    def _select(
        self,
        query_vector: Sequence[float],
        candidate_vectors: Sequence[Sequence[float]],
        top_k: int,
    ) -> list[int]:
        """Greedy MMR. Returns candidate indices in selection order."""
        relevance = [cosine_similarity(query_vector, v) for v in candidate_vectors]

        selected: list[int] = [max(range(len(relevance)), key=relevance.__getitem__)]
        remaining = set(range(len(candidate_vectors))) - set(selected)

        while len(selected) < min(top_k, len(candidate_vectors)) and remaining:
            best_index, best_value = -1, -math.inf

            for index in remaining:
                redundancy = max(
                    cosine_similarity(candidate_vectors[index], candidate_vectors[chosen])
                    for chosen in selected
                )
                value = (
                    self.lambda_param * relevance[index]
                    - (1.0 - self.lambda_param) * redundancy
                )
                if value > best_value:
                    best_index, best_value = index, value

            selected.append(best_index)
            remaining.discard(best_index)

        return selected
```

### The Theory: greedy is not optimal, and that is fine

MMR is greedy: it commits to the best first pick and never reconsiders. The globally optimal
"relevant and diverse set of 5" would require evaluating every 5-subset of 20 — 15,504 combinations,
each needing 10 pairwise similarities. Tractable at these numbers, intractable the moment `top_k`
grows, and the greedy solution is close enough in practice that nobody uses the exact one.

What the greedy structure does guarantee is a useful property: **the first selection is always the
most relevant candidate.** Whatever λ you choose, you never lose the best hit to diversity. Only
positions 2 through k are traded, which is the right place to trade them — the LLM will read the first
passage most attentively regardless.

The λ intuition, concretely:

| λ | Behaviour | When |
| :-- | :--- | :--- |
| 1.0 | Pure relevance; identical to no MMR | Precise, single-fact queries |
| 0.7 | Mild de-duplication | Default. Drops obvious near-copies, keeps ordering |
| 0.5 | Genuinely diverse | "What are all the parties' obligations?" |
| 0.0 | Maximally different, relevance ignored | Exploration only; will return junk |

### Failure Modes

**MMR reorders nothing.** With `len(candidates) <= top_k` it short-circuits by design. Raise
`RETRIEVAL_TOP_K`.

**Results are diverse but wrong.** λ too low. At 0.3 the algorithm will happily pick an irrelevant
passage because it differs from everything selected.

**MMR and the cross-encoder disagree, and MMR wins.** If you run both, MMR runs second and its
selection is final — it operates on the cross-encoder's output. That ordering is deliberate (§11):
score first with the accurate model, then diversify within the good candidates. Reversed, you would
diversify the *retrieval* order and then rerank, and the cross-encoder would simply undo the
diversification.

---

## 10. File 8 — `src/retrieval/parents.py`

### The Problem

Everything up to here searched and refined **children** — ~400 tokens each, precise, and frequently
incomplete. A child chunk might read:

> "...shall not exceed the amount set forth in Section 4.02(b), except as provided in the following
> subsection."

Perfectly retrievable, and useless to answer from. The cap is in Section 4.02(b), the exception is in
the next subsection, and neither is in this chunk.

Phase 2 built the fix: each child knows its `parent_id`, and the parent is ~1,600 tokens of
surrounding context in the same collection. Something has to make the swap, and three things make it
non-trivial.

**Multiple children share a parent.** Five reranked children can easily map to two parents. Substitute
naively and you send the same 1,600-token parent three times — the LLM sees triplicated text and, in
my experience, treats repetition as emphasis.

**The context grows 4×.** Five children is ~2,000 tokens. Five parents is ~8,000. Deduplication helps,
but a budget still has to be enforced somewhere, or a wide retrieval silently overruns the model's
window and the last sources get dropped by the provider rather than by you.

**Citations change identity.** `Citation.chunk_id` points at what was cited. After substitution the
text in the prompt belongs to the *parent*, so the citation must name the parent — otherwise Phase 6's
citation validator compares an answer against text that was never in the prompt.

### Design Decision

**Fetch parents by ID, deduplicate, preserve child order, enforce a token budget.** Order comes from
the best-ranked child that maps to each parent, so the most relevant material stays first.

**A missing parent is not an error.** `fetch_by_ids` returns only what exists. A child whose parent was
deleted — mid-migration, or from a corpus ingested before parent-child chunking — falls back to the
child's own text. That is a correct degradation: slightly less context, still a right answer. Raising
would take down a request over a stale pointer.

**The budget truncates by dropping whole sources, never by cutting one in half.** A half-clause is
worse than an absent clause, because the LLM cannot tell it was cut and will answer from the fragment.
Dropping whole sources from the end is loud (it is logged and countable) and leaves every surviving
source complete.

```python
from collections.abc import Sequence

from config.settings import settings
from src.core.interfaces import BaseVectorStore
from src.core.logging import logger
from src.core.models import ChunkLevel, ScoredChunk
from src.core.telemetry import telemetry


class ParentSubstituter:
    """Replaces retrieved children with their larger parent chunks.

    Small-to-search, large-to-generate. This is the second half of Phase 2's
    parent-child design; without it, that phase's extra indexing work buys nothing.
    """

    def __init__(
        self, store: BaseVectorStore, max_context_tokens: int | None = None
    ) -> None:
        self.store = store
        self.max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS

    @telemetry.measure_async("retrieval.parent_substitution")
    async def substitute(self, scored_children: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return one `ScoredChunk` per distinct parent, in child-rank order.

        The returned chunks carry the PARENT's `chunk_id`, because that is the text
        the LLM will see and therefore the text a citation must point at. Scores and
        ranks come from the best child that mapped to each parent — the child is
        what was actually scored, and inventing a parent score would be fiction.
        """
        if not scored_children or not settings.ENABLE_PARENT_SUBSTITUTION:
            return list(scored_children)

        parent_ids = [
            c.chunk.parent_id
            for c in scored_children
            if c.chunk.parent_id and c.chunk.chunk_level == ChunkLevel.CHILD
        ]
        if not parent_ids:
            logger.debug("No parents to substitute — corpus may predate parent-child chunking")
            return list(scored_children)

        # Deduplicated, so a parent shared by three children is fetched once.
        parents = await self.store.fetch_by_ids(list(dict.fromkeys(parent_ids)))
        by_id = {parent.chunk_id: parent for parent in parents}

        missing = set(parent_ids) - set(by_id)
        if missing:
            logger.warning(
                "Some parent chunks are missing; falling back to child text",
                extra={"missing": len(missing)},
            )

        substituted: list[ScoredChunk] = []
        seen: set[str] = set()

        for child in scored_children:
            parent = by_id.get(child.chunk.parent_id or "")
            target = parent or child.chunk

            if target.chunk_id in seen:
                continue
            seen.add(target.chunk_id)

            substituted.append(
                ScoredChunk(
                    chunk=target,
                    score=child.score,
                    rank=len(substituted),
                    method=child.method,
                    rerank_score=child.rerank_score,
                )
            )

        capped = self._apply_budget(substituted)

        logger.info(
            "Parent substitution complete",
            extra={
                "children_in": len(scored_children),
                "sources_out": len(capped),
                "tokens": sum(c.chunk.token_count for c in capped),
            },
        )
        return capped

    def _apply_budget(self, sources: list[ScoredChunk]) -> list[ScoredChunk]:
        """Drop whole sources from the end until the context fits.

        Never truncates a source's text. A half-clause reads as a complete one to
        the LLM, which will then answer from a fragment whose qualifying proviso
        you removed. Dropping a source is visible in the count; cutting one is not.

        Uses each chunk's stored `token_count`, which Phase 1 documents as
        approximate. That is fine for a budget with headroom, and it is why the
        default (12,000) sits well below any model's actual window.
        """
        kept: list[ScoredChunk] = []
        total = 0

        for source in sources:
            tokens = source.chunk.token_count
            if kept and total + tokens > self.max_context_tokens:
                logger.warning(
                    "Context budget reached; dropping lower-ranked sources",
                    extra={
                        "kept": len(kept),
                        "dropped": len(sources) - len(kept),
                        "budget": self.max_context_tokens,
                    },
                )
                break
            kept.append(source)
            total += tokens

        # `if kept and ...` above guarantees at least one source survives even if a
        # single parent exceeds the whole budget. An empty context guarantees a
        # non-answer; one oversized source at least has a chance.
        return kept
```

### The store methods this needs

Add to `QdrantStore` (Phase 3, `src/vectorstores/qdrant_store.py`):

```python
    async def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Retrieve points by ID. Missing IDs are skipped, not reported.

        `with_vectors=False` — the caller wants text and metadata. Vectors would
        add 384 floats per point to the response for nothing.
        """
        if not ids:
            return []

        records = await self._retry(
            "retrieve",
            lambda: self._client.retrieve(
                collection_name=self.collection_name,
                ids=list(ids),
                with_payload=True,
                with_vectors=False,
            ),
        )
        return [Chunk.from_payload(record.payload) for record in records if record.payload]
```

Add to `ChromaStore` (`src/vectorstores/chroma_store.py`):

```python
    async def fetch_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        response = await asyncio.to_thread(
            self._collection.get, ids=list(ids), include=["metadatas"]
        )
        return [Chunk.from_payload(meta) for meta in (response.get("metadatas") or []) if meta]
```

Both need `Chunk` imported and `Sequence` from `collections.abc`.

### The Theory: the token arithmetic, done honestly

This is where a design decision meets a budget, so it is worth working through with real numbers.

| Configuration | Sources | Tokens each | Context |
| :--- | ---: | ---: | ---: |
| Children only, no substitution | 5 | ~400 | ~2,000 |
| Parents, no deduplication | 5 | ~1,600 | ~8,000 |
| Parents, deduplicated (typical) | 2–3 | ~1,600 | ~3,200–4,800 |

Deduplication is doing most of the work, and it is not a micro-optimisation: five reranked children
frequently come from two or three parents, because RRF rewards agreement and children of the same
parent are about the same thing. A realistic context is therefore ~4,000 tokens rather than 8,000 —
double the children-only case, for materially more complete context.

The second-order effect is worth knowing too. Long contexts suffer from **position bias**: models
attend most reliably to the beginning and end of a prompt, and material in the middle is measurably
less likely to be used. Fewer, larger, deduplicated sources put more of your context in the
well-attended regions than many small overlapping ones do. Deduplication improves answer quality for a
reason that has nothing to do with saving tokens.

### Failure Modes

**Every source is 400 tokens after substitution.** `parent_id` is None on your chunks. Either
`ENABLE_PARENT_SUBSTITUTION` is off, or the corpus was ingested with `SentenceChunker` rather than
`ParentChildChunker`.

**"Some parent chunks are missing" on every query.** Parents were never indexed. Phase 2's chunker
emits both levels in one list — if a custom ingestion loop filtered to children before storing, the
parents do not exist. Re-index.

**Citations point at chunks the user cannot find.** After substitution, `chunk_id` is the parent's,
whose `chunk_index` is ≥ 1,000,000 (Phase 2's offset). That is correct and intended, but Phase 9's UI
should display `contract_name` and `section_title`, never a raw chunk index — 1,000,003 is not a
section number and will look like a bug to a user.

**Context still overruns the model.** The budget uses stored `token_count`, which Phase 1 documents as
approximate, and the prompt template plus the question add more. Keep `MAX_CONTEXT_TOKENS` well below
the real window; 12,000 against a 128k model is deliberately conservative because Phase 6's prompts,
Phase 12's map-reduce, and Phase 13's graph context all draw from the same budget.

---

## 11. File 9 — `src/retrieval/pipeline.py`

### The Problem

Eight components, one order, several optional. The order is not arbitrary and getting it wrong
produces a system that works and underperforms — the hardest kind of bug to find, because nothing
fails.

### Design Decision

**One class, one method, stages in a fixed order, every optional stage able to no-op.**

The order and its justification, in one place:

| # | Stage | Why here |
| :-- | :--- | :--- |
| 1 | Expand + HyDE, concurrently | Both are independent LLM calls; sequencing them doubles latency for nothing |
| 2 | Hybrid search, all probes concurrently | Filtered to children. Server fuses dense+sparse per probe |
| 3 | RRF across probes | Ranks are the only comparable quantity across different query vectors |
| 4 | Cross-encoder rerank | On children, which fit the 512-token window. On the *fused* set, so it sees the best of every probe |
| 5 | MMR | After reranking, so it diversifies within accurate scores rather than fighting them |
| 6 | Parent substitution | Last. Reranking parents would truncate them; substituting before reranking wastes the good model on the wrong text |

**Everything is injected.** Phase 5's graph nodes will hold one of these; Phase 8 builds it once at
startup; Phase 10 passes fakes. Constructing dependencies inside would make all three awkward.

**The result carries diagnostics, not just chunks.** `RetrievalResult` has `expanded_queries`,
`total_candidates`, and `latency_ms` because Phase 7 reads them and because they are what you inspect
when recall is poor. A retrieval function that returns only a list of chunks is unmeasurable.

```python
import asyncio
import time
from typing import Any

from config.settings import settings
from src.core.interfaces import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseReranker,
    BaseVectorStore,
)
from src.core.logging import logger
from src.core.models import RetrievalResult
from src.core.telemetry import telemetry

from .fusion import reciprocal_rank_fusion
from .hybrid import HybridRetriever
from .hyde import HyDEGenerator
from .multi_query import QueryExpander
from .parents import ParentSubstituter
from .rerankers.cross_encoder import CrossEncoderReranker


class RetrievalPipeline:
    """Pipeline 2's search half: question in, ranked context out.

    Knows nothing about generation. Phase 5's graph calls this and hands the
    result to an LLM; keeping that boundary sharp is what lets Phase 7 evaluate
    retrieval quality independently of answer quality.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        store: BaseVectorStore,
        llm: BaseLLMProvider | None = None,
        reranker: BaseReranker | None = None,
        diversifier: BaseReranker | None = None,
    ) -> None:
        self.retriever = HybridRetriever(embedder, store)
        self.expander = QueryExpander(llm)
        self.hyde = HyDEGenerator(llm)
        self.reranker = reranker or CrossEncoderReranker()
        self.diversifier = diversifier
        self.parents = ParentSubstituter(store)

    @telemetry.span("retrieval.pipeline", warn_over_ms=3000)
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval cascade.

        Raises:
            RetrievalError: every search arm failed. Everything else degrades.
        """
        started = time.perf_counter()
        final_k = top_k or settings.RERANK_TOP_K

        # ── 1. probes ──────────────────────────────────────────────────────
        # Independent LLM calls, issued together. Sequential awaits here would
        # add 400ms to every request for no reason.
        queries, hypothetical = await asyncio.gather(
            self.expander.expand(query),
            self.hyde.generate(query),
        )

        # ── 2. search ──────────────────────────────────────────────────────
        outcome = await self.retriever.search_many(queries, filters=filters)
        result_lists = list(outcome.results)

        if hypothetical:
            # The HyDE text pretends to be a document, so it is embedded with the
            # passage-side transformation, not the query-side one.
            dense, sparse = await self.retriever.embed_probe(hypothetical, as_document=True)
            try:
                result_lists.append(
                    await self.retriever.search_vector(dense, sparse, filters=filters)
                )
            except Exception as exc:
                logger.warning("HyDE arm failed", extra={"error": type(exc).__name__})

        # ── 3. fuse ────────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(result_lists, top_k=settings.RETRIEVAL_TOP_K)

        # ── 4. rerank ──────────────────────────────────────────────────────
        # On children. A parent would exceed the cross-encoder's 512-token window
        # and be scored on its first quarter.
        reranked = await self.reranker.rerank(query, fused, top_k=final_k)

        # ── 5. diversify ───────────────────────────────────────────────────
        if self.diversifier is not None:
            reranked = await self.diversifier.rerank(query, reranked, top_k=final_k)

        # ── 6. substitute parents ──────────────────────────────────────────
        # Last, because everything above wanted small precise text and generation
        # wants large complete text.
        final = await self.parents.substitute(reranked)

        elapsed_ms = (time.perf_counter() - started) * 1000

        result = RetrievalResult(
            original_query=query,
            expanded_queries=queries[1:],
            chunks=final,
            total_candidates=outcome.total_candidates,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "Retrieval complete",
            extra={
                "probes": len(result_lists),
                "failed_arms": outcome.failed_queries,
                "candidates": outcome.total_candidates,
                "fused": len(fused),
                "returned": result.chunk_count,
                "latency_ms": round(elapsed_ms, 1),
            },
        )
        return result

    def warmup(self) -> None:
        """Load the cross-encoder. Call at startup; the first query should not pay."""
        warm = getattr(self.reranker, "warmup", None)
        if warm is not None:
            warm()
```

### Wiring it up

```python
from src.embeddings.factory import get_embedding_provider
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.rerankers.mmr import MMRReranker
from src.vectorstores.factory import get_vector_store

embedder = get_embedding_provider()
store = get_vector_store()

pipeline = RetrievalPipeline(
    embedder=embedder,
    store=store,
    llm=None,                                    # Phase 6 supplies this
    diversifier=MMRReranker(embedder) if settings.ENABLE_MMR else None,
)
pipeline.warmup()

result = await pipeline.retrieve("what happens if we terminate early?")
for scored in result.chunks:
    print(scored.rank, scored.chunk.section_title, round(scored.effective_score, 4))
```

Note `llm=None`. With no LLM this is single-query hybrid search plus reranking plus parent
substitution — which is already a good retrieval system, and is exactly what Phase 10 tests against.
Phase 5 passes the real provider once Phase 6 exists.

### The Theory: the latency budget, stage by stage

Rough figures on a laptop with a warm cache, for one query. Know these numbers; "why is retrieval
slow" is answerable only if you know where the time goes.

| Stage | Cost | Notes |
| :--- | ---: | :--- |
| Expansion (LLM) | 200–600ms | Concurrent with HyDE. Cacheable in Phase 6 |
| HyDE (LLM) | 300–800ms | Off by default. Concurrent, so it replaces expansion's cost rather than adding |
| Embed 4 probes | 20–40ms | Local ONNX, concurrent |
| 4 hybrid searches | 50–150ms | Concurrent; roughly one search's latency |
| RRF fusion | <1ms | Pure Python over ~80 items |
| Cross-encoder, 20 candidates | 20–40ms | Linear in candidates |
| MMR | 30–50ms | Mostly the re-embedding |
| Parent fetch | 10–30ms | One `retrieve` round trip |
| **Total, LLM stages on** | **~700–1,100ms** | |
| **Total, LLM stages off** | **~120–260ms** | |

The shape of that table is the point: **the two LLM calls are 80% of the latency and they happen
before any retrieval starts.** Everything Phases 3 and 4 built — HNSW, server-side fusion, ONNX
reranking — lives inside a 200ms envelope. If retrieval feels slow, the answer is almost always to
cache or skip the query rewriting, not to tune the vector search.

That is precisely why Phase 5 has a `router_node` (skip expansion for queries that do not need it) and
Phase 6 has a semantic cache (skip it for queries you have seen). Neither is an optimisation looking
for a problem; both target the dominant term.

### Failure Modes

**Retrieval takes 3 seconds and trips the span warning.** Check the logged `probes` count. If it is
larger than expected, expansion returned more variations than requested. If probes is 4 and it is
still slow, one arm is timing out — `failed_arms` will be non-zero.

**`expanded_queries` is empty but multi-query is enabled.** No LLM was injected. Check the INFO log at
startup: `Query expansion disabled reason=no LLM`.

**Results are worse after adding MMR.** Turn it off. It is off by default for this reason, and Phase 7
is how you decide rather than guessing.

**Answers cite text not in the retrieved chunks.** Not this phase — that is Phase 6's citation
validator and Phase 5's grader. But check that parent substitution ran: if the LLM saw parents and
your validator checks children, everything will look like a hallucination when the pipeline is
actually correct.

---

## 12. Verification (deferred)

Save as `scripts/verify_phase4.py`. Run after `verify_phase3.py` passes, with Qdrant running. It
builds its own tiny collection so it does not depend on a full ingestion run.

```python
"""Phase 4 verification. Run from the project root, with Qdrant running:

    python scripts/verify_phase4.py

Creates and destroys a temporary collection. No LLM required — the expansion and
HyDE paths are verified in their disabled state, which is exactly how Phase 4 runs
before Phase 6 exists.
"""
import asyncio
import sys

from src.core.models import Chunk, ChunkLevel
from src.core.utils import make_chunk_id, make_doc_id, make_section_id
from src.embeddings.fastembed_provider import FastEmbedProvider
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.parents import ParentSubstituter
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.rerankers.cross_encoder import CrossEncoderReranker
from src.retrieval.rerankers.mmr import MMRReranker
from src.vectorstores.qdrant_store import QdrantStore

TEMP_COLLECTION = "verify_phase4_tmp"
DOC_ID = make_doc_id("data/contracts/2003/verify4.txt")

#: Five distinct topics, each as a parent with two children. The children are
#: deliberately incomplete — the middle one references a section it does not
#: contain, which is the exact case parent substitution exists for.
TOPICS = {
    "Termination": (
        "Either party may terminate this Agreement for convenience upon ninety (90) "
        "days prior written notice to the other party. Upon such termination the "
        "Company shall pay all amounts accrued through the effective date.",
        "Either party may terminate this Agreement for convenience upon ninety (90) days prior written notice.",
    ),
    "Indemnification": (
        "The Company shall indemnify and hold harmless the Purchaser from any Losses "
        "arising out of a breach of this Agreement, provided that the aggregate "
        "liability shall not exceed the Cap set forth in Section 4.02(b).",
        "The aggregate liability shall not exceed the Cap set forth in Section 4.02(b).",
    ),
    "Confidentiality": (
        "Confidential Information shall not be disclosed to any third party without "
        "prior written consent, and shall be protected with no less than reasonable "
        "care for a period of five (5) years following disclosure.",
        "Confidential Information shall not be disclosed to any third party without prior written consent.",
    ),
    "Governing Law": (
        "This Agreement shall be governed by and construed in accordance with the "
        "laws of the State of Delaware, without regard to its conflict of laws "
        "principles, and the parties consent to exclusive jurisdiction therein.",
        "This Agreement shall be governed by the laws of the State of Delaware.",
    ),
    "Payment": (
        "The Purchase Price shall be paid in immediately available funds at Closing "
        "by wire transfer to an account designated by the Seller no later than two "
        "(2) Business Days prior to the Closing Date.",
        "The Purchase Price shall be paid in immediately available funds at Closing.",
    ),
}


def build_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for order, (title, (parent_text, child_text)) in enumerate(TOPICS.items()):
        section_id = make_section_id(DOC_ID, title, order)
        parent_index = 1_000_000 + order          # Phase 2's parent offset
        parent_id = make_chunk_id(DOC_ID, section_id, parent_index)

        def make(text: str, index: int, level: ChunkLevel, parent: str | None) -> Chunk:
            return Chunk(
                chunk_id=make_chunk_id(DOC_ID, section_id, index),
                doc_id=DOC_ID,
                section_id=section_id,
                text=text,
                chunk_index=index,
                token_count=len(text) // 4,
                contract_name="Phase 4 Verification Agreement",
                file_name="verify4.txt",
                section_title=title,
                year=2003,
                chunk_level=level,
                parent_id=parent,
            )

        chunks.append(make(parent_text, parent_index, ChunkLevel.PARENT, None))
        chunks.append(make(child_text, order * 10, ChunkLevel.CHILD, parent_id))
        # A second, near-duplicate child so MMR has something to de-duplicate.
        chunks.append(make(child_text + " The foregoing applies to each party.",
                           order * 10 + 1, ChunkLevel.CHILD, parent_id))
    return chunks


def check_fusion() -> None:
    """RRF must reward agreement and must not mutate its inputs."""
    corpus = build_corpus()
    children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]

    from src.core.models import RetrievalMethod, ScoredChunk

    def rank_list(order: list[int]) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=children[i], score=1.0 - n * 0.1, rank=n,
                        method=RetrievalMethod.HYBRID)
            for n, i in enumerate(order)
        ]

    # Chunk 2 is 2nd, 1st, and 2nd. Chunk 0 is 1st once and absent twice.
    fused = reciprocal_rank_fusion([rank_list([0, 2, 4]), rank_list([2, 1]), rank_list([3, 2])], k=60)

    assert fused[0].chunk.chunk_id == children[2].chunk_id, (
        "RRF failed to reward the chunk that three probes agreed on"
    )
    assert fused[0].rank == 0 and fused[1].rank == 1, "fused ranks must be 0-based and dense"
    assert all(f.rerank_score is None for f in fused), (
        "fusion must not carry a rerank_score — nothing has been reranked yet"
    )
    assert len({f.chunk.chunk_id for f in fused}) == len(fused), "fusion returned duplicates"

    # A single-list fusion must preserve that list's order exactly.
    single = reciprocal_rank_fusion([rank_list([3, 1, 4, 0])], k=60)
    assert [s.chunk.chunk_id for s in single] == [children[i].chunk_id for i in [3, 1, 4, 0]], (
        "fusing one list changed its order"
    )
    print("✓ fusion: agreement rewarded, ranks dense, no duplicates, single list preserved")


async def check_reranker(provider: FastEmbedProvider) -> None:
    """The cross-encoder must beat the retrieval order on a query it should win."""
    from src.core.models import RetrievalMethod, ScoredChunk

    corpus = build_corpus()
    children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]

    # Deliberately adversarial: the correct answer is placed LAST in the input.
    query = "how much notice is required to terminate for convenience?"
    ordered = [c for c in children if c.section_title != "Termination"]
    ordered += [c for c in children if c.section_title == "Termination"]

    candidates = [
        ScoredChunk(chunk=c, score=1.0 - n * 0.01, rank=n, method=RetrievalMethod.HYBRID)
        for n, c in enumerate(ordered)
    ]

    reranker = CrossEncoderReranker()
    reranker.warmup()
    result = await reranker.rerank(query, candidates, top_k=3)

    assert len(result) == 3, f"expected 3 results, got {len(result)}"
    assert result[0].chunk.section_title == "Termination", (
        f"cross-encoder ranked {result[0].chunk.section_title!r} first; it should have "
        "promoted the Termination clause from last place"
    )
    assert result[0].rerank_score is not None, "rerank_score must be populated"
    assert result[0].effective_score == result[0].rerank_score, (
        "effective_score must prefer rerank_score once it is set"
    )
    assert result[0].score != result[0].rerank_score, (
        "the retrieval score must be preserved, not overwritten"
    )
    assert [r.rank for r in result] == [0, 1, 2], "ranks must be rewritten after sorting"

    scores = [round(r.rerank_score, 3) for r in result]
    print(f"✓ reranker: promoted the buried correct answer to rank 0; logits={scores}")


async def check_mmr(provider: FastEmbedProvider) -> None:
    """MMR must drop a near-duplicate that pure relevance would keep."""
    from src.core.models import RetrievalMethod, ScoredChunk

    corpus = build_corpus()
    conf = [c for c in corpus
            if c.chunk_level == ChunkLevel.CHILD and c.section_title == "Confidentiality"]
    other = [c for c in corpus
             if c.chunk_level == ChunkLevel.CHILD and c.section_title != "Confidentiality"]

    # Two near-identical confidentiality children first, then everything else.
    ordered = conf + other
    candidates = [
        ScoredChunk(chunk=c, score=1.0 - n * 0.01, rank=n, method=RetrievalMethod.HYBRID)
        for n, c in enumerate(ordered)
    ]

    mmr = MMRReranker(provider, lambda_param=0.5)
    selected = await mmr.rerank("confidentiality obligations", candidates, top_k=2)

    titles = [s.chunk.section_title for s in selected]
    assert len(set(titles)) == 2, (
        f"MMR kept two chunks from the same section: {titles}. With lambda=0.5 the "
        "second slot should have gone to a different topic."
    )
    print(f"✓ mmr: second slot went to a different topic ({titles})")


async def check_parents(store: QdrantStore) -> None:
    """Substitution must deduplicate, keep order, and survive a missing parent."""
    from src.core.models import RetrievalMethod, ScoredChunk

    corpus = build_corpus()
    children = [c for c in corpus if c.chunk_level == ChunkLevel.CHILD]

    # Both children of Indemnification, plus one child of Termination.
    indem = [c for c in children if c.section_title == "Indemnification"]
    term = [c for c in children if c.section_title == "Termination"][:1]
    picked = indem + term

    candidates = [
        ScoredChunk(chunk=c, score=0.9 - n * 0.1, rank=n, method=RetrievalMethod.HYBRID,
                    rerank_score=5.0 - n)
        for n, c in enumerate(picked)
    ]

    substituter = ParentSubstituter(store)
    result = await substituter.substitute(candidates)

    assert len(result) == 2, (
        f"expected 2 deduplicated parents from 3 children, got {len(result)}"
    )
    assert all(r.chunk.chunk_level == ChunkLevel.PARENT for r in result), (
        "substitution returned children — parents missing from the index?"
    )
    assert result[0].chunk.section_title == "Indemnification", "child rank order not preserved"
    assert "Section 4.02(b)" in result[0].chunk.text and "Cap set forth" in result[0].chunk.text
    assert result[0].chunk.token_count > candidates[0].chunk.token_count, (
        "the parent should be larger than the child it replaced"
    )
    assert result[0].rerank_score == candidates[0].rerank_score, (
        "the child's scores must carry onto its parent"
    )
    assert [r.rank for r in result] == [0, 1], "ranks must be recomputed after deduplication"

    # A dangling parent_id must degrade to the child, not raise.
    orphan = candidates[0].chunk.model_copy(update={"parent_id": make_chunk_id(DOC_ID, "x", 99)})
    degraded = await substituter.substitute(
        [ScoredChunk(chunk=orphan, score=0.5, rank=0, method=RetrievalMethod.HYBRID)]
    )
    assert len(degraded) == 1 and degraded[0].chunk.chunk_id == orphan.chunk_id, (
        "a missing parent must fall back to the child, not vanish and not raise"
    )

    print("✓ parents: 3 children → 2 parents, order and scores preserved, orphan degraded")


async def check_pipeline(provider: FastEmbedProvider, store: QdrantStore) -> None:
    """End to end, with no LLM — the configuration Phase 4 ships in."""
    pipeline = RetrievalPipeline(embedder=provider, store=store, llm=None)
    pipeline.warmup()

    result = await pipeline.retrieve("what notice is needed to end the agreement early?", top_k=3)

    assert result.chunk_count > 0, "pipeline returned nothing"
    assert result.expanded_queries == [], "no LLM was injected, so there can be no variations"
    assert result.total_candidates > 0, "candidate count not recorded"
    assert result.latency_ms > 0, "latency not recorded"
    assert result.original_query.startswith("what notice"), "original query not preserved"

    assert all(c.chunk.chunk_level == ChunkLevel.PARENT for c in result.chunks), (
        "final context should be parents after substitution"
    )
    assert result.chunks[0].chunk.section_title == "Termination", (
        f"top result was {result.chunks[0].chunk.section_title!r}; the vocabulary gap "
        "between 'end the agreement early' and 'Termination for Convenience' was not bridged"
    )

    # Every returned chunk must have gone through the full chain.
    assert all(c.rerank_score is not None for c in result.chunks), "reranking did not run"
    assert [c.rank for c in result.chunks] == list(range(result.chunk_count))

    print(f"✓ pipeline: {result.chunk_count} sources in {result.latency_ms:.0f}ms, "
          f"top = {result.chunks[0].chunk.section_title!r}")


async def main() -> int:
    provider = FastEmbedProvider()
    provider.warmup()

    check_fusion()

    store = QdrantStore(collection_name=TEMP_COLLECTION)
    try:
        await store.initialize(provider.dense_dimensions)

        corpus = build_corpus()
        texts = [c.text for c in corpus]
        dense = await provider.embed_dense(texts)
        sparse = await provider.embed_sparse(texts)
        await store.delete_by_doc_ids([DOC_ID])
        await store.upsert_points(corpus, dense, sparse)
        print(f"  indexed {len(corpus)} chunks "
              f"({sum(1 for c in corpus if c.chunk_level == ChunkLevel.PARENT)} parents)")

        await check_reranker(provider)
        await check_mmr(provider)
        await check_parents(store)
        await check_pipeline(provider, store)
    except AssertionError as exc:
        print(f"\n✗ FAILED: {exc}")
        return 1
    finally:
        try:
            await store._client.delete_collection(TEMP_COLLECTION)
            print(f"  (cleaned up {TEMP_COLLECTION})")
        finally:
            await store.close()

    print("\nPhase 4 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

`check_mmr` sets `lambda_param=0.5` explicitly and `ENABLE_MMR` must be true for it to do anything —
if that assertion fails first, check the flag before suspecting the algorithm.

### What to look at, beyond the assertions

**The reranker's logits.** You should see a clear gap — something like `[7.2, -3.1, -8.4]`. If all
three are within a point of each other, the cross-encoder is not discriminating, which on this tiny
synthetic corpus would suggest the wrong model loaded.

**The pipeline's latency line.** With no LLM, expect 120–260ms. Substantially more means the
cross-encoder is loading inside the request rather than at `warmup()`.

**The `overlap_ratio` in the fusion log during the pipeline check.** With one probe it will be 0.0 by
definition. Once Phase 6 supplies an LLM, watch this number — §6 explains how to read it.

**After a real ingestion run,** try the three vocabulary-gap queries from §3 by hand. Those are the
queries this entire phase exists to answer, and they are the ones to check before believing any
benchmark.

---

## 13. What Phase 4 Bought You

**Retrieval that survives the vocabulary gap.** Four probes instead of one, fused by rank. The casual
question now reaches the drafted clause.

**A precision stage that a bi-encoder cannot provide.** The cross-encoder reads query and passage
together over the top 20, which is worth more than any amount of embedding-model tuning and is
affordable only because retrieval narrowed the field first. `rerank_score` sits beside `score`, so
you can prove it.

**Phase 2's parent-child work finally paying off.** Search precision from 400-token children, answer
completeness from 1,600-token parents, deduplicated so five children become two or three sources
rather than five copies.

**A measurable pipeline.** `RetrievalResult` carries the expansions, the candidate count, and the
latency. Phase 7 evaluates against those fields; the `overlap_ratio` and `top_changed` diagnostics
tell you whether each optional stage is earning its cost.

**Graceful degradation at every optional stage.** No LLM, expansion off. Expansion fails, use the
original. Reranker fails, use the RRF order. Parent missing, use the child. One search arm fails,
fuse the rest. The only unrecoverable failure is every search arm failing, which is a genuine outage
and correctly raises with `retryable=True`.

### What is deliberately not here

**Any LLM provider.** Phase 6. This phase depends on the `BaseLLMProvider` interface and takes an
optional instance. That is what let it be written before Phase 6 exists, and it is why Phase 10 can
test retrieval without an API key.

**Generation, grading, and the retry loop.** Phase 5. `RetrievalPipeline.retrieve` returns context and
stops. The temptation to have it also produce an answer is exactly what makes retrieval quality
unmeasurable.

**Query routing.** Phase 5's `router_node`. §11's latency table shows why it matters — the LLM stages
are 80% of the cost — but the decision of *whether* to expand belongs to the graph that owns the
request, not to the retriever.

**Aggregation across many documents.** "How many contracts contain a Delaware governing-law clause"
cannot be answered by top-k retrieval at all, no matter how good the reranker. That is Phase 12's
map-reduce, and recognising that it is a different problem rather than a tuning failure is the point.

---

## Next

**Phase 5 — the LangGraph agent.** It wraps this pipeline in a state machine that can grade its own
answer and retry with a rewritten query. Two things to do before writing it: verify LangGraph's
current `StateGraph` API by web search (it moves fast, and the handoff flags this), and note that
Phase 5 will need Phase 6's LLM provider for real — the optional-LLM trick that carried Phase 4 does
not extend to a generation node.
