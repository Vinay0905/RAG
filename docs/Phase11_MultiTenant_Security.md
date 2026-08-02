# Phase 11 — Multi-Tenant Security and RBAC

> **Part II begins.** Phases 11–16 solve the six problems in `future_ideas_problems.md`, in your
> order. This is #1: *Enterprise Permission Leakage & Multi-Tenant Access Control*.
>
> **Prerequisite:** Phases 1–10. This phase modifies shared surfaces — it is the only Part II phase
> that does, which is why it goes first and alone.
>
> **Budget: ~610 lines of Python across 5 files, on budget.** §2 lists what was cut.

---

## 1. The Problem, In Your Words

> **The Failure:** In real companies (SharePoint, Confluence, Google Drive), User A (Junior
> Associate) should NOT see executive compensation contracts, while User B (Partner) can see
> everything.
>
> **Why Our Baseline Fails:** Qdrant currently searches across all indexed contracts globally
> without evaluating user identity or role permissions during vector similarity lookup.
>
> **The Fix Required:** Attach User/Group ACL tokens to chunk metadata
> (`allowed_roles: ["legal_partner", "admin"]`). Inject security session context into Qdrant queries
> to execute pre-search RBAC payload filtering.

That is the design, and this phase implements it. What is worth adding is *why the obvious
implementations of it fail*, because there are three, and all three look like they work.

**Filtering after retrieval is a leak, not a filter.** The natural first attempt is to retrieve the
top 20 and drop the ones the user cannot see. Two things break. The result count collapses
unpredictably — a partner gets 20 sources and an associate gets 3, from the same query. And the
leak is still there: the *scores* were computed over documents the associate cannot see, the LLM
never sees them but the ranking was shaped by them, and any diagnostic surface (Phase 8's
`/admin/stats`, Phase 9's trace, an error message naming a contract) can expose their existence.
Filtering has to happen **inside** the vector search — which is exactly what "pre-search RBAC payload
filtering" means and why you wrote it that way.

**A filter that can be forgotten will be forgotten.** There are five call sites that reach the store
across Phases 4, 5, 7, and 16. An ACL that depends on each of them remembering to pass a filter is
an ACL with five ways to fail open, and the failure is invisible — the query returns *more* results,
which nobody reports as a bug.

**Caches ignore identity unless you make them.** Phase 6's answer cache is keyed on the question. Two
users asking "what is the CEO's severance?" get one cache entry, and whoever asks second receives the
first user's answer *and their source documents*. This is the single most likely way to build a
system that passes an ACL review and leaks anyway.

### The three things Part I already did for this phase

Not coincidence — these were built knowing this phase was coming, and they are why the budget is 610
rather than 1,200:

| Already in place | Where | Why it matters here |
| :--- | :--- | :--- |
| `_build_filter` **raises** on unknown fields | Phase 3 §8 | A typo'd `allowed_roles` cannot be silently dropped, returning everything |
| `cache_scope()` partitions both cache layers | Phase 6 §7–8 | Add the principal to it and the cross-user cache leak closes |
| Payload indexes are declared in one tuple | Phase 3 §7 | One line adds a `keyword` index on the ACL field |

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| 1 | `src/security/principal.py` | 90 | Who is asking, and what they may see |
| 2 | `src/security/acl.py` | 120 | Document → `allowed_roles`, at ingestion |
| 3 | `src/security/filters.py` | 110 | Principal → an un-forgettable store filter |
| 4 | `src/security/middleware.py` | 130 | Request → principal, and per-request context |
| 5 | Modifications to Part I | 160 | Store, cache, agent, API |

```text
src/security/
├── __init__.py
├── principal.py
├── acl.py
├── filters.py
└── middleware.py
```

---

## 2. What Was Cut

**A real identity provider.** No OAuth, no OIDC, no SAML, no user database. The principal comes from
a signed JWT whose issuance is somebody else's problem, plus a dev-mode header for local use.
Integrating an IdP teaches you about that IdP, not about RAG security.

**Row-level permissions and per-document sharing.** Role-based, not user-based: a document carries
`allowed_roles`, not `allowed_users`. Per-user ACLs mean a membership table, invalidation when
membership changes, and a filter with unbounded cardinality. Roles cover your stated example exactly.

**An admin API for managing roles.** Roles come from the token; documents get their `allowed_roles`
at ingestion from a rules file. Managing that in a UI is a CRUD app.

**Audit logging to a durable store.** Access decisions are logged through Phase 1's structured
logger with the principal attached. Shipping them to a tamper-evident store is deployment work.

**Encryption at rest and field-level encryption.** Real requirements in this domain, and entirely
about infrastructure — Qdrant's storage, Redis' persistence — not about this codebase.

That is ~900 lines declined. The rule from Phase 7 onward holds.

---

## 3. File 1 — `src/security/principal.py`

### Design Decision

**One frozen object that travels the whole request.** Every decision in this phase is a function of
it, and making it immutable means no node, node, or middleware downstream can widen its own
permissions.

**A deny-by-default constructor.** `Principal()` with no arguments is anonymous with no roles and
sees nothing. Every failure path in this phase produces that object rather than `None`, because
`None` invites `if principal:` and a missing principal must not read as "skip the check".

```python
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, computed_field

#: Sees everything. One role, named once, so a `role == "admin"` string comparison
#: never appears anywhere else.
ADMIN_ROLE = "admin"

#: Attached to documents nobody has been granted explicitly. Everyone has it, which
#: makes the default behaviour of an un-classified document "visible to staff" rather
#: than "visible to nobody" — see the note on fail-closed below.
PUBLIC_ROLE = "public"


class Principal(BaseModel):
    """Who is asking. Immutable, and carried through the whole request.

    Deny by default: the no-argument construction is an anonymous principal with no
    roles, which matches nothing. Every error path in this phase returns THIS rather
    than None, because a nullable principal invites `if principal:` — and a missing
    principal must never read as "no check required".
    """

    model_config = ConfigDict(frozen=True)

    subject: str = Field(default="anonymous", description="Stable user identifier.")
    #: Roles from the token. `frozenset` rather than `list`: order is meaningless,
    #: membership is the only operation, and it cannot be appended to in flight.
    roles: frozenset[str] = Field(default_factory=frozenset)
    tenant: str | None = None
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def is_admin(self) -> bool:
        return ADMIN_ROLE in self.roles

    @computed_field
    @property
    def is_anonymous(self) -> bool:
        return not self.roles

    def visible_roles(self) -> frozenset[str]:
        """The role set a document must intersect for this principal to see it.

        `public` is added implicitly for any authenticated principal, so a document
        with no explicit classification is visible to staff rather than to nobody.
        That is a deliberate usability choice and it is the ONE place this phase is
        not fail-closed: an operator who forgets to classify a document exposes it
        to everyone with a login, not to the internet. §4 explains why the
        alternative is worse in practice.
        """
        if self.is_anonymous:
            return frozenset()
        return self.roles | {PUBLIC_ROLE}

    def can_see(self, allowed_roles: list[str] | None) -> bool:
        """Whether this principal may see a document with these roles.

        Used for defence in depth AFTER the store filter, not instead of it. If this
        ever returns False on a retrieved chunk, the pre-search filter failed and
        that is a bug worth an alert — see `filters.assert_permitted`.
        """
        if self.is_admin:
            return True
        if not allowed_roles:
            # An unclassified document. Treated as public, consistent with
            # `visible_roles`, and refused outright for anonymous principals.
            return not self.is_anonymous
        return bool(self.visible_roles() & set(allowed_roles))

    def for_log(self) -> dict[str, str]:
        """Identity for structured logs. Never the token, never a name."""
        return {"subject": self.subject, "roles": ",".join(sorted(self.roles)),
                "tenant": self.tenant or "-"}


#: The principal used when authentication is disabled entirely (local development
#: and Phase 10's tests). Explicit and named, so it can be grepped for before a
#: deployment and never appears by accident.
DEV_PRINCIPAL = Principal(subject="dev", roles=frozenset({ADMIN_ROLE}))
ANONYMOUS = Principal()
```

---

## 4. File 2 — `src/security/acl.py`

### The Problem

Your fix says "attach User/Group ACL tokens to chunk metadata". Something has to decide *which* roles,
for 650,000 documents, without a human classifying each one.

### Design Decision

**Rules evaluated at ingestion, written to the chunk payload.** Classification happens once, offline,
and the query path only ever reads. The alternative — resolving permissions at query time from a
separate system — adds a network call to every search and a second source of truth.

**Path and content rules, in that order, first match wins.** Directory structure carries most of the
signal in a real document store (`/executive/`, `/hr/`), and a content pattern catches the rest.

**An unmatched document gets `["public"]`, and that is a deliberate, argued choice.** Fail-closed —
`[]`, visible to nobody — is the textbook answer and it is wrong here. With 650,000 documents and
rules that will not cover every case, fail-closed means a large fraction of the corpus silently
disappears from every search, and the symptom is "retrieval got worse" rather than "permissions are
misconfigured". Nobody debugs that to the ACL. Failing open *to authenticated staff only* keeps the
system usable while `report_coverage()` makes the unclassified fraction a number you look at.

The honest statement: **this trades a confidentiality risk for a discoverability one, bounded by
authentication.** In a deployment where the corpus contains material that must never be
over-exposed, invert `DEFAULT_ROLES` to `[]` and accept the support burden. That is a one-constant
change, and it is a policy decision rather than an engineering one.

```python
import re
from dataclasses import dataclass
from pathlib import Path

from src.core.logging import logger
from src.core.models import Chunk, Document

from .principal import PUBLIC_ROLE

#: The payload field. One constant: it appears in the chunk model, the payload
#: index, the query filter, and the ingestion rules, and a typo in any of them
#: produces a filter that matches nothing or everything.
ACL_FIELD = "allowed_roles"

#: What an unmatched document gets. See the argument above before changing it.
DEFAULT_ROLES = [PUBLIC_ROLE]


@dataclass(frozen=True)
class AclRule:
    """One classification rule. `path_pattern` is checked first — it is cheaper and
    directory structure is the stronger signal."""

    name: str
    roles: list[str]
    path_pattern: str | None = None
    content_pattern: str | None = None


#: Ordered, first match wins. Most restrictive first, because a compensation
#: agreement filed under /hr/ must get the executive rule, not the HR one.
DEFAULT_RULES: tuple[AclRule, ...] = (
    AclRule(
        name="executive_compensation",
        roles=["legal_partner", "board"],
        path_pattern=r"(executive|compensation|severance|board)",
        content_pattern=r"\b(Chief Executive Officer|Executive Severance|"
                        r"Change in Control Agreement)\b",
    ),
    AclRule(name="hr", roles=["legal_partner", "hr"], path_pattern=r"(employment|hr|personnel)"),
    AclRule(name="litigation", roles=["legal_partner", "litigation"],
            path_pattern=r"(litigation|dispute|settlement)"),
    AclRule(name="commercial", roles=["legal_partner", "associate", PUBLIC_ROLE],
            path_pattern=r"(purchase|supply|vendor|license)"),
)


class AclClassifier:
    """Assigns `allowed_roles` to documents at ingestion time.

    Runs inside Phase 2's multiprocessing workers, so it holds no state beyond
    compiled patterns and is safe to pickle.
    """

    def __init__(self, rules: tuple[AclRule, ...] = DEFAULT_RULES) -> None:
        self.rules = rules
        self._paths = {
            r.name: re.compile(r.path_pattern, re.IGNORECASE)
            for r in rules if r.path_pattern
        }
        self._contents = {
            r.name: re.compile(r.content_pattern, re.IGNORECASE)
            for r in rules if r.content_pattern
        }
        self.counts: dict[str, int] = {}

    def classify(self, document: Document) -> list[str]:
        """Roles permitted to see this document."""
        path = Path(document.source_path).as_posix()

        for rule in self.rules:
            pattern = self._paths.get(rule.name)
            if pattern and pattern.search(path):
                return self._record(rule.name, rule.roles)

        # Content rules scan only the head. A rule that needs page 40 is a rule that
        # costs a full read of 37 GB, and the classifying language in a contract is
        # in its title and recitals.
        head = document.content[:4000]
        for rule in self.rules:
            pattern = self._contents.get(rule.name)
            if pattern and pattern.search(head):
                return self._record(rule.name, rule.roles)

        return self._record("unclassified", DEFAULT_ROLES)

    def _record(self, name: str, roles: list[str]) -> list[str]:
        self.counts[name] = self.counts.get(name, 0) + 1
        return list(roles)

    def report_coverage(self) -> dict[str, int]:
        """Rule hit counts. Log this after ingestion and READ it.

        A high `unclassified` count is the number that matters: it is the fraction
        of the corpus visible to every authenticated user, and it is the difference
        between an ACL and the appearance of one.
        """
        total = sum(self.counts.values()) or 1
        unclassified = self.counts.get("unclassified", 0)
        if unclassified / total > 0.5:
            logger.warning(
                "Most documents matched no ACL rule and are visible to all staff",
                extra={"unclassified": unclassified, "total": total},
            )
        return dict(self.counts)


def apply_acl(chunks: list[Chunk], roles: list[str]) -> list[Chunk]:
    """Stamp roles onto every chunk of a document.

    Denormalised onto the CHUNK, not looked up from the document at query time,
    because the filter runs inside Qdrant and Qdrant can only filter on the payload
    in front of it. This is the same reasoning as Phase 1's denormalised
    `contract_name`.
    """
    return [c.model_copy(update={"allowed_roles": roles}) for c in chunks]
```

### The Phase 1 and Phase 2 changes this needs

**`src/core/models.py`** — add to `Chunk`:

```python
    #: Roles permitted to retrieve this chunk. Empty means unclassified; see
    #: `acl.DEFAULT_ROLES` for how that is treated.
    allowed_roles: list[str] = Field(default_factory=list)
```

and to both `to_payload()` and `from_payload()`:

```python
            "allowed_roles": self.allowed_roles,      # to_payload
            allowed_roles=payload.get("allowed_roles", []),   # from_payload
```

Phase 10's `test_payload_round_trip_preserves_every_field` covers the new field automatically only if
you add it to that test's field list. **Add it** — a permission field that does not survive the round
trip is an ACL that evaporates on write.

**`src/vectorstores/base.py`** — add one entry to `PAYLOAD_INDEX_FIELDS`:

```python
    (ACL_FIELD, "keyword"),
```

Without it, every permission-filtered search is a full scan of 39 million points, which is the
difference between 40ms and a minute.

**`src/ingestion/pipeline.py`** — in `_init_worker` and `_process_one`:

```python
_classifier = AclClassifier()          # in _init_worker
...
chunks = apply_acl(chunks, _classifier.classify(document))    # after chunking
```

---

## 5. File 3 — `src/security/filters.py`

### The Problem

Five call sites reach the store. An ACL that each of them must remember to apply has five ways to
fail open.

### Design Decision

**A wrapper store that applies the filter, rather than a helper each caller invokes.** `SecureStore`
implements `BaseVectorStore` and holds a principal. Every search through it is filtered, and there is
no method on it that performs an unfiltered one. Forgetting is not possible, because there is nothing
to forget — this is the same reasoning as Phase 4 putting the `chunk_level` filter inside
`HybridRetriever` rather than trusting callers.

**A post-retrieval assertion as well.** Belt and braces: after the store returns, verify every chunk
against the principal. If the pre-search filter worked, this never fires. If it ever does, that is a
security bug and it raises rather than filters, because a silently-corrected leak is a leak you never
learn about.

```python
from typing import Any

from src.core.exceptions import AuthorizationError
from src.core.interfaces import BaseVectorStore
from src.core.logging import logger
from src.core.models import Chunk, ScoredChunk

from .acl import ACL_FIELD
from .principal import Principal


def acl_filter(principal: Principal) -> dict[str, Any]:
    """The filter fragment for this principal.

    An admin gets `{}` — no ACL constraint. Everyone else gets a `MatchAny` over
    their visible roles, which Phase 3's `_build_filter` turns into a Qdrant
    condition. An anonymous principal gets a filter matching a role that cannot
    exist, so it returns nothing rather than everything: the failure mode of a
    permission filter must be an empty result, never an unfiltered one.
    """
    if principal.is_admin:
        return {}
    visible = principal.visible_roles()
    if not visible:
        return {ACL_FIELD: ["__no_access__"]}
    return {ACL_FIELD: sorted(visible)}


class SecureStore(BaseVectorStore):
    """A `BaseVectorStore` that cannot perform an unfiltered search.

    Wraps the real store and injects the principal's ACL filter into every query.
    Phases 4, 5, 7, and 16 all call the store; making the filter a property of the
    OBJECT rather than of each call means none of them can omit it, and a future
    phase that adds a sixth call site inherits the protection without knowing this
    file exists.
    """

    def __init__(self, inner: BaseVectorStore, principal: Principal) -> None:
        self._inner = inner
        self._principal = principal

    async def hybrid_search(
        self, dense_query: list[float], sparse_query: dict[str, list],
        limit: int = 20, filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        merged = {**(filters or {}), **acl_filter(self._principal)}
        # The ACL fragment is merged LAST, so a caller cannot override it by passing
        # its own `allowed_roles`. Order in a dict merge is a security property here.
        results = await self._inner.hybrid_search(dense_query, sparse_query, limit, merged)
        return assert_permitted(results, self._principal)

    async def fetch_by_ids(self, ids) -> list[Chunk]:
        """Parent substitution fetches by ID, which bypasses every filter.

        This is the subtle hole. Phase 4 retrieves permitted CHILDREN and then fetches
        their parents by ID — and an ID lookup has no filter. A child could in
        principle be classified more loosely than its parent, and the parent's full
        text would then reach the prompt. Filtering here closes it.
        """
        chunks = await self._inner.fetch_by_ids(ids)
        permitted = [c for c in chunks if self._principal.can_see(c.allowed_roles)]
        if len(permitted) != len(chunks):
            logger.warning(
                "Filtered parent chunks the principal may not see",
                extra={"removed": len(chunks) - len(permitted),
                       **self._principal.for_log()},
            )
        return permitted

    # ── writes are admin-only ───────────────────────────────────────────────

    async def upsert_points(self, chunks, dense, sparse) -> int:
        self._require_admin("upsert")
        return await self._inner.upsert_points(chunks, dense, sparse)

    async def delete_by_doc_ids(self, doc_ids) -> int:
        self._require_admin("delete")
        return await self._inner.delete_by_doc_ids(doc_ids)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        return await self.delete_by_doc_ids([doc_id])

    async def initialize(self, vector_size: int) -> None:
        self._require_admin("initialize")
        await self._inner.initialize(vector_size)

    async def count(self) -> int:
        return await self._inner.count()

    async def close(self) -> None:
        await self._inner.close()

    def _require_admin(self, operation: str) -> None:
        if not self._principal.is_admin:
            raise AuthorizationError(
                f"{operation} requires an administrator",
                details={"operation": operation, **self._principal.for_log()},
            )


def assert_permitted(results: list[ScoredChunk], principal: Principal) -> list[ScoredChunk]:
    """Verify what came back. Raises if the pre-search filter let something through.

    Defence in depth, and deliberately loud. The tempting implementation quietly
    drops the offending chunks — which works, and means a broken filter runs in
    production indefinitely while the post-filter silently cleans up after it. A
    permission bug you cannot observe is the one that eventually leaks.
    """
    violations = [r for r in results if not principal.can_see(r.chunk.allowed_roles)]
    if violations:
        logger.error(
            "PRE-SEARCH ACL FILTER FAILED — results contained forbidden chunks",
            extra={"violations": len(violations), **principal.for_log()},
        )
        raise AuthorizationError(
            "Retrieval returned material the principal may not access",
            details={"violations": len(violations)},
        )
    return results
```

---

## 6. File 4 — `src/security/middleware.py`

### Design Decision

**A `contextvar` for the principal, plus a FastAPI dependency.** The dependency is how routes get it
explicitly; the contextvar is how the *logger* gets it without threading a parameter through forty
functions — the same mechanism Phase 1 already uses for the request ID, and it propagates through
`asyncio.to_thread` (which Phase 1's correction pass established).

**This is also where Phase 8's `last_trace` bug gets fixed.** That field was per-agent, so two
concurrent streams read each other's progress. A per-request contextvar is exactly the right home for
it, and the reason Phase 8 deferred the fix to here.

```python
import time
from contextvars import ContextVar
from typing import Annotated

import jwt
from fastapi import Depends, Header

from config.settings import settings
from src.core.exceptions import AuthorizationError
from src.core.logging import logger

from .principal import ANONYMOUS, DEV_PRINCIPAL, Principal

#: The current request's principal. Defaults to ANONYMOUS — deny by default, even
#: for code that runs outside a request.
_principal: ContextVar[Principal] = ContextVar("principal", default=ANONYMOUS)

#: Per-request agent trace. This is Phase 8 §5's fix: `RAGAgent.last_trace` was an
#: instance attribute, so two concurrent SSE streams saw an interleaving of both
#: runs. A contextvar is per-task, which is per-request.
_trace: ContextVar[list[str]] = ContextVar("trace", default=[])


def current_principal() -> Principal:
    return _principal.get()


def set_principal(principal: Principal) -> None:
    _principal.set(principal)


def current_trace() -> list[str]:
    return list(_trace.get())


def set_trace(trace: list[str]) -> None:
    _trace.set(list(trace))


def decode_principal(token: str) -> Principal:
    """Verify a JWT and build a principal from it.

    Signature verification is not optional and there is no "trust the claims"
    fallback: an unverified JWT is a client-supplied role list, which is the same as
    no access control at all.

    Raises:
        AuthorizationError: missing, malformed, expired, or badly signed.
    """
    if not settings.JWT_SECRET:
        raise AuthorizationError("JWT_SECRET is not configured; refusing to authenticate")

    try:
        claims = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise AuthorizationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        # The reason is deliberately not echoed to the client: "invalid signature"
        # versus "malformed payload" is a probing oracle.
        logger.warning("Rejected a token", extra={"error": type(exc).__name__})
        raise AuthorizationError("Invalid token") from exc

    roles = claims.get("roles", [])
    if not isinstance(roles, list):
        raise AuthorizationError("Token roles claim must be a list")

    return Principal(
        subject=str(claims.get("sub", "unknown")),
        roles=frozenset(str(r) for r in roles),
        tenant=claims.get("tenant"),
    )


async def principal_dependency(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """FastAPI dependency: request → principal.

    In development with `REQUIRE_AUTH=false`, returns the admin `DEV_PRINCIPAL` so
    the system is usable without issuing tokens. That bypass is loud — it logs on
    every request and `validate_runtime()` warns about it in production — because a
    development bypass that reaches production silently is how this whole phase
    becomes theatre.
    """
    if not settings.REQUIRE_AUTH:
        logger.debug("Auth disabled; using the development principal")
        set_principal(DEV_PRINCIPAL)
        return DEV_PRINCIPAL

    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthorizationError("Missing bearer token")

    principal = decode_principal(authorization.split(" ", 1)[1])
    set_principal(principal)
    logger.info("Authenticated", extra=principal.for_log())
    return principal


PrincipalDep = Annotated[Principal, Depends(principal_dependency)]


def issue_dev_token(subject: str, roles: list[str], ttl_seconds: int = 3600) -> str:
    """Mint a token for local testing and for Phase 10's tests.

    Deliberately in this module rather than a script: it is the only place that
    knows the claim shape, and a second implementation would drift from the decoder.
    """
    if not settings.JWT_SECRET:
        raise AuthorizationError("JWT_SECRET is not configured")
    now = int(time.time())
    return jwt.encode(
        {"sub": subject, "roles": roles, "iat": now, "exp": now + ttl_seconds},
        settings.JWT_SECRET,
        algorithm="HS256",
    )
```

### Settings

```python
    REQUIRE_AUTH: bool = Field(default=False)
    JWT_SECRET: str = Field(default="")
    ACL_ENABLED: bool = Field(default=True)
```

and in `validate_runtime()`:

```python
        if self.is_production and not self.REQUIRE_AUTH:
            warnings.append(
                "REQUIRE_AUTH is false in production: every request runs as an "
                "administrator and the ACL is inert."
            )
        if self.REQUIRE_AUTH and len(self.JWT_SECRET) < 32:
            warnings.append("JWT_SECRET is short; use at least 32 random characters.")
        if not self.ACL_ENABLED and self.is_production:
            warnings.append("ACL_ENABLED is false in production: no permission filtering.")
```

Add `pyjwt` to `pyproject.toml`.

---

## 7. File 5 — the Part I modifications

Four changes, each small, each closing a hole that would otherwise make the rest of this phase
decorative.

### The cache — the leak that survives a correct filter

```python
# src/cache/redis_cache.py — inside cache_scope()

def cache_scope() -> dict[str, object]:
    from src.security.middleware import current_principal

    principal = current_principal()
    return {
        ...,
        # WITHOUT THIS, THE ACL DOES NOT WORK. Two users ask "what is the CEO's
        # severance?"; the key is the question, so the second user is served the
        # first user's answer AND their source documents — through a store filter
        # that worked perfectly. Phase 6 already partitions both cache layers by
        # this dict, so one line closes it in the exact and semantic caches at once.
        "roles": sorted(principal.visible_roles()),
    }
```

Note what makes this cheap: Phase 6's `cache_scope()` was extracted specifically so the exact key and
the semantic index could not disagree about identity. Adding the principal to one function covers
both.

### The agent — per-request trace

```python
# src/graph/builder.py — in RAGAgent.answer()

from src.security.middleware import set_trace

        async for state in self.graph.astream(state, config, stream_mode="values"):
            final = state
            set_trace(state.get("trace", []))     # was: self.last_trace = ...
```

and Phase 8's SSE endpoint reads `current_trace()` instead of `service.agent.last_trace`. That makes
`tests/e2e/test_api.py::test_concurrent_streams_do_not_interleave` a real test rather than a
documented boundary.

### The service — a principal-scoped view

```python
# src/app.py — on RAGService

    def for_principal(self, principal: Principal) -> "RAGService":
        """A view of this service scoped to one principal.

        Returns a shallow copy with the store wrapped in `SecureStore` and a fresh
        agent bound to it. Cheap — no models are reloaded, only the two objects that
        must not be shared. The alternative, passing a principal down through
        `answer` → `RAGAgent` → `RetrievalPipeline` → `HybridRetriever` → store, is
        five signature changes and five opportunities to forget.
        """
        if not settings.ACL_ENABLED:
            return self

        secure = SecureStore(self.store, principal)
        pipeline = RetrievalPipeline(embedder=self.embedder, store=secure, llm=self.llm)
        return replace(self, store=secure,
                       agent=RAGAgent(llm=self.llm, pipeline=pipeline))
```

`RAGService` becomes `@dataclass(frozen=False)` with `replace` from `dataclasses`. The
cross-encoder inside the new pipeline comes from the same `@lru_cache`d model, so this costs a few
object allocations rather than a model load.

### The API — one dependency, one line per route

```python
# src/api/routes/query.py

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, service: ServiceDep,
                principal: PrincipalDep) -> QueryResponse:
    answer = await service.for_principal(principal).answer(
        request.question, top_k=request.top_k, filters=request.to_filters()
    )
    return QueryResponse.from_answer(answer)
```

`/admin/*` gains an admin check, and `AuthorizationError` already carries `status_code = 403`, so
Phase 8's error handler maps it with no changes.

---

## 8. The Theory: why pre-search filtering is the only correct place

Worth being precise, because "filter the results" is the intuitive design and it is wrong in three
distinct ways.

**Correctness.** HNSW returns approximate nearest neighbours from the whole index. Retrieve 20 and
discard 17 the user cannot see, and you have 3 results — not the top 3 *permitted* results, which
might have been at ranks 40, 60, and 200. The user gets an answer built from whatever fragments
survived, and there is no error to notice. **Post-filtering does not restrict a result set, it
corrupts it.** Qdrant's filtered search instead applies the condition *during* graph traversal, so
you get the true top 20 among permitted documents.

**Performance.** To reliably obtain 20 permitted results by post-filtering, you must over-fetch by
whatever your worst-case exclusion rate is — 10×, 50× — and you cannot know that rate in advance
because it varies per principal. Filtered search costs one index lookup.

**Leakage.** The scores, the candidate counts, and the diagnostics were all computed over documents
the principal cannot see. Phase 8's `/admin/stats`, Phase 9's trace, Phase 7's `total_candidates`, and
any error message naming a collection or a contract can each expose the existence and shape of hidden
material. **Existence is information.** Salary bands are inferable from the fact that a document
matched.

The general principle, which recurs through this project: **a constraint enforced by the component
that owns the data is a guarantee; a constraint applied by the caller afterwards is a convention.**
It is the same reasoning as Phase 2 giving the storage consumer the commit decision, and Phase 4
putting the `chunk_level` filter inside the retriever.

---

## 9. Verification (deferred)

Add `tests/unit/test_security.py` and extend Phase 10's suite. `scripts/verify_phase11.py` mirrors it
for the corpus machine.

```python
"""Phase 11 verification. No services required for the unit portion."""
import pytest

from src.core.exceptions import AuthorizationError
from src.security.acl import ACL_FIELD, AclClassifier, apply_acl
from src.security.filters import SecureStore, acl_filter, assert_permitted
from src.security.middleware import decode_principal, issue_dev_token
from src.security.principal import ADMIN_ROLE, ANONYMOUS, Principal
from tests.fakes import FakeStore, build_corpus, scored

PARTNER = Principal(subject="p", roles=frozenset({"legal_partner"}))
ASSOCIATE = Principal(subject="a", roles=frozenset({"associate"}))
ADMIN = Principal(subject="root", roles=frozenset({ADMIN_ROLE}))


class TestPrincipal:
    def test_deny_by_default(self) -> None:
        assert ANONYMOUS.is_anonymous
        assert not ANONYMOUS.can_see(["public"]), (
            "an anonymous principal must see nothing, including public documents"
        )

    def test_admin_sees_everything(self) -> None:
        assert ADMIN.can_see(["board", "legal_partner"])

    def test_your_stated_example(self) -> None:
        # "User A (Junior Associate) should NOT see executive compensation
        # contracts, while User B (Partner) can see everything."
        executive = ["legal_partner", "board"]
        assert not ASSOCIATE.can_see(executive)
        assert PARTNER.can_see(executive)

    def test_principal_is_immutable(self) -> None:
        with pytest.raises(Exception):
            PARTNER.roles = frozenset({ADMIN_ROLE})


class TestFilter:
    def test_admin_gets_no_constraint(self) -> None:
        assert acl_filter(ADMIN) == {}

    def test_anonymous_matches_nothing(self) -> None:
        # The critical direction: an empty role set must produce a filter that
        # returns NOTHING, never an absent filter that returns everything.
        assert acl_filter(ANONYMOUS)[ACL_FIELD] == ["__no_access__"]

    async def test_caller_cannot_override_the_acl(self) -> None:
        captured: dict = {}

        class Spy(FakeStore):
            async def hybrid_search(self, dense, sparse, limit=20, filters=None):
                captured.update(filters or {})
                return []

        store = SecureStore(Spy(), ASSOCIATE)
        # A caller passing its own allowed_roles must not widen its permissions.
        await store.hybrid_search([0.1], {"indices": [], "values": []},
                                  filters={ACL_FIELD: ["board"]})
        assert captured[ACL_FIELD] == sorted(ASSOCIATE.visible_roles())


class TestSecureStore:
    async def test_writes_require_admin(self) -> None:
        store = SecureStore(FakeStore(), PARTNER)
        with pytest.raises(AuthorizationError):
            await store.delete_by_doc_ids(["d"])

    async def test_parent_fetch_is_filtered(self) -> None:
        # The subtle hole: parent substitution fetches by ID, and an ID lookup has
        # no filter.
        inner = FakeStore()
        for chunk in build_corpus():
            inner.points[chunk.chunk_id] = chunk.model_copy(
                update={"allowed_roles": ["board"]}
            )
        store = SecureStore(inner, ASSOCIATE)
        assert await store.fetch_by_ids(list(inner.points)) == []

    def test_post_filter_raises_rather_than_cleaning_up(self) -> None:
        forbidden = scored([c.model_copy(update={"allowed_roles": ["board"]})
                            for c in build_corpus()[:1]])
        with pytest.raises(AuthorizationError):
            assert_permitted(forbidden, ASSOCIATE)


class TestClassifier:
    def test_path_rules_win_over_content(self) -> None:
        from src.core.models import Document

        doc = Document(doc_id="d", source_path="data/contracts/executive/ceo.txt",
                       file_name="ceo.txt", content="A supply agreement." * 10,
                       content_hash="h")
        assert "board" in AclClassifier().classify(doc)

    def test_acl_reaches_every_chunk(self) -> None:
        stamped = apply_acl(build_corpus(), ["legal_partner"])
        assert all(c.allowed_roles == ["legal_partner"] for c in stamped)

    def test_payload_round_trip(self) -> None:
        # A permission field that does not survive to_payload/from_payload is an ACL
        # that evaporates the moment it is written.
        from src.core.models import Chunk

        stamped = apply_acl(build_corpus()[:1], ["board"])[0]
        assert Chunk.from_payload(stamped.to_payload()).allowed_roles == ["board"]


class TestTokens:
    def test_round_trip(self, monkeypatch) -> None:
        from config.settings import settings

        monkeypatch.setattr(settings, "JWT_SECRET", "x" * 40)
        token = issue_dev_token("alice", ["associate"])
        assert decode_principal(token).roles == frozenset({"associate"})

    def test_tampered_token_rejected(self, monkeypatch) -> None:
        from config.settings import settings

        monkeypatch.setattr(settings, "JWT_SECRET", "x" * 40)
        token = issue_dev_token("alice", ["associate"])
        monkeypatch.setattr(settings, "JWT_SECRET", "y" * 40)
        with pytest.raises(AuthorizationError):
            decode_principal(token)


@pytest.mark.integration
async def test_cache_does_not_cross_principals(redis_cache) -> None:
    """The leak that survives a perfect store filter."""
    from src.cache.redis_cache import make_cache_key
    from src.security.middleware import set_principal
    from tests.fakes import answer

    question = "what is the CEO's severance?"

    set_principal(PARTNER)
    partner_key = make_cache_key(question, top_k=5, filters=None, prompt_version="v1.0")
    await redis_cache.set(partner_key, answer(answer="Two years' salary."))

    set_principal(ASSOCIATE)
    associate_key = make_cache_key(question, top_k=5, filters=None, prompt_version="v1.0")
    assert associate_key != partner_key, "the cache key must include the principal"
    assert await redis_cache.get(associate_key) is None, (
        "an associate received a partner's cached answer and sources"
    )
```

### Check by hand, after a real ingestion

1. `classifier.report_coverage()` — read the `unclassified` count. If it is most of the corpus, the
   rules do not fit your directory layout and the ACL is close to inert.
2. Issue two tokens, ask the same question, and diff the source lists.
3. Ask the associate's question **twice**. The second must not be a cache hit of the partner's.

---

## 10. What Phase 11 Bought You

**Your stated failure, fixed as stated.** `allowed_roles` on chunk metadata, injected into the Qdrant
query as a pre-search payload filter. The junior associate's search never traverses the executive
compensation vectors.

**A filter that cannot be forgotten.** `SecureStore` has no unfiltered search method, so the five
existing call sites and every future one inherit the constraint without knowing this phase exists.

**The cache hole closed.** One line in `cache_scope()`, because Phase 6 extracted that function
precisely so the exact and semantic layers could not disagree about identity.

**Phase 8's concurrency bug fixed properly**, with the per-request contextvar that this phase was
always going to introduce — and the test Phase 10 wrote as a documented boundary becomes a real one.

**A defensible answer to "how do you know it works?"** — a post-retrieval assertion that raises rather
than quietly cleaning up, so a broken pre-filter is an incident rather than a silence.

### What is deliberately not here

Everything in §2. The one to revisit first is per-user (not per-role) permissions, and only if a real
deployment needs document-level sharing — it changes the filter's cardinality and needs a membership
store, which is a phase of its own.

The honest weak point is `DEFAULT_ROLES`. Unclassified documents are visible to all authenticated
staff, which is a usability choice with a confidentiality cost, argued in §4. It is one constant, and
whether it should be `[]` is a question about your corpus rather than about this code.

---

## Next

**Phases 14 and 15, in parallel** — they are the two Part II phases that touch nothing shared. Phase
14 (layout-aware PDF and vision OCR) extends Phase 2's loaders only; Phase 15 (embedding drift and
shadow indexing) extends Phase 3's store only. Then 12, 13, and 16, which all add nodes to Phase 5's
graph and should be written with one consistent view of that surface.
