# Phase 9 — CLI and Web Dashboard

> **Prerequisite:** Phase 8. Both interfaces are clients of that API and contain no RAG logic.
>
> **Budget: ~640 lines across 6 files** (~390 Python, ~250 HTML/CSS/JS), on budget. §2 lists the cuts.
>
> **This file replaces the previous `Phase9_CLI_Web_UI.md`.**

---

## 1. What Makes This Phase Hard

Nothing about this phase is algorithmically hard, which is exactly the trap: it is where scope goes
to die. A dashboard can absorb unlimited effort and teach nothing about RAG.

So the discipline is different here. The question for every feature is not "would this be nice" but
**"does this let me see something about the system I cannot currently see?"** Three things qualify.

**The trace is the product.** Six phases built a self-correcting agent, and from the outside a good
answer and a lucky answer look identical. Showing `router → retrieval → generation → grading:fail →
rewrite → grading:pass` is the only way anyone — including you, debugging — sees the machine work.

**Grounding needs to be visible, not asserted.** An answer with a `[SOURCE 2]` marker that nobody can
click is a claim. The interface's job is making provenance one keystroke away, because that is what a
lawyer would actually check.

**Status is not decoration.** `UNVERIFIED` and `LOW_CONFIDENCE` exist because Phase 5 and 6 fought to
keep them distinct. An interface that renders every answer identically discards that work.

### The files

| # | File | Lines | Role |
| :-- | :--- | ---: | :--- |
| 1 | `src/cli/client.py` | 80 | Thin async HTTP client for the Phase 8 API |
| 2 | `src/cli/render.py` | 120 | Rich rendering: answer, sources, trace, status |
| 3 | `src/cli/main.py` | 190 | Typer commands and the interactive REPL |
| 4 | `src/web/index.html` | 90 | Structure |
| 5 | `src/web/styles.css` | 80 | Presentation |
| 6 | `src/web/app.js` | 80 | SSE consumption and rendering |

```text
src/cli/
├── __init__.py
├── client.py
├── render.py
└── main.py

src/web/
├── index.html
├── styles.css
└── app.js
```

---

## 2. What Was Cut

**A React/Vue build.** Three files with no build step, no `node_modules`, and no bundler. The
dashboard is a diagnostic tool for one user; a framework would be more lines of tooling than
application.

**Conversation history and multi-turn chat.** The agent has no conversational memory — Phase 5's
state is per-question by design, and Phase 8's threads are per-question. A chat UI would imply a
capability that does not exist, which is worse than not having it.

**Authentication in the UI.** Nothing to authenticate against until Phase 11.

**A results table with filtering, sorting, and export.** The CLI can print JSON; `jq` exists.

**Syntax-highlighted diffs between runs, and an evaluation dashboard.** Phase 7 writes JSON files. A
chart of two numbers is not worth 200 lines.

That is ~500 lines declined, and the constraint is what makes the trace panel affordable.

---

## 3. File 1 — `src/cli/client.py`

### Design Decision

**The CLI talks to the API over HTTP, not to `RAGService` directly.** Importing the service would be
faster to write and would load the ONNX models into the CLI process — five seconds of startup per
command, and a second copy of everything if the server is also running. More importantly, going
through HTTP means the CLI exercises the same path a real client does, so it finds API bugs.

```python
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class RAGClient:
    """Async client for the Phase 8 API. Thin on purpose."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        # Long default: a cold query is retrieval plus up to four LLM calls, and a
        # 30-second timeout would abort perfectly healthy requests on first run.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def query(self, question: str, top_k: int | None = None,
                    year: int | None = None) -> dict[str, Any]:
        """Ask a question. Raises `RAGAPIError` with the server's own message."""
        response = await self._client.post(
            "/query",
            json={"question": question, "top_k": top_k, "year": year},
        )
        if response.status_code >= 400:
            raise RAGAPIError.from_response(response)
        return response.json()

    async def stream(self, question: str, top_k: int | None = None,
                     year: int | None = None) -> AsyncIterator[tuple[str, dict]]:
        """Yield `(event_name, payload)` from the SSE stream.

        Hand-parsed rather than pulling in an SSE client library: the format is
        `event:` and `data:` lines separated by blank lines, and one dependency for
        fifteen lines of parsing is not a trade worth making.
        """
        async with self._client.stream(
            "POST", "/query/stream",
            json={"question": question, "top_k": top_k, "year": year},
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise RAGAPIError.from_response(response)

            event = "message"
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                    if payload:
                        yield event, json.loads(payload)
                    event = "message"

    async def health(self) -> dict[str, Any]:
        return (await self._client.get("/ready")).json()

    async def stats(self) -> dict[str, Any]:
        return (await self._client.get("/admin/stats")).json()

    async def close(self) -> None:
        await self._client.aclose()


class RAGAPIError(Exception):
    """An error the server reported, carrying its structured detail."""

    def __init__(self, status: int, error: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message
        self.retryable = retryable

    @classmethod
    def from_response(cls, response: httpx.Response) -> "RAGAPIError":
        """Rebuild from the envelope `errors.py` produces.

        Falls back to the raw body when the response is not ours — a 502 from a
        proxy is not JSON, and a `JSONDecodeError` here would replace a useful
        status code with a confusing parse error.
        """
        try:
            body = response.json()
            return cls(response.status_code, body.get("error", "Error"),
                       body.get("message", ""), body.get("retryable", False))
        except Exception:
            return cls(response.status_code, "HTTPError", response.text[:200])
```

---

## 4. File 2 — `src/cli/render.py`

### Design Decision

**Rendering is separated from the commands.** `main.py` decides what to ask; this decides how it
looks. That split is what lets the REPL and the one-shot command share every panel.

**Status drives colour, and the four cases look different.** This is the whole reason Phase 5 fought
for `AnswerStatus`: a `LOW_CONFIDENCE` answer that renders like an `ANSWERED` one has thrown away the
grading pipeline.

```python
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

#: (colour, label) per status. The distinction between "we checked and found
#: problems" (yellow) and "we could not check" (magenta) is the one users most need
#: and the one an interface most easily collapses.
_STATUS = {
    "answered": ("green", "✓ verified"),
    "low_confidence": ("yellow", "⚠ unverified claims"),
    "unverified": ("magenta", "⚠ not checked"),
    "no_match": ("blue", "no matching passages"),
    "unsupported": ("blue", "unsupported question"),
}

_STAGE_LABELS = {
    "router": "understanding",
    "retrieval": "searching",
    "generation": "drafting",
    "grading": "verifying",
    "rewrite": "refining",
}


def render_answer(payload: dict) -> None:
    """The answer panel, coloured by status and annotated with warnings."""
    status = payload.get("status", "answered")
    colour, label = _STATUS.get(status, ("white", status))

    body: list = [Text(payload["answer"])]

    for warning in payload.get("warnings", []):
        body.append(Text(f"\n! {warning}", style="yellow"))

    grading = payload.get("grading")
    if grading:
        body.append(Text(
            f"\ngrounded={grading['grounded']} relevant={grading['relevant']} "
            f"confidence={grading['confidence']:.2f}",
            style="dim",
        ))

    subtitle = f"{label}  ·  {payload.get('latency_ms', 0):.0f}ms"
    if payload.get("cache_hit"):
        subtitle += "  ·  cached"
    if payload.get("retry_count"):
        subtitle += f"  ·  {payload['retry_count']} retry"

    console.print(Panel(Group(*body), title="Answer", subtitle=subtitle,
                        border_style=colour))


def render_sources(payload: dict) -> None:
    """Sources as a table, numbered to match the [SOURCE N] markers.

    The numbering is the point. An answer citing [SOURCE 2] is only checkable if
    row 2 here is the same source the model saw — that correspondence is built in
    Phase 5's `format_context` and this is where a user cashes it in.
    """
    sources = payload.get("sources", [])
    if not sources:
        return

    cited = set(payload.get("citations", []))
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("#", width=3)
    table.add_column("Contract", max_width=28)
    table.add_column("Section", max_width=22)
    table.add_column("Excerpt", overflow="fold")

    for index, source in enumerate(sources, start=1):
        marker = f"[bold]{index}[/bold]" if index in cited else f"[dim]{index}[/dim]"
        table.add_row(
            marker,
            source["contract_name"],
            source["section_title"],
            source["excerpt"][:200].replace("\n", " "),
        )

    uncited = len(sources) - len(cited)
    caption = f"{len(cited)} cited" + (f", {uncited} retrieved but unused" if uncited else "")
    console.print(Panel(table, title="Sources", subtitle=caption, border_style="blue"))


def render_trace(trace: list[str]) -> None:
    """The agent's path. The single most informative thing on screen.

    A retry is invisible in the answer text and obvious here — and 'retrieval →
    generation → grading:fail → rewrite → retrieval → grading:pass' is the clearest
    demonstration of what this system does that exists anywhere in the project.
    """
    if not trace:
        return
    steps = " → ".join(_STAGE_LABELS.get(step.split(":")[0], step) for step in trace)
    console.print(f"[dim]{steps}[/dim]\n")


def render_error(error: Exception) -> None:
    console.print(Panel(str(error), title="Error", border_style="red"))
```

---

## 5. File 3 — `src/cli/main.py`

### Design Decision

**Typer for commands, and an interactive REPL as the default.** Ninety percent of use is asking
several questions in a row, and re-running a command that spawns a process each time is friction for
no reason.

**Stream by default in the REPL.** A cold query takes two to four seconds, and progress makes that
feel responsive. `--no-stream` exists for scripting.

```python
import asyncio
import json

import httpx
import typer
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from src.cli.client import RAGAPIError, RAGClient
from src.cli.render import console, render_answer, render_error, render_sources, render_trace

app = typer.Typer(help="Legal Contract RAG — command line interface.", no_args_is_help=False)

_URL_OPTION = typer.Option("http://127.0.0.1:8000", "--url", help="API base URL.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    top_k: int = typer.Option(None, "--top-k"),
    year: int = typer.Option(None, "--year", help="Restrict to a filing year."),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    as_json: bool = typer.Option(False, "--json", help="Raw JSON, for scripting."),
    url: str = _URL_OPTION,
) -> None:
    """Ask one question and exit."""
    asyncio.run(_ask(url, question, top_k, year, stream and not as_json, as_json))


@app.command()
def chat(url: str = _URL_OPTION) -> None:
    """Interactive session. Ctrl-C or 'exit' to leave."""
    asyncio.run(_chat(url))


@app.command()
def status(url: str = _URL_OPTION) -> None:
    """Show what the server is running and whether it is ready."""
    asyncio.run(_status(url))


async def _ask(url: str, question: str, top_k: int | None, year: int | None,
               stream: bool, as_json: bool) -> None:
    client = RAGClient(url)
    try:
        if as_json:
            print(json.dumps(await client.query(question, top_k, year), indent=2))
            return
        payload = await (
            _stream_with_progress(client, question, top_k, year)
            if stream else client.query(question, top_k, year)
        )
        if payload:
            _show(payload)
    except RAGAPIError as exc:
        render_error(exc)
        # A non-zero exit matters: this gets used in shell pipelines, and a failed
        # query that exits 0 breaks every `&&` chain silently.
        raise typer.Exit(code=1)
    except httpx.ConnectError:
        render_error(Exception(f"Cannot reach the API at {url}. Is uvicorn running?"))
        raise typer.Exit(code=1)
    finally:
        await client.close()


async def _stream_with_progress(client: RAGClient, question: str, top_k: int | None,
                                year: int | None) -> dict | None:
    """Consume the SSE stream, showing a live spinner per stage.

    The `answer` event carries the finished, graded answer. An `error` event is a
    failure that happened AFTER the 200 was sent, which is why it arrives as data
    rather than a status code.
    """
    trace: list[str] = []
    result: dict | None = None

    with Live(Spinner("dots", text="starting"), console=console, transient=True) as live:
        async for event, payload in client.stream(question, top_k, year):
            if event == "progress":
                trace.append(payload["stage"])
                live.update(Spinner("dots", text=Text(payload["label"], style="cyan")))
            elif event == "answer":
                result = payload
            elif event == "error":
                raise RAGAPIError(500, payload.get("error", "Error"),
                                  payload.get("message", ""))

    render_trace(trace)
    return result


def _show(payload: dict) -> None:
    render_answer(payload)
    render_sources(payload)


async def _chat(url: str) -> None:
    client = RAGClient(url)
    console.print("[bold]Legal Contract RAG[/bold]  ·  ask a question, or 'exit'\n")

    try:
        ready = await client.health()
        if not ready.get("ready"):
            console.print(f"[yellow]Server not ready: {ready.get('reason')}[/yellow]\n")
    except Exception:
        console.print(f"[red]Cannot reach {url}. Start it with:[/red]")
        console.print("  uvicorn src.api.app:app --host 127.0.0.1 --port 8000\n")
        await client.close()
        return

    try:
        while True:
            try:
                question = console.input("[bold cyan]?[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", ":q"}:
                break

            try:
                payload = await _stream_with_progress(client, question, None, None)
                if payload:
                    _show(payload)
            except RAGAPIError as exc:
                # A failed question must not end the session — the next one may
                # work, and being dropped back to the shell for one 503 is hostile.
                render_error(exc)
            console.print()
    finally:
        await client.close()
        console.print("[dim]bye[/dim]")


async def _status(url: str) -> None:
    client = RAGClient(url)
    try:
        ready = await client.health()
        stats = await client.stats()
        console.print(f"ready:      {ready.get('ready')}  ({ready.get('points', 0):,} points)")
        console.print(f"collection: {stats['collection']}")
        console.print(f"models:     {stats['models']}")
        console.print(f"flags:      {stats['flags']}")
        console.print(f"cache:      {stats['cache']}")
    except Exception as exc:
        render_error(exc)
        raise typer.Exit(code=1)
    finally:
        await client.close()


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """With no subcommand, start the REPL — the most common use."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_chat("http://127.0.0.1:8000"))


if __name__ == "__main__":
    app()
```

Add `typer`, `rich`, and `httpx` to `pyproject.toml`, plus a console script:

```toml
[project.scripts]
rag = "src.cli.main:app"
```

---

## 6. Files 4–6 — the web dashboard

### Design Decision

**Three static files, no build step.** Open `index.html` in a browser, or serve it from any static
server. Adding a bundler would mean more configuration than application, for a tool with one user.

**The trace panel is the reason this exists.** A browser is better than a terminal at showing a
process unfolding, and this is the artefact to demo.

### `src/web/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Legal Contract RAG</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header>
    <h1>Legal Contract RAG</h1>
    <span id="status-pill" class="pill pill-muted">checking…</span>
  </header>

  <main>
    <form id="query-form" autocomplete="off">
      <input id="question" type="text" placeholder="What notice is required to terminate early?"
             maxlength="1000" required />
      <button id="submit" type="submit">Ask</button>
    </form>

    <!-- The trace. Six phases of machinery are invisible without this. -->
    <section id="trace" class="trace" hidden></section>

    <section id="answer" class="card" hidden>
      <div class="card-header">
        <h2>Answer</h2>
        <span id="answer-status" class="pill"></span>
      </div>
      <div id="answer-body"></div>
      <ul id="warnings" class="warnings"></ul>
      <div id="meta" class="meta"></div>
    </section>

    <section id="sources" class="card" hidden>
      <div class="card-header"><h2>Sources</h2><span id="source-count" class="meta"></span></div>
      <ol id="source-list"></ol>
    </section>

    <section id="error" class="card card-error" hidden></section>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

### `src/web/styles.css`

```css
:root {
  --bg: #0f1117; --panel: #171a21; --line: #262b36;
  --text: #e6e8ec; --muted: #8b93a7; --accent: #5b9dff;
  --ok: #3fb950; --warn: #d29922; --unverified: #a371f7; --bad: #f85149;
}

* { box-sizing: border-box; }

body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 -apple-system, "Segoe UI", system-ui, sans-serif;
}

header {
  display: flex; align-items: center; gap: 12px;
  padding: 16px 24px; border-bottom: 1px solid var(--line);
}
header h1 { font-size: 16px; margin: 0; font-weight: 600; }

main { max-width: 860px; margin: 0 auto; padding: 24px; }

#query-form { display: flex; gap: 8px; margin-bottom: 20px; }
#question {
  flex: 1; padding: 12px 14px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text);
}
#question:focus { outline: none; border-color: var(--accent); }
button {
  padding: 12px 20px; border: 0; border-radius: 8px;
  background: var(--accent); color: #fff; font-weight: 600; cursor: pointer;
}
button:disabled { opacity: .5; cursor: not-allowed; }

.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 18px; margin-bottom: 16px;
}
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.card-header h2 { font-size: 13px; text-transform: uppercase; color: var(--muted); margin: 0; }
.card-error { border-color: var(--bad); color: var(--bad); }

/* Status colours must differ: "checked and flawed" and "not checked" are
   different things to tell someone relying on a contract term. */
.pill { font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid; }
.pill-answered { color: var(--ok); border-color: var(--ok); }
.pill-low_confidence { color: var(--warn); border-color: var(--warn); }
.pill-unverified { color: var(--unverified); border-color: var(--unverified); }
.pill-no_match, .pill-unsupported, .pill-muted { color: var(--muted); border-color: var(--line); }

.trace { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.step {
  font-size: 12px; padding: 4px 10px; border-radius: 6px;
  background: var(--panel); border: 1px solid var(--line); color: var(--muted);
}
.step-active { color: var(--accent); border-color: var(--accent); }
.step-fail { color: var(--warn); border-color: var(--warn); }

.warnings { margin: 12px 0 0; padding-left: 18px; color: var(--warn); font-size: 13px; }
.meta { color: var(--muted); font-size: 12px; margin-top: 12px; }

#source-list { margin: 0; padding-left: 22px; }
#source-list li { margin-bottom: 12px; }
.source-title { font-weight: 600; }
.source-cited { color: var(--accent); }
.source-excerpt { color: var(--muted); font-size: 13px; }
```

### `src/web/app.js`

```javascript
const API = "http://127.0.0.1:8000";

const el = (id) => document.getElementById(id);
const STAGES = {
  router: "understanding", retrieval: "searching", generation: "drafting",
  grading: "verifying", rewrite: "refining", no_context: "no matches",
};

async function checkReady() {
  const pill = el("status-pill");
  try {
    const ready = await (await fetch(`${API}/ready`)).json();
    pill.textContent = ready.ready ? `${ready.points.toLocaleString()} passages` : ready.reason;
    pill.className = `pill ${ready.ready ? "pill-answered" : "pill-low_confidence"}`;
  } catch {
    pill.textContent = "API unreachable";
    pill.className = "pill pill-unverified";
  }
}

function addStep(stage) {
  const trace = el("trace");
  trace.hidden = false;
  document.querySelectorAll(".step-active").forEach((s) => s.classList.remove("step-active"));
  const step = document.createElement("span");
  // A failed grade is the interesting event — it is the moment the system decides
  // to correct itself, and it should be visible rather than blended in.
  step.className = `step step-active${stage.includes("fail") ? " step-fail" : ""}`;
  step.textContent = STAGES[stage] || stage;
  trace.appendChild(step);
}

function showAnswer(data) {
  el("answer").hidden = false;
  el("answer-body").textContent = data.answer;

  const pill = el("answer-status");
  pill.textContent = data.status.replace("_", " ");
  pill.className = `pill pill-${data.status}`;

  const warnings = el("warnings");
  warnings.innerHTML = "";
  (data.warnings || []).forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    warnings.appendChild(item);
  });

  const bits = [`${Math.round(data.latency_ms)}ms`];
  if (data.cache_hit) bits.push("cached");
  if (data.retry_count) bits.push(`${data.retry_count} retry`);
  if (data.grading) bits.push(`confidence ${data.grading.confidence.toFixed(2)}`);
  el("meta").textContent = bits.join("  ·  ");

  const cited = new Set(data.citations || []);
  const list = el("source-list");
  list.innerHTML = "";
  (data.sources || []).forEach((source, index) => {
    const item = document.createElement("li");
    const isCited = cited.has(index + 1);
    item.innerHTML =
      `<div class="source-title ${isCited ? "source-cited" : ""}">` +
      `${escapeHtml(source.contract_name)} — ${escapeHtml(source.section_title)}` +
      `${isCited ? " ✓" : ""}</div>` +
      `<div class="source-excerpt">${escapeHtml(source.excerpt)}</div>`;
    list.appendChild(item);
  });
  el("sources").hidden = (data.sources || []).length === 0;
  el("source-count").textContent = `${cited.size} of ${(data.sources || []).length} cited`;
}

// Source text comes from contracts, which are untrusted input all the way to the
// browser. innerHTML with unescaped document text is stored XSS.
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function showError(message) {
  el("error").hidden = false;
  el("error").textContent = message;
}

async function ask(question) {
  ["answer", "sources", "error"].forEach((id) => (el(id).hidden = true));
  el("trace").innerHTML = "";
  el("submit").disabled = true;

  try {
    // fetch + ReadableStream rather than EventSource: EventSource cannot issue a
    // POST, and the question belongs in a body rather than a query string.
    const response = await fetch(`${API}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      // SSE frames are separated by a blank line. Keep the trailing partial frame
      // in the buffer — chunk boundaries do not respect message boundaries.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        const event = (frame.match(/^event:\s*(.*)$/m) || [])[1] || "message";
        const raw = (frame.match(/^data:\s*(.*)$/m) || [])[1];
        if (!raw) continue;
        const payload = JSON.parse(raw);
        if (event === "progress") addStep(payload.stage);
        else if (event === "answer") showAnswer(payload);
        else if (event === "error") showError(payload.message || "Request failed");
      }
    }
  } catch (error) {
    showError(`${error.message}. Is the API running on ${API}?`);
  } finally {
    el("submit").disabled = false;
    document.querySelectorAll(".step-active").forEach((s) => s.classList.remove("step-active"));
  }
}

el("query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el("question").value.trim();
  if (question) ask(question);
});

checkReady();
```

### The XSS point, because it is the one real security issue in this phase

`escapeHtml` is not boilerplate. Source excerpts are **contract text from the corpus**, and the
corpus is 650,000 documents scraped from EDGAR that nobody has reviewed. A filing containing
`<img src=x onerror=...>` is entirely possible, and rendering it with `innerHTML` executes it.

This is the same threat model as Phase 6's indirect prompt injection, one layer out: untrusted
document content flowing into an interpreter. There it was an LLM reading instructions; here it is a
browser executing script. **The corpus is untrusted at every boundary it crosses**, and there are
three of them now — the prompt, the API response, and the DOM.

---

## 7. Verification (deferred)

Save as `scripts/verify_phase9.py`. The CLI's rendering and client parsing are testable without a
server; the dashboard is checked by hand.

```python
"""Phase 9 verification. No server required."""
import sys

from src.cli.client import RAGAPIError
from src.cli.render import _STATUS, render_answer, render_sources, render_trace

SAMPLE = {
    "question": "what notice is required?",
    "answer": "Ninety (90) days written notice [SOURCE 1].",
    "status": "low_confidence",
    "sources": [
        {"chunk_id": "a", "contract_name": "Acme Agreement",
         "section_title": "Termination", "year": 2003,
         "excerpt": "Either party may terminate...", "score": 0.91},
        {"chunk_id": "b", "contract_name": "Beta Agreement",
         "section_title": "Notices", "year": 2004,
         "excerpt": "All notices shall be...", "score": 0.62},
    ],
    "citations": [1],
    "grading": {"verified": True, "grounded": False, "relevant": True, "confidence": 0.8},
    "warnings": ["1 figure stated without a citation."],
    "retry_count": 2, "cache_hit": False, "latency_ms": 2841.0,
}


def check_status_coverage() -> None:
    """Every AnswerStatus must render distinctly."""
    from src.core.models import AnswerStatus

    for status in AnswerStatus:
        assert status.value in _STATUS, f"no rendering defined for {status.value}"
    colours = {colour for colour, _ in _STATUS.values()}
    assert _STATUS["low_confidence"][0] != _STATUS["unverified"][0], (
        "'checked and flawed' and 'not checked' must not look the same — Phases 5 "
        "and 6 kept them distinct precisely so a user could tell"
    )
    assert len(colours) >= 3
    print(f"✓ status: {len(_STATUS)} statuses, distinct colours")


def check_rendering() -> None:
    """Rendering must not raise on any shape the API can return."""
    render_trace(["router:vague", "retrieval:5", "generation", "grading:fail",
                  "rewrite:1", "retrieval:5", "generation", "grading:pass"])
    render_answer(SAMPLE)
    render_sources(SAMPLE)

    # Degenerate shapes: a no-match answer has no sources, no citations, no grading.
    render_answer({"answer": "No passages matched.", "status": "no_match",
                   "latency_ms": 120.0})
    render_sources({"sources": [], "citations": []})
    render_trace([])
    print("✓ rendering: full and empty payloads both render")


def check_error_parsing() -> None:
    """The client must surface the server's own error envelope."""
    class FakeResponse:
        status_code = 503

        def json(self):
            return {"error": "RetrievalError", "message": "Search unavailable",
                    "retryable": True}

    error = RAGAPIError.from_response(FakeResponse())
    assert error.status == 503 and error.retryable
    assert "Search unavailable" in str(error)

    class NotJson:
        status_code = 502
        text = "<html>Bad Gateway</html>"

        def json(self):
            raise ValueError("not json")

    fallback = RAGAPIError.from_response(NotJson())
    assert fallback.status == 502, (
        "a non-JSON response from a proxy must keep its status, not become a "
        "JSONDecodeError"
    )
    print("✓ client: error envelopes parsed, non-JSON responses survive")


def main() -> int:
    try:
        check_status_coverage()
        check_rendering()
        check_error_parsing()
    except AssertionError as exc:
        print(f"\n✗ FAILED: {exc}")
        return 1
    print("\nPhase 9 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Check by hand

Start the API, open `src/web/index.html`, and confirm four things the script cannot:

1. The trace panel fills in **during** the query, not after. If steps appear all at once, the stream
   is being buffered — usually a proxy, or `EventSourceResponse` not being flushed.
2. Ask something that will fail grading (a question about an amount the corpus does not state). The
   trace should show `verifying → refining → searching` again, and the answer should carry a caveat.
3. Stop the API mid-query. The dashboard should show a readable error, not hang.
4. Ask about a contract whose text contains `<` or `&`. It must render as text.

---

## 8. What Phase 9 Bought You

**The system is visible.** The trace panel is the demo: six phases of retrieval, grading, and
self-correction were previously observable only in logs, and now a failed grade and its retry are
something you watch happen.

**Grounding is checkable in one glance.** Cited sources are marked, uncited retrieved sources are
shown greyed, and the numbering matches the `[SOURCE N]` markers in the answer.

**Four answer statuses that look different.** `UNVERIFIED` is not `LOW_CONFIDENCE`, and neither is
`ANSWERED`. Phases 5 and 6 fought to keep that distinction in the data; this is where it reaches a
person.

**A CLI that is genuinely faster to use than curl**, exits non-zero on failure so it composes in
shell pipelines, and never loads a model into its own process.

### What is deliberately not here

Everything in §2 — a framework build, chat history, filtering and export, evaluation charts. The
cuts are what kept this at 640 lines for a phase that could absorb five thousand.

The one honest gap: the dashboard is **single-user by construction**. It has no auth, points at
`127.0.0.1`, and inherits the `last_trace` limitation from Phase 8 §5, so two browser tabs querying
at once will see each other's progress events. Phase 11 is where that becomes worth fixing, because
that is where requests gain a principal.

---

## Next

**Phase 10 — testing and verification.** The largest remaining phase at 1,100 lines: `pytest`
fixtures, unit tests mirroring `src/`, integration tests against real Qdrant and Redis, and the CI
wiring that runs Phase 7's regression gate. It also absorbs the deferred verification scripts from
every phase — nine `scripts/verify_phaseN.py` files that currently duplicate one another's setup.
