# Phase 9 Study Guide: Rich CLI & Modern Glassmorphic Web Dashboard

Welcome to **Phase 9** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will build user interfaces across two modalities: a **Rich Terminal CLI** with progress spinners, formatted ASCII banners, and tables, and a **Modern Glassmorphic Web Dashboard** with real-time LangGraph execution node timelines and streaming search results.

---

## 📁 Directory Structure for Phase 9

Ensure the following subdirectories exist inside `src/`:

```text
src/
├── cli/
│   ├── __init__.py
│   ├── main.py
│   └── ui.py
└── web/
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/cli/ui.py`
**Location**: `src/cli/ui.py`  
**Purpose**: Component library using `rich` for terminal banners, tables, and formatted Markdown rendering.

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def print_banner():
    """Prints styled ASCII header banner."""
    banner_text = "[bold cyan]CUAD LEGAL CONTRACTS[/bold cyan]\n[dim]Enterprise Self-Corrective LangGraph RAG System[/dim]"
    console.print(Panel(banner_text, expand=False, border_style="cyan"))


def print_answer(answer: str, retry_count: int):
    """Prints formatted Markdown response with retry counts."""
    console.print(Panel("[bold green]================ FINAL ANSWER ================[/bold green]", border_style="green"))
    console.print(Markdown(answer))
    console.print(f"\n[dim]Self-Correction Retries Performed: {retry_count}[/dim]\n")


def print_sources(sources: list):
    """Prints source contract chunks in a formatted Rich table."""
    table = Table(title="Retrieved Contract Sources", border_style="blue")
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Contract Name", style="bold white")
    table.add_column("Section", style="magenta")
    table.add_column("Score", style="yellow")

    for idx, c in enumerate(sources):
        table.add_row(
            f"SOURCE {idx+1}",
            c.get("contract_name", "Agreement")[:30],
            c.get("section", "Clause")[:25],
            f"{c.get('rerank_score', 0.0):.4f}" if c.get('rerank_score') else "N/A"
        )
    console.print(table)
```

> **Deep 2-Line Explanation**:  
> *Renders colorful terminal panels, Markdown responses, and interactive tables using Python's `rich` library.*  
> *Transforms plain CLI outputs into an attractive developer dashboard for interactive contract Q&A.*

---

### File 2: `src/cli/main.py`
**Location**: `src/cli/main.py`  
**Purpose**: CLI entry point supporting `ingest`, `query`, `eval`, and `server` commands.

```python
import argparse
import sys
from src.cli.ui import print_banner, print_answer, print_sources, console
from src.graph.builder import RAGAgentGraphBuilder
from src.ingestion.pipeline import IngestionPipeline
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory


def main():
    parser = argparse.ArgumentParser(description="CUAD Production RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Ingest subcommand
    ingest_p = subparsers.add_parser("ingest", help="Ingest contract files into Qdrant")
    ingest_p.add_argument("--limit", type=int, default=50, help="Max documents to index")

    # Query subcommand
    query_p = subparsers.add_parser("query", help="Run self-corrective RAG query")
    query_p.add_argument("text", type=str, help="Question to ask about contracts")

    # Server subcommand
    subparsers.add_parser("server", help="Launch FastAPI Web Server")

    args = parser.parse_args()

    print_banner()

    if args.command == "ingest":
        console.print(f"[bold yellow]Initiating ingestion (limit={args.limit})...[/bold yellow]")
        pipeline = IngestionPipeline()
        chunks = pipeline.run(limit=args.limit)

        if chunks:
            embedder = EmbeddingProviderFactory.get_provider("fastembed")
            store = VectorStoreFactory.get_store("qdrant")

            store.setup_collection()
            texts = [c["chunk_text"] for c in chunks]
            denses = embedder.embed_dense(texts)
            sparses = embedder.embed_sparse(texts)
            store.upsert_chunks(chunks, denses, sparses)
            console.print(f"[bold green]✅ Ingested and indexed {len(chunks)} chunks in Qdrant![/bold green]")

    elif args.command == "query":
        console.print(f"[bold cyan]Running Agent for Query:[/bold cyan] '{args.text}'")
        builder = RAGAgentGraphBuilder()
        app = builder.build()

        initial_state = {
            "original_query": args.text,
            "current_query": args.text,
            "expanded_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "response": "",
            "retry_count": 0,
            "max_retries": 2,
            "grading_report": {},
            "intent": "",
            "is_grounded": False,
            "is_relevant": False
        }

        res = app.invoke(initial_state)
        print_answer(res["response"], res["retry_count"])
        if res.get("reranked_chunks"):
            print_sources(res["reranked_chunks"])

    elif args.command == "server":
        import uvicorn
        console.print("[bold green]🚀 Launching FastAPI Web Server on http://localhost:8000[/bold green]")
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

> **Deep 2-Line Explanation**:  
> *Provides a unified command-line entry point (`ingest`, `query`, `server`) using standard `argparse`.*  
> *Executes ingestion, initiates agent graph runs, or launches the FastAPI web server with a single terminal command.*

---

### File 3: `src/web/index.html`
**Location**: `src/web/index.html`  
**Purpose**: HTML5 single-page application template with modern glassmorphic interface layout.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUAD Legal RAG Dashboard</title>
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <header class="navbar">
            <div class="logo">
                <span class="icon">⚖️</span>
                <h1>CUAD Legal RAG Engine</h1>
            </div>
            <div class="status-badge" id="statusBadge">Connecting...</div>
        </header>

        <main class="dashboard-grid">
            <section class="card query-section">
                <h2>Ask Legal Question</h2>
                <div class="input-group">
                    <input type="text" id="queryInput" placeholder="e.g. What is the limit of liability under the contract?" />
                    <button id="sendBtn">Ask Assistant</button>
                </div>

                <div class="live-timeline">
                    <h3>Execution Node Timeline</h3>
                    <div id="timelineContainer" class="timeline">
                        <div class="node-step ready">Idle</div>
                    </div>
                </div>
            </section>

            <section class="card response-section">
                <h2>Grounded Response</h2>
                <div id="responseBox" class="response-box">
                    <p class="placeholder">Results will appear here...</p>
                </div>
            </section>
        </main>
    </div>
    <script src="app.js"></script>
</body>
</html>
```

> **Deep 2-Line Explanation**:  
> *Structures a clean single-page web layout featuring a question input area, real-time node timeline, and answer display.*  
> *Imports Google Fonts ('Inter') and modern semantic HTML elements for an enterprise look and feel.*

---

### File 4: `src/web/styles.css`
**Location**: `src/web/styles.css`  
**Purpose**: Modern glassmorphic CSS stylesheet using CSS variables, dark mode palette, and subtle micro-animations.

```css
:root {
    --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    --card-bg: rgba(255, 255, 255, 0.05);
    --card-border: rgba(255, 255, 255, 0.1);
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

body {
    margin: 0;
    font-family: 'Inter', sans-serif;
    background: var(--bg-gradient);
    color: var(--text-main);
    min-height: 100vh;
}

.app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 2rem;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1fr solid var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.input-group {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

input[type="text"] {
    flex: 1;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    border: 1px solid var(--card-border);
    background: rgba(0, 0, 0, 0.2);
    color: white;
}

button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    transition: background 0.2s ease;
}

button:hover {
    background: var(--accent-hover);
}

.timeline {
    display: flex;
    gap: 0.5rem;
}

.node-step {
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    font-size: 0.85rem;
}

.node-step.active {
    background: var(--accent);
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
```

> **Deep 2-Line Explanation**:  
> *Applies a sleek dark glassmorphic design system using CSS backdrop filters, glowing gradients, and CSS variables.*  
> *Includes smooth micro-animations (`@keyframes pulse`) for active LangGraph execution nodes.*

---

### File 5: `src/web/app.js`
**Location**: `src/web/app.js`  
**Purpose**: Frontend JavaScript client connecting to WebSocket backend for real-time node state visualization.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/graph');
const statusBadge = document.getElementById('statusBadge');
const sendBtn = document.getElementById('sendBtn');
const queryInput = document.getElementById('queryInput');
const responseBox = document.getElementById('responseBox');
const timelineContainer = document.getElementById('timelineContainer');

ws.onopen = () => {
    statusBadge.innerText = 'Connected';
    statusBadge.style.color = '#4ade80';
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === 'node_start') {
        timelineContainer.innerHTML = `<div class="node-step active">Node: ${data.node}</div>`;
    } else if (data.event === 'graph_complete') {
        timelineContainer.innerHTML = `<div class="node-step">Completed (${data.retry_count} retries)</div>`;
        responseBox.innerHTML = `<p>${data.response}</p>`;
    }
};

sendBtn.addEventListener('click', () => {
    const query = queryInput.value.trim();
    if (!query) return;
    responseBox.innerHTML = '<p class="placeholder">Executing LangGraph agent...</p>';
    ws.send(JSON.stringify({ query: query }));
});
```

> **Deep 2-Line Explanation**:  
> *Manages real-time WebSocket communication to send user queries and listen for node execution step events.*  
> *Updates DOM elements dynamically, highlighting active graph execution nodes and rendering incoming answer text.*

---

## 🎯 Phase 9 Checkpoint Verification

To verify Phase 9:
1. Run a query in the CLI:
   ```bash
   python -m src.cli.main query "What are the termination terms?"
   ```
2. Start the web server and open the web dashboard:
   ```bash
   python -m src.cli.main server
   ```
   Open `http://localhost:8000` or inspect `src/web/index.html` in your browser.

When you are ready, let me know to proceed to **Phase 10: Pytest Suite & End-to-End Verification**!
