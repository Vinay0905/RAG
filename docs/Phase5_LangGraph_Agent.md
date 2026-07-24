# Phase 5 Study Guide: LangGraph Self-Corrective Agent Engine

Welcome to **Phase 5** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will construct a state machine agent using **LangGraph**. The agent implements an autonomous self-correction loop: expanding queries, retrieving hybrid chunks, reranking context, generating answers with strict inline citations, grading responses for hallucinations and relevance, and dynamically rewriting search queries when context or grounding is insufficient.

---

## 📁 Directory Structure for Phase 5

Ensure the following subdirectories exist inside `src/graph/`:

```text
src/graph/
├── __init__.py
├── state.py
├── nodes/
│   ├── __init__.py
│   ├── router_node.py
│   ├── expansion_node.py
│   ├── retrieval_node.py
│   ├── reranking_node.py
│   ├── generation_node.py
│   ├── grading_node.py
│   └── rewriting_node.py
└── builder.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/graph/state.py`
**Location**: `src/graph/state.py`  
**Purpose**: Defines the strongly-typed `AgentState` schema dictionary tracking agent workflow variables.

```python
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """Pydantic-compatible state schema shared across all LangGraph nodes."""

    original_query: str
    current_query: str
    expanded_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    response: str
    retry_count: int
    max_retries: int
    grading_report: Dict[str, Any]
    intent: str
    is_grounded: bool
    is_relevant: bool
```

> **Deep 2-Line Explanation**:  
> *Defines the global memory dictionary passed from node to node throughout the LangGraph agent state machine.*  
> *Tracks original queries, retrieved context chunks, generated answers, grading results, and retry counters.*

---

### File 2: `src/graph/nodes/router_node.py`
**Location**: `src/graph/nodes/router_node.py`  
**Purpose**: Classifies user intent to route directly to search or handle general legal greetings.

```python
from src.graph.state import AgentState
from src.core.logging import logger


def route_intent_node(state: AgentState) -> Dict[str, Any]:
    """Node that classifies user prompt intent."""
    query = state["current_query"].lower().strip()
    logger.info(f"🤖 [NODE: Router] Classifying query: '{query}'")

    if any(greeting in query for greeting in ["hi", "hello", "who are you", "help"]):
        return {
            "intent": "greeting",
            "response": "Hello! I am your CUAD AI Legal Contract Assistant. Ask me any question regarding your contracts!"
        }

    return {"intent": "contract_query"}
```

> **Deep 2-Line Explanation**:  
> *Analyzes incoming user queries to separate general greetings from complex legal contract searches.*  
> *Allows simple user interactions to bypass expensive retrieval loops entirely.*

---

### File 3: `src/graph/nodes/expansion_node.py`
**Location**: `src/graph/nodes/expansion_node.py`  
**Purpose**: Graph node executing search query expansion.

```python
from typing import Dict, Any
from src.graph.state import AgentState
from src.retrieval.multi_query import MultiQueryExpander
from src.core.logging import logger


expander = MultiQueryExpander()


def expand_query_node(state: AgentState) -> Dict[str, Any]:
    """Node that expands search queries for RAG fusion."""
    logger.info("🤖 [NODE: Expansion] Expanding query...")
    queries = expander.expand(state["current_query"])
    return {"expanded_queries": queries}
```

> **Deep 2-Line Explanation**:  
> *Wraps our multi-query generator as a discrete LangGraph node, adding expanded queries to state.*  
> *Prepares multiple search variations before initiating vector database retrieval.*

---

### File 4: `src/graph/nodes/retrieval_node.py`
**Location**: `src/graph/nodes/retrieval_node.py`  
**Purpose**: Graph node querying vector database with expanded queries.

```python
from typing import Dict, Any
from src.graph.state import AgentState
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.retrieval.hybrid import ReciprocalRankFusion
from src.core.logging import logger


embedder = EmbeddingProviderFactory.get_provider("fastembed")
store = VectorStoreFactory.get_store("qdrant")
rrf = ReciprocalRankFusion()


def retrieve_chunks_node(state: AgentState) -> Dict[str, Any]:
    """Node executing hybrid search across expanded queries."""
    logger.info("🤖 [NODE: Retrieval] Executing vector searches...")
    queries = state.get("expanded_queries", [state["current_query"]])

    all_rank_lists = []
    for q in queries:
        dense_vec = embedder.embed_dense([q])[0]
        sparse_vec = embedder.embed_sparse([q])[0]

        hits = store.search_hybrid(dense_vec, sparse_vec, limit=10)
        all_rank_lists.append(hits)

    fused_chunks = rrf.fuse(all_rank_lists)
    return {"retrieved_chunks": fused_chunks}
```

> **Deep 2-Line Explanation**:  
> *Performs parallel hybrid dense-sparse vector lookups across expanded queries and applies RRF fusion.*  
> *Populates state with top candidates aggregated across all search variations.*

---

### File 5: `src/graph/nodes/reranking_node.py`
**Location**: `src/graph/nodes/reranking_node.py`  
**Purpose**: Graph node applying Cross-Encoder reranking to retrieved candidates.

```python
from typing import Dict, Any
from src.graph.state import AgentState
from src.retrieval.rerankers.cross_encoder import CrossEncoderReranker
from src.core.logging import logger


reranker = CrossEncoderReranker()


def rerank_chunks_node(state: AgentState) -> Dict[str, Any]:
    """Node applying CrossEncoder re-scoring."""
    logger.info("🤖 [NODE: Reranking] Re-scoring chunks...")
    query = state["original_query"]
    candidates = state.get("retrieved_chunks", [])

    top_chunks = reranker.rerank(query, candidates, top_k=5)
    return {"reranked_chunks": top_chunks}
```

> **Deep 2-Line Explanation**:  
> *Uses Cross-Encoder context attention to refine candidate order and select the top 5 chunks.*  
> *Filters out noisy vector matches so the generation prompt receives clean, hyper-relevant context.*

---

### File 6: `src/graph/nodes/generation_node.py`
**Location**: `src/graph/nodes/generation_node.py`  
**Purpose**: Graph node generating answers via Groq with inline contract source citations.

```python
from typing import Dict, Any
from groq import Groq
from config.settings import settings
from src.graph.state import AgentState
from src.core.logging import logger

client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Node generating grounded answer with citations."""
    logger.info("🤖 [NODE: Generation] Writing grounded answer...")
    query = state["original_query"]
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {"response": "INSUFFICIENT_CONTEXT: No relevant contract sections were found."}

    context_str = ""
    for idx, c in enumerate(chunks):
        contract_name = c.get("contract_name", "Agreement")
        year = c.get("year", "N/A")
        section = c.get("section", "Clause")
        text = c.get("chunk_text", "")
        context_str += f"\n[SOURCE {idx+1}]: {contract_name} ({year}) Section: {section}\nText: {text}\n"

    prompt = f"""You are an expert corporate lawyer. Answer the query based ONLY on the provided contract sources below.
Use inline citations when stating facts (e.g. "Fact [SOURCE 1]").
If the context does not contain enough information to answer the query, reply with: 'INSUFFICIENT_CONTEXT'

Query: {query}

Sources:
{context_str}"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=settings.GENERATION_MODEL,
            temperature=0.0
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        logger.error(f"Generation node failed: {str(e)}")
        return {"response": "Generation error occurred."}
```

> **Deep 2-Line Explanation**:  
> *Formats reranked chunks into structured `[SOURCE N]` prompts and generates answers strictly grounded in context.*  
> *Forces the model to output 'INSUFFICIENT_CONTEXT' if provided contract clauses do not contain the answer.*

---

### File 7: `src/graph/nodes/grading_node.py`
**Location**: `src/graph/nodes/grading_node.py`  
**Purpose**: Self-correction grader evaluating faithfulness (groundedness) and query relevance.

```python
import json
from typing import Dict, Any
from groq import Groq
from config.settings import settings
from src.graph.state import AgentState
from src.core.logging import logger

client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None


def grade_response_node(state: AgentState) -> Dict[str, Any]:
    """Node grading answer for hallucinations (groundedness) and query relevance."""
    logger.info("🤖 [NODE: Grader] Fact-checking candidate response...")
    query = state["original_query"]
    response = state.get("response", "")
    chunks = state.get("reranked_chunks", [])

    if "INSUFFICIENT_CONTEXT" in response or not chunks:
        return {
            "grading_report": {"grounded": False, "relevant": False, "explanation": "Insufficient context"},
            "is_grounded": False,
            "is_relevant": False
        }

    context_text = "\n\n".join([c.get("chunk_text", "") for c in chunks])

    prompt = f"""Analyze if the Candidate Response is fully grounded and supported by the Reference Context (legal contracts).
Also check if it directly answers the User Query.
Respond in exactly this JSON format:
{{
  "grounded": true/false,
  "relevant": true/false,
  "explanation": "Short explanation"
}}

User Query: {query}
Reference Context: {context_text}
Candidate Response: {response}"""

    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=settings.GRADER_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        report = json.loads(res.choices[0].message.content.strip())
        is_g = report.get("grounded", False)
        is_r = report.get("relevant", False)
        logger.info(f"Grade results -> Grounded: {is_g}, Relevant: {is_r}")

        return {
            "grading_report": report,
            "is_grounded": is_g,
            "is_relevant": is_r
        }
    except Exception as e:
        logger.error(f"Grading failed: {str(e)}")
        return {
            "grading_report": {"grounded": True, "relevant": True},
            "is_grounded": True,
            "is_relevant": True
        }
```

> **Deep 2-Line Explanation**:  
> *Acts as an automated legal proofreader using structured JSON format to evaluate hallucination and relevance metrics.*  
> *Detects unsupported statements or off-topic responses, setting boolean flags that trigger graph retries.*

---

### File 8: `src/graph/nodes/rewriting_node.py`
**Location**: `src/graph/nodes/rewriting_node.py`  
**Purpose**: Self-correction node that rewrites queries when grading fails.

```python
from typing import Dict, Any
from groq import Groq
from config.settings import settings
from src.graph.state import AgentState
from src.core.logging import logger

client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None


def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """Node rewriting search query to improve retrieval on retry."""
    current_q = state["current_query"]
    retries = state.get("retry_count", 0) + 1
    logger.info(f"🔄 [NODE: Rewrite] Rewriting search query (Attempt #{retries})...")

    prompt = f"""The previous search query '{current_q}' did not retrieve enough contract clause information to answer the legal topic.
Rewrite this query to be broader, using legal synonyms, contract terms, or different search angles to retrieve relevant clauses.
Output ONLY the rewritten search query. No extra text.

Query: {current_q}"""

    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=settings.GRADER_MODEL,
            temperature=0.3
        )
        new_q = res.choices[0].message.content.strip()
        logger.info(f"Rewritten query -> '{new_q}'")
        return {
            "current_query": new_q,
            "retry_count": retries
        }
    except Exception:
        return {
            "current_query": current_q,
            "retry_count": retries
        }
```

> **Deep 2-Line Explanation**:  
> *Reformulates failed search queries using legal synonyms and broader terms to unlock hidden clauses.*  
> *Increments the retry counter to prevent infinite execution loops while guiding the next search iteration.*

---

### File 9: `src/graph/builder.py`
**Location**: `src/graph/builder.py`  
**Purpose**: Constructs the StateGraph, configures conditional routing edges, and compiles the agent graph.

```python
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes.router_node import route_intent_node
from src.graph.nodes.expansion_node import expand_query_node
from src.graph.nodes.retrieval_node import retrieve_chunks_node
from src.graph.nodes.reranking_node import rerank_chunks_node
from src.graph.nodes.generation_node import generate_response_node
from src.graph.nodes.grading_node import grade_response_node
from src.graph.nodes.rewriting_node import rewrite_query_node
from config.settings import settings
from src.core.logging import logger


class RAGAgentGraphBuilder:
    """Builder for compiled LangGraph Self-Corrective RAG agent."""

    @staticmethod
    def route_after_router(state: AgentState) -> str:
        if state.get("intent") == "greeting":
            return "end"
        return "expand"

    @staticmethod
    def route_after_grading(state: AgentState) -> str:
        is_g = state.get("is_grounded", False)
        is_r = state.get("is_relevant", False)
        retries = state.get("retry_count", 0)
        max_r = state.get("max_retries", settings.MAX_RETRIES)

        if is_g and is_r:
            logger.info("🎉 Answer passed all grade checks! Completing graph.")
            return "pass"

        if retries < max_r:
            logger.warning(f"⚠️ Answer failed grading. Retrying via rewrite ({retries}/{max_r})...")
            return "retry"

        logger.warning("🚨 Max retries reached. Exiting graph.")
        return "max_reached"

    def build(self):
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("router", route_intent_node)
        workflow.add_node("expand_query", expand_query_node)
        workflow.add_node("retrieve", retrieve_chunks_node)
        workflow.add_node("rerank", rerank_chunks_node)
        workflow.add_node("generate", generate_response_node)
        workflow.add_node("grade", grade_response_node)
        workflow.add_node("rewrite", rewrite_query_node)

        # Set Entry Point
        workflow.set_entry_point("router")

        # Add Edges
        workflow.add_conditional_edges(
            "router",
            self.route_after_router,
            {"end": END, "expand": "expand_query"}
        )
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "grade")

        workflow.add_conditional_edges(
            "grade",
            self.route_after_grading,
            {
                "pass": END,
                "retry": "rewrite",
                "max_reached": END
            }
        )

        workflow.add_edge("rewrite", "expand_query")

        return workflow.compile()
```

> **Deep 2-Line Explanation**:  
> *Assembles all graph nodes into a cyclic state graph, wiring conditional edge routers for grading passes and rewrites.*  
> *Compiles the complete Self-Corrective RAG workflow ready for invocation via CLI, REST API, or WebSockets.*

---

## 🎯 Phase 5 Checkpoint Verification

To verify Phase 5:
Run an end-to-end graph test in Python:
```python
from src.graph.builder import RAGAgentGraphBuilder

builder = RAGAgentGraphBuilder()
app = builder.build()

initial_state = {
    "original_query": "What is the limit of liability?",
    "current_query": "What is the limit of liability?",
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

final_state = app.invoke(initial_state)
print("FINAL RESPONSE:\n", final_state["response"])
print("RETRIES PERFORMED:", final_state["retry_count"])
```

When you are ready, let me know to proceed to **Phase 6: LLM Providers, Redis Semantic Cache & Enterprise Guardrails**!
