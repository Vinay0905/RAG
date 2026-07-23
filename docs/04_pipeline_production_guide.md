# Phase 4: Production Pipeline Packaging Guide
**Target Location**: `Pipelines/`

Once you have successfully executed the notebooks in your `notesbooks/` directory and are satisfied with the retrieval performance, we move the code into production scripts.

We will organize the codebase inside the `Pipelines/` folder. This set of production scripts accesses the local CUAD contract data from your `dataset/` folder and indexes it using **Recursive Sentence-Aligned Chunking**.

---

## 1. Directory Structure

Ensure the `Pipelines/` folder contains the following files:
```
Pipelines/
├── config.py         # Configs (Local Paths, Qdrant, Groq Models, parameters)
├── ingestion.py      # Local file batch parsing & Qdrant ingestion
├── reranker.py       # Cross-Encoder scoring module
├── prompts.py        # System prompts for expansion, QA, and grading
├── agent.py          # LangGraph state graph compilation
└── main.py           # Command-Line Interface (CLI) entry point
```

---

## 2. Production Code Modules

Below are the updated contents for the packaged pipeline:

### File 1: `Pipelines/config.py`
```python
import os
from dotenv import load_dotenv

# Load keys
load_dotenv(dotenv_path="../.env")

# Local Data Path
DATASET_PATH = os.getenv("DATASET_PATH", "/Users/mast/Documents/VInayPrograming/RAG/dataset")

# Qdrant Configs
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "cuad_advanced"

# Models Configs
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-6-v2"

# Groq Configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXPANSION_MODEL = "llama-3.1-8b-instant"
GENERATION_MODEL = "llama-3.1-70b-versatile"
GRADER_MODEL = "llama-3.1-8b-instant"

# RAG Hyperparameters
MAX_TOKENS = 400
SENTENCE_OVERLAP = 2
MAX_RETRIES = 2
```

### File 2: `Pipelines/prompts.py`
```python
QUERY_EXPANSION_PROMPT = """You are a legal research search assistant.
Generate exactly 3 variations of the following search query to help retrieve relevant clauses (such as termination, indemnity, limitation of liability) from a legal contract vector database.
Generate ONLY the 3 queries, one per line. Do not number them or add any introductory text.

Original Query: {query}"""

GENERATION_PROMPT = """You are an expert corporate lawyer. Answer the query based ONLY on the provided contract sources below.
Use inline citations when stating facts (e.g. "Fact [SOURCE 1]").
If the context does not contain enough information to answer the query, reply with: 'INSUFFICIENT_CONTEXT'

Query: {query}

Sources:
{contexts}
"""

GRADER_PROMPT = """Analyze if the Candidate Response is fully grounded and supported by the Reference Context (legal contracts).
Also check if it directly answers the User Query.
Respond in exactly this JSON format:
{{
  "grounded": true/false,
  "relevant": true/false,
  "explanation": "Short explanation"
}}

User Query: {query}
Reference Context: {context}
Candidate Response: {response}"""

QUERY_REWRITE_PROMPT = """The previous search query '{query}' did not retrieve enough contract clause information to answer the legal topic.
Rewrite this query to be broader, using legal synonyms, contract terms, or different search angles to retrieve relevant clauses.
Output ONLY the rewritten search query. No extra conversational text.

Query: {query}"""
```

### File 3: `Pipelines/reranker.py`
```python
from sentence_transformers import CrossEncoder
import config

class CrossEncoderReranker:
    def __init__(self):
        print(f"Loading reranker model '{config.RERANK_MODEL_NAME}'...")
        self.model = CrossEncoder(config.RERANK_MODEL_NAME)
        
    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        if not candidates:
            return []
            
        pairs = [[query, item["chunk_text"]] for item in candidates]
        scores = self.model.predict(pairs)
        
        scored_candidates = []
        for idx, score in enumerate(scores):
            item = candidates[idx]
            item["rerank_score"] = float(score)
            scored_candidates.append(item)
            
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
```

### File 4: `Pipelines/ingestion.py`
```python
import os
import re
import uuid
import tiktoken
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector, PayloadSchemaType
from fastembed import TextEmbedding, SparseTextEmbedding
import config

tokenizer = tiktoken.get_encoding("cl100k_base")

class IngestionPipeline:
    def __init__(self):
        print("Initializing Qdrant Local Ingestion pipeline...")
        self.client = QdrantClient(config.QDRANT_HOST)
        self.dense_model = TextEmbedding(config.DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(config.SPARSE_MODEL_NAME)
        
    def setup_collection(self):
        print(f"Setting up Qdrant collection: {config.COLLECTION_NAME}")
        # Create base collection with Dual-Vector Indexes
        self.client.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                "dense-bge": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse-bm25": SparseVectorParams(index=SparseIndexParams(on_disk=True))
            }
        )
        
        # Create Payload Indexes for fast metadata filtering
        print("Registering metadata payload indexes...")
        self.client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name="year",
            field_schema=PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name="section",
            field_schema=PayloadSchemaType.KEYWORD
        )

    def split_recursive_sentence(self, text: str) -> list:
        text = " ".join(text.split())
        if not text:
            return []
            
        raw_sentences = text.split(". ")
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                if not s.endswith("."):
                    s += "."
                sentences.append(s)
                
        chunks = []
        i = 0
        num_sentences = len(sentences)
        
        while i < num_sentences:
            current_chunk_sentences = []
            current_tokens = 0
            
            j = i
            while j < num_sentences:
                s_text = sentences[j]
                s_len = len(tokenizer.encode(s_text))
                
                if s_len > config.MAX_TOKENS:
                    if current_chunk_sentences:
                        break
                    words = s_text.split(" ")
                    w_idx = 0
                    while w_idx < len(words):
                        sub_words = words[w_idx : w_idx + config.MAX_TOKENS]
                        chunks.append(" ".join(sub_words))
                        w_idx += config.MAX_TOKENS
                    j += 1
                    i = j
                    break
                    
                if current_tokens + s_len <= config.MAX_TOKENS:
                    current_chunk_sentences.append(s_text)
                    current_tokens += s_len
                    j += 1
                else:
                    break
                    
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                
            if j >= num_sentences:
                break
                
            i = max(j - config.SENTENCE_OVERLAP, i + 1)
            
        return chunks

    def stream_local_documents(self, limit: int):
        # Support passing either the parent directory or the 'contracts' directory itself
        if os.path.basename(config.DATASET_PATH.rstrip("/")) == "contracts":
            contracts_dir = config.DATASET_PATH
        else:
            contracts_dir = os.path.join(config.DATASET_PATH, "contracts")
            
        if not os.path.exists(contracts_dir):
            raise FileNotFoundError(f"Contracts directory not found at: {contracts_dir}")
            
        count = 0
        for root, dirs, files in os.walk(contracts_dir):
            for file in sorted(files):
                if file.endswith(".txt"):
                    if count >= limit:
                        break
                    
                    file_path = os.path.join(root, file)
                    year = os.path.basename(root)
                    
                    try:
                        with open(file_path, mode="r", encoding="utf-8") as f:
                            text = f.read()
                    except Exception:
                        continue
                    
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    contract_name = lines[0] if lines else "Unknown Contract"
                    if len(contract_name) > 100:
                        contract_name = contract_name[:97] + "..."
                        
                    yield {
                        "doc_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)),
                        "contract_name": contract_name,
                        "file_name": file,
                        "year": year,
                        "text": text
                    }
                    count += 1
            if count >= limit:
                break

    def parse_document(self, doc: dict) -> list:
        text = doc["text"]
        meta = {
            "doc_id": doc["doc_id"],
            "contract_name": doc["contract_name"],
            "file_name": doc["file_name"],
            "year": doc["year"]
        }
        
        section_pattern = re.compile(r'^(SECTION\s+\d+\.\d+|ARTICLE\s+[IVXLCDM]+|EXHIBIT\s+[A-Z]|\bINDEMNITY\b|\bTERMINATION\b|\bLIMITATION\b)', re.IGNORECASE)
        lines = text.split("\n")
        current_section = "Preamble"
        section_text_blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            if section_pattern.match(stripped) and len(stripped) < 80:
                if current_block:
                    section_text_blocks.append((current_section, "\n".join(current_block)))
                    current_block = []
                current_section = stripped
            else:
                current_block.append(stripped)
                
        if current_block:
            section_text_blocks.append((current_section, "\n".join(current_block)))
            
        chunks = []
        for section_name, section_text in section_text_blocks:
            if len(section_text.strip()) > 50:
                for c_text in self.split_recursive_sentence(section_text):
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_text": c_text,
                        "section": section_name
                    }
                    chunk.update(meta)
                    chunks.append(chunk)
        return chunks

    def run(self, limit: int = 100):
        self.setup_collection()
        
        print(f"Reading contracts locally from {config.DATASET_PATH}...")
        doc_stream = self.stream_local_documents(limit)
        
        batch = []
        count = 0
        point_id = 1
        
        for doc in tqdm(doc_stream, total=limit, desc="Processing contracts"):
            chunks = self.parse_document(doc)
            for chunk in chunks:
                batch.append(chunk)
                
                if len(batch) >= 100:
                    self.upload_batch(batch, point_id)
                    point_id += len(batch)
                    batch = []
            count += 1
            
        if batch:
            self.upload_batch(batch, point_id)
            
        print(f"\nCompleted! Ingested {count} contracts and registered chunks in Qdrant.")

    def upload_batch(self, batch_data: list, start_id: int):
        texts = [item["chunk_text"] for item in batch_data]
        dense_embs = list(self.dense_model.embed(texts))
        sparse_embs = list(self.sparse_model.embed(texts))
        
        points = []
        for idx, item in enumerate(batch_data):
            sparse_obj = sparse_embs[idx]
            q_sparse = SparseVector(
                indices=sparse_obj.indices.tolist(),
                values=sparse_obj.values.tolist()
            )
            
            point = PointStruct(
                id=start_id + idx,
                vector={"dense-bge": dense_embs[idx].tolist(), "sparse-bm25": q_sparse},
                payload=item
            )
            points.append(point)
            
        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)
```

### File 5: `Pipelines/agent.py`
```python
import json
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
from langgraph.graph import StateGraph, END

import config
import prompts
from reranker import CrossEncoderReranker

class RAGAgentPipeline:
    def __init__(self):
        print("Compiling agent nodes...")
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.qdrant_client = QdrantClient(config.QDRANT_HOST)
        self.dense_model = TextEmbedding(config.DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(config.SPARSE_MODEL_NAME)
        self.reranker = CrossEncoderReranker()
        self.app = self.build_graph()

    def rrf(self, results_lists: list) -> list:
        scores = {}
        for rank_list in results_lists:
            for idx, hit in enumerate(rank_list):
                payload = hit.payload
                c_id = payload["chunk_id"]
                score = 1.0 / (60 + idx + 1)
                
                if c_id not in scores:
                    scores[c_id] = {"payload": payload, "score": 0.0}
                scores[c_id]["score"] += score
        sorted_scores = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["payload"] for item in sorted_scores]

    def expand_query(self, state: dict) -> dict:
        query = state["current_query"]
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.QUERY_EXPANSION_PROMPT.format(query=query)}],
            model=config.EXPANSION_MODEL,
            temperature=0.2
        )
        resp = comp.choices[0].message.content.strip()
        queries = [q.strip() for q in resp.split("\n") if q.strip()]
        if query not in queries:
            queries.append(query)
        return {"expanded_queries": queries}

    def retrieve(self, state: dict) -> dict:
        queries = state["expanded_queries"]
        results = []
        for q in queries:
            q_dense = list(self.dense_model.embed([q]))[0].tolist()
            res = self.qdrant_client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=("dense-bge", q_dense),
                limit=10
            )
            results.append(res)
        return {"retrieved_chunks": self.rrf(results)}

    def rerank(self, state: dict) -> dict:
        top_chunks = self.reranker.rerank(state["original_query"], state["retrieved_chunks"])
        return {"reranked_chunks": top_chunks}

    def generate(self, state: dict) -> dict:
        query = state["original_query"]
        contexts = ""
        for idx, c in enumerate(state["reranked_chunks"]):
            contexts += f"\nSOURCE [{idx+1}]: {c['contract_name']} ({c['year']}) Section: {c['section']}\nContent: {c['chunk_text']}\n"
            
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.GENERATION_PROMPT.format(query=query, contexts=contexts)}],
            model=config.GENERATION_MODEL,
            temperature=0.0
        )
        return {"response": comp.choices[0].message.content.strip()}

    def grade_response(self, state: dict) -> dict:
        query = state["original_query"]
        resp = state["response"]
        context = "\n\n".join([c["chunk_text"] for c in state["reranked_chunks"]])
        
        if "INSUFFICIENT_CONTEXT" in resp:
            return {"grading_report": {"grounded": False, "relevant": False}}
            
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.GRADER_PROMPT.format(query=query, context=context, response=resp)}],
            model=config.GRADER_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        report = json.loads(comp.choices[0].message.content.strip())
        return {"grading_report": report}

    def feedback_rewrite(self, state: dict) -> dict:
        query = state["current_query"]
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.QUERY_REWRITE_PROMPT.format(query=query)}],
            model=config.GRADER_MODEL,
            temperature=0.3
        )
        return {"current_query": comp.choices[0].message.content.strip(), "retry_count": state["retry_count"] + 1}

    def route_grading(self, state: dict) -> str:
        report = state["grading_report"]
        if report.get("grounded", False) and report.get("relevant", False):
            return "pass"
        if state["retry_count"] < state["max_retries"]:
            return "fail"
        return "max_reached"

    def build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("expand_query", self.expand_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        workflow.add_node("grade_response", self.grade_response)
        workflow.add_node("feedback_rewrite", self.feedback_rewrite)
        
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "grade_response")
        
        workflow.add_conditional_edges(
            "grade_response",
            self.route_grading,
            {"pass": END, "fail": "feedback_rewrite", "max_reached": END}
        )
        workflow.add_edge("feedback_rewrite", "expand_query")
        return workflow.compile()

    def run_agent(self, query: str):
        initial_state = {
            "original_query": query,
            "current_query": query,
            "expanded_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "response": "",
            "retry_count": 0,
            "max_retries": config.MAX_RETRIES,
            "grading_report": {}
        }
        return self.app.invoke(initial_state)
```

### File 6: `Pipelines/main.py`
Exposes the CLI.

```python
import argparse
import sys
from ingestion import IngestionPipeline
from agent import RAGAgentPipeline

def main():
    parser = argparse.ArgumentParser(description="Industrial CUAD Legal Contracts LangGraph RAG CLI Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Ingestion subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CUAD contracts into Qdrant")
    ingest_parser.add_argument("--limit", type=int, default=100, help="Number of contracts to process and index")
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run the RAG query agent")
    query_parser.add_argument("text", type=str, help="Search query question")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        pipeline = IngestionPipeline()
        pipeline.run(limit=args.limit)
    elif args.command == "query":
        agent = RAGAgentPipeline()
        res = agent.run_agent(args.text)
        print("\n================ ANSWER ================")
        print(res["response"])
        print("========================================")
        print(f"Correction retries: {res['retry_count']}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```
