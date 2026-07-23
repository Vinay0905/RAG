# Product Requirements Document (PRD): Industrial LangGraph Self-Corrective RAG for CUAD Legal Contracts

## 1. Title and Summary
*   **Title**: CUAD Legal Contract Self-Corrective RAG Agent
*   **Summary**: A production-grade, local-first Retrieval-Augmented Generation (RAG) system utilizing the **CUAD (Contract Understanding Atticus Dataset)** commercial legal corpus. The system leverages **LangGraph** to model an agentic state machine that validates its own responses, **Qdrant** for dense-sparse hybrid search, and **Groq Cloud** for high-speed generation.

---

## 2. Problem Statement or Objective (Goal)
Legal counsel, contract managers, and compliance analysts spend significant time reviewing long, complex commercial legal agreements. 
*   **Naive RAG Limitations**: Standard RAG yields vague semantic matches, cuts clauses blindly at arbitrary character limits, and suffers from hallucinations that can lead to severe legal or financial liabilities.
*   **Legal Precision**: Legal Q&A requires exact clause identification (e.g., termination clauses, indemnity, limitation of liability) and strict grounding in the text of specific contracts.
*   **Objective**: Build a system that ingests CUAD contract text files locally using memory-efficient generators, segments them into logical contract sections, runs precise dense-sparse vector lookups, reranks results using a local Cross-Encoder transformer, and implements self-correction logic (fact-checking) in LangGraph to ensure final answers are 100% grounded in contract terms.

---

## 3. Target Users / Personas
*   **Corporate Lawyers / Legal Counsel**: Users seeking fast, precise answers to contract questions (e.g., *"What is the limit of liability under the Applied Materials contract?"* or *"What are the termination notice periods?"*). They require exact text citations, section references, and agreement names/years.
*   **RAG Developers**: Engineers looking for a scalable, modular template to index and query unstructured legal agreements or large structured corpuses locally.

---

## 4. Requirements & Features (What should it do?)

### Ingestion Requirements:
*   **Local File Streaming**: Stream contract text files directly from directory trees (e.g., `dataset/contracts/<year>/*.txt`) using a Python generator (keeping RAM usage < 50MB).
*   **Section-Aware Parser**: Segment agreements by section headers and article clauses (e.g., `SECTION ...` or `ARTICLE ...`).
*   **Recursive Sentence-Aligned Chunking**: Group text within sections into chunks of ~400 tokens, aligning boundaries on sentence endings (`. `) with a 2-sentence overlap.

### Database & Retrieval Requirements:
*   **Dense-Sparse Hybrid Index**: Index chunks in Qdrant using both dense embeddings (`BAAI/bge-small-en-v1.5`) and sparse BM25 indices (`Qdrant/bm25`).
*   **Metadata & Payload Indexing**: Store contract name, file name, year, and section name in Qdrant, configuring indexes for fast payload filtering (e.g., filtering by contract year or specific clause sections).

### Agent & Logic Requirements:
*   **Query Expansion (RAG Fusion)**: Generate 3 variations of the search query via Groq (`llama-3.1-8b-instant`).
*   **Reciprocal Rank Fusion (RRF)**: Merge retrieval rankings from query variations, de-duplicating results by `chunk_id`.
*   **Cross-Encoder Reranking**: Re-score the top 20 candidate chunks against the original query using a local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to capture deep contextual alignment.
*   **Grounded Response Generation**: Groq (`llama-3.1-70b-versatile`) writes answers citing source contracts and sections (e.g. `[SOURCE 1: Credit Agreement, SECTION 2.01]`).
*   **Self-Correction Grading**:
    *   **Grounding Grader**: LLM grades if the answer has outside facts (hallucinations).
    *   **Relevance Grader**: LLM grades if the answer matches the query.
    *   **Fallback / Self-Healing**: If grading fails and retry limits are not hit, rewrite the search query and loop back to retrieval.

---

## 5. User Flows or Use Cases

### Use Case 1: Ingesting CUAD Contracts
1.  Developer starts Qdrant via Docker.
2.  Developer runs `python main.py ingest --limit 50`.
3.  The pipeline recursively walks `dataset/contracts`, parses the text files, chunks text at sentence boundaries, generates dense/sparse vectors, and uploads points to Qdrant.

### Use Case 2: Querying the Legal RAG Agent
1.  User inputs query: *"What is the governing law in the 2000 Applied Materials agreement?"*.
2.  LangGraph initiates:
    *   `expand_query`: Rewrites the query.
    *   `retrieve`: Queries Qdrant, fuses lists via RRF.
    *   `rerank`: Re-orders top chunks via local Cross-Encoder.
    *   `generate`: Groq generates a candidate response.
    *   `grader`: Response is graded. If it finds conflicting data (e.g., grounding fails), it rewrites the query and executes the search loop again.
3.  Terminal prints the final grounded answer with contract citations.

---

## 6. Non-Functional Requirements
*   **Performance**:
    *   Ingestion parsing: > 50 contracts per minute.
    *   Retrieval & fusion: < 500ms.
    *   Reranking: < 150ms on standard CPUs.
*   **Security & Safety**:
    *   No external API logging of database contents.
    *   All credentials stored locally in `.env` files.
*   **Memory Efficiency**: Keep local Python script footprint below 1.5GB of RAM during local Cross-Encoder execution.

---

## 7. Out of Scope (What is not included?)
*   **Front-end Web UI**: The interface is strictly a Terminal Command-Line (CLI) interface and Jupyter Notebooks.
*   **Model Fine-Tuning**: No model training or parameter adjustments are performed.

---

## 8. Assumptions, Risks, and Dependencies
*   **Dependencies**: Requires local Docker Desktop/CLI to run Qdrant.
*   **Risks**:
    *   *Groq Rate Limits*: If a loop executes multiple retries, it might hit Groq API rate limits. (Mitigation: Implement retry boundaries and back-offs).
*   **Assumptions**: We assume the local environment has Python 3.9+ and active internet connectivity to download FastEmbed models on the first run.

---

## 9. Success Metrics (How will we know it worked?)
*   **Zero Hallucination Rate**: Grader scores 100% on grounded response tests.
*   **Reranking Delta**: Check that the top-ranked chunk after Cross-Encoder reranking is more semantically aligned than the initial Qdrant vector-distance rank.
*   **Pipeline Completion**: The LangGraph state machine successfully exits and outputs answers within 3 seconds.

---

## 10. Timeline or Milestones (When will it ship?)
*   **Milestone 1 (Day 1)**: Core Setup & Collection Configuration.
*   **Milestone 2 (Day 2)**: Local File Streaming Ingestion & Chunking.
*   **Milestone 3 (Day 3)**: LangGraph state agent node coding & routing setup.
*   **Milestone 4 (Day 4)**: Grader node evaluation, feedback rewrites, and final CLI packaging.
