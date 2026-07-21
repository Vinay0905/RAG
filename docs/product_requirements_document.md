# Product Requirements Document (PRD): Industrial LangGraph Self-Corrective RAG for CORD-19

## 1. Title and Summary
*   **Title**: CORD-19 Self-Corrective RAG Agent
*   **Summary**: A production-grade, local-first Retrieval-Augmented Generation (RAG) system utilizing the **CORD-19 (18.6GB)** scientific corpus. The system leverages **LangGraph** to model an agentic state machine that validates its own generation, **Qdrant** for dense-sparse hybrid search, and **Groq Cloud** for high-speed generation.

---

## 2. Problem Statement or Objective (Goal)
Scientific researchers and clinical analysts face immense bottlenecks when querying massive literature collections (like CORD-19, which contains hundreds of thousands of papers on coronaviruses). 
*   **Naive RAG Limitations**: Standard RAG yields "vague" semantic matches, cuts text blindly at character limits, and suffers from LLM hallucinations.
*   **Dataset Scale**: The 18.6GB size makes simple in-memory indexing impossible on standard local machines.
*   **Objective**: Build a system that ingests the dataset locally using low-memory generators, runs precise dense-sparse vector lookups, reranks them using a local Cross-Encoder transformer, and implements self-correction logic (fact-checking) in LangGraph to ensure final responses are 100% grounded in literature.

---

## 3. Target Users / Personas
*   **Medical Researchers / Epidemiologists**: Users seeking fast, precise answers to complex clinical questions (e.g., *"What is the efficacy of Remdesivir against placebo in hospitalized patients?"*). They require exact text citations, publication dates, and DOIs for validation.
*   **RAG Developers**: Engineers looking for a scalable, modular template to index and query large unstructured documents locally.

---

## 4. Requirements & Features (What should it do?)

### Ingestion Requirements:
*   **Local File Streaming**: Ingest the 18.6GB CORD-19 dataset by streaming rows from `metadata.csv` (keeping RAM usage < 50MB) and resolving paths to local full-text JSON files.
*   **Section-Aware Parser**: Segment papers by their logical structural headers (`Abstract`, `Introduction`, `Methods`, `Results`, `Discussion`, `Conclusion`).
*   **Recursive Sentence-Aligned Chunking**: Group text within sections into chunks of ~400 tokens, aligning boundaries on sentence endings (`. `) with a 2-sentence overlap.

### Database & Retrieval Requirements:
*   **Dense-Sparse Hybrid Index**: Index chunks in Qdrant using both dense embeddings (`BAAI/bge-small-en-v1.5`) and sparse BM25 indices (`Qdrant/bm25`).
*   **Metadata & Payload Indexing**: Store publication dates, titles, DOIs, and section headers in Qdrant, configuring indexes for fast payload filtering (e.g., filtering by dates or sections).

### Agent & Logic Requirements:
*   **Query Expansion (RAG Fusion)**: Generate 3 variations of the search query via Groq (`llama-3.1-8b-instant`).
*   **Reciprocal Rank Fusion (RRF)**: Merge retrieval rankings from query variations, de-duplicating results by `chunk_id`.
*   **Cross-Encoder Reranking**: Re-score the top 20 candidate chunks against the original query using a local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to capture deep contextual alignment.
*   **Grounded Response Generation**: Groq (`llama-3.1-70b-versatile`) writes answers citing source indexes (e.g. `[SOURCE 1]`).
*   **Self-Correction Grading**:
    *   **Grounding Grader**: LLM grades if the answer has outside facts (hallucinations).
    *   **Relevance Grader**: LLM grades if the answer matches the query.
    *   **Fallback / Self-Healing**: If grading fails and retry limits are not hit, rewrite the search query and loop back to retrieval.

---

## 5. User Flows or Use Cases

### Use Case 1: Ingesting CORD-19
1.  Developer starts Qdrant via Docker.
2.  Developer runs `python main.py ingest --limit 1000`.
3.  The pipeline opens `metadata.csv`, parses the first 1000 JSONs, chunks text at sentence boundaries, generates dense/sparse vectors, and uploads points to Qdrant.

### Use Case 2: Querying the Self-Corrective Agent
1.  User inputs query: *"Efficacy of hydroxychloroquine trials"*.
2.  LangGraph initiates:
    *   `expand_query`: Rewrites the query.
    *   `retrieve`: Queries Qdrant, fuses lists via RRF.
    *   `rerank`: Re-orders top chunks via local Cross-Encoder.
    *   `generate`: Groq generates a candidate response.
    *   `grader`: Response is graded. If it finds conflicting data (e.g., grounding fails), it rewrites the query and executes the search loop again.
3.  Terminal prints the final grounded answer with citations.

---

## 6. Non-Functional Requirements
*   **Performance**:
    *   Ingestion parsing: > 50 documents per minute.
    *   Retrieval & fusion: < 500ms.
    *   Reranking: < 150ms on standard CPUs.
*   **Security & Safety**:
    *   No external API logging of database contents.
    *   All credentials stored locally in `.env` files.
*   **Memory Efficiency**: Keep local Python script footprint below 1.5GB of RAM during local Cross-Encoder execution.

---

## 7. Out of Scope (What is not included?)
*   **Front-end Web UI**: The interface is strictly a Terminal Command-Line (CLI) interface and Jupyter Notebooks.
*   **Live Dataset Sync**: We work with a pre-downloaded, static zip of the CORD-19 corpus.
*   **Model Fine-Tuning**: No model training or parameter adjustments are performed.

---

## 8. Assumptions, Risks, and Dependencies
*   **Dependencies**: Requires local Docker Desktop/CLI to run Qdrant.
*   **Risks**:
    *   *Groq Rate Limits*: If a loop executes multiple retries, it might hit Groq API rate limits. (Mitigation: Implement retry boundaries and back-offs).
    *   *Disk Space*: CORD-19 requires > 30GB of unpacked storage space.
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
