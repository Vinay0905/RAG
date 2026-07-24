# Phase 2 Study Guide: Pipeline 1 — Data Ingestion Pipeline

Welcome to **Phase 2**! In this phase, you will build **Pipeline 1 (Data Ingestion Pipeline)** as shown in standard enterprise RAG architectures:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE 1: DATA INGESTION PIPELINE                  │
│                                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐    ┌─────┐  │
│  │ DATA INGEST  ├────►│ DATA PARSING ├────►│  EMBEDDING  ├───►│Vector│ │
│  │ (Read Data)  │     │  (Chunking)  │     │(Text->Vecs) │    │Store│  │
│  └──────────────┘     └──────────────┘     └─────────────┘    └─────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure for Phase 2 (Pipeline 1)

Ensure the following subdirectories exist inside `src/ingestion/`:

```text
src/ingestion/
├── __init__.py
├── loaders/
│   ├── __init__.py
│   ├── base.py
│   ├── txt_loader.py
│   └── pdf_loader.py
├── parsers/
│   ├── __init__.py
│   └── legal_parser.py
├── chunkers/
│   ├── __init__.py
│   ├── sentence_chunker.py
│   └── parent_child_chunker.py
├── extractors/
│   ├── __init__.py
│   └── metadata_extractor.py
└── pipeline.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### Step 1: DATA INGESTION — File Loaders

#### File 1.1: `src/ingestion/loaders/base.py`
**Location**: `src/ingestion/loaders/base.py`  
**Purpose**: Base document loader defining standard interface for reading files.

```python
import os
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLoader(ABC):
    """Abstract base loader for document format readers."""

    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Reads document and extracts text, metadata, and hash."""
        pass

    def compute_sha256(self, file_path: str) -> str:
        """Calculates SHA-256 checksum of raw file for differential tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
```

> **Deep 2-Line Explanation**:  
> *Defines an abstract loader equipped with SHA-256 binary checksum computation for tracking file modifications.*  
> *Enables differential ingestion so our vector database never wastes compute re-indexing unchanged documents.*

---

#### File 1.2: `src/ingestion/loaders/txt_loader.py`
**Location**: `src/ingestion/loaders/txt_loader.py`  
**Purpose**: Handles reading UTF-8 encoded text files (such as raw CUAD contract `.txt` files).

```python
import os
import uuid
from typing import Dict, Any
from src.ingestion.loaders.base import BaseLoader
from src.core.exceptions import IngestionError


class TextDocumentLoader(BaseLoader):
    """Loader for plaintext contract agreements (.txt files)."""

    def load(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise IngestionError(f"File not found: {file_path}")

        try:
            with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            raise IngestionError(f"Failed to read file {file_path}: {str(e)}")

        file_name = os.path.basename(file_path)
        year = os.path.basename(os.path.dirname(file_path))
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        contract_title = lines[0] if lines else file_name

        if len(contract_title) > 120:
            contract_title = contract_title[:117] + "..."

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path))
        checksum = self.compute_sha256(file_path)

        return {
            "doc_id": doc_id,
            "contract_name": contract_title,
            "file_name": file_name,
            "year": year,
            "file_path": file_path,
            "checksum": checksum,
            "text": text
        }
```

> **Deep 2-Line Explanation**:  
> *Extracts full document text, infers contract titles and year metadata, and assigns deterministic DNS UUIDs.*  
> *Ensures bulletproof text loading even when encountering minor encoding errors in large legal corpuses.*

---

### Step 2: DATA PARSING & CHUNKING — Section Parser & Chunkers

#### File 2.1: `src/ingestion/parsers/legal_parser.py`
**Location**: `src/ingestion/parsers/legal_parser.py`  
**Purpose**: Regex-based section boundary parser that splits legal agreements into logical articles and clauses.

```python
import re
from typing import List, Tuple


class LegalContractParser:
    """Parses raw agreement text into structured sections based on legal headers."""

    HEADER_PATTERN = re.compile(
        r'^(SECTION\s+\d+(\.\d+)?|ARTICLE\s+[IVXLCDM\d]+|EXHIBIT\s+[A-Z]|\bINDEMNIFICATION\b|\bTERMINATION\b|\bGOVERNING LAW\b|\bLIMITATION OF LIABILITY\b|\bCONFIDENTIALITY\b)',
        re.IGNORECASE
    )

    def parse_sections(self, text: str) -> List[Tuple[str, str]]:
        """Splits full contract text into (section_header, section_text) tuples."""
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        current_section = "Preamble"
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if self.HEADER_PATTERN.match(stripped) and len(stripped) < 90:
                if current_lines:
                    sections.append((current_section, "\n".join(current_lines)))
                    current_lines = []
                current_section = stripped
            else:
                current_lines.append(stripped)

        if current_lines:
            sections.append((current_section, "\n".join(current_lines)))

        return sections
```

> **Deep 2-Line Explanation**:  
> *Scans text for standard legal headers (Articles, Sections, Indemnifications, Governing Law) to chunk along actual clause boundaries.*  
> *Prevents arbitrary context splits so clauses like 'Limitation of Liability' remain intact within a single logical block.*

---

#### File 2.2: `src/ingestion/chunkers/parent_child_chunker.py`
**Location**: `src/ingestion/chunkers/parent_child_chunker.py`  
**Purpose**: Creates small child vectors linked to large parent section contexts (Small-to-Big Retrieval).

```python
import uuid
from typing import List, Dict, Any
from src.ingestion.parsers.legal_parser import LegalContractParser
from src.ingestion.chunkers.sentence_chunker import SentenceAlignedChunker


class ParentChildChunker:
    """Generates small child chunks for vector indexing linked to parent section context."""

    def __init__(self):
        self.parser = LegalContractParser()
        self.child_chunker = SentenceAlignedChunker()

    def process_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sections = self.parser.parse_sections(doc_data["text"])
        final_chunks = []

        for section_name, section_text in sections:
            if len(section_text.strip()) < 40:
                continue

            parent_id = str(uuid.uuid4())
            child_text_blocks = self.child_chunker.split_text(section_text)

            for idx, c_text in enumerate(child_text_blocks):
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "parent_id": parent_id,
                    "doc_id": doc_data["doc_id"],
                    "contract_name": doc_data["contract_name"],
                    "file_name": doc_data["file_name"],
                    "year": doc_data["year"],
                    "section": section_name,
                    "chunk_index": idx,
                    "chunk_text": c_text,
                    "parent_context": section_text[:1500]  # Store full parent section for LLM context
                }
                final_chunks.append(chunk)

        return final_chunks
```

> **Deep 2-Line Explanation**:  
> *Links small, precise child vector chunks (indexed for vector search) to full parent section text (passed to the LLM).*  
> *Solves the 'Lost in the Middle' RAG problem by retrieving granular matches while feeding complete section background to the generator.*

---

### Step 3: EMBEDDING & VECTORSTORE STORAGE — Master Pipeline

#### File 3.1: `src/ingestion/pipeline.py`
**Location**: `src/ingestion/pipeline.py`  
**Purpose**: Master Data Ingestion Pipeline connecting Loaders -> Parsing/Chunking -> Embedding -> VectorStore.

```python
import os
import json
from typing import List, Dict, Any, Generator
from tqdm import tqdm
from config.settings import settings
from src.ingestion.loaders.txt_loader import TextDocumentLoader
from src.ingestion.chunkers.parent_child_chunker import ParentChildChunker
from src.ingestion.extractors.metadata_extractor import LegalMetadataExtractor
from src.embeddings.factory import EmbeddingProviderFactory
from src.vectorstores.factory import VectorStoreFactory
from src.core.logging import logger
from src.core.telemetry import tracer


class DataIngestionPipeline:
    """PIPELINE 1: Data Ingestion -> Data Parsing -> Embedding -> VectorStore Storage."""

    def __init__(self):
        self.txt_loader = TextDocumentLoader()
        self.chunker = ParentChildChunker()
        self.extractor = LegalMetadataExtractor()
        self.embedder = EmbeddingProviderFactory.get_provider("fastembed")
        self.store = VectorStoreFactory.get_store("qdrant")
        self.checksum_registry_file = os.path.join(settings.DATASET_PATH, ".checksum_registry.json")
        self.processed_checksums = self._load_checksum_registry()

    def _load_checksum_registry(self) -> Dict[str, str]:
        if os.path.exists(self.checksum_registry_file):
            try:
                with open(self.checksum_registry_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_checksum_registry(self) -> None:
        with open(self.checksum_registry_file, "w") as f:
            json.dump(self.processed_checksums, f, indent=2)

    def stream_documents(self, limit: int = 100) -> Generator[Dict[str, Any], None, None]:
        contracts_dir = os.path.join(settings.DATASET_PATH, "contracts")
        if not os.path.exists(contracts_dir):
            contracts_dir = settings.DATASET_PATH

        count = 0
        for root, _, files in os.walk(contracts_dir):
            for file in sorted(files):
                if file.endswith(".txt"):
                    if count >= limit:
                        return
                    file_path = os.path.join(root, file)
                    doc_data = self.txt_loader.load(file_path)

                    if self.processed_checksums.get(file_path) == doc_data["checksum"]:
                        logger.info(f"⏭️ Skipping unchanged file: {file}")
                        continue

                    yield doc_data
                    self.processed_checksums[file_path] = doc_data["checksum"]
                    count += 1

    @tracer.trace_span("ingestion_pipeline_run")
    def run(self, limit: int = 100) -> List[Dict[str, Any]]:
        logger.info(f"⚡ [PIPELINE 1 START] Ingesting documents (limit={limit})...")
        self.store.setup_collection()
        all_chunks = []

        for doc in tqdm(self.stream_documents(limit), desc="Processing Documents"):
            chunks = self.chunker.process_document(doc)
            for c in chunks:
                c["clause_tags"] = self.extractor.extract_clause_tags(c["chunk_text"])
                all_chunks.append(c)

        if all_chunks:
            logger.info("Generating Dense and Sparse Embeddings for VectorStore...")
            texts = [c["chunk_text"] for c in all_chunks]
            dense_vectors = self.embedder.embed_dense(texts)
            sparse_vectors = self.embedder.embed_sparse(texts)

            logger.info("Upserting vectors and metadata into Qdrant VectorStore...")
            self.store.upsert_chunks(all_chunks, dense_vectors, sparse_vectors)

        self._save_checksum_registry()
        logger.info(f"✅ [PIPELINE 1 COMPLETE] Ingested and stored {len(all_chunks)} chunks in VectorStore.")
        return all_chunks
```

> **Deep 2-Line Explanation**:  
> *Implements Pipeline 1: reading raw text documents, parsing clauses into parent-child chunks, vectorizing text, and saving to Qdrant.*  
> *Tracks file checksums so re-running Pipeline 1 instantly skips already-indexed contract files.*

---

## 🎯 Phase 2 Checkpoint Verification

To verify Pipeline 1:
```python
from src.ingestion.pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline()
chunks = pipeline.run(limit=5)
print(f"Pipeline 1 Success! Ingested {len(chunks)} chunks into VectorStore.")
```

Next, see **Pipeline 2: Retrieval & Generation Pipeline** in [docs/Phase4_Hybrid_Retrieval_Reranking.md](file:///Users/mast/Documents/VInayPrograming/RAG/docs/Phase4_Hybrid_Retrieval_Reranking.md) and [docs/Dual_Pipeline_Architecture.md](file:///Users/mast/Documents/VInayPrograming/RAG/docs/Dual_Pipeline_Architecture.md)!
