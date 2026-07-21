# Phase 1: Local Ingestion & Recursive Sentence-Aligned Chunking Guide
**Notebook Target**: `notesbooks/01_ingestion_chunking.ipynb`

Welcome to Phase 1. In this updated guide, we implement a highly intuitive, production-ready **Recursive Sentence-Aligned Chunking** strategy to ingest our CORD-19 data.

We will learn how to:
1. **Stream CORD-19 locally** from `metadata.csv` using a generator.
2. **Parse structured sections** (`Abstract`, `Methods`, etc.).
3. **Recursively split text** at sentence boundaries (`. `) to build chunks of ~400 tokens, avoiding cutting sentences in half and removing parent-child mapping complexity.

---

## 1. Directory Structure

Ensure your downloaded dataset directory (`dataset/`) is placed in the project root:
```
RAG/
├── dataset/
│   ├── metadata.csv
│   └── document_parses/
├── notesbooks/
│   └── 01_ingestion_chunking.ipynb
└── docs/
    └── 01_ingestion_chunking_guide.md
```

---

## 2. Notebook Code Cells

Create a new notebook named `01_ingestion_chunking.ipynb` in the `notesbooks/` directory and copy the following cells:

### Cell 1: Imports and Helper Functions
Initializes libraries and the token length calculator.

```python
import os
import csv
import json
import uuid
import tiktoken
from tqdm import tqdm

# Initialize the encoder
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_len(text: str) -> int:
    """Helper to return the token length of a given text string."""
    return len(tokenizer.encode(text))

print("Prerequisites imported successfully!")
```

### Cell 2: Recursive Sentence-Aligned Chunk Splitter
This groups sentences together until they reach a target size (e.g., 400 tokens), carrying over a sentence overlap (e.g., 2 sentences) to maintain continuity.

```python
def split_recursive_sentence(text: str, max_tokens: int = 400, sentence_overlap: int = 2) -> list:
    """
    Splits text into chunks of max_tokens, aligning cuts on sentence boundaries.
    Applies an overlap of 'sentence_overlap' sentences at boundaries.
    """
    # Clean text whitespace
    text = " ".join(text.split())
    if not text:
        return []
        
    # Split text into sentences
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
            s_len = get_token_len(s_text)
            
            # If a single sentence exceeds max_tokens, we force-split it by words
            if s_len > max_tokens:
                if current_chunk_sentences:
                    break  # Commit current chunk first
                # Split single giant sentence by words
                words = s_text.split(" ")
                w_idx = 0
                while w_idx < len(words):
                    sub_words = words[w_idx : w_idx + max_tokens]
                    chunks.append(" ".join(sub_words))
                    w_idx += max_tokens
                j += 1
                i = j
                break
                
            if current_tokens + s_len <= max_tokens:
                current_chunk_sentences.append(s_text)
                current_tokens += s_len
                j += 1
            else:
                break
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        # If we reached the end, terminate
        if j >= num_sentences:
            break
            
        # Step forward, subtracting the overlap count to maintain context continuity
        i = max(j - sentence_overlap, i + 1)
        
    return chunks
```

### Cell 3: Local CORD-19 Generator Stream
Streams metadata CSV and reads relative JSON file paths from disk.

```python
def stream_local_cord19(dataset_dir: str, limit: int = 100):
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")
        
    with open(metadata_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            if count >= limit:
                break
            
            json_paths = row.get("pdf_json_files") or row.get("pmc_json_files")
            body_sections = []
            
            if json_paths:
                first_rel_path = json_paths.split(";")[0].strip()
                full_json_path = os.path.join(dataset_dir, first_rel_path)
                
                if os.path.exists(full_json_path):
                    try:
                        with open(full_json_path, mode="r", encoding="utf-8") as jf:
                            json_data = json.load(jf)
                            raw_body = json_data.get("body_text", [])
                            for section in raw_body:
                                body_sections.append({
                                    "title": section.get("section", "Section"),
                                    "text": section.get("text", "")
                                })
                    except Exception:
                        continue
            
            yield {
                "doc_id": row.get("cord_uid") or str(uuid.uuid4()),
                "title": row.get("title") or "Untitled Document",
                "doi": row.get("doi") or "N/A",
                "date": row.get("publish_time") or "N/A",
                "abstract": row.get("abstract") or "",
                "body": body_sections
            }
            count += 1
```

### Cell 4: Section-Aware Parser & Ingestion Execution
We run the parser over the first **5 local files**.

```python
def parse_local_document(doc: dict) -> list:
    metadata = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "doi": doc["doi"],
        "date": doc["date"]
    }
    all_chunks = []
    
    # 1. Parse Abstract
    abstract_text = doc["abstract"]
    if abstract_text and len(abstract_text.strip()) > 50:
        abs_chunks = split_recursive_sentence(abstract_text)
        for chunk_text in abs_chunks:
            chunk = {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text}
            chunk.update(metadata)
            chunk["section"] = "Abstract"
            all_chunks.append(chunk)
            
    # 2. Parse Body Sections
    for idx, section in enumerate(doc["body"]):
        title = section["title"]
        text = section["text"]
        if text and len(text.strip()) > 50:
            sec_chunks = split_recursive_sentence(text)
            for chunk_text in sec_chunks:
                chunk = {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text}
                chunk.update(metadata)
                chunk["section"] = title
                all_chunks.append(chunk)
                
    return all_chunks

DATASET_PATH = "/Users/mast/Documents/VInayPrograming/RAG/dataset"

print(f"Reading local documents from: {DATASET_PATH}")
sample_chunks = []

doc_stream = stream_local_cord19(DATASET_PATH, limit=5)
for doc in tqdm(doc_stream, total=5, desc="Parsing local files"):
    doc_chunks = parse_local_document(doc)
    sample_chunks.extend(doc_chunks)

print(f"\nGenerated a total of {len(sample_chunks)} Sentence-Aligned chunks from local files.")
```

### Cell 5: Inspect Ingestion Output
Let's print a sample chunk to verify.

```python
if sample_chunks:
    sample = sample_chunks[0]
    print("=== INSPECTING LOCAL SAMPLE CHUNK ===")
    print(f"Document Title : {sample['title']}")
    print(f"Section Name   : {sample['section']}")
    print(f"Chunk ID       : {sample['chunk_id']}")
    print(f"Chunk Tokens   : {get_token_len(sample['chunk_text'])} tokens")
    print("\n--- Chunk Text ---")
    print(sample['chunk_text'])
else:
    print("No chunks generated. Make sure CORD-19 is downloaded to the dataset/ folder.")
```
