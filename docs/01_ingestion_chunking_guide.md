# Phase 1: Local Ingestion & Recursive Sentence-Aligned Chunking Guide
**Notebook Target**: `notesbooks/01_ingestion_chunking.ipynb`

Welcome to Phase 1. In this updated guide, we implement a highly intuitive, production-ready **Recursive Sentence-Aligned Chunking** strategy to ingest our CUAD (Contract Understanding Atticus Dataset) commercial legal agreements.

We will learn how to:
1. **Stream CUAD contracts locally** from subdirectories in `dataset/contracts/` using a generator.
2. **Parse structured sections** (`Preamble`, `SECTION ...`, `ARTICLE ...`, etc.) from raw text files.
3. **Recursively split text** at sentence boundaries (`. `) to build chunks of ~400 tokens, avoiding cutting sentences in half and maintaining context.

---

## 1. Directory Structure

Ensure your downloaded dataset directory (`dataset/`) is placed in the project root:
```
RAG/
├── dataset/
│   └── contracts/
│       ├── 2000/
│       ├── 2001/
│       └── ...
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
import re
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

### Cell 3: Local CUAD Contract Generator Stream
Recursively walks `dataset/contracts/` to stream text agreements along with metadata (year, contract name, etc.).

```python
def stream_local_cuad(dataset_dir: str, limit: int = 100):
    # Support passing either the parent directory or the 'contracts' directory itself
    if os.path.basename(dataset_dir.rstrip("/")) == "contracts":
        contracts_dir = dataset_dir
    else:
        contracts_dir = os.path.join(dataset_dir, "contracts")
        
    if not os.path.exists(contracts_dir):
        raise FileNotFoundError(f"Contracts directory not found at: {contracts_dir}")
        
    count = 0
    for root, dirs, files in os.walk(contracts_dir):
        for file in sorted(files):
            if file.endswith(".txt"):
                if count >= limit:
                    break
                
                file_path = os.path.join(root, file)
                # In CUAD directory structure, the parent folder name represents the contract year
                year = os.path.basename(root)
                
                try:
                    with open(file_path, mode="r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    continue
                
                # Determine contract name/title from first non-empty lines
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
```

### Cell 4: Section-Aware Contract Parser & Ingestion Execution
Reads raw contract text and segments it into sections based on typical contract headings (e.g. `SECTION ...` or `ARTICLE ...`), then runs the sentence-aligned splitter on the text within each section. We run the parser over the first **3 contracts**.

```python
def parse_cuad_document(doc: dict) -> list:
    text = doc["text"]
    metadata = {
        "doc_id": doc["doc_id"],
        "contract_name": doc["contract_name"],
        "file_name": doc["file_name"],
        "year": doc["year"]
    }
    
    # Identify Section or Article patterns
    section_pattern = re.compile(r'^(SECTION\s+\d+\.\d+|ARTICLE\s+[IVXLCDM]+|EXHIBIT\s+[A-Z]|\bINDEMNITY\b|\bTERMINATION\b|\bLIMITATION\b)', re.IGNORECASE)
    
    lines = text.split("\n")
    current_section = "Preamble"
    section_text_blocks = []
    current_block = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check if line indicates a section boundary / heading
        if section_pattern.match(stripped) and len(stripped) < 80:
            if current_block:
                section_text_blocks.append((current_section, "\n".join(current_block)))
                current_block = []
            current_section = stripped
        else:
            current_block.append(stripped)
            
    if current_block:
        section_text_blocks.append((current_section, "\n".join(current_block)))
        
    all_chunks = []
    for section_name, section_text in section_text_blocks:
        if len(section_text.strip()) > 50:
            chunks = split_recursive_sentence(section_text)
            for chunk_text in chunks:
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_text": chunk_text,
                    "section": section_name
                }
                chunk.update(metadata)
                all_chunks.append(chunk)
                
    return all_chunks

DATASET_PATH = "/Users/mast/Documents/VInayPrograming/RAG/dataset"

print(f"Reading local CUAD documents from: {DATASET_PATH}")
sample_chunks = []

doc_stream = stream_local_cuad(DATASET_PATH, limit=3)
for doc in tqdm(doc_stream, total=3, desc="Parsing contract text"):
    doc_chunks = parse_cuad_document(doc)
    sample_chunks.extend(doc_chunks)

print(f"\nGenerated a total of {len(sample_chunks)} Sentence-Aligned chunks from contracts.")
```

### Cell 5: Inspect Ingestion Output
Let's print a sample chunk to verify.

```python
if sample_chunks:
    sample = sample_chunks[0]
    print("=== INSPECTING LOCAL SAMPLE CHUNK ===")
    print(f"Contract Name : {sample['contract_name']}")
    print(f"File Name     : {sample['file_name']}")
    print(f"Year          : {sample['year']}")
    print(f"Section Name  : {sample['section']}")
    print(f"Chunk ID      : {sample['chunk_id']}")
    print(f"Chunk Tokens  : {get_token_len(sample['chunk_text'])} tokens")
    print("\n--- Chunk Text ---")
    print(sample['chunk_text'])
else:
    print("No chunks generated. Make sure CUAD is downloaded to the dataset/ contracts directory.")
```
