# Phase 2: Hybrid Indexing in Qdrant Guide (Updated)
**Notebook Target**: `notesbooks/02_qdrant_indexing.ipynb`

Welcome to Phase 2. In this guide, we adapt Qdrant to index our sentence-aligned chunks.

We will learn how to:
1. **Connect to Qdrant** (Docker container or local memory).
2. **Configure a Hybrid Search Collection** (Dense semantic vectors + Sparse BM25 keyword indices).
3. **Setup Payload Indexes** to enable fast metadata filtering on publication date and document section.
4. **Use FastEmbed** to vectorize our chunk text.
5. **Insert data and payloads** directly.
6. **Run test queries** evaluating Dense, Sparse, and Hybrid searches.

---

## 1. Notebook Code Cells

Create `02_qdrant_indexing.ipynb` in the `notesbooks/` directory and add the following cells:

### Cell 1: Imports and Qdrant Setup
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

try:
    client = QdrantClient("http://localhost:6333", timeout=5)
    client.get_collections()
    print("Successfully connected to Qdrant Docker on port 6333.")
except Exception as e:
    print("Could not connect to local Docker Qdrant. Falling back to local in-memory instance.")
    client = QdrantClient(":memory:")

COLLECTION_NAME = "cord19_advanced"
```

### Cell 2: Creating the Hybrid Search Collection
```python
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense-bge": VectorParams(
            size=384,  # Dimension of BAAI/bge-small-en-v1.5
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse-bm25": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=True
            )
        )
    }
)

print(f"Collection '{COLLECTION_NAME}' created and configured successfully for Hybrid Search!")
```

### Cell 2.5: Setting up Payload Indexes (New)
To allow our LangGraph agent to filter papers instantly by section (e.g., only "Methods") or by date, we create payload indexes on these fields:

```python
from qdrant_client.models import PayloadSchemaType

print("Creating payload indexes for metadata filtering...")

# Index the 'date' field (Keyword index for fast comparison / filtering)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="date",
    field_schema=PayloadSchemaType.KEYWORD
)

# Index the 'section' field (Keyword index)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="section",
    field_schema=PayloadSchemaType.KEYWORD
)

print("Payload indexes established successfully!")
```

### Cell 3: Loading FastEmbed Models
```python
from fastembed import TextEmbedding, SparseTextEmbedding

print("Loading Dense Embedding Model (BAAI/bge-small-en-v1.5)...")
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")

print("Loading Sparse BM25 Model (Qdrant/bm25)...")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

print("FastEmbed models loaded and ready!")
```

### Cell 4: Generating Sample Data
Simulated chunks matching the output of our Phase 1 sentence-aligned splitter:

```python
sample_parsed_chunks = [
    {
        "chunk_id": "c1-uuid",
        "chunk_text": "Clinical trials for Remdesivir (GS-5734) in patients hospitalized with COVID-19 have shown faster recovery rates compared to placebo controls. In a double-blind study of 1000 patients, patients receiving Remdesivir had a median recovery time of 10 days compared to 15 days for the placebo group. Antiviral efficacy was most pronounced in patients requiring oxygen support.",
        "title": "Efficacy of Remdesivir in Hospitalized Patients",
        "doi": "10.1056/NEJMoa2007764",
        "date": "2020-10-12",
        "section": "Results"
    },
    {
        "chunk_id": "c2-uuid",
        "chunk_text": "A severe symptom of SARS-CoV-2 is acute respiratory distress syndrome (ARDS), leading to respiratory failure and requiring mechanical ventilation. Cytokine storm, characterized by elevated IL-6 and TNF-alpha, plays a major role in triggering ARDS and causing multi-organ dysfunction in critical patients.",
        "title": "Pathophysiology of SARS-CoV-2 ARDS",
        "doi": "10.1038/s41584-020-00448-w",
        "date": "2020-05-18",
        "section": "Introduction"
    },
    {
        "chunk_id": "c3-uuid",
        "chunk_text": "Chloroquine and Hydroxychloroquine did not demonstrate clinical benefit in randomized trials of hospitalized COVID-19 patients. The RECOVERY trial showed no significant reduction in 28-day mortality for patients allocated to hydroxychloroquine compared to standard care. Consequently, clinical guidelines recommend against their use.",
        "title": "Hydroxychloroquine Trials Review",
        "doi": "10.1056/NEJMoa2022926",
        "date": "2020-11-19",
        "section": "Discussion"
    }
]

print(f"Loaded {len(sample_parsed_chunks)} chunks for indexing.")
```

### Cell 5: Vectorizing and Ingesting to Qdrant
```python
from qdrant_client.models import PointStruct, SparseVector

# 1. Extract texts
chunk_texts = [item["chunk_text"] for item in sample_parsed_chunks]

# 2. Compute embeddings
dense_embeddings = list(dense_model.embed(chunk_texts))
sparse_embeddings = list(sparse_model.embed(chunk_texts))

# 3. Compile Points
points = []
for idx, item in enumerate(sample_parsed_chunks):
    sparse_obj = sparse_embeddings[idx]
    qdrant_sparse = SparseVector(
        indices=sparse_obj.indices.tolist(),
        values=sparse_obj.values.tolist()
    )
    
    point = PointStruct(
        id=idx + 1,
        vector={
            "dense-bge": dense_embeddings[idx].tolist(),
            "sparse-bm25": qdrant_sparse
        },
        payload={
            "chunk_id": item["chunk_id"],
            "chunk_text": item["chunk_text"],
            "title": item["title"],
            "doi": item["doi"],
            "date": item["date"],
            "section": item["section"]
        }
    )
    points.append(point)

# 4. Upload
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} vectors to Qdrant successfully!")
```

### Cell 6: Running Search Queries
```python
query_text = "Does Remdesivir speed up recovery time?"

# 1. Generate query vectors
query_dense = list(dense_model.embed([query_text]))[0].tolist()
query_sparse_obj = list(sparse_model.embed([query_text]))[0]
query_sparse = SparseVector(
    indices=query_sparse_obj.indices.tolist(),
    values=query_sparse_obj.values.tolist()
)

# 2. Search Dense-Only
dense_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=("dense-bge", query_dense),
    limit=2
)
print("=== DENSE SEARCH RESULTS ===")
for res in dense_results:
    print(f"Score: {res.score:.4f} | Document: {res.payload['title']} | Chunk: '{res.payload['chunk_text'][:80]}...'")

# 3. Search Sparse-Only
sparse_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=("sparse-bm25", query_sparse),
    limit=2
)
print("\n=== SPARSE SEARCH RESULTS ===")
for res in sparse_results:
    print(f"Score: {res.score:.4f} | Document: {res.payload['title']} | Chunk: '{res.payload['chunk_text'][:80]}...'")
```
