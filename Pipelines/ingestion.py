
import os 
import re
import uuid
import tiktoken
from tqdm import tqdm
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector, PayloadSchemaType,SparseTextEmbedding
from fastembed import TextEmbedding,SparseEmbedding
import config

tokenizer=tiktoken.get_encoding("cl100k_base")
class IngestPipeline:
    def __init__(self):
        print("Initializing Qdrant Local Ingestion pipeline...")
        self.client = QdrantClient(config.QDRANT_HOST)
        self.dense_model = TextEmbedding(config.DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(config.SPARSE_MODEL_NAME)

    def setup_collection(self):
        print(f"Setting up Qdrant Collection: {config.COLLECTION_NAME}")
        self.client.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                "dense-bge": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse-bm25": SparseVectorParams(index=SparseIndexParams(on_disk=True))
            }
        )

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




    