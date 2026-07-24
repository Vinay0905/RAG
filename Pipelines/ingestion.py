
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
    def stream_local_document(self,limit:int ):
        if os.path.basename(config.DATASET_PATH.rstrip("/"))=="contracts":
            contracts_dir=config.DATASET_PATH
        else:
            contracts_dir=os.path.join(config.DATASET_PATH,"contracts")
        if not os.path.exists(contracts_dir):
            raise FileNotFoundError (f"Contracts directory not found at {contracts_dir}")
        count=0
        for root,dirs,files in os.walk(contracts_dir):
            

