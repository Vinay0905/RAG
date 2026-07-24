import os 
from dotenv import load_dotenv
load_dotenv()
DATASET_PATH=os.getenv("DATASET_PATH","/Users/mast/Documents/VInayPrograming/RAG/dataset")
QDRANT_HOST=os.getenv("QDRANT_HOST","http://locolhost:6333")
COLLECTION_NAME="cuad_advanced"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq Configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXPANSION_MODEL = "llama-3.1-8b-instant"
GENERATION_MODEL = "llama-3.1-70b-versatile"
GRADER_MODEL = "llama-3.1-8b-instant"

# RAG Hyperparameters
MAX_TOKENS = 400
SENTENCE_OVERLAP = 2
MAX_RETRIES = 2
