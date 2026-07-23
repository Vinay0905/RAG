from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

def test_qdrant_connection():
    print("Attempting to connect to local Qdrant container...")
    try:
        # Explicitly connect to the IPv4 loopback IP on port 6333
        client = QdrantClient(url="http://127.0.0.1:6333", timeout=5, check_compatibility=False)
        client.get_collections()
        print("🎉 SUCCESS: Connected to Qdrant Docker on http://127.0.0.1:6333!")
    except Exception as e:
        print("⚠️ FALLBACK: Could not connect to Docker Qdrant. Using in-memory database.")
        print(f"Connection error details: {e}")
        client = QdrantClient(":memory:")
    
    collection_name = "cuad_advanced"
    print(f"\nSetting up collection '{collection_name}'...")
    
    # Recreate the collection using the modern API
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense-bge": VectorParams(
                size=384,
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
    print(f"🎉 SUCCESS: Collection '{collection_name}' is active and ready!")

if __name__ == "__main__":
    test_qdrant_connection()
