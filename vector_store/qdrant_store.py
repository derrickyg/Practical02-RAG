from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

def store_embeddings_qdrant(embedded_data, collection_name):
    vectors = [
        PointStruct(id=i, vector=chunk["embedding"], payload=chunk)
        for i, chunk in enumerate(embedded_data)
    ]

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embedded_data[0]["embedding"]), distance=Distance.COSINE)
    )

    client.upsert(collection_name=collection_name, points=vectors)
    print(f"Stored {len(vectors)} vectors in Qdrant collection '{collection_name}'")

def query_qdrant(collection_name, query_vector, top_k=5):
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [r.payload["text"] for r in results]

def delete_qdrant_collection(collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f" Deleted Qdrant collection '{collection_name}'")
