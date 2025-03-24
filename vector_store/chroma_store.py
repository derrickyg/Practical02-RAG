# vector_store/chroma_store.py

import chromadb

def create_chroma_client():
    return chromadb.HttpClient(host="localhost", port=8000)


def store_embeddings_chroma(embedded_data, collection_name="ds4300_notes"):
    client = create_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    for item in embedded_data:
        collection.add(
            ids=[item["chunk_id"]],
            embeddings=[item["embedding"]],
            documents=[item["text"]],
            metadatas=[{"source": item["source"]}]
        )

    print(f"Stored {len(embedded_data)} chunks in Chroma collection '{collection_name}'")
    return collection

def query_chroma(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
