import time
import pandas as pd
from scripts.ingest import ingest_documents
from scripts.embed import embed_chunks
from scripts.llm_respond import generate_answer_ollama
from vector_store.chroma_store import store_embeddings_chroma, query_chroma, delete_chroma_collection
from vector_store.qdrant_store import store_embeddings_qdrant, query_qdrant, delete_qdrant_collection
from vector_store.redis_store import store_embeddings_redis, query_redis, flush_redis_index
from sentence_transformers import SentenceTransformer


def run_experiment(params, save_to_csv=True, csv_path="results.csv"):
    # Pre-step: Clear vector DB before starting timer
    collection_name = f"ds4300_{params['vector_db']}"
    if params["vector_db"] == "chroma":
        try:
            delete_chroma_collection(collection_name)
            print(f"Deleted Chroma collection '{collection_name}'")
        except Exception:
            print(f"Collection '{collection_name}' does not exist, skipping delete.")

    elif params["vector_db"] == "redis":
        try:
            flush_redis_index(collection_name)
        except Exception:
            print(f"Redis index '{collection_name}' does not exist or failed to delete, skipping.")

    elif params["vector_db"] == "qdrant":
        try:
            delete_qdrant_collection(collection_name)
        except Exception:
            print(f"Qdrant collection '{collection_name}' does not exist or failed to delete, skipping.")

    # Step 1: Ingest and chunk
    t0 = time.time()
    chunked_data = ingest_documents(
        notes_path='Notes/',
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        text_prep=params.get("text_prep", None)
    )
    t1 = time.time()

    # Step 2: Embed
    embedded_data = embed_chunks(
        chunked_data,
        model_key=params["embedding_model"]
    )
    t2 = time.time()

    # Step 3: Store in Vector DB
    if params["vector_db"] == "chroma":
        collection = store_embeddings_chroma(embedded_data, collection_name=collection_name)
        query_fn = lambda embedding: query_chroma(collection, embedding, top_k=5)["documents"][0]
    elif params["vector_db"] == "redis":
        client = None
        from vector_store.redis_store import create_redis_client
        client = create_redis_client()
        store_embeddings_redis(client, embedded_data, index_name=collection_name)
        query_fn = lambda embedding: query_redis(client, top_k=5, query_embedding=embedding, index_name=collection_name)
    elif params["vector_db"] == "qdrant":
        store_embeddings_qdrant(embedded_data, collection_name=collection_name)
        query_fn = lambda embedding: query_qdrant(collection_name, embedding, top_k=5)
    else:
        raise ValueError("Unsupported vector DB")
    t3 = time.time()

    # Step 4: Query
    model_map = {
        "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "instructor": "hkunlp/instructor-xl"
    }
    model = SentenceTransformer(model_map[params["embedding_model"]])
    query_embedding = model.encode(params["query"])
    retrieved_chunks = query_fn(query_embedding)
    t4 = time.time()

    # Step 5: LLM response
    llm_answer = generate_answer_ollama(retrieved_chunks, params["query"], model=params["llm_model"])
    t5 = time.time()

    # Step 6: Log to CSV
    if save_to_csv:
        row = {
            "chunk_size": params["chunk_size"],
            "chunk_overlap": params["chunk_overlap"],
            "embedding_model": params["embedding_model"],
            "text_prep": params.get("text_prep", None),
            "vector_db": params["vector_db"],
            "llm_model": params["llm_model"],
            "query": params["query"],
            "retrieved_chunks": "\n\n".join(retrieved_chunks),
            "llm_answer": llm_answer,
            "time_ingest_chunk": round(t1 - t0, 2),
            "time_embed": round(t2 - t1, 2),
            "time_store": round(t3 - t2, 2),
            "time_query": round(t4 - t3, 2),
            "time_llm": round(t5 - t4, 2),
            "total_elapsed": round(t5 - t0, 2)
        }

        try:
            df_existing = pd.read_csv(csv_path)
            df = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame([row])

        df.to_csv(csv_path, index=False)

    return llm_answer
