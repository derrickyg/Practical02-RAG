import redis
import numpy as np
from redis.commands.search.query import Query


def create_redis_client():
    return redis.Redis(host='localhost', port=6380, decode_responses=False)


def flush_redis_index(index_name):
    client = create_redis_client()
    try:
        client.execute_command(f"FT.DROPINDEX {index_name} DD")
        print(f"Flushed Redis index '{index_name}'")
    except redis.exceptions.ResponseError:
        print(f"Index '{index_name}' does not exist, skipping flush.")


def create_redis_index(client, vector_dim, index_name):
    try:
        client.execute_command(f"FT.DROPINDEX {index_name} DD")
    except redis.exceptions.ResponseError:
        pass

    client.execute_command(
        f"""
        FT.CREATE {index_name} ON HASH PREFIX 1 doc:
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {vector_dim} TYPE FLOAT32 DISTANCE_METRIC COSINE
        """
    )
    print(f"Index '{index_name}' created successfully.")


def store_embeddings_redis(client, embedded_data, index_name='ds4300_notes'):
    vector_dim = len(embedded_data[0]['embedding'])
    create_redis_index(client, vector_dim=vector_dim, index_name=index_name)

    for idx, item in enumerate(embedded_data):
        key = f"doc:{idx}"
        embedding = np.array(item['embedding'], dtype=np.float32).tobytes()
        client.hset(key, mapping={'text': item['text'], 'embedding': embedding})

    print(f"Stored {len(embedded_data)} chunks in Redis index '{index_name}'")


def query_redis(client, query_embedding, top_k=5, index_name='ds4300_notes'):
    q = (
        Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )

    response = client.ft(index_name).search(
        q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
    )

    # Return the 'text' field from each result
    return [doc.text for doc in response.docs]
