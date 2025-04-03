import redis
import numpy as np
from redis.commands.search.query import Query


def create_redis_client():
    return redis.Redis(host='localhost', port='6380', decode_responses=True)


def create_redis_index(client, vector_dim, index_name='ds4300_notes'):

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
    print("Index created successfully.")


def store_embeddings_redis(client, embedded_data, index_name='ds4300_notes'):
    create_redis_index(client=client, vector_dim=len(embedded_data), index_name=index_name)

    for idx, item in enumerate(embedded_data):
        key = f"doc:{idx}"
        embedding = np.array(item['embedding'], dtype=np.float32).tobytes()
        client.hset(key, mapping={'text': item['text'], 'embedding': embedding})

    print(f"Stored {len(embedded_data)} chunks in Redis collection '{index_name}'")


def query_redis(client, collection, top_k, query_embedding, index_name='ds4300_notes'):
    q = (
        Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("source", "vector_distance")
        .dialect(2)
    )

    ans = client.ft(index_name).search(
        q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
    )

    return ans

