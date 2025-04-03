# vector_store/milvus_store.py

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np

def connect_milvus():
    connections.connect(host="localhost", port="19530")

def create_milvus_collection(collection_name="ds4300_notes", dim=768):
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
    ]
    schema = CollectionSchema(fields, description="Embedding store for notes")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={
        "metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}
    })
    collection.load()
    return collection

def ensure_correct_collection(collection_name="ds4300_notes", dim=768):
    if utility.has_collection(collection_name):
        existing = Collection(collection_name)
        field_names = [f.name for f in existing.schema.fields]
        if "chunk_id" in field_names and "embedding" in field_names:
            return existing
        else:
            print(f"[INFO] Dropping mismatched collection '{collection_name}'")
            utility.drop_collection(collection_name)

    # Create fresh collection
    return create_milvus_collection(collection_name, dim)


def store_embeddings_milvus(embedded_data, collection_name="ds4300_notes", dim=768):
    connect_milvus()
    collection = ensure_correct_collection(collection_name, dim)

    data = [
        [item["chunk_id"] for item in embedded_data],
        [item["embedding"] for item in embedded_data],
        [item["text"] for item in embedded_data],
        [item["source"] for item in embedded_data],
    ]

    collection.insert(data)
    print(f"Stored {len(embedded_data)} chunks in Milvus collection '{collection_name}'")
    return collection

def query_milvus(collection, query_embedding, top_k=3):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "text", "source"]
    )
    return results
