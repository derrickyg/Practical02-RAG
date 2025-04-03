#scripts/embed.py

from sentence_transformers import SentenceTransformer
import numpy as np
import re

# add new models here 
SUPPORTED_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "BAAI": "BAAI/bge-base-en-v1.5",
}


def load_embedding_model(model_key):
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"{model_key} not in SUPPORTED_MODELS (embed.py)")

    model_name = SUPPORTED_MODELS[model_key]
    model = SentenceTransformer(model_name)
    return model

def embed_chunks(chunked_data, model_key="miniLM"):
    model = load_embedding_model(model_key)
    embedded_data = []

    # save embedded text and metadata
    for chunk in chunked_data:
        chunk_text = chunk["text"]
        embedding = model.encode(chunk_text)

        embedded_data.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk_text,
            "embedding": embedding.tolist()
        })

    return embedded_data

