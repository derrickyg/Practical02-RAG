from sentence_transformers import SentenceTransformer
import numpy as np


SUPPORTED_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}


def load_embedding_model(model_key):
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f" Unsupported model key: {model_key}")
    
    model_name = SUPPORTED_MODELS[model_key]
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def get_embedding_input(model_key, text):
    if model_key == "instructor":
        return ("Represent the DS4300 course note chunk:", text)
    return text


def embed_chunks(chunked_data, model_key="miniLM"):
    model = load_embedding_model(model_key)
    embedded_data = []

    for chunk in chunked_data:
        chunk_text = chunk["text"]
        embed_input = get_embedding_input(model_key, chunk_text)

        embedding = model.encode(embed_input)

        embedded_data.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk_text,
            "embedding": embedding.tolist()  #
        })

    print(f"Embedded {len(embedded_data)} chunks using model '{model_key}'")
    return embedded_data
