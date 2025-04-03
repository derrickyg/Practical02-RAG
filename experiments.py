from driver import run_experiment

def run_experiments():
    query = "What is a sparse index?"

    # batch1 - testing embedding models, vector_dbs, and llm models while chunks are constant
    default_chunk_size = 500
    default_chunk_overlap = 50
    default_text_prep = None

    embedding_models = ["BAAI","miniLM", "mpnet"]
    vector_dbs = ["chroma", "redis", "qdrant"]
    llm_models = ["tinyllama", "mistral"]


    for embed_model in embedding_models:
        for vector_db in vector_dbs:
            for llm_model in llm_models:
                print(f"\n[BATCH1] {embed_model} | {vector_db} | {llm_model}", flush=True)
                params = {
                    "chunk_size": default_chunk_size,
                    "chunk_overlap": default_chunk_overlap,
                    "text_prep": default_text_prep,
                    "embedding_model": embed_model,
                    "vector_db": vector_db,
                    "llm_model": llm_model,
                    "query": query
                }
                run_experiment(params)

    # batch2 - testing chunk sizes while keeping all params constant
    chunk_sizes = [200, 500, 1000]
    for size in chunk_sizes:
        print(f"\n[BATCH2] size={size}", flush=True)
        params = {
            "chunk_size": size,
            "chunk_overlap": default_chunk_overlap,
            "text_prep": default_text_prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

    # batch3 - testing chunk overlaps while keeping all params constant
    chunk_overlaps = [0, 50, 100]
    for overlap in chunk_overlaps:
        print(f"\n[BATCH3] overlap={overlap}", flush=True)
        params = {
            "chunk_size": default_chunk_size,
            "chunk_overlap": overlap,
            "text_prep": default_text_prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

    # batch4 - testing text processing strategies keeping all params constant
    preps = [None, "remove_whitespace", "remove_punctuation", "remove_noise"]
    for prep in preps:
        print(f"\n[BATCH4] strategy={prep}", flush=True)
        params = {
            "chunk_size": default_chunk_size,
            "chunk_overlap": default_chunk_overlap,
            "text_prep": prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

if __name__ == "__main__":
    run_experiments()
