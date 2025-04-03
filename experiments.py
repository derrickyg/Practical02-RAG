from driver import run_experiment

def run_experiments():
    query = "What is a sparse index?"
    fixed_chunk_size = 500
    fixed_chunk_overlap = 50
    text_prep = None

    embedding_models = ["miniLM", "mpnet", "instructor"]
    vector_dbs = ["chroma", "redis", "qdrant"]
    llm_models = ["tinyllama", "mistral"]

    # Section 1: Baseline - all embeddings, DBs, and LLMs
    for embed_model in embedding_models:
        for vector_db in vector_dbs:
            for llm_model in llm_models:
                print(f"\n[BASELINE] {embed_model} | {vector_db} | {llm_model}", flush=True)
                params = {
                    "chunk_size": fixed_chunk_size,
                    "chunk_overlap": fixed_chunk_overlap,
                    "text_prep": text_prep,
                    "embedding_model": embed_model,
                    "vector_db": vector_db,
                    "llm_model": llm_model,
                    "query": query
                }
                run_experiment(params)

    # Section 2: Chunk size variation
    chunk_sizes = [200, 500, 1000]
    for size in chunk_sizes:
        print(f"\n[CHUNK SIZE TEST] size={size}", flush=True)
        params = {
            "chunk_size": size,
            "chunk_overlap": fixed_chunk_overlap,
            "text_prep": text_prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

    # Section 3: Chunk overlap variation
    chunk_overlaps = [0, 50, 100]
    for overlap in chunk_overlaps:
        print(f"\n[CHUNK OVERLAP TEST] overlap={overlap}", flush=True)
        params = {
            "chunk_size": fixed_chunk_size,
            "chunk_overlap": overlap,
            "text_prep": text_prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

    # Section 4: Text preprocessing strategies
    preps = [None, "remove_whitespace", "remove_punctuation", "remove_noise"]
    for prep in preps:
        print(f"\n[TEXT PREP TEST] strategy={prep}", flush=True)
        params = {
            "chunk_size": fixed_chunk_size,
            "chunk_overlap": fixed_chunk_overlap,
            "text_prep": prep,
            "embedding_model": "mpnet",
            "vector_db": "chroma",
            "llm_model": "tinyllama",
            "query": query
        }
        run_experiment(params)

if __name__ == "__main__":
    run_experiments()
