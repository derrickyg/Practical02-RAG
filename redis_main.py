# main.py

from scripts.ingest import ingest_documents
from scripts.embed import embed_chunks
from scripts.llm_respond import generate_answer_ollama
from vector_store.redis_store import *
from sentence_transformers import SentenceTransformer

client = create_redis_client()

chunked_data = ingest_documents(notes_path='Notes/', chunk_size=500, chunk_overlap=100)

model_key = 'mpnet'
embedded_data = embed_chunks(chunked_data, model_key=model_key)

collection = store_embeddings_redis(client=client, embedded_data=embedded_data)

model_name_map = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}

model = SentenceTransformer(model_name_map[model_key])

user_query = input("Whats your question? ")
query_embedding = model.encode(user_query)

results = query_redis(client=client, collection=collection, query_embedding=query_embedding, top_k=3)
retrieved_chunks = results.docs

answer = generate_answer_ollama(retrieved_chunks, user_query, model="tinyllama")

print("\n LLM Response:")
print(answer)
