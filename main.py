# main.py

from scripts.ingest import ingest_documents
from scripts.embed import embed_chunks
from scripts.llm_respond import generate_answer_ollama
from vector_store.chroma_store import store_embeddings_chroma, query_chroma
from sentence_transformers import SentenceTransformer  

chunked_data = ingest_documents(notes_path='Notes/', chunk_size=500, chunk_overlap=100)

model_key = 'mpnet'  
embedded_data = embed_chunks(chunked_data, model_key=model_key)

collection = store_embeddings_chroma(embedded_data, collection_name="ds4300_notes")

model_name_map = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}
model = SentenceTransformer(model_name_map[model_key])

user_query = input("Whats your question")
query_embedding = model.encode(user_query)

results = query_chroma(collection, query_embedding, top_k=5)
retrieved_chunks = results["documents"][0]

#print("\n Retrieved Chunks:")
#for i, chunk in enumerate(retrieved_chunks):
#    print(f"\nChunk {i + 1}:\n{chunk.encode('utf-8', errors='replace').decode()}")

answer = generate_answer_ollama(retrieved_chunks, user_query, model="tinyllama")

print("\n LLM Response:")
print(answer)