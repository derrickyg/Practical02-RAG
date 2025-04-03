from pymilvus import utility
from scripts.ingest import ingest_documents
from scripts.embed import embed_chunks
from scripts.llm_respond import generate_answer_ollama
from vector_store.milvus_store import connect_milvus, store_embeddings_milvus, query_milvus
from sentence_transformers import SentenceTransformer  

connect_milvus()
if utility.has_collection("ds4300_notes"):
    print("[INFO] Dropping old collection")
    utility.drop_collection("ds4300_notes")

chunked_data = ingest_documents(notes_path='Notes/', chunk_size=500, chunk_overlap=100)

model_key = 'mpnet'  
embedded_data = embed_chunks(chunked_data, model_key=model_key)

collection = store_embeddings_milvus(embedded_data, collection_name="ds4300_notes")

model_name_map = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}
model = SentenceTransformer(model_name_map[model_key])

user_query = input("What's your question? ")
query_embedding = model.encode(user_query)

results = query_milvus(collection, query_embedding, top_k=5)
retrieved_chunks = [hit.entity.get("text") for hit in results]

#print("\n Retrieved Chunks:")
#for i, chunk in enumerate(retrieved_chunks):
#    print(f"\nChunk {i + 1}:\n{chunk.encode('utf-8', errors='replace').decode()}")

answer = generate_answer_ollama(retrieved_chunks, user_query, model="mistral")

print("\n LLM Response:")
print(answer)