import requests

def generate_answer_ollama(context_chunks, query, model="mistral"):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful TA for a university-level data systems course (DS4300).
You will be given a user question and relevant course notes. Provide a clear, direct answer.

User Question: {query}

Relevant Notes:
{context}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
