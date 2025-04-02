import requests

def generate_answer_ollama(context_chunks, query, model="llama2"):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful TA for a university-level data systems course (DS4300).
You will be given a user question and relevant course notes. Provide a clear, direct answer.

User Question: {query}

Relevant Notes:
{context}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            print(" Unexpected response structure:", data)
            return "[Error: No 'response' key in Ollama output]"

        return data["response"]

    except requests.exceptions.RequestException as e:
        print(f" Error calling Ollama API: {e}")
        return "[Error: Ollama API call failed]"

    except Exception as e:
        print(f" Unexpected error: {e}")
        return "[Error: Unexpected failure in LLM response]"
