# scripts/llm_respond

import requests

def generate_answer_ollama(context_chunks, query, model="llama2"):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    
    You are a helpful TA for a University Database Systems course (DS4300).
    You will be given a student question and relevant course notes. Provide a concise answer
    to aid the student.

    User Question: {query}

    Relevant Notes:
    {context}

    """

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

        # error logging when ollama does not return a response object
        if "response" not in data:
            print("Unexpected response", data)
            return "No response returned from Ollama"

        return data["response"]

    except requests.exceptions.RequestException as e:
        print(f" Error calling Ollama API: {e}")
        return f"{e}"

    except Exception as e:
        print(f" Unexpected error: {e}")
        return f"{e}"
