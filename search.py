from sentence_transformers import SentenceTransformer

def load_models():
    """Initialzing models"""
    models = {
        "MiniLM": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        "MPNet": SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
        "InstructorXL": None  # Assume placeholder if specific loading required
    }
    # If InstructorXL requires special handling, initialize it here
    # Example: models["InstructorXL"] = load_instructorxl_model()
    
    return models

def generate_embeddings(models, text):
    """ Function to process text through each model and generate embeddings. """
    embeddings = {}
    for name, model in models.items():
        if model is not None:
            embeddings[name] = model.encode(text)
        else:
            print(f"Model {name} is not loaded.")
    return embeddings


def main():
    models = load_models()
    sample_text = "Example query text for embedding."

    # Generate embeddings
    embeddings = generate_embeddings(models, sample_text)

    # Here you might add logic to compare these embeddings in some tasks,
    # such as clustering, similarity search, etc.
    print(embeddings)

if __name__ == "__main__":
    main()