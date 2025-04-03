# scripts/ingest.py

import os
import re
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(folder_path):
    all_text = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
            all_text[filename] = text
    return all_text

def preprocess_text(text, strategy=None):
    if strategy == "remove_whitespace":
        text = re.sub(r'\s+', ' ', text)
    elif strategy == "remove_punctuation":
        text = re.sub(r'[^\w\s]', '', text)
    elif strategy == "remove_noise":
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def ingest_documents(notes_path='Notes/', chunk_size=500, chunk_overlap=100, text_prep=None):
    raw_text = extract_text_from_pdfs(notes_path)
    processed_texts = {filename: preprocess_text(text, text_prep) for filename, text in raw_text.items()}

    # save chunked data and metadata
    chunked_data = []
    for filename, text in processed_texts.items():
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "source": filename,
                "chunk_id": f"{filename}_{i}",
                "text": chunk
            })

    return chunked_data
