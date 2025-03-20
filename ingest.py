import os
import pytesseract
import numpy as np
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

notes_path = 'Practical02-RAG\\Notes'
raw_text = extract_text_from_pdfs(notes_path)

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^\w\s]', '', text)  
    return text.strip().lower()  

processed_texts = {filename: preprocess_text(text) for filename, text in raw_text.items()}




def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Apply chunking to all documents
chunked_texts = {filename: chunk_text(text) for filename, text in processed_texts.items()}

print(chunked_texts)



