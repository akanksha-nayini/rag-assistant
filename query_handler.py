import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os

# Paths
DOCS_FOLDER = "docs"
INDEX_FILE = "vector_store.index"

# Step 1: Reload documents (to map FAISS results back to text)
def load_pdfs(folder):
    text_data = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text_data += page.extract_text() + "\n"
    return text_data

print(" Loading documents...")
raw_text = load_pdfs(DOCS_FOLDER)

# Same splitter used in corpus_prep.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(raw_text)

# Step 2: Load FAISS index
print(" Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Query function
def query_rag(question, top_k=3):
    # Encode query
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Fetch top chunks
    results = [chunks[i] for i in indices[0]]
    return results

# ------------------------------
# Example: Interactive query
# ------------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answers = query_rag(q)
        print("\n Top retrieved chunks:")
        for i, ans in enumerate(answers, 1):
            print(f"{i}. {ans[:200]}...\n")
