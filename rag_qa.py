import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Paths
# ------------------------------
DOCS_FOLDER = "docs"
INDEX_FILE = "vector_store.index"

# ------------------------------
# Load PDFs
# ------------------------------
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

# ------------------------------
# Split text into chunks
# ------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# ------------------------------
# Load FAISS index
# ------------------------------
print(" Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

# ------------------------------
# Load embedding model
# ------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# Load HuggingFace Flan-T5-large model
# ------------------------------
print(" Loading HuggingFace Flan-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# ------------------------------
# Function to generate detailed answer using Flan-T5
# ------------------------------
def generate_answer(context_chunks, question):
    context_text = " ".join(context_chunks)
    prompt = f"""
You are an expert AI assistant. Using the context provided, answer the question in 2-3 detailed sentences.

Context: {context_text}

Question: {question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    output_ids = llm_model.generate(**inputs, max_length=512)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# ------------------------------
# Query function
# ------------------------------
def query_rag(question, top_k=5):  # increased top_k for richer context
    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    top_chunks = [chunks[i] for i in indices[0]]
    answer = generate_answer(top_chunks, question)
    return answer, top_chunks

# ------------------------------
# Interactive session
# ------------------------------
if __name__ == "__main__":
    print("\n RAG assistant ready. Type your question or 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() == "exit":
            break
        answer, retrieved_chunks = query_rag(q)
        print("\n Top retrieved chunks:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"{i}. {chunk[:200]}...\n")
        print(" Generated Answer:")
        print(answer)
