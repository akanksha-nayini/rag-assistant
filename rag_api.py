from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# ------------------ Paths ------------------
DOCS_FOLDER = "docs"
INDEX_FILE = "vector_store.index"

# ------------------ Load PDFs ------------------
def load_pdfs(folder):
    text_data = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text_data += page.extract_text() + "\n"
    return text_data

raw_text = load_pdfs(DOCS_FOLDER)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# ------------------ Load FAISS ------------------
index = faiss.read_index(INDEX_FILE)

# ------------------ Embedding model ------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ LLM ------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# ------------------ Helper functions ------------------
def generate_answer(context_chunks, question):
    context_text = " ".join(context_chunks)
    prompt = f"""
You are an expert AI assistant. Using the context provided, answer the question thoroughly in 3-5 sentences.

Context: {context_text}

Question: {question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    output_ids = llm_model.generate(**inputs, max_length=700, min_length=150)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

def query_rag(question, top_k=5):
    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    top_chunks = [chunks[i] for i in indices[0]]
    answer = generate_answer(top_chunks, question)
    return answer

# ------------------ API Route ------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    answer = query_rag(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print(" RAG API running on http://127.0.0.1:5000")
    app.run(debug=True)
