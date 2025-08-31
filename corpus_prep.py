import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Path to your docs
DOCS_FOLDER = "docs"

# Load PDF files
def load_pdfs(folder):
    text_data = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text_data += page.extract_text() + "\n"
    return text_data

# Step 1: Read PDFs
print(" Reading documents...")
raw_text = load_pdfs(DOCS_FOLDER)

# Step 2: Split into chunks
print(" Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(raw_text)

# Step 3: Generate embeddings
print(" Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Step 4: Store in FAISS
print(" Saving to FAISS vector DB...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "vector_store.index")

print("✅ Corpus preparation done! Chunks stored in FAISS.")

