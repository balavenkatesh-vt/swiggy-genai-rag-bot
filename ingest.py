import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("--- STEP 1: Reading Local Knowledge Base File ---")

file_path = "swiggy_policy_raw.txt"

if not os.path.exists(file_path):
    print(f"❌ Error: {file_path} file not found!")
    exit()

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"✅ Successfully read {len(raw_text)} characters from '{file_path}'")

# ==========================================
# STEP 2: Text Chunking & Vector DB Storage
# ==========================================
print("\n--- STEP 2: Text Chunking & Vector DB Storage ---")

# Text-ஐ சிறிய பகுதிகளாகப் பிரித்தல்
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " "]
)

docs = [Document(page_content=raw_text, metadata={"source": file_path})]
chunks = text_splitter.split_documents(docs)

print(f"Total Text Chunks Created: {len(chunks)}")

# Embedding Model லோட் செய்தல்
print("\nLoading HuggingFace Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector DB (ChromaDB) உருவாக்குதல்
print("Storing Chunks into Chroma Vector Database...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_swiggy"
)

print("✅ Success: All policy data converted to Embeddings and saved in './chroma_db_swiggy'!")