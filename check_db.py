from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Embedding Model & Vector DB லோட் செய்தல்
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./chroma_db_swiggy",
    embedding_function=embeddings
)

# 2. ChromaDB-யில் உள்ள அனைத்து Chunks-ஐயும் எடுத்தல்
data = vector_db.get()

print("==========================================")
print(f"Total Chunks inside ChromaDB: {len(data['documents'])}")
print("==========================================\n")

# 3. முதல் 3 Chunks-ல் என்ன உரை இருக்கிறது என்று பிரின்ட் செய்து பார்ப்பது
for i, doc in enumerate(data['documents'][:3], 1):
    print(f"--- CHUNK {i} ---")
    print(doc[:300])  # முதல் 300 எழுத்துக்கள்
    print("\n" + "="*40 + "\n")