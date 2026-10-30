from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Embedding Model-ஐ லோட் செய்தல்
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. சேமிக்கப்பட்ட Chroma Vector DB-ஐ லோட் செய்தல்
print("Loading Existing Vector Database...")
vector_db = Chroma(
    persist_directory="./chroma_db_swiggy",
    embedding_function=embeddings
)

# 3. கேள்வி (Sample Question)
user_query = "What is the cancellation fee policy after ordering?"

print(f"\n🔍 Searching Vector DB for Question: '{user_query}'\n")

# 4. Similarity Search - Top 2 Chunks எடுப்பது
results = vector_db.similarity_search(user_query, k=2)

# பிரின்ட் செய்தல்
print("--- RETRIEVED CONTEXT FROM VECTOR DB ---")
for i, doc in enumerate(results, 1):
    print(f"\n[Result {i}]")
    print(doc.page_content)
    print("-" * 40)