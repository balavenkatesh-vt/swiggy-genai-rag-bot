import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Groq API Key அமைத்தல் (உங்கள் API Key-ஐ இங்கே போடவும்)
os.environ["GROQ_API_KEY"] = "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"

# 2. Embedding Model & Existing Chroma Vector DB லோட் செய்தல்
print("Loading Embedding Model and Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./chroma_db_swiggy",
    embedding_function=embeddings
)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)
# 4. System Prompt Design (Hallucination-ஐத் தவிர்க்க தெளிவான விதிமுறைகள்)
system_prompt = """
You are a helpful customer support AI assistant for Swiggy.
Use ONLY the following context to answer the user's question. 
If the answer is not contained in the context, politely say: "I'm sorry, I don't have that information in Swiggy policy guidelines." Do not make up answers.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 5. Question Answer Function
def ask_swiggy_bot(user_question):
    print(f"\n❓ User Question: {user_question}")
    
    # Vector DB-லிருந்து தொடர்புடைய Top 3 Chunks-ஐ எடுத்தல்
    docs = vector_db.similarity_search(user_question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Prompt உருவாக்குதல்
    formatted_prompt = prompt.format(context=context_text, question=user_question)
    
    # LLM-க்கு அனுப்பி பதில் பெறுதல்
    response = llm.invoke(formatted_prompt)
    
    print("\n🤖 Swiggy Assistant Response:")
    print(response.content)

# 6. Test Questions
if __name__ == "__main__":
    ask_swiggy_bot("What is the cancellation fee if I cancel after 60 seconds?")
    ask_swiggy_bot("How long does it take to get a refund for Swiggy Wallet?")