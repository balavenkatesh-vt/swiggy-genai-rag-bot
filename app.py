import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Page Configuration
st.set_page_config(
    page_title="Swiggy GenAI Support Assistant",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Swiggy GenAI Support Assistant")
st.caption("Powered by RAG, LangChain & Groq Llama 3")

# 1. Setup API Key & Vector DB Resources
os.environ["GROQ_API_KEY"] = "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"

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./chroma_db_swiggy",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )
    return vector_db, llm

vector_db, llm = load_rag_pipeline()

# System Prompt
system_prompt = """
You are a helpful and polite customer support AI assistant for Swiggy.
Use ONLY the provided policy context to answer the user's question. 
If the information is not present in the context, politely inform the user that you don't have that specific detail in Swiggy policy guidelines.

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Swiggy GenAI Support Assistant. How can I help you today?"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Past Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. User Input Handling & Smart RAG Querying
if user_question := st.chat_input("Ask about Swiggy policies..."):
    # User message UI
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Contextual Search Query Generation
    # Chat History இருந்தால் கேள்வியை முழுமையான கேள்வியாக மாற்றுதல்
    search_query = user_question
    if len(st.session_state.chat_history) > 0:
        reformulate_prompt = f"Given the chat history and the follow-up question below, rephrase the follow-up question to be a standalone search query. Do not answer it, just rephrase it.\nFollow-up question: {user_question}"
        rephrased = llm.invoke(st.session_state.chat_history + [HumanMessage(content=reformulate_prompt)])
        search_query = rephrased.content

    # Assistant Response UI
    with st.chat_message("assistant"):
        with st.spinner("Searching Swiggy Knowledge Base..."):
            # Context Retrieval using Reformulated Query
            docs = vector_db.similarity_search(search_query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Format Prompt
            formatted_prompt = prompt_template.format_messages(
                context=context_text,
                chat_history=st.session_state.chat_history,
                question=user_question
            )
            
            # LLM Call
            response = llm.invoke(formatted_prompt)
            ai_answer = response.content
            
            st.markdown(ai_answer)

    # Update State
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=ai_answer))