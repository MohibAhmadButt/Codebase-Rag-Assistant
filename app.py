import streamlit as st
import google.generativeai as genai
import os
import shutil
from git import Repo

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Code-Aware RAG Assistant", layout="wide")
st.title("🚀 Code-Aware RAG Assistant")
st.markdown("Analyze GitHub repositories using FAISS and Gemini 2.5 Flash.")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Local Embeddings (Runs fast and free)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Using password type so it's hidden on your portfolio
    gemini_key = st.text_input("Gemini API Key", type="password")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")

    if st.button("Initialize & Index Repo"):
        if not repo_url or not gemini_key:
            st.error("Missing API Key or URL")
        else:
            with st.spinner("Cloning & Indexing..."):
                try:
                    repo_path = os.path.abspath("./cloned_repo")
                    if os.path.exists(repo_path):
                        shutil.rmtree(repo_path, ignore_errors=True)
                    
                    Repo.clone_from(repo_url, to_path=repo_path)
                    
                    # Stable Text Loader for Python files
                    loader = DirectoryLoader(
                        repo_path, 
                        glob="**/*.py", 
                        loader_cls=TextLoader,
                        show_progress=True
                    )
                    docs = loader.load()
                    
                    if not docs:
                        st.warning("No Python files found in this repository.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = splitter.split_documents(docs)
                        
                        st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
                        st.success(f"Successfully indexed {len(docs)} files!")
                except Exception as e:
                    st.error(f"Error during indexing: {e}")

    if st.button("Reset Session"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.rerun()

# --- CHAT LOGIC ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

if user_input := st.chat_input("Ask about the codebase..."):
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not gemini_key:
        st.error("Please enter your Gemini API Key.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    context = ""
                    if st.session_state.vector_db:
                        results = st.session_state.vector_db.similarity_search(user_input, k=3)
                        context = "\n\nCODE CONTEXT:\n" + "\n---\n".join([d.page_content for d in results])

                    # Build prompt with history so it doesn't forget
                    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in st.session_state.chat_history[-5:]])
                    full_prompt = f"Previous Chat:\n{history_text}\n\nUser Question: {user_input}\n{context}\n\nAnswer the question based on the code context if provided."
                    
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "text": response.text})
                except Exception as e:
                    st.error(f"Generation Error: {e}")