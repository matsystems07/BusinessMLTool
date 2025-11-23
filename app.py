# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import docx
import PyPDF2
from typing import Optional
from utils import llm_tools

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Business AI App", layout="wide", page_icon="💼")

# -------------------------------
# Initialize Session State
# -------------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  # {session_name: [messages]}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Default Chat"

# -------------------------------
# ----- MEMORY ANALYTICS -----
# -------------------------------
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def init_memory(short_term_limit: int = 50):
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = deque(maxlen=short_term_limit)
    if 'long_term_memory' not in st.session_state:
        st.session_state.long_term_memory = []

def store_message(role: str, content: str, long_term: bool = False):
    entry = {"role": role, "content": content}
    st.session_state.short_term_memory.append(entry)
    if long_term:
        st.session_state.long_term_memory.append(entry)

def retrieve_memory(n: int = 5, long_term: bool = False):
    memory = st.session_state.long_term_memory if long_term else st.session_state.short_term_memory
    return list(memory)[-n:]

def cluster_memory_topics(num_clusters: int = 3, long_term: bool = False):
    memory = st.session_state.long_term_memory if long_term else st.session_state.short_term_memory
    if not memory:
        return []
    texts = [m["content"] for m in memory]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    try:
        kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=42)
        kmeans.fit(X)
        clusters = {i: [] for i in range(kmeans.n_clusters)}
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(texts[idx])
        return clusters
    except Exception as e:
        return {"error": str(e)}

def clear_memory(long_term: bool = False):
    if long_term:
        st.session_state.long_term_memory = []
    else:
        st.session_state.short_term_memory.clear()

def get_memory_text(long_term: bool = False) -> str:
    memory = st.session_state.long_term_memory if long_term else st.session_state.short_term_memory
    return "\n".join([f"{m['role']}: {m['content']}" for m in memory])

# Initialize memory at app start
init_memory()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Business AI App")
mode = st.sidebar.radio(
    "Select Mode", 
    ["Customer Support", "Business Insights", "Upload Report/Dataset", "View Analytics"]
)

# -------------------------------
# Chat Session Controls
# -------------------------------
if mode in ["Customer Support", "Business Insights"]:
    st.sidebar.subheader("Chat Sessions")
    session_names = list(st.session_state.chat_sessions.keys())
    if not session_names:
        session_names = ["Default Chat"]
        st.session_state.chat_sessions["Default Chat"] = []
    selected_session = st.sidebar.selectbox(
        "Choose Chat",
        session_names,
        index=session_names.index(st.session_state.current_session) if st.session_state.current_session in session_names else 0
    )
    st.session_state.current_session = selected_session

    if st.sidebar.button("Start New Chat"):
        new_name = f"Session {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_name] = []
        st.session_state.current_session = new_name

    if st.sidebar.button("Delete Current Chat"):
        if st.session_state.current_session in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[st.session_state.current_session]
            st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0] if st.session_state.chat_sessions else "Default Chat"

# -------------------------------
# --------- FILE PROCESSING HELPERS ---------
# -------------------------------
def extract_text_from_pdf(file: io.BytesIO) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file: io.BytesIO) -> str:
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_text_from_txt(file: io.BytesIO) -> str:
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return content.strip()

def process_uploaded_file(file, history: Optional[list] = None) -> str:
    if file is None:
        return "⚠️ No file uploaded."
    filename = file.name.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif filename.endswith(".txt"):
        text = extract_text_from_txt(file)
    else:
        return "⚠️ Unsupported file type. Please upload PDF, DOCX, or TXT."
    MAX_CHARS = 3000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[Text truncated for LLM]"
    return llm_tools.business_analysis(text, history)

def process_uploaded_dataset(file, analysis_type: str = "summary", history: Optional[list] = None):
    if file is None:
        return "⚠️ No file uploaded."
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return "⚠️ Unsupported dataset format. Please upload CSV or Excel."
    except Exception as e:
        return f"⚠️ Error reading dataset: {str(e)}"
    return llm_tools.generate_insights_from_dataframe(df, analysis_type=analysis_type, history=history)

def process_uploaded(file, analysis_type: str = "summary", history: Optional[list] = None):
    fname = file.name.lower()
    if fname.endswith((".pdf", ".docx", ".txt")):
        return process_uploaded_file(file, history)
    elif fname.endswith((".csv", ".xls", ".xlsx")):
        return process_uploaded_dataset(file, analysis_type, history)
    else:
        return "⚠️ Unsupported file type."

# -------------------------------
# CUSTOMER SUPPORT MODE
# -------------------------------
if mode == "Customer Support":
    st.subheader(f"💬 Customer Support - {st.session_state.current_session}")
    chat_placeholder = st.container()
    user_input = st.text_area("Ask a question to the business AI:")

    if st.button("Send"):
        history = st.session_state.chat_sessions.get(st.session_state.current_session, [])
        reply = llm_tools.customer_support_reply(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "ai", "content": reply})
        st.session_state.chat_sessions[st.session_state.current_session] = history
        # Store in memory analytics too
        store_message("user", user_input)
        store_message("ai", reply)

    with chat_placeholder:
        history = st.session_state.chat_sessions.get(st.session_state.current_session, [])
        for msg in history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")

# -------------------------------
# BUSINESS INSIGHTS MODE
# -------------------------------
elif mode == "Business Insights":
    st.subheader(f"📊 Business Insights - {st.session_state.current_session}")
    chat_placeholder = st.container()
    user_input = st.text_area("Enter business-related text or query:")

    if st.button("Analyze"):
        history = st.session_state.chat_sessions.get(st.session_state.current_session, [])
        reply = llm_tools.business_analysis(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "ai", "content": reply})
        st.session_state.chat_sessions[st.session_state.current_session] = history
        store_message("user", user_input)
        store_message("ai", reply)

    with chat_placeholder:
        history = st.session_state.chat_sessions.get(st.session_state.current_session, [])
        for msg in history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")

# -------------------------------
# UPLOAD REPORT / DATASET MODE
# -------------------------------
elif mode == "Upload Report/Dataset":
    st.subheader("📁 Upload Report / Dataset")
    uploaded_file = st.file_uploader("Upload a report or dataset")
    analysis_type = st.selectbox("Analysis Type", ["summary", "trends", "predictions"])
    if uploaded_file:
        history = st.session_state.chat_sessions.get(st.session_state.current_session, [])
        result = process_uploaded(uploaded_file, analysis_type, history)
        st.write(result)
        # Optionally store in memory analytics
        store_message("user", f"[Uploaded file: {uploaded_file.name}]")
        store_message("ai", result)

# -------------------------------
# MEMORY & ANALYTICS MODE
# -------------------------------
elif mode == "View Analytics":
    st.subheader("📈 Memory & Analytics")
    st.write("**Short-Term Memory (Recent Messages)**")
    st.json(retrieve_memory(n=20))
    
    st.write("**Topic Clusters (Short-Term)**")
    clusters = cluster_memory_topics(num_clusters=3)
    st.json(clusters)
    
    if st.button("Clear Short-Term Memory"):
        clear_memory()

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Business AI App | Powered by Groq LLM")
