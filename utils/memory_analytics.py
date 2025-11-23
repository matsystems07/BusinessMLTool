# utils/memory_analytics.py

import streamlit as st
from typing import List, Dict
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -------------------------------
# Memory Initialization
# -------------------------------
def init_memory(short_term_limit: int = 50):
    """
    Initialize short-term and long-term memory in Streamlit session state.
    """
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = deque(maxlen=short_term_limit)
    if 'long_term_memory' not in st.session_state:
        st.session_state.long_term_memory = []

# -------------------------------
# Store Message in Memory
# -------------------------------
def store_message(role: str, content: str, long_term: bool = False):
    """
    Stores a message in short-term or long-term memory.
    """
    memory_entry = {"role": role, "content": content}
    st.session_state.short_term_memory.append(memory_entry)
    if long_term:
        st.session_state.long_term_memory.append(memory_entry)

# -------------------------------
# Retrieve Memory
# -------------------------------
def retrieve_memory(n: int = 5, long_term: bool = False) -> List[Dict]:
    """
    Returns the last n messages from memory.
    """
    memory = st.session_state.long_term_memory if long_term else st.session_state.short_term_memory
    return list(memory)[-n:]

# -------------------------------
# Basic Topic Clustering
# -------------------------------
def cluster_memory_topics(num_clusters: int = 3, long_term: bool = False):
    """
    Performs KMeans clustering on memory content to find main topics.
    """
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

# -------------------------------
# Clean Memory
# -------------------------------
def clear_memory(long_term: bool = False):
    """
    Clears short-term or long-term memory.
    """
    if long_term:
        st.session_state.long_term_memory = []
    else:
        st.session_state.short_term_memory.clear()

# -------------------------------
# Get All Memory as Text
# -------------------------------
def get_memory_text(long_term: bool = False) -> str:
    """
    Returns all messages concatenated into a single string for LLM input.
    """
    memory = st.session_state.long_term_memory if long_term else st.session_state.short_term_memory
    return "\n".join([f"{m['role']}: {m['content']}" for m in memory])
