# utils/llm_tools.py

import os
from groq import Client
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import io
import re

# -------------------------------
# Initialize Groq Client
# -------------------------------
def init_groq_client():
    """
    Initializes the Groq client using the API key stored in environment variables.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GROQ_API_KEY not set in environment variables")
    client = Client(api_key=api_key)
    return client

# -------------------------------
# Generate LLM Response (Fixed Role Mapping)
# -------------------------------
def generate_response(prompt: str, history: List[Dict] = None, max_tokens: int = 512) -> str:
    """
    Generate response from LLaMA model via Groq API.
    Handles history and maps roles correctly for Groq chat API.
    """
    client = init_groq_client()
    try:
        messages = []

        if history:
            for h in history:
                role = h.get('role', 'user')
                if role == "ai":
                    role = "assistant"
                elif role not in ["user", "assistant", "system"]:
                    role = "user"
                messages.append({"role": role, "content": h.get("content", "")})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

# -------------------------------
# Summarize Text
# -------------------------------
def summarize_text(text: str, history: List[Dict] = None) -> str:
    prompt = f"Summarize the following text concisely:\n{text}"
    return generate_response(prompt, history)

# -------------------------------
# Classify Text
# -------------------------------
def classify_text(text: str, categories: List[str], history: List[Dict] = None) -> str:
    cats_str = ", ".join(categories)
    prompt = f"Classify the following text into one of these categories: {cats_str}\nText: {text}"
    return generate_response(prompt, history)

# -------------------------------
# Business Analysis
# -------------------------------
def business_analysis(text: str, history: List[Dict] = None) -> str:
    prompt = f"Analyze the following business-related text and provide key insights:\n{text}"
    return generate_response(prompt, history)

# -------------------------------
# Token Counting
# -------------------------------
def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# -------------------------------
# Customer Support Response
# -------------------------------
def customer_support_reply(customer_query: str, history: List[Dict] = None) -> str:
    """
    Generates professional, polite, and helpful responses to customer queries.
    """
    prompt = f"""
    You are a professional customer support agent.
    Respond to the customer query in a friendly and helpful way:
    {customer_query}
    """
    return generate_response(prompt, history)

# -------------------------------
# Generate Business Insights from CSV/DF
# -------------------------------
def generate_insights_from_dataframe(df: pd.DataFrame, analysis_type: str = "summary", history: List[Dict] = None) -> str:
    """
    Generates textual business insights from a dataframe using LLM.
    
    analysis_type: "summary", "trends", "recommendations"
    """
    csv_str = df.to_csv(index=False)
    prompt = f"""
    You are a business analyst.
    Analyze the following dataset and provide {analysis_type} insights for a business decision maker:
    {csv_str}
    """
    return generate_response(prompt, history)

# -------------------------------
# Plot simple bar chart from DataFrame
# -------------------------------
def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Chart") -> io.BytesIO:
    plt.figure(figsize=(8,5))
    plt.bar(df[x_col], df[y_col], color='skyblue')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

# -------------------------------
# Plot simple line chart from DataFrame
# -------------------------------
def plot_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Line Chart") -> io.BytesIO:
    plt.figure(figsize=(8,5))
    plt.plot(df[x_col], df[y_col], marker='o', color='green')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

# -------------------------------
# Generate Graphs Dynamically from LLM Output
# -------------------------------
def generate_graphs_from_llm_output(llm_output: str) -> dict:
    """
    Parses LLM output and generates appropriate charts.
    Returns a dictionary: {"text": llm_output, "charts": [buf1, buf2, ...]}
    """
    charts = []

    # Attempt to detect tabular-like data in LLM output (CSV-style or key: value)
    lines = llm_output.strip().split("\n")
    data_rows = []
    headers = []

    # Check if first line is comma-separated -> assume headers
    if len(lines) > 1 and "," in lines[0]:
        headers = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(headers):
                data_rows.append(parts)
    else:
        # Try key: value pairs
        kv_data = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    kv_data[key.strip()] = float(val.strip())
                except ValueError:
                    continue
        if kv_data:
            df = pd.DataFrame(list(kv_data.items()), columns=["Category", "Value"])
            buf = plot_bar_chart(df, "Category", "Value", title="LLM Generated Insights")
            charts.append(buf)

    # If table rows detected
    if headers and data_rows:
        try:
            df = pd.DataFrame(data_rows, columns=headers)
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) >= 1:
                x_col = df.select_dtypes(exclude="number").columns[0] if len(df.select_dtypes(exclude="number").columns) > 0 else numeric_cols[0]
                y_col = numeric_cols[0]
                buf = plot_bar_chart(df, x_col, y_col, title="LLM Generated Chart")
                charts.append(buf)
        except Exception:
            pass  # fallback: no chart

    return {"text": llm_output, "charts": charts}
