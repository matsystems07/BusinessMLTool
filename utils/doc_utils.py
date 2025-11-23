# utils/doc_utils.py

import streamlit as st
from typing import Optional
from utils import llm_tools
import pandas as pd
import docx
import PyPDF2
import io

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(file: io.BytesIO) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# -------------------------------
# Extract text from DOCX
# -------------------------------
def extract_text_from_docx(file: io.BytesIO) -> str:
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# -------------------------------
# Extract text from TXT
# -------------------------------
def extract_text_from_txt(file: io.BytesIO) -> str:
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return content.strip()

# -------------------------------
# Process Uploaded Report
# -------------------------------
def process_uploaded_file(file, history: Optional[list] = None) -> str:
    """
    Detects file type and extracts text, then runs business analysis via LLM.
    Supports PDF, DOCX, TXT.
    """
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

    # Optional: chunk text if too large (Groq LLM has max token limits)
    MAX_CHARS = 3000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[Text truncated for LLM]"

    # Call business_analysis (context-aware) from llm_tools
    return llm_tools.business_analysis(text, history)

# -------------------------------
# Process CSV/Excel for Insights
# -------------------------------
def process_uploaded_dataset(file, analysis_type: str = "summary", history: Optional[list] = None):
    """
    Reads CSV or Excel file and generates business insights.
    """
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

    # Call LLM for analysis
    return llm_tools.generate_insights_from_dataframe(df, analysis_type=analysis_type, history=history)

# -------------------------------
# Helper: Auto-detect file type and route accordingly
# -------------------------------
def process_uploaded(file, analysis_type: str = "summary", history: Optional[list] = None):
    """
    Unified function for Streamlit app to handle any uploaded file.
    - PDF/DOCX/TXT → business analysis
    - CSV/Excel → dataset insights
    """
    fname = file.name.lower()
    if fname.endswith((".pdf", ".docx", ".txt")):
        return process_uploaded_file(file, history)
    elif fname.endswith((".csv", ".xls", ".xlsx")):
        return process_uploaded_dataset(file, analysis_type, history)
    else:
        return "⚠️ Unsupported file type."
