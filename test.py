from groq import Client
import streamlit as st

client = Client(api_key=st.secrets["GROQ_API_KEY"])
print(client.models.list())
