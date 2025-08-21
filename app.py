import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/user_query"

st.set_page_config(page_title="Alice in Wonderland QnA", page_icon="📖", layout="centered")

st.title("📖 Alice in Wonderland QnA")
st.write("Ask me anything about the book! (powered by FastAPI + RAG + Redis)")

# User input
user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": user_query})
            if response.status_code == 200:
                answer = response.json()["User Query"]
                st.success("✅ Answer received!")
                st.write(answer)
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")