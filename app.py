import streamlit as st
import os
import requests
import pandas as pd


# Load environment variables

groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="College Info Assistant", page_icon="🎓")
st.title("🎓 College Info Assistant (Groq-powered)")

# Load CSV data
@st.cache_data
def load_college_data(path):
    df = pd.read_csv(path)
    return df

csv_file_path = "college_info.csv"
try:
    college_df = load_college_data(csv_file_path)
    college_data_text = college_df.to_string(index=False)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Function to call Groq
def call_groq_with_context(user_prompt, context):
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    system_msg = (
        "You are a helpful college information assistant. "
        "Answer questions based only on the data provided below:\n\n"
        f"{context[:12000]}\n\n"  # Avoid overly large context
        "If you don’t know the answer, say 'I don’t know based on the data provided.'"
    )

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 10000
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {e}"

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask me anything about the colleges...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_groq_with_context(prompt, college_data_text)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
