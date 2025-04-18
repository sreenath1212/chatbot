import streamlit as st
import os
import pandas as pd

from langchain_community.llms import WatsonxLLM  



# Watsonx credentials
api_key = st.secrets["WATSONX_API_KEY"]
url = st.secrets["WATSONX_URL"]
project_id = st.secrets["WATSONX_PROJECT_ID"]


# Streamlit page setup
st.set_page_config(page_title="Chat with Your Data (Watsonx)", page_icon="🤖")
st.title("🤖 Chat with Your Data - Powered by Watsonx")

# Safety check for missing environment variables
if not api_key or not url or not project_id:
    st.error(
        "Missing Watsonx credentials. Please check your `.env` file. "
        "Required variables: WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID."
    )
    st.stop()

# Try to create the LLM instance with error handling
try:
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-1-8b-instruct",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 2048,
            "min_new_tokens": 1,
            "stop_sequences": ["</s>", "<|endoftext|>"],
            "temperature": 0.7
        }
    )
except Exception as e:
    st.error(f"Failed to initialize Watsonx LLM: {e}")
    st.stop()

# Load the CSV file as a dataframe
csv_path = "college_info.csv"  # Adjust if you move the file elsewhere
try:
    df = pd.read_csv(csv_path)
    data_context = df.to_string(index=False)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear button
if st.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt input
prompt = st.chat_input("Ask me anything about the data...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build context-aware prompt
    system_context = (
        "You are a helpful assistant. Use the following college data to answer the user's question.\n\n"
        f"{data_context}\n\n"
    )

    chat_prompt = system_context
    for i in range(0, len(st.session_state.messages), 2):
        user_msg = st.session_state.messages[i]["content"]
        assistant_msg = st.session_state.messages[i+1]["content"] if i+1 < len(st.session_state.messages) else ""
        chat_prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"

    chat_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm(chat_prompt)
                if isinstance(response, dict):
                    response = response.get("generated_text", str(response))
                elif not isinstance(response, str):
                    response = str(response)
            except Exception as e:
                response = f"❌ Error generating response: {e}"

            st.markdown(response)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": response})
