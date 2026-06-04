import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from google import genai

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="🔬"
)
st.title("🔬 Research Paper Q&A")
st.markdown("Ask anything about **Transformer** and **ResNet** papers.")

api_key = st.sidebar.text_input("Paste Gemini API Key here", type="password")

if not api_key:
    st.warning("👈 Enter your Gemini API key in the sidebar to begin")
    st.stop()

try:
    gemini = genai.Client(api_key=api_key)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=DefaultEmbeddingFunction()
    )
    st.sidebar.success(f"✅ {collection.count()} chunks loaded")
except Exception as e:
    st.error(f"Connection error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                results = collection.query(
                    query_texts=[prompt],
                    n_results=4
                )
                context = "\n\n".join(results['documents'][0])
                full_prompt = f"""You are a research assistant.
Answer based only on the context below.
If the answer is not in the context, say so.

Context:
{context}

Question: {prompt}
Answer:"""
                response = gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error: {e}")