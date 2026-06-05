import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil

CHROMA_PATH = "chroma_db"

def ingest_uploaded_files(pdf_paths):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            st.warning(f"Skipped {os.path.basename(path)}: could not read file")
            continue

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("research_papers")
    except:
        pass

    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=DefaultEmbeddingFunction()
    )

    for i, chunk in enumerate(all_chunks):
        collection.add(
            documents=[chunk.page_content.encode('ascii', 'ignore').decode('ascii')],
            metadatas=[{"source": str(chunk.metadata.get("source", "")),
                        "page": int(chunk.metadata.get("page", 0))}],
            ids=[f"chunk_{i}"]
        )
    return collection, len(all_chunks)


def rewrite_query(gemini, original_query, chat_history):
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent
        )

    rewrite_prompt = f"""You are a search query optimizer for academic papers.
Rewrite the following question into a better search query that will find 
relevant chunks in research papers. Make it more specific with technical terms.
Return ONLY the rewritten query, nothing else.

Recent conversation (for context):
{history_text}

Original question: {original_query}
Rewritten query:"""
    response = gemini.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=rewrite_prompt
    )
    return response.text.strip()


def evaluate_retrieval(gemini, query, chunks):
    relevant = 0
    for chunk in chunks:
        eval_prompt = f"""Is the following text chunk relevant to answering this question?
Question: {query}
Chunk: {chunk[:300]}
Answer with ONLY 'yes' or 'no'."""
        response = gemini.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=eval_prompt
        )
        if "yes" in response.text.strip().lower():
            relevant += 1
    return relevant, len(chunks)


def build_prompt_with_history(context, chat_history, current_question):
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:300]}" for m in recent
        )

    prompt = f"""You are a research assistant helping with literature surveys.
Answer based only on the context below from research papers.
If the answer is not in the context, say so clearly.
Be specific and cite which paper the information comes from when possible.
Use the conversation history to understand follow-up questions and references like "it", "they", "this method" etc.

Context from papers:
{context}

Conversation history:
{history_text}

Current question: {current_question}
Answer:"""
    return prompt


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Research Literature Assistant", page_icon="🔬")
st.title("🔬 Research Literature Assistant")
st.markdown("Upload your research papers and ask questions across all of them.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF papers",
    type="pdf",
    accept_multiple_files=True
)

if not api_key:
    st.warning("👈 Enter your Gemini API key in the sidebar to begin")
    st.stop()

if not uploaded_files:
    st.info("👈 Upload one or more PDF research papers in the sidebar to begin")
    st.stop()

# ── Ingestion ────────────────────────────────────────────────────────────────
file_names = sorted([f.name for f in uploaded_files])

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "collection" not in st.session_state:
    st.session_state.collection = None

if file_names != st.session_state.loaded_files:
    with st.spinner(f"Processing {len(uploaded_files)} paper(s)..."):
        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            pdf_paths.append(path)

        collection, total_chunks = ingest_uploaded_files(pdf_paths)
        st.session_state.collection = collection
        st.session_state.loaded_files = file_names
        shutil.rmtree(temp_dir)

    st.sidebar.success(f"✅ {len(uploaded_files)} paper(s) loaded — {total_chunks} chunks")

collection = st.session_state.collection

# ── Gemini connection ────────────────────────────────────────────────────────
try:
    gemini = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Gemini connection error: {e}")
    st.stop()

# ── Loaded papers list ───────────────────────────────────────────────────────
st.sidebar.markdown("**Loaded papers:**")
for name in file_names:
    st.sidebar.markdown(f"- {name}")

# ── Chat interface ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question across your papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching across papers..."):
            try:
                # Step 1 — Rewrite query (history-aware)
                rewritten = rewrite_query(gemini, prompt, st.session_state.messages[:-1])
                st.caption(f"🔍 Search query: *{rewritten}*")

                # Step 2 — Retrieve chunks
                results = collection.query(
                    query_texts=[rewritten],
                    n_results=3
                )
                context = "\n\n".join(results['documents'][0])
                sources = results['metadatas'][0]

                # Step 3 — Generate answer (with conversation history)
                full_prompt = build_prompt_with_history(
                    context,
                    st.session_state.messages[:-1],
                    prompt
                )
                response = gemini.models.generate_content(
                    model="models/gemini-2.0-flash",
                    contents=full_prompt
                )
                answer = response.text
                st.markdown(answer)

                # Step 4 — Retrieval evaluation
                relevant, total = evaluate_retrieval(
                    gemini, prompt, results['documents'][0]
                )
                precision = round((relevant / total) * 100)
                st.markdown(f"**📊 Retrieval Quality:** {relevant}/{total} chunks relevant — {precision}%")

                # Step 5 — Citations
                st.markdown("---")
                st.markdown("**📄 Citations:**")
                seen = set()
                for s in sources:
                    src = f"{os.path.basename(s['source'])} — Page {s['page']+1}"
                    if src not in seen:
                        st.markdown(f"- {src}")
                        seen.add(src)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error: {e}")
