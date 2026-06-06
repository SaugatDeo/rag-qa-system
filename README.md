# 🔬 Research Literature Survey Assistant

A Retrieval-Augmented Generation (RAG) application that lets researchers upload multiple research papers and ask natural language questions across all of them — with source citations, retrieval quality scores, and conversation history awareness.

## 🚀 Live Demo

**Deployed:** [Open App on Streamlit Cloud](https://rag-app-system-2jmeywtk4bmhdc8fwfc3a7.streamlit.app)

Ask questions like:
- *"How does the attention mechanism work in the Transformer?"*
- *"Compare how ResNet and VGG handle deep network training"*
- *"What datasets are used across these papers?"*
- *"How does Stable Diffusion use cross-attention differently from Swin Transformer?"*

---

## 🧠 How It Works

```
User uploads PDFs
        ↓
Text extracted + split into 1000-char overlapping chunks
        ↓
Each chunk converted to 384-dim vector (all-MiniLM-L6-v2)
        ↓
Vectors stored in ChromaDB (persistent vector database)
        ↓
User asks question
        ↓
Gemini rewrites question into optimised search query
        ↓
ChromaDB finds top-k most similar chunks (cosine similarity)
        ↓
Gemini evaluates chunk relevance (precision@k)
        ↓
Gemini generates grounded answer with citations
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Dynamic PDF upload | Upload any number of research papers through the UI |
| 🔍 Query rewriting | Gemini rewrites user questions into precise technical search queries |
| 🗃️ Semantic search | ChromaDB vector search across all uploaded papers simultaneously |
| 📊 Retrieval evaluation | Precision@k metric shows percentage of retrieved chunks that are relevant |
| 📖 Source citations | Every answer shows the exact paper name and page number |
| 💬 Conversation history | Follow-up questions understand context from previous messages |
| ✅ Unit tested | 8 pytest unit tests with 100% function coverage |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Vector Database | ChromaDB (persistent) |
| Embeddings | all-MiniLM-L6-v2 via ONNX runtime |
| LLM | Google Gemini 2.5 Flash |
| PDF Loading | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter (chunk=1000, overlap=200) |
| Web Interface | Streamlit |
| Testing | pytest + unittest.mock |

---

## 📁 Project Structure

```
rag-qa-system/
├── app.py            # Streamlit chat interface — main application
├── ingest.py         # Standalone PDF ingestion pipeline
├── tests.py          # 8 unit tests (pytest)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/SaugatDeo/rag-qa-system.git
cd rag-qa-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Gemini API key
Go to [aistudio.google.com](https://aistudio.google.com) → Get API Key → Create API Key (free, no credit card)

### 5. Launch the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), paste your API key in the sidebar, upload PDFs and start asking questions.

---

## 🧪 Running Tests

```bash
python -m pytest tests.py -v
```

Expected output:
```
8 passed in ~80s
```

Tests cover: query rewriting, retrieval evaluation (all relevant, none relevant, partial), prompt building with and without history, and PDF ingestion pipeline.

---

## 📐 Architecture Decisions

**Why ChromaDB?** Persistent local vector DB with zero infrastructure cost. No API calls for storage — vectors live on disk.

**Why query rewriting?** Raw user questions are often vague. Rewriting to include technical terms (e.g. "W-MSA", "cross-attention", "residual connections") significantly improves retrieval precision.

**Why precision@k?** Without evaluation, users can't tell if the retrieved chunks are actually relevant. A single Gemini call judges all chunks at once and shows a confidence score.

**Why conversation history?** Multi-turn research conversations use pronouns like "it", "they", "this method". Without history context, follow-up questions fail.

---

## 📄 Sample Papers to Test With

These are freely available on arxiv:

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) — Transformer
- [Deep Residual Learning](https://arxiv.org/pdf/1512.03385) — ResNet
- [Swin Transformer](https://arxiv.org/pdf/2103.14030) — Swin
- [High-Resolution Image Synthesis](https://arxiv.org/pdf/2112.10752) — Stable Diffusion
- [An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929) — ViT

---

## 👤 Author

**Saugat Deo**
B.Tech Electronics & Instrumentation Engineering — NIT Rourkela (First Class, CGPA 7.30)
Research background: Computer vision, deep learning, gait analysis, medical AI

[GitHub](https://github.com/SaugatDeo) · [LinkedIn](https://linkedin.com/in/SaugatDeo)
