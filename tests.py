import pytest
from unittest.mock import MagicMock, patch
from app import rewrite_query, evaluate_retrieval, build_prompt_with_history, ingest_uploaded_files

# ── Helper: create a fake Gemini client ──────────────────────────────────────
def make_fake_gemini(return_text):
    gemini = MagicMock()
    gemini.models.generate_content.return_value.text = return_text
    return gemini

# ── Test 1: rewrite_query returns non-empty string ───────────────────────────
def test_rewrite_query_returns_string():
    gemini = make_fake_gemini("attention mechanism transformer architecture")
    result = rewrite_query(gemini, "how does attention work", [])
    assert isinstance(result, str)
    assert len(result) > 0

# ── Test 2: rewrite_query returns something different from original ───────────
def test_rewrite_query_is_different():
    original = "how does attention work"
    gemini = make_fake_gemini("attention mechanism query key value transformer")
    result = rewrite_query(gemini, original, [])
    assert result != original

# ── Test 3: evaluate_retrieval single call returns correct counts ─────────────
def test_evaluate_retrieval_counts():
    # New version makes ONE call returning comma-separated yes/no
    gemini = make_fake_gemini("yes,yes,yes,no,no")
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    relevant, total = evaluate_retrieval(gemini, "test query", chunks)
    assert total == 5
    assert relevant == 3

# ── Test 4: evaluate_retrieval all relevant ───────────────────────────────────
def test_evaluate_retrieval_all_relevant():
    gemini = make_fake_gemini("yes,yes,yes")
    chunks = ["chunk1", "chunk2", "chunk3"]
    relevant, total = evaluate_retrieval(gemini, "test query", chunks)
    assert total == 3
    assert relevant == 3

# ── Test 5: evaluate_retrieval none relevant ──────────────────────────────────
def test_evaluate_retrieval_none_relevant():
    gemini = make_fake_gemini("no,no,no")
    chunks = ["chunk1", "chunk2", "chunk3"]
    relevant, total = evaluate_retrieval(gemini, "test query", chunks)
    assert total == 3
    assert relevant == 0

# ── Test 6: build_prompt_with_history includes history ───────────────────────
def test_prompt_includes_history():
    history = [
        {"role": "user", "content": "how does attention work"},
        {"role": "assistant", "content": "attention uses query key value pairs"}
    ]
    prompt = build_prompt_with_history(
        context="some paper context here",
        chat_history=history,
        current_question="how is it different in Swin?"
    )
    assert "how does attention work" in prompt
    assert "attention uses query key value pairs" in prompt
    assert "how is it different in Swin?" in prompt

# ── Test 7: build_prompt_with_history works with empty history ────────────────
def test_prompt_works_with_no_history():
    prompt = build_prompt_with_history(
        context="some paper context here",
        chat_history=[],
        current_question="what is a transformer?"
    )
    assert "what is a transformer?" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 0

# ── Test 8: ingest creates collection with correct chunk count ────────────────
def test_ingest_chunk_count():
    with patch("app.PyPDFLoader") as mock_loader:
        from langchain_core.documents import Document
        mock_loader.return_value.load.return_value = [
            Document(page_content="This is page one content about transformers " * 20,
                     metadata={"source": "test.pdf", "page": 0}),
            Document(page_content="This is page two content about attention " * 20,
                     metadata={"source": "test.pdf", "page": 1}),
            Document(page_content="This is page three content about vision " * 20,
                     metadata={"source": "test.pdf", "page": 2}),
        ]
        collection, total_chunks = ingest_uploaded_files(["test.pdf"])
        assert total_chunks > 0
        assert collection is not None
