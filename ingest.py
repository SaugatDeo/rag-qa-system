import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

def ingest_pdfs():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path1> <pdf_path2> ...")
        return

    pdf_paths = sys.argv[1:]
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
            print(f"Loaded {path} — {len(chunks)} chunks")
        except Exception as e:
            print(f"Skipped {path}: {e}")

    client = chromadb.PersistentClient(path="chroma_db")
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
        if i % 30 == 0:
            print(f"Processed {i}/{len(all_chunks)} chunks...")

    print(f"Done! Stored {len(all_chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    ingest_pdfs()
