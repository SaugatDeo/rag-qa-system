import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

def ingest_pdfs():
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader("pdfs/")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=DefaultEmbeddingFunction()
    )

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"chunk_{i}"]
        )
        if i % 50 == 0:
            print(f"Processed {i}/{len(chunks)} chunks...")

    print(f"Done! Stored {len(chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    ingest_pdfs()