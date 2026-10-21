"""PDF ingestion and cached ChromaDB access for the RAG assistant."""
from pathlib import Path
import shutil

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, DATA_PATH


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model():
    """Load one embedding model per running app process."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Open the persistent vector store, reusing the connection across reruns."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model(),
    )


def ingest_pdf(pdf_path: str | None = None):
    """Replace the vector index with chunks from the selected PDF."""
    source = Path(pdf_path or DATA_PATH)
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge-base PDF not found: {source}")

    # Parse and chunk first, so a bad PDF does not destroy a previously built index.
    documents = PyPDFLoader(str(source)).load()
    if not documents:
        raise ValueError("The selected PDF did not contain any readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be extracted from the selected PDF.")

    # Clear cached handles before replacing on-disk Chroma data. This also avoids
    # accidentally appending duplicate chunks after a repeat ingestion.
    load_vectorstore.clear()
    chroma_dir = Path(CHROMA_PATH)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=CHROMA_PATH,
    )
    load_vectorstore.clear()
    return vectorstore


def ensure_vectorstore():
    """Build the index on first startup if Chroma is absent or empty."""
    chroma_dir = Path(CHROMA_PATH)
    if chroma_dir.exists():
        try:
            vectorstore = load_vectorstore()
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception:
            # An incomplete/empty cache can safely be rebuilt from the source PDF.
            load_vectorstore.clear()

    return ingest_pdf(DATA_PATH)


if __name__ == "__main__":
    ingest_pdf()
    print("Knowledge base indexed successfully.")
