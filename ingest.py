# ingest.py
# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_PATH = "chroma_db"
DATA_PATH = "data/knowledge_base.pdf"

def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages")

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks")

    # Create embeddings using free HuggingFace model
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Store in ChromaDB
    print("💾 Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print("✅ ChromaDB ready!")
    return vectorstore

def load_vectorstore():
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    return vectorstore

if __name__ == "__main__":
    ingest_pdf()