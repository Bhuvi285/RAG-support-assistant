"""Retrieve relevant chunks from the cached Chroma vector store."""
from config import TOP_K_RESULTS
from ingest import load_vectorstore


def retrieve_context(query: str, k: int = TOP_K_RESULTS):
    """Return retrieved context and a basic confidence label."""
    vectorstore = load_vectorstore()
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)

    if not docs:
        return "No relevant information found.", "low"

    context = "\n\n".join(doc.page_content for doc in docs)
    confidence = "high" if len(docs) >= 2 else "low"
    return context, confidence
