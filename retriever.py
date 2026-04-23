# retriever.py
from ingest import load_vectorstore

def retrieve_context(query: str, k: int = 4):
    """
    Retrieve top-k relevant chunks from ChromaDB.
    Returns (context_text, confidence_level)
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found.", "low"

    # Build context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Simple confidence check: if we got fewer than 2 docs, low confidence
    confidence = "high" if len(docs) >= 2 else "low"

    return context, confidence