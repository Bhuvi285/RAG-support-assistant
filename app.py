# app.py
import streamlit as st
import os
from graph import run_graph
from hitl import log_escalation, get_pending_escalations, resolve_escalation
from ingest import ingest_pdf

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Based Customer Support Assistant")
st.caption("Powered by Groq LLM · ChromaDB · LangGraph · HITL")

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload Knowledge Base PDF", type="pdf")
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        with open("data/knowledge_base.pdf", "wb") as f:
            f.write(uploaded_file.read())
        if st.button("📥 Ingest PDF into ChromaDB"):
            with st.spinner("Processing PDF..."):
                ingest_pdf()
            st.success("✅ PDF ingested successfully!")

    st.divider()
    st.header("📋 Pending Escalations")
    escalations = get_pending_escalations()
    if escalations:
        for esc in escalations:
            with st.expander(f"❗ {esc['query'][:40]}..."):
                st.write(f"**Query:** {esc['query']}")
                st.write(f"**Time:** {esc['timestamp']}")
                human_reply = st.text_area("Your Reply:", key=esc['timestamp'])
                if st.button("✅ Resolve", key=f"resolve_{esc['timestamp']}"):
                    resolve_escalation(esc['timestamp'], human_reply)
                    st.success("Resolved!")
                    st.rerun()
    else:
        st.info("No pending escalations")

# ─── Chat Interface ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("escalated"):
            st.warning("🔺 This query was escalated to a human agent")
        if msg.get("context"):
            with st.expander("📄 Retrieved Context"):
                st.text(msg["context"])

# Chat Input
if prompt := st.chat_input("Ask a support question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_graph(prompt)

        answer = result["answer"]
        escalated = result["needs_escalation"]
        context = result.get("context", "")

        st.markdown(answer)

        if escalated:
            st.warning("🔺 Escalated to human agent")
            log_escalation(prompt, "Low confidence or escalation keyword")

        with st.expander("📄 Retrieved Context"):
            st.text(context if context else "No context retrieved")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "escalated": escalated,
        "context": context
    })