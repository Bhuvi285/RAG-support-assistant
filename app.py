"""Streamlit UI for the RAG Support Assistant."""
import hmac
import os
from pathlib import Path

import streamlit as st

from config import AGENT_PASSWORD, DATA_PATH, GROQ_API_KEY
from graph import run_graph
from hitl import log_escalation, get_pending_escalations, resolve_escalation
from ingest import ensure_vectorstore, ingest_pdf

st.set_page_config(page_title="RAG Support Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG-Based Customer Support Assistant")
st.caption("Groq · ChromaDB · LangGraph · Human-in-the-loop review")

# The checked-in sample PDF is indexed automatically on first start. The index can
# be rebuilt from that PDF after a free host restarts and clears its local storage.
if "knowledge_base_ready" not in st.session_state:
    try:
        with st.spinner("Preparing the knowledge base on first startup..."):
            ensure_vectorstore()
        st.session_state.knowledge_base_ready = True
    except Exception:
        st.error("The knowledge base could not be initialized. Check the app logs and confirm data/knowledge_base.pdf exists.")
        st.stop()

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY is not configured. Add it under the app's Secrets settings to enable generated answers.")

with st.sidebar:
    st.header("🔐 Agent tools")
    if not AGENT_PASSWORD:
        st.info("Agent tools are locked. The app owner must configure AGENT_PASSWORD in Streamlit Secrets to enable PDF updates and the escalation queue.")
        agent_authenticated = False
    else:
        if "agent_authenticated" not in st.session_state:
            st.session_state.agent_authenticated = False

        if not st.session_state.agent_authenticated:
            entered_password = st.text_input("Agent password", type="password", key="agent_password")
            if st.button("Unlock agent tools"):
                st.session_state.agent_authenticated = hmac.compare_digest(
                    entered_password.encode("utf-8"), AGENT_PASSWORD.encode("utf-8")
                )
                if not st.session_state.agent_authenticated:
                    st.error("Incorrect password.")
                st.rerun()

        agent_authenticated = bool(st.session_state.get("agent_authenticated", False))
        if agent_authenticated and st.button("Lock agent tools"):
            st.session_state.agent_authenticated = False
            st.rerun()

    if agent_authenticated:
        st.divider()
        st.subheader("Knowledge base")
        uploaded_pdf = st.file_uploader("Replace knowledge-base PDF", type=["pdf"])
        if uploaded_pdf and st.button("Rebuild index from uploaded PDF"):
            target = Path(DATA_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with st.spinner("Rebuilding the knowledge base..."):
                    target.write_bytes(uploaded_pdf.getvalue())
                    ingest_pdf(str(target))
                st.session_state.knowledge_base_ready = True
                st.success("Knowledge base updated.")
                st.rerun()
            except Exception:
                st.error("The PDF could not be indexed. Check the app logs.")

        st.divider()
        st.subheader("📋 Pending escalations")
        try:
            escalations = get_pending_escalations()
        except Exception:
            escalations = []
            st.warning("The escalation log could not be read. Check the app logs.")

        if escalations:
            for esc in escalations:
                with st.expander(f"❗ {esc['query'][:40]}..."):
                    st.write(f"**Query:** {esc['query']}")
                    st.write(f"**Time:** {esc['timestamp']}")
                    human_reply = st.text_area("Agent reply", key=f"reply_{esc['timestamp']}")
                    if st.button("Resolve", key=f"resolve_{esc['timestamp']}"):
                        resolve_escalation(esc["timestamp"], human_reply)
                        st.success("Escalation resolved.")
                        st.rerun()
        else:
            st.info("No pending escalations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("escalated"):
            st.warning("This query was flagged for human review.")
        if msg.get("context"):
            with st.expander("📄 Retrieved context"):
                st.text(msg["context"])

if prompt := st.chat_input("Ask a support question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                result = run_graph(prompt)
                answer = result["answer"]
                escalated = result["needs_escalation"]
                context = result.get("context", "")
                if escalated:
                    try:
                        log_escalation(prompt, "Escalation keyword detected or insufficient retrieval confidence")
                    except Exception:
                        # Keep the user-facing answer available even if the host's
                        # ephemeral local filesystem cannot save this record.
                        st.warning("The escalation could not be saved. The app owner should check deployment logs.")
                st.markdown(answer)
                if escalated:
                    st.warning("This query was flagged for human review.")
                with st.expander("📄 Retrieved context"):
                    st.text(context or "No context retrieved.")
            except Exception:
                answer = "Sorry, I couldn't process that request. The app owner may need to check the deployment configuration or logs."
                escalated = False
                context = ""
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "escalated": escalated,
        "context": context,
    })
