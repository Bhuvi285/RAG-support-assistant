"""LangGraph workflow for retrieval, answer generation, and escalation."""
from typing import TypedDict, Literal

import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from config import ESCALATION_KEYWORDS, GROQ_API_KEY, GROQ_MODEL
from retriever import retrieve_context


class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    confidence: str
    needs_escalation: bool
    human_response: str


@st.cache_resource(show_spinner=False)
def get_llm():
    """Initialize Groq only when generation is actually needed."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your local .env file."
        )
    return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)


def retrieve_node(state: GraphState) -> GraphState:
    context, confidence = retrieve_context(state["query"])
    return {**state, "context": context, "confidence": confidence}


def generate_node(state: GraphState) -> GraphState:
    prompt = f"""You are a helpful and friendly customer support assistant for ShopEase.
Use ONLY the context below to answer the user's question. If the context does not
contain the answer, say that you do not have enough information. Do not invent policy.
Give a clear, well-structured answer using bullet points where useful.

Context:
{state['context']}

User question: {state['query']}

Answer:"""
    response = get_llm().invoke(prompt)
    return {**state, "answer": response.content, "needs_escalation": False}


def escalate_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "⚠️ This query has been flagged for review by a human support agent.",
        "needs_escalation": True,
    }


def route_query(state: GraphState) -> Literal["generate", "escalate"]:
    query = state["query"].lower()
    if any(keyword in query for keyword in ESCALATION_KEYWORDS):
        return "escalate"
    if state["confidence"] == "low":
        return "escalate"
    return "generate"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate", escalate_node)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_query,
        {"generate": "generate", "escalate": "escalate"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)
    return graph.compile()


def run_graph(query: str, human_response: str = ""):
    initial_state: GraphState = {
        "query": query,
        "context": "",
        "answer": "",
        "confidence": "high",
        "needs_escalation": False,
        "human_response": human_response,
    }
    return build_graph().invoke(initial_state)
