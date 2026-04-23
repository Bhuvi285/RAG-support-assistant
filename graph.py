# graph.py
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from retriever import retrieve_context

load_dotenv()

# ─── State Object ───────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    confidence: str        # "high" | "low"
    needs_escalation: bool
    human_response: str

# ─── LLM Setup ──────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ─── Node 1: Retrieve Context ────────────────────────────────────
def retrieve_node(state: GraphState) -> GraphState:
    query = state["query"]
    context, confidence = retrieve_context(query)
    return {
        **state,
        "context": context,
        "confidence": confidence
    }

# ─── Node 2: Generate Answer ─────────────────────────────────────
def generate_node(state: GraphState) -> GraphState:
    query = state["query"]
    context = state["context"]

    prompt = f"""You are a helpful and friendly customer support assistant for ShopEase.
Use ONLY the context below to answer the user's question.
Give a clear, well-structured answer using bullet points where needed.
If the context doesn't have enough information, say so clearly.

Context:
{context}

User Question: {query}

Answer in a clear, friendly and formatted way:"""

    response = llm.invoke(prompt)
    answer = response.content

    return {
        **state,
        "answer": answer,
        "needs_escalation": False
    }

# ─── Node 3: Escalate to Human ───────────────────────────────────
def escalate_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "⚠️ This query has been escalated to a human agent. You will receive a response shortly.",
        "needs_escalation": True
    }

# ─── Conditional Router ──────────────────────────────────────────
def route_query(state: GraphState) -> Literal["generate", "escalate"]:
    query = state["query"].lower()
    confidence = state["confidence"]

    # Escalation triggers
    escalation_keywords = [
        "refund", "legal", "lawsuit", "urgent", "angry",
        "complaint", "fraud", "cancel account", "speak to human",
        "manager", "supervisor"
    ]

    if any(kw in query for kw in escalation_keywords):
        return "escalate"

    if confidence == "low":
        return "escalate"

    return "generate"

# ─── Build the Graph ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate", escalate_node)

    # Entry point
    graph.set_entry_point("retrieve")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieve",
        route_query,
        {
            "generate": "generate",
            "escalate": "escalate"
        }
    )

    # Both end after their node
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()

# Run the graph
def run_graph(query: str, human_response: str = ""):
    app = build_graph()
    initial_state = GraphState(
        query=query,
        context="",
        answer="",
        confidence="high",
        needs_escalation=False,
        human_response=human_response
    )
    result = app.invoke(initial_state)
    return result