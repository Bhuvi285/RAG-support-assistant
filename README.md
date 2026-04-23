# 🤖 RAG-Based Customer Support Assistant

> **Powered by Groq LLM · ChromaDB · LangGraph · HITL · Streamlit**

A production-grade AI Customer Support system built using Retrieval-Augmented Generation (RAG), LangGraph workflow orchestration, and Human-in-the-Loop (HITL) escalation. The system reads a PDF knowledge base, retrieves relevant context using vector embeddings, and generates accurate answers using Groq's LLaMA 3.3 model.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Core Concepts Explained](#-core-concepts-explained)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Step-by-Step Development Flow](#-step-by-step-development-flow)
- [Input & Output Flow](#-input--output-flow)
- [LangGraph Workflow](#-langgraph-workflow)
- [HITL Escalation System](#-hitl-escalation-system)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Testing the System](#-testing-the-system)
- [Errors Faced & Solutions](#-errors-faced--solutions)
- [Key Design Decisions](#-key-design-decisions)
- [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

### What is this project?

This project is a **RAG-Based Customer Support Assistant** for a fictional e-commerce company called **ShopEase**. Instead of a static chatbot with hardcoded answers, this system:

1. **Reads** a company PDF knowledge base (policies, FAQs, procedures)
2. **Understands** customer questions using semantic search
3. **Retrieves** the most relevant sections from the PDF
4. **Generates** accurate, context-aware answers using an LLM
5. **Escalates** complex or sensitive queries to human agents automatically

### What Problem Does It Solve?

Traditional customer support either uses:
- **Rule-based bots** → rigid, can't handle variations in questions
- **Pure LLMs** → hallucinate answers not based on actual company policy

**RAG solves this** by grounding the LLM's answers in real, verified documents.

### Input → Output Example

```
Customer:  "What is the return policy for electronics?"
   ↓
System searches PDF → Finds Section 4.1 → Retrieves relevant chunks
   ↓
Groq LLaMA 3.3 forms answer from context
   ↓
Bot: "Electronics can be returned within 7 days in an unopened box only."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Customer:  "I want a refund, this is FRAUD!"
   ↓
LangGraph detects "fraud" keyword → Routes to ESCALATION node
   ↓
Bot: "⚠️ This query has been escalated to a human agent."
     Query logged in escalation_log.json for human review
```

---

## 🧠 Core Concepts Explained

### 1. RAG (Retrieval-Augmented Generation)

RAG is a technique that combines two things:
- **Retrieval**: Finding relevant information from a document/database
- **Generation**: Using an LLM to form a human-like answer from that information

```
Without RAG:  User Question → LLM → Answer (may hallucinate)
With RAG:     User Question → Search Documents → Retrieve Context → LLM → Grounded Answer
```

**Why RAG?**
- LLMs have a knowledge cutoff and don't know your company's specific policies
- RAG lets the LLM answer based on YOUR documents, not its training data
- Answers are accurate, verifiable, and up-to-date

---

### 2. Embeddings

An **embedding** is a numerical representation (vector) of text. Words or sentences with similar meaning have similar vectors.

```
"return policy"     → [0.23, 0.87, 0.12, ...]  ← vector of numbers
"refund procedure"  → [0.21, 0.85, 0.14, ...]  ← similar vector!
"cricket match"     → [0.91, 0.03, 0.67, ...]  ← very different
```

We use the `all-MiniLM-L6-v2` model from SentenceTransformers to convert text chunks into embeddings.

---

### 3. Vector Database (ChromaDB)

ChromaDB stores embeddings so we can search them efficiently. When a user asks a question:

1. The question is converted to an embedding
2. ChromaDB finds the most **similar embeddings** (nearest neighbors)
3. The corresponding text chunks are returned as context

```
Query Embedding → ChromaDB Similarity Search → Top-K Relevant Chunks
```

**Why ChromaDB?**
- Free and open source
- Runs locally, no external API needed
- Persistent storage — embeddings survive app restarts
- Easy integration with LangChain

---

### 4. Chunking

PDFs are split into smaller pieces called **chunks** before storing in ChromaDB.

```
Full PDF (10 pages)
    ↓
Chunks of 500 characters with 50 character overlap
    ↓
[Chunk 1][Chunk 2][Chunk 3]...[Chunk N]
```

**Why overlap?** So that context at the boundary of two chunks is not lost.

**Settings used:**
- `chunk_size = 500` characters
- `chunk_overlap = 50` characters
- Strategy: `RecursiveCharacterTextSplitter`

---

### 5. LangGraph

LangGraph is a framework for building **stateful, graph-based AI workflows**. Instead of a simple linear chain, it allows:

- **Nodes**: Individual processing steps (retrieve, generate, escalate)
- **Edges**: Connections between nodes
- **Conditional Routing**: Different paths based on logic
- **State**: Data that flows between all nodes

```
[User Query]
     ↓
[RETRIEVE Node] ← searches ChromaDB
     ↓
[ROUTER] ← decides the path
   ↙        ↘
[GENERATE]  [ESCALATE]
   ↓              ↓
[Answer]    [Human Alert]
```

---

### 6. HITL (Human-in-the-Loop)

HITL is a design pattern where **humans are brought into the AI workflow** for cases the AI cannot or should not handle alone.

**Escalation triggers in this system:**
- Keywords: "fraud", "refund", "lawsuit", "manager", "supervisor", "complaint", "urgent"
- Low confidence: fewer than 2 relevant chunks retrieved
- Explicitly requests human: "speak to human", "talk to agent"

**What happens after escalation:**
1. Customer gets an immediate acknowledgment message
2. Query is logged in `escalation_log.json` with timestamp
3. Human agent sees it in the Streamlit sidebar
4. Agent types a reply and clicks "Resolve"
5. Query is marked as resolved

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                      │
│         (app.py - User chat interface)                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  LANGGRAPH WORKFLOW                       │
│                    (graph.py)                            │
│                                                          │
│   [Retrieve Node] → [Router] → [Generate Node]          │
│                          └───→ [Escalate Node]           │
└────────┬───────────────────────────────┬────────────────┘
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌─────────────────────┐
│   RETRIEVER     │            │    GROQ LLM          │
│  (retriever.py) │            │  llama-3.3-70b       │
│                 │            │  (via langchain-groq) │
└────────┬────────┘            └─────────────────────┘
         │
         ▼
┌─────────────────┐            ┌─────────────────────┐
│    CHROMADB     │            │    HITL MODULE       │
│  Vector Store   │            │    (hitl.py)         │
│  (chroma_db/)   │            │ escalation_log.json  │
└────────┬────────┘            └─────────────────────┘
         │
         ▼
┌─────────────────┐
│  INGEST MODULE  │
│  (ingest.py)    │
│  PDF → Chunks   │
│  → Embeddings   │
└─────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (LLaMA 3.3 70B) | Fast, free inference |
| **Vector DB** | ChromaDB | Store & search embeddings |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Convert text to vectors |
| **Workflow** | LangGraph | Graph-based AI pipeline |
| **PDF Loading** | LangChain PyPDFLoader | Extract text from PDF |
| **Chunking** | RecursiveCharacterTextSplitter | Split PDF into chunks |
| **Web UI** | Streamlit | Interactive chat interface |
| **Env Config** | python-dotenv | Manage API keys securely |
| **Language** | Python 3.11 | Core language |

---

## 📁 Project Structure

```
rag_support_assistant/
│
├── app.py                  # Streamlit UI — main entry point
├── ingest.py               # PDF loading, chunking, ChromaDB storage
├── graph.py                # LangGraph workflow definition
├── retriever.py            # ChromaDB query & context retrieval
├── hitl.py                 # Human-in-the-Loop escalation logic
│
├── .env                    # API keys (never commit to git!)
├── requirements.txt        # All dependencies
│
├── data/
│   └── knowledge_base.pdf  # Company PDF knowledge base
│
├── chroma_db/              # Auto-created — ChromaDB storage
│   └── ...
│
├── escalation_log.json     # Auto-created — HITL escalation logs
│
└── venv/                   # Virtual environment (Python 3.11)
```

---

## 🔨 Step-by-Step Development Flow

### Phase 1: Environment Setup

**Goal:** Create an isolated Python environment with all required packages.

**Why virtual environment?**
- Keeps project dependencies separate from system Python
- Avoids version conflicts between projects
- Reproducible setup for other developers

```bash
# Create venv with Python 3.11
C:\Users\...\Python311\python.exe -m venv venv

# Activate
venv\Scripts\activate

# Install all packages
pip install langchain langchain-community langchain-groq langgraph
pip install chromadb sentence-transformers streamlit pypdf python-dotenv
pip install langchain-text-splitters langchain-chroma
```

**Key concept:** Python 3.11 was specifically required because newer versions (3.13, 3.14) don't have pre-built wheels for ML packages like `tokenizers`.

---

### Phase 2: Knowledge Base Creation (ingest.py)

**Goal:** Load a PDF, split it into chunks, create embeddings, store in ChromaDB.

**Step-by-step flow inside ingest.py:**

```
PDF File
   ↓ PyPDFLoader
Raw Text (page by page)
   ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
List of Chunks
   ↓ SentenceTransformerEmbeddings (all-MiniLM-L6-v2)
List of Embedding Vectors
   ↓ Chroma.from_documents()
ChromaDB (stored in chroma_db/ folder)
```

**Why `all-MiniLM-L6-v2`?**
- Free, runs locally — no API needed
- Small and fast (only 22MB)
- Good quality for semantic search tasks

**Important fix:** In newer ChromaDB versions, `.persist()` is automatic. Calling it manually throws `AttributeError: 'Chroma' object has no attribute 'persist'`. Solution: remove the `.persist()` call entirely.

---

### Phase 3: Retrieval Module (retriever.py)

**Goal:** Given a user query, find the most relevant chunks from ChromaDB.

**How it works:**
1. Load existing ChromaDB vectorstore
2. Convert query to embedding using the same model
3. Find top-4 most similar chunks (cosine similarity)
4. Return chunks as context + confidence level

**Confidence logic:**
```python
confidence = "high" if len(docs) >= 2 else "low"
```
If fewer than 2 chunks are found, the system doesn't have enough information → triggers HITL escalation.

---

### Phase 4: LangGraph Workflow (graph.py)

**Goal:** Define the AI workflow as a graph with nodes, edges, and conditional routing.

**State object** — data shared across all nodes:
```python
class GraphState(TypedDict):
    query: str          # user's question
    context: str        # retrieved PDF chunks
    answer: str         # LLM's response
    confidence: str     # "high" or "low"
    needs_escalation: bool
    human_response: str
```

**Three nodes:**

| Node | What it does |
|---|---|
| `retrieve_node` | Calls retriever.py to get relevant chunks from ChromaDB |
| `generate_node` | Sends context + query to Groq LLM, gets formatted answer |
| `escalate_node` | Returns escalation message, skips LLM entirely |

**Conditional router logic:**
```python
def route_query(state):
    # Check for escalation keywords
    if any(keyword in query for keyword in escalation_keywords):
        return "escalate"
    # Check for low confidence
    if confidence == "low":
        return "escalate"
    # Otherwise generate answer
    return "generate"
```

**Graph edges:**
```
Entry → retrieve_node
retrieve_node → [conditional] → generate_node OR escalate_node
generate_node → END
escalate_node → END
```

---

### Phase 5: HITL Module (hitl.py)

**Goal:** Log escalated queries, allow human agents to review and respond.

**Three functions:**

```python
log_escalation(query, reason)
# → Saves to escalation_log.json with timestamp + status="pending"

get_pending_escalations()
# → Returns all entries where status="pending"

resolve_escalation(timestamp, human_reply)
# → Updates status="resolved" and saves human's reply
```

**Data structure in escalation_log.json:**
```json
[
  {
    "timestamp": "2026-04-23T10:15:30",
    "query": "I want a refund, this is fraud!",
    "reason": "Escalation keyword detected",
    "status": "resolved",
    "human_reply": "We apologize for the inconvenience..."
  }
]
```

---

### Phase 6: Streamlit UI (app.py)

**Goal:** Build an interactive web chat interface with sidebar controls.

**UI Components:**
- **Sidebar**: PDF upload button, "Ingest PDF" button, Pending Escalations list
- **Main area**: Chat history, message input box
- **Per message**: Answer text, escalation warning, "Retrieved Context" expander

**Session state** is used to maintain chat history:
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

**Full request flow in app.py:**
```
User types question
    ↓
st.chat_input captures it
    ↓
run_graph(prompt) called
    ↓
LangGraph executes (retrieve → route → generate/escalate)
    ↓
Result displayed in chat
    ↓
If escalated → log_escalation() called → appears in sidebar
```

---

## 🔄 Input & Output Flow

### Normal Query Flow
```
1. User: "How do I reset my password?"
2. app.py → calls run_graph("How do I reset my password?")
3. graph.py → retrieve_node → retriever.py → ChromaDB search
4. ChromaDB returns top 4 chunks from Section 6.2 of PDF
5. confidence = "high" (4 chunks found)
6. Router → "generate"
7. generate_node → Groq LLaMA 3.3 → formatted answer
8. Answer displayed in chat UI
```

### Escalation Flow
```
1. User: "I want to file a lawsuit against you!"
2. app.py → calls run_graph(...)
3. graph.py → retrieve_node (still retrieves context)
4. Router checks: "lawsuit" in escalation_keywords → True
5. Router → "escalate" (skips LLM entirely)
6. escalate_node → returns standard escalation message
7. app.py → log_escalation() → saved to escalation_log.json
8. Sidebar shows new pending escalation
9. Human agent types reply → resolve_escalation() called
```

---

## 🔀 LangGraph Workflow

```
                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   RETRIEVE   │  ← searches ChromaDB
                    │    Node      │  ← sets confidence level
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   ROUTER     │  ← conditional edge
                    └──────┬───────┘
              ┌────────────┴────────────┐
              │ "generate"              │ "escalate"
              ▼                         ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │  GENERATE Node  │      │   ESCALATE Node      │
    │  Groq LLM call  │      │   No LLM needed      │
    │  Returns answer │      │   Returns alert msg  │
    └────────┬────────┘      └──────────┬───────────┘
             │                          │
             └────────────┬─────────────┘
                          ▼
                        [END]
```

---

## 👤 HITL Escalation System

### When Does Escalation Happen?

| Trigger | Example Query |
|---|---|
| Keyword: fraud | "This is fraud!" |
| Keyword: refund | "I want a refund now" |
| Keyword: lawsuit | "I'll file a lawsuit" |
| Keyword: manager | "Get me a manager" |
| Keyword: supervisor | "I want your supervisor" |
| Keyword: complaint | "I want to make a complaint" |
| Keyword: urgent | "This is urgent!" |
| Low confidence | Query outside PDF scope |

### Human Agent Workflow

1. Agent opens Streamlit sidebar
2. Sees pending escalation with customer's query
3. Types a response in the text box
4. Clicks "Resolve"
5. Query marked as resolved in `escalation_log.json`

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 (specifically — not 3.12, 3.13, or 3.14)
- Groq API key (free at console.groq.com)
- Git (optional)

### Step 1: Clone / Create Project Folder
```bash
mkdir rag_support_assistant
cd rag_support_assistant
```

### Step 2: Create Virtual Environment
```bash
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --timeout 120 langchain langchain-community langchain-groq langgraph python-dotenv pypdf
pip install --timeout 120 streamlit
pip install --timeout 120 chromadb
pip install --timeout 300 sentence-transformers
pip install langchain-text-splitters langchain-chroma
```

### Step 4: Create `.env` File
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Place Knowledge Base PDF
```
data/knowledge_base.pdf
```

---

## ▶️ Running the Application

```bash
# Activate virtual environment
venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

Open browser at: `http://localhost:8501`

**First-time setup in UI:**
1. Click "Upload" in sidebar → select `knowledge_base.pdf`
2. Click "Ingest PDF into ChromaDB" → wait for ✅
3. Start asking questions!

---

## 🧪 Testing the System

### Normal RAG Queries
```
"What is the return policy for electronics?"
"How do I reset my password?"
"What payment methods are accepted?"
"How long does delivery take?"
"What are the ShopEase membership tiers?"
```

### HITL Escalation Queries
```
"I want a refund, this is fraud!"
"I need to speak to a manager urgently"
"I will file a lawsuit against your company"
"This is a complaint, get me a supervisor now"
```

### Edge Case Queries (outside PDF scope)
```
"What is the weather today?"
"Tell me about iPhone 15 features"
```

---

## 🐛 Errors Faced & Solutions

### Error 1: Python Version Incompatibility
```
Error: failed-wheel-build-for-install → tokenizers
ModuleNotFoundError: No module named 'tokenizers'
```
**Cause:** Python 3.14 is too new — `tokenizers` package hasn't built wheels for it yet.

**Solution:** Installed Python 3.11 specifically and created the virtual environment using its executable path:
```bash
C:\Users\...\Python311\python.exe -m venv venv
```

---

### Error 2: Streamlit Torchvision Warnings
```
ModuleNotFoundError: No module named 'torchvision'
[transformers] Accessing __path__ from .models...
```
**Cause:** Streamlit scans all installed packages including `transformers`, which has vision models requiring `torchvision`.

**Solution:** These are harmless warnings, not errors. Silenced by creating `.streamlit/config.toml`:
```toml
[logger]
level = "error"
```

---

### Error 3: Text Splitter Import Error
```
ModuleNotFoundError: No module named 'langchain.text_splitter'
```
**Cause:** In newer LangChain versions, `text_splitter` was moved to a separate package.

**Solution:** 
1. Installed the new package: `pip install langchain-text-splitters`
2. Updated import in `ingest.py`:
```python
# Old (broken):
from langchain.text_splitter import RecursiveCharacterTextSplitter

# New (correct):
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

### Error 4: ChromaDB Persist Error
```
AttributeError: 'Chroma' object has no attribute 'persist'
```
**Cause:** ChromaDB version 0.4+ automatically persists data — the `.persist()` method was removed.

**Solution:** Removed the `vectorstore.persist()` call from `ingest.py`. ChromaDB now auto-saves when `persist_directory` is specified.

---

### Error 5: Groq Model Decommissioned
```
groq.BadRequestError: The model `llama3-8b-8192` has been decommissioned
```
**Cause:** Groq retired the `llama3-8b-8192` model.

**Solution:** Updated model name in `graph.py`:
```python
# Old (broken):
model_name="llama3-8b-8192"

# New (correct):
model_name="llama-3.3-70b-versatile"
```

---

### Error 6: pip Download Timeout
```
TimeoutError: The read operation timed out
pip._vendor.urllib3.exceptions.ReadTimeoutError
```
**Cause:** Large packages (like `onnxruntime` at 12.9MB) timing out on slow internet.

**Solution:** Used `--timeout` flag and installed packages in separate smaller groups:
```bash
pip install --timeout 300 sentence-transformers
```

---

### Error 7: Two Commands Merged Accidentally
```
ERROR: Could not find a version that satisfies the requirement chromadbpip
```
**Cause:** Copy-pasted two commands together as one — `chromadb` and `pip` got merged into `chromadbpip`.

**Solution:** Run each command separately, wait for the prompt to return before typing the next command.

---

## 🎯 Key Design Decisions

### Why chunk_size=500?
- Small enough to be specific (precise retrieval)
- Large enough to contain full sentences and context
- Too small → fragmented context; Too large → irrelevant content mixed in

### Why top-k=4 chunks?
- Provides enough context for the LLM to form a complete answer
- Keeps prompt size manageable (avoids token limit issues)
- More than 4 chunks → dilutes relevance

### Why SentenceTransformers over OpenAI Embeddings?
- Completely free — no API cost
- Runs locally — no network dependency
- `all-MiniLM-L6-v2` is fast and accurate for semantic search

### Why LangGraph over simple LangChain chains?
- Supports conditional routing (generate vs escalate)
- Stateful — data flows cleanly between nodes
- Easy to add more nodes later (e.g., memory, feedback)
- Better for production-grade systems

### Why Groq over OpenAI?
- Free tier available
- Extremely fast inference (LLaMA 3.3 70B in milliseconds)
- No credit card required to get started

---

## 🚀 Future Enhancements

| Enhancement | Description |
|---|---|
| **Multi-document support** | Ingest multiple PDFs into the same ChromaDB |
| **Conversation memory** | Remember previous messages in the same session |
| **Feedback loop** | Users rate answers → improve retrieval over time |
| **Re-ranking** | Use a cross-encoder to re-rank retrieved chunks |
| **Authentication** | Login system for human agents |
| **Email notifications** | Alert agents via email on new escalations |
| **Analytics dashboard** | Track common queries, escalation rates |
| **Deployment** | Host on Streamlit Cloud for public access |
| **Multi-language** | Support queries in Hindi and other languages |

---

## 📊 Project Evaluation Mapping

| Criteria | Where It's Implemented |
|---|---|
| RAG Implementation | `ingest.py` + `retriever.py` |
| LangGraph Workflow | `graph.py` — nodes, edges, state |
| Conditional Routing | `route_query()` in `graph.py` |
| HITL Escalation | `hitl.py` + sidebar in `app.py` |
| Vector Database | ChromaDB in `chroma_db/` folder |
| Embedding Model | SentenceTransformers in `ingest.py` |
| LLM Integration | Groq LLaMA 3.3 in `graph.py` |
| Web Interface | Streamlit in `app.py` |

---

## 👨‍💻 Author

Built as part of the **Advanced GenAI Internship Program**

**Tech Stack:** Python 3.11 · LangChain · LangGraph · ChromaDB · Groq · Streamlit · SentenceTransformers

---

*This project demonstrates a production-ready RAG system with intelligent routing, vector search, and human oversight — not just a chatbot, but a scalable AI decision-making system.*
