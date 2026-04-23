# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ChromaDB Configuration
CHROMA_PATH = "chroma_db"
DATA_PATH = "data/knowledge_base.pdf"

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Configuration
TOP_K_RESULTS = 4

# Escalation Keywords
ESCALATION_KEYWORDS = [
    "refund", "legal", "lawsuit", "urgent", "angry",
    "complaint", "fraud", "cancel account", "speak to human",
    "manager", "supervisor"
]