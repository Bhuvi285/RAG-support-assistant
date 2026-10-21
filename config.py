"""Runtime configuration for local development and Streamlit Community Cloud."""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a Streamlit Cloud secret first, then fall back to environment/.env."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value)
    except Exception:
        # st.secrets may be unavailable when running outside Streamlit or when no
        # secrets file has been configured. Falling back to environment is expected.
        pass
    value = os.getenv(name, default)
    return value if value is None or str(value).strip() else default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
AGENT_PASSWORD = get_secret("AGENT_PASSWORD")

# Local Chroma storage is rebuildable from the PDF. On free hosted tiers, treat it
# as cache rather than durable storage because the filesystem may be recycled.
CHROMA_PATH = str(Path(get_secret("CHROMA_PATH", str(BASE_DIR / "chroma_db"))))
DATA_PATH = str(Path(get_secret("DATA_PATH", str(BASE_DIR / "data" / "knowledge_base.pdf"))))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4

ESCALATION_KEYWORDS = [
    "refund", "legal", "lawsuit", "urgent", "angry", "complaint", "fraud",
    "cancel account", "speak to human", "manager", "supervisor",
]
