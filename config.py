from dotenv import load_dotenv
import os

load_dotenv()

# ── Qdrant ────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Documents")

# ── OpenAI ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ── Embedding ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# ── Search ────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 3))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# ── Data ──────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")