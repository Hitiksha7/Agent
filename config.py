from dotenv import load_dotenv
import os

load_dotenv(override=True)

# ── Qdrant ────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Documents")

# ── LLM Provider ──────────────────────────────────────────
# Switch provider by changing LLM_PROVIDER in .env
# Options: openai, openrouter
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# # ── OpenAI ────────────────────────────────────────────────
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

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

# ── Max tokens ────────────────────────────────────────────
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
 