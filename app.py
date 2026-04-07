import os
import uuid
import streamlit as st
from ingestion.collection import store_document, init_collection
from agent.agent import run_agent
from config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, TEMPERATURE,
    QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY
)

# ── Startup validation ────────────────────────────────────
missing = [
    name for name, val in {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not val
]
if missing:
    st.error(
        f"❌ Missing env vars: {', '.join(missing)}. Check your .env file."
    )
    st.stop()

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("🤖 Agentic RAG")

# ── Sidebar Settings ──────────────────────────────────────
st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size", 100, 2000, CHUNK_SIZE, step=100
)
chunk_overlap = st.sidebar.slider(
    "Chunk Overlap", 0, 200, CHUNK_OVERLAP, step=10
)
top_k = st.sidebar.slider("Top K Results", 1, 10, TOP_K)
temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, TEMPERATURE, step=0.1
)

# ── Session State ─────────────────────────────────────────
for key, default in {
    "session_id": str(uuid.uuid4()),    # unique ID per browser session
    "last_response": None,              # latest tool_search answer
    "chat_history": [],
    "collection_ready": False,
    "last_uploaded_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.markdown("---")
st.sidebar.caption(f"🔑 Session: `{st.session_state.session_id[:8]}...`")

# ── File Upload ───────────────────────────────────────────
st.subheader("📤 Upload Document")
uploaded_file = st.file_uploader(
    "Upload a document", type=["pdf", "docx", "txt"]
)

if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded_file:
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_path = os.path.join(DATA_DIR, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            init_collection()
            num_chunks = store_document(
                file_path=temp_path,
                session_id=st.session_state.session_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.session_state.collection_ready = True
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.last_response = None
            st.session_state.chat_history = []

        st.success(
            f"✅ {uploaded_file.name} stored! ({num_chunks} chunks)"
        )
    else:
        st.info(f"📄 {uploaded_file.name} already indexed.")

# ── Chat Interface ────────────────────────────────────────
st.subheader("💬 Chat with your Document")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])
        if chat.get("tool_used"):
            st.caption(f"🔧 Tool used: `{chat['tool_used']}`")
        if chat.get("file_path"):
            st.caption(f"📁 File saved: `{chat['file_path']}`")

user_input = st.chat_input(
    "Ask a question or say 'save this to a file'..."
)

if user_input:
    if not st.session_state.collection_ready:
        st.warning("⚠️ Please upload a document first.")
    else:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Agent is thinking..."):
            result = run_agent(
                user_message=user_input,
                session_id=st.session_state.session_id,
                last_response=st.session_state.last_response,
                top_k=top_k,
                temperature=temperature
            )

        with st.chat_message("assistant"):
            st.write(result["response"])
            if result["tool_used"]:
                st.caption(f"🔧 Tool used: `{result['tool_used']}`")
            if result["file_path"]:
                st.caption(f"📁 File saved: `{result['file_path']}`")

        # ── Fix: use last_search_response to always track
        # the latest search answer correctly
        if result.get("last_search_response"):
            st.session_state.last_response = result["last_search_response"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "tool_used": result["tool_used"],
            "file_path": result["file_path"],
            "last_search_response": result["last_search_response"]
        })