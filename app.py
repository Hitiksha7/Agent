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

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;}
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        min-width: 300px !important;
        transform: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Startup validation ────────────────────────────────────
missing = [
    name for name, val in {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not val
]
if missing:
    st.error(f"Missing env vars: {', '.join(missing)}")
    st.stop()


# ── Helper ────────────────────────────────────────────────
def new_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "documents": [],
        "session_id": str(uuid.uuid4()),
        "last_response": None,
    }
    return chat_id


# ── Session State ─────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = new_chat()

if "settings" not in st.session_state:
    st.session_state.settings = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "temperature": TEMPERATURE,
    }

# ── Sidebar ───────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("### 🤖 Agentic RAG")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.current_chat_id = new_chat()
        st.rerun()

    st.divider()

    # Chat list
    st.caption("CHATS")
    for chat_id, chat in list(st.session_state.chats.items()):
        is_active = chat_id == st.session_state.current_chat_id
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'▶ ' if is_active else ''}{chat['title']}"
            if st.button(
                label,
                key=f"chat_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{chat_id}"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat_id == chat_id:
                    if st.session_state.chats:
                        st.session_state.current_chat_id = list(
                            st.session_state.chats.keys()
                        )[-1]
                    else:
                        st.session_state.current_chat_id = new_chat()
                st.rerun()

    st.divider()

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.settings["chunk_size"] = st.slider(
            "Chunk Size", 100, 2000,
            st.session_state.settings["chunk_size"], step=100
        )
        st.session_state.settings["chunk_overlap"] = st.slider(
            "Chunk Overlap", 0, 200,
            st.session_state.settings["chunk_overlap"], step=10
        )
        st.session_state.settings["top_k"] = st.slider(
            "Top K", 1, 10,
            st.session_state.settings["top_k"]
        )
        st.session_state.settings["temperature"] = st.slider(
            "Temperature", 0.0, 1.0,
            st.session_state.settings["temperature"], step=0.1
        )

    st.divider()

    # Documents
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    st.caption("DOCUMENTS")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        key=f"upload_{st.session_state.current_chat_id}"
    )

    if uploaded_file:
        already = [d["name"] for d in current_chat["documents"]]
        if uploaded_file.name not in already:
            os.makedirs(DATA_DIR, exist_ok=True)
            temp_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing..."):
                init_collection()
                num_chunks = store_document(
                    file_path=temp_path,
                    session_id=current_chat["session_id"],
                    chunk_size=st.session_state.settings["chunk_size"],
                    chunk_overlap=st.session_state.settings["chunk_overlap"]
                )
                current_chat["documents"].append({
                    "name": uploaded_file.name,
                    "chunks": num_chunks
                })
            st.success(f" {uploaded_file.name}")
        else:
            st.info("Already indexed.")

    if current_chat["documents"]:
        for doc in current_chat["documents"]:
            st.caption(f"📎 {doc['name']} · {doc['chunks']} chunks")

    st.divider()

    if st.button(" Reset Chat", use_container_width=True):
        current_chat["messages"] = []
        current_chat["last_response"] = None
        st.rerun()

# ── Main chat area ────────────────────────────────────────
current_chat = st.session_state.chats[st.session_state.current_chat_id]

st.subheader(f" {current_chat['title']}")
st.divider()

# Messages
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_used"):
            st.caption(f"🔧 `{msg['tool_used']}`")
        if msg.get("file_path"):
            st.caption(f" `{msg['file_path']}`")

# Empty state
if not current_chat["messages"]:
    st.markdown(
        "<div style='text-align:center; color:#888; margin-top:5rem;'>"
        " Upload a document and start chatting"
        "</div>",
        unsafe_allow_html=True
    )

# ── Input ─────────────────────────────────────────────────
user_input = st.chat_input("Ask a question...")

if user_input:
    current_chat["messages"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        result = run_agent(
            user_message=user_input,
            session_id=current_chat["session_id"],
            last_response=current_chat["last_response"],
            top_k=st.session_state.settings["top_k"],
            temperature=st.session_state.settings["temperature"]
        )

    with st.chat_message("assistant"):
        st.write(result["response"])
        if result["tool_used"]:
            st.caption(f" `{result['tool_used']}`")
        if result["file_path"]:
            st.caption(f" `{result['file_path']}`")

    if result.get("last_search_response"):
        current_chat["last_response"] = result["last_search_response"]

    current_chat["messages"].append({
        "role": "assistant",
        "content": result["response"],
        "tool_used": result["tool_used"],
        "file_path": result["file_path"]
    })