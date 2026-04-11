import os
import time
import uuid
import gradio as gr
from ingestion.collection import store_document, init_collection
from agent.agent import run_agent
from config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, TEMPERATURE
)

# ── Document upload handler ───────────────────────────────
def upload_document(file, session_id, chunk_size, chunk_overlap, doc_list):
    if file is None:
        return doc_list, "No file selected."

    already = [d["name"] for d in doc_list]
    filename = os.path.basename(file.name)

    if filename in already:
        return doc_list, f" {filename} already indexed."

    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, filename)
    with open(file.name, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())

    init_collection()
    num_chunks = store_document(
        file_path=dest,
        session_id=session_id,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )

    doc_list.append({"name": filename, "chunks": num_chunks})
    doc_display = "\n".join(
        [f"📎 {d['name']} · {d['chunks']} chunks" for d in doc_list]
    )
    return doc_list, f" {filename} stored! ({num_chunks} chunks)\n\n{doc_display}"


# ── Chat handler ──────────────────────────────────────────
def _normalize_history(history):
    """Ensure chatbot history is always in messages format."""
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(item)
            continue

        # Backward compatibility for tuple/pair format: (user, assistant)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})

    return normalized


def _stream_chunks(text: str, chunk_size: int = 25):
    """Yield growing text chunks for UI streaming."""
    if not text:
        yield ""
        return

    for i in range(chunk_size, len(text) + chunk_size, chunk_size):
        yield text[:i]


def chat(
    user_message, history, session_id, last_response,
    doc_list, top_k, temperature
):
    if not user_message.strip():
        yield _normalize_history(history), last_response
        return

    history = _normalize_history(history)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})
    yield history, last_response

    result = run_agent(
        user_message=user_message,
        session_id=session_id,
        last_response=last_response,
        top_k=int(top_k),
        temperature=float(temperature)
    )

    response = result["response"]
    if result.get("tool_used"):
        response += f"\n\n🔧 *Tool: `{result['tool_used']}`*"
    if result.get("file_path"):
        response += f"\n *Saved: `{result['file_path']}`*"

    for partial in _stream_chunks(response):
        history[-1]["content"] = partial
        yield history, last_response
        time.sleep(0.02)

    new_last = result.get("last_search_response") or last_response
    yield history, new_last


# ── New chat handler ──────────────────────────────────────
def new_chat():
    return [], str(uuid.uuid4()), None, [], "No documents uploaded yet."


# ── Reset chat handler ────────────────────────────────────
def reset_chat():
    return [], None


# ── Build UI ──────────────────────────────────────────────
with gr.Blocks(
    title="Agentic RAG"
) as demo:

    # ── State ─────────────────────────────────────────────
    session_id = gr.State(str(uuid.uuid4()))
    last_response = gr.State(None)
    doc_list = gr.State([])

    # ── Layout ────────────────────────────────────────────
    with gr.Row():

        # ── Sidebar ───────────────────────────────────────
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            gr.Markdown("## ")

            new_chat_btn = gr.Button("＋ New Chat", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Settings")

            top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="Top K")
            temperature = gr.Slider(
                0.0, 1.0, value=TEMPERATURE, step=0.1, label="Temperature"
            )

            with gr.Accordion("Advanced", open=False):
                chunk_size = gr.Slider(
                    100, 2000, value=CHUNK_SIZE, step=100, label="Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    0, 200, value=CHUNK_OVERLAP, step=10, label="Chunk Overlap"
                )

            gr.Markdown("---")
            gr.Markdown("###  Documents")

            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )
            doc_status = gr.Textbox(
                value="No documents uploaded yet.",
                label="Status",
                interactive=False,
                lines=3
            )

            gr.Markdown("---")
            reset_btn = gr.Button(" Reset Chat", variant="secondary")

        # ── Main chat area ────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="New Chat",
                height=520,
                show_label=True
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

    # ── Event handlers ────────────────────────────────────

    # Upload document
    file_upload.change(
        fn=upload_document,
        inputs=[file_upload, session_id, chunk_size, chunk_overlap, doc_list],
        outputs=[doc_list, doc_status]
    )

    # Send message via button
    send_btn.click(
        fn=chat,
        inputs=[
            msg_input, chatbot, session_id,
            last_response, doc_list, top_k, temperature
        ],
        outputs=[chatbot, last_response]
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )

    # Send message via Enter key
    msg_input.submit(
        fn=chat,
        inputs=[
            msg_input, chatbot, session_id,
            last_response, doc_list, top_k, temperature
        ],
        outputs=[chatbot, last_response]
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )

    # New chat
    new_chat_btn.click(
        fn=new_chat,
        outputs=[chatbot, session_id, last_response, doc_list, doc_status]
    )

    # Reset chat
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot, last_response]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css="""
        .sidebar { min-width: 260px; max-width: 260px; }
        .chatbot { height: 500px; overflow-y: auto; }
        footer { display: none !important; }
        """,
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True
    )