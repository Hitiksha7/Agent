import os
import uuid
import gradio as gr
from ingestion.collection import store_document, init_collection, search
from agent.agent import run_agent
from agent.prompt import stream_response
from config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, TEMPERATURE
)
from db import init_db, create_chat, save_chat, load_chat, list_chats, delete_chat

init_db()


def _get_chat_choices():
    chats = list_chats()
    id_map = {}
    seen = {}
    for c in chats:
        t = c["title"]
        if t in seen:
            seen[t] += 1
            t = f"{t} ({seen[t]})"
        else:
            seen[t] = 1
        id_map[t] = c["id"]
    choices = list(id_map.keys())
    return choices, id_map


def _doc_status(doc_list):
    if not doc_list:
        return "No documents uploaded yet."
    return "\n".join([f" {d['name']} · {d['chunks']} chunks" for d in doc_list])


def upload_document(file, session_id, chunk_size, chunk_overlap, doc_list):
    if file is None:
        return doc_list, "No file selected."
    filename = os.path.basename(file)
    already = [d["name"] for d in doc_list]
    if filename in already:
        return doc_list, _doc_status(doc_list)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, filename)
    with open(file, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    init_collection()
    num_chunks = store_document(
        file_path=dest,
        session_id=session_id,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )
    doc_list = doc_list + [{"name": filename, "chunks": num_chunks}]
    save_chat(session_id, [], doc_list)
    return doc_list, _doc_status(doc_list)


def chat(
    user_message, history, session_id, last_response,
    doc_list, top_k, temperature, db_creds
):
    if not user_message.strip():
        yield history, last_response
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""}
    ]
    yield history, last_response

    result = run_agent(
        user_message=user_message,
        session_id=session_id,
        last_response=last_response,
        top_k=int(top_k),
        temperature=float(temperature),
        db_credentials=db_creds
    )

    tool_used = result.get("tool_used")
    file_path = result.get("file_path")

    if tool_used == "tool_search":
        chunks = search(
            query=user_message,
            session_id=session_id,
            top_k=int(top_k)
        )
        streamed = ""
        for partial in stream_response(user_message, chunks, float(temperature)):
            streamed = partial
            history[-1] = {"role": "assistant", "content": streamed}
            yield history, last_response
        suffix = f"\n\n🔧 *Tool: `{tool_used}`*"
        if file_path:
            suffix += f"\n📁 *Saved: `{file_path}`*"
        history[-1] = {"role": "assistant", "content": streamed + suffix}
        new_last = streamed
    else:
        response = result.get("response") or ""
        if tool_used:
            response += f"\n\n🔧 *Tool: `{tool_used}`*"
        if file_path:
            response += f"\n📁 *Saved: `{file_path}`*"
        history[-1] = {"role": "assistant", "content": response}
        new_last = result.get("last_search_response") or last_response

    save_chat(session_id, history, doc_list)
    choices, _ = _get_chat_choices()
    yield history, new_last


def new_chat(current_history, current_session, current_docs):
    if current_history:
        save_chat(current_session, current_history, current_docs)
    new_id = str(uuid.uuid4())
    create_chat(new_id)
    choices, id_map = _get_chat_choices()
    return (
        [], new_id, None, [],
        _doc_status([]),
        gr.Dropdown(choices=choices, value=None),
        id_map
    )


def select_chat(selected_title):
    if not selected_title:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    _, id_map = _get_chat_choices()
    if selected_title not in id_map:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    chat_id = id_map[selected_title]
    messages, docs = load_chat(chat_id)
    return messages, chat_id, None, docs, _doc_status(docs)


def delete_selected(selected_title, current_session):
    if not selected_title:
        choices, id_map = _get_chat_choices()
        return gr.Dropdown(choices=choices, value=None), id_map, gr.update(), gr.update(), gr.update(), gr.update()
    _, id_map = _get_chat_choices()
    if selected_title not in id_map:
        choices, new_map = _get_chat_choices()
        return gr.Dropdown(choices=choices, value=None), new_map, gr.update(), gr.update(), gr.update(), gr.update()
    chat_id = id_map[selected_title]
    delete_chat(chat_id)
    choices, new_map = _get_chat_choices()
    if chat_id == current_session:
        new_id = str(uuid.uuid4())
        create_chat(new_id)
        choices, new_map = _get_chat_choices()
        return (
            gr.Dropdown(choices=choices, value=None),
            new_map, [], new_id, [], _doc_status([])
        )
    return (
        gr.Dropdown(choices=choices, value=None),
        new_map,
        gr.update(), gr.update(), gr.update(), gr.update()
    )


def reset_chat():
    return [], None


def refresh_chat_list():
    choices, id_map = _get_chat_choices()
    return gr.Dropdown(choices=choices, value=None), id_map


def connect_db(host, port, name, user, password):
    if not all([host, name, user, password]):
        return None, "⚠️ Please fill all fields."
    credentials = {
        "host": host,
        "port": int(port),
        "dbname": name,
        "user": user,
        "password": password
    }
    from postgres_db import test_connection
    success, message = test_connection(credentials)
    if success:
        return credentials, f"✅ Connected to `{name}`"
    else:
        return None, f"❌ {message}"


# ── Build UI ──────────────────────────────────────────────

_init_choices, _init_map = _get_chat_choices()

with gr.Blocks(
    title="Agentic RAG",
    theme=gr.themes.Soft(),
    css="""
    :root { color-scheme: dark; }
    .gradio-container { background-color: #1a1a2e !important; }
    footer { display: none !important; }
    .chat-dropdown select { max-height: 300px; overflow-y: auto; }
    """
) as demo:

    session_id    = gr.State(str(uuid.uuid4()))
    last_response = gr.State(None)
    doc_list      = gr.State([])
    id_map        = gr.State(_init_map)
    db_creds      = gr.State(None)

    with gr.Row():

        # ── Left: DB Connection ───────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## Agentic RAG")
            gr.Markdown("#### 🗄️ Database Connection")
            db_host     = gr.Textbox(placeholder="localhost", label="Host", value="localhost")
            db_port     = gr.Textbox(value="5432", label="Port")
            db_name     = gr.Textbox(placeholder="postgres", label="Database Name")
            db_user     = gr.Textbox(placeholder="postgres", label="Username")
            db_password = gr.Textbox(placeholder="password", label="Password", type="password")
            db_connect_btn = gr.Button(" Connect", variant="primary", size="sm")
            db_status   = gr.Textbox(value="Not connected", label="Status", interactive=False)

        # ── Middle: Chat ──────────────────────────────────
        with gr.Column(scale=2, min_width=500):
            chatbot = gr.Chatbot(
                label="Chat",
                height=700,
                show_label=False,
                placeholder="Upload a document/database and start chatting...",
                render_markdown=True
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False,
                    scale=9,
                    container=False,
                    autofocus=True
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        # ── Right: Settings + Chats + Documents ──────────
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("#### ⚙️ Settings")
            top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="Top K")
            temperature = gr.Slider(0.0, 1.0, value=TEMPERATURE, step=0.1, label="Temperature")

            with gr.Accordion("Advanced", open=False):
                chunk_size = gr.Slider(100, 2000, value=CHUNK_SIZE, step=100, label="Chunk Size")
                chunk_overlap = gr.Slider(0, 200, value=CHUNK_OVERLAP, step=10, label="Chunk Overlap")

            gr.Markdown("---")
            reset_btn = gr.Button("🔄 Reset Chat", variant="secondary", size="sm")

            gr.Markdown("---")
            gr.Markdown("#### 🕘 Recent Chats")
            new_chat_btn = gr.Button("＋ New Chat", variant="primary", size="sm")
            chat_selector = gr.Dropdown(
                choices=_init_choices,
                value=None,
                label="",
                show_label=False,
                interactive=True,
                elem_classes="chat-dropdown"
            )
            delete_btn = gr.Button("🗑 Delete Selected", variant="stop", size="sm")

            gr.Markdown("---")
            gr.Markdown("#### 📄 Documents")
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )
            doc_status = gr.Textbox(
                value="No documents uploaded yet.",
                label="Status",
                interactive=False,
                lines=2
            )

    # ── Event wiring ──────────────────────────────────────

    db_connect_btn.click(
        fn=connect_db,
        inputs=[db_host, db_port, db_name, db_user, db_password],
        outputs=[db_creds, db_status]
    )

    file_upload.change(
        fn=upload_document,
        inputs=[file_upload, session_id, chunk_size, chunk_overlap, doc_list],
        outputs=[doc_list, doc_status]
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, session_id, last_response, doc_list, top_k, temperature, db_creds],
        outputs=[chatbot, last_response]
    ).then(fn=lambda: "", outputs=msg_input
    ).then(fn=refresh_chat_list, outputs=[chat_selector, id_map])

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, session_id, last_response, doc_list, top_k, temperature, db_creds],
        outputs=[chatbot, last_response]
    ).then(fn=lambda: "", outputs=msg_input
    ).then(fn=refresh_chat_list, outputs=[chat_selector, id_map])

    new_chat_btn.click(
        fn=new_chat,
        inputs=[chatbot, session_id, doc_list],
        outputs=[chatbot, session_id, last_response, doc_list, doc_status, chat_selector, id_map]
    )

    chat_selector.change(
        fn=select_chat,
        inputs=[chat_selector],
        outputs=[chatbot, session_id, last_response, doc_list, doc_status]
    )

    delete_btn.click(
        fn=delete_selected,
        inputs=[chat_selector, session_id],
        outputs=[chat_selector, id_map, chatbot, session_id, doc_list, doc_status]
    )

    reset_btn.click(fn=reset_chat, outputs=[chatbot, last_response])

    demo.load(fn=lambda: _init_map, outputs=[id_map])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True
    )