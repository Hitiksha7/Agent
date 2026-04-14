import sqlite3
from datetime import datetime

DB_PATH = "chat_history.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_documents (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id  TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunks   INTEGER NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def create_chat(chat_id: str):
    """
    Insert a new empty chat row immediately.
    Called when New Chat is clicked — before any message is sent.
    This ensures the chat appears in the list right away.
    """
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chats (id, title, created_at, updated_at) "
            "VALUES (?, 'New Chat', ?, ?)",
            (chat_id, now, now)
        )
        conn.commit()


def save_chat(chat_id: str, messages: list, doc_list: list):
    """
    Save messages and docs for a chat.
    Updates title from first user message.
    Uses upsert so it never creates duplicates.
    """
    if not chat_id:
        return

    # Build title from first user message
    title = "New Chat"
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if str(content).strip():
                text = str(content).strip()
                title = text[:40] + ("..." if len(text) > 40 else "")
                break

    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM chats WHERE id=?", (chat_id,)
        ).fetchone()

        if exists:
            conn.execute(
                "UPDATE chats SET title=?, updated_at=? WHERE id=?",
                (title, now, chat_id)
            )
        else:
            conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) "
                "VALUES (?,?,?,?)",
                (chat_id, title, now, now)
            )

        # Replace messages
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        valid_msgs = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if isinstance(role, str) and isinstance(content, str):
                valid_msgs.append((chat_id, role, content, now))
        if valid_msgs:
            conn.executemany(
                "INSERT INTO messages (chat_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                valid_msgs
            )

        # Replace docs
        conn.execute("DELETE FROM chat_documents WHERE chat_id=?", (chat_id,))
        if doc_list:
            conn.executemany(
                "INSERT INTO chat_documents (chat_id, filename, chunks) "
                "VALUES (?,?,?)",
                [(chat_id, d["name"], d["chunks"]) for d in doc_list]
            )

        conn.commit()


def load_chat(chat_id: str):
    """Load messages and docs for a chat."""
    with get_conn() as conn:
        msg_rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE chat_id=? ORDER BY id",
            (chat_id,)
        ).fetchall()
        messages = [
            {"role": r["role"], "content": r["content"]}
            for r in msg_rows
        ]

        doc_rows = conn.execute(
            "SELECT filename, chunks FROM chat_documents WHERE chat_id=?",
            (chat_id,)
        ).fetchall()
        docs = [
            {"name": r["filename"], "chunks": r["chunks"]}
            for r in doc_rows
        ]

    return messages, docs


def list_chats():
    """Return ALL chats ordered by most recently updated."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM chats "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {"id": r["id"], "title": r["title"], "updated_at": r["updated_at"]}
        for r in rows
    ]


def delete_chat(chat_id: str):
    """Delete a chat and all its messages/docs (cascade)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        conn.commit()