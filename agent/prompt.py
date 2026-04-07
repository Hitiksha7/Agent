import streamlit as st
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

MAX_TOKENS = 1024

# ── System Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Answer only from the provided context
- If the answer is not in the context, say "I don't know based on the provided document"
- Be concise and clear
- Do not make up information
"""


@st.cache_resource          # ✅ OpenAI client created once, reused
def get_openai_client() -> OpenAI:
    """Create OpenAI client once and reuse across all reruns."""
    return OpenAI(api_key=OPENAI_API_KEY)


def build_prompt(query: str, context_chunks: list[str]) -> list[dict]:
    """
    Build the messages payload for OpenAI from query and retrieved chunks.

    Args:
        query: User's question
        context_chunks: List of relevant text chunks from Qdrant

    Returns:
        List of messages for OpenAI chat completion
    """
    context = "\n\n".join(
        [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]


def generate_response(
    query: str,
    context_chunks: list[str],
    temperature: float
) -> str:
    """
    Generate a response from OpenAI given query and context chunks.

    Args:
        query: User's question
        context_chunks: Retrieved chunks from Qdrant
        temperature: LLM temperature

    Returns:
        Generated answer string
    """
    if not context_chunks:
        return "No relevant information found in the document."

    client = get_openai_client()
    messages = build_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        messages=messages
    )

    return response.choices[0].message.content