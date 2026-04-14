from typing import Generator
from agent.llm_client import get_llm_client
from config import LLM_MODEL, MAX_TOKENS

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Answer only from the provided context
- If the answer is not in the context, say "I don't know based on the provided document"
- Be concise and clear
- Do not make up information
"""


def build_prompt(query: str, context_chunks: list[str]) -> list[dict]:
    """Build messages payload for LLM from query and retrieved chunks."""
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
    """Generate full response (non-streaming) from LLM."""
    if not context_chunks:
        return "No relevant information found in the document."

    messages = build_prompt(query, context_chunks)

    response = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        messages=messages
    )
    return response.choices[0].message.content


def stream_response(
    query: str,
    context_chunks: list[str],
    temperature: float
) -> Generator[str, None, None]:
    """
    Stream response token by token from LLM.
    Yields growing text chunks as tokens arrive.
    """
    if not context_chunks:
        yield "No relevant information found in the document."
        return

    messages = build_prompt(query, context_chunks)

    stream = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        messages=messages,
        stream=True        # ← enable streaming
    )

    full_text = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_text += token
            yield full_text  # yield growing text for Gradio
