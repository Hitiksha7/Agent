from functools import lru_cache
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

MAX_TOKENS = 512

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Answer only from the provided context
- If the answer is not in the context, say "I don't know based on the provided document"
- Be concise and clear
- Do not make up information
"""


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Create OpenAI client once and reuse."""
    return OpenAI(api_key=OPENAI_API_KEY)


def build_prompt(query: str, context_chunks: list[str]) -> list[dict]:
    """Build messages payload for OpenAI."""
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
    """Generate response from OpenAI given query and context chunks."""
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
