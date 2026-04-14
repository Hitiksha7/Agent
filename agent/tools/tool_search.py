from agent.tools.registry import tool
from ingestion.collection import search
from agent.prompt import generate_response, stream_response


@tool(
    description=(
        "Search the uploaded document and answer the user's "
        "question using the vector database. Use this for any "
        "factual or content question about the document."
    ),
    params={
        "query": {
            "type": "string",
            "description": "The user's question to search for"
        }
    },
    required=["query"]
)
def tool_search(
    query: str,
    session_id: str,
    top_k: int = 3,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
):
    """
    Tool 1: Retrieve relevant chunks from Qdrant and generate answer.
    Supports both streaming and non-streaming modes.

    Args:
        stream: If True returns a generator, else returns a string
    """
    chunks = search(query, session_id=session_id, top_k=top_k)

    if stream:
        return stream_response(query, chunks, temperature)
    return generate_response(query, chunks, temperature)
