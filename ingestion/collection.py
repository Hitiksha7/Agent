import uuid
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from openai import OpenAI

from ingestion.loader import read_document
from ingestion.chunking import split_text
from config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    EMBEDDING_MODEL, VECTOR_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    OPENAI_API_KEY
)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant once and reuse."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    """OpenAI client for embeddings — created once and reused."""
    return OpenAI(api_key=OPENAI_API_KEY)


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embeddings API.
    Returns a list of float vectors in the same order as input.
    """
    client = get_embedding_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    # Sort by index to guarantee order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def init_collection() -> None:
    """Create Qdrant collection only if it doesn't exist."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


def store_document(
    file_path: str,
    session_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> int:
    """Load, chunk, embed and upsert a document into Qdrant."""
    client = get_qdrant_client()

    raw_text = read_document(file_path)
    chunks = split_text(raw_text, chunk_size, chunk_overlap)

    # Embed all chunks in one API call (more efficient)
    embeddings = embed(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "source": file_path,
                "chunk_index": i,
                "session_id": session_id
            }
        )
        for i, (chunk, embedding) in enumerate[tuple[str, list[float]]][tuple[str, list[float]]](zip[tuple[str, list[float]]](chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)


def search(
    query: str,
    session_id: str,
    top_k: int = TOP_K
) -> list[str]:
    """Embed query and search Qdrant filtered by session_id."""
    client = get_qdrant_client()

    # Embed the query using same model
    query_vector = embed([query])[0]

    session_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(value=session_id)
            )
        ]
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=session_filter,
        limit=top_k
    )
    return [r.payload["text"] for r in results.points]