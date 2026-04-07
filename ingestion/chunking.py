import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Null-byte placeholder — won't appear in any real document text
_PLACEHOLDER = "\x00PROTECT\x00"


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences robustly.
    Handles abbreviations (Dr., Mr., e.g., etc.) and decimals.
    """
    # Protect common abbreviations
    abbreviations = r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|e\.g|i\.e|vs|Fig)\.'
    text = re.sub(abbreviations, rf'\1{_PLACEHOLDER}', text)

    # Protect decimals like 3.14
    text = re.sub(r'(\d)\.(\d)', rf'\1{_PLACEHOLDER}\2', text)

    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore protected dots
    sentences = [s.replace(_PLACEHOLDER, '.') for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into chunks with overlap while preserving sentence boundaries.

    Args:
        text: Raw input text
        chunk_size: Max characters per chunk
        chunk_overlap: Characters of overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    sentences = _split_sentences(text.replace('\n', ' '))
    chunks = []
    current_chunk: list[str] = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # Overlap: carry over sentences from end of current chunk
            overlap_chunk: list[str] = []
            overlap_size = 0
            for s in reversed(current_chunk):
                s_size = len(s) + 1  # +1 for joining space
                if overlap_size + s_size <= chunk_overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += s_size
                else:
                    break

            current_chunk = overlap_chunk + [sentence]
            current_size = overlap_size + sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for joining space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks