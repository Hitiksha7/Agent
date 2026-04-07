import os
import docx
from pypdf import PdfReader


def read_text_file(file_path: str) -> str:
    """Read content from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file using pypdf."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def read_docx_file(file_path: str) -> str:
    """Read content from a Word document."""
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_document(file_path: str) -> str:
    """
    Read document content based on file extension.
    Supports: .txt, .pdf, .docx
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.txt':
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")