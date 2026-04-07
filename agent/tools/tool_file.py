import os
from datetime import datetime
from agent.tools.registry import tool
from config import DATA_DIR


@tool(
    description=(
        "Save the last response to a text file. Use this when "
        "the user asks to save, export, download, or create a "
        "file from the previous answer."
    ),
    params={
        "filename": {
            "type": "string",
            "description": "Optional name for the output file"
        }
    },
    required=[]
)
def tool_file(filename: str = None, last_response: str = None, **kwargs) -> str:
    """
    Tool 2: Save the last RAG response to a .txt file under DATA_DIR.
    """
    if not last_response:
        return "NO_CONTENT"

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.txt"

    if not filename.endswith(".txt"):
        filename += ".txt"

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(last_response)

    return file_path