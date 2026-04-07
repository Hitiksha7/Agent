from datetime import datetime
from agent.tools.registry import tool


@tool(
    description=(
        "Get the current date and time. Use this when the user "
        "asks what time it is, today's date, or current datetime."
    ),
    params={},
    required=[]
)
def tool_time(**kwargs) -> str:
    """Tool 3: Return the current date and time."""
    now = datetime.now()
    return now.strftime("🕐 Current time: %H:%M:%S | 📅 Date: %A, %B %d, %Y")