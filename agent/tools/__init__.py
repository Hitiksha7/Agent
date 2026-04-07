# Importing each tool module triggers the @tool decorator,
# which registers them into TOOL_REGISTRY automatically.
from agent.tools import tool_search  # noqa: F401
from agent.tools import tool_file    # noqa: F401
from agent.tools import tool_time    # noqa: F401

from agent.tools.registry import get_tool_schemas, call_tool

__all__ = ["get_tool_schemas", "call_tool"]