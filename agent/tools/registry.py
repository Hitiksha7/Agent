import inspect

TOOL_REGISTRY: dict = {}  # name -> {"func": callable, "schema": dict}


def tool(description: str, params: dict = None, required: list = None):
    """
    Decorator to register a function as an agent tool.

    Args:
        description: What this tool does (shown to the LLM)
        params: OpenAI-style parameter properties dict
        required: List of required parameter names
    """
    def decorator(func):
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params or {},
                    "required": required or []
                }
            }
        }

        TOOL_REGISTRY[func.__name__] = {
            "func": func,
            "schema": schema
        }

        return func
    return decorator


def get_tool_schemas() -> list[dict]:
    """Return all registered tool schemas for OpenAI function calling."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def call_tool(name: str, args: dict, **kwargs):
    """
    Execute a registered tool by name.
    Only passes kwargs that the tool function actually accepts,
    preventing TypeError from unexpected keyword arguments.

    Args:
        name: Tool function name
        args: Arguments parsed from OpenAI tool call
        **kwargs: Extra runtime args (top_k, temperature, last_response)
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: '{name}'")

    func = TOOL_REGISTRY[name]["func"]
    sig = inspect.signature(func)
    params = sig.parameters

    # Check if function accepts **kwargs — if so pass everything
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    if accepts_var_keyword:
        # Function has **kwargs — pass all
        return func(**args, **kwargs)
    else:
        # Filter to only accepted parameter names
        accepted = {
            k: v for k, v in {**args, **kwargs}.items()
            if k in params
        }
        return func(**accepted)