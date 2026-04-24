import json
import agent.tools

from agent.tools.registry import get_tool_schemas, call_tool
from agent.llm_client import get_llm_client
from postgres_db import get_schema
from config import LLM_MODEL


MAX_ITERATIONS = 2


def _format_result(tool_name: str, raw) -> dict:
    """Format raw tool output into a result dict."""
    if tool_name == "tool_search":
        return {
            "response": raw,
            "tool_used": "tool_search",
            "file_path": None,
            "last_search_response": raw
        }
    elif tool_name == "tool_sql":
        return {
            "response": raw,
            "tool_used": "tool_sql",
            "file_path": None,
            "last_search_response": None
        }
    elif tool_name == "tool_file":
        if raw == "NO_CONTENT":
            return {
                "response": "No answer to save yet. Please ask a question first.",
                "tool_used": "tool_file",
                "file_path": None,
                "last_search_response": None
            }
        return {
            "response": f"Response saved to: `{raw}`",
            "tool_used": "tool_file",
            "file_path": raw,
            "last_search_response": None
        }
    elif tool_name == "tool_time":
        return {
            "response": raw,
            "tool_used": "tool_time",
            "file_path": None,
            "last_search_response": None
        }
    return {
        "response": f"Unknown tool: {tool_name}",
        "tool_used": None,
        "file_path": None,
        "last_search_response": None
    }


def run_agent(
    user_message: str,
    session_id: str,
    last_response: str = None,
    top_k: int = 3,
    temperature: float = 0.7,
    db_credentials: dict = None
) -> dict:
    """
    Agent loop: keeps calling tools until the LLM returns
    a final answer with no more tool calls.
    """

    # ✅ Fetch schema and inject into system prompt
    schema_context = ""
    if db_credentials:
        try:
            schema = get_schema(credentials=db_credentials)
            schema_context = (
                f"\n\nDatabase Schema (use this to understand the DB):\n"
                f"{schema}\n"
                f"Use this schema to understand what tables and columns "
                f"exist before deciding to call tool_sql.\n"
            )
        except Exception:
            schema_context = "\n\nDatabase: connection available but schema fetch failed.\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools.\n\n"
                "Tool usage rules:\n"
                "- Use tool_search when the user asks about the content "
                "of an uploaded document.\n"
                "- Use tool_sql when the user asks about data from a "
                "database — for example counts, totals, records, filters, "
                "statistics, or when they explicitly say 'from the db', "
                "'query the database', 'in the database'.\n"
                "- When using tool_sql, ALWAYS pass the user's COMPLETE "
                "original question as-is — never split or simplify it. "
                "The tool handles splitting internally.\n"
                "- IMPORTANT: Call tool_sql ONLY ONCE per user message.\n"
                "- Use tool_file when the user explicitly asks to save, "
                "export, or create a file.\n"
                "- Use tool_time when the user asks for the current time "
                "or date.\n"
                "- For greetings, small talk, or anything unrelated — "
                "respond directly WITHOUT calling any tool.\n\n"
                "Examples:\n"
                "- 'hi' → no tool\n"
                "- 'what is the refund policy?' → tool_search\n"
                "- 'how many orders in 2023?' → tool_sql\n"
                "- 'save this to a file' → tool_file\n"
                "- 'what time is it?' → tool_time\n"
                + schema_context  # ✅ DB schema injected here
            )
        },
        {"role": "user", "content": user_message}
    ]

    final_result = None
    tools_used = []
    current_last_response = last_response

    for _ in range(MAX_ITERATIONS):
        response = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            if not tools_used:
                return {
                    "response": message.content,
                    "tool_used": None,
                    "file_path": None,
                    "last_search_response": None
                }
            if final_result is None:
                return {
                    "response": "Agent completed but produced no result.",
                    "tool_used": None,
                    "file_path": None,
                    "last_search_response": None
                }
            if final_result.get("tool_used") != "tool_sql":
                final_result["response"] = (
                    message.content if message.content
                    else final_result["response"]
                )
            return final_result

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            raw = call_tool(
                tool_name,
                tool_args,
                session_id=session_id,
                last_response=current_last_response,
                top_k=top_k,
                temperature=temperature,
                db_credentials=db_credentials  # ✅ pass credentials to tools
            )

            final_result = _format_result(tool_name, raw)
            tools_used.append(tool_name)

            if tool_name == "tool_search":
                current_last_response = raw

            if tool_name == "tool_sql":
                return final_result  # ✅ return immediately

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(raw)
            })

    return final_result or {
        "response": "Agent reached max iterations without a final answer.",
        "tool_used": None,
        "file_path": None,
        "last_search_response": None
    }