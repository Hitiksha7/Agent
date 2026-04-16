import json
import agent.tools

from agent.tools.registry import get_tool_schemas, call_tool
from agent.llm_client import get_llm_client
from config import LLM_MODEL


MAX_ITERATIONS = 5


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
    temperature: float = 0.7
) -> dict:
    """
    Agent loop: keeps calling tools until the LLM returns
    a final answer with no more tool calls.
    """
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
                "- Use tool_file when the user explicitly asks to save, "
                "export, or create a file.\n"
                "- Use tool_time when the user asks for the current time "
                "or date.\n"
                "- For greetings, small talk, or anything unrelated — "
                "respond directly WITHOUT calling any tool.\n\n"
                "- Use tool_sql when the user asks about data from a database...\n"
                "- 'how many users signed up this month?' → tool_sql\n"
                "- 'show me all orders above 1000' → tool_sql\n"
                "- 'query db for total revenue' → tool_sql\n"
                "Examples:\n"
                "- 'hi' → no tool\n"
                "- 'what is the refund policy?' → tool_search\n"
                "- 'how many users signed up this month?' → tool_sql\n"
                "- 'show me all orders above 1000' → tool_sql\n"
                "- 'query db for total revenue' → tool_sql\n"
                "- 'save this to a file' → tool_file\n"
                "- 'what time is it?' → tool_time\n"
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
                temperature=temperature
            )

            final_result = _format_result(tool_name, raw)
            tools_used.append(tool_name)

            if tool_name == "tool_search":
                current_last_response = raw

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