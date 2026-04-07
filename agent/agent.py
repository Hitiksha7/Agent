import json
import streamlit as st
from openai import OpenAI

# Importing tools package triggers all @tool decorators → auto-registration
import agent.tools  # noqa: F401

from agent.tools.registry import get_tool_schemas, call_tool
from agent.tools.tool_search import tool_search
from config import OPENAI_API_KEY, LLM_MODEL

MAX_ITERATIONS = 5


@st.cache_resource
def get_openai_client() -> OpenAI:
    """Create OpenAI client once and reuse across all reruns."""
    return OpenAI(api_key=OPENAI_API_KEY)


def _format_result(tool_name: str, raw) -> dict:
    """Format raw tool output into a result dict."""
    if tool_name == "tool_search":
        return {
            "response": raw,
            "tool_used": "tool_search",
            "file_path": None,
            "last_search_response": raw
        }

    elif tool_name == "tool_file":
        if raw == "NO_CONTENT":
            return {
                "response": (
                    "No answer to save yet. "
                    "Please ask a question first."
                ),
                "tool_used": "tool_file",
                "file_path": None,
                "last_search_response": None
            }
        return {
            "response": f"✅ Response saved to: `{raw}`",
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

    The LLM naturally decides when to use tools and when to
    respond directly (greetings, small talk, off-topic, etc.)

    Args:
        user_message: The user's input from Streamlit
        session_id: Current session ID for scoped Qdrant search
        last_response: Last RAG answer (used by tool_file)
        top_k: Number of chunks to retrieve from Qdrant
        temperature: LLM temperature

    Returns:
        dict: response, tool_used, file_path, last_search_response
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful document assistant with access to tools.\n\n"
                "Tool usage rules:\n"
                "- Use tool_search ONLY when the user asks a question "
                "about the content of the uploaded document.\n"
                "- Use tool_file ONLY when the user explicitly asks to "
                "save, export, or create a file.\n"
                "- Use tool_time ONLY when the user asks for the current "
                "time or date.\n"
                "- For greetings, small talk, or anything unrelated to "
                "the document — respond directly WITHOUT calling any tool.\n\n"
                "Examples:\n"
                "- 'hi' → respond friendly, no tool\n"
                "- 'how are you' → respond friendly, no tool\n"
                "- 'what is the refund policy?' → use tool_search\n"
                "- 'save this to a file' → use tool_file\n"
                "- 'what time is it?' → use tool_time\n"
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    final_result = None
    tools_used = []
    current_last_response = last_response

    # ── Agent loop ────────────────────────────────────────
    for _ in range(MAX_ITERATIONS):

        response = get_openai_client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto"
        )

        message = response.choices[0].message

        # No tool calls → LLM responded directly (greeting, small talk etc.)
        if not message.tool_calls:

            # No tool was ever called and LLM gave a direct answer
            # → trust the LLM's direct response (greeting, off-topic etc.)
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

            # Use LLM summary if available, else keep last tool result
            final_result["response"] = (
                message.content
                if message.content
                else final_result["response"]
            )
            return final_result

        # Add assistant message to history
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

        # Execute each tool call
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

            # Always update so tool_file gets freshest search answer
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