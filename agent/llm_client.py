from functools import lru_cache

from openai import OpenAI

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def _valid_key(key: str | None) -> bool:
    """Basic guard against empty or placeholder API keys."""
    return bool(key and key.strip() and not key.strip().endswith("..."))


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Create LLM client once based on provider configuration."""
    if LLM_PROVIDER == "openrouter":
        # Common setup mistake: key is placed in OPENAI_API_KEY while using openrouter.
        openrouter_key = (
            OPENROUTER_API_KEY if _valid_key(OPENROUTER_API_KEY) else OPENAI_API_KEY
        )
        return OpenAI(
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
        )

    return OpenAI(api_key=OPENAI_API_KEY)