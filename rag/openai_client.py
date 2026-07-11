from functools import lru_cache

from openai import OpenAI

from rag.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    validate_azure_settings,
)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Crée et conserve un client Azure OpenAI réutilisable.
    """

    validate_azure_settings()

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
        timeout=90.0,
        max_retries=2,
    )