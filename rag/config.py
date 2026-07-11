import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "Assistant DORA")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "")

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ASSETS_DIR = BASE_DIR / "assets"

PDF_PATH = DATA_DIR / "dora.pdf"
INDEX_PATH = VECTORSTORE_DIR / "dora_index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "dora_chunks.pkl"
LOGO_PATH = ASSETS_DIR / "logo.png"
STYLE_PATH = ASSETS_DIR / "style.css"


def validate_azure_settings() -> None:
    values = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
        "EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    }

    missing = [name for name, value in values.items() if not value]

    if missing:
        raise ValueError(
            "Variables Azure absentes : "
            + ", ".join(missing)
            + ". Vérifiez votre fichier .env ou la configuration Azure App Service."
        )
