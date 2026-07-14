import os
from pathlib import Path

from dotenv import load_dotenv


# ==========================================================
# DOSSIERS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ASSETS_DIR = BASE_DIR / "assets"


# ==========================================================
# APPLICATION
# ==========================================================

APP_NAME = "Assistant Documentaire IA"


# ==========================================================
# AZURE OPENAI
# ==========================================================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    ""
).rstrip("/")

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    ""
)

CHAT_DEPLOYMENT = os.getenv(
    "CHAT_DEPLOYMENT",
    ""
)

EMBEDDING_DEPLOYMENT = os.getenv(
    "EMBEDDING_DEPLOYMENT",
    ""
)


# ==========================================================
# INTERFACE
# ==========================================================

LOGO_PATH = ASSETS_DIR / "logo.png"
STYLE_PATH = ASSETS_DIR / "style.css"


# ==========================================================
# LIVRES DISPONIBLES
# ==========================================================

BOOKS = {

    "dora": {

        "label": "DORA",

        "pdf": DATA_DIR / "dora.pdf",

        "index": VECTORSTORE_DIR / "dora" / "index.faiss",

        "chunks": VECTORSTORE_DIR / "dora" / "chunks.pkl",

    },

    "iso27001": {

        "label": "ISO 27001",

        "pdf": DATA_DIR / "iso27001.pdf",

        "index": VECTORSTORE_DIR / "iso27001" / "index.faiss",

        "chunks": VECTORSTORE_DIR / "iso27001" / "chunks.pkl",

    },

    "ebios": {

    "label": "EBIOS RM",

    "pdf": DATA_DIR / "ebios.pdf",

    "index": VECTORSTORE_DIR / "ebios" / "index.faiss",

    "chunks": VECTORSTORE_DIR / "ebios" / "chunks.pkl",

},

"dora_certification": {
    "label": "DORA Certification",

    "pdf": DATA_DIR / "dora_certification.pdf",

    "index": VECTORSTORE_DIR / "dora_certification" / "index.faiss",

    "chunks": VECTORSTORE_DIR / "dora_certification" / "chunks.pkl",
},

}


# ==========================================================
# FONCTIONS
# ==========================================================

def get_book_config(book_id: str) -> dict:
    """
    Retourne la configuration du livre demandé.
    """

    if book_id not in BOOKS:

        raise KeyError(
            f"Livre inconnu : {book_id}"
        )

    return BOOKS[book_id]


def validate_azure_settings():
    """
    Vérifie les variables Azure.
    """

    values = {

        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,

        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,

        "CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,

        "EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,

    }

    missing = [

        key

        for key, value in values.items()

        if not value

    ]

    if missing:

        raise ValueError(

            "Variables Azure manquantes : "

            + ", ".join(missing)

        )