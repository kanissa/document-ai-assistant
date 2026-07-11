import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "dora.pdf"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "dora_index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "dora_chunks.pkl"


def split_text(
    text: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[str]:
    """
    Découpe le texte en passages avec un chevauchement.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_client() -> OpenAI:
    """
    Crée le client Azure OpenAI.
    """

    if not AZURE_ENDPOINT:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT est absent du fichier .env."
        )

    if not AZURE_API_KEY:
        raise ValueError(
            "AZURE_OPENAI_API_KEY est absent du fichier .env."
        )

    if not EMBEDDING_DEPLOYMENT:
        raise ValueError(
            "EMBEDDING_DEPLOYMENT est absent du fichier .env."
        )

    return OpenAI(
        api_key=AZURE_API_KEY,
        base_url=f"{AZURE_ENDPOINT}/openai/v1/",
    )


def extract_chunks() -> list[dict]:
    """
    Lit le PDF et crée les passages avec leurs numéros de page.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Le fichier PDF est introuvable : {PDF_PATH}"
        )

    print(f"Lecture du PDF : {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            print(
                f"Avertissement : aucun texte extrait de la page {page_number}."
            )
            continue

        page_chunks = split_text(text)

        for passage in page_chunks:
            chunks.append(
                {
                    "page": page_number,
                    "text": passage,
                }
            )

    if not chunks:
        raise ValueError(
            "Aucun texte n'a été extrait du PDF. "
            "Le document est peut-être composé uniquement d'images."
        )

    print(f"{len(chunks)} passages créés.")

    return chunks


def create_embeddings(
    chunks: list[dict],
    client: OpenAI,
    batch_size: int = 50,
) -> np.ndarray:
    """
    Crée les embeddings Azure pour tous les passages.
    """

    print("Création des embeddings Azure...")

    vectors = []

    for position in range(0, len(chunks), batch_size):
        batch = chunks[position : position + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=[item["text"] for item in batch],
        )

        vectors.extend(
            item.embedding
            for item in response.data
        )

        completed = min(
            position + batch_size,
            len(chunks),
        )

        print(
            f"{completed}/{len(chunks)} passages traités."
        )

    if not vectors:
        raise ValueError(
            "Aucun embedding n'a été créé."
        )

    matrix = np.array(
        vectors,
        dtype="float32",
    )

    return matrix


def save_vectorstore(
    matrix: np.ndarray,
    chunks: list[dict],
) -> None:
    """
    Crée et enregistre l'index FAISS et les métadonnées.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(
        matrix.shape[1]
    )

    index.add(matrix)

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(
            chunks,
            file,
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "L'index FAISS n'a pas été créé."
        )

    if INDEX_PATH.stat().st_size == 0:
        raise ValueError(
            "L'index FAISS créé est vide."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Le fichier des passages n'a pas été créé."
        )

    if CHUNKS_PATH.stat().st_size == 0:
        raise ValueError(
            "Le fichier des passages créé est vide."
        )


def main() -> None:
    """
    Construit la base vectorielle à partir du livre DORA.
    """

    print("Démarrage de la construction de la base DORA...")

    client = create_client()

    chunks = extract_chunks()

    matrix = create_embeddings(
        chunks,
        client,
    )

    save_vectorstore(
        matrix,
        chunks,
    )

    print()
    print("Base créée avec succès.")
    print(f"PDF : {PDF_PATH}")
    print(f"Index : {INDEX_PATH}")
    print(f"Passages : {CHUNKS_PATH}")
    print(f"Nombre de passages : {len(chunks)}")
    print(f"Dimension des embeddings : {matrix.shape[1]}")


if __name__ == "__main__":
    main()