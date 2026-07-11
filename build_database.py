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

PDF_PATH = Path("dora.pdf")
INDEX_PATH = Path("dora_index.faiss")
CHUNKS_PATH = Path("dora_chunks.pkl")


def split_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    """Découpe le texte en passages avec un petit chevauchement."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main() -> None:
    if not AZURE_ENDPOINT or not AZURE_API_KEY or not EMBEDDING_DEPLOYMENT:
        raise ValueError(
            "Vérifie AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY "
            "et EMBEDDING_DEPLOYMENT dans le fichier .env."
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError("Le fichier dora.pdf est introuvable.")

    client = OpenAI(
        api_key=AZURE_API_KEY,
        base_url=f"{AZURE_ENDPOINT}/openai/v1/",
    )

    print("Lecture du PDF...")

    reader = PdfReader(PDF_PATH)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for passage in split_text(text):
            chunks.append(
                {
                    "page": page_number,
                    "text": passage,
                }
            )

    if not chunks:
        raise ValueError(
            "Aucun texte n'a été extrait. Le PDF est peut-être composé d'images."
        )

    print(f"{len(chunks)} passages créés.")
    print("Création des embeddings Azure...")

    vectors = []

    for position in range(0, len(chunks), 50):
        batch = chunks[position : position + 50]

        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=[item["text"] for item in batch],
        )

        vectors.extend(item.embedding for item in response.data)

        completed = min(position + 50, len(chunks))
        print(f"{completed}/{len(chunks)} passages traités.")

    matrix = np.array(vectors, dtype="float32")

    # Normalisation pour utiliser une similarité cosinus.
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print("\nBase créée avec succès.")
    print(f"Index : {INDEX_PATH}")
    print(f"Passages : {CHUNKS_PATH}")


if __name__ == "__main__":
    main()