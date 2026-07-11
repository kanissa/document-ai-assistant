import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from rag.config import (
    DATA_DIR,
    EMBEDDING_DEPLOYMENT,
    VECTORSTORE_DIR,
)
from rag.openai_client import get_openai_client


def normalize_book_id(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "")


def get_paths(book_id: str) -> tuple[Path, Path, Path]:
    pdf_path = DATA_DIR / f"{book_id}.pdf"
    book_directory = VECTORSTORE_DIR / book_id
    index_path = book_directory / "index.faiss"
    chunks_path = book_directory / "chunks.pkl"

    return pdf_path, index_path, chunks_path


def split_text(
    text: str,
    chunk_size: int = 1600,
    overlap: int = 250,
) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def extract_chunks(
    pdf_path: Path,
    book_id: str,
) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}\n"
            f"Ajoute le fichier data/{book_id}.pdf."
        )

    if pdf_path.stat().st_size == 0:
        raise ValueError(
            f"Le fichier {pdf_path.name} est vide."
        )

    reader = PdfReader(str(pdf_path))
    chunks = []

    print(
        f"Lecture de {pdf_path.name} "
        f"({len(reader.pages)} pages)..."
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = (page.extract_text() or "").strip()

        if not text:
            print(
                f"Page {page_number} : "
                "aucun texte détecté."
            )
            continue

        for passage in split_text(text):
            chunks.append(
                {
                    "book_id": book_id,
                    "document": pdf_path.name,
                    "page": page_number,
                    "text": passage,
                }
            )

    if not chunks:
        raise ValueError(
            "Aucun texte n'a été extrait. "
            "Le PDF est peut-être composé d'images."
        )

    return chunks


def create_embeddings(
    chunks: list[dict],
    batch_size: int = 32,
) -> np.ndarray:
    client = get_openai_client()
    vectors = []

    print("Création des embeddings Azure...")

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start : start + batch_size
        ]

        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=[
                item["text"]
                for item in batch
            ],
        )

        vectors.extend(
            item.embedding
            for item in response.data
        )

        completed = min(
            start + batch_size,
            len(chunks),
        )

        print(
            f"{completed}/{len(chunks)} "
            "passages traités."
        )

    matrix = np.asarray(
        vectors,
        dtype="float32",
    )

    if matrix.size == 0:
        raise ValueError(
            "Aucun embedding n'a été créé."
        )

    return matrix


def save_vectorstore(
    matrix: np.ndarray,
    chunks: list[dict],
    index_path: Path,
    chunks_path: Path,
) -> None:
    index_path.parent.mkdir(
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
        str(index_path),
    )

    with chunks_path.open("wb") as file:
        pickle.dump(
            chunks,
            file,
        )

    if index_path.stat().st_size == 0:
        raise ValueError(
            "L'index FAISS créé est vide."
        )

    if chunks_path.stat().st_size == 0:
        raise ValueError(
            "Le fichier chunks.pkl créé est vide."
        )


def build_book(book_id: str) -> None:
    pdf_path, index_path, chunks_path = get_paths(
        book_id
    )

    print()
    print(
        f"Construction de la base : {book_id}"
    )
    print(
        f"PDF : {pdf_path}"
    )

    chunks = extract_chunks(
        pdf_path=pdf_path,
        book_id=book_id,
    )

    print(
        f"{len(chunks)} passages créés."
    )

    matrix = create_embeddings(
        chunks
    )

    save_vectorstore(
        matrix=matrix,
        chunks=chunks,
        index_path=index_path,
        chunks_path=chunks_path,
    )

    print()
    print("Base créée avec succès.")
    print(f"Index : {index_path}")
    print(f"Passages : {chunks_path}")
    print(
        f"Dimension : {matrix.shape[1]}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit l'index FAISS "
            "d'un livre PDF."
        )
    )

    parser.add_argument(
        "book",
        help=(
            "Nom du livre sans extension. "
            "Exemples : dora ou iso27001"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    book_id = normalize_book_id(
        args.book
    )

    build_book(
        book_id
    )


if __name__ == "__main__":
    main()