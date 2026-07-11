import pickle

import faiss

from rag.config import get_book_config


# Cache simple pour éviter de recharger plusieurs fois
# la même base vectorielle pendant l'exécution.
_DATABASE_CACHE: dict[str, tuple] = {}


def load_vectorstore(book_id: str):
    """
    Charge l'index FAISS et les passages du livre sélectionné.
    """

    if book_id in _DATABASE_CACHE:
        return _DATABASE_CACHE[book_id]

    book_config = get_book_config(book_id)

    index_path = book_config["index"]
    chunks_path = book_config["chunks"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable pour {book_config['label']}.\n"
            f"Exécutez : python build_database.py {book_id}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Fichier chunks.pkl introuvable pour "
            f"{book_config['label']}.\n"
            f"Exécutez : python build_database.py {book_id}"
        )

    if index_path.stat().st_size == 0:
        raise ValueError(
            f"L'index FAISS de {book_config['label']} est vide."
        )

    if chunks_path.stat().st_size == 0:
        raise ValueError(
            f"Le fichier chunks.pkl de "
            f"{book_config['label']} est vide."
        )

    index = faiss.read_index(
        str(index_path)
    )

    with chunks_path.open("rb") as file:
        chunks = pickle.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            f"L'index FAISS et les passages de "
            f"{book_config['label']} ne correspondent pas.\n"
            f"Reconstruisez la base avec : "
            f"python build_database.py {book_id}"
        )

    _DATABASE_CACHE[book_id] = (
        index,
        chunks,
    )

    return index, chunks


def clear_database_cache() -> None:
    """
    Vide le cache des bases vectorielles.
    """

    _DATABASE_CACHE.clear()