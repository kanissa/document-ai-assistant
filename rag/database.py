import pickle
from functools import lru_cache

import faiss

from rag.config import CHUNKS_PATH, INDEX_PATH


@lru_cache(maxsize=1)
def load_vectorstore():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS absent : {INDEX_PATH}. "
            "Exécutez d’abord python build_database.py."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Métadonnées absentes : {CHUNKS_PATH}. "
            "Exécutez d’abord python build_database.py."
        )

    if INDEX_PATH.stat().st_size == 0 or CHUNKS_PATH.stat().st_size == 0:
        raise ValueError(
            "La base vectorielle est vide. "
            "Exécutez de nouveau build_database.py."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with CHUNKS_PATH.open("rb") as file:
        chunks = pickle.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            "L’index FAISS et les passages ne correspondent pas. "
            "Reconstruisez la base."
        )

    return index, chunks