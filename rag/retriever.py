import faiss
import numpy as np

from rag.config import EMBEDDING_DEPLOYMENT


def retrieve_passages(
    question: str,
    client,
    index,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Recherche les passages les plus pertinents dans l'index FAISS.
    """

    # Création de l'embedding de la question
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=question,
    )

    query_vector = np.asarray(
        [response.data[0].embedding],
        dtype="float32",
    )

    # Normalisation pour la similarité cosinus
    faiss.normalize_L2(query_vector)

    # Recherche des passages
    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx].copy()

        chunk["score"] = round(float(score), 4)

        results.append(chunk)

    return results