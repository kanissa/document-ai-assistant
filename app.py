import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

INDEX_PATH = Path("dora_index.faiss")
CHUNKS_PATH = Path("dora_chunks.pkl")


def create_client() -> OpenAI:
    if not all(
        [
            AZURE_ENDPOINT,
            AZURE_API_KEY,
            CHAT_DEPLOYMENT,
            EMBEDDING_DEPLOYMENT,
        ]
    ):
        raise ValueError("Vérifie les informations présentes dans le fichier .env.")

    return OpenAI(
        api_key=AZURE_API_KEY,
        base_url=f"{AZURE_ENDPOINT}/openai/v1/",
    )


@st.cache_resource
def load_database():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "La base documentaire est absente. Exécute d'abord build_database.py."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def retrieve_passages(
    question: str,
    client: OpenAI,
    index,
    chunks,
    number_of_results: int = 5,
):
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=question,
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype="float32",
    )

    faiss.normalize_L2(query_vector)

    _, positions = index.search(query_vector, number_of_results)

    return [
        chunks[position]
        for position in positions[0]
        if 0 <= position < len(chunks)
    ]


def generate_answer(question: str, passages: list[dict], client: OpenAI) -> str:
    context = "\n\n".join(
        f"[Page {passage['page']}]\n{passage['text']}"
        for passage in passages
    )

    prompt = f"""
Tu es un assistant spécialisé dans le règlement DORA.

Réponds uniquement à partir des passages du livre fournis ci-dessous.

Règles :
- N'utilise aucune connaissance extérieure.
- Si la réponse n'est pas présente dans les passages, réponds :
  "Je ne trouve pas cette information dans le livre DORA."
- Réponds clairement et simplement en français.
- Mentionne les pages utilisées à la fin de la réponse.

Question :
{question}

Passages du livre :
{context}
"""

    response = client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=prompt,
    )

    return response.output_text


st.set_page_config(
    page_title="Assistant DORA",
    page_icon="📘",
)

st.title("📘 Assistant DORA")
st.write("Posez une question. La réponse sera basée uniquement sur le livre DORA.")

try:
    client = create_client()
    index, chunks = load_database()

    question = st.text_input(
        "Votre question",
        placeholder="Exemple : Quelles sont les exigences relatives aux tests de résilience ?",
    )

    if st.button("Obtenir la réponse", type="primary"):
        if not question.strip():
            st.warning("Veuillez saisir une question.")
        else:
            with st.spinner("Recherche dans le livre DORA..."):
                passages = retrieve_passages(
                    question,
                    client,
                    index,
                    chunks,
                )

                answer = generate_answer(
                    question,
                    passages,
                    client,
                )

            st.subheader("Réponse")
            st.write(answer)

except Exception as error:
    st.error(f"Erreur : {error}")