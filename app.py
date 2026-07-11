import time

import streamlit as st

from rag.citations import build_references
from rag.config import (
    APP_NAME,
    CHUNKS_PATH,
    INDEX_PATH,
    LOGO_PATH,
    PDF_PATH,
    STYLE_PATH,
)
from rag.database import load_vectorstore
from rag.generator import generate_answer
from rag.openai_client import get_openai_client
from rag.retriever import retrieve_passages


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_css() -> None:
    """Charge le fichier CSS personnalisé."""

    if STYLE_PATH.exists():
        st.markdown(
            f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def reset_conversation() -> None:
    """Réinitialise la conversation et l'historique."""

    st.session_state.messages = []
    st.session_state.question_history = []


def stream_text(text: str):
    """Affiche progressivement une réponse déjà générée."""

    for word in text.split():
        yield word + " "
        time.sleep(0.015)


def display_sources(passages: list[dict]) -> None:
    """Affiche les références et les passages réellement récupérés."""

    if not passages:
        return

    references = build_references(passages)

    st.markdown("---")
    st.markdown("### 📚 Références")

    if references:
        st.markdown(references)

    with st.expander("Afficher les passages sources"):
        for position, passage in enumerate(passages, start=1):
            document = passage.get("document", "dora.pdf")
            page = passage.get("page", "?")
            score = passage.get("score")

            st.markdown(
                f"#### Source {position} — {document}, page {page}"
            )

            st.write(
                passage.get(
                    "text",
                    "Aucun texte disponible.",
                )
            )

                

            st.divider()


# ============================================================
# INITIALISATION
# ============================================================

load_css()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=165,
        )

    st.markdown("## Assistant DORA")

    st.caption(
        "Azure OpenAI · RAG · FAISS · Streamlit"
    )

    if st.button(
        "➕ Nouvelle conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    st.markdown("### Document")

    if PDF_PATH.exists():
        file_size_mb = PDF_PATH.stat().st_size / (
            1024 * 1024
        )

        st.success(
            f"✅ {PDF_PATH.name}"
        )

        st.caption(
            f"Taille : {file_size_mb:.1f} Mo"
        )
    else:
        st.error(
            "Le fichier data/dora.pdf est absent."
        )

    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        st.success(
            "✅ Base vectorielle prête"
        )
    else:
        st.warning(
            "Exécutez python build_database.py"
        )

    st.divider()

    top_k = st.slider(
        "Nombre de passages recherchés",
        min_value=3,
        max_value=10,
        value=5,
        help=(
            "Détermine combien de passages du livre "
            "seront envoyés au modèle."
        ),
    )

    st.divider()

    st.markdown("### Historique")

    if st.session_state.question_history:
        for previous_question in reversed(
            st.session_state.question_history[-10:]
        ):
            st.caption(
                f"💬 {previous_question}"
            )
    else:
        st.caption(
            "Aucune question pour le moment."
        )

    st.divider()

    st.caption(
        "Réponses fondées uniquement sur le document chargé."
    )


# ============================================================
# EN-TÊTE PRINCIPAL
# ============================================================

left_column, right_column = st.columns(
    [4, 1],
    vertical_alignment="center",
)

with left_column:

    st.title("📘 Assistant DORA")

    st.write(
        "Posez une question sur le règlement DORA. "
        "L’assistant recherche d’abord dans votre livre, "
        "puis formule une réponse sourcée."
    )

with right_column:

    st.metric(
        label="Document",
        value="DORA",
        delta="Actif",
    )

st.divider()


# ============================================================
# APPLICATION PRINCIPALE
# ============================================================

try:

    client = get_openai_client()

    index, chunks = load_vectorstore()

    # Message d'accueil
    if not st.session_state.messages:

        with st.chat_message("assistant"):

            st.markdown(
                """
👋 **Bonjour !**

Posez votre première question sur le règlement DORA.
                """
            )

    # Affichage des anciens messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                display_sources(
                    message["sources"]
                )

    # Champ de saisie
    question = st.chat_input(
        "Posez votre question sur DORA…"
    )

    if question and question.strip():

        clean_question = question.strip()

        # Enregistrer la question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )

        st.session_state.question_history.append(
            clean_question
        )

        # Afficher la question
        with st.chat_message("user"):

            st.markdown(
                clean_question
            )

        # Générer et afficher la réponse
        with st.chat_message("assistant"):

            with st.spinner(
                "Recherche dans le livre DORA…"
            ):

                passages = retrieve_passages(
                    question=clean_question,
                    client=client,
                    index=index,
                    chunks=chunks,
                    top_k=top_k,
                )

                answer = generate_answer(
                    question=clean_question,
                    passages=passages,
                    client=client,
                )

            st.write_stream(
                stream_text(answer)
            )

            display_sources(
                passages
            )

        # Enregistrer la réponse complète
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": passages,
            }
        )

        # Mise à jour immédiate de la sidebar
        st.rerun()


# ============================================================
# GESTION DES ERREURS
# ============================================================

except Exception as error:

    st.error(
        f"Erreur : {error}"
    )

    st.info(
        "Vérifiez le fichier .env, les déploiements Azure, "
        "le fichier data/dora.pdf et les fichiers du dossier vectorstore."
    )