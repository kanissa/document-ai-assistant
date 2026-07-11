import time

import streamlit as st

from rag.citations import build_references
from rag.config import (
    APP_NAME,
    BOOKS,
    LOGO_PATH,
    STYLE_PATH,
    get_book_config,
)
from rag.database import load_vectorstore
from rag.generator import generate_answer
from rag.openai_client import get_openai_client
from rag.retriever import retrieve_passages


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(
            f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.question_history = []


def stream_text(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.015)


def display_sources(passages: list[dict]) -> None:
    if not passages:
        return

    st.markdown("---")
    st.markdown("### 📚 Références")
    st.markdown(build_references(passages))

    with st.expander("Afficher les passages sources"):
        for position, passage in enumerate(passages, start=1):
            document = passage.get("document", "document.pdf")
            page = passage.get("page", "?")

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


def retrieve_from_selected_books(
    question: str,
    selected_book_ids: list[str],
    client,
    top_k: int,
) -> list[dict]:
    """
    Recherche dans tous les livres sélectionnés,
    fusionne les résultats et conserve les meilleurs passages.
    """

    combined_passages = []

    for book_id in selected_book_ids:
        index, chunks = load_vectorstore(book_id)

        book_passages = retrieve_passages(
            question=question,
            client=client,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )

        combined_passages.extend(book_passages)

    combined_passages.sort(
        key=lambda passage: passage.get("score", 0),
        reverse=True,
    )

    return combined_passages[:top_k]


load_css()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "selected_books" not in st.session_state:
    st.session_state.selected_books = ["dora"]


with st.sidebar:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=165,
        )

    st.markdown("## Assistant documentaire IA")

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

    st.markdown("### Livres à interroger")

    book_ids = list(BOOKS.keys())

    book_labels = {
        book_id: BOOKS[book_id]["label"]
        for book_id in book_ids
    }

    default_labels = [
        book_labels[book_id]
        for book_id in st.session_state.selected_books
        if book_id in book_labels
    ]

    selected_labels = st.multiselect(
        "Bases documentaires",
        options=list(book_labels.values()),
        default=default_labels,
        placeholder="Sélectionnez un ou plusieurs livres",
    )

    selected_book_ids = [
        book_id
        for book_id, label in book_labels.items()
        if label in selected_labels
    ]

    if selected_book_ids != st.session_state.selected_books:
        st.session_state.selected_books = selected_book_ids
        reset_conversation()
        st.rerun()

    st.divider()
    st.markdown("### Documents")

    if not selected_book_ids:
        st.warning(
            "Sélectionnez au moins un livre."
        )

    for book_id in selected_book_ids:
        book_config = get_book_config(book_id)

        st.markdown(
            f"**{book_config['label']}**"
        )

        if book_config["pdf"].exists():
            file_size_mb = (
                book_config["pdf"].stat().st_size
                / (1024 * 1024)
            )

            st.success(
                f"✅ {book_config['pdf'].name}"
            )

            st.caption(
                f"Taille : {file_size_mb:.1f} Mo"
            )
        else:
            st.error(
                f"PDF absent : {book_config['pdf'].name}"
            )

        if (
            book_config["index"].exists()
            and book_config["chunks"].exists()
        ):
            st.success(
                "✅ Base vectorielle prête"
            )
        else:
            st.warning(
                "Base vectorielle absente."
            )

            st.code(
                f"python build_database.py {book_id}"
            )

    st.divider()

    top_k = st.slider(
        "Nombre total de passages",
        min_value=3,
        max_value=10,
        value=5,
        help=(
            "Nombre maximal de passages conservés "
            "après la recherche dans tous les livres sélectionnés."
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


active_labels = [
    BOOKS[book_id]["label"]
    for book_id in st.session_state.selected_books
]

active_books_text = (
    " + ".join(active_labels)
    if active_labels
    else "Aucun livre"
)


left_column, right_column = st.columns(
    [4, 1],
    vertical_alignment="center",
)

with left_column:

    st.title("📘 Assistant documentaire IA")

    st.write(
        "Posez une question sur les livres sélectionnés. "
        "L’assistant recherchera uniquement dans ces documents."
    )

with right_column:

    st.metric(
        label="Bases actives",
        value=len(active_labels),
        delta=active_books_text,
    )

st.divider()


try:

    if not st.session_state.selected_books:
        st.warning(
            "Sélectionnez au moins un livre dans la barre latérale."
        )
        st.stop()

    client = get_openai_client()

    if not st.session_state.messages:

        with st.chat_message(
            "assistant",
            avatar="📘",
        ):

            st.markdown(
                f"""
👋 **Bonjour !**

Livres sélectionnés : **{active_books_text}**

Posez votre première question.
                """
            )

    for message in st.session_state.messages:

        avatar = (
            "👤"
            if message["role"] == "user"
            else "📘"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
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

    question = st.chat_input(
        f"Posez votre question sur {active_books_text}…"
    )

    if question and question.strip():

        clean_question = question.strip()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )

        st.session_state.question_history.append(
            clean_question
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):
            st.markdown(
                clean_question
            )

        with st.chat_message(
            "assistant",
            avatar="📘",
        ):

            with st.spinner(
                f"Recherche dans {active_books_text}…"
            ):

                passages = retrieve_from_selected_books(
                    question=clean_question,
                    selected_book_ids=(
                        st.session_state.selected_books
                    ),
                    client=client,
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

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": passages,
            }
        )

        st.rerun()


except Exception as error:

    st.error(
        f"Erreur : {error}"
    )

    st.info(
        "Vérifiez les PDF sélectionnés, leurs bases vectorielles "
        "et les paramètres Azure."
    )