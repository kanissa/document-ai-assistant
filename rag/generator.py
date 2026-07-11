from rag.config import CHAT_DEPLOYMENT
from rag.prompts import build_prompt


def generate_answer(
    question: str,
    passages: list[dict],
    client,
) -> str:
    """
    Génère uniquement la réponse.
    Les références seront ajoutées par l'application.
    """

    if not passages:
        return "I cannot find this information in the provided document."

    prompt = build_prompt(
        question=question,
        passages=passages,
    )

    response = client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=prompt,
    )

    answer = response.output_text.strip()

    # Nettoyage : on retire une éventuelle section "References"
    if "## References" in answer:
        answer = answer.split("## References")[0].strip()

    elif "# References" in answer:
        answer = answer.split("# References")[0].strip()

    elif "References" in answer:
        answer = answer.split("References")[0].strip()

    return answer