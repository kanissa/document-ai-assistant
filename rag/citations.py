from collections import OrderedDict


def build_references(passages: list[dict]) -> str:
    """
    Construit automatiquement la liste des références
    à partir des passages récupérés par FAISS.
    """

    references = OrderedDict()

    for passage in passages:
        page = passage.get("page", "?")
        document = passage.get("document", "DORA.pdf")

        references[(document, page)] = True

    lines = []

    for index, (document, page) in enumerate(references.keys(), start=1):
        lines.append(f"[{index}] {document} — Page {page}")

    return "\n\n".join(lines)