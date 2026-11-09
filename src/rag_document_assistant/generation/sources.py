"""Display original passages supplied to answer generation."""


def format_sources(
    sources: list[dict],
    *,
    cited_ids: set[str],
) -> str:
    """Render source metadata and original text without another model call."""
    blocks = []

    for source in sources:
        citation_id = source["citation_id"]
        pages = (
            str(source["page_start"])
            if source["page_start"] == source["page_end"]
            else f"{source['page_start']}-{source['page_end']}"
        )
        usage = "cited in answer" if citation_id in cited_ids else "not cited in answer"

        blocks.append(
            f"[{citation_id}] {source['title']}\n"
            f"Page: {pages} | Version: {source['version']} | {usage}\n"
            f"File: {source['source_path']}\n"
            f"{source['text']}"
        )

    return "\n\n".join(blocks)