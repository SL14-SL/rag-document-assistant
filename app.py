"""Streamlit interface for the local document assistant."""

from pathlib import Path

import streamlit as st

from rag_document_assistant.generation.answer import DEFAULT_GENERATION_MODEL
from rag_document_assistant.generation.chat import answer_question, save_report
from rag_document_assistant.generation.openai_backend import OPENAI_MODEL
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "artifacts" / "index" / "semantic.npz"
REPORT_DIRECTORY = ROOT / "artifacts" / "answers" / "streamlit"


@st.cache_resource(show_spinner=False)
def get_search_index(index_modified_ns: int):
    """Keep the encoder and index loaded between questions."""
    encoder = LocalSentenceEncoder(local_files_only=True)
    return load_index(
        INDEX_PATH,
        root=ROOT,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )


def display_report(report: dict) -> None:
    st.subheader("Generated answer")
    st.caption(f"Question: {report['question']}")
    st.caption(
        f"Provider: {report.get('provider', 'ollama')} | "
        f"Model: {report['model']}"
    )
    st.write(report["answer"])

    check = report["citation_check"]

    if check["unknown_ids"]:
        st.error(
            "The answer mentions source IDs that were not supplied: "
            + ", ".join(check["unknown_ids"])
        )
    elif not check["has_citations"]:
        st.warning("The answer contains no detected source IDs.")

    if report["done_reason"] == "length":
        st.warning("The answer reached the output limit and may be incomplete.")

    st.caption(
        "Source IDs identify passages. They do not prove that every claim "
        "is supported or that the answer includes every relevant condition."
    )

    st.subheader("Original source passages")
    st.write(
        "These are all passages supplied to the model, "
        "in retrieval order."
    )

    cited_ids = set(check["cited_ids"])

    for source in report["sources"]:
        citation_id = source["citation_id"]
        citation_status = (
            "cited in answer"
            if citation_id in cited_ids
            else "not cited in answer"
        )
        page_start = source["page_start"]
        page_end = source["page_end"]
        pages = (
            str(page_start)
            if page_start == page_end
            else f"{page_start}–{page_end}"
        )

        label = (
            f"[{citation_id}] {source['title']} · "
            f"page {pages} · {citation_status}"
        )

        with st.expander(label, expanded=citation_id in cited_ids):
            st.caption(
                f"Version: {source['version']} | "
                f"Status: {source['status']} | "
                f"Section: {source['heading']}"
            )
            st.write(source["text"])
            st.caption(f"Source file: {source['source_path']}")

            pdf_path = (ROOT / source["source_path"]).resolve()
            if pdf_path.is_relative_to(ROOT) and pdf_path.is_file():
                st.download_button(
                    "Download source PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"pdf-{citation_id}",
                )

    timing = report["timing"]
    st.caption(
        f"Search: {timing['search_seconds']:.3f} seconds | "
        f"Generation: {timing['generation_request_seconds']:.2f} seconds"
    )

    with st.expander("Run details"):
        st.json(
            {
                "model": report["model"],
                "prompt_version": report["prompt_version"],
                "runtime_metadata": report.get("runtime_metadata"),
                "generation_options": (
                    report.get("generation_request") or {}
                ).get("options"),
                "citation_check": check,
                "provider": report.get("provider", "ollama"),
                "api_usage": (report.get("api_response") or {}).get("usage"),
            }
        )

    report_path = st.session_state.get("report_path")
    if report_path:
        st.caption(f"Saved report: {report_path}")


def main() -> None:
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("Document Assistant")
    st.write("Ask questions about the fictional Nordlicht company documents.")
    st.info(
        "Portfolio demo. Generated answers may omit important conditions. "
        "Review the original passages before acting on an answer."
    )

    with st.sidebar:
        st.header("Generation")

        provider_label = st.selectbox(
            "Answer provider",
            options=("Local Ollama", "OpenAI API"),
        )
        provider = (
            "ollama" if provider_label == "Local Ollama" else "openai"
        )
        model = (
            DEFAULT_GENERATION_MODEL
            if provider == "ollama"
            else OPENAI_MODEL
        )

        st.write(f"Model: {model}")
        st.write("Search: local semantic retrieval, top five passages")

        if provider == "openai":
            st.warning(
                "Questions and retrieved source passages are sent to "
                "OpenAI. Each submitted question incurs API charges."
            )
        else:
            st.caption("Answer generation runs through local Ollama.")

        st.caption(
            "Each question is independent. Previous answers are not "
            "included in subsequent requests."
        )

    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            placeholder="Can I book a hotel in Germany for EUR 140 per night?",
            max_chars=2000,
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            st.session_state.pop("report", None)
            st.session_state.pop("report_path", None)

            try:
                with st.spinner(
                    "Searching documents and generating an answer"
                    + (
                        " locally. This may take several minutes."
                        if provider == "ollama"
                        else " via OpenAI."
                    )
                ):
                    index = get_search_index(
                        INDEX_PATH.stat().st_mtime_ns
                    )
                    report = answer_question(
                        index,
                        question.strip(),
                        model=model,
                        provider=provider,
                    )

                st.session_state["report"] = report

                try:
                    path = save_report(report, REPORT_DIRECTORY)
                    st.session_state["report_path"] = str(
                        path.relative_to(ROOT)
                    )
                except OSError:
                    st.warning(
                        "The answer is available, but its report "
                        "could not be saved."
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Request failed: {exc}")

    if "report" in st.session_state:
        display_report(st.session_state["report"])


if __name__ == "__main__":
    main()