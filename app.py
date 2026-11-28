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
from rag_document_assistant.retrieval.storage import (
    corpus_fingerprint,
    load_index,
)
from rag_document_assistant.ui.config import (
    AppConfig,
    load_app_config,
)
from rag_document_assistant.ui.theme import apply_ui_theme

ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def get_search_index(
    corpus_root: Path,
    index_path: Path,
    index_modified_ns: int,
    fingerprint: str,
):
    """Keep the encoder and validated index loaded between questions.

    index_modified_ns and fingerprint are cache keys used to
    invalidate the cached index when the index or corpus changes.
    """
    encoder = LocalSentenceEncoder(local_files_only=True)

    return load_index(
        index_path,
        root=corpus_root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )


def display_report(
    report: dict,
    config: AppConfig,
) -> None:
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

            pdf_path = (
                config.corpus_root / source["source_path"]
            ).resolve()

            if (
                pdf_path.is_relative_to(config.corpus_root)
                and pdf_path.is_file()
            ):
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
                    (report.get("generation_request") or {}).get("options")
                    if report.get("provider", "ollama") == "ollama"
                    else {
                        key: (report.get("generation_request") or {}).get(key)
                        for key in (
                            "temperature",
                            "reasoning",
                            "max_output_tokens",
                            "store",
                        )
                    }
                ),
                "citation_check": check,
                "provider": report.get("provider", "ollama"),
                "api_usage": (report.get("api_response") or {}).get("usage"),
            }
        )

    report_path = st.session_state.get("report_path")
    if report_path:
        st.caption(f"Saved report: {report_path}")


def reset_question() -> None:
    """Clear the current question and displayed answer."""
    st.session_state["question"] = ""
    st.session_state.pop("report", None)
    st.session_state.pop("report_path", None)


def main() -> None:
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📚",
        layout="wide",
    )

    apply_ui_theme()
    try:
        config = load_app_config(ROOT)
    except (OSError, ValueError) as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
        return

    previous_config = st.session_state.get("app_config")

    if previous_config != config:
        reset_question()
        st.session_state["app_config"] = config

    st.title("Document Assistant")

    subtitle = (
        "Ask questions about the fictional Nordlicht company documents."
        if config.is_demo
        else "Ask questions about your indexed documents."
    )

    st.markdown(
        f'<p class="app-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )

    if config.is_demo:
        st.info(
            "Portfolio demo. Generated answers may omit important conditions. "
            "Review the original passages before acting on an answer."
        )
    else:
        st.info(
            "Answers are generated from retrieved document passages. "
            "Review the original sources before acting on an answer."
        )

    with st.sidebar:
        st.header("Generation")

        provider_labels = {
            "ollama": "Local Ollama",
            "openai": "OpenAI API",
        }

        if len(config.allowed_providers) == 1:
            provider = config.allowed_providers[0]

            st.write(
                f"Answer provider: {provider_labels[provider]}"
            )
        else:
            provider = st.selectbox(
                "Answer provider",
                options=config.allowed_providers,
                format_func=lambda value: provider_labels[value],
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
            placeholder=(
                "Can I book a hotel in Germany for EUR 140 per night?"
                if config.is_demo
                else "Enter a question about your documents."
            ),
            max_chars=2000,
            key="question",
        )

        ask_column, reset_column, _ = st.columns([1, 2, 5])

        with ask_column:
            submitted = st.form_submit_button(
                "Ask",
                type="primary",
            )

        with reset_column:
            st.form_submit_button(
                "New question",
                type="secondary",
                on_click=reset_question,
            )

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
                    if provider not in config.allowed_providers:
                        raise ValueError(
                            "The selected provider is not permitted"
                        )

                    fingerprint = corpus_fingerprint(
                        config.corpus_root
                    )

                    index = get_search_index(
                        config.corpus_root,
                        config.index_path,
                        config.index_path.stat().st_mtime_ns,
                        fingerprint,
                    )
                    report = answer_question(
                        index,
                        question.strip(),
                        model=model,
                        provider=provider,
                    )

                st.session_state["report"] = report

                try:
                    path = save_report(
                        report,
                        config.report_directory,
                    )

                    st.session_state["report_path"] = str(
                        path.relative_to(config.corpus_root)
                    )
                except OSError:
                    st.warning(
                        "The answer is available, but its report "
                        "could not be saved."
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Request failed: {exc}")

    if "report" in st.session_state:
        display_report(
            st.session_state["report"],
            config,
        )


if __name__ == "__main__":
    main()