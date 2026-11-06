import numpy as np

from rag_document_assistant.evaluation.retrieval import evaluate_retrieval
from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.semantic import SemanticIndex


class FixedEncoder:
    def encode(self, texts):
        vectors = [
            [0.0, 1.0] if "Password" in text else [1.0, 0.0]
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


def make_section(document_id, title):
    return DocumentSection(
        document_id=document_id,
        title=title,
        version="1.0",
        status="active",
        source_path=f"documents_pdf/{document_id}.pdf",
        section_id="S1",
        heading="Rules",
        page_start=1,
        page_end=1,
        text=title,
    )


def test_semantic_index_works_with_retrieval_evaluation():
    index = SemanticIndex(
        [
            make_section("NL-01", "Hotel policy"),
            make_section("NL-02", "Password policy"),
        ],
        encoder=FixedEncoder(),
    )
    questions = [
        {
            "id": "Q001",
            "question": "Accommodation rules",
            "answerability": "answerable",
            "evidence": [{"document_id": "NL-01", "section": "S1"}],
        }
    ]

    report = evaluate_retrieval(index, questions, ks=(1, 3))

    assert report["evaluated_count"] == 1
    assert report["skipped_count"] == 0
    assert report["questions"][0]["retrieved"][0] == "NL-01:S1"
    assert report["metrics"]["1"] == {
        "recall": 1.0,
        "complete_evidence": 1.0,
        "reciprocal_rank": 1.0,
    }