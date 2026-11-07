import json

import pytest

from rag_document_assistant.generation.extract import validate_extraction

SOURCE = (
    "Two price quotations and the trade fair dates must be recorded "
    "in the travel request before booking."
)


def content(quote=SOURCE, source_id="NL-11:S2"):
    return json.dumps(
        {"quotes": [{"source_id": source_id, "quote": quote}]}
    )


def test_exact_quote_is_accepted():
    extraction = validate_extraction(content(), {"NL-11:S2": SOURCE})

    assert extraction.quotes[0].quote == SOURCE


def test_whitespace_changes_are_accepted():
    extraction = validate_extraction(
        content(quote=SOURCE.replace(" and ", "\n and  ")),
        {"NL-11:S2": SOURCE},
    )

    assert len(extraction.quotes) == 1


def test_invented_quote_is_rejected():
    with pytest.raises(ValueError, match="not present verbatim"):
        validate_extraction(
            content(quote="No price quotations are required before booking."),
            {"NL-11:S2": SOURCE},
        )


def test_quote_from_wrong_source_is_rejected():
    with pytest.raises(ValueError, match="not present verbatim"):
        validate_extraction(
            content(),
            {"NL-11:S2": "Finance reviews complete claims within ten working days."},
        )


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown source"):
        validate_extraction(
            content(source_id="NL-99:S1"),
            {"NL-11:S2": SOURCE},
        )


def test_duplicate_quote_is_rejected():
    item = {"source_id": "NL-11:S2", "quote": SOURCE}

    with pytest.raises(ValueError, match="Duplicate"):
        validate_extraction(
            json.dumps({"quotes": [item, item]}),
            {"NL-11:S2": SOURCE},
        )


def test_empty_extraction_is_allowed():
    extraction = validate_extraction(
        '{"quotes": []}',
        {"NL-11:S2": SOURCE},
    )

    assert extraction.quotes == []