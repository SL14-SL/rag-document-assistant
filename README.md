# Document Assistant with Source Citations and Evaluation

[![CI](https://github.com/SL14-SL/rag-document-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/SL14-SL/rag-document-assistant/actions/workflows/ci.yml)

An English-language RAG portfolio application for querying fictional
company documents. It combines local PDF retrieval with either local
Ollama generation or an explicitly selected OpenAI API backend.

Answers are displayed alongside the original passages supplied to the
model, including document titles, versions, page references and PDF
downloads.

## Features

- PDF ingestion with document, section and page metadata.
- BM25 baseline and semantic retrieval.
- A persisted semantic index and an in-memory search model.
- Local generation through Ollama.
- Optional OpenAI generation through the Responses API.
- Streamlit interface with explicit provider selection.
- Original source passages and downloadable source PDFs.
- Saved requests, answers, timings and provider metadata.
- Retrieval evaluation and manual answer-quality review.
- Unit tests and a GitHub Actions lint/test workflow.

Each question is independent. Previous questions and answers are not
included in subsequent requests.

## Architecture

1. Extract sections from the PDFs.
2. Build a semantic index of active document sections.
3. Retrieve the five most relevant sections for a question.
4. Send the question and source passages to the selected language model.
5. Check whether detected source IDs belong to the supplied passages.
6. Display the answer and all supplied source passages.

The search model and index remain loaded between questions in Streamlit.
Generation providers share the same prompt construction and retrieval
pipeline. Provider failures do not trigger an automatic fallback.

## Requirements

- Python 3.12.
- uv.
- Ollama with `qwen2.5:14b` for local generation.
- An OpenAI API key and available API credit for API generation.

The first installation downloads Python packages. Semantic retrieval
also requires an initial download of
`sentence-transformers/all-MiniLM-L6-v2`.

Local generation was tested on an Intel i7-8665U laptop with approximately
32 GB RAM and CPU-only inference. It is functional but slow.

## Installation

From the repository root:

```bash
uv sync --locked --dev
uv run python scripts/validate_dataset.py
```

## Prepare the local index

Extract PDF sections:

```bash
uv run python -m rag_document_assistant.ingestion
```

Build the semantic index:

```bash
uv run python -m rag_document_assistant.retrieval.build_index
```

The initial build may download the embedding model. After it is cached,
the index can be rebuilt offline:

```bash
HF_HUB_OFFLINE=1 uv run python \
    -m rag_document_assistant.retrieval.build_index \
    --offline
```

The indexed corpus contains 48 sections from 24 active documents.
The archived policy is excluded from the active search index.

## Run the application

```bash
HF_HUB_OFFLINE=1 uv run streamlit run app.py \
    --server.address 127.0.0.1
```

Open http://localhost:8501.

### Local Ollama

Ensure Ollama is running and install the tested model:

```bash
ollama pull qwen2.5:14b
```

Select **Local Ollama** in the sidebar.

After the initial downloads, retrieval and generation can run locally
without an external model API.

### OpenAI API

Set `OPENAI_API_KEY` in the same terminal before starting Streamlit.

For zsh, execute this line, paste the key at the hidden prompt and press
Enter:

```bash
read -rs "OPENAI_API_KEY?OpenAI API key: "
```

Then:

```bash
export OPENAI_API_KEY
printf '\n'
```

Start Streamlit and select **OpenAI API** in the sidebar.

The configured API model is `gpt-5.4-2026-03-05`.

Questions and the five retrieved passages are sent to OpenAI. PDFs are
not uploaded by this implementation. API requests incur usage charges.

`HF_HUB_OFFLINE=1` applies to the local embedding model; it does not
disable selected OpenAI requests.

Never commit API keys or Streamlit secrets.

## Example questions

- How many days per week may I work remotely?
- Can I book a hotel in Germany for EUR 140 per night?
- Does P1 mean the issue will be resolved within 30 minutes?
- What happens if an import file contains duplicate external_id values?
- How long are internal knowledge platform backups retained?
- How much annual leave am I entitled to?

The corpus deliberately includes exceptions, conflicting policies and
information gaps.

## Evaluation

The dataset contains 30 questions: 24 development questions and six
validation questions.

### Retrieval baseline

On the 22 development questions eligible for the retrieval metric,
semantic retrieval achieved:

- Evidence recall@5: 1.000.
- Complete evidence@5: 1.000.
- MRR@5: 0.955.

These scores describe retrieval of annotated sections, not answer quality.
The historical-policy case and the unanswerable case were excluded from
that retrieval metric.

### Local versus API generation

A development comparison used identical prompt messages and source
passages for all 24 questions.

| Measure | Local Qwen 2.5 14B | OpenAI GPT-5.4 |
| --- | --- | --- |
| Content pass, including editorial notes | 22 | 22 |
| Needs revision | 1 | 1 |
| Insufficient historical source coverage | 1 | 1 |
| Median generation request time | 146.26 seconds | 1.80 seconds |

These are manual, assistant-assisted content assessments. They do not
certify perfect citation formatting or production readiness.

The comparison demonstrates a latency advantage for the API on the
tested CPU-only laptop. It does not demonstrate a higher content pass
rate on this small development set.

See
[the comparison report](reports/answer-evaluation/local-vs-openai-v12.md).

### Validation

The local grounded-v1.2 candidate met all predefined content criteria
on four of six validation questions. Two answers were correct but
incomplete under the existing rubric.

See
[the validation review](reports/answer-evaluation/qwen25-14b-v12-validation.md).

The validation questions have now been inspected. Future changes informed
by these results require fresh held-out questions for an independent
final evaluation.

## Known limitations

- Generated answers can omit necessary conditions or process steps.
- Valid source IDs do not establish factual support or completeness.
- Showing original passages supports review but does not repair an answer.
- Historical questions may require archived sources absent from the
  active index.
- The dataset and questions were jointly AI-assisted and are small.
- PDFs are short, text-based documents, normally one page each.
- Scanned documents, OCR, complex tables and diverse real-world layouts
  have not been evaluated.
- The application has no user authentication or document-level access
  control and is intended for a local portfolio demonstration.
- Evaluation results do not establish reliability on real company data.

## Development checks

```bash
uv run ruff check app.py src tests scripts
uv run pytest -q
```

Tests use substitutes for model and API calls. They do not require a
running Ollama service, downloaded embedding weights or an API key.

The CI workflow installs locked dependencies and runs lint and tests.
It does not execute the answer-quality evaluation or paid API requests.

## Repository structure

- `app.py`: Streamlit interface.
- `src/rag_document_assistant/ingestion/`: PDF loading.
- `src/rag_document_assistant/retrieval/`: search and index persistence.
- `src/rag_document_assistant/generation/`: prompts and generation backends.
- `src/rag_document_assistant/evaluation/`: evaluation tools.
- `documents/`: editable Markdown authoring sources.
- `documents_pdf/`: PDF retrieval corpus.
- `evaluation/`: questions, evidence references and content criteria.
- `reports/`: versioned evaluation findings.
- `scripts/`: dataset and comparison utilities.
- `tests/`: automated tests.
- `artifacts/`: generated indexes and answer reports; ignored by Git.

Markdown and PDF files represent the same documents. Do not index both
copies.

## Dataset

Nordlicht Software and all policies in this repository are fictional.
The corpus contains 25 English-language documents, including one archived
policy.

The fictional document snapshot is 6 November 2026. Active policies take
effect on 1 October 2026.

`manifest.json` records document metadata and file hashes. Intentional
document changes require updated hashes and a review of evidence
references.

The materials are for demonstration and are not legal, tax or
operational advice.