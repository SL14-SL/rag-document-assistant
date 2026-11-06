# Nordlicht — Local Document Assistant Dataset v0.2 (English)

25 synthetic English-language company documents and 30 questions with reference evidence. Created for a portfolio project. All policies are fictional and are not legal, tax or operational advice. You may modify and publish these materials as part of your project; no claim of exclusive rights in AI-generated output is made.

## Getting started (WSL)

Extract the archive into a new project directory. Read documents/09.md and documents/10.md to compare the current and archived travel policies. Compare documents/23.md and documents/24.md for an intentionally unresolved conflict.

Run `python3 scripts/validate_dataset.py` from this directory. This uses the Python standard library only, installs nothing and makes no network requests.

## Implementation sequence

1. Load Markdown while preserving metadata and section IDs.
2. Search active documents for current questions; keep archived documents available for historical questions.
3. Implement local retrieval returning original passages and citation IDs such as NL-09:S1.
4. Evaluate retrieval using development.jsonl before adding a language model.
5. Connect a local model to selected passages and validate citation IDs.
6. Assess answer quality separately using required_claims and actual evidential support.

Not yet included: a search application, a language model, model installation or PDF extraction. Markdown sections are the initial source locations. Do not invent page numbers.

## Evaluation and limitations

There are 24 development questions and six held-out validation questions. This small split is an initial smoke test, not a robust general benchmark. Documents and questions were jointly AI-assisted; the validation set is therefore not independent. Add separately authored, manually reviewed test questions before final portfolio evaluation. Do not use validation questions for prompt tuning.

required_claims are semantic criteria, not exact-match strings. Different wording can be correct. The conflict question requires both contradictory sources. For unanswerable questions, an explicit information gap is appropriate; absence of a reimbursement rule does not prove reimbursement is forbidden.

Recall@k measures retrieval of annotated evidence sections. A valid citation ID alone does not prove that a source supports a claim. Review answers manually before introducing automated judges. Results on this known synthetic dataset do not establish reliability on real company documents.

## Document snapshot

Fictional evaluation date: 6 November 2026. All active documents take effect on 1 October 2026. The archived travel policy was replaced on that date. manifest.json records file hashes; update it after intentional document changes.

## English edition

Document content, evaluation questions, criteria, metadata and guidance are in English. German locations, EUR amounts and German local time are retained as the fictional company context. Document and section identifiers remain unchanged from v0.1. This edition replaces the earlier German-language package.
