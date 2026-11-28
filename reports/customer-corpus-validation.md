# Customer Corpus Validation

**Date:** 2026-11-28  
**Scope:** Technical validation of the Starter deployment scope  
**Environment:** Windows 11, WSL Ubuntu, Docker Desktop, Intel i7-8665U, approximately 32 GB RAM

## Objective

Verify that the document assistant can ingest, index and retrieve information from 20 text-based PDFs containing 200 pages, including a dense corpus exceeding 70,000 words.

## Test corpora

Two synthetic corpora were generated using fictional Westbridge Manufacturing documents.

- **Validation corpus:** Individual policy sections with selected longer passages.
- **Stress corpus:** Expanded versions containing additional operational text. Paragraph templates are not repeated within an individual page but may recur across documents.

The stress corpus is intended for technical load testing rather than answer-quality evaluation.

## Technical results

| Metric | Validation corpus | Stress corpus |
| --- | ---: | ---: |
| Documents | 20 | 20 |
| Pages | 200 | 200 |
| Total words | 6,866 | 72,115 |
| Mean words per page | 34.3 | 360.6 |
| Indexed chunks | 206 | 600 |
| Pages with multiple chunks | 4 | 200 |
| Index build time | 26.70 s | 43.13 s |
| Peak build RSS | 651 MiB | 698 MiB |
| Index size | 318,138 bytes | 985,000 bytes |
| Index setup time | 12.30 s | 10.91 s |
| Median retrieval latency | 16.66 ms | 15.62 ms |
| Maximum embedding input tokens | 207 | 210 |
| Missing indexed pages | 0 | 0 |
| Empty chunks | 0 | 0 |
| Duplicate citation IDs | 0 | 0 |
| Oversized embedding inputs | 0 | 0 |

All indexed embedding inputs remained below the model limit of 256 tokens.

Retrieval latency includes query encoding and similarity search. Measurements were taken after encoder warm-up, using five questions and three repetitions per question.

## Docker verification

The stress corpus was successfully indexed offline and loaded within the customer Docker deployment.

| Metric | Result |
| --- | ---: |
| Indexed documents | 20 |
| Indexed pages | 200 |
| Chunks | 600 |
| Index build time | 48.83 s |
| Index setup time | 12.68 s |
| Median retrieval latency | 17.34 ms |
| Missing pages | 0 |
| Oversized embedding inputs | 0 |

### Application container memory

| Application state | Observed memory |
| --- | ---: |
| Before first request | 68.38 MiB |
| During generation | 322.1 MiB |
| After completed request | 446.2 MiB |

These measurements are point-in-time observations, not peak memory measurements. The external Ollama process is excluded.

### End-to-end answer test

**Question:** How often must hydraulic presses undergo preventive maintenance?

**Provider:** Ollama  
**Model:** qwen2.5:14b

**Generated answer:**

Hydraulic presses require preventive servicing every 500 operating hours or six months, whichever occurs first [DOC-009:S10].

**Result:** Passed.

The answer included both maintenance intervals and the required condition. The correct source passage was retrieved at rank 1, referencing page 4 of `machine_maintenance.pdf`.

- Retrieval time: 0.023 seconds.
- Generation time: 538.39 seconds.

Local generation latency remains a significant limitation on the tested CPU-only hardware.

## Tokenizer warning

During ingestion, full-page tokenization produces a sequence-length warning before the page is split into smaller chunks.

Subsequent index validation confirmed that no actual embedding input exceeded the model token limit.

## Reproducibility

The following scripts support corpus generation and technical validation:

- `scripts/build_customer_validation_corpus.py`
- `scripts/fixtures/customer_validation_documents.json`
- `scripts/build_customer_stress_corpus.py`
- `scripts/validate_customer_index.py`

Generated PDFs, indexes and answer reports are stored outside version control under `customer_data/`.

## Limitations

- Both corpora are synthetic.
- Stress corpus text contains recurring paragraph templates across documents.
- Page coverage does not independently prove lossless extraction of every character.
- Technical retrieval measurements do not establish retrieval relevance.
- One successful end-to-end answer does not establish general answer quality.
- Scanned PDFs, OCR and complex table layouts are outside the validated scope.
- Performance measurements depend on hardware, document content and runtime conditions.

## Conclusion

The application successfully processed the intended Starter document volume of 20 text-based PDFs and 200 pages, including a stress corpus containing 72,115 words.

Technical ingestion, chunking, index persistence, source references and Docker compatibility were verified.

Systematic retrieval and answer-quality evaluation with both supported generation providers remains a separate validation step.