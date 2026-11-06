# Initial prompt comparison

## Scope

Model: llama3:latest
Questions: Q003, Q027, Q029 from the development set.
Retrieval and reference criteria were unchanged between runs.

These are single-run observations on three known development questions.
They do not establish general model quality or a reliable latency benchmark.

## Findings

| Question | grounded-v1 | grounded-v2 |
|---|---|---|
| Q003: Workation | All required conditions stated. Repetitive and unnecessarily uncertain. Citation brackets missing. | Incorrectly claims that two weeks exceed ten working days. Required conditions and bracketed citations present. |
| Q027: Backup retention | Correctly describes conflicting retention periods and refers to the system owner. Does not explicitly state that both documents are active and no precedence rule exists. | Selects 30 days and omits the conflicting 60-day policy. |
| Q029: Annual leave | Correctly acknowledges missing information. Citation brackets missing. | Correctly acknowledges missing information with a bracketed citation. |

## Decision

Retain grounded-v1 as the current baseline.
Do not adopt grounded-v2: formatting improvements do not compensate for
incorrect conclusions and omission of conflicting evidence.

grounded-v1 still requires broader evaluation.

## Review method

Answers were checked against the supplied source passages and required claims.
Valid source IDs were assessed separately from factual correctness and completeness.
This is a qualitative, assistant-assisted review, not an independent human benchmark.

## Remaining limitations

- No automatic verification of factual support.
- No claim-level citation coverage check.
- No repeated generation runs.
- Only three development questions reviewed so far.
- Validation questions remain unused.