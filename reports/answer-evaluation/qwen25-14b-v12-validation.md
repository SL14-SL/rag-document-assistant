# Qwen 2.5 14B: held-out validation review

## Evaluated configuration

- Model: qwen2.5:14b
- Model digest: 7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6
- Ollama version: 0.40.9
- Prompt/input version: grounded-v1.2
- Temperature: 0
- Seed: 42
- Context window: 4096 tokens
- Maximum generated tokens: 256
- Retrieval: semantic search, top five sections
- Run: artifacts/evaluation/answers/qwen25-14b-v12-validation-01

The six validation questions were evaluated after selecting the candidate
configuration. Reference answers and review criteria were not included
in generation requests.

This is a manual, assistant-assisted content review, not an automated
factual correctness test.

## Per-question results

| Question | Result | Findings |
| --- | --- | --- |
| Q005 | Pass | Includes the seven-calendar-day limit and the requirement for a new request afterwards. |
| Q010 | Needs revision | Correct hotel limit and written manager approval before booking. Omits the required travel request. |
| Q015 | Pass | Correctly states ten working days for complete expense claims. |
| Q020 | Pass | Correctly identifies P2 and the four-hour initial response target during support hours. |
| Q025 | Needs revision | Correctly states that open tickets are not deleted under this rule. Omits the required explanation that the 365-day rule applies to closed tickets. |
| Q030 | Pass | Correctly states that reimbursement is not specified and does not invent a reimbursement commitment. |

## Interpretation

- Four of six answers meet all predefined content criteria.
- Two answers omit information required by those criteria.
- No false factual claims were identified in this review.
- All expected evidence passages were present in the actual model requests.
- All six requests completed with stop reason `stop`.
- Runtime metadata collection reported no errors.

Q010 omits a required process step. Q025 answers the direct question
correctly but lacks an explanatory detail required by the existing rubric.
These findings differ in severity, but neither is counted as a full pass.

The sample is too small to estimate general system reliability.
These results must not be combined with earlier development results
from different prompt/input versions.

## Known development limitation

In the grounded-v1.2 development spot check, Q013 still omitted the
two price quotations, trade fair dates and explicit before-booking
requirement despite receiving the relevant source passages.

## Release scope

Suitable for a portfolio demonstration with visible original passages
and clearly stated limitations.

Not approved for unattended policy decisions or production use.

Source ID membership checks do not establish factual support or
answer completeness. Displaying original passages supports human
review but does not correct the generated summary.

The validation questions have now been inspected. Future changes
informed by these results require a fresh held-out set for an
independent evaluation.