# Local versus OpenAI generation: development comparison

## Setup

- Dataset: 24 development questions.
- Prompt/input version: grounded-v1.2.
- Identical system messages, questions and source passages for both models.
- Local model: qwen2.5:14b, Ollama 0.40.9.
- Local digest: 7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6
- API model: gpt-5.4-2026-03-05, reasoning effort none.
- Temperature: 0.
- Output limit: 256 tokens.
- Local seed: 42. No API seed was supplied.
- No new retrieval during replay.
- Review: manual, assistant-assisted content assessment.

## Results

| Measure | Local | OpenAI |
| --- | --- | --- |
| Content pass, including editorial notes | 22 | 22 |
| Needs revision: Q013 | 1 | 1 |
| Insufficient historical source coverage: Q011 | 1 | 1 |
| Median generation request time | 146.26 seconds | 1.80 seconds |
| Total generation request time | 3726.62 seconds | 49.82 seconds |

All requests completed normally. Original messages were compared directly
and matched for all 24 questions.

Q013 has different omissions: the local answer omits the two quotations,
trade fair dates and explicit approval timing. The API answer omits the
explicit requirement for written manager approval.

Q011 lacks the historical evidence needed to explain the replaced policy.
Both answers correctly identify the current EUR 120 limit. The API's
claim that the documents do not mention EUR 100 is too broad for the
retrieved excerpts.

Minor citation and editorial issues remain, including unnecessary
additions and statements about absent information.

## Interpretation

This development comparison does not demonstrate a higher content pass
rate for the API model. It demonstrates substantially lower request
latency than CPU-only inference on the tested laptop.

The pass counts assess content and do not certify perfect citation
formatting, production readiness or general reliability.

The six separately evaluated validation questions are not included.
Results informed development and are not an independent final benchmark.

## Usage decision

Keep local generation available for offline use. Offer the API as an
explicit alternative, with disclosure that questions and retrieved
passages leave the machine and API usage incurs charges.

Never switch providers automatically after a failure.