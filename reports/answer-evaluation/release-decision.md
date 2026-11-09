# Current release decision

Candidate: qwen2.5:14b with grounded-v1.1 and five semantic passages.

## Decision

The candidate is not approved for production use.
It remains a development prototype.

## Observed result

- 24 development answers reviewed.
- 20 passed the recorded review.
- 3 require revision.
- 1 has a mismatch between reference expectations and the active corpus.
- Known issues: missing material conditions and an unsupported inference.
- Original source passages are displayed, but this does not repair the answer.

## Promotion policy

For the next candidate:
- No observed material contradiction or unsupported actionable conclusion.
- No omitted condition that materially changes the user's action.
- Factual claims must have appropriate source support.
- Scope mismatches must be documented separately.
- Evaluate the full development set using the same criteria.
- Use the held-out questions only after selecting a candidate.

These gates are development acceptance rules, not proof of universal safety.
They were formalized after inspecting the current development results.

## Evaluation limitations

Reviews are assistant-assisted and not independent human evaluations.
The dataset is small and synthetic.
Repeated use of development questions creates a risk of overfitting.
A low sample error rate cannot establish production reliability.