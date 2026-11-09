"""Record an assistant-assisted review of a specific saved development run."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SOURCE = Path(
    "artifacts/evaluation/answers/"
    "qwen25-14b-v11-development-full-01/combined-review.json"
)
OUTPUT = Path(
    "reports/answer-evaluation/qwen25-14b-v11-reviewed.json"
)
EXPECTED_ANSWERS_HASH = (
    "fecdfdf71f66d2a18d5f0860372974d45"
    "d9fb623c61feec2210c51e359d778bc"
)

PASSED_IDS = {
    "Q001", "Q002", "Q004", "Q006", "Q007",
    "Q008", "Q009", "Q012", "Q014", "Q016",
    "Q017", "Q019", "Q021", "Q022", "Q023",
    "Q024", "Q026", "Q027", "Q028", "Q029",
}

EXCEPTIONS = {
    "Q003": {
        "verdict": "needs_revision",
        "error_types": ["unsupported_inference"],
        "failed_claim_indices": [],
        "unsupported_additions_found": True,
        "notes": (
            "All required claims are present. The answer additionally says "
            "the user needs to request ten working days for two weeks. "
            "The actual number of working days is not established by the "
            "question or supplied evidence."
        ),
    },
    "Q013": {
        "verdict": "needs_revision",
        "error_types": ["missing_condition"],
        "failed_claim_indices": [3],
        "unsupported_additions_found": False,
        "notes": (
            "Price quotations and trade fair dates are now included. "
            "The answer does not explicitly preserve the requirement "
            "to obtain the approvals before booking."
        ),
    },
    "Q018": {
        "verdict": "needs_revision",
        "error_types": ["missing_condition"],
        "failed_claim_indices": [1],
        "unsupported_additions_found": False,
        "notes": (
            "Correctly distinguishes initial response from resolution, "
            "but omits that the target applies during support hours."
        ),
    },
    "Q011": {
        "verdict": "scope_mismatch",
        "error_types": ["reference_scope_mismatch"],
        "failed_claim_indices": [1],
        "unsupported_additions_found": False,
        "notes": (
            "The current EUR 120 limit is answered correctly. "
            "The reference also expects an explanation of the old version. "
            "NL-10:S2 is archived and excluded from the active index; "
            "NL-09:S2 was not among the supplied passages. "
            "Exclude this question from the overall answer pass rate."
        ),
    },
}


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Review already exists: {OUTPUT}")

    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    answers = {
        question["id"]: question["answer"]
        for question in original["questions"]
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            answers,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    if fingerprint != EXPECTED_ANSWERS_HASH:
        raise SystemExit(
            "Answers differ from the reviewed snapshot. "
            "Do not apply these judgments without reviewing the differences."
        )

    expected_ids = PASSED_IDS | set(EXCEPTIONS)
    if set(answers) != expected_ids:
        raise SystemExit("Unexpected question set")

    reviewed = []
    for question in original["questions"]:
        question_id = question["id"]
        decision = EXCEPTIONS.get(
            question_id,
            {
                "verdict": "pass",
                "error_types": [],
                "failed_claim_indices": [],
                "unsupported_additions_found": False,
                "notes": (
                    "Required claims are covered; no material unsupported "
                    "addition identified in this review."
                ),
            },
        )

        claim_reviews = [
            {
                "criterion": claim["criterion"],
                "met": index not in decision["failed_claim_indices"],
            }
            for index, claim in enumerate(question["required_claims"])
        ]

        formatting_note = ""
        if question_id == "Q027":
            formatting_note = (
                "Retention values are explicitly attributed to source IDs "
                "in prose. Not all IDs use individual square brackets. "
                "Treat as presentation, not missing source attribution."
            )

        reviewed.append({
            **question,
            "status": "content_review_completed",
            "required_claims": claim_reviews,
            "recorded_review": {
                **decision,
                "required_claims_complete": all(
                    claim["met"] for claim in claim_reviews
                ),
                "citation_format_note": formatting_note,
            },
        })

    counts = Counter(
        question["recorded_review"]["verdict"]
        for question in reviewed
    )
    eligible = counts["pass"] + counts["needs_revision"]

    result = {
        "run": original["run"],
        "review_metadata": {
            "recorded_at": datetime.now(UTC).isoformat(),
            "method": "Assistant-assisted qualitative source review",
            "independent_human_review": False,
            "source_report": str(SOURCE),
            "answers_sha256": fingerprint,
            "scope": (
                "One known synthetic development set, one generation "
                "per question. Not a production reliability estimate."
            ),
        },
        "review_summary": {
            "total": len(reviewed),
            "verdict_counts": dict(counts),
            "eligible_for_pass_rate": eligible,
            "pass_rate": counts["pass"] / eligible,
            "release_decision": "not_approved_for_production",
        },
        "questions": reviewed,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Review counts:", dict(counts))
    print(f"Passed: {counts['pass']}/{eligible}")
    print("Production release: not approved")
    print("Report:", OUTPUT)


if __name__ == "__main__":
    main()