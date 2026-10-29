import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW_PATH = ROOT / "eval" / "manual_review.jsonl"


def rate(rows, field):
    reviewed = [r for r in rows if isinstance(r.get(field), bool)]
    if not reviewed:
        return {"reviewed": 0, "correct": None}
    return {
        "reviewed": len(reviewed),
        "correct": sum(r[field] for r in reviewed),
        "rate": sum(r[field] for r in reviewed) / len(reviewed),
    }


def main():
    with REVIEW_PATH.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    summary = {
        "total_questions": len(rows),
        "factual_correctness_answerable": rate(answerable, "factual_correct"),
        "citation_support_answerable": rate(answerable, "citation_supported"),
        "abstention_decision_accuracy": rate(rows, "abstention_correct"),
        "correct_abstentions": sum(
            r.get("abstention_correct") is True and not r["answerable"]
            for r in unanswerable
        ),
        "unanswerable_questions": len(unanswerable),
        "manual_review_remaining": sum(
            r.get("factual_correct") is None
            or r.get("citation_supported") is None
            for r in answerable
        ),
    }

    output = ROOT / "eval" / "quality_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
