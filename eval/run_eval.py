"""Run a basic evaluation of The Librarian on the evaluation dataset."""

import json
import math
import time
from pathlib import Path

from src.generation import generate_answer, ABSTENTION

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"
SUMMARY_PATH = ROOT / "eval" / "summary.json"


def load_questions():
    """Load evaluation questions from JSONL."""
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_latency_ms(result, elapsed_ms):
    """Use reported total latency when available; otherwise use wall time."""
    timings = result.get("timings_ms", {})

    if isinstance(timings, dict):
        for key in ("total", "total_ms", "total_time"):
            value = timings.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

    return round(elapsed_ms, 2)


def percentile(values, percentile_value):
    """Calculate a percentile using the nearest-rank method."""
    if not values:
        return None

    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def evaluate_question(item):
    """Run one question and evaluate abstention and citation-source matching."""
    started = time.perf_counter()

    try:
        result = generate_answer(item["question"])
        elapsed_ms = (time.perf_counter() - started) * 1000

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        cited_urls = sorted({
            source.get("source_url", "")
            for source in sources
            if source.get("source_url")
        })
        expected_urls = sorted(set(item.get("supporting_urls", [])))

        answerable = bool(item["answerable"])
        abstained = (
            answer.strip().casefold() == ABSTENTION.strip().casefold()
        )

        # Correct decision means answering answerable questions and
        # abstaining on unanswerable questions.
        abstention_correct = abstained == (not answerable)

        if not answerable:
            # This checks that the system abstained without returning sources.
            citation_match = abstained and len(cited_urls) == 0
            citation_precision = None
            citation_recall = None
        else:
            # This measures URL overlap, not whether the passage truly
            # supports the generated answer.
            matches = set(cited_urls) & set(expected_urls)
            citation_match = bool(matches)
            citation_precision = (
                len(matches) / len(cited_urls) if cited_urls else 0.0
            )
            citation_recall = (
                len(matches) / len(expected_urls)
                if expected_urls else None
            )

        return {
            "id": item["id"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "answer": answer,
            "answerable": answerable,
            "abstained": abstained,
            "abstention_correct": abstention_correct,
            "expected_urls": expected_urls,
            "cited_urls": cited_urls,
            "citation_match": citation_match,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "latency_ms": round(get_latency_ms(result, elapsed_ms), 2),
            # Requires human review; not inferred automatically.
            "manual_correctness": None,
            "error": None,
        }

    except Exception as error:
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "id": item["id"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "answer": "",
            "answerable": bool(item["answerable"]),
            "abstained": False,
            "abstention_correct": False,
            "expected_urls": item.get("supporting_urls", []),
            "cited_urls": [],
            "citation_match": False,
            "citation_precision": None,
            "citation_recall": None,
            "latency_ms": round(elapsed_ms, 2),
            "manual_correctness": None,
            "error": str(error),
        }


def main():
    questions = load_questions()
    results = []

    print(f"Evaluating {len(questions)} questions...")
    print("This may take a few minutes on the first run.\n")

    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['question']}")

        result = evaluate_question(item)
        results.append(result)

        status = "ERROR" if result["error"] else (
            "ABSTAIN" if result["abstained"] else "ANSWER"
        )
        print(f"  {status} | {result['latency_ms']:.0f} ms")

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    successful = [result for result in results if not result["error"]]
    answerable = [r for r in successful if r["answerable"]]
    unanswerable = [r for r in successful if not r["answerable"]]

    latencies = [r["latency_ms"] for r in successful]

    correct_decisions = sum(
        r["abstention_correct"] for r in successful
    )
    correct_abstentions = sum(
        r["abstained"] for r in unanswerable
    )
    missed_abstentions = sum(
        not r["abstained"] for r in unanswerable
    )

    citation_matches = [r["citation_match"] for r in answerable]

    summary = {
        "total_questions": len(questions),
        "successful_runs": len(successful),
        "errors": len(results) - len(successful),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),

        # Overall decision correctness across successful runs.
        "decision_accuracy": (
            correct_decisions / len(successful)
            if successful else None
        ),

        # Correct refusals among unanswerable questions.
        "correct_abstentions": correct_abstentions,
        "abstention_recall": (
            correct_abstentions / len(unanswerable)
            if unanswerable else None
        ),
        "missed_abstentions": missed_abstentions,

        # URL overlap only; this does not prove semantic support.
        "answerable_questions_with_matching_citations": sum(
            citation_matches
        ),
        "citation_source_match_rate": (
            sum(citation_matches) / len(citation_matches)
            if citation_matches else None
        ),

        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),

        "manual_correctness": (
            "Not measured automatically; review eval/results.jsonl."
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nEvaluation summary")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()