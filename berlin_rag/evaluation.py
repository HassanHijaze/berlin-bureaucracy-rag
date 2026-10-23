"""Evaluation harness: run questions, print results, and summarize retrieval and refusal."""


def parse_question(question, number):
    """Read one evaluation entry, which may be a dict or a plain string."""
    if isinstance(question, dict):
        return {
            "id": question.get("id", f"Q{number}"),
            "text": question.get("question", ""),
            "sources": question.get("source_files", []),
            "coverage": question.get("coverage", "-"),
            "expected": question.get("expected"),
        }

    return {
        "id": f"Q{number}",
        "text": str(question),
        "sources": [],
        "coverage": "-",
        "expected": None,
    }


def print_result(number, total, entry, answer, hits, refused, found):
    """Print the answer and retrieved chunks for one evaluation question."""
    print("=" * 80)
    print(f"[{number}/{total}] {entry['id']}   coverage: {entry['coverage']}")
    print(f"Q: {entry['text']}")
    print("-" * 80)
    print(answer.strip())
    print("-" * 80)

    if entry["expected"]:
        print(f"EXPECTED: {entry['expected']}")

    print(f"RETRIEVED TOP-{len(hits)}:")
    for rank, (chunk, final, semantic, bm25) in enumerate(hits, 1):
        print(
            f"  {rank:2d}. final={final:.4f} semantic={semantic:.4f} "
            f"bm25={bm25:.4f} | {chunk['source']} (chunk {chunk['index']})"
        )

    if entry["sources"]:
        print(
            f"CORRECT SOURCE FOUND: {'yes' if found else 'NO'} "
            f"(wanted: {', '.join(entry['sources'])})"
        )

    if refused:
        print("FLAGS: REFUSED")

    print()


def print_summary(results):
    """Print retrieval and refusal rates for the evaluation run."""
    should_answer = [r for r in results if r["coverage"] != "none"]
    should_refuse = [r for r in results if r["coverage"] == "none"]

    print("=" * 80)
    print("SUMMARY EVALUATION")
    print("=" * 80)
    print("Retrieval:")

    if should_answer:
        total = len(should_answer)
        retrieved = sum(r["retrieved_correct"] for r in should_answer)
        answered = sum(not r["refused"] for r in should_answer)
        print(f"  right source retrieved: {retrieved}/{total} ({100 * retrieved / total:.1f}%)")
        print(f"  produced an answer: {answered}/{total} ({100 * answered / total:.1f}%)")

    if should_refuse:
        total = len(should_refuse)
        refused = sum(r["refused"] for r in should_refuse)
        print(f"  correctly refused: {refused}/{total} ({100 * refused / total:.1f}%)")

    print()
    print("NOTE:")
    print(
        "This evaluation measures retrieval and refusal behavior. "
        "It does NOT automatically determine whether an answer is legally correct."
    )


def run_review(system, questions, ids=None):
    """Run evaluation questions, print each result, and return a summary list."""
    if isinstance(questions, dict):
        questions = questions.get("questions", [])

    if ids:
        questions = [
            q for q in questions if isinstance(q, dict) and q.get("id") in ids
        ]

    results = []

    for number, question in enumerate(questions, 1):
        entry = parse_question(question, number)
        answer, hits = system.ask(entry["text"])

        refused = "do not cover" in answer.lower()
        retrieved_sources = [chunk["source"] for chunk, _, _, _ in hits]
        found = any(source in entry["sources"] for source in retrieved_sources)

        print_result(number, len(questions), entry, answer, hits, refused, found)

        results.append(
            {
                "id": entry["id"],
                "coverage": entry["coverage"],
                "refused": refused,
                "retrieved_correct": found,
                "answer": answer.strip(),
            }
        )

    print_summary(results)
    return results