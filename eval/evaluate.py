"""Evaluation workflow for retrieval quality and answer accuracy.

Retrieval metrics (no LLM): Hit@k and MRR over the expected source files.
Answer metrics (LLM-as-judge): correctness vs ground truth, plus groundedness
flag from the pipeline itself.

Usage: python eval/evaluate.py [dataset.json] [report.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.llm import invoke_structured  # noqa: E402
from app.vectorstore import get_vectorstore  # noqa: E402


class Verdict(BaseModel):
    correct: bool = Field(description="True if the answer is consistent with the reference answer.")
    reason: str


def retrieval_scores(question: str, expected: list[str], k: int) -> tuple[float, float, float, float]:
    """Returns (hit@k, mrr, precision@k, recall@k).

    precision@k: share of the k retrieved chunks that come from an expected source.
    recall@k: share of the expected source files that appear among the k chunks.
    """
    docs = get_vectorstore().similarity_search(question, k=k)
    sources = [d.metadata.get("source") for d in docs]
    ranks = [i + 1 for i, s in enumerate(sources) if s in expected]
    hit = 1.0 if ranks else 0.0
    mrr = 1.0 / ranks[0] if ranks else 0.0
    precision = len(ranks) / len(sources) if sources else 0.0
    recall = len({s for s in sources if s in expected}) / len(expected)
    return hit, mrr, precision, recall


def main(dataset_path: str, report_path: str) -> None:
    s = get_settings()
    data = json.loads(Path(dataset_path).read_text())
    graph = build_graph()
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict grader. Compare the answer to the reference."),
        ("human", "Question: {q}\nReference: {ref}\nAnswer: {ans}"),
    ])

    rows = []
    for item in data:
        hit, mrr, prec, rec = retrieval_scores(item["question"], item["expected_sources"], s.top_k)
        out = graph.invoke({"question": item["question"], "trace": []})
        v = invoke_structured(
            judge_prompt, Verdict,
            {"q": item["question"], "ref": item["ground_truth"], "ans": out["generation"]},
            default=Verdict(correct=False, reason="judge call failed"))
        rows.append({
            "question": item["question"],
            "hit_at_k": hit,
            "mrr": mrr,
            "precision_at_k": prec,
            "recall_at_k": rec,
            "answer_correct": v.correct,
            "grounded": out.get("grounded", False),
            "judge_reason": v.reason,
            "answer": out["generation"],
        })

    n = len(rows)
    summary = {
        f"hit@{s.top_k}": sum(r["hit_at_k"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        f"precision@{s.top_k}": sum(r["precision_at_k"] for r in rows) / n,
        f"recall@{s.top_k}": sum(r["recall_at_k"] for r in rows) / n,
        "answer_accuracy": sum(r["answer_correct"] for r in rows) / n,
        "groundedness": sum(r["grounded"] for r in rows) / n,
        "n": n,
    }
    Path(report_path).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.json",
        sys.argv[2] if len(sys.argv) > 2 else "eval/report.json",
    )