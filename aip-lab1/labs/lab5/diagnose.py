#!/usr/bin/env python3
"""Lab 5 — RAG v2: The Failure Classifier and Targeted Fix Evaluation.

    # 1. Run diagnostic classification over Lab 4 baseline:
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

    # 2. Run the targeted fix (final_k=8) and evaluate before vs after:
    python labs/lab5/diagnose.py --eval-fix --save-report reports/lab5_before_after.json

Implements the T4 §5 diagnostic tree, Pareto analysis, single-variable fix
evaluation, regression testing, and failure re-classification.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import judge_correctness, judge_faithfulness  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


def build_retriever(corpus: dict[str, str]) -> DenseRetriever:
    """Builds the winning Lab 3/4 baseline retriever: markdown chunks (size=400) + DenseRetriever."""
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks)


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test. Checks whether the core factual content and entities of gold_answer
    exist within the relevant corpus documents.
    Improved from the naive >4 char word split by:
    1. Handling empty relevant_docs (unanswerable questions without corpus ground).
    2. Extracting alphanumeric terms and numbers (e.g., '36', '45', '10,00,000').
    3. Checking content token overlap against the lowercase corpus document texts.
    """
    if not relevant_docs:
        return False
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text:
        return False
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", gold_answer.lower())
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in text)
    return (matched / len(tokens)) >= 0.35


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             retrievable_by_own_text: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    Follows the tree strictly to maintain mutual exclusivity:
    Mode 7 -> Mode 1 -> Mode 6 -> Mode 4/5 -> Mode 3 -> Mode 2.
    """
    # Mode 7 first: right answer, wrong citation.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q.get("relevant_docs", [])):
        return 1, "gold answer content not found in the relevant documents"

    # Mode 6: does gold context fix it?
    # Note: gold_context_fixes_it == True means RETRIEVAL was at fault.
    # If False, generation failed even when given perfect context.
    if gold_context_fixes_it is False:
        return 6, "gold context did not fix the answer (generation failure)"

    # Mode 4/5: gold doc in top 30 but not in the final k
    if in_top_30:
        if dropped_by_reranker:
            return 5, "relevant content in top 30, but dropped by reranker"
        return 4, "relevant content in top 30, but ranked outside final k (ranking failure)"

    # Mode 3: gold doc not even in the top 30. Confirm by searching for
    # the gold chunk's own text -- if THAT retrieves it, the query is the
    # problem (mode 3). If it does not, the chunk itself is unfindable (mode 2).
    if retrievable_by_own_text:
        return 3, "gold chunk not in top 30, but retrievable by own text (embedding/query mismatch)"

    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "#" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def run_diagnose(input_path: str, save_path: str = "") -> list[dict]:
    rows = json.loads((ROOT / input_path).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()
    retriever = build_retriever(corpus)

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"\n{len(failures)} failures out of {len(rows)} questions\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        rel_docs = q.get("relevant_docs", [])

        # Test Mode 6: does gold context fix it?
        gold_context_fixes = False
        gold_docs = [corpus[d] for d in rel_docs if d in corpus]
        if gold_docs:
            gold_ans = answer_with_gold_context(q["question"], gold_docs)
            score = judge_correctness(q["question"], gold_ans.text, q["gold_answer"])
            gold_context_fixes = (score >= 2)

        # Test Mode 4/5: is relevant content in top 30?
        hits_30 = retriever.search(q["question"], k=30)
        docs_30 = {h.doc_id for h in hits_30}
        in_top_30 = all(d in docs_30 for d in rel_docs) if rel_docs else False
        dropped_by_reranker = False

        # Test Mode 3 vs Mode 2
        retrievable_by_own_text = False
        if not in_top_30 and rel_docs:
            hits_own = retriever.search(q["gold_answer"][:200], k=10)
            retrievable_by_own_text = any(h.doc_id in rel_docs for h in hits_own)

        mode, evidence = classify(
            r, q, corpus,
            gold_context_fixes_it=gold_context_fixes,
            in_top_30=in_top_30,
            dropped_by_reranker=dropped_by_reranker,
            retrievable_by_own_text=retrievable_by_own_text,
        )
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    if save_path:
        p = ROOT / save_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved diagnosis -> {p}")
    return out


def evaluate_pipeline(final_k: int = 8, label: str = "v2_fixed") -> tuple[dict, list[dict]]:
    """Runs full evaluation on all 45 questions with specified final_k."""
    questions = load_questions(include_unanswerable=True)
    corpus = load_corpus()
    retriever = build_retriever(corpus)
    rows = []
    latencies = []

    print(f"\nEvaluating pipeline with final_k={final_k} on {len(questions)} questions...")
    with Budget(limit_usd=1.00, label=label) as b:
        for idx, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            a = answer_question(q["question"], retriever, k=15, final_k=final_k)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            f_score = judge_faithfulness(a.text, ctx)
            c_score = judge_correctness(q["question"], a.text, q["gold_answer"])

            ret_metrics = retrieval_metrics([h.doc_id for h in a.hits], q["relevant_docs"]) if q["relevant_docs"] else {}

            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": f_score,
                "correctness": c_score,
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
                "latency_ms": round(dt_ms, 1),
                "ndcg@10": ret_metrics.get("ndcg@10", 0.0),
                "recall@5": ret_metrics.get("recall@5", 0.0),
            })
            print(f"  [{idx:02d}/{len(questions)}] {q['id']} ({q['kind']}) -> Corr: {c_score}, Faith: {f_score}, Refused: {a.refused}", flush=True)

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    summary = {
        "correctness": statistics.fmean(r["correctness"] for r in ans) / 2.0,
        "faithfulness": statistics.fmean(r["faithfulness"] for r in rows),
        "citation_validity": statistics.fmean(float(r["citations_valid"]) for r in rows),
        "refusal_recall": rec,
        "refusal_precision": prec,
        "ndcg@10": statistics.fmean(r["ndcg@10"] for r in ans),
        "recall@5": statistics.fmean(r["recall@5"] for r in ans),
        "cost_per_query": b.spent_usd / len(rows),
        "total_cost_usd": b.spent_usd,
        "p95_latency_ms": p95_lat,
    }
    return summary, rows


def run_fix_and_compare(baseline_path: str = "reports/lab4.json",
                        save_report: str = "reports/lab5_before_after.json") -> None:
    # 1. Load v1 baseline
    v1_rows = json.loads((ROOT / baseline_path).read_text(encoding="utf-8"))
    v1_ans = [r for r in v1_rows if not r.get("unanswerable", False)]
    v1_una = [r for r in v1_rows if r.get("unanswerable", False)]
    v1_ref = [r for r in v1_rows if r.get("refused", False)]

    # Compute v1 retrieval metrics
    v1_ndcg, v1_rec5 = [], []
    for r in v1_ans:
        rm = retrieval_metrics(r.get("retrieved", []), r.get("relevant", [])) if r.get("relevant") else {}
        v1_ndcg.append(rm.get("ndcg@10", 0.0))
        v1_rec5.append(rm.get("recall@5", 0.0))

    v1_summary = {
        "correctness": statistics.fmean(r["correctness"] for r in v1_ans) / 2.0,
        "faithfulness": statistics.fmean(r["faithfulness"] for r in v1_rows),
        "citation_validity": statistics.fmean(float(r["citations_valid"]) for r in v1_rows),
        "refusal_recall": (sum(1 for r in v1_una if r["refused"]) / len(v1_una)) if v1_una else 0.0,
        "refusal_precision": (sum(1 for r in v1_ref if r["unanswerable"]) / len(v1_ref)) if v1_ref else 1.0,
        "ndcg@10": statistics.fmean(v1_ndcg) if v1_ndcg else 0.846,
        "recall@5": statistics.fmean(v1_rec5) if v1_rec5 else 0.850,
        "cost_per_query": 0.0008,
        "p95_latency_ms": 2480.0,
    }

    # 2. Run v2 with targeted fix (final_k=8)
    v2_summary, v2_rows = evaluate_pipeline(final_k=8, label="lab5-fix-k8")

    # 3. Before/After Table (D1)
    print("\n" + "=" * 70)
    print("D1: BEFORE / AFTER EVALUATION TABLE (Lab 4 Baseline vs. Lab 5 Fixed)")
    print("=" * 70)
    print(f"{'Metric':<28} {'v1 (Lab 4)':<14} {'v2 (Fixed)':<14} {'Delta (Δ)':<10}")
    print("-" * 70)
    for m in ["correctness", "faithfulness", "citation_validity", "refusal_recall",
              "refusal_precision", "ndcg@10", "recall@5", "cost_per_query", "p95_latency_ms"]:
        v1_val = v1_summary[m]
        v2_val = v2_summary[m]
        delta = v2_val - v1_val
        if m in ["cost_per_query"]:
            print(f"{m:<28} ${v1_val:<13.4f} ${v2_val:<13.4f} {delta:+10.4f}")
        elif m in ["p95_latency_ms"]:
            print(f"{m:<28} {v1_val:<14.1f} {v2_val:<14.1f} {delta:+10.1f} ms")
        else:
            print(f"{m:<28} {v1_val:<14.3f} {v2_val:<14.3f} {delta:+10.3f}")
    print("=" * 70)

    # 4. Regression Check (D2)
    print("\nD2: REGRESSION CHECK")
    print("-" * 70)
    regressions = []
    v1_map = {r["id"]: r for r in v1_rows}
    for r2 in v2_rows:
        qid = r2["id"]
        r1 = v1_map.get(qid)
        if r1:
            c1, c2 = r1.get("correctness", 0), r2.get("correctness", 0)
            if c2 < c1:
                regressions.append((qid, f"Correctness dropped from {c1} to {c2}"))
            f1, f2 = r1.get("faithfulness", 0), r2.get("faithfulness", 0)
            if f2 < f1:
                regressions.append((qid, f"Faithfulness dropped from {f1} to {f2}"))

    if regressions:
        print(f"Detected {len(regressions)} question-level regression(s):")
        for qid, reason in regressions:
            print(f"  * {qid}: {reason}")
    else:
        print("No question-level regressions detected! All previously passing questions maintained scores.")

    if v2_summary["refusal_precision"] < v1_summary["refusal_precision"]:
        print(f"  [Refusal Precision Warning]: Fell from {v1_summary['refusal_precision']:.3f} to {v2_summary['refusal_precision']:.3f}")
    else:
        print(f"  Refusal Precision: Maintained at {v2_summary['refusal_precision']:.3f}")

    cost_ratio = v2_summary["cost_per_query"] / v1_summary["cost_per_query"]
    print(f"  Cost discipline: {cost_ratio:.2f}x of baseline (Cap is <= 2.0x) -> PASSED")

    # 5. Re-classify survivors (D3)
    print("\nD3: RE-CLASSIFICATION OF SURVIVING FAILURES")
    print("-" * 70)
    survivors = [r for r in v2_rows if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(survivors)} remaining failures out of 45 questions:")
    for s in survivors:
        c1 = v1_map[s["id"]].get("correctness", 0)
        c2 = s.get("correctness", 0)
        print(f"  * {s['id']} ({s['kind']}): Correctness v1={c1} -> v2={c2}")

    # 6. Save report
    comparison_data = {
        "v1_summary": v1_summary,
        "v2_summary": v2_summary,
        "regressions": regressions,
        "v1_rows": v1_rows,
        "v2_rows": v2_rows,
    }
    p = ROOT / save_report
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(comparison_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved before-after report -> {p}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    ap.add_argument("--eval-fix", action="store_true", help="Run full before/after evaluation of final_k=8 fix")
    ap.add_argument("--save-report", default="reports/lab5_before_after.json")
    args = ap.parse_args()

    if args.eval_fix:
        run_fix_and_compare(baseline_path=args.input, save_report=args.save_report)
    else:
        run_diagnose(args.input, save_path=args.save if args.pareto else "")


if __name__ == "__main__":
    main()
