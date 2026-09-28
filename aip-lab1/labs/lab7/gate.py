#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cost import Budget  # noqa: E402
from aip.retrieval import format_context  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402
from labs.lab4.evaluate import (  # noqa: E402
    build_retriever,
    judge_correctness,
    judge_faithfulness,
)
from labs.lab4.rag import answer_question  # noqa: E402


def measure() -> dict[str, float]:
    """TODO D1: run your golden set and return the metric dict.

    Keys must match thresholds.yml. Run under AIP_OFFLINE=1 so CI replays the
    committed cache and costs nothing.
    """
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []
    latencies = []

    with Budget(limit_usd=1.00, label="regression-gate") as b:
        for q in questions:
            t0 = time.perf_counter()
            a = answer_question(q["question"], retriever, k=12, final_k=5)
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)

            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            f_score = judge_faithfulness(a.text, ctx)
            c_score = judge_correctness(q["question"], a.text, q["gold_answer"])

            rows.append({
                "id": q["id"],
                "unanswerable": unanswerable,
                "answer": a.text,
                "refused": a.refused,
                "citations_valid": a.citations_valid,
                "faithfulness": f_score,
                "correctness": c_score,
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    hit_rate_5 = statistics.fmean(
        1.0 if any(doc in r["retrieved"][:5] for doc in r["relevant"]) else 0.0
        for r in ans
    )

    lat_sorted = sorted(latencies)
    p95_lat = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0

    return {
        "correctness": round(statistics.fmean(r["correctness"] for r in ans) / 2.0, 4),
        "faithfulness": round(statistics.fmean(r["faithfulness"] for r in rows), 4),
        "citation_validity": round(statistics.fmean(1.0 if r["citations_valid"] else 0.0 for r in rows), 4),
        "refusal_recall": round(rec, 4),
        "refusal_precision": round(prec, 4),
        "hit_rate_at_5": round(hit_rate_5, 4),
        "cost_per_query_usd": round(b.spent_usd / len(questions), 4),
        "p95_latency_ms": round(p95_lat, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
