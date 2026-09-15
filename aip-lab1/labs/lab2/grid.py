#!/usr/bin/env python3
"""Lab 2 grid runner. Written for you.

    python labs/lab2/grid.py --variants zero_shot few_shot --split dev
    python labs/lab2/grid.py --all --split dev --save reports/lab2_grid.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from aip.evals import compare, field_accuracy, load_cases, run_eval
from aip.llm import model_latency_ms, repair_attempts
from labs.lab1.run_eval import GRADED_FIELDS
from labs.lab2.stats import paired_test, wilson_interval
from labs.lab2.variants import FEW_SHOT_IDS, VARIANTS

TICKETS_PER_DAY = 10_000


def metric(pred, gold) -> dict[str, float]:
    if hasattr(pred, "model_dump"):
        pred = pred.model_dump()
    m = field_accuracy(pred, gold, GRADED_FIELDS)
    m["schema_valid"] = float(isinstance(pred, dict) and bool(pred)
                              and not pred.get("_used_fallback"))
    m["fallback_rate"] = float(bool(isinstance(pred, dict)
                                     and pred.get("_used_fallback")))
    m["repair_rate"] = float(bool(isinstance(pred, dict)
                                  and pred.get("_repair_attempts", 0)))
    m["escalated"] = float(isinstance(pred, dict) and pred.get("_path") == "large")
    if isinstance(pred, dict) and pred.get("_small_record"):
        m["small_correct"] = field_accuracy(pred["_small_record"], gold, GRADED_FIELDS)["record_accuracy"]
        if pred.get("_small_agreement") is not None:
            m["small_agreement"] = float(pred["_small_agreement"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    names = list(VARIANTS) if args.all else args.variants
    if not names:
        ap.error("give --variants or --all")

    cases = load_cases(ROOT / f"data/eval/extraction_{args.split}.jsonl")
    if args.split == "dev" and any("few_shot" in name for name in names):
        # The labelled examples are part of the prompt. Never grade them.
        cases = [case for case in cases if case.id not in FEW_SHOT_IDS]
        print(f"held out {len(FEW_SHOT_IDS)} few-shot examples; evaluating {len(cases)} dev cases")
    if args.n:
        cases = cases[: args.n]

    def measured_variant(name: str):
        def run(ticket: str) -> dict:
            token = repair_attempts.set(0)
            latency_token = model_latency_ms.set(0.0)
            try:
                result = VARIANTS[name](ticket)
                return {**result, "_repair_attempts": repair_attempts.get(),
                        "_model_latency_ms": model_latency_ms.get()}
            finally:
                repair_attempts.reset(token)
                model_latency_ms.reset(latency_token)
        return run

    reports = []
    for name in names:
        if name not in VARIANTS:
            ap.error(f"unknown variant {name!r}; choose from {', '.join(VARIANTS)}")
        rep = run_eval(name, cases, measured_variant(name), metric,
                       budget_usd=args.budget, workers=args.workers)
        print(rep.summary())
        agg = rep.aggregate()
        if agg.get("fallback_rate", 0) > 0.20:
            first = next((r.output.get("review_reason", "") for r in rep.results
                          if isinstance(r.output, dict) and r.output.get("_used_fallback")), "unknown")
            print(f"   !! {agg['fallback_rate']:.1%} fell back to heuristics. First reason: {first[:240]}")
            print("   Provider or schema failed too often; stopping before invalid comparisons.")
            return

        # Every case errored -- almost always an unimplemented TODO or a raise
        # inside the variant. Say so plainly instead of dying on a KeyError.
        if "record_accuracy" not in agg:
            first = next((r.error for r in rep.results if r.error), "unknown")
            print(f"   !! every case failed. First error:\n      {first}")
            print("   Fix the variant before reading anything else here.\n")
            continue

        correct = round(agg["record_accuracy"] * rep.n)
        lo, hi = wilson_interval(correct, rep.n)
        print(f"   record_accuracy 95% CI      [{lo:.3f}, {hi:.3f}]")

        # Cost, honestly. Some providers publish no per-token price, and a
        # cost meter that prints $0.00 for a paid API is the exact bug this
        # module teaches against -- so when prices are missing we report the
        # thing we DID measure (tokens) instead of a confident zero.
        if rep.budget.get("unpriced_calls", 0):
            tok = (rep.budget.get("prompt_tokens", 0)
                   + rep.budget.get("completion_tokens", 0)) / max(rep.n, 1)
            out_tok = rep.budget.get("completion_tokens", 0) / max(rep.n, 1)
            print(f"   cost                        UNPRICED "
                  f"({rep.budget['unpriced_calls']} calls to models with no "
                  f"published price)")
            print(f"   tokens / ticket             {tok:,.0f} total, "
                  f"{out_tok:,.0f} out")
            print(f"   tokens / 1k tickets         {tok * 1000:,.0f}")
            print("   -> compare variants on TOKENS. Cost is proportional to")
            print("      tokens at a fixed price, so the ranking still holds.")
        else:
            per_ticket = rep.budget.get("estimated_cost_usd", rep.budget["cost_usd"]) / max(rep.n, 1)
            print(f"   cost / 1k tickets           ${per_ticket * 1000:.2f}")
            print(f"   cost / year @ {TICKETS_PER_DAY:,}/day  "
                  f"${per_ticket * TICKETS_PER_DAY * 365:,.0f}")
        if agg.get("escalated", 0) > 0:
            print(f"   escalation rate             {agg['escalated']:.1%}")
        print()
        reports.append(rep)

    if reports:
        print("\nGrid table (estimated deployment cost includes cached calls)")
        header = ("variant", "record_accuracy", "field_accuracy", "schema_valid",
                  "repair_rate", "fallback_rate", "cost_usd", "cost_per_1k_tickets", "p50_ms", "p95_ms")
        print(" | ".join(header))
        for rep in reports:
            agg = rep.aggregate()
            durations = sorted(result.output.get("_model_latency_ms", result.latency_ms)
                               if isinstance(result.output, dict) else result.latency_ms
                               for result in rep.results)
            p50 = statistics.median(durations) if durations else 0.0
            p95 = durations[min(len(durations)-1, round(.95*(len(durations)-1)))] if durations else 0.0
            deployment_cost = rep.budget.get("estimated_cost_usd", rep.budget["cost_usd"])
            cost = "UNPRICED" if rep.budget.get("unpriced_calls") else f"{deployment_cost:.6f}"
            per_1k = "UNPRICED" if rep.budget.get("unpriced_calls") else f"{deployment_cost * 1000 / rep.n:.3f}"
            print(" | ".join((rep.name, *(f"{agg.get(k, 0):.3f}" for k in header[1:6]),
                              cost, per_1k, f"{p50:.0f}", f"{p95:.0f}")))

    if len(reports) > 1:
        print(compare(*reports, metrics=["record_accuracy", "field_accuracy",
                                         "schema_valid", "error_rate"]))
        print()
        print("paired comparisons vs. the first variant:")
        base = {r.id: r.metrics.get("record_accuracy", 0) == 1.0
                for r in reports[0].results}
        for rep in reports[1:]:
            other = {r.id: r.metrics.get("record_accuracy", 0) == 1.0
                     for r in rep.results}
            ids = [c.id for c in cases if c.id in base and c.id in other]
            res = paired_test([base[i] for i in ids], [other[i] for i in ids])
            print(f"  {reports[0].name} vs {rep.name}: "
                  f"b={res['b']} c={res['c']} -> {res['verdict']}")
        print()
        # If any run was unpriced, dominance on "cost" would compare zeros and
        # declare everything dominated. Fall back to total tokens, which is
        # what cost is proportional to anyway.
        unpriced = any(r.budget.get("unpriced_calls", 0) for r in reports)
        axis = "tokens" if unpriced else "cost"
        print(f"Dominated configurations (worse on quality AND {axis} AND latency):")
        rows = [(r.name, r.aggregate()["record_accuracy"],
                 (r.budget.get("prompt_tokens", 0) + r.budget.get("completion_tokens", 0))
                 if unpriced else r.budget.get("estimated_cost_usd", r.budget["cost_usd"]),
                 sorted(x.output.get("_model_latency_ms", x.latency_ms)
                        if isinstance(x.output, dict) else x.latency_ms for x in r.results)
                 [min(r.n-1, round(.95*(r.n-1)))]) for r in reports]
        dominated = [
            a[0] for a in rows
            if any(b[1] >= a[1] and b[2] <= a[2] and b[3] <= a[3] and b[0] != a[0]
                   for b in rows)
        ]
        print("  " + (", ".join(dominated) if dominated else "none"))

    if args.save:
        out = ROOT / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({r.name: r.to_dict() for r in reports},
                                  indent=2, default=str), encoding="utf-8")
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
