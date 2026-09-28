#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, cost, tracing  # noqa: E402
from aip.chunking import markdown_chunks  # noqa: E402
from aip.config import resolve_model, settings  # noqa: E402
from aip.cost import Budget, BudgetExceeded, global_budget  # noqa: E402
from aip.embed import cosine, embed  # noqa: E402
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import _is_retryable, chat  # noqa: E402
from aip.retrieval import DenseRetriever, Hit, format_context  # noqa: E402
from labs.lab3.search import load_corpus  # noqa: E402
from labs.lab4.rag import REFUSAL, validate_answer  # noqa: E402
from labs.lab6.agent import ToolGuard, filter_output, run_agent  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_STARTED = time.time()

# ---------------------------------------------------------------------------
# Caching layers (Part B1)
# ---------------------------------------------------------------------------
_EXACT_CACHE: dict[str, dict[str, Any]] = {}
_SEMANTIC_CACHE: list[dict[str, Any]] = []
SEMANTIC_THRESHOLD: float = float(os.getenv("AIP_SEMANTIC_THRESHOLD", "0.96"))


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _exact_cache_get(mode: str, top_k: int, question: str) -> dict[str, Any] | None:
    norm_q = _normalize_query(question)
    key = f"{mode}::{top_k}::{norm_q}"
    return _EXACT_CACHE.get(key)


def _exact_cache_put(mode: str, top_k: int, question: str, data: dict[str, Any]) -> None:
    norm_q = _normalize_query(question)
    key = f"{mode}::{top_k}::{norm_q}"
    _EXACT_CACHE[key] = data


def _semantic_cache_get(mode: str, top_k: int, question: str,
                        threshold: float = SEMANTIC_THRESHOLD) -> dict[str, Any] | None:
    if not _SEMANTIC_CACHE:
        return None
    candidates = [e for e in _SEMANTIC_CACHE if e["mode"] == mode and e["top_k"] == top_k]
    if not candidates:
        return None
    q_vec = embed(question, input_type="query")
    best_sim = -1.0
    best_entry = None
    for cand in candidates:
        sim = float(cosine(q_vec, cand["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best_entry = cand
    if best_sim >= threshold and best_entry is not None:
        tracing.event("cache.semantic_hit", similarity=round(best_sim, 4),
                      matched=best_entry["question"][:80])
        return best_entry["response"]
    return None


def _semantic_cache_put(mode: str, top_k: int, question: str,
                        data: dict[str, Any]) -> None:
    try:
        q_vec = embed(question, input_type="query")
        _SEMANTIC_CACHE.append({
            "mode": mode,
            "top_k": top_k,
            "question": question,
            "embedding": q_vec,
            "response": data,
        })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline class (Labs 3-5 pipeline + Lab 6 guards)
# ---------------------------------------------------------------------------
class PolicyPipeline:
    def __init__(self):
        corpus = load_corpus()
        chunks = [
            c for doc_id, text in corpus.items()
            for c in markdown_chunks(text, doc_id, size=800)
        ]
        self.retriever = DenseRetriever(chunks, show_progress=False)
        self.n_chunks = len(chunks)
        self.n_docs = len(corpus)

    def answer_rag(self, question: str, top_k: int = 5) -> tuple[str, bool, list[Hit]]:
        with tracing.trace("rag.retrieve", k=top_k) as span:
            hits = self.retriever.search(question, k=top_k)
            span["n_hits"] = len(hits)
            span["top_doc"] = hits[0].doc_id if hits else None

        if not hits:
            return REFUSAL, True, []

        context_str = delimit_untrusted(format_context(hits))
        prompt = f"{context_str}\n\nQuestion: {question}\n\nAnswer with citations:"

        from labs.lab4.rag import ANSWER_SYSTEM
        with tracing.trace("rag.generate", n_sources=len(hits)):
            res = chat(
                prompt,
                system=ANSWER_SYSTEM,
                tier="MAIN",
                temperature=0.0,
                max_tokens=600,
                return_full=True,
            )
            raw_text = res["text"].strip() if isinstance(res, dict) else str(res).strip()
            finish_reason = res.get("finish_reason") if isinstance(res, dict) else None

        # Guard: Filter output
        filtered_text = filter_output(raw_text)

        # Validation
        val = validate_answer(filtered_text, len(hits), finish_reason)
        if not val["valid"] and not val["refused"]:
            # One repair attempt
            repair_prompt = (
                f"{prompt}\n\nYour previous response was:\n{filtered_text}\n\n"
                f"It failed validation because: {val['reason']}.\n"
                f"Please rewrite the answer strictly following the rules: cite only valid sources "
                f"[1] to [{len(hits)}]. If the sources do not contain the answer, reply exactly:\n"
                f"{REFUSAL}"
            )
            res_rep = chat(repair_prompt, system=ANSWER_SYSTEM, tier="MAIN",
                           temperature=0.0, max_tokens=600, return_full=True)
            text_rep = res_rep["text"].strip() if isinstance(res_rep, dict) else str(res_rep).strip()
            val_rep = validate_answer(text_rep, len(hits), res_rep.get("finish_reason"))
            if val_rep["valid"]:
                filtered_text, val = text_rep, val_rep
            else:
                filtered_text = REFUSAL
                val["refused"] = True

        return filtered_text, val["refused"], hits

    def answer_tools(self, question: str) -> tuple[str, bool, list[Hit]]:
        with tracing.trace("tools.agent", question=question[:80]):
            guard = ToolGuard(max_calls=5)
            agent_res = run_agent(question, guard=guard, layers={1, 2, 3, 4, 5, 6})
            answer_text = agent_res.get("answer", "")
            refused = (
                REFUSAL in answer_text
                or "not enough information" in answer_text.lower()
                or "denied by policy" in answer_text.lower()
            )
            return answer_text, refused, []


def pipeline() -> PolicyPipeline:
    """TODO A2: build your Labs 3-5 pipeline once, at startup, and cache it."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = PolicyPipeline()
    return _PIPELINE


@app.on_event("startup")
def startup_event() -> None:
    pipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """TODO A1. Return cost and trace_id in the response -- they are how
    anyone debugs this later."""
    t0 = time.perf_counter()
    b_before = global_budget().spent_usd
    pipe = pipeline()

    try:
        with tracing.trace("http.ask", question=req.question[:120], mode=req.mode) as span:
            span_id = span.get("span_id", "trace-unknown")

            # Check Exact Cache
            exact_hit = _exact_cache_get(req.mode, req.top_k, req.question)
            if exact_hit is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                tracing.event("cache.exact_hit", question=req.question[:80])
                return AskResponse(
                    answer=exact_hit["answer"],
                    refused=exact_hit["refused"],
                    citations=[Citation(**c) for c in exact_hit["citations"]],
                    latency_ms=round(elapsed_ms, 2),
                    cost_usd=0.0,
                    cached=True,
                    trace_id=span_id,
                )

            # Check Semantic Cache
            sem_hit = _semantic_cache_get(req.mode, req.top_k, req.question)
            if sem_hit is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return AskResponse(
                    answer=sem_hit["answer"],
                    refused=sem_hit["refused"],
                    citations=[Citation(**c) for c in sem_hit["citations"]],
                    latency_ms=round(elapsed_ms, 2),
                    cost_usd=0.0,
                    cached=True,
                    trace_id=span_id,
                )

            # Execute Pipeline
            if req.mode == "tools":
                answer_text, refused, hits = pipe.answer_tools(req.question)
            else:
                answer_text, refused, hits = pipe.answer_rag(req.question, top_k=req.top_k)

            # Extract Citations
            citations: list[Citation] = []
            if not refused and hits:
                cited_idx = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer_text)})
                for idx in cited_idx:
                    if 1 <= idx <= len(hits):
                        hit = hits[idx - 1]
                        citations.append(Citation(index=idx, doc_id=hit.doc_id,
                                                  excerpt=hit.text[:400]))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            cost_usd = max(0.0, global_budget().spent_usd - b_before)

            resp_payload = {
                "answer": answer_text,
                "refused": refused,
                "citations": [c.dict() for c in citations],
                "latency_ms": round(elapsed_ms, 2),
                "cost_usd": round(cost_usd, 6),
                "cached": False,
                "trace_id": span_id,
            }

            # Update cache layers
            _exact_cache_put(req.mode, req.top_k, req.question, resp_payload)
            _semantic_cache_put(req.mode, req.top_k, req.question, resp_payload)

            return AskResponse(**resp_payload)

    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=f"spend budget exceeded: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc).lower()
        if (_is_retryable(exc) or "rate limit" in err_msg or "resource_exhausted" in err_msg
                or "unavailable" in err_msg or "503" in err_msg or "connection" in err_msg):
            raise HTTPException(
                status_code=503,
                detail=f"upstream model unavailable: {exc}",
                headers={"Retry-After": "30"},
            ) from exc
        raise HTTPException(status_code=500, detail=f"internal service error: {exc}") from exc


@app.get("/health")
def health() -> dict:
    """TODO C: index size, model profile, cache stats, uptime."""
    p = pipeline()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "corpus_docs": getattr(p, "n_docs", 0),
        "corpus_chunks": getattr(p, "n_chunks", 0),
        "model_profile": resolve_model("MAIN"),
        "cache": cache.stats(),
    }


@app.get("/metrics")
def metrics() -> dict:
    """TODO C2: cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    traces = tracing.read_traces()
    ask_spans = [s for s in traces if s.get("name") == "http.ask"]
    err_spans = [s for s in traces if s.get("status") == "error"]

    err_counts: dict[str, int] = {}
    for es in err_spans:
        ek = es.get("error_kind") or es.get("error", "Unknown").split(":")[0]
        err_counts[ek] = err_counts.get(ek, 0) + 1

    tool_calls = sum(1 for s in traces if s.get("name") in ("tools.agent", "guard.call"))
    n_queries = max(1, len(ask_spans))
    cached_queries = sum(1 for s in ask_spans if s.get("cached", False))

    latencies = sorted(s.get("duration_ms", 0.0) for s in ask_spans)
    p50 = latencies[int(len(latencies) * 0.5)] if latencies else b.percentile(50)
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else b.percentile(95)
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else b.percentile(99)

    return {
        "cost_today_usd": round(b.spent_usd, 6),
        "total_queries": len(ask_spans),
        "cost_per_query_usd": round(b.spent_usd / n_queries, 6),
        "cache_hit_rate": round(cached_queries / n_queries, 3) if ask_spans else 0.0,
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "latency_p99_ms": round(p99, 1),
        "error_count": len(err_spans),
        "errors_by_type": err_counts,
        "tool_call_counts": tool_calls,
    }


# ---------------------------------------------------------------------------
# POST /ask/stream (Part B2 & B3)
# ---------------------------------------------------------------------------
@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    """Stream answer tokens via SSE.

    B3 Defense (Strategy 3: Stream the prose, hold citations to the end):
    Tokens of the prose response stream immediately for low TTFT (<=1,500 ms).
    Because citation validity cannot be verified before generation finishes,
    citations and validation status are withheld during streaming and delivered
    in the final terminal SSE event ('done'). This guarantees fast human-perceived
    interactivity without compromising citation correctness.
    """
    pipe = pipeline()
    t0 = time.perf_counter()
    b_before = global_budget().spent_usd

    async def event_generator():
        with tracing.trace("http.ask.stream", question=req.question[:120]) as span:
            span_id = span.get("span_id", "trace-unknown")
            ttft_ms: float | None = None

            # Retrieve sources
            hits = pipe.retriever.search(req.question, k=req.top_k)
            context_str = delimit_untrusted(format_context(hits))
            prompt = f"{context_str}\n\nQuestion: {req.question}\n\nAnswer with citations:"

            from labs.lab4.rag import ANSWER_SYSTEM
            full_text = ""

            # Check exact cache first
            cached_resp = _exact_cache_get(req.mode, req.top_k, req.question)
            if cached_resp is not None:
                # Stream cached answer in small chunks
                words = cached_resp["answer"].split(" ")
                for i, w in enumerate(words):
                    chunk = w + (" " if i < len(words) - 1 else "")
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    yield {"event": "token", "data": json.dumps({"token": chunk})}
                    await asyncio.sleep(0.01)
                full_text = cached_resp["answer"]
            elif settings.offline:
                # In offline mode, use chat() which reads from cache
                res = chat(prompt, system=ANSWER_SYSTEM, tier="MAIN", temperature=0.0,
                           max_tokens=600, return_full=True)
                full_text = res["text"].strip() if isinstance(res, dict) else str(res).strip()
                words = full_text.split(" ")
                for i, w in enumerate(words):
                    chunk = w + (" " if i < len(words) - 1 else "")
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    yield {"event": "token", "data": json.dumps({"token": chunk})}
                    await asyncio.sleep(0.01)
            else:
                # Live online streaming via litellm
                import litellm
                resp = litellm.completion(
                    model=resolve_model("MAIN"),
                    messages=[{"role": "system", "content": ANSWER_SYSTEM},
                              {"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.0,
                    max_tokens=600,
                )
                for chunk in resp:
                    delta = getattr(chunk.choices[0].delta, "content", "") or ""
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        full_text += delta
                        yield {"event": "token", "data": json.dumps({"token": delta})}
                        await asyncio.sleep(0.001)

            # Post-generation filtering and citation validation (Part B3)
            clean_text = filter_output(full_text.strip())
            val = validate_answer(clean_text, len(hits), None)
            refused = val["refused"]

            citations: list[dict[str, Any]] = []
            if not refused and hits:
                cited_idx = sorted({int(m) for m in re.findall(r"\[(\d+)\]", clean_text)})
                for idx in cited_idx:
                    if 1 <= idx <= len(hits):
                        hit = hits[idx - 1]
                        citations.append({
                            "index": idx,
                            "doc_id": hit.doc_id,
                            "excerpt": hit.text[:400],
                        })

            elapsed_ms = (time.perf_counter() - t0) * 1000
            cost_usd = max(0.0, global_budget().spent_usd - b_before)

            # Terminal Done Event
            done_payload = {
                "answer": clean_text,
                "refused": refused,
                "citations": citations,
                "ttft_ms": round(ttft_ms or elapsed_ms, 1),
                "latency_ms": round(elapsed_ms, 1),
                "cost_usd": round(cost_usd, 6),
                "valid": val["valid"],
                "trace_id": span_id,
            }
            yield {"event": "done", "data": json.dumps(done_payload)}

    return EventSourceResponse(event_generator())
