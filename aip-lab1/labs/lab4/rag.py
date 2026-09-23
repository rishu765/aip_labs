#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

ANSWER_SYSTEM = f"""\
You are an insurance information assistant. Answer the user's question using ONLY the numbered sources provided below.

Strict Rules:
1. Answer ONLY from the provided sources. Do not use any outside knowledge, assumptions, or general training data.
2. If the sources do not contain the answer, reply with this exact refusal string and nothing else:
2. If the sources answer part of the question but not all of it, answer the supported part with citations and explicitly state that the remaining information is not available in the provided sources.
3. If the sources contain NO relevant information to answer the question, reply with this exact refusal string and nothing else:
   "{REFUSAL}"
3. If the sources answer part of the question but not all of it, answer the supported part with citations and explicitly state what information is missing.
4. Every factual claim or sentence must end with an inline citation referring to the source number, formatted as [1], [2], or [1][3].
5. Never cite a source number that was not provided in the context.
6. If different sources disagree or contradict each other, explicitly describe the contradiction and cite all conflicting sources.
7. Be concise, direct, and factual. Limit answers to two or three sentences unless a detailed breakdown is strictly necessary.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """Validate that the answer satisfies citation, truncation, and refusal rules.

    Returns a dict with:
        {"valid": bool, "refused": bool, "invalid_citations": [ints],
         "n_citations": int, "truncated": bool, "reason": str}
    """
    cleaned = text.strip()
    truncated = (finish_reason == "length")
    refused = (cleaned == REFUSAL or REFUSAL in cleaned or cleaned.startswith(REFUSAL[:40]))

    citations = [int(m) for m in re.findall(r"\[(\d+)\]", cleaned)]
    n_citations = len(citations)
    invalid_citations = sorted({c for c in citations if c < 1 or c > n_sources})

    if not cleaned:
        return {
            "valid": False,
            "refused": False,
            "invalid_citations": [],
            "n_citations": 0,
            "truncated": False,
            "reason": "empty_answer",
        }

    if truncated:
        return {
            "valid": False,
            "refused": refused,
            "invalid_citations": invalid_citations,
            "n_citations": n_citations,
            "truncated": True,
            "reason": "truncated_by_length",
        }

    if refused:
        return {
            "valid": True,
            "refused": True,
            "invalid_citations": [],
            "n_citations": 0,
            "truncated": False,
            "reason": "refusal",
        }

    if invalid_citations:
        return {
            "valid": False,
            "refused": False,
            "invalid_citations": invalid_citations,
            "n_citations": n_citations,
            "truncated": False,
            "reason": f"invalid_citations: {invalid_citations}",
        }

    if n_citations == 0:
        return {
            "valid": False,
            "refused": False,
            "invalid_citations": [],
            "n_citations": 0,
            "truncated": False,
            "reason": "missing_citations",
        }

    return {
        "valid": True,
        "refused": False,
        "invalid_citations": [],
        "n_citations": n_citations,
        "truncated": False,
        "reason": "valid",
    }


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    """Full RAG pipeline: retrieve -> (rerank) -> generate -> validate -> maybe repair.

    Failure policy (B3 defense):
    If validation fails (e.g. invalid citation index or missing citations), we
    first attempt a single corrective repair prompt. If validation fails again,
    we fall back to safe refusal (REFUSAL). In an enterprise/insurance domain,
    declining to answer is always preferred over returning an unverified or
    hallucinated citation.
    """
    hits = retriever.search(question, k=k)
    if reranker is not None:
        final_hits = reranker.rerank(question, hits, k=final_k)
    else:
        final_hits = list(hits)[:final_k]

    if not final_hits:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=[],
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=False,
        )

    context_str = delimit_untrusted(format_context(final_hits))
    prompt = f"{context_str}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip() if isinstance(res, dict) else str(res).strip()
    finish_reason = res.get("finish_reason") if isinstance(res, dict) else None

    val = validate_answer(text, len(final_hits), finish_reason)

    if not val["valid"]:
        # Attempt 1 targeted repair
        repair_prompt = (
            f"{prompt}\n\nYour previous response was:\n{text}\n\n"
            f"It failed validation because: {val['reason']}.\n"
            f"Please rewrite the answer strictly following the rules: cite only valid sources "
            f"[1] to [{len(final_hits)}]. If the sources do not contain the answer, reply exactly:\n"
            f"{REFUSAL}"
        )
        res_rep = chat(repair_prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
        text_rep = res_rep["text"].strip() if isinstance(res_rep, dict) else str(res_rep).strip()
        finish_rep = res_rep.get("finish_reason") if isinstance(res_rep, dict) else None
        val_rep = validate_answer(text_rep, len(final_hits), finish_rep)

        if val_rep["valid"]:
            text, val = text_rep, val_rep
        else:
            # Fall back to safe refusal
            text = REFUSAL
            val = validate_answer(text, len(final_hits), None)

    return Answer(
        question=question,
        text=text,
        hits=list(final_hits),
        refused=val["refused"],
        citations_valid=val["valid"] or val["refused"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """Generate using the gold documents directly (no retrieval stage).

    The difference in correctness between this and answer_question() isolates
    the performance penalty attributable to retrieval vs. generation.
    """
    from aip.chunking import Chunk

    if not gold_docs:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=[],
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=False,
        )

    pseudo_hits = [
        Hit(Chunk(text=doc_text, doc_id=f"gold_{i}", chunk_id=f"gold_{i}"), score=1.0, rank=i - 1)
        for i, doc_text in enumerate(gold_docs, 1)
    ]

    context_str = delimit_untrusted(format_context(pseudo_hits))
    prompt = f"{context_str}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip() if isinstance(res, dict) else str(res).strip()
    finish_reason = res.get("finish_reason") if isinstance(res, dict) else None

    val = validate_answer(text, len(pseudo_hits), finish_reason)

    if not val["valid"]:
        repair_prompt = (
            f"{prompt}\n\nYour previous response was:\n{text}\n\n"
            f"It failed validation because: {val['reason']}.\n"
            f"Please rewrite the answer strictly following the rules: cite only valid sources "
            f"[1] to [{len(pseudo_hits)}]. If the sources do not contain the answer, reply exactly:\n"
            f"{REFUSAL}"
        )
        res_rep = chat(repair_prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
        text_rep = res_rep["text"].strip() if isinstance(res_rep, dict) else str(res_rep).strip()
        finish_rep = res_rep.get("finish_reason") if isinstance(res_rep, dict) else None
        val_rep = validate_answer(text_rep, len(pseudo_hits), finish_rep)

        if val_rep["valid"]:
            text, val = text_rep, val_rep
        else:
            text = REFUSAL
            val = validate_answer(text, len(pseudo_hits), None)

    return Answer(
        question=question,
        text=text,
        hits=pseudo_hits,
        refused=val["refused"],
        citations_valid=val["valid"] or val["refused"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )
