#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import structured
from labs.lab1.extract import (
    SYSTEM_PROMPT,
    TicketRecordC,
    _fallback_c,
    apply_business_rules,
    extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# Chosen dev-set tickets covering edge cases (T2 §2.2):
#   - T0097: Billing/complaint boundary — angry ombudsman threat over double debit is billing, not complaint.
#   - T0054: Absent policy number — teaches null when no policy appears; Aurora mis-selling is complaint.
#   - T0200: Hinglish code-mixing — transliterated Hindi ('Koi solution batayiye') requires language: 'hi-en'.
#   - T0029: Satisfied-with-query trap — polite thank-you with follow-up is information (urgency 1, satisfied).
#   - T0238: Quoted history & non-AUR ref — email reply reference 'SR-100238' is not an AUR- policy number (null).
#   - T0010: High-urgency claim denial — formal claim rejection with same-day request ('today') is urgency 5.
FEW_SHOT_IDS: list[str] = [
    "T0097",  # teaches: billing vs complaint boundary (double debit/refund dispute is billing despite threat)
    "T0054",  # teaches: explicit null for missing policy_number, and mis-selling conduct is complaint
    "T0200",  # teaches: transliterated Hindi ('Koi solution batayiye') and claim deduction inquiry
    "T0029",  # teaches: satisfied tone paired with routine follow-up inquiry belongs to information at urgency 1
    "T0238",  # teaches: non-AUR reference codes (SR-*) and quoted reply history must be ignored (policy is null)
    "T0010",  # teaches: formal claim rejection demanding immediate action ('today') sets urgency to 5 under claims
]

FEW_SHOT_DATA: dict[str, dict] = {
    "T0097": {
        "reasoning": (
            "Customer repeatedly demands refund for double debit and threatens ombudsman. "
            "Financial transactions and refund disputes are categorized as billing. "
            "Urgency is 4 due to repeated failure over 45 days. Tone is angry."
        ),
        "evidence": "THIS IS THE THIRD TIME I am writing about teh double debit on AUR-7548999. Rs 8750 taken twice",
        "category": "billing",
        "urgency": 4,
        "sentiment": "angry",
    },
    "T0054": {
        "reasoning": (
            "Customer complains about agent mis-selling regarding maternity waiting period and demands full refund. "
            "Aurora conduct and mis-selling is classified as complaint. "
            "Urgency is 4 due to escalation. Tone is angry. No policy number is given."
        ),
        "evidence": "Your agent mis-sold me this policy. Nobody told me maternity has a 5-month waiting period",
        "category": "complaint",
        "urgency": 4,
        "sentiment": "angry",
    },
    "T0200": {
        "reasoning": (
            "Customer inquires about proportionate deduction on a settled hospital bill claim. "
            "Claim deductions and settlements belong to claims. "
            "Urgency is 3 (unexplained deduction awaiting resolution). Tone is frustrated."
        ),
        "evidence": "My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher. Nobody explained the deduction.",
        "category": "claims",
        "urgency": 3,
        "sentiment": "frustrated",
    },
    "T0029": {
        "reasoning": (
            "Customer thanks Aurora for quick settlement and asks a general question about no-claim bonus impact. "
            "With no active claim pending, this is an information query. "
            "Urgency is 1 (general knowledge). Tone is satisfied."
        ),
        "evidence": "Just wanted to confirm whether my no-claim bonus is affected by this claim.",
        "category": "information",
        "urgency": 1,
        "sentiment": "satisfied",
    },
    "T0238": {
        "reasoning": (
            "Customer submitted a portability request 21 days ago with no response. "
            "Porting or updating policy terms is policy_change. "
            "Urgency is 3 (request stuck in flight). Tone is frustrated."
        ),
        "evidence": "I submitted a portability request 21 days ago and heard nothing.",
        "category": "policy_change",
        "urgency": 3,
        "sentiment": "frustrated",
    },
    "T0010": {
        "reasoning": (
            "Customer received a formal claim rejection for a 14-day hospitalisation and demands the rejection clause today. "
            "Claim rejection belongs to claims. "
            "Urgency is 5 (formal denial needing immediate same-day response). Tone is angry."
        ),
        "evidence": "You have REJECTED my claim on AUR-5021488 without giving any clause.",
        "category": "claims",
        "urgency": 5,
        "sentiment": "angry",
    },
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str], reasoned: bool = False) -> str:
    """A2: render the examples into the prompt.

    Examples contain every required judgement field of the requested schema.
    """
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        tid = ex["id"]
        meta = FEW_SHOT_DATA.get(tid, {})
        evidence = meta.get("evidence", "")
        if evidence not in ex["input"]:
            raise ValueError(f"example {tid} evidence is not verbatim")
        rec = {
            "evidence": evidence,
            "category": ex["expected"]["category"],
            "urgency": ex["expected"]["urgency"],
            "sentiment": ex["expected"]["sentiment"],
        }
        if reasoned:
            rec = {"reasoning": meta.get("reasoning", ""), **rec}
        rec_json = json.dumps(rec, indent=2)
        blocks.append(f"Ticket:\n{ex['input']}\n\nRecord:\n{rec_json}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """B: Lab 1 Part C baseline (no few-shot examples)."""
    deterministic = extract_deterministic(ticket)
    try:
        model_rec = structured(
            f"Ticket:\n{ticket}",
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier=tier,
            temperature=0.0,
            max_tokens=700,
        ).model_dump()
    except Exception as exc:  # noqa: BLE001 - a failed provider call becomes a reviewable fallback
        model_rec = _fallback_c(ticket, str(exc))
        model_rec["_used_fallback"] = True

    return apply_business_rules({**model_rec, **deterministic}, ticket)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """B: zero_shot + the 6-example few-shot block."""
    deterministic = extract_deterministic(ticket)
    examples_text = few_shot_block(FEW_SHOT_IDS, reasoned=False)
    prompt = (
        f"Reference examples:\n\n{examples_text}\n\n"
        f"---\n\nNow extract the target ticket:\nTicket:\n{ticket}"
    )
    try:
        model_rec = structured(
            prompt,
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier=tier,
            temperature=0.0,
            max_tokens=700,
        ).model_dump()
    except Exception as exc:  # noqa: BLE001 - a failed provider call becomes a reviewable fallback
        model_rec = _fallback_c(ticket, str(exc))
        model_rec["_used_fallback"] = True

    return apply_business_rules({**model_rec, **deterministic}, ticket)


class TicketRecordReasoned(TicketRecordC):
    """B: add a `reasoning: str` field FIRST (T2 §3.3).

    Field order in the JSON Schema influences generation order. Putting
    reasoning first makes it condition the answer rather than serve as a post-hoc
    rationalisation.
    """
    reasoning: str = Field(
        max_length=400,
        description="Brief step-by-step reasoning explaining category, urgency, and sentiment.",
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        schema = super().model_json_schema(*args, **kwargs)
        props = schema.get("properties", {})
        if "reasoning" in props:
            reordered = {"reasoning": props["reasoning"]}
            for k, v in props.items():
                if k != "reasoning":
                    reordered[k] = v
            schema["properties"] = reordered
        return schema


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """B: few_shot with TicketRecordReasoned."""
    deterministic = extract_deterministic(ticket)
    examples_text = few_shot_block(FEW_SHOT_IDS, reasoned=True)
    prompt = (
        f"Reference examples:\n\n{examples_text}\n\n"
        f"---\n\nNow extract the target ticket:\nTicket:\n{ticket}"
    )
    try:
        model_rec = structured(
            prompt,
            schema=TicketRecordReasoned,
            system=SYSTEM_PROMPT,
            tier=tier,
            temperature=0.0,
            max_tokens=1000,
        ).model_dump()
    except Exception as exc:  # noqa: BLE001 - a failed provider call becomes a reviewable fallback
        model_rec = _fallback_c(ticket, str(exc))
        model_rec["_used_fallback"] = True

    return apply_business_rules({**model_rec, **deterministic}, ticket)


def cascade(ticket: str) -> dict:
    """C: SMALL first; escalate to MAIN on disagreement, missing evidence, or validation error.

    Avoids the temperature=0 cache trap by sampling a second candidate at T=0.7.
    Sets rec['_path'] = 'small' | 'large'.
    """
    rec_small = zero_shot(ticket, tier="SMALL")

    should_escalate = False
    agreement = None
    # Trigger 1: schema fallback or missing evidence
    if rec_small.get("needs_human_review") or not rec_small.get("evidence") or len(str(rec_small.get("evidence", "")).strip()) < 5:
        should_escalate = True
    else:
        # Trigger 2: check disagreement against a second sample drawn at T=0.7
        try:
            sample2 = structured(
                f"Ticket:\n{ticket}",
                schema=TicketRecordC,
                system=SYSTEM_PROMPT,
                tier="SMALL",
                temperature=0.7,
                max_tokens=700,
            ).model_dump()

            agreement = (sample2.get("category") == rec_small.get("category")
                         and sample2.get("urgency") == rec_small.get("urgency")
                         and sample2.get("sentiment") == rec_small.get("sentiment"))
            should_escalate = not agreement
        except Exception:  # noqa: BLE001 - disagreement probe failure requires escalation
            should_escalate = True

    if should_escalate:
        rec_main = zero_shot(ticket, tier="MAIN")
        rec_main["_path"] = "large"
        rec_main["_small_agreement"] = agreement
        rec_main["_small_record"] = rec_small
        return rec_main
    else:
        rec_small["_path"] = "small"
        rec_small["_small_agreement"] = agreement
        rec_small["_small_record"] = {k: v for k, v in rec_small.items() if not k.startswith("_")}
        return rec_small


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
