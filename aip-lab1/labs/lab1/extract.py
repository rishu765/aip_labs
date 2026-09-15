#!/usr/bin/env python3
"""Lab 1, Parts B and C - the extractor you actually ship."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal[
    "billing",
    "claims",
    "policy_change",
    "technical",
    "complaint",
    "information",
]

POLICY_RE = re.compile(r"\bAUR-\d{7}\b")
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)
OWN_EMAILS = {"support@aurorahealth.example", "grievance@aurorahealth.example"}

CATEGORY_DESCRIPTION = (
    "Choose one: billing=payments, premiums, debits, refunds, invoices, tax "
    "certificate; claims=actual or intended cashless/reimbursement/settlement/"
    "deduction/rejection; policy_change=add/remove member, upgrade, port, "
    "contact change; technical=app, portal, OTP, login, locator, upload broken; "
    "complaint=Aurora conduct such as mis-selling, hold time, ignored grievance; "
    "information=general question with no pending transaction. Angry claim "
    "requests are claims, not complaint."
)

URGENCY_DESCRIPTION = (
    "1=general knowledge/self-service, no account lookup. 2=look up this "
    "customer, act, fix a defect, or transaction in flight. 3=something already "
    "went wrong or is stuck and customer waits. 4=repeated failure, money/access "
    "at risk now, or explicit escalation threat. 5=emergency, formal denial "
    "needing immediate reversal, or customer is filing with Ombudsman. Add 1, "
    "capped at 5, for same-day/next-morning deadline."
)

SENTIMENT_DESCRIPTION = (
    "Tone only: angry=hostile, shouting, threatening; frustrated=unhappy after "
    "prior failure/delay/repeat attempt, still civil; neutral=matter-of-fact "
    "first-time request; satisfied=thanks or praise."
)

PRODUCT_DESCRIPTION = (
    "Return bronze, silver, gold, or platinum only when the plan is named in "
    "the ticket. Never infer from sum insured or context; use unknown when no "
    "plan name appears."
)

LANGUAGE_DESCRIPTION = (
    "Return hi-en when Hindi words are mixed into English, including "
    "transliterated Hindi such as kripya, jaldi, bahut, turant; otherwise en."
)


class TicketRecord(BaseModel):
    """Part B schema: the model decides all graded fields except escalation."""

    # Evidence comes first so the model grounds the judgement before choosing
    # the category that the evidence supports.
    evidence: str = Field(
        max_length=200,
        description="Quote the shortest verbatim span that determines category.",
    )
    category: CATEGORIES = Field(description=CATEGORY_DESCRIPTION)
    urgency: int = Field(ge=1, le=5, description=URGENCY_DESCRIPTION)
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description=SENTIMENT_DESCRIPTION
    )
    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description=PRODUCT_DESCRIPTION
    )
    language: Literal["en", "hi-en"] = Field(description=LANGUAGE_DESCRIPTION)
    policy_number: str | None = Field(
        default=None,
        pattern=r"^AUR-\d{7}$",
        description=(
            "Copy an exact policy number only if it appears as AUR- followed by "
            "7 digits. Return null when absent. Never invent, reformat, or copy "
            "from quoted reply history."
        ),
    )
    contains_pii: bool = Field(
        default=False,
        description=(
            "True if the ticket contains a phone number or non-Aurora email "
            "address. Names alone do not count; support@aurorahealth.example and "
            "grievance@aurorahealth.example do not count."
        ),
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        # Empty strings and literal "null" are treated as absent; anything else
        # malformed is rejected so the repair loop can fix it.
        if v is None:
            return None
        if v.strip().lower() in {"", "null", "none"}:
            return None
        if not POLICY_RE.fullmatch(v):
            raise ValueError("policy_number must be exactly AUR- followed by 7 digits")
        return v


SYSTEM_PROMPT = """\
Role: You are Aurora Health's support-ticket extraction service.
Task: Convert one raw support ticket into the requested structured record.
Audience: The record is used by routing, compliance review, and evaluation.
Source policy: Use only text present in the ticket; do not infer hidden facts.
Decision policy: Follow the JSON Schema field descriptions exactly.
Reliability: If uncertain, choose the closest valid value and brief evidence.
Output: Return only the JSON object requested by the schema.
"""


def _fallback_b(reason: str) -> TicketRecord:
    heuristic = _heuristic_judgements("")
    return TicketRecord(
        evidence="",
        category=heuristic["category"],
        urgency=heuristic["urgency"],
        sentiment=heuristic["sentiment"],
        product="unknown",
        language="en",
        policy_number=None,
        contains_pii=False,
        needs_human_review=True,
        review_reason=reason[:200],
    )


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(
            f"Ticket:\n{ticket}",
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
            tier="SMALL",
            temperature=0.0,
            max_tokens=900,
        )
    except (StructuredOutputError, Exception) as exc:
        return _fallback_b(str(exc))


def _live_message(ticket: str) -> str:
    live = QUOTE_MARKER.split(ticket, maxsplit=1)[0]
    return re.split(
        r"(?im)^\s*(?:--|thanks(?:\s*&\s*regards)?|regards|yours sincerely),?\s*$",
        live,
        maxsplit=1,
    )[0]


def _plain_live_message(ticket: str) -> str:
    text = html.unescape(_live_message(ticket))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_product(ticket: str) -> str:
    text = ticket.lower()
    for product in ("bronze", "silver", "gold", "platinum"):
        if re.search(rf"\b(?:aurora\s+)?{product}\b", text):
            return product
    return "unknown"


def _detect_language(ticket: str) -> str:
    hindi_words = (
        "kripya", "jaldi", "bahut", "turant", "koi", "batayiye", "nahi",
        "hai", "ho", "mera", "meri", "please karo", "kar do",
    )
    text = ticket.lower()
    return "hi-en" if any(re.search(rf"\b{re.escape(w)}\b", text) for w in hindi_words) else "en"


def _heuristic_judgements(ticket: str) -> dict:
    live = _plain_live_message(ticket)
    text = live.lower()

    if re.search(r"\b(mis-?sold|on hold|ignored grievance|service is a disgrace)\b", text):
        category = "complaint"
    elif re.search(r"\b(claim|cashless|reimbursement|settled|settlement|deduction|denied|hospital bill)\b", text):
        category = "claims"
    elif re.search(r"\b(premium|debit|debited|refund|invoice|80d|tax certificate|instalment|payment)\b", text):
        category = "billing"
    elif re.search(r"\b(add|remove|newborn|upgrade|portability|port\b|change.*(?:address|mobile|email|contact))\b", text):
        category = "policy_change"
    elif re.search(r"\b(app|portal|otp|login|upload|locator|crash|crashes|down|broken)\b", text):
        category = "technical"
    else:
        category = "information"

    urgency = 1
    if re.search(r"\b(icu|emergency|cashless is denied|cashless denied|am filing.*ombudsman|filing a complaint with the ombudsman)\b", text):
        urgency = 5
    elif re.search(r"\b(third time|repeated|standing at the hospital|at the hospital desk|going to.*ombudsman|escalat|over an hour|twice today)\b", text):
        urgency = 4
    elif re.search(r"\b(heard nothing|debited twice|settled.*but|stuck|no response|not received|submitted .* days ago|delayed|nobody explained)\b", text):
        urgency = 3
    elif (
        POLICY_RE.search(live)
        or category in {"policy_change", "technical"}
        or re.search(r"\b(my|me|currently have|please add|please change|refund|deduction|claim status)\b", text)
    ):
        urgency = 2

    if urgency < 5 and re.search(r"\b(today|same day|tonight|tomorrow|next morning|by morning)\b", text):
        urgency += 1

    if re.search(r"\b(disgrace|unacceptable|ombudsman|mis-?sold|full refund|third time)\b", text) or re.search(r"[A-Z]{4,}", live):
        sentiment = "angry"
    elif re.search(r"\b(again|twice|heard nothing|still|stuck|delayed|nobody|not working|crashes|failed|failure)\b", text):
        sentiment = "frustrated"
    elif re.search(r"\b(thank you|thanks for|appreciate|great service|helpful)\b", text):
        sentiment = "satisfied"
    else:
        sentiment = "neutral"

    return {
        "evidence": live[:200],
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
    }


def extract_deterministic(ticket: str) -> dict:
    """Return deterministic fields without a model call."""
    # General rule: choose the first policy in the live customer message only.
    # Quoted lines are prior history, and signature tails are identity/contact
    # material rather than the request being routed.
    match = POLICY_RE.search(_live_message(ticket))

    emails = [
        m.group(0).lower()
        for m in _PII_PATTERNS["EMAIL"].finditer(ticket)
        if m.group(0).lower() not in OWN_EMAILS
    ]
    phones = list(_PII_PATTERNS["PHONE_IN"].finditer(ticket))

    return {
        "policy_number": match.group(0) if match else None,
        "contains_pii": bool(emails or phones),
        "product": _detect_product(ticket),
        "language": _detect_language(ticket),
    }


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute auditable routing business rules."""
    out = dict(rec_fields)
    out["escalate"] = int(out.get("urgency", 1)) >= 4 or "ombudsman" in ticket.lower()
    return out


class TicketRecordC(BaseModel):
    """Part C schema: the model only handles judgement fields."""

    # Evidence still comes first so category is selected after grounding.
    evidence: str = Field(
        max_length=200,
        description="Quote the shortest verbatim span that determines category.",
    )
    category: CATEGORIES = Field(description=CATEGORY_DESCRIPTION)
    urgency: int = Field(ge=1, le=5, description=URGENCY_DESCRIPTION)
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description=SENTIMENT_DESCRIPTION
    )
    needs_human_review: bool = False
    review_reason: str = ""


def _fallback_c(ticket: str, reason: str) -> dict:
    return {
        **_heuristic_judgements(ticket),
        "needs_human_review": True,
        "review_reason": reason[:200],
    }


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for deterministic fields and rules."""
    deterministic = extract_deterministic(ticket)
    try:
        model_rec = structured(
            f"Ticket:\n{ticket}",
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier="SMALL",
            temperature=0.0,
            max_tokens=700,
        ).model_dump()
    except (StructuredOutputError, Exception) as exc:
        model_rec = _fallback_c(ticket, str(exc))

    return apply_business_rules({**model_rec, **deterministic}, ticket)


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
