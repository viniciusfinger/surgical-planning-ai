"""
Centralized safety patterns.

Kept as plain regex / wordlists so:
- They run with no network calls (low latency, deterministic)
- They are auditable and version-controlled
- They can be unit-tested independently from the LLM

These are a deny-list reinforcement, NOT a primary defense. The primary defense
remains the structured Pydantic schema (typed fields, enums, ranges).
"""

from __future__ import annotations

import re

PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bignore\b[^.\n]{0,60}\b(instructions?|rules?|prompts?|guidelines?|directives?)\b",
        r"\bdisregard\b[^.\n]{0,60}\b(instructions?|rules?|prompts?|guidelines?|directives?)\b",
        r"\bforget\s+(everything|all|previous|prior|the\s+rules)\b",
        r"\b(reveal|show|print|output|leak|expose)\s+(your|the)?\s*(system\s*)?(prompt|instructions?|rules?|guidelines?)\b",
        r"\byou\s+are\s+now\s+(?!a\s+patient\b)",
        r"\bact\s+as\s+(?:a\s+)?(?!patient\b)",
        r"\bpretend\s+(to\s+be|you\s+are)\b",
        r"\brole[\s-]*play\s+as\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"\bdeveloper\s+mode\b",
        r"\bsudo\s+mode\b",
        r"<\s*/?\s*(system|assistant|user|tool|instruction)s?\s*>",
        r"\[\s*(system|assistant|instruction)s?\s*\]",
        r"\bnew\s+(instructions?|rules?|task)\s*[:=]",
        r"\boverride\b[^.\n]{0,40}\b(instructions?|rules?|prompts?)\b",
        r"\bend\s+of\s+(prompt|instructions?|system\s*message)\b",
        r"\b(execute|run|eval)\s+(this\s+)?code\b",
    )
)

OFF_TOPIC_TERMS: tuple[str, ...] = (
    "porn",
    "pornography",
    "nude",
    "nudes",
    "naked",
    "nudity",
    "sexual",
    "sex tape",
    "erotic",
    "fetish",
    "horny",
    "explicit content",
    "rape",
    "incest",
    "pedophil",
    "child porn",
    "weapon",
    "firearm",
    "bomb",
    "explosive",
    "improvised explosive",
    "terrorist",
    "kill yourself",
    "suicide method",
    "self-harm method",
    "buy drugs",
    "illegal drugs",
    "make meth",
    "synthesize cocaine",
    "pelada",
    "pelado",
    "putaria",
    "transar",
)

SLUR_TERMS: tuple[str, ...] = (
    "nigger",
    "nigga",
    "faggot",
    "tranny",
    "retard",
    "retarded",
    "kike",
    "spic",
    "chink",
    "raghead",
    "viado",
    "bicha",
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ),
    (
        "cpf",
        re.compile(r"\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[-]?\d{2}\b"),
    ),
    (
        "rg",
        re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-?[\dXx]\b"),
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    ),
    (
        "phone_br",
        re.compile(r"\b(?:\+?55\s*)?\(?\d{2}\)?[\s-]?9?\d{4}[-\s]?\d{4}\b"),
    ),
    (
        "url",
        re.compile(r"https?://\S+", re.IGNORECASE),
    ),
)

MAX_FREE_TEXT_LENGTH = 200
"""Hard upper bound for any free-text field reaching an LLM prompt."""

MIN_FREE_TEXT_LENGTH = 2
"""Reject empty / single-character noise."""


def find_prompt_injection(value: str) -> list[str]:
    """Return the list of triggered injection patterns (empty if clean)."""
    matches: list[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(value):
            matches.append(pattern.pattern)
    return matches


def find_off_topic_terms(value: str) -> list[str]:
    lower = value.lower()
    return [term for term in OFF_TOPIC_TERMS if term in lower]


def find_slurs(value: str) -> list[str]:
    lower = value.lower()
    return [term for term in SLUR_TERMS if term in lower]


def find_pii(value: str) -> list[str]:
    """Return the list of PII categories detected in `value`."""
    return [name for name, pattern in PII_PATTERNS if pattern.search(value)]
