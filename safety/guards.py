from __future__ import annotations

from functools import lru_cache
from typing import Any

from guardrails import Guard

from safety.exceptions import GuardrailViolation
from safety.validators import (
    ClinicalScopeOnly,
    LengthBounds,
    NoPII,
    NoPromptInjection,
)


@lru_cache(maxsize=1)
def _input_guard() -> Guard:
    return Guard().use(
        LengthBounds(on_fail="noop"),
        NoPromptInjection(on_fail="noop"),
        ClinicalScopeOnly(on_fail="noop"),
        NoPII(on_fail="noop"),
    )


@lru_cache(maxsize=1)
def _output_guard() -> Guard:
    return Guard().use(
        ClinicalScopeOnly(on_fail="noop"),
        NoPII(on_fail="noop"),
    )


def _extract_failures(outcome: Any) -> list[dict[str, Any]]:
    """
    Normalize a Guardrails ValidationOutcome into a flat list of failures.
    """
    failures: list[dict[str, Any]] = []
    summaries = getattr(outcome, "validation_summaries", None) or []
    for summary in summaries:
        validator_status = (
            getattr(summary, "validator_status", None)
            or getattr(summary, "outcome", None)
            or ""
        )
        if str(validator_status).lower() not in {"fail", "failed", "failure"}:
            continue
        failures.append(
            {
                "validator": getattr(summary, "validator_name", None)
                or getattr(summary, "name", None)
                or "unknown",
                "error_message": getattr(summary, "error_message", None)
                or getattr(summary, "failure_reason", None)
                or "validation_failed",
                "metadata": getattr(summary, "metadata", None) or {},
            }
        )

    if not failures and getattr(outcome, "validation_passed", True) is False:
        failures.append(
            {
                "validator": "unknown",
                "error_message": "validation_failed",
                "metadata": {},
            }
        )

    return failures


def validate_clinical_text_input(value: str, field: str) -> str:
    """
    Validate a free-text field that will be passed verbatim into an LLM prompt.
    """
    
    cleaned = (value or "").strip()
    outcome = _input_guard().validate(cleaned)
    if getattr(outcome, "validation_passed", True):
        return cleaned

    failures = _extract_failures(outcome)
    primary = failures[0] if failures else {"error_message": "validation_failed"}
    raise GuardrailViolation(
        rule=primary.get("error_message", "validation_failed"),
        stage="input",
        field=field,
        detail=f"{len(failures)} validator(s) rejected field '{field}'",
        metadata={"failures": failures},
    )


def validate_llm_text_output(value: str, field: str) -> str:
    """Validate a free-text field produced by an LLM before exposing it."""

    if not value:
        return value
    outcome = _output_guard().validate(value)
    if getattr(outcome, "validation_passed", True):
        return value

    failures = _extract_failures(outcome)
    primary = failures[0] if failures else {"error_message": "validation_failed"}
    raise GuardrailViolation(
        rule=primary.get("error_message", "validation_failed"),
        stage="output",
        field=field,
        detail=f"{len(failures)} validator(s) rejected output field '{field}'",
        metadata={"failures": failures},
    )
