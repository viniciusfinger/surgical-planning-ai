"""
Custom Guardrails AI validators for the surgical planning domain.
"""

from __future__ import annotations

from typing import Any, Callable

from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)

from safety.patterns import (
    MAX_FREE_TEXT_LENGTH,
    MIN_FREE_TEXT_LENGTH,
    find_off_topic_terms,
    find_pii,
    find_prompt_injection,
    find_slurs,
)


@register_validator(name="surgical/no-prompt-injection", data_type="string")
class NoPromptInjection(Validator):
    """Reject text that matches known prompt-injection / jailbreak heuristics."""

    def __init__(self, on_fail: Callable | str | None = None) -> None:
        super().__init__(on_fail=on_fail)

    def _validate(self, value: Any, metadata: dict) -> ValidationResult:
        if not isinstance(value, str):
            return PassResult()
        matches = find_prompt_injection(value)
        if matches:
            return FailResult(
                error_message="prompt_injection_detected",
                metadata={"patterns": matches},
            )
        return PassResult()


@register_validator(name="surgical/clinical-scope-only", data_type="string")
class ClinicalScopeOnly(Validator):
    """
    Reject text that strays outside the perioperative clinical scope: explicit
    sexual content, slurs, weapons/violence, illegal drugs, etc.
    """

    def __init__(self, on_fail: Callable | str | None = None) -> None:
        super().__init__(on_fail=on_fail)

    def _validate(self, value: Any, metadata: dict) -> ValidationResult:
        if not isinstance(value, str):
            return PassResult()

        off_topic = find_off_topic_terms(value)
        slurs = find_slurs(value)
        triggered: list[str] = []
        if off_topic:
            triggered.append("off_topic")
        if slurs:
            triggered.append("slur")

        if triggered:
            return FailResult(
                error_message="out_of_clinical_scope",
                metadata={
                    "categories": triggered,
                    "off_topic_terms": off_topic,
                    "slur_terms": slurs,
                },
            )
        return PassResult()


@register_validator(name="surgical/no-pii", data_type="string")
class NoPII(Validator):
    """
    Reject text containing personal identifiers (email, CPF, RG, phone, credit
    card, URL). Surgical planning never needs these and storing them risks
    LGPD / HIPAA non-compliance.
    """

    def __init__(self, on_fail: Callable | str | None = None) -> None:
        super().__init__(on_fail=on_fail)

    def _validate(self, value: Any, metadata: dict) -> ValidationResult:
        if not isinstance(value, str):
            return PassResult()
        categories = find_pii(value)
        if categories:
            return FailResult(
                error_message="pii_detected",
                metadata={"categories": categories},
            )
        return PassResult()


@register_validator(name="surgical/length-bounds", data_type="string")
class LengthBounds(Validator):
    """Reject empty / overly long free-text values reaching the prompt."""

    def __init__(
        self,
        min_length: int = MIN_FREE_TEXT_LENGTH,
        max_length: int = MAX_FREE_TEXT_LENGTH,
        on_fail: Callable | str | None = None,
    ) -> None:
        super().__init__(
            on_fail=on_fail,
            min_length=min_length,
            max_length=max_length,
        )
        self.min_length = min_length
        self.max_length = max_length

    def _validate(self, value: Any, metadata: dict) -> ValidationResult:
        if not isinstance(value, str):
            return PassResult()
        stripped = value.strip()
        if len(stripped) < self.min_length:
            return FailResult(
                error_message="text_too_short",
                metadata={"length": len(stripped), "min": self.min_length},
            )
        if len(stripped) > self.max_length:
            return FailResult(
                error_message="text_too_long",
                metadata={"length": len(stripped), "max": self.max_length},
            )
        return PassResult()
