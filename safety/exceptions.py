from typing import Any


class GuardrailViolation(Exception):
    """
    Raised when a safety guardrail rejects an input or output.

    Carries enough structured detail for the API layer to translate it into a
    safe, non-leaky HTTP error while keeping a precise audit trail in logs.
    Never include raw patient text or full LLM outputs in `detail`.
    """

    def __init__(
        self,
        rule: str,
        stage: str,
        field: str | None = None,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.rule = rule
        self.stage = stage
        self.field = field
        self.detail = detail
        self.metadata = metadata or {}
        super().__init__(self.public_message())

    def public_message(self) -> str:
        """User-facing message; intentionally generic to avoid leaking heuristics."""
        if self.stage == "input":
            return (
                "The submitted clinical data did not pass safety validation. "
                "Please review free-text fields and remove any non-clinical content."
            )
        return "The generated plan did not pass safety validation and was blocked."

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "stage": self.stage,
            "field": self.field,
            "detail": self.detail,
            "metadata": self.metadata,
        }
