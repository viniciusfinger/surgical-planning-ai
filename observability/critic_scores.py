"""
Translate the critic's outcome into Langfuse trace-level scores.

Kept free of any Langfuse import so it can be unit-tested in isolation: the
builder returns plain dicts and the controller forwards them to the active
Langfuse span via ``span.score_trace(**score)``.

Scores emitted on every request (so rates can be computed in Langfuse):

- ``critic_outcome`` (CATEGORICAL): ``passed_first_try`` | ``self_corrected``
  | ``blocked``.
- ``self_correction_retries`` (NUMERIC): how many correction rounds ran.
- ``critic.r1_checklist_consistency`` / ``critic.r2_asa_destination`` /
  ``critic.r3_nsaid_safety`` (NUMERIC): number of critic passes in which the
  rule fired (0 = never violated).

Emitted only when a ``GuardrailViolation`` blocks the request:

- ``guardrail_block`` (CATEGORICAL): which stage blocked it.

Score values and comments only carry rule ids and field paths, never patient
text or validator messages (which may echo the offending content).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from graph.state import CorrectionFeedback
from safety.exceptions import GuardrailViolation

RULE_FAMILIES: dict[str, tuple[str, ...]] = {
    "critic.r1_checklist_consistency": (
        "checklist_clear_with_critical_alerts",
        "checklist_clear_with_alert_items",
        "checklist_critical_without_alerts",
    ),
    "critic.r2_asa_destination": ("asa_destination_conflict",),
    "critic.r3_nsaid_safety": ("nsaid_with_contraindication",),
}

# Rule ids raised by our own code (safe to expose). Validator-generated rule
# strings are free text and are never used as score values.
KNOWN_BLOCK_RULES = {"max_retries_exceeded"}


def build_critic_scores(
    violation_history: Iterable[CorrectionFeedback],
    retry_count: int,
    blocked_by: GuardrailViolation | None = None,
) -> list[dict[str, Any]]:
    history = list(violation_history)
    rule_counts = Counter(v["rule"] for v in history)

    if blocked_by is not None:
        outcome = "blocked"
    elif retry_count > 0:
        outcome = "self_corrected"
    else:
        outcome = "passed_first_try"

    scores: list[dict[str, Any]] = [
        {"name": "critic_outcome", "value": outcome, "data_type": "CATEGORICAL"},
        {
            "name": "self_correction_retries",
            "value": float(retry_count),
            "data_type": "NUMERIC",
        },
    ]

    for score_name, rules in RULE_FAMILIES.items():
        fired = {rule: rule_counts[rule] for rule in rules if rule_counts[rule]}
        score: dict[str, Any] = {
            "name": score_name,
            "value": float(sum(fired.values())),
            "data_type": "NUMERIC",
        }
        if fired:
            score["comment"] = ", ".join(f"{r} x{n}" for r, n in fired.items())
        scores.append(score)

    if blocked_by is not None:
        rule = blocked_by.rule if blocked_by.rule in KNOWN_BLOCK_RULES else "guard_validator"
        scores.append({
            "name": "guardrail_block",
            "value": f"{blocked_by.stage}:{rule}",
            "data_type": "CATEGORICAL",
            "comment": f"field={blocked_by.field}" if blocked_by.field else None,
        })

    return scores
