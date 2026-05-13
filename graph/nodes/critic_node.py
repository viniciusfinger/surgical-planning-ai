"""
Post-LLM deterministic critic with self-correction support.

Applies hard, auditable safety rules over the structured outputs produced by
the LLM nodes. Correctable violations (R1-R3) are returned as structured
feedback so the graph can route back to the relevant LLM node for a retry.
Non-correctable violations (R4 — output text safety) raise `GuardrailViolation`
immediately. After `MAX_RETRIES` failed correction attempts, any remaining
violation also raises.

Rules implemented:

R1. Internal consistency of the perioperative checklist:
    - overall_status == "clear" must imply empty critical_alerts AND no item
      flagged alert=True.
    - overall_status == "critical" must imply non-empty critical_alerts.

R2. ASA / postoperative destination conflict:
    - ASA IV or V routed to "ward" is a hard conflict.

R3. NSAID safety:
    - NSAIDs in the analgesia plan are forbidden when the patient profile
      includes chronic kidney disease, severe coagulopathy, or active
      heart failure (mirroring the rule already encoded in the postop
      prompt). This catches LLM regressions.

R4. Output text safety (hard fail — not correctable via retry):
    - All free-text strings emitted by the LLM are revalidated against the
      narrower output guard (clinical scope + PII). Prevents echoed
      injection attempts or leaked identifiers from reaching the caller.
"""

from __future__ import annotations

from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput
from graph.schema.postoperative_care_output import PostoperativeCareOutput
from graph.state import CorrectionFeedback, GraphState
from safety.exceptions import GuardrailViolation
from safety.guards import validate_llm_text_output

MAX_RETRIES = 2

NSAID_TOKENS: tuple[str, ...] = (
    "nsaid",
    "ibuprofen",
    "ibuprofeno",
    "ketorolac",
    "cetorolaco",
    "diclofenac",
    "diclofenaco",
    "naproxen",
    "naproxeno",
    "celecoxib",
    "etoricoxib",
    "parecoxib",
    "aspirin",
    "ácido acetilsalicílico",
    "acido acetilsalicilico",
)

NSAID_CONTRAINDICATION_TOKENS: tuple[str, ...] = (
    "kidney",
    "renal",
    "ckd",
    "irc",
    "nephropathy",
    "nefropatia",
    "coagulopath",
    "coagulopat",
    "thrombocytopenia",
    "trombocitopenia",
    "heart failure",
    "insuficiência cardíaca",
    "insuficiencia cardiaca",
)


def _check_checklist_consistency(
    checklist: PerioperativeChecklistOutput,
) -> list[CorrectionFeedback]:
    violations: list[CorrectionFeedback] = []

    has_alert_item = any(
        item.alert
        for phase in (checklist.sign_in, checklist.time_out, checklist.sign_out)
        for item in phase
    )

    if checklist.overall_status == "clear":
        if checklist.critical_alerts:
            violations.append({
                "rule": "checklist_clear_with_critical_alerts",
                "field": "perioperative_checklist",
                "detail": (
                    "overall_status='clear' but critical_alerts is non-empty; "
                    "either set overall_status to 'hold'/'critical' or remove critical_alerts"
                ),
            })
        if has_alert_item:
            violations.append({
                "rule": "checklist_clear_with_alert_items",
                "field": "perioperative_checklist",
                "detail": (
                    "overall_status='clear' but at least one item has alert=True; "
                    "either set overall_status to 'hold' or remove alert=True from all items"
                ),
            })

    if checklist.overall_status == "critical" and not checklist.critical_alerts:
        violations.append({
            "rule": "checklist_critical_without_alerts",
            "field": "perioperative_checklist",
            "detail": (
                "overall_status='critical' but critical_alerts is empty; "
                "add at least one entry to critical_alerts describing the unresolved issue"
            ),
        })

    return violations


def _check_destination_conflict(state: GraphState) -> list[CorrectionFeedback]:
    asa = state["asa"].asa
    destination = state["postoperative_care"].destination
    if asa in ("IV", "V") and destination == "ward":
        return [{
            "rule": "asa_destination_conflict",
            "field": "postoperative_care.destination",
            "detail": (
                f"ASA {asa} patient routed to 'ward' is unsafe; "
                "set destination to 'ICU' for ASA IV/V patients"
            ),
        }]
    return []


def _check_nsaid_safety(state: GraphState) -> list[CorrectionFeedback]:
    comorbidity_blob = " ".join(c.name.lower() for c in state["comorbidities"])
    has_contraindication = any(
        token in comorbidity_blob for token in NSAID_CONTRAINDICATION_TOKENS
    )
    if not has_contraindication:
        return []

    plan: PostoperativeCareOutput = state["postoperative_care"]
    offenders: list[str] = []
    for protocol in plan.analgesia_recommendation:
        haystack = f"{protocol.agent} {protocol.notes or ''}".lower()
        if any(token in haystack for token in NSAID_TOKENS):
            offenders.append(protocol.agent)

    if offenders:
        return [{
            "rule": "nsaid_with_contraindication",
            "field": "postoperative_care.analgesia_recommendation",
            "detail": (
                f"NSAIDs {offenders} prescribed despite contraindicating comorbidity; "
                "replace with paracetamol, dexamethasone, or regional/opioid-sparing alternatives"
            ),
        }]
    return []


def _validate_output_text(state: GraphState) -> None:
    asa = state["asa"]
    validate_llm_text_output(asa.justification, field="asa.justification")

    checklist: PerioperativeChecklistOutput = state["perioperative_checklist"]
    for phase_name, phase in (
        ("sign_in", checklist.sign_in),
        ("time_out", checklist.time_out),
        ("sign_out", checklist.sign_out),
    ):
        for index, item in enumerate(phase):
            validate_llm_text_output(item.item, field=f"checklist.{phase_name}[{index}].item")
            if item.notes:
                validate_llm_text_output(
                    item.notes,
                    field=f"checklist.{phase_name}[{index}].notes",
                )
    for index, alert in enumerate(checklist.critical_alerts):
        validate_llm_text_output(alert, field=f"checklist.critical_alerts[{index}]")
    for index, recommendation in enumerate(checklist.recommendations):
        validate_llm_text_output(
            recommendation,
            field=f"checklist.recommendations[{index}]",
        )

    plan: PostoperativeCareOutput = state["postoperative_care"]
    validate_llm_text_output(plan.destination_rationale, field="postop.destination_rationale")
    for index, item in enumerate(plan.eras_recommendations):
        validate_llm_text_output(item, field=f"postop.eras_recommendations[{index}]")
    for index, item in enumerate(plan.early_mobilization):
        validate_llm_text_output(item, field=f"postop.early_mobilization[{index}]")
    for index, item in enumerate(plan.follow_up_plan):
        validate_llm_text_output(item, field=f"postop.follow_up_plan[{index}]")
    for index, item in enumerate(plan.critical_alerts):
        validate_llm_text_output(item, field=f"postop.critical_alerts[{index}]")
    for index, item in enumerate(plan.recommendations):
        validate_llm_text_output(item, field=f"postop.recommendations[{index}]")


def critic_node(state: GraphState) -> dict:
    checklist_violations = _check_checklist_consistency(state["perioperative_checklist"])
    postop_violations = _check_destination_conflict(state) + _check_nsaid_safety(state)

    _validate_output_text(state)

    retry_count = state.get("retry_count", 0)
    has_violations = bool(checklist_violations or postop_violations)

    if has_violations and retry_count >= MAX_RETRIES:
        raise GuardrailViolation(
            rule="max_retries_exceeded",
            stage="output",
            detail=f"Self-correction failed after {MAX_RETRIES} attempt(s)",
            metadata={
                "violations": checklist_violations + postop_violations,
                "retry_count": retry_count,
            },
        )

    return {
        "checklist_feedback": checklist_violations,
        "postop_feedback": postop_violations,
        "retry_count": retry_count + 1 if has_violations else retry_count,
    }
