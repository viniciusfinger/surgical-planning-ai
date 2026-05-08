from __future__ import annotations

import pytest

from domain.schema.comorbidity import Comorbidity
from domain.schema.severity import Severity
from domain.schema.urgency import Urgency
from graph.nodes.critic_node import critic_node
from graph.schema.ASA_output import ASAOutput
from graph.schema.perioperative_checklist_output import (
    ChecklistItem,
    PerioperativeChecklistOutput,
)
from graph.schema.postoperative_care_output import (
    AnalgesiaProtocol,
    DischargeCriteria,
    PostoperativeCareOutput,
    ProphylaxisItem,
)
from safety.exceptions import GuardrailViolation


def _comorbidity(
    name: str,
    severity: Severity = Severity.mild,
    controlled: bool = True,
) -> Comorbidity:
    return Comorbidity(name=name, severity=severity, controlled=controlled)


def _checklist(
    overall_status: str = "clear",
    critical_alerts: list[str] | None = None,
    sign_in_alert: bool = False,
    recommendations: list[str] | None = None,
) -> PerioperativeChecklistOutput:
    return PerioperativeChecklistOutput(
        sign_in=[ChecklistItem(item="Patient identity confirmed", alert=sign_in_alert)],
        time_out=[ChecklistItem(item="Surgical site marked")],
        sign_out=[ChecklistItem(item="Instrument count complete")],
        overall_status=overall_status,
        critical_alerts=critical_alerts or [],
        recommendations=recommendations or [],
    )


def _postop(
    destination: str = "PACU",
    analgesia: list[AnalgesiaProtocol] | None = None,
) -> PostoperativeCareOutput:
    return PostoperativeCareOutput(
        destination=destination,
        destination_rationale="ASA II patient with stable hemodynamics.",
        analgesia=analgesia
        or [
            AnalgesiaProtocol(
                agent="Paracetamol",
                route="IV",
                dose_or_regimen="1g q6h",
                who_step="step_1",
            )
        ],
        prophylaxis=[
            ProphylaxisItem(target="TEV", intervention="LMWH per Caprini score")
        ],
        eras_recommendations=["Early oral intake"],
        early_mobilization=["Sit at edge of bed within 6 h post-op"],
        discharge_criteria=[
            DischargeCriteria(
                scale="Aldrete", minimum_score=9, specific_criteria=["Stable SpO2"]
            )
        ],
        follow_up_plan=["Outpatient visit at 7 days"],
        critical_alerts=[],
        recommendations=[],
    )


def _state(
    asa_class: str = "II",
    comorbidities: list[Comorbidity] | None = None,
    checklist: PerioperativeChecklistOutput | None = None,
    postop: PostoperativeCareOutput | None = None,
) -> dict:
    return {
        "age": 50,
        "comorbidities": comorbidities or [],
        "surgical_type": "Apendicectomia laparoscopica",
        "urgency": Urgency.elective,
        "asa": ASAOutput(asa=asa_class, confidence=0.92, justification="baseline"),
        "perioperative_checklist": checklist or _checklist(),
        "postoperative_care": postop or _postop(),
    }


class TestHappyPath:
    def test_clean_state_passes(self):
        """A coherent ASA II case with paracetamol-only analgesia must pass."""
        assert critic_node(_state()) == {}


class TestChecklistConsistency:
    def test_blocks_clear_with_critical_alerts(self):
        """`overall_status='clear'` is incompatible with non-empty critical_alerts."""
        state = _state(
            checklist=_checklist(
                overall_status="clear",
                critical_alerts=["Severe uncontrolled bleeding risk"],
            )
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "checklist_clear_with_critical_alerts"
        assert exc_info.value.stage == "output"

    def test_blocks_clear_with_alert_items(self):
        """`overall_status='clear'` is incompatible with any item flagged alert=True."""
        state = _state(checklist=_checklist(overall_status="clear", sign_in_alert=True))
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "checklist_clear_with_alert_items"

    def test_blocks_critical_without_alerts(self):
        """`overall_status='critical'` requires at least one critical_alert."""
        state = _state(
            checklist=_checklist(overall_status="critical", critical_alerts=[])
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "checklist_critical_without_alerts"

    def test_allows_critical_with_alerts(self):
        """`overall_status='critical'` with non-empty critical_alerts is consistent."""
        state = _state(
            checklist=_checklist(
                overall_status="critical",
                critical_alerts=["Unresolved difficult airway"],
            )
        )
        assert critic_node(state) == {}

    def test_allows_hold_with_alert_items(self):
        """`overall_status='hold'` is allowed to have item-level alerts."""
        state = _state(
            checklist=_checklist(overall_status="hold", sign_in_alert=True)
        )
        assert critic_node(state) == {}


class TestAsaDestinationConflict:
    @pytest.mark.parametrize("asa_class", ["IV", "V"])
    def test_blocks_high_asa_routed_to_ward(self, asa_class: str):
        """ASA IV and ASA V patients must never be routed to a general ward."""
        state = _state(asa_class=asa_class, postop=_postop(destination="ward"))
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "asa_destination_conflict"
        assert exc_info.value.field == "postoperative_care.destination"

    @pytest.mark.parametrize("destination", ["PACU", "ICU"])
    def test_allows_high_asa_routed_to_higher_care(self, destination: str):
        """ASA IV with PACU or ICU is acceptable; only `ward` is blocked."""
        state = _state(asa_class="IV", postop=_postop(destination=destination))
        assert critic_node(state) == {}

    def test_allows_low_asa_routed_to_ward(self):
        """ASA II routed to ward is a normal pathway and must pass."""
        state = _state(asa_class="II", postop=_postop(destination="ward"))
        assert critic_node(state) == {}


class TestNsaidSafety:
    @pytest.mark.parametrize(
        "comorbidity_name",
        [
            "Chronic kidney disease",
            "CKD stage 3",
            "Severe coagulopathy",
            "Decompensated heart failure",
        ],
    )
    def test_blocks_nsaid_with_contraindication(self, comorbidity_name: str):
        """NSAIDs must be blocked when a contraindicating comorbidity is present."""
        state = _state(
            comorbidities=[
                _comorbidity(comorbidity_name, severity=Severity.severe, controlled=True)
            ],
            postop=_postop(
                analgesia=[
                    AnalgesiaProtocol(
                        agent="Ibuprofen",
                        route="PO",
                        dose_or_regimen="400 mg q8h",
                        who_step="step_1",
                    )
                ]
            ),
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "nsaid_with_contraindication"

    def test_allows_paracetamol_with_contraindication(self):
        """Paracetamol-only analgesia is safe even when NSAIDs are forbidden."""
        state = _state(
            comorbidities=[
                _comorbidity(
                    "Chronic kidney disease", severity=Severity.severe, controlled=True
                )
            ],
            postop=_postop(),
        )
        assert critic_node(state) == {}

    def test_allows_nsaid_without_contraindication(self):
        """NSAIDs are allowed when no contraindicating comorbidity is present."""
        state = _state(
            comorbidities=[_comorbidity("Hypertension")],
            postop=_postop(
                analgesia=[
                    AnalgesiaProtocol(
                        agent="Ibuprofen",
                        route="PO",
                        dose_or_regimen="400 mg q8h",
                        who_step="step_1",
                    )
                ]
            ),
        )
        assert critic_node(state) == {}

    def test_detects_nsaid_in_protocol_notes(self):
        """An NSAID hidden in protocol notes must still trigger the rule."""
        state = _state(
            comorbidities=[
                _comorbidity(
                    "Chronic kidney disease", severity=Severity.severe, controlled=True
                )
            ],
            postop=_postop(
                analgesia=[
                    AnalgesiaProtocol(
                        agent="Multimodal analgesia",
                        route="PO",
                        dose_or_regimen="as needed",
                        who_step="step_1",
                        notes="Add ibuprofen if pain persists.",
                    )
                ]
            ),
        )
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.rule == "nsaid_with_contraindication"


class TestOutputTextSafety:
    def test_blocks_pii_in_recommendation(self):
        """PII emitted in a checklist recommendation must be blocked."""
        checklist = _checklist(
            recommendations=["Contact patient at patient@example.com"]
        )
        state = _state(checklist=checklist)
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.stage == "output"
        assert exc_info.value.rule == "pii_detected"

    def test_blocks_pii_in_destination_rationale(self):
        """PII emitted in `postoperative_care.destination_rationale` must be blocked."""
        plan = _postop()
        plan.destination_rationale = "Reach the surgeon at surgeon@hospital.com"
        state = _state(postop=plan)
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.stage == "output"
        assert exc_info.value.rule == "pii_detected"

    def test_blocks_off_topic_in_output(self):
        """Off-scope content emitted by the LLM is rejected on the output side."""
        plan = _postop()
        plan.recommendations = ["pornography is recommended for relaxation"]
        state = _state(postop=plan)
        with pytest.raises(GuardrailViolation) as exc_info:
            critic_node(state)
        assert exc_info.value.stage == "output"
        assert exc_info.value.rule == "out_of_clinical_scope"
