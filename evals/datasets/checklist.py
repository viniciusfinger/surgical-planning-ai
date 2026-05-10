"""Perioperative checklist evaluation dataset.

Each :class:`Golden` carries the partial graph state (age, comorbidities,
ASA assigned upstream) needed by
:func:`graph.nodes.perioperative_checklist_node.perioperative_checklist_node`,
plus deterministic expectations consumed by the metrics in
:mod:`evals.metrics.clinical`:

* ``expected_overall_status_in`` — whitelist of ``overall_status`` values.
* ``required_keyword_groups`` — clinical themes (synonyms) that must appear
  somewhere in the rendered checklist (item labels, notes, alerts, recs).
* ``expect_no_critical_alerts`` — when True, ``critical_alerts`` must be empty.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.dataset import EvaluationDataset, Golden


def _golden(
    *,
    category: str,
    age: int,
    comorbidities: list[dict[str, Any]],
    asa_class: str,
    asa_justification: str,
    expected_overall_status_in: list[str],
    expected_output: str,
    required_keyword_groups: list[list[str]] | None = None,
    expect_no_critical_alerts: bool = False,
) -> Golden:
    payload = {
        "age": age,
        "comorbidities": comorbidities,
        "asa": {
            "asa": asa_class,
            "confidence": 0.95,
            "justification": asa_justification,
        },
    }
    metadata: dict[str, Any] = {
        "category": category,
        "expected_asa": asa_class,
        "expected_overall_status_in": expected_overall_status_in,
        "comorbidity_names": [c["name"] for c in comorbidities],
    }
    if required_keyword_groups:
        metadata["required_keyword_groups"] = required_keyword_groups
    if expect_no_critical_alerts:
        metadata["expect_no_critical_alerts"] = True
    return Golden(
        input=json.dumps(payload, ensure_ascii=False),
        expected_output=expected_output,
        additional_metadata=metadata,
    )


CHECKLIST_GOLDENS: list[Golden] = [
    _golden(
        category="healthy_adult_asa_i_clear",
        age=32,
        comorbidities=[],
        asa_class="I",
        asa_justification="No systemic disease.",
        expected_overall_status_in=["clear"],
        expected_output=(
            "Overall Status: clear\n"
            "All three phases populated with standard WHO checklist items.\n"
            "No critical alerts for a healthy ASA I patient.\n"
            "Recommendations may include routine postoperative monitoring."
        ),
        expect_no_critical_alerts=True,
    ),
    _golden(
        category="mild_controlled_hypertension_clear_or_hold",
        age=52,
        comorbidities=[
            {"name": "Controlled hypertension", "severity": "mild", "controlled": True}
        ],
        asa_class="II",
        asa_justification="Single mild controlled systemic disease.",
        expected_overall_status_in=["clear", "hold"],
        expected_output=(
            "Overall Status is either 'clear' or 'hold' (never 'critical').\n"
            "Sign-In or Time-Out includes a blood pressure-related item or an "
            "antihypertensive medication review note.\n"
            "Critical Alerts list is empty for this mild controlled condition.\n"
            "Recommendations mention perioperative blood pressure monitoring or "
            "antihypertensive continuation."
        ),
        required_keyword_groups=[
            ["blood pressure", "hypertens", "antihypertens"],
        ],
        expect_no_critical_alerts=True,
    ),
    _golden(
        category="moderate_uncontrolled_copd_airway_alert",
        age=63,
        comorbidities=[
            {"name": "COPD", "severity": "moderate", "controlled": False}
        ],
        asa_class="III",
        asa_justification="Moderate uncontrolled COPD implies severe systemic limitation.",
        expected_overall_status_in=["hold", "critical"],
        expected_output=(
            "Overall Status: hold or critical\n"
            "Sign-In includes difficult airway assessment and bronchodilator availability.\n"
            "Time-Out references oxygen reserve and ventilation plan.\n"
            "Alerts or recommendations address COPD exacerbation risk."
        ),
        required_keyword_groups=[
            ["airway", "respiratory", "pulmonary", "copd", "bronchospasm", "oxygen"],
        ],
    ),
    _golden(
        category="severe_uncontrolled_heart_failure_critical",
        age=74,
        comorbidities=[
            {"name": "Heart failure", "severity": "severe", "controlled": False}
        ],
        asa_class="IV",
        asa_justification=(
            "Severe uncontrolled heart failure is a life-threatening systemic disease."
        ),
        expected_overall_status_in=["hold", "critical"],
        expected_output=(
            "Overall Status is 'hold' or 'critical'.\n"
            "Critical Alerts (when present) reference cardiac/hemodynamic risk such as "
            "hemodynamic instability, decompensation, or anesthetic risk from ASA IV.\n"
            "At least one Sign-In item addresses cardiac stability verification.\n"
            "Recommendations include specialist consultation and/or advanced monitoring."
        ),
        required_keyword_groups=[
            ["cardiac", "heart", "hemodynamic", "ecg", "echocardiogr"],
        ],
    ),
    _golden(
        category="diabetes_recommendations_present",
        age=60,
        comorbidities=[
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True}
        ],
        asa_class="II",
        asa_justification="Moderate controlled diabetes is mild systemic disease.",
        expected_overall_status_in=["clear", "hold"],
        expected_output=(
            "Critical Alerts list is empty for this controlled ASA II diabetic patient.\n"
            "Recommendations include diabetes-specific perioperative concerns such as "
            "glucose monitoring, glycemic control targets, insulin management, or "
            "wound healing."
        ),
        required_keyword_groups=[
            ["glucose", "glycemic", "diabetes", "insulin"],
        ],
        expect_no_critical_alerts=True,
    ),
    _golden(
        category="pediatric_healthy_clear",
        age=7,
        comorbidities=[],
        asa_class="I",
        asa_justification="Healthy pediatric patient with no systemic disease.",
        expected_overall_status_in=["clear"],
        expected_output=(
            "Overall Status is 'clear'.\n"
            "Sign-In includes a weight-based drug dosing verification item and a "
            "parental/guardian informed-consent item.\n"
            "At least one item addresses pediatric airway considerations.\n"
            "Critical Alerts list is empty."
        ),
        required_keyword_groups=[
            ["pediatric", "child", "weight", "consent", "parent", "guardian"],
        ],
        expect_no_critical_alerts=True,
    ),
    _golden(
        category="elderly_multimorbidity_hold",
        age=81,
        comorbidities=[
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True},
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True},
        ],
        asa_class="III",
        asa_justification=(
            "Two moderate comorbidities in an elderly patient escalate to ASA III."
        ),
        expected_overall_status_in=["hold", "critical"],
        expected_output=(
            "Overall Status: hold\n"
            "Sign-In includes renal function review and nephrotoxic agent avoidance.\n"
            "Sign-In or Time-Out includes glucose level verification.\n"
            "Recommendations address elderly-specific perioperative risks."
        ),
        required_keyword_groups=[
            ["renal", "kidney", "nephro"],
            ["glucose", "glycemic", "diabetes", "insulin"],
        ],
    ),
    _golden(
        category="moribund_septic_shock_critical",
        age=59,
        comorbidities=[
            {
                "name": "Septic shock with multi-organ failure",
                "severity": "severe",
                "controlled": False,
            }
        ],
        asa_class="V",
        asa_justification="Moribund patient not expected to survive without immediate intervention.",
        expected_overall_status_in=["critical"],
        expected_output=(
            "Overall Status: critical\n"
            "Critical Alerts reference hemodynamic instability, vasopressor "
            "dependence, and organ failure.\n"
            "Sign-In includes ICU-level monitoring requirements and resuscitation "
            "readiness.\n"
            "Multiple items carry alert=True."
        ),
        required_keyword_groups=[
            ["icu", "intensive", "vasopressor", "resuscitation", "septic", "shock"],
        ],
    ),
    _golden(
        category="severe_uncontrolled_copd_alert_notes",
        age=68,
        comorbidities=[
            {"name": "Severe COPD", "severity": "severe", "controlled": False}
        ],
        asa_class="III",
        asa_justification="Severe uncontrolled COPD is a severe systemic limitation.",
        expected_overall_status_in=["hold", "critical"],
        expected_output=(
            "Every item with alert=True includes a non-empty notes field explaining "
            "the specific risk and the recommended corrective action or precaution."
        ),
        required_keyword_groups=[
            ["airway", "respiratory", "pulmonary", "copd", "bronchodilator", "oxygen"],
        ],
    ),
    _golden(
        category="dual_comorbidity_coverage",
        age=66,
        comorbidities=[
            {"name": "Ischemic heart disease", "severity": "moderate", "controlled": True},
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": False},
        ],
        asa_class="III",
        asa_justification="Two moderate comorbidities; uncontrolled diabetes escalates concern.",
        expected_overall_status_in=["hold", "critical"],
        expected_output=(
            "Checklist covers both ischemic heart disease (ECG review, anti-anginal "
            "medication, hemodynamic monitoring) and uncontrolled diabetes (pre-op "
            "glucose, insulin protocol, glycemic targets). overall_status is 'hold' "
            "or 'critical'."
        ),
        required_keyword_groups=[
            ["cardiac", "coronary", "ischemic", "heart", "ecg"],
            ["glucose", "glycemic", "diabetes", "insulin", "blood sugar"],
        ],
    ),
    _golden(
        category="asa_i_no_critical_alerts",
        age=28,
        comorbidities=[],
        asa_class="I",
        asa_justification="Healthy young adult with no systemic disease.",
        expected_overall_status_in=["clear"],
        expected_output=(
            "Overall Status is 'clear'.\n"
            "Critical Alerts list is empty.\n"
            "All checklist items have alert=False."
        ),
        expect_no_critical_alerts=True,
    ),
]


checklist_dataset = EvaluationDataset(goldens=CHECKLIST_GOLDENS)


__all__ = ["CHECKLIST_GOLDENS", "checklist_dataset"]
