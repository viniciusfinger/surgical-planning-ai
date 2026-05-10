"""Postoperative care evaluation dataset.

Each :class:`Golden` carries the partial graph state required by
:func:`graph.nodes.postoperative_care_node.postoperative_care_node` and
expectations consumed by the metrics in :mod:`evals.metrics.clinical`:

* ``expected_destination_in`` — whitelist for ``postop.destination``.
* ``required_keyword_groups`` — clinical themes (synonyms) that must appear
  somewhere in the rendered care plan.
* ``forbid_step3`` — when True, no WHO step_3 (strong opioid) agents allowed.
* ``comorbidity_names`` — used by :class:`NsaidContraindicationMetric`.
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
    surgical_type: str,
    urgency: str,
    expected_destination_in: list[str],
    expected_output: str,
    required_keyword_groups: list[list[str]] | None = None,
    forbid_step3: bool = False,
    expected_prophylaxis_targets: list[str] | None = None,
) -> Golden:
    payload = {
        "age": age,
        "comorbidities": comorbidities,
        "surgical_type": surgical_type,
        "urgency": urgency,
        "asa": {
            "asa": asa_class,
            "confidence": 0.95,
            "justification": asa_justification,
        },
    }
    metadata: dict[str, Any] = {
        "category": category,
        "expected_asa": asa_class,
        "expected_destination_in": expected_destination_in,
        "comorbidity_names": [c["name"] for c in comorbidities],
    }
    if required_keyword_groups:
        metadata["required_keyword_groups"] = required_keyword_groups
    if forbid_step3:
        metadata["forbid_step3"] = True
    if expected_prophylaxis_targets is not None:
        metadata["expected_prophylaxis_targets"] = list(expected_prophylaxis_targets)
    return Golden(
        input=json.dumps(payload, ensure_ascii=False),
        expected_output=expected_output,
        additional_metadata=metadata,
    )


POSTOP_GOLDENS: list[Golden] = [
    _golden(
        category="healthy_adult_minor_elective",
        age=32,
        comorbidities=[],
        asa_class="I",
        asa_justification="No systemic disease.",
        surgical_type="Elective inguinal hernia repair",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Destination is 'PACU' or 'ward' for an ASA I patient with elective "
            "minor surgery.\n"
            "Analgesia starts with non-opioid agents (WHO step_1) such as "
            "paracetamol or NSAIDs.\n"
            "Prophylaxis covers TEV, IRAS and NVPO targets.\n"
            "Discharge criteria use Aldrete and/or PADSS appropriate for ambulatory "
            "pathway.\n"
            "Critical Alerts list is empty for a healthy patient with low-risk surgery."
        ),
        forbid_step3=True,
    ),
    _golden(
        category="three_prophylaxis_targets_always_covered",
        age=50,
        comorbidities=[
            {"name": "Controlled hypertension", "severity": "mild", "controlled": True}
        ],
        asa_class="II",
        asa_justification="Single mild controlled systemic disease.",
        surgical_type="Elective laparoscopic cholecystectomy",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Prophylaxis section contains exactly the three required targets: TEV, "
            "IRAS, NVPO.\n"
            "Each prophylaxis item has a concrete intervention.\n"
            "Any item with alert=True includes notes describing the patient-specific "
            "risk."
        ),
    ),
    _golden(
        category="analgesia_step1_first",
        age=45,
        comorbidities=[],
        asa_class="I",
        asa_justification="Healthy adult with no systemic disease.",
        surgical_type="Elective minor outpatient lipoma excision",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Analgesia begins with WHO step_1 agents (paracetamol, NSAIDs, or COX-2 "
            "inhibitors).\n"
            "No strong opioids (step_3) are prescribed for this minor outpatient "
            "procedure.\n"
            "Each analgesic agent has a clear route and dose/regimen."
        ),
        forbid_step3=True,
    ),
    _golden(
        category="asa_iv_emergency_icu",
        age=72,
        comorbidities=[
            {"name": "Heart failure", "severity": "severe", "controlled": False}
        ],
        asa_class="IV",
        asa_justification=(
            "Severe uncontrolled heart failure with life-threatening systemic disease."
        ),
        surgical_type="Emergency open exploratory laparotomy for ischemic bowel",
        urgency="emergency",
        expected_destination_in=["ICU"],
        expected_output=(
            "Destination is 'ICU' for ASA IV emergency surgery with severe "
            "uncontrolled heart failure.\n"
            "Critical Alerts list is non-empty and references cardiac/hemodynamic "
            "risk or ICU-level monitoring.\n"
            "Analgesia is multimodal and may include strong opioids when justified.\n"
            "Prophylaxis covers TEV, IRAS, NVPO with patient-specific notes."
        ),
        required_keyword_groups=[
            ["hemodynamic", "cardiac", "vasopressor", "icu", "intensive"],
        ],
    ),
    _golden(
        category="ckd_avoids_nsaids",
        age=65,
        comorbidities=[
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True}
        ],
        asa_class="III",
        asa_justification="Moderate CKD with severe systemic burden.",
        surgical_type="Elective open inguinal hernia repair",
        urgency="elective",
        expected_destination_in=["PACU", "ICU", "ward"],
        expected_output=(
            "Analgesia EXCLUDES NSAIDs (ibuprofen, ketorolac, diclofenac) for this "
            "CKD patient; paracetamol and other non-nephrotoxic agents are preferred.\n"
            "The plan references renal function review and avoidance of nephrotoxic "
            "agents.\n"
            "Recommendations or critical_alerts explicitly mention renal monitoring "
            "and CKD-related perioperative considerations."
        ),
        required_keyword_groups=[
            ["renal", "kidney", "nephro"],
        ],
    ),
    _golden(
        category="high_apfel_nvpo_multimodal",
        age=38,
        comorbidities=[
            {
                "name": "History of postoperative nausea and vomiting",
                "severity": "moderate",
                "controlled": False,
            },
            {"name": "Motion sickness", "severity": "mild", "controlled": False},
        ],
        asa_class="II",
        asa_justification="Mild systemic burden with high Apfel risk.",
        surgical_type=(
            "Elective laparoscopic cholecystectomy in non-smoker female with prior NVPO"
        ),
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "NVPO prophylaxis names at least one specific antiemetic agent "
            "(ondansetron, dexamethasone, droperidol).\n"
            "NVPO prophylaxis is multimodal (combines two or more agents from "
            "different classes) given high Apfel score.\n"
            "At least one NVPO item has alert=True referencing the high Apfel score."
        ),
        required_keyword_groups=[
            ["ondansetron", "dexamethasone", "droperidol", "scopolamine", "antiemetic"],
        ],
    ),
    _golden(
        category="coagulopathy_tev_conflict",
        age=58,
        comorbidities=[
            {
                "name": "Severe thrombocytopenia with active coagulopathy",
                "severity": "severe",
                "controlled": False,
            }
        ],
        asa_class="III",
        asa_justification="Severe coagulopathy elevating bleeding risk.",
        surgical_type="Elective open abdominal hysterectomy",
        urgency="elective",
        expected_destination_in=["PACU", "ICU", "ward"],
        expected_output=(
            "TEV prophylaxis has alert=True due to coagulopathy / bleeding risk.\n"
            "The plan favors mechanical measures (IPC, elastic stockings) over "
            "pharmacological anticoagulation, OR documents the bleeding-vs-clotting "
            "conflict explicitly.\n"
            "Notes describe the coagulopathy as the contraindication driver."
        ),
        required_keyword_groups=[
            ["mechanical", "compression", "stocking", "ipc", "bleeding", "coagulopath"],
        ],
    ),
    _golden(
        category="colorectal_specific_eras",
        age=62,
        comorbidities=[
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True}
        ],
        asa_class="II",
        asa_justification="Moderate controlled diabetes.",
        surgical_type="Elective laparoscopic colorectal resection (sigmoidectomy)",
        urgency="elective",
        expected_destination_in=["PACU", "ICU", "ward"],
        expected_output=(
            "ERAS recommendations include at least three core ERAS items applied to "
            "colorectal surgery: early oral feeding, opioid-sparing multimodal "
            "analgesia, euvolemic fluid management, minimization of drains/catheters, "
            "and early mobilization.\n"
            "ERAS or recommendations also include at least one distinctive colorectal "
            "element such as postoperative ileus prevention, early bowel function "
            "recovery, chewing gum, or early urinary catheter removal.\n"
            "Glycemic control is mentioned because the patient has Type 2 diabetes."
        ),
        required_keyword_groups=[
            ["ileus", "bowel function", "gastrointestinal", "gut motility", "chewing gum"],
            ["glucose", "glycemic", "diabetes", "insulin"],
        ],
    ),
    _golden(
        category="pediatric_tonsillectomy",
        age=7,
        comorbidities=[],
        asa_class="I",
        asa_justification="Healthy pediatric patient with no systemic disease.",
        surgical_type="Elective tonsillectomy in healthy child",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Destination is 'PACU' or 'ward' for healthy pediatric tonsillectomy.\n"
            "Analgesia uses non-opioid agents with weight-based dosing (e.g., "
            "paracetamol 10–15 mg/kg, ibuprofen 5–10 mg/kg).\n"
            "Strong opioids (step_3) are NOT prescribed.\n"
            "Prophylaxis covers IRAS and NVPO. Pharmacological TEV prophylaxis is "
            "not indicated for a healthy 7-year-old undergoing minor ambulatory "
            "surgery (very low Caprini/Padua risk).\n"
            "Follow-up plan includes parental education and signs of bleeding or "
            "dehydration."
        ),
        required_keyword_groups=[
            ["weight", "pediatric", "child", "kg", "guardian", "parent"],
        ],
        forbid_step3=True,
        # TEV prophylaxis is not indicated in healthy ambulatory pediatric
        # surgery. The default {TEV, IRAS, NVPO} would be a clinical error.
        expected_prophylaxis_targets=["IRAS", "NVPO"],
    ),
    _golden(
        category="elderly_multimorbidity_urgent",
        age=82,
        comorbidities=[
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": False},
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True},
            {"name": "Ischemic heart disease", "severity": "moderate", "controlled": True},
        ],
        asa_class="III",
        asa_justification="Multiple moderate comorbidities in elderly patient.",
        surgical_type="Urgent open abdominal surgery for incarcerated hernia",
        urgency="urgent",
        expected_destination_in=["ICU", "PACU"],
        expected_output=(
            "Destination is 'ICU' or 'PACU' (never 'ward') for elderly ASA III urgent "
            "abdominal surgery.\n"
            "The plan addresses renal function (CKD), glycemic control (uncontrolled "
            "diabetes), and cardiac/hemodynamic monitoring (ischemic heart disease).\n"
            "Recommendations or critical_alerts mention elderly-specific risks "
            "(delirium, frailty, polypharmacy)."
        ),
        required_keyword_groups=[
            ["renal", "kidney", "nephro"],
            ["glucose", "glycemic", "diabetes", "insulin"],
            ["cardiac", "ecg", "ischemic", "hemodynamic"],
            # Elderly-specific risks may surface as any of: explicit
            # geriatric labels, age-related framing, cognitive risks, or
            # pharmacology cautions tied to advanced age. Broadened to
            # avoid penalizing valid plans that frame the same theme
            # differently.
            [
                "delirium",
                "frailty",
                "fragility",
                "polypharmacy",
                "elderly",
                "geriatric",
                "advanced age",
                "older adult",
                "age-related",
                "cognitive",
            ],
        ],
    ),
    _golden(
        category="severe_copd_thoracic",
        age=68,
        comorbidities=[
            {"name": "COPD", "severity": "severe", "controlled": False}
        ],
        asa_class="III",
        asa_justification="Severe uncontrolled COPD with major respiratory limitation.",
        surgical_type="Elective open thoracotomy for lung lobectomy",
        urgency="elective",
        expected_destination_in=["ICU", "PACU"],
        expected_output=(
            "Destination is 'ICU' or 'PACU' (never 'ward') for severe COPD undergoing "
            "thoracotomy.\n"
            "Early mobilization includes respiratory physiotherapy, incentive "
            "spirometry, or breathing exercises.\n"
            "Plan references bronchodilator availability and oxygen/ventilation "
            "strategy.\n"
            "Critical alerts or recommendations address COPD exacerbation and "
            "atelectasis risk."
        ),
        required_keyword_groups=[
            [
                "respiratory",
                "pulmonary",
                "bronchodilator",
                "oxygen",
                "ventilation",
                "physiotherapy",
                "spirometry",
                "incentive",
            ],
        ],
    ),
    _golden(
        category="discharge_criteria_scales",
        age=40,
        comorbidities=[],
        asa_class="I",
        asa_justification="Healthy adult.",
        surgical_type="Elective day-case knee arthroscopy",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Discharge criteria include Aldrete (target ≥9) for PACU and/or PADSS "
            "(target ≥9) for ambulatory home discharge.\n"
            "Each scale lists patient-specific criteria such as pain control, "
            "oxygenation, ambulation, oral intake, nausea, and surgical site status."
        ),
    ),
    _golden(
        category="follow_up_medication_reconciliation",
        age=58,
        comorbidities=[
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True},
            {"name": "Hypertension", "severity": "mild", "controlled": True},
        ],
        asa_class="II",
        asa_justification="Mild systemic burden, fully controlled.",
        surgical_type="Elective laparoscopic cholecystectomy",
        urgency="elective",
        expected_destination_in=["PACU", "ward"],
        expected_output=(
            "Follow-up plan includes medication reconciliation for antihypertensive "
            "and antidiabetic agents.\n"
            "Follow-up plan schedules a return visit (e.g., 7-14 days post-op) and "
            "wound assessment.\n"
            "Patient education covers warning signs and activity restrictions."
        ),
        required_keyword_groups=[
            ["medication", "reconciliation", "antihypertens", "antidiabet", "insulin"],
        ],
    ),
]


postop_dataset = EvaluationDataset(goldens=POSTOP_GOLDENS)


__all__ = ["POSTOP_GOLDENS", "postop_dataset"]
