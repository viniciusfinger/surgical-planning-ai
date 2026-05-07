from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from graph.nodes.postoperative_care_node import postoperative_care_node
from graph.schema.ASA_output import ASAOutput


def _make_state(
    age: int,
    comorbidities: list,
    asa_class: str,
    surgical_type: str,
    urgency: str = "elective",
    justification: str = "",
):
    return {
        "age": age,
        "comorbidities": comorbidities,
        "surgical_type": surgical_type,
        "urgency": urgency,
        "asa": ASAOutput(
            asa=asa_class,
            confidence=0.95,
            justification=justification or f"ASA {asa_class} assigned based on provided comorbidities.",
        ),
    }


def _format_output(care) -> str:
    def _format_list(values):
        if not values:
            return "  (none)"
        return "\n".join(f"  - {v}" for v in values)

    analgesia = "\n".join(
        f"  [{a.who_step}] {a.agent} ({a.route}, {a.dose_or_regimen}) notes={a.notes}"
        for a in care.analgesia
    ) or "  (none)"

    prophylaxis = "\n".join(
        f"  [{p.target}] {p.intervention} alert={p.alert} notes={p.notes}"
        for p in care.prophylaxis
    ) or "  (none)"

    discharge = "\n".join(
        f"  [{d.scale} ≥{d.minimum_score}] criteria={'; '.join(d.specific_criteria) or '(none)'}"
        for d in care.discharge_criteria
    ) or "  (none)"

    return (
        f"Destination: {care.destination}\n"
        f"Destination Rationale: {care.destination_rationale}\n"
        f"Analgesia:\n{analgesia}\n"
        f"Prophylaxis:\n{prophylaxis}\n"
        f"ERAS Recommendations:\n{_format_list(care.eras_recommendations)}\n"
        f"Early Mobilization:\n{_format_list(care.early_mobilization)}\n"
        f"Discharge Criteria:\n{discharge}\n"
        f"Follow-up Plan:\n{_format_list(care.follow_up_plan)}\n"
        f"Critical Alerts:\n{_format_list(care.critical_alerts)}\n"
        f"Recommendations:\n{_format_list(care.recommendations)}"
    )


def _prophylaxis_targets(care) -> set:
    return {p.target for p in care.prophylaxis}


def _all_text(care) -> str:
    parts = [care.destination_rationale]
    parts += [f"{a.agent} {a.route} {a.dose_or_regimen} {a.notes or ''}" for a in care.analgesia]
    parts += [f"{p.target} {p.intervention} {p.notes or ''}" for p in care.prophylaxis]
    parts += list(care.eras_recommendations)
    parts += list(care.early_mobilization)
    for d in care.discharge_criteria:
        parts += [d.scale, *d.specific_criteria]
    parts += list(care.follow_up_plan)
    parts += list(care.critical_alerts)
    parts += list(care.recommendations)
    return " ".join(parts).lower()


def test_healthy_adult_elective_minor_surgery_pacu_or_ward():
    """Healthy ASA I patient undergoing elective minor surgery should go to PACU or ward."""
    state = _make_state(
        32, [], "I", "Elective inguinal hernia repair", "elective", "No systemic disease."
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert care.destination in ("PACU", "ward")
    assert len(care.analgesia) >= 1
    assert any(a.who_step == "step_1" for a in care.analgesia)
    assert _prophylaxis_targets(care) == {"TEV", "IRAS", "NVPO"}
    assert len(care.eras_recommendations) >= 1
    assert len(care.early_mobilization) >= 1
    assert len(care.discharge_criteria) >= 1
    assert len(care.follow_up_plan) >= 1
    assert care.critical_alerts == []

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Destination is 'PACU' or 'ward' for an ASA I patient with elective minor surgery.\n"
            "Analgesia starts with non-opioid agents (WHO step_1) such as paracetamol or NSAIDs.\n"
            "Prophylaxis covers TEV, IRAS and NVPO targets.\n"
            "Discharge criteria use Aldrete and/or PADSS appropriate for ambulatory pathway.\n"
            "Critical Alerts list is empty for a healthy patient with low-risk surgery."
        ),
    )

    metric = GEval(
        name="Healthy Adult Elective Postop Plan",
        criteria="""
        Evaluate whether:
        1. destination is 'PACU' or 'ward' (never 'ICU') for an ASA I elective minor surgery
        2. Analgesia includes at least one non-opioid (WHO step_1) agent
        3. Prophylaxis explicitly covers TEV, IRAS and NVPO
        4. Discharge criteria are present and appropriate for ambulatory recovery
        5. critical_alerts list is empty for this low-risk profile
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_three_prophylaxis_targets_always_covered():
    """Every postoperative plan must address TEV, IRAS and NVPO targets."""
    state = _make_state(
        50,
        [{"name": "Controlled hypertension", "severity": "mild", "controlled": True}],
        "II",
        "Elective laparoscopic cholecystectomy",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    targets = _prophylaxis_targets(care)
    assert targets == {"TEV", "IRAS", "NVPO"}, f"Missing prophylaxis targets, got {targets}"
    for p in care.prophylaxis:
        assert p.intervention.strip(), f"Empty intervention for {p.target}"
        if p.alert:
            assert p.notes and p.notes.strip(), (
                f"Prophylaxis target {p.target} has alert=True but no notes."
            )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Prophylaxis section contains exactly the three required targets: TEV, IRAS, NVPO.\n"
            "Each prophylaxis item has a concrete intervention.\n"
            "Any item with alert=True includes notes describing the patient-specific risk."
        ),
    )

    metric = GEval(
        name="Prophylaxis Three-Target Coverage",
        criteria="""
        Evaluate whether:
        1. Prophylaxis includes the targets TEV, IRAS and NVPO (each at least once)
        2. Each prophylaxis item has a concrete, named intervention (not vague)
        3. Items with alert=True include a notes field explaining the conflict or risk
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_analgesia_starts_with_non_opioid():
    """The analgesia plan must always begin with a WHO step_1 (non-opioid) agent."""
    state = _make_state(
        45,
        [],
        "I",
        "Elective minor outpatient lipoma excision",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert any(a.who_step == "step_1" for a in care.analgesia), (
        "At least one step_1 (non-opioid) agent is required."
    )
    assert not any(a.who_step == "step_3" for a in care.analgesia), (
        "Strong opioids (step_3) are not justified for ASA I outpatient surgery."
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Analgesia begins with WHO step_1 agents (paracetamol, NSAIDs, or COX-2 inhibitors).\n"
            "No strong opioids (step_3) are prescribed for this minor outpatient procedure.\n"
            "Each analgesic agent has a clear route and dose/regimen."
        ),
    )

    metric = GEval(
        name="WHO Ladder Step 1 First",
        criteria="""
        Evaluate whether:
        1. At least one analgesic agent is at WHO step_1 (non-opioid)
        2. No strong opioids (step_3) are prescribed for an ASA I minor outpatient surgery
        3. Each analgesic agent specifies route and dose/regimen
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_asa_iv_emergency_destination_icu_with_critical_alerts():
    """ASA IV emergency surgery should be admitted to ICU with critical alerts."""
    state = _make_state(
        72,
        [{"name": "Heart failure", "severity": "severe", "controlled": False}],
        "IV",
        "Emergency open exploratory laparotomy for ischemic bowel",
        "emergency",
        "Severe uncontrolled heart failure with life-threatening systemic disease.",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert care.destination == "ICU"
    assert len(care.critical_alerts) > 0
    assert _prophylaxis_targets(care) == {"TEV", "IRAS", "NVPO"}

    text = _all_text(care)
    assert any(
        kw in text for kw in ("hemodynamic", "cardiac", "vasopressor", "icu", "intensive")
    ), "Plan must reference cardiac/hemodynamic monitoring or ICU-level support."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Destination is 'ICU' for ASA IV emergency surgery with severe uncontrolled heart failure.\n"
            "Critical Alerts list is non-empty and references cardiac/hemodynamic risk or ICU-level monitoring.\n"
            "Analgesia is multimodal and may include strong opioids when justified by severe pain.\n"
            "Prophylaxis covers TEV, IRAS, NVPO with patient-specific notes."
        ),
    )

    metric = GEval(
        name="ASA IV Emergency Destination and Alerts",
        criteria="""
        Evaluate whether:
        1. destination is 'ICU' for ASA IV emergency surgery with severe uncontrolled heart failure
        2. critical_alerts list is non-empty and references cardiac/hemodynamic risk
        3. destination_rationale clearly justifies the ICU placement using ASA, urgency and comorbidity
        4. The plan includes advanced monitoring or vasopressor readiness
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_chronic_kidney_disease_avoids_nsaids():
    """Patients with moderate chronic kidney disease should avoid NSAIDs or flag the conflict."""
    state = _make_state(
        65,
        [{"name": "Chronic kidney disease", "severity": "moderate", "controlled": True}],
        "III",
        "Elective open inguinal hernia repair",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    nsaid_keywords = ("ibuprofen", "ketorolac", "diclofenac", "naproxen", "nsaid", "aine")
    nsaid_agents = [
        a for a in care.analgesia
        if any(kw in (a.agent.lower() + " " + (a.notes or "").lower()) for kw in nsaid_keywords)
    ]

    if nsaid_agents:
        for a in nsaid_agents:
            assert a.notes and any(
                kw in a.notes.lower()
                for kw in ("renal", "kidney", "nephro", "avoid", "caution", "contraindic")
            ), f"NSAID '{a.agent}' prescribed without renal caveat."

    text = _all_text(care)
    assert any(
        kw in text for kw in ("renal", "kidney", "nephro")
    ), "Plan must explicitly address renal considerations."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Analgesia EXCLUDES NSAIDs (ibuprofen, ketorolac, diclofenac) for this CKD patient; "
            "paracetamol and other non-nephrotoxic agents are preferred.\n"
            "The plan references renal function review and avoidance of nephrotoxic agents.\n"
            "Recommendations or critical_alerts explicitly mention renal monitoring and CKD-related "
            "perioperative considerations."
        ),
    )

    metric = GEval(
        name="CKD NSAID Safety",
        criteria="""
        Evaluate whether:
        1. The analgesia plan EXCLUDES NSAIDs entirely OR documents an explicit renal
           contraindication note for any NSAID listed
        2. The plan references renal function review or avoidance of nephrotoxic agents
        3. Recommendations or critical_alerts include renal-specific perioperative considerations
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_high_apfel_risk_nvpo_multimodal_prophylaxis():
    """High Apfel score patient (female, non-smoker, prior NVPO, opioid use) should get multimodal NVPO prophylaxis."""
    state = _make_state(
        38,
        [
            {"name": "History of postoperative nausea and vomiting", "severity": "moderate", "controlled": False},
            {"name": "Motion sickness", "severity": "mild", "controlled": False},
        ],
        "II",
        "Elective laparoscopic cholecystectomy in non-smoker female with prior NVPO",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    nvpo_items = [p for p in care.prophylaxis if p.target == "NVPO"]
    assert len(nvpo_items) >= 1
    nvpo_text = " ".join((p.intervention + " " + (p.notes or "")).lower() for p in nvpo_items)
    assert any(
        kw in nvpo_text
        for kw in ("ondansetron", "dexamethasone", "droperidol", "scopolamine", "antiemetic")
    ), "NVPO prophylaxis must name at least one antiemetic agent."

    assert any(p.alert for p in nvpo_items), (
        "High Apfel score should be flagged with alert=True on NVPO prophylaxis."
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "NVPO prophylaxis names at least one specific antiemetic agent "
            "(ondansetron, dexamethasone, droperidol).\n"
            "NVPO prophylaxis is multimodal (combines two or more agents from different classes) "
            "given high Apfel score.\n"
            "At least one NVPO item has alert=True referencing the high Apfel score."
        ),
    )

    metric = GEval(
        name="High Apfel NVPO Coverage",
        criteria="""
        Evaluate whether:
        1. NVPO prophylaxis names at least one specific antiemetic agent
        2. Multimodal antiemetic strategy is used (two or more drug classes) for high Apfel
        3. The high Apfel score is explicitly acknowledged in notes or alert
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_coagulopathy_tev_conflict_flagged():
    """Active coagulopathy must trigger an alert on TEV pharmacological prophylaxis."""
    state = _make_state(
        58,
        [
            {"name": "Severe thrombocytopenia with active coagulopathy", "severity": "severe", "controlled": False}
        ],
        "III",
        "Elective open abdominal hysterectomy",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    tev_items = [p for p in care.prophylaxis if p.target == "TEV"]
    assert len(tev_items) >= 1
    assert any(p.alert for p in tev_items), (
        "Coagulopathy should trigger alert=True on TEV prophylaxis."
    )
    text = _all_text(care)
    assert any(
        kw in text
        for kw in ("mechanical", "compression", "stocking", "ipc", "bleeding", "coagulopath")
    ), "Plan must consider mechanical TEV prophylaxis or document the bleeding conflict."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "TEV prophylaxis has alert=True due to coagulopathy / bleeding risk.\n"
            "The plan favors mechanical measures (IPC, elastic stockings) over pharmacological "
            "anticoagulation, OR documents the bleeding-vs-clotting conflict explicitly.\n"
            "Notes describe the coagulopathy as the contraindication driver."
        ),
    )

    metric = GEval(
        name="Coagulopathy TEV Conflict",
        criteria="""
        Evaluate whether:
        1. TEV prophylaxis has at least one item with alert=True
        2. Mechanical prophylaxis (IPC or elastic stockings) is preferred or the bleeding conflict
           with anticoagulation is explicitly documented
        3. Notes explicitly reference the coagulopathy as the reason for the alert
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_eras_colorectal_specific_items():
    """Elective colorectal surgery should yield colorectal-specific ERAS items."""
    state = _make_state(
        62,
        [{"name": "Type 2 diabetes", "severity": "moderate", "controlled": True}],
        "II",
        "Elective laparoscopic colorectal resection (sigmoidectomy)",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    eras_text = " ".join(care.eras_recommendations).lower()
    full_text = _all_text(care)
    general_eras_keywords = (
        "early oral",
        "early diet",
        "early feed",
        "opioid-sparing",
        "opioid sparing",
        "euvolemic",
        "fluid",
        "drain",
        "catheter",
        "mobiliz",
    )
    assert sum(kw in eras_text for kw in general_eras_keywords) >= 3, (
        f"ERAS must include at least three core colorectal ERAS items. Got: {care.eras_recommendations}"
    )
    assert any(
        kw in full_text
        for kw in ("ileus", "bowel function", "gastrointestinal", "gut motility", "chewing gum")
    ), "ERAS for colorectal surgery must address ileus prevention or bowel function recovery."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "ERAS recommendations include at least three core ERAS items applied to colorectal "
            "surgery: early oral feeding, opioid-sparing multimodal analgesia, euvolemic fluid "
            "management, minimization of drains/catheters, and early mobilization.\n"
            "ERAS or recommendations also include at least one distinctive colorectal element such "
            "as postoperative ileus prevention, early bowel function recovery, chewing gum, or "
            "early urinary catheter removal.\n"
            "Glycemic control is mentioned because the patient has Type 2 diabetes."
        ),
    )

    metric = GEval(
        name="Colorectal ERAS Specificity",
        criteria="""
        Evaluate whether:
        1. ERAS recommendations contain at least three core ERAS items applicable to colorectal
           surgery (early oral feeding, opioid-sparing analgesia, euvolemic fluid management,
           drain/catheter minimization, early mobilization)
        2. ERAS or recommendations include at least one distinctive colorectal element
           (postoperative ileus prevention, early bowel function recovery, chewing gum, or
           early urinary catheter removal)
        3. Glycemic control is mentioned given the patient's Type 2 diabetes
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.8,
    )

    assert_test(test_case, [metric])


def test_pediatric_postoperative_care_age_appropriate():
    """A healthy child should get age-appropriate analgesia and PACU/ward destination."""
    state = _make_state(
        7,
        [],
        "I",
        "Elective tonsillectomy in healthy child",
        "elective",
        "Healthy pediatric patient with no systemic disease.",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert care.destination in ("PACU", "ward")
    assert any(a.who_step == "step_1" for a in care.analgesia)
    assert not any(a.who_step == "step_3" for a in care.analgesia), (
        "Strong opioids are not appropriate for a healthy pediatric tonsillectomy."
    )

    text = _all_text(care)
    assert any(
        kw in text
        for kw in ("weight", "pediatric", "child", "kg", "age-appropriate", "guardian", "parent")
    ), "Pediatric considerations (weight-based dosing, parental education) must be present."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Destination is 'PACU' or 'ward' for healthy pediatric tonsillectomy.\n"
            "Analgesia uses non-opioid agents with weight-based dosing (e.g., paracetamol "
            "10–15 mg/kg, ibuprofen 5–10 mg/kg).\n"
            "Strong opioids (step_3) are NOT prescribed.\n"
            "Follow-up plan includes parental education and signs of bleeding or dehydration."
        ),
    )

    metric = GEval(
        name="Pediatric Postoperative Plan",
        criteria="""
        Evaluate whether:
        1. destination is 'PACU' or 'ward' (not 'ICU') for a healthy ASA I tonsillectomy
        2. Analgesia uses weight-based dosing of non-opioid agents
        3. No strong opioids (WHO step_3) are prescribed
        4. The plan includes pediatric-specific elements (parental education, weight-based)
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_elderly_multimorbidity_destination_and_alerts():
    """Elderly ASA III patient with multiple comorbidities should receive escalated postop care."""
    state = _make_state(
        82,
        [
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": False},
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True},
            {"name": "Ischemic heart disease", "severity": "moderate", "controlled": True},
        ],
        "III",
        "Urgent open abdominal surgery for incarcerated hernia",
        "urgent",
        "Multiple moderate comorbidities in elderly patient.",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert care.destination in ("ICU", "PACU"), (
        "Elderly ASA III with multimorbidity and urgent open abdominal surgery should not go to general ward."
    )
    text = _all_text(care)
    assert any(kw in text for kw in ("renal", "kidney", "nephro"))
    assert any(kw in text for kw in ("glucose", "glycemic", "diabetes", "insulin"))
    assert any(kw in text for kw in ("cardiac", "ecg", "ischemic", "hemodynamic"))
    assert any(
        kw in text for kw in ("delirium", "frailty", "fragility", "polypharmacy", "elderly", "geriatric")
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Destination is 'ICU' or 'PACU' (never 'ward') for elderly ASA III urgent abdominal surgery.\n"
            "The plan addresses renal function (CKD), glycemic control (uncontrolled diabetes), "
            "and cardiac/hemodynamic monitoring (ischemic heart disease).\n"
            "Recommendations or critical_alerts mention elderly-specific risks "
            "(delirium, frailty, polypharmacy)."
        ),
    )

    metric = GEval(
        name="Elderly Multimorbidity Postop Plan",
        criteria="""
        Evaluate whether:
        1. destination is 'ICU' or 'PACU' (not 'ward') for this risk profile
        2. The plan addresses renal, glycemic, and cardiac considerations
        3. Recommendations or critical_alerts include elderly-specific risks
        4. ERAS or analgesia plan is adapted to renal function (e.g., NSAID caution)
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_severe_copd_thoracic_surgery_respiratory_focus():
    """Severe uncontrolled COPD undergoing thoracic surgery should escalate respiratory care."""
    state = _make_state(
        68,
        [{"name": "COPD", "severity": "severe", "controlled": False}],
        "III",
        "Elective open thoracotomy for lung lobectomy",
        "elective",
        "Severe uncontrolled COPD with major respiratory limitation.",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert care.destination in ("ICU", "PACU")
    text = _all_text(care)
    assert any(
        kw in text
        for kw in (
            "respiratory",
            "pulmonary",
            "bronchodilator",
            "oxygen",
            "ventilation",
            "physiotherapy",
            "spirometry",
            "incentive",
        )
    ), "Plan must include explicit respiratory rehabilitation/monitoring elements."
    mob_text = " ".join(care.early_mobilization).lower()
    assert any(
        kw in mob_text
        for kw in ("respiratory", "pulmonary", "physiotherapy", "breath", "spirometry")
    ), "Early mobilization must include respiratory physiotherapy for severe COPD + thoracotomy."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Destination is 'ICU' or 'PACU' (never 'ward') for severe COPD undergoing thoracotomy.\n"
            "Early mobilization includes respiratory physiotherapy, incentive spirometry, "
            "or breathing exercises.\n"
            "Plan references bronchodilator availability and oxygen/ventilation strategy.\n"
            "Critical alerts or recommendations address COPD exacerbation and atelectasis risk."
        ),
    )

    metric = GEval(
        name="Severe COPD Thoracic Postop Plan",
        criteria="""
        Evaluate whether:
        1. destination is 'ICU' or 'PACU' for severe COPD with thoracotomy
        2. Early mobilization includes respiratory physiotherapy or incentive spirometry
        3. Plan references bronchodilator availability and oxygen/ventilation strategy
        4. Critical alerts or recommendations address COPD exacerbation risk
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_discharge_criteria_use_appropriate_scale():
    """Discharge criteria must use Aldrete (PACU) or PADSS (home) coherent with destination."""
    state = _make_state(
        40,
        [],
        "I",
        "Elective day-case knee arthroscopy",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert len(care.discharge_criteria) >= 1
    scales = {d.scale for d in care.discharge_criteria}
    assert scales.issubset({"Aldrete", "PADSS"})
    for d in care.discharge_criteria:
        assert d.minimum_score >= 9, f"{d.scale} should target ≥9, got {d.minimum_score}"
        assert len(d.specific_criteria) >= 1, f"{d.scale} requires specific patient-tailored criteria"

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Discharge criteria include Aldrete (target ≥9) for PACU and/or PADSS (target ≥9) "
            "for ambulatory home discharge.\n"
            "Each scale lists patient-specific criteria such as pain control, oxygenation, "
            "ambulation, oral intake, nausea, and surgical site status."
        ),
    )

    metric = GEval(
        name="Discharge Criteria Scales",
        criteria="""
        Evaluate whether:
        1. Discharge criteria use Aldrete and/or PADSS appropriate to the care pathway
        2. Each scale targets minimum score ≥9
        3. Each scale lists at least one patient-tailored specific criterion
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_follow_up_plan_includes_medication_reconciliation():
    """Follow-up for a hypertensive diabetic must include medication reconciliation and education."""
    state = _make_state(
        58,
        [
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True},
            {"name": "Hypertension", "severity": "mild", "controlled": True},
        ],
        "II",
        "Elective laparoscopic cholecystectomy",
        "elective",
    )

    result = postoperative_care_node(state)
    care = result["postoperative_care"]

    assert len(care.follow_up_plan) >= 2
    follow_text = " ".join(care.follow_up_plan).lower()
    assert any(
        kw in follow_text
        for kw in ("medication", "reconciliation", "antihypertens", "antidiabet", "insulin")
    ), "Follow-up must reference medication reconciliation for chronic medications."
    assert any(
        kw in follow_text
        for kw in ("warning", "sign", "education", "wound", "return", "visit", "follow")
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(care),
        expected_output=(
            "Follow-up plan includes medication reconciliation for antihypertensive and "
            "antidiabetic agents.\n"
            "Follow-up plan schedules a return visit (e.g., 7–14 days post-op) and wound assessment.\n"
            "Patient education covers warning signs and activity restrictions."
        ),
    )

    metric = GEval(
        name="Follow-up Medication Reconciliation",
        criteria="""
        Evaluate whether:
        1. Follow-up plan explicitly mentions medication reconciliation for chronic drugs
        2. A scheduled return visit is included (timing specified)
        3. Patient education on warning signs is mentioned
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.85,
    )

    assert_test(test_case, [metric])
