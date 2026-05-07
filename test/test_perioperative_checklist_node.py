from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from graph.nodes.perioperative_checklist_node import perioperative_checklist_node
from graph.schema.ASA_output import ASAOutput


def _make_state(age: int, comorbidities: list, asa_class: str, justification: str = ""):
    return {
        "age": age,
        "comorbidities": comorbidities,
        "asa": ASAOutput(
            asa=asa_class,
            confidence=0.95,
            justification=justification or f"ASA {asa_class} assigned based on provided comorbidities.",
        ),
    }


def _format_output(checklist) -> str:
    sign_in = "\n".join(
        f"  [{item.item}] alert={item.alert} notes={item.notes}" for item in checklist.sign_in
    )
    time_out = "\n".join(
        f"  [{item.item}] alert={item.alert} notes={item.notes}" for item in checklist.time_out
    )
    sign_out = "\n".join(
        f"  [{item.item}] alert={item.alert} notes={item.notes}" for item in checklist.sign_out
    )
    critical = "\n".join(f"  - {a}" for a in checklist.critical_alerts)
    recommendations = "\n".join(f"  - {r}" for r in checklist.recommendations)
    return (
        f"Overall Status: {checklist.overall_status}\n"
        f"Sign-In:\n{sign_in}\n"
        f"Time-Out:\n{time_out}\n"
        f"Sign-Out:\n{sign_out}\n"
        f"Critical Alerts:\n{critical}\n"
        f"Recommendations:\n{recommendations}"
    )


def test_healthy_adult_asa_i_clear_status():
    """Healthy adult with no comorbidities should yield overall_status='clear'."""
    state = _make_state(32, [], "I", "No systemic disease.")

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status == "clear"
    assert len(checklist.sign_in) > 0
    assert len(checklist.time_out) > 0
    assert len(checklist.sign_out) > 0
    assert len(checklist.critical_alerts) == 0

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: clear\n"
            "All three phases populated with standard WHO checklist items.\n"
            "No critical alerts for a healthy ASA I patient.\n"
            "Recommendations may include routine postoperative monitoring."
        ),
    )

    metric = GEval(
        name="Healthy Patient Checklist Evaluation",
        criteria="""
        Evaluate whether:
        1. overall_status is 'clear' for a healthy ASA I patient
        2. All three phases (Sign-In, Time-Out, Sign-Out) have at least one item
        3. No critical alerts are present
        4. Items are clinically relevant and not overly generic
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


def test_mild_controlled_hypertension_asa_ii_clear_or_hold():
    """Mild controlled hypertension (ASA II) may produce hold status with relevant alerts."""
    state = _make_state(
        52,
        [{"name": "Controlled hypertension", "severity": "mild", "controlled": True}],
        "II",
        "Single mild controlled systemic disease.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status in ("clear", "hold")
    assert len(checklist.sign_in) > 0
    assert len(checklist.time_out) > 0
    assert len(checklist.sign_out) > 0

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: clear or hold\n"
            "Sign-In includes blood pressure verification and antihypertensive medication review.\n"
            "No critical alerts expected for a controlled mild condition.\n"
            "Recommendations include perioperative blood pressure monitoring."
        ),
    )

    metric = GEval(
        name="Mild Hypertension Checklist Evaluation",
        criteria="""
        Evaluate whether:
        1. The checklist includes blood pressure-related items in Sign-In or Time-Out
        2. overall_status is 'clear' or 'hold' (not 'critical') for mild controlled disease
        3. Relevant notes reference hypertension monitoring or antihypertensive management
        4. The checklist is not identical to a healthy patient's checklist
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


def test_all_three_phases_populated():
    """All three checklist phases must be populated for any patient."""
    state = _make_state(45, [], "I")

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert len(checklist.sign_in) >= 1
    assert len(checklist.time_out) >= 1
    assert len(checklist.sign_out) >= 1

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Sign-In, Time-Out, and Sign-Out phases each contain at least one checklist item "
            "that reflects the WHO Surgical Safety Checklist structure."
        ),
    )

    metric = GEval(
        name="Three-Phase Structure Completeness",
        criteria="""
        Evaluate whether:
        1. Sign-In, Time-Out, and Sign-Out phases each have at least one distinct item
        2. Items are distributed correctly across phases (not duplicated verbatim)
        3. Phase names correspond to the correct timing in the surgical workflow
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


def test_moderate_uncontrolled_copd_asa_iii_airway_alert():
    """Moderate uncontrolled COPD (ASA III) should trigger airway or respiratory alerts."""
    state = _make_state(
        63,
        [{"name": "COPD", "severity": "moderate", "controlled": False}],
        "III",
        "Moderate uncontrolled COPD implies severe systemic limitation.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status in ("hold", "critical")

    all_items = checklist.sign_in + checklist.time_out + checklist.sign_out
    all_notes = " ".join(
        (item.notes or "").lower() + item.item.lower() for item in all_items
    )
    assert any(
        keyword in all_notes
        for keyword in ("airway", "respiratory", "pulmonary", "copd", "bronchospasm", "oxygen")
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: hold or critical\n"
            "Sign-In includes difficult airway assessment and bronchodilator availability.\n"
            "Time-Out references oxygen reserve and ventilation plan.\n"
            "Alerts or recommendations address COPD exacerbation risk."
        ),
    )

    metric = GEval(
        name="Uncontrolled COPD Respiratory Safety Evaluation",
        criteria="""
        Evaluate whether:
        1. overall_status is 'hold' or 'critical' given moderate uncontrolled COPD and ASA III
        2. At least one checklist item addresses respiratory/airway risk
        3. Notes explain the specific COPD-related risk and suggested precaution
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


def test_severe_uncontrolled_heart_failure_asa_iv_critical_status():
    """Severe uncontrolled heart failure (ASA IV) should result in critical overall_status."""
    state = _make_state(
        74,
        [{"name": "Heart failure", "severity": "severe", "controlled": False}],
        "IV",
        "Severe uncontrolled heart failure is a life-threatening systemic disease.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status in ("hold", "critical")
    assert len(checklist.critical_alerts) > 0 or checklist.overall_status == "critical"

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: critical\n"
            "Critical Alerts include hemodynamic instability warning and cardiac decompensation risk.\n"
            "Sign-In includes echocardiography review and hemodynamic monitoring setup.\n"
            "Recommendations include cardiology consult before proceeding."
        ),
    )

    metric = GEval(
        name="Severe Heart Failure Critical Safety Evaluation",
        criteria="""
        Evaluate whether:
        1. overall_status is 'hold' or 'critical' for ASA IV severe uncontrolled heart failure
        2. critical_alerts or alert=True items reference cardiac/hemodynamic risk
        3. At least one item addresses pre-induction cardiac stability verification
        4. Recommendations include specialist consultation or advanced monitoring
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


def test_recommendations_present_for_asa_ii_plus():
    """Patients with ASA II or above should always receive non-blocking recommendations."""
    state = _make_state(
        60,
        [{"name": "Type 2 diabetes", "severity": "moderate", "controlled": True}],
        "II",
        "Moderate controlled diabetes is mild systemic disease.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert len(checklist.recommendations) > 0

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Recommendations include glucose monitoring, glycemic control targets, "
            "and wound healing considerations for a diabetic patient."
        ),
    )

    metric = GEval(
        name="Diabetes Recommendations Evaluation",
        criteria="""
        Evaluate whether:
        1. At least one recommendation is present for a diabetic ASA II patient
        2. Recommendations address diabetes-specific perioperative concerns
           (e.g., glucose monitoring, insulin management, wound care)
        3. Recommendations are non-blocking (informational, not critical alerts)
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


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_pediatric_healthy_patient():
    """Healthy child (ASA I) should get age-appropriate items without critical alerts."""
    state = _make_state(7, [], "I", "Healthy pediatric patient with no systemic disease.")

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status == "clear"
    assert len(checklist.critical_alerts) == 0

    all_items_text = " ".join(
        item.item.lower() + " " + (item.notes or "").lower()
        for item in checklist.sign_in + checklist.time_out + checklist.sign_out
    )
    assert any(
        keyword in all_items_text
        for keyword in ("pediatric", "child", "weight", "age", "consent", "parent", "guardian")
    )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: clear\n"
            "Sign-In includes weight-based dosing verification and parental/guardian consent.\n"
            "Items are tailored to pediatric context (age-appropriate drug doses, small airway considerations).\n"
            "No critical alerts."
        ),
    )

    metric = GEval(
        name="Pediatric Healthy Patient Checklist",
        criteria="""
        Evaluate whether:
        1. overall_status is 'clear' for a healthy 7-year-old with no comorbidities
        2. At least one item reflects pediatric-specific concerns
           (e.g., weight-based dosing, parental consent, small airway)
        3. No critical alerts are present
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


def test_elderly_multiple_comorbidities_asa_iii_hold():
    """Elderly patient with multiple moderate comorbidities (ASA III) should produce hold status."""
    state = _make_state(
        81,
        [
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": True},
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True},
        ],
        "III",
        "Two moderate comorbidities in an elderly patient escalate to ASA III.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status in ("hold", "critical")

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: hold\n"
            "Sign-In includes renal function review and nephrotoxic agent avoidance.\n"
            "Sign-In or Time-Out includes glucose level verification.\n"
            "Recommendations address elderly-specific perioperative risks."
        ),
    )

    metric = GEval(
        name="Elderly Multi-Morbidity Checklist Evaluation",
        criteria="""
        Evaluate whether:
        1. overall_status is 'hold' or 'critical' for ASA III elderly patient
        2. At least one item addresses renal considerations (CKD)
        3. At least one item addresses glycemic control (diabetes)
        4. Items or recommendations reflect elderly-specific risks
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


def test_moribund_patient_asa_v_critical_status_and_alerts():
    """Moribund ASA V patient should yield critical status with multiple critical alerts."""
    state = _make_state(
        59,
        [{"name": "Septic shock with multi-organ failure", "severity": "severe", "controlled": False}],
        "V",
        "Moribund patient not expected to survive without immediate intervention.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status == "critical"
    assert len(checklist.critical_alerts) > 0

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: critical\n"
            "Critical Alerts reference hemodynamic instability, vasopressor dependence, and organ failure.\n"
            "Sign-In includes ICU-level monitoring requirements and resuscitation readiness.\n"
            "Multiple items carry alert=True."
        ),
    )

    metric = GEval(
        name="Moribund ASA V Critical Checklist",
        criteria="""
        Evaluate whether:
        1. overall_status is 'critical' for a moribund ASA V patient
        2. critical_alerts list is non-empty and references septic shock or organ failure
        3. Multiple checklist items have alert=True
        4. Notes describe urgent interventions (vasopressors, ICU handover, hemodynamic monitoring)
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


def test_alert_items_always_have_notes():
    """Any checklist item with alert=True must have non-empty notes."""
    state = _make_state(
        68,
        [{"name": "Severe COPD", "severity": "severe", "controlled": False}],
        "III",
        "Severe uncontrolled COPD is a severe systemic limitation.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    all_items = checklist.sign_in + checklist.time_out + checklist.sign_out
    for item in all_items:
        if item.alert:
            assert item.notes is not None and len(item.notes.strip()) > 0, (
                f"Item '{item.item}' has alert=True but missing notes."
            )

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Every item with alert=True includes a non-empty notes field explaining "
            "the specific risk and the recommended corrective action or precaution."
        ),
    )

    metric = GEval(
        name="Alert Notes Completeness",
        criteria="""
        Evaluate whether:
        1. Every item where alert=True has a notes field
        2. Notes for alerted items include both the risk identified and the recommended action
        3. Notes are clinically specific (not generic statements like "this is important")
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


def test_two_comorbidities_checklist_covers_both():
    """Checklist should reference both conditions when two comorbidities are present."""
    state = _make_state(
        66,
        [
            {"name": "Ischemic heart disease", "severity": "moderate", "controlled": True},
            {"name": "Type 2 diabetes", "severity": "moderate", "controlled": False},
        ],
        "III",
        "Two moderate comorbidities; uncontrolled diabetes escalates concern.",
    )

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    all_text = " ".join(
        item.item.lower() + " " + (item.notes or "").lower()
        for item in checklist.sign_in + checklist.time_out + checklist.sign_out
    ) + " ".join(checklist.critical_alerts).lower() + " ".join(checklist.recommendations).lower()

    has_cardiac = any(kw in all_text for kw in ("cardiac", "coronary", "ischemic", "heart", "ecg"))
    has_glucose = any(kw in all_text for kw in ("glucose", "glycemic", "diabetes", "insulin", "blood sugar"))

    assert has_cardiac, "Checklist should address ischemic heart disease."
    assert has_glucose, "Checklist should address uncontrolled diabetes."

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Checklist covers both ischemic heart disease (ECG review, anti-anginal medication, "
            "hemodynamic monitoring) and uncontrolled diabetes (pre-op glucose, insulin protocol, "
            "glycemic targets). overall_status is 'hold' or 'critical'."
        ),
    )

    metric = GEval(
        name="Dual Comorbidity Coverage Evaluation",
        criteria="""
        Evaluate whether:
        1. At least one item or note references cardiac/ischemic heart disease management
        2. At least one item or note references glucose/glycemic/diabetes management
        3. overall_status reflects the elevated risk from dual uncontrolled/moderate conditions
        4. Recommendations address at least one of the two comorbidities
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


def test_no_critical_alerts_for_asa_i():
    """ASA I patients must not produce critical_alerts."""
    state = _make_state(28, [], "I", "Healthy young adult with no systemic disease.")

    result = perioperative_checklist_node(state)
    checklist = result["perioperative_checklist"]

    assert checklist.overall_status == "clear"
    assert checklist.critical_alerts == []

    test_case = LLMTestCase(
        input=str(state),
        actual_output=_format_output(checklist),
        expected_output=(
            "Overall Status: clear\n"
            "Critical Alerts: none\n"
            "Standard WHO checklist items for a healthy patient."
        ),
    )

    metric = GEval(
        name="No Critical Alerts for ASA I",
        criteria="""
        Evaluate whether:
        1. critical_alerts is empty for a healthy ASA I patient
        2. overall_status is 'clear'
        3. No individual item has alert=True that would imply a critical safety conflict
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
