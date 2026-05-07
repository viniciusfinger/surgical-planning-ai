from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from graph.nodes.ASA_classifier_node import ASA_classifier_node


def test_asa_healthy_patient():
    """Test ASA for healthy patient"""
    state = {
        "age": 25,
        "comorbidities": [],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "I"
    assert asa_output.confidence >= 0.85
    assert len(asa_output.justification) > 10

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: I
        Healthy patient with no systemic disease.
        """,
    )

    metric = GEval(
        name="ASA Clinical Reasoning",
        criteria="""
        Evaluate whether:

        1. The ASA classification is clinically correct
        2. The confidence score is coherent with the certainty of the case
        3. The justification clearly references:
           - absence or presence of disease
           - severity
           - disease control
        4. The reasoning follows ASA guidelines
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


def test_asa_severe_uncontrolled():
    """Test ASA for severe uncontrolled heart failure"""
    state = {
        "age": 72,
        "comorbidities": [
            {
                "name": "Heart Failure",
                "severity": "severe",
                "controlled": False,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "IV"
    assert asa_output.confidence >= 0.85
    assert "heart" in asa_output.justification.lower() or "failure" in asa_output.justification.lower()

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: IV
        Severe uncontrolled heart failure represents a major systemic disease
        with possible threat to life.
        """,
    )

    metric = GEval(
        name="Severe Disease Clinical Evaluation",
        criteria="""
        Evaluate whether:

        1. Severe uncontrolled heart failure is classified appropriately
        2. ASA IV is preferred for life-threatening disease
        3. The justification references:
           - severity
           - lack of control
           - systemic impact
        4. Confidence should be high for this straightforward severe case
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


def test_asa_mild_controlled_single_condition():
    """Test ASA for mild controlled hypertension"""
    state = {
        "age": 48,
        "comorbidities": [
            {
                "name": "Controlled hypertension",
                "severity": "mild",
                "controlled": True,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "II"
    assert asa_output.confidence >= 0.85
    assert len(asa_output.justification) > 10

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: II
        Single mild systemic disease that is medically controlled warrants ASA II.
        """,
    )

    metric = GEval(
        name="Mild Controlled Clinical Evaluation",
        criteria="""
        Evaluate whether:
        1. Mild controlled hypertension is ASA II per guidelines
        2. Confidence matches a straightforward mild controlled case
        3. Justification cites control and mild severity
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


def test_asa_moderate_controlled_single_condition():
    """Test ASA for moderate controlled COPD"""
    state = {
        "age": 61,
        "comorbidities": [
            {
                "name": "COPD",
                "severity": "moderate",
                "controlled": True,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "II"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: II
        Moderate but well-controlled COPD fits ASA II.
        """,
    )

    metric = GEval(
        name="Moderate Controlled COPD Evaluation",
        criteria="""
        Evaluate whether:
        1. Moderate COPD under control is ASA II according to stated rules
        2. Justification references severity (moderate) and control status
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


def test_asa_moderate_uncontrolled():
    """Test ASA for moderate uncontrolled asthma"""
    state = {
        "age": 54,
        "comorbidities": [
            {
                "name": "Asthma",
                "severity": "moderate",
                "controlled": False,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "III"
    assert asa_output.confidence >= 0.85
    justification_lower = asa_output.justification.lower()
    assert "control" in justification_lower or "uncontrolled" in justification_lower or "poorly" in justification_lower

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: III
        Moderate uncontrolled asthma implies severe systemic disease burden with functional limitation.
        """,
    )

    metric = GEval(
        name="Moderate Uncontrolled Evaluation",
        criteria="""
        Evaluate whether:
        Moderate uncontrolled disease maps to ASA III
        Justification addresses clinical impact
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


def test_asa_severe_controlled():
    """Test ASA for severe controlled chronic kidney disease"""
    state = {
        "age": 69,
        "comorbidities": [
            {
                "name": "Chronic kidney disease",
                "severity": "severe",
                "controlled": True,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "III"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: III
        Severe CKD that is medically optimized remains severe systemic disease (ASA III).
        """,
    )

    metric = GEval(
        name="Severe Controlled Evaluation",
        criteria="""
        Evaluate whether:
        1. Severe controlled comorbidity is ASA III not IV
        2. Justification distinguishes severe disease from imminent life threat
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


def test_asa_elderly_no_comorbidities_age_alone_not_enough():
    """Test ASA for elderly patient with no comorbidities"""
    state = {
        "age": 88,
        "comorbidities": [],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "I"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: I
        Advanced age without listed systemic disease does not by itself elevate ASA class.
        """,
    )

    metric = GEval(
        name="Elderly Without Disease Edge Case",
        criteria="""
        Evaluate whether:
        1. ASA I is appropriate when no comorbidities are listed regardless of age
        2. Justification explicitly notes age does not alone define ASA
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


def test_asa_pediatric_healthy():
    """Test ASA for pediatric healthy patient"""
    state = {
        "age": 9,
        "comorbidities": [],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "I"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: I
        Healthy child with no systemic comorbidities.
        """,
    )

    metric = GEval(
        name="Pediatric Healthy Patient",
        criteria="""
        Evaluate whether:
        1. No comorbidities yields ASA I
        2. Reasoning is coherent for a pediatric healthy presentation
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


def test_asa_mild_uncontrolled_boundary():
    """Test ASA for mild uncontrolled hypothyroidism"""
    state = {
        "age": 33,
        "comorbidities": [
            {
                "name": "Hypothyroidism",
                "severity": "mild",
                "controlled": False,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "II"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: II
        Mild hypothyroidism without adequate control counts as mild systemic disease. Perioperative
        risk remains limited compared with moderate/severe disease, so ASA II is appropriate:
        the uncontrolled status is acknowledged, but there is no indication of functional
        limitation or life-threatening instability that would justify III.
        """,
    )

    metric = GEval(
        name="Mild Uncontrolled Boundary Case",
        criteria="""
        Evaluate whether:
        1. Classification stays with II for mild uncontrolled disease (boundary case)
        2. Justification explains why II (severity mild, lack of control, limited perioperative impact)
        3. Do not penalize a longer or more detailed Actual justification if it agrees with II
           and the same clinical reasoning as the Expected Output
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


def test_asa_two_moderate_controlled_escalation():
    """Test ASA for two moderate controlled comorbidities"""
    state = {
        "age": 67,
        "comorbidities": [
            {
                "name": "Type 2 diabetes mellitus",
                "severity": "moderate",
                "controlled": True,
            },
            {
                "name": "Chronic kidney disease",
                "severity": "moderate",
                "controlled": True,
            },
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "III"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: III
        Two or more moderate comorbidities escalate overall burden to ASA III.
        """,
    )

    metric = GEval(
        name="Two Moderate Comorbidities Rule",
        criteria="""
        Evaluate whether:
        1. ≥2 moderate conditions produce ASA III per multi-morbidity rule
        2. Justification references combined burden or multiple moderate diseases
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


def test_asa_mixed_moderate_severe_uncontrolled_range():
    """Test ASA for mixed moderate and severe uncontrolled cardiopulmonary disease"""

    state = {
        "age": 71,
        "comorbidities": [
            {
                "name": "Ischemic heart disease",
                "severity": "moderate",
                "controlled": False,
            },
            {
                "name": "Severe COPD",
                "severity": "severe",
                "controlled": False,
            },
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "IV"
    assert asa_output.confidence >= 0.85

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: IV
        Mixed moderate and severe uncontrolled cardiopulmonary disease warrants at least III; IV if life-threatening.
        """,
    )

    metric = GEval(
        name="Mixed Moderate Severe Evaluation",
        criteria="""
        Evaluate whether:
        1. Combined moderate + severe pathology is classified III or IV appropriately
        2. Justification cites severity, control, and systemic risk
        3. Conservative bias toward higher class when uncertainty exists is respected
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


def test_asa_moribund_septic_shock():
    """Test ASA for moribund patient with septic shock and multi-organ failure"""
    
    state = {
        "age": 58,
        "comorbidities": [
            {
                "name": "Septic shock with multi-organ failure",
                "severity": "severe",
                "controlled": False,
            }
        ],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "V"
    assert asa_output.confidence >= 0.85
    j = asa_output.justification.lower()
    assert "septic" in j or "shock" in j or "organ" in j or "failure" in j

    test_case = LLMTestCase(
        input=str(state),
        actual_output=f"""
        ASA: {asa_output.asa}
        Confidence: {asa_output.confidence}
        Justification: {asa_output.justification}
        """,
        expected_output="""
        ASA: V
        Moribund patient with septic shock and organ failure not expected to survive without immediate intervention.
        """,
    )

    metric = GEval(
        name="Moribund Septic Shock Case",
        criteria="""
        Evaluate whether:
        1. Imminent perioperative mortality risk is reflected as ASA V
        2. Justification matches extreme physiologic collapse
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