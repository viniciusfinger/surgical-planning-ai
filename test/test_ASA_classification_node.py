from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from graph.nodes.ASA_classifier_node import ASA_classifier_node


def test_asa_healthy_patient():
    state = {
        "age": 25,
        "comorbidities": [],
    }

    result = ASA_classifier_node(state)

    asa_output = result["asa"]

    assert asa_output.asa == "I"
    assert asa_output.confidence >= 0.7
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
        threshold=0.85,
    )

    assert_test(test_case, [metric])


def test_asa_severe_uncontrolled():
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
    assert asa_output.confidence >= 0.8
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
        threshold=0.85,
    )

    assert_test(test_case, [metric])