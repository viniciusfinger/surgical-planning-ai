"""ASA classifier evaluation dataset.

Each :class:`Golden` carries a JSON ``input`` (the partial graph state fed to
:func:`graph.nodes.ASA_classifier_node.ASA_classifier_node`), an
``expected_output`` rendered as prose for ``GEval``, and a metadata dict that
the deterministic metrics in :mod:`evals.metrics.clinical` consume.

Categories double as ``pytest`` parametrization ids and as group keys for
aggregated reporting (``per_category`` averages).
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
    expected_asa: str | None = None,
    expected_asa_in: list[str] | None = None,
    expected_output: str,
    min_confidence: float = 0.85,
    extra_metadata: dict[str, Any] | None = None,
) -> Golden:
    if expected_asa is None and not expected_asa_in:
        raise ValueError(
            "Each ASA golden must declare expected_asa or expected_asa_in"
        )
    payload = {"age": age, "comorbidities": comorbidities}
    metadata: dict[str, Any] = {"category": category, "min_confidence": min_confidence}
    if expected_asa is not None:
        metadata["expected_asa"] = expected_asa
    if expected_asa_in:
        metadata["expected_asa_in"] = list(expected_asa_in)
    if extra_metadata:
        metadata.update(extra_metadata)
    return Golden(
        input=json.dumps(payload, ensure_ascii=False),
        expected_output=expected_output,
        additional_metadata=metadata,
    )


ASA_GOLDENS: list[Golden] = [
    _golden(
        category="healthy_adult",
        age=25,
        comorbidities=[],
        expected_asa="I",
        expected_output=(
            "ASA: I\n"
            "Healthy patient with no systemic disease."
        ),
    ),
    _golden(
        category="severe_uncontrolled_heart_failure",
        age=72,
        comorbidities=[
            {"name": "Heart Failure", "severity": "severe", "controlled": False}
        ],
        expected_asa="IV",
        expected_output=(
            "ASA: IV\n"
            "Severe uncontrolled heart failure represents a major systemic disease "
            "with possible threat to life."
        ),
        extra_metadata={
            "required_keyword_groups": [["heart", "failure"]],
        },
    ),
    _golden(
        category="mild_controlled_hypertension",
        age=48,
        comorbidities=[
            {"name": "Controlled hypertension", "severity": "mild", "controlled": True}
        ],
        expected_asa="II",
        expected_output=(
            "ASA: II\n"
            "Single mild systemic disease that is medically controlled warrants ASA II."
        ),
    ),
    _golden(
        category="moderate_controlled_copd",
        age=61,
        comorbidities=[{"name": "COPD", "severity": "moderate", "controlled": True}],
        expected_asa="II",
        expected_output=(
            "ASA: II\n"
            "Moderate but well-controlled COPD fits ASA II."
        ),
    ),
    _golden(
        category="moderate_uncontrolled_asthma",
        age=54,
        comorbidities=[{"name": "Asthma", "severity": "moderate", "controlled": False}],
        expected_asa="III",
        expected_output=(
            "ASA: III\n"
            "Moderate uncontrolled asthma implies severe systemic disease burden "
            "with functional limitation."
        ),
        extra_metadata={
            "required_keyword_groups": [
                ["control", "uncontrolled", "poorly"],
            ],
        },
    ),
    _golden(
        category="severe_controlled_ckd",
        age=69,
        comorbidities=[
            {"name": "Chronic kidney disease", "severity": "severe", "controlled": True}
        ],
        expected_asa="III",
        expected_output=(
            "ASA: III\n"
            "Severe CKD that is medically optimized remains severe systemic disease "
            "(ASA III)."
        ),
    ),
    _golden(
        category="elderly_no_comorbidities",
        age=88,
        comorbidities=[],
        expected_asa="I",
        expected_output=(
            "ASA: I\n"
            "Advanced age without listed systemic disease does not by itself elevate "
            "ASA class."
        ),
    ),
    _golden(
        category="pediatric_healthy",
        age=9,
        comorbidities=[],
        expected_asa="I",
        expected_output=(
            "ASA: I\n"
            "Healthy child with no systemic comorbidities."
        ),
    ),
    _golden(
        category="mild_uncontrolled_hypothyroidism_boundary",
        age=33,
        comorbidities=[
            {"name": "Hypothyroidism", "severity": "mild", "controlled": False}
        ],
        expected_asa="II",
        expected_output=(
            "ASA: II\n"
            "Mild hypothyroidism without adequate control counts as mild systemic "
            "disease. Perioperative risk remains limited compared with moderate or "
            "severe disease, so ASA II is appropriate."
        ),
    ),
    _golden(
        category="two_moderate_controlled_escalation",
        age=67,
        comorbidities=[
            {"name": "Type 2 diabetes mellitus", "severity": "moderate", "controlled": True},
            {"name": "Chronic kidney disease", "severity": "moderate", "controlled": True},
        ],
        expected_asa="III",
        expected_output=(
            "ASA: III\n"
            "Two or more moderate comorbidities escalate overall burden to ASA III."
        ),
    ),
    _golden(
        category="mixed_moderate_severe_uncontrolled",
        age=71,
        comorbidities=[
            {"name": "Ischemic heart disease", "severity": "moderate", "controlled": False},
            {"name": "Severe COPD", "severity": "severe", "controlled": False},
        ],
        expected_asa="IV",
        expected_output=(
            "ASA: IV\n"
            "Mixed moderate and severe uncontrolled cardiopulmonary disease warrants "
            "at least III; IV when life-threatening."
        ),
    ),
    _golden(
        category="moribund_septic_shock",
        age=58,
        comorbidities=[
            {
                "name": "Septic shock with multi-organ failure",
                "severity": "severe",
                "controlled": False,
            }
        ],
        # Boundary case — both IV and V are clinically defensible:
        #   - V: moribund patient not expected to survive without surgery.
        #   - IV: severe systemic disease that is a constant threat to life.
        # The exact label depends on whether mortality is judged imminent
        # without the surgical intervention. We accept either.
        expected_asa_in=["IV", "V"],
        expected_output=(
            "ASA: IV or V (boundary case).\n"
            "Septic shock with multi-organ failure justifies ASA V (moribund) when "
            "mortality without surgery is judged imminent, or ASA IV (constant threat "
            "to life) when the patient remains a surgical candidate after "
            "resuscitation. Both classifications are clinically acceptable as long "
            "as the justification cites septic shock and organ failure."
        ),
        extra_metadata={
            "required_keyword_groups": [
                ["septic", "shock", "organ", "failure"],
            ],
        },
    ),
]


asa_dataset = EvaluationDataset(goldens=ASA_GOLDENS)


__all__ = ["ASA_GOLDENS", "asa_dataset"]
