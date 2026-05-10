"""
Reusable metric presets for the surgical-planning evaluations.
"""

from __future__ import annotations

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import SingleTurnParams

from evals.metrics.clinical import (
    AlertNotesCompletenessMetric,
    ASAConfidenceMinMetric,
    ASAExactMatchMetric,
    DestinationSafetyMetric,
    KeywordCoverageMetric,
    NsaidContraindicationMetric,
    OverallStatusMetric,
    PhaseCompletenessMetric,
    ProphylaxisCoverageMetric,
    WHOLadderMetric,
)

JUDGE_MODEL = "gpt-4o-mini"

"""
GEval thresholds are calibrated per node based on the empirical score
distribution of the first full run. The deterministic clinical metrics
above remain at threshold=1.0 — they are the hard KPIs of the product.
GEval is the "spice": a softer, subjective signal whose threshold is
pegged a bit below the observed median to absorb judge variance while
still flagging genuine regressions.
"""
GEVAL_THRESHOLD_ASA = 0.80
GEVAL_THRESHOLD_CHECKLIST = 0.75
GEVAL_THRESHOLD_POSTOP = 0.80


def asa_metrics() -> list[BaseMetric]:
    return [
        ASAExactMatchMetric(),
        ASAConfidenceMinMetric(),
        KeywordCoverageMetric(),
        GEval(
            name="ASA Clinical Reasoning",
            criteria=(
                "Evaluate whether: 1. The ASA classification is clinically correct. "
                "2. The confidence score is coherent with the certainty of the case. "
                "3. The justification clearly references absence/presence of disease, "
                "severity, and disease control. 4. The reasoning follows ASA guidelines. "
                "Note: when the Expected Output explicitly mentions that two adjacent "
                "ASA classes are both clinically defensible (e.g., 'IV or V'), do not "
                "penalize the Actual Output for picking either of them, as long as the "
                "justification is clinically coherent."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=JUDGE_MODEL,
            threshold=GEVAL_THRESHOLD_ASA,
        ),
    ]


def checklist_metrics() -> list[BaseMetric]:
    return [
        OverallStatusMetric(),
        PhaseCompletenessMetric(),
        AlertNotesCompletenessMetric(),
        KeywordCoverageMetric(),
        GEval(
            name="Checklist Clinical Coverage",
            criteria=(
                "Evaluate whether: 1. overall_status reflects the patient's risk level "
                "(clear / hold / critical). 2. Items are clinically relevant (not "
                "generic). 3. Critical alerts are consistent with overall_status. "
                "4. Recommendations cover the comorbidities in the input. "
                "Do not penalize formatting differences or reasonable alternative "
                "phrasings as long as the clinical content is sound."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=JUDGE_MODEL,
            threshold=GEVAL_THRESHOLD_CHECKLIST,
        ),
    ]


def postop_metrics() -> list[BaseMetric]:
    return [
        DestinationSafetyMetric(),
        ProphylaxisCoverageMetric(),
        NsaidContraindicationMetric(),
        WHOLadderMetric(),
        KeywordCoverageMetric(),
        GEval(
            name="Postop Clinical Coverage",
            criteria=(
                "Evaluate whether: 1. Destination matches risk profile (ASA, urgency, "
                "comorbidities). 2. Analgesia is multimodal and starts with WHO step_1. "
                "3. Prophylaxis covers the targets clinically appropriate to the case "
                "(omitting TEV in healthy ambulatory pediatric surgery is acceptable). "
                "4. ERAS items are specialty-appropriate. 5. Discharge criteria use the "
                "right scale. 6. Follow-up plan is concrete (timing, medication "
                "reconciliation, education). "
                "Do not penalize formatting differences or reasonable alternative "
                "phrasings as long as the clinical content is sound."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=JUDGE_MODEL,
            threshold=GEVAL_THRESHOLD_POSTOP,
        ),
    ]


__all__ = [
    "asa_metrics",
    "checklist_metrics",
    "postop_metrics",
    "JUDGE_MODEL",
    "GEVAL_THRESHOLD_ASA",
    "GEVAL_THRESHOLD_CHECKLIST",
    "GEVAL_THRESHOLD_POSTOP",
]
