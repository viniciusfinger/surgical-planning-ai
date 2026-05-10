"""Runner that turns a postop :class:`Golden` into a populated
:class:`LLMTestCase` by invoking
:func:`graph.nodes.postoperative_care_node.postoperative_care_node`.
"""

from __future__ import annotations

import json

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency
from evals.metrics.clinical import POSTOP_OUTPUT_KEY
from graph.nodes.postoperative_care_node import postoperative_care_node
from graph.schema.ASA_output import ASAOutput
from graph.schema.postoperative_care_output import PostoperativeCareOutput


def _state_from_golden(golden: Golden) -> dict:
    payload = json.loads(golden.input)
    asa_payload = payload["asa"]
    return {
        "age": payload["age"],
        "comorbidities": [Comorbidity(**c) for c in payload.get("comorbidities", [])],
        "surgical_type": payload["surgical_type"],
        "urgency": Urgency(payload["urgency"]),
        "asa": ASAOutput(
            asa=asa_payload["asa"],
            confidence=asa_payload.get("confidence", 0.95),
            justification=asa_payload.get("justification", "baseline"),
        ),
    }


def _format_list(values) -> str:
    if not values:
        return "  (none)"
    return "\n".join(f"  - {v}" for v in values)


def _format_postop(care: PostoperativeCareOutput) -> str:
    analgesia = "\n".join(
        f"  [{a.who_step}] {a.agent} ({a.route}, {a.dose_or_regimen}) notes={a.notes}"
        for a in care.analgesia_recommendation
    ) or "  (none)"

    prophylaxis = "\n".join(
        f"  [{p.target}] {p.intervention} alert={p.alert} notes={p.notes}"
        for p in care.prophylaxis_recommendation
    ) or "  (none)"

    discharge = "\n".join(
        f"  [{d.scale} ≥{d.minimum_score}] criteria="
        f"{'; '.join(d.specific_criteria) or '(none)'}"
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


async def build_postop_test_case(golden: Golden) -> LLMTestCase:
    """Run the postop node on ``golden`` and assemble an LLMTestCase."""
    state = _state_from_golden(golden)
    result = await postoperative_care_node(state)
    care: PostoperativeCareOutput = result["postoperative_care"]

    metadata = dict(golden.additional_metadata or {})
    metadata[POSTOP_OUTPUT_KEY] = care
    metadata.setdefault("comorbidity_names", [c.name for c in state["comorbidities"]])

    return LLMTestCase(
        input=golden.input,
        actual_output=_format_postop(care),
        expected_output=golden.expected_output,
        additional_metadata=metadata,
    )


__all__ = ["build_postop_test_case"]
