"""Runner that turns a checklist :class:`Golden` into a populated
:class:`LLMTestCase` by invoking
:func:`graph.nodes.perioperative_checklist_node.perioperative_checklist_node`.
"""

from __future__ import annotations

import json

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from domain.schema.comorbidity import Comorbidity
from evals.metrics.clinical import CHECKLIST_OUTPUT_KEY
from graph.nodes.perioperative_checklist_node import perioperative_checklist_node
from graph.schema.ASA_output import ASAOutput
from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput


def _state_from_golden(golden: Golden) -> dict:
    payload = json.loads(golden.input)
    asa_payload = payload["asa"]
    return {
        "age": payload["age"],
        "comorbidities": [Comorbidity(**c) for c in payload.get("comorbidities", [])],
        "asa": ASAOutput(
            asa=asa_payload["asa"],
            confidence=asa_payload.get("confidence", 0.95),
            justification=asa_payload.get("justification", "baseline"),
        ),
    }


def _format_items(items) -> str:
    if not items:
        return "  (none)"
    return "\n".join(
        f"  [{item.item}] alert={item.alert} notes={item.notes}" for item in items
    )


def _format_list(values) -> str:
    if not values:
        return "  (none)"
    return "\n".join(f"  - {value}" for value in values)


def _format_checklist(checklist: PerioperativeChecklistOutput) -> str:
    return (
        f"Overall Status: {checklist.overall_status}\n"
        f"Sign-In:\n{_format_items(checklist.sign_in)}\n"
        f"Time-Out:\n{_format_items(checklist.time_out)}\n"
        f"Sign-Out:\n{_format_items(checklist.sign_out)}\n"
        f"Critical Alerts:\n{_format_list(checklist.critical_alerts)}\n"
        f"Recommendations:\n{_format_list(checklist.recommendations)}"
    )


async def build_checklist_test_case(golden: Golden) -> LLMTestCase:
    """Run the checklist node on ``golden`` and assemble an LLMTestCase."""
    state = _state_from_golden(golden)
    result = await perioperative_checklist_node(state)
    checklist: PerioperativeChecklistOutput = result["perioperative_checklist"]

    metadata = dict(golden.additional_metadata or {})
    metadata[CHECKLIST_OUTPUT_KEY] = checklist
    metadata.setdefault("comorbidity_names", [c.name for c in state["comorbidities"]])

    return LLMTestCase(
        input=golden.input,
        actual_output=_format_checklist(checklist),
        expected_output=golden.expected_output,
        additional_metadata=metadata,
    )


__all__ = ["build_checklist_test_case"]
