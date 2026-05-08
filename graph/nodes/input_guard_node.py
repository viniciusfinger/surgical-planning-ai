"""
Pre-LLM safety gate.
"""

from __future__ import annotations

from graph.state import GraphState
from safety.guards import validate_clinical_text_input


def input_guard_node(state: GraphState) -> dict:
    validate_clinical_text_input(state["surgical_type"], field="surgical_type")
    for index, comorbidity in enumerate(state["comorbidities"]):
        validate_clinical_text_input(
            comorbidity.name,
            field=f"comorbidities[{index}].name",
        )
    return {}
