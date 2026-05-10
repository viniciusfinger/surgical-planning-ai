"""Runner that turns an ASA :class:`Golden` into a populated
:class:`LLMTestCase` by invoking the real
:func:`graph.nodes.ASA_classifier_node.ASA_classifier_node`.
"""

from __future__ import annotations

import json

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from domain.schema.comorbidity import Comorbidity
from evals.metrics.clinical import ASA_OUTPUT_KEY
from graph.nodes.ASA_classifier_node import ASA_classifier_node
from graph.schema.ASA_output import ASAOutput


def _state_from_golden(golden: Golden) -> dict:
    payload = json.loads(golden.input)
    return {
        "age": payload["age"],
        "comorbidities": [Comorbidity(**c) for c in payload.get("comorbidities", [])],
    }


def _format_asa(asa: ASAOutput) -> str:
    return (
        f"ASA: {asa.asa}\n"
        f"Confidence: {asa.confidence:.2f}\n"
        f"Justification: {asa.justification}"
    )


async def build_asa_test_case(golden: Golden) -> LLMTestCase:
    """
    Run the ASA classifier on ``golden`` and assemble an LLMTestCase.
    """
    state = _state_from_golden(golden)
    result = await ASA_classifier_node(state)
    asa: ASAOutput = result["asa"]

    metadata = dict(golden.additional_metadata or {})
    metadata[ASA_OUTPUT_KEY] = asa
    metadata.setdefault("comorbidity_names", [c.name for c in state["comorbidities"]])

    return LLMTestCase(
        input=golden.input,
        actual_output=_format_asa(asa),
        expected_output=golden.expected_output,
        additional_metadata=metadata,
    )


__all__ = ["build_asa_test_case"]
