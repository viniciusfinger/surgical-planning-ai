"""Perioperative checklist node — DeepEval suite.

Pytest harness over :data:`evals.datasets.checklist.CHECKLIST_GOLDENS`. The
dataset is shared with the standalone batch runner in
:mod:`evals.run_checklist`, so the same regression seen in CI can be
reproduced locally with ``uv run python -m evals.run_checklist``.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.datasets.checklist import CHECKLIST_GOLDENS
from evals.presets import checklist_metrics
from evals.runners.checklist import build_checklist_test_case


@pytest.mark.parametrize(
    "golden",
    CHECKLIST_GOLDENS,
    ids=[g.additional_metadata["category"] for g in CHECKLIST_GOLDENS],
)
async def test_perioperative_checklist(golden: Golden) -> None:
    test_case = await build_checklist_test_case(golden)
    assert_test(test_case, checklist_metrics())


@pytest.mark.parametrize(
    "golden",
    [g for g in CHECKLIST_GOLDENS if g.additional_metadata.get("expect_no_critical_alerts")],
    ids=lambda g: g.additional_metadata["category"],
)
async def test_no_critical_alerts(golden: Golden) -> None:
    """Goldens flagged as ``expect_no_critical_alerts`` must keep that list empty.

    Kept as an explicit deterministic check (not via ``BaseMetric``) because
    it asserts a structural property of the output that should never be
    relaxed across runs — failing here is a hard guardrail.
    """
    test_case = await build_checklist_test_case(golden)
    checklist = test_case.additional_metadata["checklist_output"]
    assert checklist.critical_alerts == [], (
        f"Expected empty critical_alerts for {golden.additional_metadata['category']}, "
        f"got {checklist.critical_alerts}"
    )
