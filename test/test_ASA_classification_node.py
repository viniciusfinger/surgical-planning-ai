"""ASA classifier node — DeepEval suite.

This file used to inline the goldens, the metrics and the assertion in one
function per scenario. The whole dataset lives in :mod:`evals.datasets.asa` and
is shared by the pytest harness and the standalone batch runner in
:mod:`evals.run_asa`.

"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.datasets.asa import ASA_GOLDENS
from evals.presets import asa_metrics
from evals.runners.asa import build_asa_test_case


@pytest.mark.parametrize(
    "golden",
    ASA_GOLDENS,
    ids=[g.additional_metadata["category"] for g in ASA_GOLDENS],
)
async def test_asa_classifier(golden: Golden) -> None:
    test_case = await build_asa_test_case(golden)
    assert_test(test_case, asa_metrics())
