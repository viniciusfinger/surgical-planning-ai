"""Postoperative care node — DeepEval suite.

Pytest harness over :data:`evals.datasets.postop.POSTOP_GOLDENS`. The dataset
is shared with the standalone batch runner in :mod:`evals.run_postop`, so
the same regression seen in CI can be reproduced locally with
``uv run python -m evals.run_postop``.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.datasets.postop import POSTOP_GOLDENS
from evals.presets import postop_metrics
from evals.runners.postop import build_postop_test_case


@pytest.mark.parametrize(
    "golden",
    POSTOP_GOLDENS,
    ids=[g.additional_metadata["category"] for g in POSTOP_GOLDENS],
)
async def test_postoperative_care(golden: Golden) -> None:
    test_case = await build_postop_test_case(golden)
    assert_test(test_case, postop_metrics())
