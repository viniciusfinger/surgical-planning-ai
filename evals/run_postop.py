"""Standalone evaluation script for the postoperative care node.

Run with::

    uv run python -m evals.run_postop
    uv run python -m evals.run_postop --concurrency 16
    EVAL_CONCURRENCY=16 uv run python -m evals.run_postop
"""

from __future__ import annotations

import asyncio

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig

from evals._runner_helpers import parse_concurrency
from evals.datasets.postop import POSTOP_GOLDENS
from evals.presets import postop_metrics
from evals.runners.postop import build_postop_test_case


async def _build_test_cases():
    return await asyncio.gather(
        *(build_postop_test_case(g) for g in POSTOP_GOLDENS)
    )


def main() -> None:
    concurrency = parse_concurrency()
    test_cases = asyncio.run(_build_test_cases())
    evaluate(
        test_cases=test_cases,
        metrics=postop_metrics(),
        async_config=AsyncConfig(run_async=concurrency > 1, max_concurrent=concurrency),
    )


if __name__ == "__main__":
    main()
