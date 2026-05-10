"""Standalone evaluation script for the perioperative checklist node.

Run with::

    uv run python -m evals.run_checklist
    uv run python -m evals.run_checklist --concurrency 16
    EVAL_CONCURRENCY=16 uv run python -m evals.run_checklist
"""

from __future__ import annotations

import asyncio

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig

from evals._runner_helpers import parse_concurrency
from evals.datasets.checklist import CHECKLIST_GOLDENS
from evals.presets import checklist_metrics
from evals.runners.checklist import build_checklist_test_case


async def _build_test_cases():
    return await asyncio.gather(
        *(build_checklist_test_case(g) for g in CHECKLIST_GOLDENS)
    )


def main() -> None:
    concurrency = parse_concurrency()
    test_cases = asyncio.run(_build_test_cases())
    evaluate(
        test_cases=test_cases,
        metrics=checklist_metrics(),
        async_config=AsyncConfig(run_async=concurrency > 1, max_concurrent=concurrency),
    )


if __name__ == "__main__":
    main()
