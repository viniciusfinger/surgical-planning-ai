"""Standalone evaluation script for the ASA classifier.

Run with::

    uv run python -m evals.run_asa
    uv run python -m evals.run_asa --concurrency 16
    EVAL_CONCURRENCY=16 uv run python -m evals.run_asa

It iterates over :data:`evals.datasets.asa.ASA_GOLDENS`, invokes the real
:func:`graph.nodes.ASA_classifier_node.ASA_classifier_node` for each one,
and pipes the resulting :class:`LLMTestCase`s into :func:`deepeval.evaluate`,
relying on DeepEval's built-in per-case report.
"""

from __future__ import annotations

import asyncio

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig

from evals._runner_helpers import parse_concurrency
from evals.datasets.asa import ASA_GOLDENS
from evals.presets import asa_metrics
from evals.runners.asa import build_asa_test_case


async def _build_test_cases():
    return await asyncio.gather(*(build_asa_test_case(g) for g in ASA_GOLDENS))


def main() -> None:
    concurrency = parse_concurrency()
    test_cases = asyncio.run(_build_test_cases())
    evaluate(
        test_cases=test_cases,
        metrics=asa_metrics(),
        async_config=AsyncConfig(run_async=concurrency > 1, max_concurrent=concurrency),
    )


if __name__ == "__main__":
    main()
