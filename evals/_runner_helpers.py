"""
Shared helpers for the ``evals.run_*`` scripts.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_CONCURRENCY = 8


def parse_concurrency(default: int = DEFAULT_CONCURRENCY) -> int:
    """Pick the concurrency level for an ``evals.run_*`` invocation.

    Resolution order:

    1. ``--concurrency N`` CLI flag (or ``-c N``).
    2. ``EVAL_CONCURRENCY`` env var.
    3. ``default`` (8).

    A value of ``0`` or ``1`` disables async parallelism — useful when
    debugging a single golden.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--concurrency", "-c", type=int, default=None)
    args, _ = parser.parse_known_args()

    if args.concurrency is not None:
        return max(1, args.concurrency)

    env = os.environ.get("EVAL_CONCURRENCY")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass

    return default


__all__ = ["DEFAULT_CONCURRENCY", "parse_concurrency"]
