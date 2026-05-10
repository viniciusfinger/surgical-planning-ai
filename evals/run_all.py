"""Run all three node-level evaluations sequentially.
Useful for end-of-day regression sweeps and CI nightly jobs:
    uv run python -m evals.run_all
"""

from __future__ import annotations

from evals.run_asa import main as run_asa
from evals.run_checklist import main as run_checklist
from evals.run_postop import main as run_postop


def main() -> None:
    run_asa()
    run_checklist()
    run_postop()


if __name__ == "__main__":
    main()
