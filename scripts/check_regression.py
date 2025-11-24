#!/usr/bin/env python3
"""Script to check for performance regression in benchmarks."""

import sys
from pathlib import Path

from llmchess.benchmark import BenchmarkHarness


def main() -> int:
    """Run benchmark and check for regression against baseline."""
    harness = BenchmarkHarness(num_iterations=100, mock_response_time=0.05)
    harness.run_benchmark()

    baseline_path = Path("benchmark_baseline.json")
    passed, message = harness.check_regression(baseline_path, threshold_percent=15.0)

    print(message)

    if not passed and baseline_path.exists():
        return 1

    # Save current results as new baseline
    harness.save_results(baseline_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
