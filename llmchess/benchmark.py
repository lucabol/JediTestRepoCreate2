"""Performance benchmark module for AI move latency."""

import json
import statistics
from pathlib import Path
from typing import Any

from llmchess.ai_player import AIPlayer, MockAzureAIClient


class BenchmarkHarness:
    """Harness for running performance benchmarks with mocked Azure responses."""

    def __init__(self, num_iterations: int = 10, mock_response_time: float = 0.05) -> None:
        """Initialize benchmark harness.

        Args:
            num_iterations: Number of benchmark iterations to run
            mock_response_time: Simulated response time for mocked Azure client
        """
        self.num_iterations = num_iterations
        self.mock_response_time = mock_response_time
        self.results: list[float] = []
        self._results_dict: dict[str, Any] | None = None

    def run_benchmark(self) -> dict[str, Any]:
        """Run AI move latency benchmark.

        Returns:
            Dictionary containing benchmark results with statistics
        """
        mock_client = MockAzureAIClient(response_time=self.mock_response_time)
        ai_player = AIPlayer(client=mock_client)

        # Standard starting position in FEN notation
        board_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        latencies: list[float] = []

        for _ in range(self.num_iterations):
            _, latency = ai_player.get_move_with_timing(board_state)
            latencies.append(latency)

        self.results = latencies

        self._results_dict = {
            "num_iterations": self.num_iterations,
            "mock_response_time": self.mock_response_time,
            "latencies": latencies,
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "min": min(latencies),
            "max": max(latencies),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99),
        }

        return self._results_dict

    @staticmethod
    def _percentile(data: list[float], percentile: float) -> float:
        """Calculate percentile of data.

        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        weight = index - lower

        if upper >= len(sorted_data):
            return sorted_data[-1]

        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def save_results(self, filepath: Path) -> None:
        """Save benchmark results to JSON file.

        Args:
            filepath: Path to save results
        """
        if not self._results_dict:
            raise ValueError("No benchmark results to save. Run benchmark first.")

        with open(filepath, "w") as f:
            json.dump(self._results_dict, f, indent=2)

    def check_regression(
        self, baseline_filepath: Path, threshold_percent: float = 10.0
    ) -> tuple[bool, str]:
        """Check for performance regression against baseline.

        Args:
            baseline_filepath: Path to baseline results JSON file
            threshold_percent: Maximum acceptable increase in mean latency (percentage)

        Returns:
            Tuple of (passed, message)
        """
        if not self._results_dict:
            return False, "No benchmark results available. Run benchmark first."

        if not baseline_filepath.exists():
            return True, f"No baseline found at {baseline_filepath}. Creating baseline."

        with open(baseline_filepath) as f:
            baseline = json.load(f)

        current_mean = self._results_dict["mean"]
        baseline_mean = baseline["mean"]

        if baseline_mean == 0:
            return False, "Invalid baseline: mean latency is zero"

        percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100

        if percent_change > threshold_percent:
            return False, (
                f"Performance regression detected! "
                f"Mean latency increased by {percent_change:.2f}% "
                f"(baseline: {baseline_mean:.4f}s, current: {current_mean:.4f}s, "
                f"threshold: {threshold_percent}%)"
            )

        return True, (
            f"Performance check passed. "
            f"Mean latency change: {percent_change:.2f}% "
            f"(baseline: {baseline_mean:.4f}s, current: {current_mean:.4f}s)"
        )


def main() -> None:
    """Run benchmark and save results."""
    harness = BenchmarkHarness(num_iterations=100)
    results = harness.run_benchmark()

    print("Benchmark Results:")
    print(f"  Iterations: {results['num_iterations']}")
    print(f"  Mean latency: {results['mean']:.4f}s")
    print(f"  Median latency: {results['median']:.4f}s")
    print(f"  Std dev: {results['stdev']:.4f}s")
    print(f"  Min: {results['min']:.4f}s")
    print(f"  Max: {results['max']:.4f}s")
    print(f"  P95: {results['p95']:.4f}s")
    print(f"  P99: {results['p99']:.4f}s")

    # Save results
    results_path = Path("benchmark_results.json")
    harness.save_results(results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
