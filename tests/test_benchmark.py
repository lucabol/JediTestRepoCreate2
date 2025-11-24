"""Tests for benchmark module."""

import json
import tempfile
from pathlib import Path

import pytest

from llmchess.benchmark import BenchmarkHarness


class TestBenchmarkHarness:
    """Tests for BenchmarkHarness."""

    def test_initialization(self) -> None:
        """Test BenchmarkHarness initialization."""
        harness = BenchmarkHarness(num_iterations=5, mock_response_time=0.03)
        assert harness.num_iterations == 5
        assert harness.mock_response_time == 0.03
        assert harness.results == []

    def test_run_benchmark_returns_results(self) -> None:
        """Test that run_benchmark returns result dictionary."""
        harness = BenchmarkHarness(num_iterations=5, mock_response_time=0.01)
        results = harness.run_benchmark()

        assert "num_iterations" in results
        assert "mean" in results
        assert "median" in results
        assert "stdev" in results
        assert "min" in results
        assert "max" in results
        assert "p95" in results
        assert "p99" in results
        assert "latencies" in results

        assert results["num_iterations"] == 5
        assert len(results["latencies"]) == 5

    def test_run_benchmark_measures_latency(self) -> None:
        """Test that benchmark measures latency correctly."""
        mock_response_time = 0.05
        harness = BenchmarkHarness(num_iterations=3, mock_response_time=mock_response_time)
        results = harness.run_benchmark()

        # All latencies should be at least the mock response time
        for latency in results["latencies"]:
            assert latency >= mock_response_time * 0.9

        # Mean should be close to mock response time
        assert results["mean"] >= mock_response_time * 0.9

    def test_run_benchmark_populates_results(self) -> None:
        """Test that run_benchmark populates results list."""
        harness = BenchmarkHarness(num_iterations=5)
        harness.run_benchmark()

        assert len(harness.results) == 5
        for latency in harness.results:
            assert latency > 0

    def test_percentile_calculation(self) -> None:
        """Test percentile calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        p50 = BenchmarkHarness._percentile(data, 50)
        assert 5.0 <= p50 <= 6.0

        p95 = BenchmarkHarness._percentile(data, 95)
        assert p95 >= 9.0

        p99 = BenchmarkHarness._percentile(data, 99)
        assert p99 >= 9.5

    def test_save_results_creates_file(self) -> None:
        """Test that save_results creates a JSON file."""
        harness = BenchmarkHarness(num_iterations=3, mock_response_time=0.01)
        harness.run_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            harness.save_results(filepath)

            assert filepath.exists()

            with open(filepath) as f:
                saved_results = json.load(f)

            assert "mean" in saved_results
            assert "latencies" in saved_results
            assert len(saved_results["latencies"]) == 3

    def test_save_results_without_running_raises_error(self) -> None:
        """Test that save_results raises error if benchmark not run."""
        harness = BenchmarkHarness(num_iterations=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            with pytest.raises(ValueError, match="No benchmark results to save"):
                harness.save_results(filepath)

    def test_check_regression_with_no_baseline(self) -> None:
        """Test regression check when no baseline exists."""
        harness = BenchmarkHarness(num_iterations=3, mock_response_time=0.01)
        harness.run_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            passed, message = harness.check_regression(baseline_path)

            assert passed
            assert "No baseline found" in message

    def test_check_regression_passes_when_under_threshold(self) -> None:
        """Test regression check passes when latency is under threshold."""
        # Create baseline
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_data = {
                "mean": 0.05,
                "median": 0.05,
                "stdev": 0.001,
                "min": 0.049,
                "max": 0.051,
            }
            with open(baseline_path, "w") as f:
                json.dump(baseline_data, f)

            # Run current benchmark with similar latency
            harness = BenchmarkHarness(num_iterations=3, mock_response_time=0.05)
            harness.run_benchmark()

            passed, message = harness.check_regression(baseline_path, threshold_percent=10.0)

            assert passed
            assert "Performance check passed" in message

    def test_check_regression_fails_when_over_threshold(self) -> None:
        """Test regression check fails when latency exceeds threshold."""
        # Create baseline with low latency
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_data = {
                "mean": 0.01,
                "median": 0.01,
                "stdev": 0.001,
                "min": 0.009,
                "max": 0.011,
            }
            with open(baseline_path, "w") as f:
                json.dump(baseline_data, f)

            # Run current benchmark with much higher latency
            harness = BenchmarkHarness(num_iterations=3, mock_response_time=0.05)
            harness.run_benchmark()

            passed, message = harness.check_regression(baseline_path, threshold_percent=10.0)

            assert not passed
            assert "Performance regression detected" in message
            assert "increased by" in message

    def test_check_regression_without_running_benchmark(self) -> None:
        """Test regression check returns error if benchmark not run."""
        harness = BenchmarkHarness(num_iterations=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            passed, message = harness.check_regression(baseline_path)

            assert not passed
            assert "No benchmark results available" in message

    def test_check_regression_with_zero_baseline(self) -> None:
        """Test regression check handles zero baseline gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_data = {
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            with open(baseline_path, "w") as f:
                json.dump(baseline_data, f)

            harness = BenchmarkHarness(num_iterations=3, mock_response_time=0.01)
            harness.run_benchmark()

            passed, message = harness.check_regression(baseline_path)

            assert not passed
            assert "Invalid baseline" in message


class TestBenchmarkValidation:
    """Tests that validate the benchmark harness itself."""

    def test_benchmark_consistency(self) -> None:
        """Test that benchmark produces consistent results across runs."""
        mock_response_time = 0.05
        harness1 = BenchmarkHarness(num_iterations=10, mock_response_time=mock_response_time)
        harness2 = BenchmarkHarness(num_iterations=10, mock_response_time=mock_response_time)

        results1 = harness1.run_benchmark()
        results2 = harness2.run_benchmark()

        # Means should be similar (within 50% to account for timing variance)
        mean_diff = abs(results1["mean"] - results2["mean"])
        avg_mean = (results1["mean"] + results2["mean"]) / 2
        assert mean_diff / avg_mean < 0.5

    def test_benchmark_scales_with_response_time(self) -> None:
        """Test that benchmark latency scales with mock response time."""
        harness_fast = BenchmarkHarness(num_iterations=5, mock_response_time=0.01)
        harness_slow = BenchmarkHarness(num_iterations=5, mock_response_time=0.05)

        results_fast = harness_fast.run_benchmark()
        results_slow = harness_slow.run_benchmark()

        # Slow should have higher mean latency
        assert results_slow["mean"] > results_fast["mean"]

    def test_benchmark_statistics_validity(self) -> None:
        """Test that benchmark statistics are mathematically valid."""
        harness = BenchmarkHarness(num_iterations=20, mock_response_time=0.02)
        results = harness.run_benchmark()

        # Min should be <= median <= max
        assert results["min"] <= results["median"] <= results["max"]

        # Mean should be within reasonable range of median
        assert abs(results["mean"] - results["median"]) < results["max"] - results["min"]

        # P95 should be >= P99 is conceptually wrong, P99 >= P95
        assert results["p99"] >= results["p95"]

        # P95 should be >= median
        assert results["p95"] >= results["median"]

    def test_benchmark_handles_varying_latencies(self) -> None:
        """Test that benchmark correctly handles varying latencies."""
        harness = BenchmarkHarness(num_iterations=10, mock_response_time=0.05)
        results = harness.run_benchmark()

        # Should have some variance in latencies
        assert results["stdev"] >= 0
        assert results["max"] >= results["min"]

    def test_benchmark_isolation(self) -> None:
        """Test that multiple benchmark instances don't interfere."""
        harness1 = BenchmarkHarness(num_iterations=3, mock_response_time=0.01)
        harness2 = BenchmarkHarness(num_iterations=5, mock_response_time=0.02)

        results1 = harness1.run_benchmark()
        results2 = harness2.run_benchmark()

        assert len(results1["latencies"]) == 3
        assert len(results2["latencies"]) == 5
        assert results1["mock_response_time"] == 0.01
        assert results2["mock_response_time"] == 0.02
