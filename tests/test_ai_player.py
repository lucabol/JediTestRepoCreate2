"""Tests for AI player module."""

import time
from unittest.mock import MagicMock

import pytest

from llmchess.ai_player import AIPlayer, MockAzureAIClient


class TestMockAzureAIClient:
    """Tests for MockAzureAIClient."""

    def test_get_completion_returns_move(self) -> None:
        """Test that get_completion returns a chess move."""
        client = MockAzureAIClient()
        move = client.get_completion("test prompt")
        assert move == "e2e4"

    def test_get_completion_respects_response_time(self) -> None:
        """Test that get_completion simulates response time."""
        response_time = 0.1
        client = MockAzureAIClient(response_time=response_time)

        start = time.perf_counter()
        client.get_completion("test prompt")
        elapsed = time.perf_counter() - start

        # Allow for some timing variance
        assert elapsed >= response_time * 0.9


class TestAIPlayer:
    """Tests for AIPlayer."""

    def test_initialization_with_client(self) -> None:
        """Test AIPlayer can be initialized with a client."""
        mock_client = MockAzureAIClient()
        player = AIPlayer(client=mock_client)
        assert player.client == mock_client

    def test_initialization_without_credentials_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AIPlayer raises error when credentials are missing."""
        monkeypatch.delenv("AZURE_AI_FOUNDRY_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_AI_MODEL", raising=False)
        with pytest.raises(ValueError, match="Azure AI credentials required"):
            AIPlayer()

    def test_initialization_with_endpoint_and_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AIPlayer can be initialized with endpoint and model."""
        monkeypatch.setenv("AZURE_AI_FOUNDRY_ENDPOINT", "https://test.endpoint")
        monkeypatch.setenv("AZURE_AI_MODEL", "test-model")

        player = AIPlayer(endpoint="https://test.endpoint", model="test-model")
        assert player.client is not None

    def test_get_move_returns_move(self) -> None:
        """Test that get_move returns a chess move."""
        mock_client = MockAzureAIClient()
        player = AIPlayer(client=mock_client)

        move = player.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert move == "e2e4"

    def test_get_move_calls_client(self) -> None:
        """Test that get_move calls the client correctly."""
        mock_client = MagicMock()
        mock_client.get_completion.return_value = "d2d4"
        player = AIPlayer(client=mock_client)

        board_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = player.get_move(board_state)

        assert move == "d2d4"
        mock_client.get_completion.assert_called_once()
        call_args = mock_client.get_completion.call_args[0][0]
        assert board_state in call_args

    def test_get_move_with_timing_returns_move_and_latency(self) -> None:
        """Test that get_move_with_timing returns move and latency."""
        mock_client = MockAzureAIClient(response_time=0.05)
        player = AIPlayer(client=mock_client)

        move, latency = player.get_move_with_timing(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        assert move == "e2e4"
        assert latency >= 0.04  # Allow for timing variance
        assert latency < 0.2  # Reasonable upper bound

    def test_get_move_with_timing_measures_accurately(self) -> None:
        """Test that timing measurement is accurate."""
        response_time = 0.1
        mock_client = MockAzureAIClient(response_time=response_time)
        player = AIPlayer(client=mock_client)

        _, latency = player.get_move_with_timing(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        # Latency should be at least the response time
        assert latency >= response_time * 0.9
