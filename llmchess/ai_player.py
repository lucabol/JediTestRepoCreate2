"""AI player module for LLMChess using Azure AI Foundry."""

import os
import time
from typing import Protocol


class AzureAIClient(Protocol):
    """Protocol for Azure AI client interface."""

    def get_completion(self, prompt: str) -> str:
        """Get completion from AI model."""
        ...


class MockAzureAIClient:
    """Mock Azure AI client for testing and benchmarking."""

    def __init__(self, response_time: float = 0.05) -> None:
        """Initialize mock client with configurable response time.

        Args:
            response_time: Simulated response time in seconds
        """
        self.response_time = response_time

    def get_completion(self, prompt: str) -> str:
        """Return a mock chess move after simulated delay.

        Args:
            prompt: The prompt sent to the AI

        Returns:
            A mock chess move in UCI format
        """
        time.sleep(self.response_time)
        # Return a standard opening move for simplicity
        return "e2e4"


class AIPlayer:
    """AI player that uses Azure AI Foundry to make chess moves."""

    def __init__(
        self,
        client: AzureAIClient | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize AI player.

        Args:
            client: Azure AI client (or mock for testing)
            endpoint: Azure AI Foundry endpoint
            model: Azure AI model name
        """
        if client is not None:
            self.client = client
        else:
            # In real implementation, would create actual Azure client
            endpoint = endpoint or os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", "")
            model = model or os.environ.get("AZURE_AI_MODEL", "")
            if not endpoint or not model:
                raise ValueError(
                    "Azure AI credentials required: set AZURE_AI_FOUNDRY_ENDPOINT "
                    "and AZURE_AI_MODEL environment variables"
                )
            # For now, use mock client as placeholder
            self.client = MockAzureAIClient()

    def get_move(self, board_state: str) -> str:
        """Get next move from AI.

        Args:
            board_state: Current board state in FEN notation

        Returns:
            Move in UCI format (e.g., 'e2e4')
        """
        prompt = f"Given this chess board state: {board_state}, what is the best move?"
        return self.client.get_completion(prompt)

    def get_move_with_timing(self, board_state: str) -> tuple[str, float]:
        """Get move with latency measurement.

        Args:
            board_state: Current board state in FEN notation

        Returns:
            Tuple of (move, latency_in_seconds)
        """
        start_time = time.perf_counter()
        move = self.get_move(board_state)
        latency = time.perf_counter() - start_time
        return move, latency
