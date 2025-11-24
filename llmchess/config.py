"""Configuration validation for LLMChess."""

import os
import re
from typing import Optional, Dict, Any


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class Config:
    """Configuration class for LLMChess with validation."""

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, verbose: bool = False):
        """
        Initialize configuration.

        Args:
            endpoint: Azure AI Foundry endpoint URL (or None to read from env)
            model: Azure AI model name (or None to read from env)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        # Use endpoint parameter if provided and not empty, otherwise fall back to env var
        self.endpoint = endpoint if endpoint else os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        # Use model parameter if provided and not empty, otherwise fall back to env var
        self.model = model if model else os.environ.get("AZURE_AI_MODEL")

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []

        # Validate endpoint
        if not self.endpoint:
            errors.append(
                "AZURE_AI_FOUNDRY_ENDPOINT is required. "
                "Set it via environment variable or command line argument."
            )
        elif not self._is_valid_url(self.endpoint):
            errors.append(
                f"AZURE_AI_FOUNDRY_ENDPOINT is not a valid URL: '{self.endpoint}'. "
                "Expected format: https://your-resource.azure.com"
            )

        # Validate model
        if not self.model:
            errors.append(
                "AZURE_AI_MODEL is required. "
                "Set it via environment variable or command line argument."
            )
        elif not self._is_valid_model_name(self.model):
            errors.append(
                f"AZURE_AI_MODEL has invalid format: '{self.model}'. "
                "Expected format: deployment-name or provider/model-name"
            )

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_message)

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        if not url or not isinstance(url, str):
            return False
        # Basic URL validation - must start with http:// or https://
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))

    @staticmethod
    def _is_valid_model_name(model: str) -> bool:
        """Check if model name is valid."""
        if not model or not isinstance(model, str):
            return False
        # Model name should not be empty or contain only whitespace
        if not model.strip():
            return False
        # Model name should not contain invalid characters
        # Valid characters: alphanumeric, dash, underscore, slash, dot
        model_pattern = re.compile(r'^[a-zA-Z0-9/_.-]+$')
        return bool(model_pattern.match(model))

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "verbose": self.verbose
        }
