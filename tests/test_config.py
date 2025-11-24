"""Tests for configuration validation."""

import os
import pytest
from unittest.mock import patch

from llmchess.config import Config, ConfigurationError


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config_with_env_vars(self):
        """Test valid configuration from environment variables."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            config = Config()
            config.validate()  # Should not raise

    def test_valid_config_with_params(self):
        """Test valid configuration from parameters."""
        config = Config(
            endpoint="https://test.azure.com",
            model="gpt-4",
            verbose=True
        )
        config.validate()  # Should not raise

    def test_missing_endpoint_env_var(self):
        """Test missing AZURE_AI_FOUNDRY_ENDPOINT environment variable."""
        with patch.dict(os.environ, {
            "AZURE_AI_MODEL": "gpt-4"
        }, clear=True):
            config = Config()
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            assert "AZURE_AI_FOUNDRY_ENDPOINT is required" in str(exc_info.value)

    def test_missing_model_env_var(self):
        """Test missing AZURE_AI_MODEL environment variable."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com"
        }, clear=True):
            config = Config()
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            assert "AZURE_AI_MODEL is required" in str(exc_info.value)

    def test_missing_both_env_vars(self):
        """Test missing both required environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            error_msg = str(exc_info.value)
            assert "AZURE_AI_FOUNDRY_ENDPOINT is required" in error_msg
            assert "AZURE_AI_MODEL is required" in error_msg

    def test_invalid_endpoint_url(self):
        """Test invalid endpoint URL format."""
        config = Config(
            endpoint="not-a-valid-url",
            model="gpt-4"
        )
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "not a valid URL" in str(exc_info.value)

    def test_empty_endpoint(self):
        """Test empty endpoint string."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(endpoint="", model="gpt-4")
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            assert "AZURE_AI_FOUNDRY_ENDPOINT is required" in str(exc_info.value)

    def test_empty_model(self):
        """Test empty model string."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(endpoint="https://test.azure.com", model="")
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            assert "AZURE_AI_MODEL is required" in str(exc_info.value)

    def test_whitespace_model(self):
        """Test model with only whitespace."""
        config = Config(endpoint="https://test.azure.com", model="   ")
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "invalid format" in str(exc_info.value)

    def test_malformed_model_name(self):
        """Test malformed model name with invalid characters."""
        config = Config(endpoint="https://test.azure.com", model="model@#$%")
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "invalid format" in str(exc_info.value)

    def test_valid_model_formats(self):
        """Test various valid model name formats."""
        valid_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "azure/gpt-4",
            "deployment_name",
            "model.name",
            "provider/model-name"
        ]
        for model in valid_models:
            config = Config(endpoint="https://test.azure.com", model=model)
            config.validate()  # Should not raise

    def test_valid_endpoint_formats(self):
        """Test various valid endpoint URL formats."""
        valid_endpoints = [
            "https://test.azure.com",
            "https://test.openai.azure.com",
            "https://test.azure.com:443",
            "https://test.azure.com/path",
            "http://localhost:8080"
        ]
        for endpoint in valid_endpoints:
            config = Config(endpoint=endpoint, model="gpt-4")
            config.validate()  # Should not raise

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config(
            endpoint="https://test.azure.com",
            model="gpt-4",
            verbose=True
        )
        config_dict = config.to_dict()
        assert config_dict["endpoint"] == "https://test.azure.com"
        assert config_dict["model"] == "gpt-4"
        assert config_dict["verbose"] is True

    def test_verbose_flag_default(self):
        """Test verbose flag defaults to False."""
        config = Config(endpoint="https://test.azure.com", model="gpt-4")
        assert config.verbose is False

    def test_verbose_flag_explicit(self):
        """Test explicit verbose flag setting."""
        config = Config(endpoint="https://test.azure.com", model="gpt-4", verbose=True)
        assert config.verbose is True

    def test_params_override_env_vars(self):
        """Test that parameters override environment variables."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://env.azure.com",
            "AZURE_AI_MODEL": "env-model"
        }):
            config = Config(
                endpoint="https://param.azure.com",
                model="param-model"
            )
            assert config.endpoint == "https://param.azure.com"
            assert config.model == "param-model"

    def test_none_params_use_env_vars(self):
        """Test that None parameters fall back to environment variables."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://env.azure.com",
            "AZURE_AI_MODEL": "env-model"
        }):
            config = Config(endpoint=None, model=None)
            assert config.endpoint == "https://env.azure.com"
            assert config.model == "env-model"
