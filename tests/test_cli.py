"""Tests for CLI argument parsing."""

import os
import sys
import pytest
from io import StringIO
from unittest.mock import patch

from llmchess.cli import parse_args, main, create_parser
from llmchess.config import ConfigurationError


class TestCLIParsing:
    """Test CLI argument parsing."""

    def test_parse_args_no_arguments(self):
        """Test parsing with no arguments."""
        args = parse_args([])
        assert args.endpoint is None
        assert args.model is None
        assert args.verbose is False

    def test_parse_args_endpoint(self):
        """Test parsing with --endpoint argument."""
        args = parse_args(["--endpoint", "https://test.azure.com"])
        assert args.endpoint == "https://test.azure.com"

    def test_parse_args_model(self):
        """Test parsing with --model argument."""
        args = parse_args(["--model", "gpt-4"])
        assert args.model == "gpt-4"

    def test_parse_args_verbose_long(self):
        """Test parsing with --verbose argument."""
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_parse_args_verbose_short(self):
        """Test parsing with -v argument."""
        args = parse_args(["-v"])
        assert args.verbose is True

    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments."""
        args = parse_args([
            "--endpoint", "https://test.azure.com",
            "--model", "gpt-4",
            "--verbose"
        ])
        assert args.endpoint == "https://test.azure.com"
        assert args.model == "gpt-4"
        assert args.verbose is True

    def test_parse_args_help(self):
        """Test that --help triggers SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_parse_args_version(self):
        """Test that --version triggers SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_parse_args_invalid_argument(self):
        """Test parsing with invalid argument."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--invalid-arg"])
        assert exc_info.value.code != 0

    def test_parser_has_epilog(self):
        """Test that parser includes environment variable documentation."""
        parser = create_parser()
        assert "AZURE_AI_FOUNDRY_ENDPOINT" in parser.epilog
        assert "AZURE_AI_MODEL" in parser.epilog


class TestCLIMain:
    """Test main CLI entry point."""

    def test_main_success_with_env_vars(self, capsys):
        """Test successful execution with environment variables."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            exit_code = main([])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "initialized successfully" in captured.out.lower()

    def test_main_success_with_args(self, capsys):
        """Test successful execution with command-line arguments."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "https://test.azure.com",
                "--model", "gpt-4"
            ])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "initialized successfully" in captured.out.lower()

    def test_main_verbose_mode(self, capsys):
        """Test verbose mode output."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            exit_code = main(["--verbose"])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Verbose mode enabled" in captured.out
            assert "Validating configuration" in captured.out

    def test_main_missing_endpoint(self, capsys):
        """Test error handling for missing endpoint."""
        with patch.dict(os.environ, {
            "AZURE_AI_MODEL": "gpt-4"
        }, clear=True):
            exit_code = main([])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "AZURE_AI_FOUNDRY_ENDPOINT is required" in captured.err

    def test_main_missing_model(self, capsys):
        """Test error handling for missing model."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com"
        }, clear=True):
            exit_code = main([])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "AZURE_AI_MODEL is required" in captured.err

    def test_main_invalid_endpoint(self, capsys):
        """Test error handling for invalid endpoint URL."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "invalid-url",
                "--model", "gpt-4"
            ])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "not a valid URL" in captured.err

    def test_main_malformed_model(self, capsys):
        """Test error handling for malformed model name."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "https://test.azure.com",
                "--model", "invalid@model"
            ])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "invalid format" in captured.err

    def test_main_keyboard_interrupt(self, capsys):
        """Test handling of keyboard interrupt."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            with patch('llmchess.cli.Config.validate', side_effect=KeyboardInterrupt):
                exit_code = main([])
                assert exit_code == 130
                captured = capsys.readouterr()
                assert "Interrupted" in captured.err

    def test_main_unexpected_error(self, capsys):
        """Test handling of unexpected errors."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            with patch('llmchess.cli.Config.validate', side_effect=RuntimeError("Unexpected")):
                exit_code = main([])
                assert exit_code == 1
                captured = capsys.readouterr()
                assert "Unexpected error" in captured.err

    def test_main_verbose_with_unexpected_error(self, capsys):
        """Test verbose mode shows traceback on unexpected errors."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://test.azure.com",
            "AZURE_AI_MODEL": "gpt-4"
        }):
            with patch('llmchess.cli.Config.validate', side_effect=RuntimeError("Test error")):
                exit_code = main(["--verbose"])
                assert exit_code == 1
                captured = capsys.readouterr()
                assert "Traceback" in captured.err or "Test error" in captured.err

    def test_main_args_override_env_vars(self, capsys):
        """Test command-line args override environment variables."""
        with patch.dict(os.environ, {
            "AZURE_AI_FOUNDRY_ENDPOINT": "https://env.azure.com",
            "AZURE_AI_MODEL": "env-model"
        }):
            exit_code = main([
                "--endpoint", "https://cli.azure.com",
                "--model", "cli-model"
            ])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "https://cli.azure.com" in captured.out
            assert "cli-model" in captured.out

    def test_main_empty_endpoint_arg(self, capsys):
        """Test empty endpoint argument."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "",
                "--model", "gpt-4"
            ])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "AZURE_AI_FOUNDRY_ENDPOINT is required" in captured.err

    def test_main_empty_model_arg(self, capsys):
        """Test empty model argument."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "https://test.azure.com",
                "--model", ""
            ])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "AZURE_AI_MODEL is required" in captured.err

    def test_main_whitespace_model(self, capsys):
        """Test model with only whitespace."""
        with patch.dict(os.environ, {}, clear=True):
            exit_code = main([
                "--endpoint", "https://test.azure.com",
                "--model", "   "
            ])
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "invalid format" in captured.err


class TestCLIEdgeCases:
    """Test edge cases in CLI handling."""

    def test_verbose_flag_variations(self):
        """Test different ways to specify verbose flag."""
        # Long form
        args = parse_args(["--verbose"])
        assert args.verbose is True

        # Short form
        args = parse_args(["-v"])
        assert args.verbose is True

        # Default (not specified)
        args = parse_args([])
        assert args.verbose is False

    def test_argument_order_independence(self):
        """Test that argument order doesn't matter."""
        args1 = parse_args(["--verbose", "--endpoint", "https://test.com", "--model", "gpt-4"])
        args2 = parse_args(["--model", "gpt-4", "--endpoint", "https://test.com", "--verbose"])
        args3 = parse_args(["--endpoint", "https://test.com", "--verbose", "--model", "gpt-4"])

        assert args1.verbose == args2.verbose == args3.verbose is True
        assert args1.endpoint == args2.endpoint == args3.endpoint == "https://test.com"
        assert args1.model == args2.model == args3.model == "gpt-4"

    def test_special_characters_in_model(self):
        """Test model names with various special characters."""
        # Valid characters
        valid_models = ["model-name", "model_name", "provider/model", "model.v1"]
        for model in valid_models:
            args = parse_args(["--model", model])
            assert args.model == model

    def test_unicode_in_arguments(self):
        """Test handling of unicode characters in arguments."""
        args = parse_args(["--model", "模型"])
        # Parser should accept it, but validation will catch it
        assert args.model == "模型"

    def test_very_long_arguments(self):
        """Test handling of very long argument values."""
        long_url = "https://" + "a" * 1000 + ".com"
        args = parse_args(["--endpoint", long_url])
        assert args.endpoint == long_url
