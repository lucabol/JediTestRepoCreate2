"""Command-line interface for LLMChess."""

import argparse
import sys
from typing import Optional, List

from llmchess.config import Config, ConfigurationError


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="llmchess",
        description="Play chess against an AI powered by Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  AZURE_AI_FOUNDRY_ENDPOINT  Azure AI Foundry endpoint URL (required)
  AZURE_AI_MODEL             Azure AI model name (required)

Examples:
  llmchess --verbose
  llmchess --endpoint https://my-resource.azure.com --model gpt-4
        """
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        help="Azure AI Foundry endpoint URL (overrides AZURE_AI_FOUNDRY_ENDPOINT env var)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Azure AI model name (overrides AZURE_AI_MODEL env var)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments with validation.

    Args:
        args: List of arguments to parse (None = sys.argv)

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If argument parsing fails
    """
    parser = create_parser()
    
    try:
        parsed_args = parser.parse_args(args)
        return parsed_args
    except SystemExit as e:
        # Re-raise SystemExit to maintain standard argparse behavior
        raise
    except Exception as e:
        parser.error(f"Error parsing arguments: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for LLMChess CLI.

    Args:
        argv: Command-line arguments (None = sys.argv)

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    try:
        # Parse arguments
        args = parse_args(argv)

        if args.verbose:
            print("Verbose mode enabled")
            print(f"Arguments: endpoint={args.endpoint}, model={args.model}")

        # Create and validate configuration
        config = Config(
            endpoint=args.endpoint,
            model=args.model,
            verbose=args.verbose
        )

        if args.verbose:
            print("Validating configuration...")

        config.validate()

        if args.verbose:
            print("Configuration valid:")
            for key, value in config.to_dict().items():
                print(f"  {key}: {value}")

        # If we get here, configuration is valid
        print("LLMChess initialized successfully!")
        print(f"Connected to: {config.endpoint}")
        print(f"Using model: {config.model}")

        # TODO: Implement actual chess game logic
        print("\nChess game functionality to be implemented.")

        return 0

    except ConfigurationError as e:
        print(f"Configuration Error:\n{e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
