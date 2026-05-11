"""CLI entry point for the PU extraction agent."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path so we can import agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import load_config
from agent.agent import PUExtractionAgent


def setup_logging(log_dir: Path):
    """Setup logging to console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console)

    # File
    fh = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(fh)


def main():
    parser = argparse.ArgumentParser(description="PU Literature Data Extraction Agent")
    parser.add_argument("--config", default="agent_config.yaml", help="Config YAML path")
    parser.add_argument("--env", default=".env", help=".env file path")
    parser.add_argument("--folders", nargs="+", help="Folders to process (overrides config)")
    parser.add_argument("--steps", nargs="+", choices=["literature", "curves", "mechanical"],
                        help="Run only specific extraction steps")
    args = parser.parse_args()

    config = load_config(args.config, args.env)

    # Override folders from CLI
    folders = args.folders if args.folders else config.folders
    if not folders:
        parser.error("No folders configured. Add folders in agent_config.yaml or pass --folders PU_001 PU_002.")

    # Setup logging
    setup_logging(config.root_dir / "combined_agent_output" / "logs")

    # Run agent
    try:
        agent = PUExtractionAgent(config)
    except ValueError as exc:
        parser.error(str(exc))
    agent.run(folders, steps=args.steps)


if __name__ == "__main__":
    main()
