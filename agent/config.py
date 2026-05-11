"""Configuration loader: .env + YAML."""

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    vision_model: str = ""
    temperature: float = 0.1
    max_tokens: int = 8192


@dataclass
class AgentConfig:
    root_dir: Path = field(default_factory=lambda: Path("."))
    folders: list[str] = field(default_factory=list)
    render_dpi: int = 400
    max_image_dim: int = 1600
    retries: int = 3
    backoff_base: int = 2
    api_timeout: int = 180
    min_curve_points: int = 50
    min_confidence: str = "low"
    output_dir_name: str = "agent_output"
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(config_path: str = "agent_config.yaml", env_path: str = ".env") -> AgentConfig:
    """Load configuration from .env and YAML files."""
    project_root = Path(__file__).parent.parent
    config_file = _resolve_user_path(config_path, project_root)
    env_file = _resolve_user_path(env_path, project_root)

    # Load .env
    load_dotenv(env_file)

    llm = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        vision_model=os.getenv("LLM_VISION_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
    )

    # Load YAML
    yaml_cfg = {}
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    root_dir = Path(yaml_cfg.get("root_dir", project_root))
    if not root_dir.is_absolute():
        root_dir = (config_file.parent / root_dir).resolve()

    config = AgentConfig(
        root_dir=root_dir,
        folders=yaml_cfg.get("folders", []),
        render_dpi=yaml_cfg.get("render_dpi", 400),
        max_image_dim=yaml_cfg.get("max_image_dim", 1600),
        retries=yaml_cfg.get("retries", 3),
        backoff_base=yaml_cfg.get("backoff_base", 2),
        api_timeout=yaml_cfg.get("api_timeout", 180),
        min_curve_points=yaml_cfg.get("min_curve_points", 50),
        min_confidence=yaml_cfg.get("min_confidence", "low"),
        output_dir_name=yaml_cfg.get("output_dir_name", "agent_output"),
        llm=llm,
    )

    return config


def _resolve_user_path(path_value: str, base_dir: Path) -> Path:
    """Resolve CLI-supplied config/env paths relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path
