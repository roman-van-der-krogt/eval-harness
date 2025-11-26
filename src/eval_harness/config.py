import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    judge_mapping: dict[str, str]  # maps response provider to judge provider
    judge_models: dict[str, str]   # maps judge provider to model name


def load_config(path: Path) -> Config:
    """Load config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        judge_mapping=data["judge_mapping"],
        judge_models=data["judge_models"]
    )


def get_provider_from_model(model: str) -> str:
    """Determine provider from model name."""
    if model.startswith(("gpt-", "o1-")):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        raise ValueError(f"Unknown model provider for: {model}")
