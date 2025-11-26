import pytest
import yaml
from pathlib import Path
from eval_harness.config import Config, load_config, get_provider_from_model


def test_load_config_successfully(tmp_path):
    """Test loading a valid YAML config file."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "judge_mapping": {
            "openai": "anthropic",
            "anthropic": "openai"
        },
        "judge_models": {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514"
        }
    }
    config_file.write_text(yaml.dump(config_data))

    result = load_config(config_file)

    assert isinstance(result, Config)
    assert result.judge_mapping == config_data["judge_mapping"]
    assert result.judge_models == config_data["judge_models"]


def test_load_config_returns_correct_judge_mapping(tmp_path):
    """Test that load_config returns the correct judge_mapping."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "judge_mapping": {
            "openai": "anthropic",
            "anthropic": "openai"
        },
        "judge_models": {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514"
        }
    }
    config_file.write_text(yaml.dump(config_data))

    result = load_config(config_file)

    assert result.judge_mapping["openai"] == "anthropic"
    assert result.judge_mapping["anthropic"] == "openai"


def test_load_config_returns_correct_judge_models(tmp_path):
    """Test that load_config returns the correct judge_models."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "judge_mapping": {
            "openai": "anthropic",
            "anthropic": "openai"
        },
        "judge_models": {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514"
        }
    }
    config_file.write_text(yaml.dump(config_data))

    result = load_config(config_file)

    assert result.judge_models["openai"] == "gpt-4o-mini"
    assert result.judge_models["anthropic"] == "claude-sonnet-4-20250514"


def test_get_provider_from_model_openai_gpt():
    """Test that 'gpt-4o' returns 'openai'."""
    result = get_provider_from_model("gpt-4o")
    assert result == "openai"


def test_get_provider_from_model_openai_o1():
    """Test that 'o1-preview' returns 'openai'."""
    result = get_provider_from_model("o1-preview")
    assert result == "openai"


def test_get_provider_from_model_anthropic():
    """Test that 'claude-sonnet-4-20250514' returns 'anthropic'."""
    result = get_provider_from_model("claude-sonnet-4-20250514")
    assert result == "anthropic"


def test_get_provider_from_model_unknown_raises():
    """Test that 'unknown-model' raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model provider for: unknown-model"):
        get_provider_from_model("unknown-model")
