from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AVAILABLE_MODELS, ProviderError, complete, list_available_models


def test_list_available_models_count():
    """list_available_models returns at least 6 entries."""
    models = list_available_models()
    assert len(models) >= 6


def test_list_available_models_schema():
    """Every entry returned by list_available_models has the required keys."""
    required = {"alias", "display_name", "model_id", "provider", "cost_per_1k_tokens", "key_configured"}
    for entry in list_available_models():
        assert required.issubset(entry.keys()), f"Entry missing keys: {entry}"


def test_list_available_models_aliases_match_registry():
    """Aliases returned by list_available_models match keys in AVAILABLE_MODELS."""
    returned_aliases = {m["alias"] for m in list_available_models()}
    assert returned_aliases == set(AVAILABLE_MODELS.keys())


def test_provider_error_openai_key_missing():
    """complete() raises ProviderError(provider='openai') when OPENAI_API_KEY is unset."""
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(ProviderError) as exc_info:
            complete("system prompt", ["hello"], "gpt4o-mini")
    assert exc_info.value.provider == "openai"


def test_provider_error_openai_empty_string_key():
    """complete() raises ProviderError when OPENAI_API_KEY is an empty string."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        with pytest.raises(ProviderError) as exc_info:
            complete("system", ["test"], "gpt4o")
    assert exc_info.value.provider == "openai"


def test_unknown_alias_raises_key_error():
    """complete() raises KeyError for an unrecognized model alias."""
    with pytest.raises(KeyError):
        complete("system", ["hello"], "not-a-real-model-xyz")


def test_complete_joins_user_turns():
    """complete() joins multiple user_turns with double newlines before dispatch."""
    captured = {}

    def fake_handler(config, system_prompt, user_content, max_tokens):
        captured["user_content"] = user_content
        return "ok"

    alias = "haiku"
    config = AVAILABLE_MODELS[alias]
    from src.models import ModelProvider
    with patch("src.models._DISPATCH", {config.provider: fake_handler}):
        complete("sys", ["turn one", "turn two", "turn three"], alias)

    assert captured["user_content"] == "turn one\n\nturn two\n\nturn three"
