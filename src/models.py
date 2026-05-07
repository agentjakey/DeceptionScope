from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
GOOGLE_AI_API_KEY: str | None = os.environ.get("GOOGLE_AI_API_KEY")


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_id: str
    display_name: str
    cost_per_1k_tokens: float


class ProviderError(Exception):
    """Raised when an API key is missing or a provider call fails."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message)
        self.provider = provider


AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-3-5",
        display_name="Claude Haiku 3.5",
        cost_per_1k_tokens=0.0008,
    ),
    "sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        cost_per_1k_tokens=0.003,
    ),
    "gpt4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        cost_per_1k_tokens=0.00015,
    ),
    "gpt4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_1k_tokens=0.005,
    ),
    "gemini-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        cost_per_1k_tokens=0.000075,
    ),
    "gemini-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        cost_per_1k_tokens=0.00125,
    ),
}

_PROVIDER_KEYS: dict[ModelProvider, str | None] = {
    ModelProvider.ANTHROPIC: ANTHROPIC_API_KEY,
    ModelProvider.OPENAI: OPENAI_API_KEY,
    ModelProvider.GOOGLE: GOOGLE_AI_API_KEY,
}


def _complete_anthropic(
    config: ModelConfig,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> str:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text
    except Exception as exc:
        raise ProviderError(f"Anthropic call failed: {exc}", "anthropic") from exc


def _complete_openai(
    config: ModelConfig,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> str:
    try:
        import openai

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise ProviderError(f"OpenAI call failed: {exc}", "openai") from exc


def _complete_google(
    config: ModelConfig,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_AI_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.model_id,
            system_instruction=system_prompt,
        )
        generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
        response = model.generate_content(
            user_content,
            generation_config=generation_config,
        )
        return response.text
    except Exception as exc:
        raise ProviderError(f"Google call failed: {exc}", "google") from exc


_DISPATCH: dict[ModelProvider, object] = {
    ModelProvider.ANTHROPIC: _complete_anthropic,
    ModelProvider.OPENAI: _complete_openai,
    ModelProvider.GOOGLE: _complete_google,
}


def complete(
    system_prompt: str,
    user_turns: list[str],
    model_alias: str,
    max_tokens: int = 800,
) -> str:
    """Return a single completion from the specified model alias.

    user_turns are joined with double newlines into a single user message so that
    multi-turn scenario context is preserved without injecting fake assistant
    responses. Raises KeyError for unknown aliases, ProviderError for missing
    keys or failed calls.
    """
    if model_alias not in AVAILABLE_MODELS:
        raise KeyError(
            f"Unknown model alias {model_alias!r}. "
            f"Available: {list(AVAILABLE_MODELS)}"
        )

    config = AVAILABLE_MODELS[model_alias]

    if config.provider == ModelProvider.ANTHROPIC and not ANTHROPIC_API_KEY:
        raise ProviderError("ANTHROPIC_API_KEY is not set", "anthropic")
    if config.provider == ModelProvider.OPENAI and not OPENAI_API_KEY:
        raise ProviderError("OPENAI_API_KEY is not set", "openai")
    if config.provider == ModelProvider.GOOGLE and not GOOGLE_AI_API_KEY:
        raise ProviderError("GOOGLE_AI_API_KEY is not set", "google")

    user_content = "\n\n".join(user_turns)
    handler = _DISPATCH[config.provider]
    return handler(config, system_prompt, user_content, max_tokens)  # type: ignore[operator]


def list_available_models() -> list[dict]:
    """Return all registry entries formatted for the web UI.

    Each entry includes whether the provider key is currently set in the
    environment, so the frontend can indicate which models are usable.
    """
    result: list[dict] = []
    for alias, config in AVAILABLE_MODELS.items():
        key_present = bool(_PROVIDER_KEYS[config.provider])
        result.append(
            {
                "alias": alias,
                "display_name": config.display_name,
                "model_id": config.model_id,
                "provider": config.provider.value,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "key_configured": key_present,
                "available": key_present,
            }
        )
    return result
