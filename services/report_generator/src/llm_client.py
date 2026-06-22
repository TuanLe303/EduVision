"""LLM client wrappers for Google Gemini and OpenAI GPT."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* to the model and return the generated text."""


class GeminiClient(LLMClient):
    """Google Gemini via the ``google-generativeai`` SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ) -> None:
        import google.generativeai as genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable or pass api_key."
            )
        genai.configure(api_key=key)
        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._model = genai.GenerativeModel(model, generation_config=gen_config)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text


class GPTClient(LLMClient):
    """OpenAI GPT via the ``openai`` SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""


def build_llm_client(provider: str, cfg: dict) -> LLMClient:
    """Instantiate the correct LLMClient from *provider* name and config dict."""
    provider_cfg: dict = cfg.get(provider, {})

    if provider == "gemini":
        return GeminiClient(
            model=provider_cfg.get("model", "gemini-2.0-flash"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_output_tokens=int(provider_cfg.get("max_output_tokens", 2048)),
        )

    if provider == "gpt":
        return GPTClient(
            model=provider_cfg.get("model", "gpt-4o-mini"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_tokens=int(provider_cfg.get("max_tokens", 2048)),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported values: 'gemini', 'gpt'."
    )
