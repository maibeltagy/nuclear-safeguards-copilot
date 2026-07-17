"""LLM clients for OpenAI and Ollama, selected via configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod

import os
import requests

from google import genai

from Configuration.settings import LLMSettings, settings


class BaseLLMClient(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    """Call OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMSettings | None = None) -> None:
        self.config = config or settings.llm
        self.openai = self.config.openai

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        api_key = self.openai.api_key
        if not api_key:
            raise ValueError(
                f"Missing OpenAI API key. Set environment variable {self.openai.api_key_env}."
            )

        client_kwargs = {"api_key": api_key}
        if self.openai.base_url:
            client_kwargs["base_url"] = self.openai.base_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.openai.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiClient(BaseLLMClient):
    """Call Google's Gemini API."""

    def __init__(self, config: LLMSettings | None = None) -> None:
        self.config = config or settings.llm
        self.gemini = self.config.gemini

        api_key = self.gemini.api_key
        if not api_key:
            raise ValueError(
                f"Missing Gemini API key. Set environment variable {self.gemini.api_key_env}."
            )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        print("===================================")
        print("Model from settings:", repr(self.gemini.model))
        print("===================================")
        
        response = self.client.models.generate_content(
            model=self.gemini.model,
            contents=prompt,
        
        )
        return response.text or ""


class OllamaClient(BaseLLMClient):
    """Call a local Ollama server (Lab 9 pattern)."""

    def __init__(self, config: LLMSettings | None = None) -> None:
        self.config = config or settings.llm
        self.ollama = self.config.ollama

    def generate(self, prompt: str) -> str:
        url = f"{self.ollama.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def create_llm_client(config: LLMSettings | None = None) -> BaseLLMClient:
    """Factory: pick OpenAI or Ollama from settings."""
    config = config or settings.llm
    provider = config.provider.lower().strip()

    if provider == "openai":
        return OpenAIClient(config)
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "gemini":
        return GeminiClient(config)

    raise ValueError(f"Unsupported LLM provider: {config.provider}. Use 'openai', 'ollama', or 'gemini'.")
