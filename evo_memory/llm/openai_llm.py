"""OpenAI LLM backend."""

import os
from typing import List, Dict, Optional, Any

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI API backend (GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI LLM.

        Args:
            model_name: OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: Optional custom API base URL
            **kwargs: Additional BaseLLM parameters
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(model_name, api_key, api_base, **kwargs)
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client

    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
