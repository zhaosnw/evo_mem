"""Anthropic LLM backend."""

import os
from typing import List, Dict, Optional

from .base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    """Anthropic API backend (Claude models)."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic LLM.

        Args:
            model_name: Claude model name
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional BaseLLM parameters
        """
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(model_name, api_key, **kwargs)
        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
        return self._client

    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Create message
        create_kwargs = {
            "model": self.model_name,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)

        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response,
        )
