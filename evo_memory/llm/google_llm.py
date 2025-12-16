"""Google Gemini LLM backend."""

import os
from typing import List, Dict, Optional

from .base import BaseLLM, LLMResponse


class GoogleLLM(BaseLLM):
    """Google Gemini API backend."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google Gemini LLM.

        Args:
            model_name: Gemini model name (e.g., gemini-2.5-flash, gemini-2.5-pro)
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            **kwargs: Additional BaseLLM parameters
        """
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        super().__init__(model_name, api_key, **kwargs)
        self._model = None

    @property
    def model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install google-generativeai"
                )
        return self._model

    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Google Gemini API."""
        # Convert messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [content]})

        # Configure generation
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }

        # Generate response
        if system_instruction:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction,
            )
        else:
            model = self.model

        response = model.generate_content(
            contents,
            generation_config=generation_config,
        )

        # Extract content
        content = ""
        if response.text:
            content = response.text

        # Get usage metadata
        usage = {}
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", 0),
                "completion_tokens": getattr(meta, "candidates_token_count", 0),
                "total_tokens": getattr(meta, "total_token_count", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model_name,
            usage=usage,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
            raw_response=response,
        )
