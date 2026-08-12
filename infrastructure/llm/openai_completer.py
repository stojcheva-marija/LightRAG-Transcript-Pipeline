from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import ModelConfig


class OpenAITextCompleter:
    """``TextCompleter`` backed by the OpenAI chat completions API."""

    def __init__(self, api_key: str, model_config: ModelConfig) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model_config.llm_model

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_object: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"temperature": temperature}
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return response.choices[0].message.content or ""
