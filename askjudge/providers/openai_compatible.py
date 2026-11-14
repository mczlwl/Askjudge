"""Provider for OpenAI-compatible chat-completions APIs."""

from __future__ import annotations

from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # The integration is an optional dependency.
    AsyncOpenAI = None  # type: ignore[assignment,misc]


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat-completions endpoint asynchronously."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        json_mode: bool = True,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if client is None:
            if AsyncOpenAI is None:
                raise ImportError(
                    "OpenAI-compatible provider requires the optional dependency; "
                    'install it with: pip install "askjudge[openai]"'
                )
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )

        self.model = model
        self.json_mode = json_mode
        self.call_count = 0
        self._client = client

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return one non-streaming model response."""

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "stream": False,
        }
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}

        self.call_count += 1
        response = await self._client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model provider returned an empty text response")
        return content

    async def list_model_ids(self) -> list[str]:
        """List model identifiers exposed by the configured endpoint."""

        response = await self._client.models.list()
        return [model.id for model in response.data]

