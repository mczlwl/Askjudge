from types import SimpleNamespace
from typing import Any

import pytest

from askjudge.providers import OpenAICompatibleProvider
from tests.async_helpers import run_immediate


class FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeModels:
    async def list(self) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
        )


class FakeClient:
    def __init__(self, content: str | None = "response") -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = FakeModels()


def test_provider_sends_openai_compatible_json_request() -> None:
    client = FakeClient("model response")
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        client=client,
    )

    result = run_immediate(
        provider.complete(system_prompt="system", user_prompt="user")
    )

    assert result == "model response"
    assert provider.call_count == 1
    assert client.completions.requests == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_can_disable_json_mode() -> None:
    client = FakeClient()
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        json_mode=False,
        client=client,
    )

    run_immediate(provider.complete(system_prompt="system", user_prompt="user"))

    assert "response_format" not in client.completions.requests[0]


def test_provider_rejects_empty_model_response() -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        client=FakeClient(None),
    )

    with pytest.raises(RuntimeError, match="empty text response"):
        run_immediate(
            provider.complete(system_prompt="system", user_prompt="user")
        )


def test_provider_lists_model_ids() -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        client=FakeClient(),
    )

    assert run_immediate(provider.list_model_ids()) == ["model-a", "model-b"]


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (
            "api_key",
            {"api_key": "", "base_url": "url", "model": "model"},
        ),
        (
            "base_url",
            {"api_key": "key", "base_url": "", "model": "model"},
        ),
        (
            "model",
            {"api_key": "key", "base_url": "url", "model": ""},
        ),
    ],
)
def test_provider_rejects_empty_configuration(
    field: str,
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        OpenAICompatibleProvider(**kwargs, client=FakeClient())

