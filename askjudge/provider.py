"""Minimal model-provider boundary used by AskJudge."""

from __future__ import annotations

from typing import Protocol


class ModelProvider(Protocol):
    """An asynchronous text-completion provider.

    The first MVP will implement one concrete provider while tests can use a
    lightweight fake that follows this protocol.
    """

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text response."""

        ...

