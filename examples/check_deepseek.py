"""Check DeepSeek credentials without running the 20-case evaluation."""

from __future__ import annotations

import asyncio
import getpass
import os

from askjudge.providers import OpenAICompatibleProvider


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    return getpass.getpass("Paste DEEPSEEK_API_KEY (input hidden): ")


async def _main() -> None:
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    provider = OpenAICompatibleProvider(
        api_key=_api_key(),
        model=model,
        base_url=os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
    )
    model_ids = await provider.list_model_ids()
    print("DeepSeek API connection succeeded.")
    print(f"Configured model: {model}")
    print("Available models:")
    for model_id in model_ids:
        print(f"- {model_id}")
    if model not in model_ids:
        print(
            "Warning: the configured model was not returned by /models. "
            "Check DEEPSEEK_MODEL before running the batch demo."
        )


if __name__ == "__main__":
    asyncio.run(_main())

