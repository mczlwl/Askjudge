from dataclasses import dataclass, field


@dataclass
class FakeProvider:
    responses: list[str | Exception]
    calls: list[dict[str, str]] = field(default_factory=list)

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider has no response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

