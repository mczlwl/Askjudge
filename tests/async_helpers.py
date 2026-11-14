from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


def run_immediate(coroutine: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine whose fake dependencies complete without yielding."""

    try:
        coroutine.send(None)
    except StopIteration as exc:
        return exc.value
    coroutine.close()
    raise AssertionError("test coroutine unexpectedly yielded")


async def sequential_gather(*awaitables: Any) -> list[Any]:
    """Test substitute for asyncio.gather that needs no system event loop."""

    return [await awaitable for awaitable in awaitables]

