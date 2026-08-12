"""Streaming a long-running use case to the browser as server-sent events.

This is the only place that knows a use case's progress is delivered over SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi.responses import StreamingResponse

from application.progress import ProgressReporter

logger = logging.getLogger(__name__)

MEDIA_TYPE = "text/event-stream"


def format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class QueueProgress(ProgressReporter):
    """Publishes progress onto the queue feeding the response stream."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def report(self, message: str) -> None:
        self._queue.put_nowait(("log", {"message": message}))


def stream_use_case(
    run: Callable[[ProgressReporter], Awaitable[dict[str, Any]]],
    on_finish: Callable[[], None] | None = None,
) -> StreamingResponse:
    """Run ``run`` in the background, streaming its progress and final payload."""
    queue: asyncio.Queue = asyncio.Queue()

    async def worker() -> None:
        try:
            payload = await run(QueueProgress(queue))
            await queue.put(("done", payload))
        except Exception as exc:
            logger.exception("Streamed use case failed")
            await queue.put(("error", {"message": str(exc)}))
        finally:
            if on_finish:
                on_finish()

    asyncio.create_task(worker())
    return StreamingResponse(_events(queue), media_type=MEDIA_TYPE)


async def _events(queue: asyncio.Queue) -> AsyncIterator[str]:
    while True:
        event, data = await queue.get()
        yield format_event(event, data)
        if event in ("done", "error"):
            break
