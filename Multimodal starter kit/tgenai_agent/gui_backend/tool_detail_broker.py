"""Broker optional custom tool detail messages for GUI clients."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
from collections import deque
from typing import Any, Deque, Optional, Set

# Third Party
# None

# Local
from .events import ToolDetailMessage


# ==================================== CLASSES ====================================

class ToolDetailBroker:
    """Async pub/sub broker with bounded replay for tool detail messages."""

    def __init__(self, buffer_size: int = 200) -> None:
        self._buffer: Deque[ToolDetailMessage] = deque(maxlen=buffer_size)
        self._queues: Set[asyncio.Queue[Optional[ToolDetailMessage]]] = set()

    async def publish(self, message: ToolDetailMessage) -> None:
        """Publish a message to current subscribers and the replay buffer."""
        self._buffer.append(message)
        dead: Set[asyncio.Queue[Optional[ToolDetailMessage]]] = set()
        for queue in self._queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead.add(queue)
        self._queues -= dead

    def replay(self) -> list[ToolDetailMessage]:
        """Return buffered detail messages for a newly connected client."""
        return list(self._buffer)

    def subscribe(self) -> "ToolDetailSubscription":
        """Return an async subscription for detail messages."""
        return ToolDetailSubscription(self)


class ToolDetailSubscription:
    """Async context manager for a per-client detail message queue."""

    def __init__(self, broker: ToolDetailBroker, maxsize: int = 256) -> None:
        self._broker = broker
        self._queue: asyncio.Queue[Optional[ToolDetailMessage]] = asyncio.Queue(maxsize=maxsize)

    async def __aenter__(self) -> "ToolDetailSubscription":
        self._broker._queues.add(self._queue)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._broker._queues.discard(self._queue)
        while not self._queue.empty():
            self._queue.get_nowait()

    def __aiter__(self) -> "ToolDetailSubscription":
        return self

    async def __anext__(self) -> ToolDetailMessage:
        message = await self._queue.get()
        if message is None:
            raise StopAsyncIteration
        return message


# ==================================== GLOBALS ====================================

_default_broker: Optional[ToolDetailBroker] = None


# ================================ MAIN FUNCTIONS ================================

def get_tool_detail_broker() -> ToolDetailBroker:
    """Return the default tool detail broker, creating it on first use."""
    global _default_broker
    if _default_broker is None:
        from tgenai_agent.config import GUI_TOOL_DETAIL_BUFFER_SIZE
        _default_broker = ToolDetailBroker(buffer_size=GUI_TOOL_DETAIL_BUFFER_SIZE)
    return _default_broker