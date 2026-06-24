"""Redis Streams publisher for backend events."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import logging
from typing import Optional, Protocol

# Third Party
from redis.asyncio import Redis

# Local
from .schema import BackendEvent


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)


# ==================================== PROTOCOLS ====================================

class EventPublisher(Protocol):
    """Minimal interface shared by concrete event publishers."""

    async def publish(self, event: BackendEvent) -> None:
        """Publish one backend event."""

    async def ping(self) -> bool:
        """Return whether the publisher backend is reachable."""

    async def close(self) -> None:
        """Release any publisher resources."""


# ==================================== CLASSES ====================================

class NullEventPublisher:
    """No-op publisher used when backend eventing is disabled."""

    async def publish(self, event: BackendEvent) -> None:
        """Ignore events when backend eventing is disabled."""
        return None

    async def ping(self) -> bool:
        """Report that no backend publisher is active."""
        return False

    async def close(self) -> None:
        """Nothing to close for the no-op publisher."""
        return None


class RedisEventPublisher:
    """Small Redis Streams publisher with graceful failure behavior."""

    def __init__(self, redis_url: str, stream_name: str, maxlen: int) -> None:
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._maxlen = maxlen
        self._client: Optional[Redis] = None
        self._disabled = False

    async def publish(self, event: BackendEvent) -> None:
        """Publish an event to Redis Streams."""
        if self._disabled:
            return None

        client = self._get_client()
        try:
            await client.xadd(
                self._stream_name,
                event.to_redis_fields(),
                maxlen=self._maxlen,
                approximate=True,
            )
        except Exception as exc:
            await self._disable(exc)

    async def ping(self) -> bool:
        """Return whether Redis responds to a ping."""
        if self._disabled:
            return False

        try:
            return bool(await self._get_client().ping())
        except Exception as exc:
            await self._disable(exc)
            return False

    async def close(self) -> None:
        """Close the Redis connection if it was opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> Redis:
        """Return a lazy Redis client."""
        if self._client is None:
            self._client = Redis.from_url(self._redis_url, decode_responses=False)
        return self._client

    async def _disable(self, exc: Exception) -> None:
        """Disable further publish attempts after a Redis failure."""
        if self._disabled:
            return None

        self._disabled = True
        LOGGER.warning(
            "Disabling backend event publishing because Redis is unavailable at %s (%s: %s)",
            self._redis_url,
            type(exc).__name__,
            exc,
        )
        await self.close()


# ==================================== GLOBALS ====================================

_default_publisher: Optional[EventPublisher] = None
_null_publisher = NullEventPublisher()


# ================================ MAIN FUNCTIONS ================================

def get_event_publisher() -> EventPublisher:
    """Return the configured module-level Redis event publisher."""
    global _default_publisher
    if _default_publisher is None:
        from tgenai_agent.config import (
            BACKEND_EVENTS_ENABLED,
            BACKEND_EVENT_STREAM,
            REDIS_STREAM_MAXLEN,
            REDIS_URL,
        )

        if not BACKEND_EVENTS_ENABLED or not REDIS_URL:
            if not BACKEND_EVENTS_ENABLED:
                LOGGER.info("Backend event publishing disabled by configuration")
            else:
                LOGGER.info("Backend event publishing disabled because REDIS_URL is empty")
            _default_publisher = _null_publisher
        else:
            _default_publisher = RedisEventPublisher(
                redis_url=REDIS_URL,
                stream_name=BACKEND_EVENT_STREAM,
                maxlen=REDIS_STREAM_MAXLEN,
            )
    return _default_publisher


async def publish_event(event: BackendEvent) -> None:
    """Publish an event and never let backend eventing break the caller."""
    await get_event_publisher().publish(event)


def publish_event_background(event: BackendEvent) -> None:
    """Schedule event publishing from sync and async call sites."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(publish_event(event))
        return

    loop.create_task(publish_event(event))
