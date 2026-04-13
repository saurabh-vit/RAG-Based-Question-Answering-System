from __future__ import annotations

from cachetools import TTLCache


class QueryCache:
    """
    Simple in-memory TTL cache (bonus feature).
    Good for repeated questions in demos and small deployments.
    For multi-instance production, swap with Redis.
    """

    def __init__(self, *, maxsize: int = 1024, ttl_seconds: int = 120) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, value) -> None:
        self._cache[key] = value

