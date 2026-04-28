"""Semantic cache for claim detection results.

Uses an in-memory LRU cache keyed by normalized text. In production,
this would be replaced with Redis for cross-instance sharing.

Future enhancement: use CLS embedding similarity for semantic dedup
(e.g., "the earth is round" and "the earth is spherical" hit same cache).
"""

from cachetools import TTLCache

from src.config import settings


class PredictionCache:
    """LRU + TTL cache for prediction results."""

    def __init__(
        self,
        max_size: int | None = None,
        ttl: int | None = None,
    ):
        max_size = max_size or settings.cache_max_size
        ttl = ttl or settings.cache_ttl_seconds
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for cache key (lowercase, strip whitespace)."""
        return text.strip().lower()

    def get(self, text: str) -> dict | None:
        """Look up a cached prediction. Returns None on miss."""
        key = self._normalize(text)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            return {**result, "cached": True}
        self._misses += 1
        return None

    def put(self, text: str, result: dict) -> None:
        """Cache a prediction result."""
        key = self._normalize(text)
        self._cache[key] = result

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._cache.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
