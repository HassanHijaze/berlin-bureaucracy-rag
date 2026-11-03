"""A per-instance daily request limit that resets at midnight UTC."""

from collections.abc import Callable
from datetime import UTC, datetime

from ..config import DAILY_REQUEST_LIMIT


def utc_today() -> str:
    """Return the current UTC date.

    Returns:
        The date as ``YYYY-MM-DD``.
    """
    return datetime.now(UTC).date().isoformat()


class DailyRequestLimiter:
    """Counts answered requests and blocks new ones once today's limit is hit."""

    def __init__(
        self,
        limit: int = DAILY_REQUEST_LIMIT,
        today: Callable[[], str] = utc_today,
    ) -> None:
        """Start with zero requests counted.

        Args:
            limit: Maximum number of requests per day.
            today: Returns the current date; replaceable in tests.
        """
        self._limit = limit
        self._today = today
        self._day = today()
        self._requests_today = 0
        self._requests_total = 0

    @property
    def requests_today(self) -> int:
        """Requests counted since the last daily reset."""
        return self._requests_today

    @property
    def requests_total(self) -> int:
        """Requests counted since the limiter was created."""
        return self._requests_total

    def is_exhausted(self) -> bool:
        """Check whether today's limit is reached, resetting it on a new day.

        Returns:
            True if no more requests are allowed today.
        """
        today = self._today()
        if today != self._day:
            self._day = today
            self._requests_today = 0

        return self._requests_today >= self._limit

    def record(self) -> None:
        """Count one request."""
        self._requests_today += 1
        self._requests_total += 1
