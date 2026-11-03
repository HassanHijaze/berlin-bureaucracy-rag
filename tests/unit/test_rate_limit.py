import re

import pytest

from berlin_rag.api.rate_limit import DailyRequestLimiter, utc_today


@pytest.fixture
def calendar():
    return {"today": "2026-11-02"}


@pytest.fixture
def limiter(calendar):
    return DailyRequestLimiter(limit=2, today=lambda: calendar["today"])


def test_utc_today():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", utc_today())


def test_limiter_blocks_after_the_limit(limiter):
    results = []
    for _ in range(3):
        results.append(limiter.is_exhausted())
        limiter.record()

    assert results == [False, False, True]


def test_limiter_resets_on_a_new_day(limiter, calendar):
    limiter.record()
    limiter.record()
    calendar["today"] = "2026-11-03"

    assert limiter.is_exhausted() is False
    assert limiter.requests_today == 0
    assert limiter.requests_total == 2
