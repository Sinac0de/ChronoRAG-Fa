from __future__ import annotations
from datetime import datetime, timezone


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def days_between(later: datetime, earlier: datetime) -> float:
    later_utc = ensure_utc(later)
    earlier_utc = ensure_utc(earlier)
    delta = later_utc - earlier_utc
    return max(0.0, delta.total_seconds() / 86400.0)
