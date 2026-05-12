from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from src.common.schemas import IntentLabel


@dataclass(slots=True)
class IntentDetector:
    fresh_keywords: tuple[str, ...]
    historical_keywords: tuple[str, ...]

    def detect(self, query: str) -> IntentLabel:
        q = (query or "").strip().lower()
        if any(k in q for k in self.fresh_keywords):
            return "fresh"
        if any(k in q for k in self.historical_keywords):
            return "historical"
        return "balanced"


def build_default_intent_detector() -> IntentDetector:
    return IntentDetector(
        fresh_keywords=("آخرین", "امروز", "فوری", "جدید", "latest", "up-to-date"),
        historical_keywords=("تاریخچه", "روند", "بلندمدت", "گذشته", "historical", "trend"),
    )
