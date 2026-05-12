# src/common/schemas.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Literal


IntentLabel = Literal["fresh", "balanced", "historical"]


@dataclass(slots=True)
class QueryContext:
    query: str
    now: datetime
    user_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateDoc:
    doc_id: str
    text: str
    published_at: datetime
    category: str
    semantic_score: float
    source: Optional[str] = None

    def __post_init__(self) -> None:
        self.category = (self.category or "general").strip().lower()
        self.semantic_score = float(min(1.0, max(0.0, self.semantic_score)))


@dataclass(slots=True)
class ScoredDoc(CandidateDoc):
    intent: IntentLabel = "balanced"
    delta_t_days: float = 0.0
    lambda_dynamic: float = 0.08
    recency_dynamic: float = 1.0
    alpha_dynamic: float = 0.5
    final_score: float = 0.0

    def __post_init__(self) -> None:
        self.delta_t_days = max(0.0, float(self.delta_t_days))
        self.lambda_dynamic = max(1e-8, float(self.lambda_dynamic))
        self.recency_dynamic = float(min(1.0, max(0.0, self.recency_dynamic)))
        self.alpha_dynamic = float(min(1.0, max(0.0, self.alpha_dynamic)))
        self.final_score = float(min(1.0, max(0.0, self.final_score)))

