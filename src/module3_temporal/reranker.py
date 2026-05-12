from __future__ import annotations
from dataclasses import dataclass
from typing import List

from src.common.schemas import QueryContext, CandidateDoc, ScoredDoc
from src.module3_temporal.scorer import TemporalScorer


@dataclass(slots=True)
class TemporalReranker:
    scorer: TemporalScorer
    top_k: int = 5

    def rerank(self, query_ctx: QueryContext, candidates: List[CandidateDoc]) -> List[ScoredDoc]:
        scored = self.scorer.score(query_ctx, candidates)
        scored.sort(key=lambda d: d.final_score, reverse=True)
        return scored

    def top(self, query_ctx: QueryContext, candidates: List[CandidateDoc]) -> List[ScoredDoc]:
        return self.rerank(query_ctx, candidates)[: self.top_k]
