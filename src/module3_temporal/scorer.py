from __future__ import annotations
from dataclasses import dataclass
from typing import List

from src.common.schemas import QueryContext, CandidateDoc, ScoredDoc
from src.common.utils_time import days_between
from src.module3_temporal.intent import IntentDetector
from src.module3_temporal.recency import exponential_recency
from src.module3_temporal.lambda_policy import LambdaPolicy
from src.module3_temporal.alpha_policy import AlphaPolicy


@dataclass(slots=True)
class TemporalScorer:
    intent_detector: IntentDetector
    lambda_policy: LambdaPolicy
    alpha_policy: AlphaPolicy

    def score(self, query_ctx: QueryContext, candidates: List[CandidateDoc]) -> List[ScoredDoc]:
        intent = self.intent_detector.detect(query_ctx.query)
        alpha = self.alpha_policy.get_alpha(query_ctx, intent)

        scored: List[ScoredDoc] = []
        for doc in candidates:
            dt = days_between(query_ctx.now, doc.published_at)
            lam = self.lambda_policy.get_lambda(doc, intent)
            rec = exponential_recency(dt, lam)
            final_score = rec * (1.0 - alpha) + doc.semantic_score * alpha

            scored.append(
                ScoredDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    published_at=doc.published_at,
                    category=doc.category,
                    semantic_score=doc.semantic_score,
                    source=doc.source,
                    intent=intent,
                    delta_t_days=dt,
                    lambda_dynamic=lam,
                    recency_dynamic=rec,
                    alpha_dynamic=alpha,
                    final_score=final_score,
                )
            )
        return scored
