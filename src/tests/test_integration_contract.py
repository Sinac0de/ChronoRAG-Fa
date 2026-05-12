from datetime import datetime, timezone, timedelta

from src.common.schemas import QueryContext, CandidateDoc
from src.module3_temporal.intent import build_default_intent_detector
from src.module3_temporal.lambda_policy import RuleBasedLambdaPolicy
from src.module3_temporal.alpha_policy import RuleBasedAlphaPolicy
from src.module3_temporal.scorer import TemporalScorer
from src.module3_temporal.reranker import TemporalReranker


def test_contract_and_rerank_order():
    now = datetime.now(timezone.utc)

    candidates = [
        CandidateDoc(
            doc_id="recent_mid_rel",
            text="recent",
            published_at=now - timedelta(hours=6),
            category="economy",
            semantic_score=0.70
        ),
        CandidateDoc(
            doc_id="old_high_rel",
            text="old",
            published_at=now - timedelta(days=90),
            category="economy",
            semantic_score=0.95
        ),
    ]

    scorer = TemporalScorer(
        intent_detector=build_default_intent_detector(),
        lambda_policy=RuleBasedLambdaPolicy(
            base_lambda=0.08,
            category_weights={"economy": 1.2, "general": 1.0},
            intent_weights={"fresh": 1.3, "balanced": 1.0, "historical": 0.6}
        ),
        alpha_policy=RuleBasedAlphaPolicy(
            base_alpha={"fresh": 0.25, "balanced": 0.5, "historical": 0.8}
        ),
    )
    reranker = TemporalReranker(scorer=scorer, top_k=2)

    ctx = QueryContext(query="آخرین وضعیت بازار", now=now, user_profile={"freshness_preference": 0.9})
    out = reranker.rerank(ctx, candidates)

    assert len(out) == 2
    assert out[0].final_score >= out[1].final_score
    assert out[0].doc_id in {"recent_mid_rel", "old_high_rel"}
