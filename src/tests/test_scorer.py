from datetime import datetime, timezone, timedelta

from src.common.schemas import QueryContext, CandidateDoc
from src.module3_temporal.intent import build_default_intent_detector
from src.module3_temporal.lambda_policy import RuleBasedLambdaPolicy
from src.module3_temporal.alpha_policy import RuleBasedAlphaPolicy
from src.module3_temporal.scorer import TemporalScorer


def _build_scorer() -> TemporalScorer:
    return TemporalScorer(
        intent_detector=build_default_intent_detector(),
        lambda_policy=RuleBasedLambdaPolicy(
            base_lambda=0.08,
            category_weights={"economy": 1.2, "general": 1.0},
            intent_weights={"fresh": 1.3, "balanced": 1.0, "historical": 0.6}
        ),
        alpha_policy=RuleBasedAlphaPolicy(
            base_alpha={"fresh": 0.25, "balanced": 0.5, "historical": 0.8}
        )
    )


def test_scorer_output_fields():
    scorer = _build_scorer()
    now = datetime.now(timezone.utc)
    ctx = QueryContext(query="آخرین اخبار اقتصاد", now=now, user_profile={"freshness_preference": 0.8})
    docs = [
        CandidateDoc("a", "t1", now - timedelta(days=1), "economy", 0.8),
        CandidateDoc("b", "t2", now - timedelta(days=30), "economy", 0.9),
    ]

    out = scorer.score(ctx, docs)
    assert len(out) == 2
    for d in out:
        assert 0.0 <= d.final_score <= 1.0
        assert d.lambda_dynamic > 0
        assert 0.0 <= d.recency_dynamic <= 1.0
        assert 0.0 <= d.alpha_dynamic <= 1.0
