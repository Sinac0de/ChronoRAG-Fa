from datetime import datetime, timezone
from src.common.schemas import QueryContext
from src.module3_temporal.alpha_policy import RuleBasedAlphaPolicy


def test_alpha_rule_based_with_pref():
    policy = RuleBasedAlphaPolicy(base_alpha={"fresh": 0.25, "balanced": 0.5, "historical": 0.8})
    ctx = QueryContext(query="x", now=datetime.now(timezone.utc), user_profile={"freshness_preference": 0.9})
    a = policy.get_alpha(ctx, "fresh")
    assert 0.0 <= a <= 1.0
    assert a < 0.25  


def test_alpha_rule_based_no_pref():
    policy = RuleBasedAlphaPolicy(base_alpha={"fresh": 0.25, "balanced": 0.5, "historical": 0.8})
    ctx = QueryContext(query="x", now=datetime.now(timezone.utc), user_profile={})
    a = policy.get_alpha(ctx, "historical")
    assert abs(a - 0.8) < 1e-12
