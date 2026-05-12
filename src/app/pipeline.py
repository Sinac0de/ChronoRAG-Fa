from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import yaml

from src.common.schemas import QueryContext
from src.module3_temporal.intent import IntentDetector
from src.module3_temporal.lambda_policy import (
    RuleBasedLambdaPolicy, LearnedLambdaPolicy
)
from src.module3_temporal.alpha_policy import (
    RuleBasedAlphaPolicy, LearnedAlphaPolicy
)
from src.module3_temporal.scorer import TemporalScorer
from src.module3_temporal.reranker import TemporalReranker


def _load_config(path: str = "configs/temporal.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_temporal_reranker(config_path: str = "configs/temporal.yaml") -> TemporalReranker:
    cfg = _load_config(config_path)["temporal"]

    detector = IntentDetector(
        fresh_keywords=tuple(cfg["intent"]["fresh_keywords"]),
        historical_keywords=tuple(cfg["intent"]["historical_keywords"]),
    )

    # Lambda policy
    lambda_cfg = cfg["lambda_policy"]
    rb_lambda = RuleBasedLambdaPolicy(
        base_lambda=float(lambda_cfg["base_lambda"]),
        category_weights=dict(lambda_cfg["category_weights"]),
        intent_weights=dict(lambda_cfg["intent_weights"]),
    )
    if lambda_cfg["mode"] == "learned":
        model_path = Path(lambda_cfg["model_path"])
        if model_path.exists():
            lambda_policy = LearnedLambdaPolicy.load(str(model_path), rb_lambda)
        else:
            lambda_policy = LearnedLambdaPolicy.train_synthetic(
                base_lambda=rb_lambda.base_lambda,
                category_weights=rb_lambda.category_weights,
                intent_weights=rb_lambda.intent_weights,
                n_samples=int(cfg["learning"]["synthetic_samples"]),
                noise_std=float(cfg["learning"]["noise_std_lambda"]),
                seed=int(cfg["learning"]["random_seed"]),
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            lambda_policy.save(str(model_path))
    else:
        lambda_policy = rb_lambda

    alpha_cfg = cfg["alpha_policy"]
    rb_alpha = RuleBasedAlphaPolicy(base_alpha=dict(alpha_cfg["base_alpha"]))
    if alpha_cfg["mode"] == "learned":
        model_path = Path(alpha_cfg["model_path"])
        if model_path.exists():
            alpha_policy = LearnedAlphaPolicy.load(str(model_path), rb_alpha)
        else:
            alpha_policy = LearnedAlphaPolicy.train_synthetic(
                base_alpha=rb_alpha.base_alpha,
                n_samples=int(cfg["learning"]["synthetic_samples"]),
                noise_std=float(cfg["learning"]["noise_std_alpha"]),
                seed=int(cfg["learning"]["random_seed"]),
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            alpha_policy.save(str(model_path))
    else:
        alpha_policy = rb_alpha

    scorer = TemporalScorer(
        intent_detector=detector,
        lambda_policy=lambda_policy,
        alpha_policy=alpha_policy
    )
    return TemporalReranker(scorer=scorer, top_k=int(cfg["top_k"]))


def build_query_context(query: str, user_profile: dict | None = None) -> QueryContext:
    return QueryContext(
        query=query,
        now=datetime.now(timezone.utc),
        user_profile=user_profile or {},
    )
