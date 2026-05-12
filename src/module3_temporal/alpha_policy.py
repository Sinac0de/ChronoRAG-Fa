from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

from src.common.schemas import IntentLabel, QueryContext


class AlphaPolicy(Protocol):
    def get_alpha(self, query_ctx: QueryContext, intent: IntentLabel) -> float: ...


@dataclass(slots=True)
class RuleBasedAlphaPolicy:
    base_alpha: Dict[str, float]

    def get_alpha(self, query_ctx: QueryContext, intent: IntentLabel) -> float:
        a = self.base_alpha.get(intent, 0.5)
        pref = query_ctx.user_profile.get("freshness_preference", None)
        if pref is not None:
            pref = float(min(1.0, max(0.0, pref)))
            a = 0.85 * a + 0.15 * (1.0 - pref)
        return float(min(1.0, max(0.0, a)))


@dataclass
class LearnedAlphaPolicy:
    model: Pipeline
    fallback: RuleBasedAlphaPolicy

    def get_alpha(self, query_ctx: QueryContext, intent: IntentLabel) -> float:
        pref = query_ctx.user_profile.get("freshness_preference", 0.5)
        pref = float(min(1.0, max(0.0, pref)))
        x = [{"intent": intent, "freshness_preference": pref}]
        try:
            pred = float(self.model.predict(x)[0])
            return float(min(1.0, max(0.0, pred)))
        except Exception:
            return self.fallback.get_alpha(query_ctx, intent)

    @staticmethod
    def train_synthetic(
        base_alpha: Dict[str, float],
        n_samples: int = 2000,
        noise_std: float = 0.02,
        seed: int = 42,
    ) -> "LearnedAlphaPolicy":
        rng = np.random.default_rng(seed)
        intents = list(base_alpha.keys())

        X: List[Dict[str, Any]] = []
        y: List[float] = []

        for _ in range(n_samples):
            i = rng.choice(intents)
            pref = float(rng.uniform(0.0, 1.0))
            target = 0.85 * base_alpha[i] + 0.15 * (1.0 - pref)
            target += float(rng.normal(0.0, noise_std))
            target = float(min(1.0, max(0.0, target)))

            X.append({"intent": i, "freshness_preference": pref})
            y.append(target)

        pipe = Pipeline([
            ("vec", DictVectorizer(sparse=False)),
            ("reg", Ridge(alpha=1.0, random_state=seed))
        ])
        pipe.fit(X, y)

        fallback = RuleBasedAlphaPolicy(base_alpha=base_alpha)
        return LearnedAlphaPolicy(model=pipe, fallback=fallback)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str, fallback: RuleBasedAlphaPolicy) -> "LearnedAlphaPolicy":
        model = joblib.load(path)
        return LearnedAlphaPolicy(model=model, fallback=fallback)
