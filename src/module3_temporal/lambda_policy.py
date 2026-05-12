from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

from src.common.schemas import IntentLabel, CandidateDoc


class LambdaPolicy(Protocol):
    def get_lambda(self, doc: CandidateDoc, intent: IntentLabel) -> float: ...


@dataclass(slots=True)
class RuleBasedLambdaPolicy:
    base_lambda: float
    category_weights: Dict[str, float]
    intent_weights: Dict[str, float]

    def get_lambda(self, doc: CandidateDoc, intent: IntentLabel) -> float:
        cw = self.category_weights.get(doc.category, self.category_weights.get("general", 1.0))
        iw = self.intent_weights.get(intent, 1.0)
        return max(1e-8, self.base_lambda * cw * iw)


@dataclass
class LearnedLambdaPolicy:
    model: Pipeline
    fallback: RuleBasedLambdaPolicy

    def get_lambda(self, doc: CandidateDoc, intent: IntentLabel) -> float:
        x = [{"category": doc.category, "intent": intent}]
        try:
            pred = float(self.model.predict(x)[0])
            return max(1e-8, pred)
        except Exception:
            return self.fallback.get_lambda(doc, intent)

    @staticmethod
    def train_synthetic(
        base_lambda: float,
        category_weights: Dict[str, float],
        intent_weights: Dict[str, float],
        n_samples: int = 2000,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> "LearnedLambdaPolicy":
        rng = np.random.default_rng(seed)
        categories = list(category_weights.keys())
        intents = list(intent_weights.keys())

        X: List[Dict[str, Any]] = []
        y: List[float] = []

        for _ in range(n_samples):
            c = rng.choice(categories)
            i = rng.choice(intents)
            target = base_lambda * category_weights[c] * intent_weights[i]
            target += float(rng.normal(0.0, noise_std))
            target = max(1e-8, target)
            X.append({"category": c, "intent": i})
            y.append(target)

        pipe = Pipeline([
            ("vec", DictVectorizer(sparse=False)),
            ("reg", Ridge(alpha=1.0, random_state=seed))
        ])
        pipe.fit(X, y)

        fallback = RuleBasedLambdaPolicy(
            base_lambda=base_lambda,
            category_weights=category_weights,
            intent_weights=intent_weights,
        )
        return LearnedLambdaPolicy(model=pipe, fallback=fallback)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str, fallback: RuleBasedLambdaPolicy) -> "LearnedLambdaPolicy":
        model = joblib.load(path)
        return LearnedLambdaPolicy(model=model, fallback=fallback)
