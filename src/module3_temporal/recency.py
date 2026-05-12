from __future__ import annotations
import math


def exponential_recency(delta_t_days: float, lambda_dynamic: float) -> float:
    dt = max(0.0, float(delta_t_days))
    lam = max(1e-8, float(lambda_dynamic))
    score = math.exp(-lam * dt)
    return float(min(1.0, max(0.0, score)))
