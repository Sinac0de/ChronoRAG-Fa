import math
from src.module3_temporal.recency import exponential_recency


def test_recency_zero_dt():
    assert exponential_recency(0, 0.08) == 1.0


def test_recency_monotonic():
    r1 = exponential_recency(1, 0.08)
    r2 = exponential_recency(10, 0.08)
    assert r1 > r2


def test_recency_known_value():
    v = exponential_recency(10, 0.1)
    assert abs(v - math.exp(-1.0)) < 1e-9
