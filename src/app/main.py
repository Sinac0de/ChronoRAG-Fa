from __future__ import annotations
from datetime import datetime, timezone, timedelta

from src.common.logger import setup_logger, get_logger
from src.common.schemas import CandidateDoc
from src.app.pipeline import build_temporal_reranker, build_query_context


def demo_candidates() -> list[CandidateDoc]:
    now = datetime.now(timezone.utc)
    return [
        CandidateDoc(
            doc_id="1",
            text="خبر جدید اقتصادی...",
            published_at=now - timedelta(hours=8),
            category="economy",
            semantic_score=0.82
        ),
        CandidateDoc(
            doc_id="2",
            text="گزارش تحلیلی قدیمی...",
            published_at=now - timedelta(days=120),
            category="economy",
            semantic_score=0.91
        ),
    ]


def main() -> None:
    setup_logger()
    log = get_logger()

    reranker = build_temporal_reranker()
    query_ctx = build_query_context("آخرین وضعیت تورم", {"freshness_preference": 0.9})

    ranked = reranker.top(query_ctx, demo_candidates())
    for i, d in enumerate(ranked, 1):
        log.info(
            f"{i}) id={d.doc_id} score={d.final_score:.4f} "
            f"rec={d.recency_dynamic:.4f} rel={d.semantic_score:.4f} intent={d.intent}"
        )


if __name__ == "__main__":
    main()
