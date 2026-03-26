from __future__ import annotations

import asyncio
from typing import Any
from operator import itemgetter

from FlagEmbedding import FlagReranker

_reranker: FlagReranker | None = None


def _get_reranker(model: str) -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(model, use_fp16=True)
    return _reranker


async def bge_rerank(
    query: str,
    documents: list[str],
    model: str,
    top_n: int,
) -> list[dict[str, Any]]:
    reranker = _get_reranker(model)
    pairs = [[query, doc] for doc in documents]

    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, lambda: reranker.compute_score(pairs, normalize=True))

    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(enumerate(scores), key=itemgetter(1), reverse=True)
    return [{"index": i, "relevance_score": float(s)} for i, s in ranked[:top_n]]
