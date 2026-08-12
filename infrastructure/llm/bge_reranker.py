"""Cross-encoder reranking of LightRAG's retrieved chunks, via BGE."""

from __future__ import annotations

import asyncio
from typing import Any
from operator import itemgetter

import torch
from FlagEmbedding import FlagReranker

_reranker: FlagReranker | None = None
# The shared reranker isn't safe to call from two threads at once.
_reranker_lock = asyncio.Lock()


def _get_reranker(model: str) -> FlagReranker:
    """Loaded once per process and reused — construction loads the model weights."""
    global _reranker
    if _reranker is None:
        # fp16 ("Half") is a GPU-only optimization — PyTorch's CPU backend
        # can't run these ops in fp16, so it must stay off unless CUDA is
        # actually available.
        _reranker = FlagReranker(model, use_fp16=torch.cuda.is_available())
    return _reranker


async def bge_rerank(
    query: str,
    documents: list[str],
    model: str,
    top_n: int,
) -> list[dict[str, Any]]:
    """Score each document against the query; return the top ``top_n`` as LightRAG expects them."""
    reranker = _get_reranker(model)
    pairs = [[query, doc] for doc in documents]

    loop = asyncio.get_event_loop()
    async with _reranker_lock:
        scores = await loop.run_in_executor(None, lambda: reranker.compute_score(pairs, normalize=True))

    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(enumerate(scores), key=itemgetter(1), reverse=True)
    return [{"index": i, "relevance_score": float(s)} for i, s in ranked[:top_n]]
