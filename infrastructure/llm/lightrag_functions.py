"""The plain callables LightRAG expects for completion, embedding and reranking."""

from __future__ import annotations

from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from config.settings import ModelConfig, RerankerConfig
from infrastructure.llm.bge_reranker import bge_rerank


def make_llm_func(model_config: ModelConfig):
    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            model=model_config.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    return llm_func


def make_embed_func(model_config: ModelConfig):
    async def embed_func(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return await openai_embed(texts, model=model_config.embedding_model, **kwargs)
    return embed_func


def make_rerank_func(reranker_config: RerankerConfig):
    async def rerank_func(query: str, documents: list, top_n: int = None):
        return await bge_rerank(
            query=query,
            documents=documents,
            model=reranker_config.model,
            top_n=top_n if top_n is not None else reranker_config.top_n,
        )
    return rerank_func
