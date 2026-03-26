from __future__ import annotations

from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from config.settings import Config
from rerankers.bge import bge_rerank


def make_llm_func(config: Config):
    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            model=config.model.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    return llm_func


def make_embed_func(config: Config):
    async def embed_func(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return await openai_embed(texts, model=config.model.embedding_model, **kwargs)
    return embed_func


def make_rerank_func(config: Config):
    async def rerank_func(query: str, documents: list):
        return await bge_rerank(
            query=query,
            documents=documents,
            model=config.reranker.model,
            top_n=config.reranker.top_n,
        )
    return rerank_func
