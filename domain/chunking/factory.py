from __future__ import annotations

from typing import Callable

from domain.chunking.base import BaseChunker
from domain.chunking.block import BlockChunker
from domain.chunking.contextual import ContextualChunker
from domain.parsing.base import BaseParser
from domain.parsing.transcript import TranscriptParser
from domain.ports import TextCompleter

# Registry of chunking strategies keyed by the configured chunker type.
# Add a strategy by implementing BaseChunker and registering it here.
_BUILDERS: dict[str, Callable[..., BaseChunker]] = {
    "block": lambda parser, completer, block_window, context_window: BlockChunker(
        parser=parser,
        window_size=block_window,
    ),
    "contextual": lambda parser, completer, block_window, context_window: ContextualChunker(
        completer,
        parser=parser,
        context_window=context_window,
    ),
}


def make_chunker(
    chunker_type: str,
    *,
    completer: TextCompleter,
    block_window_size: int,
    contextual_window_size: int,
    parser: BaseParser | None = None,
) -> BaseChunker:
    """Build the requested chunking strategy."""
    builder = _BUILDERS.get(chunker_type.strip().lower())
    if builder is None:
        raise ValueError(
            f"Unknown chunker type '{chunker_type}'. "
            f"Expected one of: {', '.join(sorted(_BUILDERS))}."
        )
    return builder(parser or TranscriptParser(), completer, block_window_size, contextual_window_size)
