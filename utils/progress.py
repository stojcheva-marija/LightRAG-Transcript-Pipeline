from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class Progress:
    def __init__(self, fn: Callable[[int, str], None] | None = None) -> None:
        self._fn = fn

    def __call__(self, pct: int, message: str) -> None:
        logger.info(f"[{pct}%] {message}")
        if self._fn:
            self._fn(pct, message)
