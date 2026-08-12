"""Interface for reporting progress from long-running use cases.

Callers don't know where the message ends up — a log, a stream, etc.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    def report(self, message: str) -> None: ...


class LoggingProgress:
    """Default reporter: writes progress to the log and nowhere else."""

    def report(self, message: str) -> None:
        logger.info("%s", message)
