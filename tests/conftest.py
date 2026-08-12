from __future__ import annotations

import pytest

from tests.fakes import RecordingProgress


@pytest.fixture
def progress() -> RecordingProgress:
    return RecordingProgress()
