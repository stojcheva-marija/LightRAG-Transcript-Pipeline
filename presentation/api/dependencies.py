from __future__ import annotations

from fastapi import Request

from composition import Container


def get_container(request: Request) -> Container:
    """The container built once at startup and shared by every request."""
    return request.app.state.container
