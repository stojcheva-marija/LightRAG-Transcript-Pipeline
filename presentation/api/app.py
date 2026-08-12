"""FastAPI app: wires the container into ``lifespan`` and mounts the routers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from composition import build_container
from presentation.api.routes import pipeline, query, transcripts


ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost"]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container()
        await app.state.container.startup()
        yield
        await app.state.container.shutdown()

    app = FastAPI(title="LightRAG Transcript API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (query.router, pipeline.router, transcripts.router):
        app.include_router(router, prefix="/api")

    return app


app = create_app()
