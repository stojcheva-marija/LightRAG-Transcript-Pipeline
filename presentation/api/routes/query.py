from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from composition import Container
from presentation.api.dependencies import get_container
from presentation.api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, container: Container = Depends(get_container)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = await container.query.ask(request.question, request.history)
    return QueryResponse.from_answer(answer)
