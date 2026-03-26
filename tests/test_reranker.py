import pytest
from rerankers.bge import bge_rerank

MODEL = "BAAI/bge-reranker-v2-m3"

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is in Paris.",
    "London is the capital of England.",
]


@pytest.mark.asyncio
class TestBgeRerank:

    async def test_returns_correct_number_of_results(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=3)
        assert len(results) == 3

    async def test_returns_all_when_top_n_exceeds_documents(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=100)
        assert len(results) == len(DOCUMENTS)

    async def test_result_keys(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=2)
        for r in results:
            assert "index" in r
            assert "relevance_score" in r

    async def test_scores_are_floats(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=2)
        assert all(isinstance(r["relevance_score"], float) for r in results)

    async def test_sorted_by_score_descending(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=4)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_most_relevant_document_ranked_first(self):
        results = await bge_rerank(query=QUERY, documents=DOCUMENTS, model=MODEL, top_n=4)
        assert results[0]["index"] == 0  # "Paris is the capital of France."

    async def test_single_document(self):
        results = await bge_rerank(query=QUERY, documents=["Paris is the capital."], model=MODEL, top_n=1)
        assert len(results) == 1
        assert results[0]["index"] == 0
