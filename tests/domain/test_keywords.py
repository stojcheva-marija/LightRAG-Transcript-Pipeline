from __future__ import annotations

import json

import pytest

from domain.keywords import extract_keywords
from tests.fakes import FakeCompleter


async def extract(response: str, known_speakers=None):
    return await extract_keywords(
        "Кога зборуваше Ана?", FakeCompleter([response]), known_speakers or ["Ана"]
    )


@pytest.mark.asyncio
async def test_extracts_speakers_and_dates():
    keywords = await extract(json.dumps({"speakers": ["Ана"], "dates": ["2024-01-01"]}))
    assert keywords.speakers == ["Ана"]
    assert keywords.dates == ["2024-01-01"]


@pytest.mark.asyncio
async def test_accepts_month_and_month_day_formats():
    keywords = await extract(json.dumps({"speakers": [], "dates": ["03", "03-14", "2024-03-14"]}))
    assert keywords.dates == ["03", "03-14", "2024-03-14"]


@pytest.mark.asyncio
async def test_drops_dates_of_unexpected_shape():
    keywords = await extract(json.dumps({"speakers": [], "dates": ["yesterday", "2024"]}))
    assert keywords.dates == []


@pytest.mark.asyncio
async def test_reads_json_embedded_in_prose():
    keywords = await extract('Еве: {"speakers": ["Ана"], "dates": []} — готово')
    assert keywords.speakers == ["Ана"]


@pytest.mark.asyncio
async def test_unparseable_response_yields_no_keywords():
    keywords = await extract("no json at all")
    assert keywords.speakers == [] and keywords.dates == []


@pytest.mark.asyncio
async def test_malformed_json_yields_no_keywords():
    keywords = await extract('{"speakers": [')
    assert keywords.speakers == [] and keywords.dates == []


@pytest.mark.asyncio
async def test_known_speakers_are_offered_to_the_model():
    completer = FakeCompleter(['{"speakers": [], "dates": []}'])
    await extract_keywords("Кој?", completer, ["Ана Ристеска"])
    assert "Ана Ристеска" in completer.prompts[0]
