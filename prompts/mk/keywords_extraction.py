from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


async def extract_keywords(
    question: str,
    llm_func,
    known_speakers: list[str],
) -> tuple[list[str], list[str]]:
    prompt = build_keyword_extraction_prompt(question, known_speakers)
    resp = None
    try:
        resp = await llm_func(prompt=prompt)
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {repr(resp)}")
        data = json.loads(match.group())
        speakers = data.get("speakers", [])
        ll_keywords = [d for d in data.get("dates", []) if len(d) in {2, 5, 10}]
        return speakers, ll_keywords
    except Exception as e:
        logger.warning("Keyword extraction failed: %s | raw resp: %r", e, resp)
        return [], []


def build_keyword_extraction_prompt(question: str, known_speakers: list[str]) -> str:
    return f"""
Од следниот текст наведен подолу, извлечи:
1. Датуми доколку се споменати:
   - Ако е целосен датум (ден, месец, година) врати го во формат YYYY-MM-DD
   - Ако е само ден и месец без година врати го во формат MM-DD
   - Ако е само месец врати го во формат MM
2. Имиња ако се споменати и се присутни во оваа листа: {known_speakers}
   Дури и делумни совпаѓања се прифатливи, но доколку има повеќе совпаѓања врати ги сите.
   (пр. "Елена" -> ["Елена Ристеска", "Елена Спасовска"])

Текст: {question}

Одговори САМО во JSON без дополнителен текст: {{"dates": [], "speakers": []}}
"""