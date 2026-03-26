from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from prompts.mk.transcript_enrichment import (
    build_topics_and_summary_system_prompt,
    build_topics_and_summary_user_prompt,
)

logger = logging.getLogger(__name__)


def build_metadata_header(metadata: str) -> str:
    if not metadata:
        return ""
    try:
        m = json.loads(metadata)
        parts = [
            f"[DATE={m['date']}]" if "date" in m else "",
            f"[SHOW={m['show']}]" if "show" in m else "",
        ]
        parts = [p for p in parts if p]
        return "\n".join(parts) + "\n" if parts else ""
    except Exception:
        return ""


async def generate_topics_and_summary_async(
    transcript: list[dict],
    final_mapping: dict[str, str],
    openai_api_key: str,
    model: str,
) -> dict:
    client = AsyncOpenAI(api_key=openai_api_key)

    lines = [
        f"{final_mapping.get(turn['speaker'], turn['speaker'])}: {turn['text']}"
        for turn in transcript
    ]
    full_text = "\n".join(lines)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_topics_and_summary_system_prompt()},
                {"role": "user", "content": build_topics_and_summary_user_prompt(full_text)},
            ],
            max_tokens=800,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("LLM summary generation failed: %s", e)
        return {"topics": [], "summary": "", "language": "mk"}


def build_metadata_json(
    transcript: list[dict],
    final_mapping: dict[str, str],
    date: str,
    time_str: str,
    show_name: str,
    location: str,
    llm_summary: dict,
) -> dict:
    duration = int(transcript[-1]["end"] - transcript[0]["start"]) if transcript else 0
    speakers_ordered = list(dict.fromkeys(
        final_mapping.get(turn["speaker"], turn["speaker"]) for turn in transcript
    ))

    return {
        "date": date,
        "time": time_str,
        "duration_seconds": duration,
        "speakers": speakers_ordered,
        "speaker_count": len(speakers_ordered),
        "topics": llm_summary.get("topics", []),
        "language": llm_summary.get("language", "mk"),
        "show": show_name,
        "location": location,
        "summary": llm_summary.get("summary", ""),
    }


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_transcript_txt(transcript: list[dict], final_mapping: dict[str, str]) -> str:
    lines = [
        f"{final_mapping.get(turn['speaker'], turn['speaker'])} ({_fmt_time(turn['start'])} - {_fmt_time(turn['end'])}): {turn['text']}"
        for turn in transcript
    ]
    return "\n\n".join(lines)
