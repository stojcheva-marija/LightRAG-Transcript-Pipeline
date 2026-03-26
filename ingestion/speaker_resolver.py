from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from openai import AsyncOpenAI

from prompts.speaker_identification import (
    build_speaker_identification_system_prompt,
    build_speaker_identification_user_prompt,
)

logger = logging.getLogger(__name__)


def resolve_speakers_from_db(
    speaker_centroids: dict[str, np.ndarray],
    speaker_repo,
    threshold: float,
) -> dict[str, dict]:
    results = {}
    for spk_id, centroid in speaker_centroids.items():
        match = speaker_repo.match_speaker(centroid, threshold=threshold)
        if match:
            results[spk_id] = {
                "name": match["name"],
                "similarity": round(match["similarity"], 4),
                "source": "db",
            }
        else:
            results[spk_id] = {
                "name": spk_id,
                "similarity": None,
                "source": "unknown",
            }
    return results


async def llm_identify_speakers_async(
    transcript: list[dict],
    unknown_speaker_ids: list[str],
    user_hints: list[str],
    openai_api_key: str,
    model: str,
) -> dict[str, str]:
    if not unknown_speaker_ids:
        return {}

    client = AsyncOpenAI(api_key=openai_api_key)
    excerpt = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in transcript)
    user_prompt = build_speaker_identification_user_prompt(
        hints=user_hints,
        excerpt=excerpt,
        unknown_ids=unknown_speaker_ids,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_speaker_identification_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"LLM identity inference failed: {e}")
        return {}


def llm_identify_speakers(
    transcript: list[dict],
    unknown_speaker_ids: list[str],
    user_hints: list[str],
    openai_api_key: str,
    model: str,
) -> dict[str, str]:
    return asyncio.run(
        llm_identify_speakers_async(transcript, unknown_speaker_ids, user_hints, openai_api_key, model)
    )


def build_final_speaker_mapping(
    speaker_results: dict[str, dict],
    llm_identities: dict[str, str],
) -> dict[str, str]:
    return {
        spk_id: (
            info["name"] if info["source"] == "db"
            else llm_identities[spk_id] if llm_identities.get(spk_id)
            else spk_id
        )
        for spk_id, info in speaker_results.items()
    }


def generate_resolution_report(
    speaker_results: dict[str, dict],
    llm_identities: dict[str, str],
    final_mapping: dict[str, str],
) -> str:
    lines = ["## Speaker Resolution Report", ""]
    for spk_id in sorted(speaker_results.keys()):
        info = speaker_results[spk_id]
        final_name = final_mapping[spk_id]

        if info["source"] == "db":
            lines.append(f"**{spk_id}** → `{final_name}` matched from DB (cosine similarity: {info['similarity']:.3f})")
        elif llm_identities.get(spk_id):
            lines.append(f"**{spk_id}** → `{final_name}` identified by LLM")
        else:
            lines.append(f"**{spk_id}** → `{final_name}` unknown (not in DB, not identified)")
    return "\n".join(lines)
