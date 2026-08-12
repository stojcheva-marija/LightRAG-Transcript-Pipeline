"""Prompts for domain.speaker_identification: name a voice from what it said."""


def build_speaker_identification_system_prompt() -> str:
    return """You are an expert at identifying speakers in transcripts.
Given a transcript excerpt and a list of possibly present speakers, identify which speaker ID corresponds to which person.
Return ONLY a JSON object mapping speaker_id to name. If you cannot identify someone, map them to null.
Do not add any explanation."""


def build_speaker_identification_user_prompt(
    hints: list[str],
    excerpt: str,
    unknown_ids: list[str],
) -> str:
    hints_str = ", ".join(hints) if hints else "None provided"
    return f"""Known speakers possibly in the audio: {hints_str}

Transcript excerpt:
{excerpt}

Unknown speaker IDs to identify: {unknown_ids}

Return JSON like: {{"speaker_1": "Name", "speaker_2": null}}"""