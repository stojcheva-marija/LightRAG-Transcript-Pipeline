def build_transcript_enrichment_prompt(*, prev_block: str, current_line: str, next_block: str) -> str:
    return f"""
Даден е сегмент од транскрипт на разговор на македонски јазик.

Задача:

Врати го тековниот параграф збогатен со краток контекст, користејќи ги претходниот и следниот параграф.
   - НЕ додавај информации кои не се експлицитно кажани.
   - Ако нема доволно контекст, врати го тековниот параграф непроменет.

ПРЕТХОДЕН ПАРАГРАФ:
{prev_block}

ТЕКОВЕН ПАРАГРАФ:
{current_line}

СЛЕДЕН ПАРАГРАФ:
{next_block}

Одговори само со збогатениот текст (без никакви додатоци пред или после).
""".strip()


def build_topics_and_summary_system_prompt() -> str:
    return """You are analyzing a Macedonian TV show transcript.
Extract topics and generate a summary. Return ONLY valid JSON, no markdown, no explanation.
The summary should be in Macedonian language. Topics should be short phrases in Macedonian."""


def build_topics_and_summary_user_prompt(transcript_text: str) -> str:
    return f"""Transcript:
{transcript_text}

Return JSON with these exact keys:
{{
  "topics": ["topic1", "topic2", ...],
  "summary": "summary in Macedonian",
  "language": "mk"
}}"""
