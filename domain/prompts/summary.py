"""Prompts for domain.summarization: topics + summary from a transcript (Macedonian)."""


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
