"""Prompt for domain.keywords: pull speaker names and dates out of a question (Macedonian)."""


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
