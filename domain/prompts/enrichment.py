"""Prompt for the contextual chunker: rewrite one line using its neighbours (Macedonian)."""


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
