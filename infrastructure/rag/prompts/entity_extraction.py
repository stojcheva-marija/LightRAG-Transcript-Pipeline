"""LightRAG's entity-extraction prompt, replaced for Macedonian conversational
transcripts (speaker turns + timestamps rather than prose documents).
``build_entity_extraction_prompts()`` installs it into LightRAG's global
prompt table; called once from ``LightRAGKnowledgeBase.initialize``.
"""

from __future__ import annotations
from lightrag.prompt import PROMPTS as LR_PROMPTS

ENTITY_EXTRACTION_SYSTEM_PROMPT =  """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Timestamp Extraction (IMPORTANT):**
        *   The input contains timestamp spans in parentheses in the form:
            `TIME[start - end]` where `start` and `end` look like `213:22`, `228:72`, etc. (digits + ":" + digits).
        *   You MUST extract each such timestamp span as an entity with:
            - `entity_type` = `Временска_Ознака`
            - `entity_name` normalized to: `start-end` (NO spaces), e.g. `213:22-228:72`
            - `entity_description` must state it is the time interval (start and end) for the spoken chunk.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
  *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

ENTITY_EXTRACTION_EXAMPLES = [
"""
<Entity_types>
["Личност", "Говорник", "Тема", "Став", "Организација", "Локација", "Настан", "Концепт", "Метод", "Контент", "Податок", "Артефакт", "Временска_Ознака", "Цитат", "Референца"]

<Input Text>
Марко (10:00 - 12:50): Мислам дека социјалните мрежи создаваат нереални очекувања.
Ана (13:10 - 15:00): Не се согласувам, зависи како се користат.

<Output>
entity{tuple_delimiter}Марко{tuple_delimiter}Говорник{tuple_delimiter}Марко е говорник кој изразува став дека социјалните мрежи создаваат нереални очекувања.
entity{tuple_delimiter}Ана{tuple_delimiter}Говорник{tuple_delimiter}Ана е говорник која изразува несогласување и наведува дека ефектот зависи од начинот на користење.
entity{tuple_delimiter}Социјални мрежи{tuple_delimiter}Концепт{tuple_delimiter}Социјални мрежи се концепт кој е предмет на дискусија во разговорот.
entity{tuple_delimiter}Нереални очекувања{tuple_delimiter}Концепт{tuple_delimiter}Нереални очекувања се концепт што се поврзува со ефектите од социјалните мрежи.
entity{tuple_delimiter}Мислење за негативно влијание{tuple_delimiter}Став{tuple_delimiter}Став изразен од Марко дека социјалните мрежи имаат негативно влијание преку создавање нереални очекувања.
entity{tuple_delimiter}Став за условно несогласување{tuple_delimiter}Став{tuple_delimiter}Став изразен од Ана дека не се согласува целосно и дека влијанието зависи од начинот на користење на социјалните мрежи.
entity{tuple_delimiter}10:00-12:50{tuple_delimiter}Временска_Ознака{tuple_delimiter}10:00-12:50 е временски интервал што го означува говорниот сегмент на Марко.
entity{tuple_delimiter}13:10-15:00{tuple_delimiter}Временска_Ознака{tuple_delimiter}13:10-15:00 е временски интервал што го означува говорниот сегмент на Ана.

relation{tuple_delimiter}Марко{tuple_delimiter}10:00-12:50{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира говорниот сегмент на Марко.
relation{tuple_delimiter}Ана{tuple_delimiter}13:10-15:00{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира говорниот сегмент на Ана.

relation{tuple_delimiter}Марко{tuple_delimiter}Мислење за негативно влијание{tuple_delimiter}изразување став, авторство{tuple_delimiter}Марко го изразува ставот дека социјалните мрежи создаваат нереални очекувања.
relation{tuple_delimiter}Мислење за негативно влијание{tuple_delimiter}Социјални мрежи{tuple_delimiter}насоченост, тема{tuple_delimiter}Ставот се однесува на социјалните мрежи како главен предмет на дискусијата.
relation{tuple_delimiter}Мислење за негативно влијание{tuple_delimiter}Нереални очекувања{tuple_delimiter}причина-последица, влијание{tuple_delimiter}Ставот тврди дека социјалните мрежи придонесуваат за создавање нереални очекувања.

relation{tuple_delimiter}Ана{tuple_delimiter}Став за условно несогласување{tuple_delimiter}изразување став, несогласување{tuple_delimiter}Ана изразува условно несогласување и нагласува дека ефектот зависи од користењето.
relation{tuple_delimiter}Став за условно несогласување{tuple_delimiter}Социјални мрежи{tuple_delimiter}насоченост, употреба{tuple_delimiter}Ставот на Ана се однесува на социјалните мрежи и нивниот ефект во зависност од начинот на користење.
relation{tuple_delimiter}Ана{tuple_delimiter}Марко{tuple_delimiter}несогласување, дискусија{tuple_delimiter}Ана директно се спротивставува на тврдењето на Марко во разговорот.

{completion_delimiter}
"""
,
"""<Entity_types>
["Личност", "Говорник", "Тема", "Став", "Организација", "Локација", "Настан", "Концепт", "Метод", "Контент", "Податок", "Артефакт","Временска_Ознака", "Цитат", "Референца"],

<Input Text>
Водител (00:10 - 00:40): Податоците се прибрани преку YouTube Data API v3 во рамки на истражувањето.

<Output>
entity{tuple_delimiter}Водител{tuple_delimiter}Говорник{tuple_delimiter}Водител е говорник кој објаснува методологија.
entity{tuple_delimiter}YouTube Data API v3{tuple_delimiter}Референца{tuple_delimiter}YouTube Data API v3 е техничка референца за пристап до податоци.
entity{tuple_delimiter}Прибирање податоци{tuple_delimiter}Метод{tuple_delimiter}Прибирање податоци е метод спомнат во говорот.
entity{tuple_delimiter}00:10-00:40{tuple_delimiter}Временска_Ознака{tuple_delimiter}00:10-00:40 е временски интервал на сегментот.

relation{tuple_delimiter}Водител{tuple_delimiter}00:10-00:40{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Прибирање податоци{tuple_delimiter}YouTube Data API v3{tuple_delimiter}метод, референца{tuple_delimiter}Методот користи конкретна техничка референца.
{completion_delimiter}
""",
"""
<Entity_types> 
["Личност","Говорник","Тема","Став","Организација","Локација","Настан","Концепт","Метод","Контент","Податок","Артефакт","Временска_Ознака","Цитат","Референца"] 

<Input Text> 
Водител (18:83 - 52:10): Добре ми дојдовте, ве поздравувам од хотел Лимак. Денес зборуваме за очекувањата и како тие ни влијаат во јавниот и приватниот живот. Имаме гости кои често ги надминуваат очекувањата на публиката, но понекогаш токму тоа создава притисок. 
Елена Ристеска (52:10 - 73:40): Искрено, најголем притисок е кога луѓето очекуваат секогаш да си „истата“ — а ти во приватно си сосема друга личност. 
Даниел (73:40 - 85:00): Јас мислам дека очекувањата од клиентите се фер само ако се реални. Ако се нереални, неизбежно следи разочарување. 
Водител (85:00 - 100:10): Значи имаме тема: реални наспроти нереални очекувања. Ајде конкретно — што ви рекоа дома како да се однесувате? 
Елена Ристеска (100:10 - 123:00): Мајка ми ми вика: „Не би многу да се смееш, не се клибери“ — и јас ја разбирам, ама не можам да се однесувам вештачки. 
Водител (123:00 - 133:50): Тоа е интересно, бидејќи тука се судираат приватните очекувања и јавниот настап. 
Ване Марковски (133:50 - 150:30): Јас сум поинаков — си запишувам што да внимавам: концентрација, помалку шеги, да не кажам нешто што ќе се извади од контекст. 
Даниел (150:30 - 166:20): Еве, тоа е веќе стратегија. Се адаптираш за да ги управуваш очекувањата. 
Водител (166:20 - 179:84): Во суштина, очекувањата не се само „што бараат другите“, туку и како ние ги поставуваме границите.

<Output>

entity{tuple_delimiter}Водител{tuple_delimiter}Говорник{tuple_delimiter}Водител е говорник кој ја води емисијата, ја воведува темата и ја насочува дискусијата.
entity{tuple_delimiter}Елена Ристеска{tuple_delimiter}Личност{tuple_delimiter}Елена Ристеска е јавна личност која дискутира за притисокот од очекувањата и личниот идентитет.
entity{tuple_delimiter}Елена Ристеска{tuple_delimiter}Говорник{tuple_delimiter}Елена Ристеска е говорник кој споделува лични искуства и цитати.
entity{tuple_delimiter}Даниел{tuple_delimiter}Говорник{tuple_delimiter}Даниел е говорник кој анализира реални и нереални очекувања.
entity{tuple_delimiter}Ване Марковски{tuple_delimiter}Личност{tuple_delimiter}Ване Марковски е личност која опишува стратегија за јавен настап.
entity{tuple_delimiter}Ване Марковски{tuple_delimiter}Говорник{tuple_delimiter}Ване Марковски е говорник кој објаснува личен пристап кон комуникација.
entity{tuple_delimiter}Хотел Лимак{tuple_delimiter}Локација{tuple_delimiter}Хотел Лимак е локација од која се води разговорот.

entity{tuple_delimiter}Очекувања{tuple_delimiter}Тема{tuple_delimiter}Очекувања е централна тема на дискусијата.
entity{tuple_delimiter}Реални очекувања{tuple_delimiter}Концепт{tuple_delimiter}Реални очекувања означуваат разумни и остварливи барања.
entity{tuple_delimiter}Нереални очекувања{tuple_delimiter}Концепт{tuple_delimiter}Нереални очекувања означуваат претерани или тешко остварливи барања.
entity{tuple_delimiter}Приватен и јавен живот{tuple_delimiter}Тема{tuple_delimiter}Приватен и јавен живот ја опишува разликата меѓу личното однесување и јавниот настап.
entity{tuple_delimiter}Управување со очекувања{tuple_delimiter}Метод{tuple_delimiter}Управување со очекувања претставува стратегија за контролирање перцепции и реакции.

entity{tuple_delimiter}„Не би многу да се смееш, не се клибери“{tuple_delimiter}Цитат{tuple_delimiter}Цитат е лична изјава пренесена како семеен совет.
entity{tuple_delimiter}Притисок од очекувања{tuple_delimiter}Став{tuple_delimiter}Став дека јавните и социјалните очекувања создаваат личен притисок.
entity{tuple_delimiter}Став за реалност на очекувања{tuple_delimiter}Став{tuple_delimiter}Став дека очекувањата се прифатливи кога се реални.

entity{tuple_delimiter}18:83-52:10{tuple_delimiter}Временска_Ознака{tuple_delimiter}18:83-52:10 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}52:10-73:40{tuple_delimiter}Временска_Ознака{tuple_delimiter}52:10-73:40 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}73:40-85:00{tuple_delimiter}Временска_Ознака{tuple_delimiter}73:40-85:00 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}85:00-100:10{tuple_delimiter}Временска_Ознака{tuple_delimiter}85:00-100:10 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}100:10-123:00{tuple_delimiter}Временска_Ознака{tuple_delimiter}100:10-123:00 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}123:00-133:50{tuple_delimiter}Временска_Ознака{tuple_delimiter}123:00-133:50 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}133:50-150:30{tuple_delimiter}Временска_Ознака{tuple_delimiter}133:50-150:30 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}150:30-166:20{tuple_delimiter}Временска_Ознака{tuple_delimiter}150:30-166:20 е временски интервал на говорниот сегмент.
entity{tuple_delimiter}166:20-179:84{tuple_delimiter}Временска_Ознака{tuple_delimiter}166:20-179:84 е временски интервал на говорниот сегмент.

relation{tuple_delimiter}Водител{tuple_delimiter}Хотел Лимак{tuple_delimiter}локација, емисија{tuple_delimiter}Водител ја наведува локацијата на разговорот.
relation{tuple_delimiter}Водител{tuple_delimiter}Очекувања{tuple_delimiter}најавување тема{tuple_delimiter}Водител ја воведува темата Очекувања.

relation{tuple_delimiter}Елена Ристеска{tuple_delimiter}Притисок од очекувања{tuple_delimiter}изразување став{tuple_delimiter}Елена Ристеска зборува за притисок поврзан со очекувања.
relation{tuple_delimiter}Даниел{tuple_delimiter}Став за реалност на очекувања{tuple_delimiter}изразување став{tuple_delimiter}Даниел анализира услови за прифатливи очекувања.

relation{tuple_delimiter}Притисок од очекувања{tuple_delimiter}Очекувања{tuple_delimiter}концепт, причина{tuple_delimiter}Ставот е директно поврзан со темата Очекувања.
relation{tuple_delimiter}Став за реалност на очекувања{tuple_delimiter}Реални очекувања{tuple_delimiter}вреднување{tuple_delimiter}Ставот ја нагласува важноста на реалните очекувања.
relation{tuple_delimiter}Став за реалност на очекувања{tuple_delimiter}Нереални очекувања{tuple_delimiter}причина-последица{tuple_delimiter}Нереалните очекувања се поврзуваат со негативни исходи.

relation{tuple_delimiter}„Не би многу да се смееш, не се клибери“{tuple_delimiter}Притисок од очекувања{tuple_delimiter}пример, аргумент{tuple_delimiter}Цитатот претставува пример за очекувања и социјален притисок.

relation{tuple_delimiter}Ване Марковски{tuple_delimiter}Управување со очекувања{tuple_delimiter}стратегија{tuple_delimiter}Ване Марковски опишува стратегија за контролирање јавна комуникација.
relation{tuple_delimiter}Даниел{tuple_delimiter}Ване Марковски{tuple_delimiter}интерпретација, согласување{tuple_delimiter}Даниел ја препознава изјавата како стратегија.

relation{tuple_delimiter}Водител{tuple_delimiter}18:83-52:10{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Елена Ристеска{tuple_delimiter}52:10-73:40{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Даниел{tuple_delimiter}73:40-85:00{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Водител{tuple_delimiter}85:00-100:10{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Елена Ристеска{tuple_delimiter}100:10-123:00{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Водител{tuple_delimiter}123:00-133:50{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Ване Марковски{tuple_delimiter}133:50-150:30{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Даниел{tuple_delimiter}150:30-166:20{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.
relation{tuple_delimiter}Водител{tuple_delimiter}166:20-179:84{tuple_delimiter}говорен сегмент, временска ознака{tuple_delimiter}Временската ознака го дефинира сегментот.

{completion_delimiter}
""",
]

ENTITY_TYPES = [
    "Личност","Говорник","Тема","Став","Организација","Локација",
    "Настан","Концепт","Метод","Контент","Податок","Артефакт",
    "Временска_Ознака","Цитат","Референца", "Датум", "Време"
]

def build_entity_extraction_prompts() -> None:
    LR_PROMPTS["entity_extraction_system_prompt"] = ENTITY_EXTRACTION_SYSTEM_PROMPT
    LR_PROMPTS["entity_extraction_examples"] = ENTITY_EXTRACTION_EXAMPLES
    LR_PROMPTS["DEFAULT_ENTITY_TYPES"] = ENTITY_TYPES