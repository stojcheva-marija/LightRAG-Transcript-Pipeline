from __future__ import annotations

import json
import os
import re
from typing import Any

import psycopg2
from dotenv import load_dotenv
from pyvis.network import Network

from config.settings import DatabaseConfig

load_dotenv()

AGE_GRAPH_NAME = os.environ.get("AGE_GRAPH_NAME", "openai_graph_chunk_entity_relation")
EDGE_LIMIT     = int(os.environ.get("GRAPH_EDGE_LIMIT", "2000"))
OUTPUT_FILE    = "transcript_graph.html"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]

_AGTYPE_SUFFIX_RE = re.compile(r"::agtype\s*$", re.IGNORECASE)


def _strip_agtype_suffix(s: str) -> str:
    return _AGTYPE_SUFFIX_RE.sub("", s.strip())


def _extract_first_json_object(s: str) -> str | None:
    """Extract the first balanced {...} from an AGE agtype string."""
    s = _strip_agtype_suffix(str(s).strip())
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_node(node_ag: Any) -> tuple[str, str, str]:
    """Returns (node_uid, entity_name, entity_type)."""
    raw = str(node_ag)
    js = _extract_first_json_object(raw)
    if not js:
        return f"node_{hash(raw)}", raw[:60], "UNKNOWN"

    obj = json.loads(js)
    internal_id = obj.get("id", hash(js))
    props = obj.get("properties") or {}
    entity_name = (
        props.get("entity_id")
        or props.get("name")
        or props.get("title")
        or props.get("id")
        or f"node_{internal_id}"
    )
    entity_type = props.get("entity_type") or "UNKNOWN"
    return f"node_{internal_id}", str(entity_name), str(entity_type)


def _parse_rel_type(rel_ag: Any) -> str:
    raw = str(rel_ag).strip().strip('"')
    return raw if raw and raw != "null" else "REL"


def _type_color_map(entity_types: set[str]) -> dict[str, str]:
    return {
        entity_type: PALETTE[idx % len(PALETTE)]
        for idx, entity_type in enumerate(sorted(entity_types))
    }


def fetch_edges(cur, graph_name: str, limit: int | None) -> list[tuple]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    cur.execute(
        f"""
        SELECT *
        FROM cypher(%s, $$
            MATCH (n)-[r]->(m)
            RETURN n, type(r), m
            {limit_clause}
        $$) AS (source agtype, rel agtype, target agtype);
        """,
        (graph_name,),
    )
    return cur.fetchall()


def parse_graph(rows: list[tuple]) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str, str]]]:
    nodes: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str, str]] = []
    for src_ag, rel_ag, tgt_ag in rows:
        src_uid, src_name, src_type = _parse_node(src_ag)
        tgt_uid, tgt_name, tgt_type = _parse_node(tgt_ag)
        nodes[src_uid] = (src_name, src_type)
        nodes[tgt_uid] = (tgt_name, tgt_type)
        edges.append((src_uid, tgt_uid, _parse_rel_type(rel_ag)))
    return nodes, edges


def build_network(
    nodes: dict[str, tuple[str, str]],
    edges: list[tuple[str, str, str]],
    type_color: dict[str, str],
) -> Network:
    net = Network(
        height="900px", width="100%",
        bgcolor="white", font_color="black",
        directed=True,
    )
    for uid, (name, entity_type) in nodes.items():
        net.add_node(
            uid,
            label=name,
            title=f"Type: {entity_type}\nUID: {uid}",
            color=type_color.get(entity_type, "#cccccc"),
        )
    for src_uid, tgt_uid, rel_type in edges:
        net.add_edge(src_uid, tgt_uid, title=rel_type, label="", arrows="to")

    net.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 200},
        },
        "edges": {
            "smooth": {"type": "dynamic"},
            "arrows": {"to": {"enabled": False}, "from": {"enabled": False}},
            "color": {"inherit": True},
        },
        "nodes": {"shape": "dot", "size": 12, "font": {"size": 16}},
        "interaction": {"hover": True, "navigationButtons": True, "multiselect": True},
    }))
    return net


def print_stats(nodes: dict, edges: list, type_color: dict[str, str]) -> None:
    print(f"\n{'=' * 44}")
    print(f"  Nodes: {len(nodes)}   Edges: {len(edges)}")
    print(f"{'=' * 44}\n")
    print("Entity type → color:")
    for entity_type, color in sorted(type_color.items()):
        print(f"  {entity_type:<25} {color}")
    print()


def main() -> None:
    db = DatabaseConfig.from_env()
    conn = psycopg2.connect(**db.to_dict())
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            print("Fetching graph data...")
            rows = fetch_edges(cur, AGE_GRAPH_NAME, EDGE_LIMIT)

        if not rows:
            print("No edges found. Insert documents first.")
            return

        nodes, edges = parse_graph(rows)
        entity_types = {entity_type for _, entity_type in nodes.values()}
        type_color = _type_color_map(entity_types)

        print_stats(nodes, edges, type_color)

        net = build_network(nodes, edges, type_color)
        net.save_graph(OUTPUT_FILE)
        print(f"Graph saved to {OUTPUT_FILE}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
