#!/usr/bin/env python3
"""
SPARQL query / analytics service over FOREST-KG: named analytic queries
plus a read-only raw-SPARQL passthrough, as a CLI or Flask HTTP service.

Usage:
    python scripts/kg_query_service.py --list
    python scripts/kg_query_service.py --run context_counts
    python scripts/kg_query_service.py --serve --port 5057
"""
import argparse
import os
import sys

import requests

QUERIES = {
    "context_counts": {
        "description": "Sound recordings per AcousticContext, most to least common.",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?context (COUNT(?s) AS ?n) WHERE {
                ?s a fkg:Sound ; fkg:hasAcousticContext ?c .
                ?c rdfs:label ?context .
            } GROUP BY ?context ORDER BY DESC(?n)
        """,
    },
    "soundtype_counts": {
        "description": "Sound recordings per SoundType (Urban vs Environmental).",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?type (COUNT(?s) AS ?n) WHERE {
                ?s a fkg:Sound ; fkg:inferredSoundType ?st .
                ?st rdfs:label ?type .
            } GROUP BY ?type ORDER BY DESC(?n)
        """,
    },
    "mean_aci_by_context": {
        "description": "Mean acoustic complexity index (aci), averaged per Frame then per "
                        "AcousticContext -- 2-hop aggregation: Frame -> owning Sound -> AcousticContext.",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?context (AVG(?aci) AS ?mean_aci) (COUNT(?f) AS ?n_frames) WHERE {
                ?s a fkg:Sound ; fkg:hasAcousticContext ?c ; fkg:hasFrame ?f .
                ?f fkg:aci ?aci .
                ?c rdfs:label ?context .
            } GROUP BY ?context ORDER BY DESC(?mean_aci)
        """,
    },
    "richest_recordings": {
        "description": "Top 10 Sound recordings by mean acoustic richness (amr) across their Frames.",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?sourceFile ?context (AVG(?amr) AS ?mean_richness) WHERE {
                ?s a fkg:Sound ; fkg:hasAcousticContext ?c ; fkg:hasFrame ?f ; fkg:sourceFile ?sourceFile .
                ?f fkg:acousticRichness ?amr .
                ?c rdfs:label ?context .
            } GROUP BY ?sourceFile ?context ORDER BY DESC(?mean_richness) LIMIT 10
        """,
    },
    "sensor_ambiguity": {
        "description": "Sensors that recorded more than one AcousticContext -- the reason "
                        "structure-only link prediction (KGE) underperforms a content-feature "
                        "GNN (see .carlos/notes.md), shown directly from the graph.",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?sensorId (COUNT(DISTINCT ?context) AS ?n_contexts)
                   (GROUP_CONCAT(DISTINCT ?context; separator=", ") AS ?contexts) WHERE {
                ?s a fkg:Sound ; fkg:recordedBy ?sensor ; fkg:hasAcousticContext ?c .
                ?sensor fkg:sensorId ?sensorId .
                ?c rdfs:label ?context .
            } GROUP BY ?sensorId
            HAVING (COUNT(DISTINCT ?context) > 1)
            ORDER BY DESC(?n_contexts) LIMIT 15
        """,
    },
    "municipality_breakdown": {
        "description": "Sound count per municipality -- uses fkg:inMunicipality, the "
                        "rule-derived property from kg_rules.py (LO2), in a service query (LO11).",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            SELECT ?municipality (COUNT(?s) AS ?n) WHERE {
                ?s a fkg:Sound ; fkg:inMunicipality ?municipality .
            } GROUP BY ?municipality ORDER BY DESC(?n)
        """,
    },
    "rule_consistency_check": {
        "description": "Data-quality check: every Sound's inferredSoundType (rule-derived) "
                        "should agree with its AcousticContext's own hasSoundType (schema-level). "
                        "Counts mismatches -- should be 0.",
        "sparql": """
            PREFIX fkg: <https://forest-kg.example.org/ontology#>
            SELECT (COUNT(?s) AS ?n_mismatches) WHERE {
                ?s a fkg:Sound ; fkg:hasAcousticContext ?c ; fkg:inferredSoundType ?inferred .
                ?c fkg:hasSoundType ?schema .
                FILTER (?inferred != ?schema)
            }
        """,
    },
}


def run_sparql(base_url: str, repo_id: str, query: str):
    resp = requests.get(
        f"{base_url}/repositories/{repo_id}",
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def bindings_to_rows(sparql_json: dict) -> list:
    rows = []
    for binding in sparql_json.get("results", {}).get("bindings", []):
        rows.append({k: v["value"] for k, v in binding.items()})
    return rows


def is_read_only(query: str) -> bool:
    q = query.strip().upper()
    lines = [l for l in q.splitlines() if not l.strip().startswith(("PREFIX", "BASE"))]
    q = " ".join(lines).strip()
    return q.startswith(("SELECT", "ASK", "CONSTRUCT", "DESCRIBE"))


def print_rows(rows: list):
    if not rows:
        print("(no results)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def make_app(base_url: str, repo_id: str):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/")
    def index():
        return jsonify({
            "service": "FOREST-KG query/analytics service",
            "graphdb": f"{base_url}/repositories/{repo_id}",
            "analytics_endpoints": {
                name: {"description": q["description"], "url": f"/analytics/{name}"}
                for name, q in QUERIES.items()
            },
            "sparql_endpoint": {"method": "POST", "url": "/sparql", "body": {"query": "<SELECT/ASK/CONSTRUCT/DESCRIBE only>"}},
        })

    @app.get("/analytics/<name>")
    def analytics(name):
        if name not in QUERIES:
            return jsonify({"error": f"unknown query '{name}'", "available": list(QUERIES)}), 404
        result = run_sparql(base_url, repo_id, QUERIES[name]["sparql"])
        return jsonify({"name": name, "description": QUERIES[name]["description"], "rows": bindings_to_rows(result)})

    @app.post("/sparql")
    def sparql():
        body = request.get_json(silent=True) or {}
        query = body.get("query", "")
        if not query:
            return jsonify({"error": "missing 'query' in request body"}), 400
        if not is_read_only(query):
            return jsonify({"error": "only SELECT/ASK/CONSTRUCT/DESCRIBE queries are allowed"}), 400
        try:
            result = run_sparql(base_url, repo_id, query)
        except requests.exceptions.HTTPError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"rows": bindings_to_rows(result)})

    return app


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graphdb-url", default=os.environ.get("GRAPHDB_URL", "http://localhost:7200"))
    parser.add_argument("--repo", default="forest-kg")
    parser.add_argument("--list", action="store_true", help="List available named queries and exit.")
    parser.add_argument("--run", metavar="NAME", help="Run one named query and print results.")
    parser.add_argument("--query", metavar="SPARQL", help="Run a raw SPARQL query and print results.")
    parser.add_argument("--serve", action="store_true", help="Start the Flask HTTP service instead.")
    parser.add_argument("--port", type=int, default=5057)
    args = parser.parse_args()

    if args.list:
        for name, q in QUERIES.items():
            print(f"{name}\n  {q['description']}\n")
        return

    if args.serve:
        app = make_app(args.graphdb_url, args.repo)
        print(f"Serving on http://0.0.0.0:{args.port}  (GraphDB: {args.graphdb_url}/repositories/{args.repo})")
        app.run(host="0.0.0.0", port=args.port)
        return

    if args.run:
        if args.run not in QUERIES:
            sys.exit(f"Unknown query '{args.run}'. --list to see available names.")
        q = QUERIES[args.run]
        print(f"{args.run}: {q['description']}\n")
        result = run_sparql(args.graphdb_url, args.repo, q["sparql"])
        print_rows(bindings_to_rows(result))
        return

    if args.query:
        if not is_read_only(args.query):
            sys.exit("Only SELECT/ASK/CONSTRUCT/DESCRIBE queries are allowed here.")
        result = run_sparql(args.graphdb_url, args.repo, args.query)
        print_rows(bindings_to_rows(result))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
