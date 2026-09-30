#!/usr/bin/env python3
"""
Rule-based inference over FOREST-KG (LO2, basic proficiency: simple
non-recursive property-chain rules, not a full reasoner -- GraphDB's
built-in reasoner is intentionally left off, see scripts/load_kg.py).

Two rules, each a SPARQL 1.1 Update INSERT...WHERE, idempotent via
FILTER NOT EXISTS so re-running never duplicates triples:

  R1  Sound hasRegion Region hasSoundClass SoundClass
      => Sound inferredSoundClass SoundClass

  R2  Sound recordedBy Sensor municipality M
      => Sound inMunicipality M

New properties (fkg:inferredSoundClass, fkg:inMunicipality) are declared
in scripts/kg_ontology.ttl and never asserted by scripts/build_kg_triples.py
-- only by this script, so a triple on either property is, by construction,
a materialized fact rather than raw extracted data.

Usage:
    python scripts/kg_rules.py
    python scripts/kg_rules.py --graphdb-url http://localhost:7200 --repo forest-kg

Environment: only needs `requests` (present in the tpyforest conda env).
Requires the KG already loaded -- see scripts/load_kg.py.
"""
import argparse
import os
import sys

import requests

RULES = {
    "R1: Sound inferredSoundClass (via hasRegion -> hasSoundClass)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        INSERT { ?sound fkg:inferredSoundClass ?class }
        WHERE {
            ?sound a fkg:Sound ; fkg:hasRegion ?region .
            ?region fkg:hasSoundClass ?class .
            FILTER NOT EXISTS { ?sound fkg:inferredSoundClass ?class }
        }
    """,
    "R2: Sound inMunicipality (via recordedBy -> municipality)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        INSERT { ?sound fkg:inMunicipality ?m }
        WHERE {
            ?sound a fkg:Sound ; fkg:recordedBy ?sensor .
            ?sensor fkg:municipality ?m .
            FILTER NOT EXISTS { ?sound fkg:inMunicipality ?m }
        }
    """,
}

SANITY_QUERIES = {
    "inferredSoundClass count": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(*) AS ?n) WHERE { ?s fkg:inferredSoundClass ?c }
    """,
    "inMunicipality count": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(*) AS ?n) WHERE { ?s fkg:inMunicipality ?m }
    """,
    "Sound with no inferredSoundClass (should be 0)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(?s) AS ?n) WHERE {
            ?s a fkg:Sound .
            FILTER NOT EXISTS { ?s fkg:inferredSoundClass ?c }
        }
    """,
}


def run_update(base_url: str, repo_id: str, update: str):
    resp = requests.post(
        f"{base_url}/repositories/{repo_id}/statements",
        data=update.encode("utf-8"),
        headers={"Content-Type": "application/sparql-update; charset=utf-8"},
        timeout=120,
    )
    resp.raise_for_status()


def run_query(base_url: str, repo_id: str, query: str):
    resp = requests.get(
        f"{base_url}/repositories/{repo_id}",
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--graphdb-url", default=os.environ.get("GRAPHDB_URL", "http://localhost:7200")
    )
    parser.add_argument("--repo", default="forest-kg")
    args = parser.parse_args()

    try:
        requests.get(args.graphdb_url, timeout=5)
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Could not reach GraphDB at {args.graphdb_url}.\n"
            f"Start it with: docker compose -f scripts/graphdb-docker-compose.yml up -d"
        )

    for label, update in RULES.items():
        run_update(args.graphdb_url, args.repo, update)
        print(f"Applied {label}")

    print("\nSanity checks:")
    for label, query in SANITY_QUERIES.items():
        rows = run_query(args.graphdb_url, args.repo, query)
        value = rows[0]["n"]["value"] if rows else "?"
        print(f"  {label}: {value}")


if __name__ == "__main__":
    main()
