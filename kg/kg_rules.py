#!/usr/bin/env python3
"""
Rule-based inference over FOREST-KG: two non-recursive SPARQL 1.1 Update
rules, idempotent via FILTER NOT EXISTS.

  R1  Sound hasAcousticContext AcousticContext hasSoundType SoundType
      => Sound inferredSoundType SoundType

  R2  Sound recordedBy Sensor municipality M
      => Sound inMunicipality M

Usage:
    python scripts/kg_rules.py --graphdb-url http://localhost:7200 --repo forest-kg
"""
import argparse
import os
import sys

import requests

RULES = {
    "R1: Sound inferredSoundType (via hasAcousticContext -> hasSoundType)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        INSERT { ?sound fkg:inferredSoundType ?type }
        WHERE {
            ?sound a fkg:Sound ; fkg:hasAcousticContext ?ctx .
            ?ctx fkg:hasSoundType ?type .
            FILTER NOT EXISTS { ?sound fkg:inferredSoundType ?type }
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
    "inferredSoundType count": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(*) AS ?n) WHERE { ?s fkg:inferredSoundType ?c }
    """,
    "inMunicipality count": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(*) AS ?n) WHERE { ?s fkg:inMunicipality ?m }
    """,
    "Sound with no inferredSoundType (should be 0)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(?s) AS ?n) WHERE {
            ?s a fkg:Sound .
            FILTER NOT EXISTS { ?s fkg:inferredSoundType ?c }
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
