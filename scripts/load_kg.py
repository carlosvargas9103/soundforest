#!/usr/bin/env python3
"""
Load FOREST-KG (scripts/kg_ontology.ttl + out/data/kg/triples.nt, from
scripts/build_kg_triples.py) into a GraphDB repository.

Creates the repository if it doesn't exist yet (empty ruleset -- inference
is handled explicitly by a later rule-based-reasoning step, not by
GraphDB's built-in reasoner), then loads the ontology and the generated
triples, then runs a couple of sanity SPARQL queries.

GraphDB itself: scripts/graphdb-docker-compose.yml
    docker compose -f scripts/graphdb-docker-compose.yml up -d

The endpoint is read from --graphdb-url / $GRAPHDB_URL rather than
hardcoded, since the eventual Gradio app (on Hugging Face) needs to reach
GraphDB wherever it ends up running, not just localhost.

Usage:
    python scripts/load_kg.py
    python scripts/load_kg.py --graphdb-url http://localhost:7200 --repo forest-kg
    GRAPHDB_URL=https://my-graphdb.example.org python scripts/load_kg.py

Environment: only needs `requests` (present in the tpyforest conda env).
"""
import argparse
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONTOLOGY = REPO_ROOT / "scripts" / "kg_ontology.ttl"
DEFAULT_TRIPLES = REPO_ROOT / "out" / "data" / "kg" / "triples.nt"

REPO_CONFIG_TEMPLATE = """\
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep: <http://www.openrdf.org/config/repository#> .
@prefix sr: <http://www.openrdf.org/config/repository/sail#> .
@prefix sail: <http://www.openrdf.org/config/sail#> .
@prefix graphdb: <http://www.ontotext.com/config/graphdb#> .

[] a rep:Repository ;
    rep:repositoryID "{repo_id}" ;
    rdfs:label "FOREST-KG" ;
    rep:repositoryImpl [
        rep:repositoryType "graphdb:SailRepository" ;
        sr:sailImpl [
            sail:sailType "graphdb:Sail" ;
            graphdb:read-only "false" ;
            graphdb:ruleset "empty" ;
            graphdb:disable-sameAs "true" ;
            graphdb:check-for-inconsistencies "false" ;
            graphdb:enablePredicateList "true" ;
            graphdb:base-URL "https://forest-kg.example.org/resource/" ;
            graphdb:repository-type "file-repository" ;
            graphdb:storage-folder "storage" ;
            graphdb:entity-index-size "10000000" ;
            graphdb:enable-literal-index "true" ;
        ]
    ] .
"""

SANITY_QUERIES = {
    "total triples": "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }",
    "Sound count": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        SELECT (COUNT(?s) AS ?n) WHERE { ?s a fkg:Sound }
    """,
    "Sound count per AcousticContext": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?context (COUNT(?s) AS ?n) WHERE {
            ?s a fkg:Sound ; fkg:hasAcousticContext ?c .
            ?c rdfs:label ?context .
        } GROUP BY ?context ORDER BY DESC(?n)
    """,
    "Sound count per SoundType (via hasAcousticContext -> hasSoundType)": """
        PREFIX fkg: <https://forest-kg.example.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?type (COUNT(?s) AS ?n) WHERE {
            ?s a fkg:Sound ; fkg:hasAcousticContext ?c .
            ?c fkg:hasSoundType ?st .
            ?st rdfs:label ?type .
        } GROUP BY ?type
    """,
}


def repo_exists(base_url: str, repo_id: str) -> bool:
    resp = requests.get(f"{base_url}/rest/repositories", timeout=30)
    resp.raise_for_status()
    return any(r.get("id") == repo_id for r in resp.json())


def create_repo(base_url: str, repo_id: str):
    config = REPO_CONFIG_TEMPLATE.format(repo_id=repo_id)
    resp = requests.post(
        f"{base_url}/rest/repositories",
        files={"config": ("config.ttl", config, "text/turtle")},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Created repository '{repo_id}'.")


def load_file(base_url: str, repo_id: str, path: Path, content_type: str):
    with open(path, "rb") as fh:
        resp = requests.post(
            f"{base_url}/repositories/{repo_id}/statements",
            data=fh,
            headers={"Content-Type": content_type},
            timeout=600,
        )
    resp.raise_for_status()
    print(f"Loaded {path} ({content_type}).")


def run_sparql(base_url: str, repo_id: str, query: str):
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
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the repository if it already exists.",
    )
    args = parser.parse_args()

    if not args.ontology.exists():
        sys.exit(f"Ontology not found: {args.ontology}")
    if not args.triples.exists():
        sys.exit(f"Triples not found: {args.triples} (run scripts/build_kg_triples.py first)")

    try:
        requests.get(args.graphdb_url, timeout=5)
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Could not reach GraphDB at {args.graphdb_url}.\n"
            f"Start it with: docker compose -f scripts/graphdb-docker-compose.yml up -d"
        )

    exists = repo_exists(args.graphdb_url, args.repo)
    if exists and args.recreate:
        requests.delete(f"{args.graphdb_url}/rest/repositories/{args.repo}", timeout=30).raise_for_status()
        print(f"Deleted existing repository '{args.repo}'.")
        exists = False
    if not exists:
        create_repo(args.graphdb_url, args.repo)
    else:
        print(f"Repository '{args.repo}' already exists, loading into it.")

    load_file(args.graphdb_url, args.repo, args.ontology, "text/turtle")
    load_file(args.graphdb_url, args.repo, args.triples, "application/n-triples")

    print("\nSanity checks:")
    for label, query in SANITY_QUERIES.items():
        rows = run_sparql(args.graphdb_url, args.repo, query)
        print(f"\n{label}:")
        for row in rows:
            print("  " + ", ".join(f"{k}={v['value']}" for k, v in row.items()))


if __name__ == "__main__":
    main()
