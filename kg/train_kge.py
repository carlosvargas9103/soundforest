#!/usr/bin/env python3
"""
KG embeddings (TransE, ComplEx) on FOREST-KG: link prediction for
Sound.hasAcousticContext (13-way) and Sound.inferredSoundType (2-way),
one prediction per recording. Reads relational triples directly from
out/data/kg/triples.nt (not GraphDB).

Usage:
    python scripts/train_kge.py
    python scripts/train_kge.py --sample-fraction 0.10
    python scripts/train_kge.py --models transe --epochs 50 --embedding-dim 128
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIPLES = REPO_ROOT / "out" / "data" / "kg" / "triples.nt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "data" / "kg" / "kge"

FKG = "https://forest-kg.example.org/ontology#"
RES = "https://forest-kg.example.org/resource/"

FOREST_CONTEXTS = {"Plantation", "Pasture", "NaturalRegeneration", "RefForest"}
ALL_CONTEXTS = FOREST_CONTEXTS | {
    "Noise", "Engine", "MachineryImpact", "NonMachineryImpact", "PoweredSaw",
    "AlertSignal", "Music", "HumanVoice", "Dog",
}


def sound_type_for(context: str) -> str:
    return "Environmental" if context in FOREST_CONTEXTS else "Urban"


FRAME_SUFFIX_RE = re.compile(r"_s\d+_b\d+$")


def parent_sound_of(local_name: str) -> str:
    if local_name.startswith("Sound_"):
        return local_name
    if local_name.startswith("Frame_"):
        stem = FRAME_SUFFIX_RE.sub("", local_name)
        return "Sound_" + stem[len("Frame_"):]
    raise ValueError(f"Not a Sound or Frame local name: {local_name}")


def context_of_sound(sound_local: str) -> str:
    rest = sound_local[len("Sound_"):]
    for ctx in sorted(ALL_CONTEXTS, key=len, reverse=True):
        if rest.startswith(ctx + "_"):
            return ctx
    raise ValueError(f"Could not determine AcousticContext for {sound_local}")


NT_LINE_RE = re.compile(
    r'^<' + re.escape(RES) + r'([^>]+)>\s+<' + re.escape(FKG) + r'([^>]+)>\s+<' + re.escape(RES) + r'([^>]+)>\s*\.\s*$'
)


def count_files_per_context(extraction_dir: Path) -> dict:
    counts = {}
    for ctx in ALL_CONTEXTS:
        d = extraction_dir / ctx
        counts[ctx] = len(list(d.glob("*.pkl"))) if d.exists() else 0
    return counts


def parse_object_triples(path: Path, wanted_predicates: set, per_context_cap: dict | None):
    triples = []
    per_context_files = defaultdict(set)
    kept_groups = set()
    skipped_groups = set()
    t0 = time.time()
    n_lines = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            m = NT_LINE_RE.match(line)
            if not m:
                continue
            s, p, o = m.groups()
            if p not in wanted_predicates:
                continue

            if s.startswith("Sound_") or s.startswith("Frame_"):
                group = parent_sound_of(s)
                if group in skipped_groups:
                    continue
                if group not in kept_groups:
                    if per_context_cap is not None:
                        ctx = context_of_sound(group)
                        if len(per_context_files[ctx]) >= per_context_cap.get(ctx, 0):
                            skipped_groups.add(group)
                            continue
                        per_context_files[ctx].add(group)
                    kept_groups.add(group)

            triples.append((s, p, o))

            if n_lines % 5_000_000 == 0:
                print(f"  ...{n_lines:,} lines scanned, {len(triples):,} triples kept "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr)

    print(f"Scanned {n_lines:,} lines in {time.time() - t0:.0f}s, "
          f"kept {len(triples):,} object-property triples, "
          f"{len(kept_groups):,} Sound recordings.", file=sys.stderr)
    return triples, kept_groups


def add_schema_and_derived(triples):
    """Adds the 13 hasSoundType schema edges plus inferredSoundType per
    Sound, mirroring kg_rules.py's R1 locally instead of via GraphDB."""
    out = list(triples)
    for ctx in ALL_CONTEXTS:
        out.append((f"AcousticContext_{ctx}", "hasSoundType", f"SoundType_{sound_type_for(ctx)}"))
    for s, p, o in triples:
        if p == "hasAcousticContext":
            ctx = o[len("AcousticContext_"):]
            out.append((s, "inferredSoundType", f"SoundType_{sound_type_for(ctx)}"))
    return out


def stratified_group_split(kept_groups, seed, train_frac):
    by_context = defaultdict(list)
    for g in kept_groups:
        by_context[context_of_sound(g)].append(g)

    rng = random.Random(seed)
    split_of = {}
    for ctx, groups in by_context.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, min(n - 1, int(round(n * train_frac)))) if n > 1 else 1
        for g in groups[:n_train]:
            split_of[g] = "train"
        for g in groups[n_train:]:
            split_of[g] = "test"
    return split_of


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--extraction-dir", type=Path, default=REPO_ROOT / "out" / "data" / "extraction",
                         help="Only used with --sample-fraction, to count each context's true file total.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-files-per-context", type=int, default=None,
        help="Flat cap on source recordings per AcousticContext (smoke test). "
             "Prefer --sample-fraction for a representative subsample.",
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=None,
        help="Proportional subsample: keep this fraction of EACH context's own "
             "recordings (at least 1), preserving class balance. E.g. 0.10 for ~10%%.",
    )
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--models", nargs="+", default=["transe", "complex"],
                         choices=["transe", "complex"])
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9103)
    args = parser.parse_args()

    if not args.triples.exists():
        sys.exit(f"Triples file not found: {args.triples} (run scripts/build_kg_triples.py first)")

    import torch
    from pykeen.models import model_resolver
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    if args.sample_fraction is not None:
        true_counts = count_files_per_context(args.extraction_dir)
        per_context_cap = {
            ctx: max(1, round(n * args.sample_fraction)) for ctx, n in true_counts.items()
        }
        print(f"--sample-fraction {args.sample_fraction} -> per-context recording caps: {per_context_cap}")
    elif args.max_files_per_context is not None:
        per_context_cap = {ctx: args.max_files_per_context for ctx in ALL_CONTEXTS}
    else:
        per_context_cap = None

    wanted = {"hasAcousticContext", "recordedBy", "hasFrame", "hasBand"}
    triples, kept_groups = parse_object_triples(args.triples, wanted, per_context_cap)
    triples = add_schema_and_derived(triples)

    split_of = stratified_group_split(kept_groups, args.seed, args.train_frac)

    def split_for_triple(s, p, o):
        if p in ("hasAcousticContext", "inferredSoundType") and s.startswith("Sound_"):
            return split_of[s]
        return "train"

    train, test = [], []
    for s, p, o in triples:
        (train if split_for_triple(s, p, o) == "train" else test).append((s, p, o))

    print(f"Split: train={len(train):,}  test={len(test):,}")

    full_array = np.array(triples, dtype=str)
    full_tf = TriplesFactory.from_labeled_triples(full_array, compact_id=True)
    e2id, r2id = full_tf.entity_to_id, full_tf.relation_to_id

    def factory(subset):
        arr = np.array(subset, dtype=str)
        return TriplesFactory.from_labeled_triples(
            arr, entity_to_id=e2id, relation_to_id=r2id, compact_id=False,
        )

    train_tf, test_tf = factory(train), factory(test)
    print(f"Entities: {full_tf.num_entities:,}  Relations: {full_tf.num_relations}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def closed_set_eval(model, mapped_triples, relation_label, candidate_prefix, label):
        """Ranks the true tail against only the real candidate set, not
        PyKEEN's default full-vocabulary corruption (meaningless here since
        the target is closed-world, fixed-cardinality)."""
        rel_id = r2id[relation_label]
        mask = mapped_triples[:, 1] == rel_id
        subset = mapped_triples[mask].cpu().numpy()
        if len(subset) == 0:
            return None

        candidate_ids = sorted(
            eid for label_, eid in e2id.items() if label_.startswith(candidate_prefix)
        )
        candidate_pos = {cid: i for i, cid in enumerate(candidate_ids)}

        hr = torch.tensor(subset[:, [0, 1]], dtype=torch.long, device=model.device)
        tails = torch.tensor(candidate_ids, dtype=torch.long, device=model.device)
        with torch.no_grad():
            scores = model.score_t(hr, tails=tails)

        true_pos = torch.tensor(
            [candidate_pos[t] for t in subset[:, 2]], device=model.device
        )
        true_scores = scores.gather(1, true_pos.unsqueeze(1))
        ranks = (scores > true_scores).sum(dim=1).cpu().numpy() + 1

        metrics = {
            "n_triples": int(len(subset)),
            "n_candidates": len(candidate_ids),
            "mrr": float(np.mean(1.0 / ranks)),
            "hits@1": float(np.mean(ranks <= 1)),
            "hits@3": float(np.mean(ranks <= 3)) if len(candidate_ids) >= 3 else None,
        }
        print(f"    {label}: {metrics}")
        return metrics

    all_results = {}
    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        t0 = time.time()

        model_cls = model_resolver.lookup(model_name)
        model = model_cls(
            triples_factory=train_tf, embedding_dim=args.embedding_dim, random_seed=args.seed,
        )
        if device == "cuda":
            model = model.to("cuda")

        # Low-level loop, not pykeen.pipeline(): that always runs its own
        # full-vocabulary evaluation, wasted cost since we use closed_set_eval.
        training_loop = SLCWATrainingLoop(model=model, triples_factory=train_tf, optimizer="adam")
        training_loop.train(
            triples_factory=train_tf, num_epochs=args.epochs, use_tqdm=True,
        )
        elapsed = time.time() - t0
        print(f"Trained {model_name} in {elapsed:.0f}s")

        print(f"  Per-target test metrics ({model_name}, closed-set ranking):")
        primary = closed_set_eval(model, test_tf.mapped_triples,
                                   "hasAcousticContext", "AcousticContext_",
                                   "primary (hasAcousticContext, 13-way)")
        secondary = closed_set_eval(model, test_tf.mapped_triples,
                                     "inferredSoundType", "SoundType_",
                                     "secondary (inferredSoundType, 2-way)")

        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model_state.pt")
        all_results[model_name] = {
            "train_seconds": elapsed,
            "primary_hasAcousticContext": primary,
            "secondary_inferredSoundType": secondary,
        }

    with open(args.output_dir / "summary.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nWrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
