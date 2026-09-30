#!/usr/bin/env python3
"""
Train a heterogeneous GNN (PyTorch Geometric) on FOREST-KG for node
classification, evaluating on BOTH targets (LO3, commit 6), one prediction
per RECORDING (same task as scripts/train_kge.py's LO1/commit 5):
  - primary:   Sound.hasAcousticContext  (13-way)
  - secondary: Sound.hasSoundType (via inferredSoundType) (2-way)

Unlike train_kge.py (structure only: recordedBy/hasFrame/hasBand, no
acoustic values), this script gives Frame nodes their real 17 acoustic
index values (standardized) as node features. Message passing propagates
that signal from a Sound's Frame neighbors (up to 4490 for a forest
recording) into the Sound's own representation, which a classification
head reads off. This is the natural LO12 comparison: KGE (structure-only)
vs. GNN (structure + real content features) on the identical task and
split methodology.

Graph (heterogeneous, PyG HeteroData):
    Sound  --hasFrame-->   Frame   (+ reverse edge, auto-added)
    Frame  --hasBand-->    Band    (+ reverse edge)
    Sound  --recordedBy--> Sensor  (+ reverse edge)
  Frame.x = the 17 acoustic indices + second, standardized (z-score).
  Sound/Sensor/Band.x = small constant seed vectors -- deliberately
  uninformative, so any classification skill has to come from message
  passing over Frame features, not from a memorized per-node embedding.

Split: same as train_kge.py -- stratified by AcousticContext, split by
RECORDING (a Sound's Frames always travel with it, never separately
train/test -- there's no frame-level split to leak across here since
Frame nodes carry no target label of their own).

Reads straight from out/data/kg/triples.nt (object AND literal triples),
not GraphDB -- same reasoning as train_kge.py: local parsing is much
faster than a SPARQL round trip for this volume.

Usage:
    python scripts/train_gnn.py --sample-fraction 0.10   # recommended first
    python scripts/train_gnn.py                            # full data (CPU: slow, see notes.md)
    python scripts/train_gnn.py --epochs 50 --hidden-dim 64

Environment: needs torch_geometric (installed this session) + torch.
"""
import argparse
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIPLES = REPO_ROOT / "out" / "data" / "kg" / "triples.nt"

FKG = "https://forest-kg.example.org/ontology#"
RES = "https://forest-kg.example.org/resource/"

FOREST_CONTEXTS = {"Plantation", "Pasture", "NaturalRegeneration", "RefForest"}
ALL_CONTEXTS = FOREST_CONTEXTS | {
    "Noise", "Engine", "MachineryImpact", "NonMachineryImpact", "PoweredSaw",
    "AlertSignal", "Music", "HumanVoice", "Dog",
}
CONTEXT_LIST = sorted(ALL_CONTEXTS)
CONTEXT_TO_IDX = {c: i for i, c in enumerate(CONTEXT_LIST)}

# 17 acoustic indices + second = 18 Frame features, in a fixed order.
FRAME_LITERAL_PROPS = [
    "meanEnergy", "medianEnergy", "sumEnergy", "maxEnergy", "minEnergy",
    "aci", "aca", "adi", "bioacousticIndex", "temporalMedian", "numberPeaks",
    "entropyFrequency", "entropyTemporal", "entropy", "acousticEvenness",
    "soundscapeIndex", "acousticRichness", "second",
]
PROP_TO_COL = {p: i for i, p in enumerate(FRAME_LITERAL_PROPS)}
SEED_DIM = 8  # dim of the deliberately-uninformative Sound/Sensor/Band seed features

# Fixed relation set (+ their auto-added reverse, via T.ToUndirected()) --
# hardcoded here (not read off a live HeteroData.edge_types) so this module
# can build a fresh, empty skeleton graph and a matching model without any
# training data on hand, e.g. for inference on one new recording.
BASE_EDGE_TYPES = [
    ("sound", "hasFrame", "frame"),
    ("frame", "hasBand", "band"),
    ("sound", "recordedBy", "sensor"),
]


def empty_hetero_skeleton():
    """A HeteroData with all 4 node types (0 nodes each) and all 3 base
    edge types (0 edges each), then made undirected -- gives a canonical
    edge_types list (base + reverse) without depending on real data."""
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    data = HeteroData()
    for node_type, dim in [("sound", SEED_DIM), ("frame", len(FRAME_LITERAL_PROPS)),
                            ("sensor", SEED_DIM), ("band", SEED_DIM)]:
        data[node_type].x = torch.zeros((0, dim), dtype=torch.float32)
    for src, rel, dst in BASE_EDGE_TYPES:
        data[src, rel, dst].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return T.ToUndirected()(data)


class GNN(torch.nn.Module):
    """2-layer heterogeneous GraphSAGE + two classification heads (primary
    AcousticContext 13-way, secondary SoundType 2-way) reading off Sound's
    final representation. See module docstring for the design rationale
    (Frame carries real features, Sound/Sensor/Band start uninformative)."""

    def __init__(self, hidden_dim, edge_types):
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv

        self.hidden_dim = hidden_dim
        self.lin_in = torch.nn.ModuleDict({
            "sound": torch.nn.Linear(SEED_DIM, hidden_dim),
            "frame": torch.nn.Linear(len(FRAME_LITERAL_PROPS), hidden_dim),
            "sensor": torch.nn.Linear(SEED_DIM, hidden_dim),
            "band": torch.nn.Linear(SEED_DIM, hidden_dim),
        })
        # Explicit (not lazy (-1,-1)) in/out dims: every node type is
        # projected to hidden_dim by lin_in first, so every conv layer's
        # input is uniformly hidden_dim -- and explicit shapes mean a saved
        # state_dict can be loaded straight away, no dummy forward pass
        # needed first to materialize lazy parameters.
        self.conv1 = HeteroConv({
            et: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for et in edge_types
        }, aggr="mean")
        self.conv2 = HeteroConv({
            et: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for et in edge_types
        }, aggr="mean")
        self.head_primary = torch.nn.Linear(hidden_dim, len(CONTEXT_LIST))
        self.head_secondary = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_in[k](v).relu() for k, v in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        sound_repr = x_dict["sound"]
        return self.head_primary(sound_repr), self.head_secondary(sound_repr)


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


NT_OBJ_RE = re.compile(
    r'^<' + re.escape(RES) + r'([^>]+)>\s+<' + re.escape(FKG) + r'([^>]+)>\s+<' + re.escape(RES) + r'([^>]+)>\s*\.\s*$'
)
NT_LIT_RE = re.compile(
    r'^<' + re.escape(RES) + r'([^>]+)>\s+<' + re.escape(FKG) + r'([^>]+)>\s+"([^"]*)"(?:\^\^<[^>]+>)?\s*\.\s*$'
)


def count_files_per_context(extraction_dir: Path) -> dict:
    counts = {}
    for ctx in ALL_CONTEXTS:
        d = extraction_dir / ctx
        counts[ctx] = len(list(d.glob("*.pkl"))) if d.exists() else 0
    return counts


class IdMap:
    def __init__(self):
        self._ids = {}

    def get(self, name: str) -> int:
        i = self._ids.get(name)
        if i is None:
            i = len(self._ids)
            self._ids[name] = i
        return i

    def __len__(self):
        return len(self._ids)


def build_graph(path: Path, per_context_cap: dict | None):
    """Single streaming pass over triples.nt. Keeps only Sound recordings
    within per_context_cap (and their Frames/Sensor). Returns a dict with
    edge index lists, Frame feature rows, Sound context labels, and id maps.
    """
    sound_ids, frame_ids, sensor_ids, band_ids = IdMap(), IdMap(), IdMap(), IdMap()
    kept_groups, skipped_groups = set(), set()
    per_context_files = defaultdict(set)

    hasframe_edges = []   # (sound_idx, frame_idx)
    hasband_edges = []    # (frame_idx, band_idx)
    recordedby_edges = [] # (sound_idx, sensor_idx)
    frame_feat_rows = {}  # frame_idx -> list[18] (filled in, may stay partially NaN if a prop is absent)
    sound_context = {}    # sound_idx -> context label (string)

    t0 = time.time()
    n_lines = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            s = p = o = None
            m = NT_OBJ_RE.match(line)
            is_literal = False
            if m:
                s, p, o = m.groups()
            else:
                m = NT_LIT_RE.match(line)
                if m:
                    s, p, o = m.groups()
                    is_literal = True
                else:
                    continue

            if p not in ("hasFrame", "hasBand", "recordedBy", "hasAcousticContext") and p not in PROP_TO_COL:
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

            if p == "hasFrame":
                si, fi = sound_ids.get(s), frame_ids.get(o)
                hasframe_edges.append((si, fi))
            elif p == "hasBand":
                fi, bi = frame_ids.get(s), band_ids.get(o)
                hasband_edges.append((fi, bi))
            elif p == "recordedBy":
                si, sei = sound_ids.get(s), sensor_ids.get(o)
                recordedby_edges.append((si, sei))
            elif p == "hasAcousticContext":
                si = sound_ids.get(s)
                sound_context[si] = o[len("AcousticContext_"):]
            elif p in PROP_TO_COL and is_literal:
                fi = frame_ids.get(s)
                row = frame_feat_rows.setdefault(fi, [float("nan")] * len(FRAME_LITERAL_PROPS))
                row[PROP_TO_COL[p]] = float(o)

            if n_lines % 5_000_000 == 0:
                print(f"  ...{n_lines:,} lines scanned ({time.time() - t0:.0f}s)", file=sys.stderr)

    print(f"Scanned {n_lines:,} lines in {time.time() - t0:.0f}s: "
          f"{len(sound_ids):,} Sound, {len(frame_ids):,} Frame, "
          f"{len(sensor_ids):,} Sensor, {len(band_ids):,} Band.", file=sys.stderr)

    return {
        "sound_ids": sound_ids, "frame_ids": frame_ids,
        "sensor_ids": sensor_ids, "band_ids": band_ids,
        "hasframe_edges": hasframe_edges, "hasband_edges": hasband_edges,
        "recordedby_edges": recordedby_edges,
        "frame_feat_rows": frame_feat_rows, "sound_context": sound_context,
        "kept_groups": kept_groups,
    }


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
    parser.add_argument("--extraction-dir", type=Path, default=REPO_ROOT / "out" / "data" / "extraction")
    parser.add_argument("--sample-fraction", type=float, default=None,
                         help="Stratified subsample: fraction of EACH context's recordings. E.g. 0.10.")
    parser.add_argument("--max-files-per-context", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=9103)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "data" / "kg" / "gnn")
    args = parser.parse_args()

    if not args.triples.exists():
        sys.exit(f"Triples file not found: {args.triples}")

    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    if args.sample_fraction is not None:
        true_counts = count_files_per_context(args.extraction_dir)
        per_context_cap = {ctx: max(1, round(n * args.sample_fraction)) for ctx, n in true_counts.items()}
        print(f"--sample-fraction {args.sample_fraction} -> per-context recording caps: {per_context_cap}")
    elif args.max_files_per_context is not None:
        per_context_cap = {ctx: args.max_files_per_context for ctx in ALL_CONTEXTS}
    else:
        per_context_cap = None

    g = build_graph(args.triples, per_context_cap)
    split_of = stratified_group_split(g["kept_groups"], args.seed, args.train_frac)

    n_sound, n_frame, n_sensor, n_band = (
        len(g["sound_ids"]), len(g["frame_ids"]), len(g["sensor_ids"]), len(g["band_ids"])
    )

    # ---- Frame features: 18-d (17 indices + second), z-score standardized,
    # any missing value (shouldn't happen, but be safe) filled with 0 after
    # standardization. ----
    frame_x = np.zeros((n_frame, len(FRAME_LITERAL_PROPS)), dtype=np.float32)
    for fi, row in g["frame_feat_rows"].items():
        frame_x[fi] = row
    col_mean = np.nanmean(frame_x, axis=0)
    col_std = np.nanstd(frame_x, axis=0)
    col_std[col_std == 0] = 1.0
    frame_x = (frame_x - col_mean) / col_std
    frame_x = np.nan_to_num(frame_x, nan=0.0)
    frame_x = torch.tensor(frame_x, dtype=torch.float32)

    # ---- Sound/Sensor/Band: deliberately uninformative seed features (see
    # module docstring) -- classification skill must come from Frame
    # message passing, not a memorized per-node embedding. ----
    g_torch = torch.Generator().manual_seed(args.seed)
    sound_x = torch.randn(n_sound, SEED_DIM, generator=g_torch) * 0.01
    sensor_x = torch.randn(n_sensor, SEED_DIM, generator=g_torch) * 0.01
    band_x = torch.randn(n_band, SEED_DIM, generator=g_torch) * 0.01

    data = HeteroData()
    data["sound"].x = sound_x
    data["frame"].x = frame_x
    data["sensor"].x = sensor_x
    data["band"].x = band_x

    def edge_tensor(pairs):
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        arr = np.array(pairs, dtype=np.int64).T
        return torch.tensor(arr, dtype=torch.long)

    data["sound", "hasFrame", "frame"].edge_index = edge_tensor(g["hasframe_edges"])
    data["frame", "hasBand", "band"].edge_index = edge_tensor(g["hasband_edges"])
    data["sound", "recordedBy", "sensor"].edge_index = edge_tensor(g["recordedby_edges"])
    data = T.ToUndirected()(data)

    # ---- Labels + train/test masks, Sound nodes only ----
    y_primary = torch.full((n_sound,), -1, dtype=torch.long)
    y_secondary = torch.full((n_sound,), -1, dtype=torch.long)
    train_mask = torch.zeros(n_sound, dtype=torch.bool)
    test_mask = torch.zeros(n_sound, dtype=torch.bool)

    sound_name_by_id = {v: k for k, v in g["sound_ids"]._ids.items()}
    for si, ctx in g["sound_context"].items():
        y_primary[si] = CONTEXT_TO_IDX[ctx]
        y_secondary[si] = 0 if sound_type_for(ctx) == "Urban" else 1
        split = split_of[sound_name_by_id[si]]
        (train_mask if split == "train" else test_mask)[si] = True

    print(f"Sound={n_sound:,} Frame={n_frame:,} Sensor={n_sensor:,} Band={n_band:,}  "
          f"train={int(train_mask.sum())}  test={int(test_mask.sum())}")

    model = GNN(args.hidden_dim, data.edge_types)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_primary, out_secondary = model(data.x_dict, data.edge_index_dict)
        loss_p = F.cross_entropy(out_primary[train_mask], y_primary[train_mask])
        loss_s = F.cross_entropy(out_secondary[train_mask], y_secondary[train_mask])
        loss = loss_p + loss_s
        loss.backward()
        optimizer.step()

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                out_primary, out_secondary = model(data.x_dict, data.edge_index_dict)
                pred_p = out_primary[test_mask].argmax(dim=1)
                pred_s = out_secondary[test_mask].argmax(dim=1)
                acc_p = (pred_p == y_primary[test_mask]).float().mean().item()
                acc_s = (pred_s == y_secondary[test_mask]).float().mean().item()
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  "
                  f"test_acc primary={acc_p:.4f}  secondary={acc_s:.4f}")

    elapsed = time.time() - t0
    print(f"\nTrained in {elapsed:.0f}s")

    model.eval()
    with torch.no_grad():
        out_primary, out_secondary = model(data.x_dict, data.edge_index_dict)
        pred_p = out_primary[test_mask].argmax(dim=1)
        pred_s = out_secondary[test_mask].argmax(dim=1)
        acc_p = (pred_p == y_primary[test_mask]).float().mean().item()
        acc_s = (pred_s == y_secondary[test_mask]).float().mean().item()

        # top-3 accuracy for the 13-way primary target, comparable to KGE's hits@3
        top3 = out_primary[test_mask].topk(3, dim=1).indices
        acc_p_top3 = (top3 == y_primary[test_mask].unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"\nFinal test metrics (n={int(test_mask.sum())}):")
    print(f"  primary (AcousticContext, 13-way):   acc={acc_p:.4f}  top3_acc={acc_p_top3:.4f}  (random~{1/len(CONTEXT_LIST):.3f}/{3/len(CONTEXT_LIST):.3f})")
    print(f"  secondary (SoundType, 2-way):         acc={acc_s:.4f}  (random~0.5-0.75 by class balance)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model_state.pt")

    # Everything scripts/gnn_inference.py needs to rebuild this exact model
    # and preprocess a brand-new recording's features the same way training
    # did -- without this, inference on new audio would silently use the
    # wrong normalization (different mean/std) and get nonsense scores.
    import json
    with open(args.output_dir / "model_meta.json", "w") as fh:
        json.dump({
            "hidden_dim": args.hidden_dim,
            "frame_literal_props": FRAME_LITERAL_PROPS,
            "context_list": CONTEXT_LIST,
            "col_mean": col_mean.tolist(),
            "col_std": col_std.tolist(),
        }, fh, indent=2)

    with open(args.output_dir / "summary.json", "w") as fh:
        json.dump({
            "train_seconds": elapsed,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "primary_acc": acc_p, "primary_top3_acc": acc_p_top3,
            "secondary_acc": acc_s,
        }, fh, indent=2)
    print(f"Wrote {args.output_dir / 'summary.json'} and model_meta.json")


if __name__ == "__main__":
    main()
