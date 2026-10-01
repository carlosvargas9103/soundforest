#!/usr/bin/env python3
"""
Map out/data/extraction/*.pkl into FOREST-KG triples (N-Triples).
One Sound per recording, owning its per-(second, band) Frames via
fkg:hasFrame. Parallel by file across worker processes.

Schema: scripts/kg_ontology.ttl

Usage:
    python scripts/build_kg_triples.py --workers 19
    python scripts/build_kg_triples.py --limit-per-region 50   # smoke test
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "out" / "data" / "extraction"
DEFAULT_OUTPUT = REPO_ROOT / "out" / "data" / "kg" / "triples.nt"
SOURCE_DIR = REPO_ROOT / "source"

if SOURCE_DIR.exists() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    from extraction import Metrics
except Exception as e:
    sys.exit(
        f"Could not import source/extraction.py ({e}).\n"
        f"Run with an env that has joblib/pandas/numpy/scikit-maad, e.g.\n"
        f"  /home/cvargas/miniconda3/envs/tpyforest/bin/python {Path(__file__).name} ..."
    )

FKG = "https://forest-kg.example.org/ontology#"
RES = "https://forest-kg.example.org/resource/"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"

FOREST_CONTEXTS = {"Plantation", "Pasture", "NaturalRegeneration", "RefForest"}

INDEX_PROPERTY = {
    Metrics.MEAN.value: "meanEnergy",
    Metrics.MEDIAN.value: "medianEnergy",
    Metrics.SUM.value: "sumEnergy",
    Metrics.MAX.value: "maxEnergy",
    Metrics.MIN.value: "minEnergy",
    Metrics.ACOUSTIC_COMPLEXITY.value: "aci",
    Metrics.ACOUSTIC_COMPLEXITY_ALTERNATIVE.value: "aca",
    Metrics.ACOUSTIC_DIVERSITY.value: "adi",
    Metrics.BIOACOUSTIC_INDEX_BETA.value: "bioacousticIndex",
    Metrics.TEMPORAL_MEDIAN.value: "temporalMedian",
    Metrics.NUMBER_PEAKS.value: "numberPeaks",
    Metrics.ENTROPY_FREQUENCY.value: "entropyFrequency",
    Metrics.ENTROPY_TEMPORAL.value: "entropyTemporal",
    Metrics.ENTROPY.value: "entropy",
    Metrics.ACOUSTIC_EVENNESS.value: "acousticEvenness",
    Metrics.SOUNDSCAPE_INDEX.value: "soundscapeIndex",
    Metrics.ACOUSTIC_RICHNESS.value: "acousticRichness",
}

# No per-sensor GPS available; every sensor in a group shares one centroid.
LOCATION_BY_GROUP = {
    "nicoya": {"latitude": 10.000, "longitude": -85.417, "municipality": "Nicoya Peninsula"},
    "nyc": {"latitude": 40.7128, "longitude": -74.0060, "municipality": "New York City"},
}


def group_for_region(region: str) -> str:
    return "nicoya" if region in FOREST_CONTEXTS else "nyc"


def source_stem(pkl_path: Path) -> str:
    name = pkl_path.stem
    marker = "_dict_y_split_"
    return name.split(marker)[0] if marker in name else name


def sensor_id_for(region: str, stem: str) -> str:
    if region in FOREST_CONTEXTS:
        parts = stem.split("_")
        return "_".join(parts[2:]) if len(parts) > 2 else stem
    return stem.split("_")[0]


def escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def uri(local: str) -> str:
    return f"<{RES}{local}>"


def lit_str(s: str) -> str:
    return f'"{escape_literal(s)}"'


def lit_float(x: float) -> str:
    return f'"{x:.6f}"^^<http://www.w3.org/2001/XMLSchema#float>'


def lit_int(n: int) -> str:
    return f'"{n}"^^<http://www.w3.org/2001/XMLSchema#integer>'


def sanitize_id(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in s)


def load_pkl(path: Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def process_files(worker_id, file_list, extraction_dir: Path, out_path: Path, stats_queue):
    seen_sensors = set()
    n_sounds = 0
    n_frames = 0
    n_errors = 0
    per_region = {}

    with open(out_path, "w", encoding="utf-8") as out:
        for region, fp in file_list:
            group = group_for_region(region)
            region_uri = uri("AcousticContext_" + region)

            try:
                df = load_pkl(fp)
            except Exception as e:
                print(f"  [WARN][w{worker_id}] could not read {fp}: {e}", file=sys.stderr)
                n_errors += 1
                continue

            col_names = [str(c) for c in df.columns]
            required = ["sec", "ban", "sid"]
            if not all(c in col_names for c in required):
                print(f"  [WARN][w{worker_id}] {fp} missing {required}, skipping", file=sys.stderr)
                n_errors += 1
                continue

            stem = source_stem(fp)
            sensor_id = sensor_id_for(region, stem)
            sensor_local = f"Sensor_{group}_{sanitize_id(sensor_id)}"
            sensor = uri(sensor_local)
            sound_local = f"Sound_{sanitize_id(region)}_{sanitize_id(stem)}"
            sound = uri(sound_local)
            frame_prefix = f"Frame_{sanitize_id(region)}_{sanitize_id(stem)}"
            source_file_lit = lit_str(fp.name)
            pkl_path_lit = lit_str(str(fp.relative_to(extraction_dir)))

            if sensor_local not in seen_sensors:
                seen_sensors.add(sensor_local)
                loc = LOCATION_BY_GROUP[group]
                out.write(f"{sensor} {RDF_TYPE} <{FKG}Sensor> .\n")
                out.write(f"{sensor} <{FKG}sensorId> {lit_str(sensor_id)} .\n")
                out.write(f"{sensor} <{FKG}latitude> {lit_float(loc['latitude'])} .\n")
                out.write(f"{sensor} <{FKG}longitude> {lit_float(loc['longitude'])} .\n")
                out.write(f"{sensor} <{FKG}municipality> {lit_str(loc['municipality'])} .\n")

            idx_cols = [
                (prop, col_names.index(metric_value))
                for metric_value, prop in INDEX_PROPERTY.items()
                if metric_value in col_names
            ]
            sec_i, ban_i, sid_i = (col_names.index(c) for c in required)

            n_rows = len(df)
            max_sec = -1
            sid_value = None

            out.write(f"{sound} {RDF_TYPE} <{FKG}Sound> .\n")
            out.write(f"{sound} <{FKG}hasAcousticContext> {region_uri} .\n")
            out.write(f"{sound} <{FKG}recordedBy> {sensor} .\n")
            out.write(f"{sound} <{FKG}sourceFile> {source_file_lit} .\n")
            out.write(f"{sound} <{FKG}pklPath> {pkl_path_lit} .\n")
            out.write(f"{sound} <{FKG}nFrames> {lit_int(n_rows)} .\n")

            for row in df.itertuples(index=False, name=None):
                sec, ban, sid = int(row[sec_i]), int(row[ban_i]), int(row[sid_i])
                max_sec = max(max_sec, sec)
                sid_value = sid
                frame = uri(f"{frame_prefix}_s{sec}_b{ban}")

                out.write(f"{sound} <{FKG}hasFrame> {frame} .\n")
                out.write(f"{frame} {RDF_TYPE} <{FKG}Frame> .\n")
                out.write(f"{frame} <{FKG}hasBand> {uri('Band_' + str(ban))} .\n")
                out.write(f"{frame} <{FKG}second> {lit_int(sec)} .\n")
                for prop, col_i in idx_cols:
                    out.write(f"{frame} <{FKG}{prop}> {lit_float(float(row[col_i]))} .\n")

                n_frames += 1

            out.write(f"{sound} <{FKG}durationSeconds> {lit_int(max_sec + 1)} .\n")
            if sid_value is not None:
                out.write(f"{sound} <{FKG}soundscapeId> {lit_int(sid_value)} .\n")

            n_sounds += 1
            per_region[region] = per_region.get(region, 0) + 1

    stats_queue.put({
        "worker_id": worker_id, "n_sounds": n_sounds, "n_frames": n_frames,
        "n_errors": n_errors, "per_region": per_region, "n_sensors": len(seen_sensors),
    })


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit-per-region", type=int, default=None,
        help="Only process the first N files per region (smoke test). Default: all files.",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default: cpu_count - 1).",
    )
    args = parser.parse_args()

    if not args.extraction_dir.exists():
        sys.exit(f"Extraction directory not found: {args.extraction_dir}")

    region_dirs = sorted(p for p in args.extraction_dir.iterdir() if p.is_dir())
    if not region_dirs:
        sys.exit(f"No region subfolders found in {args.extraction_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_files = []
    for region_dir in region_dirs:
        region = region_dir.name
        files = sorted(region_dir.glob("*.pkl"))
        if args.limit_per_region is not None:
            files = files[: args.limit_per_region]
        all_files.extend((region, fp) for fp in files)
    all_files.sort(key=lambda rf: rf[1].stat().st_size, reverse=True)

    n_workers = max(1, min(args.workers, len(all_files)))
    chunks = [all_files[i::n_workers] for i in range(n_workers)]
    print(f"{len(all_files)} files across {len(region_dirs)} regions, "
          f"{n_workers} workers ({sum(len(c) for c in chunks)} files assigned)")

    part_paths = [args.output.with_suffix(f".part{i}.nt") for i in range(n_workers)]
    stats_queue = mp.Queue()
    procs = []
    t0 = time.time()
    for i, (chunk, part_path) in enumerate(zip(chunks, part_paths)):
        p = mp.Process(target=process_files, args=(i, chunk, args.extraction_dir, part_path, stats_queue))
        p.start()
        procs.append(p)

    all_stats = [stats_queue.get() for _ in procs]
    for p in procs:
        p.join()
    elapsed = time.time() - t0

    n_sounds = sum(s["n_sounds"] for s in all_stats)
    n_frames = sum(s["n_frames"] for s in all_stats)
    n_errors = sum(s["n_errors"] for s in all_stats)
    per_region = {}
    for s in all_stats:
        for region, n in s["per_region"].items():
            per_region[region] = per_region.get(region, 0) + n
    for region in sorted(per_region):
        print(f"  {region:20s} -> {per_region[region]} Sound recordings")

    print(f"\nConcatenating {n_workers} part files...")
    with open(args.output, "wb") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as fh:
                out.write(fh.read())
            part_path.unlink()

    print(f"\n{n_sounds} Sound recordings, {n_frames} Frame nodes, {n_errors} files skipped on error.")
    print(f"Wrote {args.output} in {elapsed:.1f}s ({n_workers} workers)")
    print(f"Load alongside scripts/kg_ontology.ttl (schema) into the graph DB.")


if __name__ == "__main__":
    main()
