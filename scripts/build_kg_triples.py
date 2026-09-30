#!/usr/bin/env python3
"""
Map out/data/extraction/*.pkl into FOREST-KG triples (N-Triples).

For every source audio file's feature DataFrame (one row per second x band,
see scripts/read_extraction_stats.py for the column layout) this emits one
fkg:Sound individual with:
  - acoustic index properties = the mean of each scalar Metrics column
    across the file's rows (aci, adi, entropy, ... -- see kg_ontology.ttl)
  - fkg:hasRegion  -> a fkg:Region individual: one of all 13
    source/extraction.py:SoundscapeRegion categories (the 9 SONYC-UST
    event classes and the 4 FOREST land-use classes alike)
  - fkg:recordedBy -> a fkg:Sensor individual, shared across all files from
    the same sensor_id, with a fixed lat/lon/municipality per group (see
    LOCATION_BY_GROUP below -- we don't have per-sensor GPS for most sounds)

Schema: scripts/kg_ontology.ttl
Pipeline diagram: scripts/kg_pipeline_plan.html

Usage:
    python scripts/build_kg_triples.py
    python scripts/build_kg_triples.py --extraction-dir out/data/extraction-data
    python scripts/build_kg_triples.py --limit-per-region 50   # smoke test
    python scripts/build_kg_triples.py --output out/data/kg/triples.nt

Environment: same as read_extraction_stats.py -- unpickling needs
source/extraction.py importable (joblib/pandas/numpy/scikit-maad), e.g.:
    /home/cvargas/miniconda3/envs/tpyforest/bin/python scripts/build_kg_triples.py
"""
import argparse
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

# Every Sound gets fkg:hasRegion -> Region_<folder name>, all 13 alike (see
# scripts/kg_ontology.ttl). This set is only used to pick the Sensor's fixed
# location group (Nicoya Peninsula vs NYC) and to parse the sensor/site id.
FOREST_REGIONS = {"Plantation", "Pasture", "NaturalRegeneration", "RefForest"}

# Sound (Metrics value -> ontology property). Excludes reg/sid/sec/ban
# (structural, not acoustic) and vec_* (kept out of the KG, see pklPath).
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

# Fixed centroid per recording group -- we don't have per-sensor GPS for
# most sounds, so every sensor in a group shares its group's coordinates.
LOCATION_BY_GROUP = {
    "nicoya": {"latitude": 10.000, "longitude": -85.417, "municipality": "Nicoya Peninsula"},
    "nyc": {"latitude": 40.7128, "longitude": -74.0060, "municipality": "New York City"},
}


def group_for_region(region: str) -> str:
    return "nicoya" if region in FOREST_REGIONS else "nyc"


def source_stem(pkl_path: Path) -> str:
    """'3_01_Pasture11_dict_y_split_11945_NULL_1769012640.pkl' -> '3_01_Pasture11'"""
    name = pkl_path.stem
    marker = "_dict_y_split_"
    return name.split(marker)[0] if marker in name else name


def sensor_id_for(region: str, stem: str) -> str:
    if region in FOREST_REGIONS:
        # '3_01_Pasture11' -> 'Pasture11' (site code is everything after the
        # 2nd underscore; the leading two fields are a local region/index
        # counter, not identifying information).
        parts = stem.split("_")
        return "_".join(parts[2:]) if len(parts) > 2 else stem
    # SONYC: '40_010020' -> '40' (matches annotations.csv sensor_id).
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
    args = parser.parse_args()

    if not args.extraction_dir.exists():
        sys.exit(f"Extraction directory not found: {args.extraction_dir}")

    region_dirs = sorted(p for p in args.extraction_dir.iterdir() if p.is_dir())
    if not region_dirs:
        sys.exit(f"No region subfolders found in {args.extraction_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen_sensors = set()
    n_sounds = 0
    n_errors = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        # ---- Region individuals are declared in kg_ontology.ttl; only
        #      fkg:Sound and fkg:Sensor individuals are emitted here. ----
        for region_dir in region_dirs:
            region = region_dir.name
            group = group_for_region(region)
            files = sorted(region_dir.glob("*.pkl"))
            if args.limit_per_region is not None:
                files = files[: args.limit_per_region]

            for fp in files:
                try:
                    df = load_pkl(fp)
                except Exception as e:
                    print(f"  [WARN] could not read {fp}: {e}", file=sys.stderr)
                    n_errors += 1
                    continue

                col_names = [str(c) for c in df.columns]
                stem = source_stem(fp)
                sensor_id = sensor_id_for(region, stem)
                sound_local = f"Sound_{sanitize_id(region)}_{sanitize_id(stem)}"
                sensor_local = f"Sensor_{group}_{sanitize_id(sensor_id)}"
                sound = uri(sound_local)
                sensor = uri(sensor_local)

                out.write(f"{sound} {RDF_TYPE} <{FKG}Sound> .\n")
                out.write(f"{sound} <{FKG}hasRegion> {uri('Region_' + region)} .\n")
                out.write(f"{sound} <{FKG}recordedBy> {sensor} .\n")
                out.write(f"{sound} <{FKG}sourceFile> {lit_str(fp.name)} .\n")
                out.write(f"{sound} <{FKG}pklPath> {lit_str(str(fp.relative_to(args.extraction_dir)))} .\n")
                out.write(f"{sound} <{FKG}nRows> {lit_int(len(df))} .\n")

                if "sec" in col_names:
                    sec_col = df.columns[col_names.index("sec")]
                    out.write(f"{sound} <{FKG}durationSeconds> {lit_int(int(df[sec_col].max()) + 1)} .\n")
                if "sid" in col_names:
                    sid_col = df.columns[col_names.index("sid")]
                    out.write(f"{sound} <{FKG}soundscapeId> {lit_int(int(df[sid_col].iloc[0]))} .\n")

                for metric_value, prop in INDEX_PROPERTY.items():
                    if metric_value in col_names:
                        col = df.columns[col_names.index(metric_value)]
                        mean_val = float(df[col].mean())
                        out.write(f"{sound} <{FKG}{prop}> {lit_float(mean_val)} .\n")

                if sensor_local not in seen_sensors:
                    seen_sensors.add(sensor_local)
                    loc = LOCATION_BY_GROUP[group]
                    out.write(f"{sensor} {RDF_TYPE} <{FKG}Sensor> .\n")
                    out.write(f"{sensor} <{FKG}sensorId> {lit_str(sensor_id)} .\n")
                    out.write(f"{sensor} <{FKG}latitude> {lit_float(loc['latitude'])} .\n")
                    out.write(f"{sensor} <{FKG}longitude> {lit_float(loc['longitude'])} .\n")
                    out.write(f"{sensor} <{FKG}municipality> {lit_str(loc['municipality'])} .\n")

                n_sounds += 1

            print(f"  {region:20s} -> {len(files)} Sound nodes ({group})")

    elapsed = time.time() - t0
    print(f"\n{n_sounds} Sound nodes, {len(seen_sensors)} distinct Sensor nodes, "
          f"{n_errors} files skipped on error.")
    print(f"Wrote {args.output} in {elapsed:.1f}s")
    print(f"Load alongside scripts/kg_ontology.ttl (schema) into the graph DB.")


if __name__ == "__main__":
    main()
