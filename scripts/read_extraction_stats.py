#!/usr/bin/env python3
"""
Statistics reader for the acoustic feature extraction output.

Each region folder under `out/data/extraction/` (e.g. Engine, Dog, Pasture, ...)
contains one .pkl file per source audio file. Every .pkl holds a pandas
DataFrame with one row per (second, frequency-band) window and:
  - scalar acoustic-index columns (reg, sid, sec, ban, men, aci, ...)
  - a flattened feature vector per row (vec_0 ... vec_N)

This script scans those folders and reports, per region:
  - number of entries (i.e. number of source audio files / .pkl files)
  - "length" of each file: rows per DataFrame and derived duration in seconds
  - the columns/variables present (scalar acoustic indices + vector size)

Usage:
    python scripts/read_extraction_stats.py
    python scripts/read_extraction_stats.py --extraction-dir out/data/extraction-data
    python scripts/read_extraction_stats.py --sample-per-region 50
    python scripts/read_extraction_stats.py --full                 # read every file (slow, ~30GB)
    python scripts/read_extraction_stats.py --csv out/extraction_stats_summary.csv

Environment:
    The .pkl files use `Metrics` enum members (source/extraction.py) as column
    labels, so unpickling needs that module importable, which in turn needs
    joblib / pandas / numpy / scikit-maad. The repo's own venvs/ (tpy310f_h,
    tpy310f_i) are broken symlinks outside the cluster; use a working env
    instead, e.g.:
        /home/cvargas/miniconda3/envs/tpyforest/bin/python scripts/read_extraction_stats.py
"""
import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "out" / "data" / "extraction"
SOURCE_DIR = REPO_ROOT / "source"

if SOURCE_DIR.exists() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    from extraction import Metrics
    METRIC_DESCRIPTIONS = {m.value: m.name for m in Metrics}
except Exception as e:
    Metrics = None
    METRIC_DESCRIPTIONS = {}
    print(f"[WARN] Could not import source/extraction.py ({e}). "
          f"Column names will still work, but descriptions will be blank.\n"
          f"       Try running with an env that has joblib/pandas/numpy/scikit-maad, "
          f"e.g. /home/cvargas/miniconda3/envs/tpyforest/bin/python\n", file=sys.stderr)


def find_region_dirs(extraction_dir: Path):
    return sorted(p for p in extraction_dir.iterdir() if p.is_dir())


def load_pkl(path: Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def col_stats(values):
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    s = pd.Series(values)
    return {"min": s.min(), "median": s.median(), "mean": round(s.mean(), 1), "max": s.max()}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR,
        help=f"Directory with one subfolder per region (default: {DEFAULT_EXTRACTION_DIR})",
    )
    parser.add_argument(
        "--sample-per-region", type=int, default=20,
        help="Files to unpickle per region for row/duration stats (default: 20). Ignored with --full.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Unpickle every file instead of sampling (slow — reads the whole dataset).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional path to write the per-region summary table as CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=9103,
        help="Random seed for sampling (default: 9103, matches source/extraction.py).",
    )
    args = parser.parse_args()

    if not args.extraction_dir.exists():
        sys.exit(f"Extraction directory not found: {args.extraction_dir}")

    random.seed(args.seed)
    region_dirs = find_region_dirs(args.extraction_dir)
    if not region_dirs:
        sys.exit(f"No region subfolders found in {args.extraction_dir}")

    print(f"Reading extraction data from: {args.extraction_dir}")
    print(f"Found {len(region_dirs)} region folders")
    print(f"Mode: {'FULL scan' if args.full else f'sample of {args.sample_per_region} files/region'}\n")

    summary_records = []
    scalar_cols, n_vec_cols = None, None
    global_col_signatures = set()
    t0 = time.time()

    for region_dir in region_dirs:
        region = region_dir.name
        files = sorted(region_dir.glob("*.pkl"))
        n_entries = len(files)

        if n_entries == 0:
            summary_records.append({"region": region, "n_entries": 0})
            continue

        sample_files = files if args.full else random.sample(files, min(args.sample_per_region, n_entries))

        n_rows_list, n_seconds_list, n_bands_list, size_kb_list = [], [], [], []
        n_cols_list = []
        col_signatures = set()
        for fp in sample_files:
            size_kb_list.append(fp.stat().st_size / 1024)
            try:
                df = load_pkl(fp)
            except Exception as e:
                print(f"  [WARN] could not read {fp.name}: {e}", file=sys.stderr)
                continue

            n_rows_list.append(len(df))
            col_names = [str(c) for c in df.columns]
            n_cols_list.append(len(col_names))
            col_signatures.add(tuple(col_names))
            global_col_signatures.add(tuple(col_names))

            if scalar_cols is None:
                scalar_cols = [c for c in col_names if not c.startswith("vec_")]
                n_vec_cols = sum(1 for c in col_names if c.startswith("vec_"))

            if "sec" in col_names:
                sec_col = df.columns[col_names.index("sec")]
                n_seconds_list.append(int(df[sec_col].max()) + 1)
            if "ban" in col_names:
                ban_col = df.columns[col_names.index("ban")]
                n_bands_list.append(int(df[ban_col].nunique()))

        rows_s = col_stats(n_rows_list)
        dur_s = col_stats(n_seconds_list)
        cols_s = col_stats(n_cols_list)
        same_shape = (rows_s["min"] == rows_s["max"]) and (cols_s["min"] == cols_s["max"])
        same_columns = len(col_signatures) <= 1

        summary_records.append({
            "region": region,
            "n_entries": n_entries,
            "n_sampled": len(sample_files),
            "rows_per_file_min": rows_s["min"],
            "rows_per_file_median": rows_s["median"],
            "rows_per_file_max": rows_s["max"],
            "n_cols_min": cols_s["min"],
            "n_cols_median": cols_s["median"],
            "n_cols_max": cols_s["max"],
            "same_shape_in_region": same_shape,
            "same_columns_in_region": same_columns,
            "duration_sec_min": dur_s["min"],
            "duration_sec_median": dur_s["median"],
            "duration_sec_max": dur_s["max"],
            "avg_file_size_kb": round(sum(size_kb_list) / len(size_kb_list), 1) if size_kb_list else None,
        })

        print(f"  {region:20s} entries={n_entries:6d}  sampled={len(sample_files):4d}  "
              f"rows/file(median)={rows_s['median']}  duration_sec(median)={dur_s['median']}  "
              f"same_shape={same_shape}  same_columns={same_columns}")

    elapsed = time.time() - t0
    print(f"\nScan finished in {elapsed:.1f}s\n")

    print("=" * 78)
    print("COLUMNS / VARIABLES")
    print("=" * 78)
    if scalar_cols:
        print(f"{len(scalar_cols)} scalar columns (acoustic indices, source/extraction.py Metrics enum):")
        for c in scalar_cols:
            print(f"  - {c:6s} {METRIC_DESCRIPTIONS.get(c, '')}")
        band_width = n_vec_cols // 10 if n_vec_cols else "?"
        print(f"\n+ {n_vec_cols} 'vec_0' .. 'vec_{n_vec_cols - 1}' columns: flattened per-row feature vector "
              f"({band_width} points x 10 frequency bands, see DICT_BANDAS in source/extraction.py)")
    else:
        print("No files could be read to determine columns.")

    print("\n" + "=" * 78)
    print("PER-REGION SUMMARY")
    print("=" * 78)
    summary_df = pd.DataFrame(summary_records)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df.to_string(index=False))

    print(f"\nTotal entries (files) across all regions: {summary_df['n_entries'].sum()}")
    print(f"Same columns (name+order) across ALL sampled files, all regions: {len(global_col_signatures) <= 1}")
    if len(global_col_signatures) > 1:
        print(f"  -> {len(global_col_signatures)} distinct column layouts found")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.csv, index=False)
        print(f"Saved summary CSV to: {args.csv}")


if __name__ == "__main__":
    main()
