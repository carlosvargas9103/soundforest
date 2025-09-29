# make_sid_index.py
import os, json
import pandas as pd
from pathlib import Path

COLS = ['npp', 'bet', 'htp', 'hfq', 'aei']

def build_sid_json(mapping_csv: str, extraction_root: str, out_json: str):
    m = pd.read_csv(mapping_csv, sep=";", index_col=0)
    m.index.name = "sid"
    m = m.reset_index()
    m["sid"] = m["sid"].astype(int)
    m["mp3_name"] = m["filename"].apply(lambda p: os.path.basename(str(p)))

    regions = sorted(m["region"].unique())
    dfs = []
    for region in regions:
        region_dir = Path(extraction_root) / region
        if not region_dir.exists():
            continue
        for pkl in region_dir.rglob("*.pkl"):
            try:
                df = pd.read_pickle(pkl)
            except Exception:
                continue
            if "sid" not in df.columns:
                continue
            keep = ["sid"] + [c for c in COLS if c in df.columns]
            if len(keep) == 1:
                continue
            df = df[keep].copy()
            df["region"] = region
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid pickle data with 'sid' and EAI columns found.")

    df_all = pd.concat(dfs, ignore_index=True)
    have = [c for c in COLS if c in df_all.columns]
    g = df_all.groupby(["region", "sid"], as_index=False)[have].mean(numeric_only=True)

    j = pd.merge(g, m[["sid", "region", "mp3_name"]], on=["sid", "region"], how="left")

    result = {}
    for _, row in j.iterrows():
        sid = int(row["sid"])
        eai = {k: float(row[k]) for k in COLS if k in row and pd.notna(row[k])}
        result[sid] = {"region": row["region"], "filename": row["mp3_name"], "eai": eai}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json} with {len(result)} entries.")
    return result

if __name__ == "__main__":
    cwd = os.getcwd()
    if cwd.endswith("/source"):
        cwd = str(Path(cwd).parents[0])

    mapping_csv = f"{cwd}/out/audio_extraction_2025-01-11.csv"
    extraction_root = f"{cwd}/out/data/extraction"
    out_json = f"{cwd}/out/data/visual_regions/sid_index.json"

    build_sid_json(mapping_csv, extraction_root, out_json)
