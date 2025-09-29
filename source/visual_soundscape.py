import os, time, json, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def visualisation_regions():
    t0 = time.time()
    cwd = os.getcwd()
    cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
    regions = ['Plantation', 'NaturalRegeneration', 'RefForest', 'Pasture']
    path_in = f'{cwd}/out/data/extraction/'
    path_out_visual = f'{cwd}/out/data/visual_regions/'
    os.makedirs(path_out_visual, exist_ok=True)

    cols = ['npp', 'bet', 'htp', 'hfq', 'aei']

    sid_json_path = f'{path_out_visual}/sid_index.json'
    sid_map = {}
    if os.path.exists(sid_json_path):
        with open(sid_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            try:
                sid = int(k)
            except:
                continue
            fn = v.get('filename', '')
            rg = v.get('region', '')
            sid_map[sid] = {'region': rg, 'filename': fn}
    else:
        print(f'[warn] sid_index.json not found at {sid_json_path}')

    def safe_stem(name):
        stem = Path(name).stem
        stem = re.sub(r'[^\w\-.]+', '_', stem).strip('_')
        return stem or f'sid'

    region_filepaths = {r: [] for r in regions}
    for dirpath, _, filenames in os.walk(path_in):
        rname = os.path.basename(dirpath)
        if rname in regions:
            for fn in filenames:
                if fn.endswith('.pkl'):
                    region_filepaths[rname].append(os.path.join(dirpath, fn))

    for region in regions:
        files = sorted(region_filepaths.get(region, []))
        if not files:
            continue
        frames = []
        for idx, fp in enumerate(files, start=1):
            try:
                df_i = pd.read_pickle(fp)
            except Exception as e:
                print(f'[skip] {fp}: {e}')
                continue
            if 'sid' not in df_i.columns:
                df_i = df_i.copy()
                df_i['sid'] = idx
            df_i['region'] = region
            frames.append(df_i)
        if not frames:
            continue

        df_r = pd.concat(frames, ignore_index=True)
        missing = [c for c in cols if c not in df_r.columns]
        if missing:
            print(f'[warn] {region}: missing columns {missing}, skipping region')
            continue

        df_r = df_r[['sid', 'region'] + cols].copy()
        g = df_r.groupby('sid', as_index=False)[cols].mean()

        df_norm = g.copy()
        for c in cols:
            mn, mx = g[c].min(), g[c].max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                df_norm[c] = 30.0
            else:
                df_norm[c] = 10 + (g[c] - mn) * 40.0 / (mx - mn)

        out_dir = Path(path_out_visual) / region
        out_dir.mkdir(parents=True, exist_ok=True)

        def plot_spider(row):
            cats = cols
            N = len(cats)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            vals = df_norm.loc[row, cats].values.tolist()
            vals += vals[:1]

            sid_val = int(g.loc[row, 'sid'])
            mp3_name = sid_map.get(sid_val, {}).get('filename', f'sid_{sid_val}.mp3')
            title_name = Path(mp3_name).name
            file_stem = safe_stem(title_name)

            plt.figure(figsize=(5,5), dpi=200)
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(pi/2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], cats, color='black', size=9)
            ax.set_rlabel_position(0)
            plt.yticks([10,20,30,40], ["10","20","30","40"], color="grey", size=8)
            plt.ylim(0, 51)
            ax.plot(angles, vals, linewidth=2)
            ax.fill(angles, vals, alpha=0.35)
            plt.title(f'{region} · {title_name}', size=11, y=1.08)
            fp = out_dir / f'{file_stem}.png'
            plt.savefig(fp, bbox_inches='tight')
            plt.close()

        for i in range(len(df_norm)):
            plot_spider(i)

    print('DONE in', round(time.time()-t0, 3), 's')

if __name__ == '__main__':
    visualisation_regions()
