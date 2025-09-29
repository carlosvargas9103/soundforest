import os, time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def visualisation_regions(files_path: list[tuple[str,str]] = [],
                          dir_json_results_in: str = '',
                          ncols: int = 6016, *,
                          si: int = 1964,
                          sr: int = 48000,
                          b_band: int = 0,
                          u_band: int = 10000,
                          bandas: int = 10,
                          bandwidth: int = 1000,
                          path_data: str = '',
                          path_out: str = '',
                          samples_s: int = 1800,
                          isamples_s: int = 3,
                          secs_b: int = 6,
                          secs_o: int = 1.9,
                          hanning: bool = True,
                          w_size_mins: float = 0.06,
                          n_jobs: int = 1,
                          job_id: str = 'NULL',
                          verbose: bool = False,
                          f_pattern_out: str = 'visual_regions',
                          windows_13: bool = True,
                          horas: int = 30,
                          metric_names: list[str] = None,
                          dev_mode: bool = True,
                          n_epochs: int = 11,
                          df_stats: bool = False) -> None:
    t0 = time.time()
    cwd = os.getcwd()
    cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
    regions = ['Plantation', 'NaturalRegeneration', 'RefForest', 'Pasture']
    path_in = f'{cwd}/out/data/extraction/'
    path_out_visual = f'{cwd}/out/data/visual_regions/'
    os.makedirs(path_out_visual, exist_ok=True)

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

        exclude = {'region', 'sid'}
        metrics = [c for c in df_r.columns
                   if c not in exclude
                   and not str(c).startswith('vec_')
                   and pd.api.types.is_numeric_dtype(df_r[c])]

        if not metrics:
            print(f'[warn] No numeric metrics in {region}')
            continue

        g = df_r.groupby('sid', as_index=False)[metrics].mean()

        # normalize per-region to 10..50 (like your earlier scaling)
        df_norm = g.copy()
        for col in metrics:
            minv = g[col].min()
            maxv = g[col].max()
            if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
                df_norm[col] = 30.0
            else:
                df_norm[col] = 10 + (g[col] - minv) * (50 - 10) / (maxv - minv)

        out_dir = Path(path_out_visual) / region
        out_dir.mkdir(parents=True, exist_ok=True)

        def plot_spider(row):
            cats = metrics
            N = len(cats)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            vals = df_norm.loc[row, cats].values.tolist()
            vals += vals[:1]

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
            sid_val = int(g.loc[row, 'sid'])
            plt.title(f'{region} · sid {sid_val}', size=11, y=1.08)
            fp = out_dir / f'spider_sid_{sid_val}.png'
            plt.savefig(fp, bbox_inches='tight')
            plt.close()

        for i in range(len(df_norm)):
            plot_spider(i)

    print('DONE in', round(time.time()-t0, 3), 's')

if __name__ == '__main__':
    visualisation_regions()
