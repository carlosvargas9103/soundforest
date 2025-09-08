import gc

gc.collect()

import os
import io
import time
import math
import json
import random
import joblib
import datetime

from glob import glob
from pathlib import Path
from typing import List, Tuple

from joblib import Parallel, delayed
from joblib import effective_n_jobs
from itertools import cycle, combinations
from multiprocessing import Pool

from math import pi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from extraction import Metrics as M
from modelling import *

random.seed("9103")

t00 = time.time()

### IDENTIFY PATH ###
cwd = os.getcwd()
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)


def visualisation_regions(files_path: List[Tuple[str, str]] = [],
                          dir_json_results_in: str = '',
                          ncols: int = 6016, *,
                          # region: str = '',
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
                          metric_names: List[str] = M.list(),
                          dev_mode: bool = True,
                          n_epochs: int = 11,
                          df_stats: bool = False
                          ) -> None:
    print('#### #### HOI VISUAL_REGIONS #### ####')

    t00 = time.time()
    regions = ['Plantation', 'NaturalRegeneration', 'RefForest', 'Pasture']
    cols = ['npp', 'bet', 'htp', 'hfq', 'aei']
    folders_in, folder_out, f_ext_in, prefix_vec = 'extraction', 'visual_regions', '.pkl', 'vec_'
    path_in_observation = f'{cwd}/out/data/{folders_in}/'
    path_out_visual_region = f'{cwd}/out/data/{folder_out}/'

    global df_data
    df_data = None
    npy_arrays = []
    print('####', 'loading vectors', '####')
    region_filepaths = {region: [] for region in regions}
    for dirpath, dirnames, filenames in os.walk(path_in_observation):
        region_name = os.path.basename(dirpath)
        if region_name in regions:
            for filename in filenames:
                if filename.endswith(f_ext_in):
                    filepath = os.path.join(dirpath, filename)
                    region_filepaths[region_name].append(filepath)
    # just describe
    frames = []
    expected_cols = {'aca', 'adi', 'dsi'}
    for region, files in region_filepaths.items():
        for fp in files:
            try:
                df_i = pd.read_pickle(fp)
                missing = expected_cols - set(df_i.columns)
                if missing:
                    print(f"Warning: {os.path.basename(fp)} missing columns {sorted(missing)} — skipping")
                    continue
                df_i = df_i[['aca', 'adi', 'dsi']].copy()
                df_i['region'] = region
                frames.append(df_i)
            except Exception as e:
                print(f"Error reading {fp}: {e}")

    if not frames:
        raise RuntimeError("No valid data loaded. Check input paths and column names: 'aca', 'adi', 'dsi'.")

    df_data_d = pd.concat(frames, ignore_index=True)

    # Print overall describe (only numeric cols)
    print("#### Overall describe (aca, adi, dsi) ####")
    print(df_data_d[['aca', 'adi', 'dsi']].describe())

    # Print per-region describe
    print("\n#### Per-region describe (aca, adi, dsi) ####")
    for region, df_r in df_data_d.groupby('region'):
        print(f"\n-- {region} --")
        print(df_r[['aca', 'adi', 'dsi']].describe())

    exit()

    # for region, files in region_filepaths.items():
    #     print(f'Region: {region}, Files: {files}')
    # exit()
    # print('####', 'loading vectors', '####')

    # Set data
    # df = pd.DataFrame({
    #     'group': ['A', 'B', 'C', 'D'],
    #     'var1': [38, 1.5, 30, 4],
    #     'var2': [29, 10, 9, 34],
    #     'var3': [8, 39, 23, 24],
    #     'var4': [7, 31, 33, 14],
    #     'var5': [28, 15, 32, 14]
    # })
    #
    # print(type(df))
    # print(df.head())
    # exit()

    data_spider = []
    for reg in regions:
        try:
            print('####', 'loading', reg, 'loading', '####')
            # files_path = region_filepaths.get(reg, [])
            # print(files_path[0])
            df_data = pd.concat((pd.read_pickle(f) for f in region_filepaths.get(reg, [])), ignore_index=True)
            df_data.columns = df_data.columns.map(str)
        except Exception as e:
            print(f"Error loading {files_path}: {e}")
            exit()
        df_data = df_data.drop(columns=[col for col in df_data.columns if col.startswith(prefix_vec)])
        print(df_data.shape)
        # print(type(df_data))
        # print(df_data.head())
        # print(df_data.describe())
        # Select columns and aggregate (mean)
        values = df_data[cols].mean().values.tolist()
        data_spider.append(values)

    df_spider = pd.DataFrame(data_spider, columns=cols)
    df_spider['group'] = regions
    df_spider = df_spider[['group'] + cols]
    print(df_spider)
    df_norm = df_spider.copy()
    for col in cols:
        min_val = df_spider[col].min()
        max_val = df_spider[col].max()
        df_norm[col] = 10 + (df_spider[col] - min_val) * (50 - 10) / (max_val - min_val)
    print(df_norm)

    # exit()

    def make_spider(df, row, title, color):
        categories = cols
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(2, 2, row + 1, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, color='black', size=13)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40], ["10", "20", "30", "40"], color="grey", size=9)
        plt.ylim(0, 51)
        values = df_norm.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
        plt.title(title, size=13, color=color, y=1.09103)#, weight='bold')
    # initialize the figure
    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    # my_palette = plt.cm.get_cmap("Set2", len(df_spider.index)) # NO
    # my_palette = plt.cm.get_cmap("Set1", len(df_spider.index)) # SI
    # my_palette = plt.cm.get_cmap("tab20", len(df_spider.index))# NO
    my_palette = plt.cm.get_cmap("tab10", len(df_spider.index))  # SI
    # my_palette = plt.cm.get_cmap("Paired", len(df_spider.index)) # NO
    # my_palette = plt.cm.get_cmap("Set3", len(df_spider.index)) # NO
    # my_palette = plt.cm.get_cmap("Dark2", len(df_spider.index)) # SI

    # for row in range(0, len(df_norm.index)):
    for row in range(0, len(df_spider.index)):
        make_spider(df=df_spider, row=row, title=df_spider['group'][row], color=my_palette(row))

    # plt.suptitle("SpiderNet Chart of Ecological Acoustic Indices (EAIs) by Region", fontsize=13, y=0.98)
    plt.savefig(f'{path_out_visual_region}/visual_regions_spider.png', dpi=369, bbox_inches="tight")
    plt.close()
    # exit()
    print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
    print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')
    print('#### TIMES #### Visual Regions - TOTAL TOTAL ==>>', round(time.time() - t00, 3))


if __name__ == '__main__':
    visualisation_regions()
    print('Mirá ve.. oís?? alles gut oder was??')
