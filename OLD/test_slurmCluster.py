import os
import io
import time

from itertools import cycle
from typing import List
import urllib.request
from glob import glob

from scipy.io import wavfile
import soundfile as sf
import librosa
import librosa.display
import IPython
import IPython.display as ipd
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import seaborn as sns

import dask
from dask import delayed
import dask.config
import dask.dataframe as dd
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

t0 = time.time()

cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas
os.chdir('../')
print(cwd)
path_data = f"{cwd}/data/"
path_out = f"{cwd}/out/"

job_id = os.environ.get('SLURM_JOB_ID') or "NULL"
outfile_name = "cities_" + job_id + ".txt"

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
               for dirpath, dirnames, files in os.walk(path_data)
               for f in files if f.endswith('.mp3')]

sreg = 0  # NaturalRegeneration
uniqueRegions = list(set(r[0] for r in configfiles))
region = uniqueRegions[sr]  # Region by region
audio_files = [f[1] for f in configfiles if f[0] == region]
print(audio_files)
print(f"{sreg}", f'{region} =>', len(audio_files))

### Let's run this for only one audio file
si = 0

y, sr = librosa.load(audio_files[si], sr=None)  # , duration=1800)
y_da = da.array(y)
time = int(len(y) / sr)
print(f'y: {y[:9]}')
print(f'total samples in y: {y.shape} in time: {time} secs')
print(f'samples rate per second:  {sr}')


### Let's plot one audio sample from one region ###
# Reduce Noise Stationary

def audio_denoise(y: da = None) -> da:
    y_rn = nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)
    # Reduce Noise NON-Stationary
    y_rn_ns = nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)


data = ['cali',
        'thornbury',
        'mexico',
        'wien',
        'porto',
        'lausanne',
        'rapperswil'
        ]

with open(path_out + outfile_name, 'w') as file:
    time.sleep(0.3)  # Sleep for 0.3 seconds
    t1 = time.time()
    t_diff = t1 - t0
    file.write('\n'.join(data) + '\n')

file.close()

t1 = time.time()
t_diff = t1 - t0

print(f'cwd: {cwd}',
      f'file: {path_out + outfile_name}',
      f'time: {t_diff}',
      f'job_id: {job_id}')
print(f"Sleeping time: {round(t_diff, 3)}")
print("Alles gut parcero, let's aller!")
