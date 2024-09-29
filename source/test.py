import os
import io
from typing import List
import urllib.request
from glob import glob

from itertools import cycle

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

import time

t0 = time.time()

cwd = os.getcwd()

path = f"{cwd}/out/"
job_id = os.environ.get('SLURM_JOB_ID')
outfile_name = "cities_" + job_id + ".txt"

data = ['cali',
        'thornbury',
        'mexico',
        'wien',
        'porto',
        'lausanne',
        'rapperswil'
        ]

with open(path + outfile_name, 'w') as file:
    time.sleep(0.3)  # Sleep for 0.3 seconds
    t1 = time.time()
    t_diff = t1 - t0
    file.write('\n'.join(data) + '\n')

file.close()

t1 = time.time()
t_diff = t1 - t0

print(f'cwd: {cwd}',
      f'file: {path+outfile_name}',
      f'time: {t_diff}',
      f'job_id: {job_id}')
print(f"Sleeping time: {round(t_diff, 3)}")
print("Alles gut parcero, let's aller!")
