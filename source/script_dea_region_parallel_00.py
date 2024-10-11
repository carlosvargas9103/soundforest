import gc
gc.collect()

import os
import io
import copy
import time

from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import urllib.request

from glob import glob
from joblib import Parallel, delayed
from itertools import cycle, chain

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

### IDENTIFY PATH ###
t0 = time.time()
cwd = os.getcwd()
# path = f"{cwd}/out/"

### DEFINE ENVIRONMENT VARIABLES ###
# TODO: define all the environment variables
global job_id, N_JOBS, sreg, si, bandas, sr, b_band, u_band, bandwidth
job_id = os.environ.get('SLURM_JOB_ID') or "NULL"
N_JOBS = 8  # os.environ.get('N_JOBS') or 4
sreg, si, bandas = 0, 0, 10
b_band, u_band, bandwidth = 0, 10000, 1000

### LOAD AUDIO FILES ###
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith("/source") else cwd
# cwd = str(Path(cwd).parents[0])
print(cwd)
path_data = f"{cwd}/data/"
path_out = f"{cwd}/out/"

configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
               for dirpath, dirnames, files in os.walk(path_data)
               for f in files if f.endswith('.mp3')]
print(path_data, path_out, configfiles)
uniqueRegions = list(set(r[0] for r in configfiles))
region = uniqueRegions[sreg]  # for r in uniqueRegions
audio_files = [f[1] for f in configfiles if f[0] == region]
print(audio_files[:3])
print(f'{region} =>', len(audio_files))

# This function is meant to be used in a parallel fashion
print(f'Starting with parallel jobs: {len(audio_files)} time: {int(time.time())}')


def process_soundscape(audio_file: str = audio_files[si],
                       region: str = region,
                       si: int = si,
                       bandas: int = bandas,
                       path_data=f"{cwd}/data/",
                       path_out=f"{cwd}/out/",
                       ):
    ### LOAD COLOUR TEMPLATES ###
    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # TODO: CREATE PARALLEL JOB PER FILE
    y_c, sr = librosa.load(audio_file, sr=None)  # , duration=1800)
    tt = int(len(y_c) / sr)
    print(f'y: {y_c[:9]}')
    print(f'total samples in y: {y_c.shape} in time: {tt} secs')
    print(f'samples rate per second:  {sr}')
    # y_c = copy.deepcopy(y)
    print('###', 'ORIGINAL Y_C', len(y_c), y_c[:9], type(y_c))
    # [1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec, 1800 => per sec]
    my_chunks = 1800 / 30
    split_y = np.hsplit(y_c, my_chunks)
    print('###', 'CHUNKS Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size: {len(split_y[0])}',
          type(split_y), type(split_y[0])
          )

    # wrap audio_denoise with parameters
    def audio_denoise_st(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)

    def audio_denoise_ns(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)

    t1 = time.time()
    # parallel audio_denoise => ~30sec
    split_y_rn_st = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(audio_denoise_st)(y_i) for y_i in split_y)
    y_rn_st_p = np.array(split_y_rn_st).flatten()

    split_y_rn_ns = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(audio_denoise_ns)(y_i) for y_i in split_y)

    print('###', 'PARALLEL DENOISE Y_C REARRANGED', type(y_rn_st_p), len(y_rn_st_p), y_rn_st_p[:9],
          int(time.time() - t1))

    def plot_raw_signal(y=y_c, sr=sr, color_pal=color_pal,
                        region=region, si=si,
                        path_out=f"{cwd}/out/") -> None:
        # Calculate the time for each sample
        tt = np.arange(0, len(y)) / sr
        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(tt, y, lw=1, color=color_pal[0])
        title = f'Raw Audio Region {region} {si}'
        plt.title(title)
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        # plt.show()
        # save the figure to file
        plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                    format="png", dpi=600)
        plt.clf()

    plot_raw_signal()

    def split_freq_band(s: np.memmap = None,
                        sr: int = sr,
                        b_band: int = b_band,
                        u_band: int = u_band,
                        bandwidth: int = bandwidth) -> List[np.ndarray]:
        # Calculate the FFT
        y_fft = np.fft.fft(s)
        # Calculate the frequencies for the FFT
        fft_freq = np.fft.fftfreq(len(s), 1.0 / sr)
        signals = []
        for f in range(b_band, u_band, bandwidth):
            mask = (fft_freq >= f) & (fft_freq < f + bandwidth)
            y_fft_band = y_fft[mask]
            fft_freq_band = fft_freq[mask]
            y_band = np.fft.ifft(y_fft_band)
            y_band = np.real(y_band)  # Take only the real part
            signals.append(y_band)
        return signals

    # PARALLEL split_freq_band => ~6sec
    print('###', 'PARALLEL split_freq_band')
    t1 = time.time()
    split_y_rn_st_split = np.array(
        Parallel(n_jobs=N_JOBS, verbose=1)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_st))
    split_y_rn_ns_split = np.array(
        Parallel(n_jobs=N_JOBS, verbose=1)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_ns))

    print('####', 'PARALLEL SPLIT DENOISE Y_C SPLIT PER FREQUENCY',
          len(split_y_rn_st_split), len(split_y_rn_st_split[0]),
          len(split_y_rn_st_split[0][0]), split_y_rn_st_split[0][0][:9],
          time.time() - t1)

    # REARRANGE THE SIGNAL
    # https://chatgpt.com/share/670726c0-e354-8006-84cc-330ef17beb5a
    t1 = time.time()
    print('#### TIMES #### List comprehension')
    y_rn_split_st_10 = [split_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    y_rn_split_ns_10 = [split_y_rn_ns_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### List comprehension:', time.time() - t1)

    # t1 = time.time()
    # print('#### TIMES #### For loop:')
    # y_rn_split_st_10 = []
    # # y_rn_split_ns_10 = []
    # for b in range(0, bandas):
    #     y_rn_split_st_10.append(split_y_rn_st_split[:, b, :].flatten())
    #     y_rn_split_ns_10.append(split_y_rn_ns_split[:, b, :].flatten())
    # print('#### TIMES #### For loop:', time.time() - t1)

    print('####', 'REARRANGED THE SIGNAL PER 10 FREQUENCIES', type(y_rn_split_st_10), len(y_rn_split_st_10),
          len(y_rn_split_st_10[0]))

    def plot_split_signal(s_split_0: List[np.ndarray] = None,
                          s_split_1: List[np.ndarray] = None,
                          bandwidth: int = 1000,
                          region: str = region,
                          si: int = si,
                          path_out: str = path_out
                          ) -> None:
        if not s_split_1:
            for i, y_band_0 in enumerate(s_split_0):
                # Calculate the time for each sample
                # time = np.arange(0, len(y_rn)) / sr
                time_band = np.arange(0, len(y_band_0)) / bandwidth
                # Plot the signal
                plt.figure(figsize=(10, 5))
                plt.plot(time_band, y_band_0, lw=1, color=color_pal[1], label='Stationary Denoised')
                title = f'Signals for {i * bandwidth}-{(i + 1) * bandwidth} Hz Band in {region} {si}'
                plt.title(title)
                plt.xlabel('Time (sec)')
                plt.ylabel('Amplitude')
                plt.legend(loc='upper left')
                # plt.show()
                # save the figure to file
                plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                            format="png", dpi=600)
                plt.clf()
        elif (s_split_1 and s_split_1):
            for i, (y_band_0, y_band_1) in enumerate(zip(s_split_0, s_split_1)):
                # Calculate the time for each sample
                # time = np.arange(0, len(y_rn)) / sr
                time_band = np.arange(0, max([len(y_band_0), len(y_band_1)])) / bandwidth
                # Plot the signal
                plt.figure(figsize=(10, 5))
                plt.plot(time_band, y_band_0, lw=1, color=color_pal[1], label='Stationary Denoised')
                plt.plot(time_band, y_band_1, lw=1, color=color_pal[2], label='Non-Stationary Denoised')
                title = f'Signals for {i * bandwidth}-{(i + 1) * bandwidth} Hz Band in {region} {si}'
                plt.title(title)
                plt.xlabel('Time (sec)')
                plt.ylabel('Amplitude')
                plt.legend(loc='upper left')
                # plt.show()
                # save the figure to file
                plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                            format="png", dpi=600)
                plt.clf()

    plot_split_signal(y_rn_split_st_10, y_rn_split_ns_10)

    print('###', 'COMPUTE SLIDING WINDOW WITH SUM')
    # bandwidth, secs, w_size_mins = 1000, 60, 0.06  # every 3.6 secs











process_soundscape()









outfile_name = f"cities_{job_id}_{int(time.time())}.txt"

data = ['cali',
        'thornbury',
        'mexico',
        'wien',
        'porto',
        'lausanne',
        'rapperswil'
        ]

with open(path_out + outfile_name, 'w') as file:
    file.write('\n'.join(data) + '\n')
file.close()

print(f'cwd: {cwd}',
      f'file: {path_out + outfile_name}',
      f'total_time: {round(time.time() - t0, 3)}',
      f'job_id: {job_id}')
print('####', f"SLEEPING TIME:", f'{round(time.time() - t0, 3)} sec', '####')
print("Alles good parcero, aller!")
