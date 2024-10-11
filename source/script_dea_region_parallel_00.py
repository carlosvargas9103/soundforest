import gc

gc.collect()

import os
import io
import copy
import time
import joblib

from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import urllib.request

from glob import glob
from joblib import Parallel, delayed
from joblib import effective_n_jobs
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
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

### IDENTIFY PATH ###
t0 = time.time()
cwd = os.getcwd()
# path = f"{cwd}/out/"

### DEFINE ENVIRONMENT VARIABLES ###
# TODO: define all the environment variables
global job_id, N_JOBS, sreg, si, bandas, sr, b_band, u_band, bandwidth, audio_files, samples_s, isamples_s, verbose, f_progress
job_id = os.environ.get('SLURM_JOB_ID') or "NULL"
N_JOBS = int(effective_n_jobs(-1)) or -1  # os.environ.get('N_JOBS') or 4
sreg, si, bandas = 0, 0, 10
b_band, u_band, bandwidth, samples_s, isamples_s, verbo = 0, 10000, 1000, 1800, 10, True

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
print(path_data, path_out, configfiles[:3])


# uniqueRegions = list(set(r[0] for r in configfiles))

# TODO: USE GLOBAL VARIABLE sreg TO SUBMIT PARALLEL JOBS VIA args in batch script
# region = uniqueRegions[sreg]  # for r in uniqueRegions
# audio_files = [f[1] for f in configfiles if f[0] == region]
# print(audio_files[:3])
# print(f'{region} =>', len(audio_files))


def process_soundscape(audio_file: str = "",
                       region: str = "",
                       si: int = 1991, *,
                       bandas: int = bandas,
                       bandwidth: int = bandwidth,
                       path_data=f"{cwd}/data/",
                       path_out=f"{cwd}/out/",
                       verbose:bool=verbo
                       ):
    print(f'{si} - audio_file: {audio_file} - region: {region}')
    # return
    ### LOAD COLOUR TEMPLATES ###
    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # PARALLEL JOBS PER FILE
    y, sr = librosa.load(audio_file, sr=None)  # , duration=1800)
    y_c = None
    y_c = copy.deepcopy(y)
    tt = int(len(y_c) / sr)
    print(f'y: {y_c[:9]}')
    print(f'total samples in y: {y_c.shape} in time: {tt} secs')
    print(f'samples rate per second:  {sr}')
    print('###', 'ORIGINAL Y_C', len(y_c), y_c[:9], type(y_c))
    # [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
    my_chunks = samples_s / isamples_s
    split_y = np.hsplit(y_c, my_chunks)
    print('###', 'CHUNKS Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size: {len(split_y[0])}',
          type(split_y), type(split_y[0])
          )

    # wrap methods audio_denoise with parameters
    def audio_denoise_st(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)

    def audio_denoise_ns(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)

    t1 = time.time()
    # parallel audio_denoise => ~30sec
    split_y_rn_st = Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(audio_denoise_st)(y_i) for y_i in split_y)
    split_y_rn_ns = Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(audio_denoise_ns)(y_i) for y_i in split_y)

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
        matplotlib.pyplot.close()

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
        Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_st)
    )
    split_y_rn_ns_split = np.array(
        Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_ns)
    )

    print('####', 'PARALLEL SPLIT DENOISE Y_C SPLIT PER FREQUENCY',
          len(split_y_rn_st_split), len(split_y_rn_st_split[0]),
          len(split_y_rn_st_split[0][0]), split_y_rn_st_split[0][0][:9],
          time.time() - t1)

    # REARRANGE THE SIGNAL
    t1 = time.time()
    print('#### TIMES #### List comprehension')
    y_rn_split_st_10 = [split_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    y_rn_split_ns_10 = [split_y_rn_ns_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### List comprehension:', time.time() - t1)

    print('####', 'REARRANGED THE SIGNAL PER 10 FREQUENCIES', type(y_rn_split_st_10), len(y_rn_split_st_10),
          type(y_rn_split_st_10[0]), len(y_rn_split_st_10[0]))

    def plot_split_signal(s_split_0: List[np.ndarray] = None,
                          s_split_1: List[np.ndarray] = None,
                          bandwidth: int = bandwidth,
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
                matplotlib.pyplot.close()

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
                matplotlib.pyplot.close()

    plot_split_signal(y_rn_split_st_10, y_rn_split_ns_10)

    print('###', 'COMPUTE SLIDING WINDOW WITH SUM')
    bandwidth, secs, w_size_mins = bandwidth, 60, 0.06  # 1000, 60, 0.06 => every 3.6 secs

    def sum_this_sliding_window(data, size=int(bandwidth * secs), stepsize=int(bandwidth * secs * w_size_mins),
                                padded=False, axis=-1, copy=False, suma=True, average=False) -> np.ndarray:
        if axis >= data.ndim:
            raise ValueError("Axis value out of range")
        if stepsize < 1:
            raise ValueError("Stepsize may not be zero or negative")
        if size > data.shape[axis]:
            raise ValueError("Sliding window size may not exceed size of selected axis")
        shape = list(data.shape)
        shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
        shape.append(size)
        strides = list(data.strides)
        strides[axis] *= stepsize
        strides.append(data.strides[axis])
        # TODO: Maybe with dask??
        strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        if suma:
            return [np.sum(s, axis=0) for s in strided.copy()] if copy else [np.sum(s, axis=0) for s in strided]
        elif average:
            return [np.mean(s, axis=0) for s in strided.copy()] if copy else [np.mean(s, axis=0) for s in strided]
        else:
            return strided.copy() if copy else strided

    # SLIDING WINDOWS FOR THE SIGNAL
    t1 = time.time()
    print('#### RUNNING #### sum_this_sliding_window')
    sum_y_rn_st_split = np.array(Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_st_10))
    sum_y_rn_ns_split = np.array(Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_ns_10))
    # NOT NEEDED - REARRANGE THE SIGNAL
    # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### sum_this_sliding_window', time.time() - t1)

    print('####', 'REARRANGED THE SIGNAL SUM SLIDING_WINDOW PER 10 FREQUENCIES',
          type(sum_y_rn_st_split), len(sum_y_rn_st_split),
          type(sum_y_rn_st_split[0]), len(sum_y_rn_st_split[0])
          )

    def plot_sum_split_signal(s_split_0: List[np.ndarray] = None, s_split_1: List[np.ndarray] = None, y_label: str = 'SUM',
                              bandwidth: int = bandwidth, secs: int = secs, w_size_mins: int = w_size_mins,
                              region: str = region, si: int = si, path_out: str = path_out) -> None:
        if s_split_1 is None:
            for i, y_band_0 in enumerate(s_split_0):
                # time = np.arange(0, len(y_rn)) / sr
                time_band = np.arange(0, len(y_band_0)) * w_size_mins  # / bandwidth
                plt.figure(figsize=(10, 5))
                plt.plot(time_band, y_band_0, lw=1, color=color_pal[1], label='Stationary Denoised')
                title = f'{y_label} Signals for {i * bandwidth}-{(i + 1) * bandwidth} Hz Band in {region} {si}'
                plt.title(title)
                plt.xlabel('Time (min)')
                plt.ylabel(f'{y_label}(Amplitude) on {round(secs * w_size_mins, 1)} secs')
                plt.legend(loc='upper left')
                # plt.show()
                # save the figure to file
                plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                            format="png", dpi=600)
                plt.clf()
                matplotlib.pyplot.close()

        # elif (s_split_1 and s_split_1):
        elif (s_split_1 is not None and
              s_split_0 is not None):
            for i, (y_band_0, y_band_1) in enumerate(zip(s_split_0, s_split_1)):
                # time = np.arange(0, len(y_rn)) / sr
                time_band = np.arange(0, max([len(y_band_0), len(y_band_1)])) * w_size_mins  # / bandwidth
                plt.figure(figsize=(10, 5))
                plt.plot(time_band, y_band_0, lw=1, color=color_pal[1], label='Stationary Denoised')
                plt.plot(time_band, y_band_1, lw=1, color=color_pal[2], label='Non-Stationary Denoised')
                title = f'{y_label} Signals for {i * bandwidth}-{(i + 1) * bandwidth} Hz Band in {region} {si}'
                plt.title(title)
                plt.xlabel('Time (min)')
                plt.ylabel(f'{y_label}(Amplitude) on {round(secs * w_size_mins, 1)} secs')
                plt.legend(loc='upper left')
                # plt.show()
                # save the figure to file
                plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                            format="png", dpi=600)
                plt.clf()
                matplotlib.pyplot.close()

    plot_sum_split_signal(sum_y_rn_st_split, sum_y_rn_ns_split)

    def plot_3d_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                             region: str = region, si: int = si, path_out: str = path_out) -> None:
        fig = plt.figure(figsize=(60, 30))
        ax = fig.add_subplot(111, projection='3d')
        for i, y in enumerate(np.array(s_split)):
            x = np.full(y.shape, i)
            time_band = np.arange(0, len(y)) / bandwidth
            ax.plot(x, time_band, y)
        title = f'{y_label} 3D Signals for all frequency bands in {region} {si}'
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Time (Seconds)')
        ax.set_zlabel('Amplitude')
        fig.tight_layout()
        # plt.show()
        # save the figure to file
        plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                    format="png", dpi=300)
        plt.clf()
        matplotlib.pyplot.close()

    plot_3d_split_signal(sum_y_rn_st_split)

    def plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                               region: str = region, si: int = si, path_out: str = path_out) -> None:
        fig = plt.figure(figsize=(60, 30))
        ax = fig.add_subplot(111, projection='3d')

        # Convert the split signals into 2D arrays for X, Y, and Z
        num_bands = len(s_split)
        signal_length = len(s_split[0])
        X, Y = np.meshgrid(np.arange(num_bands), np.arange(signal_length) / bandwidth)
        Z = np.array(s_split).T

        # Plot the surface with transparency (alpha)
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25, cmap='viridis')

        # Add contour plots on all three axes
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 10, cmap='viridis')  # Contour on Z-axis
        ax.contour(X, Y, Z, zdir='x', offset=-1, cmap='viridis')  # Contour on X-axis
        ax.contour(X, Y, Z, zdir='y', offset=np.max(Y) + 1, cmap='viridis')  # Contour on Y-axis

        # Set title and labels
        title = f'{y_label} SURF Signals for all frequency bands in {region} {si}'
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Time (Seconds)')
        ax.set_zlabel('Amplitude')

        # Rotate the view further to the left (increase azimuth angle)
        # ax.view_init(elev=33, azim=-66)  # Adjust elevation and azimuth for left-hand side view

        # Adjust layout and save the figure
        fig.tight_layout()
        plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                    format="png", dpi=300)
        plt.clf()
        matplotlib.pyplot.close()
        # plt.close()

    plot_surf_split_signal(sum_y_rn_st_split)

    def r_plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                                 region: str = region, si: int = si, path_out: str = path_out) -> None:
        fig = plt.figure(figsize=(60, 30))
        ax = fig.add_subplot(111, projection='3d')

        # Convert the split signals into 2D arrays for X, Y, and Z
        num_bands = len(s_split)
        signal_length = len(s_split[0])
        X, Y = np.meshgrid(np.arange(num_bands), np.arange(signal_length) / bandwidth)
        Z = np.array(s_split).T

        # Plot the surface with coolwarm colormap and transparency (alpha)
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25, cmap='coolwarm')

        # Add contour plots only on the Z (amplitude) and X (frequency) axes
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 10, cmap='coolwarm')  # Contour on Z-axis (Amplitude)
        ax.contour(X, Y, Z, zdir='x', offset=-1, cmap='coolwarm')  # Contour on X-axis (Frequency)

        # Set title and labels
        title = f'{y_label} R-SURF Signals for all frequency bands in {region} {si}'
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Time (Seconds)')
        ax.set_zlabel('Amplitude')

        # Rotate the view to see contours on amplitude and frequency only
        ax.view_init(elev=30, azim=-60)  # Adjust angles for better viewing of X and Z contours

        # Adjust layout and save the figure
        fig.tight_layout()
        plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                    format="png", dpi=300)
        plt.clf()
        matplotlib.pyplot.close()
        # plt.close()

    r_plot_surf_split_signal(sum_y_rn_st_split)

    def rr_plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                                  region: str = region, si: int = si, path_out: str = path_out) -> None:
        fig = plt.figure(figsize=(60, 30))
        ax = fig.add_subplot(111, projection='3d')

        # Convert the split signals into 2D arrays for X, Y, and Z
        num_bands = len(s_split)
        signal_length = len(s_split[0])
        X, Y = np.meshgrid(np.arange(num_bands), np.arange(signal_length) / bandwidth)
        Z = np.array(s_split).T

        # Plot the surface with reduced opacity and coolwarm colormap
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.15, cmap='coolwarm')

        # Add contour plots on all three axes
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z) - 10, cmap='coolwarm')  # Contour on Z-axis (Amplitude)
        ax.contour(X, Y, Z, zdir='x', offset=-1, cmap='coolwarm')  # Contour on X-axis (Frequency)
        ax.contour(X, Y, Z, zdir='y', offset=np.max(Y) + 1, cmap='coolwarm')  # Contour on Y-axis (Time)

        # Set title and labels
        title = f'{y_label} RR-SURF Signals for all frequency bands in {region} {si}'
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Time (Seconds)')
        ax.set_zlabel('Amplitude')

        # Rotate the view further to the left (increase azimuth angle)
        ax.view_init(elev=25, azim=-50)  # Adjust elevation and azimuth for left-hand side view

        # Adjust layout and save the figure
        fig.tight_layout()
        plt.savefig(f'{path_out}{int(time.time())}_{"_".join(title.split())}_{job_id}.png',
                    format="png", dpi=300)
        plt.clf()
        matplotlib.pyplot.close()
        # plt.close()

    rr_plot_surf_split_signal(sum_y_rn_st_split)


t00 = time.time()
f_progress = []
outfile_name = f"f_progress_{job_id}_{int(time.time())}.txt"
print('#### #### HOI FOREST #### ####')
# Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(process_soundscape)(audio_file=a) for a in audio_files)

# This function is meant to be used in a parallel fashion
print(f'Starting with parallel jobs: {len(configfiles)} time: {int(time.time())}')
for i, f in enumerate(configfiles[:2]):
    t11 = time.time()
    f_progress.append(f'{i} - {f} -  {t11}')
    print(i, 'REGION:', '====>>>>', f[0], '#### RUNNING #### AUDIO:', '====>>>>', f[1])
    # print(f'audio_file: {f[1]} region_: {f[0]}')
    # try:
    process_soundscape(f[1], f[0], i)
    # except Exception as e:
    #     print ('HHHHHHHOOOOOOOOOOOORRRRRRRRRRRRRRRIIIIIIIIIIIIIIIBBBBBBBBBBBBLLLLLLLLLLLLLLLEEEEEEEEE')
    #     print ('HHHHHHHOOOOOOOOOOOORRRRRRRRRRRRRRRIIIIIIIIIIIIIIIBBBBBBBBBBBBLLLLLLLLLLLLLLLEEEEEEEEE', e)
    # finally:
    #     with open(path_out + outfile_name, 'w') as file:
    #         file.write('\n'.join(f_progress) + '\n')
    #     file.close()
    #     print('GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
    print(i, '#### TIMES #### process_soundscape TOTAL TOTAL ==>>', time.time() - t11)
print('#### TIMES #### process_soundscape TOTAL TOTAL ==>>', time.time() - t00)

data = ['cali',
        'thornbury',
        'mexico',
        'wien',
        'porto',
        'lausanne',
        'rapperswil'
        ]

outfile_name = f"cities_{job_id}_{int(time.time())}.txt"
with open(path_out + outfile_name, 'w') as file:
    file.write('\n'.join(data) + '\n')
file.close()

print('#### #### NO DA MAS #### TERMINO #### FINITO #### ####')
print(f'cwd: {cwd}',
      f'file: {path_out + outfile_name}',
      f'total_time: {round(time.time() - t0, 3)}',
      f' TOTAL_JOBS: {effective_n_jobs(N_JOBS)}',
      f'job_id: {job_id}')
print('####', f"SLEEPING TIME:", f'{round(time.time() - t0, 3)} sec', '####')
print("All guert parcero, aller!")
