import gc

gc.collect()

import re
import ast
import csv
import copy
import time
import joblib

from typing import List, Tuple

from joblib import Parallel, delayed
from itertools import cycle

import librosa
import noisereduce as nr

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

# print(f'{si} - audio_file: {audio_file} - region: {region}')
### LOAD COLOUR TEMPLATES ###
sns.set_theme(style='white', palette=None)
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


"""

def process_soundscape(audio_file: str = '',
                       region: str = '',
                       si: int = 1964, *,
                       b_band: int = 0,
                       u_band: int = 10000,
                       bandas: int = 10,
                       bandwidth: int = 1000,
                       path_data: str = '',
                       path_out: str = '',
                       samples_s: int = 1800,
                       isamples_s: int = 3,
                       verbose: bool = False,
                       n_jobs: int = 1,
                       job_id: str = 'NULL'
                       ) -> None:
    # print(f'{si} - audio_file: {audio_file} - region: {region}')
    ### LOAD COLOUR TEMPLATES ###
    sns.set_theme(style='white', palette=None)
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # PARALLEL JOBS PER FILE
    y, y_c = None, None
    y, sr = librosa.load(audio_file, sr=None)  # , duration=1800)
    y_c = copy.deepcopy(y)
    tt = int(len(y_c) / sr)
    # print('RAW y:', y_c[:6])
    # print('total samples in y:', y_c.shape, 'total time:', tt,  'secs', 'samples rate per second:', sr)
    print('####', 'ORIGINAL Y_C', len(y_c), y_c[:3], type(y_c))
    # [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
    my_chunks = samples_s / isamples_s
    split_y = np.hsplit(y_c, my_chunks)
    print('####', 'CHUNKS Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size: {len(split_y[0])}',
          type(split_y), type(split_y[0])
          )

    # wrap methods audio_denoise with parameters
    def audio_denoise_st(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)

    def audio_denoise_ns(y=None, sr: int = 48000):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)

    # TODO: implement the case of no-noise reduction
    t1 = time.time()
    # parallel audio_denoise => ~30sec
    split_y_rn_st = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_st)(y_i, sr) for y_i in split_y)
    split_y_rn_ns = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_ns)(y_i, sr) for y_i in split_y)

    def split_freq_band(s: np.memmap = None,
                        sr: int = sr,
                        b_band: int = b_band,
                        u_band: int = u_band,
                        bandwidth: int = bandwidth) -> List[np.ndarray]:
        # TODO: validate if the transformation is possible using the chunk of the soundscape or needs to be the full
        # Calculate the FFT
        # 144000 48000 2.0833333333333333e-05 0 10000 1000
        # print(len(s), sr, 1/sr, b_band, u_band, bandwidth)
        y_fft = np.fft.fft(s)
        # Calculate the frequencies for the FFT
        fft_freq = np.fft.fftfreq(len(s), 1.0 / sr)
        # Original sample = 2880000 => after fft = 60000 * 10. Then, 2280000 samples are lost
        # print(len(set(np.select([fft_freq < b_band, fft_freq > u_band], [fft_freq, fft_freq]))))
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
    print('####', 'PARALLEL split_freq_band')
    t1 = time.time()
    split_y_c_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y)
    )
    split_y_rn_st_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_st)
    )
    split_y_rn_ns_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_ns)
    )

    print('####', 'PARALLEL SPLIT ORIGINAL Y_C SPLIT PER FREQUENCY',
          len(split_y_c_split), len(split_y_c_split[0]), len(split_y_c_split[0][0]), '\n',
          split_y_c_split[0][0][:3])
    print('####', 'PARALLEL SPLIT DENOISE Y_C SPLIT PER FREQUENCY',
          len(split_y_rn_st_split), len(split_y_rn_st_split[0]), len(split_y_rn_st_split[0][0]), '\n',
          split_y_rn_st_split[0][0][:3], '\n',
          '####', 'TIME', round(time.time() - t1, 3), 'seconds')

    # REARRANGE THE SIGNAL
    t1 = time.time()
    print('#### REARRANGE THE SIGNAL #### List comprehension')
    y_c_split_10 = [split_y_c_split[:, b, :].flatten() for b in range(0, bandas)]
    y_rn_split_st_10 = [split_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    y_rn_split_ns_10 = [split_y_rn_ns_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### flatten:', round(time.time() - t1, 3), 'seconds')

    print('####', 'REARRANGED THE SIGNAL PER 10 FREQUENCIES',
          type(y_rn_split_st_10), len(y_rn_split_st_10),
          type(y_rn_split_st_10[0]), len(y_rn_split_st_10[0])
          )

    print('####', 'COMPUTE SLIDING WINDOW WITH SUM')
    bandwidth, secs, w_size_mins = bandwidth, 60, 0.06  # 1000, 60, 0.06 => every 3.6 secs

    def sum_this_sliding_window(data, size=int(bandwidth * secs), stepsize=int(bandwidth * secs * w_size_mins),
                                padded=False, axis=-1, copy=False, suma=True, average=False) -> np.ndarray:
        if axis >= data.ndim:
            raise ValueError('Axis value out of range')
        if stepsize < 1:
            raise ValueError('Stepsize may not be zero or negative')
        if size > data.shape[axis]:
            raise ValueError('Sliding window size may not exceed size of selected axis')
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
    sum_y_c_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_c_split_10)
    )
    sum_y_rn_st_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_st_10)
    )
    sum_y_rn_ns_split = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_ns_10)
    )

    # NOT NEEDED - REARRANGE THE SIGNAL
    # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### sum_this_sliding_window', round(time.time() - t1, 3), 'seconds')

    print('####', 'REARRANGED THE SIGNAL SUM SLIDING_WINDOW PER 10 FREQUENCIES',
          type(sum_y_rn_st_split), len(sum_y_rn_st_split),
          type(sum_y_rn_st_split[0]), len(sum_y_rn_st_split[0])
          )

    # <class 'numpy.ndarray'> 10
    # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
    # sum_y_rn_st_split to .csv

    data_export = [
        ('sum_y_c_split', sum_y_c_split),
        ('sum_y_rn_st_split', sum_y_rn_st_split),
        ('sum_y_rn_ns_split', sum_y_rn_ns_split)
    ]
    for d_e in data_export:
        (d_n, df) = d_e
        np.save(
            f'{path_out}data/{region}_{audio_file.split("/")[-1][:-4]}_{d_n}_{si}_{job_id}_{int(time.time())}.npy',
            df)

        # pd.DataFrame(
        #     sum_y_rn_st_split.T.astype(float),
        #     columns=[f'band_{int(i)}' for i in range(sum_y_rn_st_split.shape[0])]).to_csv(
        #     f'{path_out}data/{region}_{audio_file.split("/")[-1][:-4]}_sum_y_rn_st_split_{si}_{job_id}_{int(time.time())}.csv',
        #     sep=';'
        # )
        
"""

def plot_raw_signal(y=y_c, sr=sr, color_pal=color_pal,
                    region: str = region, si: int = si,
                    path_out: str = f'{path_out}/figs/RAW/') -> None:
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
                format='png', dpi=600)
    plt.clf()
    matplotlib.pyplot.close()

plot_raw_signal()

def plot_split_signal(s_split_0: List[np.ndarray] = None,
                      s_split_1: List[np.ndarray] = None,
                      bandwidth: int = bandwidth,
                      region: str = region,
                      si: int = si,
                      path_out: str = f'{path_out}figs/BANDS/'
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
                        format='pn', dpi=600)
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
                        format='png', dpi=600)
            plt.clf()
            matplotlib.pyplot.close()

plot_split_signal(y_rn_split_st_10, y_rn_split_ns_10)

def plot_sum_split_signal(s_split_0: List[np.ndarray] = None, s_split_1: List[np.ndarray] = None,
                          y_label: str = 'SUM',
                          bandwidth: int = bandwidth, secs: int = secs, w_size_mins: int = w_size_mins,
                          region: str = region, si: int = si,
                          path_out: str = f'{path_out}figs/SUM/') -> None:
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
                        format='png', dpi=600)
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
                        format='png', dpi=600)
            plt.clf()
            matplotlib.pyplot.close()

plot_sum_split_signal(sum_y_rn_st_split, sum_y_rn_ns_split)

def plot_3d_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                         region: str = region, si: int = si,
                         path_out: str = f'{path_out}figs/3D/') -> None:
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
                format='png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()

plot_3d_split_signal(sum_y_rn_st_split)

def plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                           region: str = region, si: int = si,
                           path_out: str = f'{path_out}figs/3D/') -> None:
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
                format='png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()
    # plt.close()

plot_surf_split_signal(sum_y_rn_st_split)

def r_plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                             region: str = region, si: int = si,
                             path_out: str = f'{path_out}figs/3D/') -> None:
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
                format='png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()
    # plt.close()

# r_plot_surf_split_signal(sum_y_rn_st_split)

def rr_plot_surf_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                              region: str = region, si: int = si,
                              path_out: str = f'{path_out}figs/3D/') -> None:
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
                format='png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()
    # plt.close()

# rr_plot_surf_split_signal(sum_y_rn_st_split)

if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
    # TODO: extract visualisation modules here
