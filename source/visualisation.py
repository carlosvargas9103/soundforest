import gc

# from soupsieve.util import lower

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


def visualise_distribution(audio_files_in: pd.DataFrame = None) -> None:
    # TODO: Given all the preprocessed datasets plot distribution with box-plots
    ...


def visualise_soundscape(audio_file: str = '',
                         region: str = '',
                         si: int = 1964, *,
                         sr: int = 48000,
                         b_band: int = 0,
                         u_band: int = 10000,
                         bandas: int = 10,
                         bandwidth: int = 1000,
                         path_data: str = '',
                         path_out: str = '',
                         samples_s: int = 1800,
                         isamples_s: int = 3,
                         secs_b: int = 60,
                         w_size_mins: float = 0.06,
                         n_jobs: int = 1,
                         job_id: str = 'NULL',
                         verbose: bool = False,
                         f_pattern_out: str = 'visualisation'
                         ) -> None:
    # LOAD numpy ndarrays
    # print(path_out)
    path_fig_out = f'{path_out}data/{f_pattern_out}/{region}/'
    # print(path_fig_out)
    # print(audio_file)
    vectors = np.load(audio_file, allow_pickle=True)
    # print(len(data_export), )
    # (_, y_c), (_, sum_y_c_split), (_, sum_y_rn_st_split), (_, sum_y_rn_ns_split) = vectors[0], vectors[1], vectors[2], vectors[3]
    (_, sum_y_c_split), (_, sum_y_rn_st_split), (_, sum_y_rn_ns_split) = vectors[0], vectors[1], vectors[2]
    y_c = copy.deepcopy(sum_y_c_split)
    # print(type(y_c), len(y_c))
    # <class 'numpy.ndarray'> 10 [-0.26039539  0.83492684  0.26657928]
    print(type(sum_y_c_split), len(sum_y_c_split), sum_y_c_split[0][:3])

    ### LOAD COLOUR TEMPLATES ###
    sns.set_theme(style='white', palette=None)
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def plot_raw_signal(y=y_c, sr=sr, color_pal=color_pal,
                        region: str = region, si: int = si,
                        path_out: str = f'{path_fig_out}/RAW/') -> None:
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

    # TODO: export the y_c raw signal
    plot_raw_signal()

    # exit()

    def plot_split_signal(s_split_0: List[np.ndarray] = None,
                          s_split_1: List[np.ndarray] = None,
                          bandwidth: int = bandwidth,
                          region: str = region,
                          si: int = si,
                          path_out: str = f'{path_fig_out}/BANDS/'
                          ) -> None:
        if s_split_1 is None:
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
                            format='png', dpi=600)
                plt.clf()
                matplotlib.pyplot.close()

        elif (s_split_0 is not None and s_split_1 is not None):
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

    # plot_split_signal(sum_y_rn_st_split, sum_y_rn_ns_split)
    # exit()

    def plot_sum_split_signal(s_split_0: List[np.ndarray] = None, s_split_1: List[np.ndarray] = None,
                              y_label: str = 'SUM',
                              bandwidth: int = bandwidth, secs: int = secs_b, w_size_mins: int = w_size_mins,
                              region: str = region, si: int = si,
                              path_out: str = f'{path_fig_out}/SUM/') -> None:
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
        elif (s_split_1 is not None and s_split_0 is not None):
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

    # exit()

    def plot_3d_split_signal(s_split: List[np.ndarray] = None, bandwidth: int = bandwidth, y_label: str = 'SUM',
                             region: str = region, si: int = si,
                             path_out: str = f'{path_fig_out}/3D/') -> None:
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
                               path_out: str = f'{path_fig_out}/3D/') -> None:
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
                                 path_out: str = f'{path_fig_out}/3D/') -> None:
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
                                  path_out: str = f'{path_fig_out}/3D/') -> None:
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
