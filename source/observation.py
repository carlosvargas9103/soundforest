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
                       secs_b: int = 60,
                       w_size_mins: float = 0.06,
                       verbose: bool = False,
                       n_jobs: int = 1,
                       job_id: str = 'NULL'
                       ) -> None:
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
    split_y_c_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y))
    split_y_rn_st_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_st))
    split_y_rn_ns_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_ns))

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
    # bandwidth, secs, w_size_mins = bandwidth, secs_b, w_size_mins  # 1000, 60, 0.06 => every 3.6 secs
    def sum_this_sliding_window(data,
                                size=int(bandwidth * secs_b),
                                stepsize=int(bandwidth * secs_b * w_size_mins),
                                padded=False,
                                axis=-1,
                                copy=False,
                                suma=True,
                                average=False
                                ) -> np.ndarray:
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
    sum_y_c_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_c_split_10))
    sum_y_rn_st_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_st_10))
    sum_y_rn_ns_split = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sum_this_sliding_window)(s) for s in y_rn_split_ns_10))

    # NOT NEEDED - REARRANGE THE SIGNAL
    # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### sum_this_sliding_window', round(time.time() - t1, 3), 'seconds')

    print('####', 'REARRANGED THE SIGNAL SUM SLIDING_WINDOW PER 10 FREQUENCIES',
          type(sum_y_rn_st_split), len(sum_y_rn_st_split),
          type(sum_y_rn_st_split[0]), len(sum_y_rn_st_split[0])
          )

    # <class 'numpy.ndarray'> 10
    # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
    data_export = [
        # ('y_c', y_c),
        ('sum_y_c_split', sum_y_c_split),
        ('sum_y_rn_st_split', sum_y_rn_st_split),
        ('sum_y_rn_ns_split', sum_y_rn_ns_split)
    ]

    np.save(f'{path_out}data/observation/{region}/{audio_file.split("/")[-1][:-4]}_{si}_{job_id}_{int(time.time())}.npy',
            np.array(data_export, dtype=object),
            allow_pickle=True)

if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
    # TODO: extract observation modules here