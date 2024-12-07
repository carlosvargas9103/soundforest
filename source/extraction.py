import gc

gc.collect()

import copy
import time
import random
# import datetime

from typing import List, Tuple

import joblib
from joblib import Parallel, delayed

import librosa
import noisereduce as nr

import pandas as pd
import numpy as np


random.seed("9103")


def bootstrap_soundscape(audio_file: str = '',
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
                         f_pattern_out: str = 'extraction',
                         windows_13: bool = True,
                         horas: int = 30
                         ) -> None:
    d_re = {'NaturalRegeneration': 0, 'Pasture': 1, 'Plantation': 2, 'RefForest': 3}
    # print(f'{si} - audio_file: {audio_file} - region: {region}')
    # PARALLEL JOBS PER FILE
    y, y_c = None, None
    y, sr = librosa.load(audio_file, sr=None)  # , duration=1800)
    y_c = copy.deepcopy(y)
    tt = int(len(y_c) / sr)
    # print(f'y: {y_c[:3]}')
    # print(f'total samples in y: {y_c.shape} in time: {tt} secs')
    # print(f'samples rate per second:  {sr}')
    print('####', 'ORIGINAL Y_C', len(y_c), y_c[:3], type(y_c))
    # [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
    my_chunks = samples_s / secs_b
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

    def split_freq_band_per_chunk(s: np.memmap = None,
                                  sr: int = sr,
                                  b_band: int = b_band,
                                  u_band: int = u_band,
                                  bandwidth: int = bandwidth) -> List[np.ndarray]:
        # Calculate the FFT
        # 2880000 48000 2.0833333333333333e-05 10000 0 1000
        # print(len(s), sr, 1/sr, u_band, b_band, bandwidth)
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
    print('####', 'PARALLEL split_freq_band')
    t1 = time.time()
    split_y_split_bands = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band_per_chunk)(y_i) for y_i in split_y)
    )

    print('####', 'PARALLEL SPLIT Y_C PER FREQUENCY',
          len(split_y_split_bands), len(split_y_split_bands[0]),
          len(split_y_split_bands[0][0]), split_y_split_bands[0][0][:3],
          time.time() - t1)

    t1 = time.time()
    y_split_hours_bands = [split_y_split_bands[h, b, :].flatten() for b in range(0, bandas) for h in range(0, horas)]
    # ARRANGE in dict per BANDAS, and HORAS
    mean_y_hours_bandas = [
        {'r': d_re.get(region, 0), # CLASS (INT) 4
         'id_s': 123, #serial # DONT NEED
         # {'r': d_re.get(region.upper(), 0),
         'h': h, # TIME (int) 0 - 29 REMARK => DONT NEED!
         'b': b, # BAND (int) 0 - 9  => TODO: Consider 10-bands at once
         'm': np.mean(f := split_y_split_bands[h, b, :].flatten()), # MEAN of the VECTOR (float)
         'v': f # VECTOR (npArray[float]) => 60000 => 3600
         } for b in range(0, bandas) for h in range(0, horas)

        # TODO: Activation function (Sigmoid).
        # TODO: Evaluation Metrics for classification => Table & Matrix.
        # TODO: Extract the Benchmark from Giacomo.
        # TODO: Reduce the time of the samples.
        # TODO: Continuing with the pre-processing.
        # TODO: Next meeting => 08.01.2025.
        # TODO: Methodology PDFs FOLDER on Git?

    ]
    print('####', 'DICT VECTOR', len(mean_y_hours_bandas), list(mean_y_hours_bandas[0].keys()),
          mean_y_hours_bandas[0].get('v', [])[:3])
    print('####', 'TIMES', '####', 'FLATTEN:', round(time.time() - t1, 3))
    print('####', 'REARRANGED THE SIGNAL PER 10 FREQUENCIES', type(y_split_hours_bands), len(y_split_hours_bands),
          type(y_split_hours_bands[0]), len(y_split_hours_bands[0]))
    print('####', f'COMPUTE SLIDING WINDOW {windows_13}')

    # DEFAULT VALUES ARE: bandwidth, secs, w_size_mins = bandwidth, 60, 0.06  # 1000, 60, 0.06 => every 3.6 secs
    def w_this_sliding_window(data, size=int(bandwidth * secs_b * w_size_mins), stepsize=int(bandwidth * secs_b),
                              padded=False, axis=-1, copy=False, suma=False, average=False) -> np.ndarray:
        if axis >= data.ndim:
            raise ValueError('Axis value out of range')
        if stepsize < 1:
            raise ValueError('Stepsize may not be zero or negative')
        if size > data.shape[axis]:
            raise ValueError('Sliding window size may not exceed size of selected axis')
        shape = list(data.shape)
        # print(data.shape[axis] / stepsize - size / stepsize + 1)
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
    df_m = None
    if windows_13:
        w_vectors = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(w_this_sliding_window)(s.get('v')) for s in mean_y_hours_bandas)
        )
        print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
              type(w_vectors), len(w_vectors),
              type(w_vectors[0]), len(w_vectors[0]), len(w_vectors[0][0]),
              )
        w_vectors = w_vectors.reshape(w_vectors.shape[0], w_vectors.shape[-1])
        df, df_w = pd.DataFrame(mean_y_hours_bandas), pd.DataFrame(w_vectors)
        # DEFAULT VALUES ARE: <class 'pandas.core.frame.DataFrame'> 300 <class 'pandas.core.series.Series'> (300, 3600)
        print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
              type(df_w), len(df_w), type(df_w[0]), df_w.shape
              )

        # ARRANGE TO A PANDAS DATAFRAME FOR MODELLING
        v_prefix = 'v'
        df.drop(columns=[v_prefix], inplace=True)
        # df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
        # DEFAULT VALUES ARE: (300, 60000)
        # print(df.head(3), df.shape)
        v_prefix = 'w'
        df_m = pd.concat([df, df_w.add_prefix(f'{v_prefix}_')], axis=1)
        # DEFAULT VALUES ARE: [3 rows x 3604 columns] (300, 3604)'
        # print(df_m.head(3), df_m.shape)
        print('####', 'EXTRACTED', 'MEAN', df_m.shape)

        # NOT NEEDED - REARRANGE THE SIGNAL
        # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
        print('#### TIMES #### w_this_sliding_window', round(time.time() - t1), 3)
    else:
        # <class 'numpy.ndarray'> 10
        # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
        # sum_y_rn_st_split to .csv
        df = pd.DataFrame(mean_y_hours_bandas)
        v_prefix = 'v'
        df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
        # (300, 60000)
        # print(df_v.head(3), df_v.shape)
        df.drop(columns=['v'], inplace=True)
        df_m = pd.concat([df, df_v], axis=1)
        print(df_m.head(3), df_m.shape)


    exit()

    df_m.to_pickle(
        f'{path_out}data/{file_in_pattern}{region}_{audio_file.split("/")[-1][:-4]}_dict_y_split_{si}_{job_id}_{int(time.time())}.pkl')

    df_m.to_csv(
        f'{path_out}data/{file_in_pattern}{region}_{audio_file.split("/")[-1][:-4]}_dict_y_split_{si}_{job_id}_{int(time.time())}.csv',
        sep=';')


t00 = time.time()
print('#### TIMES #### bootstrap_soundscape TOTAL TOTAL ==>>', time.time() - t00)

if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
    # TODO: extract extraction modules here
