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
                         secs_b: int = 6,
                         secs_o: int = 1.9,
                         hanning: bool = True,
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
    print('NUMBERS', sr, isamples_s, samples_s, secs_b)
    my_chunks = samples_s / secs_b
    split_y = np.hsplit(y_c, my_chunks)
    print('####', 'CHUNKS Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size: {len(split_y[0])}',
          type(split_y), type(split_y[0])
          )

    frame_size, hop_size = sr * secs_b, int(sr * (secs_b - secs_o))

    def hanning(y: np.array = None, frame_size: int = frame_size, hop_size: int = hop_size, hanning: bool = True):
        signal = np.array(y)
        # num_frames = 1 + (len(signal) - frame_size) // hop_size
        # print(num_frames) # 449
        if hanning:
            hanning_window = np.hanning(frame_size)
            return [signal[i:i + frame_size] * hanning_window for i in range(0, len(signal) - frame_size + 1, hop_size)]
        return [signal[i:(i + frame_size)] for i in range(0, len(signal) - frame_size + 1, hop_size)]

    split_y = hanning(y_c, frame_size, hop_size, hanning)

    print('####', 'HANNING Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size first: {len(split_y[0])}',
          f'chunk_size last: {len(split_y[-1])}',
          type(split_y), type(split_y[0])
          )
    # print('####', 'HANNING', '\n',
    #       f'hanning => max: {max(split_y[0])} min: {min(split_y[0])} mean: {(split_y[0].mean())} size: {len(split_y[250])}', '\n',
    #       f'hanning => max: {max(split_y[250])} min: {min(split_y[250])} mean: {split_y[250].mean()} size: {len(split_y[250])}'
    #       )
    # exit()

    # ATM, WE DO NOT CALL THESE METHODS
    # wrap methods audio_denoise with parameters
    # def audio_denoise_st(y=None, sr: int = 48000):
    def audio_denoise_st(y=None, sr: int = sr):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)

    # def audio_denoise_ns(y=None, sr: int = 48000):
    def audio_denoise_ns(y=None, sr: int = sr):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)

    denoise = False
    if denoise:
        # PARALLEL split_freq_band => ~6sec
        print('####', 'PARALLEL audio_denoise')
        t1 = time.time()
        split_y_st = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_st)(y_i) for y_i in split_y)
        )
        split_y_ns = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_ns)(y_i) for y_i in split_y)
        )
        # 438 288000 [2.19772187e-26 1.05804673e-26 5.78876953e-27] 18.265
        print('####', 'PARALLEL DENOISE Y_C', len(split_y_st), len(split_y_st[0]), split_y_st[0][:3], round(time.time() - t1, 3))

    # exit()

    # print('####', f'COMPUTE SLIDING WINDOW {windows_13}')

    # # DEFAULT VALUES ARE: bandwidth, secs_b, w_size_mins = bandwidth, 60, 0.06  # 1000, 60, 0.06 => every 3.6 secs
    # def w_this_sliding_window(data: np.ndarray = None, size=int(bandwidth * secs_b * w_size_mins),
    #                           stepsize=int(bandwidth * secs_b),
    #                           padded=False, axis=-1, copy=False, suma=False, average=False) -> np.ndarray:
    #     if axis >= data.ndim:
    #         raise ValueError('Axis value out of range')
    #     if stepsize < 1:
    #         raise ValueError('Stepsize may not be zero or negative')
    #     if size > data.shape[axis]:
    #         raise ValueError('Sliding window size may not exceed size of selected axis')
    #     shape = list(data.shape)
    #     # print(data.shape[axis] / stepsize - size / stepsize + 1)
    #     shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    #     shape.append(size)
    #     strides = list(data.strides)
    #     strides[axis] *= stepsize
    #     strides.append(data.strides[axis])
    #     # TODO: Maybe with dask??
    #     strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    #     if suma:
    #         return [np.sum(s, axis=0) for s in strided.copy()] if copy else [np.sum(s, axis=0) for s in strided]
    #     elif average:
    #         return [np.mean(s, axis=0) for s in strided.copy()] if copy else [np.mean(s, axis=0) for s in strided]
    #     else:
    #         return strided.copy() if copy else strided

    # # SLIDING WINDOWS FOR THE SIGNAL
    # t1 = time.time()
    # print('#### RUNNING #### sum_this_sliding_window')
    # df_m = None
    # if windows_13:
    #     w_vectors = np.array(
    #         Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(w_this_sliding_window)(s.get('v')) for s in mean_y_hours_bandas)
    #     )
    #     print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
    #           type(w_vectors), len(w_vectors),
    #           type(w_vectors[0]), len(w_vectors[0]), len(w_vectors[0][0]),
    #           )
    #     w_vectors = w_vectors.reshape(w_vectors.shape[0], w_vectors.shape[-1])
    #     df, df_w = pd.DataFrame(mean_y_hours_bandas), pd.DataFrame(w_vectors)
    #     # DEFAULT VALUES ARE: <class 'pandas.core.frame.DataFrame'> 300 <class 'pandas.core.series.Series'> (300, 3600)
    #     print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
    #           type(df_w), len(df_w), type(df_w[0]), df_w.shape
    #           )
    #
    #     # ARRANGE TO A PANDAS DATAFRAME FOR MODELLING
    #     v_prefix = 'v'
    #     df.drop(columns=[v_prefix], inplace=True)
    #     # df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
    #     # DEFAULT VALUES ARE: (300, 60000)
    #     # print(df.head(3), df.shape)
    #     v_prefix = 'w'
    #     df_m = pd.concat([df, df_w.add_prefix(f'{v_prefix}_')], axis=1)
    #     # DEFAULT VALUES ARE: [3 rows x 3604 columns] (300, 3604)'
    #     # print(df_m.head(3), df_m.shape)
    #     print('####', 'EXTRACTED', 'MEAN', df_m.shape)
    #
    #     # NOT NEEDED - REARRANGE THE SIGNAL
    #     # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
    #     print('#### TIMES #### w_this_sliding_window', round(time.time() - t1), 3)
    # else:
    #     # <class 'numpy.ndarray'> 10
    #     # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
    #     # sum_y_rn_st_split to .csv
    #     df = pd.DataFrame(mean_y_hours_bandas)
    #     v_prefix = 'v'
    #     df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
    #     # (300, 60000)
    #     # print(df_v.head(3), df_v.shape)
    #     df.drop(columns=['v'], inplace=True)
    #     df_m = pd.concat([df, df_v], axis=1)
    #     print(df_m.head(3), df_m.shape)


    def split_freq_band_per_frame(s: np.memmap = None,
                                  sr: int = sr,
                                  b_band: int = b_band,
                                  u_band: int = u_band,
                                  bandwidth: int = bandwidth) -> List[np.ndarray]:
        # Calculate the FFT
        # 2880000 48000 2.0833333333333333e-05 10000 0 1000 => 60 seconds
        # 288000 48000 2.0833333333333333e-05 10000 0 1000 => 6 seconds
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
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band_per_frame)(y_i) for y_i in split_y)
    )

    print('####', 'PARALLEL SPLIT Y_C PER FREQUENCY',
          len(split_y_split_bands), len(split_y_split_bands[0]),
          len(split_y_split_bands[0][0]), split_y_split_bands[0][0][:3],
          round(time.time() - t1, 3)
          )

    t1 = time.time()

    # # y_split_hours_bands = [split_y_split_bands[h, b, :].flatten() for b in range(0, bandas) for h in range(0, horas)]
    # y_split_seconds_bands = [split_y_split_bands[s, b, :].flatten() for b in range(0, bandas) for s in range(0, secs_b)]
    # # exit()
    #
    # print('####', 'REARRANGED THE SIGNAL PER (bands = 10) FREQUENCIES', type(y_split_seconds_bands), len(y_split_seconds_bands),
    #       type(y_split_seconds_bands[0]), len(y_split_seconds_bands[0]))

    # exit()

    # ARRANGE in dict per BANDAS, and HORAS
    mean_y_seconds_bandas = [
        {'reg': d_re.get(region, 0),  # CLASS (INT) 4
         'sid': si,  # audio_file_id
         'sec': s,  # TIME (int) 0 - 6 => DONT NEED!?? - SUNDAY (05.01.25)!
         'ban': b,  # BAND (int) 0 - 9  => TODO: Consider 10-bands at once - SUNDAY (05.01.25)!
         'men': np.mean(f := split_y_split_bands[s, b, :].flatten()),  # MEAN of the VECTOR (float)
         'med': np.median(f),  # MEDIAN of the VECTOR (float)
         'sum': np.sum(f),  # SUM of the VECTOR (float)
         'max': np.max(f),  # MAX of the VECTOR (float)
         'min': np.min(f),  # MIN of the VECTOR (float)
         'vec': f  # VECTOR (npArray[float]) => 6000 => 4380
         } for b in range(0, bandas) for s in range(0, secs_b)
    ]

    # print('####', 'DICT_VECTOR',
    #       len(mean_y_seconds_bandas), list(mean_y_seconds_bandas[0].keys()),
    #       mean_y_seconds_bandas[0],
    #       # mean_y_seconds_bandas[0].get("vec", [])[:3]
    #       )
    print('####', 'TIMES', '####', 'DICT_VECTOR:', round(time.time() - t1, 3))

    # exit()

    # <class 'numpy.ndarray'> 10
    # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
    df = pd.DataFrame(mean_y_seconds_bandas)
    v_prefix = 'vec'
    df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
    # (60, 6009)
    # print(df_v.head(3), df_v.shape)
    df.drop(columns=[v_prefix], inplace=True)
    df_m = pd.concat([df, df_v], axis=1)

    # exit()

    # TODO: Pipeline (12-24.12.24):
    # TODO: MODELLING - DONE!
    # TODO: Continuing with the pre-processing - in-progress - SATURDAY (04-05.01.25)!
    #   3. Compute the mean, medium, max, min, distance, etc.. - DONE!
    #   3.6. Compute the BIO-ACOUSTIC indexes, etc.. - IN-PROGRESS
    #   4. Transform the data => filters, envelope, pitch, etc.. - SATURDAY (04-05.01.25)!
    #   4.1. These transformations need to be included here in the extraction module - SATURDAY (04-05.01.25)!
    # TODO: Activation function (Sigmoid) - SATURDAY (03.01.25)!
    # #### # ####
    # TODO: Extract the Benchmark from Giacomo - SUNDAY (02.01.25)!
    #   6. PLOTS the distribution or each frequency against a metric per region - SUNDAY (02.01.25)!
    # TODO: Evaluation Metrics for classification => Table & Matrix - THURSDAY (02.01.25)!
    # TODO: Reduce the time of the samples - DONE!
    # TODO: Next meeting => 08.01.2025.
    # TODO: Methodology PDFs FOLDER on Git?


    # TODO: Pipeline (12-24-07.01.25):
    #       0. Frame per 6 secs with 1.9 secs overlapping, make sure the vectors have all the same size. - DONE
    #       1. Apply Hanning window to smooth the frame. - DONE
    #       1.5 Denoise - DONE (no used)
    #       2. Split per frequency band. - DONE
    #       3. Compute the mean, medium, max, min, distance, etc.. - DONE
    #       4. Transform the data => filters, envelope, pitch, etc.. - 3h
    #       6. Plot the distribution or each frequency against a metric per region. - 2h
    #       7. Save the plots.. - 1h


    df_m.to_pickle(
        f'{path_out}data/{f_pattern_out}/{region}/{audio_file.split("/")[-1][:-4]}_dict_y_split_{si}_{job_id}_{int(time.time())}.pkl')

    # df_m.to_csv(
    #     f'{path_out}data/{f_pattern_out}/{region}/{audio_file.split("/")[-1][:-4]}_dict_y_split_{si}_{job_id}_{int(time.time())}.csv',
    #     sep=';')

    # exit()


t00 = time.time()
print('#### TIMES #### extraction TOTAL TOTAL ==>>', round(time.time() - t00, 3))

if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')