import gc

gc.collect()

import os
import io
import re
import ast
import csv
import copy
import time
import random
import joblib
import datetime

from glob import glob
from pathlib import Path
from typing import List, Tuple

from joblib import Parallel, delayed
from joblib import effective_n_jobs
from itertools import cycle
from multiprocessing import Pool

import urllib.request

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

from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

random.seed("9103")

### IDENTIFY PATH ###
t0 = time.time()
cwd = os.getcwd()

### DEFINE ENVIRONMENT VARIABLES ###
# TODO: define all the environment variables
global job_id, N_JOBS, sreg, si, bandas, sr, b_band, u_band, bandwidth, samples_s, isamples_s, verbose, f_progress
global d_re, windows_13
# global audio_files, verbose, f_progress
job_id = os.environ.get('SLURM_JOB_ID') or 'NULL'
N_JOBS = int(effective_n_jobs(-1)) or -1  # os.environ.get('N_JOBS') or 4
sreg, si, bandas, windows_13, file_in_pattern = 0, 0, 10, True, '000_'

# SAMPLES_S / ISAMPLES_S => [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
b_band, u_band, bandwidth, samples_s, isamples_s, verbo = 0, 10000, 1000, 1800, 60, False
d_re = {'NaturalRegeneration': 0, 'Pasture': 1, 'Plantation': 2, 'RefForest': 3}
### LOAD AUDIO FILES ###
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)
path_data = f'{cwd}/data/'
path_out = f'{cwd}/out/'

configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
               for dirpath, dirnames, files in os.walk(path_data)
               for f in files if f.endswith('.mp3')]
print(path_data, path_out, configfiles[:3])

# READ audio_files to process
audio_files = pd.DataFrame.from_records(configfiles, columns=['region', 'filename']).astype(str)
audio_files = audio_files.assign(processed=False)
# audio_files['processed'] = audio_files['processed'].astype(bool)
audio_files.to_csv(f'{path_out}audio_files_in_{job_id}_{str(datetime.date.today())[:-3]}.csv', sep=';', index=True)


def update_progress(audio_files_in: pd.DataFrame = audio_files,
                    audio_files_processed: pd.DataFrame = None,  # audio_files_processed,
                    # audio_files_processed: pd.DataFrame = audio_files_processed,
                    c_processed: str = 'processed',
                    c_filename: str = 'filename') -> pd.DataFrame:
    if audio_files_processed is None:
        return audio_files
    # audios = [t[c_filename].split("/")[-1] for _, t in audio_files_processed.iterrows() if ast.literal_eval(t[c_processed])]
    audios = [t.filename.split("/")[-1] for _, t in audio_files_processed.iterrows() if eval(t.processed)]
    # audios = []
    # for _, t in audio_files_processed.iterrows():
    #     print(t.processed)
    #     # if eval(str(t.processed)):
    #     if eval(t.processed):
    #         audios.append(t.filename.split("/")[-1])
    # print(len(audios), audios[:3])
    if audios:
        audio_files_in[c_processed] = audio_files_in[c_filename].astype(str).str.contains(
            '|'.join(map(re.escape, audios)))
        # mask = [a.split('/')[-1] in terms for a in audio_files_in[c_filename]]
    # print(audio_files_in.head())
    return audio_files_in


# TODO: USE GLOBAL VARIABLE sreg TO SUBMIT PARALLEL JOBS VIA args in batch script
# uniqueRegions = list(set(r[0] for r in configfiles))
# region = uniqueRegions[sreg]  # for r in uniqueRegions
# audio_files = [f[1] for f in configfiles if f[0] == region]
# print(audio_files[:3])
# print(f'{region} =>', len(audio_files))


def bootstrap_soundscape(audio_file: str = '',
                         region: str = '',
                         si: int = 1964, *,
                         bandas: int = bandas,
                         bandwidth: int = bandwidth,
                         path_data: str = f'{cwd}/data/',
                         path_out: str = f'{cwd}/out/',
                         samples_s: int = samples_s,
                         isamples_s: int = isamples_s,
                         windows_13: bool = windows_13 or False,
                         verbose: bool = verbo
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
    print(f'y: {y_c[:9]}')
    print(f'total samples in y: {y_c.shape} in time: {tt} secs')
    print(f'samples rate per second:  {sr}')
    print('####', 'ORIGINAL Y_C', len(y_c), y_c[:9], type(y_c))
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

    # CHUNKS OF 60 SECONDS
    # exit()

    # WE DONT DENOISE THIS TIME
    # t1 = time.time()
    # parallel audio_denoise => ~30sec
    # split_y_rn_st = Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(audio_denoise_st)(y_i) for y_i in split_y)
    # split_y_rn_ns = Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(audio_denoise_ns)(y_i) for y_i in split_y)

    def split_freq_band(s: np.memmap = None,
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
    split_y_split_bands = np.array(
        Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y)
    )
    # split_y_rn_ns_split = np.array(
    #     Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(split_freq_band)(y_i) for y_i in split_y_rn_ns)
    # )

    print('####', 'PARALLEL SPLIT Y_C PER FREQUENCY',
          len(split_y_split_bands), len(split_y_split_bands[0]),
          len(split_y_split_bands[0][0]), split_y_split_bands[0][0][:9],
          time.time() - t1)

    # exit()

    # # REARRANGE THE SIGNAL
    # t1 = time.time()
    # # print('#### TIMES #### List comprehension')
    # y_rn_split_st_10 = [split_y_split_bands[:, b, :].flatten() for b in range(0, bandas)]
    # y_rn_split_ns_10 = [split_y_rn_ns_split[:, b, :].flatten() for b in range(0, bandas)]
    # print('#### TIMES #### flatten:', time.time() - t1)
    # REARRANGE THE SIGNAL
    t1 = time.time()
    # print('#### TIMES #### List comprehension')
    horas = 30
    y_split_hours_bands = [split_y_split_bands[h, b, :].flatten() for b in range(0, bandas) for h in range(0, horas)]

    mean_y_hours_bandas = [
        {'r': d_re.get(region, 0),
        # {'r': d_re.get(region.upper(), 0),
         'h': h,
         'b': b,
         'm': np.mean(f := split_y_split_bands[h, b, :].flatten()),
         'v': f
         } for b in range(0, bandas) for h in range(0, horas)
    ]
    print(len(mean_y_hours_bandas), mean_y_hours_bandas[0])
    # y_rn_split_ns_10 = [split_y_rn_ns_split[:, b, :].flatten() for b in range(0, bandas)]
    print('#### TIMES #### flatten:', time.time() - t1)
    print('####', 'REARRANGED THE SIGNAL PER 10 FREQUENCIES', type(y_split_hours_bands), len(y_split_hours_bands),
          type(y_split_hours_bands[0]), len(y_split_hours_bands[0]))

    # exit()

    print('####', f'COMPUTE SLIDING WINDOW {windows_13}')
    bandwidth, secs, w_size_mins = bandwidth, 60, 0.06  # 1000, 60, 0.06 => every 3.6 secs

    def w_this_sliding_window(data, size=int(bandwidth * secs * w_size_mins), stepsize=int(bandwidth * secs),
    # def sum_this_sliding_window(data, size=int(bandwidth * secs), stepsize=int(bandwidth * secs * w_size_mins),
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
    # mean_y_hours_bandas = copy.deepcopy(mean_y_hours_bandas)
    # dict_y_split_hours_bands = copy.deepcopy(data)

    # mean_y_hours_bandas = [dict(item, w=np.array(sum_this_sliding_window(item['v']))) for item in copy.deepcopy(data)]

    df_m = None

    if windows_13:
        w_vectors = np.array(
            Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(w_this_sliding_window)(s.get('v')) for s in mean_y_hours_bandas)
        )

        print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
              type(w_vectors), len(w_vectors),
              type(w_vectors[0]), len(w_vectors[0]), len(w_vectors[0][0]),
              )

        w_vectors = w_vectors.reshape(w_vectors.shape[0], w_vectors.shape[-1])
        df, df_w = pd.DataFrame(mean_y_hours_bandas), pd.DataFrame(w_vectors)

        # <class 'pandas.core.frame.DataFrame'> 300 <class 'pandas.core.series.Series'> (300, 3600)
        print('####', 'REARRANGED THE SIGNAL W SLIDING_WINDOW PER 10 FREQUENCIES',
              type(df_w), len(df_w), type(df_w[0]), df_w.shape
              )

        v_prefix = 'v'
        df.drop(columns=[v_prefix], inplace=True)
        # df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
        # (300, 60000)
        # print(df.head(3), df.shape)
        v_prefix = 'w'
        df_m = pd.concat([df, df_w.add_prefix(f'{v_prefix}_')], axis=1)
        # [3 rows x 3604 columns] (300, 3604)
        # print(df_m.head(3), df_m.shape)

        # NOT NEEDED - REARRANGE THE SIGNAL
        # sum_y_rn_st_split = [split_sum_y_rn_st_split[:, b, :].flatten() for b in range(0, bandas)]
        print('#### TIMES #### w_this_sliding_window', time.time() - t1)

        # print('####', 'REARRANGED THE SIGNAL SUM SLIDING_WINDOW PER 10 FREQUENCIES',
        #       type(mean_y_hours_bandas), len(mean_y_hours_bandas),
        #       type(mean_y_hours_bandas[0]), len(mean_y_hours_bandas[0]), len(mean_y_hours_bandas[0][0]),
        #       # len(mean_y_hours_bandas[0][0]), mean_y_hours_bandas[0][0][33:39], data[0].get('v')[33:39]
        #       )

        # print('####', 'REARRANGED THE SIGNAL SUM SLIDING_WINDOW PER 10 FREQUENCIES',
        #       type(mean_y_hours_bandas), len(mean_y_hours_bandas), type(mean_y_hours_bandas[0]), mean_y_hours_bandas[0].keys()
        #       # type(mean_y_hours_bandas[0]), len(mean_y_hours_bandas[0]), len(mean_y_hours_bandas[0][0]),
        #       # len(mean_y_hours_bandas[0][0]), mean_y_hours_bandas[0][0][33:39], data[0].get('v')[33:39]
        #       )
        # for k in mean_y_hours_bandas[0].keys():
        #     print(k, type(mean_y_hours_bandas[0].get(k)))

        # exit()

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
        # print(df_m.head(3), df_m.shape)

    df_m.to_csv(f'{path_out}data/{file_in_pattern}{region}_{audio_file.split("/")[-1][:-4]}_dict_y_split_{si}_{job_id}_{int(time.time())}.csv', sep=';')

    # exit()

t00 = time.time()
# READ audio_files_progress
global audio_files_processed  # = None
try:
    audio_files_processed = pd.read_csv(
        f'{path_out}audio_files_processed_{str(datetime.date.today())[:-3]}.csv', sep=';'
    ).astype(str)
    # audio_files_processed['processed'] = audio_files_processed['processed'].astype(bool)
except Exception as e:
    print('ALWAYS PROBLEMS', e)
    audio_files_processed = None

audio_files = update_progress(audio_files_in=audio_files, audio_files_processed=audio_files_processed)
f_progress = copy.deepcopy(audio_files)

outfile_name = f'f_progress_{job_id}_{int(time.time())}.txt'
print('#### #### HOI FOREST #### ####')
# Parallel(n_jobs=N_JOBS, verbose=verbose)(delayed(bootstrap_soundscape)(audio_file=a) for a in audio_files)

# This function is meant to be used in a parallel fashion
print(f'starting with => {len(audio_files)} soundscapes => now {int(time.time())}')

try:
    for i, f in audio_files.iterrows():
        if bool(f.processed):
            # print(f)
            continue
        t11 = time.time()
        print(i, '################', '################', '################', '################')
        print(i, '#### RUNNING ####', 'REGION:', '==>>', f.region, '<<==', 'AUDIO:', '==>>', f.filename.split('/')[-1])
        bootstrap_soundscape(audio_file=f.filename, region=f.region, si=i)
        print(i, '#### TIMES #### bootstrap_soundscape TOTAL TOTAL ==>>', time.time() - t11)
        f_progress.at[i, 'processed'] = True
        f_progress.to_csv(f'{path_out}audio_files_processed_{str(datetime.date.today())[:-3]}.csv', sep=';', index=True)
except Exception as e:
    print('ALWAYS PROBLEMS', e)
    raise
finally:
    print('SE ME CUIDA MIJO, AHÍ LE DEJO PA` QUE NO TRASNOCHE TANTO ;)')
    f_progress.to_csv(f'{path_out}audio_files_processed_{str(datetime.date.today())[:-3]}.csv', sep=';', index=True)

print('#### TIMES #### bootstrap_soundscape TOTAL TOTAL ==>>', time.time() - t00)

data = [
    'cali',
    'thornbury',
    'mexico',
    'wien',
    'porto',
    'lausanne',
    'rapperswil'
]

outfile_name = f'cities_{job_id}_{str(datetime.date.today())}.txt'
with open(path_out + outfile_name, 'w') as file:
    file.write('\n'.join(data) + '\n')
file.close()

print('#### #### NO DA MAS #### TERMINO #### FINITO #### ####')
print(
    '####', '####', '\n',
    f'TOTAL_TIME: {round(time.time() - t0, 3)}', '\n',
    f'TOTAL_JOBS: {effective_n_jobs(N_JOBS)}', '\n',
    f'JOB_ID: {job_id}', '\n',
    '####', '####'
)
print('####', f'SLEEPING TIME:', f'{round(time.time() - t0, 3)} seconds', '####')
print('All guert parcero, aller!')
