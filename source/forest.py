# !/venvs/tpy310f_k/bin/python
# filename: forest.py
# -*- coding: utf-8 -*-

import gc

gc.collect()

import os
import re
import csv
import copy
import time
import random
import joblib
import datetime

import argparse
from glob import glob
from pathlib import Path
from typing import List, Tuple

from joblib import effective_n_jobs

import pandas as pd
import numpy as np

from enum import Enum

from observation import process_soundscape
from visualisation import visualise_soundscape, visualise_distribution
from extraction import bootstrap_soundscape


class Task(Enum):  # These are each of the tasks ( modules | files )
    # FRAMEWORK = 0
    OBSERVATION = 1
    VISUALISATION = 2
    EXTRACTION = 3
    SAMPLING = 4
    AUGMENTATION = 5
    MODELLING = 6
    CLASSIFICATION = 7


class FILE_PATTERN(Enum):
    sum_y_c_split = 'sum_y_c_split'
    sum_y_rn_st_split = 'sum_y_rn_st_split'
    sum_y_rn_ns_split = 'sum_y_rn_ns_split'


random.seed("9103")
t0 = time.time()

### IDENTIFY PATH ###
cwd = os.getcwd()
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)

### CONSTANTS ###
TASKS = [Task.VISUALISATION]
TASKS = [Task.OBSERVATION]
TASKS = [Task.EXTRACTION]
# TASKS = [Task.OBSERVATION, Task.VISUALISATION, Task.EXTRACTION]


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-ts",
        "--tasks",
        type=List,
        default=TASKS,
        required=False,
        help="Path of the input data",
    )

    parser.add_argument(
        "-p-in",
        "--path-in",
        type=str,
        default=f'{cwd}/data/',
        required=False,
        help="Path of the input data",
    )

    parser.add_argument(
        "-p-out",
        "--path-out",
        type=str,
        default=f'{cwd}/out/',
        required=False,
        help="Path of the output data",
    )

    # TODO: USE GLOBAL VARIABLE sreg TO SUBMIT PARALLEL JOBS VIA args in batch script
    parser.add_argument(
        "-sreg",
        "--sregions",
        type=int,
        default=0,
        required=False,
        help="Number of regions",
    )

    parser.add_argument(
        "-si",
        "--sregion-iterator",
        type=int,
        default=0,
        required=False,
        help="Iterator for regions id",
    )

    parser.add_argument(
        "-sr",
        "--sample-r",
        type=int,
        default=48000,
        required=False,
        help="Sample rate of the original audio files",
    )

    parser.add_argument(
        "-bs",
        "--bandas",
        type=int,
        default=10,
        required=False,
        help="Total bandas to play",
    )

    parser.add_argument(
        "-b-b",
        "--b-band",
        type=int,
        default=0,
        required=False,
        help="Base band in Hertz",
    )

    parser.add_argument(
        "-u-b",
        "--u-band",
        type=int,
        default=10000,
        required=False,
        help="Upper band in Hertz",
    )

    parser.add_argument(
        "-bw",
        "--bandwidth",
        type=int,
        default=1000,
        required=False,
        help="Bandwidth in Hertz",
    )
    # SAMPLES_S / ISAMPLES_S => [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
    parser.add_argument(
        "-s-s",
        "--samples-second",
        type=int,
        default=1800,
        required=False,
        help="Samples per second",
    )

    parser.add_argument(
        "-is-s",
        "--isamples-second",
        type=int,
        default=3,
        required=False,
        help="Inverse of samples per second",
    )
    parser.add_argument(
        "-s-b",
        "--seconds-bandwidth",
        type=int,
        # default=60,
        default=6,
        required=False,
        help="Seconds sampled per bandwidth",
    )

    parser.add_argument(
        "-ws-m",
        "--win-size-mins",
        type=float,
        default=0.06,
        required=False,
        help="Windows size per minutes",
    )

    parser.add_argument(
        "-w13",
        "--windows13",
        type=bool,
        default=True,
        required=False,
        help="Dimensionality reduction with windows per 3.1 seconds",
    )

    parser.add_argument(
        "-vb",
        "--verbo",
        type=bool,
        default=False,
        required=False,
        help="Verbose",
    )

    return parser.parse_args()


def update_progress(audio_files_in: pd.DataFrame = None,
                    audio_files_processed: pd.DataFrame = None,
                    c_processed: str = 'processed',
                    c_filename: str = 'filename'
                    ) -> pd.DataFrame:
    if audio_files_processed is None:
        return audio_files_in
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


def main():
    ### DEFINE ENVIRONMENT VARIABLES ###
    # TODO: define all the environment variables
    # relevant for observation, extraction, and sampling
    global TASKS, job_id, N_JOBS, path_data, path_out, verbo, windows_13
    global sreg, si, bandas, sr, b_band, u_band, bandwidth, samples_s, isamples_s, secs_b, w_size_mins
    global audio_files_processed, outfile_name
    # relevant for modelling
    global file_in_pattern

    # ASSIGN environment variables
    job_id = os.environ.get('SLURM_JOB_ID') or 'NULL'
    N_JOBS = int(effective_n_jobs(-1)) or -1  # os.environ.get('N_JOBS') or 4

    args = get_args()
    tasks = args.tasks
    path_data, path_out, sreg, si = args.path_in, args.path_out, args.sregions, args.sregion_iterator
    bandas, sr, b_band, u_band, bandwidth = args.bandas, args.sample_r, args.b_band, args.u_band, args.bandwidth
    samples_s, isamples_s, secs_b = args.samples_second, args.isamples_second, args.seconds_bandwidth
    w_size_mins, verbo, windows_13 = args.win_size_mins, args.verbo, args.windows13

    print('#### #### HOI FOREST #### ####')
    for task in tasks:
        match task:
            case Task.OBSERVATION:
                t00 = time.time()
                folders_in, f_pattern_out, f_ext_in = 'files_in', 'observation', ''
                # READ audio_files to process
                configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
                               for dirpath, dirnames, files in os.walk(path_data)
                               for f in files if f.endswith('.mp3')]
                print(path_data, path_out, configfiles[:3])
                audio_files = pd.DataFrame.from_records(configfiles, columns=['region', 'filename']).astype(str)
                audio_files = audio_files.assign(processed=False)
                audio_files.to_csv(f'{path_out}audio_{folders_in}_{job_id}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                   index=True)
                try:
                    audio_files_processed = pd.read_csv(
                        f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';').astype(str)
                except Exception as e:
                    print('ALWAYS PROBLEMS => Dónde están los Kuschifiles?', e)
                    audio_files_processed = None
                audio_files = update_progress(audio_files_in=audio_files, audio_files_processed=audio_files_processed)
                f_progress = copy.deepcopy(audio_files)
                outfile_name = f'f_progress_{job_id}_{int(time.time())}.txt'
                # PREPROCESS the soundscapes
                # TODO: This function is meant to be used in a parallel fashion
                print(f'starting with => {len(audio_files)} soundscapes => now {int(time.time())}')
                try:
                    for i, f in audio_files.iterrows():
                        if bool(f.processed):
                            # print(f)
                            continue
                        t11 = time.time()
                        print(f'{i}/{len(audio_files)}', '########', '################', '################', '########')
                        print(i, '#### OBSERVATION ####', 'REGION:', '==>>', f.region, '<<==', 'AUDIO', '==>>',
                              f.filename.split('/')[-1])

                        process_soundscape(audio_file=f.filename, region=f.region, si=i,
                                           bandas=bandas, b_band=b_band, u_band=u_band, bandwidth=bandwidth,
                                           path_data=path_data, path_out=path_out,
                                           samples_s=samples_s, isamples_s=isamples_s,
                                           secs_b=secs_b, w_size_mins=w_size_mins,
                                           verbose=verbo, n_jobs=N_JOBS, job_id=job_id)
                        print(i, '#### TIMES #### observation #### PARTIAL FILE ==>>', round(time.time() - t11, 3), 'seconds')
                        f_progress.at[i, 'processed'] = True
                        f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                          index=True)
                        break
                except Exception as e:
                    print('ALWAYS PROBLEMS', e)
                    raise
                finally:
                    print('SE ME CUIDA MIJO, AHÍ LE DEJO PA` QUE NO TRASNOCHE TANTO ;)')
                    f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                      index=True)
                print('#### TIMES #### observation TOTAL TOTAL ==>>', time.time() - t00)

            case Task.VISUALISATION:
                t00 = time.time()
                # LOAD preprocessed n-dimensional arrays to plot
                folders_in, f_pattern_out, f_ext_in = 'data/observation/', 'visualisation', '.npy'
                path_data = f'{path_out}{folders_in}'
                configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
                               for dirpath, dirnames, files in os.walk(path_data)
                               for f in files if f.endswith(f_ext_in)]
                # print(path_data, path_out, configfiles[:3])
                audio_files = pd.DataFrame.from_records(configfiles, columns=['region', 'filename']).astype(str)
                audio_files = audio_files.assign(processed=False)
                audio_files.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                   index=True)
                try:
                    audio_files_processed = pd.read_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv',
                                                        sep=';').astype(str)
                except Exception as e:
                    print('ALWAYS PROBLEMS => Dónde están los Kuschifiles?', e)
                    audio_files_processed = None
                audio_files = update_progress(audio_files_in=audio_files, audio_files_processed=audio_files_processed)
                f_progress = copy.deepcopy(audio_files)
                outfile_name = f'f_visual_{job_id}_{int(time.time())}.txt'
                # PREPROCESS the soundscapes
                # TODO: This function is meant to be used in a parallel fashion!!!
                print(f'starting with => {len(audio_files)} soundscapes => now {int(time.time())}')
                try:
                    for i, f in audio_files.iterrows():
                        if bool(f.processed):
                            # print(f)
                            continue
                        t11 = time.time()
                        print(f'{i}/{len(audio_files)}', '########', '################', '################', '########')
                        print(i, '#### VISUALISATION ####', 'REGION:', '==>>', f.region, '<<==', 'DATA', '==>>',
                              f.filename.split('/')[-1])
                        visualise_soundscape(audio_file=f.filename, region=f.region, si=i, sr=sr,
                                             bandas=bandas, b_band=b_band, u_band=u_band, bandwidth=bandwidth,
                                             path_data=path_data, path_out=path_out,
                                             samples_s=samples_s, isamples_s=isamples_s,
                                             secs_b=secs_b, w_size_mins=w_size_mins,
                                             verbose=verbo, n_jobs=N_JOBS, job_id=job_id,
                                             f_pattern_out=f_pattern_out)
                        print(i, '#### TIMES #### observation #### PARTIAL FILE ==>>', round(time.time() - t11, 3), 'seconds')
                        f_progress.at[i, 'processed'] = True
                        f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                          index=True)
                        break
                    # exit()
                except Exception as e:
                    print('ALWAYS PROBLEMS', e)
                    raise
                finally:
                    print('SE ME CUIDA MIJO, AHÍ LE DEJO PA` QUE NO TRASNOCHE TANTO ;)')
                    f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                      index=True)
                print('#### TIMES #### visualisation TOTAL TOTAL ==>>', time.time() - t00)

            case Task.EXTRACTION:
                t00 = time.time()
                path_data = args.path_in
                folders_in, f_pattern_out, f_ext_in = 'files_in', 'extraction', ''
                # READ audio_files to process
                configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
                               for dirpath, dirnames, files in os.walk(path_data)
                               for f in files if f.endswith('.mp3')]
                print(path_data, path_out, configfiles[:3])
                audio_files = pd.DataFrame.from_records(configfiles, columns=['region', 'filename']).astype(str)
                audio_files = audio_files.assign(processed=False)
                audio_files.to_csv(f'{path_out}audio_{folders_in}_{job_id}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                   index=True)
                try:
                    audio_files_processed = pd.read_csv(
                        f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';').astype(str)
                except Exception as e:
                    print('ALWAYS PROBLEMS => Dónde están los Kuschifiles?', e)
                    audio_files_processed = None
                audio_files = update_progress(audio_files_in=audio_files, audio_files_processed=audio_files_processed)
                f_progress = copy.deepcopy(audio_files)
                outfile_name = f'f_progress_{job_id}_{int(time.time())}.txt'
                # PREPROCESS the soundscapes
                # TODO: This function is meant to be used in a parallel fashion
                print(f'starting with => {len(audio_files)} soundscapes => now {int(time.time())}')
                try:
                    for i, f in audio_files.iterrows():
                        if bool(f.processed):
                            # print(f)
                            continue
                        t11 = time.time()
                        print(f'{i}/{len(audio_files)}', '########', '################', '################', '########')
                        print(i, '#### EXTRACTION ####', 'REGION:', '==>>', f.region, '<<==', 'DATA', '==>>',
                              f.filename.split('/')[-1])
                        bootstrap_soundscape(audio_file=f.filename, region=f.region, si=i, sr=sr,
                                             bandas=bandas, b_band=b_band, u_band=u_band, bandwidth=bandwidth,
                                             path_data=path_data, path_out=path_out,
                                             samples_s=samples_s, isamples_s=isamples_s,
                                             secs_b=secs_b, w_size_mins=w_size_mins,
                                             verbose=verbo, n_jobs=N_JOBS, job_id=job_id,
                                             f_pattern_out=f_pattern_out, windows_13=windows_13, horas = 30)
                        print(i, '#### TIMES #### observation #### PARTIAL FILE ==>>', round(time.time() - t11, 3), 'seconds')
                        f_progress.at[i, 'processed'] = True
                        f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                          index=True)
                        break
                    exit()
                except Exception as e:
                    print('ALWAYS PROBLEMS', e)
                    raise
                finally:
                    print('SE ME CUIDA MIJO, AHÍ LE DEJO PA` QUE NO TRASNOCHE TANTO ;)')
                    f_progress.to_csv(f'{path_out}audio_{f_pattern_out}_{str(datetime.date.today())[:-3]}.csv', sep=';',
                                      index=True)
                print('#### TIMES #### visualisation TOTAL TOTAL ==>>', time.time() - t00)


if __name__ == '__main__':
    print('Hablámelo maniño, alles gut oder was??')

    main()

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
    print('All gürkel parcero, aller!')
