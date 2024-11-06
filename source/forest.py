# !/venvs/tpy310f_k/bin/python
# filename: forest.py
# -*- coding: utf-8 -*-

import gc

gc.collect()

import os
import io
import re
import ast
import csv
import sys
import copy
import time
import random
import joblib
import datetime

import argparse
from glob import glob
from pathlib import Path
from typing import List, Tuple

from joblib import Parallel, delayed
from joblib import effective_n_jobs

from enum import Enum


# from script_dea_region_parallel_00 import job_id


class Task(Enum):
    OBSERVATION = 0
    VISUALISATION = 1
    EXTRACTION = 2
    AUGMENTATION = 3
    SAMPLING = 4
    MODELLING = 5
    CLASSIFICATION = 6


random.seed("9103")
t0 = time.time()

### IDENTIFY PATH ###
cwd = os.getcwd()
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)

### CONSTANTS ###
TASKS = [Task.OBSERVATION]


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
        "-vb",
        "--verbo",
        type=bool,
        default=False,
        required=False,
        help="Verbose",
    )

    return parser.parse_args()


def main():
    ### DEFINE ENVIRONMENT VARIABLES ###
    # TODO: define all the environment variables
    global TASKS, job_id, N_JOBS, path_data, path_out, verbo
    global sreg, si, bandas, sr, b_band, u_band, bandwidth, samples_s, isamples_s

    job_id = os.environ.get('SLURM_JOB_ID') or 'NULL'
    N_JOBS = int(effective_n_jobs(-1)) or -1  # os.environ.get('N_JOBS') or 4

    args = get_args()
    tasks = args.tasks
    path_data, path_out, sreg, si = args.path_in, args.path_out, args.sregions, args.sregion_iterator
    bandas, b_band, u_band, bandwidth = args.b_band, args.u_band, args.bandwidth, args.bandas
    samples_s, isamples_s, verbo = args.samples_second, args.isamples_second, args.verbo


























if __name__ == '__main__':
    print('Hablámelo maniño, alles gut oder was??')
    main()
    print('All gürkel parcero, aller!')
