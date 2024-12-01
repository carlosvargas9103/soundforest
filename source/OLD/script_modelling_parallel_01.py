import gc

gc.collect()

import os
import io
import re
import ast
import csv
import copy
import time
import json
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

# >>>> import libraries for CNN >>>>
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax
# <<< import libraries for CNN <<<<

from warnings import simplefilter

# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

random.seed("9103")

### IDENTIFY PATH ###
t0 = time.time()
cwd = os.getcwd()

####################################
### DEFINE ENVIRONMENT VARIABLES ###

# TODO: define all the environment variables
global job_id, N_JOBS, sreg, si, bandas, sr, b_band, u_band, bandwidth, samples_s, isamples_s, verbose, f_progress
global d_re, windows_13, N_MODELS
# global audio_files, verbose, f_progress
job_id = os.environ.get('SLURM_JOB_ID') or 'NULL'
N_JOBS = int(effective_n_jobs(-1)) or -1  # os.environ.get('N_JOBS') or 4
sreg, si, bandas, windows_13, file_in_pattern = 0, 0, 10, False, '000_'

N_MODELS = 0
windows_13 = True
num_epochs = 11

file_in_pattern = '000_' if windows_13 else '00_'

# SAMPLES_S / ISAMPLES_S => [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
b_band, u_band, bandwidth, samples_s, isamples_s, verbo = 0, 10000, 1000, 1800, 60, False
d_re = {'NaturalRegeneration': 0, 'Pasture': 1, 'Plantation': 2, 'RefForest': 3}

### LOAD AUDIO FILES ###
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)
# path_data = f'{cwd}/data/'
path_out = f'{cwd}/out/'
path_data = f'{cwd}/out/data'

configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
               for dirpath, dirnames, files in os.walk(path_data)
               for f in files if f.startswith(file_in_pattern)]

print(path_data, path_out, len(configfiles))  # , configfiles[:3])

# READ audio_files to process

audio_files = pd.DataFrame.from_records(configfiles, columns=['region', 'filename']).astype(str)
audio_files = audio_files.assign(processed=False)
# audio_files['processed'] = audio_files['processed'].astype(bool)

audio_files.to_csv(f'{path_out}{file_in_pattern}preprocessed_data_in_{job_id}_{str(datetime.date.today())[:-3]}.csv',
                   sep=';', index=True)

### END ENVIRONMENT VARIABLES ###
#################################


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


def merge_data(data_files: pd.DataFrame = None) -> pd.DataFrame:
    data_merged = None
    ncols = 3005
    with open(data_files.filename[0]) as x:
        ncols = len(x.readline().split(';'))
    try:
        df_merged = pd.concat((pd.read_csv(f, sep=';', usecols=range(1, ncols)) for f in data_files.filename),
                              ignore_index=True)
        # print(df_merged.shape)
        # print(df_merged.head(3))
        return df_merged
    except Exception as e:
        print('ALWAYS PROBLEMS', e)
    return data_merged


# model 1: CNN + LSTM
# model 2: LSTM + CNN
# model 3: CNN LSTM parallel

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # flatten
            nn.Flatten(),
            nn.LazyLinear(out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)
        out = self.cnn(out)
        return out


# model 3: CNN LSTM parallel
class ParallelCNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ParallelCNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_lstm = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # cnn takes input of shape (batch_size, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)
        out_cnn = self.cnn(x_cnn)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out_lstm, _ = self.lstm(x)
        out_lstm = self.fc_lstm(out_lstm[:, -1, :])
        out = torch.cat([out_cnn, out_lstm], dim=1)
        out = self.fc(out)
        return out


def train(models: List,
          train_loader: DataLoader,
          epochs: int = 1,
          t0: int = time.time(),
          verbose: bool = verbo or False,
          model_path: str = f'{path_out}models/{file_in_pattern}{job_id}_{str(datetime.date.today())[:-3]}'
          ) -> None:
    criterion = nn.CrossEntropyLoss()
    for model in models:
        t1 = time.time()
        print('####',
              'TRAINING MODEL',
              model.__class__.__name__,
              '####')
        model.train()
        optimizer = Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                if verbose:
                    if (i + 1) % 33 == 0:
                        print(f'Epoch [{epoch + 1}/{epochs}]',
                              f'Step [{i + 1}/{len(train_loader)}]',
                              f'Loss: {loss.item():.4f}',
                              f'Time: {round(time.time() - t1, 3)}')
                else:
                    if ((epoch + 1) % 10 == 0) and ((i + 1) % 33 == 0):
                        print('####', 'prediction', y_pred.shape)
                        print(f'Epoch [{epoch + 1}/{epochs}]',
                              f'Step [{i + 1}/{len(train_loader)}]',
                              f'Loss: {loss.item():.4f}',
                              f'Time: {round(time.time() - t1, 3)}')

        torch.save(model, f'{model_path}_{model.__class__.__name__}.model')
        print('####',
              'TRAINED MODEL',
              model.__class__.__name__,
              'training time:', round(time.time() - t1, 3),
              'total time:', round(time.time() - t0, 3),
              '####')


# test
def test(models, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        accuracy_dict = {}
        for model in models:
            model.eval()
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'Accuracy of the {model.__class__.__name__} model on the test set: {100 * correct / total:.2f} %')
            accuracy_dict[model.__class__.__name__] = 100 * correct / total
    return accuracy_dict


### MAIN ###

print('#### #### HOI FOREST #### ####')

t00 = time.time()

# READ audio_files_progress
global audio_files_processed  # = None
t0 = time.time()
try:
    audio_files_processed = pd.read_csv(
        f'{path_out}{file_in_pattern}processed_data_in_{str(datetime.date.today())[:-3]}.csv', sep=';'
    ).astype(str)
except Exception as e:
    print('ALWAYS PROBLEMS', e)
    audio_files_processed = None

audio_files = update_progress(audio_files_in=audio_files, audio_files_processed=audio_files_processed)
# print(audio_files[:3])
f_progress = copy.deepcopy(audio_files)
df_data = merge_data(audio_files)

print('####', 'MERGE', len(audio_files),
      'merge_data Dataframe shape:', df_data.shape,
      '####', 'time:', int(time.time() - t0))

t0 = time.time()
X_train, X_test, y_train, y_test = train_test_split(df_data.iloc[:, 1:], df_data.iloc[:, 0], test_size=0.2,
                                                    random_state=9103)
# normalize the data
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
# Convert the data to torch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train.values).long()
y_test = torch.from_numpy(y_test.values).long()

# Datasets
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
# Dataloaders

batch_s = 64# if windows_13 else 128
#batch_s = 64 if windows_13 else 128
train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('####', 'SPLIT', 'train_test_split and CUDA:', '???', device, torch.cuda.is_available(), '####', 'time:', int(time.time() - t0))

t0 = time.time()

# SOME CONFIG ####
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

#### PREPARE FOR FITTING THE MODEL ####
input_size = X_train.shape[-1]
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y_train))
num_classes = num_classes if num_classes >= 4 else num_classes + 1
print('####', 'CLASSES:', np.unique(y_train), 'TOTAL', num_classes)

#### MODELS ####
cnn_lstm = CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
lstm_cnn = LSTM_CNN(input_size, hidden_size, num_layers, num_classes).to(device)
cnn_lstm_parallel = ParallelCNNLSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

dict_models = {
    0: [cnn_lstm],
    1: [cnn_lstm, lstm_cnn],
    2: [cnn_lstm, lstm_cnn, cnn_lstm_parallel],
}
models = dict_models.get(N_MODELS, [cnn_lstm])

#### TRAIN ####
print('####', 'MODELS', dict_models, '####')
print('####', 'MODELS - TOTAL', len(models), '####')
print('####', 'EPOCHS', num_epochs, '####')

tt0 = time.time()
train(models, train_loader, epochs=num_epochs)
print('####', 'TRAINING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')


tt0 = time.time()
accuracy_dict = test(models, test_loader)
print('####', 'TESTING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

# plot bar chart with the accuracy of each model
# sns.barplot(x=list(accuracy_dict.keys()), y=list(accuracy_dict.values()))

with open(f'{path_out}000_models_accuracy_dict_{job_id}_{str(datetime.date.today())}.json', 'w') as fp:
    json.dump(accuracy_dict, fp, sort_keys=True, indent=4)

print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')

outfile_name = f'f_progress_{job_id}_{int(time.time())}.txt'

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
