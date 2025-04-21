import gc

gc.collect()

import os
import io
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

import pandas as pd
import numpy as np
import seaborn as sns

from extraction import Metrics as M

# >>>> import libraries for CNN >>>>
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax

# <<< import libraries for CNN <<<<
# primary source: https://github.com/mijanr/TimeSeries/blob/master/Time_Series_Classification/cnn_plus_lstm.ipynb
# second source: https://www.kaggle.com/code/orkatz2/cnn-lstm-pytorch-train
# from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

random.seed("9103")

### IDENTIFY PATH ###
t00 = time.time()


def train_with_soundscapes(files_path: List[Tuple[str, str]] = [],
                           ncols: int = 6016, *,
                           region: str = '',
                           si: int = 1964,
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
                           f_pattern_out: str = 'modelling',
                           windows_13: bool = True,
                           horas: int = 30,
                           metric_names: List[str] = M.list(),
                           dev_mode: bool = False
                           ) -> None:
    print('#### #### HOI FOREST - MODELLING #### ####')
    model_path = f'{path_out}data/{f_pattern_out}/'
    t0 = time.time()
    df_data = None
    ncols = 6016
    accuracy_dictt, accuracy_dicttt = {}, {}

    # SOME CONFIG ####
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    print('#### #### READING DATA FILES #### ####')
    try:
        print(files_path[0])
        df_data = pd.concat((pd.read_pickle(f[1]) for f in files_path), ignore_index=True)
        df_data.columns = df_data.columns.map(str)
        columns_to_train = metric_names + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
        # columns_to_train = metric_names # + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
        # print(columns_to_train)
        df_data = df_data[columns_to_train]
        print(df_data.shape, df_data.columns[:11], df_data.columns[-11:])
        # print(df_data.head(555))
        # exit()
    except Exception as e:
        print('ALWAYS PROBLEMS', 'CORRUPTED DATA =>', 'EXTRACTION', '<= DATA CORRUPTED', 'ALWAYS PROBLEMS', e)

    print('####', 'MERGE', len(files_path),
          'merge_data Dataframe shape:', df_data.shape,
          '####', 'time:', int(time.time() - t0))

    # exit()
    print('#### #### SPLIT TRAIN TEST #### ####')
    t0 = time.time()

    # Subsample 1% of the data with stratification
    label_column = df_data.columns[0]  # Assuming first column is your target
    df_data_sampled, _ = train_test_split(
        df_data,
        train_size=0.09103,
        stratify=df_data[label_column],
        random_state=9103
    )

    print('####', 'reduces subsampled shape:', df_data_sampled.shape)

    print('#### #### SPLIT TRAIN TEST #### ####')
    t0 = time.time()

    if dev_mode:
        # stratified train/test split on sampled data
        X_train, X_test, y_train, y_test = train_test_split(
            df_data_sampled.iloc[:, 1:],
            df_data_sampled.iloc[:, 0],
            test_size=0.2,
            stratify=df_data_sampled.iloc[:, 0],
            random_state=9103
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df_data.iloc[:, 1:],
            df_data.iloc[:, 0],
            test_size=0.2,
            stratify=df_data_sampled.iloc[:, 0],
            random_state=9103
        )

    # Normalize the data
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

    # Global Variables
    #### PREPARE FOR FITTING THE MODEL ####
    input_size = X_train.shape[-1]
    hidden_size = 128
    num_layers = 2
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    num_classes = 4 if len(unique_classes) <= 4 else len(unique_classes)
    # num_classes = num_classes if num_classes >= 4 else num_classes + 1
    print('####', 'CLASSES:', unique_classes, 'TOTAL', num_classes)

    # batch_s = 64 if windows_13 else 128
    batch_s = 64 if windows_13 else 64
    # batch_s = 32 if windows_13 else 32
    # batch_s = 128 if windows_13 else 128
    batch_s = 192 if windows_13 else 192
    # batch_s = 128 if windows_13 else 64
    train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('####', 'SPLIT', 'train_test_split and CUDA:', '???', device, '####', 'time:', int(time.time() - t0))

    # exit()

    t0 = time.time()

    # model 1: CNN + LSTM
    # model 2: LSTM + CNN
    # model 3: CNN LSTM parallel

    class CNN_LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(CNN_LSTM, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(64),  # Batch Normalisation
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(128),  # Batch Normalisation
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
                # nn.BatchNorm1d(64),  # Batch Normalisation
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(128),  # Batch Normalisation
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
                # nn.BatchNorm1d(64),  # Batch Normalisation
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(128),  # Batch Normalisation
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
              verbose: bool = verbose or False,
              model_path: str = f'{path_out}data/{f_pattern_out}/'
              ) -> None:
        criterion = nn.CrossEntropyLoss()
        # PARALLEL ??
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

            # torch.save(model.state_dict(), f'{model_path}{job_id}_{str(datetime.date.today())}_{model.__class__.__name__}.model')
            print('####',
                  'TRAINED MODEL',
                  model.__class__.__name__,
                  'model training time:', round(time.time() - t1, 3),
                  'total training time:', round(time.time() - t0, 3),
                  '####')

    #### MODELS ####
    cnn_lstm = CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    lstm_cnn = LSTM_CNN(input_size, hidden_size, num_layers, num_classes).to(device)
    cnn_lstm_parallel = ParallelCNNLSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

    dict_models = {
        0: [cnn_lstm],
        1: [lstm_cnn],
        11: [cnn_lstm, lstm_cnn],
        2: [cnn_lstm_parallel],
        22: [cnn_lstm, lstm_cnn, cnn_lstm_parallel],
    }
    # models = dict_models.get(22, [cnn_lstm])
    models = dict_models.get(0, [cnn_lstm])

    #### TRAIN ####
    num_epochs = 1
    # print('####', 'MODELS', dict_models, '####')
    print('####', 'MODELS - TOTAL', len(models), '####')
    print('####', 'EPOCHS', num_epochs, '####')
    # exit()

    tt0 = time.time()
    print('####', 'TRAINING', 'MODELS', '####')
    train(models, train_loader, epochs=num_epochs)
    print('####', 'TRAINING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

    # test
    def ttest(models, test_loader, metric_names_str: str = ", ".join([str(m) for m in metric_names])):
        with torch.no_grad():
            correct = 0
            total = 0
            accuracy_dict = {'METRICS': metric_names_str}
            for model in models:
                model.eval()
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                accuracy = round(100 * correct / total, 6)
                print(f'Accuracy of the {model.__class__.__name__} model on the test set: {accuracy} %')
                accuracy_dict[model.__class__.__name__] = accuracy
        return accuracy_dict

    def test(models, test_loader, metric_names_str: str = ", ".join([str(m) for m in metric_names]), e: int = num_epochs) -> dict:
        model_scores_dict = {'METRICS': metric_names_str}
        with torch.no_grad():
            for model in models:
                model.eval()
                metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                total = 0
                correct = 0
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    # Update total and correct predictions
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    # Compute TP, TN, FP, FN
                    for cls in torch.unique(y):  # Iterate over unique classes
                        cls = cls.item()
                        cls_pred = (predicted == cls)  # Predictions for the current class
                        cls_true = (y == cls)  # Ground truth for the current class
                        metrics['TP'] += (cls_pred & cls_true).sum().item()
                        metrics['FP'] += (cls_pred & ~cls_true).sum().item()
                        metrics['FN'] += (~cls_pred & cls_true).sum().item()
                        metrics['TN'] += ((~cls_pred) & (~cls_true)).sum().item()
                accuracy = round(100 * correct / total, 6)
                print(f'Accuracy of the {model.__class__.__name__} model on the test set: {accuracy} %')

                model_scores_dict[model.__class__.__name__] = {
                    'EPOCHS': e,
                    'AC': accuracy,
                    'TP': metrics['TP'],
                    'TN': metrics['TN'],
                    'FP': metrics['FP'],
                    'FN': metrics['FN'],
                }
        return model_scores_dict

    tt0 = time.time()
    accuracy_dictt = test(models, test_loader)
    # accuracy_dicttt = ttest(models, test_loader)
    print('####', 'TESTING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

    # plot bar chart with the accuracy of each model
    # sns.barplot(x=list(model_scores_dict.keys()), y=list(model_scores_dict.values()))

    with open(f'{model_path}000_models_accuracy_dict_epochs_{num_epochs}_{job_id}_{str(datetime.date.today())}.json', 'w') as fp:
        json.dump(accuracy_dictt, fp, sort_keys=True, indent=4)

    # with open(f'{model_path}000_models_accuracy_dicttt_{job_id}_{str(datetime.date.today())}.json', 'w') as fp:
    #    json.dump(accuracy_dicttt, fp, sort_keys=True, indent=4)

    print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
    print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')
    print('#### TIMES #### modelling TOTAL TOTAL ==>>', round(time.time() - t00, 3))


if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
