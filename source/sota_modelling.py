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
from itertools import cycle, combinations
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

# >>>> import libraries for SOTA >>>>

# Create a CNN object designed to recognize 3-second samples
# from opensoundscape import CNN
import torchvision.models as pymodels

# <<< import libraries for CNN <<<<
# primary source: https://github.com/mijanr/TimeSeries/blob/master/Time_Series_Classification/cnn_plus_lstm.ipynb
# second source: https://www.kaggle.com/code/orkatz2/cnn-lstm-pytorch-train
# from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

semilla = 9103
torch.manual_seed(semilla)
random.seed(semilla)
np.random.seed(semilla)

### IDENTIFY PATH ###
t00 = time.time()


def sota_train_with_soundscapes(files_path: List[Tuple[str, str]] = [],
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
                                dev_mode: bool = True,
                                n_epochs: int = 11,
                                df_stats: bool = False
                                ) -> None:
    print('#### #### HOI FOREST - MODELLING #### ####')
    model_path = f'{path_out}data/{f_pattern_out}/'
    t0 = time.time()
    df_data = None
    ncols = 6016
    # accuracy_dict, accuracy_dicttt = {}, {}
    accuracy_dict = {}

    # SOME CONFIG ####
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Fixed part: always included
    start_combi = 0  # 424
    i_fix_metrics = 7  # [reg, sid, ban, sec, men, med, sum, max, aci, bet, mmm, npp, hfq, htp, hhh, aei]
    fixed_part = metric_names[:i_fix_metrics]
    # Variable part: will be combined in all possible ways
    variable_part = metric_names[i_fix_metrics:]
    # Count total combinations
    total_combi = sum(1 for r in range(1, len(variable_part) + 1) for _ in combinations(variable_part, r))
    print('####', "COMBI", total_combi, '####', 'FIXED', len(fixed_part), 'VARIABLE', len(variable_part))
    # TOTAL combinations: 511
    # exit()
    i_r_c = 0
    for r in range(1, len(variable_part) + 1):  # r = number of items in each combination
        for combi_metrics in combinations(variable_part, r):
            i_r_c += 1
            if i_r_c <= start_combi:
                continue
            combi_metric_names = fixed_part + list(combi_metrics)
            print('#### #### READING DATA FILES #### ####')
            print(i_r_c, total_combi, '####', 'COMBI', combi_metric_names, 'METRICS', '####')
            # continue
            # exit()
            try:
                print(files_path[0])
                df_data = pd.concat((pd.read_pickle(f[1]) for f in files_path), ignore_index=True)
                df_data.columns = df_data.columns.map(str)
                # include the scalar AND temporal features
                # columns_to_train = combi_metric_names + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
                # include the ONLY scalar features
                columns_to_train = combi_metric_names  # + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
                df_data = df_data[columns_to_train]
                print(df_data.shape, df_data.columns[:11], df_data.columns[-11:])
                # print(df_data.head(555))
                # exit()
            except Exception as e:
                print('ALWAYS PROBLEMS', 'CORRUPTED DATA =>', 'EXTRACTION', '<= DATA CORRUPTED', 'ALWAYS PROBLEMS', e)

            print('####', 'MERGE', len(files_path),
                  'merge_data Dataframe shape:', df_data.shape,
                  '####', 'time:', int(time.time() - t0))

            if df_stats:
                # basic stats for numeric columns
                stats_df = df_data.describe(include='all').transpose()  # .transpose() makes it more readable
                stats_df.to_csv(f'{model_path}df_statistics_{job_id}_{str(datetime.date.today())}.csv', index=True)
                # exit()

            # exit()
            print('#### #### SPLIT TRAIN TEST #### ####')
            t0 = time.time()
            X_train, X_test, y_train, y_test = None, None, None, None
            if dev_mode:
                # subsample 1% for dev_mode
                label_column = df_data.columns[0]  # target label
                df_data_sampled, _ = train_test_split(
                    df_data,
                    train_size=semilla * 0.00001,
                    stratify=df_data[label_column],
                    random_state=semilla
                )
                print('####', 'REDUCED subsampled shape:', df_data_sampled.shape)
                X_train, X_test, y_train, y_test = train_test_split(
                    df_data_sampled.iloc[:, 1:],
                    df_data_sampled.iloc[:, 0],
                    test_size=0.2,
                    stratify=df_data_sampled.iloc[:, 0],  # to handle unbalanced classes
                    random_state=semilla
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    df_data.iloc[:, 1:],
                    df_data.iloc[:, 0],
                    test_size=0.2,
                    stratify=df_data.iloc[:, 0],  # to handle unbalanced classes
                    random_state=semilla
                )

            # exit()

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
            # label_column = df_data.columns[0]  # target label
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

            #### MODELS ####
            # model 1: SEQ => CNN >> LSTM
            # model 2: SEQ => LSTM >> CNN
            # model 3: PARALLEL => CNN || LSTM
            # model 4: Simple CNN
            # model 5: Simple LSTM
            # model 6: Simple SVM
            #### MODELS ####

            # model 1: SEQ => CNN >> LSTM
            class SEQ_CNN_LSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes):
                    super(SEQ_CNN_LSTM, self).__init__()
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

            # model 2: SEQ => LSTM >> CNN
            class SEQ_LSTM_CNN(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes):
                    super(SEQ_LSTM_CNN, self).__init__()
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

            # model 3: PARALLEL => CNN || LSTM
            class PARA_CNN_LSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes):
                    super(PARA_CNN_LSTM, self).__init__()
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

            # model 4: Simple CNN
            class Simple_CNN(nn.Module):
                def __init__(self, input_size, num_classes):
                    super(Simple_CNN, self).__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=2),
                        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=256),
                        nn.ReLU(),
                        nn.Linear(256, num_classes)
                    )

                def forward(self, x):
                    # Input (batch_size, seq_len, features) -> CNN expects (batch_size, channels, seq_len)
                    x = x.permute(0, 2, 1)
                    out = self.cnn(x)
                    return out

            # model 5: Simple LSTM
            class Simple_LSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes):
                    super(Simple_LSTM, self).__init__()
                    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.fc(out[:, -1, :])  # take the output of the last timestep
                    return out

            # model 6: Simple SVM
            class Simple_SVM(nn.Module):
                def __init__(self, input_size, num_classes):
                    super(Simple_SVM, self).__init__()
                    self.fc = nn.LazyLinear(num_classes)  # <--- LazyLinear!

                def forward(self, x):
                    x = x.mean(dim=1)  # Mean across sequence length
                    out = self.fc(x)
                    return out

            # TRAIN
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
                    if model.__class__.__name__ == "Simple_SVM":
                        criterion = nn.MultiMarginLoss()
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
                    # models sizes are ~200-500 MB
                    # torch.save(model.state_dict(), f'{model_path}{job_id}_{str(datetime.date.today())}_{model.__class__.__name__}.model')
                    print('####',
                          'TRAINED MODEL',
                          model.__class__.__name__,
                          'model training time:', round(time.time() - t1, 3),
                          'total training time:', round(time.time() - t0, 3),
                          '####')

            # if not dev_mode:
            #### PROPOSED MODELS ####
            cnn_lstm = SEQ_CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            lstm_cnn = SEQ_LSTM_CNN(input_size, hidden_size, num_layers, num_classes).to(device)
            cnn_lstm_parallel = PARA_CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            #### BASE-LINE MODELS ####
            simple_cnn = Simple_CNN(input_size, num_classes).to(device)
            simple_lstm = Simple_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            simple_svm = Simple_SVM(input_size, num_classes).to(device)

            #### SOTA MODELS ####
            class ResNet1D(nn.Module):
                def __init__(self, resnet, input_size, num_classes):
                    super(ResNet1D, self).__init__()

                    # Load a 2D ResNet model
                    self.resnet = resnet # models.resnet34(pretrained=False)

                    # Modify first conv layer: it expects 3 channels, but your data has `input_size`
                    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)

                    # Remove avgpool and fc layers, we'll do custom pooling
                    self.features = nn.Sequential(*list(resnet.children())[:-2])

                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(512, num_classes)  # 512 is output channels for resnet34

                def forward(self, x):
                    # Input: (batch_size, seq_len, input_size)
                    # CNN expects: (batch_size, channels, H, W)
                    x = x.permute(0, 2, 1).unsqueeze(2)  # (batch_size, channels, 1, seq_len)
                    x = self.features(x)
                    x = self.avgpool(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x

            resnet = pymodels.resnet34(pretrained=False)
            sota_resnet = ResNet1D(resnet, input_size, num_classes).to(device)

            # exit()

            # # Use resnet34 architecture
            # architecture = "resnet34"
            # # Can use this code to get your classes, if needed
            # # Parameters to use for label creation
            # clip_duration = 6
            # clip_overlap = 1.9
            # min_label_overlap = 0.25
            # # species_of_interest = ["NOCA", "EATO", "SCTA", "BAWW", "BCCH", "AMCR", "NOFL"]
            # # class_list = list(train_df.columns)
            # label_column = df_data.columns[0]  # target label
            # class_list = list(df_data.columns[0])  # target label
            # unique_classes = list(np.unique(np.concatenate((y_train, y_test))))
            # model = CNN(
            #     architecture=architecture,
            #     classes=class_list,
            #     sample_duration=clip_duration,
            #     single_target=True,
            #
            # )
            #
            # exit()

            dict_models = {
                # DUAL-MODELS
                00: [cnn_lstm],
                10: [lstm_cnn],
                11: [cnn_lstm, lstm_cnn],
                12: [cnn_lstm_parallel],
                22: [cnn_lstm, lstm_cnn, cnn_lstm_parallel],
                # SIMPLE-MODELS
                23: [simple_svm],
                24: [simple_lstm],
                25: [simple_cnn],
                26: [simple_cnn, simple_lstm, simple_svm],
                # SOTA-MODELS
                28: [sota_resnet]
            }
            # models = dict_models.get(22, [cnn_lstm])
            # models = dict_models.get(00, [cnn_lstm]) if dev_mode else dict_models.get(22, [cnn_lstm])
            # models = dict_models.get(25, [simple_cnn]) if dev_mode else dict_models.get(26, [simple_cnn, simple_lstm, simple_svm])
            models = dict_models.get(28, [sota_resnet]) # if dev_mode else dict_models.get(26, [simple_cnn, simple_lstm, simple_svm])

            #### TRAIN ####
            num_epochs = n_epochs
            # print('####', 'MODELS', dict_models, '####')
            print('####', 'MODELS - TOTAL', len(models), '####')
            print('####', 'EPOCHS', num_epochs, '####')
            # exit()

            tt0 = time.time()
            print('####', 'TRAINING', 'MODELS', '####')
            train(models, train_loader, epochs=num_epochs)
            print('####', 'TRAINING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

            # test
            def ttest(models, test_loader, metric_names_str: str = ", ".join([str(m) for m in combi_metric_names])):
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

            def test(models, test_loader, metric_names_str: str = ", ".join([str(m) for m in combi_metric_names]),
                     e: int = num_epochs,
                     c: int = i_r_c
                     ) -> dict:
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
            accuracy_dict = test(models, test_loader, e=n_epochs, c=i_r_c)
            # accuracy_dicttt = ttest(models, test_loader)
            print('####', 'TESTING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

            # plot bar chart with the accuracy of each model
            # sns.barplot(x=list(model_scores_dict.keys()), y=list(model_scores_dict.values()))

            with open(f'{model_path}000_models_accuracy_dict_EPOCHS_{num_epochs}_COMBI_{i_r_c}_'
                      f'JOBID_{job_id}_{str(datetime.date.today())}.json',
                      'w') as fp:
                json.dump(accuracy_dict, fp, sort_keys=True, indent=4)

            # with open(f'{model_path}000_models_accuracy_dicttt_{job_id}_{str(datetime.date.today())}.json', 'w') as fp:
            #    json.dump(accuracy_dicttt, fp, sort_keys=True, indent=4)

            print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
            print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')
            print('#### TIMES #### modelling TOTAL TOTAL ==>>', round(time.time() - t00, 3), 'seconds')


if __name__ == '__main__':
    # sota_train_with_soundscapes()
    print('Mirá ve.. oís?? alles gut oder was??')
