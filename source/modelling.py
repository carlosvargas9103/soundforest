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
from collections import defaultdict

from joblib import Parallel, delayed
from joblib import effective_n_jobs
from itertools import cycle, combinations
from multiprocessing import Pool

import pandas as pd
import numpy as np
# import seaborn as sns

from extraction import (
    Metrics as M,
    SoundscapeRegion as Region
)

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax

# >>>> import libraries for SOTA >>>>

# Create a CNN object designed to recognize 3-second samples
# from opensoundscape import CNN
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34, ResNet34_Weights,
    vgg16, VGG16_Weights,
    alexnet, AlexNet_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights
)

# import torchvision.models as pymodels


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
                           f_pattern_models_out: str = 'models',
                           windows_13: bool = True,
                           horas: int = 30,
                           metric_names: List[str] = M.list(),
                           dev_mode: bool = True,
                           n_epochs: int = 11,
                           df_stats: bool = False,
                           m_sota: int = 0,
                           s_combi: int = 0,
                           e_combi: int = 9103
                           ) -> None:
    print('#### #### HOI FOREST - MODELLING #### ####')
    model_path = f'{path_out}data/{f_pattern_out}/'
    t0 = time.time()
    df_data = None
    # ncols = 6016
    accuracy_dict = {}
    model_config = {}
    all_indices_names = None


    # SOME CONFIG ####
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Fixed part: always included
    start_combi, end_combi = s_combi, e_combi  # 424
    #### INDICES 21 => [reg, sid, ban, sec, men, med, sum, max, min, aci, aca, adi, bet, mmm, npp, hfq, htp, hhh, aei, dsi, amr]
    i_fix_metrics = 21  # [reg, sid, ban, sec, men, med, sum, max, min, aci, aca, adi, bet, mmm, npp, hfq, htp, hhh, aei, dsi, amr]
    i_fix_metrics = 14  # [reg, sid, ban, sec, men, med, sum, max, min, bet, npp, hfq, htp, aei, adi, dsi, amr]
    i_fix_metrics = 13  # [reg, sid, ban, sec, men, med, sum, max, bet, npp, hfq, htp, aei, adi, dsi, amr]
    i_fix_metrics = 13  # [reg, sid, ban, sec, men, med, sum, max, bet, npp, hfq, htp, aei]
    fixed_part = metric_names[:i_fix_metrics]  # having 14 elements => the total combinations among [adi, dsi, amr] is seven (7)
    # Variable part: will be combined in all possible ways
    variable_part = metric_names[i_fix_metrics:]
    # Count total combinations
    total_combi = sum(1 for r in range(1, len(variable_part) + 1) for _ in combinations(variable_part, r)) if variable_part else 1
    print('####', "COMBI", total_combi, '####', 'FIXED', len(fixed_part), 'VARIABLE', len(variable_part))

    # TOTAL combinations: 511
    # exit()
    i_r_c = 0
    for r in range(len(variable_part) + 1):  # r = number of items in each combination
        for indices_combination in combinations(variable_part, r):
            i_r_c += 1
            if i_r_c < start_combi:
                continue
            elif i_r_c > end_combi:
                break

            all_indices_names = fixed_part + list(indices_combination)
            print('#### #### READING DATA FILES #### ####')
            print(i_r_c - 1, total_combi, '####', 'COMBI', all_indices_names, 'METRICS', '####')
            # continue
            # exit()

            try:
                # TODO: Maybe read less files here if dev_mode
                print(files_path[0])
                # if dev_mode:
                #     files_sample = files_path.copy()
                #     n_files_sample = max(1, int(len(files_sample) * 0.11))
                #     files_path = random.sample(files_sample, n_files_sample)

                if dev_mode:
                    random.seed(42)

                    files_by_class = defaultdict(list)
                    for cls, path in files_path:
                        files_by_class[cls].append((cls, path))

                    sampled_files = []

                    for cls, files in files_by_class.items():
                        n = max(1, int(len(files) * 0.01))  # 1% per class
                        sampled_files.extend(random.sample(files, n))

                    files_path = sampled_files
                # LOAD FILES
                df_data = pd.concat((pd.read_pickle(f[1]) for f in files_path), ignore_index=True)
                print('### ###', 'SOY YO', type(df_data))
                df_data.columns = df_data.columns.map(str)
                # print('### ###', df_data.shape)
                # include the scalar AND temporal features
                # columns_to_train = all_indices_names + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
                # include the ONLY scalar features
                columns_to_train = all_indices_names  # + [col for col in df_data.columns if col.startswith(str(M.VECTOR_VEC))]
                df_data = df_data[columns_to_train]
                print(df_data.shape, df_data.columns[:11], df_data.columns[-11:])
                # print(df_data.head(555))
                # exit()
            except Exception as e:
                print('ALWAYS PROBLEMS', 'CORRUPTED DATA =>', 'EXTRACTION', '<= DATA CORRUPTED', 'ALWAYS PROBLEMS', e)
                raise

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
            # if dev_mode:
            if False:
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
            unique_classes = np.unique(np.concatenate(([Region.Unknown.value], y_train, y_test)))
            num_classes = 4 if (len(unique_classes) <= 4) else len(unique_classes)
            print('####', 'CLASSES', unique_classes, 'TOTAL', num_classes)

            batch_s = 64 if windows_13 else 64
            batch_s = 192 if windows_13 else 192

            train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print('####', 'SPLIT', 'train_test_split and CUDA:', '???', device, '####', 'time:', int(time.time() - t0))

            # exit()

            t0 = time.time()

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

            # MODEL CONFIG
            model_config = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

            # TRAIN
            def train(models: List,
                      train_loader: DataLoader,
                      epochs: int = 1,
                      t0: int = time.time(),
                      verbose: bool = verbose or False,
                      model_path: str = f'{path_out}data/{f_pattern_models_out}/',
                      *,
                      model_config: None = model_config
                      ) -> None:
                criterion = nn.CrossEntropyLoss()
                # PARALLEL ??
                for model in models:
                    model_name = model.__class__.__name__
                    if model_name == "SOTA_Model":
                        model_name = model.backbone_name
                    if model_name == "Simple_SVM":
                        criterion = nn.MultiMarginLoss()
                    t1 = time.time()
                    print('####',
                          'TRAINING MODEL',
                          model_name,
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
                    # model serialisation
                    # torch.save(
                    #     model.state_dict(),
                    #     f'{model_path}{job_id}_{str(datetime.date.today())}_{model.__class__.__name__}_COMBI_{i_r_c - 1}.pth'
                    # )
                    today = datetime.date.today()
                    model_name = model.__class__.__name__
                    checkpoint_path = Path((
                        f"{model_path}"
                        f"{job_id}_{today}_{model_name}_COMBI_{i_r_c - 1}.pth"
                    ))
                    checkpoint = {
                        "model_class": model_name,
                        "model_config": model_config,
                        "metrics": all_indices_names,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "torch_version": torch.__version__
                    }
                    print('### ###', 'CHECK-MEINE-PUNTO', checkpoint.get('model_class'))
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = checkpoint_path.with_suffix(".tmp")
                    torch.save(checkpoint, tmp_path)
                    tmp_path.rename(checkpoint_path)
                    # torch.save(checkpoint, checkpoint_path)
                    # torch.save(checkpoint, tmp_path)
                    # tmp_path.rename(checkpoint_path)

                    ### ### NOW THE MODEL ### ###
                    model.eval()
                    scripted_model = torch.jit.script(model)
                    scripted_path = Path((
                        f"{model_path}"
                        f"{job_id}_{today}_{model_name}_COMBI_{i_r_c - 1}.pt"
                    ))
                    print('### ###', 'CHECK-MEINE-PUNTO', checkpoint.get('model_class'))
                    scripted_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = scripted_path.with_suffix(".tmp")
                    torch.save(scripted_model, tmp_path)
                    tmp_path.rename(scripted_path)

                    print('####',
                          'TRAINED MODEL',
                          model_name,
                          'model training time:', round(time.time() - t1, 3),
                          'total training time:', round(time.time() - t0, 3),
                          '####')

            class SOTA_Model(nn.Module):
                def __init__(self, backbone_name, base_model, input_size, num_classes, output_dim):
                    super(SOTA_Model, self).__init__()
                    self.backbone_name = backbone_name

                    # Modify input conv layer depending on backbone
                    if backbone_name.startswith("resnet") or backbone_name.startswith("resnext") or backbone_name.startswith(
                            "regnet"):
                        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)

                    elif backbone_name.startswith("vgg") or backbone_name.startswith("alexnet"):
                        features = list(base_model.features)
                        if isinstance(features[0], nn.Conv2d) and features[0].in_channels == 3:
                            features[0] = nn.Conv2d(1, features[0].out_channels,
                                                    kernel_size=features[0].kernel_size,
                                                    stride=features[0].stride,
                                                    padding=features[0].padding)
                            base_model.features = nn.Sequential(*features)

                    elif backbone_name.startswith("efficientnet"):
                        conv_stem = base_model.features[0][0]
                        base_model.features[0][0] = nn.Conv2d(1, conv_stem.out_channels,
                                                              kernel_size=conv_stem.kernel_size,
                                                              stride=conv_stem.stride,
                                                              padding=conv_stem.padding,
                                                              bias=False)

                    elif backbone_name.startswith("convnext"):
                        conv_stem = base_model.features[0][0]
                        base_model.features[0][0] = nn.Conv2d(1, conv_stem.out_channels,
                                                              kernel_size=conv_stem.kernel_size,
                                                              stride=conv_stem.stride,
                                                              padding=conv_stem.padding,
                                                              bias=False)

                    else:
                        raise NotImplementedError(f"{backbone_name} not yet supported.")

                    # Extract features and custom classifier
                    self.features = nn.Sequential(*list(base_model.children())[:-2])
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(output_dim, num_classes)

                def forward(self, x):

                    if self.backbone_name.startswith(("vgg", "resnet", "alexnet", "convnext_tiny")):
                        # Convert to (B, C=1, H, W) if input is (B, T, F)
                        if x.ndim == 3:
                            x = x.permute(0, 2, 1).unsqueeze(1)  # e.g., (B, 1, F, T)
                        # Resize to standard input size
                        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                    else:
                        # Custom case, adjust as needed
                        x = x.permute(0, 2, 1).unsqueeze(2)  # e.g., (B, C=1, 1, W)

                    # x = x.permute(0, 2, 1).unsqueeze(3)  # (B, C=1, H=1, W)
                    x = self.features(x)
                    x = self.avgpool(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x

            # Example model initializations (device and input_size must be defined)
            sota_models = {
                0: SOTA_Model(
                    "resnet34",
                    resnet34(weights=ResNet34_Weights.DEFAULT),
                    input_size,
                    num_classes,
                    512
                ).to(device),
                1: SOTA_Model(
                    "vgg16",
                    vgg16(weights=VGG16_Weights.DEFAULT),
                    input_size,
                    num_classes,
                    512
                ).to(device),
                2: SOTA_Model(
                    # RuntimeError: CUDA error: an illegal memory access was encountered
                    # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
                    "alexnet",
                    alexnet(weights=AlexNet_Weights.DEFAULT),
                    input_size,
                    num_classes,
                    256
                ).to(device),
                3: SOTA_Model(
                    "efficientnet_v2_s",
                    efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT),
                    input_size,
                    num_classes, 1280
                ).to(device),
                4: SOTA_Model(
                    "convnext_tiny",
                    convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT),
                    input_size,
                    num_classes, 768
                ).to(device)
                # 5: SOTA_Model("swin_t", swin_t(weights=Swin_T_Weights.DEFAULT), input_size, num_classes, 768).to(device),
                # 6: SOTA_Model("vit_b_16", vit_b_16(weights=ViT_B_16_Weights.DEFAULT), input_size, num_classes, 768).to(device),
            }

            class ResNet1D(nn.Module):
                def __init__(self, resnet, input_size, num_classes):
                    super(ResNet1D, self).__init__()
                    self.resnet = resnet  # models.resnet34(pretrained=False)
                    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
                    self.features = nn.Sequential(*list(resnet.children())[:-2])
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(512, num_classes)  # 512 is output channels for resnet34

                def forward(self, x):
                    x = x.permute(0, 2, 1).unsqueeze(2)  # (batch_size, channels, 1, seq_len)
                    x = self.features(x)
                    x = self.avgpool(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x

            #### PROPOSED MODELS ####
            cnn_lstm = SEQ_CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            lstm_cnn = SEQ_LSTM_CNN(input_size, hidden_size, num_layers, num_classes).to(device)
            para_cnn_lstm = PARA_CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            #### BASE-LINE MODELS ####
            simple_cnn = Simple_CNN(input_size, num_classes).to(device)
            simple_lstm = Simple_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            simple_svm = Simple_SVM(input_size, num_classes).to(device)
            #### SOTA-MODELS ####
            resnet = ResNet1D(resnet34(weights=ResNet34_Weights.DEFAULT), input_size, num_classes).to(device)

            dict_models = {
                # DUAL-MODELS
                10: [cnn_lstm],
                11: [lstm_cnn],
                12: [cnn_lstm, lstm_cnn],
                13: [para_cnn_lstm],
                22: [cnn_lstm, lstm_cnn, para_cnn_lstm],
                # # SIMPLE-MODELS
                23: [simple_svm],
                24: [simple_lstm],
                25: [simple_cnn],
                26: [simple_cnn, simple_lstm, simple_svm],
                # SOTA-MODELS
                28: [resnet],
                30: list(sota_models.values()),
                31: [sota_models.get(0)],
                32: [sota_models.get(1)],
                33: [sota_models.get(2)],
                34: [sota_models.get(3)],
                35: [sota_models.get(4)],
                # ALL-MODELS
                36: [cnn_lstm, lstm_cnn, para_cnn_lstm, simple_cnn, simple_lstm, simple_svm, sota_models.get(0)],
                # BEST-MODELS => Vargas Secret
                37: [cnn_lstm, para_cnn_lstm, simple_cnn]
            }
            # models = dict_models.get(39, list(sota_models.values())) if dev_mode else dict_models.get(00, [sota_resnet])
            models = dict_models.get(25, [cnn_lstm])  # if not dev_mode else dict_models.get(10, [cnn_lstm])
            # models = dict_models.get(30 + m_sota, []) if 0 < m_sota < 6 else models
            models = dict_models.get(m_sota, models) if m_sota else models

            #### TRAIN ####
            num_epochs = n_epochs
            print('####', 'MODELS - TOTAL', len(models), '####')
            print('####', 'EPOCHS', num_epochs, '####')
            # exit()

            tt0 = time.time()
            print('####', 'TRAINING', 'MODELS', '####')
            train(models, train_loader, epochs=num_epochs, model_config=model_config)
            print('####', 'TRAINING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

            # test
            def test(models, test_loader, metric_names_str: str = ", ".join([str(m) for m in all_indices_names]),
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
                        model_name = model.__class__.__name__
                        if model_name == "SOTA_Model":
                            model_name = model.backbone_name
                        print(f'Accuracy of the {model_name} model on the test set: {accuracy} %')

                        model_scores_dict[model_name] = {
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
            print('####', 'TESTING', 'TOTAL TIME:', round(time.time() - tt0, 3), '####')

            metrics_dict = accuracy_dict.get('METRICS')

            for model_name, model_data in accuracy_dict.items():
                if model_name == "METRICS":
                    continue

                model_dict = {
                    'METRICS': metrics_dict,
                    model_name: model_data
                }

                output_file = (
                    f"{model_path}000_models_accuracy_dict_"
                    f"MODEL_{model_name}_"
                    f"EPOCHS_{num_epochs}_"
                    f"COMBI_{i_r_c - 1}_"
                    f"JOBID_{job_id}_"
                    f"{datetime.date.today()}.json"
                )

                with open(output_file, "w") as fp:
                    json.dump(model_dict, fp, sort_keys=True, indent=4)

            # with open(f'{model_path}000_models_accuracy_dict_MODEL{model_name}_EPOCHS_{num_epochs}_COMBI_{i_r_c}_'
            #           f'JOBID_{job_id}_{str(datetime.date.today())}.json',
            #           'w') as fp:
            #     json.dump(accuracy_dict, fp, sort_keys=True, indent=4)

            print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
            print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')
            print('#### TIMES #### modelling TOTAL TOTAL ==>>', round(time.time() - t00, 3), 'seconds')


if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
