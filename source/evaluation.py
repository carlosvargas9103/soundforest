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
import matplotlib.pyplot as plt

from extraction import Metrics as M
from modelling import *

# >>>> import libraries for CNN >>>>
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.optim import Adam
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax

# <<< import libraries for CNN <<<<
# primary source: https://github.com/mijanr/TimeSeries/blob/master/Time_Series_Classification/cnn_plus_lstm.ipynb
# second source: https://www.kaggle.com/code/orkatz2/cnn-lstm-pytorch-train
# from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

random.seed("9103")

### IDENTIFY PATH ###
t00 = time.time()

### IDENTIFY PATH ###
cwd = os.getcwd()
cwd = os.getcwd()  # cwd: /home/fs72552/vargas/forests-sounds-vargas/source
cwd = str(Path(cwd).parents[0]) if cwd.endswith('/source') else cwd
print('PATH', cwd)


def evaluation_soundscapes(files_path: List[Tuple[str, str]] = [],
                           dir_json_results_in: str = '',
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
                           f_pattern_out: str = 'evaluation',
                           windows_13: bool = True,
                           horas: int = 30,
                           metric_names: List[str] = M.list(),
                           dev_mode: bool = True,
                           n_epochs: int = 11,
                           df_stats: bool = False
                           ) -> None:
    print('#### #### HOI EVALUATION #### ####')
    print('####', 'INDICES =>', metric_names, '<= INDICES', len(metric_names), '####')
    t00 = time.time()
    model_names = ["COMPOSED", "SIMPLE_MODELS", "SOTA", "ALL"]
    model_name_filename = model_names[0]

    experiments = []

    experiments.append(('EVAL_SEQ_SEQ_PARA_2606265', model_names[0]))
    experiments.append(('EVAL_SEQ_SEQ_PARA_2616999', model_names[0]))
    experiments.append(('EVAL_SEQ_SEQ_PARA_2620555', model_names[0]))
    experiments.append(('EVAL_SIMPLE_MODELS_2610631', model_names[1]))
    experiments.append(('EVAL_ResNet1D_2616261', model_names[2]))
    experiments.append(('TESTING_EVAL_ALL', model_names[3]))
    # dir_json_results_in, model_name_filename = 'EVAL_SEQ_SEQ_PARA_2616999', model_names[0]
    # dir_json_results_in, model_name_filename = 'EVAL_SEQ_SEQ_PARA_2620555', model_names[0]
    # dir_json_results_in, model_name_filename = 'EVAL_SIMPLE_MODELS_2610631', model_names[1]
    # dir_json_results_in, model_name_filename = 'EVAL_ResNet1D_2616261', model_names[2]
    # dir_json_results_in, model_name_filename = 'TESTING_EVAL_ALL', model_names[3]

    for exp in experiments:
        print(exp)
        # exit()
        dir_json_results_in, model_name_filename = exp[0], exp[1]
        print(dir_json_results_in, model_name_filename)
        # exit()
        # continue

        path_in_evaluation = f'{cwd}/out/data/modelling/{dir_json_results_in}/'
        path_data = path_in_evaluation
        folders_in, f_pattern_out, f_ext_in = 'modelling', 'evaluation', '.json'
        path_data_in = path_in_evaluation # f'{path_out}data/{folders_in}/'
        # READ models_accuracy_dict to process

        # Read all JSON files from path_data_in
        json_dicts = []
        for dirpath, dirnames, filenames in os.walk(path_data_in):
            for filename in filenames:
            # for filename in os.listdir(path_data_in):
                if filename.endswith(f_ext_in):
                    # filepath = os.path.join(path_data_in, filename)
                    filepath = os.path.join(dirpath, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                json_dicts.append(data)
                            elif isinstance(data, list):
                                json_dicts.extend(data)  # handle list of dicts
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

        # Convert to DataFrame
        df_json = pd.DataFrame(json_dicts)
        print(df_json.shape)
        print(df_json.head())

        # flatten the json
        rows = []
        for item in json_dicts:
            metrics = item.get("METRICS", "")
            num_metrics = int(len([m.strip() for m in metrics.split(",") if m.strip()]))

            for model_name, values in item.items():
                if model_name != "METRICS":
                    model_name_filename = model_name if not model_name_filename else model_name_filename
                    if isinstance(values, dict):
                        flat_row = values.copy()
                        flat_row["MODEL"] = model_name
                        flat_row["INDICES"] = metrics
                        flat_row["NUM_INDICES"] = num_metrics
                        rows.append(flat_row)
        df = pd.DataFrame(rows)

        # calculate performance scores
        df["Precision"] = 100 * df["TP"] / (df["TP"] + df["FP"])
        df["Recall"] = 100 * df["TP"] / (df["TP"] + df["FN"])
        df["F1"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])

        df["AC"] = df["AC"].round(3)
        df["Precision"] = df["Precision"].round(3)
        df["Recall"] = df["Recall"].round(3)
        df["F1"] = df["F1"].round(3)
        df["F1"] = df["F1"].fillna(0)

        # df = df.sort_values(by="AC", ascending=False)
        df = df.sort_values(by="F1", ascending=False)

        # reorder columns
        cols = df.columns.tolist()
        cols = [col for col in cols if col not in ["MODEL", "AC", "INDICES", "NUM_INDICES"]]
        new_order = ["MODEL"] + cols + ["AC", "INDICES", "NUM_INDICES"]
        df = df[new_order]

        for c, col in enumerate(df.columns[1:-2]):
            if c > 4:
                df[col] = df[col].astype(float)
            else:
                df[col] = df[col].astype(int)

        # Ensure 'INDICES' is a string
        int_fixed_indices = 8
        df["INDICES"] = df["INDICES"].astype(str) # change name
        df["INDICES"] = df["INDICES"].apply(lambda x: ", ".join([m.strip() for m in x.split(",")[int_fixed_indices:] if m.strip()]))
        df["INDICES"] = df["INDICES"].astype(str).apply(lambda x: f'[{x}]')

        # one-hot-encoded
        # Clean and parse the INDICES column (convert string lists to actual lists)
        df['INDICES'] = df['INDICES'].str.strip('[]').str.replace(' ', '').str.split(',')
        # Extract unique indices
        all_indices = sorted(set(i if i else 'fix' for sublist in df['INDICES'] for i in sublist))
        # One-hot encode and assign AC values
        for idx in all_indices:
            df[idx] = df.apply(lambda row: row['AC'] if idx in row['INDICES'] else 0.0, axis=1)
        df['INDICES'] = df['INDICES'].apply(lambda lst: ['fix' if i == '' else i for i in lst])
        print(all_indices)

        for idx in all_indices:
            df[idx] = 0.0  # float to allow accuracy assignment
        for i, row in df.iterrows():
            for idx in row['INDICES']:
                df.at[i, idx] = row['AC']

        print(df.shape)
        print(df.head())

        # Export to CSV
        df.to_csv(f"{path_data_in}model_{model_name_filename}_performance_summary_{dir_json_results_in}.csv", index=False, sep=';')

        import math

        subset_values = [9, 10, 11, 12, 13, 14, 15, 16]  # Any length list

        def plot_combined_ac_group(ac_filter, label, filename):
            n = len(subset_values)
            ncols = 4
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
            axes = axes.flatten()  # Flatten 2D axes array for easy indexing

            legend_handles = None

            for i, num in enumerate(subset_values):
                subset = df[(df['NUM_INDICES'] == num) & ac_filter]
                subset.to_csv(f"{path_data_in}subset_{model_name_filename}_{label}_indices_{num}.csv", index=False, sep=';')
                ax = axes[i]

                if not subset.empty:
                    plot_indices = [idx for idx in all_indices if idx != 'fix']
                    normalized = subset[plot_indices].div(subset['NUM_INDICES'], axis=0)
                    grouped = normalized.groupby(subset['MODEL']).sum().T
                    # grouped = normalized.groupby(subset['MODEL']).mean().T
                    # grouped = subset.groupby('MODEL')[plot_indices].mean().T
                    # Step 3: Normalize each value in grouped to 0–100 range
                    grouped = grouped.apply(lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0, axis=0)
                    plot = grouped.plot(kind='bar', ax=ax, legend=False)

                    if legend_handles is None:
                        legend_handles = plot.containers  # capture legend handles

                    ax.set_title(f'NUM_INDICES = {num}')
                    ax.set_xlabel('indices')
                    if i % ncols == 0:
                        ax.set_ylabel('accuracy (mean)')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.set_visible(False)

            # Hide unused subplots
            for j in range(len(subset_values), len(axes)):
                axes[j].set_visible(False)

            # Add shared legend outside the plot (if any data was plotted)
            if legend_handles:
                labels = grouped.columns.tolist()
                fig.legend(legend_handles, labels, loc='center right', title='Model')

            fig.suptitle(f"indices contributions per model ({label.lower()} 79% ac)", fontsize=16)
            fig.tight_layout(rect=[0, 0, 0.92, 0.95])
            fig.savefig(f'{path_data_in}{filename}')
            plt.close(fig)
            print(f"Saved: {filename}")

        # Create and save the two combined charts
        plot_combined_ac_group(df['AC'] > 79, 'above', 'combined_chart_AC_above_79.png')
        plot_combined_ac_group(df['AC'] <= 79, 'below', 'combined_chart_AC_below_79.png')

        # per feature
        sns.set_style("whitegrid")  # white background with gridlines
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x="NUM_INDICES", y="AC", data=df)
        ax.set_xlabel("Number of Selected Features")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs. Number of Selected Features")
        # ax.set_ylim(0, 100)
        ax.yaxis.grid(True)  # add horizontal grid lines
        plt.tight_layout()
        plt.savefig(f"{path_data_in}BOXPLOT_{model_name_filename}_ac_vs_num_indices_{dir_json_results_in}.png")

        # per model
        models = sorted(df["MODEL"].unique())
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x="MODEL", y="AC", data=df, order=models)
        ax.set_xlabel("Model Name")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Model")
        # ax.set_ylim(0, 100)
        plt.xticks(rotation=45)  # rotate x labels for readability&#8203;:contentReference[oaicite:6]{index=6}
        ax.yaxis.grid(True)  # horizontal grid lines
        plt.tight_layout()
        plt.savefig(f"{path_data_in}BOXPLOT_{model_name_filename}_ac_vs_model_{dir_json_results_in}.png")

        # grouped by model
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x="NUM_INDICES", y="AC", hue="MODEL", data=df)
        ax.set_xlabel("Number of Selected Features")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Number of Features and Model")
        # ax.set_ylim(0, 100)
        ax.yaxis.grid(True)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{path_data_in}BOXPLOT_{model_name_filename}_ac_vs_num_indices_per_model_{dir_json_results_in}.png")
        # plt.show()

        # max accuracy
        df_summary = df.groupby("NUM_INDICES", as_index=False)["AC"].max()

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_summary, x="NUM_INDICES", y="AC", marker="o")
        ax.set_title("Maximum Accuracy vs. Number of Selected Features")
        ax.set_xlabel("Number of Selected Features")
        ax.set_ylabel("Maximum Accuracy (%)")
        # ax.set_ylim(0, 100)
        ax.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(f"{path_data_in}MAX_{model_name_filename}_ac_vs_num_indices_{dir_json_results_in}.png")
        # plt.show()

        # exit()
        # group-by for charts
        df_summary = df.groupby("NUM_INDICES")["AC"].max().reset_index()

        # plots
        plt.figure(figsize=(10, 6))
        plt.plot(df_summary["NUM_INDICES"], df_summary["AC"], marker='o')
        plt.title("MAX AC vs Number of Indices")
        plt.xlabel("Number of Indices")
        plt.ylabel("MAX AC")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path_data_in}MAX_{model_name_filename}_ac_vs_num_indices_{dir_json_results_in}.png")
        # plt.show()

        continue


        # exit()

        # grouped = df_json.groupby('METRICS').agg(['count', 'mean', 'std', 'min', 'median', 'max'])
        # Flatten the model dictionary into top-level columns
        rows = []
        for row in df_json.to_dict(orient='records'):
            metrics = row.get("METRICS")
            for model_name, values in row.items():
                if model_name != "METRICS":
                    flat_row = values.copy()
                    flat_row["MODEL"] = model_name # optional the model name here..
                    flat_row["METRICS"] = metrics
                    rows.append(flat_row)

        df_flat = pd.DataFrame(rows)

        # Now group by METRICS
        grouped = df_flat.groupby("METRICS")#.agg(['count', 'mean', 'min', 'median', 'max'])

        # Optional: Save to LaTeX or CSV
        # grouped.to_csv(f"{path_data_in}metrics_grouped_summary_{dir_json_results_in}.csv")

        # exit()



        configfiles = [(dirpath.split('/')[-1], os.path.join(dirpath, f))
                       for dirpath, dirnames, files in os.walk(path_data_in)
                       for f in files if f.endswith(f_ext_in)]
        # print(path_data, path_data_in, path_out, configfiles[:3])
        # exit()

        print('#### #### HOI FOREST - MODELLING #### ####')
        evaluation_path = f'{path_out}data/{f_pattern_out}/'
        t0 = time.time()
        df_data = None
        ncols = 6016
        accuracy_dict, accuracy_dicttt = {}, {}

        # SOME CONFIG ####
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"

        # Fixed part: always included
        i_fix_metrics = 7
        # fixed_part = metric_names[:-1]
        fixed_part = metric_names[:i_fix_metrics]
        # Variable part: will be combined in all possible ways
        # variable_part = metric_names[-1:]
        variable_part = metric_names[i_fix_metrics:]

        print('#### #### READING DATA FILES #### ####')
        # # exit()
        try:
            print(files_path[0])
            exit()
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
            stats_df.to_csv(f'{evaluation_path}df_statistics_{job_id}_{str(datetime.date.today())}.csv', index=True)
            # exit()

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
        models = dict_models.get(0, [cnn_lstm]) if dev_mode else dict_models.get(22, [cnn_lstm])

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

        with open(f'{evaluation_path}000_evaluation_dict_EPOCHS_{num_epochs}_COMBI_{i_r_c}_'
                  f'JOBID_{job_id}_{str(datetime.date.today())}.json',
                  'w') as fp:
            json.dump(accuracy_dict, fp, sort_keys=True, indent=4)

        # with open(f'{evaluation_path}000_models_accuracy_dicttt_{job_id}_{str(datetime.date.today())}.json', 'w') as fp:
        #    json.dump(accuracy_dicttt, fp, sort_keys=True, indent=4)

        print('####', 'TIME', '####', 'TERMINO', '####', round(time.time() - t00, 3), '####')
        print('####', 'FINITO', '####', 'TERMINO', '####', 'NO-VA-MAS', '####')
        print('#### TIMES #### modelling TOTAL TOTAL ==>>', round(time.time() - t00, 3))


if __name__ == '__main__':
    evaluation_soundscapes()
    print('Mirá ve.. oís?? alles gut oder was??')
