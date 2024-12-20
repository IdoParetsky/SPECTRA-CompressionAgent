# from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX - must be import!!!!
import glob
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from torch import nn
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX
from src.NetworkEnv import NetworkEnv
import torch.nn.utils.prune as prune

from src.utils import load_models_path, print_flush, save_times_csv


def init_conf_values(compression_rates_dict, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=1, increase_loops_from_1_to_4=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_actions = len(compression_rates_dict)
    cv = ConfigurationValues(device, compression_rates_dict=compression_rates_dict, num_actions=num_actions,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             increase_loops_from_1_to_4=increase_loops_from_1_to_4)
    StaticConf(cv)


torch.manual_seed(0)
np.random.seed(0)


def get_linear_layer(row):
    for l in row:
        if type(l) is nn.Linear:
            return l


def get_model_layers(model):
    new_model_with_rows = ModelWithRows(model)
    linear_layers = [(get_linear_layer(x).in_features, get_linear_layer(x).out_features) for x in
                     new_model_with_rows.all_rows]
    return str(linear_layers)


def prune_model(env: NetworkEnv, prune_percentage):
    accuracies_wp = []
    accuracies_np = []

    # Get the accuracy without any pruning
    initial_accuracy = env.original_acc
    # accuracies_np.append(initial_accuracy)

    model = deepcopy(env.current_model)

    # for _ in range(4):
    parameters_to_prune = []

    for l in list(model.modules()):
        if type(l) is torch.nn.Linear:
            parameters_to_prune.append((l, 'weight'))
            # prune.l1_unstructured(l, name='weight', amount=prune_percentage)

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount = prune_percentage)
    new_lh = env.create_learning_handler(model)
    new_lh.unfreeze_all_layers()
    new_lh.train_model()
    new_acc = new_lh.evaluate_model()

    num_params = calc_num_parameters(model)
    pruned_params = sum(list(
        map(lambda x: (x.weight_mask == 0).sum(), filter(lambda x: hasattr(x, 'weight_mask'), model.modules()))))
    pruned_params = int(pruned_params)

    accuracies_np.append((num_params, pruned_params, new_acc))

    return max(accuracies_np, key = lambda x: x[2])


def calc_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(mode, base_path, curr_prune_percentage):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.increase_loops_from_1_to_4)

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        state = env.reset()
        # print(i)
        origin_lh = env.create_learning_handler(env.loaded_model.model)
        origin_acc = origin_lh.evaluate_model()

        num_params, pruned_params, new_acc = prune_model(env, curr_prune_percentage)

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append({'model': model_name,
                                  'new_acc': new_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': num_params - pruned_params,
                                  'origin_param': num_params,
                                  'new_model_arch': get_model_layers(env.current_model),
                                  'origin_model_arch': get_model_layers(env.loaded_model.model)}, ignore_index=True)

    return results


def main(dataset_name, test_name):
    actions = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"./OneDatasetLearning/Classification/{dataset_name}/"

    init_conf_values(actions, num_epoch=5)
    prune_percentages = [.80]

    for curr_prune_percentage in prune_percentages:
        mode = 'all'
        results = evaluate_model(mode, base_path, curr_prune_percentage)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}_pp_{curr_prune_percentage}.csv")

        # mode = 'train'
        # results = evaluate_model(mode, base_path, curr_prune_percentage)
        # results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}_pp_{curr_prune_percentage}.csv")


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_name', type=str)
    # parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=False, nargs='?')
    # parser.add_argument('--split', type=bool, const=True, default=False, nargs='?')
    # parser.add_argument('--increase_loops_from_1_to_4', type=bool, const=True, default=False, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print_flush("Pruning")
    args = extract_args_from_cmd()
    all_datasets = glob.glob("./OneDatasetLearning/Classification/*")
    all_times = []

    for idx, curr_dataset in enumerate(all_datasets):
        dataset_name = os.path.basename(curr_dataset)
        print_flush(f"{dataset_name} {idx} / {len(all_datasets)}")
        test_name = f'Agent_{dataset_name}_pruning2'
        now = datetime.now()
        main(dataset_name=dataset_name, test_name=test_name)
        total_time = (datetime.now() - now).total_seconds()
        all_times.append(total_time)

    save_times_csv(test_name, all_times, all_datasets)
