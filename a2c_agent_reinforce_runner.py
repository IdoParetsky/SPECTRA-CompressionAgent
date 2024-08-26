from NetworkFeatureExtration.src.ModelClasses.NetX.netX import NetX # Must be imported, called from torch.serialization line 1431

import argparse
import sys

from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
from pandas import DataFrame

from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from src.A2C_Agent_Reinforce import A2C_Agent_Reinforce
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from src.NetworkEnv import NetworkEnv
from src.utils import print_flush, load_models_path, get_model_layers_str


def init_conf_values(compression_rates_dict, num_epoch=100, is_learn_new_layers_only=False,
                     total_allowed_accuracy_reduction=5, increase_loops_from_1_to_4=True, prune=False):
    if not torch.cuda.is_available():
        sys.exit("GPU was not allocated!!!!")

    device = torch.device("cuda:0")
    print_flush(f"device is {device}")
    print_flush(f"device name is {torch.cuda.get_device_name(0)}")

    MAX_TIME_TO_RUN = 60 * 60 * 24 * 2.5
    cv = ConfigurationValues(device, compression_rates_dict=compression_rates_dict,
                             num_epoch=num_epoch,
                             is_learn_new_layers_only=is_learn_new_layers_only,
                             total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                             increase_loops_from_1_to_4=increase_loops_from_1_to_4,
                             MAX_TIME_TO_RUN=MAX_TIME_TO_RUN,
                             prune=prune)
    StaticConf(cv)


def \
        evaluate_model(mode, base_path, agent):
    models_path = load_models_path(base_path, mode)
    env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.increase_loops_from_1_to_4)
    compression_rates_dict = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param',
                                 'origin_param', 'new_model_arch', 'origin_model_arch'])

    for i in range(len(env.all_networks)):
        print_flush(i)
        state = env.reset()
        total_values = sum(arr.size for tuple_arrays in state for arr in tuple_arrays)
        nan_count = sum(np.isnan(arr).sum() for tuple_arrays in state for arr in tuple_arrays)

        # Calculate the percentage of NaN values
        nan_percentage = (nan_count / total_values) * 100

        # Print the results
        print(f"Number of NaN values in the list of tuples: {nan_count}")
        print(f"Percentage of NaN values: {nan_percentage:.2f}%")
        state = [tuple(np.nan_to_num(arr, nan=0.0) for arr in tuple_arrays) for tuple_arrays in state]  # TODO: Understarnd why there are NaNs!
        done = False

        while not done:
            # Get action distribution and value prediction from the agent
            # alt. implementation - action_dist, value_pred = agent.actor_critic_model(state)
            value_pred = agent.critic_model(state)
            action_dist = agent.actor_model(state)

            action = action_dist.sample()  # Compression Rate
            compression_rate = compression_rates_dict[action.cpu().numpy()[0]]
            next_state, reward, done = env.step(compression_rate)
            state = next_state
            total_values = sum(arr.size for tuple_arrays in state for arr in tuple_arrays)
            nan_count = sum(np.isnan(arr).sum() for tuple_arrays in state for arr in tuple_arrays)

            # Calculate the percentage of NaN values
            nan_percentage = (nan_count / total_values) * 100

            # Print the results
            print(f"Number of NaN values in the list of tuples: {nan_count}")
            print(f"Percentage of NaN values: {nan_percentage:.2f}%")
            state = [tuple(np.nan_to_num(arr, nan=0.0) for arr in tuple_arrays) for tuple_arrays in state]  # TODO: Understarnd why there are NaNs!

        # Evaluate performance and model size of the new and original models
        new_lh = env.create_learning_handler(env.current_model)
        origin_lh = env.create_learning_handler(env.loaded_model.model)

        new_acc = new_lh.evaluate_model()
        origin_acc = origin_lh.evaluate_model()

        new_params = env.calc_num_parameters(env.current_model)
        origin_params = env.calc_num_parameters(env.loaded_model.model)

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        new_model_with_rows = ModelWithRows(env.current_model)

        results = results.append({'model': model_name,
                                  'new_acc': new_acc,
                                  'origin_acc': origin_acc,
                                  'new_param': new_params,
                                  'origin_param': origin_params,
                                  'new_model_arch': get_model_layers_str(env.current_model),
                                  'origin_model_arch': get_model_layers_str(env.loaded_model.model)}, ignore_index=True)

    return results


def split_dataset_to_train_test(path):
    models_path = load_models_path(path, 'all')
    all_models = models_path[0][1]
    all_models = list(map(os.path.basename, all_models))
    train_models, test_models = train_test_split(all_models, test_size=0.2)

    df_train = DataFrame(data=train_models)
    df_train.to_csv(os.path.join(path, "train_models.csv"))

    df_test = DataFrame(data=test_models)
    df_test.to_csv(os.path.join(path, "test_models.csv"))


def main(dataset_name, is_learn_new_layers_only, test_name,
         total_allowed_accuracy_reduction, is_to_split_cv=True, increase_loops_from_1_to_4=False, prune=False):
    compression_rates_dict = {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0.7,
        4: 0.6
    }
    base_path = f"C:/Users/idopa/Documents/BGU/MSc/SPECTRA-CompressionAgent/datasets/SPECTRA-csv/{dataset_name}/"  # SPECTRA
    #base_path = f"C:/Users/idopa/Documents/BGU/MSc/SPECTRA-CompressionAgent/datasets/NEON-csv/{dataset_name}/"  # NEON
    #base_path = f"C:/Users/idopa/Documents/BGU/MSc/SPECTRA-CompressionAgent/datasets/{dataset_name}/"  # NEON orig structure

    if is_to_split_cv:
        split_dataset_to_train_test(base_path)

    init_conf_values(compression_rates_dict, is_learn_new_layers_only=is_learn_new_layers_only, num_epoch=100,
                     total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
                     increase_loops_from_1_to_4=increase_loops_from_1_to_4,
                     prune=prune)
    models_path = load_models_path(base_path, 'train')

    agent = A2C_Agent_Reinforce(models_path, test_name)
    print_flush("Starting training")
    agent.train()
    print_flush("Done training")

    print_flush("Evaluating train datasets")

    mode = 'train'
    results = evaluate_model(mode, base_path, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")

    print_flush("DONE evaluating train datasets")

    print_flush("Evaluating test datasets")
    mode = 'test'
    results = evaluate_model(mode, base_path, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{test_name}_{mode}.csv")
    print_flush("DONE evaluating test datasets")


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_name', type=str, const=True, default='svhn', nargs='?')
    parser.add_argument('--dataset_name', type=str, default="svhn")
    parser.add_argument('--learn_new_layers_only', type=bool, const=True, default=True, nargs='?')
    parser.add_argument('--split', type=bool, const=True, default=True, nargs='?')
    parser.add_argument('--allowed_reduction_acc', type=int, default=5, nargs='?')  # Another recommended default is 1 (in precents)
    parser.add_argument('--increase_loops_from_1_to_4', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--prune', type=bool, const=True, default=False, nargs='?')
    parser.add_argument('--seed', type=int, const=True, default=0, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    print_flush(args)
    with_loop = '_with_loop' if args.increase_loops_from_1_to_4 else ""
    test_name = f'Agent_warmup_{args.dataset_name}_learn_new_layers_only_{args.learn_new_layers_only}_acc_reduction_{args.allowed_reduction_acc}{with_loop}'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(dataset_name=args.dataset_name, is_learn_new_layers_only=args.learn_new_layers_only, test_name=test_name,
         is_to_split_cv=args.split,
         total_allowed_accuracy_reduction=args.allowed_reduction_acc,
         increase_loops_from_1_to_4=args.increase_loops_from_1_to_4,
         prune=args.prune)
