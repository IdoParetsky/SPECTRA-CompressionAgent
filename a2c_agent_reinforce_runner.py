import numpy as np
import torch
from pandas import DataFrame
from datetime import datetime
from time import time

from NetworkFeatureExtraction.src.ModelClasses.NetX.netX import NetX  # required for compatibility with `torch.load`
from src.A2C_Agent_Reinforce import A2CAgentReinforce
from src.NetworkEnv import NetworkEnv
import src.utils as utils
from src.Configuration.StaticConf import StaticConf


def evaluate_model(mode, agent):
    # TODO: Consider implementing CV of groups of arch-dataset pairs (NEON impl. below)
    #  Iteratively run across all 'fold' options for comparison (alternatively - add 'fold' arg)
    # np.random.shuffle(datasets)
    # num_of_folds = 6
    #
    # flatten = lambda l: [item for sublist in l for item in sublist]
    #
    # all_datasets_splitted = [datasets[i:i + num_of_folds] for i in range(0, len(datasets), num_of_folds)]
    # test_datasets = all_datasets_splitted[fold]
    # train_datasets = flatten([*all_datasets_splitted[:fold], *all_datasets_splitted[fold + 1:]]

    env = NetworkEnv(mode)  # TODO: implement accordingly - utilize train / val / test / all datasets
    conf = StaticConf.get_instance().conf_values

    results = DataFrame(columns=['model', 'new_acc', 'origin_acc', 'new_param (M)',
                                 'origin_param (M)', 'new_flops (M)', 'origin_flops (M)',
                                 'new_model_arch', 'origin_model_arch', 'evaluation_time'])

    for i in range(len(env.all_networks)):
        utils.print_flush(i)
        state = env.reset()
        done = False
        step_count = 0
        t_start = time()

        while not done and (conf.rollout_limit is None or step_count < conf.rollout_limit):
            # Get action distribution from the agent
            action_dist = agent.actor_model(state)

            action = action_dist.sample()  # Compression Rate
            compression_rate = conf.compression_rates_dict[action.cpu().numpy()[0]]
            next_state, reward, done = env.step(compression_rate)
            state = next_state

            step_count += 1

        t_end = time()

        # Evaluate performance and model size of the new and original models
        new_lh = env.create_learning_handler(env.current_model)
        origin_lh = env.create_learning_handler(env.loaded_model.model)

        model_name = env.all_networks[env.net_order[env.curr_net_index - 1]][1]

        results = results.append(
            {'model': model_name,
             'new_acc': round(new_lh.evaluate_model(env.test_loader), 3),
             'origin_acc': round(origin_lh.evaluate_model(env.test_loader), 3),
             'new_param (M)': round(utils.calc_num_parameters(env.current_model) / 1e6, 3),
             'origin_param (M)': round(utils.calc_num_parameters(env.loaded_model.model) / 1e6, 3),
             'new_flops (M)': round(utils.calc_flops(env.current_model) / 1e6, 3),
             'origin_flops (M)': round(utils.calc_flops(env.loaded_model.model) / 1e6, 3),
             'new_model_arch': utils.get_model_layers_str(env.current_model),
             'origin_model_arch': utils.get_model_layers_str(env.loaded_model.model),
             'evaluation_time': t_end - t_start},
            ignore_index=True)

    return results


def main():
    """ Main function for training and evaluating the A2C agent. """
    conf = StaticConf.get_instance().conf_values
    agent = A2CAgentReinforce()
    utils.print_flush("Starting training")
    agent.train()
    utils.print_flush("Done training")

    utils.print_flush("Evaluating train datasets")

    mode = 'train'
    results = evaluate_model(mode, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{conf.test_name}_{mode}.csv")

    utils.print_flush("DONE evaluating train datasets")

    utils.print_flush("Evaluating test datasets")
    mode = 'test'
    results = evaluate_model(mode, agent)
    results.to_csv(f"./models/Reinforce_One_Dataset/results_{conf.test_name}_{mode}.csv")
    utils.print_flush("DONE evaluating test datasets")


if __name__ == "__main__":
    args = utils.extract_args_from_cmd()
    utils.print_flush(args)

    assert args.train_split + args.val_split < 1, f"{args.train_split=} + {args.val_split=} >= 1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    passes = f'_{args.passes}-passes' if args.passes else ""
    train_compressed_layer_only = "_train_compressed-layer-only" if args.train_compressed_layer_only else ""
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    utils.init_conf_values(
        # args.input_dict is left out due to file name's length limitation
        test_name=f'SPECTRA{train_compressed_layer_only}_acc-red_{args.allowed_reduction_acc}_'
                  f'gamma_{args.discount_factor}_lr_{args.learning_rate}_rollout-lim_{args.rollout_limit}_'
                  f'num-epochs_{args.num_epochs}{passes}_comp-rates_{args.compression_rates}_'
                  f'train_{args.train_split}_val_{args.val_split}_{dt_string}',
        input_dict=utils.parse_input_argument(args.input, args.train_split, args.val_split),
        compression_rates_dict=utils.parse_compression_rates(args.compression_rates),
        is_train_compressed_layer_only=args.is_train_compressed_layer_only,
        total_allowed_accuracy_reduction=args.total_allowed_accuracy_reduction,
        discount_factor=args.discount_factor,
        learning_rate=args.learning_rate,
        rollout_limit=args.rollout_limit,
        passes=args.passes,
        prune=args.prune,
        num_epochs=args.num_epochs,
        runtime_limit=args.runtime_limit,
        train_split=args.train_split,
        val_split=args.val_split,
        actor_checkpoint_path=args.actor_checkpoint_path,
        critic_checkpoint_path=args.critic_checkpoint_path
    )

    main()
