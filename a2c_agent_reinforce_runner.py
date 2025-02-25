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

TRAIN = "train"
TEST = "test"


def evaluate_model(mode, agent, train_dict=None, test_dict=None, fold_idx=None):
    """
    Evaluate models using intra-model (train/test) and inter-model (cross-validation).

    Args:
        mode (str):                 'train' or 'test' (used for intra-model evaluation).
        agent (A2CAgentReinforce):  Trained RL agent.
        train_dict (dict):          {network_path: (model, (train_loader, val_loader, test_loader))} for training
                                    (Used for inter-model evaluation via cross-validation).
        test_dict (dict):           {network_path: (model, (train_loader, val_loader, test_loader))} for testing
                                    (Used for inter-model evaluation via cross-validation).
        fold_idx (int, optional):   The index of the cross-validation fold.

    Returns:
        DataFrame: Evaluation results.
    """
    conf = StaticConf.get_instance().conf_values
    results = DataFrame(columns=['model', 'fold', 'new_acc', 'origin_acc', 'new_param (M)',
                                 'origin_param (M)', 'new_flops (M)', 'origin_flops (M)',
                                 'new_model_arch', 'origin_model_arch', 'evaluation_time'])

    # Use intra-model evaluation if no cross-validation dicts are provided
    if not all([train_dict, test_dict, fold_idx]):
        train_dict = conf.input_dict
        test_dict = conf.input_dict
        fold_idx = "N/A"  # Not applicable for intra-model evaluation

    env = NetworkEnv(train_dict)

    fold_str = f"fold {fold_idx}, " if isinstance(fold_idx, int) else ""

    for model_idx, (net_path, (model, (train_loader, val_loader, test_loader))) in enumerate(test_dict.items()):
        utils.print_flush(f"Evaluating {fold_str}model: {net_path}, {model_idx=}/{len(env.networks)}")

        dataset_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[mode]

        # Reset environment with test model instead of selecting from train_dict
        state = env.reset(test_net_path=net_path, test_model=model,
                          test_loaders=(train_loader, val_loader, test_loader))
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

        # Retrieve original & compressed models
        original_model = conf.input_dict[env.networks[env.curr_net_index]][0]
        compressed_model = env.current_model

        # Evaluate performance and model size of the compressed and original models
        new_lh = env.create_learning_handler(compressed_model)
        origin_lh = env.create_learning_handler(original_model)

        results = results.append(
            {'model': net_path,
             'fold': fold_idx,
             'new_acc': round(new_lh.evaluate_model(dataset_loader), 3),
             'origin_acc': round(origin_lh.evaluate_model(dataset_loader), 3),
             'new_param (M)': round(utils.calc_num_parameters(compressed_model) / 1e6, 3),
             'origin_param (M)': round(utils.calc_num_parameters(original_model) / 1e6, 3),
             'new_flops (M)': round(utils.calc_flops(compressed_model) / 1e6, 3),
             'origin_flops (M)': round(utils.calc_flops(original_model) / 1e6, 3),
             'new_model_arch': utils.get_model_layers_str(compressed_model),
             'origin_model_arch': utils.get_model_layers_str(original_model),
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

    # Perform standard intra-model evaluation
    for mode in [TRAIN, TEST]:
        utils.print_flush(f"Evaluating {mode} DataLoader")
        results = evaluate_model(mode, agent)
        results.to_csv(f"./models/Reinforce_One_Dataset/results_{conf.test_name}_{mode}.csv")
        utils.print_flush(f"DONE evaluating {mode} datasets")

    # Optionally, perform inter-model evaluation via cross-validation
    if conf.n_splits:  # Default is 0 (no CV), recommended value is 10
        utils.print_flush(f"Starting {conf.n_splits}-Fold Cross-Validation")
        folds = utils.get_cross_validation_splits(conf.input_dict)

        for fold_idx, (train_dict, test_dict) in enumerate(folds):
            utils.print_flush(f"Cross-Validation Fold {fold_idx + 1}")
            results = evaluate_model(TEST, agent, train_dict, test_dict, fold_idx + 1)
            results.to_csv(f"./models/Reinforce_CV/results_{conf.test_name}_fold_{fold_idx + 1}.csv")

        utils.print_flush("DONE Cross-Validation")


if __name__ == "__main__":
    args = utils.extract_args_from_cmd()
    utils.print_flush(args)

    assert args.train_split + args.val_split < 1, f"{args.train_split=} + {args.val_split=} >= 1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    passes = f'_passes_{args.passes}' if args.passes else ""
    n_splits = f'_n_splits_{args.n_splits}' if args.n_splits else ""
    train_compressed_layer_only = "_train_compressed-layer-only" if args.train_compressed_layer_only else ""
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    utils.init_conf_values(
        # args.input_dict is left out due to file name's length limitation
        test_name=f'SPECTRA{train_compressed_layer_only}_acc-red_{args.allowed_reduction_acc}_'
                  f'gamma_{args.discount_factor}_lr_{args.learning_rate}_rollout-lim_{args.rollout_limit}_'
                  f'num-epochs_{args.num_epochs}{passes}_comp-rates_{args.compression_rates}_{n_splits}'
                  f'train_{args.train_split}_val_{args.val_split}_seed_{args.seed}_{dt_string}',
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
        seed=args.seed,
        n_splits=args.n_splits,
        train_split=args.train_split,
        val_split=args.val_split,
        actor_checkpoint_path=args.actor_checkpoint_path,
        critic_checkpoint_path=args.critic_checkpoint_path,
        save_pruned_checkpoints=args.save_pruned_checkpoints
    )

    main()
