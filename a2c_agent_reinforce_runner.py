import numpy as np
import torch
from datetime import datetime
import time

from NetworkFeatureExtraction.src.ModelClasses.NetX.netX import NetX  # required for compatibility with `torch.load`
from src.A2C_Agent_Reinforce import A2CAgentReinforce
from src.NetworkEnv import *
import src.utils as utils
from src.Configuration.StaticConf import StaticConf


def evaluate_model(mode, agent, train_dict=None, test_dict=None, fold_idx="N/A"):
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

    # Use intra-model evaluation if no cross-validation dicts are provided
    if not all([train_dict, test_dict]):
        train_dict = conf.input_dict
        test_dict = conf.input_dict

    env = NetworkEnv(train_dict, mode, fold_idx)

    for model_idx, (net_path, (model, (train_loader, val_loader, test_loader))) in enumerate(test_dict.items()):
        utils.print_flush(f"Evaluating model: {net_path}, {model_idx=}/{len(env.networks)}")

        # Reset environment with test model instead of selecting from train_dict
        state = env.reset(test_net_path=net_path, test_model=model,
                          test_loaders=(train_loader, val_loader, test_loader))
        done = False
        env.t_start = time.perf_counter()

        while not done:
            # Get action distribution from the agent
            action_dist = agent.actor_model(state)

            action = action_dist.sample()  # Compression Rate
            compression_rate = conf.compression_rates_dict[action.cpu().numpy()[0]]
            next_state, reward, done = env.step(compression_rate)
            state = next_state


def main():
    """ Main function for training and evaluating the A2C agent. """
    conf = StaticConf.get_instance().conf_values
    agent = A2CAgentReinforce()

    utils.print_flush(f"Starting test: {conf.test_name}")

    # Agent training is skipped if both actor and critic checkpoints are provided (Agent is pretrained)
    if not all([conf.actor_checkpoint_path, conf.critic_checkpoint_path]):
        utils.print_flush("Starting Agent training")
        agent.train()
        utils.print_flush("Done Agent training")
    else:
        utils.print_flush(f"Agent is pre-trained, training is skipped (actor_checkpoint={conf.actor_checkpoint_path}, "
                          f"critic_checkpoint={conf.critic_checkpoint_path}")

    # Perform standard intra-model evaluation
    for mode in [EVAL_TRAIN, EVAL_TEST]:
        utils.print_flush(f"Evaluating {mode.split('_')[1]} DataLoader")
        evaluate_model(mode, agent)
        utils.print_flush(f"DONE evaluating {mode.split('_')[1]} datasets")

    # Optionally, perform inter-model evaluation via cross-validation
    if conf.n_splits:  # Default is 0 (no CV), recommended value is 10
        utils.print_flush(f"Starting {conf.n_splits}-Fold Cross-Validation")
        folds = utils.get_cross_validation_splits(conf.input_dict)

        for fold_idx, (train_dict, test_dict) in enumerate(folds):
            utils.print_flush(f"Cross-Validation Fold {fold_idx + 1}")
            evaluate_model(EVAL_TEST, agent, train_dict, test_dict, fold_idx + 1)
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
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    utils.init_conf_values(
        # args.input_dict is left out due to file name's length limitation
        test_name=f'SPECTRA{train_compressed_layer_only}_acc-red_{args.allowed_reduction_acc}_'
                  f'gamma_{args.discount_factor}_lr_{args.learning_rate}_rollout-lim_{args.rollout_limit}_'
                  f'num-epochs_{args.num_epochs}{passes}_comp-rates_{args.compression_rates}_{n_splits}'
                  f'train_{args.train_split}_val_{args.val_split}_seed_{args.seed}_{dt_string}',
        input_dict=utils.parse_input_argument(args.input, args.train_split, args.val_split),
        database_dict=utils.parse_input_argument(args.database, args.train_split, args.val_split),
        actor_checkpoint_path=args.actor_checkpoint_path,
        critic_checkpoint_path=args.critic_checkpoint_path,
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
        save_pruned_checkpoints=args.save_pruned_checkpoints,
        test_ts=dt_string
    )

    main()
