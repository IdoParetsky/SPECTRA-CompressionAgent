import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
import time
import warnings


from NetworkFeatureExtraction.src.ModelClasses.NetX.netX import NetX  # required for compatibility with `torch.load`
from src.A2C_Agent_Reinforce import A2CAgentReinforce
from src.NetworkEnv import *
import src.utils as utils
import src.distributed as ddp
import src.logging_utils as logging_utils
import src.run_recorder as run_recorder
from src.Configuration.StaticConf import StaticConf

# Distributed is opt-in: a plain `python a2c_agent_reinforce_runner.py ...` on a single
# rtx_6000 never initialises NCCL. Launch with torchrun/srun to enable multi-GPU.
ddp.maybe_init_distributed()
rank = ddp.get_rank()
# Optional PyCharm remote debug (set SPECTRA_PYDEVD=1). Off by default for Cursor/SLURM runs.
if os.environ.get("SPECTRA_PYDEVD", "").strip() in ("1", "true", "True"):
    port = 12345 + rank  # each rank gets its own port, initial value chosen arbitrarily
    utils.print_flush(f"[Rank {rank}] Connecting debugger on port {port}")
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True, suspend=False)

import logging
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)


def evaluate_model(mode, agent, train_dict=None, test_dict=None, fold_idx="N/A"):
    """
    Evaluate models using intra-model (train/test) and inter-model (cross-validation).

    Args:
        mode (str):                 'train' or 'test' (used for intra-model evaluation).
        agent (A2CAgentReinforce):  Trained RL agent.
        train_dict (dict):          {network_path: (model, dataset_name)} for training
                                    (Used for inter-model evaluation via cross-validation).
        test_dict (dict):           {network_path: (model, dataset_name)} for testing
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

    # Under a multi-GPU launch each rank evaluates a disjoint slice of the networks and
    # writes its own results file, so the work is split rather than duplicated
    world_size, rank = ddp.get_world_size(), ddp.get_rank()
    shard = list(test_dict.items())[rank::world_size]
    if world_size > 1:
        utils.print_flush(f"Rank {rank} evaluating {len(shard)}/{len(test_dict)} networks")

    for model_idx, (net_path, (net_model, net_loaders)) in enumerate(shard):
        utils.print_flush(f"Evaluating model {model_idx + 1}/{len(shard)}: {net_path}")

        # One failing network must not abort the whole evaluation sweep: the remaining
        # networks still produce results, and the failure is recorded for triage.
        try:
            with logging_utils.context(net=os.path.basename(net_path), phase=mode):
                # Reset environment with test model instead of selecting from train_dict
                env.t_start = time.perf_counter()
                state = env.reset(test_net_path=net_path, test_model=net_model, test_loaders=net_loaders)
                done = False

                env._budget_logged = False
                while not done:
                    action_dist = agent.actor_model(state)
                    legal = env.legal_action_mask(device=conf.device)
                    from src import fortify as fortify_mod
                    action_dist = fortify_mod.apply_action_mask(action_dist, legal)

                    min_ratio = fortify_mod.eval_min_param_ratio()
                    at_budget = env.param_ratio() <= min_ratio
                    if at_budget:
                        if not env._budget_logged:
                            utils.print_flush(
                                f"[eval] param budget x{env.param_ratio():.3f} <= {min_ratio}; "
                                f"identity-pad remaining steps")
                            env._budget_logged = True
                        identity = next(
                            (i for i, r in conf.compression_rates_dict.items()
                             if abs(float(r) - 1.0) < 1e-9),
                            0)
                        action = torch.tensor([identity], device=conf.device)
                    else:
                        action = action_dist.sample()

                    compression_rate = conf.compression_rates_dict[int(action.item())]
                    next_state, reward, done = env.step(compression_rate)
                    state = next_state
        except Exception as error:
            logging_utils.exception(f"Evaluation of {net_path} failed; continuing with the rest")
            run_recorder.issue("eval_network_failed", f"{type(error).__name__}: {error}",
                               network=net_path, eval_mode=mode)


def main():
    """ Main function for training and evaluating the A2C agent. """
    conf = StaticConf.get_instance().conf_values

    with logging_utils.stage("agent.construct"):
        agent = A2CAgentReinforce()

    utils.print_flush(f"Starting test: {conf.test_name}")

    # Both actor+critic paths historically meant "eval only". For warm-start continued
    # training set SPECTRA_CONTINUE_TRAIN=1 (loads weights, still runs agent.train()).
    continue_train = os.environ.get("SPECTRA_CONTINUE_TRAIN", "").strip().lower() in (
        "1", "true", "yes")
    pretrained = bool(conf.actor_checkpoint_path and conf.critic_checkpoint_path)
    if pretrained and not continue_train:
        utils.print_flush(
            f"Agent is pre-trained, training is skipped "
            f"(actor_checkpoint={conf.actor_checkpoint_path}, "
            f"critic_checkpoint={conf.critic_checkpoint_path}). "
            f"Set SPECTRA_CONTINUE_TRAIN=1 to warm-start training from these weights.")
    else:
        if pretrained and continue_train:
            utils.print_flush(
                f"Warm-start training from actor={conf.actor_checkpoint_path} "
                f"critic={conf.critic_checkpoint_path}")
        with logging_utils.stage("phase.train"):
            with logging_utils.context(phase="train"):
                agent.train()

    # Perform standard intra-model evaluation
    for mode in [EVAL_TRAIN, EVAL_TEST]:
        with logging_utils.stage(f"phase.{mode}"):
            with logging_utils.context(phase=mode):
                evaluate_model(mode, agent)

    # Optionally, perform inter-model evaluation via cross-validation
    if conf.n_splits:  # Default is 0 (no CV), recommended value is 10
        utils.print_flush(f"Starting {conf.n_splits}-Fold Cross-Validation")
        folds = utils.get_cross_validation_splits(conf.input_dict)

        for fold_idx, (train_dict, test_dict) in enumerate(folds):
            with logging_utils.stage(f"phase.cv_fold_{fold_idx + 1}"):
                with logging_utils.context(phase="cv", fold=fold_idx + 1):
                    evaluate_model(EVAL_TEST, agent, train_dict, test_dict, fold_idx + 1)
        utils.print_flush("DONE Cross-Validation")


if __name__ == "__main__":
    # Logging is configured before anything else so that even an argument-parsing failure or
    # an import-time crash lands in the run directory with a full traceback.
    RUN_DIR = logging_utils.setup()
    logging_utils.start_heartbeat()
    utils.print_flush(f"SPECTRA run {logging_utils.run_id()} -> {RUN_DIR}")
    logging_utils.log_environment()

    # Anomaly detection roughly triples backward-pass cost, so it is opt-in via
    # SPECTRA_DETECT_ANOMALY=1 rather than always-on.
    if os.environ.get("SPECTRA_DETECT_ANOMALY", "").strip() in ("1", "true", "True"):
        torch.autograd.set_detect_anomaly(True)

    # GPUs have Tensor Cores capable of speeding up float32 matmul ops (used in conv/linear layers),
    # but PyTorch doesn't enable them by default. This increases performance without needing AMP
    torch.set_float32_matmul_precision('high')

    # Stabilizing torch.compile() in practice
    torch._dynamo.config.suppress_errors = True

    # Helps Conv2D tune performance across batch shapes
    torch.backends.cudnn.benchmark = True

    #Filtering out a harmless warning
    warnings.filterwarnings("ignore", message="xindex is not in var_ranges")

    args = utils.extract_args_from_cmd()
    utils.print_flush(args)

    assert args.train_split + args.val_split < 1, f"{args.train_split=} + {args.val_split=} >= 1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    passes = f'_passes_{args.passes}' if args.passes else ""
    n_splits = f'n_splits_{args.n_splits}_' if args.n_splits else ""
    train_compressed_layer_only = "_train_compressed-layer-only" if args.train_compressed_layer_only else ""
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace("/", "-").replace(":", "-")

    # Every dataset is materialised once here and shared by all networks referencing it
    with logging_utils.stage("startup.preload_datasets", datasets=",".join(args.datasets or [])):
        preloaded_dataloaders_dict = utils.preload_datasets(args.datasets, args.train_split, args.val_split)

    utils.init_conf_values(
        # args.input_dict and args.database are left out due to file name's length limitation
        test_name=f'SPECTRA{train_compressed_layer_only}_acc-red_{args.allowed_acc_reduction}_'
                  f'gamma_{args.discount_factor}_lr_{args.learning_rate}_rollout-lim_{args.rollout_limit}_'
                  f'num-epochs_{args.num_epochs}{passes}_comp-rates_{args.compression_rates}_{n_splits}'
                  f'train_{args.train_split}_val_{args.val_split}_seed_{args.seed}_{dt_string}',
        dataloaders_dict=preloaded_dataloaders_dict,
        input_dict=utils.parse_input_argument(args.input, preloaded_dataloaders_dict),
        database_dict=utils.parse_input_argument(args.database, preloaded_dataloaders_dict),
        actor_checkpoint_path=args.actor_checkpoint_path,
        critic_checkpoint_path=args.critic_checkpoint_path,
        compression_rates_dict=utils.parse_compression_rates(args.compression_rates),
        train_compressed_layer_only=args.train_compressed_layer_only,
        allowed_acc_reduction=args.allowed_acc_reduction,
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

    # The manifest pins down what was run (config, git commit, GPUs, SPECTRA_* switches), so a
    # results directory can be interpreted months later without the launch command.
    recorder_instance = run_recorder.RunRecorder.instance()
    recorder_instance.write_manifest({
        "argv": sys.argv,
        "args": vars(args),
        "config": run_recorder.config_snapshot(StaticConf.get_instance().conf_values),
    })
    run_recorder.record("run_start", test_name=StaticConf.get_instance().conf_values.test_name)

    RUN_STARTED = time.perf_counter()
    try:
        main()
    except Exception as fatal:
        # The excepthook logs the traceback; this records the terminal status so a summary of
        # a failed run still reports how far it got and why it stopped.
        run_recorder.record("run_end", status="failed",
                            error=f"{type(fatal).__name__}: {fatal}",
                            seconds=round(time.perf_counter() - RUN_STARTED, 2))
        recorder_instance.close()
        raise
    else:
        run_recorder.record("run_end", status="ok",
                            seconds=round(time.perf_counter() - RUN_STARTED, 2),
                            counters=recorder_instance.counters())
        utils.print_flush(f"Run finished in {(time.perf_counter() - RUN_STARTED) / 60:.1f} min; "
                          f"artefacts in {RUN_DIR}")
    finally:
        logging_utils.stop_heartbeat()
        recorder_instance.close()
