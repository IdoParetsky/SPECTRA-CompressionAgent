import numpy as np
import pandas as pd
import copy
import logging
import os
import time
import gc
import torch
from torch import nn

import src.channel_groups as channel_groups
import src.distributed as ddp
import src.logging_utils as logging_utils
import src.pruning as pruning
import src.run_recorder as recorder
from src.BERTInputModeler import BERTInputModeler
from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
import src.utils as utils

AGENT_TRAIN = "agent_train"  # Mode when NetworkEnv is called from A2C_Agent_Reinforce.py
EVAL_TRAIN = "eval_train"  # Mode when NetworkEnv is called from a2c_agent_reinforce_runner.py, evaluating the train dataset
EVAL_TEST = "eval_test"  # Mode when NetworkEnv is called from a2c_agent_reinforce_runner.py, evaluating the test dataset


def reward_compression_rate(prune_outcome, compression_rate, params_before, params_after,
                            new_acc, original_acc, tau):
    """
    Rate fed to NEON ``compute_reward``. The trichotomy itself is unchanged.

    Masked fallback does not change ``numel``/FLOPs. Crediting ``(1 - rate)`` taught the
    agent to "compress" ShuffleNet / unknown ops that never shrink. In-budget masked
    no-ops therefore use rate 1.0 (zero compression credit). Over-budget accuracy drops
    still use the nominal rate so wrecking an unprunable layer is punished.
    """
    outcome = prune_outcome or {}
    if (outcome.get("mode") == "masked"
            and params_after >= params_before * (1.0 - 1e-12)
            and (new_acc - original_acc) * 100.0 >= -float(tau)):
        return 1.0
    return compression_rate


class NetworkEnv:
    """
    Implements a Reinforcement Learning Environment for structured CNN pruning.

    This environment interacts with an RL agent (e.g., 'A2CAgentReinforce') to iteratively prune convolutional (Conv2D)
    and fully connected (Linear) layers in deep neural networks. The pruning process is guided by reinforcement learning,
    aiming to reduce model complexity while maintaining accuracy.

    The environment is responsible for:
    - Loading pre-trained CNN models and datasets from 'self.conf.input_dict'.
    - Extracting model architecture features using a BERT-based state encoder.
    - Applying pruning actions to individual layers and evaluating their impact.
    - Computing rewards based on accuracy and model efficiency.
    - Logging results and optionally saving pruned models.

    Modes of Operation:
    - 'AGENT_TRAIN': Used when training an RL agent, skipping logging and evaluation.
    - 'EVAL_TRAIN': Used for evaluating pruning effectiveness on training datasets.
    - 'EVAL_TEST': Used for evaluating pruning effectiveness on test datasets.

    Attributes:
        conf (StaticConf): Static configuration instance containing hyperparameters and settings.
        layer_index (int): Index of the layer currently being pruned.
        actions_history (List[float]): History of compression rates applied during pruning.
        original_acc (float): Accuracy of the original, unpruned model (set at reset).
        selected_net_path (str): Path of the current model being evaluated.
        current_model (torch.nn.Module): The CNN model currently being pruned.
        feature_extractor (FeatureExtractor): Extractor for generating model representations for BERT.
        networks (List[str]): List of model paths from 'input_dict', used for selecting models.
                              None (default) - all the networks in conf.input_dict are retrieved.
        curr_net_index (int): Current index in 'networks', tracking which model is loaded.
        bert_modeler (BERTInputModeler): Handles BERT-based feature extraction.
        mode (str): One of 'AGENT_TRAIN', 'EVAL_TRAIN', or 'EVAL_TEST', indicating the instance's context.
        fold_idx (int or str): Fold index for cross-validation or '"N/A"' if not using cross-validation.
        t_start (float): Start time of model evaluation, used for logging and tracking execution time.

    Methods:
        reset(test_net_path=None, test_model=None, test_loaders=None):
            Resets the environment by loading a new model and dataset. If test parameters are provided,
            uses them instead of selecting from 'input_dict' (utilized in cross-validation to evaluate the test dataset
            over a train dataset environment).

        step(compression_rate: float, is_to_train: bool = True) -> Tuple[np.ndarray, float, bool]:
            Applies a pruning action, evaluates the compressed model, and moves to the next state.
            Returns the updated state, computed reward, and termination flag.

        compute_and_log_results(model_with_rows, t_curr: float = time.perf_counter()):
            Computes accuracy, model size, and FLOPs after pruning, logging them to a CSV file.

        save_pruned_checkpoint():
            Saves the final pruned model to a checkpoint file, ensuring the filename is uniquely formatted.

        create_learning_handler(new_model: torch.nn.Module) -> BasicHandler:
            Instantiates a learning handler appropriate for the current mission type, supporting both
            training and testing scenarios. SPECTRA's current implementation supports Classifications tasks only.

    Workflow:
        1. The environment is initialized with a set of models ('input_dict').
        2. The RL agent selects pruning actions via 'step()', reducing model complexity.
        3. After each pruning action, the model is evaluated and a reward is computed.
        4. Once each pruning pass (over all the network's layers) is complete, results are logged.
           Optionally, once the network's pruning process is terminated - a pruned model is saved.
    """

    def __init__(self, networks=None, mode=None, fold_idx="N/A"):
        self.conf = StaticConf.get_instance().conf_values
        self.row_idx = None  # This variable will hold the index of the row after the one to be pruned
        self.actions_history = []
        self.original_acc = None
        self.original_params = None
        self.selected_net_path = None
        self.current_model = None
        self.feature_extractor = None

        # EVAL_TRAIN / EVAL_TEST when called from a2c_agent_reinforce_runner.py's evaluate_model(),
        # used for accuracy calculation in NetworkEnv's compute_and_log_results().
        # AGENT_TRAIN when called from A2C_Agent_Reinforce.py, skipping compute_and_log_results()
        self.mode = mode

        # Full database if in agent training mode, else evaluation database (user input)
        self.data_dict = self.conf.database_dict if self.mode == AGENT_TRAIN else self.conf.input_dict
        # Callers pass either an explicit list of paths or a {path: (model, loaders)} dict
        # (evaluate_model hands over a fold's train_dict). np.random.shuffle cannot operate
        # on a dict, so normalise to a list of paths first.
        if isinstance(networks, dict):
            networks = list(networks.keys())
        self.networks = list(networks) if networks else list(self.data_dict.keys())
        self.curr_net_index = -1
        # Offsetting the seed by the rank makes each process explore a different network, so
        # a two-GPU run gathers two independent trajectories per update instead of duplicating
        # the same one. Runs stay reproducible for a given (seed, world size).
        np.random.default_rng(self.conf.seed + ddp.get_rank()).shuffle(self.networks)

        # Shared state-builder singleton (does not load frozen BERT unless
        # SPECTRA_STATE_ENCODER=bert). FeatureExtractor holds the same instance.
        self.bert_modeler = BERTInputModeler()

        # "N/A" / an integer representing the fold number within the amount of folds in cross-validation evaluation
        # when called from a2c_agent_reinforce_runner.py's evaluate_model(),
        # used for logging via NetworkEnv's compute_and_log_results().
        # None (irrelevant) when called from A2C_Agent_Reinforce.py.
        self.fold_idx = fold_idx

        # t_start is assigned in a2c_agent_reinforce_runner.py's evaluate_model(),
        # and utilized in NetworkEnv's compute_and_log_results()
        self.t_start = None  # a Model's evaluation start time

    def reset(self, test_net_path=None, test_model=None, test_loaders=None):
        """ Reset environment with a new CNN model & dataset """
        # Ensure prior memory is cleaned
        if hasattr(self, "feature_extractor"):
            del self.feature_extractor
        if hasattr(self, "current_model"):
            del self.current_model
        torch.cuda.empty_cache()
        gc.collect()

        self.row_idx = 1  # The first row to be a candidate for pruning is self.row_idx - 1 -> index 0
        self.actions_history = []

        # If a specific network is requested, use it directly (evaluation / cross-validation).
        # Previously all three arguments had to be supplied for this branch to be taken, so a
        # caller passing only test_net_path silently evaluated whatever network happened to be
        # next in this environment's own rotation.
        if test_net_path:
            self.selected_net_path = test_net_path
            if test_model is not None and test_loaders is not None:
                model, loaders = test_model, test_loaders
            else:
                model, loaders = self.data_dict[test_net_path]
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = model, loaders
        else:
            self.curr_net_index = (self.curr_net_index + 1) % len(self.networks)
            self.selected_net_path = self.networks[self.curr_net_index]

            # Load model & dataset from preloaded input_dict
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = self.data_dict[
                self.selected_net_path]

        # Each episode restarts from the pristine checkpoint, so the compression applied by the
        # previous episode does not leak into this one
        self.current_model = copy.deepcopy(self.current_model)
        self.original_params = utils.calc_num_parameters(self.current_model)

        model_with_rows = ModelWithRows(self.current_model)

        # Every log line and event emitted for this episode is tagged with the network under
        # compression, so a failure can be attributed without reading back through the file
        logging_utils.set_context(net=os.path.basename(self.selected_net_path), mode=self.mode)
        utils.print_flush(f"Loading {self.selected_net_path}")

        # Prepare feature extractor with training data
        self.feature_extractor = FeatureExtractor(self.train_loader, self.conf.device)
        with logging_utils.stage("reset.feature_extraction"):
            fm = self.feature_extractor.encode_to_bert_input(
                model_with_rows, model_with_rows.row_to_main_layer[self.row_idx - 1],
                dependency_groups=self._dependency_groups(model_with_rows),
                param_ratio=1.0)

        # Evaluate original model accuracy
        learning_handler_original_model = self.create_learning_handler(self.current_model)
        with logging_utils.stage("reset.baseline_accuracy"):
            self.original_acc = learning_handler_original_model.evaluate_model(self.val_loader)

        num_rows = max(len(model_with_rows.all_rows) - 1, 0)
        recorder.record(
            "episode_reset",
            network=self.selected_net_path,
            baseline_acc=round(float(self.original_acc), 5),
            num_layers=len(model_with_rows.all_layers),
            num_prunable_rows=num_rows,
            params_m=round(self.original_params / 1e6, 4),
        )

        # After feature extraction and setup
        torch.cuda.empty_cache()
        gc.collect()

        return fm

    def _dependency_groups(self, model_with_rows):
        """
        Channel-dependency groups for the current model, with failures made visible.

        A model that cannot be traced silently loses structured pruning *and* the exact
        coupling bias in the state encoder, so it is counted as an issue rather than being
        absorbed by a `None` return value.
        """
        try:
            groups = channel_groups.build_channel_groups(model_with_rows.model)
        except Exception as error:
            recorder.issue("fx_trace_error", f"{type(error).__name__}: {error}",
                           network=self.selected_net_path)
            return None

        if groups is None:
            recorder.issue("fx_trace_failed", "model is not symbolically traceable",
                           network=self.selected_net_path)
        return groups

    def param_ratio(self) -> float:
        """Current / original parameter count (1.0 at reset)."""
        origin = getattr(self, "original_params", None) or utils.calc_num_parameters(self.current_model)
        return utils.calc_num_parameters(self.current_model) / max(float(origin), 1.0)

    def legal_action_mask(self, device=None):
        """Bool mask over ``compression_rates_dict`` for the current row (fortify-aware)."""
        from src.fortify import legal_action_mask
        import src.pruning as pruning

        model_with_rows = ModelWithRows(self.current_model)
        row = max(0, (self.row_idx or 1) - 1)
        layer_idx = model_with_rows.row_to_main_layer[row]
        layer = model_with_rows.all_layers[layer_idx]
        alive = int(pruning.alive_filters(layer).numel()) if hasattr(layer, "weight") else 1
        dev = device if device is not None else self.conf.device
        return legal_action_mask(
            self.conf.compression_rates_dict,
            row_index=row,
            alive_count=alive,
            device=dev,
        )

    def step(self, compression_rate, is_to_train=True):
        """
        Compress the network, then move to the next state.

        Args:
            compression_rate (float): Factor to reduce layer size.
            is_to_train (bool): Whether to train after compression.

        Returns:
            Tuple: Next state, reward, and done flag.
        """
        step_timer = logging_utils.Timer().__enter__()
        model_with_rows = ModelWithRows(self.current_model)

        # Determine affected layers (from current row up to start of next row)
        current_layer_idx = model_with_rows.row_to_main_layer[self.row_idx - 1]
        next_layer_idx = model_with_rows.row_to_main_layer[self.row_idx] \
            if self.row_idx < len(model_with_rows.row_to_main_layer) else len(model_with_rows.all_layers)
        update_indices = list(range(current_layer_idx, next_layer_idx))

        step_index = len(self.actions_history)
        target_layer = model_with_rows.all_layers[current_layer_idx]
        logging_utils.set_context(step=step_index, layer=current_layer_idx)
        utils.print_flush(f"Step {self.row_idx - 1} - Layer {current_layer_idx} "
                          f"({type(target_layer).__name__}), Compression Rate: {compression_rate}")

        params_before = utils.calc_num_parameters(self.current_model)
        prune_outcome = {"mode": "identity"}

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Modify the model in-place
            with logging_utils.stage("step.prune", level=logging.DEBUG):
                if self.conf.prune:
                    model_with_rows = prune_current_model(model_with_rows, compression_rate, self.row_idx - 1)
                else:
                    model_with_rows = create_new_model_with_new_weights(model_with_rows, compression_rate,
                                                                        self.row_idx - 1)
            prune_outcome = dict(getattr(model_with_rows, "last_prune_outcome", {}) or {})

            # Prepare model handler
            learning_handler_new_model = self.create_learning_handler(model_with_rows.model)

            # Freeze/unfreeze layers based on config. Prefer the modules actually rewritten by
            # the last structural group prune (producers + consumers + norms); the old
            # "pruned row + next layer" rule left resized consumers frozen and made mild
            # compressions unrecoverable under the -5 pp reward cliff (see recovery probes).
            if self.conf.train_compressed_layer_only:
                edited = getattr(model_with_rows, "last_edited_param_ids", None)
                params_to_keep_trainable = (
                    edited if edited
                    else build_param_names_to_keep_trainable(model_with_rows, self.row_idx - 1))
                learning_handler_new_model.freeze_all_layers_but_pruned(params_to_keep_trainable)
            else:
                learning_handler_new_model.unfreeze_all_layers()

            if is_to_train:
                with logging_utils.stage("step.finetune", level=logging.DEBUG):
                    learning_handler_new_model.train_model(self.train_loader)

        # Evaluate the compressed model
        learning_handler_new_model.model.eval()
        with logging_utils.stage("step.evaluate", level=logging.DEBUG):
            new_acc = learning_handler_new_model.evaluate_model(self.val_loader)

        # Realized size before reward: CNN group edits ≠ nominal (1-rate).
        params_after = utils.calc_num_parameters(learning_handler_new_model.model)
        reward_rate = reward_compression_rate(
            prune_outcome, compression_rate, params_before, params_after,
            new_acc, self.original_acc, self.conf.allowed_acc_reduction)
        reward = utils.compute_reward(
            new_acc, self.original_acc, reward_rate,
            params_before=params_before, params_after=params_after)

        # Move to next state
        self.row_idx += 1
        learning_handler_new_model.unfreeze_all_layers()
        old_model = self.current_model
        self.current_model = learning_handler_new_model.model
        del old_model
        del learning_handler_new_model
        # Identity steps do not allocate a new graph; skipping the cache flush avoids a
        # stall on every remaining eval step after the param floor.
        if compression_rate != 1:
            torch.cuda.empty_cache()
            gc.collect()

        # Extract features for the next state. The dependency analysis is redone here because
        # the compression just applied changed the graph.
        with logging_utils.stage("step.feature_extraction", level=logging.DEBUG):
            kept = utils.calc_num_parameters(self.current_model) / max(self.original_params, 1e-9)
            fm = self.feature_extractor.encode_to_bert_input(
                model_with_rows, current_layer_idx, update_indices,
                dependency_groups=self._dependency_groups(model_with_rows),
                param_ratio=min(1.0, max(0.0, kept)))

        # Check termination condition
        num_rows = len(model_with_rows.all_rows) - 1  # Only FC and Conv layers trigger a new row
        self.actions_history.append(compression_rate)
        num_actions = len(self.actions_history)
        # As self.row_idx - 1 is the current appraised row, the index should not drop below 1
        self.row_idx = max(1, self.row_idx % (num_rows + 1))
        done = num_actions >= num_rows * self.conf.passes

        step_timer.__exit__(None, None, None)
        # One record per transition: enough to reconstruct the trajectory, the policy's
        # behaviour and the library's pruning coverage without re-running anything
        recorder.record(
            "step",
            network=self.selected_net_path,
            step_index=step_index,
            layer_index=current_layer_idx,
            layer_type=type(target_layer).__name__,
            compression_rate=compression_rate,
            reward=round(float(reward), 4),
            reward_mode=__import__("os").environ.get("SPECTRA_REWARD_MODE", "neon"),
            baseline_acc=round(float(self.original_acc), 5),
            new_acc=round(float(new_acc), 5),
            delta_acc=round(float(new_acc - self.original_acc), 5),
            params_before_m=round(params_before / 1e6, 4),
            params_after_m=round(params_after / 1e6, 4),
            param_reduction=round(1 - params_after / max(params_before, 1), 5),
            prune_mode=prune_outcome.get("mode"),
            prune_reason=prune_outcome.get("reason"),
            old_width=prune_outcome.get("old_width"),
            new_width=prune_outcome.get("new_width"),
            # Requested vs. applied: a rate the layer's width cannot express (0.9 of 6
            # channels) would otherwise look like a normal action in the trajectory
            realized_rate=(round(prune_outcome["new_width"] / prune_outcome["old_width"], 4)
                           if prune_outcome.get("old_width") else None),
            seconds=round(step_timer.seconds, 3),
            done=bool(done),
        )
        utils.print_flush(
            f"Step {step_index} done in {step_timer.seconds:.1f}s | rate={compression_rate} "
            f"acc {self.original_acc:.4f} -> {new_acc:.4f} | reward={reward:.2f} "
            f"| prune={prune_outcome.get('mode')}")

        # Log model evaluation metrics after each pass and flush to CSV.
        if self.mode != AGENT_TRAIN and (done or num_actions % num_rows == 0):
            with logging_utils.stage("step.compute_results", level=logging.DEBUG):
                self.compute_and_log_results(model_with_rows)

        # Save the final pruned model to a checkpoint file,
        # if requested by the user via self.conf.save_pruned_checkpoints = True
        if done and self.conf.save_pruned_checkpoints:
            self.save_pruned_checkpoint()

        return fm, reward, done

    def compute_and_log_results(self, model_with_rows, t_curr=None):
        """
        Compute accuracy according to eval mode (train / test datasets), number of params and FLOPs.
        Log model evaluation metrics after each pass and flush to CSV.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            t_curr (float):    Time of log, to calculate evaluation time. Defaults to now.
        """
        # A default of time.perf_counter() would be evaluated once, at import time, making
        # every logged evaluation_time a constant offset rather than an elapsed duration.
        if t_curr is None:
            t_curr = time.perf_counter()

        # Retrieve original & compressed models
        original_model = self.data_dict[self.selected_net_path][0]
        compressed_model = self.current_model

        # Create learning handlers
        new_lh = self.create_learning_handler(compressed_model)
        origin_lh = self.create_learning_handler(original_model)

        # self.mode holds EVAL_TRAIN / EVAL_TEST; comparing against the bare string "test"
        # never matched, so test-mode results were reported on the training split
        dataset_loader = self.test_loader if self.mode == EVAL_TEST else self.train_loader

        fold_str = self.fold_idx if self.fold_idx == "N/A" else f"{self.fold_idx} / {self.conf.n_splits}"

        input_shape = utils.get_input_shape(dataset_loader)

        # Store results
        result_entry = {
            'model': self.selected_net_path,
            'pass': f'{len(self.actions_history) // (len(model_with_rows.all_rows) - 1)}'
                    f' / {self.conf.passes}',
            'fold': fold_str,
            'new_acc': round(new_lh.evaluate_model(dataset_loader), 3),
            'origin_acc': round(origin_lh.evaluate_model(dataset_loader), 3),
            'new_param (M)': round(utils.calc_num_parameters(compressed_model) / 1e6, 3),
            'origin_param (M)': round(utils.calc_num_parameters(original_model) / 1e6, 3),
            'new_effective_param (M)': round(pruning.count_effective_parameters(compressed_model) / 1e6, 3),
            'new_flops (M)': round(utils.calc_flops(compressed_model, input_shape) / 1e6, 3),
            'origin_flops (M)': round(utils.calc_flops(original_model, input_shape) / 1e6, 3),
            'new_model_arch': utils.get_model_layers_str(compressed_model),
            'origin_model_arch': utils.get_model_layers_str(original_model),
            'evaluation_time': t_curr - self.t_start if self.t_start else None
        }

        # Results live alongside the run's logs and events so a run is one self-contained
        # directory; the historical ./models/Reinforce_Evaluation location is still written
        # for compatibility with existing analysis notebooks.
        results_dir = os.path.join(logging_utils.run_dir(), "results")
        os.makedirs(results_dir, exist_ok=True)
        legacy_dir = "./models/Reinforce_Evaluation"
        os.makedirs(legacy_dir, exist_ok=True)  # the directory was never created

        # Ranks evaluate disjoint shards; separate files avoid interleaved concurrent appends
        rank_suffix = f"_rank{ddp.get_rank()}" if ddp.get_world_size() > 1 else ""
        file_name = f"results_{self.mode}{rank_suffix}.csv"
        legacy_name = (f"results_{self.conf.test_name}_{self.mode}"
                       f"_{self.conf.test_ts}{rank_suffix}.csv")

        df_entry = pd.DataFrame([result_entry])
        for path in (os.path.join(results_dir, file_name), os.path.join(legacy_dir, legacy_name)):
            df_entry.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

        # The same record as a structured event, so summaries do not have to parse CSVs whose
        # columns include multi-line architecture strings
        acc_delta = result_entry['new_acc'] - result_entry['origin_acc']
        param_ratio = result_entry['new_param (M)'] / max(result_entry['origin_param (M)'], 1e-9)
        flops_ratio = result_entry['new_flops (M)'] / max(result_entry['origin_flops (M)'], 1e-9)
        effective_ratio = (result_entry['new_effective_param (M)']
                           / max(result_entry['origin_param (M)'], 1e-9))
        recorder.record(
            "eval",
            network=self.selected_net_path,
            eval_mode=self.mode,
            fold=fold_str,
            pass_index=result_entry['pass'],
            new_acc=result_entry['new_acc'],
            origin_acc=result_entry['origin_acc'],
            delta_acc=round(acc_delta, 5),
            new_param_m=result_entry['new_param (M)'],
            origin_param_m=result_entry['origin_param (M)'],
            new_effective_param_m=result_entry['new_effective_param (M)'],
            param_ratio=round(param_ratio, 5),
            effective_param_ratio=round(effective_ratio, 5),
            new_flops_m=result_entry['new_flops (M)'],
            origin_flops_m=result_entry['origin_flops (M)'],
            flops_ratio=round(flops_ratio, 5),
            actions=list(self.actions_history),
            evaluation_time=result_entry['evaluation_time'],
        )
        mask_note = ""
        if abs(effective_ratio - param_ratio) > 0.02:
            mask_note = (f" | effective-params x{effective_ratio:.3f} "
                         "(masked zeros; not a structural size cut)")
        utils.print_flush(
            f"[eval] {os.path.basename(self.selected_net_path)} pass {result_entry['pass']} | "
            f"acc {result_entry['origin_acc']:.3f} -> {result_entry['new_acc']:.3f} "
            f"({acc_delta:+.3f}) | params x{param_ratio:.3f} | FLOPs x{flops_ratio:.3f}"
            f"{mask_note}")

    def save_pruned_checkpoint(self):
        """
        Save the final pruned model to a checkpoint file.
        The filename keeps the original name but replaces the last '.' before the extension with '_pruned.'.
        """
        # Extract filename and replace only the last dot (as the filename might contain decimal points)
        filename = os.path.basename(self.selected_net_path)  # Get the file name from the path
        name_parts = filename.rsplit('.', 1)  # Split at the last dot
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Prevent overwriting
        model_name = f"{name_parts[0]}_pruned_{timestamp}.{name_parts[1]}"
        save_path = f"./pruned_models/{model_name}"
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_to_save = ddp.unwrap(self.current_model)

        # Save state dict safely (on rank 0 only). The architecture is recorded alongside the
        # weights because structured pruning changes layer widths, so the checkpoint can no
        # longer be loaded into a stock instantiation of the original architecture.
        if ddp.is_main_process():
            torch.save({
                "state_dict": model_to_save.state_dict(),
                "source_checkpoint": self.selected_net_path,
                "architecture": utils.get_model_layers_str(model_to_save),
                "actions_history": self.actions_history,
            }, save_path)
            utils.print_flush(f"Pruned model saved at {save_path}")

        ddp.barrier()  # Sync all processes if in DDP
        torch.cuda.empty_cache()
        gc.collect()

    def create_learning_handler(self, new_model) -> BasicHandler:
        """
        Create appropriate learning handler based on mission type,
        ensuring compatibility with both training & testing scenarios.
        SPECTRA's current implementation supports Classifications tasks only.
        """
        return ClassificationHandler(
            new_model,
            torch.nn.CrossEntropyLoss()
        )


def build_param_names_to_keep_trainable(model_with_rows, row_to_modify_idx):
    """
    Builds a list of the pruned layer and its subsequent's parameter IDs to keep trainable (all other layers' parameters are freezed).

    Args:
        model_with_rows (ModelWithRows):   The model wrapped with rows of layers.
        row_to_modify_idx (int):           Index of the row whose first layer is to be pruned / resized

    Returns:
        List[int]: A list of parameter IDs to freeze.
    """
    layer_to_modify_idx = model_with_rows.row_to_main_layer[row_to_modify_idx]
    layers_to_keep_trainable = model_with_rows.all_layers[layer_to_modify_idx:layer_to_modify_idx + 2]

    # Flatten and collect parameter IDs
    return [id(param) for layer in layers_to_keep_trainable for param in layer.parameters()]


def create_new_model_with_new_weights(model_with_rows, compression_rate, row_to_resize_idx):
    """
    Replace a layer with a reduced version, adjusting the subsequent layer accordingly.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be resized
        compression_rate (float):         The desired compression rate for resizing
        row_to_resize_idx (int):          Index of the row whose first layer is to be resized

    Returns:
         model_with_rows (ModelWithRows): The resized model.
    """
    model_with_rows.unwrap_model()

    layer_to_resize_idx = model_with_rows.row_to_main_layer[row_to_resize_idx]
    layer_to_resize = model_with_rows.all_layers[layer_to_resize_idx]

    if not isinstance(layer_to_resize, (nn.Linear, nn.Conv2d)):
        raise NotImplementedError("Resizing not implemented for this layer type.")

    # Keep the highest-magnitude filters instead of discarding all learned weights: this
    # path used to install a freshly initialised layer, throwing away the pretrained
    # network that the reward is measured against. It also only rebound a list entry, so
    # the model itself was never modified, and the consumer layers were left expecting the
    # original width.
    return prune_current_model(model_with_rows, compression_rate, row_to_resize_idx)


def prune_current_model(model_with_rows, compression_rate, row_to_prune_idx):
    """
    Compress the target layer by removing its least important output filters.

    The layer is physically shrunk (and its consumers resized) whenever the dependency
    chain permits; otherwise the filters are masked in place. See src/pruning.py.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be pruned
        compression_rate (float):         The desired compression rate for pruning
        row_to_prune_idx (int):           Index of the row whose first layer is to be pruned

    Returns:
        pruned_model_with_rows (ModelWithRows): The pruned model
    """
    model_with_rows.unwrap_model()

    layer_to_prune_idx = model_with_rows.row_to_main_layer[row_to_prune_idx]
    layer_to_prune = model_with_rows.all_layers[layer_to_prune_idx]
    old_width = pruning.layer_width(layer_to_prune)

    # Layers whose widths are tied together (a residual block's conv2 and whatever feeds its
    # shortcut) are compressed as one unit, so coupled convolutions shrink instead of merely
    # being masked
    try:
        groups = channel_groups.build_channel_groups(model_with_rows.model)
    except Exception as error:
        groups, trace_error = None, f"{type(error).__name__}: {error}"
    else:
        trace_error = None
    group = channel_groups.group_of(groups, layer_to_prune) if groups else None

    if group is not None and group.prunable:
        keep_idx = pruning.select_group_survivors(group, compression_rate)
        if keep_idx is not None and pruning.prune_group_structurally(model_with_rows, group, keep_idx):
            coupled = len(group.producers) + len(group.depthwise)
            utils.print_flush(
                f"Layer {layer_to_prune_idx}: width {old_width} -> {keep_idx.numel()} "
                f"across {coupled} coupled layer(s), {len(group.consumers)} consumer(s) resized")
            # Recorded on the ModelWithRows so NetworkEnv.step can fold the outcome into its
            # own step record; the caller owns the episode/network context.
            edited_ids = list(getattr(model_with_rows, "last_edited_param_ids", []) or [])
            model_with_rows.last_prune_outcome = {
                "mode": "structural", "reason": None, "old_width": old_width,
                "new_width": int(keep_idx.numel()), "coupled_layers": coupled,
                "consumers_resized": len(group.consumers),
                "trainable_param_count": len(edited_ids),
            }
            recorder.record("prune", mode="structural", layer_index=layer_to_prune_idx,
                            layer_type=type(layer_to_prune).__name__, rate=compression_rate,
                            old_width=old_width, new_width=int(keep_idx.numel()),
                            coupled_layers=coupled, consumers_resized=len(group.consumers))
            model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)
            return model_with_rows

    # "no dependency group resolved" conflated four different failures, so the logs could not
    # say whether the library needs a new fx rule, a new resize rule, or nothing at all.
    if trace_error is not None:
        reason = f"fx trace raised {trace_error}"
    elif groups is None:
        reason = "model is not symbolically traceable (dynamic control flow)"
    elif group is None:
        reason = f"layer produces no channel group (of {len(groups)} resolved)"
    elif not group.prunable:
        reason = group.reason
    else:
        reason = "structural edit rejected (target width equals current width)"

    keep_idx = pruning.select_surviving_filters(layer_to_prune, compression_rate)
    pruning.mask_layer_filters(layer_to_prune, keep_idx)
    utils.print_flush(f"Layer {layer_to_prune_idx}: masked {old_width - keep_idx.numel()}/{old_width} "
                      f"filters, shape preserved ({reason})")

    # Masking is a correctness-preserving fallback, not a success: filters are zeroed but
    # numel/FLOPs stay put. NetworkEnv withholds NEON compression credit on in-budget
    # masked no-ops (see reward_compression_rate). Counting the reasons is how we learn
    # which dependency patterns the library still cannot resize.
    #
    # A layer already down to a single channel is the one case that is *not* a library gap:
    # there is nothing left to remove. It is counted separately so the masked-fallback rate
    # keeps measuring what it is meant to measure, and because a network whose layers reach
    # width 1 is telling us the action space is too aggressive for that architecture.
    at_floor = old_width <= 1
    kind = "prune_floor_reached" if at_floor else "prune_fallback_masked"
    if at_floor:
        reason = "layer is already one channel wide"

    model_with_rows.last_prune_outcome = {
        "mode": "floor" if at_floor else "masked", "reason": reason, "old_width": old_width,
        "new_width": int(keep_idx.numel()), "coupled_layers": 0, "consumers_resized": 0,
    }
    # Masked / floor edits do not rewrite a dependency group; fall back to the row-local rule.
    model_with_rows.last_edited_param_ids = None
    recorder.issue(kind, reason, layer_index=layer_to_prune_idx,
                   layer_type=type(layer_to_prune).__name__, rate=compression_rate)
    recorder.record("prune", mode="floor" if at_floor else "masked", reason=reason,
                    layer_index=layer_to_prune_idx,
                    layer_type=type(layer_to_prune).__name__, rate=compression_rate,
                    old_width=old_width, new_width=int(keep_idx.numel()))

    model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)

    return model_with_rows
