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
import src.fortify as fortify
import src.recovery_edits as recovery_edits
import src.utils as utils

AGENT_TRAIN = "agent_train"  # Mode when NetworkEnv is called from A2C_Agent_Reinforce.py
EVAL_TRAIN = "eval_train"  # Mode when NetworkEnv is called from a2c_agent_reinforce_runner.py, evaluating the train dataset
EVAL_TEST = "eval_test"  # Mode when NetworkEnv is called from a2c_agent_reinforce_runner.py, evaluating the test dataset


def _ratio_cache_enabled() -> bool:
    """Memoize size probes within a step. Kill switch: ``SPECTRA_PREVIEW_CACHE=0``."""
    return os.environ.get("SPECTRA_PREVIEW_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


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
        self.last_val_acc = None
        self._origin_test_acc = None
        self.original_params = None
        self.original_flops = None
        self.selected_net_path = None
        self.current_model = None
        self.feature_extractor = None
        # Size probes for the *current* model, dropped whenever the model changes.
        # Eval asks for the same param/FLOP ratios several times per step (floor check,
        # look-ahead, Δparams/ΔFLOPs preference), and each miss is a deepcopy + prune or
        # a hooked forward pass. See _ratio_cache_enabled.
        self._ratio_cache = {}
        # all_layers indices of every coupled group structurally cut in the current pass
        # (fortify.group_once_per_pass). Empty unless that switch is on.
        self._pass_locked_layers = set()

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
        # v10 fixed-target episodes (fortify.fixed_target): this episode's target keep, the val
        # accuracy the last step ended at, where the episode ended, and the per-layer group
        # sensitivity channels (fortify.state_sens). The target stream has its own generator so
        # the network shuffle above is unchanged.
        self.target_keep = None
        self._target_prev_acc = None
        self._target_final = None
        self._layer_sens = None
        self._sens_cache = {}
        self._target_rng = np.random.default_rng([int(self.conf.seed), ddp.get_rank(), 10])
        self._reset_episode_reward_stats()

    def _reset_episode_reward_stats(self):
        self.episode_rho_sum = 0.0
        self.episode_overshoot_sum = 0.0
        self.episode_any_over = False
        self.episode_step_over = []
        # Smallest kept-parameter fraction reached while the cumulative val Δacc was still
        # inside the τ band — the training-side twin of the TRAJ ``val_best`` point.
        self.episode_best_inband_kept = 1.0
        # V7 probe score "area": Σ over in-band steps of (size removed by the step, as a fraction
        # of the origin) × (remaining slack / τ). Deeper-in-band and kinder-at-equal-depth both
        # score higher; the legacy ``1 − kept`` saturates at the deepest legal walk (the mild
        # clone) and is blind to Δacc — every 12/4 arm froze at the same 0.262 for that reason.
        self._episode_inband_area = 0.0
        self._area_prev_kept = 1.0

    def episode_inband_area(self) -> float:
        """Slack-weighted in-band cut area of this episode (``SPECTRA_PROBE_SCORE=area``)."""
        return float(self._episode_inband_area)

    def _account_inband_point(self, kept_now: float, delta_pp: float, tau: float) -> None:
        """Book an in-band step: deepest in-band kept (legacy score) and the slack-weighted area."""
        kept_now = float(kept_now)
        self.episode_best_inband_kept = min(self.episode_best_inband_kept, kept_now)
        removed = max(0.0, float(self._area_prev_kept) - kept_now)
        slack_frac = max(0.0, min(1.0, (float(tau) + float(delta_pp)) / max(float(tau), 1e-6)))
        self._episode_inband_area += removed * slack_frac
        self._area_prev_kept = min(float(self._area_prev_kept), kept_now)

    def episode_checkpoint_score(self) -> float:
        """In-budget compression (F1) vs discounted return. See fortify.inbudget_checkpointing."""
        return fortify.inbudget_checkpoint_score(
            self.episode_rho_sum, self.episode_overshoot_sum, self.episode_any_over)

    def episode_val_best_compression(self) -> float:
        """``1 − kept`` at the deepest in-band point of this episode (0 when nothing was cut in band)."""
        return max(0.0, 1.0 - float(self.episode_best_inband_kept))

    # ------------------------------------------------------------------ v10 fixed-target episodes

    def _episode_target(self, explicit=None) -> float:
        """This episode's target keep: ``explicit``; a draw in training; the size match in eval."""
        if explicit is not None:
            return float(explicit)
        if self.mode == AGENT_TRAIN:
            lo, hi = fortify.target_keep_range()
            return float(self._target_rng.uniform(lo, hi))
        match = fortify.eval_size_match()
        if match is not None and match[0] == "param":
            return float(match[1])
        points = [t for kind, t in fortify.eval_size_points() if kind == "param"]
        if points:
            return float(min(points))
        utils.print_flush("fixed target: eval without SPECTRA_EVAL_SIZE_MATCH=param:<keep>; "
                          "the actor is given 0.6")
        return 0.6

    def _group_sensitivity_features(self):
        """
        Per-layer group-sensitivity channels of the origin (``SPECTRA_STATE_SENS``); None on
        failure. Every episode on a network starts from the same checkpoint, so the channels are
        measured the first time the network comes round and reused after that.
        """
        import src.group_sensitivity as group_sensitivity
        cache = getattr(self, "_sens_cache", None)
        if cache is None:
            cache = self._sens_cache = {}
        if self.selected_net_path in cache:
            return cache[self.selected_net_path]
        try:
            with logging_utils.stage("reset.group_sensitivity"):
                model = self.current_model.to(self.conf.device)
                batches = group_sensitivity.calibration_batches(
                    self.train_loader, group_sensitivity.CALIB_BATCHES, self.conf.device)
                features, summary = group_sensitivity.layer_features(model, batches, self._input_shape())
        except Exception as error:  # noqa: BLE001 - the channels are an enrichment, not a precondition
            utils.print_flush(f"group sensitivity unavailable ({type(error).__name__}: {error}); zeros")
            return None
        utils.print_flush(
            f"group sensitivity: {summary['groups']} groups on {summary['layers']} layers, loss rise "
            f"median {summary['median']:.4f} min {summary['min']:.4f} max {summary['max']:.4f} "
            f"(base {summary['base_loss']:.4f}) in {summary['seconds']:.1f}s")
        cache[self.selected_net_path] = features
        return features

    def episode_target_score(self) -> float:
        """Fixed-target return of this episode: val Δacc (pp) where it ended minus any miss penalty."""
        if self.target_keep is None:
            return 0.0
        final = self._target_final
        if final is None:
            kept = self.param_ratio()
            delta_pp = (float(self.last_val_acc) - float(self.original_acc)) * 100.0
        else:
            kept, delta_pp = final["kept"], final["delta_pp"]
        return fortify.target_score(delta_pp, kept, self.target_keep)

    def _land_on_target(self, rate: float):
        """
        ``(rate, landed kept)``: when a cut at ``rate`` would take the network below the target,
        the mildest keep rate that still reaches it (bisection on the previewed size); else
        ``(rate, None)``. The episode then ends at the target instead of past it.
        """
        target = float(self.target_keep) + 1e-9
        if self.preview_param_ratio(rate) > target:
            return rate, None
        lo, hi = float(rate), 1.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if self.preview_param_ratio(mid) <= target:
                lo = mid
            else:
                hi = mid
        return lo, self.preview_param_ratio(lo)

    def reset(self, test_net_path=None, test_model=None, test_loaders=None, target_keep=None):
        """
        Reset environment with a new CNN model & dataset.

        ``target_keep`` fixes a ``SPECTRA_FIXED_TARGET`` episode's target (the fixed probe);
        otherwise training draws one and eval reads ``SPECTRA_EVAL_SIZE_MATCH``.
        """
        # Ensure prior memory is cleaned
        if hasattr(self, "feature_extractor"):
            del self.feature_extractor
        if hasattr(self, "current_model"):
            del self.current_model
        torch.cuda.empty_cache()
        gc.collect()

        self.row_idx = 1  # The first row to be a candidate for pruning is self.row_idx - 1 -> index 0
        self.actions_history = []
        self._ratio_cache = {}
        self._pass_locked_layers = set()
        self._rollback_locked_layers = set()
        self._episode_group_cuts = {}
        self.last_step_outcome = {"mode": "identity"}
        self.last_step_layer_idx = None
        self._reset_episode_reward_stats()

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
        # FLOP origin is only needed when the eval FLOP floor is on (otherwise a
        # MAC probe every episode would tax training for no reason).
        self.original_flops = None
        try:
            min_flop = float(os.environ.get("SPECTRA_EVAL_MIN_FLOP_RATIO", "0") or 0)
        except ValueError:
            min_flop = 0.0
        if min_flop > 0:
            self.original_flops = utils.calc_flops(
                self.current_model, self._input_shape())
        if utils.env_flag("SPECTRA_FT_KD"):
            self.kd_teacher = copy.deepcopy(self.current_model).eval()
            for param in self.kd_teacher.parameters():
                param.requires_grad = False
            self.kd_teacher.to(self.conf.device)
        else:
            self.kd_teacher = None

        model_with_rows = ModelWithRows(self.current_model)

        # Every log line and event emitted for this episode is tagged with the network under
        # compression, so a failure can be attributed without reading back through the file
        logging_utils.set_context(net=os.path.basename(self.selected_net_path), mode=self.mode)
        utils.print_flush(f"Loading {self.selected_net_path}")

        self.target_keep = self._episode_target(target_keep) if fortify.fixed_target() else None
        self._target_final = None
        self._layer_sens = self._group_sensitivity_features() if fortify.state_sens() else None
        target_extras = (fortify.target_channels(1.0, self.target_keep)
                         if self.target_keep is not None else None)

        # Prepare feature extractor with training data
        self.feature_extractor = FeatureExtractor(self.train_loader, self.conf.device)
        with logging_utils.stage("reset.feature_extraction"):
            fm = self.feature_extractor.encode_to_bert_input(
                model_with_rows, model_with_rows.row_to_main_layer[self.row_idx - 1],
                dependency_groups=self._dependency_groups(model_with_rows),
                param_ratio=1.0, extras=[1.0, 0.0], episode_cuts={},
                target_extras=target_extras, layer_sens=self._layer_sens)

        # Evaluate original model accuracy
        learning_handler_original_model = self.create_learning_handler(self.current_model)
        with logging_utils.stage("reset.baseline_accuracy"):
            self.original_acc = learning_handler_original_model.evaluate_model(self.val_loader)
        self.last_val_acc = float(self.original_acc)
        self._target_prev_acc = float(self.original_acc)
        self._origin_test_acc = None
        if self.target_keep is not None:
            utils.print_flush(f"fixed target: keep x{self.target_keep:.3f} of the parameters "
                              f"({self.mode}); origin val {float(self.original_acc):.4f}")

        num_rows = max(len(model_with_rows.all_rows) - 1, 0)
        recorder.record(
            "episode_reset",
            network=self.selected_net_path,
            baseline_acc=round(float(self.original_acc), 5),
            num_layers=len(model_with_rows.all_layers),
            num_prunable_rows=num_rows,
            params_m=round(self.original_params / 1e6, 4),
            target_keep=(round(float(self.target_keep), 5) if self.target_keep is not None else None),
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

    def _cached_ratio(self, key, compute):
        """``compute()`` memoized until the model changes (see ``_ratio_cache``)."""
        if not _ratio_cache_enabled():
            return compute()
        cache = getattr(self, "_ratio_cache", None)
        if cache is None:
            cache = self._ratio_cache = {}
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def param_ratio(self) -> float:
        """Current / original parameter count (1.0 at reset)."""
        def _compute():
            origin = (getattr(self, "original_params", None)
                      or utils.calc_num_parameters(self.current_model))
            return utils.calc_num_parameters(self.current_model) / max(float(origin), 1.0)

        return self._cached_ratio("param_ratio", _compute)

    def _input_shape(self):
        """
        Per-sample shape of the current train loader.

        Cached per episode rather than per step: it does not depend on the model, and
        ``get_input_shape`` materialises a batch (``next(iter(loader))``), which restarts
        the loader's worker processes every call.
        """
        loader = self.train_loader
        cached = getattr(self, "_input_shape_cache", None)
        if cached is not None and cached[0] is loader and _ratio_cache_enabled():
            return cached[1]
        shape = utils.get_input_shape(loader)
        self._input_shape_cache = (loader, shape)
        return shape

    def flops_ratio(self) -> float:
        """Current / original FLOPs (1.0 at reset). Lazy-origin if reset skipped the probe."""
        def _compute():
            current = utils.calc_flops(self.current_model, self._input_shape())
            origin = getattr(self, "original_flops", None)
            if not origin:
                # First call should be at reset (unpruned). Do not lock origin after a prune.
                self.original_flops = current
                return 1.0
            return current / max(float(origin), 1.0)

        return self._cached_ratio("flops_ratio", _compute)

    def score_test_loader(self):
        """``(new_acc, origin_acc, delta_pp)`` on the CNN test split. Caches origin."""
        origin_model = self.data_dict[self.selected_net_path][0]
        if self._origin_test_acc is None:
            origin_lh = self.create_learning_handler(origin_model)
            self._origin_test_acc = float(origin_lh.evaluate_model(self.test_loader))
        new_lh = self.create_learning_handler(self.current_model)
        new_acc = float(new_lh.evaluate_model(self.test_loader))
        origin = float(self._origin_test_acc)
        return new_acc, origin, (new_acc - origin) * 100.0

    def preview_ratios(self, compression_rate: float):
        """
        ``(param_ratio, flops_ratio)`` after a dry-run prune of the current row.

        One clone, no fine-tune. Identity is a no-op preview. FLOPs are only
        measured when ``SPECTRA_EVAL_MIN_FLOP_RATIO`` is on.
        """
        return self._cached_ratio(
            ("preview", self.row_idx, round(float(compression_rate), 6)),
            lambda: self._preview_ratios_uncached(compression_rate))

    def _preview_ratios_uncached(self, compression_rate: float):
        origin_p = float(getattr(self, "original_params", None)
                         or utils.calc_num_parameters(self.current_model))
        origin_p = max(origin_p, 1.0)
        try:
            need_flops = float(os.environ.get("SPECTRA_EVAL_MIN_FLOP_RATIO", "0") or 0) > 0
        except ValueError:
            need_flops = False
        origin_f = getattr(self, "original_flops", None)

        def _current_flops():
            return utils.calc_flops(self.current_model, self._input_shape())

        if fortify.width_ladder_max() > 0:
            compression_rate = self._ladder_keep_rate(
                ModelWithRows(self.current_model), max(0, (self.row_idx or 1) - 1), compression_rate)
        if abs(float(compression_rate) - 1.0) < 1e-9:
            p = utils.calc_num_parameters(self.current_model) / origin_p
            if not need_flops:
                return p, 1.0
            if not origin_f:
                origin_f = _current_flops()
                self.original_flops = origin_f
            return p, _current_flops() / max(float(origin_f), 1.0)

        cloned = copy.deepcopy(self.current_model)
        try:
            mwr = ModelWithRows(cloned)
            prune_current_model(
                mwr, compression_rate, max(0, (self.row_idx or 1) - 1),
                quiet=True, record=False, input_shape=self._input_shape())
            after_p = utils.calc_num_parameters(mwr.model)
            after_f = None
            if need_flops:
                if not origin_f:
                    origin_f = _current_flops()
                    self.original_flops = origin_f
                after_f = utils.calc_flops(mwr.model, self._input_shape())
        except Exception:
            after_p = utils.calc_num_parameters(self.current_model)
            after_f = None
        finally:
            del cloned
        p = after_p / origin_p
        if not need_flops or after_f is None:
            return p, 1.0
        return p, after_f / max(float(origin_f), 1.0)

    def preview_param_ratio(self, compression_rate: float) -> float:
        """
        Parameter fraction after applying ``compression_rate`` to the current row, without
        mutating the live model or running fine-tune.

        Used by eval look-ahead so a 0.8 group-cut cannot jump past
        ``SPECTRA_EVAL_MIN_PARAM_RATIO`` in one step. Identity is a no-op preview.
        """
        return self.preview_ratios(compression_rate)[0]

    def preview_flops_ratio(self, compression_rate: float) -> float:
        """FLOP fraction after a dry-run prune of the current row (eval FLOP floor)."""
        return self.preview_ratios(compression_rate)[1]

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
        owned = None
        if fortify.action_menu() == "budget":
            # Budget actions are priced per group: the mask must judge the mapped keep rate.
            owned = recovery_edits.group_param_fraction(model_with_rows, row)
        force_identity = self.group_locked(layer_idx)
        if not force_identity and fortify.protect_streams():
            force_identity = self._is_stream_row(model_with_rows, layer)
        return legal_action_mask(
            self.conf.compression_rates_dict,
            row_index=row,
            alive_count=alive,
            device=dev,
            force_identity=force_identity,
            group_param_fraction=owned,
        )

    @staticmethod
    def _is_stream_row(model_with_rows, layer) -> bool:
        """True when ``layer`` writes a residual stream (``SPECTRA_PROTECT_STREAMS``)."""
        try:
            groups = channel_groups.build_channel_groups(model_with_rows.model)
        except Exception:
            return False
        if not groups:
            return False
        return fortify.is_residual_stream(channel_groups.group_of(groups, layer))

    @staticmethod
    def _ladder_keep_rate(model_with_rows, row: int, rate: float) -> float:
        """Keep rate the width ladder applies to ``row`` (``rate`` when no ladder row; 1.0 = infeasible)."""
        value = float(rate)
        if fortify.width_ladder_max() <= 0 or not 0.0 < value < 1.0:
            return value
        target = model_with_rows.all_layers[model_with_rows.row_to_main_layer[row]]
        alive = int(pruning.alive_filters(target).numel()) if hasattr(target, "weight") else None
        keep, _is_stop, feasible = fortify.effective_rates({0: value}, 0.0, group_width=alive)[0]
        return float(keep) if feasible else 1.0

    def group_locked(self, layer_idx: int) -> bool:
        """
        True when ``layer_idx`` owns a group already cut this pass (group-once switch), or a
        group whose cut an eval rollback undid (locked for the rest of the walk).
        """
        if int(layer_idx) in (getattr(self, "_rollback_locked_layers", None) or set()):
            return True
        if not fortify.group_once_per_pass():
            return False
        return int(layer_idx) in (getattr(self, "_pass_locked_layers", None) or set())

    def rollback_snapshot(self) -> dict:
        """Pre-step copy for ``SPECTRA_EVAL_ROLLBACK``: the model and the val accuracy it scored."""
        return {"model": copy.deepcopy(self.current_model),
                "val_acc": float(self.last_val_acc if self.last_val_acc is not None
                                 else self.original_acc)}

    def rollback_to(self, snapshot: dict) -> list:
        """
        Undo the last step's cut: restore ``snapshot`` and lock the layers of the group that was
        cut (the target layer alone after a masked fallback) for the rest of the walk. The row
        counter and pass bookkeeping stay where the step left them. Returns the locked indices.
        """
        outcome = getattr(self, "last_step_outcome", None) or {}
        indices = [int(i) for i in (outcome.get("group_layer_indices") or [])]
        if not indices and self.last_step_layer_idx is not None:
            indices = [int(self.last_step_layer_idx)]
        self.current_model = snapshot["model"]
        self.last_val_acc = float(snapshot["val_acc"])
        self._ratio_cache = {}
        locked = getattr(self, "_rollback_locked_layers", None)
        if locked is None:
            locked = self._rollback_locked_layers = set()
        locked.update(indices)
        return indices

    def tau(self) -> float:
        """τ in force: ``SPECTRA_TRAIN_TAU`` in AGENT_TRAIN mode, else ``--allowed_acc_reduction``."""
        base = float(self.conf.allowed_acc_reduction)
        if self.mode == AGENT_TRAIN:
            return fortify.train_tau(base)
        return base

    def _register_group_lock(self, prune_outcome) -> None:
        """
        Remember the layers of a group that was just structurally cut.

        Always counts the cut for this episode (group-cost channel, ``episode_group_cuts``);
        locks the owner rows for the rest of the pass only under ``SPECTRA_GROUP_ONCE_PER_PASS``.
        """
        outcome = prune_outcome or {}
        if outcome.get("mode") != "structural":
            return
        indices = [int(i) for i in (outcome.get("group_layer_indices") or [])]
        if indices:
            cuts = getattr(self, "_episode_group_cuts", None)
            if cuts is None:
                cuts = self._episode_group_cuts = {}
            key = frozenset(indices)
            cuts[key] = cuts.get(key, 0) + 1
        if not fortify.group_once_per_pass():
            return
        locked = getattr(self, "_pass_locked_layers", None)
        if locked is None:
            locked = self._pass_locked_layers = set()
        locked.update(indices)

    def episode_group_cuts(self) -> dict:
        """``{frozenset(owner layer indices): structural cuts this episode}``."""
        return dict(getattr(self, "_episode_group_cuts", None) or {})

    def _end_of_pass_reset(self, num_actions: int, num_rows: int) -> bool:
        """Clear per-pass state at a pass boundary; returns True when a pass just ended."""
        if num_rows <= 0 or num_actions % num_rows != 0:
            return False
        if getattr(self, "_pass_locked_layers", None):
            self._pass_locked_layers.clear()
        return True

    def _recover_after_prune(self, handler, model_with_rows, prune_outcome, is_to_train):
        """
        Post-prune recovery under the recipe in force (``fortify.ft_recipe``).

        * **A** (live default): every parameter trainable, keep the surviving filters,
          ``train_model`` on the train loss (policy training may cap the budget with
          ``SPECTRA_TRAIN_FT_EPOCHS``; eval walks use ``--num_epochs``).
        * **B** (``--train_compressed_layer_only=True``): keep the surviving filters, train
          only the rewritten group (0/32 OK on ResNets, ledger §12).
        * **C-G** (``SPECTRA_FT_REINIT_EDITED=1``, P8): NEON layer replacement — the group the
          structural prune just resized is re-initialised at its new width
          (``pruning.reinit_group_edit``), everything else is frozen (BN-safe), and the new
          group is trained until the **val** accuracy plateaus. Gilad-literal NEON-C.
        * **C-G+** (``SPECTRA_FT_REINIT_THEN_POLISH=1``): C-G, then every parameter is
          unfrozen for a short low-LR full-net polish, also val-selected.

        A masked fallback rewrites no module, so there is nothing to replace: that step takes
        recipe A and is recorded as ``ft_recipe="A"`` / ``reinit=False``. Identity steps never
        reach this method. The handler is left frozen the way the last phase left it; ``step``
        unfreezes everything before handing the model on.
        """
        recipe = fortify.ft_recipe(bool(self.conf.train_compressed_layer_only))
        group_edit = getattr(model_with_rows, "last_group_edit", None)
        if recipe in ("C-G", "C-G+") and prune_outcome.get("mode") == "structural" and group_edit:
            reinit_summary = pruning.reinit_group_edit(model_with_rows, group_edit,
                                                       scope=fortify.ft_reinit_scope())
            prune_outcome["reinit"] = reinit_summary
            prune_outcome["ft_recipe"] = recipe
            utils.print_flush(
                f"P8 {recipe} (scope={reinit_summary['scope']}): layer replacement — "
                f"{reinit_summary['producers']} producer(s) "
                f"re-drawn at width {group_edit.get('new_width')}, {reinit_summary['norms']} norm(s) "
                f"reset, consumers full={reinit_summary['consumers_full']} "
                f"slice={reinit_summary['consumers_slice']}, "
                f"{reinit_summary['params_reinit']} params from scratch")
            edited = list(getattr(model_with_rows, "last_edited_param_ids", None) or [])
            handler.freeze_all_layers_but_pruned(edited)
            if is_to_train:
                with logging_utils.stage("step.finetune", level=logging.DEBUG):
                    select_val = self.val_loader if fortify.ft_reinit_select() == "val" else None
                    handler.train_model(
                        self.train_loader, max_epochs=fortify.ft_reinit_epochs(),
                        patience=fortify.ft_reinit_patience(), val_loader=select_val,
                        tag=f"{recipe} group")
                    if recipe == "C-G+":
                        handler.unfreeze_all_layers()
                        handler.train_model(
                            self.train_loader, max_epochs=fortify.ft_polish_epochs(),
                            patience=fortify.ft_polish_patience(), val_loader=self.val_loader,
                            lr_mult=fortify.ft_polish_lr_mult(), tag="C-G+ polish")
            return recipe

        if recipe in ("C-G", "C-G+"):
            prune_outcome["ft_recipe"] = "A"  # no structural edit to replace this step
            prune_outcome["reinit"] = {"reinit": False}
        # Freeze/unfreeze layers based on config. Prefer the modules actually rewritten by
        # the last structural group prune (producers + consumers + norms); the old
        # "pruned row + next layer" rule left resized consumers frozen and made mild
        # compressions unrecoverable under the -5 pp reward cliff (see recovery probes).
        if self.conf.train_compressed_layer_only:
            edited = getattr(model_with_rows, "last_edited_param_ids", None)
            params_to_keep_trainable = (
                edited if edited
                else build_param_names_to_keep_trainable(model_with_rows, self.row_idx - 1))
            handler.freeze_all_layers_but_pruned(params_to_keep_trainable)
        else:
            handler.unfreeze_all_layers()

        # A-LSQ / C-PCA / BN recalibration. Default off. C-G already returned above.
        # C-PCA and A-LSQ then take this same full-net fine-tune, so the comparison
        # with recipe A is the initialisation, not a different training budget.
        if prune_outcome.get("mode") == "structural":
            captured = getattr(self, "_pre_prune_io", None)
            if fortify.ft_pca_reinit():
                pca_summary = recovery_edits.apply_pca(model_with_rows, captured)
                prune_outcome["pca"] = pca_summary
                prune_outcome["ft_recipe"] = "C-PCA"
                utils.print_flush(
                    f"C-PCA: producers {pca_summary['pca_producers']}, "
                    f"consumers {pca_summary['pca_consumers']}, "
                    f"width {pca_summary['width']}, skipped {pca_summary['skipped']}")
            elif fortify.ft_lsq_consumers():
                lsq_summary = recovery_edits.apply_lsq(model_with_rows, captured)
                prune_outcome["lsq"] = lsq_summary
                prune_outcome["ft_recipe"] = "A-LSQ"
                utils.print_flush(
                    f"A-LSQ: consumers refit {lsq_summary['lsq']}, skipped {lsq_summary['skipped']}")
            if fortify.ft_bn_recal():
                n_norms = recovery_edits.recalibrate_batchnorm(
                    model_with_rows.model, self.train_loader, self.conf.device,
                    n_batches=max(8, fortify.ft_calib_batches()))
                prune_outcome["bn_recal"] = n_norms
                if n_norms:
                    utils.print_flush(f"BN recalibration: {n_norms} BatchNorm module(s)")

        group_first = fortify.ft_group_first_epochs()
        edited = list(getattr(model_with_rows, "last_edited_param_ids", None) or [])
        if (group_first and is_to_train and recipe == "A" and edited
                and prune_outcome.get("mode") == "structural"):
            handler.freeze_all_layers_but_pruned(edited)
            with logging_utils.stage("step.finetune", level=logging.DEBUG):
                handler.train_model(self.train_loader, max_epochs=group_first,
                                    patience=fortify.ft_group_first_patience(), tag="group-first")
            handler.unfreeze_all_layers()
            prune_outcome["ft_group_first"] = group_first

        if is_to_train:
            with logging_utils.stage("step.finetune", level=logging.DEBUG):
                # Policy-training episodes may use a shorter recovery budget
                # (SPECTRA_TRAIN_FT_EPOCHS); eval walks always use --num_epochs.
                ft_kwargs = {}
                if self.mode == AGENT_TRAIN and fortify.train_ft_epochs() is not None:
                    ft_kwargs = {"max_epochs": fortify.train_ft_epochs(),
                                 "patience": fortify.train_ft_patience()}
                if self.mode != AGENT_TRAIN and fortify.ft_cuda_graph():
                    ft_kwargs["cuda_graph"] = True
                handler.train_model(self.train_loader, **ft_kwargs)
        if fortify.ft_pca_reinit():
            return "C-PCA"
        if fortify.ft_lsq_consumers():
            return "A-LSQ"
        return "B" if self.conf.train_compressed_layer_only else "A"

    def step(self, compression_rate, is_to_train=True, ranking=None):
        """
        Compress the network, then move to the next state.

        Args:
            compression_rate (float): Factor to reduce layer size.
            is_to_train (bool): Whether to train after compression.
            ranking (str, optional): Filter-importance criterion chosen by the *action*
                (``--action_rankings``); ``None`` keeps the environment default
                (``SPECTRA_FILTER_IMPORTANCE``, L1).

        Returns:
            Tuple: Next state, reward, and done flag.
        """
        step_timer = logging_utils.Timer().__enter__()
        model_with_rows = ModelWithRows(self.current_model)
        self._pre_prune_io = None
        requested_rate = float(compression_rate)
        stop = fortify.is_stop_rate(requested_rate)
        if requested_rate < 0.0:
            # A negative rate is STOP under the budget menu, and identity otherwise,
            # so an accidental minus never deletes every channel.
            compression_rate = 1.0
        elif fortify.action_menu() == "budget" and requested_rate < 1.0:
            # Same mapping as the legal mask and the action-cost slots (fortify.effective_rates).
            owned = recovery_edits.group_param_fraction(model_with_rows, self.row_idx - 1)
            _target = model_with_rows.all_layers[model_with_rows.row_to_main_layer[self.row_idx - 1]]
            _alive = int(pruning.alive_filters(_target).numel()) if hasattr(_target, "weight") else None
            keep, _is_stop, feasible = fortify.effective_rates(
                {0: requested_rate}, owned, group_width=_alive)[0]
            if not feasible:
                # The mask should have hidden this action; never round it to a one-channel cut.
                utils.print_flush(
                    f"budget action: remove {requested_rate:.4f} of the network exceeds what the "
                    f"group owns ({owned:.4f}); treated as identity")
                keep = 1.0
            compression_rate = keep
            utils.print_flush(
                f"budget action: remove {requested_rate:.4f} of the network "
                f"through a group that owns {owned:.4f} -> keep rate {compression_rate:.4f}")
        elif fortify.width_ladder_max() > 0 and 0.0 < requested_rate < 1.0:
            compression_rate = self._ladder_keep_rate(model_with_rows, self.row_idx - 1, requested_rate)
            if abs(compression_rate - requested_rate) > 1e-9:
                utils.print_flush(f"width ladder: rate {requested_rate} -> keep rate {compression_rate:.4f}")
        if self.target_keep is not None and 0.0 < float(compression_rate) < 1.0:
            landed_rate, landed_kept = self._land_on_target(float(compression_rate))
            if landed_kept is not None:
                utils.print_flush(
                    f"fixed target: rate {float(compression_rate):.4f} -> {landed_rate:.4f} lands at "
                    f"params x{landed_kept:.4f} (target x{self.target_keep:.4f})")
                compression_rate = landed_rate

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
        flops_before = None
        flops_after = None
        need_flops = fortify.reward_needs_flops()
        if need_flops and compression_rate != 1:
            flops_before = utils.calc_flops(self.current_model, self._input_shape())
        prune_outcome = {"mode": "identity"}

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Modify the model in-place
            with logging_utils.stage("step.prune", level=logging.DEBUG):
                if (fortify.ft_lsq_consumers() or fortify.ft_pca_reinit()) and self.train_loader is not None:
                    self._pre_prune_io = recovery_edits.capture_pre_prune(
                        model_with_rows, self.row_idx - 1, self.train_loader, self.conf.device,
                        n_batches=fortify.ft_calib_batches())
                if pruning.normalize_importance_mode(ranking) == "taylor" and ranking:
                    # Data-dependent criterion: one forward+backward on a train batch, bound
                    # right before ranking so a resized model never reads stale scores.
                    pruning.bind_taylor_scores(self.current_model, self.train_loader, self.conf.device)
                if self.conf.prune:
                    model_with_rows = prune_current_model(
                        model_with_rows, compression_rate, self.row_idx - 1,
                        input_shape=self._input_shape(), importance=ranking)
                else:
                    model_with_rows = create_new_model_with_new_weights(model_with_rows, compression_rate,
                                                                        self.row_idx - 1)
            prune_outcome = dict(getattr(model_with_rows, "last_prune_outcome", {}) or {})
            # Group-once: later rows owning this group are identity-only for the rest of
            # the pass (fortify.group_once_per_pass; no-op when the switch is off).
            self._register_group_lock(prune_outcome)

            # Prepare model handler
            learning_handler_new_model = self.create_learning_handler(model_with_rows.model)
            self._recover_after_prune(learning_handler_new_model, model_with_rows, prune_outcome,
                                      is_to_train)
        self.last_step_outcome = prune_outcome
        self.last_step_layer_idx = current_layer_idx

        # Evaluate the compressed model
        learning_handler_new_model.model.eval()
        with logging_utils.stage("step.evaluate", level=logging.DEBUG):
            new_acc = learning_handler_new_model.evaluate_model(self.val_loader)
        self.last_val_acc = float(new_acc)

        # Realized size before reward: CNN group edits ≠ nominal (1-rate).
        params_after = utils.calc_num_parameters(learning_handler_new_model.model)
        if need_flops and compression_rate != 1:
            flops_after = utils.calc_flops(
                learning_handler_new_model.model, self._input_shape())
        tau = self.tau()
        reward_rate = reward_compression_rate(
            prune_outcome, compression_rate, params_before, params_after,
            new_acc, self.original_acc, tau)
        reward = utils.compute_reward(
            new_acc, self.original_acc, reward_rate,
            params_before=params_before, params_after=params_after,
            flops_before=flops_before, flops_after=flops_after, tau=tau)
        utils.trace_reward(
            self.selected_net_path, reward_rate, new_acc, self.original_acc, reward,
            params_before=params_before, params_after=params_after)
        target_step_reward = None
        if self.target_keep is not None:
            # v10: the step's change in val accuracy, so the return telescopes to the val Δacc
            # where the episode ends; the miss penalty is added below once it is done.
            prev = self._target_prev_acc if self._target_prev_acc is not None else self.original_acc
            target_step_reward = (float(new_acc) - float(prev)) * 100.0
            reward = target_step_reward
            self._target_prev_acc = float(new_acc)
        delta_pp = (new_acc - self.original_acc) * 100.0
        nominal = (1.0 - float(reward_rate)) * 100.0
        rho_step = utils.unified_rho(
            nominal, params_before, params_after, flops_before, flops_after)
        overshoot = max(0.0, -delta_pp - tau)
        self.episode_rho_sum += max(0.0, float(rho_step))
        self.episode_overshoot_sum += overshoot
        if overshoot > 0:
            self.episode_any_over = True
        self.episode_step_over.append(overshoot > 0)
        if overshoot <= 0 and self.original_params:
            kept_now = params_after / max(float(self.original_params), 1.0)
            self._account_inband_point(kept_now, delta_pp, tau)

        # Move to next state
        self.row_idx += 1
        learning_handler_new_model.unfreeze_all_layers()
        old_model = self.current_model
        self.current_model = learning_handler_new_model.model
        # Every memoized size probe describes the pre-step model
        self._ratio_cache = {}
        del old_model
        del learning_handler_new_model
        # Identity steps do not allocate a new graph; skipping the cache flush avoids a
        # stall on every remaining eval step after the param floor.
        if compression_rate != 1:
            torch.cuda.empty_cache()
            gc.collect()

        # Check termination / wrap before encoding so the next-state marker can
        # point at the layer the *next* action will actually prune.
        num_rows = len(model_with_rows.all_rows) - 1  # Only FC and Conv layers trigger a new row
        self.actions_history.append(compression_rate)
        num_actions = len(self.actions_history)
        # As self.row_idx - 1 is the current appraised row, the index should not drop below 1
        self.row_idx = max(1, self.row_idx % (num_rows + 1))
        done = num_actions >= num_rows * self.conf.passes
        if stop:
            # The step itself was an identity. The return for ending here is the
            # slack-weighted area already accumulated (0 when nothing in-band was kept),
            # in the same units as the per-step in-band reward (+ρ in percentage points of
            # the network removed): area is a fraction × slack fraction, hence ×100.
            done = True
            reward = fortify.stop_reward_scale() * float(self.episode_inband_area())
        if self.target_keep is not None:
            kept_now = params_after / max(float(self.original_params), 1.0)
            # Only at or below the target: the eval's size point (first point <= target) must exist
            # on every walk that reached it.
            if kept_now <= float(self.target_keep) + 1e-9:
                done = True
            reward = target_step_reward
            if done:
                self._target_final = {"kept": kept_now, "delta_pp": delta_pp}
                reward += fortify.target_score(delta_pp, kept_now, self.target_keep) - delta_pp
                utils.print_flush(
                    f"fixed target: episode ends at params x{kept_now:.4f} (target "
                    f"x{self.target_keep:.4f}) with val Δacc {delta_pp:+.2f} pp; return "
                    f"{fortify.target_score(delta_pp, kept_now, self.target_keep):+.2f}")
        # A completed pass releases the group-once locks so the next pass may cut again.
        self._end_of_pass_reset(num_actions, num_rows)
        encode_idx = current_layer_idx
        if fortify.state_align_next() and (not done) and num_rows > 0:
            encode_idx = model_with_rows.row_to_main_layer[self.row_idx - 1]

        # Extract features for the next state. The dependency analysis is redone here because
        # the compression just applied changed the graph. update_indices stay the row that
        # just changed (activation refresh); encode_idx is the layer about to be pruned
        # when SPECTRA_STATE_ALIGN=next.
        with logging_utils.stage("step.feature_extraction", level=logging.DEBUG):
            kept = utils.calc_num_parameters(self.current_model) / max(self.original_params, 1e-9)
            total_steps = max(1, num_rows * int(self.conf.passes))
            extras = [fortify.accuracy_slack(delta_pp, tau),
                      min(1.0, num_actions / total_steps)]
            # NEON "feature-maps update" (P8, SPECTRA_REFRESH_ALL_FEATURES): after a non-identity
            # step every layer's activation moments are re-extracted — the edit changed the
            # input of every downstream layer, not only the edited row's span.
            refresh = None if (fortify.refresh_all_features() and compression_rate != 1) else update_indices
            target_extras = (fortify.target_channels(kept, self.target_keep)
                             if self.target_keep is not None else None)
            fm = self.feature_extractor.encode_to_bert_input(
                model_with_rows, encode_idx, refresh,
                dependency_groups=self._dependency_groups(model_with_rows),
                param_ratio=min(1.0, max(0.0, kept)), extras=extras,
                episode_cuts=self.episode_group_cuts(),
                target_extras=target_extras, layer_sens=self._layer_sens)

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
            requested_rate=requested_rate,
            stop=int(stop),
            ranking=ranking,
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
            ft_recipe=prune_outcome.get("ft_recipe"),
            reinit_params=(prune_outcome.get("reinit") or {}).get("params_reinit"),
            old_width=prune_outcome.get("old_width"),
            new_width=prune_outcome.get("new_width"),
            # Requested vs. applied: a rate the layer's width cannot express (0.9 of 6
            # channels) would otherwise look like a normal action in the trajectory
            realized_rate=(round(prune_outcome["new_width"] / prune_outcome["old_width"], 4)
                           if prune_outcome.get("old_width") else None),
            seconds=round(step_timer.seconds, 3),
            done=bool(done),
            target_keep=(round(float(self.target_keep), 5) if self.target_keep is not None else None),
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

        # Exact counts. The ``(M)`` columns below are rounded to 3 decimals for display;
        # ratios must not be formed from them: thin r20-w2 has 4 556 parameters, so
        # ``round(n / 1e6, 3)`` quantises its kept fraction to steps of 0.2 (the ledger's
        # ``params x0.600`` on that net means anything in [0.55, 0.77)). The TRAJ counter
        # (``param_ratio``) was always exact; this makes ``pass k/K`` agree with it.
        new_params = int(utils.calc_num_parameters(compressed_model))
        origin_params = int(utils.calc_num_parameters(original_model))
        new_effective = int(pruning.count_effective_parameters(compressed_model))
        new_flops = float(utils.calc_flops(compressed_model, input_shape))
        origin_flops = float(utils.calc_flops(original_model, input_shape))

        # Store results
        result_entry = {
            'model': self.selected_net_path,
            'pass': f'{len(self.actions_history) // (len(model_with_rows.all_rows) - 1)}'
                    f' / {self.conf.passes}',
            'fold': fold_str,
            'new_acc': round(new_lh.evaluate_model(dataset_loader), 3),
            'origin_acc': round(origin_lh.evaluate_model(dataset_loader), 3),
            'new_param (M)': round(new_params / 1e6, 3),
            'origin_param (M)': round(origin_params / 1e6, 3),
            'new_effective_param (M)': round(new_effective / 1e6, 3),
            'new_flops (M)': round(new_flops / 1e6, 3),
            'origin_flops (M)': round(origin_flops / 1e6, 3),
            'new_param': new_params,
            'origin_param': origin_params,
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
        param_ratio = new_params / max(origin_params, 1)
        flops_ratio = new_flops / max(origin_flops, 1e-9)
        effective_ratio = new_effective / max(origin_params, 1)
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
            new_param=new_params,
            origin_param=origin_params,
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
        handler = ClassificationHandler(
            new_model,
            torch.nn.CrossEntropyLoss()
        )
        if getattr(self, "kd_teacher", None) is not None:
            handler.kd_teacher = self.kd_teacher
        return handler


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


def _rebind_model(model_with_rows, model):
    """Point ModelWithRows at a restored module tree after a rolled-back structural prune."""
    model_with_rows.model = model
    model_with_rows.all_layers = []
    model_with_rows.layer_parents = []
    model_with_rows.extract_layers_from_model(model)
    model_with_rows.all_rows, model_with_rows.row_to_main_layer = (
        model_with_rows.split_and_map_layers_to_rows())


def group_owner_indices(model_with_rows, group):
    """
    ``all_layers`` indices of the layers that *produce* ``group``'s channel dimension.

    Producers and depthwise members own the dimension; consumers and norms only read it
    and are not rows that could cut it again. Empty for ``None`` / unresolved groups.
    """
    if group is None:
        return []
    owners = list(getattr(group, "producers", [])) + list(getattr(group, "depthwise", []))
    index_of = {id(layer): idx for idx, layer in enumerate(model_with_rows.all_layers)}
    return sorted({index_of[id(m)] for m in owners if id(m) in index_of})


def dummy_forward_ok(model, input_shape=None):
    """
    True if a dummy batch runs. ShuffleNet grouped-conv mismatches raise here
    instead of hours into fine-tune (C9 C100 ShuffleNet, jobs 20270291/293/295).
    Checked after every structural prune, not only depthwise groups: the C100
    crash was a producer resize that left a later grouped conv mismatched.
    """
    device = next(model.parameters()).device
    in_ch = 3
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            in_ch = int(module.in_channels)
            break
    shapes = []
    if input_shape is not None:
        shapes.append(tuple(input_shape))
    for spatial in (32, 224, 28):
        cand = (in_ch, spatial, spatial)
        if cand not in shapes:
            shapes.append(cand)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for shape in shapes:
                try:
                    model(torch.zeros(1, *shape, device=device))
                    return True
                except Exception:
                    continue
    finally:
        model.train(was_training)
    return False


def prune_current_model(model_with_rows, compression_rate, row_to_prune_idx,
                        *, quiet=False, record=True, input_shape=None, importance=None):
    """
    Compress the target layer by removing its least important output filters.

    The layer is physically shrunk (and its consumers resized) whenever the dependency
    chain permits; otherwise the filters are masked in place. See src/pruning.py.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be pruned
        compression_rate (float):         The desired compression rate for pruning
        row_to_prune_idx (int):           Index of the row whose first layer is to be pruned
        quiet (bool):                     Skip log lines (eval look-ahead dry-run).
        record (bool):                    Skip run_recorder writes (eval look-ahead dry-run).
        input_shape (tuple, optional):    NCHW spatial shape for the grouped-conv dummy
            forward (CIFAR 32, ImageNet 224). Tried 32/224/28 if omitted.
        importance (str, optional):       Ranking for this cut (``l1``/``fpgm``/…); ``None``
            keeps ``SPECTRA_FILTER_IMPORTANCE``.

    Returns:
        pruned_model_with_rows (ModelWithRows): The pruned model
    """
    model_with_rows.unwrap_model()
    pruning.bind_bn_scales(model_with_rows.model)

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
    backup = copy.deepcopy(model_with_rows.model) if (group is not None and group.prunable) else None
    rolled_back_grouped = False
    # Rows that own this group's channel dimension (producers + depthwise), as all_layers
    # indices. Captured *before* the edit: the structural prune replaces those modules.
    group_layer_indices = group_owner_indices(model_with_rows, group)

    if group is not None and group.prunable:
        keep_idx = pruning.select_group_survivors(group, compression_rate, mode=importance)
        if keep_idx is not None and pruning.prune_group_structurally(
                model_with_rows, group, keep_idx, mode=importance):
            if not dummy_forward_ok(model_with_rows.model, input_shape):
                if not quiet:
                    utils.print_flush(
                        f"Layer {layer_to_prune_idx}: structural prune broke a dummy "
                        f"forward; restoring and masking")
                _rebind_model(model_with_rows, backup)
                layer_to_prune = model_with_rows.all_layers[layer_to_prune_idx]
                rolled_back_grouped = True
            else:
                coupled = len(group.producers) + len(group.depthwise)
                if not quiet:
                    utils.print_flush(
                        f"Layer {layer_to_prune_idx}: width {old_width} -> {keep_idx.numel()} "
                        f"across {coupled} coupled layer(s), {len(group.consumers)} consumer(s) resized")
                edited_ids = list(getattr(model_with_rows, "last_edited_param_ids", []) or [])
                model_with_rows.last_prune_outcome = {
                    "mode": "structural", "reason": None, "old_width": old_width,
                    "new_width": int(keep_idx.numel()), "coupled_layers": coupled,
                    "consumers_resized": len(group.consumers),
                    "trainable_param_count": len(edited_ids),
                    "group_layer_indices": group_layer_indices,
                }
                if record:
                    recorder.record("prune", mode="structural", layer_index=layer_to_prune_idx,
                                    layer_type=type(layer_to_prune).__name__, rate=compression_rate,
                                    old_width=old_width, new_width=int(keep_idx.numel()),
                                    coupled_layers=coupled, consumers_resized=len(group.consumers))
                model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)
                return model_with_rows

    # "no dependency group resolved" conflated four different failures, so the logs could not
    # say whether the library needs a new fx rule, a new resize rule, or nothing at all.
    if rolled_back_grouped:
        reason = "structural prune broke dummy forward; masked instead"
    elif trace_error is not None:
        reason = f"fx trace raised {trace_error}"
    elif groups is None:
        reason = "model is not symbolically traceable (dynamic control flow)"
    elif group is None:
        reason = f"layer produces no channel group (of {len(groups)} resolved)"
    elif not group.prunable:
        reason = group.reason
    else:
        reason = "structural edit rejected (target width equals current width)"

    keep_idx = pruning.select_surviving_filters(layer_to_prune, compression_rate, mode=importance)
    pruning.mask_layer_filters(layer_to_prune, keep_idx)
    if not quiet:
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
    model_with_rows.last_group_edit = None  # nothing for P8 to re-initialise
    if record:
        recorder.issue(kind, reason, layer_index=layer_to_prune_idx,
                       layer_type=type(layer_to_prune).__name__, rate=compression_rate)
        recorder.record("prune", mode="floor" if at_floor else "masked", reason=reason,
                        layer_index=layer_to_prune_idx,
                        layer_type=type(layer_to_prune).__name__, rate=compression_rate,
                        old_width=old_width, new_width=int(keep_idx.numel()))

    model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)

    return model_with_rows
