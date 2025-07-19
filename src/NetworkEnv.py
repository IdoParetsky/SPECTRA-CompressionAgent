import numpy as np
import pandas as pd
import os
import time
import gc
import torch
from torch import nn
from torch.nn.utils import prune
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
        reset(test_net_path=None, test_model_pkl=None, test_loaders=None):
            Resets the environment by loading a new model from a pickle file and dataset dataloaders.
            If test parameters are provided, uses them instead of selecting from 'input_dict'
            (utilized in cross-validation to evaluate the test dataset over a train dataset environment).

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
        self.selected_net_path = None
        self.current_model = None
        self.feature_extractor = None

        # EVAL_TRAIN / EVAL_TEST when called from a2c_agent_reinforce_runner.py's evaluate_model(),
        # used for accuracy calculation in NetworkEnv's compute_and_log_results().
        # AGENT_TRAIN when called from A2C_Agent_Reinforce.py, skipping compute_and_log_results()
        self.mode = mode

        # Full database if in agent training mode, else evaluation database (user input)
        self.data_dict = self.conf.database_dict if self.mode == AGENT_TRAIN else self.conf.input_dict
        self.networks = networks or list(self.data_dict.keys())
        self.curr_net_index = -1
        np.random.shuffle(self.networks)  # Randomize order

        # Initialize BERT Input Modeler
        self.bert_modeler = BERTInputModeler()

        # "N/A" / an integer representing the fold number within the amount of folds in cross-validation evaluation
        # when called from a2c_agent_reinforce_runner.py's evaluate_model(),
        # used for logging via NetworkEnv's compute_and_log_results().
        # None (irrelevant) when called from A2C_Agent_Reinforce.py.
        self.fold_idx = fold_idx

        # t_start is assigned in a2c_agent_reinforce_runner.py's evaluate_model(),
        # and utilized in NetworkEnv's compute_and_log_results()
        self.t_start = None  # a Model's evaluation start time

    def reset(self, test_net_path=None):
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

        # If test model path is provided, use it directly (for cross-validation)
        if test_net_path is None:
            self.curr_net_index = (self.curr_net_index + 1) % len(self.networks)
            self.selected_net_path = self.networks[self.curr_net_index]
        else:
            self.selected_net_path = test_net_path

        self.current_model, dataset_name = utils.load_model_from_cache(self.selected_net_path)
        self.train_loader, self.val_loader, self.test_loader = self.conf.dataloaders_dict[utils.DATASET_ALIASES[dataset_name]][0]


        model_with_rows = ModelWithRows(self.current_model)
        utils.print_flush(f"Loading {self.selected_net_path}")
        # Prepare feature extractor with training data
        self.feature_extractor = FeatureExtractor(self.train_loader, self.conf.device)
        utils.print_flush("env.reset - Starting FM Extraction")
        fm = self.feature_extractor.encode_to_bert_input(model_with_rows,
            model_with_rows.row_to_main_layer[self.row_idx - 1])
        utils.print_flush("env.reset - Finished FM Extraction")

        # Evaluate original model accuracy
        learning_handler_original_model = self.create_learning_handler(self.current_model)
        self.original_acc = learning_handler_original_model.evaluate_model(self.val_loader)

        # After feature extraction and setup
        torch.cuda.empty_cache()
        gc.collect()

        return fm

    def step(self, compression_rate, is_to_train=True):
        """
        Compress the network, then move to the next state.

        Args:
            compression_rate (float): Factor to reduce layer size.
            is_to_train (bool): Whether to train after compression.

        Returns:
            Tuple: Next state, reward, and done flag.
        """
        model_with_rows = ModelWithRows(self.current_model)

        # Determine affected layers (from current row up to start of next row)
        current_layer_idx = model_with_rows.row_to_main_layer[self.row_idx - 1]
        next_layer_idx = model_with_rows.row_to_main_layer[self.row_idx] \
            if self.row_idx < len(model_with_rows.row_to_main_layer) else len(model_with_rows.all_layers)
        update_indices = list(range(current_layer_idx, next_layer_idx))
        utils.print_flush(f"Step {self.row_idx - 1} - Layer {current_layer_idx}, Compression Rate: {compression_rate}")

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Modify the model in-place
            if self.conf.prune:
                model_with_rows = prune_current_model(model_with_rows, compression_rate, self.row_idx - 1)
            else:
                model_with_rows = create_new_model_with_new_weights(model_with_rows, compression_rate, self.row_idx - 1)

            # Prepare model handler
            learning_handler_new_model = self.create_learning_handler(model_with_rows.model)

            # Freeze/unfreeze layers based on config
            if self.conf.train_compressed_layer_only:
                params_to_keep_trainable = build_param_names_to_keep_trainable(model_with_rows, self.row_idx - 1)
                learning_handler_new_model.freeze_all_layers_but_pruned(params_to_keep_trainable)
            else:
                learning_handler_new_model.unfreeze_all_layers()

            if is_to_train:
                learning_handler_new_model.train_model(self.train_loader)

        # Evaluate the compressed model
        learning_handler_new_model.model.eval()
        new_acc = learning_handler_new_model.evaluate_model(self.val_loader)

        # Compute reward
        reward = utils.compute_reward(new_acc, self.original_acc, compression_rate)

        # Move to next state
        self.row_idx += 1
        learning_handler_new_model.unfreeze_all_layers()
        old_model = self.current_model
        self.current_model = learning_handler_new_model.model
        del old_model
        del learning_handler_new_model
        torch.cuda.empty_cache()
        gc.collect()

        # Extract features for BERT
        utils.print_flush("env.step - Starting FM Extraction")
        fm = self.feature_extractor.encode_to_bert_input(model_with_rows, current_layer_idx, update_indices)
        utils.print_flush("env.step - Finished FM Extraction")

        # Check termination condition
        num_rows = len(model_with_rows.all_rows) - 1  # Only FC and Conv layers trigger a new row
        self.actions_history.append(compression_rate)
        num_actions = len(self.actions_history)
        # As self.row_idx - 1 is the current appraised row, the index should not drop below 1
        self.row_idx = max(1, self.row_idx % (num_rows + 1))
        done = num_actions >= num_rows * self.conf.passes

        # Log model evaluation metrics after each pass and flush to CSV.
        if self.mode != AGENT_TRAIN and (done or num_actions % num_rows == 0):
            self.compute_and_log_results(model_with_rows)

        # Save the final pruned model to a checkpoint file,
        # if requested by the user via self.conf.save_pruned_checkpoints = True
        if done and self.conf.save_pruned_checkpoints:
            self.save_pruned_checkpoint()

        return fm, reward, done

    def compute_and_log_results(self, model_with_rows, t_curr=time.perf_counter()):
        """
        Compute accuracy according to eval mode (train / test datasets), number of params and FLOPs.
        Log model evaluation metrics after each pass and flush to CSV.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            t_curr (float):    Time of log, to calculate evaluation time
        """
        # Retrieve original & compressed models
        original_model = self.data_dict[self.selected_net_path][0]
        compressed_model = self.current_model

        # Create learning handlers
        new_lh = self.create_learning_handler(compressed_model)
        origin_lh = self.create_learning_handler(original_model)

        dataset_loader = self.test_loader if self.mode == "test" else self.train_loader

        fold_str = self.fold_idx if self.fold_idx == "N/A" else f"{self.fold_idx} / {self.conf.n_splits}"

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
            'new_flops (M)': round(utils.calc_flops(compressed_model) / 1e6, 3),
            'origin_flops (M)': round(utils.calc_flops(original_model) / 1e6, 3),
            'new_model_arch': utils.get_model_layers_str(compressed_model),
            'origin_model_arch': utils.get_model_layers_str(original_model),
            'evaluation_time': t_curr - self.t_start
        }

        file_path = f"./models/Reinforce_Evaluation/results_{self.conf.test_name}_{self.mode}_{self.conf.test_ts}.csv"

        # Check if file exists to determine if headers should be written
        file_exists = os.path.exists(file_path)

        # Convert result dictionary into a DataFrame row
        df_entry = pd.DataFrame([result_entry])

        # Append row to CSV (creates file if it doesn't exist)
        df_entry.to_csv(file_path, mode='a', header=not file_exists, index=False)

    def save_pruned_checkpoint(self):
        """
        Save the final pruned model to a checkpoint file.
        The filename keeps the original name but replaces the last '.' before the extension with '_pruned.'.
        """
        # Extract filename and replace only the last dot (as the filename might contain decimal points)
        filename = os.path.basename(self.selected_net_path)  # Get the file name from the path
        name_parts = filename.rsplit('.', 1)  # Split at the last dot
        model_name = f"{name_parts[0]}_pruned.{name_parts[1]}" if len(name_parts) == 2 else f"{filename}_pruned"
        # Create timestamped save path
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Prevent overwriting
        save_path = f"./pruned_models/{model_name}_{timestamp}"
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_to_save = self.current_model
        # If it's wrapped in DDP, unwrap before saving state_dict
        if isinstance(model_to_save, DDP):
            model_to_save = model_to_save.module

        # Save state dict safely (on rank 0 only)
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.save(model_to_save.state_dict(), save_path)
            utils.print_flush(f"Pruned model saved at {save_path}")

        # Sync all processes if in DDP
        if dist.is_initialized():
            dist.barrier()
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

    # Determine new size
    old_size = layer_to_resize.out_features if isinstance(layer_to_resize, nn.Linear) else layer_to_resize.out_channels
    new_size = int(np.ceil(compression_rate * old_size))

    # Replace the layer with a resized fresh version (no weight copying)
    if isinstance(layer_to_resize, nn.Linear):
        resized_layer = nn.Linear(
            in_features=layer_to_resize.in_features,
            out_features=new_size,
            bias=layer_to_resize.bias is not None
        )
    elif isinstance(layer_to_resize, nn.Conv2d):
        resized_layer = nn.Conv2d(
            in_channels=layer_to_resize.in_channels,
            out_channels=new_size,
            kernel_size=layer_to_resize.kernel_size,
            stride=layer_to_resize.stride,
            padding=layer_to_resize.padding,
            dilation=layer_to_resize.dilation,
            groups=layer_to_resize.groups,
            bias=layer_to_resize.bias is not None
        )
    else:
        raise NotImplementedError("Resizing not implemented for this layer type.")

    model_with_rows.all_layers[layer_to_resize_idx] = resized_layer

    # Rewrap for DDP compatibility
    model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)
    return model_with_rows


# TODO: 2nd arch onwards the amount of filters pruned does not match the intended percentage!
def prune_current_model(model_with_rows, compression_rate, row_to_prune_idx):
    """
    Create a new model by pruning filters in the target layer.

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

    prune.ln_structured(layer_to_prune, name='weight', amount=(1 - compression_rate), n=1, dim=0)

    # Optional debug print
    mask = getattr(layer_to_prune, 'weight_mask', None)
    if mask is not None:
        pruned_count = torch.sum(mask == 0).item()
        total_count = mask.numel()
        utils.print_flush(f"Pruned and removed {pruned_count}/{total_count} "
            f"({round(100 * pruned_count / total_count)}%) filters in layer {layer_to_prune_idx}")
    else:
        utils.print_flush(f"Warning: No pruning mask found for layer {layer_to_prune_idx}")

    prune.remove(layer_to_prune, name='weight')  # Apply mask permanently
    model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)

    return model_with_rows
