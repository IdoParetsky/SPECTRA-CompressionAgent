import copy
from typing import List
import numpy as np
import pandas as pd
import os
import time
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
        reset(test_net_path=None, test_model=None, test_loaders=None):
            Resets the environment by loading a new model and dataset. If test parameters are provided,
            uses them instead of selecting from 'input_dict' (utilized in cross-validation to evaluate the test dataset
            over a train dataset environment).

        step(compression_rate: float, is_to_train: bool = True) -> Tuple[np.ndarray, float, bool]:
            Applies a pruning action, evaluates the compressed model, and moves to the next state.
            Returns the updated state, computed reward, and termination flag.

        compute_and_log_results(t_curr: float = time.perf_counter()):
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

    def reset(self, test_net_path=None, test_model=None, test_loaders=None):
        """ Reset environment with a new CNN model & dataset """
        self.row_idx = 1  # The first row to be a candidate for pruning is self.row_idx - 1 -> index 0
        self.actions_history = []

        # If test model is provided, use it directly (for cross-validation)
        if test_net_path and test_model and test_loaders:
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = test_model, test_loaders
            self.selected_net_path = test_net_path
        else:
            self.curr_net_index = (self.curr_net_index + 1) % len(self.networks)
            self.selected_net_path = self.networks[self.curr_net_index]

            # Load model & dataset from preloaded input_dict
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = self.data_dict[
                self.selected_net_path]

        # TODO: Pinpoint which call affects runtime the most (every reset takes ~7 minutes).
        utils.print_flush(f"Loading {self.selected_net_path}")

        # Prepare feature extractor with training data
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, self.conf.device)
        # TODO: Consider caching FM, as there is only 0/1 change (prev layer) between iter.
        utils.print_flush("env.reset - Starting FM Extraction")
        fm = self.feature_extractor.encode_to_bert_input(
            ModelWithRows(self.current_model).row_to_main_layer[self.row_idx - 1])
        utils.print_flush("env.reset - Finished FM Extraction")

        # Evaluate original model accuracy
        learning_handler_original_model = self.create_learning_handler(self.current_model)
        self.original_acc = learning_handler_original_model.evaluate_model(self.val_loader)

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
        utils.print_flush(f"Step {self.row_idx - 1} - Compression Rate: {compression_rate}")
        model_with_rows = ModelWithRows(self.current_model)

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Create a pruned or resized model
            modified_model_with_rows = prune_current_model(
                model_with_rows, compression_rate, self.row_idx - 1) if self.conf.prune \
                else create_new_model_with_new_weights(model_with_rows, compression_rate, self.row_idx - 1)

            # Prepare model handler
            learning_handler_new_model = self.create_learning_handler(modified_model_with_rows.model)

            # Freeze/unfreeze layers based on config
            if self.conf.train_compressed_layer_only:
                parameters_to_freeze_ids = build_parameters_to_freeze(modified_model_with_rows, self.row_idx - 1)
                learning_handler_new_model.freeze_layers(parameters_to_freeze_ids)
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
        self.current_model = learning_handler_new_model.model

        # Extract features for BERT
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, self.conf.device)
        # TODO: Consider caching FM, as there is only 0/1 change (prev layer) between iter.
        utils.print_flush("env.step - Starting FM Extraction")
        fm = self.feature_extractor.encode_to_bert_input(model_with_rows.row_to_main_layer[self.row_idx - 1])
        utils.print_flush("env.step - Finished FM Extraction")

        # Check termination condition
        num_rows = len(self.feature_extractor.model_with_rows.all_rows) - 1  # Only FC and Conv layers trigger a new row
        num_actions = len(self.actions_history)
        self.actions_history.append(compression_rate)
        # As self.row_idx - 1 is the current appraised row, the index should not drop below 1
        self.row_idx = max(1, self.row_idx % (num_rows + 1))
        done = num_actions >= num_rows * self.conf.passes

        # Log model evaluation metrics after each pass and flush to CSV.
        if self.mode != AGENT_TRAIN and (done or num_actions % num_rows == 0):
            self.compute_and_log_results()

        # Save the final pruned model to a checkpoint file,
        # if requested by the user via self.conf.save_pruned_checkpoints = True
        if done and self.conf.save_pruned_checkpoints:
            self.save_pruned_checkpoint()

        return fm, reward, done

    def compute_and_log_results(self, t_curr=time.perf_counter()):
        """
        Compute accuracy according to eval mode (train / test datasets), number of params and FLOPs.
        Log model evaluation metrics after each pass and flush to CSV.

        Args:
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
            'pass': f'{len(self.actions_history) // (len(self.feature_extractor.model_with_rows.all_rows) - 1)}'
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

        if len(name_parts) == 2:  # Ensure the file has an extension
            model_name = f"{name_parts[0]}_pruned.{name_parts[1]}"
        else:
            model_name = f"{filename}_pruned"  # Handle cases with no extension (unlikely)

        # Create timestamped save path
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Prevent overwriting
        save_path = f"./pruned_models/{model_name}_{timestamp}"

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model checkpoint
        torch.save(self.current_model.state_dict(), save_path)

        utils.print_flush(f"Pruned model saved at {save_path}")

    def create_learning_handler(self, new_model) -> BasicHandler:
        """
        Create appropriate learning handler based on mission type,
        ensuring compatibility with both training & testing scenarios.
        SPECTRA's current implementation supports Classifications tasks only.
        """
        return ClassificationHandler(
            new_model,
            torch.nn.CrossEntropyLoss(),
            torch.optim.Adam(new_model.parameters(), lr=self.conf.learning_rate)
        )


def build_parameters_to_freeze(model_with_rows, row_to_modify_idx):
    """
    Builds a list of parameter IDs for layers to freeze.

    Args:
        model_with_rows (ModelWithRows):   The model wrapped with rows of layers.
        row_to_modify_idx (int):           Index of the row whose first layer is to be pruned / resized

    Returns:
        List[int]: A list of parameter IDs to freeze.
    """
    # TODO: In NEON, the pruned and subsequent rows were frozen instead of the layers. Was it on purpose or an overkill?
    # Extract the layers to freeze (the pruned layer and the subsequent layer)
    layer_to_modify_idx = model_with_rows.row_to_main_layer[row_to_modify_idx]
    layers_to_freeze = model_with_rows.all_layers[layer_to_modify_idx:layer_to_modify_idx + 2]

    # Flatten and collect parameter IDs
    parameters_to_freeze_ids = [id(param) for layer in layers_to_freeze for param in layer.parameters()]

    return parameters_to_freeze_ids


# TODO: Apply here all relevant fixes from prune_current_model()
def create_new_model_with_new_weights(model_with_rows, compression_rate, row_to_resize_idx):
    """
    Replace a layer with a reduced version, adjusting the subsequent layer accordingly.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be resized
        compression_rate (float):         The desired compression rate for resizing
        row_to_resize_idx (int):          Index of the row whose first layer is to be resized

    Returns:
         resized_model_with_rows (ModelWithRows): The resized model.
    """
    resized_model_with_rows = copy.deepcopy(model_with_rows)

    layer_to_resize_idx = resized_model_with_rows.row_to_main_layer[row_to_resize_idx]
    layer_to_resize = resized_model_with_rows.all_layers[layer_to_resize_idx]

    # Determine new size
    old_size = layer_to_resize.out_features if isinstance(layer_to_resize, nn.Linear) else layer_to_resize.out_channels
    new_size = int(np.ceil(compression_rate * old_size))

    # Build the resized version of the current layer
    if isinstance(layer_to_resize, nn.Linear):
        resized_layer = nn.Linear(
            in_features=layer_to_resize.in_features,
            out_features=new_size,
            bias=layer_to_resize.bias is not None
        )

        with torch.no_grad():
            resized_layer.weight.copy_(layer_to_resize.weight[:new_size, :resized_layer.in_features])
            if layer_to_resize.bias is not None:
                resized_layer.bias.copy_(layer_to_resize.bias[:new_size])

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

        with torch.no_grad():
            resized_layer.weight.copy_(layer_to_resize.weight[:new_size, :, :, :])
            if layer_to_resize.bias is not None:
                resized_layer.bias.copy_(layer_to_resize.bias[:new_size])

    # Replace resized layer in model
    resized_model_with_rows.all_layers[layer_to_resize_idx] = resized_layer

    # Adjust the next layer if necessary
    next_layer_idx = layer_to_resize_idx + 1
    next_layer = resized_model_with_rows.all_layers[next_layer_idx]

    if isinstance(next_layer, nn.Linear):
        updated_next_layer = nn.Linear(new_size, next_layer.out_features, bias=next_layer.bias is not None)
        with torch.no_grad():
            updated_next_layer.weight.copy_(
                next_layer.weight[:updated_next_layer.out_features, :updated_next_layer.in_features])
            if next_layer.bias is not None:
                updated_next_layer.bias.copy_(next_layer.bias[:updated_next_layer.out_features])

        resized_model_with_rows.all_layers[next_layer_idx] = updated_next_layer

    elif isinstance(next_layer, nn.Conv2d):
        updated_next_layer = nn.Conv2d(
            in_channels=new_size,
            out_channels=next_layer.out_channels,
            kernel_size=next_layer.kernel_size,
            stride=next_layer.stride,
            padding=next_layer.padding,
            dilation=next_layer.dilation,
            groups=next_layer.groups,
            bias=next_layer.bias is not None
        )

        # Copy weights and biases
        with torch.no_grad():
            updated_next_layer.weight.copy_(next_layer.weight[:, :new_size, :, :]
                                            )
            if next_layer.bias is not None:
                updated_next_layer.bias.copy_(next_layer.bias[:updated_next_layer.out_channels])

        resized_model_with_rows.all_layers[next_layer_idx] = updated_next_layer

    elif isinstance(next_layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        next_bn = type(next_layer)(num_features=new_size)
        with torch.no_grad():
            next_bn.weight.copy_(next_layer.weight[:new_size])
            next_bn.bias.copy_(next_layer.bias[:new_size])
            next_bn.running_mean.copy_(next_layer.running_mean[:new_size])
            next_bn.running_var.copy_(next_layer.running_var[:new_size])
            next_bn.num_batches_tracked.copy_(next_layer.num_batches_tracked)

        resized_model_with_rows.all_layers[next_layer_idx] = next_bn

    elif isinstance(next_layer, nn.Flatten):
        # If Flatten exists, get the subsequent Linear layer from the next row and update its in_features
        next_linear_idx = resized_model_with_rows.row_to_main_layer[row_to_resize_idx + 1]
        next_linear = resized_model_with_rows.all_layers[next_linear_idx]

        # Compute new in_features after Conv2D resizing
        _, _, h, w = layer_to_resize.weight.shape  # Get kernel size
        new_input_size = new_size * h * w

        updated_linear = nn.Linear(
            in_features=new_input_size,
            out_features=next_linear.out_features,
            bias=next_linear.bias is not None
        )

        with torch.no_grad():
            updated_linear.weight.copy_(next_linear.weight[:updated_linear.out_features, :updated_linear.in_features])
            if next_linear.bias is not None:
                updated_linear.bias.copy_(next_linear.bias[:updated_linear.out_features])

        resized_model_with_rows.all_layers[next_linear_idx] = updated_linear

    # Rebuild rows
    resized_model_with_rows.all_rows, resized_model_with_rows.row_to_main_layer = \
        resized_model_with_rows.split_and_map_layers_to_rows()

    return resized_model_with_rows


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
    pruned_model_with_rows = copy.deepcopy(model_with_rows)
    layer_to_prune_idx = pruned_model_with_rows.row_to_main_layer[row_to_prune_idx]
    layer_to_prune = pruned_model_with_rows.all_layers[layer_to_prune_idx]

    # Apply structured pruning to the weight of the layer (dim=0 corresponds to out_channels / out_features)
    prune.ln_structured(layer_to_prune, name='weight', amount=(1 - compression_rate), n=1, dim=0)

    # Identify the surviving indices (un-pruned)
    if isinstance(layer_to_prune, nn.Conv2d):
        mask = layer_to_prune.weight_mask.sum(dim=(1, 2, 3)) > 0  # Identify un-pruned filters
        new_out_size = mask.sum().item()
        # TODO: Explain
        prune.remove(layer_to_prune, 'weight')

        # Create a new layer with updated dimensions
        pruned_layer = nn.Conv2d(
            in_channels=layer_to_prune.in_channels,
            out_channels=new_out_size,
            kernel_size=layer_to_prune.kernel_size,
            stride=layer_to_prune.stride,
            padding=layer_to_prune.padding,
            dilation=layer_to_prune.dilation,
            groups=layer_to_prune.groups,
            bias=layer_to_prune.bias is not None
        )

        # Copy un-pruned weights
        with torch.no_grad():
            pruned_layer.weight.copy_(layer_to_prune.weight[mask, :, :, :])
            if layer_to_prune.bias is not None:
                pruned_layer.bias.copy_(layer_to_prune.bias[mask])
    else:  # elif isinstance(layer_to_prune, nn.Linear)
        mask = layer_to_prune.weight_mask.sum(dim=1) > 0  # Identify un-pruned neurons
        new_out_size = mask.sum().item()
        # TODO: Explain
        prune.remove(layer_to_prune, 'weight')

        pruned_layer = nn.Linear(
            in_features=layer_to_prune.in_features,
            out_features=new_out_size,
            bias=layer_to_prune.bias is not None
        )

        # Copy un-pruned weights
        with torch.no_grad():
            pruned_layer.weight.copy_(layer_to_prune.weight[mask, :])
            if layer_to_prune.bias is not None:
                pruned_layer.bias.copy_(layer_to_prune.bias[mask])

    # Assign pruned layer in model's all_layers
    pruned_model_with_rows.all_layers[layer_to_prune_idx] = pruned_layer

    # Post-pruning, the subsequent layers requires an adjustment according to the pruned out_features / out_channels,
    # up until the next weight-consuming layer (inclusive) which begins the next row
    next_weight_consuming_layer_idx = pruned_model_with_rows.row_to_main_layer[row_to_prune_idx + 1]

    for idx in range(layer_to_prune_idx + 1, next_weight_consuming_layer_idx):
        layer = pruned_model_with_rows.all_layers[idx]

        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            adjusted_bn = type(layer)(num_features=new_out_size)
            with torch.no_grad():
                adjusted_bn.weight.copy_(layer.weight[:new_out_size])
                adjusted_bn.bias.copy_(layer.bias[:new_out_size])
                adjusted_bn.running_mean.copy_(layer.running_mean[:new_out_size])
                adjusted_bn.running_var.copy_(layer.running_var[:new_out_size])
                adjusted_bn.num_batches_tracked.copy_(layer.num_batches_tracked)
            pruned_model_with_rows.all_layers[idx] = adjusted_bn
        # else: leave things like ReLU, Dropout, etc. as they are.

    # Now adjust the first layer of the next row (next_weight_consuming_layer_idx)
    next_layer = pruned_model_with_rows.all_layers[next_weight_consuming_layer_idx]

    if isinstance(pruned_layer, nn.Conv2d):
        if isinstance(next_layer, nn.Conv2d):
            adjusted_next_layer = nn.Conv2d(
                in_channels=new_out_size,
                out_channels=next_layer.out_channels,
                kernel_size=next_layer.kernel_size,
                stride=next_layer.stride,
                padding=next_layer.padding,
                dilation=next_layer.dilation,
                groups=next_layer.groups,
                bias=next_layer.bias is not None
            )
            with torch.no_grad():
                adjusted_next_layer.weight.copy_(next_layer.weight[:, :new_out_size, :, :])
                if next_layer.bias is not None:
                    adjusted_next_layer.bias.copy_(next_layer.bias)

        elif isinstance(next_layer, nn.Linear):
            # Handle Conv2d → Flatten → Linear case
            _, _, h, w = pruned_layer.weight.shape
            in_features = new_out_size * h * w

            adjusted_next_layer = nn.Linear(
                in_features=in_features,
                out_features=next_layer.out_features,
                bias=next_layer.bias is not None
            )
            with torch.no_grad():
                adjusted_next_layer.weight.copy_(next_layer.weight[:, :in_features])
                if next_layer.bias is not None:
                    adjusted_next_layer.bias.copy_(next_layer.bias)

    elif isinstance(pruned_layer, nn.Linear):
        if isinstance(next_layer, nn.Linear):
            adjusted_next_layer = nn.Linear(
                in_features=new_out_size,
                out_features=next_layer.out_features,
                bias=next_layer.bias is not None
            )
            with torch.no_grad():
                adjusted_next_layer.weight.copy_(next_layer.weight[:, :new_out_size])
                if next_layer.bias is not None:
                    adjusted_next_layer.bias.copy_(next_layer.bias)

    pruned_model_with_rows.all_layers[next_weight_consuming_layer_idx] = adjusted_next_layer

    # Finally rebuild rows
    pruned_model_with_rows.all_rows, pruned_model_with_rows.row_to_main_layer = pruned_model_with_rows.split_and_map_layers_to_rows()

    pruned_model_with_rows.rewrap_model(StaticConf.get_instance().conf_values.device)
    return pruned_model_with_rows

