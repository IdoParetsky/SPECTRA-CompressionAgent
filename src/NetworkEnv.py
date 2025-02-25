import copy
import itertools
from typing import Dict, Type, List
import numpy as np
import os
import time
import torch
from torch import nn
from torch.nn.utils import prune

# TODO: Incorporate FMs extraction methods
from src.BERTInputModeler import BERTInputModeler, extract_topology_features, extract_activation_statistics, extract_weight_statistics
from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtraction.src.ModelClasses.LoadedModel import LoadedModel, MissionTypes
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.ModelHandlers.RegressionHandler import RegressionHandler
import src.utils as utils


class NetworkEnv:
    """
    Reinforcement learning environment for pruning CNNs.

    Features:
    - Preloaded models & datasets from `self.conf.input_dict`
    - Feature extraction for BERT-based state encoding
    - Iterative pruning & evaluation
    """

    def __init__(self, networks=None):
        self.conf = StaticConf.get_instance().conf_values
        self.layer_index = None  # This variable will hold the index of the layer after the one to be pruned
        self.actions_history = []
        self.original_acc = None
        self.selected_net_path = None
        self.current_model = None
        self.feature_extractor = None

        self.networks = networks or list(self.conf.input_dict.keys())  # List of model paths
        self.curr_net_index = -1
        np.random.shuffle(self.networks)  # Randomize order

        # Initialize BERT Input Modeler
        self.bert_modeler = BERTInputModeler()

        # A dictionary mapping MissionTypes to corresponding Handler classes
        self.handler_by_mission_type: Dict[MissionTypes, Type[BasicHandler]] = {
            MissionTypes.Regression: RegressionHandler,
            MissionTypes.Classification: ClassificationHandler
        }

    def reset(self, test_net_path=None, test_model=None, test_loaders=None):
        """ Reset environment with a new CNN model & dataset """
        self.layer_index = 1  # The first layer to be a candidate for pruning is self.layer_index - 1 -> index 0
        self.actions_history = []

        # If test model is provided, use it directly (for cross-validation)
        if test_net_path and test_model and test_loaders:
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = test_model, test_loaders
            self.selected_net_path = test_net_path
        else:
            self.curr_net_index = (self.curr_net_index + 1) % len(self.networks)
            self.selected_net_path = self.networks[self.curr_net_index]

            # Load model & dataset from preloaded input_dict
            self.current_model, (self.train_loader, self.val_loader, self.test_loader) = self.conf.input_dict[
                self.selected_net_path]

        utils.print_flush(f"Loading {self.selected_net_path}")

        # Prepare feature extractor with training data
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, self.conf.device)
        fm = self.feature_extractor.encode_to_bert_input(self.layer_index - 1)

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
        utils.print_flush(f"Step {self.layer_index - 1} - Compression Rate: {compression_rate}")

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            model_with_rows = ModelWithRows(self.current_model)
            # Create a pruned or resized model
            modified_model_with_rows = prune_current_model(
                model_with_rows, compression_rate, self.layer_index - 1) if self.conf.prune \
                else create_new_model_with_new_weights(model_with_rows, compression_rate, self.layer_index - 1)

            # Rebuild the model with the modified structure
            new_model_layers = [layer for row in modified_model_with_rows.all_rows for layer in row]
            self.current_model = nn.Sequential(*new_model_layers)
            
            # Prepare model handler
            learning_handler_new_model = self.create_learning_handler(self.current_model)

            # Freeze/unfreeze layers based on config
            if self.conf.is_train_compressed_layer_only:
                parameters_to_freeze_ids = build_parameters_to_freeze(modified_model_with_rows, self.layer_index - 1)
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
        self.layer_index += 1
        learning_handler_new_model.unfreeze_all_layers()
        self.current_model = learning_handler_new_model.model

        # Extract features for BERT
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, self.conf.device)
        fm = self.feature_extractor.encode_to_bert_input(self.layer_index - 1)

        # Check termination condition
        num_layers = len(self.feature_extractor.model_with_rows.all_rows) - 1
        self.actions_history.append(compression_rate)
        # As self.layer_index - 1 is the current appraised layer, the index should not drop below 1
        self.layer_index = max(1, self.layer_index % (num_layers + 1))
        done = len(self.actions_history) >= num_layers * self.conf.passes

        # Save the final pruned model to a checkpoint file,
        # if requested by the user via self.conf.save_pruned_checkpoints = True
        if done and self.conf.save_pruned_checkpoints:
            self.save_pruned_checkpoint()

        return fm, reward, done

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
        """
        return self.handler_by_mission_type[self.current_model.mission_type](
            new_model,
            self.current_model.loss,
            self.current_model.optimizer
        )


def build_parameters_to_freeze(model_with_rows, layer_to_modify_idx):
    """
    Builds a list of parameter IDs for layers to freeze.

    Args:
        model_with_rows (ModelWithRows):   The model wrapped with rows of layers.
        layer_to_modify_idx (int):         Index of the layer to prune / resize

    Returns:
        List[int]: A list of parameter IDs to freeze.
    """
    # Extract the layers to freeze (the pruned layer and the subsequent layer)
    layers_to_freeze = sum(model_with_rows.all_rows[layer_to_modify_idx:layer_to_modify_idx + 2], [])

    # Flatten and collect parameter IDs
    parameters_to_freeze_ids = [id(param) for layer in layers_to_freeze for param in layer.parameters()]

    return parameters_to_freeze_ids


def create_new_model_with_new_weights(model_with_rows, compression_rate, layer_to_resize_idx):
    """
    Replace a layer with a reduced version, adjusting the subsequent layer accordingly.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be resized
        compression_rate (float):         The desired compression rate for resizing
        layer_to_resize_idx (int):        Index of the layer to resize

    Returns:
         resized_model_with_rows (ModelWithRows): The resized model.
    """
    resized_model_with_rows = copy.deepcopy(model_with_rows)

    layer_to_change = utils.get_layer_by_type(resized_model_with_rows.all_rows[layer_to_resize_idx],
                                              (nn.Linear, nn.Conv2d))
    next_layer = utils.get_layer_by_type(resized_model_with_rows.all_rows[layer_to_resize_idx + 1],
                                         (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Flatten))

    # Compute new size based on compression rate
    new_size = int(np.ceil(compression_rate * (
        layer_to_change.out_features if isinstance(layer_to_change, nn.Linear) else layer_to_change.out_channels)))

    # Replace the layer with its resized counterpart
    if isinstance(layer_to_change, nn.Linear):
        resized_layer = nn.Linear(layer_to_change.in_features, new_size, bias=layer_to_change.bias is not None)

        with torch.no_grad():
            resized_layer.weight[:, :new_size] = next_layer.weight[:, :new_size]

    elif isinstance(layer_to_change, nn.Conv2d):
        resized_layer = nn.Conv2d(layer_to_change.in_channels, new_size, layer_to_change.kernel_size,
                              layer_to_change.stride, layer_to_change.padding, bias=layer_to_change.bias is not None)

        # Copy weights and biases
        with torch.no_grad():
            resized_layer.weight[:new_size, :, :, :] = layer_to_change.weight[:new_size, :, :, :]
            if layer_to_change.bias is not None:
                resized_layer.bias[:new_size] = layer_to_change.bias[:new_size]

    resized_model_with_rows.all_rows[layer_to_resize_idx] = resized_layer  # Update layer in the model

    # Adjust the next layer if necessary
    if isinstance(next_layer, nn.Linear):
        updated_next_layer = nn.Linear(new_size, next_layer.out_features, bias=next_layer.bias is not None)
        with torch.no_grad():
            updated_next_layer.weight[:, :new_size] = next_layer.weight[:, :new_size]
        resized_model_with_rows.all_rows[layer_to_resize_idx + 1] = updated_next_layer

    elif isinstance(next_layer, nn.Conv2d):
        updated_next_layer = nn.Conv2d(new_size, next_layer.out_channels, next_layer.kernel_size,
                                       next_layer.stride, next_layer.padding, bias=next_layer.bias is not None)

        # Copy weights and biases
        with torch.no_grad():
            updated_next_layer.weight[:, :new_size, :, :] = next_layer.weight[:, :new_size, :, :]
            if layer_to_change.bias is not None:
                updated_next_layer.bias[:] = next_layer.bias[:]

        resized_model_with_rows.all_rows[layer_to_resize_idx + 1] = updated_next_layer

    elif isinstance(next_layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        updated_next_layer = type(next_layer)(num_features=new_size)
        resized_model_with_rows.all_rows[layer_to_resize_idx + 1] = updated_next_layer

    elif isinstance(next_layer, nn.Flatten):
        # If Flatten exists, find the subsequent Linear layer and update its in_features
        found_linear = False
        for idx in range(layer_to_resize_idx + 2, len(resized_model_with_rows.all_rows)):
            potential_next_layer = utils.get_layer_by_type(resized_model_with_rows.all_rows[idx], (nn.Linear,))
            if potential_next_layer is not None:
                next_linear_layer = potential_next_layer
                found_linear = True
                break  # Stop searching after finding the first Linear layer

        if not found_linear:
            raise RuntimeError("Expected a Linear layer after Flatten, but none was found.")

        next_linear_layer = utils.get_layer_by_type(resized_model_with_rows.all_rows[layer_to_resize_idx + 2],
                                                    (nn.Linear,))

        # Compute new in_features after Conv2D resizing
        _, _, h, w = layer_to_change.weight.shape  # Get kernel size
        new_input_size = new_size * h * w

        updated_next_linear_layer = nn.Linear(new_input_size, next_linear_layer.out_features,
                                              bias=next_linear_layer.bias is not None)

        with torch.no_grad():
            updated_next_linear_layer.weight[:, :new_input_size] = next_linear_layer.weight[:, :new_input_size]
            if next_linear_layer.bias is not None:
                updated_next_linear_layer.bias[:] = next_linear_layer.bias[:]

        resized_model_with_rows.all_rows[layer_to_resize_idx + 2] = updated_next_linear_layer

    resized_model_with_rows.all_layers = list(itertools.chain.from_iterable(resized_model_with_rows.all_rows))

    return resized_model_with_rows


def prune_current_model(model_with_rows, compression_rate, layer_to_prune_idx):
    """
    Create a new model by pruning filters in the target layer.

    Args:
        model_with_rows (ModelWithRows):  The model whose layer is to be pruned
        compression_rate (float):         The desired compression rate for pruning
        layer_to_prune_idx (int):         Index of the layer to prune

    Returns:
        pruned_model_with_rows (ModelWithRows): The pruned model
        
    """
    pruned_model_with_rows = copy.deepcopy(model_with_rows)
    layer_to_change = utils.get_layer_by_type(pruned_model_with_rows.all_rows[layer_to_prune_idx],
                                              (nn.Linear, nn.Conv2d))

    # Apply structured pruning to the weight of the layer (dim=0 corresponds to out_channels)
    prune.ln_structured(layer_to_change, name='weight', amount=(1 - compression_rate), n=1, dim=0)

    # Identify the surviving indices (un-pruned)
    if isinstance(layer_to_change, nn.Conv2d):
        mask = layer_to_change.weight_mask.sum(dim=(1, 2, 3)) > 0  # Identify un-pruned filters
        new_out_channels = mask.sum().item()
        
        # Create a new layer with updated dimensions
        pruned_layer = nn.Conv2d(
            in_channels=layer_to_change.in_channels,
            out_channels=new_out_channels,
            kernel_size=layer_to_change.kernel_size,
            stride=layer_to_change.stride,
            padding=layer_to_change.padding,
            dilation=layer_to_change.dilation,
            groups=layer_to_change.groups,
            bias=layer_to_change.bias is not None
        )

        # Copy un-pruned weights
        with torch.no_grad():
            pruned_layer.weight.copy_(layer_to_change.weight[mask, :, :, :])
            if layer_to_change.bias is not None:
                pruned_layer.bias.copy_(layer_to_change.bias[mask])

    elif isinstance(layer_to_change, nn.Linear):
        mask = layer_to_change.weight_mask.sum(dim=1) > 0  # Identify un-pruned neurons
        new_out_features = mask.sum().item()

        pruned_layer = nn.Linear(
            in_features=layer_to_change.in_features,
            out_features=new_out_features,
            bias=layer_to_change.bias is not None
        )

        # Copy un-pruned weights
        with torch.no_grad():
            pruned_layer.weight[:] = layer_to_change.weight[mask, :]
            if layer_to_change.bias is not None:
                pruned_layer.bias[:] = layer_to_change.bias[mask]

    # Replace pruned layer in the model
    pruned_model_with_rows.all_rows[layer_to_prune_idx] = pruned_layer

    # Post-pruning, the subsequent layer requires an adjustment according to the pruned out_features / out_channels
    next_layer = utils.get_layer_by_type(pruned_model_with_rows.all_rows[layer_to_prune_idx + 1],
                                         (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.Flatten))

    if isinstance(layer_to_change, nn.Conv2d):
        if isinstance(next_layer, nn.Conv2d):
            # Create new Conv2D layer with adjusted in_channels but same hyperparameters
            next_conv = nn.Conv2d(in_channels=layer_to_change.out_channels,  # Adjust in_channels
                                  out_channels=next_layer.out_channels,
                                  kernel_size=next_layer.kernel_size,
                                  stride=next_layer.stride,
                                  padding=next_layer.padding,
                                  dilation=next_layer.dilation,
                                  groups=next_layer.groups,
                                  bias=next_layer.bias is not None)

            # Copy weights & biases to preserve learned features
            with torch.no_grad():
                next_conv.weight[:, :pruned_layer.out_channels, :, :] = next_layer.weight[
                                                                     :, :pruned_layer.out_channels, :, :]
                if next_layer.bias is not None:
                    next_conv.bias[:] = next_layer.bias[:]

            pruned_model_with_rows.all_rows[layer_to_prune_idx + 1] = next_conv

        elif isinstance(next_layer, nn.BatchNorm2d):
            # Adjust BatchNorm2D layer to match the pruned Conv2D output channels
            next_layer.num_features = layer_to_change.out_channels
            pruned_model_with_rows.all_rows[layer_to_prune_idx + 1] = next_layer

        elif isinstance(next_layer, (nn.Linear, nn.Flatten)):
            # Handle both:
            # 1. Conv2D → Linear (No Flatten in Between)
            # 2. Conv2D → Flatten → Linear

            # If Flatten is present, get the layer after it (Linear)
            if isinstance(next_layer, nn.Flatten):
                found_linear = False
                for idx in range(layer_to_prune_idx + 2, len(pruned_model_with_rows.all_rows)):
                    potential_next_layer = utils.get_layer_by_type(pruned_model_with_rows.all_rows[idx], (nn.Linear,))
                    if potential_next_layer is not None:
                        next_layer = potential_next_layer
                        found_linear = True
                        break  # Stop searching after finding the first Linear layer

                if not found_linear:
                    raise RuntimeError("Expected a Linear layer after Flatten, but none was found.")

            # Compute new input size after pruning
            _, _, h, w = layer_to_change.weight.shape
            new_input_size = layer_to_change.out_channels * h * w

            # Create adjusted Linear layer
            next_fc = nn.Linear(
                in_features=new_input_size,
                out_features=next_layer.out_features,
                bias=next_layer.bias is not None
            )

            # Copy existing weights if shape allows
            with torch.no_grad():
                next_fc.weight[:, :new_input_size] = next_layer.weight[:, :new_input_size]
                if next_layer.bias is not None:
                    next_fc.bias[:] = next_layer.bias[:]

            # Update the model with the adjusted Linear layer
            pruned_model_with_rows.all_rows[
                layer_to_prune_idx + (2 if isinstance(next_layer, nn.Flatten) else 1)] = next_fc

    elif isinstance(layer_to_change, nn.Linear):
        if isinstance(next_layer, nn.Linear):
            # Reduce `in_features` of next Linear layer
            next_fc = nn.Linear(
                in_features=pruned_layer.out_features,  # Adjusted after pruning
                out_features=next_layer.out_features,
                bias=next_layer.bias is not None
            )

            # Preserve Weights & Biases
            with torch.no_grad():
                next_fc.weight[:, :layer_to_change.out_features] = next_layer.weight[
                                                                   :, :layer_to_change.out_features]
                if next_layer.bias is not None:
                    next_fc.bias[:] = next_layer.bias[:]

            pruned_model_with_rows.all_rows[layer_to_prune_idx + 1] = next_fc

        elif isinstance(next_layer, nn.BatchNorm1d):
            # Adjust BatchNorm1d to match pruned Linear layer's out_features
            next_layer.num_features = layer_to_change.out_features

            pruned_model_with_rows.all_rows[layer_to_prune_idx + 1] = next_layer

        elif isinstance(next_layer, nn.Conv2d):
            # Uncommon: Linear → Conv2D
            # Ensure the reshaped input still makes sense
            print("Warning: Pruning a Linear layer before a Conv2D may break shape compatibility.")

    pruned_model_with_rows.all_layers = list(itertools.chain.from_iterable(pruned_model_with_rows.all_rows))

    return pruned_model_with_rows
