import copy
import json
from typing import Dict, Type, List
import numpy as np
import torch
import importlib
from torch import nn, no_grad
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from src.BERTInputModeler import BERTInputModeler, extract_topology_features, extract_activation_statistics, extract_weight_statistics
from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtraction.src.ModelClasses.LoadedModel import LoadedModel, MissionTypes
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from NetworkFeatureExtraction.src.main import load_model_and_data
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

    def __init__(self):
        self.conf = StaticConf.get_instance().conf_values
        self.layer_index = None
        self.actions_history = []
        self.original_acc = None
        self.current_model = None
        self.feature_extractor = None

        self.all_networks = list(self.conf.input_dict.keys())  # List of model paths
        self.curr_net_index = -1
        np.random.shuffle(self.all_networks)  # Randomize order

        # Initialize BERT Input Modeler
        self.bert_modeler = BERTInputModeler()

        # A dictionary mapping MissionTypes to corresponding Handler classes
        self.handler_by_mission_type: Dict[MissionTypes, Type[BasicHandler]] = {
            MissionTypes.Regression: RegressionHandler,
            MissionTypes.Classification: ClassificationHandler
        }

    def reset(self):
        """ Reset environment with a new CNN model & dataset """
        self.layer_index = 1
        self.actions_history = []
        self.curr_net_index = (self.curr_net_index + 1) % len(self.all_networks)
        selected_net_path = self.all_networks[self.curr_net_index]

        utils.print_flush(f"Loading {selected_net_path}")

        # Load model & dataset from preloaded input_dict
        self.current_model, (self.train_loader, self.val_loader, self.test_loader) = self.conf.input_dict[selected_net_path]

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
        utils.print_flush(f"Step {self.layer_index} - Compression Rate: {compression_rate}")

        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Create a pruned or resized model
            new_model = self.prune_current_model(compression_rate) if self.conf.prune \
                else self.create_new_model_with_new_weights(compression_rate)

            # Prepare model handler
            new_model_with_rows = ModelWithRows(new_model)
            learning_handler_new_model = self.create_learning_handler(new_model)

            # Freeze/unfreeze layers based on config
            if self.conf.is_train_compressed_layer_only:
                parameters_to_freeze_ids = self.build_parameters_to_freeze(new_model_with_rows)
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
        self.layer_index = max(1, self.layer_index % (num_layers + 1))
        done = len(self.actions_history) >= num_layers * self.conf.passes

        return fm, reward, done

    def build_parameters_to_freeze(self, model_with_rows):
        """
        Builds a list of parameter IDs for layers to freeze.

        Args:
            model_with_rows (ModelWithRows): The model wrapped with rows of layers.

        Returns:
            List[int]: A list of parameter IDs to freeze.
        """
        # Extract the layers to freeze (current and previous rows)
        layers_to_freeze = sum(model_with_rows.all_rows[self.layer_index - 1:self.layer_index + 1], [])

        # Flatten and collect parameter IDs
        parameters_to_freeze_ids = [id(param) for layer in layers_to_freeze for param in layer.parameters()]

        return parameters_to_freeze_ids

    def create_learning_handler(self, new_model) -> BasicHandler:
        """ Create appropriate learning handler based on mission type. """
        return self.handler_by_mission_type[
            self.conf.input_dict[self.all_networks[self.curr_net_index]][0].mission_type](
            new_model,
            self.conf.input_dict[self.all_networks[self.curr_net_index]][0].loss,
            self.conf.input_dict[self.all_networks[self.curr_net_index]][0].optimizer
        )

    def create_new_model_with_new_weights(self, compression_rate):
        """
        Replace layer with a reduced version.

        Args:
            compression_rate (float): Compression factor.

        Returns:
            nn.Sequential: Modified model.
        """
        model_with_rows = ModelWithRows(self.current_model)
        layer_to_change = utils.get_layer_by_type(model_with_rows.all_rows[self.layer_index], (nn.Linear, nn.Conv2d))

        new_size = int(np.ceil(compression_rate * (
            layer_to_change.in_features if isinstance(layer_to_change, nn.Linear) else layer_to_change.out_channels)))

        new_model_layers = [
            nn.Linear(layer.in_features, new_size) if layer is layer_to_change and isinstance(layer, nn.Linear) else
            nn.Conv2d(layer.in_channels, new_size, layer.kernel_size, layer.stride,
                      layer.padding) if layer is layer_to_change and isinstance(layer, nn.Conv2d) else
            copy.deepcopy(layer)
            for layer in model_with_rows.all_layers
        ]

        return nn.Sequential(*new_model_layers)

        model_with_rows = ModelWithRows(self.current_model)

        # Support for both Fully-Connected and Convolutional layers
        prev_layer_to_change = utils.get_layer_by_type(model_with_rows.all_rows[self.layer_index - 1],
                                                       (nn.Linear, nn.Conv2d))
        layer_to_change = utils.get_layer_by_type(model_with_rows.all_rows[self.layer_index],
                                                  (nn.Linear, nn.Conv2d))

        new_model_layers = []
        new_size = int(np.ceil(compression_rate * (
            layer_to_change.in_features if isinstance(layer_to_change, nn.Linear) else layer_to_change.out_channels)))

        last_layer = None  # Tracks the previous layer for BatchNorm compatibility

        for l in model_with_rows.all_layers:
            if l is prev_layer_to_change:
                if isinstance(l, nn.Linear):
                    new_model_layers.append(nn.Linear(l.in_features, new_size))
                elif isinstance(l, nn.Conv2d):
                    new_model_layers.append(nn.Conv2d(l.in_channels, new_size, l.kernel_size, l.stride, l.padding))
            elif l is layer_to_change:
                if isinstance(l, nn.Linear):
                    new_model_layers.append(nn.Linear(new_size, l.out_features))
                elif isinstance(l, nn.Conv2d):
                    new_model_layers.append(nn.Conv2d(new_size, l.out_channels, l.kernel_size, l.stride, l.padding))
            elif utils.is_to_change_bn_layer(l, last_layer):
                if isinstance(last_layer, nn.Linear):
                    new_model_layers.append(nn.BatchNorm1d(last_layer.out_features))
                elif isinstance(last_layer, nn.Conv2d):
                    new_model_layers.append(nn.BatchNorm2d(last_layer.out_channels))
            else:
                new_model_layers.append(copy.deepcopy(l))

            # Update last_layer to track the layer type for BatchNorm compatibility
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                last_layer = new_model_layers[-1]

        return nn.Sequential(*new_model_layers)

    def prune_current_model(self, compression_rate):
        """
        Create a new model by pruning filters or weights in the target layer.

        Args:
            compression_rate (float): The desired compression rate for pruning.

        Returns:
            nn.Module: The pruned model.
        """
        model_with_rows = ModelWithRows(self.current_model)
        layer_to_change = utils.get_layer_by_type(model_with_rows.all_rows[self.layer_index - 1],
                                                  (nn.Linear, nn.Conv2d))

        # Apply structured pruning to the weight of the layer (dim=0 corresponds to out_channels)
        prune.ln_structured(layer_to_change, name='weight', amount=(1 - compression_rate), n=1, dim=0)

        if isinstance(layer_to_change, nn.Conv2d):
            # Adjust subsequent layer to match the new number of filters
            next_layer = utils.get_layer_by_type(model_with_rows.all_rows[self.layer_index],
                                                 (nn.Conv2d, nn.Linear))

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
                with no_grad():
                    next_conv.weight[:next_layer.out_channels, :layer_to_change.out_channels] = next_layer.weight
                    if next_layer.bias is not None:
                        next_conv.bias[:next_layer.out_channels] = next_layer.bias

                # Replace the layer in the model
                model_with_rows.all_rows[self.layer_index] = next_conv

            elif isinstance(next_layer, nn.BatchNorm2d):
                # Adjust BatchNorm layer to match the pruned Conv2D output channels
                next_layer.num_features = layer_to_change.out_channels

        return self.current_model
