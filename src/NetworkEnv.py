import copy
from typing import Dict, Type, List
import numpy as np
from torch import nn
from torch.nn.utils import prune

from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtraction.src.ModelClasses.LoadedModel import LoadedModel, MissionTypes
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from NetworkFeatureExtraction.src.main import load_model_and_data
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.ModelHandlers.RegressionHandler import RegressionHandler
from src.utils import load_cnn_dataset, get_layer_by_type, compute_reward, is_to_change_bn_layer, print_flush


class NetworkEnv:
    loaded_model: LoadedModel
    current_model: nn.Module
    actions_history: List[float]

    def __init__(self, networks_path, dataset_name_or_path, passes, train_split, val_split):
        self.layer_index = None
        self.actions_history = []
        self.original_acc = None
        self.current_model = None
        self.feature_extractor = None

        self.all_networks = []
        self.dataset_name_or_path = dataset_name_or_path
        self.passes = passes
        self.train_split = train_split
        self.val_split = val_split

        # Initialize dataset loaders
        self.train_loader, self.val_loader, self.test_loader = load_cnn_dataset(
            self.dataset_name_or_path, train_split=self.train_split, val_split=self.val_split)

        # Initialize network paths
        self.networks_path = networks_path
        for group in networks_path:
            x_path = group[0]
            nets = group[1]

            for n in nets:
                self.all_networks.append((x_path, n))

        self.curr_net_index = -1
        self.net_order = list(range(len(self.all_networks)))
        np.random.shuffle(self.net_order)

        # A dictionary mapping MissionTypes to corresponding Handler classes
        self.handler_by_mission_type: Dict[MissionTypes, Type[BasicHandler]] = {
            MissionTypes.Regression: RegressionHandler,
            MissionTypes.Classification: ClassificationHandler,}

    def reset(self):
        """ Reset environment with a random CNN and dataset """
        self.layer_index = 1
        self.actions_history = []
        self.curr_net_index = (self.curr_net_index + 1) % len(self.net_order)
        curr_group_index = self.net_order[self.curr_net_index]
        _, selected_net_path = self.all_networks[curr_group_index]

        print_flush(selected_net_path)
        device = StaticConf.get_instance().conf_values.device

        # Load model and CNN dataset
        self.loaded_model, _, _ = load_model_and_data(selected_net_path, None, None, device)
        self.current_model = self.loaded_model.model

        # Prepare feature extractor with train data
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, device)
        fm = self.feature_extractor.extract_features(self.layer_index - 1)

        # Evaluate original model accuracy
        learning_handler_original_model = self.create_learning_handler(self.current_model)
        self.original_acc = learning_handler_original_model.evaluate_model(self.val_loader)

        return fm

    # TODO: Edit step() onwards, then a2c_agent_reinforce_runner, then additional scripts and then go back to FE scripts
    #  and implement BERT input pipeline
    def step(self, compression_rate, is_to_train=True):
        """
        Compress the network by pruning filters (for CNNs) or reducing layer size,
        then advance to the next state.

        Args:
            compression_rate (float): The chosen compression_rate.
            is_to_train (bool): Whether to train the model after compression.

        Returns:
            Tuple: A tuple containing the next state (feature maps), reward, and done flag.
        """
        print_flush(f'step {self.layer_index}')

        # If no compression, proceed without modification
        if compression_rate == 1:
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            # Create new compressed model
            if StaticConf.get_instance().conf_values.prune:
                new_model = self.create_new_model_pruned(compression_rate)
            else:
                new_model = self.create_new_model_with_new_weights(compression_rate)

            # Wrap the new model
            new_model_with_rows = ModelWithRows(new_model)
            learning_handler_new_model = self.create_learning_handler(new_model)

            # Freeze layers or unfreeze all layers based on configuration
            if StaticConf.get_instance().conf_values.is_learn_new_layers_only:
                parameters_to_freeze_ids = self.build_parameters_to_freeze(new_model_with_rows)
                learning_handler_new_model.freeze_layers(parameters_to_freeze_ids)
            else:
                learning_handler_new_model.unfreeze_all_layers()

            # Train the new model if specified
            if is_to_train:
                learning_handler_new_model.train_model(self.train_loader)

        # Evaluate the new model
        learning_handler_new_model.model.eval()
        new_acc = learning_handler_new_model.evaluate_model(self.val_loader)

        # Compute the reward based on the new accuracy and compression rate
        reward = compute_reward(new_acc, self.original_acc, compression_rate)

        # Update environment state
        self.layer_index += 1
        learning_handler_new_model.unfreeze_all_layers()
        self.current_model = learning_handler_new_model.model

        # Extract features for the next state
        device = StaticConf.get_instance().conf_values.device
        self.feature_extractor = FeatureExtractor(self.current_model, self.train_loader, device)
        fm = self.feature_extractor.extract_features(self.layer_index - 1)

        # Determine if episode is done
        number_of_layers = len(self.feature_extractor.model_with_rows.all_rows) - 1
        self.actions_history.append(compression_rate)
        self.layer_index = max(1, self.layer_index % (number_of_layers + 1))
        done = len(self.actions_history) >= number_of_layers * self.passes

        return fm, reward, done

    def calc_num_parameters(self, model, is_pruned=False):
        """
        Calculate the total number of parameters in the model, considering pruned parameters if specified.

        Args:
            model (nn.Module): The model to analyze.
            is_pruned (bool): Whether to account for pruned parameters.

        Returns:
            int: The total number of remaining parameters in the model.
        """
        if not is_pruned:
            # Count all parameters in the model
            return sum(p.numel() for p in model.parameters())

        # Count pruned parameters (weights with a mask value of zero)
        pruned_params = sum(
            (module.weight_mask == 0).sum().item()
            for module in model.modules()
            if hasattr(module, 'weight_mask')
        )

        # Calculate original parameter count
        orig_params = sum(p.numel() for p in model.parameters())

        # Subtract pruned parameters
        return orig_params - pruned_params

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
        return self.handler_by_mission_type[self.loaded_model.mission_type](new_model,
                                                                            self.loaded_model.loss,
                                                                            self.loaded_model.optimizer)

    def create_new_model_with_new_weights(self, compression_rate):
        """
        Create a new model with reduced layer size, replacing weights as needed.

        Args:
            compression_rate (float): The desired compression rate for the current layer.

        Returns:
            nn.Sequential: The new model with updated layer sizes.
        """
        model_with_rows = ModelWithRows(self.current_model)

        # Support for both Fully-Connected and Convolutional layers
        prev_layer_to_change = get_layer_by_type(model_with_rows.all_rows[self.layer_index - 1], (nn.Linear, nn.Conv2d))
        layer_to_change = get_layer_by_type(model_with_rows.all_rows[self.layer_index], (nn.Linear, nn.Conv2d))

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
            elif is_to_change_bn_layer(l, last_layer):
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

    def create_new_model_pruned(self, compression_rate):
        """
        Create a new model by pruning filters or weights in the target layer.

        Args:
            compression_rate (float): The desired compression rate for pruning.

        Returns:
            nn.Module: The pruned model.
        """
        model_with_rows = ModelWithRows(self.current_model)
        layer_to_change = get_layer_by_type(model_with_rows.all_rows[self.layer_index - 1], (nn.Linear, nn.Conv2d))

        # Apply structured pruning to the weight of the layer
        prune.ln_structured(layer_to_change, name='weight', amount=(1 - compression_rate), n=1, dim=0)

        return self.current_model
