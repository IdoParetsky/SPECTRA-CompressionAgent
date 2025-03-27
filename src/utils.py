import os
import json
import sys
import importlib.util
import inspect
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf

# TODO: Change to dynamic assignment
SPECTRA_DATASETS = "C:\\Users\\idopa\\Documents\\BGU\\MSc\\spectra_datasets"

# Possible instantiation functions' parameters
NUM_CLASSES = "num_classes"
LARGE_INPUT = "large_input"
WIDTH = "width"


def print_flush(msg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} -- {msg}", flush=True)


def extract_args_from_cmd():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Script for training and evaluating SPECTRA A2C agent for CNN pruning.")

    parser.add_argument('--input', type=str, required=True,
        help=(
            "Path to a JSON file or a JSON-formatted (dict-like) string of model checkpoints for Agent Evaluation. "
            "The JSON should map network paths to configurations:\n"
            "{\n"
            "  \"network_path_1\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\", optional_kwargs],\n"
            "  \"network_path_2\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\", optional_kwargs]\n"
            "}\n\n"
            "- 'network_path': Path to the network checkpoint (.pt/.pth/.th) file.\n"
            "- 'architecture': The architecture name of the model (e.g., 'resnet18').\n"
            "- 'instantiation_script_path': Path to the script from source repository where the architecture "
            "                               instantiation function resides.\n"
            "- 'dataset_name_or_path': Path to a custom dataset, or name of a standard dataset "
            "                          (supported in utils.load_cnn_dataset(), such as 'cifar-10').\n"
            "- optional_kwargs: keyword dict for custom instantiation parameters (e.g., {num_classes=10, width=56}\n"
            "                   By default, 'num_classes' is dynamically assigned via dataset_loaders's correspondent"
            "                   train_data's 'classes' field,\n'large_input' is True is the dataset has 'imagenet' in "
            "                   its name / path and 'width' is scraped from 'network_path' via regex (e.g., 'resnet18_width56')"
        )
    )

    parser.add_argument('--database', type=str, default=None,
        help=(
            "Path to a JSON file or a JSON-formatted (dict-like) string for Agent Training. "
            "Unused if the agent is pretrained (actor_checkpoint and critic_checkpoint are provided)."
            "A full database syntax example is available in the README file."
            "The JSON should map network paths to configurations:\n"
            "{\n"
            "  \"network_path_1\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\", optional_kwargs],\n"
            "  \"network_path_2\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\", optional_kwargs]\n"
            "}\n\n"
        )
    )

    parser.add_argument('--actor_checkpoint_path', type=str, default=None,
                        help="Path to Actor Checkpoint (pre-trained agent)")

    parser.add_argument('--critic_checkpoint_path', type=str, default=None,
                        help="Path to Critic Checkpoint (pre-trained agent)")

    parser.add_argument(
        '--compression_rates', type=float, nargs='+', default=[1.0, 0.9, 0.8, 0.7, 0.6],
        help=(
            "List of compression rates for pruning layers. The first rate (1.0) means no pruning, "
            "followed by progressively higher compression levels (e.g., 0.9, 0.8). "
            "This will be converted to a dictionary where indices map to compression rates."
        )
    )

    parser.add_argument('--train_compressed_layer_only', type=bool, default=True,
                        help="Whether to train the entire network or only the new layer, post-compression.\n"
                             "Training the entire network after the compression of each layer greatly affects runtime.")

    parser.add_argument('--split', type=bool, default=True,
                        help="Whether to split the networks to train and test sets. Must be True in the first run.")

    parser.add_argument('--allowed_acc_reduction', type=int, default=5,
                        help="The permissible reduction in performance (in percents). Default value=5; 1 is also recommended.")

    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help="Discount Factor, a.k.a Gamma, controls the weight of the agent's future rewards.")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate for the agent's optimizer. Controls the step size in gradient descent.")

    parser.add_argument('--rollout_limit', type=int, default=None,
                        help="Ensures that the agent's rollout trajectory does not exceed a predefined number of steps (optional).")

    parser.add_argument('--passes', type=int, default=1,
                        help="How many per-layer compression iterations over the entire network. Default=1 4 is also recommended.")

    parser.add_argument('--prune', type=bool, default=True,
                        help="Whether to prune layers via torch.nn.utils.prune.ln_structured during compression or resize them manually.")

    parser.add_argument('--num_epochs', type=int, default=100, help="Agent's training epochs amount. Default=100.")

    parser.add_argument('--runtime_limit', type=int, default=60 * 60 * 24 * 7,
                        help="Max runtime. Default is a week in seconds")

    parser.add_argument('--seed', type=int, default=0,
                        help="Seed to be used by pytorch, numpy etc. libraries. Default=0.")

    parser.add_argument('--n_splits', type=int, default=0,
                        help="Inter-model evaluation - train/test splits for n-fold cross-validation. "
                             "Default=0 (no CV), recommended value is 10.")

    parser.add_argument('--train_split', type=float, default=0.7, help="Training data split fraction.")

    parser.add_argument('--val_split', type=float, default=0.2,
                        help="Validation data split fraction. Test data split is 1 - train_split - val_split")

    parser.add_argument('--save_pruned_checkpoints', type=bool, default=False,
                        help="Whether to save a final checkpoint for each pruned network.")

    return parser.parse_args()


def parse_input_argument(input_arg, train_split, val_split):
    """
    Parse the --input or --database arguments as a JSON-formatted string or file.

    Args:
        input_arg (str):        A JSON string or a path to a JSON file for Agent Training / Evaluation.
        train_split (float):    Fraction of the dataset to use for training.
        val_split (float):      Fraction of the dataset to use for validation.

    Returns:
        dict: {network_path: (nn.Module, (train_loader, val_loader, test_loader))}
            A dictionary mapping network paths to:
              - Instantiated model (nn.Module).
              - Dataset loaders (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If input is invalid or instantiation fails.
    """
    # --database could be None if both an actor checkpoint and a critic checkpoint are provided by the user
    # (asserted in A2CAgentReinforce's initialization)
    if input_arg is None:
        return

    # Try parsing as JSON string
    try:
        input_dict = json.loads(input_arg)
    except json.JSONDecodeError:
        pass
    else:
        return instantiate_networks_and_load_datasets(input_dict, train_split, val_split)

    # Try reading JSON file
    try:
        with open(input_arg, 'r') as f:
            input_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise ValueError("Invalid input: Provide a valid JSON string or JSON file path.")

    return instantiate_networks_and_load_datasets(input_dict, train_split, val_split)


def instantiate_networks_and_load_datasets(input_dict, train_split, val_split):
    """
    Instantiates networks from their given architecture checkpoint and instantiation script,
    then loads the corresponding standard / custom datasets.

    Args:
        input_dict (dict):      {network_path: (architecture, instantiation_script, dataset_name_or_path)}
        train_split (float):    Fraction of the dataset to use for training.
        val_split (float):      Fraction of the dataset to use for validation.

    Returns:
        dict: {network_path: (nn.Module, (train_loader, val_loader, test_loader))}

    Raises:
        ValueError: If model instantiation fails.
    """
    instantiated_networks = {}
    for net_path, values in input_dict.items():
        if len(values) == 3:
            arch, script_path, dataset_path = values
            optional_kwargs = {}  # Assign an empty dict if not assigned
        else:
            arch, script_path, dataset_path, optional_kwargs = values

        if not os.path.exists(net_path):
            raise ValueError(f"Network checkpoint not found: {net_path}")

        # Load dataset (avoiding redundant loads)
        dataset_key = dataset_path if os.path.exists(dataset_path) else dataset_path.lower()
        if dataset_key not in instantiated_networks:
            train_loader, val_loader, test_loader = load_cnn_dataset(dataset_path, train_split, val_split)
        else:
            train_loader, val_loader, test_loader = instantiated_networks[dataset_key][1]

        # Load model architecture from script
        model = load_model_from_script(arch, dataset_path, script_path, net_path, optional_kwargs)

        instantiated_networks[net_path] = (model, (train_loader, val_loader, test_loader))

    return instantiated_networks


def get_longest_matching_key(dataset_path: str) -> str:
    """
    Given a dataset path, return the longest matching key from the dataset_loaders dictionary.

    Args:
        dataset_path (str): The dataset path or name.

    Returns:
        str: The longest matching key from the dataset_loaders dictionary.
    """
    best_match = ""
    for key in dataset_loaders.keys():
        if key in dataset_path and len(key) > len(best_match):
            best_match = key
    return best_match


def load_model_from_script(arch: str, dataset_path: str, script_path: str, checkpoint_path: str,
                           optional_kwargs: dict) -> torch.nn.Module:
    """
    Dynamically loads a model architecture from a user-provided script and initializes it with a checkpoint.

    Args:
        arch (str): Model architecture (e.g., "resnet18").
        dataset_path (str): Path to the dataset / name of the dataset (e.g., "cifar-10")
        script_path (str): Path to the Python script containing model definition.
        checkpoint_path (str): Path to the model checkpoint (.pt/.pth/.th).
        optional_kwargs (dict):  Keyword dict for custom instantiation parameters (e.g., {num_classes=10, width=56}).
                                 If not assigned by the user, {} is propagated from instantiate_networks_and_load_datasets().

    Returns:
        nn.Module: The instantiated model.

    Raises:
        ValueError: If model instantiation fails.
    """
    if not os.path.exists(script_path):
        raise ValueError(f"Instantiation script not found: {script_path}")

    module_name = Path(script_path).stem  # Extract script name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Load module

    if not hasattr(module, arch):
        raise ValueError(f"Function '{arch}' not found in {script_path}")

    instantiation_func = getattr(module, arch)
    params_list = list(inspect.signature(getattr(module, arch)).parameters)
    params_dict = {}
    if NUM_CLASSES in params_list:
        # By default, 'num_classes' is dynamically assigned via dataset_loaders's correspondent train_data's 'classes' field
        params_dict[NUM_CLASSES] = optional_kwargs.get(NUM_CLASSES, len(dataset_loaders[dataset_path]()[0].classes))
    if LARGE_INPUT in params_list:
        # By default, 'large_input' is True is the dataset has 'imagenet' in its name / path
        large_input = optional_kwargs.get(LARGE_INPUT)
        # Safeguarding 'False' cases
        params_dict[LARGE_INPUT] = large_input if large_input is not None else "imagenet" in dataset_path
    if WIDTH in params_list:
        # By default, 'width' is scraped from 'network_path' via regex
        match = re.search(r'width(\d+)', checkpoint_path)
        params_dict[WIDTH] = optional_kwargs.get(WIDTH, int(match.group(1)) if match else None)

    # Extend params_dict with other optional_kwargs, excluding handled keys
    params_dict.update({k: v for k, v in optional_kwargs.items() if k not in [NUM_CLASSES, LARGE_INPUT, WIDTH]})

    model = instantiation_func(**params_dict)  # Instantiate model
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    return model


def get_cross_validation_splits(input_dict, shuffle=True):
    """
    Generate train/test splits for 10-fold cross-validation.

    Args:
        input_dict (dict): {model_path: (model, (train_loader, val_loader, test_loader))}
        shuffle (bool): Whether to shuffle models before splitting.

    Returns:
        List[Tuple[dict, dict]]: List of (train_dict, test_dict) pairs per fold.
    """
    model_paths = list(input_dict.keys())
    kf = KFold(n_splits=StaticConf.get_instance().conf_values.n_splits, shuffle=shuffle,
               random_state=StaticConf.get_instance().conf_values.seed)

    folds = []
    for train_indices, test_indices in kf.split(model_paths):
        train_dict = {model_paths[i]: input_dict[model_paths[i]] for i in train_indices}
        test_dict = {model_paths[i]: input_dict[model_paths[i]] for i in test_indices}
        folds.append((train_dict, test_dict))

    return folds


def parse_compression_rates(compression_rates):
    """
    Parse the compression rates from a list of floats into a dictionary format.

    Args:
        compression_rates (list of float): List of compression rates provided by the user.

    Returns:
        dict: Dictionary mapping indices to compression rates.
    """
    return {i: rate for i, rate in enumerate(compression_rates)}


def init_conf_values(test_name, input_dict, compression_rates_dict, train_compressed_layer_only,
                     allowed_acc_reduction, discount_factor, learning_rate, rollout_limit, passes, prune,
                     seed, num_epochs, runtime_limit, n_splits, train_split, val_split, database_dict,
                     actor_checkpoint_path, critic_checkpoint_path, save_pruned_checkpoints, test_ts):
    """
    Initialize configuration values for the A2C Agent.

    Args:
        test_name (str):                          Indicative agent training instance name.
        input_dict (dict):                        Agent Evaluation dict - {network_path:
                                                          [arch, instantiation_script_path, dataset_name_or_path], ...}
        database_dict (dict):                     Agent Training dict - {network_path:
                                                          [arch, instantiation_script_path, dataset_name_or_path], ...}.
                                                  Unused (training is skipped) if actor_checkpoint_path and
                                                  critic_checkpoint_path are provided (agent is pre-trained)
        actor_checkpoint_path (str):              Path to pre-trained Actor Checkpoint.
        critic_checkpoint_path (str):             Path to pre-trained Critic Checkpoint.
        compression_rates_dict (dict):            Mapping of actions to compression rates.
        train_compressed_layer_only (bool):    Whether to freeze existing layers and learn only new layers.
        allowed_acc_reduction (float): Maximum allowable accuracy drop (percentage).
        discount_factor (float):                  A.k.a Gamma, controls the weight of the agent's future rewards.
        learning_rate (float):                    Learning rate for the agent's optimizer. Controls the step size in
                                                  gradient descent.
        rollout_limit (int / None):               Ensures that the agent's rollout trajectory does not exceed a
                                                  predefined number of steps (optional).
        passes (int):                             Number of iterations over the layers.
        prune (bool):                             Whether to prune layers during compression or resize them manually.
        seed (int):                               Seed to be used by numpy, torch, etc. Defaults to 0.
        num_epochs (int):                         Number of training epochs per compression step.
        runtime_limit (int):                      Max runtime allowed by the user. Defaults to a week in seconds.
        n_splits (int):                           Inter-model evaluation - train/test splits for n-fold cross-validation.
                                                  Defaults to 0 (no CV), recommended value is 10.
        train_split (float):                      Intra-model evaluation - Fraction of the dataset to use for training.
                                                  Defaults to 0.7.
        val_split (float):                        Intra-model evaluation - Fraction of the dataset to use for validation.
                                                  Defaults to 0.2.
        save_pruned_checkpoints (bool):           Whether to save a final checkpoint for each pruned network.
                                                  Defaults to False.
        test_ts (str):                            Test's timestamp
    """
    if not torch.cuda.is_available():
        sys.exit("GPU was not allocated!")

    device = torch.device("cuda:0")
    print_flush(f"Device: {device}")
    print_flush(f"Device Name: {torch.cuda.get_device_name(0)}")

    cv = ConfigurationValues(
        device=device,
        test_name=test_name,
        input_dict=input_dict,
        compression_rates_dict=compression_rates_dict,
        train_compressed_layer_only=train_compressed_layer_only,
        allowed_acc_reduction=allowed_acc_reduction,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        rollout_limit=rollout_limit,
        passes=passes,
        prune=prune,
        seed=seed,
        num_epochs=num_epochs,
        runtime_limit=runtime_limit,
        n_splits=n_splits,
        train_split=train_split,
        val_split=val_split,
        database_dict=database_dict,
        actor_checkpoint_path=actor_checkpoint_path,
        critic_checkpoint_path=critic_checkpoint_path,
        save_pruned_checkpoints=save_pruned_checkpoints,
        test_ts=test_ts
    )
    StaticConf(cv)


# Define default transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset_loaders = {
    'cifar-10': lambda: (datasets.CIFAR10(root=SPECTRA_DATASETS, train=True, download=True, transform=transform),
                         datasets.CIFAR10(root=SPECTRA_DATASETS, train=False, download=True, transform=transform)),
    'cifar10': lambda: dataset_loaders['cifar-10'](),  # Alias to allow both formats
    'cifar-100': lambda: (datasets.CIFAR100(root=SPECTRA_DATASETS, train=True, download=True, transform=transform),
                          datasets.CIFAR100(root=SPECTRA_DATASETS, train=False, download=True, transform=transform)),
    'cifar100': lambda: dataset_loaders['cifar-100'](),  # Alias to allow both formats
    'mnist': lambda: (datasets.MNIST(root=SPECTRA_DATASETS, train=True, download=True, transform=transform),
                      datasets.MNIST(root=SPECTRA_DATASETS, train=False, download=True, transform=transform)),
    'svhn': lambda: (datasets.SVHN(root=SPECTRA_DATASETS, split='train', download=True, transform=transform),
                     datasets.SVHN(root=SPECTRA_DATASETS, split='test', download=True, transform=transform)),
    'imagenet1k': lambda: (datasets.ImageNet(root=SPECTRA_DATASETS, split='train', download=False, transform=transform),
                           datasets.ImageNet(root=SPECTRA_DATASETS, split='val', download=False, transform=transform)),
    'imagenet-1k': lambda: dataset_loaders['imagenet1k'](),  # Alias to allow both formats
    'imagenet': lambda: dataset_loaders['imagenet1k'](),  # Alias to allow all formats
}


def load_cnn_dataset(name_or_path: str, train_split: float, val_split: float):
    """
    Loads a standard or custom dataset, splitting it into train, validation, and test sets.
    Implicitly, test_split = 1 - train_split - val_split

    Args:
        name_or_path (str): Dataset name (e.g., 'cifar-10', 'mnist') or custom dataset path.
        train_split (float): Fraction of data for training. Defaults to 0.7.
        val_split (float): Fraction of data for validation. Defaults to 0.2.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for train, validation, and test datasets.
    """

    if name_or_path in dataset_loaders:
        train_data, test_data = dataset_loaders[name_or_path]()

        train_len = int(len(train_data) * train_split / (train_split + val_split))
        val_len = len(train_data) - train_len
        train_data, val_data = random_split(train_data, [train_len, val_len])
    elif os.path.exists(name_or_path):  # Custom dataset path
        dataset = datasets.ImageFolder(Path(name_or_path), transform=transform)
        train_len = int(len(dataset) * train_split)
        val_len = int(len(dataset) * val_split)
        test_len = len(dataset) - train_len - val_len
        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
        # Avoids repetitive loads and ensures data is accessible from within load_model_from_script()
        dataset_loaders[name_or_path] = lambda: (train_data, test_data)
    else:
        raise ValueError("Invalid dataset name or path. Provide a known dataset name or a valid directory.")

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_reward(new_acc, prev_acc, compression_rate):
    layer_reduction_size = (1 - compression_rate) * 100

    delta_acc = (new_acc - prev_acc) * 100

    if delta_acc < -StaticConf.get_instance().conf_values.allowed_acc_reduction:
        reward = -layer_reduction_size ** 3
    elif delta_acc > 0:
        reward = layer_reduction_size ** 3
    else:
        reward = layer_reduction_size

    return reward


def compute_returns(next_value, rewards, masks, gamma):
    """
    Compute the discounted returns for the agent's trajectory.

    This function calculates the cumulative discounted reward (return)
    for each time step in a trajectory. The return at time step `t` is
    defined as:
        R_t = reward_t + gamma * reward_{t+1} + gamma^2 * reward_{t+2} + ...

    Args:
        next_value (float or torch.Tensor): The estimated value of the next state
                                            (used for bootstrapping the return at the end of the trajectory).
        rewards (list[torch.Tensor]):       List of rewards collected during the trajectory.
        masks (list[torch.Tensor]):         List of binary masks indicating whether the episode
                                            is still ongoing (1) or has ended (0) at each time step.
        gamma (float):                      Discount factor, controls the weight of future rewards.

    Returns:
        list[torch.Tensor]: List of discounted returns, where each element corresponds
        to the cumulative return from that time step onward.
    """
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def is_to_change_bn_layer(curr_layer, last_layer):
    """
    Determine if the current BatchNorm layer needs to be updated.

    Args:
        curr_layer (nn.Module): The current layer in the model.
        last_layer (nn.Module): The last processed layer (Linear or Conv2D).

    Returns:
        bool: True if the BatchNorm layer needs to be updated, False otherwise.
    """
    return isinstance(curr_layer, (nn.BatchNorm1d, nn.BatchNorm2d)) and \
        last_layer is not None and \
        ((isinstance(last_layer, nn.Linear) and curr_layer.num_features != last_layer.out_features) or
         (isinstance(last_layer, nn.Conv2d) and curr_layer.num_features != last_layer.out_channels))


def get_layer_by_type(row, layer_types):
    """
    Retrieves the first layer of a specified type from a row of layers.

    Args:
        row (list):                       A list of layers (nn.Module instances) in a row.
        layer_types (tuple of Type): The types of layer to search for, e.g., nn.Linear, nn.Conv2D.

    Returns:
        nn.Module or None: The first layer of the specified type if found; otherwise, None.

    Supported Layer Types:
        - nn.Linear: Fully-connected (dense) layers.
        - nn.Conv2D: Convolutional layers.
        - nn.BatchNorm1d: Batch normalization for 1D data (e.g., dense layers).
        - nn.BatchNorm2d: Batch normalization for 2D data (e.g., images in CNNs).
        - nn.MaxPool2d: Max pooling layers for spatial down-sampling.
        - nn.AvgPool2d: Average pooling layers for spatial down-sampling.
        - nn.ReLU, nn.ELU, nn.Sigmoid, nn.Tanh, nn.Softmax: Common activation functions.
        - nn.Dropout: Dropout regularization layers.
        - nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d: Adaptive pooling layers for spatial down-sampling.
    """
    if not isinstance(layer_types, tuple):
        layer_types = tuple(layer_types)

    for layer in row:
        if isinstance(layer, layer_types):
            return layer


def get_model_layers_str(model):
    """
    Returns a string representation of the model's layers and their dimensions for Linear and Conv2D layers.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        str: A string representation of the layers with dimensions for Linear and Conv2D layers.
    """
    new_model_with_rows = ModelWithRows(model)
    layer_descriptions = []

    for row in new_model_with_rows.all_rows:
        linear_layer = get_layer_by_type(row, nn.Linear)
        conv_layer = get_layer_by_type(row, nn.Conv2d)

        if linear_layer is not None:
            layer_descriptions.append(
                f"Linear(in_features={linear_layer.in_features}, out_features={linear_layer.out_features})"
            )
        elif conv_layer is not None:
            layer_descriptions.append(
                f"Conv2d(in_channels={conv_layer.in_channels}, out_channels={conv_layer.out_channels}, "
                f"kernel_size={conv_layer.kernel_size}, stride={conv_layer.stride}, padding={conv_layer.padding})"
            )

    return "\n".join(layer_descriptions)


def get_model_layers(model, layer_types=(nn.Linear, nn.Conv2d)):
    """
    Retrieves a list of all layers of the specified types from the model.

    Args:
        model (nn.Module): The model to inspect.
        layer_types (tuple): The types of layers to retrieve, e.g., (nn.Linear, nn.Conv2d).

    Returns:
        list: A list of layers of the specified types.
    """
    new_model_with_rows = ModelWithRows(model)
    layers = []

    for row in new_model_with_rows.all_rows:
        for layer_type in layer_types:
            layer = get_layer_by_type(row, layer_type)
            if layer is not None:
                layers.append(layer)

    return layers


def calc_num_parameters(model, is_pruned=False):
    """
    Calculate the total number of parameters in the model, considering pruned parameters if specified.
    Because SPECTRA replaces layers rather than pruning them, is_pruned is always set to False.

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


def calc_flops(model, is_pruned=False):
    """
    Calculate the total number of floating-point operations (FLOPs) for the model,
    considering pruned parameters if specified.
    Because SPECTRA replaces layers rather than pruning them, is_pruned is always set to False.

    Args:
        model (nn.Module): The model to analyze.
        is_pruned (bool): Whether to account for pruned parameters.

    Returns:
        float: The total number of FLOPs in the model (in millions).
    """
    total_flops = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs for Conv2d layers
            kernel_size = module.kernel_size[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            output_height, output_width = module.output_size[2], module.output_size[3]

            flops = (kernel_size * kernel_size * in_channels * output_height * output_width * out_channels)
            total_flops += flops

        elif isinstance(module, nn.Linear):
            # FLOPs for Linear layers
            in_features = module.in_features
            out_features = module.out_features

            flops = in_features * out_features
            total_flops += flops

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # FLOPs for BatchNorm1/2d
            num_features = module.num_features
            flops = num_features  # For each feature, one operation for mean and variance
            total_flops += flops

        elif isinstance(module, (nn.ReLU, nn.ELU, nn.SiLU, nn.Softmax, nn.Tanh, nn.Sigmoid)):
            # FLOPs for activation functions
            input_size = module.input_size
            flops = input_size[1] * input_size[2] * input_size[3]  # Element-wise operation
            total_flops += flops

        elif isinstance(module, nn.Dropout):
            # Dropout does not contribute to FLOPs during inference
            pass

    if is_pruned:
        # If pruned, account for the removed weights and the corresponding FLOPs
        pruned_flops = 0
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                pruned_flops += (module.weight_mask == 0).sum().item()
        total_flops -= pruned_flops  # Remove pruned FLOPs

    return total_flops


def save_times_csv(name, times, datasets):
    dataset_names = list(map(os.path.basename, datasets))
    data = np.array([dataset_names, times]).transpose()
    pd.DataFrame(data, columns=['Dataset', 'time']).to_csv(f"./times/{name}.csv")


def normalize_2d_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def normalize_3d_data(data):
    return np.array(list(map(normalize_2d_data, data)))
