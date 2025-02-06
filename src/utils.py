import os
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf


def print_flush(msg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} -- {msg}", flush=True)


def extract_args_from_cmd():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Script for training and evaluating SPECTRA A2C agent for CNN pruning.")

    parser.add_argument(
        '--input', type=str, required=True,
        help=(
            "Path to a JSON file or a JSON-formatted (dict-like) string. "
            "The JSON should map network paths to configurations:\n"
            "{\n"
            "  \"network_path_1\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\"],\n"
            "  \"network_path_2\": [\"architecture\", \"instantiation_script_path\", \"dataset_name_or_path\"]\n"
            "}\n\n"
            "- 'network_path': Path to the network checkpoint (.pt/.pth/.th) file.\n"
            "- 'architecture': The architecture name of the model (e.g., resnet18).\n"
            "- 'instantiation_script_path': Path to the script from source repository where the architecture "
            "                               instantiation function resides.\n"
            "- 'dataset_name_or_path': Path to a custom dataset, or name of a standard dataset "
            "                          (supported in utils.load_cnn_dataset(), such as 'cifar-10')."
        )
    )

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

    parser.add_argument('--allowed_reduction_acc', type=int, default=5,
                        help="The permissible reduction in performance (in percents). Default value=5; 1 is also recommended.")

    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help="Discount Factor, a.k.a Gamma, controls the weight of the agent's future rewards.")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate for the agent's optimizer. Controls the step size in gradient descent.")

    parser.add_argument('--rollout_limit', type=int, default=None,
                        help="Ensures that the agent's rollout trajectory does not exceed a predefined number of steps (optional).")

    parser.add_argument('--passes', type=int, default=1,
                        help="How many per-layer compression iterations over the entire network. Default=1 4 is also recommended.")

    parser.add_argument('--prune', type=bool, default=True, help="Whether to prune layers during compression.")

    parser.add_argument('--num_epochs', type=int, default=100, help="Agent's training epochs amount. Default=100.")

    parser.add_argument('--runtime_limit', type=int, default=60 * 60 * 24 * 7,
                        help="Max runtime. Default is a week in seconds")

    parser.add_argument('--seed', type=int, default=0,
                        help="Seed to be used by pytorch and numpy libraries. Default=0.")

    parser.add_argument('--train_split', type=float, default=0.7, help="Training data split fraction.")

    parser.add_argument('--val_split', type=float, default=0.2,
                        help="Validation data split fraction. Test data split is 1 - train_split - val_split")

    parser.add_argument('--actor_checkpoint', type=str, default=None,
                        help="Path to Actor Checkpoint (pre-trained agent)")

    parser.add_argument('--critic_checkpoint', type=str, default=None,
                        help="Path to Critic Checkpoint (pre-trained agent)")

    return parser.parse_args()


def parse_input_argument(input_arg, train_split, val_split):
    """
    Parse the --input argument as a JSON-formatted string or file.

    Args:
        input_arg (str):        A JSON string or a path to a JSON file.
        train_split (float):    Fraction of the dataset to use for training.
        val_split (float):      Fraction of the dataset to use for validation.

    Returns:
       dict: A dictionary mapping network paths to:
              - Instantiated model (nn.Module).
              - Dataset loaders (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If input is invalid or instantiation fails.
    """
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

    return instantiate_networks_and_load_datasets(input_dict)


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

    for net_path, (arch, script_path, dataset_path) in input_dict.items():
        if not os.path.exists(net_path):
            raise ValueError(f"Network checkpoint not found: {net_path}")

        # Load model architecture from script
        model = load_model_from_script(arch, script_path, net_path)

        # Load dataset (avoiding redundant loads)
        dataset_key = dataset_path if os.path.exists(dataset_path) else dataset_path.lower()
        if dataset_key not in instantiated_networks:
            train_loader, val_loader, test_loader = load_cnn_dataset(dataset_path, train_split, val_split)
        else:
            train_loader, val_loader, test_loader = instantiated_networks[dataset_key][1]

        instantiated_networks[net_path] = (model, (train_loader, val_loader, test_loader))

    return instantiated_networks


def load_model_from_script(arch: str, script_path: str, checkpoint_path: str) -> torch.nn.Module:
    """
    Dynamically loads a model architecture from a user-provided script and initializes it with a checkpoint.

    Args:
        arch (str): Model architecture (e.g., "resnet18").
        script_path (str): Path to the Python script containing model definition.
        checkpoint_path (str): Path to the model checkpoint (.pt/.pth/.th).

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

    model = getattr(module, arch)()  # Instantiate model
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    return model


def parse_compression_rates(compression_rates):
    """
    Parse the compression rates from a list of floats into a dictionary format.

    Args:
        compression_rates (list of float): List of compression rates provided by the user.

    Returns:
        dict: Dictionary mapping indices to compression rates.
    """
    return {i: rate for i, rate in enumerate(compression_rates)}


def init_conf_values(test_name, input_dict, compression_rates_dict, is_train_compressed_layer_only,
                     total_allowed_accuracy_reduction, discount_factor, learning_rate, rollout_limit, passes, prune,
                     num_epochs, runtime_limit, train_split, val_split, actor_checkpoint_path, critic_checkpoint_path):
    """
    Initialize configuration values for the A2C Agent.

    Args:
        test_name (str):                          Indicative agent training instance name.
        input_dict (dict):                        {network_path: [arch, instantiation_script_path, dataset_name_or_path], ...}
        compression_rates_dict (dict):            Mapping of actions to compression rates.
        is_train_compressed_layer_only (bool):    Whether to freeze existing layers and learn only new layers.
        total_allowed_accuracy_reduction (float): Maximum allowable accuracy drop (percentage).
        discount_factor (float):                  A.k.a Gamma, controls the weight of the agent's future rewards.
        learning_rate (float):                    Learning rate for the agent's optimizer. Controls the step size in gradient descent.
        rollout_limit (int / None):               Ensures that the agent's rollout trajectory does not exceed a predefined number of steps (optional).
        passes (int):                             Number of iterations over the layers.
        prune (bool):                             Whether to prune layers during compression.
        num_epochs (int):                         Number of training epochs per compression step.
        runtime_limit (int):                      Max runtime allowed by the user. Defaults to a week in seconds.
        train_split (float):                      Fraction of the dataset to use for training. Defaults to 0.7.
        val_split (float):                        Fraction of the dataset to use for validation. Defaults to 0.2.
        actor_checkpoint_path (str):              Path to pre-trained Actor Checkpoint
        critic_checkpoint_path (str):             Path to pre-trained Critic Checkpoint
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
        is_train_compressed_layer_only=is_train_compressed_layer_only,
        total_allowed_accuracy_reduction=total_allowed_accuracy_reduction,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        rollout_limit=rollout_limit,
        passes=passes,
        prune=prune,
        num_epochs=num_epochs,
        runtime_limit=runtime_limit,
        train_split=train_split,
        val_split=val_split,
        actor_checkpoint_path=actor_checkpoint_path,
        critic_checkpoint_path=critic_checkpoint_path
    )
    StaticConf(cv)


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
    # Define default transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_loaders = {
        'cifar-10': lambda: (datasets.CIFAR10(root="data", train=True, download=True, transform=transform),
                             datasets.CIFAR10(root="data", train=False, download=True, transform=transform)),
        'cifar-100': lambda: (datasets.CIFAR100(root="data", train=True, download=True, transform=transform),
                              datasets.CIFAR100(root="data", train=False, download=True, transform=transform)),
        'mnist': lambda: (datasets.MNIST(root="data", train=True, download=True, transform=transform),
                          datasets.MNIST(root="data", train=False, download=True, transform=transform)),
        'svhn': lambda: (datasets.SVHN(root="data", split='train', download=True, transform=transform),
                         datasets.SVHN(root="data", split='test', download=True, transform=transform)),
        'imagenet1k': lambda: (datasets.ImageNet(root="data", split='train', download=False, transform=transform),
                               datasets.ImageNet(root="data", split='val', download=False, transform=transform)),
    }

    if name_or_path in dataset_loaders:
        train_data, test_data = dataset_loaders[name_or_path]()
        total_len = len(train_data)
        train_len = int(total_len * train_split)
        val_len = int(total_len * val_split)
        train_data, val_data = random_split(train_data, [train_len, val_len])
    elif os.path.exists(name_or_path):  # Custom dataset path
        dataset = datasets.ImageFolder(Path(name_or_path), transform=transform)
        train_len = int(len(dataset) * train_split)
        val_len = int(len(dataset) * val_split)
        test_len = len(dataset) - train_len - val_len
        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
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

    if delta_acc < -StaticConf.get_instance().conf_values.total_allowed_accuracy_reduction:
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


def convert_state_to_text(state):
    """
    Converts the feature map state into a list of textual descriptions.

    Args:
        state (list of tuples): Feature maps and their properties.

    Returns:
        list of str: Textual representations of the state.
    """
    # TODO: Implementation TBD
    state_text = []
    for layer_properties in state:
        state_text.append(f"Layer type: {layer_properties[0]}, size: {layer_properties[1]}")
    return state_text
