import os
from pathlib import Path
from datetime import datetime
from os.path import join
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
import numpy as np


def print_flush(msg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} -- {msg}", flush=True)


# TODO: Adapt to SPECTRA's loading mechanism
def load_models_path(main_path, mode='train'):
    model_paths = []

    for root, dirs, files in os.walk(main_path):
        if ('X_train.csv' not in files):
            raise FileNotFoundError(f"{root} should contain 'X_train.csv'.\nMake sure you have run "
                                    f"`Train Base Networks.ipynb' over {root.split('/')[-2]} in advance")
        train_data_path = join(root, 'X_train.csv')

        if mode in ['train', 'test']:
            model_names = pd.read_csv(join(root, f'{mode}_models.csv'))['0'].to_numpy()
        else:
            model_names = files

        model_files = list(map(lambda file: os.path.join(root, file),
                               filter(lambda file_name: file_name.endswith('.pt') and file_name in model_names, files)))
        model_paths.append((train_data_path, model_files))

    return model_paths


def load_cnn_dataset(name_or_path: str = None, train_split: float = 0.7, val_split: float = 0.2):
    """
    Load datasets commonly used for CNNs or a user-provided dataset.

    Args:
        name_or_path (str): Name of the dataset to load ('cifar-10', 'cifar-100', 'mnist', 'svhn', 'imagenet1k' or custom).
        train_split (float): Fraction of the dataset to use for training. Defaults to 0.8.
        val_split (float): Fraction of the dataset to use for validation. Defaults to 0.1.
        test_split (float): Fraction of the dataset to use for testing. Defaults to 0.1.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for train, validation, and test datasets.
    """
    assert train_split + val_split < 1, f"{train_split=} + {val_split=} >= 1"

    # Define default transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if name_or_path == 'cifar-10':
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif name_or_path == 'cifar-100':
        train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    elif name_or_path == 'mnist':
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif name_or_path == 'svhn':
        train_data = datasets.SVHN(root="data", split='train', download=True, transform=transform)
        test_data = datasets.SVHN(root="data", split='test', download=True, transform=transform)
    elif name_or_path == 'imagenet1k':
        train_data = datasets.ImageNet(root="data", split='train', download=False, transform=transform)
        test_data = datasets.ImageNet(root="data", split='val', download=False, transform=transform)  # split='test' is not an option
    elif os.path.exists(name_or_path):
        user_dataset_path = Path(name_or_path)
        dataset = datasets.ImageFolder(user_dataset_path, transform=transform)
        train_len = int(len(dataset) * train_split)
        val_len = int(len(dataset) * val_split)
        test_len = len(dataset) - train_len - val_len
        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
    else:
        # TODO: Finalize instructions
        raise ValueError("Dataset name or path was not provided / unsupported, please provide a name or a path explicitly.")

    # Split train_data into train and validation datasets if not already split
    if not os.path.exists(name_or_path):  # Custom datasets' splits are already handled above
        total_len = len(train_data)
        train_len = int(total_len * train_split)
        val_len = int(total_len * val_split)
        train_data, val_data = random_split(train_data, [train_len, val_len])

    # Create DataLoaders for train, validation, and test sets
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


# TODO: compute_reward is to be adapted to address skip-connections, when pruning Conv2D layers in ResNets & DenseNets
def compute_reward(new_acc, prev_acc, compression_rate):
    total_allowed_accuracy_reduction = StaticConf.get_instance().conf_values.total_allowed_accuracy_reduction
    layer_reduction_size = (1 - compression_rate) * 100

    delta_acc = (new_acc - prev_acc) * 100

    if delta_acc < -total_allowed_accuracy_reduction:
        reward = -layer_reduction_size ** 3
    elif delta_acc > 0:
        reward = layer_reduction_size ** 3
    else:
        reward = layer_reduction_size

    return reward


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


def get_layer_by_type(row, layer_type):
    """
    Retrieves the first layer of a specified type from a row of layers.

    Args:
        row (list): A list of layers (nn.Module instances) in a row.
        layer_type (type): The type of layer to search for, e.g., nn.Linear, nn.Conv2d.

    Returns:
        nn.Module or None: The first layer of the specified type if found; otherwise, None.

    Supported Layer Types:
        - nn.Linear: Fully-connected (dense) layers.
        - nn.Conv2d: Convolutional layers.
        - nn.BatchNorm1d: Batch normalization for 1D data (e.g., dense layers).
        - nn.BatchNorm2d: Batch normalization for 2D data (e.g., images in CNNs).
        - nn.MaxPool2d: Max pooling layers for spatial down-sampling.
        - nn.AvgPool2d: Average pooling layers for spatial down-sampling.
        - nn.ReLU, nn.ELU, nn.Sigmoid, nn.Tanh, nn.Softmax: Common activation functions.
        - nn.Dropout: Dropout regularization layers.
        - nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d: Adaptive pooling layers for spatial down-sampling.
    """
    for layer in row:
        if isinstance(layer, layer_type):
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


def save_times_csv(name, times, datasets):
    dataset_names = list(map(os.path.basename, datasets))
    data = np.array([dataset_names, times]).transpose()
    pd.DataFrame(data, columns=['Dataset', 'time']).to_csv(f"./times/{name}.csv")


def normalize_2d_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def normalize_3d_data(data):
    return np.array(list(map(normalize_2d_data, data)))
