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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
import src.distributed as ddp
import src.logging_utils as logging_utils

# Root under which torchvision datasets are downloaded / looked up.
# Override with the SPECTRA_DATASETS environment variable.
SPECTRA_DATASETS = os.environ.get("SPECTRA_DATASETS", "/home/paretsky/spectra_datasets")

# DataLoader workers per loader. Each dataset builds 3 loaders, so keep this modest:
# the previous hard-coded 8 (with persistent_workers) spawned hundreds of resident
# processes once several datasets were preloaded.
DATALOADER_WORKERS = int(os.environ.get("SPECTRA_DATALOADER_WORKERS", "4"))

# Possible instantiation functions' parameters
NUM_CLASSES = "num_classes"
LARGE_INPUT = "large_input"
WIDTH = "width"


def print_flush(msg):
    """
    Informational log line.

    Kept under its original name so the ~100 existing call sites keep working, but it is now
    a thin shim over src.logging_utils: timestamped to the millisecond, level-tagged,
    rank-tagged, annotated with the active context (episode / network / layer) and mirrored
    into the run's per-rank log file. Non-main ranks no longer discard their output; the
    console filter lives in the handler, so rank 1's messages still reach rank1.log.
    """
    logging_utils.info(str(msg))


def str2bool(value):
    """
    Parse a boolean command-line flag.

    ``type=bool`` in argparse calls ``bool(str)``, which is True for every non-empty string:
    ``--prune False`` and ``--save_pruned_checkpoints False`` both evaluated to True, so these
    switches could not be turned off from the command line at all. That silently disabled the
    masking-vs-structural comparison and forced checkpoint writing during smoke runs.
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "t", "yes", "y", "1"):
        return True
    if normalized in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


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

    parser.add_argument('--datasets', type=str, nargs='*', default=None,
        help=(
            "Datasets to preload once, up-front, and share across every network that uses them "
            "(e.g. --datasets cifar-10 cifar-100). Each entry is a known dataset name (see "
            "utils.dataset_loaders) or a path to an ImageFolder-style directory. When omitted, "
            "the dataset names referenced by --input / --database are preloaded automatically."
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

    parser.add_argument('--train_compressed_layer_only', type=str2bool, default=True,
                        help="Whether to train the entire network or only the new layer, post-compression.\n"
                             "Training the entire network after the compression of each layer greatly affects runtime.")

    parser.add_argument('--split', type=str2bool, default=True,
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
                        help="How many per-layer compression iterations over the entire network. Default=1, 4 is also recommended.")

    parser.add_argument('--prune', type=str2bool, default=True,
                        help="Whether to prune layers via torch.nn.utils.prune.ln_structured during compression or resize them manually.")

    parser.add_argument('--num_epochs', type=int, default=40,
                        help="Post-compression fine-tune epochs per pruning step. "
                             "NEON used 100; SPECTRA default is 40. Early-stop patience is "
                             "SPECTRA_FINETUNE_PATIENCE (default 10).")

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

    parser.add_argument('--save_pruned_checkpoints', type=str2bool, default=False,
                        help="Whether to save a final checkpoint for each pruned network.")

    return parser.parse_args()


class DatasetRegistry:
    """
    Loads every dataset at most once and hands the same DataLoaders to all networks
    that share it.

    The previous implementation looked the dataset key up in the *networks* dictionary,
    so the cache never hit: every entry of --database rebuilt its dataset and spawned a
    fresh set of persistent DataLoader workers. With a few hundred networks that alone
    exhausted the node's processes and shared memory.
    """

    def __init__(self, train_split: float, val_split: float):
        self.train_split = train_split
        self.val_split = val_split
        self._entries = {}

    @staticmethod
    def key_for(spec) -> str:
        """Cache key that distinguishes a dataset *and* the preprocessing applied to it."""
        name_or_path, options = parse_dataset_spec(spec)
        base = name_or_path if os.path.exists(str(name_or_path)) else canonical_dataset_name(name_or_path)
        if not options:
            return base
        detail = ",".join(f"{k}={options[k]}" for k in sorted(options))
        return f"{base}[{detail}]"

    def get(self, spec) -> dict:
        key = self.key_for(spec)
        if key not in self._entries:
            print_flush(f"Preloading dataset '{key}'")
            loaders = load_cnn_dataset(spec, self.train_split, self.val_split)
            self._entries[key] = {
                "loaders": loaders,
                "num_classes": infer_num_classes(loaders[0].dataset),
                "input_shape": get_input_shape(loaders[0]),
            }
        return self._entries[key]

    def loaders(self, spec):
        return self.get(spec)["loaders"]

    def num_classes(self, spec) -> int:
        return self.get(spec)["num_classes"]

    def input_shape(self, spec) -> tuple:
        """(C, H, W) of a single sample, used to size architectures and count MACs."""
        return self.get(spec)["input_shape"]

    def __contains__(self, spec) -> bool:
        return self.key_for(spec) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return self._entries.keys()


def infer_num_classes(dataset) -> int:
    """
    Determine the number of classes of a (possibly ``Subset``-wrapped) dataset without
    re-instantiating it.
    """
    while hasattr(dataset, "dataset"):  # unwrap random_split Subsets
        dataset = dataset.dataset

    classes = getattr(dataset, "classes", None)
    if classes:
        return len(classes)

    for attr in ("targets", "labels"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return int(len(set(int(v) for v in values)))

    raise ValueError(f"Unable to infer the number of classes for dataset {type(dataset).__name__}")


def preload_datasets(datasets, train_split: float, val_split: float) -> DatasetRegistry:
    """
    Build the shared dataset registry before any network is instantiated.

    Args:
        datasets (list[str] | None): Dataset names/paths to load eagerly. When None, datasets
                                     are loaded lazily the first time a network references them.
        train_split (float):         Fraction of the dataset to use for training.
        val_split (float):           Fraction of the dataset to use for validation.

    Returns:
        DatasetRegistry: Registry shared by --input and --database instantiation.
    """
    registry = DatasetRegistry(train_split, val_split)
    for name_or_path in datasets or []:
        registry.get(name_or_path)
    return registry


def parse_input_argument(input_arg, dataloaders_dict: DatasetRegistry):
    """
    Parse the --input or --database arguments as a JSON-formatted string or file.

    Args:
        input_arg (str):                       A JSON string or a path to a JSON file for
                                               Agent Training / Evaluation.
        dataloaders_dict (DatasetRegistry):    Shared registry of preloaded datasets.

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
        return instantiate_networks_and_load_datasets(input_dict, dataloaders_dict)

    # Try reading JSON file
    try:
        with open(input_arg, 'r') as f:
            input_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise ValueError("Invalid input: Provide a valid JSON string or JSON file path.")

    return instantiate_networks_and_load_datasets(input_dict, dataloaders_dict)


def instantiate_networks_and_load_datasets(input_dict, dataloaders_dict: DatasetRegistry):
    """
    Instantiates networks from their given architecture checkpoint and instantiation script,
    then attaches the corresponding standard / custom dataset loaders.

    Args:
        input_dict (dict):                     {network_path: (architecture, instantiation_script,
                                                dataset_name_or_path[, optional_kwargs])}
        dataloaders_dict (DatasetRegistry):    Shared registry of preloaded datasets.

    Returns:
        dict: {network_path: (nn.Module, (train_loader, val_loader, test_loader))}

    Raises:
        ValueError: If model instantiation fails.
    """
    instantiated_networks = {}
    failures = []

    for net_path, values in input_dict.items():
        try:
            if len(values) == 3:
                arch, script_path, dataset_path = values
                optional_kwargs = {}  # Assign an empty dict if not assigned
            else:
                arch, script_path, dataset_path, optional_kwargs = values

            if not os.path.exists(net_path):
                raise ValueError(f"Network checkpoint not found: {net_path}")

            # Loaded once per dataset and reused by every network trained on it
            loaders = dataloaders_dict.loaders(dataset_path)
            num_classes = dataloaders_dict.num_classes(dataset_path)
            input_shape = dataloaders_dict.input_shape(dataset_path)

            # Load model architecture from script
            model = load_model_from_script(arch, dataset_path, script_path, net_path, optional_kwargs,
                                           num_classes, input_shape)

            instantiated_networks[net_path] = (model, loaders)
        except Exception as error:
            # A single malformed entry used to abort the whole run at the first bad path, so a
            # typo in one of ~280 database entries cost an entire allocation. Collect them all
            # instead and report them together: the count of dropped networks is recorded, so
            # the effective database size is never silently wrong.
            failures.append((net_path, f"{type(error).__name__}: {error}"))
            logging_utils.exception(f"Could not instantiate {net_path}")
            try:
                import src.run_recorder as _recorder
                _recorder.issue("network_load_failed", f"{type(error).__name__}: {error}",
                                network=net_path)
            except Exception:
                pass

    if failures:
        listing = "\n".join(f"  - {path}: {reason}" for path, reason in failures)
        logging_utils.warning(
            f"{len(failures)}/{len(input_dict)} networks could not be instantiated and were "
            f"skipped:\n{listing}")

    if not instantiated_networks:
        raise ValueError(
            f"None of the {len(input_dict)} configured networks could be instantiated. "
            f"Problems:\n" + "\n".join(f"  - {p}: {r}" for p, r in failures))

    return instantiated_networks


def load_model_from_script(arch: str, dataset_path, script_path: str, checkpoint_path: str,
                           optional_kwargs: dict, num_classes: int = None,
                           input_shape: tuple = None) -> torch.nn.Module:
    """
    Dynamically loads a model architecture from a user-provided script and initializes it with a checkpoint.

    Args:
        arch (str): Model architecture (e.g., "resnet18").
        dataset_path (str): Path to the dataset / name of the dataset (e.g., "cifar-10")
        script_path (str): Path to the Python script containing model definition.
        checkpoint_path (str): Path to the model checkpoint (.pt/.pth/.th).
        optional_kwargs (dict):  Keyword dict for custom instantiation parameters (e.g., {num_classes=10, width=56}).
                                 If not assigned by the user, {} is propagated from instantiate_networks_and_load_datasets().
        num_classes (int): Class count taken from the already-loaded dataset. Passing it avoids
                           rebuilding the dataset purely to read its ``classes`` attribute.

    Returns:
        nn.Module: The instantiated model.

    Raises:
        ValueError: If model instantiation fails.
    """
    if not os.path.exists(script_path):
        # The database JSONs are hand-maintained and mix 'thin_res_net.py' with
        # 'thin-res-net.py'. Accepting the other separator turns an entire lost run into a
        # warning, while still reporting the mismatch so the JSON gets corrected.
        alternative = Path(script_path).with_name(
            Path(script_path).name.replace("-", "_") if "-" in Path(script_path).name
            else Path(script_path).name.replace("_", "-"))
        if alternative.exists():
            print_flush(f"Instantiation script {script_path} not found; using {alternative} "
                        f"(separator mismatch - please correct the database JSON)")
            script_path = str(alternative)
        else:
            raise ValueError(f"Instantiation script not found: {script_path}")

    module_name = Path(script_path).stem  # Extract script name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Load module

    if not hasattr(module, arch):
        raise ValueError(f"Function '{arch}' not found in {script_path}")

    print_flush(f"{checkpoint_path=}, {script_path=}, {optional_kwargs=}")

    instantiation_func = getattr(module, arch)
    params_list = list(inspect.signature(getattr(module, arch)).parameters)
    params_dict = {}

    if NUM_CLASSES in params_list:
        # By default, 'num_classes' is taken from the already-loaded dataset (see DatasetRegistry)
        params_dict[NUM_CLASSES] = optional_kwargs.get(NUM_CLASSES, num_classes)
        if params_dict[NUM_CLASSES] is None:
            raise ValueError(f"'{arch}' requires num_classes but it could not be resolved for {dataset_path}")

    if LARGE_INPUT in params_list:
        large_input = optional_kwargs.get(LARGE_INPUT)
        if large_input is None:
            # Decided from the actual image size rather than from the dataset's name, so any
            # high-resolution dataset selects the ImageNet-style stem
            if input_shape is not None and len(input_shape) == 3:
                large_input = min(input_shape[1], input_shape[2]) >= 128
            else:
                large_input = "imagenet" in str(dataset_path).lower()
        params_dict[LARGE_INPUT] = large_input

    if WIDTH in params_list:
        # By default, 'width' is scraped from 'network_path' via regex
        match = re.search(r'width(\d+)', checkpoint_path)
        params_dict[WIDTH] = optional_kwargs.get(WIDTH, int(match.group(1)) if match else None)

    # Extend params_dict with other optional_kwargs, excluding handled keys
    params_dict.update({k: v for k, v in optional_kwargs.items() if k not in [NUM_CLASSES, LARGE_INPUT, WIDTH]})

    device = ddp.resolve_device()

    # The CNN under evaluation is repeatedly pruned and rebuilt, so it is deliberately left
    # unwrapped: a DDP replica would be invalidated by the first structural change.
    model = instantiation_func(**params_dict).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Detect mismatch: model expects 'module.' but checkpoint doesn't have it
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if model_keys[0].startswith("module.") and not ckpt_keys[0].startswith("module."):
        # Wrap keys with 'module.' to match DistributedDataParallel
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not model_keys[0].startswith("module.") and ckpt_keys[0].startswith("module."):
        # Unwrap if model is not wrapped but checkpoint is
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
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
                     actor_checkpoint_path, critic_checkpoint_path, save_pruned_checkpoints, test_ts,
                     dataloaders_dict=None):
    """
    Initialize configuration values for the A2C Agent.

    Args:
        test_name (str):                          Indicative agent training instance name.
        dataloaders_dict (DatasetRegistry):       Shared registry of preloaded datasets.
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

    device = ddp.resolve_device()

    print_flush(f"Device: {device} (world size {ddp.get_world_size()})")
    print_flush(f"Device Name: {torch.cuda.get_device_name(device.index)}")

    cv = ConfigurationValues(
        device=device,
        dataloaders_dict=dataloaders_dict,
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


# Channel statistics per dataset. Normalising every dataset with 0.5/0.5, as before, shifts
# the activation distribution the agent reads and therefore the features it acts on.
DATASET_STATS = {
    'cifar-10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar-100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'mnist': ((0.1307,), (0.3081,)),
    'fashion-mnist': ((0.2860,), (0.3530,)),
    'kmnist': ((0.1918,), (0.3483,)),
    'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'stl10': ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    'places365': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'imagenet1k': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
DEFAULT_STATS = ((0.5,), (0.5,))


def _torchvision_split(factory, train_kwargs, test_kwargs):
    """Build a (train, test) pair from a torchvision dataset class."""
    return lambda transform: (factory(root=SPECTRA_DATASETS, transform=transform, **train_kwargs),
                              factory(root=SPECTRA_DATASETS, transform=transform, **test_kwargs))


# Canonical name -> builder. Aliases are resolved by DATASET_ALIASES so that a database JSON
# can spell a dataset however its source repository did.
DATASET_BUILDERS = {
    'cifar-10': _torchvision_split(datasets.CIFAR10, {'train': True, 'download': True},
                                   {'train': False, 'download': True}),
    'cifar-100': _torchvision_split(datasets.CIFAR100, {'train': True, 'download': True},
                                    {'train': False, 'download': True}),
    'mnist': _torchvision_split(datasets.MNIST, {'train': True, 'download': True},
                                {'train': False, 'download': True}),
    'fashion-mnist': _torchvision_split(datasets.FashionMNIST, {'train': True, 'download': True},
                                        {'train': False, 'download': True}),
    'kmnist': _torchvision_split(datasets.KMNIST, {'train': True, 'download': True},
                                 {'train': False, 'download': True}),
    'svhn': _torchvision_split(datasets.SVHN, {'split': 'train', 'download': True},
                               {'split': 'test', 'download': True}),
    'stl10': _torchvision_split(datasets.STL10, {'split': 'train', 'download': True},
                                {'split': 'test', 'download': True}),
    'places365': _torchvision_split(datasets.Places365, {'split': 'train-standard', 'small': True},
                                    {'split': 'val', 'small': True}),
    'imagenet1k': _torchvision_split(datasets.ImageNet, {'split': 'train'}, {'split': 'val'}),
    'imagenet1kv2': lambda transform: (
        datasets.ImageNet(root=SPECTRA_DATASETS, split='train', transform=transform),
        datasets.ImageFolder(root=f'{SPECTRA_DATASETS}/imagenetv2-matched-frequency', transform=transform)),
}

DATASET_ALIASES = {
    'cifar10': 'cifar-10', 'cifar_10': 'cifar-10',
    'cifar100': 'cifar-100', 'cifar_100': 'cifar-100',
    'fashionmnist': 'fashion-mnist', 'fashion_mnist': 'fashion-mnist', 'fmnist': 'fashion-mnist',
    'stl-10': 'stl10', 'places-365': 'places365', 'places': 'places365',
    'imagenet': 'imagenet1k', 'imagenet-1k': 'imagenet1k', 'imagenet1k-v1': 'imagenet1k',
    'imagenet1kv1': 'imagenet1k', 'imagenet-1k-v1': 'imagenet1k', 'imagenet-1kv1': 'imagenet1k',
    'imagenet1k-v2': 'imagenet1kv2', 'imagenet-1k-v2': 'imagenet1kv2',
    'imagenet-1kv2': 'imagenet1kv2', 'imagenetv2': 'imagenet1kv2', 'imagenet-v2': 'imagenet1kv2',
}


def canonical_dataset_name(name: str) -> str:
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


def parse_dataset_spec(spec):
    """
    Normalise a dataset entry from the input/database JSON.

    Accepts a plain name or path ("cifar-10", "/data/my_images") or a mapping that also
    carries preprocessing needed to match the checkpoint being pruned, e.g.
    ``{"name": "mnist", "image_size": 32, "to_rgb": true}`` for a 3-channel network trained
    on upscaled digits.

    Returns:
        (str, dict): the dataset name or path, and its options.
    """
    if isinstance(spec, dict):
        options = dict(spec)
        name = options.pop("name", None) or options.pop("path", None)
        if name is None:
            raise ValueError(f"Dataset specification {spec} needs a 'name' or 'path' entry")
        return name, options
    return spec, {}


def build_transform(name_or_path: str, options: dict):
    """
    Preprocessing pipeline for a dataset.

    Args:
        name_or_path (str): Dataset name or directory.
        options (dict):     May contain ``image_size`` (int or (H, W)) and ``to_rgb`` (bool),
                            which let a grayscale or small-image dataset feed a network that
                            expects something else -- the usual obstacle to reusing one agent
                            across unrelated datasets.
    """
    steps = []

    if options.get("to_rgb"):
        steps.append(transforms.Grayscale(num_output_channels=3))

    image_size = options.get("image_size")
    if image_size:
        steps.append(transforms.Resize(image_size if isinstance(image_size, (list, tuple))
                                       else (image_size, image_size)))

    steps.append(transforms.ToTensor())

    mean, std = DATASET_STATS.get(canonical_dataset_name(name_or_path), DEFAULT_STATS)
    if options.get("to_rgb") and len(mean) == 1:  # replicate grayscale statistics per channel
        mean, std = mean * 3, std * 3
    steps.append(transforms.Normalize(mean, std))

    return transforms.Compose(steps)


def load_cnn_dataset(spec, train_split: float, val_split: float):
    """
    Loads a standard or custom dataset, splitting it into train, validation, and test sets.
    Implicitly, test_split = 1 - train_split - val_split

    Args:
        spec: Dataset name (e.g., 'cifar-10', 'mnist'), a custom dataset path, or a mapping
              with a 'name'/'path' plus preprocessing options. See parse_dataset_spec().
        train_split (float): Fraction of data for training. Defaults to 0.7.
        val_split (float): Fraction of data for validation. Defaults to 0.2.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for train, validation, and test datasets.
    """
    name_or_path, options = parse_dataset_spec(spec)
    transform = build_transform(name_or_path, options)
    canonical = canonical_dataset_name(name_or_path)

    if canonical in DATASET_BUILDERS:
        train_data, test_data = DATASET_BUILDERS[canonical](transform)

        train_len = int(len(train_data) * train_split / (train_split + val_split))
        val_len = len(train_data) - train_len
        train_data, val_data = random_split(train_data, [train_len, val_len])
    elif os.path.exists(name_or_path):  # Custom dataset path
        dataset = datasets.ImageFolder(Path(name_or_path), transform=transform)
        train_len = int(len(dataset) * train_split)
        val_len = int(len(dataset) * val_split)
        test_len = len(dataset) - train_len - val_len
        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
    else:
        raise ValueError(
            f"Unknown dataset '{name_or_path}'. Provide a directory, or one of: "
            f"{', '.join(sorted(DATASET_BUILDERS))}.")

    # Create DataLoaders with optimizations. Worker-related options are only valid when
    # num_workers > 0, and the count is bounded so that preloading many datasets does not
    # exhaust the node's process/shared-memory budget.
    batch_size = get_adaptive_batch_size()
    worker_kwargs = {"num_workers": DATALOADER_WORKERS, "pin_memory": True}
    if DATALOADER_WORKERS > 0:
        worker_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **worker_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **worker_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **worker_kwargs)

    return train_loader, val_loader, test_loader


def compute_reward(new_acc, prev_acc, compression_rate, *,
                   params_before=None, params_after=None):
    """
    Preference-aware step reward (NEON lineage).

    Modes (``SPECTRA_REWARD_MODE``):
      neon (default)
          Original NEON trichotomy on nominal ``(1 - rate) * 100``.
      structural
          Same trichotomy, but the magnitude is *realized* parameter reduction
          ``(1 - params_after/params_before) * 100``. On CNNs with channel groups /
          residuals, nominal rate systematically mis-credits the edit; realized
          cost is the thesis-aligned CNN transfer of NEON's preference reward.
      structural_shaped
          Like ``structural``, but softens the hard preference cliff: within-budget
          mild losses scale the positive term by ``((τ + Δacc) / τ)^2``, and
          over-budget penalties grow with how far past τ the drop goes. Keeps
          NEON's preference semantics while fixing CNN credit cliffs we observed
          empirically (≈−5 to −11 pp recoverable steps looking like −1000).
      structural_guard
          Asymmetric RCPR: realized param reduction for in-budget / accuracy-gain
          credit, but ``max(realized, nominal)`` on over-budget penalties so tiny
          channel-group edits cannot under-penalize preference violations.

    The NEON body is preserved verbatim under ``neon``; other modes are explicit
    gated ablations for A/B experiments.
    """
    import os
    mode = os.environ.get("SPECTRA_REWARD_MODE", "neon").strip().lower()
    tau = float(StaticConf.get_instance().conf_values.allowed_acc_reduction)
    delta_acc = (new_acc - prev_acc) * 100

    nominal = (1.0 - float(compression_rate)) * 100.0
    realized = None
    if (mode in ("structural", "structural_shaped", "structural_guard")
            and params_before is not None and params_after is not None
            and float(params_before) > 0):
        realized = max(0.0, (1.0 - float(params_after) / float(params_before)) * 100.0)
        reduction = realized
        # Identity / no-op edits: keep a tiny nominal floor so the trichotomy still
        # distinguishes "safe identity" from catastrophic prune when Δacc is nonzero
        # due to FT noise; if both reduction and nominal are 0, reward is 0.
        if reduction < 1e-9:
            reduction = nominal
    else:
        reduction = nominal

    if mode == "structural_guard":
        # Prefer realized credit when safe; never softer than nominal when over τ.
        if delta_acc < -tau:
            reduction = max(reduction, nominal)
        # else keep realized (or nominal floor for true no-ops)

    if mode in ("neon", "structural", "structural_guard"):
        # NEON trichotomy (Hirsch & Katz 2022), magnitude = reduction
        if delta_acc < -tau:
            reward = -reduction ** 3
        elif delta_acc > 0:
            reward = reduction ** 3
        else:
            reward = reduction
        return reward

    if mode == "structural_shaped":
        # Soft preference shaping (still preference-aware; not AMC's -Error·log FLOPs).
        if delta_acc < -tau:
            overshoot = (-delta_acc - tau) / max(tau, 1e-6)
            reward = -(reduction ** 3) * (1.0 + overshoot)
        elif delta_acc > 0:
            reward = (reduction ** 3) * (1.0 + 0.1 * delta_acc)
        else:
            # Mild loss inside budget: taper toward 0 as we approach the cliff
            soften = ((tau + delta_acc) / max(tau, 1e-6)) ** 2
            reward = reduction * soften
        return reward

    # Unknown mode → NEON fallback
    if delta_acc < -tau:
        return -nominal ** 3
    if delta_acc > 0:
        return nominal ** 3
    return nominal


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
    # A bare class is the common call (get_layer_by_type(row, nn.Linear)), and tuple() on a
    # class raises TypeError: 'type' object is not iterable. That aborted every evaluation
    # record, since get_model_layers_str passes exactly that.
    if isinstance(layer_types, type):
        layer_types = (layer_types,)
    elif not isinstance(layer_types, tuple):
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


def get_input_shape(loader) -> tuple:
    """
    Per-sample input shape (C, H, W) taken from one batch of `loader`.

    Needed because FLOPs depend on the spatial size of every feature map, which is a
    property of the data rather than of the module definitions.
    """
    x_batch = next(iter(loader))[0]
    return tuple(x_batch.shape[1:])


def per_module_macs(model, input_shape, device=None) -> dict:
    """
    Multiply-accumulate operations attributable to each module, for a single input sample.

    The counts are collected from a real forward pass, because the spatial dimensions of
    every intermediate feature map (and therefore the cost of each convolution) are only
    known once an input has flowed through the network. The previous implementation read
    ``module.output_size`` / ``module.input_size``, attributes PyTorch modules do not have,
    so it raised ``AttributeError`` on the first Conv2d.

    Args:
        model (nn.Module):     The model to analyze.
        input_shape (tuple):   Per-sample input shape, e.g. (3, 32, 32). See get_input_shape().
        device (torch.device): Device to run the probe pass on. Defaults to the model's device.

    Returns:
        dict: ``id(module) -> MACs``. Keyed by identity so callers can look a layer up
              without needing its qualified name.
    """
    model = ddp.unwrap(model)
    if device is None:
        device = next(model.parameters()).device

    counts = {}
    handles = []

    def conv_hook(module, _inputs, output):
        # One output element costs (in_channels / groups) * k_h * k_w multiply-accumulates
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        counts[id(module)] = counts.get(id(module), 0) + output.numel() * kernel_ops

    def linear_hook(module, _inputs, output):
        counts[id(module)] = counts.get(id(module), 0) + output.numel() * module.in_features

    def elementwise_hook(module, _inputs, output):
        counts[id(module)] = counts.get(id(module), 0) + output.numel()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.ELU, nn.SiLU,
                                 nn.Softmax, nn.Tanh, nn.Sigmoid)):
            handles.append(module.register_forward_hook(elementwise_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return counts


def calc_flops(model, input_shape, device=None):
    """
    Total multiply-accumulate operations for a single input sample.

    Args:
        model (nn.Module):     The model to analyze.
        input_shape (tuple):   Per-sample input shape, e.g. (3, 32, 32).
        device (torch.device): Device to run the probe pass on.

    Returns:
        float: Total MACs for one sample.
    """
    return float(sum(per_module_macs(model, input_shape, device).values()))


def save_times_csv(name, times, datasets):
    dataset_names = list(map(os.path.basename, datasets))
    data = np.array([dataset_names, times]).transpose()
    pd.DataFrame(data, columns=['Dataset', 'time']).to_csv(f"./times/{name}.csv")


def normalize_2d_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def normalize_3d_data(data):
    return np.array(list(map(normalize_2d_data, data)))

def get_adaptive_batch_size(base=64):
    """
    Returns an adaptive batch size based on the GPU type and memory size.
    """
    if not torch.cuda.is_available():
        return base

    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0).lower()

    # Conservative default scale
    multiplier = 1

    if "1080" in name:
        multiplier = 1
    elif "2080" in name:
        multiplier = 2
    elif "3090" in name or "titan" in name:
        multiplier = 4
    elif "a100" in name or "6000" in name:
        multiplier = 6
    elif total_memory_gb > 40:
        multiplier = 8  # e.g., A6000 48GB

    # Final batch size capped to safe upper bound
    return min(base * multiplier, 1024)
