import os
import importlib
import re
import torch
import torch.nn as nn
import pickle
import bz2
import sentencepiece as spm
from transformers import BertTokenizer, BertModel


WEIGHTS = 'weights'
GRADIENTS = 'gradients'
NETWORK_INFO = 'network_info'
ACTIVATIONS = 'activations'

FILENAME = "filename"
ARCH = "architecture"
DATASET = "dataset"
REPO = "repository"
PARAMS = "parameters"
FLOPS = "flops"

INPUT_TENSOR = "input_tensor"
NUM_CLASSES = "num_classes"
WIDTH = "width"
LARGE_INPUT = "large_input"

# TODO: Add appending mechanism via parsed arguments
DATASET_INFO_DICT = {
        "cifar10": {NUM_CLASSES: 10, INPUT_TENSOR: torch.randn(1, 3, 32, 32, requires_grad=True)},  # CIFAR-10: RGB 32x32
        "mnist": {NUM_CLASSES: 10, INPUT_TENSOR: torch.randn(1, 1, 28, 28, requires_grad=True)},  # MNIST: Grayscale 28x28
        "cifar100": {NUM_CLASSES: 100, INPUT_TENSOR: torch.randn(1, 3, 32, 32, requires_grad=True)},  # CIFAR-100: RGB 32x32
        "imagenet1k": {NUM_CLASSES: 1000, INPUT_TENSOR: torch.randn(1, 3, 224, 224, requires_grad=True)},  # ImageNet-1k: RGB 224x224
        "imagenet1kv1": {NUM_CLASSES: 1000, INPUT_TENSOR: torch.randn(1, 3, 224, 224, requires_grad=True)},  # ImageNet-1k: RGB 224x224
        "imagenet1kv2": {NUM_CLASSES: 1000, INPUT_TENSOR: torch.randn(1, 3, 224, 224, requires_grad=True)},  # ImageNet-1k: RGB 224x224
        "imagenet22k": {NUM_CLASSES: 22000, INPUT_TENSOR: torch.randn(1, 3, 224, 224, requires_grad=True)},  # ImageNet-22k: RGB 224x224
    }


def save_params(state_dict, network_info_path):
    """
    Save weights and gradients of a model's convolutional and fully-connected layers (provided as a state_dict)
    to compressed files, ignoring pooling and other non-trainable layers.

    Args:
        state_dict (dict): The state_dict containing model parameters and gradients.
        network_info_path (str): Path to save dictionary of weights, gradients, activations and layer information - name, inferred type and shape.
    """

    # Initialize dictionaries to store layer information and data
    info_dict = {WEIGHTS: {}, GRADIENTS: {}, ACTIVATIONS: {}, NETWORK_INFO: {}}

    # Iterate over the state_dict
    for idx, (name, param) in enumerate(state_dict.items()):
        # print(f"{idx=}, {name=}, {param.shape=}")  # For debugging purposes
        if isinstance(param, torch.Tensor):  # Ensure it's a tensor
            shape = param.shape
            if len(shape) == 4:  # Conv2D or DownSamplingConv2D
                kernel_height, kernel_width = shape[2], shape[3]
                if kernel_height == 1 and kernel_width == 1:
                    # Part of a downsampling shortcut operation in a residual block
                    layer_type = "DownSamplingConv2D"  # 1x1 convolution
                else:
                    layer_type = "Conv2D"  # Standard convolution
            elif len(shape) == 2:  # Fully Connected
                layer_type = "FullyConnected"  # Matches general FC shape
            else:
                continue

            # TODO: Currently disregarding Batchnorm, Bias and Scalar layers (including running_mean, running_var, num_batches_tracked),
            #  the following commented code requires an update if these layer types are to be mapped

            # elif len(shape) == 1:  # BatchNorm or Bias
            #     if idx > 0 and len(state_dict[idx - 1].shape) == 4:  # Matches Conv2D or DownSamplingConv2D
            #         layer_type = "BatchNorm"
            #     else:
            #         layer_type = "Bias"  # Default if no match
            # elif len(shape) == 0:  # Scalar (e.g., num_batches_tracked)
            #     layer_type = "Scalar"
            # else:
            #     layer_type = "UnknownLayerType"

            # Store weights and gradients
            info_dict[WEIGHTS][idx] = param.cpu().numpy()
            if param.grad is not None:  # TODO: grad is None in pretrained networks! need to compute.
                print("breakpoint")
            info_dict[GRADIENTS][idx] = param.grad.cpu().numpy() if param.grad is not None else None

            # Add entry to layer_info
            info_dict[NETWORK_INFO][idx] = {
                "name": name,
                "type": layer_type,
                "shape": tuple(param.shape),
            }

            # Print layer details for debugging
            print(f"Processed layer {idx}: {name}, Type: {layer_type}, Shape: {param.shape}")

        else:
            raise ValueError(f"Parameter {name} is not a Tensor!")

    # Save weights, gradients, activations and info per layer dictionaries to a compressed file
    with bz2.BZ2File(network_info_path, mode="wb") as info_file:
        pickle.dump(info_dict, info_file)


def parse_filename(file_path):
    """ Helper function for filename parsing """
    filename = os.path.basename(file_path)

    try:
        filename_parts = filename.split("_")
        filename_parts[-1] = filename_parts[-1].rsplit(".", 1)[0]  # remove extension

        arch = filename_parts[0]
        dataset = filename_parts[1]
        repo = filename_parts[2]
        top1_acc = filename_parts[3]

        # Handle optional Params and FLOPs
        params = filename_parts[4] if len(filename_parts) > 4 else "Unknown"
        flops = filename_parts[5] if len(filename_parts) > 5 else "Unknown"

        # Print extracted metadata
        print(f"Network Architecture: {arch}")
        print(f"Dataset: {dataset}")
        print(f"Paper / Repository: {repo}")
        print(f"Top-1 Accuracy: {top1_acc}%")
        print(f"Parameters: {params}M")
        print(f"FLOPs: {flops}M")
    except IndexError as e:
        print("Error: Filename format does not match the expected format.")
        exit(1)
    return {FILENAME: filename, ARCH: arch, DATASET: dataset, REPO: repo, PARAMS: params, FLOPS: flops}


def tokenize_network(model, current_layer_index):
    """
    Tokenizes the entire network structure into a serialized format suitable for transformer input.

    Args:
        model (nn.Module): The PyTorch model to tokenize.
        current_layer_index (int): The index of the layer to prune, marked separately in the tokenized output.

    Returns:
        str: Serialized network representation with <SEP> markers.
    """
    transformer_input = []
    current_layer_token = None

    for idx, (name, module) in enumerate(model.named_modules()):
        params = [p for p in module.parameters(recurse=False)]
        if len(params) == 0:  # Skip modules without parameters
            continue

        # Use LayerExtractor-like functionality to gather layer details
        layer_metadata = {
            "layer_index": idx,
            "layer_name": name,
            "layer_type": type(module).__name__,
        }

        if isinstance(module, nn.Conv2d):
            layer_metadata["params"] = {
                "weights_shape": module.weight.data.cpu().numpy().shape,
                "gradients_shape": module.weight.grad.cpu().numpy().shape
                if module.weight.grad is not None else None,
            }
        elif isinstance(module, nn.Linear):
            layer_metadata["params"] = {
                "weights_shape": module.weight.data.cpu().numpy().shape,
                "bias_shape": module.bias.data.cpu().numpy().shape if module.bias is not None else None,
                "gradients_shape": module.weight.grad.cpu().numpy().shape
                if module.weight.grad is not None else None,
            }

        # Save current layer's tokenization separately
        layer_representation = f"{layer_metadata}"
        if idx == current_layer_index:
            current_layer_token = layer_representation

        # Append layer metadata for full network
        transformer_input.append(layer_representation)

    # Serialize the full network with <SEP> and highlight the current layer
    network_representation = " <SEP> ".join(transformer_input)
    return f"{network_representation} <SEP> Current Layer: {current_layer_token}"


def load_pretrained_model(file_path, file_info_dict):
    """
    Load a tensor or state_dict from a file, ensure it is properly moved to the appropriate device
    and convert to a Sequential model (Note: complex branching to be handled later).

    Args:
        file_path (str):        Path to the file containing the tensor or state_dict.
        file_info_dict (dict):  Indicating from which paper / repository the pretrained CNN was taken from,
                                to load the correct model, and also which model and dataset are to be used

    Returns:
        torch.Tensor or OrderedDict: The loaded tensor or state_dict on the appropriate device,
                                     or None if an error occurred.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the object from the file
        loaded_obj = torch.load(file_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Check if the loaded object is a tensor
        if isinstance(loaded_obj, torch.Tensor):
            # Ensure tensor is on the correct device
            tensor = loaded_obj.to(device)
            print(f"Tensor successfully loaded from {file_path}. Shape: {tensor.shape}, Device: {tensor.device}")
            return tensor

        # Check if the loaded object is an OrderedDict (commonly a state_dict)
        elif isinstance(loaded_obj, dict) or isinstance(loaded_obj, collections.OrderedDict):
            print(f"State dict successfully loaded from {file_path}. Keys: {list(loaded_obj.keys())}")
            # Move all tensors in the state_dict to the correct device
            state_dict = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                          for key, value in loaded_obj.items()}

            module = importlib.import_module(file_info_dict[REPO])
            base_model, args = infer_arch_args(file_info_dict[ARCH], file_info_dict[DATASET])
            model_method = getattr(module, base_model)
            # TODO: this section must be generalized, to encompass conventions from numerous repositories
            model = model_method(**args)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Extract weights
            for name, param in model.named_parameters():
                print(f"Weight for {name}: {param.data}")

            # Compute gradientsץ TODO: Is a single arbitrary backprop meaningful in any way?
            input_tensor = DATASET_INFO_DICT[file_info_dict[DATASET]][INPUT_TENSOR].to(device)
            output = model(input_tensor)
            target = torch.zeros_like(output)  # Example target
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad}")

            # Compute activations (register hooks)
            activations = {}

            def activation_hook(module, input, output):
                activations[module] = output

            for name, layer in model.named_modules():
                # TODO: Should I filter and save activations of only Conv and FC layers?
                # if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):  # Register on all Conv2D and Fully Connected layers
                    layer.register_forward_hook(activation_hook)

            with torch.no_grad():
                _ = model(input_tensor)

            for layer, activation in activations.items():
                print(f"Activations for layer {layer}: {activation.shape}")
            return model

        else:
            raise TypeError(f"Unsupported file content type: expected a tensor or OrderedDict, got {type(loaded_obj)}")

    except Exception as e:
        print(f"Error loading the object from {file_path}: {e}")
        return None

def infer_arch_args(arch, dataset):
    """
    Parse the architecture string to extract the base model name and parameters.
    Example: 'resnet20-width12' -> ('resnet20', {'width': 12})
    """
    # Split the base model and any configurations (e.g., width12)
    parts = arch.split('-')
    base_model = parts[0]
    args = {}

    args[NUM_CLASSES] = DATASET_INFO_DICT[dataset][NUM_CLASSES]
    args[LARGE_INPUT] = True if "imagenet" in dataset else False

    if parts[1] and parts[1].startswith(WIDTH):
        args[WIDTH] = int(re.search(r'\d+', parts[1]).group())

    return base_model, args


# Main script
if __name__ == "__main__":
    # User-provided file path
    # file_path = input("Enter the path to the .pt/.pth/.th network file: ")
    file_path = r"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\resnet20-width12_cifar10_thin-res-net_93.39_0.154_23.64.pt"
    # TODO: Call only if filename is in format (create a runtime flag), else demand info (arch, dataset, path to repo module) as parsed arguments
    file_info_dict = parse_filename(file_path)

    # Load the pretrained network
    model = load_pretrained_model(file_path, file_info_dict)
    if model is None:
        exit(1)

    # Save weights and gradients
    save_params(model, network_info_path=fr"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\info\{file_info_dict[FILENAME]}_pre.pbz2")


    # Tokenize the network
    network_input = tokenize_network(model, current_layer_index=0)
    print(f"Tokenized Network Input:\n{network_input}")

    # Prepare BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        network_input,
        return_tensors=filename.split(".")[-1],  # filename's extension
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Load pre-trained BERT
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Forward pass through BERT
    outputs = bert_model(**inputs)

    # Extract outputs for pruning decisions
    print("BERT outputs (hidden states):", outputs.last_hidden_state.shape)
    print("BERT outputs (pooled output):", outputs.pooler_output.shape)

