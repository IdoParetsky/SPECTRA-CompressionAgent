import numpy as np
from torch import nn
from src.Configuration.StaticConf import StaticConf


class ModelWithRows:
    """
    A utility class for analyzing and restructuring a neural network model by organizing its layers into
    logical "rows" based on their types.

    Purpose:
    - Extracts individual layers from a PyTorch model.
    - Groups layers into "rows" where each row consists of contiguous layers of the same type.
    - Provides a structured representation of the model to facilitate layer-wise pruning and resizing.

    Attributes:
        model (nn.Module): The input PyTorch model.
        all_layers (List[nn.Module]): A flat list of all layers extracted from the model.
        all_rows (np.ndarray): A structured representation of the model, grouping layers into rows.
        main_layer_types (List[Type[nn.Module]]): The primary types of layers that define new rows. Currently includes:
            - nn.Conv2d: Convolutional layers.
            - nn.Linear: Fully-connected layers.

    TODO:
        - Consider adding 'nn.BatchNorm2d', 'nn.MaxPool2d', and 'nn.AvgPool2d' to 'main_layer_types'.
          - Reason: BatchNorm and Pooling layers are typically linked to their corresponding Conv2D layers.
          - Decision: Should they be treated as independent layers or remain associated with Conv2D layers?
        - If BatchNorm and Pooling layers should remain in the same row as Conv2D, no modifications are needed.
          - Otherwise, logic needs to be updated to treat them as separate layers.
        * If other types are added to 'main_layer_types', the termination condition ('done') in NetworkEnv's step()
          must be edited accordingly.

    Methods:
        extract_layers_from_model(layer): Recursively extracts all layers from the model and stores them in 'all_layers'.
        is_to_split_row(curr_layer, curr_row): Determines whether a new row should be started based on layer type.
        split_layers_to_rows(): Groups layers into rows based on 'main_layer_types'.
    """

    model: nn.Module

    def __init__(self, model: nn.Module):
        """
        Initializes the ModelWithRows instance by extracting and structuring the model layers.

        Args:
            model (nn.Module): The neural network model to analyze.
        """

        self.model = model.to(StaticConf.get_instance().conf_values.device)
        self.all_layers = []
        # Define which layer types should trigger a new row.
        self.main_layer_types = [nn.Conv2d, nn.Linear]

        # Extract all layers and organize them into rows
        self.extract_layers_from_model(self.model)
        self.all_rows, self.row_to_main_layer = self.split_and_map_layers_to_rows()

    def extract_layers_from_model(self, layer: nn.Module):
        """
        Recursively extracts all layers from a given PyTorch model.

        Args:
            layer (nn.Module): A PyTorch module (could be the entire model or a submodule).

        Notes:
            - If a module has children (submodules), it will recursively explore them.
            - If a module does not have children, it is treated as an atomic layer.
        """
        for sub_layer in layer.children():
            if len(list(sub_layer.children())):
                self.extract_layers_from_model(sub_layer)  # Recursive call for nested layers
            else:
                self.all_layers.append(sub_layer)

    def is_to_split_row(self, curr_layer: nn.Module, curr_row: list) -> bool:
        """
        Determines whether the current layer should start a new row.

        A new row is started if the current layer is of a 'main_layer_type' (e.g., Conv2D, Linear) and the current row
        already has content (i.e., not the first layer of a new row).

        Args:
            curr_layer (nn.Module): The layer being evaluated.
            curr_row (list): The current row of layers.

        Returns:
            bool: True if the current layer should start a new row, otherwise False.
        """
        return isinstance(curr_layer, tuple(self.main_layer_types)) and len(curr_row) > 0

    def split_and_map_layers_to_rows(self) -> (np.ndarray, dict):
        """
        Groups extracted layers into rows, maintaining logical layer groupings.

        Returns:
            np.ndarray:                 A structured representation of the model, where each row contains contiguous layers.
            row_to_main_layer (dict):   Mapping row index to the main layer at the beginning of the row

        Notes:
            - Conv2D layers are currently grouped with BatchNorm and Pooling layers.
            - If BatchNorm and Pooling layers should be separate, update 'main_layer_types' accordingly.
        """

        row_idx = 0
        # The first layer is necessarily Conv2D or Linear, so it ought to start a row
        curr_row = [self.all_layers[0]]
        row_to_main_layer = {row_idx: 0}
        all_rows = []

        for layer_idx, curr_layer in enumerate(self.all_layers[1:]):
            if self.is_to_split_row(curr_layer, curr_row):
                all_rows.append(np.array(curr_row))  # Save completed row
                row_idx += 1
                row_to_main_layer[row_idx] = layer_idx + 1  # As the iteration begins with the second layer
                curr_row = []

            curr_row.append(curr_layer)

        if len(curr_row) > 0:
            all_rows.append(np.array(curr_row))  # Append the last row

        return np.array(all_rows, dtype=object), row_to_main_layer
