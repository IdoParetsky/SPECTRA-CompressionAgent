import torch
import numpy as np
from torch import nn


class ModelWithRows():
    model: nn.Module

    def __init__(self, model):
        self.model = model
        self.all_layers = []
        # TODO: Consider adding nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d to discern them from their correspondent Conv2D layer
        self.main_layer_types = [torch.nn.Conv2d, torch.nn.modules.linear.Linear]

        self.extract_layers_from_model(self.model)
        self.all_rows = self.split_layers_to_rows()

    # TODO: If pooling or BatchNorm layers should remain associated with their corresponding Conv2D layers,
    #  this method should not be modified.
    #  However, if they need to form separate rows, add logic to check for these types explicitly.
    def extract_layers_from_model(self, layer):
        for layer in layer.children():
            if len(list(layer.children())):
                self.extract_layers_from_model(layer)
            else:
                self.all_layers.append(layer)

    # TODO: Decide if BatchNorm & Pooling layers are treated as part of their respective Conv2D row.
    #  else, they should be added to self.main_layer_types.
    def is_to_split_row(self, curr_layer, curr_row):
        """
        Determine whether to split the current row based on the layer type.
        Rows should contain only layers of the same type (e.g., Conv2D or Linear).
        """
        return type(curr_layer) in self.main_layer_types and \
            len(curr_row) > 0 and \
            any(type(l) in self.main_layer_types for l in curr_row)

    def split_layers_to_rows(self):
        all_rows = []
        curr_row = []

        for curr_layer in self.all_layers:
            # TODO: Conv2D layers are currently grouped with their BatchNorm and Pooling layers
            if self.is_to_split_row(curr_layer, curr_row):
                all_rows.append(np.array(curr_row))
                curr_row = []

            curr_row.append(curr_layer)

        if len(curr_row) > 0:
            all_rows.append(np.array(curr_row))

        return np.array(all_rows, dtype=object)
