from decimal import Decimal

import numpy as np


def pad_with_columns(layer_statistics, max_layer_size):
    pad_size = max_layer_size - len(layer_statistics)
    return np.pad(layer_statistics, (0, pad_size))


def pad_with_rows(f_map, rows_target):
    return np.pad(f_map, ((0, rows_target - f_map.shape[0]), (0, 0)))


def get_scaler_exponent(matrix):
    abs_matrix = np.abs(matrix)
    min_val = np.where(abs_matrix == 0, 0, abs_matrix).min()
    val_as_string = str(min_val)
    min_val_decimal = Decimal(val_as_string)
    adder = 0
    if "." in val_as_string:
        adder = len(val_as_string.split(".")[1].split('e')[0])

    if type(min_val_decimal.as_tuple().exponent) != int:
        print(adder)
        print(abs_matrix)
        print(val_as_string)
        print(min_val_decimal)
        print(min_val_decimal.as_tuple().exponent)

    return min_val_decimal.as_tuple().exponent + adder
