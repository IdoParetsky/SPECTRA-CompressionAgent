from typing import Dict


class ConfigurationValues():
    device: str
    num_epoch: int
    num_actions: int
    is_learn_new_layers_only: bool
    total_allowed_accuracy_reduction: int
    compression_rates_dict: Dict[int, float]
    increase_loops_from_1_to_4: bool
    prune: bool

    def __init__(self, device, compression_rates_dict, MAX_TIME_TO_RUN=60 * 60 * 24 * 1.5, num_epoch=100,
                 is_learn_new_layers_only=False, total_allowed_accuracy_reduction=1, increase_loops_from_1_to_4=False,
                 prune=False) -> None:
        self.compression_rates_dict = compression_rates_dict
        self.device = device
        self.num_epoch = num_epoch
        self.num_actions = len(compression_rates_dict)
        self.is_learn_new_layers_only = is_learn_new_layers_only
        self.total_allowed_accuracy_reduction = total_allowed_accuracy_reduction
        self.increase_loops_from_1_to_4 = increase_loops_from_1_to_4
        self.prune = prune
        self.MAX_TIME_TO_RUN = MAX_TIME_TO_RUN
