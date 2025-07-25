from typing import Dict


class ConfigurationValues:

    def __init__(self, device, test_name, input_dict, compression_rates_dict, runtime_limit, num_epochs,
                 train_compressed_layer_only, allowed_acc_reduction, discount_factor, learning_rate,
                 rollout_limit, passes, prune, seed, n_splits, train_split, val_split, database_dict, dataloaders_dict,
                 actor_checkpoint_path, critic_checkpoint_path, save_pruned_checkpoints, test_ts) -> None:
        self.test_name = test_name,
        self.input_dict = input_dict
        self.compression_rates_dict = compression_rates_dict
        self.device = device
        self.num_epochs = num_epochs
        self.num_actions = len(compression_rates_dict)
        self.train_compressed_layer_only = train_compressed_layer_only
        self.allowed_acc_reduction = allowed_acc_reduction
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.rollout_limit = rollout_limit
        self.passes = passes
        self.prune = prune
        self.seed = seed
        self.n_splits = n_splits
        self.train_split = train_split
        self.val_split = val_split
        self.runtime_limit = runtime_limit
        self.database_dict = database_dict
        self. dataloaders_dict = dataloaders_dict
        self.actor_checkpoint_path = actor_checkpoint_path
        self.critic_checkpoint_path = critic_checkpoint_path
        self.save_pruned_checkpoints = save_pruned_checkpoints
        self.test_ts = test_ts
