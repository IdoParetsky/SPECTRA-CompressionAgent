import copy
from typing import Dict, Type, List

import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune

from NetworkFeatureExtration.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from NetworkFeatureExtration.src.ModelClasses.LoadedModel import LoadedModel, MissionTypes
from NetworkFeatureExtration.src.ModelWithRows import ModelWithRows
from NetworkFeatureExtration.src.main import load_model_and_data
from src.Configuration.StaticConf import StaticConf
from src.CrossValidationObject import CrossValidationObject
from src.ModelHandlers.BasicHandler import BasicHandler
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.ModelHandlers.RegressionHandler import RegressionHandler
import pandas as pd

from src.utils import print_flush

flatten_list = lambda l: [item for sublist in l for item in sublist]


class NetworkEnv:
    loaded_model: LoadedModel
    current_model: nn.Module
    actions_history: List[float]
    max_samples_for_fe: int

    def __init__(self, networks_path, increase_loops_from_1_to_4=False, max_samples_for_fe=3000):
        # self.networks_path = [networks_path[0]]
        self.all_networks = []
        self.increase_loops_from_1_to_4 = increase_loops_from_1_to_4
        self.max_samples_for_fe = max_samples_for_fe

        self.networks_path = networks_path
        for group in networks_path:
            x_path = group[0]
            nets = group[1]

            for n in nets:
                self.all_networks.append((x_path, n))

        self.curr_net_index = -1
        self.net_order = list(range(0, len(self.all_networks)))
        np.random.shuffle(self.net_order)

        # A dictionary mapping MissionTypes to corresponding Handler classes
        self.handler_by_mission_type: Dict[MissionTypes, Type[BasicHandler]] = {
            MissionTypes.Regression: RegressionHandler,
            MissionTypes.Classification: ClassificationHandler
        }

    def create_train_test_splits(self):
        X_train, X_val, Y_train, Y_val = train_test_split(self.X_train_data.values, self.Y_train_data.values,
                                                          test_size=0.2, random_state=1)
        self.cross_validation_obj = CrossValidationObject(X_train, X_val, self.X_test_data.values,
                                                          Y_train, Y_val, self.Y_test_data.values)
        # *train_test_split(self.X_train_data.values, self.Y_train_data.values, test_size=0.2, random_state=0))

    def reset(self):
        """
        Reset environment with a random network and its train data
        :return: state includes the network and the train data
        """
        self.layer_index = 1
        self.actions_history = []
        # selected_net_group_index = np.random.choice(len(self.networks_path), 1)[0]
        # selected_net_group = self.networks_path[selected_net_group_index]
        # x_train_path = selected_net_group[0]
        # selected_net_path = np.random.choice(selected_net_group[1], 1)[0]

        # print("Selected net: ", x_train_path)

        # selected_net_path = selected_net_group[1][0]
        # selected_net_path = 0
        # print("Selected net: ", selected_net_path)
        self.curr_net_index += 1
        self.curr_net_index = self.curr_net_index % len(self.net_order)
        curr_group_index = self.net_order[self.curr_net_index]
        x_train_path, selected_net_path = self.all_networks[curr_group_index]
        print_flush(selected_net_path)

        x_test_path = str.replace(x_train_path, 'X_train', 'X_val')
        y_train_path = str.replace(x_train_path, 'X_train', 'Y_train')
        y_test_path = str.replace(x_train_path, 'X_train', 'Y_val')

        device = StaticConf.getInstance().conf_values.device
        self.loaded_model, self.X_train_data, self.Y_train_data = load_model_and_data(selected_net_path,
                                                                                            x_train_path, y_train_path,
                                                                                            device)
        self.X_test_data, self.Y_test_data = pd.read_csv(x_test_path), pd.read_csv(y_test_path)

        self.current_model = self.loaded_model.model

        self.create_fe(device)
        fm = self.feature_extractor.extract_features(self.layer_index - 1)
        self.create_train_test_splits()

        learning_handler_original_model = self.create_learning_handler(self.current_model)
        self.original_acc = learning_handler_original_model.evaluate_model(validation=True)

        return fm

    def create_fe(self, device):  # Feature Extraction
        number_of_sampled = min(self.max_samples_for_fe, max(self.max_samples_for_fe, len(self.X_train_data) // 10))
        test_size = number_of_sampled / len(self.X_train_data)
        if test_size >= 1:
            data_for_feature_extraction = self.X_train_data._values
        else:
            _, data_for_feature_extraction = train_test_split(self.X_train_data._values, test_size=test_size,
                                                              stratify=self.Y_train_data)
        self.feature_extractor = FeatureExtractor(self.loaded_model.model, data_for_feature_extraction, device)

    def step(self, compression_rate, is_to_train=True):
        """
        Compress the network according to the action (compression rate) and advance to the next state.

        Args:
            compression_rate (float):   The chosen compression_rate
            is_to_train (bool):         Whether to train the new model after compression.

        Returns:
            Tuple: A tuple containing the next state (feature maps), the reward, and a flag indicating if the episode is done.
        """

        print_flush(f'step {self.layer_index}')
        # Create new layer with smaller size & Create new model with the new layer
        if compression_rate == 1:
            new_model_with_rows = ModelWithRows(self.current_model)
            learning_handler_new_model = self.create_learning_handler(self.current_model)
        else:
            if StaticConf.getInstance().conf_values.prune:
                new_model = self.create_new_model_pruned(compression_rate)
            else:
                new_model = self.create_new_model_with_new_weights(compression_rate)

            new_model_with_rows = ModelWithRows(new_model)
            learning_handler_new_model = self.create_learning_handler(new_model)

            if StaticConf.getInstance().conf_values.is_learn_new_layers_only:
                parameters_to_freeze_ids = self.build_parameters_to_freeze(new_model_with_rows)
                learning_handler_new_model.freeze_layers(parameters_to_freeze_ids)
            else:
                learning_handler_new_model.unfreeze_all_layers()

            if is_to_train:
                learning_handler_new_model.train_model()

        self.current_model.eval()
        learning_handler_new_model.model.eval()

        # learning_handler_prev_model = self.create_learning_handler(self.current_model)
        # prev_acc = learning_handler_prev_model.evaluate_model()pn
        new_acc = learning_handler_new_model.evaluate_model(validation=True)

        # compute reward
        reward = self.compute_reward3(new_acc, self.original_acc, compression_rate)

        self.layer_index += 1
        learning_handler_new_model.unfreeze_all_layers()
        self.current_model = learning_handler_new_model.model

        # get FM for the new model and the next layer.
        device = StaticConf.getInstance().conf_values.device

        self.create_fe(device)
        fm = self.feature_extractor.extract_features(self.layer_index - 1)

        # Compute done
        number_of_layers = len(self.feature_extractor.model_with_rows.all_rows) - 1
        if not self.increase_loops_from_1_to_4:
            done = self.layer_index >= number_of_layers
        else:
            self.actions_history.append(compression_rate)
            self.layer_index = max(1, self.layer_index % (number_of_layers + 1))
            done = self.is_done_more_than_one_loop(number_of_layers)
        return fm, reward, done

    def is_done_more_than_one_loop(self, number_of_layers, max_iters=4):
        """
        Checks if all last updates are 0 OR if the agent went through the whole layers max_iterations times
        :param number_of_layers:
        :param max_iters:
        :return:
        """
        return len(self.actions_history) == max_iters * number_of_layers

    def compute_reward1(self, curr_model, new_model, new_acc, prev_acc, mission_type):
        """
        compute reward with by multiplying the parameter part with 0.4
        """
        current_model_parameter_num = self.calc_num_parameters(curr_model)
        new_model_parameter_num = self.calc_num_parameters(new_model)
        C = 1 - (new_model_parameter_num / current_model_parameter_num)
        reward = 0.4 * (C * (2 - C))

        if (mission_type is MissionTypes.Classification):
            reward *= new_acc / prev_acc
        else:
            reward *= prev_acc / new_acc
        return reward

    def compute_reward2(self, curr_model, new_model, new_acc, prev_acc, mission_type):
        current_model_parameter_num = self.calc_num_parameters(curr_model)
        new_model_parameter_num = self.calc_num_parameters(new_model)
        C = 1 - (new_model_parameter_num / current_model_parameter_num)
        reward = 10 * (C * (2 - C))

        if (mission_type is MissionTypes.Classification):
            reward *= 30 * new_acc / prev_acc

            if prev_acc - new_acc > 0.2:
                reward = -10
        else:
            reward *= 30 * prev_acc / new_acc
        return reward

    def compute_reward3(self, new_acc, prev_acc, compression_rate):
        total_allowed_accuracy_reduction = StaticConf.getInstance().conf_values.total_allowed_accuracy_reduction
        layer_reduction_size = (1 - compression_rate) * 100

        delta_acc = (new_acc - prev_acc) * 100

        if delta_acc < -total_allowed_accuracy_reduction:
            reward = -layer_reduction_size ** 3
        elif delta_acc > 0:
            reward = layer_reduction_size ** 3
        else:
            reward = layer_reduction_size

        return reward

    def calc_num_parameters(self, model, is_prune=False):
        if not is_prune:
            return sum(p.numel() for p in model.parameters())
        else:
            pruned_params = np.sum(list(
                map(lambda x: (x.weight_mask == 0).cpu().detach().sum(),
                    filter(lambda x: hasattr(x, 'weight_mask'), model.modules()))))
            original_params = self.calc_num_parameters(model)
            return original_params - pruned_params

    def build_parameters_to_freeze(self, new_model_with_rows):
        layers_to_freeze = np.concatenate(new_model_with_rows.all_rows[self.layer_index - 1:self.layer_index + 1])
        parameters_to_freeze = flatten_list(list(map(lambda l: list(l.parameters()), layers_to_freeze)))
        parameters_to_freeze_ids = list(map(lambda l: id(l), parameters_to_freeze))
        return parameters_to_freeze_ids

    def create_learning_handler(self, new_model) -> BasicHandler:
        return self.handler_by_mission_type[self.loaded_model.mission_type](new_model,
                                                                            self.loaded_model.loss,
                                                                            self.loaded_model.optimizer,
                                                                            self.cross_validation_obj)

    def get_linear_layer(self, row):
        for l in row:
            if type(l) is nn.Linear:
                return l

    def get_batchnorm_layer(self, row):
        for l in row:
            if type(l) is nn.BatchNorm1d:
                return l

    def create_new_model_with_new_weights(self, compression_rate):
        model_with_rows = ModelWithRows(self.current_model)

        prev_layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index - 1])
        # Commented  out as it is unused
        # bn_to_change = None if self.layer_index is 1 else \
        #     self.get_batchnorm_layer(model_with_rows.all_rows[self.layer_index - 1])

        layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index])

        new_model_layers = []
        new_size = int(np.ceil(compression_rate * layer_to_change.in_features))
        last_linear_layer = None

        for l in model_with_rows.all_layers:
            if (l is prev_layer_to_change):
                new_model_layers.append(nn.Linear(l.in_features, new_size))
            elif l is layer_to_change:
                new_model_layers.append(nn.Linear(new_size, l.out_features))
            elif self.is_to_change_bn_layer(l, last_linear_layer):
                new_model_layers.append(nn.BatchNorm1d(last_linear_layer.out_features))
            else:
                # new_model_layers.append(l)
                new_model_layers.append(copy.deepcopy(l))

            if type(l) is nn.Linear:
                last_linear_layer = new_model_layers[-1]

        new_model = nn.Sequential(*new_model_layers)
        return new_model

    def deep_copy_model(self, model):
        new_model_layers = []
        model_with_rows = ModelWithRows(model)

        for l in model_with_rows.all_layers:
            new_model_layers.append(copy.deepcopy(l))

        return nn.Sequential(*new_model_layers)

    def is_to_change_bn_layer(self, curr_layer, last_linear_layer):
        return type(curr_layer) is nn.BatchNorm1d and \
               last_linear_layer is not None and \
               curr_layer.num_features is not last_linear_layer.out_features

    def create_new_model_pruned(self, compression_rate):
        # model_copy = self.deep_copy_model(self.current_model)  # get a new instance
        # model_copy.load_state_dict(self.current_model.state_dict())  # copy weights and stuff

        model_with_rows = ModelWithRows(self.current_model)
        layer_to_change = self.get_linear_layer(model_with_rows.all_rows[self.layer_index - 1])

        prune.ln_structured(layer_to_change, 'weight', 1 - compression_rate, n=1, dim=0)
        return self.current_model
