from typing import List, Dict

import torch

from ..FeatureExtractors import BaseFE
from ..FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from ..FeatureExtractors.TopologyFE import TopologyFE
from ..FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from src.BERTInputModeler import BERTInputModeler
import src.action_costs as action_costs
import src.utils as utils
from src.Configuration.StaticConf import StaticConf


class FeatureExtractor:
    def __init__(self, X, device):
        """
        Initializes the FeatureExtractor class for CNN model analysis.

        Args:
            X: DataLoader for activation-based feature extraction.
            device: The device (CPU/GPU) to run computations on.
        """
        self.device = device
        self.X = X
        self._input_shape = None  # resolved lazily from the loader, then reused

        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(),
            ActivationsStatisticsFE(X, device),
            WeightStatisticsFE(device)
        ]

        # Numeric state builder. Despite the historical name it does not load BERT unless
        # SPECTRA_STATE_ENCODER=bert (see BERTInputModeler).
        self.state_builder = BERTInputModeler()
        # Alias kept so older call sites / docs that still say "bert_input_modeler" work
        self.bert_input_modeler = self.state_builder

    @property
    def input_shape(self):
        """Per-sample input shape, needed to price actions in MACs."""
        if self._input_shape is None:
            self._input_shape = utils.get_input_shape(self.X)
        return self._input_shape

    def extract_features(self, model_with_rows, update_indices=None) -> Dict[str, List[List[float]]]:
        """
        Extracts CNN architecture features for the agent state encoder.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            update_indices (List[int], optional): Layer indices to update for Activations.

        Returns:
            Dict[str, List[List[float]]]: CNN feature representations categorized by:
                - "Topology": Structural representation.
                - "Activations": Layer-wise activation statistics.
                - "Weights": Weight distribution across layers.
        """
        feature_maps = {
            "Topology": self.all_feature_extractors[0].extract_feature_map(model_with_rows),
            "Activations": self.all_feature_extractors[1].extract_feature_map(model_with_rows, update_indices),
            "Weights": self.all_feature_extractors[2].extract_feature_map(model_with_rows)
        }
        return feature_maps

    def encode_to_bert_input(self, model_with_rows, curr_layer_idx, update_indices=None,
                             dependency_groups=None, param_ratio=None):
        """
        Converts the extracted CNN features into the agent's state representation.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            curr_layer_idx (int):   Index of layer to prune, so the encoder can distinguish the
                                    layer under consideration from the rest of the network.
            update_indices (List[int], optional): Layer indices to update for Activations.
            dependency_groups (list, optional):   Channel groups already computed for this
                                    model, reused to avoid a second symbolic trace.
            param_ratio (float, optional): Remaining-parameter fraction for SPECTRA_BUDGET_IN_STATE.

        Returns:
            Dict[str, torch.Tensor]: The agent state.
        """
        feature_maps = self.extract_features(model_with_rows, update_indices)
        costs = self._action_costs(model_with_rows, curr_layer_idx, dependency_groups)
        state = self.state_builder.encode_model_to_bert_input(
            model_with_rows, feature_maps, curr_layer_idx,
            dependency_groups=dependency_groups, action_costs=costs,
            param_ratio=param_ratio)
        return state

    def _action_costs(self, model_with_rows, curr_layer_idx, dependency_groups):
        """
        Fraction of the network's parameters and MACs each compression rate would remove.

        Failures degrade to zeros rather than aborting the episode: the cost features are an
        enrichment of the state, not a precondition for acting.

        Cost note: ``estimate_action_costs`` runs a MAC probe forward. Prefer passing
        precomputed ``dependency_groups``; re-tracing every step is the other expensive piece
        and is already owned by NetworkEnv.
        """
        conf = StaticConf.get_instance().conf_values
        rates = [conf.compression_rates_dict[key] for key in sorted(conf.compression_rates_dict)]
        target = model_with_rows.all_layers[min(curr_layer_idx, len(model_with_rows.all_layers) - 1)]

        try:
            return action_costs.estimate_action_costs(
                model_with_rows.model, target, rates, self.input_shape,
                groups=dependency_groups, device=self.device)
        except Exception as error:
            utils.print_flush(f"Action-cost features unavailable ({error}); falling back to zeros")
            costs = torch.zeros(len(rates), action_costs.ACTION_FEATURE_DIM, device=self.device)
            costs[:, 0] = torch.tensor(rates, device=self.device)
            return costs
