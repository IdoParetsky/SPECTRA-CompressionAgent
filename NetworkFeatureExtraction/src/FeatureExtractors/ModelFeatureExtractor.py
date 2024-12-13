from typing import List

from ..FeatureExtractors import BaseFE
from ..FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from ..FeatureExtractors.TopologyFE import TopologyFE
from ..FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from ..ModelWithRows import ModelWithRows


class FeatureExtractor:
    def __init__(self, model, X, device):
        """
        Initializes the FeatureExtractor class for a given model and input data.
        Args:
            model: The neural network model to extract features from.
            X: Input data for activation-based feature extraction.
            device: The device (CPU/GPU) to run computations on.
        """
        self.device = device

        self.model_with_rows = ModelWithRows(model)

        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(self.model_with_rows),
            ActivationsStatisticsFE(self.model_with_rows, X, self.device),
            WeightStatisticsFE(self.model_with_rows)
        ]

    def extract_features(self, layer_index):
        """
        Extracts features for a specific layer in the model.
        Args:
            layer_index: Index of the layer to extract features from.
        Returns:
            A list of feature maps extracted by all feature extractors.
        """
        a = [curr_fe.extract_feature_map(layer_index) for curr_fe in self.all_feature_extractors]
        return a
