from typing import List

from ..FeatureExtractors import BaseFE
from ..FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from ..FeatureExtractors.TopologyFE import TopologyFE
from ..FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from ..ModelWithRows import ModelWithRows
from src.BERTInputModeler import BERTInputModeler


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
        self.X = X

        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(self.model_with_rows),
            ActivationsStatisticsFE(self.model_with_rows, X, self.device),
            WeightStatisticsFE(self.model_with_rows)
        ]

        # Initialize BERT modeler
        self.bert_input_modeler = BERTInputModeler()

    def extract_features(self, layer_index):
        """
        Extracts features for a specific layer in the model.
        Args:
            layer_index: Index of the layer to extract features from.
        Returns:
            A dictionary of feature maps for BERT encoding.
        """
        feature_maps = {
            "Topology": self.all_feature_extractors[0].extract_feature_map(layer_index),
            "Activations": self.all_feature_extractors[1].extract_feature_map(layer_index),
            "Weights": self.all_feature_extractors[2].extract_feature_map(layer_index)
        }
        return feature_maps

    def encode_to_bert_input(self, layer_index):
        """
        Converts the extracted features for a specific layer into BERT input.
        Args:
            layer_index: Index of the layer to encode.
        Returns:
            BERT-compatible inputs.
        """
        feature_maps = self.extract_features(layer_index)
        return self.bert_input_modeler.encode_model_to_bert_input(self.model_with_rows, feature_maps)
