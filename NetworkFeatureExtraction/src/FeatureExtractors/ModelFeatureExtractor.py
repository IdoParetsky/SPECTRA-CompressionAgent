from typing import List, Dict
from ..FeatureExtractors import BaseFE
from ..FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from ..FeatureExtractors.TopologyFE import TopologyFE
from ..FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from ..ModelWithRows import ModelWithRows
from src.BERTInputModeler import BERTInputModeler


class FeatureExtractor:
    def __init__(self, model, X, device):
        """
        Initializes the FeatureExtractor class for CNN model analysis.

        Args:
            model: The neural network model to extract features from.
            X: DataLoader for activation-based feature extraction.
            device: The device (CPU/GPU) to run computations on.
        """
        self.device = device
        self.model_with_rows = ModelWithRows(model)
        self.X = X

        # Initialize feature extractors
        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(self.model_with_rows),
            ActivationsStatisticsFE(self.model_with_rows, X, device),
            WeightStatisticsFE(self.model_with_rows)
        ]

        # Initialize BERT modeler
        self.bert_input_modeler = BERTInputModeler()

    def extract_features(self) -> Dict[str, List[List[float]]]:
        """
        Extracts CNN architecture features for BERT encoding.

        Returns:
            Dict[str, List[List[float]]]: CNN feature representations categorized by:
                - "Topology": Structural representation.
                - "Activations": Layer-wise activation statistics.
                - "Weights": Weight distribution across layers.
        """
        feature_maps = {
            "Topology": self.all_feature_extractors[0].extract_feature_map(),
            "Activations": self.all_feature_extractors[1].extract_feature_map(),
            "Weights": self.all_feature_extractors[2].extract_feature_map()
        }
        return feature_maps

    def encode_to_bert_input(self, curr_layer_idx):
        """
        Converts the extracted CNN features into BERT-compatible input format.

        Args:
            curr_layer_idx (int):   Index of layer to prune, so BERTInputModeler is able to distinguish between local
                                    (current layer to be pruned) and global context (entire network) via a [SEP] token.

        Returns:
            Dict[str, torch.Tensor]: Tokenized CNN architecture representation for BERT.
        """
        feature_maps = self.extract_features()
        return self.bert_input_modeler.encode_model_to_bert_input(self.model_with_rows, feature_maps, curr_layer_idx)
