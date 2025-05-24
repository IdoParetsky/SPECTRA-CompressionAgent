from typing import List, Dict
from ..FeatureExtractors import BaseFE
from ..FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from ..FeatureExtractors.TopologyFE import TopologyFE
from ..FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
from ..ModelWithRows import ModelWithRows
from src.BERTInputModeler import BERTInputModeler


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

        # Initialize feature extractors
        self.all_feature_extractors: List[BaseFE] = [
            TopologyFE(),
            ActivationsStatisticsFE(X, device),
            WeightStatisticsFE(device)
        ]

        # Initialize BERT modeler
        self.bert_input_modeler = BERTInputModeler()

    def extract_features(self, model_with_rows, update_indices=None) -> Dict[str, List[List[float]]]:
        """
        Extracts CNN architecture features for BERT encoding.

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

    def encode_to_bert_input(self, model_with_rows, curr_layer_idx, update_indices=None):
        """
        Converts the extracted CNN features into BERT-compatible input format.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            curr_layer_idx (int):   Index of layer to prune, so BERTInputModeler is able to distinguish between local
                                    (current layer to be pruned) and global context (entire network) via a [SEP] token.
            update_indices (List[int], optional): Layer indices to update for Activations.

        Returns:
            Dict[str, torch.Tensor]: Tokenized CNN architecture representation for BERT.
        """
        feature_maps = self.extract_features(model_with_rows, update_indices)
        return self.bert_input_modeler.encode_model_to_bert_input(model_with_rows, feature_maps, curr_layer_idx)
