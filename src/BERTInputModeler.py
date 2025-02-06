from typing import List, Dict
import torch
from transformers import BertTokenizer, BertModel
from NetworkFeatureExtraction.src.FeatureExtractors.TopologyFE import TopologyFE
from NetworkFeatureExtraction.src.FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from NetworkFeatureExtraction.src.FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE


class BERTInputModeler:
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        """
        Initialize the BERTInputModeler with a specified BERT model.
        Args:
            bert_model_name (str): The name of the pretrained BERT model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def encode_model_to_bert_input(self, model_with_rows, feature_maps: Dict[str, List[float]]) -> Dict[str, torch.Tensor]:
        """
        Converts the extracted model features into BERT-compatible input.
        Args:
            model_with_rows: ModelWithRows instance containing the layer-by-layer model structure.
            feature_maps: A dictionary containing features extracted from Topology, Activations, and Weights.
        Returns:
            A dictionary containing tokenized BERT inputs: input_ids, attention_mask, token_type_ids.
        """
        # Flatten and concatenate features into a single sequence
        feature_sequence = []
        for key, features in feature_maps.items():
            feature_sequence.extend(features)

        # Normalize features (optional, for scaling)
        feature_sequence = torch.tensor(feature_sequence).float()
        feature_sequence = (feature_sequence - feature_sequence.mean()) / (feature_sequence.std() + 1e-5)

        # Convert features into a string representation for BERT tokenization
        feature_str = " ".join(map(str, feature_sequence.tolist()))
        encoded_input = self.tokenizer(
            feature_str,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move tokenized data to the appropriate device
        return {
            key: value.to(self.device) for key, value in encoded_input.items()
        }


def extract_topology_features(model_with_rows) -> List[float]:
    """
    Extracts topology features for BERT encoding.
    Args:
        model_with_rows: ModelWithRows instance containing the model structure.
    Returns:
        List[float]: Extracted topology features.
    """
    return TopologyFE(model_with_rows).extract_features_all_layers()


def extract_activation_statistics(model_with_rows, input_data, device) -> List[float]:
    """
    Extracts activation statistics for BERT encoding.
    Args:
        model_with_rows: ModelWithRows instance containing the model structure.
        input_data: Input data to compute activations.
        device: Device to run computations.
    Returns:
        List[float]: Extracted activation statistics.
    """
    return ActivationsStatisticsFE(model_with_rows, input_data, device).extract_features_all_layers()


def extract_weight_statistics(model_with_rows) -> List[float]:
    """
    Extracts weight statistics for BERT encoding.
    Args:
        model_with_rows: ModelWithRows instance containing the model structure.
    Returns:
        List[float]: Extracted weight statistics.
    """
    return WeightStatisticsFE(model_with_rows).extract_features_all_layers()
