from typing import List, Dict
import torch
from transformers import BertTokenizer, BertModel
from NetworkFeatureExtraction.src.FeatureExtractors.TopologyFE import TopologyFE
from NetworkFeatureExtraction.src.FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE
from NetworkFeatureExtraction.src.FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE


class BERTInputModeler:
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        """
        Initializes BERTInputModeler for encoding CNN features into BERT-compatible format.

        Args:
            bert_model_name (str): Pretrained BERT model name.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    # TODO: Propagate env's layer_index-1 to properly SEP between layer to prune and net's global repr
    def encode_model_to_bert_input(self, model_with_rows, feature_maps) -> Dict[str, torch.Tensor]:
        """
        Extracts features from the CNN model and converts them into a BERT-compatible format.

        Args:
            model_with_rows: ModelWithRows instance representing the CNN structure.
            feature_maps: Dict[str, List[List[float]]]: CNN feature representations categorized by:
                - "Topology": Structural representation.
                - "Activations": Layer-wise activation statistics.
                - "Weights": Weight distribution across layers.

        Returns:
            Dict[str, torch.Tensor]: Tokenized BERT input.
        """
        # Flatten and concatenate feature sequences
        feature_sequence = []
        for key, features in feature_maps.items():
            feature_sequence.extend([item for sublist in features for item in sublist])  # Flatten nested lists

        # Convert to Tensor and normalize
        feature_tensor = torch.tensor(feature_sequence, dtype=torch.float32)
        feature_tensor = (feature_tensor - feature_tensor.mean()) / (feature_tensor.std() + 1e-5)

        # Convert tensor to string representation for BERT tokenization
        feature_str = " ".join(map(str, feature_tensor.tolist()))
        encoded_input = self.tokenizer(
            feature_str,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move tokenized data to the appropriate device
        return {key: value.to(self.device) for key, value in encoded_input.items()}

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes tokenized input through the BERT model to obtain embeddings.

        Args:
            tokens (Dict[str, torch.Tensor]): Tokenized BERT input.

        Returns:
            torch.Tensor: Embeddings from the BERT model (last_hidden_state).
        """
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.bert_model(**tokens)
        return outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
