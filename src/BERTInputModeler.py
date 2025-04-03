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
    def encode_model_to_bert_input(self, model_with_rows, feature_maps, curr_layer_idx) -> Dict[str, torch.Tensor]:
        """
        Extracts features from the CNN model and converts them into a BERT-compatible format.

        Args:
            model_with_rows:        ModelWithRows instance representing the CNN structure.
            feature_maps:           Dict[str, List[List[float]]]: CNN feature representations categorized by:
                                        - "Topology": Structural representation.
                                        - "Activations": Layer-wise activation statistics.
                                        - "Weights": Weight distribution across layers.
            curr_layer_idx (int):   Index of layer to prune, so BERTInputModeler is able to distinguish between local
                                    (current layer to be pruned) and global context (entire network) via a [SEP] token.

        Returns:
            Dict[str, torch.Tensor]: Tokenized BERT input.
        """
        # Flatten helper
        flatten = lambda nested: [item for sublist in nested for item in sublist]

        # Separate local (layer-specific) and global (full) features
        layer_features = []
        full_features = []

        for feature_type, all_layers in feature_maps.items():
            layer_features.extend(all_layers[curr_layer_idx])
            full_features.extend(flatten(all_layers))

        # Convert to string with true [SEP] token
        layer_str = " ".join(map(str, layer_features))
        full_str = " ".join(map(str, full_features))
        combined_input = f"{layer_str} [SEP] {full_str}"

        # Tokenize with BERT tokenizer
        encoded_input = self.tokenizer(
            combined_input,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in encoded_input.items()}

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
