from typing import Dict
import torch
from transformers import BertTokenizer, BertModel
import os

import src.utils as utils


# A Singleton
class BERTInputModeler:
    _instance = None

    def __new__(cls, bert_model_name: str = "bert-base-uncased"):
        if cls._instance is None:
            cls._instance = super(BERTInputModeler, cls).__new__(cls)
            cls._instance._initialize(bert_model_name)
        return cls._instance

    def _initialize(self, bert_model_name):
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        torch.cuda.set_device(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
        # Freeze BERT parameters, as BERT is used solely for feature extraction (static embeddings of the CNN structure and statistics)
        for param in self.bert_model.parameters():
            param.requires_grad = False

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
        utils.print_flush("Started encoding model to BERT input")
        # Flatten helper
        flatten = lambda nested: [item for sublist in nested for item in sublist]

        # Separate local (layer-specific) and global (full) features
        layer_features = []
        full_features = []

        for feature_type, all_layers in feature_maps.items():
            layer_features.extend(all_layers[curr_layer_idx])
            full_features.extend(flatten(all_layers))

        # Convert to string with true [SEP] token
        combined_input = f"{' '.join(map(str, layer_features))} [SEP] {' '.join(map(str, full_features))}"

        # Tokenize with BERT tokenizer
        with torch.no_grad():
            encoded_input = self.tokenizer(
                combined_input,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        utils.print_flush("Finished encoding model to BERT input")
        return {k: v.to(self.device) for k, v in encoded_input.items()}


    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes tokenized input through the BERT model to obtain embeddings.

        Args:
            tokens (Dict[str, torch.Tensor]): Tokenized BERT input.

        Returns:
            torch.Tensor: Embeddings from the BERT model (last_hidden_state).
        """
        tokens = {k: v.to(self.device).clone() for k, v in tokens.items()}
        with torch.no_grad():  # BERT is used solely for feature extraction (static embeddings of the CNN structure and statistics)
            outputs = self.bert_model(**tokens)
        return outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
