"""
A trainable Transformer encoder over CNN layer tokens.

Motivation (see docs/BERT_INPUT_CRITIQUE.md / docs/THESIS_INSTRUCTOR_BRIEFING.md): the
"Extending BERT Input Mechanisms" document frames the state encoder as *representation
learning*, but a frozen ``bert-base-uncased`` learns nothing — only the two-layer policy
head adapts. The frozen encoder also brings ~110M parameters, a hard 512-position limit,
and weights trained on English word-pieces rather than on network statistics.

This encoder keeps the document's structural ideas — one token per layer, an explicit
marker for the layer under consideration (entity marker), and a structural signal that
reflects *channel coupling* — while being small enough to train end to end with the RL
objective (roughly 2–3M parameters at the default width).

Pooling recommendation (critique §8)
------------------------------------
A plain mean over the sequence treats the target layer as one vote among dozens. The action
is *about* that layer, so we use target-aware pooling:

    0.5 * mean(all layer tokens) + 0.5 * encoded[target]

Action-cost tokens (if present) participate in the mean so their information is not dropped,
but they are not substituted for the target term. This keeps ``output_dim == d_model`` and
is a natural baseline before any learned attention-pooling head.
"""

import math
from typing import Dict

import torch
from torch import nn

from src.action_costs import ACTION_FEATURE_DIM

# Topology feature 0 encodes the layer family: 0 unknown, 1 Linear, 2 Conv2d, 3 BatchNorm,
# 4 activation, 5 Dropout, 6 Flatten, 7 pooling (see TopologyFE)
NUM_LAYER_TYPES = 8

# Blend weight for the target layer in the pooled state vector
TARGET_POOL_WEIGHT = 0.5


def sinusoidal_positions(length: int, dim: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    angles = position * freqs.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0:2 * half:2] = torch.sin(angles)
    encoding[:, 1:2 * half:2] = torch.cos(angles)
    return encoding


class SpectraStateEncoder(nn.Module):
    """
    Encode a CNN into a fixed-size state vector for the DRL agent.

    Args:
        feature_dim (int): Width of a per-layer feature token (incl. action-cost slots).
        d_model (int):     Transformer width.
        nhead (int):       Attention heads.
        num_layers (int):  Encoder depth.
        dropout (float):   Dropout inside the encoder.
    """

    def __init__(self, feature_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.output_dim = d_model
        self.feature_dim = feature_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.type_embedding = nn.Embedding(NUM_LAYER_TYPES, d_model)

        # Entity marker for the layer the agent is about to compress. Replaces the document's
        # dual local/global segments (critique §3): same distinction, half the sequence length,
        # no dependence on BERT's segment embedding table.
        self.target_marker = nn.Parameter(torch.zeros(d_model))

        # Optional dedicated tokens for each compression rate's cost vector. Costs are *also*
        # concatenated onto the target layer's feature vector (critique §7); these tokens let
        # attention compare prices as first-class sequence elements.
        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_marker = nn.Parameter(torch.zeros(d_model))

        # Learned additive attention bias between layers that share a channel-coupling id
        # (from src/channel_groups.py). This is the Graphormer-style mechanism recommended
        # in critique §5 — connectivity enters the attention logits, not the input PE sum.
        self.block_affinity = nn.Parameter(torch.zeros(1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        # Nested tensors are incompatible with pre-norm layers and only help padded batches;
        # SPECTRA encodes one network at a time, so the fast path is irrelevant here
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state: Mapping from FeatureExtractor.encode_to_bert_input(), containing
                   ``layer_features`` (L, feature_dim), ``layer_types`` (L,),
                   ``coupling_ids`` (or legacy ``block_ids``) (L,) and ``target_index``.

        Returns:
            torch.Tensor: (1, d_model) state embedding.
        """
        features = state["layer_features"]
        device = features.device
        num_layers = features.size(0)

        types = state["layer_types"].clamp(0, NUM_LAYER_TYPES - 1)
        tokens = self.input_proj(features) + self.type_embedding(types)
        tokens = tokens + sinusoidal_positions(num_layers, tokens.size(1), device)

        target_index = int(state["target_index"])
        if 0 <= target_index < num_layers:
            marker = torch.zeros_like(tokens)
            marker[target_index] = self.target_marker
            tokens = tokens + marker

        # Prefer exact coupling ids; fall back to the legacy key name
        coupling_ids = state.get("coupling_ids", state["block_ids"])

        action_costs = state.get("action_costs")
        num_layer_tokens = num_layers
        if action_costs is not None and action_costs.numel():
            action_tokens = self.action_proj(action_costs.to(device)) + self.action_marker
            tokens = torch.cat([tokens, action_tokens], dim=0)
            target_coupling = (coupling_ids[target_index]
                               if 0 <= target_index < num_layers
                               else coupling_ids.new_zeros(()))
            coupling_ids = torch.cat([coupling_ids, target_coupling.expand(action_tokens.size(0))])

        length = tokens.size(0)
        same_group = coupling_ids.unsqueeze(0) == coupling_ids.unsqueeze(1)
        attention_bias = torch.zeros(length, length, device=device)
        attention_bias = attention_bias.masked_fill(same_group, 1.0) * self.block_affinity

        encoded = self.encoder(tokens.unsqueeze(0), mask=attention_bias)  # (1, L', d)
        encoded = self.output_norm(encoded)

        # Target-aware pooling (see module docstring)
        sequence_mean = encoded.mean(dim=1)  # (1, d)
        if 0 <= target_index < num_layer_tokens:
            target_vec = encoded[:, target_index, :]
            pooled = ((1.0 - TARGET_POOL_WEIGHT) * sequence_mean
                      + TARGET_POOL_WEIGHT * target_vec)
        else:
            pooled = sequence_mean
        return pooled
