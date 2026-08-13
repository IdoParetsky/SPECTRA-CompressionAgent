"""
Trainable encoders over CNN layer tokens.

The default is a small Transformer with a Graphormer-style coupling bias
(``SPECTRA_STATE_ENCODER=transformer``). That is how the *generic* agent *reads*
architecture; it is not a per-net policy. Sibling variants:

* ``transformer_wide`` — same inductive bias, more capacity (6 × 512)
* ``set`` — same tokens, no cross-layer attention (architecture-agnostic *read*)
* ``bert`` / ``legacy`` — handled in ``Agent``, not here

Pooling (critique §8): ``0.5 * mean(sequence) + 0.5 * encoded[target]``.
"""

import math
from typing import Dict, Tuple

import torch
from torch import nn

from src.action_costs import ACTION_FEATURE_DIM

# Topology feature 0 encodes the layer family: 0 unknown, 1 Linear, 2 Conv2d, 3 BatchNorm,
# 4 activation, 5 Dropout, 6 Flatten, 7 pooling (see TopologyFE)
NUM_LAYER_TYPES = 8

# Blend weight for the target layer in the pooled state vector
TARGET_POOL_WEIGHT = 0.5

TRAINABLE_ENCODER_KINDS = ("transformer", "transformer_wide", "transformer_deep", "set", "mlp")


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


class SpectraTokenFront(nn.Module):
    """Shared per-layer token construction (projection, type embed, target marker, action tokens)."""

    def __init__(self, feature_dim: int, d_model: int = 256):
        super().__init__()
        self.output_dim = d_model
        self.feature_dim = feature_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.type_embedding = nn.Embedding(NUM_LAYER_TYPES, d_model)
        self.target_marker = nn.Parameter(torch.zeros(d_model))
        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_marker = nn.Parameter(torch.zeros(d_model))

    def embed_tokens(self, state: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
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

        coupling_ids = state.get("coupling_ids", state["block_ids"])
        action_costs = state.get("action_costs")
        if action_costs is not None and action_costs.numel():
            action_tokens = self.action_proj(action_costs.to(device)) + self.action_marker
            tokens = torch.cat([tokens, action_tokens], dim=0)
            target_coupling = (coupling_ids[target_index]
                               if 0 <= target_index < num_layers
                               else coupling_ids.new_zeros(()))
            coupling_ids = torch.cat([coupling_ids, target_coupling.expand(action_tokens.size(0))])

        return tokens, target_index, num_layers, coupling_ids

    def pool(self, encoded: torch.Tensor, target_index: int, num_layer_tokens: int) -> torch.Tensor:
        """encoded: (1, L', d) → (1, d)."""
        sequence_mean = encoded.mean(dim=1)
        if 0 <= target_index < num_layer_tokens:
            target_vec = encoded[:, target_index, :]
            return ((1.0 - TARGET_POOL_WEIGHT) * sequence_mean
                    + TARGET_POOL_WEIGHT * target_vec)
        return sequence_mean


class SpectraStateEncoder(SpectraTokenFront):
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
        super().__init__(feature_dim=feature_dim, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.block_affinity = nn.Parameter(torch.zeros(1))
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, target_index, num_layer_tokens, coupling_ids = self.embed_tokens(state)
        length = tokens.size(0)
        device = tokens.device
        same_group = coupling_ids.unsqueeze(0) == coupling_ids.unsqueeze(1)
        attention_bias = torch.zeros(length, length, device=device)
        attention_bias = attention_bias.masked_fill(same_group, 1.0) * self.block_affinity

        encoded = self.encoder(tokens.unsqueeze(0), mask=attention_bias)
        encoded = self.output_norm(encoded)
        return self.pool(encoded, target_index, num_layer_tokens)


class SpectraSetEncoder(SpectraTokenFront):
    """
    Architecture-agnostic *read* of the same tokens: per-token MLP, then target-aware pool.

    No cross-layer attention and no coupling bias. If this matches the Transformer on a
    recoverable environment (CIFAR-10), relational encoding is not what the agent is using.
    """

    def __init__(self, feature_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__(feature_dim=feature_dim, d_model=d_model)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, target_index, num_layer_tokens, _ = self.embed_tokens(state)
        tokens = tokens + self.token_mlp(tokens)
        encoded = self.output_norm(tokens).unsqueeze(0)
        return self.pool(encoded, target_index, num_layer_tokens)


def build_state_encoder(kind: str, feature_dim: int) -> nn.Module:
    """Factory for ``SPECTRA_STATE_ENCODER`` trainable variants."""
    key = (kind or "transformer").strip().lower()
    if key == "transformer":
        return SpectraStateEncoder(feature_dim=feature_dim)
    if key in ("transformer_wide", "transformer_deep"):
        return SpectraStateEncoder(feature_dim=feature_dim, d_model=512, nhead=8,
                                   num_layers=6, dropout=0.1)
    if key in ("set", "mlp"):
        return SpectraSetEncoder(feature_dim=feature_dim)
    raise ValueError(
        f"Unknown trainable encoder {kind!r}. Expected one of {TRAINABLE_ENCODER_KINDS}")
