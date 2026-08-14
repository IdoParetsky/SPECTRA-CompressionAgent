"""
Build the agent's state from CNN feature maps.

Baseline (default): a fixed-width numeric token per layer, consumed by the trainable
``SpectraStateEncoder``. Frozen ``bert-base-uncased`` is *not* loaded unless
``SPECTRA_STATE_ENCODER=bert`` — it remains an ablation only (see
docs/THESIS_INSTRUCTOR_BRIEFING.md and docs/BERT_INPUT_CRITIQUE.md).

Token layout (per layer)
------------------------
  topology (7) | activation moments (NUM_MOMENTS) | weight moments (NUM_MOMENTS)
               | per-filter L1 shape (WEIGHT_SHAPE_DIM)
               | action-cost slots (2 * num_actions)  — filled only on the target layer

The dual "local / global" BERT segments from the document are intentionally *not* used:
entity marking of the target layer (a learned marker in the Transformer path; a
``token_type_ids`` bit in the BERT ablation) conveys the same distinction without doubling
the sequence. The 512-position ceiling therefore only matters for the optional BERT path.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import torch

import src.channel_groups as channel_groups
import src.distributed as ddp
import src.utils as utils
from src.feature_standardizer import FeatureStandardizer

# "embeds" / "text" only apply when SPECTRA_STATE_ENCODER=bert
BERT_INPUT_MODE = os.environ.get("SPECTRA_BERT_INPUT_MODE", "embeds").strip().lower()
STATE_ENCODER_KIND = os.environ.get("SPECTRA_STATE_ENCODER", "transformer").strip().lower()

# Must match BaseFE.MOMENT_NAMES / SHAPE_NAMES
NUM_MOMENTS = 12
WEIGHT_SHAPE_DIM = 7
TOPOLOGY_DIM = 7
# Base width *before* fortify channels and per-rate action-cost slots on the target layer
TOKEN_BASE_DIM = TOPOLOGY_DIM + NUM_MOMENTS + NUM_MOMENTS + WEIGHT_SHAPE_DIM  # 38

BERT_MAX_POSITIONS = 512


def _default_num_actions() -> int:
    try:
        from src.Configuration.StaticConf import StaticConf
        conf = StaticConf.get_instance()
        if conf is not None:
            return int(conf.conf_values.num_actions)
    except Exception:
        pass
    return 5


def action_cost_slot_dim(num_actions: Optional[int] = None) -> int:
    """Two slots per candidate rate: (param_fraction_removed, mac_fraction_removed)."""
    return 2 * (num_actions if num_actions is not None else _default_num_actions())


def token_feature_dim(num_actions: Optional[int] = None) -> int:
    from src.fortify import fortify_token_dim
    return TOKEN_BASE_DIM + fortify_token_dim() + action_cost_slot_dim(num_actions)


# Back-compat alias used by Agent / tests; recomputed at import from StaticConf when present
TOKEN_FEATURE_DIM = token_feature_dim()


def _signed_log1p(tensor: torch.Tensor) -> torch.Tensor:
    """Fallback squash when the database standardiser has not been fitted yet."""
    return torch.sign(tensor) * torch.log1p(tensor.abs())


def _sinusoidal_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer sinusoidal positional encoding evaluated at arbitrary positions."""
    device = positions.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    encoding = torch.zeros(positions.numel(), dim, device=device)
    encoding[:, 0:2 * half:2] = torch.sin(angles)
    encoding[:, 1:2 * half:2] = torch.cos(angles)
    return encoding


class BERTInputModeler:
    """
    State builder (and, optionally, frozen-BERT encoder for ablation).

    Despite the historical name, the default path never downloads or instantiates BERT.
    Prefer thinking of this object as the CNN → agent-state adapter.
    """

    _instance = None

    def __new__(cls, bert_model_name: str = "bert-base-uncased"):
        if cls._instance is None:
            cls._instance = super(BERTInputModeler, cls).__new__(cls)
            cls._instance._initialize(bert_model_name)
        return cls._instance

    def _initialize(self, bert_model_name):
        self.device = ddp.resolve_device()
        self.bert_model_name = bert_model_name
        self.mode = BERT_INPUT_MODE
        self._bert_loaded = False
        self.tokenizer = None
        self.bert_model = None
        self.cls_embedding = None
        self.sep_embedding = None
        self.feature_projection = None
        self.hidden_size = None

        # Load frozen BERT only for the ablation encoder — never for the trainable baseline
        if STATE_ENCODER_KIND == "bert":
            self._ensure_bert()

    def _ensure_bert(self):
        if self._bert_loaded:
            return
        # Local import so transformer-default runs never require a transformers download path
        # to succeed at import time beyond what the environment already has installed.
        from transformers import BertTokenizer, BertModel

        utils.print_flush(
            f"Loading frozen {self.bert_model_name} for SPECTRA_STATE_ENCODER=bert ablation "
            f"(not the training baseline)")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name).to(self.device)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_model.eval()
        self.hidden_size = self.bert_model.config.hidden_size

        embeddings = self.bert_model.embeddings.word_embeddings.weight
        self.cls_embedding = embeddings[self.tokenizer.cls_token_id].detach().clone()
        self.sep_embedding = embeddings[self.tokenizer.sep_token_id].detach().clone()

        # Fixed projection into BERT hidden space (frozen-encoder design; not trained)
        generator = torch.Generator(device="cpu").manual_seed(0)
        projection = torch.randn(TOKEN_BASE_DIM, self.hidden_size, generator=generator)
        projection /= math.sqrt(TOKEN_BASE_DIM)
        self.feature_projection = projection.to(self.device)
        self._bert_loaded = True

    # ------------------------------------------------------------------ token construction

    def build_base_tokens(self, feature_maps: Dict[str, List[List[float]]]) -> torch.Tensor:
        """
        Assemble raw per-layer base features → (num_layers, TOKEN_BASE_DIM), no scaling.

        Used both for online state building and for fitting FeatureStandardizer.
        """
        topology = feature_maps["Topology"]
        activations = feature_maps["Activations"]
        weights = feature_maps["Weights"]

        rows = []
        for idx in range(len(topology)):
            topo = list(topology[idx]) if idx < len(topology) else [0.0] * TOPOLOGY_DIM
            act = (list(activations[idx]) if idx < len(activations) and activations[idx]
                   else [0.0] * NUM_MOMENTS)
            wgt = list(weights[idx]) if idx < len(weights) and weights[idx] else (
                [0.0] * (NUM_MOMENTS + WEIGHT_SHAPE_DIM))

            topo = (topo + [0.0] * TOPOLOGY_DIM)[:TOPOLOGY_DIM]
            act = (act + [0.0] * NUM_MOMENTS)[:NUM_MOMENTS]
            wgt = (wgt + [0.0] * (NUM_MOMENTS + WEIGHT_SHAPE_DIM))[: NUM_MOMENTS + WEIGHT_SHAPE_DIM]
            rows.append(topo + act + wgt)

        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _scale_base_tokens(self, base: torch.Tensor) -> torch.Tensor:
        """Database-wide z-score when fitted; otherwise signed-log1p fallback."""
        std = FeatureStandardizer.instance(TOKEN_BASE_DIM)
        if std.is_fitted:
            return std.transform(base)
        return _signed_log1p(base)

    def _attach_action_cost_slots(self, base_scaled: torch.Tensor, target_index: int,
                                  action_costs: Optional[torch.Tensor],
                                  num_actions: Optional[int] = None) -> torch.Tensor:
        """
        Critique §7: extend the *target* layer's token with, for each candidate rate, the
        fraction of total parameters and MACs that pruning at that rate would remove.

        Other layers receive zeros in those slots so every token stays the same width.
        ``action_costs`` is expected as (num_actions, 3) = (rate, param_frac, mac_frac).
        """
        n_actions = num_actions if num_actions is not None else _default_num_actions()
        slot_dim = 2 * n_actions
        num_layers = base_scaled.size(0)
        slots = torch.zeros(num_layers, slot_dim, device=base_scaled.device, dtype=base_scaled.dtype)

        if action_costs is not None and action_costs.numel() and 0 <= target_index < num_layers:
            costs = action_costs.to(device=base_scaled.device, dtype=base_scaled.dtype)
            # Drop the rate column — it is already known from the action index / conf
            fracs = costs[:, 1:3].reshape(-1)
            width = min(slot_dim, fracs.numel())
            slots[target_index, :width] = fracs[:width]

        return torch.cat([base_scaled, slots], dim=1)

    def _build_layer_tokens(self, feature_maps, curr_layer_idx,
                            action_costs=None, coupling_ids=None,
                            param_ratio=None) -> torch.Tensor:
        from src.fortify import fortify_enabled, build_fortify_features, budget_in_state

        base = self._scale_base_tokens(self.build_base_tokens(feature_maps))
        if fortify_enabled():
            # Fortify channels are already ~[0,1]; append after z-score of raw moments.
            fort = build_fortify_features(
                base.size(0), coupling_ids, feature_maps.get("Topology", []),
                device=base.device, dtype=base.dtype)
            base = torch.cat([base, fort], dim=1)
        if budget_in_state() and base.size(0):
            ratio = 1.0 if param_ratio is None else float(param_ratio)
            ratio = min(1.0, max(0.0, ratio))
            col = torch.full((base.size(0), 1), ratio, device=base.device, dtype=base.dtype)
            base = torch.cat([base, col], dim=1)
        target = min(curr_layer_idx, base.size(0) - 1) if base.size(0) else 0
        return self._attach_action_cost_slots(base, target, action_costs)

    @staticmethod
    def _block_ids(model_with_rows) -> torch.Tensor:
        """
        Legacy heuristic: group layers by owning module.

        Retained for the frozen-BERT ablation and for tests; the trainable baseline prefers
        ``coupling_ids_for_layers`` derived from ``channel_groups``.
        """
        parents = getattr(model_with_rows, "layer_parents", None)
        if not parents:
            return torch.arange(len(model_with_rows.all_layers))

        ids, seen = [], {}
        for parent, _name in parents:
            key = id(parent)
            if key not in seen:
                seen[key] = len(seen)
            ids.append(seen[key])
        return torch.tensor(ids)

    def encode_model_to_bert_input(self, model_with_rows, feature_maps, curr_layer_idx,
                                   dependency_groups=None,
                                   action_costs=None,
                                   param_ratio=None) -> Dict[str, torch.Tensor]:
        """
        Package CNN features as an agent state.

        Always returns encoder-agnostic fields consumed by ``SpectraStateEncoder``. When
        ``SPECTRA_STATE_ENCODER=bert``, also fills a ``bert`` entry for the frozen ablation.
        """
        with torch.no_grad():
            topology = feature_maps["Topology"]
            # Exact channel-coupling ids when groups are available; module-parent fallback
            if dependency_groups is not None:
                coupling = channel_groups.coupling_ids_for_layers(
                    model_with_rows.all_layers, dependency_groups).to(self.device)
            else:
                coupling = self._block_ids(model_with_rows).to(self.device)

            layer_tokens = self._build_layer_tokens(
                feature_maps, curr_layer_idx, action_costs, coupling_ids=coupling,
                param_ratio=param_ratio)
            coupling = coupling[: layer_tokens.size(0)]
            layer_types = torch.tensor(
                [int(row[0]) if row else 0 for row in topology],
                dtype=torch.long, device=self.device)[: layer_tokens.size(0)]

        target_index = min(curr_layer_idx, layer_tokens.size(0) - 1) if layer_tokens.size(0) else 0
        state = {
            "layer_features": layer_tokens,
            "layer_types": layer_types,
            # ``coupling_ids`` is the preferred key; ``block_ids`` kept as an alias so older
            # checkpoints / tests that still look for the heuristic name keep working.
            "coupling_ids": coupling,
            "block_ids": coupling,
            "target_index": target_index,
        }
        if action_costs is not None:
            state["action_costs"] = action_costs

        if STATE_ENCODER_KIND == "bert":
            self._ensure_bert()
            if self.mode == "text":
                state["bert"] = self._encode_as_text(feature_maps, curr_layer_idx)
            else:
                state["bert"] = self._encode_as_embeddings(
                    model_with_rows, feature_maps, curr_layer_idx, dependency_groups)

        return state

    def _encode_as_embeddings(self, model_with_rows, feature_maps, curr_layer_idx,
                              dependency_groups=None) -> Dict[str, torch.Tensor]:
        """
        Frozen-BERT ablation with *entity marking*, not dual local/global segments.

        Sequence: ``[CLS] layer_0 … layer_L [SEP]``. The target layer is marked by
        ``token_type_ids == 1`` (entity-marker style); every other layer stays in segment 0.
        If the network exceeds BERT's 512 positions, layers are mean-pooled by coupling id
        until they fit — that ceiling is BERT's, not SPECTRA's (the trainable encoder has none).
        """
        with torch.no_grad():
            base = self._scale_base_tokens(self.build_base_tokens(feature_maps))
            num_layers = base.size(0)
            if num_layers == 0:
                raise ValueError("No layers available for BERT encoding")

            embeddings = base @ self.feature_projection  # (L, hidden)

            if dependency_groups is not None:
                coupling = channel_groups.coupling_ids_for_layers(
                    model_with_rows.all_layers, dependency_groups).to(self.device)[:num_layers]
            else:
                coupling = self._block_ids(model_with_rows).to(self.device)[:num_layers]

            # Budget: [CLS] + layers + [SEP]
            budget = BERT_MAX_POSITIONS - 2
            if num_layers > budget:
                embeddings, coupling = self._pool_by_coupling(embeddings, coupling, budget)
                num_layers = embeddings.size(0)

            target_index = min(curr_layer_idx, num_layers - 1)
            sequence = torch.cat([
                self.cls_embedding.unsqueeze(0),
                embeddings,
                self.sep_embedding.unsqueeze(0),
            ], dim=0)

            # Entity marker: only the target layer sits in segment 1
            token_type_ids = torch.zeros(sequence.size(0), dtype=torch.long, device=self.device)
            token_type_ids[1 + target_index] = 1

            return {
                "inputs_embeds": sequence.unsqueeze(0),
                "attention_mask": torch.ones(1, sequence.size(0), dtype=torch.long, device=self.device),
                "token_type_ids": token_type_ids.unsqueeze(0),
            }

    @staticmethod
    def _pool_by_coupling(embeddings: torch.Tensor, coupling_ids: torch.Tensor,
                          budget: int):
        """Mean-pool each coupling group into one token, then trim to BERT's position budget."""
        pooled, pooled_ids = [], []
        for cid in torch.unique(coupling_ids, sorted=True):
            pooled.append(embeddings[coupling_ids == cid].mean(dim=0))
            pooled_ids.append(cid)
        stacked = torch.stack(pooled)
        ids = torch.stack(pooled_ids)
        if stacked.size(0) > budget:
            return stacked[:budget], ids[:budget]
        return stacked, ids

    def _encode_as_text(self, feature_maps, curr_layer_idx) -> Dict[str, torch.Tensor]:
        """
        Legacy encoding: every feature is rendered as text and tokenized with WordPiece.

        Retained only as the historical control. Each float costs roughly ten WordPiece
        tokens, so the 512-token window holds on the order of fifty numbers.
        """
        self._ensure_bert()
        flatten = lambda nested: [item for sublist in nested for item in sublist]

        layer_features, full_features = [], []
        for _feature_type, all_layers in feature_maps.items():
            layer_features.extend(all_layers[curr_layer_idx])
            full_features.extend(flatten(all_layers))

        with torch.no_grad():
            encoded_input = self.tokenizer(
                ' '.join(map(str, layer_features)),
                ' '.join(map(str, full_features)),
                max_length=BERT_MAX_POSITIONS,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        return {k: v.to(self.device) for k, v in encoded_input.items()}

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Frozen BERT forward (ablation only)."""
        self._ensure_bert()
        tokens = {k: v.to(self.device) for k, v in state["bert"].items()}
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        return outputs.last_hidden_state

    def embed_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pool BERT's output into the fixed-size state vector for the actor/critic.

        Averaging is restricted to real positions so padding cannot dominate the mean.
        """
        hidden = self.forward(state)
        mask = state["bert"].get("attention_mask")
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.to(hidden.device).unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
