import os

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from src.utils import normalize_2d_data, normalize_3d_data
from src.BERTInputModeler import BERTInputModeler, token_feature_dim
from src.Model.StateEncoder import TRAINABLE_ENCODER_KINDS, build_state_encoder

# "transformer" (default): small Transformer trained with A2C — SPECTRA baseline.
# "transformer_wide":      same bias, 6×512 capacity check.
# "set":                   same tokens, no cross-layer attention (agnostic read).
# "bert":                  frozen bert-base-uncased ablation.
# "legacy":                NEON convolutional feature pipelines.
STATE_ENCODER = os.environ.get("SPECTRA_STATE_ENCODER", "transformer").strip().lower()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class FactoredCategorical:
    """
    Joint policy over (rate, ranking) with two independent heads (``SPECTRA_FACTORED_HEAD=1``).

    ``rate`` and ``rank`` are ``torch.distributions.Categorical`` over the rate menu and the
    ranking menu. An action is a pair ``(rate_idx, rank_idx)``. The ranking head is *inactive*
    on the identity rate: its log-probability is not added and its choice is irrelevant, so
    the policy gradient for "which filters" only flows through steps that actually cut.
    Entropy is the sum of the two heads' entropies (both regularised).
    """

    def __init__(self, rate: Categorical, rank: Categorical, identity_index: int = 0):
        self.rate = rate
        self.rank = rank
        self.identity_index = int(identity_index)

    @property
    def probs(self):
        """Rate-head probabilities (what masks, telemetry and argmax-of-rate read)."""
        return self.rate.probs

    def with_rate(self, rate: Categorical) -> "FactoredCategorical":
        return FactoredCategorical(rate, self.rank, self.identity_index)

    def sample(self):
        return self.rate.sample().reshape(-1)[0], self.rank.sample().reshape(-1)[0]

    def argmax(self):
        r = self.rate.probs.flatten().argmax()
        k = self.rank.probs.flatten().argmax()
        return r, k

    def log_prob(self, rate_idx, rank_idx):
        r = torch.as_tensor(rate_idx, device=self.rate.probs.device).reshape(1)
        k = torch.as_tensor(rank_idx, device=self.rank.probs.device).reshape(1)
        lp = self.rate.log_prob(r).reshape(-1)[0]
        active = (r.reshape(-1)[0] != self.identity_index).to(lp.dtype)
        return lp + active * self.rank.log_prob(k).reshape(-1)[0]

    def entropy(self):
        return self.rate.entropy().reshape(-1)[0] + self.rank.entropy().reshape(-1)[0]


class Agent(nn.Module):
    def __init__(self, device, num_outputs, encoder=None):
        super(Agent, self).__init__()
        self.device = device
        self.is_actor = False
        self.is_critic = False

        self.encoder_kind = (encoder or STATE_ENCODER)
        self.bert_enabled = self.encoder_kind == "bert"

        # State builder is always needed (numeric layer tokens). Frozen BERT weights are
        # loaded inside BERTInputModeler only when encoder_kind == "bert".
        self.bert_input_modeler = BERTInputModeler()

        if self.encoder_kind in TRAINABLE_ENCODER_KINDS:
            # Width includes the per-rate action-cost slots on the target layer token
            feature_dim = token_feature_dim(num_outputs)
            self.state_encoder = build_state_encoder(self.encoder_kind, feature_dim)
            self.embedding_dim = self.state_encoder.output_dim
        elif self.bert_enabled:
            self.bert_input_modeler._ensure_bert()
            self.state_encoder = None
            self.embedding_dim = self.bert_input_modeler.hidden_size
        elif self.encoder_kind == "legacy":
            self.state_encoder = None
            self.embedding_dim = 5350  # width of the concatenated legacy pipelines
        else:
            raise ValueError(
                f"Unknown SPECTRA_STATE_ENCODER={self.encoder_kind!r}. "
                f"Expected transformer|transformer_wide|set|bert|legacy")

        # DRL Head
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, num_outputs),
            nn.Softmax(dim=1),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        # V4-1: factored (rate × ranking) policy. The rate head is ``self.actor``; a second
        # head chooses the filter-importance criterion from ``conf.ranking_menu``. Off unless
        # SPECTRA_FACTORED_HEAD=1 *and* a ranking menu is configured, so every earlier actor
        # loads and behaves exactly as before.
        from src.fortify import policy_head_zero_init, factored_head
        self.ranking_menu = []
        self.ranker = None
        if factored_head():
            menu = self._configured_ranking_menu()
            if len(menu) >= 2:
                self.ranking_menu = list(menu)
                self.ranker = nn.Sequential(
                    nn.Linear(self.embedding_dim, 300),
                    nn.ReLU(),
                    nn.Linear(300, 300),
                    nn.ReLU(),
                    nn.Linear(300, len(menu)),
                    nn.Softmax(dim=1),
                )

        # SPECTRA_POLICY_HEAD_ZERO_INIT=1: start from an exactly uniform policy so the first
        # updates are driven by state features, not by the random bias of the last Linear
        # (the frozen s42 actor's argmax was its bias vector; audit 13 Sep F3).
        if policy_head_zero_init():
            last = self.actor[4]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            if self.ranker is not None:
                nn.init.zeros_(self.ranker[4].weight)
                nn.init.zeros_(self.ranker[4].bias)

    @staticmethod
    def _configured_ranking_menu():
        try:
            from src.Configuration.StaticConf import StaticConf
            conf = StaticConf.get_instance()
            menu = getattr(conf.conf_values, "ranking_menu", None) if conf is not None else None
            return list(menu or [])
        except Exception:
            return []

    @property
    def is_factored(self) -> bool:
        return self.ranker is not None

        # Original NEON feature processing pipelines (legacy). These hold ~10M parameters
        # per agent and are unreachable unless they are the selected encoder, yet they used
        # to be constructed unconditionally: they were handed to Adam (allocating optimizer
        # state for them) and forced DDP into static_graph mode to tolerate unused parameters.
        if self.encoder_kind == "legacy":
            self._build_legacy_feature_pipelines()

    def _build_legacy_feature_pipelines(self):
        self.architecture_network = nn.Sequential(
            nn.Conv2d(1, 50, (1, 8)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 5, (1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 2, (1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten()
        )

        self.weights_network = nn.Sequential(
            nn.Conv2d(8, 100, (1, 1000)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 10, (1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 5, (1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            Flatten()
        )

        self.activation_network = nn.Sequential(
            nn.Conv2d(8, 100, (1, 1000)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 10, (1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 5, (1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            Flatten()
        )

        self.architecture_layer = nn.Sequential(
            nn.Conv2d(1, 10, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(10, 10, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(10, 10, (1, 1)),
            nn.ReLU(),
            Flatten()
        )

        self.weights_layer = nn.Sequential(
            nn.Conv2d(1, 500, (8, 1000)),
            nn.ReLU(),
            nn.Conv2d(500, 500, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(500, 20, (1, 1)),
            nn.ReLU(),
            Flatten()
        )

        self.activation_layer = nn.Sequential(
            nn.Conv2d(1, 500, (8, 1000)),
            nn.ReLU(),
            nn.Conv2d(500, 500, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(500, 20, (1, 1)),
            nn.ReLU(),
            Flatten()
        )

    def forward(self, state_tokens_or_fm):
        """
        Forward pass of the DRL Agent.

        Args:
            state_tokens_or_fm: Agent state dict (transformer/bert) or legacy feature maps.

        Returns:
            Either:
                - Categorical action distribution (Actor)
                - Value prediction (Critic)
        """
        if self.encoder_kind in TRAINABLE_ENCODER_KINDS:
            # Trained end to end with the policy, so no no_grad here
            embeddings = self.state_encoder(state_tokens_or_fm)
        elif self.bert_enabled:
            with torch.no_grad():
                embeddings = self.bert_input_modeler.embed_state(state_tokens_or_fm)
        else:
            embeddings = self.extract_legacy_features(state_tokens_or_fm)

        if self.is_actor and not self.is_critic:
            probs = self.actor(embeddings)
            if self.ranker is not None:
                from src.fortify import identity_action_index
                from src.Configuration.StaticConf import StaticConf
                rates = StaticConf.get_instance().conf_values.compression_rates_dict
                return FactoredCategorical(Categorical(probs), Categorical(self.ranker(embeddings)),
                                           identity_index=identity_action_index(rates))
            return Categorical(probs)
        elif self.is_critic and not self.is_actor:
            return self.critic(embeddings)
        else:
            raise TypeError("Agent must be either Actor or Critic.")

    def extract_legacy_features(self, fm):
        """
        Legacy feature extraction pipeline (ConvNet-based) for non-BERT usage.
        """
        fm_splitted = self.split_fm(fm)
        index_of_current_layer = np.argmax((fm_splitted[0] == fm_splitted[1]).sum(axis=1))

        architecture_topology_fm_norm = self.convert_to_tensor(normalize_2d_data(fm_splitted[0]))
        architecture_weights_fm_norm = self.convert_to_tensor(normalize_3d_data(fm_splitted[2]))
        architecture_activations_fm_norm = self.convert_to_tensor(normalize_3d_data(fm_splitted[4]))

        layer_weights_fm_norm = self.convert_to_tensor(normalize_2d_data((fm_splitted[3])))
        layer_activation_fm_norm = self.convert_to_tensor(normalize_2d_data((fm_splitted[5])))

        arch_net_features = self.architecture_network(architecture_topology_fm_norm.unsqueeze(0))
        weights_net_features = self.weights_network(architecture_weights_fm_norm)
        activations_net_features = self.activation_network(architecture_activations_fm_norm)

        arch_layer_features = self.architecture_layer(
            architecture_topology_fm_norm[0][index_of_current_layer].unsqueeze(0).unsqueeze(0).unsqueeze(0))
        weights_layer_features = self.weights_layer(layer_weights_fm_norm.unsqueeze(0))
        activations_layer_features = self.activation_layer(layer_activation_fm_norm.unsqueeze(0))

        return torch.cat((
            arch_net_features,
            weights_net_features,
            activations_net_features,
            arch_layer_features,
            weights_layer_features,
            activations_layer_features
        ), 1)

    def split_fm(self, fm):
        arc = fm[0]
        net_arc = arc[0]
        layer_arch = arc[1]

        activations = fm[1]
        net_ac = activations[0]
        layer_ac = activations[1]

        weights = fm[2]
        net_weights = weights[0]
        layer_weights = weights[1]

        return net_arc, layer_arch, net_ac, layer_ac, net_weights, layer_weights

    def convert_to_tensor(self, np_ar) -> torch.Tensor:
        return torch.Tensor(np_ar).to(self.device).float().unsqueeze(0)
