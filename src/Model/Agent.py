import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from src.utils import normalize_2d_data, normalize_3d_data
from src.BERTInputModeler import BERTInputModeler


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Agent(nn.Module):
    def __init__(self, device, num_outputs):
        super(Agent, self).__init__()
        self.device = device
        self.is_actor = False
        self.is_critic = False

        # BERT mechanism
        self.bert_enabled = True
        self.bert_input_modeler = BERTInputModeler()
        self.embedding_dim = self.bert_input_modeler.bert_model.config.hidden_size

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

        # Original NEON feature processing pipelines (legacy)
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
        If BERT-based embeddings are enabled, this expects tokenized BERT inputs.
        Otherwise, expects raw feature maps (legacy).

        Args:
            state_tokens_or_fm: BERT input tokens or raw feature map tuple.

        Returns:
            Either:
                - Categorical action distribution (Actor)
                - Value prediction (Critic)
        """
        if self.bert_enabled:
            embeddings = self.bert_input_modeler.forward(state_tokens_or_fm)
            embeddings = embeddings.mean(dim=1)  # shape: (batch_size, embedding_dim)
        else:
            embeddings = self.extract_legacy_features(state_tokens_or_fm)

        if self.is_actor and not self.is_critic:
            probs = self.actor(embeddings)
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
