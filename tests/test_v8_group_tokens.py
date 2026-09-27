"""CPU tests for the V8 group-as-token state (``SPECTRA_STATE_TOKENS=groups``). Default off.

    python -m pytest tests/test_v8_group_tokens.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from src import group_tokens  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.BERTInputModeler as bim  # noqa: E402
from src.Model.StateEncoder import SpectraStateEncoder  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)
    yield


def test_flag_default_off_and_in_contract():
    assert fortify.state_tokens() == "layers"
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    assert "SPECTRA_STATE_TOKENS" in A2CAgentReinforce.POLICY_CONTRACT_KEYS


def test_token_width_grows_only_under_the_flag(monkeypatch):
    base = bim.token_feature_dim(5)
    monkeypatch.setenv("SPECTRA_STATE_TOKENS", "groups")
    assert fortify.state_tokens() == "groups"
    assert bim.token_feature_dim(5) == base + group_tokens.GROUP_TOKEN_EXTRA_DIM


def test_pooling_means_members_and_maxes_action_slots():
    torch.manual_seed(0)
    L, F, slots = 6, 7, 2
    feats = torch.randn(L, F)
    feats[:, F - slots:] = 0.0
    feats[3, F - slots:] = torch.tensor([0.4, 0.2])            # target layer carries the slots
    cids = torch.tensor([0, 0, 1, 1, 2, 3])
    state = {"layer_features": feats, "layer_types": torch.tensor([1, 2, 1, 2, 1, 3]),
             "coupling_ids": cids, "block_ids": cids, "target_index": 3}
    out = group_tokens.group_token_state(state, layers=[], groups=None, slot_dim=slots)
    pooled = out["layer_features"]
    assert pooled.shape == (4, F + group_tokens.GROUP_TOKEN_EXTRA_DIM)
    assert torch.allclose(pooled[1, : F - slots], feats[2:4, : F - slots].mean(dim=0))
    assert torch.allclose(pooled[1, F - slots: F], torch.tensor([0.4, 0.2]))   # max, not mean/2
    assert torch.allclose(pooled[0, F - slots: F], torch.zeros(slots))
    # structure columns: member share, first / last position, prunable (0 without groups)
    assert pooled[0, F + 0].item() == pytest.approx(2 / 6)
    assert pooled[1, F + 1].item() == pytest.approx(2 / 6)
    assert pooled[1, F + 2].item() == pytest.approx(3 / 6)
    assert pooled[:, F + 3].sum().item() == 0.0
    assert out["target_index"] == 1                              # layer 3 lives in unit 1
    assert out["layer_types"].tolist() == [1, 1, 1, 3]
    assert out["coupling_ids"].tolist() == [0, 1, 2, 3]
    assert out["relations"].shape == (4, 4) and out["relations"].sum().item() == 0
    assert out["token_members"] == [[0, 1], [2, 3], [4], [5]]


def _synthetic_maps(L):
    return {"Topology": [[2, 3, 8, 3, 1, 1, 1]] * L, "Activations": [[0.1] * 12] * L,
            "Weights": [[0.2] * 19] * L}


def test_modeler_layer_state_untouched_without_flag_and_pooled_with_it(monkeypatch):
    monkeypatch.delenv("SPECTRA_FORTIFY", raising=False)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    L = len(mwr.all_layers)
    builder = bim.BERTInputModeler()
    off = builder.encode_model_to_bert_input(mwr, _synthetic_maps(L), 2, dependency_groups=groups)
    assert "relations" not in off
    assert off["layer_features"].size(0) == L
    assert off["layer_features"].size(1) == bim.token_feature_dim()
    monkeypatch.setenv("SPECTRA_STATE_TOKENS", "groups")
    on = builder.encode_model_to_bert_input(mwr, _synthetic_maps(L), 2, dependency_groups=groups)
    G = on["layer_features"].size(0)
    assert 1 < G < L
    assert on["layer_features"].size(1) == bim.token_feature_dim()      # width follows the flag
    assert on["relations"].shape == (G, G)
    assert 2 in on["token_members"][int(on["target_index"])]
    # the pooled state still runs through the real encoder
    enc = SpectraStateEncoder(feature_dim=bim.token_feature_dim(), d_model=32, nhead=4,
                              num_layers=1, dropout=0.0).eval()
    assert torch.isfinite(enc(on)).all()


def test_resnet_group_state_has_fewer_tokens_and_feed_relations(monkeypatch):
    monkeypatch.setenv("SPECTRA_STATE_TOKENS", "groups")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    layers = mwr.all_layers
    cids = channel_groups.coupling_ids_for_layers(layers, groups)
    L = len(layers)
    F = 9
    feats = torch.randn(L, F)
    types = torch.ones(L, dtype=torch.long)
    target = next(i for i, m in enumerate(layers) if m is model.layer2[0].conv1)
    state = {"layer_features": feats, "layer_types": types, "coupling_ids": cids,
             "block_ids": cids, "target_index": target}
    out = group_tokens.group_token_state(state, layers, groups, slot_dim=0)
    G = out["layer_features"].size(0)
    assert 1 < G < L                                             # residual streams pooled
    rel = out["relations"]
    assert rel.shape == (G, G)
    assert (rel == group_tokens.RELATION_FEEDS).any() and (rel == group_tokens.RELATION_FED_BY).any()
    # feeds and fed-by are transposes of each other where both are set
    feeds = rel == group_tokens.RELATION_FEEDS
    fed = rel == group_tokens.RELATION_FED_BY
    assert torch.equal(feeds.T & fed, fed)
    assert rel.diagonal().sum().item() == 0
    # the target unit is prunable and owns the target layer
    t = out["target_index"]
    assert target in out["token_members"][t]
    assert out["layer_features"][t, F + 3].item() == 1.0


def test_encoder_uses_relation_bias_only_when_relations_present():
    torch.manual_seed(1)
    d = 12
    enc = SpectraStateEncoder(feature_dim=d, d_model=32, nhead=4, num_layers=1, dropout=0.0).eval()
    G = 4
    feats = torch.randn(G, d)
    state = {"layer_features": feats, "layer_types": torch.ones(G, dtype=torch.long),
             "coupling_ids": torch.arange(G), "block_ids": torch.arange(G), "target_index": 1}
    rel = torch.zeros(G, G, dtype=torch.long)
    rel[0, 1] = group_tokens.RELATION_FEEDS
    rel[1, 0] = group_tokens.RELATION_FED_BY
    with torch.no_grad():
        enc.relation_bias[1] = 2.0
        enc.relation_bias[2] = -1.0
    plain = enc(state)
    with_rel = enc(dict(state, relations=rel))
    assert torch.isfinite(with_rel).all()
    assert not torch.allclose(plain, with_rel)                   # bias changes the read
    with torch.no_grad():
        enc.relation_bias.zero_()
    assert torch.allclose(enc(state), enc(dict(state, relations=rel)), atol=1e-6)
    # index 0 (no relation) is pinned to zero even if the parameter drifts
    with torch.no_grad():
        enc.relation_bias[0] = 5.0
    assert torch.allclose(enc(dict(state, relations=torch.zeros(G, G, dtype=torch.long))), enc(state), atol=1e-6)
    # gradient reaches the relation bias when relations are present (a plain .sum() of a
    # LayerNorm output is identically constant, so weight the output first)
    enc.train()
    weights = torch.randn(1, 32)
    out = (enc(dict(state, relations=rel)) * weights).sum()
    out.backward()
    assert enc.relation_bias.grad is not None and enc.relation_bias.grad[1].abs().item() > 0
    assert enc.relation_bias.grad[0].item() == 0.0                # index 0 never learns
