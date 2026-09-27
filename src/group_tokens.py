"""
Group-as-token state (V8, ``SPECTRA_STATE_TOKENS=groups``). Default off.

SPECTRA prunes *channel groups* — the producers, depthwise owners and norms that share one
channel dimension through residual, concat or depthwise ties — but the encoder's tokens are
*layers*. With layer tokens the actor must reassemble "what does cutting this row do to the
whole net" from a learned scalar bias on same-group pairs. Here the prune unit is the token:

* one token per coupling id (``channel_groups.coupling_ids_for_layers``): the mean of its member
  layer tokens plus four structure columns — member share, first / last member position (÷ L),
  and whether the unit is structurally prunable;
* a relation matrix between tokens: 1 = "this unit's stream feeds that unit's producers",
  2 = the reverse. The encoder turns it into a learned per-relation attention bias
  (Graphormer-style edge encoding) instead of the single same-group scalar;
* the target token is the unit that owns the layer about to be cut; action-cost tokens are
  appended unchanged; the legal-rate mask stays env-side (representation may change *which*
  unit and *how hard*, never which rates exist).

Off = the layer-token state is returned untouched (byte-identical to every actor so far).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

GROUP_TOKEN_EXTRA_DIM = 4
RELATION_NONE, RELATION_FEEDS, RELATION_FED_BY = 0, 1, 2


def token_index_of_ids(coupling_ids: torch.Tensor) -> Dict[int, int]:
    """Token index per coupling id, in order of first appearance along the layer sequence."""
    order: Dict[int, int] = {}
    for cid in coupling_ids.tolist():
        if int(cid) not in order:
            order[int(cid)] = len(order)
    return order


def relation_matrix(layers, groups, coupling_ids: torch.Tensor, token_of: Dict[int, int]) -> torch.Tensor:
    """G×G long matrix: 1 where the row unit's stream feeds the column unit, 2 for the reverse."""
    n = len(token_of)
    rel = torch.zeros(n, n, dtype=torch.long)
    if not groups:
        return rel
    index_of = {id(layer): i for i, layer in enumerate(layers)}
    cids = coupling_ids.tolist()
    for group in groups:
        owners = [index_of[id(m)] for m in list(group.producers) + list(group.depthwise) if id(m) in index_of]
        if not owners:
            continue
        a = token_of.get(int(cids[owners[0]]))
        if a is None:
            continue
        for ref in group.consumers:
            j = index_of.get(id(ref.module))
            if j is None:
                continue
            b = token_of.get(int(cids[j]))
            if b is None or b == a:
                continue
            rel[a, b] = RELATION_FEEDS
            if rel[b, a] == RELATION_NONE:
                rel[b, a] = RELATION_FED_BY
    return rel


def group_token_state(state: Dict[str, torch.Tensor], layers, groups,
                      slot_dim: int = 0) -> Dict[str, torch.Tensor]:
    """
    Pool a layer-token state into a group-token state. ``layers`` = ``model_with_rows.all_layers``,
    ``groups`` = the channel groups (may be None → coupling ids alone define the units).
    The trailing ``slot_dim`` columns are the target layer's action-cost slots (zeros on every
    other layer) — they are pooled by max so the target unit keeps them undiluted.
    """
    feats = state["layer_features"]
    if not torch.is_tensor(feats) or feats.dim() != 2 or feats.size(0) == 0:
        return state
    cids = state.get("coupling_ids", state.get("block_ids"))
    if cids is None or cids.numel() != feats.size(0):
        return state
    cids = cids.detach().cpu()
    L = int(feats.size(0))
    token_of = token_index_of_ids(cids)
    G = len(token_of)
    device, dtype = feats.device, feats.dtype
    members: List[List[int]] = [[] for _ in range(G)]
    for i, cid in enumerate(cids.tolist()):
        members[token_of[int(cid)]].append(i)

    prunable = torch.zeros(G, dtype=dtype, device=device)
    if groups:
        index_of = {id(layer): i for i, layer in enumerate(layers)}
        for group in groups:
            owners = [index_of[id(m)] for m in list(group.producers) + list(group.depthwise) if id(m) in index_of]
            if owners and getattr(group, "prunable", False):
                prunable[token_of[int(cids[owners[0]])]] = 1.0

    pooled = torch.zeros(G, feats.size(1) + GROUP_TOKEN_EXTRA_DIM, device=device, dtype=dtype)
    types_src = state.get("layer_types")
    types = torch.zeros(G, dtype=torch.long, device=feats.device)
    F = int(feats.size(1))
    slot_dim = max(0, min(int(slot_dim), F))
    for g, idx in enumerate(members):
        rows = torch.tensor(idx, dtype=torch.long, device=device)
        block = feats[rows]
        pooled[g, : F - slot_dim] = block[:, : F - slot_dim].mean(dim=0)
        if slot_dim:
            pooled[g, F - slot_dim: F] = block[:, F - slot_dim:].max(dim=0).values
        pooled[g, feats.size(1) + 0] = len(idx) / max(L, 1)
        pooled[g, feats.size(1) + 1] = idx[0] / max(L, 1)
        pooled[g, feats.size(1) + 2] = idx[-1] / max(L, 1)
        pooled[g, feats.size(1) + 3] = prunable[g]
        if types_src is not None and types_src.numel() > idx[0]:
            types[g] = types_src[idx[0]]

    target_layer = int(state.get("target_index", 0))
    target_token = token_of[int(cids[min(max(target_layer, 0), L - 1)])]
    out = dict(state)
    out["layer_features"] = pooled
    out["layer_types"] = types
    out["coupling_ids"] = torch.arange(G, dtype=torch.long, device=cids.device if cids.is_cuda else device)
    out["block_ids"] = out["coupling_ids"]
    out["target_index"] = target_token
    out["relations"] = relation_matrix(layers, groups, cids, token_of).to(device)
    out["token_members"] = members
    return out
