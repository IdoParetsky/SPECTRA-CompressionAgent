"""
Channel dependency analysis.

Structured pruning cannot treat layers independently. In a ResNet block the output of
``conv2`` is added to the identity shortcut, so ``conv2`` and whatever produced the shortcut
must drop *the same* channel indices or the addition stops type-checking. In a DenseNet the
output of every dense layer is concatenated onto a growing feature bank, so removing one
layer's channels shifts the position of every later layer's channels inside that bank.

This module recovers both kinds of constraint by tracing the model with ``torch.fx`` and
describing the channel dimension of each tensor as an ordered list of *segments*:

* a Conv2d/Linear opens a new segment for its output dimension;
* BatchNorm, activations, dropout and pooling pass segments through unchanged;
* ``torch.cat`` concatenates the segment lists of its operands, so consumers see each group
  at a known offset;
* an element-wise binary op (the residual add) *merges* the segments of its operands, forcing
  them to share one channel dimension;
* a depthwise convolution ties its input and output segments together.

Anything not understood -- unknown modules, tensors escaping to the model output, a
concatenation along a non-channel axis -- marks the affected groups unprunable, and the
caller falls back to masking.

This is a lightweight rendering of the dependency-graph idea from Fang et al., "DepGraph:
Towards Any Structural Pruning" (CVPR 2023), which the thesis proposal cites.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.fx import Node

PRUNABLE_TYPES = (nn.Conv2d, nn.Linear)
NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d,
              nn.LayerNorm, nn.LocalResponseNorm)
RESIZABLE_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)

WIDTH_PRESERVING_TYPES = (
    nn.ReLU, nn.ReLU6, nn.ELU, nn.SiLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.Sigmoid,
    nn.Hardtanh, nn.Hardswish, nn.Hardsigmoid, nn.Softplus, nn.PReLU, nn.SELU, nn.CELU,
    nn.Dropout, nn.Dropout2d, nn.Identity, nn.MaxPool2d, nn.AvgPool2d,
    nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Flatten, nn.ZeroPad2d,
)

WIDTH_PRESERVING_FUNCTIONS = {
    torch.flatten, torch.relu, torch.sigmoid, torch.tanh,
    torch.nn.functional.relu, torch.nn.functional.relu6, torch.nn.functional.gelu,
    torch.nn.functional.silu, torch.nn.functional.leaky_relu, torch.nn.functional.elu,
    torch.nn.functional.hardswish, torch.nn.functional.hardtanh,
    torch.nn.functional.dropout, torch.nn.functional.dropout2d,
    torch.nn.functional.max_pool2d, torch.nn.functional.avg_pool2d,
    torch.nn.functional.adaptive_avg_pool2d, torch.nn.functional.pad,
}
WIDTH_PRESERVING_METHODS = {"flatten", "view", "reshape", "contiguous", "relu", "sigmoid",
                            "tanh", "clone", "to", "float", "size"}

# Element-wise binary operations: their operands must share a channel dimension
MERGING_FUNCTIONS = {operator.add, operator.iadd, operator.sub, operator.isub,
                     operator.mul, operator.imul, torch.add, torch.sub, torch.mul}
MERGING_METHODS = {"add", "add_", "sub", "sub_", "mul", "mul_"}

CHANNEL_DIMS = (1, -3)

# Reductions that may or may not preserve the channel dimension, depending on `dim`.
# Global average pooling is frequently written as `x.mean(dim=(2, 3))` (the thin-ResNet
# family used in SPECTRA's database, and torchvision's ResNet.forward), which used to be
# rejected as "unsupported op mean" and forced every layer feeding it into masking.
REDUCTION_FUNCTIONS = {torch.mean, torch.sum, torch.amax, torch.amin}
REDUCTION_METHODS = {"mean", "sum", "amax", "amin"}


@dataclass(frozen=True)
class Ref:
    """A layer that reads a group's channels, and where those channels sit in its input."""

    module: nn.Module
    offset: int   # index of the group's first channel within the layer's input
    total: int    # total channel count of that input


@dataclass
class ChannelGroup:
    """A set of tensor dimensions that must be pruned together."""

    width: int = 0
    producers: List[nn.Module] = field(default_factory=list)   # own this dimension as output
    norms: List[Ref] = field(default_factory=list)             # normalise over it
    consumers: List[Ref] = field(default_factory=list)         # read it as input
    depthwise: List[nn.Module] = field(default_factory=list)   # in and out are both this dim
    prunable: bool = True
    reason: str = ""

    def block(self, reason: str):
        if self.prunable:
            self.prunable = False
            self.reason = reason

    def absorb(self, other: "ChannelGroup"):
        self.producers.extend(other.producers)
        self.norms.extend(other.norms)
        self.consumers.extend(other.consumers)
        self.depthwise.extend(other.depthwise)
        self.width = self.width or other.width
        if not other.prunable:
            self.block(other.reason)


@dataclass(frozen=True)
class Segment:
    """One contiguous run of channels within a tensor, owned by a single group."""

    token: int
    width: int


class _Groups:
    """Union-find over channel-dimension tokens, carrying a ChannelGroup per root."""

    def __init__(self):
        self._parent: Dict[int, int] = {}
        self._data: Dict[int, ChannelGroup] = {}
        self._next = 0

    def new(self, width: int = 0) -> int:
        token = self._next
        self._next += 1
        self._parent[token] = token
        self._data[token] = ChannelGroup(width=width)
        return token

    def find(self, token: int) -> int:
        while self._parent[token] != token:
            self._parent[token] = self._parent[self._parent[token]]
            token = self._parent[token]
        return token

    def get(self, token: int) -> ChannelGroup:
        return self._data[self.find(token)]

    def union(self, a: int, b: int) -> int:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return root_a
        self._parent[root_b] = root_a
        group_a, group_b = self._data[root_a], self._data[root_b]
        if group_a.width and group_b.width and group_a.width != group_b.width:
            group_a.block("operands have different widths")  # broadcasting, not a channel match
        group_a.absorb(group_b)
        del self._data[root_b]
        return root_a

    def roots(self) -> List[ChannelGroup]:
        return [self._data[root] for root in {self.find(t) for t in self._parent}]


def _node_args(node: Node) -> List[Node]:
    """Node-valued arguments, i.e. the tensors flowing into this operation."""
    found = []
    for arg in list(node.args) + list(node.kwargs.values()):
        if isinstance(arg, Node):
            found.append(arg)
        elif isinstance(arg, (list, tuple)):
            found.extend(a for a in arg if isinstance(a, Node))
    return found


def _total_width(segments: Sequence[Segment]) -> int:
    return sum(segment.width for segment in segments)


def _reduces_only_spatial(node: Node) -> bool:
    """
    True when a reduction leaves the channel dimension addressable.

    ``x.mean(dim=(2, 3))`` is global average pooling: channels survive, so the group can
    still be resized. ``x.mean()`` collapses everything to a scalar and ``x.mean(dim=1)``
    reduces over channels; both destroy the layout and must block the group.
    """
    dim = node.kwargs.get("dim")
    if dim is None and len(node.args) > 1:
        dim = node.args[1]
    if dim is None:
        return False

    dims = dim if isinstance(dim, (list, tuple)) else (dim,)
    if not all(isinstance(d, int) for d in dims):
        return False
    return all(d not in CHANNEL_DIMS and d != 0 for d in dims)


def build_channel_groups(model: nn.Module) -> Optional[List[ChannelGroup]]:
    """
    Recover the channel-dependency groups of `model`.

    Returns:
        The groups, or None when the model cannot be symbolically traced (dynamic control
        flow), in which case no structural pruning should be attempted.
    """
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception:
        return None

    modules = dict(traced.named_modules())
    groups = _Groups()
    layout: Dict[Node, Optional[List[Segment]]] = {}

    def segments_of(node) -> Optional[List[Segment]]:
        return layout.get(node) if isinstance(node, Node) else None

    def first_input(node: Node) -> Optional[List[Segment]]:
        for arg in _node_args(node):
            found = segments_of(arg)
            if found:
                return found
        return None

    def block_inputs(node: Node, reason: str):
        for arg in _node_args(node):
            for segment in segments_of(arg) or []:
                groups.get(segment.token).block(reason)

    def register_readers(segments: Sequence[Segment], attribute: str, module: nn.Module):
        """Record `module` against each segment it reads, with that segment's offset."""
        total = _total_width(segments)
        offset = 0
        for segment in segments:
            getattr(groups.get(segment.token), attribute).append(Ref(module, offset, total))
            offset += segment.width

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            entry = groups.new()  # the network input width is fixed by the dataset
            groups.get(entry).block("model input")
            layout[node] = [Segment(entry, 0)]
            continue

        if node.op == "get_attr":
            layout[node] = None
            continue

        if node.op == "output":
            block_inputs(node, "model output")
            layout[node] = None
            continue

        incoming = first_input(node)

        if node.op == "call_module":
            module = modules.get(node.target)

            if isinstance(module, PRUNABLE_TYPES):
                is_conv = isinstance(module, nn.Conv2d)
                width = module.out_channels if is_conv else module.out_features
                in_width = module.in_channels if is_conv else module.in_features
                depthwise = is_conv and module.groups > 1 and module.groups == in_width == width
                other_grouped = is_conv and module.groups > 1 and not depthwise

                if depthwise:
                    # One filter per input channel: the output dimension *is* the input one
                    if not incoming or len(incoming) != 1:
                        produced = groups.new(width)
                        groups.get(produced).block("depthwise conv with unresolved input")
                        layout[node] = [Segment(produced, width)]
                        continue
                    group = groups.get(incoming[0].token)
                    group.depthwise.append(module)
                    group.width = group.width or width
                    layout[node] = list(incoming)
                    continue

                if incoming:
                    register_readers(incoming, "consumers", module)
                    if other_grouped:
                        for segment in incoming:
                            groups.get(segment.token).block("grouped convolution consumer")

                produced = groups.new(width)
                produced_group = groups.get(produced)
                produced_group.producers.append(module)
                if other_grouped:
                    produced_group.block("grouped convolution producer")
                layout[node] = [Segment(produced, width)]
                continue

            if isinstance(module, NORM_TYPES):
                if incoming:
                    register_readers(incoming, "norms", module)
                    if not isinstance(module, RESIZABLE_NORM_TYPES):
                        for segment in incoming:
                            groups.get(segment.token).block(
                                f"unsupported normalisation {type(module).__name__}")
                layout[node] = list(incoming) if incoming else None
                continue

            if isinstance(module, WIDTH_PRESERVING_TYPES):
                layout[node] = list(incoming) if incoming else None
                continue

            block_inputs(node, f"unknown module {type(module).__name__}")
            layout[node] = None
            continue

        if node.op in ("call_function", "call_method"):
            target = node.target
            merging = (target in MERGING_FUNCTIONS if node.op == "call_function"
                       else target in MERGING_METHODS)

            if target is torch.cat or (node.op == "call_method" and target == "cat"):
                dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
                operands = node.args[0] if node.args else []
                parts = [segments_of(arg) for arg in operands] if isinstance(operands, (list, tuple)) else None

                if dim not in CHANNEL_DIMS or parts is None or any(p is None for p in parts):
                    block_inputs(node, "concatenation along a non-channel axis")
                    layout[node] = None
                    continue

                # Channels stack in operand order, so each group keeps a known offset
                combined: List[Segment] = []
                for part in parts:
                    combined.extend(part)
                layout[node] = combined
                continue

            if merging:
                operand_layouts = [segments_of(arg) for arg in _node_args(node)]
                operand_layouts = [l for l in operand_layouts if l]
                if not operand_layouts:
                    layout[node] = None
                    continue

                reference = operand_layouts[0]
                compatible = all(len(other) == len(reference)
                                 and all(a.width == b.width for a, b in zip(other, reference))
                                 for other in operand_layouts[1:])
                if not compatible:
                    block_inputs(node, "element-wise op over mismatched channel layouts")
                    layout[node] = None
                    continue

                merged = []
                for position, segment in enumerate(reference):
                    token = segment.token
                    for other in operand_layouts[1:]:
                        token = groups.union(token, other[position].token)
                    merged.append(Segment(token, segment.width))
                layout[node] = merged
                continue

            preserving = (target in WIDTH_PRESERVING_FUNCTIONS if node.op == "call_function"
                          else target in WIDTH_PRESERVING_METHODS)
            if preserving:
                layout[node] = list(incoming) if incoming else None
                continue

            reducing = (target in REDUCTION_FUNCTIONS if node.op == "call_function"
                        else target in REDUCTION_METHODS)
            if reducing:
                if _reduces_only_spatial(node):
                    layout[node] = list(incoming) if incoming else None
                else:
                    name = getattr(target, "__name__", target)
                    block_inputs(node, f"{name} reduces the channel dimension")
                    layout[node] = None
                continue

            name = getattr(target, "__name__", target)
            block_inputs(node, f"unsupported op {name}")
            layout[node] = None
            continue

        block_inputs(node, f"unsupported node {node.op}")
        layout[node] = None

    resolved = groups.roots()

    # A layer owns exactly one output dimension, so finding it as a producer of two groups
    # means it is invoked at several points in the graph and would need contradictory edits.
    for role in ("producers", "depthwise"):
        owner: Dict[int, int] = {}
        for index, group in enumerate(resolved):
            for module in getattr(group, role):
                first = owner.setdefault(id(module), index)
                if first != index:
                    group.block(f"module reused across groups ({role})")
                    resolved[first].block(f"module reused across groups ({role})")

    # Reading several groups at once is not a conflict -- that is precisely what a layer
    # behind a concatenation does -- but a layer may read each group only once, and all of
    # its references must agree on how wide its input is.
    widths_seen: Dict[int, set] = {}
    for group in resolved:
        for role in ("consumers", "norms"):
            unique, seen_ids = [], set()
            for ref in getattr(group, role):
                key = id(ref.module)
                if key in seen_ids:
                    # cat([x, x]) would need two simultaneous edits at different offsets
                    group.block(f"channels read twice by the same layer ({role})")
                    continue
                seen_ids.add(key)
                unique.append(ref)
                widths_seen.setdefault(key, set()).add(ref.total)
            setattr(group, role, unique)

    ambiguous = {key for key, widths in widths_seen.items() if len(widths) > 1}
    if ambiguous:
        for group in resolved:
            for role in ("consumers", "norms"):
                if any(id(ref.module) in ambiguous for ref in getattr(group, role)):
                    group.block("layer reused with differing input widths")

    for group in resolved:
        if not group.producers and not group.depthwise:
            group.block("no resizable producer")

    return resolved


def group_of(groups: List[ChannelGroup], layer: nn.Module) -> Optional[ChannelGroup]:
    """The group whose output dimension `layer` produces."""
    for group in groups:
        if any(producer is layer for producer in group.producers):
            return group
        if any(module is layer for module in group.depthwise):
            return group
    return None


def coupling_ids_for_layers(layers: Sequence[nn.Module],
                            groups: Optional[List[ChannelGroup]]) -> torch.Tensor:
    """
    Integer id per layer reflecting *channel-coupling* rather than module ownership.

    Layers that must share a channel dimension under structured pruning (producers of the
    same residual/concat group, plus the norms and consumers that read that group) receive
    the same id. The trainable state encoder uses this as a Graphormer-style attention bias
    (docs/BERT_INPUT_CRITIQUE.md §5), which is the exact coupling that pruning cares about —
    unlike grouping by parent ``nn.Module``, which is only a heuristic proxy.
    """
    num_layers = len(layers)
    if not groups:
        return torch.arange(num_layers, dtype=torch.long)

    module_to_gid: Dict[int, int] = {}
    for gid, group in enumerate(groups):
        for module in list(group.producers) + list(group.depthwise):
            module_to_gid[id(module)] = gid
        for ref in list(group.norms) + list(group.consumers):
            # Producer membership wins when a module plays both roles
            module_to_gid.setdefault(id(ref.module), gid)

    ids = []
    next_singleton = len(groups)
    for layer in layers:
        key = id(layer)
        if key in module_to_gid:
            ids.append(module_to_gid[key])
        else:
            ids.append(next_singleton)
            next_singleton += 1
    return torch.tensor(ids, dtype=torch.long)


def describe(group: ChannelGroup) -> str:
    """Short human-readable summary, used in log lines."""
    names = lambda mods: ", ".join(type(m).__name__ for m in mods) or "-"
    status = "prunable" if group.prunable else f"blocked ({group.reason})"
    return (f"width={group.width} {status} | producers=[{names(group.producers)}] "
            f"norms={len(group.norms)} consumers=[{names([r.module for r in group.consumers])}]")
