"""
TRAJ candidate files that survive pickling: ``<stem>.pt`` holds the ``state_dict`` and ``<stem>.json``
the architecture (every Conv2d / Linear / BatchNorm's constructor arguments), the TRAJ point and the
``SPECTRA_*`` recipe. The live module is never pickled.

Zoo networks are built from classes in ``spectra_models_instantiation/*.py`` loaded by file path, so
``torch.save(module)`` has to import the class by name and fails (job 21726337: ``Can't pickle
<class 'resnet_chenyaofo.CifarResNet'>: import of module 'resnet_chenyaofo' failed``). A saved
candidate is restored by resizing a fresh copy of the original network to the recorded shapes
(:func:`rebuild_from_state_dict`) and loading the weights strictly.
"""
import copy
import glob
import json
import os

import torch
from torch import nn

import src.utils as utils

_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _module_spec(module):
    if isinstance(module, nn.Conv2d):
        padding = module.padding if isinstance(module.padding, str) else list(module.padding)
        return {"type": "Conv2d", "in_channels": module.in_channels, "out_channels": module.out_channels,
                "kernel_size": list(module.kernel_size), "stride": list(module.stride), "padding": padding,
                "dilation": list(module.dilation), "groups": module.groups, "bias": module.bias is not None,
                "padding_mode": module.padding_mode}
    if isinstance(module, nn.Linear):
        return {"type": "Linear", "in_features": module.in_features, "out_features": module.out_features,
                "bias": module.bias is not None}
    if isinstance(module, _NORMS):
        return {"type": type(module).__name__, "num_features": module.num_features, "eps": module.eps,
                "momentum": module.momentum, "affine": module.affine,
                "track_running_stats": module.track_running_stats}
    return None


def module_arch(model):
    """``{module name: constructor arguments}`` for every Conv2d / Linear / BatchNorm in ``model``."""
    arch = {}
    for name, module in model.named_modules():
        spec = _module_spec(module)
        if spec is not None:
            arch[name] = spec
    return arch


def _build(spec):
    kind = spec["type"]
    if kind == "Conv2d":
        padding = spec["padding"] if isinstance(spec["padding"], str) else tuple(spec["padding"])
        return nn.Conv2d(spec["in_channels"], spec["out_channels"], tuple(spec["kernel_size"]),
                         stride=tuple(spec["stride"]), padding=padding, dilation=tuple(spec["dilation"]),
                         groups=spec["groups"], bias=spec["bias"], padding_mode=spec["padding_mode"])
    if kind == "Linear":
        return nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
    return getattr(nn, kind)(spec["num_features"], eps=spec["eps"], momentum=spec["momentum"],
                             affine=spec["affine"], track_running_stats=spec["track_running_stats"])


def _set_module(root, name, module):
    parent = root
    *path, child = name.split(".")
    for part in path:
        parent = parent._modules[part]
    parent._modules[child] = module


def rebuild_from_state_dict(model, state_dict, arch):
    """
    Resize ``model`` — a fresh instance of the original network — to the saved ``arch`` and load
    ``state_dict`` strictly. Returns ``model`` (edited in place).
    """
    live = dict(model.named_modules())
    missing = sorted(set(arch) - set(live))
    if missing:
        raise KeyError(f"saved arch names modules the network does not have: {missing[:5]}")
    for name, spec in arch.items():
        if _module_spec(live[name]) != spec:
            if not name:
                raise ValueError("the root module itself changed shape; rebuild it from the arch instead")
            _set_module(model, name, _build(spec))
    model.load_state_dict(state_dict, strict=True)
    return model


def candidate_stem(save_dir, net_name, label, step, suffix=""):
    return os.path.join(save_dir, f"{net_name}__{label}__step{int(step)}{suffix}")


def save_candidate(model, stem, point=None, meta=None):
    """
    Write ``<stem>.pt`` (``state_dict``) and ``<stem>.json`` (arch, point, recipe). Never raises: a save
    must not cost the fine-tune that follows it. Returns ``stem`` or ``None``.
    """
    try:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, stem + ".pt")
        doc = dict(meta or {})
        doc["point"] = dict(point) if point else None
        doc["arch"] = module_arch(model)
        doc["params"] = int(sum(p.numel() for p in model.parameters()))
        doc["recipe_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("SPECTRA_")}
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, default=str)
        return stem
    except Exception as error:  # noqa: BLE001 - a failed save is logged, never fatal
        utils.print_flush(f"[eval] TRAJ save failed for {os.path.basename(stem)}: "
                          f"{type(error).__name__}: {error}; continuing")
        return None


def load_candidate(template, stem):
    """``(model, doc)``: a deep copy of ``template`` (the original network) rebuilt into ``<stem>``."""
    with open(stem + ".json", encoding="utf-8") as fh:
        doc = json.load(fh)
    state = torch.load(stem + ".pt", map_location="cpu", weights_only=True)
    model = rebuild_from_state_dict(copy.deepcopy(template).cpu(), state, doc["arch"])
    return model, doc


def load_candidates(save_dir, net_name, template):
    """
    Pre-fine-tune candidates of one network saved by a TRAJ walk, as ``{label: {"point", "model",
    "key"}}`` (the shape ``_run_final_ft`` takes). Fine-tuned copies (``…__ft<E>``) are skipped.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(save_dir, f"{glob.escape(net_name)}__*__step*.json"))):
        stem = path[:-len(".json")]
        tail = os.path.basename(stem)[len(net_name) + 2:]
        label, _, rest = tail.rpartition("__step")
        if not label or not rest.lstrip("-").isdigit() or not os.path.exists(stem + ".pt"):
            continue
        model, doc = load_candidate(template, stem)
        if doc.get("point") is None:
            continue
        out[label] = {"point": doc["point"], "model": model, "key": None}
    return out


def reinit_parameters(model):
    """
    PyTorch's default init on every module that has ``reset_parameters`` (BatchNorm stats included),
    not the zoo constructor's own init. ``origin+scratch`` is re-initialised the same way, so the
    scratch rows and their control share it.
    """
    count = 0
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
            count += 1
    return count
