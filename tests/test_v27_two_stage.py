"""CPU tests for tree_v27 (19 Oct): the two eval flags of a two-stage plan (D10), both default off.

    SPECTRA_EVAL_SAVE_FINAL=<dir>     stage 1 saves the recovered pruned model right after its final fine-tune
    SPECTRA_EVAL_ORIGIN_FROM=<stem>   stage 2 rebuilds the config's network into that model before anything reads it

No dataset, tiny nets, no GPU. The runner's hooks are exercised directly: ``_save_final_model`` (what ``_final_ft``
calls) and ``_origin_from_saved`` (what ``__main__`` wraps ``utils.parse_input_argument`` with).

    python -m pytest tests/test_v27_two_stage.py -v
"""

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.traj_models as traj_models  # noqa: E402
import src.utils as utils  # noqa: E402
from tests.test_v9c_traj_models import _Env, _candidates, _point, _quiet  # noqa: E402

V27_KEYS = ("SPECTRA_EVAL_SAVE_FINAL", "SPECTRA_EVAL_ORIGIN_FROM", "SPECTRA_SEED", "SPECTRA_EVAL_FINAL_FT_ORIGIN",
            "SPECTRA_EVAL_FINAL_FT_SCRATCH", "SPECTRA_EVAL_SAVE_TRAJ_MODELS", "SPECTRA_EVAL_POLICY",
            "SPECTRA_ALLOC_KIND", "SPECTRA_ALLOC_BUDGET", "SPECTRA_EVAL_SIZE_MATCH", "SPECTRA_FIXED_TARGET_METRIC",
            "SPECTRA_EVAL_FINAL_FT_EPOCHS", "SPECTRA_EVAL_FINAL_FT_SELECT", "SPECTRA_EVAL_FINAL_FT_KD",
            "SPECTRA_VAL_FROM_TEST", "SPECTRA_SPLIT_SEED", "SPECTRA_VAL_TEST_FRACTION")
NET = "/nets/tiny.pt"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in V27_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------- a tiny net, pruned by hand

def _tiny(width=8):
    """Conv-BN-ReLU twice, pool, Linear; ``width`` is the first conv's channels, the layer pruned by hand."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
        nn.Conv2d(width, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 4))
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):           # non-trivial buffers, so the round trip checks them
                module.running_mean.normal_()
                module.running_var.uniform_(0.5, 1.5)
                module.num_batches_tracked.fill_(7)
    return net


def _prune_by_hand(model, keep):
    """Keep channels ``keep`` of the first conv (and its BN), slice the next conv's input: a structural cut."""
    pruned = copy.deepcopy(model)
    conv0, bn0, conv1 = pruned[0], pruned[1], pruned[3]
    new0 = nn.Conv2d(3, len(keep), 3, padding=1)
    newb = nn.BatchNorm2d(len(keep))
    new1 = nn.Conv2d(len(keep), conv1.out_channels, 3, padding=1)
    with torch.no_grad():
        new0.weight.copy_(conv0.weight[keep])
        new0.bias.copy_(conv0.bias[keep])
        newb.weight.copy_(bn0.weight[keep])
        newb.bias.copy_(bn0.bias[keep])
        newb.running_mean.copy_(bn0.running_mean[keep])
        newb.running_var.copy_(bn0.running_var[keep])
        newb.num_batches_tracked.copy_(bn0.num_batches_tracked)
        new1.weight.copy_(conv1.weight[:, keep])
        new1.bias.copy_(conv1.bias)
    pruned[0], pruned[1], pruned[3] = new0, newb, new1
    return pruned.eval()


def _loader(n=4):
    return DataLoader(TensorDataset(torch.zeros(n, 3, 8, 8), torch.zeros(n, dtype=torch.long)), batch_size=2)


def _stage1_env(**over):
    """What ``_save_final_model`` reads off the eval env after a walk: seed, loader, origin totals, target."""
    fields = dict(conf=SimpleNamespace(seed=42), train_loader=_loader(), original_params=1234,
                  original_flops=5678.0, target_keep=0.5)
    fields.update(over)
    return SimpleNamespace(**fields)


def _save(runner, tmp_path, pruned, net_path=NET, **over):
    env = _stage1_env(**over)
    stem = traj_models.final_stem(str(tmp_path), os.path.basename(net_path), 42)
    saved = runner._save_final_model(env, net_path, "size_param0.50", _point(3, 0.5, -1.0), pruned, 60,
                                     0.81, 0.91, stem)
    assert saved == stem
    return stem


# ---------------------------------------------------------------- flag helpers

def test_two_stage_flags_default_off(monkeypatch):
    assert fortify.eval_save_final() == "" and fortify.eval_origin_from() == ""
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_FINAL", " /tmp/final ")
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", " /tmp/final/net.pt__seed42__final ")
    assert fortify.eval_save_final() == "/tmp/final"
    assert fortify.eval_origin_from() == "/tmp/final/net.pt__seed42__final"


def test_final_stem_and_the_path_helpers(tmp_path):
    stem = traj_models.final_stem(str(tmp_path), "net.pt", 42)
    assert os.path.basename(stem) == "net.pt__seed42__final"
    assert traj_models.same_network_path("/nets/./a.pt", "/nets/a.pt")
    assert traj_models.same_network_path("nets/a.pt", os.path.abspath("nets/a.pt"))
    assert not traj_models.same_network_path("/nets/a.pt", "/nets/b.pt")
    assert not traj_models.same_network_path(None, "/nets/a.pt")
    assert traj_models.saved_origin_network({"origin": {"network": "/x.pt"}, "network": "/y.pt"}) == "/x.pt"
    assert traj_models.saved_origin_network({"network": "/y.pt"}) == "/y.pt"       # an older TRAJ candidate
    assert traj_models.saved_origin_network({"point": None}) is None


def test_run_seed_reads_the_conf_then_the_env(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    assert runner._run_seed(SimpleNamespace(conf=SimpleNamespace(seed=7))) == 7
    assert runner._run_seed(_Env()) == 0
    monkeypatch.setenv("SPECTRA_SEED", "43")
    assert runner._run_seed(_Env()) == 43


# ---------------------------------------------------------------- round trip: stage 1's saver -> stage 2's rebuild

def test_round_trip_a_hand_pruned_net_through_a_fresh_template(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    template = _tiny()
    pruned = _prune_by_hand(template, [0, 2, 5, 7])
    assert sum(p.numel() for p in pruned.parameters()) < sum(p.numel() for p in template.parameters())
    stem = _save(runner, tmp_path, pruned)
    assert os.path.exists(stem + ".pt") and os.path.exists(stem + ".json")
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", stem)
    loaders = (_loader(), _loader(), _loader())
    fresh = _tiny()
    input_dict = {NET: (fresh, loaders)}
    rng = torch.get_rng_state()
    out = runner._origin_from_saved(input_dict)
    assert out is input_dict and out[NET][1] is loaders                 # the same dict, the same loaders
    rebuilt = out[NET][0]
    assert rebuilt is not fresh and fresh[0].out_channels == 8          # the template was copied, not edited
    assert traj_models.module_arch(rebuilt) == traj_models.module_arch(pruned)
    assert set(rebuilt.state_dict()) == set(pruned.state_dict())
    assert all(torch.equal(rebuilt.state_dict()[k], pruned.state_dict()[k]) for k in pruned.state_dict())
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(rebuilt.eval()(x), pruned(x))
    assert not torch.equal(fresh.eval()(x), pruned(x))
    assert torch.equal(torch.get_rng_state(), rng)                      # the input-shape read is undone
    assert sum(ln.startswith("[eval] origin from ") for ln in lines) == 1          # one line, no split warning
    line = next(ln for ln in lines if ln.startswith("[eval] origin from "))
    params = sum(p.numel() for p in pruned.parameters())
    flops = utils.calc_flops(pruned, (3, 8, 8))
    assert line.startswith(f"[eval] origin from {stem}: params {params} FLOPs {flops:.4g} (stage-1 meta: "
                           f"size_param0.50 tiny.pt seed 42 e60 actor/None params target 0.5 | ")
    assert "params x0.500 of 1234 | FLOPs x0.500 of 5678 | acc test 0.8100 val 0.9100 | " in line
    assert line.endswith("origin test 0.8000 val 0.9000)")


def test_the_meta_round_trips(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    monkeypatch.setenv("SPECTRA_SEED", "42")
    monkeypatch.setenv("SPECTRA_EVAL_POLICY", "alloc")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "param:0.5")
    pruned = _prune_by_hand(_tiny(), [0, 2, 5, 7])
    rng = torch.get_rng_state()
    stem = _save(runner, tmp_path, pruned)
    assert torch.equal(torch.get_rng_state(), rng)
    doc = traj_models.load_doc(stem)
    params = sum(p.numel() for p in pruned.parameters())
    flops = utils.calc_flops(pruned, (3, 8, 8))
    point = _point(3, 0.5, -1.0)                                        # what _save handed the saver
    assert doc["network"] == NET and doc["label"] == "size_param0.50" and doc["candidate"] == "final"
    assert doc["origin"] == {"network": NET, "params": 1234, "flops": 5678.0, "test_acc": 0.8, "val_acc": 0.9}
    assert doc["final"] == {"params": params, "flops": flops, "param_ratio": 0.5, "flop_ratio": 0.5,
                            "test_acc": 0.81, "val_acc": 0.91, "walk_test_acc": point["test_acc"],
                            "walk_val_acc": point["val_acc"], "step": 3, "epochs": 60}
    assert doc["seed"] == 42 and doc["spectra_seed"] == "42"
    assert doc["alloc"] == {"policy": "alloc", "kind": "agent", "budget": "params", "target": 0.5,
                            "target_metric": "param", "size_match": ["param", 0.5]}
    assert doc["point"]["step"] == 3 and doc["params"] == params and doc["arch"] == traj_models.module_arch(pruned)
    assert doc["recipe_env"]["SPECTRA_ALLOC_KIND"] == "agent"           # save_candidate's own fields are kept
    assert traj_models.saved_origin_network(doc) == NET
    assert lines == [f"[eval] saved final model to {stem} | size_param0.50 tiny.pt step=3 | "
                     f"params x0.500 ({params} of 1234) | FLOPs x0.500 ({flops:.4g} of 5678) | "
                     f"acc test 0.8100 val 0.9100 | origin test 0.8000 val 0.9000 | e60 seed 42"]


def test_a_wrong_origin_path_in_the_meta_raises(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    _quiet(monkeypatch, runner)
    pruned = _prune_by_hand(_tiny(), [0, 2, 5, 7])
    stem = _save(runner, tmp_path, pruned, net_path="/nets/other.pt")
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", stem)
    with pytest.raises(ValueError, match="saved from origin '/nets/other.pt'; the config's network is '/nets/tiny.pt'"):
        runner._origin_from_saved({NET: (_tiny(), None)})
    with pytest.raises(ValueError, match="one network per job"):
        runner._origin_from_saved({NET: (_tiny(), None), "/nets/other.pt": (_tiny(), None)})
    with pytest.raises(ValueError, match="no --input network"):
        runner._origin_from_saved(None)
    with pytest.raises(ValueError, match="no --input network"):
        runner._origin_from_saved({})
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", str(tmp_path / "missing.pt__seed42__final"))
    with pytest.raises(FileNotFoundError, match="did stage 1 run with SPECTRA_EVAL_SAVE_FINAL"):
        runner._origin_from_saved({NET: (_tiny(), None)})


def test_a_split_that_differs_from_stage_1_is_said_once(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    pruned = _prune_by_hand(_tiny(), [0, 2, 5, 7])
    monkeypatch.setenv("SPECTRA_SPLIT_SEED", "1")                       # stage 1's split, kept in recipe_env
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    stem = _save(runner, tmp_path, pruned)
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", stem)
    lines.clear()
    runner._origin_from_saved({NET: (_tiny(), None)})
    assert not any("WARNING" in ln for ln in lines)                     # same split: no warning
    monkeypatch.delenv("SPECTRA_SPLIT_SEED")
    lines.clear()
    runner._origin_from_saved({NET: (_tiny(), None)})
    warnings = [ln for ln in lines if "WARNING split flags differ" in ln]
    assert len(warnings) == 1 and "{'SPECTRA_SPLIT_SEED': ('1', None)}" in warnings[0]


def test_an_arch_that_does_not_rebuild_strictly_raises(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    _quiet(monkeypatch, runner)
    pruned = _prune_by_hand(_tiny(), [0, 2, 5, 7])
    stem = _save(runner, tmp_path, pruned)
    monkeypatch.setenv("SPECTRA_EVAL_ORIGIN_FROM", stem)
    other = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(8, 4))
    with pytest.raises(RuntimeError, match="does not rebuild strictly into tiny.pt: KeyError"):
        runner._origin_from_saved({NET: (other, None)})
    state = torch.load(stem + ".pt", weights_only=True)
    state.pop("8.bias")
    torch.save(state, stem + ".pt")
    with pytest.raises(RuntimeError, match="does not rebuild strictly into tiny.pt: RuntimeError"):
        runner._origin_from_saved({NET: (_tiny(), None)})


# ---------------------------------------------------------------- the runner's wiring

def test_wiring_is_a_noop_with_the_flags_unset(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    fresh = _tiny()
    input_dict = {NET: (fresh, None)}
    assert runner._origin_from_saved(input_dict) is input_dict and input_dict[NET][0] is fresh
    assert runner._origin_from_saved(None) is None

    def boom(*_a, **_k):
        pytest.fail("the final save ran with SPECTRA_EVAL_SAVE_FINAL unset")

    monkeypatch.setattr(runner, "_save_final_model", boom)
    monkeypatch.setattr(runner, "_final_stems", boom)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0), "origin": _point(-1, 1.0, 0.0)},
                         _candidates(), 10)
    assert [r["tag"] for r in env.runs] == ["final FT val_best", "final FT size_param0.80", "final FT origin"]
    assert not any("saved final model" in ln or "origin from" in ln for ln in lines)
    assert not list(tmp_path.iterdir())


def test_save_final_writes_the_first_pruned_candidate_and_never_the_origin(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    final_dir = tmp_path / "final"
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_FINAL", str(final_dir))
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    monkeypatch.setenv("SPECTRA_SEED", "43")
    env = _Env()
    runner._run_final_ft(env, "/nets/net.pt", {"val_best": _point(5, 0.7, -9.0), "origin": _point(-1, 1.0, 0.0)},
                         _candidates(), 10)
    assert [r["tag"] for r in env.runs] == ["final FT val_best", "final FT size_param0.80", "final FT origin"]
    assert sorted(os.listdir(final_dir)) == [
        "net.pt__seed43__final.json", "net.pt__seed43__final.pt",
        "net.pt__seed43__final__size_param0.80.json", "net.pt__seed43__final__size_param0.80.pt"]
    doc = traj_models.load_doc(str(final_dir / "net.pt__seed43__final"))
    assert doc["label"] == "val_best" and doc["seed"] == 43 and doc["final"]["epochs"] == 10
    assert doc["final"]["test_acc"] == 0.75 and doc["final"]["val_acc"] == 0.75   # the stub handler's score
    assert doc["final"]["param_ratio"] == 0.7 and doc["origin"]["params"] is None  # no totals on the stub env
    assert doc["final"]["flops"] == 1.0                                            # Linear(1, 1): one MAC
    state = torch.load(final_dir / "net.pt__seed43__final.pt", weights_only=True)
    assert torch.equal(state["weight"], torch.tensor([[5.0]]))                     # val_best's recovered copy
    saved = [ln for ln in lines if ln.startswith("[eval] saved final model to ")]
    assert len(saved) == 2
    assert saved[0] == (f"[eval] saved final model to {final_dir / 'net.pt__seed43__final'} | val_best net.pt step=5 | "
                        f"params x0.700 (1 of n/a) | FLOPs x0.700 (1 of n/a) | acc test 0.7500 val 0.7500 | "
                        f"origin test 0.8000 val 0.9000 | e10 seed 43")
    assert "__final__size_param0.80 | size_param0.80 net.pt step=3 |" in saved[1]
    # the TRAJ reader of SPECTRA_EVAL_FINAL_FT_FROM never lists a final model
    assert traj_models.load_candidates(str(final_dir), "net.pt", nn.Linear(1, 1, bias=False)) == {}


def test_save_final_skips_a_repeated_point_and_scratch_copies(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    final_dir = tmp_path / "final"
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_FINAL", str(final_dir))
    candidates = _candidates()
    candidates["size_param0.70"] = dict(candidates["val_best"])               # the one-shot line: same step
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "1")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, candidates, 10)
    assert [r["tag"] for r in env.runs] == ["final FT val_best", "final FT val_best+scratch",
                                            "final FT size_param0.80", "final FT size_param0.80+scratch"]
    assert sorted(os.listdir(final_dir)) == [
        "net.pt__seed0__final.json", "net.pt__seed0__final.pt",
        "net.pt__seed0__final__size_param0.80.json", "net.pt__seed0__final__size_param0.80.pt"]
    assert sum(ln.startswith("[eval] saved final model to ") for ln in lines) == 2
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "only")
    other = tmp_path / "only"
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_FINAL", str(other))
    runner._run_final_ft(_Env(), "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    assert os.path.isdir(other) and not os.listdir(other)                       # scratch copies are never saved


def test_save_final_never_costs_the_eval(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    pruned = _prune_by_hand(_tiny(), [0, 2, 5, 7])
    stem = traj_models.final_stem(str(tmp_path), "tiny.pt", 42)
    monkeypatch.setattr(runner.traj_models, "save_candidate", lambda *a, **k: None)
    assert runner._save_final_model(_stage1_env(), NET, "val_best", _point(3, 0.5, -1.0), pruned, 60, 0.8, 0.9,
                                    stem) is None
    assert not any("saved final model" in ln for ln in lines)

    def boom(*_a, **_k):
        raise RuntimeError("no such attribute")

    monkeypatch.setattr(runner.utils, "calc_num_parameters", boom)
    assert runner._save_final_model(_stage1_env(), NET, "val_best", _point(3, 0.5, -1.0), pruned, 60, 0.8, 0.9,
                                    stem) is None
    assert lines[-1] == "[eval] final model save failed for val_best tiny.pt: RuntimeError: no such attribute; continuing"
    assert not os.path.exists(stem + ".pt")
