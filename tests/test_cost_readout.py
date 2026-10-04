"""Cost accounting tools: scripts/cost_readout.py (login node) and scripts/bench_deploy.py (CPU path)."""
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402
import src.traj_models as traj_models  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(root):
    run = root / "job123"
    (run / "events").mkdir(parents=True)
    (run / "logs").mkdir()
    started = datetime(2026, 10, 1, 10, 0, 0)
    (run / "manifest.json").write_text(json.dumps({"environment": {
        "gpus": ["NVIDIA GeForce RTX 4090"], "hostname": "ise-4090-19.auth", "started_at": started.isoformat(),
        "spectra_env": {"SPECTRA_PROFILE": "baseline_c10_mild_traj_gonce"}}}))
    events = [{"event": "run_start", "t": 0.0}]

    def step(t, seconds, rate, ft):
        events.extend([{"event": "stage", "stage": "step.finetune", "seconds": ft, "t": t - 1},
                       {"event": "stage", "stage": "step.prune", "seconds": 0.05, "t": t - 1},
                       {"event": "stage", "stage": "step.evaluate", "seconds": 0.3, "t": t - 1},
                       {"event": "stage", "stage": "step.feature_extraction", "seconds": 0.2, "t": t - 1},
                       {"event": "step", "network": "/x/r56.pth", "compression_rate": rate, "seconds": seconds,
                        "t": t}])

    step(101.0, 101.0, 0.9, 100.0)
    step(152.0, 50.5, 0.9, 50.0)
    step(153.6, 1.0, 1, 0.0)
    events.append({"event": "eval_traj_final_ft", "network": "/x/r56.pth", "label": "val_best",
                   "minutes": 15.0, "epochs": 100, "t": 1053.0})
    events.append({"event": "run_end", "status": "ok", "seconds": 1100.0, "t": 1100.0})
    (run / "events" / "rank0.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    lines = [f"2026-10-01 10:01:{e:02d} | INFO | r0 | phase=eval_test net=r56.pth mode=eval_test step=1 layer=3 "
             f"| Epoch {e}/40: Loss = 0.1" for e in range(1, 23)]
    lines += [f"x | DEBUG | r0 | phase=eval_test net=r56.pth mode=eval_test step=2 layer=4 | Epoch {e}/40: Loss = 0.1"
              for e in range(1, 16)]
    lines += [f"x | DEBUG | r0 | phase=eval_test net=r56.pth mode=eval_test step=2 | Epoch {e}/100: Loss = 0.1"
              for e in range(1, 101)]
    lines.append("x | INFO | r0 | - | [heartbeat] up 5.0 min | idle | gpu_alloc_gb=0.4 gpu_max_alloc_gb=1.32")
    (run / "logs" / "rank0.log").write_text("\n".join(lines) + "\n")
    rows = ["timestamp, index, name, power.draw [W], utilization.gpu [%], memory.used [MiB]"]
    for second in range(0, 1101):
        stamp = (started + timedelta(seconds=second)).strftime("%Y/%m/%d %H:%M:%S.000")
        rows.append(f"{stamp}, 0, NVIDIA GeForce RTX 4090, 300.00, 95, 1200")
    (run / "gpu_samples.csv").write_text("\n".join(rows) + "\n")
    return run


def test_cost_readout_splits_a_walk(tmp_path):
    readout = _script("cost_readout")
    summary = readout.summarize(str(_fake_run(tmp_path)))
    assert summary["gpus"] == ["NVIDIA GeForce RTX 4090"] and summary["peak_alloc_gb"] == 1.32
    assert summary["run_h"] == pytest.approx(1100 / 3600, abs=1e-3)
    assert summary["energy_wh"] == pytest.approx(300 * 1100 / 3600, rel=1e-3)
    (net,) = summary["networks"]
    assert (net["steps"], net["cuts"]) == (3, 2)
    assert net["walk_min"] == pytest.approx(152.5 / 60, abs=0.05)
    assert net["ft_share"] == pytest.approx(150 / 152.5, abs=1e-3)
    assert net["between_steps_s"] == pytest.approx(0.55, abs=1e-6)
    assert net["epochs_by_budget"] == {40: 37, 100: 100}
    assert (net["finals"], net["final_min"]) == (1, 15.0)
    assert net["gpu_h"] == pytest.approx((152.5 + 900) / 3600, abs=1e-3)
    assert net["energy_wh"] == pytest.approx(300 * 1053 / 3600, rel=1e-2)


def test_cost_readout_prices_each_final_point_alone(tmp_path):
    readout = _script("cost_readout")
    run = _fake_run(tmp_path)
    path = run / "events" / "rank0.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    index = 0
    for event in events:
        if event["event"] == "step":
            event["step_index"] = index
            index += 1
        elif event["event"] == "eval_traj_final_ft":
            event.update(step=1, label="size_flop0.47", param=0.5, flop=0.47)
    events.insert(-1, {"event": "eval_traj_final_ft", "network": "/x/r56.pth", "label": "origin", "step": -1,
                       "param": 1.0, "flop": 1.0, "minutes": 1.0, "epochs": 100, "t": 1099.0})
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (net,) = readout.summarize(str(run))["networks"]
    point, origin = net["points"]
    assert (point["label"], point["step"], point["walk_min"]) == ("size_flop0.47", 1, round(152.0 / 60, 1))
    assert point["total_min"] == pytest.approx(152.0 / 60 + 15.0, abs=0.06)
    assert point["walk_wh"] == pytest.approx(300 * 152 / 3600, rel=0.02)
    assert point["final_wh"] == pytest.approx(300 * 900 / 3600, rel=0.01)
    assert (origin["walk_min"], origin["total_min"]) == (0.0, 1.0)
    assert origin["final_wh"] == pytest.approx(300 * 60 / 3600, rel=0.05) and origin["walk_wh"] == 0.0


def test_cost_readout_resolves_job_ids_under_run_roots(tmp_path, monkeypatch):
    readout = _script("cost_readout")
    _fake_run(tmp_path)
    monkeypatch.setenv("SPECTRA_RUN_ROOTS", str(tmp_path))
    assert readout.resolve("123") == str(tmp_path / "job123")
    assert readout.resolve("job123") == str(tmp_path / "job123")
    assert readout.resolve("999") is None


def test_bench_deploy_times_each_architecture_once_on_cpu(tmp_path, monkeypatch):
    bench = _script("bench_deploy")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=8).eval()
    save_dir = tmp_path / "job7" / "traj_models"
    save_dir.mkdir(parents=True)
    for label, step, suffix in (("origin", -1, ""), ("origin", -1, "__ft100"), ("size_param0.80", 5, "")):
        stem = traj_models.candidate_stem(str(save_dir), "r20.pt", label, step, suffix)
        traj_models.save_candidate(model, stem, {"step": step, "param": 1.0, "flop": 1.0},
                                   {"network": "/x/r20.pt", "label": label})
    monkeypatch.setattr(bench, "load_template",
                        lambda net_path, doc: (resnet20(num_classes=10, large_input=False, width=8), (3, 32, 32)))
    calls = []
    real = bench.bench_model
    monkeypatch.setattr(bench, "bench_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = tmp_path / "rows.jsonl"
    args = SimpleNamespace(runs=[str(tmp_path / "job7")], model=None, batches="1,2", warmup=1, min_iters=2,
                           min_seconds=0.0, max_iters=3, idle_seconds=0.0, device="cpu", threads=1, out=str(out))
    rows = bench.run(args)
    assert len(rows) == 3 and len(calls) == 1
    assert len({r["arch_key"] for r in rows}) == 1
    assert {r["label"] for r in rows} == {"origin", "size_param0.80"}
    assert {r["suffix"] for r in rows} == {"", "ft100"}
    for r in rows:
        assert set(r["by_batch"]) == {"1", "2"}
        assert r["by_batch"]["2"]["latency_ms_median"] > 0 and r["params"] == utils_params(model)
        assert r["macs"] > 0 and r["state_mb_fp16"] < r["state_mb_fp32"]
    assert len(out.read_text().splitlines()) == 3


def test_bench_deploy_times_pickled_modules_of_another_method(tmp_path):
    bench = _script("bench_deploy")
    wide, thin = tmp_path / "origin.pth", tmp_path / "pruned.pth"
    torch.save(resnet20(num_classes=10, large_input=False, width=8), wide)
    torch.save(resnet20(num_classes=10, large_input=False, width=4), thin)
    args = SimpleNamespace(runs=[], model=[f"{wide}:cifar-10:origin:other", f"{thin}:cifar-10:pruned:other"],
                           batches="1", warmup=1, min_iters=2, min_seconds=0.0, max_iters=3, idle_seconds=0.0,
                           device="cpu", threads=1, out=None)
    rows = bench.run(args)
    assert [r["label"] for r in rows] == ["origin", "pruned"] and {r["network"] for r in rows} == {"other"}
    assert rows[1]["macs"] < rows[0]["macs"] and rows[1]["params"] < rows[0]["params"]
    assert rows[1]["by_batch"]["1"]["latency_ms_median"] > 0


def test_h2h_readout_parses_a_torch_pruning_log(tmp_path):
    h2h = _script("h2h_readout")
    tag = "cifar10-global-group_sl-resnet56 INFO:"
    lines = [f"[08/28 23:20:33] {tag} Regularizing..."]
    lines += [f"[08/28 23:{21 + e // 3:02d}:{(e * 17) % 60:02d}] {tag} Epoch {e}/100, Acc=0.9{e:03d}, Val Loss=0.3, lr=0.01"
              for e in range(0, 3)]
    lines += [f"[08/29 00:37:26] {tag} Best Acc=0.9324", f"[08/29 00:37:28] {tag} Pruning...",
              f"[08/29 00:37:42] {tag} Params: 0.86 M => 0.43 M (49.81%)",
              f"[08/29 00:37:42] {tag} FLOPs: 127.12 M => 59.88 M (47.11%, 2.12X )",
              f"[08/29 00:37:42] {tag} Acc: 0.9324 => 0.8885", f"[08/29 00:37:42] {tag} Finetuning...",
              f"[08/29 00:38:01] {tag} Epoch 0/100, Acc=0.8252, Val Loss=0.5, lr=0.01",
              f"[08/29 01:07:59] {tag} Epoch 99/100, Acc=0.9383, Val Loss=0.2, lr=0.0001",
              f"[08/29 01:07:59] {tag} Best Acc=0.9389"]
    (tmp_path / "resnet56_cifar10_group_sl_2.11.log").write_text("noise\n" + "\n".join(lines) + "\n")
    rows = ["timestamp, index, name, power.draw [W]"]
    rows += [f"2026/08/29 00:{m:02d}:{s:02d}.000, 0, RTX 4090, 200.0" for m in range(37, 60) for s in range(60)]
    rows += [f"2026/08/29 01:{m:02d}:{s:02d}.000, 0, RTX 4090, 200.0" for m in range(0, 9) for s in range(60)]
    (tmp_path / "gpu_samples.csv").write_text("\n".join(rows) + "\n")
    bench = []
    for repeat, scale in ((1, 1.0), (2, 1.2), (3, 1.1)):
        for label, latency in (("origin", 2.0), ("group_sl", 1.0)):
            bench.append({"network": "depgraph_resnet56", "label": label, "params": 1, "macs": 1, "gpu": "RTX 4090",
                          "host": "n1", "by_batch": {"1": {"latency_ms_median": latency * scale,
                                                           "throughput_img_s": 1e3 / (latency * scale)}}})
    (tmp_path / "bench_r1.jsonl").write_text("\n".join(json.dumps(b) for b in bench) + "\n")
    result = h2h.readout(str(tmp_path))
    (run,) = result["runs"]
    sl, cut, ft = (run["stages"][s] for s in ("sl", "prune", "ft"))
    assert sl["minutes"] == pytest.approx(76.92, abs=0.01) and sl["best"] == 0.9324
    assert sl["energy_wh"] == pytest.approx(200 * 28 / 3600, rel=0.05)
    assert cut["minutes"] == pytest.approx(14 / 60, abs=0.01) and run["cut_acc"] == (0.9324, 0.8885)
    assert ft["minutes"] == pytest.approx(30.28, abs=0.01) and (ft["best"], ft["last"], ft["epochs"]) == (0.9389, 0.9383, 2)
    assert ft["energy_wh"] == pytest.approx(200 * 30.28 / 60, rel=0.01)
    assert (run["flop_kept"], run["speed_up"], run["param_kept"]) == (0.4711, 2.123, 0.5)
    pruned = next(c for c in result["bench"] if c["label"] == "group_sl")
    assert pruned["repeats"] == 3 and pruned["by_batch"]["1"]["speedup"] == pytest.approx(2.0)


def utils_params(model):
    return sum(p.numel() for p in model.parameters())
