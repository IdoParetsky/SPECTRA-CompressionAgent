# SPECTRA SLURM / Cursor notes

**Day-to-day commands:** see [`docs/SLURM_EXPERIMENT_CHEATSHEET.md`](SLURM_EXPERIMENT_CHEATSHEET.md)  
**Environment / GPU policy:** see [`SLURM_GPU.md`](../SLURM_GPU.md)

## PyCharm → Cursor mapping

| PyCharm | Cursor equivalent |
|---|---|
| SSH interpreter → node IP | Local edit + `slurm_resolve_node.ps1` / `slurm_run.ps1` (or Remote-SSH to `bgu-slurm-gpu`) |
| System Python path | `/home/paretsky/.conda/envs/spectra/bin/python` |
| Run configuration args | `.vscode/launch.json` or `scripts/run_agent.sh` |
| Working directory | `/home/paretsky/SPECTRA-CompressionAgent` |
| Path mapping | Not needed for the recommended local+git+remote-run loop |
| Allocate GPUs | `ssh bgu-slurm` → `bash scripts/run_spectra_sinteractive.sh` |
| Capture run output | `.\scripts\slurm_run.ps1` → `runs/slurm_logs/*.log` |

Obsolete in Cursor: PyCharm path mappings, “New SSH Interpreter” UI, mandatory `pydevd_pycharm` (gated behind `SPECTRA_PYDEVD=1`).
