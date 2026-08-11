# SPECTRA compute: local Cursor + SLURM GPU runs

**Local Windows** (`C:\SPECTRA-CompressionAgent`) = edit / git only.  
**BGU SLURM GPU node** = training / evaluation with CUDA PyTorch.

Do **not** install a full ML/CUDA stack on Windows for experiments.

## Connection

| Item | Value |
|------|--------|
| Login SSH | `ssh bgu-slurm` → `paretsky@slurm.bgu.ac.il` |
| GPU SSH | `ssh bgu-slurm-gpu` (ProxyJump; update `HostName` after each `sinteractive`) |
| Key | `~/.ssh/id_ed25519_bgu_slurm` (passwordless to login) |
| Secrets (local only) | `%USERPROFILE%\.spectra\bgu-slurm.secrets.md` |
| Conda Python | `/home/paretsky/.conda/envs/spectra/bin/python` |
| Remote repo | `/home/paretsky/SPECTRA-CompressionAgent` |
| Default input | `/home/paretsky/initial_input_no_imagenet.json` |
| Default database | `/home/paretsky/initial_database_no_imagenet.json` |

## Everyday loop (recommended)

Full step-by-step: **[`docs/SLURM_EXPERIMENT_CHEATSHEET.md`](docs/SLURM_EXPERIMENT_CHEATSHEET.md)**

1. Edit in Cursor on the local clone.
2. **Commit** + **push** changes you want on the cluster.
3. Keep a login-node terminal for allocation (`sinteractive` / `run_spectra_sinteractive.sh`).
4. From PowerShell: `.\scripts\slurm_run.ps1` (resolves node, syncs, runs, writes `runs/slurm_logs/*.log`).
5. `@`-mention that log in Cursor for debugging / next steps.
6. If `check_gpu` fails: `scancel $SLURM_JOB_ID` and allocate again.

### On the node (alternative)

```bash
cd /home/paretsky/SPECTRA-CompressionAgent
git pull --ff-only origin master
bash scripts/check_gpu.sh
bash scripts/run_agent.sh
```

## Runner defaults (`scripts/run_agent.sh`)

```text
--input /home/paretsky/initial_input_no_imagenet.json
--database /home/paretsky/initial_database_no_imagenet.json
--passes 2
--rollout_limit 10
--seed 42
--n_splits 3
--save_pruned_checkpoints True
```

Override via env (`SPECTRA_INPUT`, `PASSES`, …) or extra CLI args after `run_agent.sh`.

## Edit later

| What | Where |
|------|--------|
| SSH hosts / node hostname | `%USERPROFILE%\.ssh\config` (usually auto via `slurm_resolve_node.ps1`) |
| Force node popup | `.\scripts\slurm_resolve_node.ps1 -Prompt` or `.\scripts\slurm_run.ps1 -PromptNode` |
| Password / notes | `%USERPROFILE%\.spectra\bgu-slurm.secrets.md` |
| Runner args (Cursor debug) | `.vscode/launch.json` |
| Runner args (cluster script) | `scripts/run_agent.sh` |
| Allocation defaults | `scripts/run_spectra_sinteractive.sh` or env `GPUS`/`TIME`/`MEM` |
| Workflow for the agent | `.cursor/rules/bgu-slurm-workflow.mdc` |

## Verify CUDA on the allocated node

```bash
/home/paretsky/.conda/envs/spectra/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `True` and `device_count >= 1`. Login nodes correctly report `False`.
