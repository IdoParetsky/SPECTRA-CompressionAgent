# SPECTRA Slurm experiment cheat sheet

Local Cursor workspace: `C:\SPECTRA-CompressionAgent`  
Cluster repo: `/home/paretsky/SPECTRA-CompressionAgent`  
Conda Python: `/home/paretsky/.conda/envs/spectra/bin/python`

---

## 0) One-time (already done on this PC)

| Item | Where |
|------|--------|
| SSH login | `ssh bgu-slurm` |
| SSH GPU (auto HostName) | `ssh bgu-slurm-gpu` |
| Secrets | `%USERPROFILE%\.spectra\bgu-slurm.secrets.md` |
| Resolve / sync / run | `scripts\slurm_*.ps1` |

---

## 1) Allocate a GPU session (login node)

**Terminal A** (keep open for the whole session):

```powershell
ssh bgu-slurm
```

On the login node:

```bash
cd /home/paretsky/SPECTRA-CompressionAgent
bash scripts/run_spectra_sinteractive.sh
# equivalent:
# sinteractive --gpu 1 --time 0-3:00:00 --mem 80
```

Wait until you land on a compute host (e.g. `paretsky@cs-1080-05...`).  
Leave this shell open — closing it can end the job.

**Optional check on the node:**

```bash
nvidia-smi
bash scripts/check_gpu.sh   # after helpers are synced
```

If no CUDA: `scancel $SLURM_JOB_ID` → run `sinteractive` again.

---

## 2) Experiment loop (local Cursor → cluster run → local log)

Do this from **PowerShell in the local repo** (`C:\SPECTRA-CompressionAgent`).

### A. Finish code locally

```powershell
cd C:\SPECTRA-CompressionAgent
git status
git add -p          # or git add <files>
git commit -m "Your experiment-ready message"
git push origin master
```

Uncommitted edits do **not** reach the cluster via sync.

### B. Point SSH at the current GPU node

```powershell
.\scripts\slurm_resolve_node.ps1
```

- 1 running job → auto-detects from login `squeue` and updates `bgu-slurm-gpu`
- 0 / many jobs → Windows popup to type node or pick index
- Force popup: `.\scripts\slurm_resolve_node.ps1 -Prompt`
- Manual: `.\scripts\slurm_resolve_node.ps1 -Node cs-1080-05`

### C. Sync committed code to the cluster

```powershell
.\scripts\slurm_sync.ps1
```

Pulls `origin/master` into `/home/paretsky/SPECTRA-CompressionAgent` and redeploys `scripts/*.sh`.

### D. Run the experiment on the GPU node + save log locally

```powershell
.\scripts\slurm_run.ps1
```

That script:

1. Resolves the node (same as B)  
2. Syncs (same as C) unless `-SkipSync`  
3. On the GPU node: `check_gpu.sh` → `run_agent.sh`  
4. Tees stdout/stderr to:

`runs/slurm_logs/run_YYYYMMDD_HHMMSS_<node>.log`

Useful variants:

```powershell
.\scripts\slurm_run.ps1 -SkipSync              # code already synced
.\scripts\slurm_run.ps1 -PromptNode            # force node popup
.\scripts\slurm_run.ps1 -Node cs-1080-05
.\scripts\slurm_run.ps1 -ExtraArgs "--passes","1","--rollout_limit","5"
```

### E. Feed the log back into Cursor

1. Open the newest file under `runs/slurm_logs/` (or `@`-mention it in chat).  
2. Ask Cursor to diagnose / plan next steps from that log.  
3. Edit → commit → push → resolve → run again (back to A).

---

## 3) Default runner knobs

Set in `scripts/run_agent.sh` / `.vscode/launch.json` (override with `-ExtraArgs` or env):

| Flag | Default |
|------|---------|
| `--input` | `/home/paretsky/initial_input_no_imagenet.json` |
| `--database` | `/home/paretsky/initial_database_no_imagenet.json` |
| `--passes` | `2` |
| `--rollout_limit` | `10` |
| `--seed` | `42` |
| `--n_splits` | `3` |
| `--save_pruned_checkpoints` | `True` |
| Working dir | `/home/paretsky/SPECTRA-CompressionAgent` |
| Python | `/home/paretsky/.conda/envs/spectra/bin/python` |

---

## 4) End the session

On the login/compute shell:

```bash
exit                 # leave sinteractive
# or from login:
scancel <JOBID>      # squeue -u paretsky
```

---

## 5) Minimal “when do I run what?”

| When | Command |
|------|---------|
| Start GPU time | `ssh bgu-slurm` → `bash scripts/run_spectra_sinteractive.sh` |
| After each new allocation | `.\scripts\slurm_resolve_node.ps1` (or rely on `slurm_run`) |
| Code ready to experiment | `git commit` + `git push origin master` |
| Update cluster tree | `.\scripts\slurm_sync.ps1` |
| Launch run + local log | `.\scripts\slurm_run.ps1` |
| Debug with Cursor | `@runs/slurm_logs/<latest>.log` in chat |
| No GPU / bad node | `scancel $SLURM_JOB_ID` → new `sinteractive` |
| Job still running, re-run only | `.\scripts\slurm_run.ps1 -SkipSync` |

---

## 7) Overnight / VPN expiry (important)

Your Check Point client shows a hard **Expiration** countdown (BGU policy). Neither Cursor nor the client can turn that into “never expires.” When it hits zero, the tunnel drops and **this PC loses SSH** until you reconnect (often with a password / GUI prompt).

### What actually survives overnight

| Approach | Survives VPN drop? | Notes |
|----------|--------------------|--------|
| `sinteractive` + live SSH / `slurm_run.ps1` | **No** (access lost; interactive allocation may die) | Fine while you are awake |
| **`sbatch` via `scripts/submit.sh`** | **Yes** (job keeps running on the cluster) | Preferred for sleep / long runs |
| `tmux` inside `sinteractive` | Partial | Process may continue; still fragile vs VPN + allocation policy |

**For runs while you sleep, submit a batch job** (after commit/push/sync):

```powershell
.\scripts\slurm_sync.ps1
ssh bgu-slurm "cd /home/paretsky/SPECTRA-CompressionAgent && bash scripts/submit.sh medium"
# profiles: smoke | medium | full | probe | diag | ...
```

Logs land on the cluster under:

- `/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/spectra_<JOBID>.out`
- `/home/paretsky/SPECTRA-CompressionAgent/runs/job<JOBID>/`

Next morning (VPN back up):

```powershell
ssh bgu-slurm "squeue -u paretsky; sacct -u paretsky --starttime=now-1day --format=JobID,State,Elapsed,ExitCode -n"
scp bgu-slurm:/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/spectra_<JOBID>.out runs/slurm_logs/
```

Then `@runs/slurm_logs/spectra_<JOBID>.out` in Cursor.

### VPN hygiene (reduces mid-night *idle* drops; does not remove Expiration)

1. **VPN Options** → enable anything like Always-Connect / auto-reconnect if present; avoid “disconnect on idle/lock” if offered.  
2. Windows: **Settings → System → Power** → keep the PC awake (or “Stay on when plugged in”); disable sleep overnight.  
3. Expiration still wins: after ~hours you must re-auth. Do **not** rely on VPN staying up for the experiment itself — rely on **`sbatch`**.

Do **not** leave your VPN password in a scheduled reconnect script unless you accept that risk; BGU often still wants an interactive login.

---


```powershell
ssh bgu-slurm
# ... sinteractive ...
```

```powershell
git push origin master
ssh bgu-slurm "cd /home/paretsky/SPECTRA-CompressionAgent && git pull --ff-only origin master"
.\scripts\slurm_resolve_node.ps1
ssh bgu-slurm-gpu "cd /home/paretsky/SPECTRA-CompressionAgent && bash scripts/check_gpu.sh && bash scripts/run_agent.sh" `
  2>&1 | Tee-Object runs\slurm_logs\manual_run.log
```
