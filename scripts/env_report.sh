#!/usr/bin/env bash
# Environment report for the BGU SLURM account: interpreter, CUDA, key packages, data paths.
#
# Run locally on a node/login shell:   bash scripts/env_report.sh
# Or from Windows without quoting pain: Get-Content scripts/env_report.sh | ssh bgu-slurm 'bash -s'

PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
REPO_DIR="${SPECTRA_REPO_DIR:-$HOME/SPECTRA-CompressionAgent}"

echo "=== host ==="
hostname
date -Is
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"

echo "=== python ==="
"$PYTHON" - <<'PY'
import platform, sys
print("python", platform.python_version())
try:
    import torch
    print("torch", torch.__version__, "cuda_build", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch import failed:", exc)
for mod in ("torchvision", "transformers", "sklearn", "pandas", "numpy", "tensorboard"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as exc:
        print(mod, "MISSING", exc)
PY

echo "=== gpu ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || echo "no nvidia-smi"

echo "=== repo ==="
cd "$REPO_DIR" 2>/dev/null && git log --oneline -1 && git status --porcelain | head -20

echo "=== data ==="
ls -d "$HOME"/spectra_* 2>/dev/null
echo "pretrained networks: $(ls "$HOME"/spectra_pretrained_networks 2>/dev/null | wc -l)"
ls "$HOME"/spectra_pretrained_networks 2>/dev/null | head -5
echo "instantiation scripts:"
ls "$HOME"/spectra_models_instantiation 2>/dev/null | head -10
