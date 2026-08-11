#!/usr/bin/env bash
# Verify CUDA PyTorch on the allocated sinteractive node.
# If this fails: scancel $SLURM_JOB_ID and request a new sinteractive.
set -euo pipefail

PY="${SPECTRA_PYTHON:-/home/paretsky/.conda/envs/spectra/bin/python}"

echo "Python: $PY"
"$PY" - <<'EOF'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu[{i}]={torch.cuda.get_device_name(i)}")
else:
    raise SystemExit(
        "No CUDA GPUs visible. scancel this job and start a new sinteractive."
    )
EOF
