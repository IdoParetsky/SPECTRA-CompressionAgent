#!/usr/bin/env bash
# Submit the three-way reward A/B (matched king_fortify recipe) on the cluster.
set -euo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
cd "$REPO_DIR"
echo "=== reward A/B matrix ==="
A=$(bash scripts/submit.sh reward_neon_ab | tee /dev/stderr | tail -1)
B=$(bash scripts/submit.sh reward_structural_ab | tee /dev/stderr | tail -1)
C=$(bash scripts/submit.sh reward_shaped_ab | tee /dev/stderr | tail -1)
echo "reward_neon_ab=$A"
echo "reward_structural_ab=$B"
echo "reward_shaped_ab=$C"
squeue -u "$USER"
