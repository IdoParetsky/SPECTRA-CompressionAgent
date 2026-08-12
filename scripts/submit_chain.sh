#!/usr/bin/env bash
# Cluster-side dependent / parallel SPECTRA submits (VPN-proof).
#
# Usage:
#   bash scripts/submit_chain.sh afterok:20066522 recover_warm_king_fortify
#   bash scripts/submit_chain.sh afterok:20066521,afterok:20066522 recover_careful_fortify_ft80
#
# Or submit a full parallel matrix that chains follow-ups on the cluster:
#   bash scripts/submit_parallel_matrix.sh
#
# Dependency syntax is raw Slurm (--dependency=...). Prefer afterok so a failed
# parent does not start children. afterany is available if you intentionally want
# best-effort continuation.
set -euo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
cd "$REPO_DIR"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <dependency-spec> <profile> [gpu_count]" >&2
  echo "  e.g. $0 afterok:12345 recover_careful_fortify" >&2
  exit 1
fi

export SPECTRA_DEPENDENCY="$1"
shift
exec bash scripts/submit.sh "$@"
