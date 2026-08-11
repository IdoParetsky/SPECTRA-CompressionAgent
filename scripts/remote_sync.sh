#!/usr/bin/env bash
# Bring the SLURM copy of the repo to the pushed commit, discarding local drift.
#
# The remote checkout is a build artefact, not a place where edits happen: everything is
# authored on the laptop and pushed. A hard reset therefore keeps the two in lockstep and
# avoids merge prompts inside a non-interactive SSH session.
#
#   Get-Content scripts/remote_sync.sh | ssh bgu-slurm 'bash -s'

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-$HOME/SPECTRA-CompressionAgent}"
BRANCH="${SPECTRA_BRANCH:-master}"

cd "$REPO_DIR" || { echo "no repo at $REPO_DIR" >&2; exit 1; }

echo "before: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
git fetch --quiet origin "$BRANCH" || { echo "fetch failed" >&2; exit 1; }
git reset --hard "origin/${BRANCH}" --quiet
git clean -fd --quiet -e runs -e models -e pruned_models -e checkpoints
echo "after : $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

mkdir -p runs/slurm_logs
chmod +x scripts/*.sh 2>/dev/null

echo "status:"
git status --porcelain | head -10
