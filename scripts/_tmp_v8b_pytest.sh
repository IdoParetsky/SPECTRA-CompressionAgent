#!/usr/bin/env bash
# 28 Sep 01:20 (Fable): cluster CPU pytest of the group-as-token cell in tree_v6_dev (no jobs run from it).
T=/home/paretsky/scratch_audit/tree_v6_dev
for f in "$T"/src/*.py "$T"/src/*/*.py "$T"/tests/*.py "$T"/scripts/spectra.sbatch "$T"/scripts/submit.sh; do sed -i 's/\r$//' "$f"; done
bash -n "$T"/scripts/spectra.sbatch && bash -n "$T"/scripts/submit.sh && echo "bash -n ok"
grep -c offline_train_v8_grouptoken "$T"/scripts/spectra.sbatch "$T"/scripts/submit.sh
cat > /home/paretsky/scratch_audit/pytest_v8b.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=v8b-pytest
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --exclude=ise-cpu-intl-25
#SBATCH --output=/home/paretsky/scratch_audit/pytest_v8b.out
cd /home/paretsky/scratch_audit/tree_v6_dev
export SPECTRA_RUN_DIR=/home/paretsky/scratch_audit/tree_v6_dev/runs/pytest_v8b
mkdir -p "$SPECTRA_RUN_DIR"
echo "== targeted"
/home/paretsky/.conda/envs/spectra/bin/python -m pytest tests/test_v8_group_tokens.py tests/test_state_encoder.py tests/test_v3_recipe.py -q --no-header -p no:cacheprovider 2>&1 | tail -40
echo "== full"
/home/paretsky/.conda/envs/spectra/bin/python -m pytest tests -q --no-header -p no:cacheprovider 2>&1 | tail -15
EOF
sbatch /home/paretsky/scratch_audit/pytest_v8b.sbatch 2>&1 | grep -v "GPU Parameter"
