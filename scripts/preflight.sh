#!/usr/bin/env bash
# Cheap correctness gate to run on the login node before consuming a GPU allocation.
#
# Import errors, syntax errors and shape bugs in the pruning/state code do not need a GPU to
# surface, and a failed 4-hour job that dies in the first second wastes an allocation slot.
#
#   bash scripts/preflight.sh

REPO_DIR="${SPECTRA_REPO_DIR:-$HOME/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
cd "$REPO_DIR" || exit 1

STATUS=0

echo "=== commit ==="
git log --oneline -1

echo
echo "=== imports ==="
"$PYTHON" - <<'PY' || exit 1
import importlib, sys, traceback
modules = [
    "src.logging_utils", "src.run_recorder", "src.distributed", "src.utils",
    "src.channel_groups", "src.pruning", "src.action_costs", "src.feature_standardizer",
    "src.BERTInputModeler", "src.Model.StateEncoder", "src.Model.Agent",
    "NetworkFeatureExtraction.src.FeatureExtractors.BaseFE",
    "NetworkFeatureExtraction.src.FeatureExtractors.WeightStatisticsFE",
    "NetworkFeatureExtraction.src.FeatureExtractors.ActivationsStatisticsFE",
]
failed = 0
for name in modules:
    try:
        importlib.import_module(name)
        print(f"  ok   {name}")
    except Exception:
        failed += 1
        print(f"  FAIL {name}")
        traceback.print_exc()
sys.exit(1 if failed else 0)
PY
[ $? -ne 0 ] && STATUS=1

echo
echo "=== runner --help (argument wiring) ==="
"$PYTHON" a2c_agent_reinforce_runner.py --help >/dev/null 2>&1 && echo "  ok" || { echo "  FAIL"; STATUS=1; }

echo
echo "=== unit tests (CPU) ==="
if "$PYTHON" -m pytest --version >/dev/null 2>&1; then
  "$PYTHON" -m pytest tests/ -q --tb=short 2>&1 | tail -25
  [ "${PIPESTATUS[0]}" -ne 0 ] && STATUS=1
else
  echo "  pytest not installed in this env - skipping (install with: $PYTHON -m pip install pytest)"
fi

echo
echo "=== summarizer self-check ==="
python3 scripts/summarize_run.py --help >/dev/null 2>&1 && echo "  ok" || { echo "  FAIL"; STATUS=1; }

echo
echo "preflight status: $STATUS"
exit $STATUS
