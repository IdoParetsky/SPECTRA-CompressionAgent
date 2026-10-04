#!/bin/bash
# G2 hold-out checkpoints (roadmap §3 G2): ShuffleNetV2 x1, RepVGG-A0, MobileNetV2 x0.5 and
# DenseNet-40 on one unseen dataset, 200 epochs each, four trainings sharing one GPU.
# TEST-only nets: rows go to configs/input_g2_holdout_<svhn|fmnist>.json, never a training file.
#
#   G2_DATASET=svhn SPECTRA_REPO_DIR=$PWD sbatch --gpus=1 --job-name=g2-holdout-svhn \
#     --exclude=<submit.sh list> --output=runs/slurm_logs/g2_holdout_%j.out scripts/g2_holdout_pretrain.sbatch
#
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0-16:00:00

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:?set SPECTRA_REPO_DIR to the tree}"
DS="${G2_DATASET:?svhn or fashion-mnist}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
INST="${SPECTRA_INSTANTIATION:-/home/paretsky/spectra_models_instantiation}"
cd "$REPO_DIR" || exit 1

case "$DS" in
  svhn) TOK=svhn ;;
  fashion-mnist) TOK=fmnist ;;
  *) echo "FATAL: G2_DATASET must be svhn or fashion-mnist, got '$DS'" >&2; exit 2 ;;
esac

echo "=============================================================="
echo " G2 hold-out pretrain ${SLURM_JOB_ID:-local}  dataset=${DS}"
echo " node    : $(hostname)"
echo " started : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 || echo "nvidia-smi unavailable"
if ! "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: torch sees no CUDA device." >&2
  exit 2
fi

MDIR="runs/g2_holdout/${TOK}"
mkdir -p "$MDIR"
PIDS=()
NAMES=()
train_bg() {
  local arch=$1 script=$2 source=$3
  "$PYTHON" scripts/train_pretrained_checkpoint.py --arch "$arch" --script "$INST/$script" \
    --dataset "$DS" --source "$source" --epochs 200 --milestones 100 150 --workers 2 --seed 42 \
    --manifest "$MDIR/manifest_${arch}.json" > "$MDIR/${arch}.log" 2>&1 &
  PIDS+=($!)
  NAMES+=("$arch")
}

train_bg shufflenetv2x1  shufflenetv2_chenyaofo.py chenyaofo
train_bg repvgg_a0       repvgg_chenyaofo.py       chenyaofo
train_bg mobilenet_v2x05 mobilenetv2_chenyaofo.py  chenyaofo
train_bg densenet40      densenet_cifar.py         densenet-cifar

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "ok   ${NAMES[$i]}"
  else
    echo "FAIL ${NAMES[$i]} (see $MDIR/${NAMES[$i]}.log)"
    FAIL=$((FAIL + 1))
  fi
  tail -n 2 "$MDIR/${NAMES[$i]}.log"
done

# Single writer: the four trainings each wrote their own manifest, so no read-modify-write race.
"$PYTHON" - "$MDIR" "configs/input_g2_holdout_${TOK}.json" <<'PY'
import json
import pathlib
import sys

mdir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
db = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
for manifest in sorted(mdir.glob("manifest_*.json")):
    for ckpt, meta in json.loads(manifest.read_text(encoding="utf-8")).items():
        db[ckpt] = meta["row"]
        print(f"hold-out {ckpt}  acc={meta['acc']} paramsM={meta['params_m']} flopsM={meta['flops_m']}")
out.write_text(json.dumps(db, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(db)} nets)")
PY

echo "=============================================================="
echo " finished : $(date -Is)  failed=${FAIL}/4"
echo "=============================================================="
exit "$FAIL"
