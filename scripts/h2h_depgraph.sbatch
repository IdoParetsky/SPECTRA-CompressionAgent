#!/usr/bin/env bash
# Head-to-head with DepGraph on its two CIFAR home benchmarks: Torch-Pruning's own reproduce/main.py
# (group_sl, the README's flags) from the same pretrained checkpoints SPECTRA prunes, on our GPU, then
# scripts/bench_deploy.py on DepGraph's pruned nets and their origins (same protocol as our candidates).
#   sbatch --gpus=rtx_4090:1 scripts/h2h_depgraph.sbatch
#SBATCH --job-name=h2h-depgraph
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0-12:00:00

set -uo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-$PWD}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
TP_DIR="${TP_DIR:-$HOME/scratch_audit/third_party/Torch-Pruning}"
NETS="${SPECTRA_NETS_DIR:-$HOME/spectra_pretrained_networks}"
DATA="${H2H_DATA_ROOT:-$HOME/scratch_audit/third_party/data}"
OUT="${H2H_OUT:-$REPO_DIR/runs/h2h_depgraph/job_${SLURM_JOB_ID}}"
mkdir -p "$OUT" "$DATA/torchdata"
# Torch-Pruning's registry reads <dataroot>/torchdata/ and downloads whatever it cannot verify there.
for d in cifar-10-batches-py cifar-100-python; do
  [[ -e "$DATA/torchdata/$d" ]] || ln -s "$HOME/spectra_datasets/$d" "$DATA/torchdata/$d"
done
echo "h2h-depgraph ${SLURM_JOB_ID} on $(hostname) at $(date -Is); Torch-Pruning $(git -C "$TP_DIR" describe --tags 2>/dev/null)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,nounits -l 1 >> "$OUT/gpu_samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

STATUS=0
# v1.6.1 fine-tunes after pruning only with --finetune, which its README commands leave out.
prune() {  # run name, then main.py arguments
  local name=$1 t0
  shift
  t0=$(date +%s)
  (cd "$TP_DIR/reproduce" && "$PYTHON" main.py --mode prune --batch-size 128 --global-pruning --finetune \
     --dataroot "$DATA" --output-dir "$OUT/$name" "$@") > "$OUT/$name.log" 2>&1
  local status=$?
  [[ $status -ne 0 ]] && STATUS=$status
  echo "{\"name\": \"$name\", \"status\": $status, \"seconds\": $(( $(date +%s) - t0 ))}" | tee -a "$OUT/wallclock.jsonl"
}
prune resnet56_cifar10_group_sl_2.11 --model resnet56 --dataset cifar10 \
  --restore "$NETS/resnet56_cifar10_dep_graph_93.53.pth" --method group_sl --speed-up 2.11 --reg 5e-4
prune vgg19_cifar100_group_sl_8.84 --model vgg19 --dataset cifar100 \
  --restore "$NETS/vgg19_cifar100_dep_graph_73.5.pth" --method group_sl --speed-up 8.84 --reg 5e-4

(cd "$TP_DIR/reproduce" && "$PYTHON" - "$NETS" "$OUT" <<'EOF'
import sys
import torch
import registry
nets, out = sys.argv[1], sys.argv[2]
for name, ckpt, classes in (("resnet56", "resnet56_cifar10_dep_graph_93.53.pth", 10),
                            ("vgg19", "vgg19_cifar100_dep_graph_73.5.pth", 100)):
    model = registry.get_model(name, num_classes=classes)
    model.load_state_dict(torch.load(f"{nets}/{ckpt}", map_location="cpu"))
    torch.save(model, f"{out}/origin_{name}.pth")
EOF
) || STATUS=$?

MODELS=()
for spec in resnet56:cifar10:cifar-10:resnet56_cifar10_group_sl_2.11 vgg19:cifar100:cifar-100:vgg19_cifar100_group_sl_8.84; do
  IFS=: read -r model dataset bench_dataset name <<< "$spec"
  [[ -f "$OUT/origin_${model}.pth" ]] && MODELS+=(--model "$OUT/origin_${model}.pth:${bench_dataset}:origin:depgraph_${model}")
  pruned=$(find "$OUT/$name" -name "${dataset}_${model}_group_sl.pth" 2>/dev/null | head -1)
  [[ -n "$pruned" ]] && MODELS+=(--model "${pruned}:${bench_dataset}:group_sl:depgraph_${model}")
done
for r in $(seq 1 "${BENCH_REPEATS:-3}"); do
  PYTHONPATH="$TP_DIR/reproduce" "$PYTHON" "$REPO_DIR/scripts/bench_deploy.py" "${MODELS[@]}" ${BENCH_ARGS:-} \
    --out "$OUT/bench_r${r}.jsonl" || STATUS=$?
done
echo "h2h-depgraph finished with status ${STATUS} at $(date -Is); outputs in ${OUT}"
exit $STATUS
