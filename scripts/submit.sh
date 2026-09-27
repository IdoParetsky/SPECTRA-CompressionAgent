#!/usr/bin/env bash
# Submit a SPECTRA batch job and print everything needed to follow it.
#
#   bash scripts/submit.sh smoke            # pipeline correctness
#   bash scripts/submit.sh medium           # small DB agent
#   bash scripts/submit.sh full             # thesis-scale
#   bash scripts/submit.sh probe            # recovery matrix (no RL)
#   bash scripts/submit.sh probe_groupft    # recovery matrix under group-aware freeze
#   bash scripts/submit.sh recover          # full FT + mild rates
#   bash scripts/submit.sh recover_groupft  # group-aware freeze + mild rates
#   bash scripts/submit.sh recover_wide     # full FT + rates incl. 0.7
#   bash scripts/submit.sh recover_pref10   # full FT + mild rates, -10 pp preference
#
# Wall clocks MUST exceed the profile's --runtime_limit (agent soft-stop) so eval and
# summarize_run can finish. Matching the two caused SLURM TIMEOUT mid-eval (not a
# library crash). This script refuses to submit if wall_sec <= train_sec.
#
# HPC notes that have bitten us:
#   - CPU-Mem-per-GPU-Limit is 24G (HPC 6 Sep 2026). Use --mem-per-gpu=24G, never --mem=80G.
#     Do not pass both --mem and --mem-per-gpu. SPECTRA_MEM is ignored (warns).
#   - gpu partition QoS=gpu-part MaxTRESPU gres/gpu=6 (live 9 Sep 2026 17:08; was 5).
#     That is the running-GPU cap. Job --qos=normal has no GPU MaxTRESPU; partition QOS still applies.
#     giladkz is not in gpu AllowQos and zeros rtx_6000/4090. Do not use bypass_limits.
#     Default GRES is any GPU, not one SKU. Strongest free card; ImageNet floor 4090.
#   - gpu MaxMemPerCPU=16G; 24G/8 CPU is well under. MaxTime on gpu is 7 days.
#   - scontrol update TimeLimit is denied for users; cancel+resubmit instead
#   - #SBATCH --signal=USR1@900 is overridden by submit.sh to fire at train-end

set -euo pipefail

PROFILE="${1:-smoke}"
GPU_COUNT="${2:-}"
REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"

# train_sec = --runtime_limit inside spectra.sbatch for agent profiles (0 for probe suites).
# wall must be strictly greater; keep >=90 min buffer for eval when possible.
case "$PROFILE" in
  smoke)
    GPUS="${GPU_COUNT:-1}"; TIME="0-01:00:00"; CPUS=4; TRAIN_SEC=900 ;;
  smoke_v2)
    # Integration gate for the v2 recipe; chain offline_train_v2* afterok this job.
    GPUS="${GPU_COUNT:-1}"; TIME="0-01:30:00"; CPUS=4; TRAIN_SEC=600 ;;
  medium)
    GPUS="${GPU_COUNT:-1}"; TIME="0-10:00:00"; CPUS=6; TRAIN_SEC=25200 ;;
  full)
    GPUS="${GPU_COUNT:-2}"; TIME="0-15:00:00"; CPUS=6; TRAIN_SEC=46800 ;;
  probe)
    GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; CPUS=4; TRAIN_SEC=0 ;;
  probe_continue)
    GPUS="${GPU_COUNT:-1}"; TIME="0-02:00:00"; CPUS=4; TRAIN_SEC=0 ;;
  probe_groupft)
    GPUS="${GPU_COUNT:-1}"; TIME="0-02:00:00"; CPUS=4; TRAIN_SEC=0 ;;
  diag)
    GPUS="${GPU_COUNT:-1}"; TIME="0-04:00:00"; CPUS=4; TRAIN_SEC=7200 ;;
  recover)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_groupft)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_wide)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_pref10)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_king)
    # Compose today's winners: mild rates + -10 pp preference + full FT + 40 epochs.
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  recover_careful)
    # AMC-scale warmup + 6-net DB + standardizer; wall >> train for eval buffer.
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_careful_fortify)
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_king_fortify)
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  recover_warm_king_fortify)
    # Warm-start 6-net careful-fortify from king_fortify best actor/critic.
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_careful_fortify_ft80)
    # Same as careful_fortify but 80 FT epochs (NEON-closer recovery budget).
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  reward_neon_ab)
    # Short king_fortify-like A/B control: NEON reward (explicit).
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  reward_structural_ab)
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  reward_shaped_ab)
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  reward_band_ab)
    # Over-budget arm graded by accuracy overshoot instead of cut size (ledger §52.1).
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  careful_fortify_structural)
    # Promote RCPR onto 6-net careful+fortify (vs running 20066522 neon).
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_shaped)
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  reward_structural_seed43)
    # Seed replicate of structural reward A/B (luck check).
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; CPUS=4; TRAIN_SEC=18000 ;;
  careful_fortify_tau15)
    # Preference τ=15: careful non-id mass sits at median ≈−11 pp (just past τ=10).
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_mildrates)
    # Safer action set 1.0/0.95/0.9 under fortify (cut 0.8 toxic channel).
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_structural_guard)
    # Asymmetric RCPR: realized credit in-budget, max(realized,nominal) on violations.
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_structural_tau15)
    # Combine mid-run levers: RCPR (healthier credit; late within-10↑) + τ=15 (covers −11 pp mass).
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_cifar10)
    # Mixed-DB failure is C100; C10-only 6-net diversity (same fortify recipe).
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_cifar10_fast)
    # Experimental AMP + channels_last + skip FT empty_cache. Same C10-thin recipe.
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=8; TRAIN_SEC=36000 ;;
  c10_width_skinny_train)
    # Put skinny r20-w2 in train; held-out eval is r56-w4 only.
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=8; TRAIN_SEC=36000 ;;
  c10_budget_state)
    # Remaining-param ratio as an extra token channel (same C10-thin catalog).
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=8; TRAIN_SEC=36000 ;;
  careful_fortify_cifar10_structural)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  c100_mild_structural)
    # C100-only agent: milder rates, τ=15, 80 FT, structural reward.
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  c100_mild_neon)
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  c100_curriculum_king)
    # Warm-start king_fortify (C10) onto C100-only mild recipe.
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  probe_c100)
    GPUS="${GPU_COUNT:-1}"; TIME="0-10:00:00"; CPUS=4; TRAIN_SEC=0 ;;
  probe_c100_aug)
    # CIFAR-100 recovery with train-time RandomCrop+Flip (no RL).
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  probe_c100_recipe)
    # SGD+cosine+mixup+AutoAugment+160 ep C100 recovery (no RL).
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  probe_c100_kd)
    # Same as recipe plus knowledge distillation; no mixup.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  offline_wide)
    # 24-net recoverable catalog (C10/SVHN/FMNIST). Similar/novel/C10-thin stay held out.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=129600 ;;
  c100_wide_drl)
    # C100 DRL on 6 competent nets; start only afterok a recipe probe that recovered.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=129600 ;;
  c100_recoverable_drl)
    # CIFAR-100 DRL only on families that recover (VGG-11/16 + ShuffleNet). SGD recipe.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=129600 ;;
  c10_c100_matched_vgg_drl)
    # Same VGG-16 BN on C10 and C100. Isolates 10-way vs 100-way on a recoverable family.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=129600 ;;
  eval_c100_spoof_classes|eval_c100_residuals_sgd)
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  eval_diag_structural_c100)
    # Eval-only: structural diag agent on C100 held-out (no training).
    GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; CPUS=4; TRAIN_SEC=3600 ;;
  eval_king_fortify_c100)
    GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; CPUS=4; TRAIN_SEC=3600 ;;
  eval_neon_c100)
    GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; CPUS=4; TRAIN_SEC=3600 ;;
  careful_fortify_cifar10_mildrates)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  c10_curriculum_king)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_cifar10_neon_rates)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  careful_fortify_cifar10_fine)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  encoder_c10_set)
    # Same C10 fortify recipe; architecture-agnostic read of the same tokens.
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  encoder_c10_wide)
    # Capacity A/B: 6-layer 512-d Transformer (still task-trained, not BERT).
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  encoder_c10_bert)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  generic_c10_fortify)
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; CPUS=6; TRAIN_SEC=36000 ;;
  offline_train|offline_train_cbrt|offline_train_unified)
    # 10-net leap catalog (C10 families + SVHN + Fashion-MNIST). Floor-constrained eval.
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  offline_train_band_cbrt|offline_train_prefer|offline_train_unified_full|offline_train_prefer_floor|offline_train_neon_full|offline_train_gonce_cold|offline_train_v2a|offline_train_v2b|offline_train_v2c|offline_train_v3_fpgm|offline_train_v3_svd|offline_train_v3_bnscale|offline_train_v3_fpgm_neonraw|offline_train_v3_fpgm_structraw|offline_train_v4_factored|offline_train_v4_factored_tau6|offline_train_v5_p5b3|offline_train_v5_p5b3_cgp|offline_train_v5_ft40|offline_train_v6_inband|offline_train_v6_inband_p5b2|offline_train_v6_inband_p5b2_factored|offline_train_v7_budget)
    # Wall=7d (submit.sh default). Python runtime 6d fuse; patience is the stop.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=6; TRAIN_SEC=518400 ;;
  eval_offline_similar|eval_offline_similar_det|eval_offline_novel)
    # 8 CPUs: 3 DataLoaders x SPECTRA_DATALOADER_WORKERS=4 plus the trainer.
    # 80G needs >=5 CPUs under MaxMemPerCPU=16G.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  c100_ultra_mild)
    # Even smaller cuts (2–5%) on C100 — tests whether any prune is recoverable.
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  c100_mild_structural_seed43)
    GPUS="${GPU_COUNT:-1}"; TIME="0-16:00:00"; CPUS=6; TRAIN_SEC=43200 ;;
  probe_c100_extra)
    GPUS="${GPU_COUNT:-1}"; TIME="0-10:00:00"; CPUS=4; TRAIN_SEC=0 ;;
  eval_only|eval_offline_c100|eval_offline_c100_det|eval_imagenet_short)
    # Skip training; load SPECTRA_ACTOR/CRITIC_CHECKPOINT_PATH and evaluate.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  eval_c10_thin_flop_floor)
    # Eval-only FLOP floor 0.70 + look-ahead on C10-thin; frozen 10-net s42 actor.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  eval_c10_thin|eval_c10_thin_det|eval_c10_thin_fpgm|eval_c10_thin_bnscale|eval_c10_thin_traj|eval_c10_thin_traj_gonce)
    # Plain C10-thin held-out eval (no FLOP floor) — the §17 r20-w2 / r56-w4 comparison.
    # _traj is the unconstrained curve (floor-hold then continue); _gonce adds group-once.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  baseline_c10_l1|baseline_c10_mild|baseline_c10_random|baseline_c10_mild_traj|baseline_c10_mild_traj_gonce|baseline_c10_l1_traj|baseline_c10_l1_traj_gonce)
    # Same-loop L1 / mild-0.9 / random rate policies on C10-thin held-out (r20-w2, r56-w4).
    # *_traj / *_traj_gonce: TRAJ protocol, optionally with group-once.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  diag_reward_band)
    # Crossed dataset x outcome reward-band trace. Short wall so it hands the GPU back.
    GPUS="${GPU_COUNT:-1}"; TIME="1-00:00:00"; CPUS=8; TRAIN_SEC=0 ;;
  c100_recoverable_drl_fine|c100_recoverable_drl_fine_shaped)
    # C100 DRL on recoverable families with a fine rate ladder + SGD-80 in the loop.
    # _shaped pins SPECTRA_REWARD_MODE=structural_shaped in sbatch.
    GPUS="${GPU_COUNT:-1}"; TIME="7-00:00:00"; CPUS=8; TRAIN_SEC=129600 ;;
  *)
    echo "usage: $0 {smoke|...|c100_*|c10_c100_matched_vgg_drl|eval_c100_spoof_classes|eval_c100_residuals_sgd|careful_fortify_cifar10*|encoder_c10_*|generic_c10_fortify|offline_train|offline_train_gonce_cold|offline_wide|eval_offline_*|probe_c100*|eval_*|eval_only|eval_offline_c100|eval_imagenet_short|eval_c10_thin|eval_c10_thin_traj|eval_c10_thin_traj_gonce|eval_c10_thin_flop_floor|careful_fortify_cifar10_fast|c10_width_skinny_train|c10_budget_state|probe_c100_aug|probe_c100_recipe|probe_c100_kd|c100_wide_drl|c100_recoverable_drl|c100_recoverable_drl_fine|diag_reward_band|reward_band_ab|baseline_c10_*|baseline_c10_*_traj|baseline_c10_*_traj_gonce}" >&2
    exit 1
    ;;
esac

CPUS="${SPECTRA_CPUS:-$CPUS}"
# HPC 6 Sep 2026: --mem-per-gpu=24G only (CPU-Mem-per-GPU-Limit). Do not pass --mem.
MEM_PER_GPU="${SPECTRA_MEM_PER_GPU:-24G}"
if [[ -n "${SPECTRA_MEM:-}" ]]; then
  echo "WARNING: SPECTRA_MEM=${SPECTRA_MEM} ignored; using --mem-per-gpu=${MEM_PER_GPU} (HPC CPU-Mem-per-GPU-Limit=24G)." >&2
fi

# Slurm always needs --time (partition MaxTime=7-00:00:00). Default to that so a
# slow eval cannot be hard-killed. Training still stops at TRAIN_SEC. Override:
#   SPECTRA_WALL=0-12:00:00   or   SPECTRA_KEEP_PROFILE_WALL=1
if [[ "${SPECTRA_KEEP_PROFILE_WALL:-}" != "1" && "$PROFILE" != "smoke" && "$PROFILE" != "smoke_v2" ]]; then
  TIME="${SPECTRA_WALL:-7-00:00:00}"
fi

# Continue-train extra wall: caller may raise --runtime_limit past the profile default.
if [[ -n "${SPECTRA_RUNTIME_LIMIT:-}" ]]; then
  TRAIN_SEC="${SPECTRA_RUNTIME_LIMIT}"
fi

# Parse TIME (D-HH:MM:SS or HH:MM:SS) into seconds for the guardrail.
_wall_to_sec() {
  local t="$1" days=0 rest h m s
  if [[ "$t" == *-* ]]; then
    days="${t%%-*}"
    rest="${t#*-}"
  else
    rest="$t"
  fi
  IFS=: read -r h m s <<<"$rest"
  echo $(( days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}

WALL_SEC="$(_wall_to_sec "$TIME")"
if (( TRAIN_SEC > 0 && WALL_SEC <= TRAIN_SEC )); then
  echo "REFUSING submit: wall ${TIME} (${WALL_SEC}s) <= train runtime_limit ${TRAIN_SEC}s" >&2
  echo "Eval would be SLURM-killed. Raise TIME or lower runtime_limit." >&2
  exit 2
fi
if (( TRAIN_SEC > 0 && WALL_SEC - TRAIN_SEC < 10800 )); then
  echo "WARNING: only $(( WALL_SEC - TRAIN_SEC ))s wall buffer after train stop; prefer >=3h for 6-net eval." >&2
fi

cd "$REPO_DIR"
mkdir -p runs/slurm_logs

# GPU pick (Ido 10 Sep): strongest free SKU; floor only for OOM.
#   SPECTRA_GPU_GRES / SPECTRA_GPU_TYPE still override.
#   ImageNet  floor rtx_4090 (2080/1080 OOM; 3090 untested on 1k).
#   Train     no floor — 1080 has been running 40-ep CIFAR FT.
#   Else      any SKU (--gpus=N) so tails do not steal a pro 6000.
# Still exclude L40S (preempt) and cs-4090-09.
EXCLUDE_NODES="${SPECTRA_EXCLUDE_NODES:-ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-01,ise-6000p-02,ise-6000p-03,ise-6000p-04,ise-6000p-05,ise-6000p-06,ise-6000p-07}"

_pick_gpu() {
  # Prints the strongest free typed GRES at or above optional floor (argv: floor sku or empty).
  python3 - "$EXCLUDE_NODES" "${1:-}" <<'PY'
import re, subprocess, sys
exclude = {n.strip() for n in sys.argv[1].split(",") if n.strip()}
floor = (sys.argv[2] if len(sys.argv) > 2 else "").strip()
# Strongest first among SKUs this torch build can run.
# rtx_pro_6000 (ise-6000p-*) is Blackwell; spectra CUDA wheel has no kernel image
# ("CUDA error: no kernel image is available") — 21184403/405 FAILED 11 Sep 17:24.
order = ["rtx_6000", "rtx_4090", "rtx_3090", "rtx_2080", "gtx_1080"]
if floor:
    if floor not in order:
        print("")
        raise SystemExit
    allowed = order[: order.index(floor) + 1]
else:
    allowed = order
try:
    out = subprocess.check_output(["scontrol", "show", "nodes", "-o"], text=True, errors="replace")
except Exception:
    print("")
    raise SystemExit
free = {sku: 0 for sku in allowed}
pat = {sku: re.compile(rf"gres/gpu:{re.escape(sku)}=(\d+)") for sku in allowed}
for rec in out.splitlines():
    kv = {}
    for p in rec.split():
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k] = v
    name = kv.get("NodeName", "")
    if name in exclude:
        continue
    state = kv.get("State", "").upper()
    if any(tok in state for tok in ("DOWN", "DRAIN", "NOT_RESPONDING", "FAIL")):
        continue
    cfg_s, alloc_s = kv.get("CfgTRES", "") or "", kv.get("AllocTRES", "") or ""
    for sku, rx in pat.items():
        cfg = rx.search(cfg_s)
        if not cfg:
            continue
        alloc = rx.search(alloc_s)
        ncfg, nalloc = int(cfg.group(1)), int(alloc.group(1) if alloc else 0)
        if ncfg > nalloc:
            free[sku] += ncfg - nalloc
for sku in allowed:
    if free[sku] > 0:
        print(sku)
        raise SystemExit
print("")
PY
}

if [[ -n "${SPECTRA_GPU_GRES:-}" ]]; then
  GPU_GRES="${SPECTRA_GPU_GRES}"
elif [[ -n "${SPECTRA_GPU_TYPE:-}" ]]; then
  GPU_GRES="${SPECTRA_GPU_TYPE}:${GPUS}"
else
  case "$PROFILE" in
    eval_imagenet_short)
      _strong="$(_pick_gpu rtx_4090 || true)"
      GPU_GRES="${_strong:-rtx_4090}:${GPUS}" ;;
    offline_train|offline_train_cbrt|offline_train_band_cbrt|offline_train_unified|offline_train_prefer|offline_train_unified_full|offline_train_prefer_floor|offline_train_neon_full|offline_train_gonce_cold|offline_train_v2a|offline_train_v2b|offline_train_v2c|offline_train_v3_fpgm|offline_train_v3_svd|offline_train_v3_bnscale|offline_train_v3_fpgm_neonraw|offline_train_v3_fpgm_structraw|offline_train_v4_factored|offline_train_v4_factored_tau6|offline_train_v5_p5b3|offline_train_v5_p5b3_cgp|offline_train_v5_ft40|offline_train_v6_inband|offline_train_v6_inband_p5b2|offline_train_v6_inband_p5b2_factored|offline_train_v7_budget)
      _strong="$(_pick_gpu || true)"
      if [[ -n "${_strong:-}" ]]; then
        GPU_GRES="${_strong}:${GPUS}"
      else
        GPU_GRES="${GPUS}"
      fi ;;
    *)
      GPU_GRES="${GPUS}" ;;
  esac
fi

# Cluster-side chaining: SPECTRA_DEPENDENCY=afterok:JOBID (or afterany:JOBID).
# Survives laptop/VPN disconnects — do not rely on a local PowerShell watcher.
SBATCH_EXTRA=()
if [[ -n "${SPECTRA_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA+=(--dependency="${SPECTRA_DEPENDENCY}")
fi
if [[ -n "${SPECTRA_BEGIN:-}" ]]; then
  SBATCH_EXTRA+=(--begin="${SPECTRA_BEGIN}")
fi
if [[ -n "${SPECTRA_NICE:-}" ]]; then
  SBATCH_EXTRA+=(--nice="${SPECTRA_NICE}")
fi
# Command-line --signal overrides #SBATCH --signal=B:USR1@900.
# This is a last-resort stop before SIGKILL, not the train/eval split
# (--runtime_limit ends training). Slurm rejects huge @seconds (7d walls).
if [[ -z "${SPECTRA_USR1_SEC:-}" ]]; then
  if (( TRAIN_SEC > 0 )); then
    USR1_SEC=$(( WALL_SEC - TRAIN_SEC - 300 ))
    (( USR1_SEC < 5400 )) && USR1_SEC=5400
    (( USR1_SEC > 10800 )) && USR1_SEC=10800
    (( USR1_SEC > WALL_SEC - 600 )) && USR1_SEC=$(( WALL_SEC - 600 ))
    (( USR1_SEC < 120 )) && USR1_SEC=120
  else
    USR1_SEC=900
  fi
else
  USR1_SEC="$SPECTRA_USR1_SEC"
fi
SBATCH_EXTRA+=(--signal="B:USR1@${USR1_SEC}")
SBATCH_EXTRA+=(--output="${REPO_DIR}/runs/slurm_logs/spectra_%j.out")
SBATCH_EXTRA+=(--error="${REPO_DIR}/runs/slurm_logs/spectra_%j.out")

# Pin A/B / checkpoint switches on the --export line. This cluster's scontrol
# only shows keys listed after ALL; relying on --export=ALL alone dropped
# SPECTRA_EVAL_DETERMINISTIC / SPECTRA_REWARD_SCALE from the first A/B submit.
SBATCH_EXPORT="ALL,SPECTRA_PROFILE=${PROFILE}"
for _k in SPECTRA_EVAL_DETERMINISTIC SPECTRA_REWARD_MODE SPECTRA_REWARD_SCALE \
          SPECTRA_REWARD_TRACE SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH \
          SPECTRA_EVAL_POLICY SPECTRA_EVAL_MIN_FLOP_RATIO SPECTRA_EVAL_PREFER_PARAM_PER_FLOP \
          SPECTRA_SEED SPECTRA_CONTINUE_TRAIN SPECTRA_SKIP_TRAIN SPECTRA_SKIP_EVAL_TRAIN \
          SPECTRA_SKIP_EVAL SPECTRA_STATE_ALIGN SPECTRA_ROLLOUT_LIMIT \
          SPECTRA_WARMUP_MULTIPLIER SPECTRA_WARMUP_FLOOR \
          SPECTRA_FILTER_IMPORTANCE SPECTRA_CHECKPOINT SPECTRA_UNIFIED_EPS \
          SPECTRA_STANDARDIZER_PATH SPECTRA_TRAIN_RESPECT_FLOOR \
          SPECTRA_BUDGET_IN_STATE SPECTRA_EVAL_LOOKAHEAD SPECTRA_ACTOR_SKIP_OVERBUDGET \
          SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN SPECTRA_RUNTIME_LIMIT \
          SPECTRA_EVAL_TRAJECTORY SPECTRA_GROUP_ONCE_PER_PASS SPECTRA_SNAPSHOT_BASELINE \
          SPECTRA_ALGO SPECTRA_AGENT_LR SPECTRA_TRAIN_FT_EPOCHS SPECTRA_TRAIN_FT_PATIENCE \
          SPECTRA_STATE_SLACK SPECTRA_ENCODER_DROPOUT SPECTRA_POLICY_CONFIG \
          SPECTRA_PPO_EPISODES SPECTRA_PPO_EPOCHS SPECTRA_ENTROPY_COEF SPECTRA_ENTROPY_MIN \
          SPECTRA_EVAL_PASSES SPECTRA_STATE_GROUPCOST SPECTRA_TRAIN_TAU \
          SPECTRA_PROBE_EVERY SPECTRA_PROBE_NETS SPECTRA_MIN_EPISODES SPECTRA_PATIENCE_EPISODES \
          SPECTRA_REWIND_BEST SPECTRA_REWIND_PATIENCE SPECTRA_REWIND_MAX SPECTRA_REWIND_ENTROPY \
          SPECTRA_V3_ENTROPY_COEF SPECTRA_V3_ENTROPY_MIN SPECTRA_FACTORED_HEAD SPECTRA_V4_RANKING_MENU \
          SPECTRA_FT_RECIPE SPECTRA_FT_REINIT_EDITED SPECTRA_FT_REINIT_THEN_POLISH SPECTRA_REFRESH_ALL_FEATURES \
          SPECTRA_FT_REINIT_EPOCHS SPECTRA_FT_REINIT_PATIENCE SPECTRA_FT_REINIT_SELECT SPECTRA_FT_REINIT_SCOPE \
          SPECTRA_FT_POLISH_EPOCHS SPECTRA_FT_POLISH_PATIENCE SPECTRA_FT_POLISH_LR_MULT \
          SPECTRA_V5_DATABASE SPECTRA_INPUT SPECTRA_DATABASE SPECTRA_NUM_EPOCHS SPECTRA_FINETUNE_PATIENCE \
          SPECTRA_DATASET_NAMES SPECTRA_EVAL_COUNTERFACTUAL SPECTRA_PROBE_SCORE SPECTRA_FT_LR SPECTRA_FT_OPTIM SPECTRA_FT_SGD_LR \
          SPECTRA_FT_SCHEDULE SPECTRA_FT_LR_MIN SPECTRA_FT_WARMUP_EPOCHS SPECTRA_FT_WD \
          SPECTRA_FT_LSQ_CONSUMERS SPECTRA_FT_BN_RECAL SPECTRA_FT_CALIB_BATCHES SPECTRA_FT_CALIB_IMAGES \
          SPECTRA_ACTION_MENU SPECTRA_STOP_REWARD_SCALE; do
  _v="${!_k-}"
  if [[ -n "$_v" ]]; then
    SBATCH_EXPORT+=",${_k}=${_v}"
  fi
done

JOB_ID=$(sbatch --parsable \
  --gpus="${GPU_GRES}" \
  --mem-per-gpu="${MEM_PER_GPU}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --exclude="${EXCLUDE_NODES}" \
  --job-name="${SPECTRA_JOB_NAME:-spectra-${PROFILE}}" \
  --export="${SBATCH_EXPORT}" \
  "${SBATCH_EXTRA[@]}" \
  scripts/spectra.sbatch)

LOG="${REPO_DIR}/runs/slurm_logs/spectra_${JOB_ID}.out"
    echo "submitted job ${JOB_ID} (profile=${PROFILE}, gpus=${GPU_GRES}, cpus=${CPUS}, mem-per-gpu=${MEM_PER_GPU}, time=${TIME}, train_limit=${TRAIN_SEC}s, usr1=${USR1_SEC}s, exclude=${EXCLUDE_NODES}${SPECTRA_DEPENDENCY:+, dependency=${SPECTRA_DEPENDENCY}}${SPECTRA_BEGIN:+, begin=${SPECTRA_BEGIN}})"
echo "log     : ${LOG}"
echo "run dir : ${REPO_DIR}/runs/job${JOB_ID}"
# Confirm the scheduler accepted the Timelimit we asked for (qos/partition can silently clamp).
sleep 1
TL=$(squeue -j "${JOB_ID}" -h -o "%l" 2>/dev/null || sacct -j "${JOB_ID}" -n -X -o Timelimit --parsable2 2>/dev/null | head -1 || true)
echo "Timelimit (scheduler): ${TL:-unknown}"
N_R=$(squeue -u "${USER}" -t R -h -o %i 2>/dev/null | wc -l | tr -d ' ')
echo "QOS gpu-part running: ${N_R:-?}/6 (MaxTRESPU gres/gpu=6; extras PD until a GPU frees)"
echo "follow  : tail -f ${LOG}"
echo "status  : squeue -j ${JOB_ID}"
echo "cancel  : scancel ${JOB_ID}"
echo "${JOB_ID}"
