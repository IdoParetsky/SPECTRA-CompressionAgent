# V5 — no-agent recovery probes before any DRL GPU

**Stamped:** 18 Sep 2026 03:40 IDT (Fable V5 sitting). Default-off code on the local tree; ops overlays leap only when v3/V4 stop or Ido says. Nothing here scancels a train.

Three probes, in GPU order after the in-flight frozen-snap TRAJs (neonraw → bnscale → V4 ep0083):

| # | Probe | Question it answers | Gate it opens |
|---|---|---|---|
| **A** | **P8 recovery** — A vs C-G vs C-G+, same mild 2-pass walk, thin r20-w2 / r56-w4 + Catalog L chenyaofo ResNet-56 | Does NEON layer replacement recover a real 2–5 % cut on CNN groups at all? What group budget does it need? | `offline_train_v5_p5b3_cgp` (P8 DRL) and the C-G+ caption for heuristics |
| **B** | **C100 gate** — mild 2-pass walk on the five C100 candidates under the *train* FT (12/4) | Which C100 nets have a non-empty in-band band under the recipe the actor is trained with? | admission rows in `configs/v5_p5b3_c100_gate.json` → `database_offline_v5_p5b3_admitted.json` → `offline_train_v5_p5b3` |
| **C** | **Isolated train FT 12/4 vs 40/10** — `offline_train_v5_ft40` vs live v3-fpgm `21385158` | Is the 12/4 cost cut equivalent to 40/10, or does it teach a more conservative stop? | caption of every v2/v3/V4 row; budget of the P5-B3 job |

All three are **no new science until they land**. Same τ = 10, same 2-pass group-once walk, same `det=1` TRAJ, quote `[eval] TRAJ val_best` only.

---

## A. P8 recovery probe (A vs C-G vs C-G+)

**Why a no-agent probe.** Ledger §12 killed recipe B (freeze-rest, keep filters) 0/32 on ResNets. C-G throws away *more* (the surviving filters) and trains a group from scratch inside a frozen net; residual adds and frozen BN make the rest non-independent of the fresh group. If C-G / C-G+ cannot recover the mild walk, a DRL-C GPU is an expensive way to learn "never prune".

**Walk.** `SPECTRA_EVAL_POLICY=mild` (0.9 on every legal row) × 2 passes × group-once — byte-identical actions across the three arms, so the only difference is the recovery. Control rows already exist for recipe A: thin **§93** (`21413236`).

**Arms (same input, same profile, one flag):**

```bash
# thin (r20-w2, r56-w4) — recipe A already landed as §93; submit only C-G and C-G+
SPECTRA_EVAL_PASSES=2 SPECTRA_FT_RECIPE=cg  SPECTRA_JOB_NAME=p8-thin-mild-cg  SPECTRA_NICE=100 \
  bash scripts/submit.sh baseline_c10_mild_traj_gonce
SPECTRA_EVAL_PASSES=2 SPECTRA_FT_RECIPE=cgp SPECTRA_JOB_NAME=p8-thin-mild-cgp SPECTRA_NICE=100 \
  bash scripts/submit.sh baseline_c10_mild_traj_gonce

# Catalog L chenyaofo ResNet-56 C10 (94.37 %) — all three arms (no A row exists for this net)
for r in a cg cgp; do
  SPECTRA_EVAL_PASSES=2 SPECTRA_FT_RECIPE=$r SPECTRA_JOB_NAME=p8-catl-r56-mild-$r SPECTRA_NICE=100 \
  SPECTRA_INPUT=/home/paretsky/SPECTRA-CompressionAgent/configs/input_catalog_l_c10_r56.json \
    bash scripts/submit.sh baseline_c10_mild_traj_gonce
done
```

`SPECTRA_FT_RECIPE=cg` sets `SPECTRA_FT_REINIT_EDITED=1 SPECTRA_REFRESH_ALL_FEATURES=1`; `cgp` adds `SPECTRA_FT_REINIT_THEN_POLISH=1`. Group budget at eval scale: `SPECTRA_FT_REINIT_EPOCHS=60 SPECTRA_FT_REINIT_PATIENCE=6` on **val**, polish `8 / 3` at `0.1×` lr (all overridable; the sbatch prints them). The `FLAGS` line must show `ft_recipe=C-G` / `C-G+` and `refresh_all=1`; every non-identity step logs `P8 C-G: layer replacement — N producer(s) re-drawn at width W …`.

**Cost.** Thin A-walk was 13 h (r20) / 3 h (r56) at 40/10. C-G trains the group up to 60 epochs per step (backward still traverses the frozen rest), so budget **~1.5–2×** per arm; Catalog L r56 (0.86 M params, 251 MFLOPs) is ~3× a thin epoch — expect 1–2 days per arm. Run at `SPECTRA_NICE=100` in QOS holes; never scancel a train for it.

**What to read (per arm, per net).**

1. `[eval] TRAJ val_best`: kept params / FLOPs, val Δacc, **test Δacc**. In-band if val Δacc ≥ −10.
2. Per-step `Step k done … acc a→b`: the **recovery curve** A vs C-G vs C-G+ at identical cuts.
3. `finetune` records (`run_records.jsonl`): `epochs_ran` with `phase="C-G group"` — the **median epochs to plateau** is the group budget for `offline_train_v5_p5b3_cgp` (`SPECTRA_FT_REINIT_EPOCHS`, `_PATIENCE`).

**Decision table.**

| Outcome at val_best | Read | Next |
|---|---|---|
| C-G ≈ A on all three nets (within ~0.5 pp at equal keep) | Throw-away is harmless on CNN groups; NEON-C is a valid CNN recipe | Skip the polish; P8 DRL cell = **C-G**; heuristics under C-G |
| C-G ≪ A but C-G+ ≈ A | Frozen rest + skip path needs the coupling polish | P8 DRL cell = **C-G+** (assigned); caption C-G as Gilad-literal ablation |
| C-G+ ≪ A (no in-band cut, or > 2 pp worse at equal keep) | NEON-C is dense-DNN-specific | **No P8 DRL GPU.** Thesis caption: SPECTRA's recoverable CNN recipe is A |
| C-G / C-G+ **beats** A at equal keep | Reinit regularises the leftover filters — the interesting case | P8 DRL cell + re-run the L1 2-pass control (§94) under the winning recipe |

Do **not** caption any arm as a DRL result; these are heuristic walks. Do not pick val_best on the test loader.

---

## B. C100 gate probe (P5-B3 admission)

**Rule (gate table `configs/v5_p5b3_c100_gate.json`):** a candidate is `admitted` when the mild 2-pass walk under the **train** recovery budget has a TRAJ `val_best` with kept ≤ 0.98 and val Δacc ≥ −10 (a real 2–5 % in-band cut, not `within_budget` at ≥ 98 % params). Identity / empty band → `rejected`. Unrecovered C100 residuals must not enter the C10 actor (Gilad 3 Sep; C6/§7; `20202760`; v2b C100 §87 identity).

**Candidates (all on leap, none in a TEST catalog, none of the unlike family):** `vgg11_bn_cifar100` (70.78, the only C100 net with any recovery precedent — C6 under 160-ep SGD), `vgg13_bn_cifar100` (74.63), `mobilenet-v2x1_cifar100` (74.2; §7 val DROP under the old FT), `densenet40_cifar100` (70.25), `resnet32_cifar100_chenyaofo` (70.16). **Not** `vgg16_bn_cifar100` / `shufflenetv2x1_cifar100` — both are C9 TEST rows (`input_offline_c100.json`) and ShuffleNet is the unlike family; the old `database_c100_recoverable.json` trio cannot be the P5-B C100 slice (see `configs/v5_diversity_plan.json` → `fable_18sep_correction`).

```bash
# one job, five nets, train budget, recipe of the intended train job (A here; cgp if P8 wins)
SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 SPECTRA_FT_RECIPE=a \
SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_JOB_NAME=p5b3-c100-gate-mild-ft12 SPECTRA_NICE=100 \
SPECTRA_INPUT=/home/paretsky/SPECTRA-CompressionAgent/configs/v5_p5b3_c100_candidates_input.json \
  bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

(`v5_p5b3_c100_candidates_input.json` = the five C100 rows of `configs/database_offline_v5_p5b3.json`; build it with `python - <<'PY'` … or `jq` from that file — the rows are identical.) Then fill each candidate's `status`, `probe_job`, `val_best_kept`, `val_best_delta_pp` in the gate table and run

```bash
python scripts/build_v5_catalog.py --emit-admitted       # -> configs/database_offline_v5_p5b3_admitted.json
python scripts/build_v5_catalog.py --check-admitted configs/database_offline_v5_p5b3_admitted.json
```

`offline_train_v5_p5b3` refuses to start unless every C100 row in its `--database` is admitted. If **fewer than 2** candidates are admitted, P5-B3 collapses to "one C100 exemplar" — then run **P5-B2** (keep one SVHN net in train, hold Fashion-MNIST + ImageNet) rather than a 1-net C100 slice, and say so in the caption.

---

## C. Isolated train FT 12/4 vs 40/10

Never run (LOOP_ALGORITHMS §6.1). One cell: the live v3-fpgm recipe on the 24-net with `SPECTRA_TRAIN_FT_EPOCHS=40 / PATIENCE=10`; nothing else changes.

```bash
SPECTRA_JOB_NAME=v5-ft40-v3fpgm SPECTRA_NICE=50 bash scripts/submit.sh offline_train_v5_ft40
```

Control = `21385158` (v3-fpgm, 12/4). Compare: probe-score trajectory (same probe nets), `reward_band_report.py` over-budget fraction (12/4 = 18–22 % of non-identity steps on the live v3 traces vs 2 % on v2b's 1-pass walk), and the first-freeze thin TRAJ vs §93/§94. Expect ~3× fewer episodes per GPU-day; that is the cost side of the caption.

**Read.** If ft40's first freeze keeps *less* than mild at equal Δacc, the 12/4 budget was teaching conservatism (train sees harsher Δacc than TEST) and every v2/v3/V4 row must be captioned "trained under a harsher recovery than it was TESTed with". If not, 12/4 is a free cost cut and the caption stays.

---

## GPU order (unchanged from `PROMPT_FABLE_V5.md` §0d)

1. In-flight frozen-snap TRAJs (neonraw `21442936` → bnscale ep0011 → **V4 ep0083**) — evidence.
2. Probe **A** (P8), thin C-G / C-G+ first (cheap), then Catalog L r56 ×3.
3. Probe **C** (ft40) — it is a train job; it can share the window with A on a second hole.
4. Probe **B** (C100 gate) under the recipe A picked.
5. Next DRL: `offline_train_v5_p5b3` (or `_cgp`) — only after B admitted ≥ 2 nets and A picked the recipe.

Do not scancel v3/V4 to start any of these.
