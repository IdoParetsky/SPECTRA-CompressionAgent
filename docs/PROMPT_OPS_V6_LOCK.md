# Ops handoff — V6 sitting lock (Fable, 21 Sep 2026 18:45 IDT) — paste into "SPECTRA overnight operations"

You are the SPECTRA ops agent (Grok 4.6). Fable's 21 Sep sitting locked the next cycle
(`docs/PROMPT_FABLE_V6.md` §8). Standing rules hold: do not overlay leap `src/`; do not scancel
v3 / ft40 / in-band; QOS cap **6**; quote `[eval] TRAJ val_best` only; skip r32; never wrap /
`pass 1/1` / terminals over τ; do not edit `SPECTRA_draft.md`; no ImageNet DRL; no C-G DRL; no
new ranking-menu train; do not reopen BERT.

## 1. The hole is filled — do not resubmit

**`21535193` `traj-v5-ft40-ep0059`** R since 17:36 (`cs-pheno-05`), submitted by Fable from
`/home/paretsky/scratch_audit/tree` (ft40's tree). Skip-train thin TRAJ of `21443408
snapshots/ep0059` (probe 0.2679). Pins verified: passes 2, 5-action fpgm, `ft_recipe=A`, train FT
40/10, `cbrt`, group-once, align next, group-cost, det=1, TRAJ. QOS **6/6**.

- Log: `/home/paretsky/scratch_audit/tree/runs/slurm_logs/spectra_21535193.out`
- Greps: `grep -E '\[eval\] TRAJ (floor_hold|val_best|terminal)|Traceback'`
- On COMPLETED: ledger **§112** (PRELIM), quote `val_best` for r20-w2 and r56-w4 vs **§93**
  (2-pass mild), **§95** (v3-fpgm 12/4 twin), **§111** (in-band), **§99** (neonraw). The question
  the row answers: does the 40/10 freeze walk differently from the 12/4 arms (deeper keep in band,
  or ≥ 1 pp kinder at equal keep on **r56-w4**)? r20-w2 will be 0.536/0.655 for every method —
  say so, do not read it as a comparison (Fable §8.2). Ping Ido with the r56 line.

## 2. Fill order for the next holes (submit one per freed GPU; never a 7th)

| # | Job | Command (from **leap**, recipe A, det TRAJ) | Nice | Why |
|---|---|---|---|---|
| 1 | 3-pass mild thin | `SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=ctl-thin-mild-3pass bash scripts/submit.sh baseline_c10_mild_traj_gonce` | 0 | matched-keep yardstick for the 0.756 r56 rows (§99, §111, maybe §112) |
| 2 | 3-pass L1 thin | `SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=ctl-thin-l1-3pass bash scripts/submit.sh baseline_c10_l1_traj_gonce` | 0 | same |
| 3 | Catalog L twins mild | `SPECTRA_EVAL_PASSES=2 SPECTRA_DATASET_NAMES="cifar-10 cifar-100" SPECTRA_INPUT=<repo>/configs/input_catalog_l_twins.json SPECTRA_JOB_NAME=ctl-catl-twins-mild bash scripts/submit.sh baseline_c10_mild_traj_gonce` | 5 | bar 2 of the Catalog L lock (`CATALOG_L_TEST_PLAN.md` §5) |
| 4 | Catalog L twins L1 | same with `baseline_c10_l1_traj_gonce`, `SPECTRA_JOB_NAME=ctl-catl-twins-l1` | 5 | same |
| 5 | **Next DRL train** `offline_train_v6_inband_p5b2` | see §3 — only after #1 has a job id and after Ido's morning reply | 20 | thesis actor for Catalog L |

`configs/input_catalog_l_twins.json` is on Ido's laptop tree and in `tree_v6_dev`; copy it to leap
`configs/` before #3 (it is a data file, not `src/`). Ledger each control as PRELIM, no-agent.

## 2b. V7 re-gate probes (Fable 21 Sep 19:40) — after the controls in §2, before any DRL train

Ledger §109 admitted no CIFAR-100 net under the train FT **Adam 1e-3** 12/4. Fable's hypothesis: the LR,
not the nets (`docs/V7_TRAIN_CATALOG.md` §4). Four heuristic jobs, ~4 h each, from a tree that carries
the new `SPECTRA_FT_LR` env (`tree_v6_dev` → copy to `tree_v7` first; **not** leap):

```bash
rsync -a --delete --exclude runs --exclude '*.out' --exclude .git --exclude __pycache__ \
  /home/paretsky/scratch_audit/tree_v6_dev/ /home/paretsky/scratch_audit/tree_v7/
cd /home/paretsky/scratch_audit/tree_v7 && mkdir -p runs/slurm_logs && export SPECTRA_REPO_DIR=$PWD
C=$PWD/configs/v7_c100_candidates_input.json
# C100 candidates, two LR arms
SPECTRA_FT_LR=1e-4 SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 SPECTRA_DATASET_NAMES=cifar-100 \
  SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_JOB_NAME=v7-c100-regate-adam1e4 SPECTRA_NICE=10 bash scripts/submit.sh baseline_c10_mild_traj_gonce
SPECTRA_FT_OPTIM=sgd SPECTRA_FT_SGD_LR=0.01 SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 SPECTRA_DATASET_NAMES=cifar-100 \
  SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_JOB_NAME=v7-c100-regate-sgd01 SPECTRA_NICE=10 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# CIFAR-10 controls of the same two arms on thin (compare to §93 at 40/10 and to the 12/4 Adam 1e-3 walk)
SPECTRA_FT_LR=1e-4 SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 \
  SPECTRA_JOB_NAME=v7-thin-ctl-adam1e4 SPECTRA_NICE=12 bash scripts/submit.sh baseline_c10_mild_traj_gonce
SPECTRA_FT_OPTIM=sgd SPECTRA_FT_SGD_LR=0.01 SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 \
  SPECTRA_JOB_NAME=v7-thin-ctl-sgd01 SPECTRA_NICE=12 bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

Admission rule per C100 net: `val_best` kept ≤ 0.98 and val Δacc ≥ −10 → `admitted` in
`configs/v7_c100_gate.json` (Ido's tree). Confirm in each log that `Fine-tune recipe: optim=… lr=…` shows
the arm. Report the five-line verdict per arm to Ido; do not decide the recipe from ops.

## 3. Next DRL train — ready, **not** submitted (Ido GO required)

Profile `offline_train_v6_inband_p5b2` (in-band linear reward × Catalog-L-clean catalog
`configs/database_offline_v6_p5b2.json`: 9 C10 nets with VGG-13, + VGG-11 SVHN; probes
`vgg13_bn_cifar10_,resnet56-width6`; train FT 12/4). It exists only in
`/home/paretsky/scratch_audit/tree_v6_dev` (and Ido's laptop). Before submitting:

```bash
rsync -a --delete --exclude runs --exclude '*.out' --exclude .git --exclude __pycache__ \
  /home/paretsky/scratch_audit/tree_v6_dev/ /home/paretsky/scratch_audit/tree_v6_train/
cd /home/paretsky/scratch_audit/tree_v6_train && mkdir -p runs/slurm_logs
SPECTRA_REPO_DIR=$PWD SPECTRA_JOB_NAME=v6-inband-p5b2 SPECTRA_NICE=20 bash scripts/submit.sh offline_train_v6_inband_p5b2
```

**Fable 19:40 addendum:** submit with **`SPECTRA_PROBE_SCORE=area`** (the legacy selection score
saturates at the mild walk — `docs/V7_OVERHAUL_PROPOSAL.md` §1.1; the flag is in `tree_v6_dev`) and,
if the §2b re-gate admitted ≥ 4 CIFAR-100 nets, switch the database to
`configs/database_offline_v7_diverse_admitted.json` with that arm's FT LR flags (then it is the **V7**
train — Ido names it). Add `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10` **only if** §112 says the 40/10
freeze walks differently (Fable §8.6 rule); otherwise 12/4. Confirm FLAGS at start:
`SPECTRA_REWARD_SCALE=cbrt_cubes`, `SPECTRA_REWARD_MODE=structural`, database = `..._v6_p5b2.json`,
`SPECTRA_PROBE_NETS=vgg13_bn_cifar10_,resnet56-width6`, `FT_REINIT_EDITED=0`. Snapshot pings as
for in-band; do not auto-TEST a freeze — Ido or Fable says GO.

## 4. Counterfactual probe (new, default off) — use on the next actor TRAJ

`SPECTRA_EVAL_COUNTERFACTUAL=1` logs, per actor step, `[cf] act= zero= shuf= blind= pmax=
content_used= state_used=` (does the frozen policy read the state?). Code = `src/fortify.py` +
`a2c_agent_reinforce_runner.py` from Ido's tree / `tree_v6_dev` (269 tests green). Overlay those
two files onto `tree_v6_inband` and `tree` **only when no job is starting from that tree**
(running jobs already imported; a PD job about to start would import mid-copy). Then set the
flag on the next actor TRAJ (e.g. in-band ep0095 if Ido asks for it, or the next freeze). Report:

```bash
grep -h '^\[cf\]' <log> | awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="content_used")c+=a[2];if(a[1]=="state_used")s+=a[2]};n++} END{printf "steps=%d content_used=%.1f%% state_used=%.1f%%\n",n,100*c/n,100*s/n}'
```

## 5. Docs Fable wrote this sitting (read, do not re-derive)

- `docs/PROMPT_FABLE_V6.md` §8 — all §7 answers, §7.8 lock, fill order.
- `docs/paper/CATALOG_L_TEST_PLAN.md` §5 (locked protocol) + §6 (thesis §4.1 draft). Ido signs.
- `docs/V6_REPRESENTATION_DESIGN.md` — identify-first ladder, group-as-token spec, kill list.
- `docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md` — Q1–Q3 Fable slots filled (send to Gilad).
- `docs/paper/LOOP_ALGORITHMS.md` §7 — grocery list replaced by the lock.
- Ledger §98 addendum — ft40 and in-band gain census (both 0). No new TEST row from Fable.
- Configs: `catalog_l_map.json` flags; `database_offline_v5_p5b3*` VGG-16→VGG-13;
  `database_offline_v6_p5b2.json`; `input_catalog_l_twins.json`; `input_v6_svhn_remaining.json`.

## 6. Never (this handoff)

Resubmit `21535193`. Submit the DRL train before Ido's GO. C-G / C-G+ DRL. Catalog L DRL of any
current actor (all are in-catalog on L1/L2). TEST in-band ep0095 or bnscale ep0107 without a GO.
Touch `tree_v6_dev` (Fable's) or leap `src/`. Release JobHeldUser heuristics. Read r20-w2 as a
policy comparison.
