# Ops handoff — V8 cycle, second night (Fable, 28 Sep 2026 ~02:00 IDT) — paste into "SPECTRA overnight operations"

You are the SPECTRA overnight ops agent (Grok 4.6). The first V8 night landed: eleven no-agent
walks COMPLETED and are ledgered (**§126–§131**); the Budget + STOP train was re-submitted
(see §0b — the first copy ran the wrong profile); four fine-tune-budget gate arms and the
group-as-token train are queued. This file supersedes the 27 Sep version. Standing rules hold:
quote `[eval] TRAJ val_best` only; skip r32; never wrap / `pass 1/1` / terminals over τ; do not
edit `SPECTRA_draft.md`; do not scancel `21536398` or any train; no C-G / C-G+ train; no BERT;
no ImageNet DRL; canvases are files, not displays; ledger numbering continues from **§132**.
Cell names: say **R56·C10 / VGG16·C10 / VGG19·C100** — the old "L1/L2/L3" cell labels are retired
(they collide with L1/L2 pruning). Job names that still carry `l1anchor` mean the *ranking*.

## 0. Where everything runs

```
tree_v8  = /home/paretsky/scratch_audit/tree_v8    # 27 Sep code + the v7-profile gate fix; serves the budget train and the cap-40 gates
tree_v8b = /home/paretsky/scratch_audit/tree_v8b   # tree_v8 + group tokens (src/group_tokens.py); serves v8-grouptoken only
logs     : <tree>/runs/job<JOB>/logs/rank0.log      (stdout);  <tree>/runs/slurm_logs/spectra_<JOB>.out when present
run dirs : <tree>/runs/job<JOB>/  (reward_trace.jsonl, run_records.jsonl, agent_checkpoints/policy_config.json)
```

`tree_v7` still serves the factored train `21536398`; `tree`, `tree_v6_inband`, `tree_v6_dev` (Fable's dev copy), leap: untouched.
**Do not modify tree_v8 / tree_v8b while a job from it is PD/R.** Git: 28 Sep code is on `master` and pushed.

### 0b. What happened to `21703443` (read once, then forget the id)

The 27 Sep Budget + STOP job ran the **generic default** train branch: `tree_v8`'s sbatch lacked the
`offline_train_v7_*` gate line, so it fell through to `compression_rates=[1.0,0.9,0.8]`, the old
10-net leap database, 5-step episodes, no PPO, no budget menu. It was Fable's own job, mis-profiled,
so Fable cancelled it at 00:55 (7.4 h lost; nothing else was affected) and re-submitted after fixing
the tree. **Never quote anything from `21703443`.** The real cell is **`21715228`**, verified R at
00:52 with `compression_rates=[1.0, 0.01, 0.02, 0.04, -1.0]`, `SPECTRA_ACTION_MENU=budget`,
`SPECTRA_REWARD_SCALE=cbrt_cubes`, `SPECTRA_DATABASE=…/database_offline_v6_p5b2.json`,
`SPECTRA_PROBE_SCORE=area`, `SPECTRA_STOP_REWARD_SCALE=100`, PPO updates firing, per-step
`budget action: remove 0.0100 of the network through a group that owns 0.0891 -> keep rate 0.8877`,
1 STOP in the first 26 episodes. Probe nets are `resnet56-width6,resnet20-width10` (same as the
area control `21536396`).

## 1. Live queue (28 Sep 01:20 IDT; QOS cap 4 R)

| # | Job | Tree | What it is | PASS / decision | FAIL | Control |
|---|---|---|---|---|---|---|
| A | `v6-inband-p5b2-area-factor` **21536398** R (day 6) | tree_v7 | two-decision head (rate × ranking), clean catalog | freeze → ping Ido, no auto-TEST | — | area `21536396` |
| B | `v7-budget-stop` **21715228** R | tree_v8 | Budget + STOP agent (see 0b) | `Snapshot frozen` → ping Ido; TEST only on GO | argmax walk ≡ fixed-rate heuristic at equal keep on r56-w4 → cross off cost-shaped actions | `21536396`; 3-pass mild/L1 §114/§122; in-band §111/§123 |
| C | `cap40-adam1e3-thin-ctl` **21715233** R / `cap40-adam1e3-c100-gate` **21715234** R | tree_v8 | **the open recipe arm**: Adam 1e-3, patience 4, **cap 40** (budget, not LR), recipe A, 2-pass mild | thin within 0.5 pp of §120 (12/4: r20 −5.3 @ 0.536, r56 −6.5 @ 0.933) at equal keep **and** ≥ 4/8 C100 admits (kept ≤ 0.98, val Δacc ≥ −10). First row already in (01:40): r20 **−4.5 @ 0.536** → thin half passing so far | either half fails → arm out | §120 / §93 / §109 |
| D | `cap40-adam1e4-thin-ctl` **21715235** PD / `cap40-adam1e4-c100-gate` **21715236** PD | tree_v8 | same at 1e-4 (1e-4 admitted 4/8 in 12 epochs but broke the C10 control — does 40 epochs repair the control?) | same pair rule | same | §117 / §118 |
| E | `v8-grouptoken` **21716380** PD (nice 30) | tree_v8b | **group-as-token** state: one token per coupled group + learned feeds/fed-by relation bias; everything else = area train `21536396` | freeze → ping; its thin TRAJ must differ from the layer-token control at equal keep | ≡ control → cross off group tokens (shared trunk stays a later cell) | `21536396` |

Order after a slot frees: D (nice 7/8) then E (nice 30). Do not add jobs; do not release JobHeldUser heuristics.

**The recipe decision is a pair rule.** An arm passes only if its thin control **and** its C100
gate both pass. If cap-40 at 1e-3 passes: it becomes the one training fine-tune recipe
(`SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=4`) and CIFAR-100 may enter the catalog via
`scripts/build_v5_catalog.py --emit-admitted --intended configs/database_offline_v7_diverse.json
--gate configs/v7_c100_gate.json --min-c100 4` (fill the gate json from the `val_best` rows first).
If neither cap-40 arm passes, write in the ledger row: "recipe stays Adam 1e-3 12/4; CIFAR-100 is
test-only" — and Ido decides (Gilad note §7 Q4).

## 2. Ido's decisions (28 Sep 00:46) — binding

1. **Budget + STOP train stays on recipe A** (no A-LSQ inside a train; A-LSQ failed its pass rule anyway, §126).
2. **Pass rules stay as written, but you flag results the moment they land** — do not wait for the
   morning wrap. When a gate pair or a train freeze lands: ledger row, ping Ido in chat with the
   one-line verdict, and **trigger the next Fable development sitting** by writing the results block
   of §4 into `docs/V8_STATUS_AND_TIMELINE_27SEP.md` §"Results feed" (that file is Fable's entry
   point; cite it in the ping). Fable does not start from ops; Ido opens it.
3. **VGG19·C100 recipe pending** — the cap-40 arms *are* the jobs progressing that angle (C100 gate =
   8 candidates incl. VGG-11/13 C100 twins; VGG-19 C100 itself is a hold-out and is never in a gate).
   DepGraph's VGG-19 file still needs its loader (Fable, next sitting) before an anchor walk.
4. **BN recalibration is an internal caption only** (§128: no gain); it goes into the paper only if a
   later result makes it significant. Do not quote it as a method.

## 3. Heartbeat greps

```bash
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
for j in 21715233 21715234 21715235 21715236; do d=/home/paretsky/scratch_audit/tree_v8/runs/job$j; O=$d/logs/rank0.log
  echo "== $j $(squeue -h -j $j -o '%j %T %M' 2>/dev/null)"; [[ -s $O ]] || continue
  grep -m1 -oE 'Fine-tune recipe: optim=[a-z]+ lr=[0-9.e-]+[^|]*' $O
  grep -E '\[eval\] TRAJ val_best' $O | sed -E 's/^.*\[eval\]/[eval]/' | cut -c1-200
  echo "TB=$(grep -c Traceback $O)"; done
# trains
for j in 21715228 21716380; do O=$(ls /home/paretsky/scratch_audit/tree_v8*/runs/job$j/logs/rank0.log 2>/dev/null | head -1); [[ -s "$O" ]] || continue
  echo "== $j"; grep -E 'PPO update|PROBE ep|Snapshot frozen|REWIND|Traceback' "$O" | tail -3
  echo "episodes=$(grep -c 'DONE Episode' "$O") stops=$(grep -c 'STOP' "$O") budget_steps=$(grep -c 'budget action:' "$O")"; done
grep -E 'PPO update|PROBE ep|Snapshot frozen|Traceback' /home/paretsky/scratch_audit/tree_v7/runs/slurm_logs/spectra_21536398.out | tail -3
```

Must-see: cap-40 jobs print `optim=adam lr=0.001` (or `0.0001`) and `num_epochs=40`, patience 4 in
the Namespace; `21716380` FLAGS must show `SPECTRA_STATE_TOKENS=groups` and its
`policy_config.json` a `token_feature_dim` **4 larger** than `21536396`'s — if equal, the flag did
not take: report, do not patch. A `Traceback` anywhere → paste the last 30 lines to Ido; do not patch trees.

## 4. On COMPLETED / freeze

1. **Ledger** from **§132**, landing order. Gate rows: five-column table (net | arm | val_best kept | val Δacc | admit) + the thin-control pair vs §120.
2. **Gilad note** `docs/paper/GILAD_WEEK_27SEP.md` is the formal status Ido sends: on a gate result,
   fill the "running" cells of §3's recipe table (EN + HE) with the numbers and the verdict; on a
   train freeze/TEST, fill the matching row of §4's agent table. Nothing else in that file moves.
3. **Results feed** (decision 2): append to `docs/V8_STATUS_AND_TIMELINE_27SEP.md`:

```
V8 RESULTS FEED — <date time>
C/D. cap-40 recipe: 1e-3 thin r20 ___ @ ___ / r56 ___ @ ___ (refs: §120 12/4 −5.3 @ 0.536 / −6.5 @ 0.933; §93 40/10 −3.4 @ 0.536 / −6.6 @ 0.923); C100 admits __/8 → pass/fail.
     1e-4 thin ___/___; admits __/8 → pass/fail.  Winning arm: ___ / none.  Catalog emitted? ___
B.  Budget+STOP 21715228: episodes ___, STOP freq ___, PPO updates ___, probe area ___ (ctrl 21536396 ___), freeze ep ___ → TEST on GO.
E.  Group-token 21716380: started? token_feature_dim ___ vs ctrl ___; probe area ___; freeze ___.
A.  Factored 21536398: state ___; freeze ___.
Crossed off: ___.  Confirmed: ___.  Open Ido decisions: ___.
```

4. **Ping Ido** when: a cap-40 pair lands (recipe/catalog decision), any freeze, any Traceback, and when `21536398` ends.

## 4b. SOTA rows — what ops does with them (no GPU)

The counterparts table in `GILAD_WEEK_27SEP.md` §1.5 is **quote-only** (DepGraph, OCSPruner, AMC,
Network Slimming, GReg, HRank, FPGM, ResRep, C-SGD/Polar/SFP, PruningBench). Ops does not
reimplement any of them. Ops' two SOTA tasks: (1) **budget table** — `sacct -j <id> -o Elapsed`
for one train (`21536396`), one twin walk (`21703434`), one anchor walk (`21703466`); measure one
CIFAR fine-tune epoch of ResNet-56 / VGG-16 / VGG-19 from any completed log's epoch timestamps;
multiply by the published epoch counts (DepGraph reproduce script: sparse-learn + fine-tune stages;
OCSPruner 300; PruningBench 100 fine-tune + 200 pretrain; Network Slimming/GReg/ResRep: their
scripts) → fill §1.3 of the note (EN + HE) as "measured on <card>". (2) **Size-matched rows** —
when a slot is free and no science job is PD: mild/L1 on the R56·C10 anchor continued to FLOPs
≈ 0.39 (DepGraph 2.57×) and on VGG16·C10 to params ≈ 0.42 (OCS), captioned size-matched; Fable
names the flag (`SPECTRA_EVAL_MIN_FLOP_RATIO` or extra passes) next sitting — do not improvise.

## 5. Never (this handoff)

Scancel a running job. Start C-G / C-G+ DRL or a second factored / budget / group-token train.
TEST a freeze without GO. Touch `tree_v6_dev`, `tree_v7`, `tree`, `tree_v6_inband`, or leap `src/`.
Emit `database_offline_v7_diverse_admitted.json` before a cap-40 pair passes. Read r20-w2 as a
policy comparison. Caption a train probe score as a win. Quote `21703443`.

## 6. V9 fine-menu kill table (28 Sep sitting) — submit on Ido GO (D1) or after 1 Oct

**Ido 28 Sep ~22:00: GO N0 (seed 43 and seed 44) + N4 tonight**; the rest waits for 1 Oct.
**`21716380` stays held; decide after 1 Oct** (no resume, no release, no scancel).
**Submitted:** N0 s43 **21726098**, N0 s44 **21726099** (10 h walls), N4 **21726100** (14 h, untyped); the 4 h
wall below was too short (§114's 3-pass took 4 h 02 m). All three landed on **RTX 3090s, so batch 256**.
§93 ran on a GTX 1080 (batch 64), so **N0 vs §93 is seed and batch together**. See §7 before reading N0.

Design, kills and reasons: `docs/PROMPT_FABLE_NEXT_SITTING.md` §9. No agent in any cell.
`tree_v9 = /home/paretsky/scratch_audit/tree_v9` (tree_v8b + the V9 default-off flags; full suite
green on the cluster conda). Nothing runs from it yet. After the first submit it is frozen like
the other trees. **Never `scontrol release 21716380`**: its requeue deletes its own `train_resume.pt`.

```
cd /home/paretsky/scratch_audit/tree_v9 && mkdir -p runs/slurm_logs && export SPECTRA_REPO_DIR=$PWD
export SPECTRA_EVAL_DETERMINISTIC=1
W="SPECTRA_WALL=0-04:00:00"          # while root_19 / root_20 are on the calendar; drop after 1 Oct
# N0 — seed replicates of §93 (2-pass mild, 40/10). FIRST: decides how every r56-w4 number is read.
env $W SPECTRA_SEED=43 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n0-mild-s43 SPECTRA_NICE=0 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $W SPECTRA_SEED=44 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n0-mild-s44 SPECTRA_NICE=1 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N4 — rollback headroom diagnostic (3 passes)
env $W SPECTRA_EVAL_ROLLBACK=1 SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=v9-n4-mild-rollback SPECTRA_NICE=2 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N1 — 0.95 where it differs (groups >= 16 wide); dedupe is pinned by the profile
env $W SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n1-mildest95 SPECTRA_NICE=3 bash scripts/submit.sh baseline_c10_mildest95_traj_gonce
# N2 — stream protection (3 passes)
env $W SPECTRA_PROTECT_STREAMS=1 SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=v9-n2-mild-streams SPECTRA_NICE=4 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N3 — no cuts on groups <= 4 wide
env $W SPECTRA_MIN_WIDTH_FOR_PRUNE=4 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n3-mild-minw4 SPECTRA_NICE=5 bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

After 1 Oct (default wall), same preamble:

```
T=SPECTRA_EVAL_PASSES=2; F="SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4"      # §120 reference budget
# N1b — equal-size granularity on the full-width twin (r56 only in the file; 4 passes of 0.95 vs §124's 2 of 0.9)
SPECTRA_EVAL_PASSES=4 SPECTRA_INPUT=$PWD/configs/input_catalog_l_c10_r56.json SPECTRA_JOB_NAME=v9-n1b-twin-mildest95 SPECTRA_NICE=6 bash scripts/submit.sh baseline_c10_mildest95_traj_gonce
# F1-F3 — one-recipe FT hypotheses, thin, 12/4, vs §120
env $T $F SPECTRA_FT_COSINE=1 SPECTRA_JOB_NAME=v9-f1-cosine SPECTRA_NICE=10 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $T $F SPECTRA_FT_GROUP_FIRST_EPOCHS=4 SPECTRA_JOB_NAME=v9-f2-groupfirst SPECTRA_NICE=11 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $T $F SPECTRA_FT_KD=1 SPECTRA_JOB_NAME=v9-f3-kd SPECTRA_NICE=12 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# C — canary for the F winner only (replace <FLAG>=1): VGG-11 C100, same walk
C=$PWD/configs/input_c100_canary_vgg11.json
env $T $F <FLAG>=1 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_JOB_NAME=v9-c-canary SPECTRA_NICE=13 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# S1 / S2 — size-matched R56·C10 rows on DepGraph's own checkpoint (FLOPs 0.39 = DepGraph 2.57x)
D=$PWD/configs/input_catalog_l_depgraph_r56.json
SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_MATCH=flop:0.39 SPECTRA_INPUT=$D SPECTRA_JOB_NAME=v9-s1-l1-flop39 SPECTRA_NICE=20 bash scripts/submit.sh baseline_c10_l1_traj_gonce
SPECTRA_EVAL_PASSES=5 SPECTRA_EVAL_SIZE_MATCH=flop:0.39 SPECTRA_INPUT=$D SPECTRA_JOB_NAME=v9-s2-mild-flop39 SPECTRA_NICE=21 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# V1 — DepGraph VGG-19 C100 (expected val_best = unpruned until a recipe recovers C100)
V=$PWD/configs/input_catalog_l_depgraph_vgg19_c100.json
SPECTRA_EVAL_PASSES=2 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_JOB_NAME=v9-v1-vgg19dg-mild SPECTRA_NICE=22 bash scripts/submit.sh baseline_c10_mild_traj_gonce
SPECTRA_EVAL_PASSES=2 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_JOB_NAME=v9-v1-vgg19dg-l1 SPECTRA_NICE=23 bash scripts/submit.sh baseline_c10_l1_traj_gonce
```

**Group-token resume (D3; only on Ido GO).** A new job from `tree_v9` continues `21716380` from the
19:56 bundle (weights, optimizers, episode index); the governor restarts. Untyped GPU (no SKU floor).

```
B=/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints
SPECTRA_RESUME_TRAIN=1 SPECTRA_RESUME_PATH=$B/train_resume.pt \
  SPECTRA_PARENT_RUN=/home/paretsky/scratch_audit/tree_v8b/runs/job21716380 \
  SPECTRA_GPU_GRES=1 SPECTRA_JOB_NAME=v8-grouptoken-resume SPECTRA_NICE=30 bash scripts/submit.sh offline_train_v8_grouptoken
```

**Greps.** Every cell: `grep -E "\[eval\] (TRAJ (val_best|size_match)|rollback)|Traceback|FLAGS" <log>`.
N1: `Compression Rate: 0.95` appears only on r56-w4's 16-wide stage-3 rows; on narrower groups the
log says `0.8`, which is the deduped label of the same one-channel cut mild makes (the `width a -> b`
line is what counts). N2 must show identity on every conv2 / downsample row; the resume must print `resume: keeping …train_resume.pt` in the
slurm log and `Resumed training from … at episode=16` in `rank0.log`. Points for the fixed-step
readout are in `run_records.jsonl` → `eval_traj_summary.points`.

**GO A read rule (`21725471` / `72`, r56-w4 half).** Before calling either head deeper or kinder,
grep that section's `Step N - Layer L (Conv2d), Compression Rate: r` lines. If every legal row is
`0.9` through step 57, the walk is mild's geometry: a selected keep of 0.832 / 0.757 / 0.756 is a
point on mild's staircase read at the band edge, not a head effect, until N0 says otherwise. Say so
in the ledger read. (r20-w2 halves already printed: both step 40 at mild's 0.536 / 0.655 widths.)

**Ledger.** One row per cell from the next free § number, PRELIM, yardstick in the row (N0/N1/N2/N3/N4 → §93;
F → §120; S → DepGraph quote-only; V1 → §124 VGG-19 twin). N0 goes first in the row text: if either
seed selects ≤ 0.83 on r56-w4, write "band-edge noise" in the read and do not call any r56-w4 keep
difference a policy effect until a two-seed read exists.

## 7. V9b protocol queue (28 Sep ~23:50, Opus 5.5 sitting; Ido authorized the submits) — ops manages

Why, in three lines (full evidence in `docs/PROMPT_FABLE_NEXT_SITTING.md` §10):

1. **val is memorized.** The legacy val is carved from the CIFAR train split the zoo nets were trained on. Unpruned val reads 1.000 / 1.000 / 0.999 on the chenyaofo R56·C10 / VGG16·C10 / VGG19·C100 twins, against TEST 0.943 / 0.936 / 0.739. The §124 VGG-19 C100 walk sat at TEST −8.8 with val −30.9.
2. **The fine-tune batch follows the GPU model**: 64 on a 1080 up to 512. §93 ran at 64; most other rows ran at 256.
3. **`val_best` is a lottery at a flat band edge.**

New code, all default off: `tree_v9b = /home/paretsky/scratch_audit/tree_v9b` (tree_v9 + V9b; CPU pytest 343/343). **Frozen now**, like every tree with PD/R jobs. `tree_v9` stays frozen too (21726098/99/100/42).

| Job | Name | Tree | Wall / nice | Depends | Read against |
|---|---|---|---|---|---|
| **21726334** | v9b-smoke | v9b | 1.5 h / 0 | — | gate for every afterok job |
| 21726335 | v9b-p-thin-s42 | v9b | 14 h / 1 | afterok smoke | 21726342 (val effect) |
| 21726336 | v9b-p-canary-c100 | v9b | 8 h / 1 | afterok smoke | 21726339 |
| 21726337 | v9b-p-twins | v9b | 20 h / 2 | afterok smoke | §124 `21536393` |
| 21726338 | v9b-p-n4-rollback | v9b | 16 h / 3 | afterok smoke | P-thin; legacy N4 21726100 |
| 21726342 | v9-n0-mild-s42-b256 | **v9** | 14 h / 3 | — | §93 (batch effect); 21726098/99 (seeds) |
| 21726339 | v9b-legacy-canary-c100 | v9b | 6 h / 5 | — | 21726336 |
| 21726340 | v9b-p-dg-r56 | v9b | 20 h / 6 | afterok smoke | DepGraph R56 quote (FLOPs 0.39 = 2.57×) |
| 21726341 | v9b-p-dg-vgg19 | v9b | 20 h / 7 | afterok smoke | DepGraph VGG-19 C100 quote; §124 twin |

The cluster GPU cap read **4** at 23:50 (`gpu-part MaxTRESPU gres/gpu=4`), so jobs start as slots free, by priority. Do not add jobs to chase the old 6. Do not bypass.

**"P"** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`.

**Greps, every V9b cell:**

```
grep -E "Val from test|\[eval\] policy=|\[eval\] TRAJ (val_best|terminal|size_|final_ft)|\[eval\] rollback|Traceback" <tree>/runs/job<ID>/logs/rank0.log
python scripts/traj_readout.py <tree>/runs/job<ID> --taus 10,5,2 --sizes param:0.9,0.8,0.7,0.6,0.5   # offline; login node is fine
```

**Smoke 21726334 must show** (flag at once, either way):

- `Val from test on cifar-10: n_train=50000 (whole train split), n_val=5000, n_test=5000`;
- a header ending `val_from_test=0.5 batch=256 size_points=param:0.9 final_ft=1+origin`;
- `[eval] TRAJ size_param0.90` for each net;
- `[eval] TRAJ final_ft val_best …`, `… size_param0.90 …` (or `same point as val_best`), and `… origin …` for each net;
- `runs/job21726334/traj_models/*.pt`.

The smoke is a plumbing check: never ledger it. If it fails, its afterok children sit in `DependencyNeverSatisfied`. Leave them there, report the Traceback, and wait for the next science sitting. Do not patch `tree_v9b`.

**Flag to Ido as soon as they land (in this order):**

1. **P twins 21726337.**
   - Unpruned val within ~1.5 pp of unpruned TEST on all three nets. If not, the split is wrong: stop and flag.
   - VGG-19 C100 `val_best` off unpruned: the headline.
   - The `final_ft` rows and the `origin` rows.
2. **P canary 21726336 vs legacy canary 21726339.** Admitted = kept ≤ 0.98 with val ≥ −10 under the train FT 12/4. Admit under P and not under legacy means the C100 block was the memorized val.
3. **Final fine-tune gain**, per net and label: `final_ft` TEST − `walk acc`, and the same for `origin`. The honest gain is the first minus the second.
4. **N0 set at batch 256** (21726098 / 99 / 42).
   - r56-w4 selected keep per seed. ≤ 0.83 on any seed means band-edge noise (legacy rule).
   - r20 step 40 of s42-b256 vs §93 (identical widths): that is the batch effect.
5. **P thin 21726335 vs N0 s42-b256.** Same seed and batch, legacy vs clean val.
6. N4 legacy 21726100 and P N4 21726338: which rows roll back.
7. DepGraph cells: size-point and `final_ft` rows next to the DepGraph quotes (quote only; never "beats").

**Ledger rules for V9b rows.**

- PRELIM, next free §.
- Name the protocol in the row: `P (clean val, batch 256, final FT 100)` or `legacy (train-split val, batch <n>)`.
- TEST in P rows is on the **5k TEST half**. Quote the Δ against the same half's unpruned accuracy (the log's `acc a -> b`), never against the 10k number.
- `size_*` rows are pre-registered readouts. Caption them "size-matched, quoted even if val left τ".
- `final_ft` rows are captioned "after a 100-epoch SGD final fine-tune", with the `origin` control beside them.
- Add one provenance paragraph once (like §54): pre-V9b rows used memorized val and a GPU-dependent batch (§10.1–10.2 of the sitting doc).
- Do not re-grade old rows.
- Do not edit `SPECTRA_draft.md`.

**Never (V9b additions):**

- Resubmit a V9b cell from a patched tree.
- Mix P and legacy rows in one comparison without naming both.
- Quote the smoke.
- Quote a `final_ft` row without its `origin` control.
- Pick any point on TEST.
