# Run record — V9b / V9c sitting, 29 Sep 16:00 → 30 Sep ~03:30 IDT (Opus 5.5 MAX)

What was built, submitted and learned in this sitting, in one place. Numbers are copied from the ledger (`docs/paper/RESULTS_LEDGER.md`), which stays the record of record. The way ahead is `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. The live queue is `docs/SITTING_GPU_QUEUE.md`.

## 0. Mandate

- **Ido 29 Sep 15:49.** QOS 4 stays full with independent no-agent TESTs. The sitting ranks, writes the queue and sbatches. Still needs Ido: a DRL train, releasing `21716380`, the diverse catalog emit, freeze TESTs.
- **Ido 30 Sep 01:56.** GO for Stage 4 (one DRL train with the clean-val reward), unless something more urgent is due.
- **Standing.** No `src/` overlay on the leap tree. No patch to a tree while its jobs are PD/R. No `SPECTRA_draft.md` edits. No ImageNet DRL. Quote `[eval] TRAJ` rows only; never pick on TEST.

## 1. Code

`tree_v9c` = `tree_v9b` + the items below (commit `502ec84`, CPU pytest **367/367** on the cluster conda). All default off.

| Item | Where | What it does |
|---|---|---|
| TRAJ candidate saves | `src/traj_models.py`, runner | `state_dict` + arch/recipe JSON, never the live module. Zoo classes load by file path, so pickling the module crashed 21726337. A failed save logs `TRAJ save failed` and never raises. |
| Per-candidate isolation | `a2c_agent_reinforce_runner.py` (`_run_final_ft`) | One failing final-FT candidate prints `final_ft_failed` and the others still run. |
| Scratch-B control | `SPECTRA_EVAL_FINAL_FT_SCRATCH=both\|only` | Re-initialise the walk's architecture and train 200 ep SGD 0.1 (Liu et al. ICLR 2019), plus `origin+scratch`. |
| Final FT from a saved walk | `SPECTRA_EVAL_FINAL_FT_FROM=<run>/traj_models` | Loads `val_best` / size points / origin and runs the final FT only: new recipes cost no walk. |
| Offline readers | `scripts/paired_steps.py`, `final_ft_readout.py`, `crossfit_readout.py` | Paired early read by step (or by params); honest gain; 10k cross-fit and positive-Δ census. |

**Not in `tree_v9c` (next tree only, commit `5b6398d`).** The final-FT KD builds its own frozen teacher from the unpruned original when the env has none. Before, `SPECTRA_EVAL_FINAL_FT_KD=1` alone printed `kd=1` and distilled from nothing (latent since V9b; no queued cell used it). On `tree_v9c`, the NEXT N1 line carries `SPECTRA_FT_KD=1` instead.

**Bugs fixed while building V9c.**
- `reinit_parameters` skipped the root module.
- The test env lacked `param_ratio` / `flops_ratio`.
- A float boundary in a test.
- `torch.load` FutureWarning (now `weights_only=True`).
- `print_flush` capture in the full suite.

Tooling: `rexec.ps1 -Command` breaks on format strings, so use `-File`.

## 2. Trees on the cluster

| Tree | Path | Serves |
|---|---|---|
| `tree_v9b` | `/home/paretsky/scratch_audit/tree_v9b` | wave 1 (21729550–58): P walks, final FT with nothing saved |
| `tree_v9c` | `/home/paretsky/scratch_audit/tree_v9c` | wave 2 (21730498–516), wave 3 (21737104/05), P-train 21737095 |
| `tree_v7` | `/home/paretsky/scratch_audit/tree_v7` | control train 21536396 (area, legacy val) |
| `tree_v8b` | `/home/paretsky/scratch_audit/tree_v8b` | held group-token 21716380 |

All frozen. Wave-3 config outside any tree: `/home/paretsky/scratch_audit/configs_s30/input_catalog_l_twins_vgg.json` (VGG-16 C10 + VGG-19 C100 of the twins file).

## 3. Jobs

P = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256`: val is one 5k half of the test split, TEST the other. FT = 100-ep SGD final fine-tune + origin.

| Job | Name | Tree | State 30 Sep 03:00 | Ledger |
|---|---|---|---|---|
| 21729550 | v9b-smoke-ft | v9b | COMPLETED (plumbing) | never |
| 21729551 | v9b-ft100-dg-vgg19 | v9b | COMPLETED 00:27 | §149 |
| 21729552 | v9b-p-gate-c100 | v9b | R, net 7/8, wall ~08:21 | §148 |
| 21729553 | v9b-aug-twins | v9b | R, R56 near done; VGG-16 cannot finish | Pri 6 |
| 21729554 | v9b-aug-gate-c100 | v9b | R, net 7/8, wall ~08:29 | §148 |
| 21729555 | v9b-p-thin-12x4 | v9b | COMPLETED 01:45 | §150 |
| 21729556 | v9b-aug-thin-12x4 | v9b | COMPLETED 03:12, training rule passed | §150 |
| 21729557 | v9b-aug-thin | v9b | PD | Pri 11 |
| 21729558 | v9b-p-n2-streams | v9b | PD | Pri 17 |
| 21730498 | v9c-smoke-save | v9c | COMPLETED 00:34, **passed** | never |
| 21730499 | v9c-smoke-from | v9c | COMPLETED 03:14, passed | never |
| 21730500 / 01 / 06 | v9c-ft100-dg-r56 / thin / twins-c10 | v9c | PD | Pri 7 / 10 / 12 |
| 21730507 / 16 | v9c-scratch-thin / dg-r56 | v9c | PD afterok 501 / 500 | Pri 13 / 16 |
| 21730509 / 14 | v9c-cg-neon-twins / thin | v9c | PD | Pri 14 / 15 |
| 21737095 | v9c-p-area-train | v9c | submitted 02:10, held, **cancelled 03:11** (never started) | §151 |
| **21737123** | v9c-paug-area-train | v9c | **Stage-4 train**, released 03:11, R 03:14 on `ise-cpu256-32` (RTX 6000 Ada), start checks green | §151 |
| 21737104 | v9c-aug-twins-vgg | v9c | PD | Pri 6b |
| 21737105 | v9c-aug-ft100-dg-vgg19 | v9c | PD | N4 |

Wave scripts (for a shutdown resubmit; each skips names already queued): `scripts/_tmp_v9c_wave1.sh`, `_tmp_v9c_wave2.sh`, `_tmp_s30_wave3.sh`. Train lines: `_tmp_s30_paug_submit.sh` (Stage 4, P + aug) and `_tmp_s30_train_submit.sh` (the P-only line, kept for N9). A train that dies of infrastructure resumes with the ops §9.4 line, not a fresh submit. Ops heartbeat: `_tmp_s30_ops_hb.sh`.

## 4. Results (walk rows are P, 5k TEST half unless marked 10k)

1. **Census and cross-fit (§147, zero GPU).** 0 of 343 cut points on six full-width nets under P have val Δ > 0. Thin r20-w2 gains on 8 of 18. The mild walk reads neither half, so a two-fold τ rule gives 10k numbers for free, e.g. DepGraph R56 at FLOPs 0.39: −3.98 (10k).
2. **Crop+flip in the walk FT, C100 gate (§148, 6 of 8 nets).** Kinder at 12 of 12 equal-width size points, +0.9 to +6.2 pp TEST, mean +3.3. Paired read: `ADOPT?` on 5 of 6 nets. Both arms admit 6/6 at the live 12/4 recipe. The gate adopt rule is met.
3. **100-ep SGD final FT, DepGraph VGG-19 C100 (§149).** Honest gain +4.08 / +4.46 / +5.52 pp at `val_best` / size 0.70 / size 0.60. Final −2.52 @ 0.684 (10k −2.39). Origin +0.10. The O2 adopt rule is met on this cell. The re-walk moved one size point by 0.8 pp against §145: SKU / cuDNN noise at equal widths.
4. **Thin pair at the train FT 12/4 (§150): crop+flip passes the training rule.**
   - P control: r20 size 0.80 −0.1 @ 0.774; r56-w4 −10.6 @ 0.741.
   - 12/4 against 40/10: −1.03 pp paired val on r56-w4 (60 pairs), so the train recipe is harsher than TEST's.
   - Crop+flip on r56-w4, the decider: **+2.3 pp TEST at equal keep** (size 0.80, 10k +2.8). `val_best` goes from −10.6 @ 0.741 to −5.1 @ 0.622. Paired +5.03 over 60 cuts, better on 100 %.
   - Crop+flip on r20-w2, the guard: −1.0 / −1.3 / −3.1 pp TEST at equal keep, mean −1.8, inside the 2 pp guard. A 5k-param net that underfits.
5. **C10 twin R56 with crop+flip (Pri 6, val only).** +2.06 pp paired val over 56 cuts, better on 98 %. Step 104: −0.40 vs −3.16.
6. **Smoke-save 21730498 and smoke-from 21730499 passed.**
   - Smoke-save: header `final_ft=1+origin+scratch:both`. `traj_models/*.pt` + `.json` for `val_best`, the size point, origin, the `__ft1` copies and `+scratch`. 0 Traceback / PicklingError / `TRAJ save failed`.
   - Smoke-from (03:14, 2 min): `final_ft from …/job21730498/traj_models: ['size_param0.90', 'val_best']` for both nets. `val_best`, size and origin rows all `init=inherit`, 0 errors.
   - The smoke-from exposed a reader flaw. Its 1-epoch FT cost the origin −5.84 pp, and `final_ft_readout.py` then printed "honest +5.78 ADOPT" on a raw gain of −0.06. Fixed: `ORIGIN-HURT` when the origin loses > 0.5 pp, with a regression test (13/13 on the cluster conda). The fixed readers run from `/home/paretsky/scratch_audit/readers_s30/scripts/` until `tree_v9d`.
7. **Stage-4 train released (§151).** 21737123 = the area train under P + crop+flip.
   - Both candidate arms were held until §150's TEST rows printed. Then the P+aug arm was released and the P-only arm cancelled.
   - The P-only line stays as the attribution train (N9), to run only if 21737123 leaves mild.

## 5. Readouts (login node)

```bash
PY=/home/paretsky/.conda/envs/spectra/bin/python; cd /home/paretsky/scratch_audit/tree_v9c
$PY scripts/paired_steps.py <tree>/runs/job<ARM> <tree>/runs/job<CONTROL> [--by params]
$PY scripts/final_ft_readout.py <tree>/runs/job<ID>
$PY scripts/crossfit_readout.py <tree>/runs/job<ID> --taus 10,5 --sizes <size points>
```

Ready-made: `scripts/_tmp_s30_poll.sh` (queue, sacct, reservations, per-job TRAJ lines) and `scripts/_tmp_s30_reads.sh` (the paired reads and readouts of this wave). Run with `powershell -NoProfile -File scripts/rexec.ps1 -Quiet -File <script>`.

## 6. Scheduler findings

- **Maintenance `root_20` (29 Sep 21:00 → 30 Sep 18:00) did not touch us.** MAINT, IGNORE_JOBS, SPEC_NODES, on nodes none of our jobs used. All Restarts=0. No further reservation listed.
- **QOS cap is 4 running GPUs** (`QOSMaxGRESPerUser`). GPUs are not scarce: at 02:10, 31 RTX 6000, 35 RTX 4090 and 40 RTX 3090 were free.
- **Untyped jobs land on the slowest cards by design.** Node weights: 1080 11,421 < 2080 12,421 < 3090 14,842 < 4090 15,321 < RTX 6000 16,842, and Slurm fills the lowest weight first. That is why every `SPECTRA_GPU_GRES=1` cell ran on a 1080.
  - Nodes carry SKU features (`gpu,rtx_4090` …). The mixed `cs-pheno-*` nodes have both `rtx_3090` and `gtx_1080`, so a 3090 feature does not guarantee a 3090.
  - Fix applied 30 Sep 02:10: `scontrol update JobId=<j> Features="rtx_6000|rtx_4090"` on every PD no-agent cell except the 5-min smoke-from. Wave 3 got it at submit.
  - Batch is pinned (256) under P, so the card changes speed, not the recipe.
  - If a slot sits idle > 1 h because no 4090/6000 is free, clear it with `scontrol update JobId=<j> Features=`.
- **Priority = age + fairshare (202) − nice**, so small nice values reorder our own queue. Users can raise nice, never lower it, and cannot raise TimeLimit.

## 7. Commits of this sitting

`502ec84` (V9c code + readers + sitting §13 + queue doc + ops §8), `5b6398d` (final-FT KD teacher, next tree), `3c22f83` (aug r20-w13 read, stamps). This record, the way-ahead doc, ledger §148–§151, the queue / ops updates and ops' pending ledger / status / Gilad-note edits: the 30 Sep commit that adds this file.
