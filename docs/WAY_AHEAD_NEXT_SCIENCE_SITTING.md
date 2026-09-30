# Way ahead — for the next science sitting (written 30 Sep ~03:00 IDT, Opus 5.5 MAX; restamped 30 Sep ~11:55 at the ops handoff)

Read this first, then the §7 log at the bottom (what ops saw since the handoff), `docs/SITTING_GPU_QUEUE.md` (live queue) and the ledger rows it cites. The ops handoff, pre-authorized actions and milestones M1–M7 are `docs/OPS_HANDOFF_RUNBOOK.md` §10. The diverse train is `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. The Gilad explanation of the three 29–30 Sep findings is `docs/paper/GILAD_NEWS_30SEP.md`. What was built and run is in `docs/RUN_RECORD_29SEP_V9C.md`. The earlier option list and ladder are `docs/PROMPT_FABLE_NEXT_SITTING.md` §13. This file supersedes §13.3's statuses.

## 0. State of play

- **Protocol P is the walk and train protocol.** Val = one 5k half of the test split, TEST = the other; batch 256 pinned. Every verdict closed before 28 Sep was measured on a val the zoo nets had memorized (§141): agent ≡ mild, C100 unrecoverable, C-G dead, and the reward-shape reading.
- **Under P the walk fine-tune is the lever.**
  - Crop+flip in the walk FT is kinder at every equal-width C100 point measured (12/12, mean +3.3 pp TEST, §148).
  - On the C10 R56 twin it is −0.06 pp TEST at 0.661 keep, against −2.84 without it (§152). On VGG-16 it is +2.3 to +2.6 pp at equal keep (§152).
  - The 100-epoch SGD final FT adds an honest +4.1 to +5.5 pp on the C100 bar-3 cell (§149), and +1.2 to +1.8 (HOLD) on DepGraph's R56 C10 (§153): 10k −1.52 / −2.11 at DepGraph's FLOPs points, against their +0.24 / +0.11.
- **Stage 4 is running.** **21737123**: the area train under P + crop+flip (§151). The training rule passed at 03:11 (§150): r56-w4 +2.3 pp TEST at equal keep. The P-only arm was cancelled before it started; its line is kept for the attribution train (N9). Freeze TESTs are pre-authorized (runbook §10.3).
- **C100 is in the catalog.** The aug gate admitted 8/8 (§148); `configs/database_offline_v7_diverse_admitted.json` has 16 nets (8 C10 + 8 C100), emitted 30 Sep 11:45. Nothing trains on it until N8 (roadmap).
- **No-agent ladder.** 12 cells (3 R, 9 PD) at 11:55. 21729552 / 554 / 557 and 21730500 have finished; N3 / N1 / N2 were added, and N3 is R. The pending ones are pinned to fast cards (`Features=rtx_6000|rtx_4090`).
- **The train is slow by design, and the fuse is covered.**
  - *Pace.* Mean 2,315 s per episode over the first 12 (median 1,296).
  - *When the fuse fires.* The 6-day fuse (`runtime_limit` 518,400 s from the train's start) fires **~6 Oct 03:15, near episode ~200**, short of the 250 minimum. The 03:55 estimate of "~10 Oct, ~episode 160" was wrong.
  - *The resume.* 21767188 is chained `afterok`, so the train runs to its stopping rule (decision f).

## 1. Decisions waiting on Ido (recommendation first)

Decisions (a), (b)-emit, (c) and (f) were settled by Ido's 30 Sep 11:08 GO and done by 11:45. Their facts stay below for the record. (d) was met at 11:55; the 21730506 conversion waits for Ido's reply. Still open: (b)-train (N8) and (e).

**(a) `21716380` (group-token, held since 28 Sep) — DONE: scancelled 30 Sep 11:29.**
- *Facts.* `tree_v8b`, legacy val, 7-day cold train. A release would have re-run the prologue, deleting `train_resume.pt` and restarting cold. Its 12 episodes and the `ep0011` freeze are backed up in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` (`train_resume.pt`, `latest_best_*`, `policy_config.json`, `standardizer.pt`) and in `tree_v8b/runs/job21716380/snapshots/ep0011`. Its question (group tokens) is confounded by memorized val, like every legacy train.
- *Still true.* Re-run group tokens later as a one-change arm on top of the Stage-4 recipe, only if that train leaves mild. Do not TEST `ep0011`.

**(b) C100 catalog emit — DONE 30 Sep 11:45. Diverse (C10 + C100) train → `docs/N8_DIVERSE_TRAIN_ROADMAP.md`.**
- *The emit.* `configs/v7_c100_gate.json` now carries the aug gate 21729554: 8/8 admitted, val-selected keep 0.647–0.696, val Δ −1.18 to −9.70. The no-aug gate values sit beside it as an audit column: 21729552 admitted 6/6 of the nets it finished. `build_v5_catalog.py --emit-admitted --min-c100 8` wrote `configs/database_offline_v7_diverse_admitted.json`: 16 nets, 8 C10 + 8 C100, no SVHN, r20-w8 dropped. `--check-admitted` passes; `tests/test_v5_catalog.py` 16/16.
- *Still to build for the train.* A `DATASET_NAMES` override: the profile hard-sets `cifar-10 svhn`. A C100 probe net. Requeue safety. All are `tree_v9d` items; see roadmap §5.
- *Recommendation (unchanged).*
  - Hold N8 until the Stage-4 train's first freeze TEST shows it is not a mild clone (runbook M1; roadmap §3). Two parallel trains that both copy mild waste two of four slots for a week.
  - If speed matters more than that risk, run it in parallel after M2: one slot, ~6 days.
- *Q4 answer for Gilad* (`docs/paper/GILAD_NEWS_30SEP.md` §1). One recipe (Adam 1e-3, 12/4) admits C100 under clean val: 6/6 finished without aug, 8/8 with crop+flip. Crop+flip, the augmentation every zoo net was trained with, improves it uniformly. No per-dataset recipe.

**(c) Freeze TESTs of the Stage-4 train — PRE-AUTHORIZED 30 Sep 11:08.**
- *What ops runs.* Runbook §10.3 and line §10.5 (a): the first freeze written after PPO update 20, then at most one a day, never two in flight. Each goes on the thin pair, under P + crop+flip at 40/10, against mild in 21729557 at equal keep, with the compression-rate census for the mild-clone read.
- *Fallback.* If there is still no freeze by episode 120, ops reports.

**(d) Crop+flip as the TEST walk recipe (bar 2 for every method) — MET 30 Sep 11:55 (§152; runbook M3). Waiting on Ido: the 21730506 conversion.**
- *The evidence (§152).*
  - R56 meets the rule with margin: −0.16 vs −2.68 at size 0.80, −0.50 vs −2.70 at size 0.70, −0.06 vs −2.84 at `val_best` 0.661.
  - VGG-16 meets it: +2.6 / +2.5 / +2.3 pp at size 0.80, size 0.70 and `val_best`. That makes **2 of 3 twins**. VGG-19 C100 is still walking in 21737104; its paired read is +3.21 over 7 cuts.
  - Thin guard (21729557 COMPLETED): r56-w4 is **+5.0 pp** at size 0.80 (−2.6 vs −7.6 at 0.795), and its `val_best` is deeper and kinder (−4.5 @ 0.622 vs −10.1 @ 0.739). r20 is 1.3 / 0.7 / 1.6 pp worse, inside its 2 pp.
- *Rule.* TEST ≥ 1 pp kinder at equal keep on ≥ 2 of 3 twins, and the thin guard holds. Then every TEST walk (agent and heuristics) switches to aug, and the no-aug rows stay as an audit column.
- *Recommendation.* **Convert 21730506** (parked at nice 70) to the aug line, runbook §10.5 (c): scancel 21730506, submit the line, then set `Features`. Do it on Ido's reply, in the ops chat or here.
- *Why it matters.* A P+aug-trained agent should be TESTed under the walk recipe it was trained with.

**(e) Attribution train (only after a success).**
- If 21737123 leaves mild, one P-only train (the cancelled 21737095 line, `scripts/_tmp_s30_train_submit.sh`) separates the val fix from the augmentation. If it does not leave mild, skip it: the weaker recipe cannot do better.

**(f) Resume 21737123 past its 6-day fuse — DONE: chained 30 Sep 11:40 as 21767188.**
- *Why.* The train runs at ~2.3× the control's time per episode (runbook §9.2 "Speed"): P's batch 256 against the adaptive 384, the whole 50k split, and +18 % for aug. The fuse fires ~6 Oct 03:15, near episode ~200. It is not ~10 Oct and not ~160, as first written here. Either way it comes before the 250-episode minimum, and Ido chose to let the train run its course.
- *The job.* 21767188 runs `afterok:21737123` at nice 0, with `Features` and `Requeue=0`.
- *What it restores.* Weights, both optimisers, the episode index, the standardizer, and the governor's best score and since-improvement count. Only the rewind count resets.
- *When it stops.* At the governor's rule (episode ≥ 250 and 150 episodes without a better probe), or at its own 6-day fuse (~12 Oct). A second resume needs Ido.
- *The requeue trap.* The cluster requeues on preemption. A requeue under the same job id runs the sbatch "always cold" block, which deletes `train_resume.pt`. `Requeue=0` is now set on both jobs, and the heartbeat keeps a daily bundle backup in `~/spectra_backups/`. The permanent fix is `tree_v9d` item 9.
- *A mild clone at ~episode 200* is still reported, not killed. The runbook's M1-neg calls the sitting.

## 2. Insights gathered (ledger refs)

1. **Memorized val (§141).** Zoo nets read val 1.000 / 1.000 / 0.999 unpruned against TEST 0.943 / 0.936 / 0.739. Every legacy reward and gate scored a cut on memorization loss.
2. **No cut gains on full-width nets under the walk FT, even under P (§147: 0/343).** Published light cuts recover to ≥ 0 under SGD + crop/flip, e.g. DepGraph R56 2.11× +0.24 and Network Slimming VGG-19 +0.14. The missing "gain" is a recipe property, not a reward or val property.
3. **Crop+flip helps every capable net and hurts only the tiniest (§148, §150).**
   - The C100 gate is +0.9 to +6.2 pp TEST at equal widths.
   - The r56-w4 decider at 12/4: +2.3 pp TEST at equal keep, and in-band to the end of the walk (−5.1 @ 0.622 vs −10.6 @ 0.741).
   - The full-width C10 R56 twin at 40/10 (§152): −0.06 pp TEST at 0.661 keep, against −2.84 without aug. That is almost lossless at 1.51×, with no final FT.
   - The 5k-param r20-w2 (64.8 % accuracy) loses 1.0–3.1 pp TEST. That net underfits, and augmentation hurts underfitting nets (NetAug, Cai et al. ICLR 2022). r20-w2 is a hold-out diagnostic, not a train net.
4. **Under P, C100 admits at the live recipe (§148).** Without aug, 6/6 finished nets; with crop+flip, 8/8, with val-selected points at 0.65–0.70 keep. "C100 unrecoverable" was memorized val. The catalog is emitted.
5. **The final 100-ep SGD FT recovers +4 to +5.5 pp at fixed widths (§149).** Origin moves +0.10. SOTA-facing rows (bar 3) must carry it; same-loop rows (bar 2) stay on the walk recipe.
6. **The train FT 12/4 is ~1 pp harsher than the TEST FT 40/10 on r56-w4 (§150).** The agent trains in a harsher world than it is tested in.
7. **Re-walk noise.** Up to 0.8 pp TEST at equal widths across GPU SKUs (§149). Caption walk gaps below ~1 pp as noise.
8. **Probe area is protocol-dependent.** A P-train's area is not comparable with 0.0586 (legacy). Compare freezes by TEST only.
9. **Scheduler.** Untyped GPU requests land on the lowest-weight (slowest) nodes. `Features=rtx_6000|rtx_4090` fixed the no-agent cells without starving on one SKU (run record §6).
10. **Honest gain needs a healthy origin.** Subtracting a negative origin change inflates it: the 1-epoch smoke printed "+5.78 ADOPT" on a raw gain of −0.06. The reader now says `ORIGIN-HURT` when the origin loses > 0.5 pp. That case matters for every new final recipe (KD, AutoAugment, SWA).
11. **DepGraph R56 C10 under the final FT: HOLD, ~2 pp short (§153).**
    - Honest +1.18 / +1.80 / +1.74; origin +0.42.
    - 10k final −1.52 at FLOPs 0.463 and −2.11 at 0.380. DepGraph publishes +0.24 and +0.11 on the same checkpoint.
    - Our walk here is no-aug mild, and there are no KD, AutoAugment or scratch rows yet. N3 (aug walk), N1 (KD) and N2 (AutoAugment) measure how much of the gap closes.
    - The re-walk matches 21726340 to −0.04 over 150 cuts, so the walk itself is reproducible.
12. **VGG-16 and the thin guard confirm crop+flip at TEST FT (§152).**
    - VGG-16 is +2.3 to +2.6 pp at equal keep, and the arm is better on 100 % of 28 paired cuts.
    - On the thin r56-w4 it is +5.0 pp at 0.795. At 40/10 it recovers 3.3 pp more than at 12/4 (§150).
    - Decision (d) is met.
13. **The requeue trap (runbook §10.2).**
    - *The mechanism.* The cluster requeues on preemption (`JobRequeue=1`, `PreemptMode=REQUEUE`). A requeue keeps the job id, so the sbatch's "always cold" block runs again and deletes `train_resume.pt`. A requeued resume would also re-copy the parent bundle over its own (prologue lines 48–54).
    - *For now.* `Requeue=0` on every train.
    - *The permanent fix* is in `tree_v9d` (§4 item 9).
14. **The Stage-4 critic starts flat.** ev 0.010 / −0.129 / 0.026 at PPO updates 1–3, against 0.45–0.88 in the control. That could be the harder P reward, or just the early updates; M2 at update 10 reads it (runbook §10.4). A flag, not a kill.

## 3. Options re-ranked (status 30 Sep ~11:55)

P = projected probability that the option passes its own adopt rule. Cells: `SITTING_GPU_QUEUE.md`.

| # | Option | Status | P now | Next |
|---|---|---|---|---|
| O1 | crop+flip in the walk FT | **gate rule met (§148); training rule passed (§150); TEST-walk rule met (§152)**: R56 +2.2 to +2.8, VGG-16 +2.3 to +2.6, thin r56-w4 +5.0; r20 inside its guard | adopted for the TEST walk | the 21730506 conversion (Ido) |
| O2 | 100-ep SGD final FT + origin | **met on C100 (§149)**; **HOLD on DG R56 C10 (§153, +1.2 to +1.8)**; r20 thin cross-off so far (21730501, origin +3.5) | 0.50 ≥ 2 pp on full-width C10 | 21730501 r56-w4; 21730506 after (d) |
| O18 | P gate at the live recipe | **passed (6/6 finished; TIMEOUT at net 7)** | — | — |
| O3 | crop+flip in the C100 gate | **passed (8/8); emitted 30 Sep 11:45** | — | N8 (roadmap) |
| O17 | P-val reward train + crop+flip | **Stage 4: 21737123 → resume 21767188** | 0.40–0.50 leave mild | freeze TESTs pre-authorized (runbook §10.3) → M1 |
| O22 | scratch-B at the walk architecture (Liu et al. 2019) | PD: 21730507 after 501; 21730516 eligible (500 done) | 0.45–0.55 | — |
| N3 | aug walk + final FT, DG R56 | **submitted** 21767189 (nice 3) | 0.35 (≥ 1 pp after final FT) | pairs with 21730500 (§153) |
| N4 | aug walk + final FT, DG VGG-19 | **submitted** 21737105 | 0.35 (≥ 1 pp after final FT) | pairs with 21729551 |
| O4 | KD in the final FT (N1) | **submitted** 21767190 (nice 60), from 21730500's saves | 0.35 | ≥ +0.5 pp over plain final FT → M5 |
| O12 | AutoAugment in the final FT (N2) | **submitted** 21767192 (nice 61), same saves | 0.30 | ≥ +0.5 pp → M5 |
| O20/21 | C-G under P, NEON train-loss stop | PD 21730509 / 14 | 0.25 real cut; 0.05–0.08 ≥ A | 5-pair big-effect kill |
| O13 | N2 stream protection | PD 21729558 | 0.20 | pairs by params |
| O42 | cubic (NEON) reward under P (+aug): the item-2 A/B | new; a train | 0.25 | only if an aug census shows cuts with val Δ > 0; otherwise cubic just penalises drops harder (→ mildest) |
| O41 | diverse P train (C10 + C100) = N8 | catalog emitted; a train | 0.35 | `docs/N8_DIVERSE_TRAIN_ROADMAP.md`: after M1 + `tree_v9d` + Ido GO |
| O40 | attribution train | new; a train | — | only after O17 leaves mild |
| O39 | capacity-conditioned aug (off when the net underfits; NetAug) | new | 0.30 | only if tiny nets enter the catalog |
| O38 | reward replay of the P walks (linear / cubic / NEON-exact returns) | zero GPU, not built | 0.80 informative | next sitting, ~50 lines |
| O26 | memorization census of the train catalog | zero GPU, not built | 0.95 informative | next sitting |
| O25 | selection readouts on P runs (`traj_readout.py`) | zero GPU | 0.50 | any time |
| O7 / O8 | F2 group-first / F1 cosine at 12/4 (N5 / N6) | not queued: §150 passed | 0.15 | revisit only if aug trains badly |
| O11 | SGD 0.01 + aug gate (N7) | condition met, low value | 0.20 | not queued |
| O5 / O6 / O14 / O29 | SWA-EMA / batch 64 / per-stage rates / mixup-LS | later | 0.20–0.25 | after the train reads |
| O9 / O15 / O16 | LAMB / 0.95 rung / rollback | crossed off | — | — |

## 4. Next dev items (`tree_v9d`; build only when no `tree_v9c` job needs the change)

1. **Provenance.** Add `SPECTRA_FT_AUG` / `SPECTRA_FT_AUTOAUG` to `POLICY_INFO_KEYS` (`src/A2C_Agent_Reinforce.py`). Today a P+aug train records aug only in the log's `SPECTRA_* env` dump and its job name.
2. **Diverse train profile** (roadmap §5). Honour a pre-set `SPECTRA_DATASET_NAMES` (or add `SPECTRA_V6_DATASET_NAMES`) and point `SPECTRA_V6_DATABASE` at the admitted v7 catalog. A new profile `offline_train_v9_diverse` keeps the Stage-4 recipe otherwise unchanged.
3. ~~**Emit.**~~ Done 30 Sep 11:45: the v7 gate table was filled from the gate log and `build_v5_catalog.py --emit-admitted` reused. Generalising `emit_v5_admitted_from_gate_log.py` is no longer needed.
4. **Final-FT KD teacher** (in git `5b6398d`) goes live with `tree_v9d`. Drop the `SPECTRA_FT_KD=1` workaround from N1 then.
5. **O38 reward replay** (`scripts/reward_replay.py`). Per-step (val Δ, keep) from finished P walks through the linear in-band, cubic and NEON-exact returns; print where each return would stop.
6. **O26 memorization census** (`scripts/memorization_census.py`). Baseline val in each train log vs the TEST accuracy in the checkpoint name, per catalog net.
7. **Optional O39.** `SPECTRA_FT_AUG_MIN_TRAIN_ACC` (aug off when the unpruned net's train accuracy is below a bar). Only if an underfitting net enters a catalog.
8. **Optional: GPU-side crop+flip.** Pad and crop plus flip on the batch tensor on the GPU, not per image in the loader. It recovers up to the measured +18 % per epoch. Build it only for the next train; never swap it into a live one (the recipe must not change mid-train).
9. **Requeue safety** (insight 13). Three changes:
   - Make train profiles skip the "always cold" delete when `SLURM_RESTART_COUNT` > 0, or submit them with `--no-requeue`.
   - Stop a requeued resume from re-copying the parent bundle over its own (prologue lines 48–54).
   - Correct the stale "the governor restarts" comment near the resume block.
10. **A C100 probe net for N8** (roadmap §5 item 4). Both probe sets (`SPECTRA_PROBE_SET` v7 and thin) are C10 only. Add one admitted C100 net through `SPECTRA_PROBE_NETS` (recommended: `resnet20-width13_cifar100`), so the governor's probe sees both datasets.

## 5. First moves for the next sitting

1. **Why you were called.** Runbook §10.4 names the milestone (M1, M1-neg or M7) and the §7 log below has ops' numbers. Ledger rows ops wrote start at §154.
2. **Freeze TESTs** ops already ran (pre-authorized). Read them by TEST at equal keep against 21729557 and the census; never by probe area.
3. **Build `tree_v9d`**: roadmap §5, plus §4 items 1, 2, 4, 5, 6, 9 and 10. CPU pytest on the cluster conda, then a smoke.
4. **With Ido's GO, N8** per the roadmap (on M1). On M1-neg, see roadmap §3, "If G1 fails".
5. **Gilad.** `docs/paper/GILAD_NEWS_30SEP.md` covers the three 29–30 Sep findings; extend it with M1 when it lands.

## 6. Literature used in this cycle

He et al. 2016 (crop+flip CIFAR recipe) · Li et al. ICLR 2017 (filter pruning, FT recipe, per-stage sensitivity) · Liu et al. ICLR 2019 (rethinking pruning: scratch-B) · Le & Hua ICLR 2021 (retraining schedule matters) · Fang et al. CVPR 2023 (DepGraph; published R56 / VGG-19 rows) · PruningBench 2024 (100-ep FT protocol) · Cai et al. ICLR 2022 (NetAug: augmentation hurts tiny nets) · Hinton et al. 2015 (KD) · Cubuk et al. CVPR 2019 (AutoAugment) · Izmailov et al. 2018 (SWA) · Hirsch & Katz, Information Sciences 2022 (NEON: cubic reward, train-loss stop, patience 10).

## 7. Ops annotations (append-only, dated; ops writes here, the next sitting reads)

Format: `- <date time> | <job / event> | <number, ledger §> | <implication for the next sitting>`.

**MILESTONE M3** (30 Sep 11:55, the sitting). Decision (d) is met (§152).
- 21729557 COMPLETED: thin r56-w4 +5.0 pp at equal keep (−2.6 vs −7.6 at 0.795), with a deeper, kinder `val_best` (−4.5 @ 0.622 vs −10.1 @ 0.739). r20 is inside 2 pp.
- The twins are 2/3.
- Ido was told in the sitting chat. Converting 21730506 (runbook §10.5 c) waits for his reply.

- 30 Sep 03:11 | sitting | Stage-4 train released after §150; P-only arm cancelled | first read: FLAGS + `policy_config` diff (ops §9.2)
- 30 Sep 03:16 | 21737123 R 03:14, `ise-cpu256-32` RTX 6000 Ada | start checks green: env header; val-from-test on cifar-10 and svhn; aug on cifar-10 only; `policy_config` diff = P keys only | read its curve's shape against 21536396, never its probe-area values
- 30 Sep 03:14 | 21730499 smoke-from COMPLETED, passed | 1-epoch FT from saved; origin −5.84 pp printed "honest +5.78 ADOPT" on a raw gain of −0.06 | reader fixed (`ORIGIN-HURT`, git + `readers_s30/`). For `tree_v9d`, and for any new recipe (KD, AutoAugment), check the origin row before the verdict
- 30 Sep 03:49 | 21729553 scancelled after its R56 rows (pre-registered) | §152: R56 aug −0.06 @ 0.661 vs P −2.84; size 0.80 −0.16 vs −2.68; census val Δ > 0 on 0/62, max −0.12 | TEST-walk rule 1/3 twins. Aug brings the best cut to −0.12, near the cubic's positive branch: watch the VGG census in 21737104 for N10
- 30 Sep 03:49 | 21730500 R, `ise-4090-18` | took 553's slot | first final-FT cell on C10; read it with the fixed reader
- 30 Sep 03:55 | 21737123 pace | episode 0 993 s vs control 422 s (same 24 steps); epoch 6.7 s vs 2.9 s; aug alone +18 % (556 vs 555, same node) | ~2.3× → the 6-day fuse lands near episode ~160; decision (f). `tree_v9d` option: GPU-side crop+flip (saves up to the 18 %). **Corrected 11:45:** ~6 Oct 03:15, near episode ~200 (next line)
- 30 Sep 11:29 | 21716380 scancelled (Ido GO 11:08) | bundle kept in `spectra_pre_maint_28sep/` | decision (a) closed
- 30 Sep 11:40 | resume 21767188 chained `afterok:21737123` (nice 0, `Features`, `Requeue=0`); N3 21767189 (nice 3), N1 21767190 (nice 60), N2 21767192 (nice 61) submitted; 21730506 parked at nice 70 | decision (f) closed; 21730506 waits on (d) and Ido
- 30 Sep 11:45 | fuse re-read | mean 2,315 s per episode over 12 (median 1,296) → `runtime_limit` fires ~6 Oct 03:15 near episode ~200, not ~10 Oct / ~160 | §151; the resume covers it
- 30 Sep 11:45 | requeue trap closed for now | the cluster requeues on preemption, and a requeue under the same id deletes `train_resume.pt`. `Requeue=0` on 21737123 / 21767188; bundle backup `~/spectra_backups/job21737123_20260930` (the heartbeat keeps the last 3 daily) | `tree_v9d` item 9
- 30 Sep 11:45 | C100 catalog emitted (§148) | 16 nets, 8 C10 + 8 C100; `--check-admitted` passes; pytest 16/16 | decision (b) emit closed; N8 → roadmap
- 30 Sep 11:50 | 21730500 COMPLETED (§153) | HOLD: honest +1.18 / +1.80 / +1.74, origin +0.42; 10k −1.52 / −2.11 vs DepGraph +0.24 / +0.11 | N3 / N1 / N2 read against it
- 30 Sep 11:50 | 21737104 VGG-16 done (§152) | +2.3 to +2.6 pp at equal keep; twins 2/3 met; thin guard: r20 inside 2 pp, r56-w4 paired +4.99 over 54 cuts | ops pings "(d) met" when 557's r56-w4 TEST rows land inside the guard
- 30 Sep 11:55 | 21729557 COMPLETED (4 h 15 m); N3 21767189 R on `ise-4090-19` in its slot | §152 thin guard: r56-w4 +5.0 pp at 0.795 | MILESTONE M3 above
- 30 Sep 11:55 | handoff | ops now runs from `docs/OPS_HANDOFF_RUNBOOK.md` §10 (renamed from `PROMPT_OPS_V8_QUEUE.md`) | ops appends here; milestones M1–M7 in §10.4
