# Way ahead — for the next science sitting (written 30 Sep ~03:00 IDT, Opus 5.5 MAX; restamped 30 Sep ~13:20 after Ido's 12:34 GO)

Read this first, then the §7 log at the bottom (what ops saw since the handoff), `docs/SITTING_GPU_QUEUE.md` (live queue) and the ledger rows it cites. The ops handoff, pre-authorized actions and milestones M1–M7 are `docs/OPS_HANDOFF_RUNBOOK.md` §10. The diverse train is `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. The Gilad explanation of the three 29–30 Sep findings is `docs/paper/GILAD_NEWS_30SEP.md`. What was built and run is in `docs/RUN_RECORD_29SEP_V9C.md`. The earlier option list and ladder are `docs/PROMPT_FABLE_NEXT_SITTING.md` §13. This file supersedes §13.3's statuses.

## 0. State of play

- **Protocol P is the walk and train protocol.** Val = one 5k half of the test split, TEST = the other; batch 256 pinned. Every verdict closed before 28 Sep was measured on a val the zoo nets had memorized (§141): agent ≡ mild, C100 unrecoverable, C-G dead, and the reward-shape reading.
- **Under P the walk fine-tune is the lever.**
  - Crop+flip in the walk FT is kinder at every equal-width C100 point measured (12/12, mean +3.3 pp TEST, §148).
  - On the C10 R56 twin it is −0.06 pp TEST at 0.661 keep, against −2.84 without it (§152). On VGG-16 it is +2.3 to +2.6 pp at equal keep, and on the VGG-19 C100 twin +3.8 to +4.9: twins 3/3 (§152).
  - The 100-epoch SGD final FT adds an honest +4.1 to +5.5 pp on the C100 bar-3 cell (§149), and +1.2 to +1.8 (HOLD) on DepGraph's R56 C10 (§153): 10k −1.52 / −2.11 at DepGraph's FLOPs points, against their +0.24 / +0.11.
- **Stage 4 is running.** **21737123**: the area train under P + crop+flip (§151). The training rule passed at 03:11 (§150): r56-w4 +2.3 pp TEST at equal keep. The P-only arm was cancelled before it started; its line is kept for the attribution train (N9). Freeze TESTs are pre-authorized (runbook §10.3).
- **C100 is in the catalog.** The aug gate admitted 8/8 (§148); `configs/database_offline_v7_diverse_admitted.json` has 16 nets (8 C10 + 8 C100), emitted 30 Sep 11:45. Nothing trains on it until N8 (roadmap).
- **No-agent ladder.** 12 cells (3 R, 9 PD) at 13:20. 21729552 / 554 / 557, 21730500 and 21737104 have finished. N3 / N1 / N2 were added; N3 and N4 are R. 21730506 became 21809595, and the size-matched VGG-16 cell 21814029 was added. The pending ones are pinned to fast cards (`Features=rtx_6000|rtx_4090`).
- **The train is slow by design, and the fuse is covered.**
  - *Pace.* Mean 2,315 s per episode over the first 12 (median 1,296).
  - *When the fuse fires.* The 6-day fuse (`runtime_limit` 518,400 s from the train's start) fires **~6 Oct 03:15, near episode ~200**, short of the 250 minimum. The 03:55 estimate of "~10 Oct, ~episode 160" was wrong.
  - *The resume.* 21767188 is chained `afterok`, so the train runs to its stopping rule (decision f).

## 1. Decisions waiting on Ido (recommendation first)

Decisions (a), (b)-emit, (c) and (f) were settled by Ido's 30 Sep 11:08 GO and done by 11:45. Their facts stay below for the record. (d) was met at 11:55, and the conversion ran at 12:47 on Ido's 12:34 GO (21809595). Still open: (b)-train (N8: the catalog choice and G5, roadmap §2b and §3) and (e).

**(a) `21716380` (group-token, held since 28 Sep) — DONE: scancelled 30 Sep 11:29.**
- *Facts.* `tree_v8b`, legacy val, 7-day cold train. A release would have re-run the prologue, deleting `train_resume.pt` and restarting cold. Its 12 episodes and the `ep0011` freeze are backed up in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` (`train_resume.pt`, `latest_best_*`, `policy_config.json`, `standardizer.pt`) and in `tree_v8b/runs/job21716380/snapshots/ep0011`. Its question (group tokens) is confounded by memorized val, like every legacy train.
- *Still true.* Re-run group tokens later as a one-change arm on top of the Stage-4 recipe, only if that train leaves mild. Do not TEST `ep0011`. **Enqueued 1 Oct (Ido 08:42)**: smoke 21940318, train **21940319 held** on that gate (queue file, agent-design arms), with budget + STOP, the two-decision head and the 40/10 train FT.

**(b) C100 catalog emit — DONE 30 Sep 11:45. Diverse (C10 + C100) train → `docs/N8_DIVERSE_TRAIN_ROADMAP.md`.**
- *The emit.* `configs/v7_c100_gate.json` now carries the aug gate 21729554: 8/8 admitted, val-selected keep 0.647–0.696, val Δ −1.18 to −9.70. The no-aug gate values sit beside it as an audit column: 21729552 admitted 6/6 of the nets it finished. `build_v5_catalog.py --emit-admitted --min-c100 8` wrote `configs/database_offline_v7_diverse_admitted.json`: 16 nets, 8 C10 + 8 C100, no SVHN, r20-w8 dropped. `--check-admitted` passes; `tests/test_v5_catalog.py` 16/16.
- *Still to build for the train.* A `DATASET_NAMES` override: the profile hard-sets `cifar-10 svhn`. A C100 probe net. Requeue safety. All are `tree_v9d` items; see roadmap §5.
- *Recommendation (unchanged).*
  - Hold N8 until the Stage-4 train's first freeze TEST shows it is not a mild clone (runbook M1; roadmap §3). Two parallel trains that both copy mild waste two of four slots for a week.
  - If speed matters more than that risk, run it in parallel after M2: one slot, ~6 days.
  - *Catalog (Ido 12:34).* Keep design A: 8 C10 + 8 C100, with SVHN, Fashion-MNIST and ImageNet held out. Add the three additions of roadmap §2b: grow the SVHN / Fashion-MNIST hold-out rows with unseen families, caption H2 as the net pool change, and pre-register N8b.
  - *G5.* Pre-register it now as a conditional GO (roadmap §3). The science sitting launches N8 on M1 + a clean smoke + catalog A + a free slot. Anything marginal comes back to Ido.
- *Q4 answer for Gilad* (`docs/paper/GILAD_NEWS_30SEP.md` §1). One recipe (Adam 1e-3, 12/4) admits C100 under clean val: 6/6 finished without aug, 8/8 with crop+flip. Crop+flip, the augmentation every zoo net was trained with, improves it uniformly. No per-dataset recipe.

**(c) Freeze TESTs of the Stage-4 train — PRE-AUTHORIZED 30 Sep 11:08.**
- *What ops runs.* Runbook §10.3 and line §10.5 (a): the first freeze written after PPO update 20, then at most one a day, never two in flight. Each goes on the thin pair, under P + crop+flip at 40/10, against mild in 21729557 at equal keep, with the compression-rate census for the mild-clone read.
- *Fallback.* If there is still no freeze by episode 120, ops reports.

**(d) Crop+flip as the TEST walk recipe (bar 2 for every method) — MET 30 Sep 11:55, twins 3/3 at 12:50 (§152; runbook M3). DONE: 21730506 → 21809595 (Ido GO 12:34).**
- *The evidence (§152).*
  - R56 meets the rule with margin: −0.16 vs −2.68 at size 0.80, −0.50 vs −2.70 at size 0.70, −0.06 vs −2.84 at `val_best` 0.661.
  - VGG-16 meets it: +2.6 / +2.5 / +2.3 pp at size 0.80, size 0.70 and `val_best`. That made **2 of 3 twins** at 11:55.
  - VGG-19 C100 (21737104 COMPLETED ~12:45): +3.8 / +4.9 / +4.2 pp at size 0.80 / 0.70 / `val_best`. **3 of 3 twins.**
  - Thin guard (21729557 COMPLETED): r56-w4 is **+5.0 pp** at size 0.80 (−2.6 vs −7.6 at 0.795), and its `val_best` is deeper and kinder (−4.5 @ 0.622 vs −10.1 @ 0.739). r20 is 1.3 / 0.7 / 1.6 pp worse, inside its 2 pp.
- *Rule.* TEST ≥ 1 pp kinder at equal keep on ≥ 2 of 3 twins, and the thin guard holds. Then every TEST walk (agent and heuristics) switches to aug, and the no-aug rows stay as an audit column.
- *Done.* 21730506 was cancelled while PD at 12:47 and resubmitted as **21809595**: runbook §10.5 (c), nice 40, `Features`. Its readout is runbook §10.3 item 4.
- *The gap it did not close.* No cell had reached a published VGG-16 size: the 2-pass walks stop at 0.66 kept. **21814029** (§10.5 d) walks VGG-16 for 10 passes to HRank's 46.5 % and OCSPruner's 21.2 % FLOPs. "OCS VGG-16 ≈ 0.42 params" in older notes is OCSPruner's ResNet-56 point.
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
15. **Allocation, not selection (§188, §191, §192; design §0 item 7).** Once the per-layer widths are fixed, no mask beats L1 beyond noise after a 40-epoch recovery, not even the ablation oracle's. Magnitude is still necessary: random −1.4 / −1.5 pp, anti-L1 −79 pp on MobileNet-V2. A learned score ranks like the oracle within the ResNet family (τ 0.42 vs L1 0.29 on R56-C100) and not on MobileNet. Lesson: the agent's job is how many, and any learned score needs a family hold-out.
16. **The walk is input-bound only on small nets (D5; EFFICIENCY §3.3).** With crop+flip moved to the GPU, R56-w4 runs 3.75 s/epoch against 5.29 with the loader (1.41×), and is then GPU-bound. R20-w2 still took 4.29 s/epoch with the loader, though ten times smaller, so the gain should be larger there. D5-bis measures it.

## 3. Options re-ranked (status 30 Sep ~11:55)

P = projected probability that the option passes its own adopt rule. Cells: `SITTING_GPU_QUEUE.md`.

| # | Option | Status | P now | Next |
|---|---|---|---|---|
| O1 | crop+flip in the walk FT | **gate rule met (§148); training rule passed (§150); TEST-walk rule met (§152)**: R56 +2.2 to +2.8, VGG-16 +2.3 to +2.6, thin r56-w4 +5.0; r20 inside its guard | adopted for the TEST walk; twins 3/3 | 21809595 (converted), 21814029 (L2) |
| O2 | 100-ep SGD final FT + origin | **met on C100 (§149)**; **HOLD on DG R56 C10 (§153, +1.2 to +1.8)**; r20 thin cross-off so far (21730501, origin +3.5) | 0.50 ≥ 2 pp on full-width C10 | 21730501 r56-w4; 21809595; 21814029 |
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
| O38 | reward replay of the P walks (linear / cubic / NEON-exact returns) | zero GPU, **built 1 Oct** (`scripts/reward_replay.py`) | table in `docs/SITTING_GPU_QUEUE.md` | done |
| O26 | memorization census of the train catalog | zero GPU, **built 1 Oct** (`scripts/memorization_census.py`) | ledger §169: legacy 24 / 24 memorized, P 0 / 10 | done |
| O25 | selection readouts on P runs (`traj_readout.py`) | zero GPU | 0.50 | any time |
| O7 / O8 | F2 group-first / F1 cosine at 12/4 (N5 / N6) | not queued: §150 passed | 0.15 | revisit only if aug trains badly |
| O11 | SGD 0.01 + aug gate (N7) | condition met, low value | 0.20 | not queued |
| O5 / O6 / O14 / O29 | SWA-EMA / batch 64 / per-stage rates / mixup-LS | later | 0.20–0.25 | after the train reads |
| O9 / O15 / O16 | LAMB / 0.95 rung / rollback | crossed off | — | — |

## 4. Next dev items (`tree_v9d`; build only when no `tree_v9c` job needs the change)

*Status 3 Oct 00:50 (2 Oct sitting, `docs/RUN_RECORD_02OCT_SITTING.md` §1):*
- **Built, `tree_v9d`:** items 1, 2, 4, 5, 6, 8, 9, 10, plus the new items 11 and 12 below.
- **Not built:** item 7 (optional).

1. **Provenance.** *Done 1 Oct; extended 2 Oct.* Add `SPECTRA_FT_AUG` / `SPECTRA_FT_AUTOAUG` to `POLICY_INFO_KEYS` (`src/A2C_Agent_Reinforce.py`). 2 Oct adds `SPECTRA_FT_AUG_HOLDOUT` and `SPECTRA_FT_AUG_GPU`. These "info" keys are written to `policy_config.json` and never re-applied on replay.
2. **Diverse train profile** (roadmap §5). *Done 1 Oct:* `offline_train_v9_diverse`; G2 smoke 21938898 passed. Honour a pre-set `SPECTRA_DATASET_NAMES` (or add `SPECTRA_V6_DATASET_NAMES`) and point `SPECTRA_V6_DATABASE` at the admitted v7 catalog. A new profile `offline_train_v9_diverse` keeps the Stage-4 recipe otherwise unchanged.
3. ~~**Emit.**~~ Done 30 Sep 11:45: the v7 gate table was filled from the gate log and `build_v5_catalog.py --emit-admitted` reused. Generalising `emit_v5_admitted_from_gate_log.py` is no longer needed.
4. **Final-FT KD teacher** (in git `5b6398d`) goes live with `tree_v9d`. *Done 1 Oct* (`3b72ea3`). Drop the `SPECTRA_FT_KD=1` workaround from N1 then.
5. **O38 reward replay** (`scripts/reward_replay.py`). *Done 1 Oct 03:10* (queue file "O38"). Per-step (val Δ, keep) from finished P walks through the linear in-band, cubic and NEON-exact returns; print where each return would stop.
6. **O26 memorization census** (`scripts/memorization_census.py`). *Done 1 Oct* (ledger §169). Baseline val in each train log vs the TEST accuracy in the checkpoint name, per catalog net.
7. **Optional O39.** `SPECTRA_FT_AUG_MIN_TRAIN_ACC` (aug off when the unpruned net's train accuracy is below a bar). Only if an underfitting net enters a catalog.
8. **Optional: GPU-side crop+flip.** *Built 2 Oct:* `SPECTRA_FT_AUG_GPU=1`, default off, `tests/test_gpu_ft_aug.py` 9/9. The D5 A/B 21982372 / 73 measured **1.41×** on R56-w4 (5.29 vs 3.75 s/epoch); ADOPT-PENDING on D5-bis 21990184 (3 Oct sitting). Pad and crop plus flip on the batch tensor on the GPU, not per image in the loader. It recovers up to the measured +18 % per epoch. Build it only for the next train; never swap it into a live one (the recipe must not change mid-train).
9. **Requeue safety** (insight 13). *Done 1 Oct:* trains submit with `--no-requeue` (`submit.sh`); the sbatch comment is corrected. Three changes:
   - Make train profiles skip the "always cold" delete when `SLURM_RESTART_COUNT` > 0, or submit them with `--no-requeue`.
   - Stop a requeued resume from re-copying the parent bundle over its own (prologue lines 48–54).
   - Correct the stale "the governor restarts" comment near the resume block.
10. **A C100 probe net for N8** (roadmap §5 item 4). *Done 1 Oct:* `offline_train_v9_diverse` probes `resnet56-width6,resnet20-width10,resnet20-width13_cifar100`. Both probe sets (`SPECTRA_PROBE_SET` v7 and thin) are C10 only. Add one admitted C100 net through `SPECTRA_PROBE_NETS` (recommended: `resnet20-width13_cifar100`), so the governor's probe sees both datasets.
11. **Decision timer.** *Built 2 Oct.*
    - `SPECTRA_TIME_DECIDE=1` (default off) wraps each eval-walk decision in a `step.decide` stage. For the actor that is its forward and pick; for a heuristic, its pick. Counterfactual probes stay outside.
    - `scripts/cost_readout.py` prints `decide … ms` and the count of decisions.
    - Tests: `tests/test_decide_timer.py` 8/8.
    - It replaces EFFICIENCY §3.4's between-steps upper bound once the next frozen-actor TEST sets the flag.
12. **Hold-out FT augmentation.** *Built 2 Oct.* `SPECTRA_FT_AUG_HOLDOUT=1` (default off) gives the SVHN train split RandomCrop and Fashion-MNIST RandomCrop + Flip, the recipes their checkpoints were trained with. Tests: `tests/test_holdout_ft_aug.py` 6/6. H0 uses it.

## 5. First moves for the next sitting

1. **Why you were called.** Read `docs/RUN_RECORD_02OCT_SITTING.md` (§1–§7: the 2 Oct sitting; §8: the short 3 Oct morning sitting). Then read ops' §7 lines dated after 3 Oct 11:45. The ledger's next section is **§193**.
2. **S2: closed (3 Oct sitting).** G2 HARM. The paper-facing negative is written (design §0 item 7, §6.5, §8): keep L1; S3 closed. S1b (the mask datamodel) reopens only if pf-w makes a BN-only proxy valid in the loop. Any later learned score needs a family hold-out in its fit (S1's ranking did not transfer to MobileNet).
3. **pf-w calls** (ops runs `--sets where` at 4/4, or at 21970089's wall). A ceiling below 0.5 stops the pf line. If 40x10 is valid and 12x4 is not, releasing 21940321 is Ido's call.
4. **H0 rows** are the hold-out bars N8's H5 / H7 read against. Ops ledgers them on COMPLETED.
5. **D5: ADOPT-PENDING (3 Oct sitting).** D5-bis **21990184** settles it at five TEST points against 21729557. On EQUIVALENT, new cells may set `SPECTRA_FT_AUG_GPU=1`; never a live train, a resume or a freeze TEST. Then re-time 12/4 vs 40/10 with it, and expect more than 1.41× on loader-bound nets.
6. **Freeze TESTs** are ops'. Stage-4 ep0095 is **21990060**: read it by TEST at equal keep against 21729557 and the census, never by probe area. **RW43 21990185** (the control re-walked with seed 43) says how much of M1's 0.5 pp margin is noise. The arms follow runbook §10.0c: no TEST of a pre-update-20 freeze; ARM-FLAT at episode 120; ARM-NEG at the stop.
7. **Still Ido's GO:** N8 / N9 / G5, S3, releasing 21940319 / 21940321, nvidia-ml-py / PUE, `SPECTRA_draft.md`. `tree_v9d` only; never patch `tree_v9b` / `tree_v9c`.
8. **Gilad 8 Oct.** Tracker B3 / B7, slides 3–4 and question 3 carry the selection negative. Add M1 when 21990060 lands.
9. **After D5-bis and RW43 the GPU ladder is empty.** The next cells depend on M1 (21990060) and on the pf-w calls. Register them in that sitting; ops pings if a slot idles.

## 6. Literature used in this cycle

He et al. 2016 (crop+flip CIFAR recipe) · Li et al. ICLR 2017 (filter pruning, FT recipe, per-stage sensitivity) · Liu et al. ICLR 2019 (rethinking pruning: scratch-B) · Le & Hua ICLR 2021 (retraining schedule matters) · Fang et al. CVPR 2023 (DepGraph; published R56 / VGG-19 rows) · PruningBench 2024 (100-ep FT protocol) · Cai et al. ICLR 2022 (NetAug: augmentation hurts tiny nets) · Hinton et al. 2015 (KD) · Cubuk et al. CVPR 2019 (AutoAugment) · Izmailov et al. 2018 (SWA) · Hirsch & Katz, Information Sciences 2022 (NEON: cubic reward, train-loss stop, patience 10).

## 7. Ops annotations (append-only, dated; ops writes here, the next sitting reads)

Format: `- <date time> | <job / event> | <number, ledger §> | <implication for the next sitting>`.

- 4 Oct 20:45 | **v10 fixed-target train** (sitting, Ido GO 19:23; gate A0 HEADROOM 6/6) | Code in `tree_v10`. Smokes 22155996 / 97. Train **22156018** HELD with resume 22156019. Mild-landed controls **22156061 / 62** (κ 0.8 / 0.6) R | The next sitting reads M1-v10 by the queue file's "v10" rule: freezes after update 20 only; actor vs mild-landed at κ 0.6 on r56-w4, Δ ≥ +0.5 pp = WIN. A0b's consequences decide whether r20-w2 and κ 0.8 also gate. Probe scores are never results. The walk's done rule was changed to "kept ≤ κ" after the smoke; the train runs the new rule.

**M1-neg mechanism** (4 Oct ~12:45, sitting; ledger **§200**, report `docs/paper/GILAD_1OCT_POINTS_REPORT.md` Part III).
- *Census:* every TESTed actor plays one action at every free decision. Stage-4 (×2 seeds), C2 and ep0131 play 0.8: 16/16 on r20, 60/60 on r56. Budget plays its largest budget and never STOPs.
- *Cause:* the trains' reward (`structural`, τ 10, cumulative vs origin) pays +ρ inside the band whatever the accuracy. With a fixed number of decisions, "always the largest cut" maximizes the return. The val replay pays R56-w4's 0.8 walk 270.5 vs mild's 126.6 (equal depth: 124.9 vs 126.0, across 3 pp of val).
- *Implication:* M1-neg = uniform 0.8 vs uniform 0.9. FR43's stability is trivial. A menu change alone cannot help. N10 is covered by C1 / C2.
- *Next:* A0 (allocation headroom, 22127527–29), then a reward that passes the replay check. F1 `structural_unified` with `SPECTRA_TRAIN_TAU=5` is the first shape that pays mild's walks more on both nets. A train needs Ido's GO.

**MILESTONE M1-neg** (4 Oct 06:16, ops). Stage-4 freeze TEST **21990060** (§193) plus C2 **22056144** (§198) and Budget **22059501** (§197) are each > 0.5 pp worse than mild on both thin nets at the first equal-keep cut. FR43 **22059502** (§199) COMPLETED 06:20: widths match 17/17 and 61/61; **R56 kinder band replicates**; r20 first-point deficit does not. M1 on ep0095 stays 21990060’s. Do **not** start N8.

**M1 does not fire** (4 Oct 01:31 / 06:16, ops; sitting equal-keep addendum ~02:15). Stage-4 **21990060** vs 21729557: first size points worse by > 0.5 pp on both nets. At equal keep vs the three-walk mild mean, r56 is kinder at 16 of 19 shared keeps. Closed as **M1-neg** above.

**MILESTONE M8** (2 Oct 01:04, ops). S0 `21945105/06/07` all COMPLETED. Keep 0.6 lever ≥ threshold on **3 of 3** cells (r56 budget 0; vgg16 budgets 0 and 1; vgg19 budgets 0, 1 **and 40**). Not M8-neg. Not only ≤ 3. Ledger probe **§188**. S1 is sitting, zero GPU. Do **not** start S1–S3 from ops.

**MILESTONE M6** (weak, 30 Sep 19:15, ops). N4 **21737105** census: 2/46 full-width cuts with val Δ > 0 (max +0.28). Letter of M6; N10 still a sitting design, not a launch.

**MILESTONE M3** (30 Sep 11:55, the sitting). Decision (d) is met (§152).
- 21729557 COMPLETED: thin r56-w4 +5.0 pp at equal keep (−2.6 vs −7.6 at 0.795), with a deeper, kinder `val_best` (−4.5 @ 0.622 vs −10.1 @ 0.739). r20 is inside 2 pp.
- The twins were 2/3 at 11:55 and 3/3 at 12:50 (VGG-19 C100 +3.8 to +4.9).
- Ido GO'd the conversion at 12:34. It ran at 12:47: 21730506 → 21809595.

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
- 30 Sep 12:44 | 21737123 first freeze `ep0011` (probe 0.282, after PPO update 3) | before update 20: not a TEST (runbook §10.3 item 1) | the pre-authorized TEST waits for the first freeze after update 20
- 30 Sep 12:45 | 21737104 COMPLETED (2 h 48 m, 0 Tracebacks); N4 21737105 R on `cs-4090-07` in its slot | §152: VGG-19 C100 twin +3.8 to +4.9 pp at equal keep → twins 3/3 | (d) fully met
- 30 Sep 12:47 | 21730506 cancelled while PD → **21809595** (Ido GO 12:34), nice 40 | runbook §10.5 (c) | read its re-walk ≈ 0 against 21737104 / 21729553 before any final_ft row
- 30 Sep 13:10 | **21814029** L2 submitted: VGG-16 C10, 10 passes, `flop:0.465,0.212`, nice 42 | "OCS VGG-16 ≈ 0.42 params" was OCSPruner's ResNet-56 point. The published VGG-16 sizes are HRank 46.5 % FLOPs / 17.1 % params and OCSPruner 21.2 % / 13.7 % | the frozen agent's Catalog L TEST needs the same size points (roadmap §5 item 9)
- 30 Sep 13:20 | ops chat absorbs runbook §10 + Ido addendum | 4 R: train 21737123 (13 eps, freeze ep0011 only — not a TEST); N3 21767189; N4 21737105; 501 still R with r56 `final_ft` lines appearing | conversion 21809595 / L2 21814029 PD; SSH handshake dropped once, retry OK; `root_20` until 18:00
- 30 Sep 13:20 | Ido catalog / G2 / G5 answers in ops chat | keep design A; G2 = first of M1 freeze-TEST submit or ladder drain; G5 = Ido GO to launch N8 (conditional, science sitting) | ops never edits catalog, never launches N8, never TESTs ep0011
- 30 Sep 19:12 | **KILL** C-G 21730509 / 21730514 scancelled (§156) | twins mean −30.7 / −11.6 pp; thin −27.8 / −54.0 vs mild | do not resubmit C-G; slots → 21809595 / 21814029 R on 4090s
- 30 Sep 19:15 | drain over: `root_20` gone; idle includes `ise-6000-[01-03,06-07]`, `ise-6000p-02`, `ise-4090-[11,13]` | QOS still 4; do not scancel live jobs to migrate SKUs | keep `Features=rtx_6000|rtx_4090` on **new** TESTs; login handshake still drops
- 30 Sep 19:15 | 21730501 COMPLETED §154; 21737105 COMPLETED §155 | r56-w4 long FT ADOPT; r20 CROSS-OFF; N4 bar-3 CROSS-OFF (walk already recovered) | N3 still the DepGraph R56 walk+FT question
- 30 Sep 19:11 | train 21737123 | 21 eps, PPO update 5, ev 0.191, freeze still ep0011, `gap_to_uniform=+0.25` on ep20 | M2 still at update 10; do not TEST ep0011
- 1 Oct 00:09 | **QOS `gpu-part` `gres/gpu=8`** (`DenyOnLimit`) | was 4 at 30 Sep 19:28; 3 R + 1 PD resume; 5 idle | do not invent cells; G2 open
- 1 Oct 00:15 | N3 21767189 COMPLETED §157 | 10k **−0.46 @ FLOPs 0.463** vs DepGraph +0.24 → **M4**; census 70/152 val Δ>0 | bar-3 R56 = crop+flip walk; long FT CROSS-OFF
- 1 Oct 00:15 | scratch 507/516, N1/N2, streams 558 COMPLETED | §158–§162 | thin scratch fails; DG scratch 10k −0.16; KD/AA not M5; streams split
- 1 Oct 00:20 | **G2 OPEN** | ladder drained | ping Ido; sitting builds `tree_v9d` + hold-out ckpts; ops does not start Fable or N8
- 1 Oct 01:00 | Ido **starts G2 sitting tonight** | prompt `docs/PROMPT_FABLE_G2_SITTING.md` | fill QOS 8; O38 + hold-out ckpts + P+aug heuristic/FT/C-G+ A/Bs; one-change cubic/NEON-raw trains from tree_v9d; **no N8**; ops 08:15 restamp GILAD_NEWS
- 1 Oct 01:45 | **KILL** C-G+ 21938280 scancelled (§163) | r20 5 pairs mean **−20.82 pp** vs 21729557 (0 % better; last −21.44 vs +2.36); r56 never started | close C-G+ under P+aug; do not resubmit; slot → PD hold-out / G2 tails
- 1 Oct 01:46 | twins+FT **21809595** COMPLETED §164 | walk ≈ §152; 100-ep **CROSS-OFF** | bar-3 zoo C10 = crop+flip walk
- 1 Oct 02:00 | VGG-16 10-pass **21814029** COMPLETED §165 | 10k **−0.25 @ FLOPs 0.464**, **−2.02 @ 0.211**; params 0.444 / 0.187 vs HRank 0.171 / OCS 0.137 | never "beats"; pending in Gilad table 2.5 filled
- 1 Oct 02:34 | Adam 1e-4 / SGD 0.01 thin COMPLETED §166–§167 | 1e-4 kills r20 (−5.3 / −14.5); SGD misses r56 by 0.7–2.2 pp | train FT stays Adam 1e-3 12/4
- 1 Oct 03:04 | Fashion-MNIST hold-outs **21938296** COMPLETED | 4/4 ckpts, acc 94.8–95.3 | `input_g2_holdout_fmnist.json`; SVHN **21938295** still R
- 1 Oct 03:04 | C1 cubic-gain train **21938807** R | FLAGS `scale=cbrt_miss` (gain stays +ρ³); P+aug, p5b2 | report never scancel; C2 neon-raw still PD
- 1 Oct 03:24 | SVHN hold-outs **21938295** COMPLETED | 4/4, acc 96.7–97.0 | both hold-out sets ready; never in a train catalog
- 1 Oct 03:26 | SGD 0.01 C100 t2 COMPLETED §170 | harsher on 3/4 equal-keep vs Adam 1e-3 | SGD CROSS-OFF as train FT on C10 and C100
- 1 Oct 03:44 | **3h briefing** | QOS 8: train ep 30 PPO-7; C1+C2 R ep0 identical; smoke 21938898 R | M2 still update 10; do not TEST ep0011
- 1 Oct 03:55 | F1 cosine **21938286** COMPLETED §171 | r56 +3.3 / +1.0 vs plateau; r20 0.60 **−4.2** | cosine not Stage-4; F2 still R
- 1 Oct 04:24 | F2 group-first **21938287** COMPLETED §172 | r56 +2.3 / +0.9; r20 0.60 **−2.0** | A5 both CROSS-OFF as train FT
- 1 Oct 04:36 | Stage-4 **PPO update 8** | ev **0.735**, batch_score 0.527, freeze still ep0011 | M2 at update 10; do not TEST
- 1 Oct 04:44 | v9 diverse smoke **21938898** COMPLETED | 16 nets, 2 ep, 0 TB | G2 smoke met; **do not launch N8** (G5)
- 1 Oct 05:25 | greedy **21938279** COMPLETED §173 | keeps unmatched; r56 −5.7 @ 0.743 vs mild −2.6 @ 0.795 | keep mild as bar-2
- 1 Oct 05:31 | C1 PPO update 1 | ev **0.000**, clipfrac 0, gap_to_uniform still 0 | report, never scancel (kill is ev≤0 by **update 10**)
- 1 Oct 06:05–06:11 | random r56 s42 + r20 s44 COMPLETED | r20 3-draw mean §174 ≈ mild | r56 mean waits on s43/s44; QOS 5 R, ops does not fill
- 1 Oct 06:44 | **3h briefing** | PPO-8 ev 0.735; freeze ep0011; C1 ev 0 at update 1; 3 idle GPUs | Gilad 08:45; ops 08:15 GILAD_NEWS; do not invent; do not N8
- 1 Oct 08:11 | random r56 s44 **21938929** COMPLETED | 3-draw mean §175 **−5.1 / −7.2 / −6.5** vs mild **−2.6 / none / −4.5** | keep mild; QOS 3 R / 5 idle
- 1 Oct 08:15 | **Gilad pack** `GILAD_NEWS_30SEP.md` | PPO-9 ev 0.774; C1/C2 PPO-2; G2 sitting closed except the two reward trains | meeting 08:45; do not N8; do not invent
- 1 Oct 09:20 | **LR KILL** C-G thin + producers-only thin; C-PCA r20 COMPLETED | §176–§178; crop+flip does not close replacement on the diagnostic pair | full-width LR still PD; do not release held trains
- 1 Oct 10:50 | **M2 pass** Stage-4 PPO-10 | last-3 ev 0.735/0.774/0.916; last-8 gap_to_uniform min +0.078 | not a kill; freeze still ep0011; do not TEST until after update 20
- 1 Oct 13:50 | **LR KILL** C-G+ r56-w4 **21940182** | 10 pairs mean −11.6 pp vs mild §180 | r20 already §163; full-width still PD
- 1 Oct 15:50 | **3h briefing** | PPO-12; C1/C2 PPO-4 ev+; S0 still PD behind 5 pf jobs | 8 Oct board; do not TEST ep0011; do not S1–S3
- 1 Oct 18:22 | **LR KILL** C-G full-width R56 **21940183** | 8 pairs mean −34.4 pp vs twin §181 | C-G construction **CROSS-OFF** 3/4; VGG-16 21940184 still PD
- 1 Oct 18:52 | **3h briefing** | S0 `21945105` R flags ok; pf 3/6; C-G VGG-16 4 pairs −9.1 CONTINUE | never TEST S0; kill VGG-16 at ≥5 pairs
- 1 Oct 19:22 | **LR KILL** C-G VGG-16 **21940184** | 14 pairs mean −6.9 pp vs twin §182 | C-G construction **4/4**; slot should free for sel-vgg16
- 1 Oct 21:22 | **LR KILL** producers-only R56 **21940186** | 20 pairs mean −4.3 pp vs twin §183 | construction **CROSS-OFF** 3/4; C2 froze ep0023 — not a TEST
- 1 Oct 21:52 | **3h briefing** | all 3 S0 R flags ok; pf 5/6; producers VGG-16 just started | never TEST S0 or C2 ep0023
- 1 Oct 22:22 | **LR KILL** producers-only VGG-16 **21940187** | 5 pairs mean −7.9 pp vs twin §184 | construction **4/4**; slot should free for C-PCA or bench
- 1 Oct 22:52 | **S0 2/3 COMPLETED** `21945106` vgg16 + `21945105` r56 | Kendall/[lever] in design §8; never TEST | M8 waits for vgg19 `21945107`
- 1 Oct 23:53 | **LR KILL** C-G+ R56 `21940191` | 9 pairs mean −7.02 pp vs twin §185 | construction **CROSS-OFF** 3/4; leave `21940192`; do not resubmit 21940191
- 1 Oct 23:53 | C-PCA VGG-16 `21940189` COMPLETED §186 | TEST −2.8 / −1.5 / −1.5 vs mild −0.1 / −0.7 / −0.4 | construction **CROSS-OFF** 3/4; leave `21940188`
- 1 Oct 23:53 | bench `21942378` started | `ise-4090-18`, TB=0 | never ledger; readout on COMPLETED → EFFICIENCY §5.3
- 2 Oct 00:54 | **LR KILL** C-G+ VGG-16 `21940192` | 14 pairs mean −4.39 pp vs twin §187 | construction **4/4**; do not resubmit; slot → h2h
- 2 Oct 00:54 | **3h briefing** | PPO-14 ev 0.863 freeze ep0011; C-G+ 4/4; S0 34/36; pf 5/6 | never TEST S0 or ep0011; M8 after vgg19
- 2 Oct 01:04 | **MILESTONE M8** S0 vgg19 `21945107` COMPLETED | 3/3 cells; vgg19 also budget 40; ledger **§188** | S1 sitting, zero GPU; do not start S1–S3
- 2 Oct 01:08 | bench `21942378` COMPLETED | 1.25 h, TB=0; EFFICIENCY §5.3 | never ledger
- 2 Oct 01:04 | budgetstop `21940311` + factored `21940316` **started** | FLAGS ok, `Requeue=0`; budgetstop PPO-3 ev −0.822 | report, never scancel; freeze after update 20 only
- 2 Oct 00:54 | h2h `21943448` started | Torch-Pruning v1.6.1, TB=0 | never ledger, never “beats”
- 2 Oct 01:29 | pf **6/6 COMPLETED** `21941348` | ceiling **+0.41 < 0.5** uninformative §189 | **21940321 stays held**; next sitting: widen cuts
- 2 Oct 01:40 | budgetstop froze **ep0011** score 0.060 | PPO-4 ev 0.132 | **not a TEST**; never scancel
- 2 Oct 03:54 | **3h briefing** | M8 §188; pf 6/6 ceiling dead §189; SSH down since 02:23 | last confirmed 7/8; never TEST ep0011; 21940321 held; do not invent
- 2 Oct 06:54 | **3h briefing** | SSH still down (~4.5 h); no new numbers since 01:53 | VPN is the login fix; do not invent; canvas 09:30
- 2 Oct 02:01 | C-PCA zoo R56 `21940188` COMPLETED | TEST −2.1 / −2.4 / −2.7 vs mild §190 | construction **4/4**; do not resubmit
- 2 Oct 03:11 | h2h `21943448` COMPLETED | R56 85 min / 93.80; VGG-19 45 min / 70.78 | never ledger, never “beats”; EFFICIENCY §4.5
- 2 Oct 08:10 | VPN back; QOS **5/8** (3 idle since ~03:11) | Stage-4 PPO-16 freeze ep0011; C1 PPO-9; C2 PPO-9; budgetstop PPO-10 ev −0.05; factored PPO-3 | **trigger sitting now**; do not invent from ops; 21940321 held
- 2 Oct 09:23 | sitting filled QOS: pf-w `21970086/87/88` R keep 0.6; `21970089` PD keep 0.36 | start flags ok (proxy=0.6, WHERE_ROWS=8, FT_AUG=1, VAL_FROM_TEST=1) | never TRAJ rows; readout at 4/4; 21940321 held
- 2 Oct 09:31 | **S1 G1 3/3** `tree_v9d/runs/selection_scorer_s1/s1_result.json` | held-out τ 0.572 / 0.657 / 0.643 vs hand 0.240 / 0.417 / 0.170 | never TEST; sitting pastes design §8 + tracker B7; S2 afterok first pf-w, not from ops
- 2 Oct 19:08 | Stage-4 **PPO-20 / ep 80** freeze still ep0011; QOS 8/8; pf-w ~9.75 h TB=0 | **not a TEST**; wait new freeze after update 20 (or ep 120); evening prompt `docs/PROMPT_FABLE_OCT2_EVENING.md`
- 2 Oct 21:39 | pf-w `21970086` (15 cand) + `21970088` (15 cand) COMPLETED; `21970087` already 19:36 | **3/4**; do not readout; slots → S2 r56c100 `21982335` + pf-w `21970089`
- 2 Oct 22:11 | **3h briefing** | S1 pasted §191; S2 both R; pf-w 89 R 32 min; Stage-4 PPO-21 freeze ep0011 | never TEST ep0011; readout at 4/4; S2 G2 is H_40 not Kendall; canvas 23:00
- 2 Oct 22:51 | S2 mbv2 `21982334` COMPLETED exit 0 | Kendall nap_f vs ablation 0.25 < L1 0.29; [lever] budget 40 best=ablation +0.23 (σ 0.72) — G2 after both cells; never TEST
- 2 Oct 22:51–52 | H0 `21982353` SVHN + `21982354` FMNIST **FAILED** 37s/51s | **database**, not input JSON: profile default `database_c10_thin.json` (three C10 nets) filtered to zero under `--datasets` svhn/fmnist; sitting resubmitted `21986700/01` with `SPECTRA_DATABASE` = each job's input file
- 2 Oct 22:53 | D5-off `21982372` started (slot after H0 fail); D5-on `21982373` PD | SPECTRA_FT_AUG_GPU=0; FT_AUG=1 VAL_FROM_TEST=1 size_match 0.6
- 2 Oct 23:11 | **23:00 canvas** | QOS 8/8; Budget+STOP PPO-20 freeze still ep0023 ev −0.07 rewind; pf-w 89 step ~47 | do not TEST ep0011/ep0023; readout at 4/4
- 3 Oct 00:40 | D5-off `21982372` COMPLETED TB=0 | TRAJ val_best −2.48 pp @ keep 0.757; size_match NONE (did not hit 0.6); pair waits on D5-on `21982373` started 00:41 FT_AUG_GPU=1
- 3 Oct 01:11 | **3h briefing** | Stage-4 PPO-22 ep 88 freeze ep0011; S2 r56c100 ablation s2/5; pf-w 89 step 116; H0 retries PD | never TEST ep0011; G2 after 21982335; readout at 4/4; canvas 09:30
- 3 Oct 01:57 | D5-on `21982373` COMPLETED TB=0 | TRAJ val_best −3.18 pp @ keep 0.757 size_match NONE vs off −2.48; GPU aug did not help; never TEST
- 3 Oct 02:09 | S2 r56c100 `21982335` COMPLETED exit 0 | Kendall nap_f vs ablation 0.42 > L1 0.29; [lever] budget 40 best=ablation **−0.68** vs L1 (σ 0.86) — H_40 not met; G2 after sitting paste; never TEST
- 3 Oct 02:12 | H0 retries `21986700` SVHN + `21986701` FMNIST **R** | densenet hold-outs, haug=crop / crop+flip, P val; first pair FAILED stays dead
- 3 Oct 04:12 | **3h briefing** | S2 2/2 G2 H_40 not met; D5 pair in; Stage-4 PPO-23 **ep 93** freeze ep0011; pf-w 89 step 214; H0 retries ~2 h | never TEST; no S3; readout at 4/4; ep-120 fallback ~27 ep; canvas 09:30
- 3 Oct 07:12 | **3h briefing** | SSH down since 05:12 (~2 h); last poll 04:42 QOS 8/8 pf-w 89 step 231 | VPN is the login fix; do not invent; never TEST ep0011; canvas 09:30
- 3 Oct 09:42 | **09:30 canvas** (SSH still down) | last confirmed 04:42; sleeper re-armed | do not invent; canvas 16:00
- 3 Oct 09:54 | VPN back. S2 **G2 HARM** `21982334/35` | *H_40* MBV2 +0.21 / R56-C100 **−0.87** (σ 0.86); cheap-FT both cells: none | **§192**; no S3; never TEST
- 3 Oct 09:54 | D5 **1.41×** `21982372/73` both 4090 | TEST −2.8 vs −2.7 @ keep 0.757; size_match NONE (min_param 0.70) | no ADOPT/NO-GAIN/DIVERGE; EFFICIENCY §3.3; never ledger
- 3 Oct 10:00 | Stage-4 freeze **ep0095** score 0.286; TEST **21990060** PD | first snapshot after PPO-20; `tree_v9c` no timer; vs 21729557 | do not TEST ep0011; Budget+STOP still freeze ep0023 at PPO-27 / ep 107
- 3 Oct 10:58 | PC-off prep | QOS 8/8 R; PD fill (no laptop): freeze TEST **21990060** (nice 0) → D5-bis **21990184** (nice 30, 4090) → RW43 **21990185** (nice 31). All three `Requeue=0`. Train resumes afterok. Held 21940319/21 stay held. After RW43 the ladder is empty — ping, do not invent
- 3 Oct 11:30 | pf-w `21970089` COMPLETED (13.9 h, 9 cand) | readout `--sets where` **§195**: 2/6 ranked sets, ceiling +0.71; bn/none/12x4/40x10 all not valid; 12x4 NOT validated; 40x10−12x4 **−0.07** | **21940321 stays held**; next = SGD-proxy sitting
- 3 Oct 15:40 | Stage-4 freeze TEST `21990060` COMPLETED (`cs-4090-01`, 4.2 h) | **§193** vs mild 21729557: r56 `val_best` **−7.1 @ 0.389** vs **−4.5 @ 0.622** | **M1 does not fire**; not a mild clone; M1-neg waits on a second freeze TEST
- 3 Oct 18:27–19:30 | H0 `21986700` SVHN + `21986701` FMNIST COMPLETED | **§194** TESTs (P half); origin inside 0.2 pp; size 0.8/0.6 not printed | never in a training catalog
- 3 Oct 18:28 | D5-bis `21990184` COMPLETED (`cs-4090-01`, GPU banner, 2-pass) | r20 terminal |ΔTEST| 1.1 pp vs 21729557 = UNCLEAR until RW43 | s/epoch r20 1.47 / r56 3.73 vs control 4.29 / 5.24
- 3 Oct 20:12 | C2 froze **ep0083** score 0.2893 (after PPO-20) | first arm freeze eligible for TEST | wait until no freeze TEST in flight, then `TIME_DECIDE=1`
- 3 Oct 22:38 | RW43 `21990185` COMPLETED (`cs-4090-08`, seed 43) | largest |ΔTEST| vs s42 **1.2 pp** (r20 size 0.60); r20 terminal also 1.1 | D5-bis UNCLEAR → **EQUIVALENT (re-walk noise) ⇒ ADOPT D5 for new cells**; M1 0.5 pp margin is inside noise (do not change the bar)
- 4 Oct 01:38 | PC-off catch-up; C2 freeze TEST **22056144 R** `ise-4090-21` | `tree_v9d`, ep0083, `TIME_DECIDE=1`, `Requeue=0`, Features 6000\|4090, P+aug, size 0.8/0.6 | at most one freeze TEST in flight; do **not** TEST Budget ep0131 until 22056144 ends
- 4 Oct 01:38 | afterok audit | 21767188 / 21938809 / 21938811 / 21940314 / 21940317 all PD `afterok` of live trains, `Requeue=0`; held 21940319/21 + their r1s | nothing waits on a laptop GO; fuse ~6 Oct 03:15 still covered
- 4 Oct 01:38 | QOS **6/8** (2 idle) | 5 trains + C2 TEST; ladder empty | **ping**; do not invent; do not N8/S3; do not release held trains
- 4 Oct 01:38 | ARM notes | Budget **did** freeze ep0131 after PPO-20 (not ARM-FLAT). C1 freeze still **only ep0011** (ARM-FLAT if still true at ep 120). Factored freeze **ep0047** is pre-PPO-20 — never TEST it
- 4 Oct 02:01 | Sitting close (Ido GO "fill all 3"): budgetstop freeze TEST **22059501** + FR43 **22059502** (Stage-4 ep0095, seed 43) R; the sitting's duplicate C2 TEST 22059499 scancelled | QOS **8/8**; budget menu pinned (start check 02:25) | both arm TESTs at once by a one-time exception (runbook §10.0d); read both before writing M1 or M1-neg
- 4 Oct 02:13 | ops absorbed sitting close / §10.0d | QOS **8/8**; 22056144 + 22059501 + 22059502 all R, TB 0; 22059499 stays cancelled; `Requeue=0` set on 01/02 | read both arm TESTs before M1/M1-neg; FR43 three reads on COMPLETED; STOP-EARLY ping if Budget ends a net above keep 0.80; after the three, ping, do not invent
- 4 Oct 06:20 | FR43 **22059502 COMPLETED** (`ise-4090-03`, 4.3 h) | **§199**; keeps match 17/17 + 61/61; mean \|ΔTEST\| 0.53 / 0.52; **R56 kinder band replicates** (14/17); r20 first-cut deficit does not | no call; M1 on ep0095 stays 21990060; QOS 5/8; **3 idle — ping, do not invent**; ladder empty
- 4 Oct 09:46 | **3h briefing** | QOS **5/8** (3 idle); freezes still ep0095 / ep0011 / ep0083 / ep0131 / ep0047; C2 ev 0.024; canvas 09:16 | **M1-neg is the sitting call**; register SGD-proxy + two-walk mild; ops does not invent; C1 ARM-FLAT at ep 120 if freeze still ep0011
- 4 Oct 10:48 | Stage-4 freeze **ep0131** (probe 0.2863); TEST **22124693 R** (`cs-4090-01`, `tree_v9c`, no timer, `Requeue=0`, Features) | §10.3 one-a-day newest since **21990060** | one freeze TEST in flight; do not queue a second; 2 idle remain — ping, do not invent
- 4 Oct 12:18 | sitting (Ido GO 11:41, metrics/dev) filled QOS: FW **22127216 R** `ise-4090-03`; A0 **22127527 R** `cs-pheno-03`, **22127528/29 PD** QOS | QOS **8/8**; ops does not invent; freeze TEST **22124693** still R 1.5 h TB=0; C1 freeze still ep0011
- 4 Oct 12:47 | **3h briefing** | QOS **8/8**; FW R 52 min TB=0; A0 thin R 35 min; **22124693** R 2.0 h no TRAJ val_best; C1 freeze still ep0011 ev 0.722 | leave sitting cells; one freeze TEST in flight; canvas 16:00
- 4 Oct 13:17 | factored **21940316** froze **ep0083** (probe 0.2947; episode ≥ 80) | first post-update-20 freeze | do **not** TEST while **22124693** is R; after it, §10.5 (a) + `TIME_DECIDE=1` in `tree_v9d`; never TEST ep0047
- 4 Oct 13:40 | A0 thin **22127527 COMPLETED** | **§201** budget-40 **HEADROOM** both keeps (0.6 bar 1.39; 0.35 bar 2.00) | per-net A0-HEADROOM on r56-w4; dg **22127528 R**; cy **22127529 PD**; never TEST
- 4 Oct 15:07 | Stage-4 freeze TEST **22124693 COMPLETED** (`cs-4090-01`, 4.3 h) | **§202** vs mild: first cut **−1.04 / −0.62**; r56 kinder **16/19**; named TEST r20 −4.8 @ 0.702 / r56 −3.9 @ 0.743 | **M1 does not fire; M1-neg stands**; same 0.8 keeps as ep0095
- 4 Oct 15:10 | FW **22127216 COMPLETED** (`ise-4090-03`, 3.2 h) | **§203 SLOWER**: 2.11× **108.4 min** > 85.1; 10k **−1.24** vs N3 −0.46; widths 136/210/267 | never an agent row; K* measured **1.3** (2.11×) / **1.8** (keep 0.36)
- 4 Oct 15:21 | factored freeze TEST **22132735 R** `ise-4090-21` | `tree_v9d` ep0083, `TIME_DECIDE=1`, `SPECTRA_FACTORED_HEAD` pin, 24G, `Requeue=0` | one freeze TEST in flight; never TEST ep0047
- 4 Oct 17:01 | A0 dg-r56 **22127528 COMPLETED** | **§204** budget-40 **HEADROOM** both keeps (0.6 bar 0.50 tight; 0.35 bar 0.94, sens +2.24 / +1.99) | per-net A0-HEADROOM on DepGraph R56; cy **22127529** still R keep 0.35; never TEST; QOS 7/8 — 1 idle, do not invent
- 4 Oct 17:28 | A0 cy-vgg16 **22127529 COMPLETED** | **§205** budget-40 **HEADROOM** both keeps (0.6 random1 +1.76 / +1.33; 0.35 sens2 +0.57 / +0.72) | **cross-net A0-HEADROOM 3/3**; never TEST; no train from ops; QOS 6/8 — 2 idle, do not invent
- 4 Oct 18:20 | **3h briefing** | QOS **6/8** (2 idle); A0 3/3 HEADROOM; factored TEST **22132735** R 3.0 h r20 TRAJ in r56 walking; C1 ep 113 freeze ep0011 | do not invent; ARM-FLAT at 120; canvas 23:00
- 4 Oct 19:26 | Ido **"stop3"**: C1 / C2 / factored trains + held 21940319/21 scancelled | bundles on disk | do **not** resubmit; do **not** write ARM-NEG (Ido stopped them)
- 4 Oct 19:34 | factored freeze TEST **22132735 COMPLETED** | **§206** first cut vs mild **+0.38 / +0.30**; Taylor vs L1 mean **+0.43 / −0.17** (inside FR43 noise) | not M1; not M1-neg; ranking-menu mean is noise; decide 5.4 / 3.6 ms
- 4 Oct 19:32 | A0b **22155641–44 R** (r20 / r56 keep 0.8 / VGG FLOPs / R56-C100) | start checks green | never TEST; idle slot for the fixed-target smoke; ops does not fill
- 4 Oct 20:04 | A0b r56-w4 keep 0.8 **22155642 COMPLETED** | **§208 HEADROOM** (sens +2.63 / +2.01) | κ = 0.8 first-cut read stands; never TEST
- 4 Oct 20:11 | A0b r20-w2 **22155641 COMPLETED** | **§207** HEADROOM at 0.8 and 0.35; **FLAT at 0.6** | R20 stays in v10 M1 (drop needed FLAT at both 0.8 and 0.6)
- 4 Oct 20:21 | v10 smoke **22155996 R** 13 min; mild-landed **22156061 / 62 R** start checks green (keep x0.800 / x0.600, rate=0.9); train **22156018 HELD** | do not release; do not quote smoke; A0b 43/44 still R
- 4 Oct 12:45 | sitting: §200 constant-policy census + reward replay; runbook **§10.0e** rows for FW and A0; report `GILAD_1OCT_POINTS_REPORT.md` | ep0131's TEST (22124693) is already 0.8 at 16/16 r20 and 12/12 r56 decisions | its M1 read will repeat M1-neg's comparison; next sitting: A0 calls, FW call, reward design (GO)
- 3 Oct 10:00 | H0 retries `21986700/01` R ~8 h | origin TEST DN-40 / MBV2 inside 0.12 pp; start checks ok | first net TRAJ in; do not kill; ledger on COMPLETED
- 3 Oct 10:00 | pf-w 89 still R step 281, 7 `[proxy]` lines | 24 h wall ~21:39 | readout at end; freeze TEST takes that GPU; 21940321 held

