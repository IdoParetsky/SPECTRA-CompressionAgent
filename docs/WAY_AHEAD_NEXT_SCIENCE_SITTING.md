# Way ahead — for the next science sitting (written 30 Sep ~03:00 IDT, Opus 5.5 MAX)

Read this first, then `docs/SITTING_GPU_QUEUE.md` (live queue) and the ledger rows it cites. What was built and run is in `docs/RUN_RECORD_29SEP_V9C.md`. The earlier option list and ladder are `docs/PROMPT_FABLE_NEXT_SITTING.md` §13. This file supersedes §13.3's statuses.

## 0. State of play

- **Protocol P is the walk and train protocol.** Val = one 5k half of the test split, TEST = the other; batch 256 pinned. Every verdict closed before 28 Sep was measured on a val the zoo nets had memorized (§141): agent ≡ mild, C100 unrecoverable, C-G dead, and the reward-shape reading.
- **Under P the walk fine-tune is the lever.**
  - Crop+flip in the walk FT is kinder at every equal-width C100 point measured (12/12, mean +3.3 pp TEST, §148).
  - On the C10 R56 twin it reads +2.0 pp paired val.
  - The 100-epoch SGD final FT adds an honest +4.1 to +5.5 pp on the C100 bar-3 cell (§149).
- **Stage 4 is running.** **21737123**: the area train under P + crop+flip (§151). The training rule passed at 03:11 (§150): r56-w4 +2.3 pp TEST at equal keep. The P-only arm was cancelled before it started; its line is kept for the attribution train (N9).
- **No-agent ladder.** 14 cells PD, now on fast cards (`Features=rtx_6000|rtx_4090`).

## 1. Decisions waiting on Ido (recommendation first)

**(a) `21716380` (group-token, held since 28 Sep).**
- *Facts.* `tree_v8b`, legacy val, 7-day cold train. A release re-runs the prologue, which deletes `train_resume.pt` and restarts cold. Its 12 episodes and the `ep0011` freeze are backed up in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` (`train_resume.pt`, `latest_best_*`, `policy_config.json`, `standardizer.pt`) and in `tree_v8b/runs/job21716380/snapshots/ep0011`. Its question (group tokens) is confounded by memorized val, like every legacy train.
- *Recommendation.* **scancel** it: nothing is lost, and a held 7-day legacy job is one mistyped release from burning a slot for a week. Re-run group tokens later as a one-change arm on top of the Stage-4 train, only if that train leaves mild. Never release it; do not TEST `ep0011`.

**(b) C100 catalog emit, and a diverse (C10 + C100) train.**
- *Facts.*
  - Under P both gates admit **6/6** so far (rule: ≥ 4/8). Nets 7–8 finish by ~08:30 or wall out.
  - The emit writes a file (`configs/database_offline_v7_diverse_admitted.json`); nothing trains on it until a profile points there.
  - Three steps. First, fill `configs/v7_c100_gate.json` from the gate log whose recipe matches the train. The Stage-4 recipe is P + aug, so that is the aug gate 21729554. Second, `build_v5_catalog.py --intended configs/database_offline_v7_diverse.json --gate configs/v7_c100_gate.json --out configs/database_offline_v7_diverse_admitted.json`. Third, a `DATASET_NAMES` override in the sbatch, because the profile hard-sets `cifar-10 svhn` (a `tree_v9d` item).
  - The diverse catalog has no SVHN and drops r20-w8.
- *Recommendation.*
  - **GO the emit** once both gates finish: it is reversible and it answers Gilad's Q4 on paper.
  - Hold the **diverse train** until the Stage-4 train's first freeze TEST shows it is not a mild clone. Two parallel trains that both copy mild waste two of four slots for a week.
  - If speed matters more than that risk, run it in parallel: one slot, ~6 days.
- *Draft Q4 answer.* One recipe (Adam 1e-3, 12/4) admits C100 under clean val. Crop+flip, the augmentation every zoo net was trained with, improves it uniformly. No per-dataset recipe.

**(c) Freeze TESTs of the Stage-4 train.**
- *Recommendation.* Pre-authorize ops to TEST the first freeze written after PPO update 20, plus later freezes at most once per day. Use the exact line in ops §9, on the thin pair, against mild under the same protocol. Otherwise each TEST waits for Ido.

**(d) Crop+flip as the TEST walk recipe (bar 2 for every method).**
- *When.* After the twins TEST rows: R56 from 21729553, the VGG twins from 21737104, and the thin 40/10 guard 21729557.
- *Rule.* TEST ≥ 1 pp kinder at equal keep on ≥ 2 of 3 twins, and the thin guard holds. Then every TEST walk (agent and heuristics) switches to aug, and the no-aug rows stay as an audit column.
- *Why it matters.* A P+aug-trained agent should be TESTed under the walk recipe it was trained with.

**(e) Attribution train (only after a success).**
- If 21737123 leaves mild, one P-only train (the cancelled 21737095 line, `scripts/_tmp_s30_train_submit.sh`) separates the val fix from the augmentation. If it does not leave mild, skip it: the weaker recipe cannot do better.

## 2. Insights gathered (ledger refs)

1. **Memorized val (§141).** Zoo nets read val 1.000 / 1.000 / 0.999 unpruned against TEST 0.943 / 0.936 / 0.739. Every legacy reward and gate scored a cut on memorization loss.
2. **No cut gains on full-width nets under the walk FT, even under P (§147: 0/343).** Published light cuts recover to ≥ 0 under SGD + crop/flip, e.g. DepGraph R56 2.11× +0.24 and Network Slimming VGG-19 +0.14. The missing "gain" is a recipe property, not a reward or val property.
3. **Crop+flip helps every capable net and hurts only the tiniest (§148, §150).**
   - The C100 gate is +0.9 to +6.2 pp TEST at equal widths.
   - The r56-w4 decider at 12/4: +2.3 pp TEST at equal keep, and in-band to the end of the walk (−5.1 @ 0.622 vs −10.6 @ 0.741).
   - The 5k-param r20-w2 (64.8 % accuracy) loses 1.0–3.1 pp TEST. That net underfits, and augmentation hurts underfitting nets (NetAug, Cai et al. ICLR 2022). r20-w2 is a hold-out diagnostic, not a train net.
4. **Under P, C100 admits at the live recipe (§148: 6/6).** Four of six nets stay inside τ = 10 to the deepest 2-pass mild point (~0.66 keep). "C100 unrecoverable" was memorized val.
5. **The final 100-ep SGD FT recovers +4 to +5.5 pp at fixed widths (§149).** Origin moves +0.10. SOTA-facing rows (bar 3) must carry it; same-loop rows (bar 2) stay on the walk recipe.
6. **The train FT 12/4 is ~1 pp harsher than the TEST FT 40/10 on r56-w4 (§150).** The agent trains in a harsher world than it is tested in.
7. **Re-walk noise.** Up to 0.8 pp TEST at equal widths across GPU SKUs (§149). Caption walk gaps below ~1 pp as noise.
8. **Probe area is protocol-dependent.** A P-train's area is not comparable with 0.0586 (legacy). Compare freezes by TEST only.
9. **Scheduler.** Untyped GPU requests land on the lowest-weight (slowest) nodes. `Features=rtx_6000|rtx_4090` fixed the no-agent cells without starving on one SKU (run record §6).
10. **Honest gain needs a healthy origin.** Subtracting a negative origin change inflates it: the 1-epoch smoke printed "+5.78 ADOPT" on a raw gain of −0.06. The reader now says `ORIGIN-HURT` when the origin loses > 0.5 pp. That case matters for every new final recipe (KD, AutoAugment, SWA).

## 3. Options re-ranked (status 30 Sep ~03:00)

P = projected probability that the option passes its own adopt rule. Cells: `SITTING_GPU_QUEUE.md`.

| # | Option | Status | P now | Next |
|---|---|---|---|---|
| O1 | crop+flip in the walk FT | **gate rule met (§148); training rule passed (§150)**; TEST-walk rule pending twins | 0.85 TEST walk | 21729553 R56 TEST, 21737104, 21729557 |
| O2 | 100-ep SGD final FT + origin | **met on C100 (§149)** | 0.85 on C10 | 21730500 / 01 / 06 |
| O18 | P gate at the live recipe | **passed (6/6)** | — | nets 7–8 by ~08:30 |
| O3 | crop+flip in the C100 gate | **passed** | — | nets 7–8 |
| O17 | P-val reward train + crop+flip | **Stage 4: 21737123** | 0.40–0.50 leave mild | telemetry, then freeze TEST (GO) |
| O22 | scratch-B at the walk architecture (Liu et al. 2019) | PD after 500 / 501 | 0.45–0.55 | 21730507 / 16 |
| N4 | aug walk + final FT, DG VGG-19 | **submitted** 21737105 | 0.35 (≥ 1 pp after final FT) | pairs with 21729551 |
| O4 | KD in the final FT (N1) | waits on 21730500 | 0.35 | ops §8 line |
| O12 | AutoAugment in the final FT (N2) | waits on 21730500 | 0.30 | ops §8 line |
| O20/21 | C-G under P, NEON train-loss stop | PD 21730509 / 14 | 0.25 real cut; 0.05–0.08 ≥ A | 5-pair big-effect kill |
| O13 | N2 stream protection | PD 21729558 | 0.20 | pairs by params |
| O42 | cubic (NEON) reward under P (+aug): the item-2 A/B | new; a train | 0.25 | only if an aug census shows cuts with val Δ > 0; otherwise cubic just penalises drops harder (→ mildest) |
| O41 | diverse P train (C10 + C100) | new; a train | 0.35 | after (b) |
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
2. **Diverse train profile.** Honour a pre-set `SPECTRA_DATASET_NAMES` (or add `SPECTRA_V6_DATASET_NAMES`) and point `SPECTRA_V6_DATABASE` at the admitted v7 catalog.
3. **Emit.** Generalise `emit_v5_admitted_from_gate_log.py` (hardcoded to v5 paths) to `--gate-log / --intended / --out`, or write the v7 gate table from the gate log and reuse `build_v5_catalog.py`.
4. **Final-FT KD teacher** (in git `5b6398d`) goes live with `tree_v9d`. Drop the `SPECTRA_FT_KD=1` workaround from N1 then.
5. **O38 reward replay** (`scripts/reward_replay.py`). Per-step (val Δ, keep) from finished P walks through the linear in-band, cubic and NEON-exact returns; print where each return would stop.
6. **O26 memorization census** (`scripts/memorization_census.py`). Baseline val in each train log vs the TEST accuracy in the checkpoint name, per catalog net.
7. **Optional O39.** `SPECTRA_FT_AUG_MIN_TRAIN_ACC` (aug off when the unpruned net's train accuracy is below a bar). Only if an underfitting net enters a catalog.

## 5. First moves for the next sitting

1. Stage-4 train telemetry against the control. Ops §9 has the checklist and the control's numbers at PPO updates 1 / 10 / 20 / 30.
2. Ledger rows ops wrote since §151; the twins TEST for O1 (decision (d)).
3. With GO, TEST of the first freeze under the train's own walk recipe, against mild under the same recipe, at equal keep. Include the compression-rate census for the mild-clone read.
4. Build `tree_v9d` (§4 items 1–5), CPU pytest on the cluster conda.
5. Gilad summary. Items 1–4 of the status note, answered with the P evidence (§147–§150).

## 6. Literature used in this cycle

He et al. 2016 (crop+flip CIFAR recipe) · Li et al. ICLR 2017 (filter pruning, FT recipe, per-stage sensitivity) · Liu et al. ICLR 2019 (rethinking pruning: scratch-B) · Le & Hua ICLR 2021 (retraining schedule matters) · Fang et al. CVPR 2023 (DepGraph; published R56 / VGG-19 rows) · PruningBench 2024 (100-ep FT protocol) · Cai et al. ICLR 2022 (NetAug: augmentation hurts tiny nets) · Hinton et al. 2015 (KD) · Cubuk et al. CVPR 2019 (AutoAugment) · Izmailov et al. 2018 (SWA) · Hirsch & Katz, Information Sciences 2022 (NEON: cubic reward, train-loss stop, patience 10).

## 7. Ops annotations (append-only, dated; ops writes here, the next sitting reads)

Format: `- <date time> | <job / event> | <number, ledger §> | <implication for the next sitting>`.

- 30 Sep 03:11 | sitting | Stage-4 train released after §150; P-only arm cancelled | first read: FLAGS + `policy_config` diff (ops §9.2)
- 30 Sep 03:16 | 21737123 R 03:14, `ise-cpu256-32` RTX 6000 Ada | start checks green: env header; val-from-test on cifar-10 and svhn; aug on cifar-10 only; `policy_config` diff = P keys only | read its curve's shape against 21536396, never its probe-area values
- 30 Sep 03:14 | 21730499 smoke-from COMPLETED, passed | 1-epoch FT from saved; origin −5.84 pp printed "honest +5.78 ADOPT" on a raw gain of −0.06 | reader fixed (`ORIGIN-HURT`, git + `readers_s30/`). For `tree_v9d`, and for any new recipe (KD, AutoAugment), check the origin row before the verdict
