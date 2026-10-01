# SPECTRA sitting GPU queue

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger, and the pre-authorized actions in the runbook §10.3 — do not invent cells.
**Rule (Ido 29 Sep 15:49):** QOS stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells. Cap is **live `gpu-part` MaxTRESPU** (8 as of 1 Oct 00:09). Do not invent cells when the ladder is empty.
**Pre-authorized (Ido 30 Sep 11:08):** freeze TESTs of the Stage-4 train (first after PPO update 20, then ≤ 1 a day); its resume past the 6-day fuse (chained: 21767188). **Done on that GO:** 21716380 scancelled; C100 catalog emitted (§148). **Done on Ido's 12:34 GO:** 21730506 converted to the crop+flip walk → **21809595**.
**Done on Ido's 1 Oct 01:03 GO:** the two one-change reward trains (C1 **21938807**, C2 **21938810**; resumes chained).
**Done on Ido's 1 Oct asks:** 08:29 the layer-replacement grid (15 jobs); 08:42 the agent-design arms (11 jobs, two trains held on gates); 08:56 the FT proxy-fidelity cell (zero-GPU look + 6 jobs). Sections below.
**Still Ido GO:** a DRL train (N8, N9; for N8 see the conditional-GO proposal, roadmap §3 G5); a second resume.

Ops handoff, lines, greps and kill rules: **`docs/OPS_HANDOFF_RUNBOOK.md` §10** (current), §8 (cells). Options, decisions and dev items: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. N8: `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. What was built and run: `docs/RUN_RECORD_29SEP_V9C.md`. Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12.

**Stamped:** 1 Oct 2026, ~09:20 IDT. **QOS `gpu-part` `gres/gpu=8`**. After four LR KILLs: Stage-4, C1, C2, budget-stop smoke R; LR CG draining. **PD** fill the freed slots (pca r56, smokes, pf-*, remaining LR). Do **not** invent; do **not** launch N8; do **not** release held trains.
- *09:20 KILL.* C-G thin **21940176/177** §176; producers-only thin **21940178/179** §177. C-PCA r20 **21940180** COMPLETED §178 (harsher at equal keep). Full-width LR still PD.
- *G2 sitting (charge 1 Oct 01:03).* **A1** hold-outs 8/8. **A2** greedy §173; random r20 §174; random r56 **§175**. **A3–A5** CROSS-OFF / KILL. **C** C1/C2 R (PPO-2). **D** smoke passed; **no N8**.
- *Overnight COMPLETED.* N3 **21767189** §157 **M4**; scratch-thin **21730507** §158 CROSS-OFF; scratch-DG **21730516** §159 ADOPT; N1 **21767190** §160 mixed; N2 **21767192** §161 not M5; streams **21729558** §162 split.
- *Ledger.* Next **§179**. LR KILLs §176–§177; C-PCA r20 §178.
- *Trees.* `tree_v9b` / `tree_v9c` frozen. **`tree_v9d`** = v9c + the G2 dev pass, default-off for every existing profile (`PROVENANCE_v9d.txt`). Train `Requeue=0`.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c`). **Paired read** = `readers_s30/scripts/paired_steps.py <arm> <control>`: val only, same step = same widths under mild. **Honest gain**: only `readers_s30/scripts/final_ft_readout.py` (prints `ORIGIN-HURT` when the origin loses > 0.5 pp).

## Live rank

Equal keep = the size points (`param:0.8,0.6` thin; `0.9,0.8` C100) and `val_best`, read on the 5k TEST half. Greedy and random cut differently per step, so `paired_steps.py` labels are valid only between two mild walks.

| Pri | Cell | Job | State (1 Oct 03:15) | Tree | Checks | Read so far / hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | **Stage-4 train**: area train under P + crop+flip (§151) | **21737123** → resume **21767188** | R since 30 Sep 03:14 (`ise-cpu256-32`); **~1d 05h**; PPO **9** at 08:03: ev **0.774**, batch_score 0.528, probe best 0.282; freeze still **ep0011** only; fuse ~6 Oct 03:15 | v9c | leave mild under P + crop+flip? | M2 at update 10 (next PPO) | no freeze by 250; mild clone at TEST | M1 |
| 1b | Freeze TESTs of Pri 1 | — | none yet; first freeze after PPO update 20 | v9c | vs 21729557 + census | — | mild clone | M1 |
| 2 | **C1 cubic-gain train**: the Stage-4 line, reward scale only → `cbrt_miss` (gain +ρ³, band +ρ, miss −ρ) | **21938807** → r1 **21938809** | **R since 03:04** (`cs-4090-07`); PPO **2** at 08:03: ev **−0.014**, batch_score 0.407, best −inf; `scale=cbrt_miss` | v9d | ev and batch_score vs 21737123 by PPO update; `Requeue=0` | O38: at 12/4 the gain arm is rare (3 of 76 thin cuts, 0 on C100), so C1 may train close to live | **report, never scancel**: ev ≤ 0 by update 10, or the freeze is a ≥ 90 % mild clone | first freeze TEST beats the Stage-4 freeze at equal keep |
| 3 | **C2 NEON-raw train**: scale → `raw` (+ρ³ / +ρ / −ρ³ on the realised cut) | **21938810** → r1 **21938811** | **R since 03:24** (`ise-4090-15`); PPO **2** at 08:03: ev **−0.072**, batch_score 0.422, best −inf; `scale=raw` | v9d | as C1 | O38: = C1 on every walk without a miss; differs only on the miss arm | same | same |
| 4 | A1 hold-out checkpoints, SVHN: ShuffleNetV2 ×1, RepVGG-A0, MBV2 ×0.5, DN-40, 200 ep | **21938295** | **COMPLETED** 03:24, 0/4 failed; last-epoch test acc ShuffleNetV2 96.55, RepVGG-A0 96.71, MBV2×0.5 96.64, DN-40 96.34 | v9d | `configs/input_g2_holdout_svhn.json` (4 rows, in git); `test_v5_catalog.py` 17 passed with both hold-out files | hold-out set | a net < 90 % | hold-out set (roadmap §2b); never in a train catalog |
| 5 | A1, Fashion-MNIST | **21938296** | **COMPLETED** 03:04, 0/4 failed; best-epoch test acc (in the checkpoint name) ShuffleNetV2 94.82, RepVGG-A0 94.96, MBV2×0.5 94.93, DN-40 95.29 | v9d | `input_g2_holdout_fmnist.json` (4 rows, in git) | hold-out set | a net < 90 % | hold-out set (roadmap §2b) |
| 6 | A2 **greedy** (profile `l1`: Ido's "L1" = Gilad's "greedy"), thin, P + crop+flip 40/10 | **21938279** | **COMPLETED** 05:25 §173 | v9d | vs mild **21729557**; keeps unmatched | r20 −4.0 @ 0.595 vs −4.9 @ 0.584; r56 −5.7 @ 0.743 vs −2.6 @ 0.795; greedy `val_best` deeper and worse | **CROSS-OFF** as bar-2 walk | kinder at equal keep |
| 7 | A2b **random**, r56-w4, seed 42 | **21938285** | **COMPLETED** 06:04; one draw in §174 | v9d | vs 21729557 r56-w4 | `val_best` −6.8 @ 0.450 | — (baseline row) | mean of 3 draws |
| 8 | A3 SGD 0.01, C100 tight-2 (r20-w13, r56-w9), 12/4 | **21938284** | **COMPLETED** 03:26 → ledger **§170** | v9d | vs **21729554** | harsher at 3 of 4 equal-keep points: r20-w13 −7.2 / −8.4 vs −5.2 / −6.9; r56-w9 −12.6 vs −7.7 at 0.86, −9.2 vs −9.5 at 0.79 | **CROSS-OFF** as train FT (with §167); cap-40 stays crossed | — |
| 9 | A5 F1 cosine, thin 12/4 | **21938286** | **COMPLETED** 03:55 §171 | v9d | vs **21729556** | r56 **+3.3 / +1.0**; r20 0.60 **−4.2** | **CROSS-OFF** as train FT | r56 kinder and r20 within 0.5 |
| 10 | A5 F2 group-first 4, thin 12/4 | **21938287** | **COMPLETED** 04:24 §172 | v9d | vs 21729556 | r56 **+2.3 / +0.9**; r20 0.60 **−2.0** | **CROSS-OFF** as train FT | r56 kinder and r20 within 0.5 |
| 11 | D **smoke** `offline_train_v9_diverse`, 2 episodes | **21938898** | **COMPLETED** 04:44, 0 TB, 16 nets, stop after 2 ep | v9d | plumbing only; **never ledger** | passed | any fail → fix before N8 | G2 smoke **met**; N8 still needs G5 |
| 12 | A2b random, r20-w2, seed 42 | **21938894** | **COMPLETED** 04:41 (one draw; wait s43/s44) | v9d | vs 21729557 r20 | `val_best` −6.7 @ 0.471; size 0.80 −0.9 @ 0.782 | — | mean of 3 draws |
| 13 | A2b random, draws 2 and 3 (seeds 43, 44), r56-w4 / r20-w2 | s43 **21938895** / **21938896**; s44 **21938929** / **21938930** | r20 **§174**; r56 **§175** (s44 COMPLETED 08:11) | v9d | random row = mean of the 3 draws per net, never the best draw | mean harsher than mild on r56 | — | — |

## NEXT (conditional)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| N8 | diverse P train (8 C10 + 8 C100, catalog A): profile **`offline_train_v9_diverse`** in `tree_v9d` | roadmap §3: G1 (M1) + G2 (**smoke passed 04:44**) + G5 | NEON's multi-dataset offline train; then frozen → ImageNet, SVHN, Fashion-MNIST | roadmap §4 | roadmap §4 |
| H0 | mild walks on the G2 hold-out checkpoints | A1 COMPLETED **and** a loader check: P (val from test) and crop+flip on SVHN / Fashion-MNIST (flip is not label-safe on digits) | the hold-out bar for N8 H5 / H7 | — | sitting GO; not in the 1 Oct charge |
| LA | accuracy look-ahead heuristic | today "look-ahead" = the floor guard, off under TRAJ, so it equals greedy | a one-step look-ahead costs ~3× FT per step | — | design question for Gilad |
| D5 | GPU-side crop+flip (roadmap §5 item 5) | before the N8 launch | up to +18 % per epoch | never into a live train | — |
| N8b | N8 + SVHN nets in training (pre-registered, roadmap §2b) | N8 passes on CIFAR (H2, H3) but is below mild on the dataset hold-outs (H5, H7) | read on Fashion-MNIST, ImageNet and the unlike families | — | — |
| N9 | attribution train: P-only (the 21737095 line) | Pri 1 leaves mild | P vs P + aug in training | — | — |
| O26 | `scripts/memorization_census.py` | **built 1 Oct** (ledger §169) | legacy v3 train 21385158: 24 / 24 MEMORIZED, val − TEST +3.35 to +7.19 pp; Stage-4 P train 21737123: 0 / 10, −0.63 to +0.08 pp | — | done |

## Layer-replacement grid under P + crop+flip (Ido GO 1 Oct 08:29)

Every replacement construction that had not run under P + crop+flip, one job per net, `tree_v9d`, 2-pass mild TRAJ, 40/10. C-G family = NEON's own stop (train loss, patience 10, cap 100), as §156 / §163. Before this grid: C-G clean without aug (§156); C-G+ clean + aug on r20-w2 only (§163, r56-w4 never started); producers-only (§108) and C-PCA (§127) memorized val only.

| Construction | r20-w2 | r56-w4 | R56 C10 | VGG-16 C10 |
|---|---|---|---|---|
| C-G (group redraw) | **KILL §176** | **KILL §176** | 21940183 | 21940184 |
| C-G producers-only ("the pruned layer only") | **KILL §177** | **KILL §177** | 21940186 | 21940187 |
| C-PCA (principal-direction layer) | **COMPLETED §178** | 21940181 | 21940188 | 21940189 |
| C-G+ (C-G + 0.1× polish) | §163 KILL | 21940182 | 21940191 | 21940192 |

Bold = R since 08:35 (start flags checked); the rest PD nice 25–34, thin first. Controls: thin **21729557** (tree_v9b), twins **21809595** (tree_v9c; walk ≈ 21729553 R56 / 21737104 VGG-16).
- *Kill (pre-authorized, per job).* `paired_steps.py` vs its control: ≥ 5 pairs, mean ≤ −3 pp val, ≥ 4/5 worse → scancel and ledger.
- *Read (on COMPLETED).* TEST at equal keep (size points, `val_best`) vs the control's rows.
- *Cross off a construction.* Killed, or worse than keep-the-survivors at equal keep, on ≥ 3 of its 4 nets.
- *Re-open.* Within 0.5 pp of the control, or kinder, at equal keep on ≥ 3 of 4 nets including one full-width net.

## Agent-design arms: one change each on the Stage-4 line (Ido 1 Oct 08:42)

Each arm is 21737123's recipe with one switch changed: P5-B2 catalog (CIFAR-10 + SVHN), live in-band reward, area probe, P + crop+flip, its seed and governor. All on `tree_v9d`, `Requeue=0`, resume chained `afterok`. Every one of these features was trained only under memorized val, so none has a verdict yet (the re-open rule): factored head §110 / §134 / §137, budget + STOP §135 (no freeze), group tokens 21716380 (scancelled at 12 episodes). The three reward options are Stage-4 (live), C1 and C2 (Live rank Pri 1–3).

| Arm | Change vs 21737123 | Smoke (4 ep, never ledger) | Train | Resume | Gate |
|---|---|---|---|---|---|
| Budget + STOP | `offline_train_v7_budget`: cut 0 / 1 / 2 / 4 % of the net's params through this group (L1), or STOP (scale 100) | **21940310** | **21940311** nice 40, `afterok` smoke | 21940314 | released |
| Two-decision head | `offline_train_v6_inband_p5b2_factored`: keep {1.0, 0.9, 0.8} × criterion {L1, FPGM, BN-scale, SVD, Taylor} | **21940315** | **21940316** nice 41, `afterok` smoke | 21940317 | released |
| Group-as-token state | `offline_train_v8_grouptoken`: one encoder token per dependency group | **21940318** | **21940319 held** nice 42 | 21940320 | Stage-4's first post-PPO-20 freeze TEST is not a mild clone (way-ahead (a)), or Ido |
| 40/10 train FT | `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10` (the TEST's FT budget in the loop; §150: 12/4 is ~1 pp harsher on r56-w4) | none (Stage-4's code path) | **21940321 held** nice 43 | 21940322 | the FT proxy-fidelity check shows 12/4 misranks cuts that 40/10 ranks right, or Ido |

- *Order.* Smokes run right after the five running LR jobs; the released trains start once the whole LR grid has started.
- *Slots.* Slurm does not preempt, so a train holds its GPU ~6 days per leg. At most **5 trains R** (Stage-4, C1, C2 and the two released arms): that leaves 3 GPUs for freeze TESTs, the LR grid and the fidelity cell. When a held arm's gate passes, ops pings Ido; the release (`scontrol release <id>`) is his call, best timed with the end of a train leg (Stage-4 fuse ~6 Oct, C1 / C2 ~7 Oct). N8 on GO goes ahead of any held arm.
- *Smoke pass.* `Stopping PPO training after 4 episodes`, one `PPO update`, no Traceback. A failed smoke leaves its train in `DependencyNeverSatisfied`: report; do not patch `tree_v9d` from ops.
- *Progress.* Per PPO update vs 21737123, indexed by episode, not wall clock (40/10 runs ~2–3× slower per episode): probe area, critic ev, `gap_to_uniform`, `batch_score`.
- *Preliminary read.* Each arm's first freeze after PPO update 20, TESTed exactly like Stage-4's (runbook §10.3: thin pair, P + crop+flip 40/10, vs mild 21729557 at equal keep, with the compression-rate census).
- *Final read.* The last freeze: the same TEST, plus the G2 hold-outs once H0's loader check passes.
- *Adopt.* Kinder than the Stage-4 freeze by > 1 pp at equal keep on both thin nets **and** not a ≥ 90 % mild clone → a second seed before it changes the recipe. One seed each, so smaller gaps are noise.
- *Report, never scancel* (as C1 / C2): ev ≤ 0 by PPO update 10, or a freeze that is a ≥ 90 % mild clone.
- *Not enqueued.* SGD 0.01, cosine and group-first FT in the loop: each failed the pair rule as the walk FT under P + crop+flip (§167, §170–§172). The literature SGD-cosine recipe stays the final FT.

## FT proxy fidelity (Ido GO 1 Oct 08:56)

The question: does the agent's in-loop recovery (recipe A, Adam 1e-3, 12/4, crop+flip) rank candidate cuts the way the final fine-tune does (SGD 0.01, momentum 0.9, wd 5e-4, cosine, crop+flip, 100 epochs)? Le & Hua (ICLR 2021) show the retraining schedule can reorder pruning methods; EagleEye (ECCV 2020) runs the same correlation for candidate evaluators, with re-estimated BN statistics as the cheap proxy. Gates the held 40/10 train (21940321) and N8's in-loop recipe.

**Zero-GPU look (1 Oct 09:05; every inherit final FT on disk, 1-epoch smokes excluded):**
- *With crop+flip in the walk*, the 40/10 walk TEST is within 0.5 pp of the final FT at **16 of 16** size points: DepGraph R56 (21767189), the chenyaofo R56 and VGG-16 twins (21809595), VGG-16 L2 (21814029), DepGraph VGG-19 C100 (21737105).
- *Without it*, the final FT adds +1.6 to +2.2 pp on DepGraph R56 (21730500), +4.2 to +5.6 on VGG-19 C100 (21729551), +5.6 to +5.8 on r56-w4, and −0.3 to −1.0 on r20-w2 (both 21730501).
- The final FT never reorders the two walk variants (with vs without crop+flip) on DepGraph R56 and VGG-19: **7 of 7** size points keep their order; the gap shrinks from 2.6–6.4 pp to 0.3–1.0 pp.
- It cannot answer the question: no cell tests the 12/4 budget, candidate cuts within one state, or a thin net under crop+flip with a final FT.

**GPU cell.** `tree_v9d`, `SPECTRA_EVAL_PROXY_FIDELITY` (`src/proxy_fidelity.py`, default off; `tests/test_proxy_fidelity.py` 6 passed). Mild TRAJ walk under P + crop+flip 40/10, seed 42; at the first point ≤ the target the battery runs, then `SPECTRA_EVAL_SIZE_MATCH` ends the walk. All nice 24, `Features=rtx_6000|rtx_4090`.

| Net | keep ≤ 0.9 | keep ≤ 0.7 |
|---|---|---|
| r56-w4 (thin probe; 12/4 is ~1 pp harsher than 40/10 here, §150) | **21941343** | **21941344** |
| ResNet-56 ×6 (P5-B2 training net) | **21941345** | **21941346** |
| MobileNet-V2 ×0.5 (P5-B2 training net, depthwise) | **21941347** | **21941348** |

- *Candidates per state (≤ 12).* `identity`; `menu` = keep 0.9 / 0.8 × L1 / FPGM on one row (the walk's next row, or the next one ≥ 10 channels wide); `crit` = that row at keep 0.8 under BN-scale, SVD, Taylor; `where` = up to 4 other groups cut by L1 at the same share of the network as the menu's 0.8 L1 cut. Exact duplicates are recorded, not scored.
- *Scores.* Proxies on **val**: `none` (raw cut), `bn` (BN re-estimated), `12x4`, `40x10` (recipe A). Final on **TEST**: from the raw cut, seeds 0 and 1 (the second seed is the noise ceiling). ~30 min per candidate, ~7–9 h per job.
- *Readout.* `python scripts/proxy_fidelity_readout.py runs/job21941343 … runs/job21941348` (CPU). Ranked sets = `crit` and `where` (equal size, ≥ 3 distinct candidates); `menu` is read as the depth penalty only.
- *Registered calls (written before any result; the readout prints them):*
  - Ceiling = mean ρ(final s0, final s1). Below **0.5** → uninformative: widen the cuts before reading any proxy.
  - A proxy is **valid** if its mean ρ(proxy val, final TEST) ≥ max(0.6, 0.8 × ceiling) **and** its median top-1 regret ≤ 0.5 pp.
  - **12x4 valid** → Adam 1e-3 12/4 stays the in-loop proxy; N8 unchanged; 21940321 stays held.
  - **12x4 not valid**, and 40x10 valid or ρ(40x10) − ρ(12x4) ≥ 0.2 → ping Ido to release 21940321.
  - Neither valid → next cell: SGD variants (0.05 / 0.1, cosine, short budgets) as proxies against the same finals.
  - `bn` valid and within 0.1 of 12x4 → a cheap-proxy train arm is a sitting question (3–4× more episodes per GPU-day).
  - Depth penalty: median Δ(0.9) − Δ(0.8) under 12x4 over the final's. Above 1.5 means the proxy over-penalises the deeper cut, which pushes the agent toward mild: report, no action.
- *Never* ledger these walks' TRAJ rows (truncated at the target). One ledger section at the readout, with the zero-GPU look.

## O38 reward replay (zero GPU, val only; 1 Oct 03:10)

`scripts/reward_replay.py <run_dir...> --tau 10 --gamma 1` replays finished no-agent walks through the live `src.utils.compute_reward` under each shape. Identity steps earn no size credit. Return = undiscounted sum along the walk; in brackets, the share of the positive return paid on cuts with cumulative val Δ > 0 (the gain arm). Not a TEST.

| Walk | Cuts: gain / band / miss | Ends (keep, val Δ) | Live `cbrt_cubes` | Cubic gain `cbrt_miss` (C1) | NEON raw, realised ρ (C2) | Full `cbrt` | NEON literal, nominal ρ |
|---|---|---|---|---|---|---|---|
| N3 DG R56, P + aug 40/10, 5-pass (21767189) | 150: 69 / 81 / 0 | 0.356, −1.10 | 101.4 (33 %) | 493.1 (86 %) | 493.1 (86 %) | 94.7 (36 %) | 6.98e4 (99 %) |
| DG R56, no aug (21730500; 21726340) | 150: 0 / 150 / 0 | 0.356, −4.62; −4.04 | 101.4 | 101.4 | 101.4 | 105.2 | 1500 |
| thin aug 40/10, r20-w2 (21729557) | 16: 6 / 10 / 0 | 0.522, −4.34 | 102.9 (31 %) | 2267 (97 %) | 2267 (97 %) | 51.2 (63 %) | 6100 (98 %) |
| thin aug 40/10, r56-w4 (21729557) | 60: 0 / 60 / 0 | 0.622, −4.34 | 126.6 | 126.6 | 126.6 | 60.5 | 600 |
| thin aug 12/4, r20-w2 (21729556) | 16: 3 / 13 / 0 | 0.522, −5.18 | 102.9 (7 %) | 134.2 (29 %) | 134.2 (29 %) | 31.8 (22 %) | 3130 (96 %) |
| thin aug 12/4, r56-w4 (21729556) | 60: 0 / 60 / 0 | 0.622, −5.30 | 126.6 | 126.6 | 126.6 | 60.5 | 600 |
| aug C100 gate 12/4, the 7 nets without a miss (21729554) | 18–76: 0 gain, 0 miss | 0.658–0.696, −1.18 to −8.84 | 36.1–46.4 | = live | = live | 21.7–58.1 | 180–760 |
| aug C100 gate 12/4, r56-w9 (21729554) | 60: 0 / 52 / 8 | 0.642, −10.46 | 18.04 | 18.04 | **−368.9** | 22.5 | **−7480** |

Every shape's return peaks at the walk's last cut, except on r56-w9: cut 51 of 60 (keep 0.742, val −7.66), NEON literal at 45 (0.756, −9.44).

1. **The live reward is blind to Δacc inside the band.** With no miss it pays Σρ: the aug and no-aug R56 walks both score 101.4 with a 3.5 pp val gap.
2. Full cbrt pays the no-aug walk more than the aug walk (105.2 vs 94.7): ρ^⅓ > ρ for ρ < 1, so it rewards many tiny cuts.
3. **C1 = C2 on every walk without a miss.** The two trains differ only on misses (−ρ vs −ρ³). One 7.3 % cut 0.76 pp past τ costs −7.3 live, −394 raw.
4. Cubic-gain credit sits on a few large cuts made above the origin (86–97 % of the return from ρ > 1 cuts); tiny cuts earn ≈ 0. C1 pushes toward large cuts while the net is above origin.
5. τ = 10 binds on 1 of 15 walks. The shapes differ in what they pay along a walk, not in where a return-maximiser stops.
6. At the train FT (12/4) the gain arm is rare: 3 of 76 thin cuts, none on C100. C1 may see few gain-arm steps.
7. Band miss ties live on r56-w9 by coincidence (18.036 vs 18.044). It also charges identity steps taken past τ; live charges them 0.

## `tree_v9d` (G2 sitting, 1 Oct)

`/home/paretsky/scratch_audit/tree_v9d` = `tree_v9c` + the G2 dev pass; each install is stamped in `PROVENANCE_v9d.txt`. Pytest: 346 passed, plus the 3 files the login-node watchdog killed, run single-threaded (densenet 1, v2 recipe 15, v4 factored 11) and policy_config 2; `test_v5_catalog.py` 17 passed after catalog A.

| Roadmap §5 item | State |
|---|---|
| 1 profile `offline_train_v9_diverse` | **built.** Stage-4 recipe pinned in the profile (P, batch 256, crop+flip, area probe; in-band linear reward, keep-rate menu, 12/4). Catalog A, `cifar-10 cifar-100`, `--check-admitted --gate configs/v7_c100_gate.json`; refuses a catalog with no C100 rows. If M1 changes the recipe, change the pin |
| catalog A on the cluster | **was missing**: v9b / v9c / v9d all held the 21 Sep C10-only admitted file (8 rows, gate `pending_regate`), which `--check-admitted` passes. Deployed git `d92a1a0` (16 rows, 8 admitted) into v9d only |
| 2 provenance | `SPECTRA_FT_AUG` / `SPECTRA_FT_AUTOAUG` in `POLICY_INFO_KEYS` |
| 3 requeue safety | `--no-requeue` on every `offline_train*` submit; the prologue keeps a run's own `train_resume.pt`; resume comment fixed |
| 4 probe set | + `resnet20-width13_cifar100` (3 probe nets), v9 profile only |
| 5 GPU crop+flip | not built (D5) |
| 6 tests + smoke | pytest done; smoke **21938898** R, start greps pass |
| 8 hold-out checkpoints | A1 R; the job writes the input files; the disjointness test covers them |
| 9 size points | `eval_size_match` docstring fixed (OCS ≈ 0.42 params is R56; VGG-16 matches on FLOPs) |
| reward | `cbrt_miss` scale; `SPECTRA_REWARD_SCALE_ARM` (in-band train profiles only); `scripts/reward_replay.py` + test |
| final-FT KD teacher (way-ahead §4 item 4) | git `5b6398d` deployed 03:17 (it was not in v9c, so v9d lacked it): `SPECTRA_EVAL_FINAL_FT_KD=1` alone now distils from a frozen copy of the original. `test_v9c_traj_models.py` 15 passed. Drop the walk-side `SPECTRA_FT_KD=1` workaround on v9d final-FT lines |

## Done (ledgered; do not re-run)

| Cell | Job | Ledger | One-line result (5k TEST half; 10k = cross-fit / both halves) |
|---|---|---|---|
| final_ft P thin, no-aug walk | 21730501 | §154 | r20 **CROSS-OFF** honest −3.8 (origin +3.46); r56-w4 **ADOPT** +5.3 / +5.5 @ 0.795 / 0.756 |
| N4 aug walk + final FT, DG VGG-19 | 21737105 | §155 | walk kinder (+4.48 val); honest **CROSS-OFF** −0.50 to −0.94; 10k −1.62 @ 0.684 vs §149 −2.39 |
| C-G NEON-rule P | 21730509 / 21730514 | §156 | **KILL** scancelled 19:12; mean −12 to −54 pp vs mild |
| C-G+ P+aug thin | 21938280 | §163 | **KILL** scancelled 01:45; r20 mean **−20.8 pp** vs 21729557 (0/5 better); r56 never started |
| aug twins + 100-ep FT, C10 | 21809595 | §164 | walk ≈ §152; long FT **CROSS-OFF** (R56 honest ~0; VGG-16 origin +0.88) |
| 10-pass VGG-16 HRank/OCS FLOPs | 21814029 | §165 | 10k **−0.25 @ FLOPs 0.464** (HRank −0.53 at 17 % params; we keep 44 %); **−2.02 @ 0.211** (OCS −0.44); long FT CROSS-OFF |
| Adam 1e-4 thin 12/4 | 21938281 | §166 | **CROSS-OFF** vs 21729556: r20 −5.3 / −14.5 at equal keep; r56 0.80 **+3.1** |
| SGD 0.01 thin 12/4 | 21938282 | §167 | **CROSS-OFF**: r56 −0.7 / −2.2 vs Adam 1e-3 |
| Adam 1e-4 C100 t2 | 21938283 | §168 | kinder than §148 on both nets; C10 failed so not a train switch |
| SGD 0.01 C100 t2 | 21938284 | §170 | **CROSS-OFF** vs 21729554: harsher on 3/4 equal-keep |
| F1 cosine thin 12/4 | 21938286 | §171 | r56 **+3.3 / +1.0**; r20 0.60 **−4.2** → **CROSS-OFF** as train FT |
| F2 group-first thin 12/4 | 21938287 | §172 | r56 **+2.3 / +0.9**; r20 0.60 **−2.0** → **CROSS-OFF** as train FT |
| greedy L1 thin 40/10 | 21938279 | §173 | not size-matched; r56 worse at smaller keep; **CROSS-OFF** as bar-2 walk |
| random r20-w2 3-draw | 21938894 / 96 / 930 | §174 | mean ≈ mild (−1.0 / −4.3 / −5.5 vs −1.2 / −4.9 / −5.3) |
| N3 aug walk + final FT, DG R56 | 21767189 | §157 | **M4** 10k **−0.46 @ 2.11×** vs DepGraph +0.24; long FT CROSS-OFF |
| scratch-B thin | 21730507 | §158 | r20 CROSS-OFF; r56 ORIGIN-HURT |
| scratch-B DG R56 | 21730516 | §159 | ADOPT 10k **−0.16 @ 2.11×** |
| N1 KD from saved, DG R56 | 21767190 | §160 | +0.62 only at keep 0.60; not vs N3 walk |
| N2 AutoAugment from saved | 21767192 | §161 | not M5; origin +1.24 |
| N2-streams P | 21729558 | §162 | r20 kinder; r56-w4 not deeper vs crop+flip |
| aug twins VGG (VGG-16 C10 + VGG-19 C100) | 21737104 | §152 | VGG-16 **+2.3 to +2.6 pp** at equal keep; VGG-19 C100 **+3.8 to +4.9** (−2.6 / −1.9 / −2.5 vs −6.4 / −6.8 / −6.7 at 0.796 / 0.688 / 0.657) → **twins 3/3** |
| final_ft twins C10, no-aug walk | 21730506 | — | **cancelled while PD** 12:47 on Ido's GO; replaced by 21809595 (crop+flip walk) |
| aug thin 40/10 (the (d) thin guard) | 21729557 | §152 | r56-w4 **+5.0 pp** at 0.795 (−2.6 vs −7.6); `val_best` −4.5 @ 0.622 vs −10.1 @ 0.739; r20 1.3 / 0.7 / 1.6 worse (inside 2 pp) → **(d) met**. The mild control for freeze TESTs |
| final_ft DG R56 C10 | 21730500 | §153 | honest +1.18 / +1.80 / +1.74 (HOLD); 10k final −1.06 @ FLOPs 0.599, **−1.52 @ 0.463** (DepGraph +0.24), **−2.11 @ 0.380** (DepGraph +0.11); origin +0.42; re-walk vs 21726340 −0.04 |
| aug gate C100 12/4 | 21729554 | §148 | **8/8 admitted** at 0.647–0.696 kept (val −1.18 to −9.70); 12/12 size points kinder than no aug (+0.9 to +6.2, mean +3.3); emitted |
| P gate C100 12/4 | 21729552 | §148 | 6/6 finished admitted; TIMEOUT at mbv2x1, densenet40 not started |
| final_ft DG VGG-19 C100 | 21729551 | §149 | honest +4.08 / +4.46 / +5.52; final −2.52 @ 0.684 (10k −2.39), −3.28 @ 0.599 (10k −3.04); origin +0.10 |
| P thin 12/4 | 21729555 | §150 | r20 size 0.80 −0.1 @ 0.774 (10k +0.64); r56-w4 −10.6 @ 0.741, size 0.80 −8.2; 12/4 is −1.03 pp vs 40/10 on r56-w4 |
| aug thin 12/4 | 21729556 | §150 | **training rule passed**: r56-w4 size 0.80 −5.9 (+2.3), `val_best` −5.1 @ 0.622; r20 −1.0 / −1.3 / −3.1 at equal keep |
| aug twins 40/10 (R56 only) | 21729553 | §152 | R56 size 0.80 −0.16 vs −2.68 (10k −0.37 vs −2.85), size 0.70 −0.50 vs −2.70, `val_best` −0.06 vs −2.84 @ 0.661; census val Δ > 0 on 0/62 (max −0.12); scancelled at VGG-16 step 1 |
| smoke-save (v9c) | 21730498 | never | saves + scratch lines + no PicklingError: passed |
| smoke-from (v9c) | 21730499 | never | loads `val_best` + size from the saved walk, `init=inherit`, origin row: passed. Exposed the ORIGIN-HURT reader flaw (fixed) |
| smoke-ft (v9b) | 21729550 | never | final_ft path prints with SAVE unset; plumbing only |
| P twins | 21726337 | §142 | R56 −2.8 @ 0.661 (10k −3.03); VGG-16 −2.8 @ 0.657 (10k −2.78); VGG-19 −6.7 @ 0.657 (10k −6.55) |
| P thin | 21726335 | §143 | r20 −3.7 @ 0.536 (10k −2.57); r56-w4 −10.1 @ 0.739 (10k −9.86) |
| P N4 | 21726338 | §144 | first undo 77 not 39; cross-fit invalid (rollback reads val) |
| P canary | 21726336 | §140 | admitted under P; 10k −7.86 @ 0.659 |
| DepGraph VGG-19 P | 21726341 | §145 | −7.9 @ 0.534 (10k −7.55); size 0.70 10k −6.09 |
| DepGraph R56 P | 21726340 | §146 | −4.0 @ 0.356 (10k −4.02); flop 0.47 10k −3.32; flop 0.39 10k −3.98 vs +0.11 |
| N0 3-seed b256 | 21726098/99 + 21726342 | §138 | r56 all 0.923; not band-edge noise |
| GO A area / factored | 21725471 / 72 | §136 / §137 | Drop factored head |
| Census + cross-fit (zero GPU) | — | §147 | 0/343 full-width cut points with val Δ > 0 under P; r20-w2 8/18 |
| group-token train (legacy val) | 21716380 | — | **scancelled** 30 Sep 11:29 (Ido GO); bundle kept in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` |

## Held

| Job | Why |
|---|---|
| 20412… / 20715… | Old FLOP-70 heuristics. Nice 1000+. Spent. |

## Blocked on Ido

| Item | Recommendation |
|---|---|
| N8 catalog | **keep design A**: 8 C10 + 8 C100; SVHN, Fashion-MNIST and ImageNet held out; plus additions 1–3 (roadmap §2b) |
| G5: the N8 GO | **pre-register now as a conditional GO** (roadmap §3): the science sitting launches on M1 + smoke + catalog A + a free slot; anything marginal comes back to Ido |
| Diverse train in parallel after M2 | only if speed matters more than the risk (roadmap §3) |
| A second resume of Pri 1 | only if the resume's own fuse (~12 Oct) comes before the governor stops the train |
