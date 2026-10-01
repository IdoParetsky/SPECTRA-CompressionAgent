# SPECTRA sitting GPU queue

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger, and the pre-authorized actions in the runbook §10.3 — do not invent cells.
**Rule (Ido 29 Sep 15:49):** QOS stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells. Cap is **live `gpu-part` MaxTRESPU** (8 as of 1 Oct 00:09). Do not invent cells when the ladder is empty.
**Pre-authorized (Ido 30 Sep 11:08):** freeze TESTs of the Stage-4 train (first after PPO update 20, then ≤ 1 a day); its resume past the 6-day fuse (chained: 21767188). **Done on that GO:** 21716380 scancelled; C100 catalog emitted (§148). **Done on Ido's 12:34 GO:** 21730506 converted to the crop+flip walk → **21809595**.
**Done on Ido's 1 Oct 01:03 GO:** the two one-change reward trains (C1 **21938807**, C2 **21938810**; resumes chained).
**Still Ido GO:** a DRL train (N8, N9; for N8 see the conditional-GO proposal, roadmap §3 G5); a second resume.

Ops handoff, lines, greps and kill rules: **`docs/OPS_HANDOFF_RUNBOOK.md` §10** (current), §8 (cells). Options, decisions and dev items: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. N8: `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. What was built and run: `docs/RUN_RECORD_29SEP_V9C.md`. Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12.

**Stamped:** 1 Oct 2026, ~03:15 IDT (G2 sitting). **QOS `gpu-part` `gres/gpu=8`, full: 8 R** (train **21737123** + 7 G2 cells). **6 PD** take freed slots in nice order (reward trains 50 / 51, v9 smoke 52, random 60 / 70 / 71), plus 3 `afterok` resumes. Every new job: `tree_v9d`, `Features=rtx_6000|rtx_4090`, `--gpus=1`. Do **not** launch N8.
- *G2 sitting (charge 1 Oct 01:03).* **A1** hold-out checkpoints R. **A2** greedy and random R. **A3** Adam 1e-4 thin, SGD 0.01 thin, Adam 1e-4 C100 COMPLETED (§166–§168); SGD 0.01 C100 R. **A4** C-G+ KILL (§163). **A5** F1 / F2 R. **B** O38 replay below. **C** two one-change reward trains PD. **D** `tree_v9d` built and tested; `offline_train_v9_diverse` + catalog A deployed; smoke PD.
- *Overnight COMPLETED.* N3 **21767189** §157 **M4**; scratch-thin **21730507** §158 CROSS-OFF; scratch-DG **21730516** §159 ADOPT; N1 **21767190** §160 mixed; N2 **21767192** §161 not M5; streams **21729558** §162 split.
- *Ledger.* Next **§169**. C-G+ KILL §163; twins FT §164; VGG-16 10-pass §165; Adam 1e-4 thin §166 CROSS-OFF; SGD thin §167 CROSS-OFF; Adam 1e-4 C100 t2 §168 kinder, not a train switch.
- *Trees.* `tree_v9b` / `tree_v9c` frozen. **`tree_v9d`** = v9c + the G2 dev pass, default-off for every existing profile (`PROVENANCE_v9d.txt`). Train `Requeue=0`.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c`). **Paired read** = `readers_s30/scripts/paired_steps.py <arm> <control>`: val only, same step = same widths under mild. **Honest gain**: only `readers_s30/scripts/final_ft_readout.py` (prints `ORIGIN-HURT` when the origin loses > 0.5 pp).

## Live rank

Equal keep = the size points (`param:0.8,0.6` thin; `0.9,0.8` C100) and `val_best`, read on the 5k TEST half. Greedy and random cut differently per step, so `paired_steps.py` labels are valid only between two mild walks.

| Pri | Cell | Job | State (1 Oct 03:15) | Tree | Checks | Read so far / hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | **Stage-4 train**: area train under P + crop+flip (§151) | **21737123** → resume **21767188** | R since 30 Sep 03:14 (`ise-cpu256-32`); **29** eps; PPO **7** at 01:14: ev **0.592**, batch_score 0.540, probe best 0.282; freeze still **ep0011** only; fuse ~6 Oct 03:15 | v9c | leave mild under P + crop+flip? | M2 at update 10 (~1 Oct midday) | no freeze by 250; mild clone at TEST | M1 |
| 1b | Freeze TESTs of Pri 1 | — | none yet; first freeze after PPO update 20 | v9c | vs 21729557 + census | — | mild clone | M1 |
| 2 | **C1 cubic-gain train**: the Stage-4 line, reward scale only → `cbrt_miss` (gain +ρ³, band +ρ, miss −ρ) | **21938807** → r1 **21938809** | PD nice 50; next free slot | v9d | FLAGS `SPECTRA_REWARD_SCALE_ARM=cbrt_miss`; `PPO training: … scale=cbrt_miss`; `Requeue=0` | O38: at 12/4 the gain arm is rare (3 of 76 thin cuts, 0 on C100), so C1 may train close to live | **report, never scancel**: ev ≤ 0 by update 10, or the freeze is a ≥ 90 % mild clone | first freeze TEST beats the Stage-4 freeze at equal keep |
| 3 | **C2 NEON-raw train**: scale → `raw` (+ρ³ / +ρ / −ρ³ on the realised cut) | **21938810** → r1 **21938811** | PD nice 51 | v9d | `scale=raw` | O38: = C1 on every walk without a miss; differs only on the miss arm | same | same |
| 4 | A1 hold-out checkpoints, SVHN: ShuffleNetV2 ×1, RepVGG-A0, MBV2 ×0.5, DN-40, 200 ep | **21938295** | R (`ise-4090-15`); epochs 109–162 at 02:50, test acc 96.1–96.5 % | v9d | `runs/g2_holdout/svhn/*.log`; writes `configs/input_g2_holdout_svhn.json` | done ~04:15 | a net < 90 % → retrain that net | hold-out set (roadmap §2b); never in a train catalog |
| 5 | A1, Fashion-MNIST | **21938296** | R (`cs-4090-07`); epochs 144–190, test acc 94.5–95.0 % | v9d | `…/fmnist/*.log`; `input_g2_holdout_fmnist.json` | done ~03:20 | same | same |
| 6 | A2 **greedy** (profile `l1`: Ido's "L1" = Gilad's "greedy"), thin, P + crop+flip 40/10 | **21938279** | R (`cs-4090-08`); r20 done, r56-w4 in walk | v9d | vs mild **21729557** at equal keep | r20: **−4.0 @ 0.595** vs mild −4.9 @ 0.584; −5.1 @ 0.702 (mild −1.2 @ 0.774); reaches **−8.2 @ 0.417** in band (val −6.96; mild's 2 passes stop at 0.536) | worse on both nets at equal keep | kinder, or deeper in band at equal keep |
| 7 | A2b **random**, r56-w4, seed 42 | **21938285** | R since 02:33 | v9d | vs 21729557 r56-w4 | 4 cuts in | — (baseline row) | — |
| 8 | A3 SGD 0.01, C100 tight-2 (r20-w13, r56-w9), 12/4 | **21938284** | R; r20-w13 done | v9d | vs **21729554** | r20-w13: **−7.2 @ 0.861** vs −5.2; **−8.4 @ 0.787** vs −6.9 | SGD thin failed (§167): not a train FT either way | — |
| 9 | A5 F1 cosine, thin 12/4 | **21938286** | R; r20 done | v9d | vs **21729556** | r20: +0.4 @ 0.774, **−4.2 @ 0.584**, −2.6 @ `val_best` 0.536 | > 0.5 pp worse on r20 already: CROSS-OFF unless r56-w4 is clearly kinder | within 0.5 pp on r20 and kinder on r56-w4 |
| 10 | A5 F2 group-first 4, thin 12/4 | **21938287** | R since 02:49 | v9d | vs 21729556 | r20, 4 cuts: val −1.37 mean, 0 of 4 better | same | same |
| 11 | D **smoke** `offline_train_v9_diverse`, 2 episodes | **21938898** | PD nice 52 | v9d | seed 50 → episodes on r20-w13 C100 and MBV2 ×1 C100. Grep `PPO training: networks=16`, `probe_nets=` (3), `Val from test on cifar-100`, `FT aug on cifar-100`, step counts, `Stopping PPO training after 2 episodes`, no `Traceback` | plumbing only; **never ledger** | any fail → fix before N8 | N8 launchable on G5 |
| 12 | A2b random, r20-w2, seed 42 | **21938894** | PD nice 60 | v9d | vs 21729557 r20 | — | — | — |
| 13 | A2b random, second draws (seed 43), r56-w4 / r20-w2 | **21938895** / **21938896** | PD nice 70 / 71 | v9d | random row = mean of the draws, never the best draw | — | — | — |

## NEXT (conditional)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| N8 | diverse P train (8 C10 + 8 C100, catalog A): profile **`offline_train_v9_diverse`** in `tree_v9d` | roadmap §3: G1 (M1) + G2 (**met** once smoke 21938898 passes) + G5 | NEON's multi-dataset offline train; then frozen → ImageNet, SVHN, Fashion-MNIST | roadmap §4 | roadmap §4 |
| H0 | mild walks on the G2 hold-out checkpoints | A1 COMPLETED **and** a loader check: P (val from test) and crop+flip on SVHN / Fashion-MNIST (flip is not label-safe on digits) | the hold-out bar for N8 H5 / H7 | — | sitting GO; not in the 1 Oct charge |
| LA | accuracy look-ahead heuristic | today "look-ahead" = the floor guard, off under TRAJ, so it equals greedy | a one-step look-ahead costs ~3× FT per step | — | design question for Gilad |
| D5 | GPU-side crop+flip (roadmap §5 item 5) | before the N8 launch | up to +18 % per epoch | never into a live train | — |
| N8b | N8 + SVHN nets in training (pre-registered, roadmap §2b) | N8 passes on CIFAR (H2, H3) but is below mild on the dataset hold-outs (H5, H7) | read on Fashion-MNIST, ImageNet and the unlike families | — | — |
| N9 | attribution train: P-only (the 21737095 line) | Pri 1 leaves mild | P vs P + aug in training | — | — |
| O26 | `scripts/memorization_census.py` | zero GPU | — | — | not built yet |

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
| 6 tests + smoke | pytest done; smoke **21938898** PD |
| 8 hold-out checkpoints | A1 R; the job writes the input files; the disjointness test covers them |
| 9 size points | `eval_size_match` docstring fixed (OCS ≈ 0.42 params is R56; VGG-16 matches on FLOPs) |
| reward | `cbrt_miss` scale; `SPECTRA_REWARD_SCALE_ARM` (in-band train profiles only); `scripts/reward_replay.py` + test |

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
