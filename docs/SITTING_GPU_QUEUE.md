# SPECTRA sitting GPU queue

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger, and the pre-authorized actions in the runbook §10.3 — do not invent cells.
**Rule (Ido 29 Sep 15:49):** QOS **4** stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells.
**Pre-authorized (Ido 30 Sep 11:08):** freeze TESTs of the Stage-4 train (first after PPO update 20, then ≤ 1 a day); its resume past the 6-day fuse (chained: 21767188). **Done on that GO:** 21716380 scancelled; C100 catalog emitted (§148).
**Still Ido GO:** a DRL train (N8, N9, N10); a second resume; converting 21730506 once decision (d) is met.

Ops handoff, lines, greps and kill rules: **`docs/OPS_HANDOFF_RUNBOOK.md` §10** (current), §8 (cells). Options, decisions and dev items: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. N8: `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. What was built and run: `docs/RUN_RECORD_29SEP_V9C.md`. Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12.

**Stamped:** 30 Sep 2026, ~11:55 IDT (Opus 5.5 sitting, handoff to ops). Cap 4: **4 R** (train 21737123, N3 21767189, 21730501, 21737104), 10 PD (the resume on its dependency). **Decision (d) met 11:55** (§152); the 21730506 conversion waits for Ido. Trees `tree_v9b` / `tree_v9c` frozen. Pending cells carry `Features=rtx_6000|rtx_4090`. Both train jobs `Requeue=0`.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c`). **Paired read** = `readers_s30/scripts/paired_steps.py <arm> <control>`: val only, same step = same widths under mild. **Honest gain**: only `readers_s30/scripts/final_ft_readout.py` (prints `ORIGIN-HURT` when the origin loses > 0.5 pp).

## Live rank

| Pri | Cell | Job | State (30 Sep 11:50) | Tree | Checks | Read so far / hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | **Stage-4 train**: area train under P + crop+flip (§151) | **21737123** → resume **21767188** | R since 03:14 (`ise-cpu256-32`, RTX 6000 Ada); 12 episodes / 3 PPO updates by 11:50, mean 2,315 s per episode; fuse ~6 Oct 03:15 near episode ~200; resume PD `afterok` | v9c | does the agent leave mild when the reward reads clean val and recovery uses crop+flip? | ev ~0 at updates 1–3 (control 0.45–0.88): watch M2 at update 10 | no freeze by episode 250; mild clone at the freeze TEST → report, never scancel | runbook M1 |
| 1b | Freeze TESTs of Pri 1 | — | none yet; first after PPO update 20 (~2 Oct) | v9c | thin pair, P + crop+flip, 40/10, vs 21729557 (final, §152) at equal keep + census | — | mild clone (0.9 on ≥ 95 % of legal r56-w4 rows) | M1 |
| 2 | **N3** aug walk + final FT, DG R56 | **21767189** | **R** since ~11:50, `ise-4090-19` (21729557's slot) | v9c | 21730500's line + `SPECTRA_FT_AUG=1`; pairs with §153 | close part of the ~2 pp gap to DepGraph (10k −1.52 / −2.11) | walk kinder but final FT within 0.5 pp | final FT ≥ 1 pp kinder at equal keep |
| 4 | aug twins, VGG | 21737104 | R, `cs-4090-07`; VGG-16 done (§152), VGG-19 C100 walking | v9c | VGG twins with aug vs 21726337, saved | **VGG-16 +2.3 to +2.6 pp at equal keep**: twins 2/3 met | — | extend §152 |
| 5 | final_ft P thin | 21730501 | R, `ise-4090-21`; r20 final rows in | v9c | fastest C10 honest-gain cell; saves for scratch | r20 **cross-off** so far: origin +3.5, pruned points −0.3 to −1.1 raw | honest < 0.5 pp | ≥ 2 pp |
| 6 | **N4** aug walk + final FT, DG VGG-19 | 21737105 | PD, nice 45 | v9c | does a kinder walk survive the 100-ep final FT? vs 21729551 | 21729551 final −2.52 @ 0.684 | final FT within 0.5 pp of 21729551 at equal keep | final FT ≥ 1 pp kinder |
| 7 | scratch-B thin | 21730507 | PD afterok 501, nice 45 | v9c | Liu'19: re-init the walk's saved architectures, 200 ep SGD 0.1, + origin scratch | network-level "regenerate" matches inheritance | scratch < inherit − 1 pp on both nets | scratch ≥ inherit − 0.5 pp |
| 8 | C-G NEON-rule twins | 21730509 | PD, nice 50 | v9c | NEON-literal redraw under clean val and NEON's train-loss stop (p10, cap 100) | first real in-band C-G cut on full R56 | 5 paired cuts: mean ≤ −3 pp, ≥ 4/5 worse → **scancel** | ≥ A at equal keep on ≥ 2/3 nets |
| 9 | C-G NEON-rule thin | 21730514 | PD, nice 52 | v9c | same on the skinny pair vs P thin | same | same | same |
| 10 | scratch-B DG R56 | 21730516 | PD, nice 55 | v9c | scratch at DepGraph's size points + origin scratch | scratch ≈ final FT → bar-3 scratch column | scratch < inherit − 1 pp | scratch ≥ inherit − 0.5 pp |
| 11 | **N1** final-FT KD from saved, DG R56 | **21767190** | PD, nice 60 | v9c | KD from the unpruned net on top of 100-ep SGD (`SPECTRA_FT_KD=1` on the line) vs 21730500's `final_ft` rows | — | ≤ +0.3 pp over plain final FT | ≥ +0.5 pp, healthy origin → M5 |
| 12 | **N2** final-FT AutoAugment from saved, DG R56 | **21767192** | PD, nice 61 | v9c | AutoAugment (CIFAR policy) in the final FT, same saves | — | ≤ +0.3 pp | ≥ +0.5 pp → M5 |
| 13 | final_ft twins C10 (no-aug walk) | 21730506 | PD, **parked at nice 70**; (d) met 11:55 | v9c | VGG-16 (OCS / HRank cell) + R56 twin, same walk as 21726337 | **on Ido's reply**: convert to the aug line (runbook §10.5 c) | honest < 0.5 pp | ≥ 2 pp |
| 14 | N2 streams P | 21729558 | PD, nice 80 | v9b | block internals only, 3 passes, vs P thin **by params** | deeper in-band r56-w4 | no deeper in-band r56-w4 and r20 > 0.5 pp worse at equal keep | deeper in band and TEST no worse |

## NEXT (conditional)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| (d) | aug twins + final FT, C10 (replaces 21730506) | (d) **met 11:55**; waits for Ido's reply | runbook §10.5 (c); re-walk ≈ 0 vs 21737104 (VGG-16) / 21729553 (R56) | honest < 0.5 pp | ≥ 2 pp; the bar-3 VGG-16 / R56 rows |
| N5 / N6 | F2 group-first / F1 cosine 12/4 under P | ~~only if §150 fails~~ §150 passed: not queued | skinny-group recovery without aug | — | — |
| N7 | SGD 0.01 + aug gate 12/4 | met, low value | does aug rescue SGD | admits ≤ Pri 4 | not queued |
| N8 | diverse P train (8 C10 + 8 C100, emitted catalog) | runbook M1 + `tree_v9d` + Ido GO (`docs/N8_DIVERSE_TRAIN_ROADMAP.md`) | NEON's multi-dataset offline train; then frozen → ImageNet | roadmap §4 | roadmap §4 |
| N9 | attribution train: P-only (the 21737095 line) | Pri 1 leaves mild | P vs P + aug in training | — | — |
| N10 | cubic (NEON) reward under the Pri 1 recipe | an aug census shows cuts with val Δ > 0 on full-width nets (M6) | the item-2 reward A/B | — | — |

## Done (ledgered; do not re-run)

| Cell | Job | Ledger | One-line result (5k TEST half; 10k = cross-fit / both halves) |
|---|---|---|---|
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

| Item | Recommendation (way-ahead §1) |
|---|---|
| Convert 21730506 to the aug line | **yes, now**: (d) met 11:55 (§152). Runbook §10.5 (c) |
| Diverse train (N8) | after runbook M1 and `tree_v9d` (roadmap §3); in parallel after M2 only if speed matters more than the risk |
| A second resume of Pri 1 | only if the resume's own fuse (~12 Oct) comes before the governor stops the train |
