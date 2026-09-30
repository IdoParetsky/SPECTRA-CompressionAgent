# SPECTRA sitting GPU queue

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger — do not invent cells. NEXT lines below may be submitted by ops **only when their condition is met** and a GPU would otherwise idle.
**Rule (Ido 29 Sep 15:49):** QOS **4** stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells.
**Still Ido GO:** a second DRL train; freeze TESTs; the diverse catalog emit; `21716380`.

Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12. Options, decisions and dev items: **`docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`**. What was built and run: `docs/RUN_RECORD_29SEP_V9C.md`. Exact lines, greps and kill rules: `docs/PROMPT_OPS_V8_QUEUE.md` §8 (cells) and **§9** (train + handoff).

**Stamped:** 30 Sep 2026, ~03:55 IDT (Opus 5.5 sitting, handoff to ops). Cap 4: **4 R** (train 21737123, 21730500, gates 21729552 / 54), 12 PD, 21716380 held. Trees `tree_v9b` / `tree_v9c` frozen. Pending no-agent cells carry `Features=rtx_6000|rtx_4090` (untyped requests landed on 1080s by node weight).

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c`). **Paired read** = `scripts/paired_steps.py <arm> <control>`: val only, same step = same widths under mild.

**Honest-gain reader (fixed 30 Sep 03:20).** Run `/home/paretsky/scratch_audit/readers_s30/scripts/final_ft_readout.py`, not the frozen trees' copy. When the recipe costs the unpruned origin more than 0.5 pp, subtracting that change inflates honest gain. The fixed reader prints `ORIGIN-HURT` there instead of `ADOPT`. The smoke-from showed it: a 1-epoch FT, origin −5.84 pp, "honest +5.78 ADOPT" on a raw gain of −0.06. §149 is unaffected (origin +0.10). The fix is in git for `tree_v9d`; 13/13 reader tests pass on the cluster conda.

## Live rank

| Pri | Cell | Job | State (30 Sep 03:00) | Tree | Checks | Read so far / hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | **Stage-4 train**: area train under P + crop+flip (§151) | **21737123** (P-only arm 21737095 cancelled, never started) | **R since 03:14**, `ise-cpu256-32` RTX 6000 Ada; start checks green 03:16; ~2.3× the control's s/episode, so the 6-day fuse lands near episode ~160 (way-ahead §1f) | v9c | does the agent leave mild when the reward reads clean val and recovery uses crop+flip? | control 21536396 PPO u10 ev 0.88, best area 0.055; first freeze ep0023 | no freeze by episode 250; mild clone at the freeze TEST (0.9 on ≥ 95 % of legal r56-w4 rows) → report, never scancel | frozen agent ≥ mild at equal keep on the thin pair under P + aug (control 21729557), then the coverage set |
| 2 | smoke-from | 21730499 | **COMPLETED 03:14, passed** (2 min) | v9c | final FT from the saved walk, no new walk | smoke-save **passed** (00:34) | no `final_ft from` line / wrong labels | loads `val_best` + size points, prints origin; **found the ORIGIN-HURT reader flaw** (see below) |
| 3 | aug thin 12/4 (training rule) | 21729556 | **COMPLETED 03:12 (§150)** | v9b | crop+flip as the **training** recipe vs P thin 12/4 21729555 | **PASSED**: r56-w4 +2.3 pp TEST at equal keep (0.795), `val_best` −5.1 @ 0.622 vs −10.6 @ 0.741; r20 guard −1.8 mean (one point −3.1) | — | crop+flip = Stage-4 train recipe |
| 4 | aug gate C100 12/4 | 21729554 | R, net 7/8, wall ~08:29 | v9b | crop+flip vs P gate, same steps | **rule met (§148)**: 12/12 size points kinder, mean +3.3 pp TEST; 6/6 admit | — | done; extend §148 with nets 7–8 |
| 5 | P gate C100 12/4 | 21729552 | R, net 7/8, wall ~08:21 | v9b | 8 C100 nets under clean val at the live recipe (Q4) | **passed: 6/6 admitted** | — | Q4 evidence "yes" (no emit) |
| 6 | aug twins 40/10 | 21729553 | **scancelled 03:49 after its R56 rows** (pre-registered) | v9b | aug as the TEST walk recipe vs P twins 21726337 | **R56 TEST at equal keep (§152): size 0.80 −0.16 vs −2.68, size 0.70 −0.50 vs −2.70, `val_best` −0.06 vs −2.84 @ 0.661.** 1 of 3 twins meets the rule | paired ≤ −1 pp over 25 % of cuts | TEST ≥ 1 pp kinder at equal keep on ≥ 2/3 twins (with 6b) |
| 6b | aug twins, VGG only | 21737104 | PD, nice 25 | v9c | VGG-16 C10 + VGG-19 C100 twins with aug, saved | completes Pri 6's 2/3 rule | as Pri 6 | as Pri 6 |
| 7 | final_ft DG R56 | 21730500 | **R since 03:49**, `ise-4090-18` | v9c | honest gain at DepGraph's 2.11× / 2.57× (flop 0.47 / 0.39) + saves | 10k walk −3.32 / −3.98 → within ~1–2 pp of +0.24 / +0.11 | honest < 0.5 pp | ≥ 2 pp → bar-3 row = final_ft |
| 8 | aug thin 40/10 | 21729557 | PD, nice 30 | v9b | aug at TEST FT on the skinny pair vs P thin 21726335 | the thin guard for Pri 6; r56-w4 cliff deeper than 0.739 | paired ≤ −1 pp over 25 % of cuts | TEST kinder ≥ 1 pp at equal keep on r56-w4 |
| 9 | final_ft P thin | 21730501 | PD, nice 20 | v9c | fastest C10 honest-gain cell; saves for scratch | r56-w4 size 0.80 −7.4 (10k) recovers | honest < 0.5 pp | ≥ 2 pp |
| 10 | final_ft twins C10 | 21730506 | PD, nice 40 | v9c | VGG-16 (OCS / HRank cell) + R56 twin, same walk as 21726337 | VGG-16 −2.8 → ≤ −1 at 0.66 | honest < 0.5 pp | ≥ 2 pp |
| 11 | **N4** aug walk + final FT, DG VGG-19 | 21737105 | PD, nice 45 | v9c | does a kinder walk survive the 100-ep final FT? vs 21729551 | 21729551 final −2.52 @ 0.684 | final_ft within 0.5 pp of 21729551 at equal keep | final_ft ≥ 1 pp kinder |
| 12 | scratch-B thin | 21730507 | PD afterok 501 | v9c | Liu'19: re-init the walk's saved architectures, 200 ep SGD 0.1, + origin scratch | network-level "regenerate" matches inheritance | scratch < inherit − 1 pp on both nets | scratch ≥ inherit − 0.5 pp |
| 13 | C-G NEON-rule twins | 21730509 | PD, nice 50 | v9c | NEON-literal redraw under clean val and NEON's train-loss stop (p10, cap 100) | first real in-band C-G cut on full R56 | 5 paired cuts: mean ≤ −3 pp, ≥ 4/5 worse → **scancel** | ≥ A at equal keep on ≥ 2/3 nets |
| 14 | C-G NEON-rule thin | 21730514 | PD, nice 52 | v9c | same on the skinny pair vs P thin | same | same | same |
| 15 | scratch-B DG R56 | 21730516 | PD afterok 500 | v9c | scratch at DepGraph's size points + origin scratch | scratch ≈ final_ft → bar-3 scratch column | scratch < inherit − 1 pp | scratch ≥ inherit − 0.5 pp |
| 16 | N2 streams P | 21729558 | PD, nice 80 | v9b | block internals only, 3 passes, vs P thin **by params** | deeper in-band r56-w4 | no deeper in-band r56-w4 and r20 > 0.5 pp worse at equal keep | deeper in band and TEST no worse |

**Pre-registered ops action: done by the sitting.** 21729553 printed its `resnet56` rows at ~03:45 and was scancelled at 03:49. Ledger **§152**. 21730500 took the slot on an RTX 4090.

## NEXT (conditional; exact lines in ops §8)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| N1 | final-FT KD from saved, DG R56 | 21730500 COMPLETED with honest gain ≥ 0.5 pp | KD from the unpruned net on top of 100-ep SGD (on `tree_v9c` the line must carry `SPECTRA_FT_KD=1`) | ≤ +0.3 pp over plain final_ft | ≥ +0.5 pp |
| N2 | final-FT AutoAugment from saved, DG R56 | same | AutoAugment (CIFAR policy) in the final FT | ≤ +0.3 pp | ≥ +0.5 pp |
| N3 | aug walk + final FT, DG R56 | Pri 6 + 6b TEST adopt | better walk recipe under the C10 bar-3 row | walk kinder but final_ft equal | final_ft ≥ 1 pp kinder |
| N4 | aug walk + final FT, DG VGG-19 | ~~Pri 5 or 6 adopt~~ met (§148) | — | — | **submitted: Pri 11** |
| N5 / N6 | F2 group-first / F1 cosine 12/4 under P | ~~only if §150 fails~~ §150 passed: not queued | skinny-group recovery without aug | — | — |
| N7 | SGD 0.01 + aug gate 12/4 | met, low value | does aug rescue SGD | admits ≤ Pri 4 | not queued |
| N8 | diverse P train (C10 + C100, admitted v7 catalog) | Ido GO (way-ahead §1b) | NEON's multi-dataset offline train | — | — |
| N9 | attribution train: P-only (the 21737095 line) | Pri 1 leaves mild | P vs P + aug in training | — | — |
| N10 | cubic (NEON) reward under the Pri 1 recipe | an aug census shows cuts with val Δ > 0 on full-width nets | the item-2 reward A/B | — | — |

## Done (ledgered; do not re-run)

| Cell | Job | Ledger | One-line result (5k TEST half; 10k = cross-fit / both halves) |
|---|---|---|---|
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

## Held

| Job | Why |
|---|---|
| 21716380 | Group-token, legacy val. Recommendation: scancel (bundle backed up, way-ahead §1a). Never release: requeue deletes `train_resume.pt`. |
| 20412… / 20715… | Old FLOP-70 heuristics. Nice 1000+. Spent. |

## Blocked on Ido

| Item | Recommendation (way-ahead §1) |
|---|---|
| `21716380` | scancel |
| C100 emit | GO after both gates finish (~08:30), from the aug gate 21729554 (the train's recipe) |
| Diverse train (N8) | after the Stage-4 freeze TEST shows it is not a mild clone, or in parallel if speed matters more |
| Freeze TESTs of Pri 1 | pre-authorize ops: first freeze after PPO update 20, then at most one a day |
| Crop+flip as the TEST walk recipe | decide on Pri 6 + 6b + 8 TEST rows; R56 (Pri 6) already meets it (§152) |
| Resume Pri 1 after the 6-day fuse (~10 Oct, ~episode 160) | only if a freeze TEST is ≥ mild at equal keep, or the probe area is still rising |
