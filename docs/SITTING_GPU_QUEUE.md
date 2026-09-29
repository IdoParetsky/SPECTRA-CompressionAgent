# SPECTRA sitting GPU queue

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger — do not invent cells. NEXT lines below may be submitted by ops **only when their condition is met** and a GPU would otherwise idle.
**Rule (Ido 29 Sep 15:49):** QOS **4** stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells.
**Still Ido GO:** DRL train; release `21716380`; emit diverse catalog; TEST PPO-8 / Budget / GT `ep0011`.

Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12. Ranked options, kill/adopt rules and the A/B ladder: **§13**. Exact submit lines: `docs/PROMPT_OPS_V8_QUEUE.md` §8.

**Stamped:** 29 Sep 2026, ~17:45 IDT (Opus 5.5 sitting). 4 R, 13 PD. Trees: `tree_v9b` frozen (wave 1), `tree_v9c` = v9b + state_dict saves / scratch / final FT from saved (CPU pytest **367/367**), frozen now.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c` only). **Paired read** = `scripts/paired_steps.py <arm run> <control run>`: val only, same step = same widths under mild.

## Live rank

| Pri | Cell | Job | State | Tree | Checks | Hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | smoke-save | 21730498 | PD (next GPU) | v9c | `.pt` = state_dict + `.json` arch; scratch lines; no PicklingError | wave 2 unblocked | any Traceback → fix on a new tree, children wait | `traj_models/*.pt`+`.json`, `init=scratch` lines |
| 2 | smoke-from | 21730499 | PD afterok 498 | v9c | final FT from the saved walk, no new walk | fan-out FT variants cost no walk | no `final_ft from` line / wrong labels | loads `val_best` + size points, prints origin |
| 3 | final_ft DG VGG-19 C100 | 21729551 | R cs-1080-05 | v9b | honest gain of 100-ep SGD at val_best / size 0.70 / 0.60 | close part of −7.9 vs DepGraph −3.11 | honest < 0.5 pp (and one C10 cell < 0.5) | honest ≥ 2 pp → tables use final_ft |
| 4 | P gate C100 12/4 | 21729552 | R cs-1080-05 | v9b | 8 C100 nets under clean val at the live recipe (Q4) | ≥ 4/8 admitted: one recipe already holds | ≤ 2/8 → aug gate decides | ≥ 4/8 → Q4 evidence "yes" (no emit) |
| 5 | aug gate C100 12/4 | 21729554 | R cs-1080-01 | v9b | crop+flip in the walk FT vs Pri 4, same steps | **early read: r20-w13 +4.60 pp val over 16 cuts, 94 % better (`ADOPT?`)** | paired mean ≤ −1 pp, ≥ 75 % worse | admits ≥ Pri 4 and kinder TEST at equal keep on ≥ 5/8 |
| 6 | aug twins 40/10 | 21729553 | R cs-1080-05 | v9b | aug as the TEST walk recipe vs P twins 21726337 | R56 / VGG-16 cuts reach val Δ ≥ 0 (census 0/343 today) | paired ≤ −1 pp over 25 % of cuts | TEST ≥ 1 pp kinder at equal keep on ≥ 2/3 twins |
| 7 | final_ft DG R56 | 21730500 | PD afterok 498 | v9c | honest gain at DepGraph's 2.11× / 2.57× (flop 0.47 / 0.39) + saves | 10k −3.32 / −3.98 → within ~1–2 pp of +0.24 / +0.11 | honest < 0.5 pp | ≥ 2 pp → bar-3 row = final_ft (PruningBench-style caption) |
| 8 | P thin 12/4 | 21729555 | PD | v9b | P reference for the aug train-recipe pair rule | — (control) | — | — |
| 9 | aug thin 12/4 | 21729556 | PD | v9b | aug as the **training** recipe: thin C10 control vs Pri 8 | r56-w4 no worse, r20 kinder | r56-w4 > 0.5 pp worse at equal keep (r20 guard 2 pp) | pass + Pri 5 pass → recipe candidate for Q4 |
| 10 | final_ft P thin | 21730501 | PD afterok 498 | v9c | fastest C10 honest-gain cell; saves for scratch | r56-w4 size 0.80 −7.4 (10k) recovers | honest < 0.5 pp | ≥ 2 pp |
| 11 | aug thin 40/10 | 21729557 | PD | v9b | aug at TEST FT on the skinny pair vs P thin 21726335 | r56-w4 cliff moves deeper than 0.739 in band | paired ≤ −1 pp over 25 % of cuts | TEST kinder ≥ 1 pp at equal keep on r56-w4 |
| 12 | final_ft twins C10 | 21730506 | PD afterok 498 | v9c | VGG-16 (OCS / HRank cell) + R56 twin, same walk as 21726337 | VGG-16 −2.8 → ≤ −1 at 0.66 | honest < 0.5 pp | ≥ 2 pp |
| 13 | scratch-B thin | 21730507 | PD afterok 501 | v9c | Liu'19: re-init the walk's saved architectures, 200 ep SGD 0.1, + origin scratch | network-level "regenerate" matches inheritance | scratch < inherit − 1 pp on both nets | scratch ≥ inherit − 0.5 pp → item-1 answer + architecture metric |
| 14 | C-G NEON-rule twins | 21730509 | PD afterok 498 | v9c | NEON-literal redraw under clean val **and** NEON's train-loss stop (p10, cap 100) | first real in-band C-G cut on full R56 | 5 paired cuts: mean ≤ −3 pp, ≥ 4/5 worse → **scancel**, item 1 closed | ≥ A at equal keep on ≥ 2/3 nets |
| 15 | C-G NEON-rule thin | 21730514 | PD afterok 498 | v9c | same on the skinny pair vs P thin | same | same | same |
| 16 | scratch-B DG R56 | 21730516 | PD afterok 500 | v9c | scratch at DepGraph's size points + origin scratch | scratch ≈ final_ft → bar-3 scratch column | scratch < inherit − 1 pp | scratch ≥ inherit − 0.5 pp |
| 17 | N2 streams P | 21729558 | PD | v9b | block internals only, 3 passes, vs P thin **by params** | deeper in-band r56-w4 | no deeper in-band r56-w4 and r20 > 0.5 pp worse at equal keep | deeper in band and TEST no worse |

## NEXT (conditional; exact lines in ops §8)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| N1 | final-FT KD from saved, DG R56 | 21730500 COMPLETED with honest gain ≥ 0.5 pp | KD from the unpruned net on top of 100-ep SGD | ≤ +0.3 pp over plain final_ft | ≥ +0.5 pp |
| N2 | final-FT AutoAugment from saved, DG R56 | same | AutoAugment (CIFAR policy) in the final FT | ≤ +0.3 pp | ≥ +0.5 pp |
| N3 | aug walk + final FT, DG R56 | Pri 6 TEST adopt | better walk recipe under the bar-3 row | walk kinder but final_ft equal | final_ft ≥ 1 pp kinder |
| N4 | aug walk + final FT, DG VGG-19 | Pri 5 or 6 adopt | same on the C100 bar-3 cell | same | same |
| N5 | F2 group-first 12/4 under P | Pri 9 not adopt | skinny-group recovery without aug | same keep as Pri 8 within noise, no kinder r20 | r56-w4 deeper in band |
| N6 | F1 cosine 12/4 under P | Pri 9 not adopt | schedule only | same | same |
| N7 | SGD 0.01 + aug gate 12/4 | Pri 5 adopt | does aug rescue SGD (legacy SGD failed without aug) | admits ≤ Pri 5 | admits > Pri 5, thin pair passes |

## Done (walk TESTs already ledgered — do not re-run)

| Cell | Job | Ledger | One-line result (5k TEST half; 10k = cross-fit / both halves, zero GPU) |
|---|---|---|---|
| smoke-ft (v9b) | 21729550 | never | final_ft path prints with SAVE unset; plumbing only |
| P twins | 21726337 | §142 | R56 −2.8 @ 0.661 (10k −3.03); VGG-16 −2.8 @ 0.657 (10k −2.78); VGG-19 −6.7 @ 0.657 (10k −6.55) |
| P thin | 21726335 | §143 | r20 −3.7 @ 0.536 (10k −2.57); r56-w4 −10.1 @ 0.739 (10k −9.86) |
| P N4 | 21726338 | §144 | first undo 77 not 39; cross-fit invalid (rollback reads val) |
| P canary | 21726336 | §140 | admitted under P; 10k −7.86 @ 0.659 |
| DepGraph VGG-19 P | 21726341 | §145 | −7.9 @ 0.534 (10k −7.55); size 0.70 10k −6.09 |
| DepGraph R56 P | 21726340 | §146 | −4.0 @ 0.356 (10k −4.02); flop 0.47 10k −3.32; flop 0.39 10k −3.98 vs +0.11 |
| N0 3-seed b256 | 21726098/99 + 21726342 | §138 | r56 all 0.923; not band-edge noise |
| GO A area / factored | 21725471 / 72 | §136 / §137 | Drop factored head |
| Census (zero GPU) | — | §147 (ops) | 0/343 full-width cut points with val Δ > 0 under P; r20-w2 8/18 |

## Held (do not release)

| Job | Why |
|---|---|
| 21716380 | Group-token. Ido after 1 Oct. Requeue deletes `train_resume.pt`. |
| 20412… / 20715… | Old FLOP-70 heuristics. Nice 1000+. Spent. |

## Blocked on Ido

| Cell | Why |
|---|---|
| Next DRL train: P-val reward (one change), + crop+flip if Pri 5 / 9 adopt | the reward read memorized val in every train so far (§141) |
| Catalog emit | Q4 evidence from Pri 4 / 5 first; never from the gate alone |
