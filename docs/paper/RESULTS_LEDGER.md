# SPECTRA results ledger

**As of:** 21 Aug 2026 06:30 IDT. **8/8 + 52 PD.** No running orphans. 06:30 Ido status. Similar FLOP+prefer PRELIM unchanged: s43 MobileNet **−2.0 @ 0.767/0.912**; s44 r44 **−2.6 @ 0.702/0.872**. s43 still DenseNet **eval_test**; s44 now VGG-19 **eval_test** (r32 skipped); s42 still DenseNet **eval_train**. τ=5 s44 r56 **eval_train**. ImageNet **20360208** ~14 h, train-loader — **no TEST**. Look-ahead similar still **eval_train**. Do not quote train-loader. Afterok through 13:30 if VPN drops.  
**Backfill:** every important result since git `ecefe78` (8 Aug 2026, “Transfer to new PC”) through the 10-net leap. Later jobs only *extend* these tables.  
**Protocol (current defaults):** τ = 10 pp; eval floor 0.70 params kept; rates 1.0 / 0.9 / 0.8 unless noted; full-net FT 40 ep / patience 10 on C10; NEON reward; small Transformer encoder; Fortify on.  
**Quote TEST only.** Skip akamaster ResNet-32.  
**10-net train catalog:** `configs/database_offline_train.json` (C10 + SVHN + Fashion-MNIST; no C100; **no** r20-w2 / r56-w4).  
**Held-out C10:** similar `input_offline_similar.json`; unlike `input_offline_novel.json`; thin `input_c10_thin.json`.

### Quoting rules (do not regress)

- `param_ratio` / `flops_ratio` = fraction **kept** (rebuilt shapes, not masked zeros).
- `eval_test` vs `eval_train` in logs is the **CNN data loader** (CIFAR test vs train images), **not** “held-out architecture vs in-catalog.” On r56-w4 job 20189046, eval_train is −1.7 pp at 0.667 params while TEST is −15.9 pp at 0.704 — that is fine-tune looking healthy on train images, not “the agent trained on this net.”
- Do **not** quote overnight-matrix “Eval Δacc” or train-step “within −10 %” as paper TEST. Those mixes are in §10 and §13–15 for archaeology only.
- Do **not** describe early C100 failure as “C100 was missing from the 10-net train set.” Early tests were recovery probes and mixed-catalog RL. Frozen 10-net → C100 TEST is ledger §21 (claim C9). Recoverability (no agent) remains §7.
- A probe cell `within_budget=True` at ≥98% params is **not** a 2–5% cut.

---

## 1. Claims that the paper can already make

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | One offline agent prunes **similar-family** C10 nets (new width/source) inside τ=10, except skinny-deep ResNet-56. | LOCKED | §3 three seeds |
| C2 | Same agent prunes **unlike-family** C10 nets (ShuffleNet, RepVGG; never in train) inside τ=10. | LOCKED | §4 seeds 42 / 43 / 44 |
| C3 | Easy thin ResNet-20 w2 is a size-matched **tie** vs greedy (~−4 pp at 60% params). | LOCKED | §5 |
| C4 | Skinny-deep ResNet-56 w4 **misses** τ=10 at matched DRL size across **three** seeds. | LOCKED | §5 s42 −15.9, s43 −16.2, s44 −17.2 @ 0.704/0.550 |
| C5 | On hard ResNets, DRL beats greedy: similar r56-w10 ~10 pp at matched params; held-out r56-w4 ~9 pp vs look-ahead greedy that kept **more** params. | LOCKED | §6 r56-w10; §5 job 20213131 −24.9 @ 0.722/0.477 vs DRL −15.9 @ 0.704/0.550 |
| C6 | C10 is a recoverable FT environment. C100 is **not**, except VGG-11 BN under the 160-ep SGD recipe. | LOCKED | §7. Probe **20204214 COMPLETED**. Residuals/DenseNet/MobileNet are tiny cuts or val DROP. |
| C7 | Encoder capacity (BERT / wider / set) did not fix r56-w4. Catalog diversity **3-net → 10-net** moved it to −15.9; **10-net → 24-net did not** (C8 miss). | LOCKED | §16 encoder ~−24 pp; §17; C8 **−25.0 @ 0.704/0.499** |
| C8 | 24-net train catalog moves r56-w4 further. | **LOCKED miss** | **20201263 COMPLETED** 22:30. r56-w4 **−25.0 @ 0.704/0.499** vs 10-net s42 **−15.9 @ 0.704/0.550**. Worse, same params, fewer FLOPs. Easy r20-w2 **−5.7 @ 0.600/0.748**. |
| C9 | Frozen C10-trained agent on held-out **CIFAR-100**: VGG-16 BN and ShuffleNet-v2×1 inside τ (three seeds); thin ResNets and RepVGG-A0 miss. | LOCKED mixed | §21; ShuffleNet **−3.9 / −3.4 / −4.3** |
| C10 | Eval-only FLOP floor 0.70 puts held-out r56-w4 **inside τ=10** (same frozen 10-net actor). | LOCKED | §5 s42 **−8.9 @ 0.907/0.702**; s43 **−9.2 @ 0.926/0.703**; s44 **−9.6 @ 0.907/0.703**. Not the 0.704-param operating point of C4. |

---

## 2. Headline Pareto (C10 TEST)

NEON Figure 5 grammar (Gilad 18 Aug): plot **compression vs TEST Δacc** for DRL operating points **and** same-loop heuristics. Coverage matrix is a different artifact (family × dataset transfer). See `GILAD_DIRECTIVES_18AUG.md`. Maintain by appending TEST rows; do not rebuild from chat.

### 2.1 Easy thin ResNet-20 w2 · CIFAR-10 (held out)

| Job | Agent | Δacc (pp) | params / FLOPs | Status |
|---|---|---|---|---|
| 20066579 | 2-net C10-ResNet train, structural reward | −3.1 | 0.60 / 0.79 | LOCKED |
| 20189046 | 10-net s42 | −4.2 | 0.600 / 0.760 | LOCKED |
| 20189050 | 10-net s44 | −4.6 | 0.600 / 0.760 | LOCKED |

Do **not** quote overnight-matrix **−1.2 pp** for 20066579. That was not TEST. TEST is **−3.1**.

### 2.2 Hard thin ResNet-56 w4 · CIFAR-10 — first NEON-style frontier

Same frozen 10-net actors unless noted. One net, several stop rules. Not a fix of each other.

| Point | Jobs | TEST Δacc (s42 / s43 / s44) | params / FLOPs (typical) | vs τ=10 |
|---|---|---|---|---|
| DRL default (param floor 0.70) | 20189046 / 048 / 050 | **−15.9 / −16.2 / −17.2** | 0.704 / 0.550 | miss |
| 24-net DRL s42 (same param floor) | **20201263** | **−25.0** (s42) | 0.704 / 0.499 | miss — worse than 10-net |
| DRL FLOP floor 0.70 | 20238166 / 20238653 / 20238655 | **−8.9 / −9.2 / −9.6** | ~0.91 / 0.70 | inside |
| DRL prefer Δparams/ΔFLOPs under FLOP floor 0.70 | 20307291 / 20308033 / 20317562 | **−8.9 / −8.0 / −8.1** | 0.704 / 0.872 | inside |
| DRL param floor 0.80 (no FLOP floor) | 20289103 / 20289105 / 20317564 | **−21.7 / −25.7 / −22.5** | ~0.80 / 0.54 | miss |
| Look-ahead greedy | 20213131 | **−24.9** (s42) | 0.722 / 0.477 | miss |
| DRL + L2 ranking s42 / s43 / s44 | **20353533 / 20353573 / 20353577** | **−24.1 / −21.2 / −23.4** | unmatched sizes (s43 0.593) | miss — greedy cliff |
| Greedy L2 / SVD s42 | **20353569 / 20353571** | **−26.6 / −25.8** | 0.667 / 0.454 | miss — DRL L2 −24.1 @ 0.667/0.482 slightly better, still cliff |
| DRL + SVD ranking s42 / s43 / s44 | **20353536 / 20353575 / 20357696** | **−25.6 / −19.8 / −23.7** | unmatched (s43 0.593; s44 0.685) | miss — keep L1 |

Unmatched always-0.8 greedy / mild / random sit near **−24 @ 0.667** (not size-matched to DRL 0.704). Overlay literature ResNet-56 CIFAR-10 stars only with a different-FT caption; do not invent numbers here.

---

## 3. Similar-family TEST (C10, 10-net agent)

Same families as train, different width / depth / source. Skip r32.

| Net | s42 (20158274 eval) | s43 (20163257) | s44 (20164515) | Status |
|---|---|---|---|---|
| ResNet-20 w16 | −5.4 @ 0.603/0.639 | −4.5 @ 0.673/0.696 | −5.6 @ 0.669/0.634 | LOCKED |
| ResNet-56 w10 | −9.2 @ 0.658/0.488 | −12.4 @ 0.664/0.433 | −13.0 @ 0.604/0.397 | LOCKED (seed-sensitive miss) |
| ResNet-44 | −4.3 @ 0.632/0.519 | −4.2 @ 0.700/0.586 | −4.3 @ 0.635/0.582 | LOCKED |
| VGG-19 BN | −2.7 @ 0.879/0.882 | −2.6 @ 0.788/0.802 | −2.7 @ 0.814/0.796 | LOCKED |
| MobileNet-v2×0.75 | −2.5 @ 0.689/0.630 | −2.1 @ 0.697/0.651 | −1.6 @ 0.706/0.684 | LOCKED |
| DenseNet-100 | −2.0 @ 0.847/0.849 | −2.1 @ 0.805/0.833 | −2.1 @ 0.834/0.851 | LOCKED |

s43 similar job **20163257** COMPLETED 16 Aug (1d 23h 56m).

**Similar-family FLOP floor 0.70 — PRELIM** (s42 **20360209** / s43 **20360210** / s44 **20360211** COMPLETED). Skip r32. r56-w10: s42 **−5.9 @ 0.902/0.702** inside; s43 **−14.3 @ 0.946/0.700** still a miss at 95% params; s44 **−5.6 @ 0.911/0.702** inside. DenseNet **−2.2 / −2.4 / −2.2** @ 0.837/0.833, 0.822/0.827, 0.780/0.797. Operating point, not a three-seed rescue of the similar r56-w10 miss. Table §29.

**Similar-family FLOP 0.70 + prefer — PRELIM** (jobs still running). s43 **20381786**: r20 **−3.5 @ 0.717/0.880**; r56-w10 **−4.0 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872**; VGG-19 **−2.6 @ 0.837/0.923**; MobileNet **−2.0 @ 0.767/0.912**. s44 **20381787**: r20 **−3.8 @ 0.717/0.880**; r56-w10 **−4.4 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872** (same size as s43). Skip r32. Do not lock. Table §33.

---

## 4. Unlike-family TEST (C10, 10-net agent)

Families never in train. Same dataset (C10).

| Net | s42 (20189047) | s44 (20189051) | s43 (20189049) | Status |
|---|---|---|---|---|
| ShuffleNet-v2×1 | −1.1 @ 0.800/0.825 | −1.5 @ 0.814/0.831 | −1.3 @ 0.800/0.825 | LOCKED |
| ShuffleNet-v2×1.5 | −2.4 @ 0.818/0.801 | −2.1 @ 0.877/0.826 | −1.8 @ 0.818/0.801 | LOCKED |
| RepVGG-A0 | −4.8 @ 0.681/0.565 | −4.8 @ 0.681/0.565 | −4.6 @ 0.681/0.565 | LOCKED (same size s42/s43/s44) |
| RepVGG-A1 | −4.7 @ 0.650/0.521 | −4.3 @ 0.650/0.521 | −4.4 @ 0.650/0.521 | LOCKED (same size) |

**Caveat (ShuffleNet s44 log):** 28% of actions fell back to masking (`concatenation along a non-channel axis`, `getitem`). Quote TEST Δacc; do not claim every ShuffleNet layer was structurally resized. Same-loop greedy on ShuffleNet **crashed** (depthwise groups) — not a DRL failure. Same-loop greedy RepVGG: A0 −7.1 @ 0.654/0.484; A1 −6.4 @ 0.639/0.477 (job 20202686).

**Unlike FLOP floor 0.70 — LOCKED three-seed** (look-ahead on). Default unlike was already inside τ; this is a milder operating point, not a new transfer win. Full table §28.

**Unlike FLOP 0.70 + prefer Δparams/ΔFLOPs — LOCKED three-seed** (s42 **20381788** / s43 **20381798** / s44 **20381799**). Same size on all three seeds. Table §30.

| Net | s42 (20353537) | s43 (20353579) | s44 (20359431) |
|---|---|---|---|
| ShuffleNet-v2×1 | −1.9 @ 0.809/0.826 | **−1.8 @ 0.872/0.843** | **−1.5 @ 0.887/0.868** |
| ShuffleNet-v2×1.5 | −2.0 @ 0.821/0.799 | **−2.3 @ 0.836/0.801** | **−2.8 @ 0.856/0.846** |
| RepVGG-A0 | −3.9 @ 0.792/0.702 | **−4.6 @ 0.879/0.748** | **−4.6 @ 0.858/0.754** |
| RepVGG-A1 | −4.3 @ 0.888/0.735 | **−4.2 @ 0.887/0.735** | **−4.2 @ 0.850/0.738** |

Generic 5-family C10 agent **20140553** already had ShuffleNet-v2×1 TEST −1.4 @ 0.827/0.821 and VGG-19 −2.4 @ 0.800/0.796 — unlike-family transfer is not unique to the 10-net catalog.

---

## 5. C10-thin held-out (r20-w2 easy / r56-w4 hard)

| Job | Policy | r20-w2 TEST | r56-w4 TEST | Status |
|---|---|---|---|---|
| 20189046 | 10-net DRL s42 | −4.2 @ 0.600/0.760 | −15.9 @ 0.704/0.550 | LOCKED |
| 20189050 | 10-net DRL s44 | −4.6 @ 0.600/0.760 | −17.2 @ 0.704/0.550 | LOCKED |
| 20189048 | 10-net DRL s43 | −4.9 @ 0.600/0.760 | −16.2 @ 0.704/0.550 | LOCKED |
| 20213131 | Look-ahead greedy | −4.0 @ 0.600/0.753 | **−24.9 @ 0.722/0.477** | LOCKED |
| 20140552 | C10-thin-only DRL (3 ResNets) | −5.2 @ 0.600/0.760 | −23.8 @ 0.685/0.494 | LOCKED (catalog control) |
| 20189043 | Greedy always 0.8 (L1) | −4.3 @ 0.600/0.734 | −24.0 @ 0.667/0.454 | LOCKED (hard net **not** size-matched) |
| 20202687 | Greedy floor 0.71 | −4.4 @ 0.600/0.734 | −24.2 @ 0.667/0.454 | LOCKED (overshot to 0.667) |
| 20202690 / 689 | Greedy L2 / SVD | −4.8 / −7.2 @ 0.600 | −24.9 / −24.1 @ 0.667 | LOCKED |
| 20189044 / 045 | Mild 0.9 / random | −4.1 / −5.0 @ 0.600 | −24.1 / −24.1 @ 0.667 | LOCKED |
| **20238166** | **FLOP floor 0.70 + look-ahead, frozen s42** | **−2.1 @ 0.600/0.773** | **−8.9 @ 0.907/0.702** | **LOCKED** (s42) |
| **20238653** | **FLOP floor 0.70, frozen s43** | **−3.7 @ 0.600/0.763** | **−9.2 @ 0.926/0.703** | **LOCKED** (s43) |
| **20238655** | **FLOP floor 0.70, frozen s44** | **−4.0 @ 0.800/0.799** | **−9.6 @ 0.907/0.703** | **LOCKED** (s44; r20 is a larger net than s42/s43) |
| **20307291** | FLOP floor 0.70 + prefer Δparams/ΔFLOPs s42 | **−2.7 @ 0.600/0.886** | **−8.9 @ 0.704/0.872** | **LOCKED** (s42) |
| **20308033** | FLOP floor 0.70 + prefer Δparams/ΔFLOPs s43 | **−2.7 @ 0.600/0.886** | **−8.0 @ 0.704/0.872** | **LOCKED** (s43) |
| **20317562** | FLOP floor 0.70 + prefer Δparams/ΔFLOPs s44 | **−1.9 @ 0.600/0.886** | **−8.1 @ 0.704/0.872** | **LOCKED** (s44). Same 0.704/0.872 point as s42/s43. |
| **20353533** | DRL + L2 ranking s42 (Gilad same-loop) | **−5.3 @ 0.600/0.748** | **−24.1 @ 0.667/0.482** | PRELIM s42. Not size-matched to L1 DRL 0.704/0.550. Greedy-cliff. |
| **20353536** | DRL + SVD ranking s42 | **−5.2 @ 0.600/0.748** | **−25.6 @ 0.667/0.482** | PRELIM s42. Same 0.667 point as unmatched greedy. |
| **20353569** | Greedy L2 ranking s42 | **−4.9 @ 0.600/0.734** | **−26.6 @ 0.667/0.454** | PRELIM. Same 0.667 params as DRL L2 −24.1 @ 0.482 FLOPs. Both cliff. |
| **20353571** | Greedy SVD ranking s42 | **−6.2 @ 0.600/0.734** | **−25.8 @ 0.667/0.454** | PRELIM. Tie with DRL SVD −25.6 @ 0.667/0.482. |
| **20353573** | DRL + L2 ranking s43 | **−4.8 @ 0.600/0.780** | **−21.2 @ 0.593/0.475** | PRELIM. Smaller net than s42/s44. Still miss. |
| **20353577** | DRL + L2 ranking s44 | **−3.8 @ 0.600/0.794** | **−23.4 @ 0.685/0.490** | PRELIM s44. Three-seed L2 miss. |
| **20353575** | DRL + SVD ranking s43 | **−5.4 @ 0.600/0.780** | **−19.8 @ 0.593/0.475** | **COMPLETED.** Same size as L2 s43 −21.2. Still miss. |
| **20357696** | DRL + SVD ranking s44 | **−4.0 @ 0.600/0.794** | **−23.7 @ 0.685/0.490** | **COMPLETED.** Same size as L2 s44 −23.4. Three-seed SVD miss. |
| **20382193** | Frozen 10-net s42, eval τ=5 | **−4.4 @ 0.600/0.748** | **−23.9 @ 0.667/0.481** | PRELIM one seed. Easy-net tie. Hard net at the unmatched greedy size / cliff. Tightening τ did not make this actor milder. §32. |

Do **not** quote unmatched always-0.8 −24 as the only r56-w4 greedy. Look-ahead greedy (param floor 0.70) is **−24.9 @ 0.722/0.477** — more params than DRL 0.704, fewer FLOPs than DRL 0.550, and still the cliff. DRL’s −15.9 is not “milder because it kept more weights.”

**FLOP floor (same actors as 20189046/048/050):** r56-w4 TEST is **inside τ=10** on three seeds: s42 **−8.9 @ 0.907/0.702**, s43 **−9.2 @ 0.926/0.703**, s44 **−9.6 @ 0.907/0.703**. Cost vs param-floor DRL: ~91–93% params vs 0.704, ~70% FLOPs vs 0.550. Easy r20-w2: s42 **−2.1 @ 0.600/0.773**; s43 **−3.7 @ 0.600/0.763**; s44 **−4.0 @ 0.800/0.799** (FLOP floor bound earlier on this seed — not the 0.60-param tie). Prefer-Δparams/ΔFLOPs under that FLOP floor (**LOCKED three seeds**, same 0.704/0.872 point): r56-w4 s42 **−8.9**, s43 **−8.0**, s44 **−8.1**. Easy r20-w2 **−2.7 / −2.7 / −1.9 @ 0.600/0.886**. Inside τ at the C4 param point, with more FLOPs kept (0.872 vs FLOP-floor control ~0.70 vs C4 0.550). Do **not** quote eval_train r56-w4 **+7.4**. C4 stays LOCKED at 0.704/0.550.

On 20189046 the **same** r56-w4 checkpoint is −1.7 pp on the CNN **train** loader @ 0.667/0.472 (`eval_train`) vs −15.9 TEST. On 20189048, eval_train is **−0.8 pp @ 0.667/0.472** vs TEST **−16.2 @ 0.704/0.550**. Quote TEST. That gap is FT generalization on the hard net, not catalog leakage (r56-w4 is not in `database_offline_train.json`). Look-ahead 20213131 eval_train r56-w4 **+6.0 pp @ 0.722/0.477** vs TEST **−24.9** — same trap, larger.

---

## 6. Similar-family heuristics vs DRL (C10 TEST, skip r32)

| Net | DRL s44 | Greedy 20202684 | Mild 20202691 | Random 20202692 |
|---|---|---|---|---|
| ResNet-20 w16 | −5.6 @ 0.669/0.634 | −7.7 @ 0.640/0.494 | −5.5 @ 0.669/0.649 | −6.0 @ 0.688/0.603 |
| ResNet-56 w10 | −13.0 @ 0.604/0.397 | −23.1 @ 0.607/0.336 | −12.6 @ 0.661/0.421 | −17.2 @ 0.622/0.357 |
| ResNet-44 | −4.3 @ 0.635/0.582 | −9.0 @ 0.614/0.353 | −4.3 @ 0.699/0.542 | −6.2 @ 0.589/0.408 |
| VGG-19 BN | −2.7 @ 0.814/0.796 | −3.0 @ 0.669/0.661 | −2.4 @ 0.811/0.819 | −2.6 @ 0.714/0.716 |
| MobileNet-v2×0.75 | −1.6 @ 0.706/0.684 | −3.5 @ 0.662/0.494 | −1.7 @ 0.689/0.666 | −2.4 @ 0.698/0.583 |
| DenseNet-100 | −2.1 @ 0.834/0.851 | −2.3 @ 0.700/0.679 | −2.0 @ 0.823/0.828 | −2.2 @ 0.754/0.747 |

Greedy r56-w10 is the **size-matched** hard-net contrast (0.607 vs DRL 0.604). Mild is the honest easy-net control. Greedy DenseNet-100 is **not** size-matched (0.700 vs DRL 0.834); mild DenseNet-100 **−2.0 @ 0.823/0.828** is closer. Random DenseNet-100 **−2.2 @ 0.754/0.747**. Skip r32.

---

## 7. CIFAR-100 recoverability (no DRL unless noted)

**Rule:** C100 is an FT/env question first. Early “does not recover” tests were **recovery probes** (fixed rate + fine-tune), not C10→C100 agent transfer. The 10-net agent has **not** been evaluated on `input_offline_c100.json`. Mixed-catalog RL **20158277** had **no real 2–5% cut** inside τ (2/34 val cells inside τ, both at ≥98.5% params). That is why C100 was kept **out** of the 10-net / 24-net train catalogs until a recipe recovers a real 2–5% cut. Putting C100 *into* a train catalog did not help those probes.

### 7.1 VGG-11 BN, 160-ep SGD + cosine + MixUp + AutoAugment (job 20202759) — LOCKED

Baseline TEST 70.77%. Quote TEST (val still drops ~3 pp).

| Keep-rate | TEST acc | Δacc TEST (pp) | params kept | within_budget (val) |
|---|---|---|---|---|
| 0.9 | 72.19% | **+1.42** | 0.952 | yes |
| 0.85 | 71.86% | **+1.09** | 0.928 | yes |
| 0.8 | 72.12% | **+1.35** | 0.904 | yes |

This is a real ~10% param cut with a TEST **gain**. VGG on C100 is an action menu.

**Independent confirm, same recipe, job 20204214 (LOCKED):** baseline TEST 70.77%. Val still drops ~3 pp; quote TEST.

| Keep-rate | TEST acc | Δacc TEST (pp) | params kept | within_budget (val) |
|---|---|---|---|---|
| 0.9 | 71.64% | **+0.87** | 0.952 | yes |
| 0.85 | 72.00% | **+1.23** | 0.928 | yes |
| 0.8 | 72.17% | **+1.40** | 0.904 | yes |

Same direction as 20202759 (gain at 90–95% params). Do not average the two jobs; they are two runs of the same recipe.

### 7.2 Residuals, DenseNet-40, MobileNet-v2×1, same recipe (job 20204214) — LOCKED (job COMPLETED 16 Aug 23:07)

Keep-rate 0.8 leaves **~95–96%** params on residuals and **~98–99%** on DenseNet-40 / MobileNet — not a 10% menu. TEST inside τ (or a TEST gain) because almost nothing was cut. Quote TEST; `within_budget` is **val**. Probe: 14/18 val-OK (the four DROPs are r56-w9 keep 0.8 and all three MobileNet rates).

| Net | Keep-rate | TEST Δacc (pp) | params kept | val within τ=10 | Status |
|---|---|---|---|---|---|
| r20-w13 | 0.9 | −0.87 | 0.972 | yes | LOCKED |
| r20-w13 | 0.85 | −0.56 | 0.963 | yes | LOCKED |
| r20-w13 | 0.8 | −1.13 | 0.954 | yes | LOCKED |
| r56-w9 | 0.9 | −1.96 | 0.977 | yes | LOCKED |
| r56-w9 | 0.85 | −3.22 | 0.966 | yes | LOCKED |
| r56-w9 | 0.8 | −2.84 | 0.954 | **no** (val −10.8 pp) | LOCKED |
| r56-w15 | 0.9 | −1.02 | 0.979 | yes | LOCKED |
| r56-w15 | 0.85 | −1.13 | 0.972 | yes | LOCKED |
| r56-w15 | 0.8 | −1.72 | 0.959 | yes | LOCKED |
| DenseNet-40 | 0.9 | −0.65 | 0.995 | yes | LOCKED (tiny cut) |
| DenseNet-40 | 0.85 | −1.05 | 0.993 | yes | LOCKED (tiny cut) |
| DenseNet-40 | 0.8 | −0.43 | 0.990 | yes | LOCKED (tiny cut) |
| MobileNet-v2×1 | 0.9 | **+1.50** | 0.993 | **no** (val −12.4 pp) | LOCKED (tiny cut) |
| MobileNet-v2×1 | 0.85 | **+1.05** | 0.988 | **no** (val −12.5 pp) | LOCKED (tiny cut) |
| MobileNet-v2×1 | 0.8 | **+1.46** | 0.984 | **no** (val −12.5 pp) | LOCKED (tiny cut) |

VGG remains the only C100 family with a real ~10% structured cut and a TEST gain. Do not mix C100 residuals/DenseNet/MobileNet into the C10 agent on the strength of these cells.

### 7.3 Adam 40–80 ep probes — LOCKED as negative recipe

No real 2–5% structured cut inside −10 pp.

| Job | What | Val cells inside τ | Real cut? |
|---|---|---|---|
| 20158277 | Mixed C100 recovery / RL-adjacent probe | 2/34 | **No.** Both OK cells ≥98.5% params; TEST 60.3%→54.4% and 53.7% |
| 20168590 | Same recipe + crop/flip aug, rates 0.99/0.98/0.95 | 1/26 | **No.** The one OK cell is rate 0.95 @ 80 ep, params **0.995**, TEST 70.05%→68.3% (−1.75 pp) |
| 20140557 | C100-thin DRL 0.98/0.95 | cancelled ~1 h, 0% train-within −10 | Not an encoder problem |

### 7.4 C100 DRL (job 20202760) — PRELIM, do not quote as TEST

Train catalog `database_c100_wide.json` (mixes VGG with unrecovered residuals). Rates 1.0/0.98/0.95; FT 80 ep (probe was 160). ~ep 102 at 16 Aug 23:38. **Not a result.** Do not start a second C100 DRL. Do not fold C100 into the C10 10-net/24-net agent.

**24-net** (`database_offline_wide.json`) is extra **C10** widths/sources. It is the lever for r56-w4, **not** for C100 recoverability.

---

## 8. Running / queued (ops, not paper tables)

| Job | Role | State at 11:30 IDT 18 Aug |
|---|---|---|
| **20276582 / 583 / 584** | Digit-MNIST LeNet s42/s43/s44 | **COMPLETED**. TEST **+2.8 / +2.7 / +2.9 pp**. §22 three-seed. |
| **20276586 / 587 / 588** | SVHN r20-w8 frozen eval | **COMPLETED**. TEST **−2.0 / −1.5 / −2.0**. §23. |
| **20270291 / 293 / 295** | C9 frozen 10-net → C100 | **COMPLETED**. TEST in §21. ShuffleNet hole filled by **20289197**. |
| **20201235** | 24-net DRL s42 | **COMPLETED** 04:29. Post-train similar-pool DenseNet-100 −2.1 @ 0.793/0.825. **Not r56-w4.** |
| **20202693** | 24-net DRL s43 | **COMPLETED** 12:12. Similar-pool TEST in §26. Skip r32. Not r56-w4. |
| **20204215** | 24-net DRL s44 | **COMPLETED** 01:09. Similar-pool TEST: r20 **−5.6 @ 0.680/0.653**; r56-w10 **−8.6 @ 0.616/0.538** inside τ; r44 **−3.3**; VGG-19 **−2.6**; MobileNet **−2.2 @ 0.721/0.630**; DenseNet-100 **−2.4 @ 0.798/0.813**. Skip r32. Not skinny-w4. §26. |
| **20201260** | 24-net similar held-out (skip-train afterok) | **COMPLETED** 21:08. r20 **−5.9**; r56-w10 **−12.6** miss; r44 **−4.5**; VGG-19 **−2.8**; MobileNet **−1.7 @ 0.668/0.651**; DenseNet-100 **−1.7 @ 0.835/0.831**. Skip r32. Not skinny-w4. |
| **20201263** | 24-net skinny r56-w4 (the C8 eval) | **COMPLETED** 22:30. r56-w4 TEST **−25.0 @ 0.704/0.499** miss (10-net s42 **−15.9 @ 0.704/0.550**). Easy r20-w2 **−5.7 @ 0.600/0.748**. |
| **20201265** | 24-net unlike afterok | **COMPLETED** 01:19. TEST RepVGG-A0 **−4.9 @ 0.671/0.545**; A1 **−3.9 @ 0.662/0.552**. ShuffleNet-v2×1 / ×1.5 **no eval_test FINAL** (grouping traceback; `step.finetune` ×4). Do not quote the job-mean −0.02 pp. §27. |
| **20353533 / 20353536** | C10-thin DRL ranking L2 / SVD s42 | **COMPLETED.** r20 **−5.3 / −5.2 @ 0.600/0.748**; r56-w4 **−24.1 / −25.6 @ 0.667/0.482**. |
| **20353577** | C10-thin L2 s44 | **COMPLETED.** r20 **−3.8 @ 0.600/0.794**; r56-w4 **−23.4 @ 0.685/0.490** cliff. SVD s43 **20353575** took the GPU. |
| **20353569 / 20353571** | Greedy L2 / SVD same-loop | **COMPLETED.** r56 **−26.6 / −25.8 @ 0.667/0.454**. r20 **−4.9 / −6.2**. |
| **20353573 / 20353575** | C10-thin L2 s43 / SVD s43 | Both **COMPLETED.** L2 r56 **−21.2 @ 0.593/0.475**. SVD r20 **−5.4 @ 0.600/0.780**; r56 **−19.8 @ 0.593/0.475**. |
| **20353537 / 20353579 / 20359431** | Unlike-family FLOP floor 0.70 s42/s43/s44 | **COMPLETED three-seed.** §28. All four nets inside τ (milder than default unlike). |
| **20353538 / 20353581 / 20359432** | C100 residual SGD + FLOP floor 0.70 | **COMPLETED three-seed.** §25. s43 r56 **−10.4 @ 0.950/0.702** miss. **20359432** used full C100 catalog (extra VGG/ShuffleNet/RepVGG TEST). |
| **20289097 / 20307394 / 20308031 / 20318168** | ImageNet MobileNet-v2 | **20318168 TIMEOUT** 2 d 0 h. **No TEST.** Do not quote 82.8% or train-loader. afterany started s43 **20360208**. |
| **20353581** | C100 residual FLOP s43 | **COMPLETED.** r20 **−7.0 @ 0.881/0.702** inside; r56 **−10.4 @ 0.950/0.702** miss. |
| **20357696** | SVD ranking s44 | **COMPLETED.** r20 **−4.0 @ 0.600/0.794**; r56 **−23.7 @ 0.685/0.490**. Three-seed SVD miss. |
| **20353579 / 20359431** | Unlike FLOP s43 / s44 | **COMPLETED.** §28. |
| **20359432** | C100 FLOP s44 (full C100 catalog) | **COMPLETED** 13h 11m. Residuals + VGG **+0.5** / ShuffleNet **−0.8** / RepVGG-A0 **−1.1**. §25. |
| **20353582** | C100 DRL **20307403** actor → held-out residuals | **COMPLETED** 6 h 30 m. r20-w16 **−8.3 @ 0.673/0.627**; r56-w15 **−8.4 @ 0.662/0.469**. One seed, both inside τ. Do not overwrite §21. §31. |
| **20360208** | ImageNet MobileNet-v2 s43 | **RUNNING** ~14 h. 7-day wall. eval_train step 12 — **no TEST.** Do not quote train-loader. s44 **afterany**. |
| **20360209 / 20360210 / 20360211** | Similar-family FLOP floor 0.70 s42/s43/s44 | **COMPLETED** three-seed. DenseNet **−2.2 / −2.4 / −2.2** @ 0.837/0.833, 0.822/0.827, 0.780/0.797. Skip r32. §29. |
| **20360212 / 213 / 214** | C100 residual FLOP + prefer Δparams/ΔFLOPs s42/s43/s44 | **COMPLETED three-seed.** r56 **−8.2 / −4.5 / −8.5 @ 0.703/0.872** inside. r20 s44 **−10.4** miss. §25. |
| **20289103 / 20289105** | C10-thin param floor 0.80 | **COMPLETED**. §24. r56-w4 still cliffs. |
| **20289099** | C100 residual SGD 80-ep s42 | **COMPLETED** 08:51. §25. r56-w15 **−9.1 @ 0.612/0.442** inside τ; r20-w16 **−12.2** still miss. |
| **20289197 / 20307286 / 20307395** | C100 ShuffleNet dummy-forward s42/s43/s44 | **COMPLETED**. TEST **−3.9 / −3.4 / −4.3**. Three-seed inside τ. §21. |
| **20307289** | C100 residual SGD 80-ep s43 | **COMPLETED** 18:53. r20 **−10.1 @ 0.691/0.675** miss; r56-w15 **−9.4 @ 0.672/0.450** inside τ. §25. |
| **20307396** | C100 residual SGD 80-ep s44 | **COMPLETED** 20:45. r20 **−8.8 @ 0.698/0.613** inside τ; r56-w15 **−10.8 @ 0.571/0.360** miss. §25. |
| **20307291 / 20308033 / 20317562** | FLOP floor + prefer Δparams/ΔFLOPs | **LOCKED three-seed.** r56 **−8.9 / −8.0 / −8.1 @ 0.704/0.872**. r20 **−2.7 / −2.7 / −1.9 @ 0.600/0.886**. |
| **20317564** | Param floor 0.80 s44 | **COMPLETED** 17:02. r20 **−5.0 @ 0.600/0.760**; r56-w4 **−22.5 @ 0.796/0.537**. Three-seed miss. §24. |
| **20307403** | CIFAR-100 DRL on VGG-11/16 + ShuffleNet (SGD recipe) | **COMPLETED** 08:48 (1 d 20 h). **Train only — do not quote train returns.** afterok residual eval **20353582**. |
| **20381785 / 786 / 787** | Similar-family FLOP 0.70 + prefer Δparams/ΔFLOPs s42/s43/s44 | **RUNNING** ~1 d 6 h. s42 DenseNet **eval_train**. s43 PRELIM + MobileNet **−2.0 @ 0.767/0.912** (DenseNet **eval_test**). s44 r44 **−2.6 @ 0.702/0.872** (VGG-19 **eval_test**; r32 skipped). §33. afterok C100 prefer. |
| **20381788 / 798 / 799** | Unlike-family FLOP 0.70 + prefer s42 / s43 / s44 | **COMPLETED three-seed.** §30. Same size. All four nets inside τ. |
| **20381800 / 801 / 802** | C100 C9 Adam-40 FLOP 0.70 + prefer s42/s43/s44 | **PENDING** afterok **20381785 / 786 / 787**. afterok FLOP-only **20382180–182**. |
| **20382177 / 178 / 179** | Similar look-ahead greedy s42/s43/s44 | **RUNNING** ~1 d 2 h / 1 d 1 h / 19 h. s42/s43 MobileNet **eval_train**; s44 DenseNet **eval_train**. **No TEST.** afterok mild **20382184–186**. |
| **20382180 / 181 / 182** | C100 Adam-40 FLOP-floor only (no prefer) | **PENDING** afterok C100 prefer. VGG Pareto middle. |
| **20382184 / 185 / 186** | Similar mild s42/s43/s44 | **PENDING** afterok look-ahead greedy. |
| **20382187 / 188 / 189** | Similar random s42/s43/s44 | **PENDING** afterok mild. |
| **20382192** | ImageNet MobileNet-v2 s44 | **PENDING afterany:20360208**. 7-day wall, rtx_4090. TIMEOUT hole closed. |
| **20382193 / 194 / 195** | C10-thin τ=5 s42/s43/s44 | s42 **20382193 COMPLETED**. s43 **20382194 COMPLETED** 6 h 4 m. TEST §32 (r56 cliff). s44 **20382195 RUNNING** ~2 h 35 m, r56 **eval_train**. afterok similar FLOP-floor look-ahead serial. |
| **20382196 / 197 / 198** | Unlike look-ahead greedy s42/s43/s44 | **PENDING** afterok similar random. afterok unlike mild **20412380 / 382 / 384**. |
| **20412380 / 382 / 384** | Unlike mild s42/s43/s44 | **PENDING** afterok unlike look-ahead. Wave I. |
| **20412385 / 386 / 387** | Unlike random s42/s43/s44 | **PENDING** afterok unlike mild. Wave J. |
| **20412388 / 389 / 390** | C100 Adam-40 look-ahead greedy s42/s43/s44 | **PENDING** afterok C100 FLOP-only. Wave K. VGG Pareto heuristic. |
| **20412391 / 392 / 393** | Similar FLOP-floor look-ahead serial s42→s43→s44 | **PENDING** afterok τ=5 s44 **20382195**. Wave L. Fills the tau5 GPU. |
| **20412394 / 395 / 396** | Unlike FLOP-floor look-ahead serial | **PENDING** **afterany** ImageNet s44 **20382192**, then afterok. Wave M. |
| **20412530 / 531 / 532 → 20412533 / 534 / 536** | C100 Adam-40 mild then random s42/s43/s44 | **PENDING** afterok C100 look-ahead. Wave N. VGG Pareto heuristics. |
| **20412538 / 540 / 542 → 20412545 / 546 / 548** | Similar FLOP-floor mild then random serial | **PENDING** afterok similar FLOP-floor look-ahead s44. Wave O. |
| **20412549–551 → 20412552–554** | Unlike FLOP-floor mild then random serial | **PENDING** afterok unlike FLOP-floor look-ahead s44. Wave P. |
| **20412555 / 556 / 557** | Similar FLOP+prefer look-ahead greedy s42/s43/s44 | **PENDING** afterok unlike random. Wave Q. Prefer-point heuristic. |

**PC cadence (20 Aug – ~3 Sep):** morning few minutes; PC **off** morning→afternoon; afternoon→morning **open**. **21 Aug 06:30 status written.** Ido back ~13:30. PC on; VPN may drop. afterok is the submit path. Night note: `/home/paretsky/SPECTRA_NIGHT_CHAIN_21AUG.txt`. ImageNet s42 **TIMEOUT**; s43 **20360208 RUNNING**; s44 **afterany**. **8/8 + 52 PD.** Every running job has a child.

**20189049 is done.** Overlay of `e985d5e` onto `/home/paretsky/SPECTRA-CompressionAgent` is now allowed. New jobs use the leap tree after `git pull` (ShuffleNet rollback is not in SPECTRA-night).

**C100 next lever (not more C10 catalog diversity):** Frozen CIFAR-10 agent + Adam-40 misses on thin residuals. SGD-80 three-seed: r56-w15 **−9.1 / −9.4 / −10.8** (s44 miss at a smaller net); r20 only s44 inside τ. FLOP-floor three-seed: r20 inside; r56 s43 **−10.4 @ 0.950/0.702** miss. FLOP+prefer: r56 **−8.2 / −4.5 / −8.5 @ 0.703/0.872** three-seed inside; r20 s44 **−10.4** miss. Do not overwrite §21. CIFAR-100 DRL stays on VGG+ShuffleNet (**20307403**). Do not mix unrecovered residuals into that train catalog.

**Evening 19 Aug 22:50:** **8/8 GPUs.** Idle slots filled by Gilad rank: similar FLOP+prefer **20381785–787**, unlike prefer s42 **20381788**. Unlike s43/s44 wait on similar-FLOP DenseNet **20360209 / 611**. C100 Adam-40 FLOP+prefer **20381800–802** wait on similar-prefer (VGG · C100 Pareto). Do not grow catalog. Do not restart encoder / BERT / AMP / skinny-in-train. Uniform keep-rate is not in code (do not alias it to greedy). FPGM/BN-scale/Taylor not tonight.

**Floors:** They are eval-time stop rules on how far a *held-out network* may be pruned. They do not use test labels to train the agent, and they do not change the test images. They **do** limit how small that network gets. Prefer-Δparams/ΔFLOPs under a FLOP floor (**LOCKED three seeds**): r56-w4 **−8.9 / −8.0 / −8.1 @ 0.704/0.872** — inside τ at the C4 param point, without spending FLOPs down to 0.55.

**20189049 is done.** Overlay of `e985d5e` onto `/home/paretsky/SPECTRA-CompressionAgent` is now allowed (not done this turn).

---

## 9. Defaults to keep unless a LOCKED row says otherwise

Floor 0.70. Small Transformer. Rates 1.0/0.9/0.8. τ=10. Full-net FT, 40 ep, patience 10 (C10). NEON reward. Fortify on. Look-ahead off unless `SPECTRA_EVAL_LOOKAHEAD=1`. No AMP. No BERT default. No extra C100 DRL train.

---

## 10. What not to quote (archaeology)

These numbers appeared in canvases / chat and must not migrate into paper tables.

| Source | Number | Why it is not TEST |
|---|---|---|
| Overnight matrix 20066579 | **−1.2 pp** @ 0.60 | Not the eval_test row. TEST is −3.1 @ 0.600/0.790 |
| Overnight 20066578 / 580 / 692 | −5.1 / −4.1 / −4.0 | TEST is −6.4 / −5.1 / −4.1 (same jobs, §13) |
| Overnight 20061144 / 20063793 | −13.6 / −12.2 @ 0.40 | TEST is −14.8 / −13.0 @ 0.400 |
| Mixed 6-net “45–48% within −10” | train-step share | Confounded by C100; not held-out TEST |
| 20140552/53 “100% within −10” | train-step share | Training-env health. Held-out r56-w4 is still ~−24 until 10-net |
| 20189046 eval_train r56-w4 | **−1.7 pp** | CNN train-loader, same held-out checkpoint. TEST −15.9 |
| 20189048 eval_train r56-w4 | **−0.8 pp** @ 0.667 | Same trap. TEST −16.2 @ 0.704/0.550 |
| 20213131 eval_train r56-w4 | **+6.0 pp** @ 0.722 | Look-ahead train-loader. TEST is **−24.9 @ 0.722/0.477** |
| 20238166 eval_train r56-w4 | **+7.4 pp** @ 0.926/0.703 | FLOP-floor train-loader. TEST is **−8.9 @ 0.907/0.702** |
| 20238653 eval_train r56-w4 | **+7.4 pp** @ 0.926/0.703 | Same trap. TEST is **−9.2 @ 0.926/0.703** |
| C100 `within_budget` at 99% params | “recovered” | Not a 2–5% cut |

---

## 11. Engineering overhaul since `ecefe78` (method, not a TEST table)

Starting point: 8 Aug 2026 checkpoint. These fixes made later tables *mean* something. Paper appendix / method, not results.

1. **Inherited checkout could not run; compression was fake.** `torch.nn` prune zeroed weights but left shapes (param/FLOP counts never fell; a second prune re-selected zeros). Resize assigned into a Python list and never rebound the module (would drop pretrained weights). Fine-tune was `for epoch in range(0)`, then infinite reinit recursion. Episode return stored the last step, not the trajectory. Entropy was computed and discarded. Nested DDP wrapped the CNN every step (~7 s → ~21 s). Dataset cache keyed on the wrong dict. **Fixed:** structural rebuild, deepcopy of baseline, `DatasetRegistry`, real FT, usable entropy, correct returns, single-GPU default.
2. **Structured prune with CNN connectivity.** `torch.fx` channel groups: residual adds, DenseNet concat, ShuffleNet/depthwise ties. Masked zeros get **no** size credit. Classifier output never shrunk. Uniform 0.8 pass actually drops 33–58% params on VGG/ResNet/DenseNet/MobileNet.
3. **State encoder:** small Transformer (~2–3M) trained with A2C is the default. Frozen BERT is an ablation (§16: it does not fix r56-w4).
4. **Honest protocol:** Fortify (a requested cut must drop ≥1 channel). Stem / width-1 cannot take fake prunes. Eval identity-pads at 0.70 params. Greedy/mild/random share the same FT + floor loop.
5. **Catalogs and cluster:** 287 checkpoints mapped; train is a subset (`offline_pools_manifest.json`). Similar vs novel splits. `--datasets` lazy-load no longer silently pulls C100 into C10 jobs (`6cedbe0`). Train soft-stops on timer; eval still runs; `afterok` successors; 7-day wall.

**Fortify insight:** on thin ResNets (6–16 channels) `ceil(0.9×6)=6` used to apply nothing while A2C still scored a step. A 0.8 *rate* is also not a 20% *size* cut — residual groups decide params/FLOPs.

---

## 12. CIFAR-10 recoverability probes (no RL) — LOCKED as protocol

Jobs **20018419** / **20025708** (early week; τ was 5 pp on the first matrix). Single-layer prune then FT.

| Setting | Result |
|---|---|
| Full-net FT | OK only at keep-rate **0.9 / 0.8 @ 40 epochs** (some rows). Rates ≤0.7 never recovered. 20 ep not enough. |
| Layer-only FT (then agent default) | **0/32 OK** on ResNet-20 and ResNet-56. Stem 0.9@40 = −9.2 pp. |
| Group-aware freeze + BN-safe (**20053627**) | Still **0/32 OK**. Stem worse (−33 pp @ 0.9/40). |
| Full-pass then FT | **0/4 OK**. Rate 0.9@40 still **−18 pp** at params ×0.58; 0.8@40 **−27 pp**. |

**Consequence (LOCKED protocol):** `train_compressed_layer_only=False`, ≥40 FT epochs, action menu **1.0 / 0.9 / 0.8**, τ=10. Cold start was reward + actions + FT, not “BERT cannot see CNNs.”

Early RL with the old FT (**20013559** medium, **20018420** diag): entropy stuck ~1.61, huge negative returns, one eval 0.65→0.22 at params ×0.40. Do not quote as a SPECTRA result.

---

## 13. Reward / fortify / eval-floor A/B (2-net C10 diag) — LOCKED TEST

Train: C10 ResNets only. Held-out TEST: thin r20-w2. Overnight canvas quoted slightly optimistic “Eval Δacc”; **TEST rows below are canonical.**

| Job | Profile | TEST Δacc | params / FLOPs | Status |
|---|---|---|---|---|
| 20061144 | king, no fortify, no 0.70 floor | −14.8 | 0.400 / 0.341 | LOCKED (why the floor exists) |
| 20063793 | king_fortify, no floor | −13.0 | 0.400 / 0.346 | LOCKED |
| 20066578 | NEON reward + floor | −6.4 | 0.600 / 0.748 | LOCKED |
| **20066579** | **structural reward + floor** | **−3.1** | **0.600 / 0.790** | **LOCKED headline Pareto** |
| 20066580 | shaped reward + floor | −5.1 | 0.600 / 0.748 | LOCKED |
| 20067692 | structural, seed 43 | −4.1 | 0.600 / 0.734 | LOCKED (replicates training health) |

**Keep:** Fortify (entropy 0.88 vs stuck 1.10). Eval floor 0.70 (moves held-out from ~−13–15 pp @ 40% params to ~−3–6 @ 60%). Default **paper reward remains NEON**; structural won this 2-net diag but was not promoted as the 10-net default. τ=15 / mild rates / FT80 / king warm-start did **not** fix mixed C100 (§14).

**Why not 0.7 / 0.6 in the menu:** unconstrained eval **20140546** (on mixed structural ckpt 20122326, `EVAL_MIN_PARAM_RATIO=0`) hit r56-w4 **−42.2 pp @ 0.167 / 0.284**. Greedy 0.8 is already −24 on that net. Deeper menu rates would be a one-shot path past a recoverable size.

---

## 14. Mixed 6-net (C10+C100) — LOCKED as confound, not a C10 result

Every reward / τ / rate / FT-budget / warm-start knob left mixed-DB train-steps at **45–48%** within −10. Per-net split on **20066522** (train steps, not TEST): C10 ResNets ~100% except r56-w4 (9%, med −23.9); C100 ResNets **0%** (med −12 to −19).

| Job | Profile | Held-out TEST | Status |
|---|---|---|---|
| 20066522 | careful_fortify | r20-w2 −14.4 @ 0.43 (incomplete protocol vs later floor) | do not headline |
| 20118369 | C10-labeled mixed | r20-w2 **−4.5 @ 0.600/0.767**; r56-w4 **−18.9 @ 0.685/0.507**; C100 r20-w8 **−21.8 @ 0.681/0.539** | LOCKED confound |
| 20122326 | structural mixed | r20-w2 **−5.4 @ 0.600/0.760**; r56-w4 **−23.6 @ 0.593/0.441**; C100 r20-w8 **−20.7 @ 0.694/0.604** | LOCKED confound |

Failed evals **20122394 / 20122352 / 20118370**: actor path `/runs/job…` (repo prefix eaten) or TIMEOUT — not missing checkpoints. Do not interpret as scientific failures.

**Do not mix C10 and C100 in one agent** until C100 recovers a real cut. Mixing is how train returns go to −100, not how dataset transfer is shown.

---

## 15. Unconfounded C10 training-env (13 Aug) — LOCKED as env, not TEST

Train-step “within −10” after the C10/C100 split. This is **why C10 is a solved training environment**. Held-out TEST for these jobs is §16–17 (r56-w4 still ~−24 until 10-net).

| Job | Catalog | Train-step within −10 | Median train Δacc | Notes |
|---|---|---|---|---|
| 20140552 | C10-thin 3 ResNets | 100% | −4.0 | Control. min_episodes=130 |
| 20140553 | Generic C10 5-family | 100% | −5.1 | VGG-16 BN med −6.5; MobileNet −5.0; chenyaofo r32 −5.1; all 100% |
| 20140554 | Encoder = set, C10-thin | 100% | −3.7 | Tied on train |
| 20140555 | Encoder = wide 6×512 | 100% (warmup) | −3.7 | Tied on train |
| 20140556 | Encoder = frozen BERT | 100% (warmup) | −3.6 | Tied on train |
| 20140557 | C100-thin 0.98/0.95 | 0% | −12.6 | Cancelled |

---

## 16. Encoder A/B TEST (same C10-thin train catalog) — LOCKED

Held-out thin nets. Encoder capacity does **not** separate on the hard net. Do not reopen BERT as the default.

| Job | Encoder | r20-w2 TEST | r56-w4 TEST |
|---|---|---|---|
| 20140552 | small Transformer (default) | −5.2 @ 0.600/0.760 | −23.8 @ 0.685/0.494 |
| 20140554 | set encoder | −4.9 @ 0.600/0.734 | −24.5 @ 0.667/0.471 |
| 20140555 | wide 6×512 | −3.8 @ 0.600/0.749 | −23.5 @ 0.667/0.483 |
| 20140556 | frozen BERT | −2.1 @ 0.600/0.774 | −23.7 @ 0.667/0.473 |

BERT is slightly kinder on the *easy* net and tied on r56-w4. Afterok eval jobs 20140558–62 were empty on the live tree (rechain / path); the TEST rows above come from the train-job eval_test logs.

---

## 17. Catalog-size ladder on held-out r56-w4 (C10 TEST) — LOCKED

Same hard net. The move is **train-catalog diversity**, not encoder, AMP, skinny-in-train, or budget-in-state.

| Job | Train catalog | r56-w4 TEST | r20-w2 TEST |
|---|---|---|---|
| 20118369 | mixed C10+C100 | −18.9 @ 0.685/0.507 | −4.5 @ 0.600/0.767 |
| 20122326 | mixed structural | −23.6 @ 0.593/0.441 | −5.4 @ 0.600/0.760 |
| 20140552 | C10-thin 3 ResNets | −23.8 @ 0.685/0.494 | −5.2 @ 0.600/0.760 |
| 20140553 | generic C10 5-family (no DenseNet, no SVHN/Fashion) | −24.1 @ 0.667/0.476 | −4.7 @ 0.600/0.743 |
| 20148105 | C10 + DenseNet-40 in train | −24.5 @ 0.667/0.481 | −2.4 @ 0.600/0.742 |
| 20168587 | AMP on, thin catalog | −23.8 @ 0.667/0.484 | −6.3 @ 0.600/0.760 |
| 20168588 | skinny r20-w2 **in train**; eval r56-w4 only | −23.8 @ 0.574/0.439 | (not held out) |
| 20168589 | budget token in state | −25.4 @ 0.667/0.470 | −3.8 @ 0.600/0.748 |
| **20189046** | **10-net C10+SVHN+Fashion s42** | **−15.9 @ 0.704/0.550** | −4.2 @ 0.600/0.760 |
| **20189048** | **10-net s43** | **−16.2 @ 0.704/0.550** | −4.9 @ 0.600/0.760 |
| **20189050** | **10-net s44** | **−17.2 @ 0.704/0.550** | −4.6 @ 0.600/0.760 |
| **20201263** | **24-net s42 actor, skip-train eval** | **−25.0 @ 0.704/0.499** | **−5.7 @ 0.600/0.748** |

24-net did **not** continue the 3-net→10-net gain. Same 0.704 param point, fewer FLOPs (0.499 vs 0.550), TEST back at the greedy cliff. Do not train a 48-net catalog hoping for another −8 pp. Do not restart encoder/BERT/AMP/skinny-in-train.

**20148105** other TEST (skip r32): ShuffleNet −1.7 @ 0.881/0.843; VGG-19 −2.5 @ 0.802/0.794; DenseNet-100 −2.1 @ 0.831/0.828. Putting DenseNet in train did **not** move r56-w4; it did make the easy thin net look better (−2.4).

**20168588** landed at 0.574 params (below the usual 0.70 quoting floor) and still −23.8. Putting one thin net in train did not teach width transfer to r56-w4.

24-net (`database_offline_wide.json`) is the next catalog step (**TBD**, claim C8).

---

## 18. Night A/Bs that did not move r56-w4 — LOCKED negative

Isolated tree `/home/paretsky/SPECTRA-night` (not a git overlay of the live leap). AMP **off** unless a later job matches 20140552 quality — 20168587 did not.

| Job | Knob | Verdict |
|---|---|---|
| 20168587 | AMP | Same ~−24 on r56-w4; easy net worse (−6.3 vs −5.2). AMP off. |
| 20168588 | skinny-in-train | r56-w4 still −23.8 at even smaller size. |
| 20168589 | budget-in-state | r56-w4 **worse** (−25.4). Easy net −3.8 (not a reason to change default). |
| 20168590 | C100 FT crop+flip | 1/26 val-OK at 0.995 params. Not a recipe. |

---

## 19. Pretrain job 20123034 — LOCKED as pool, not agent TEST

DenseNet-40 C10 **93.2%**, DenseNet-100 C10 **94.9%**, DenseNet-40 C100 **70.3%**, plus ResNet-20 and VGG-11 on SVHN and Fashion-MNIST. Enabled DenseNet-40 in train and DenseNet-100 as similar-family held-out.

**Pool rule:** 287 checkpoints mapped; do **not** train on every file. Cover the pool by held-out eval. Grafting / DeiT / ViT / MaxViT stay out. ImageNet CNNs are eval-only (no overnight FT). Manifest: `configs/offline_pools_manifest.json`.

---

## 20. Ops incidents (so they are not re-litigated as science)

| Incident | What happened | Read as |
|---|---|---|
| `--datasets` lazy-load | C10 jobs silently loaded C100 | Fixed `6cedbe0`. Morning 46% within −10 was this confound. |
| Actor path `/runs/job…` | Eval jobs 20122394 / 352 / 370 failed | Path bug, not missing ckpts. Rechain with literal paths. |
| Leap afterok nice 0 vs nice=10000 | Look-ahead grabbed a GPU; 20204212 scanceled; look-ahead requeued as **20213131** | Keep leap afteroks at nice 0; backlog nice=10000. |
| Overlay | Live `/home/paretsky/SPECTRA-CompressionAgent` may lag git `e985d5e` | Do not overlay until 20189048/049 finish. |
| Probe `set -e` on 0 OK cells | 20018419 aborted mid-suite | Scientific 0-OK is not a job failure. |

Git checkpoint for night code: **`e985d5e`** (16 Aug). Paper due **30 Sep**. Experiment freeze **15 Sep**.

---

## 21. Claim C9 — frozen 10-net → CIFAR-100 TEST (17 Aug) — LOCKED mixed

Same frozen actors as C1–C5 (`job20158274` / `20163257` / `20164515`). Catalog `configs/input_offline_c100.json`. Skip-train. Param floor 0.70 (no FLOP floor). Failed first submit 20270015/019/022 (`--database` was the C10 train JSON).

**Quote TEST.** Do not mix with §7 recoverability probes (no agent) or cancelled C100 DRL 20202760.

| Net | s42 (20270291) | s43 (20270293) | s44 (20270295) | vs τ=10 |
|---|---|---|---|---|
| VGG-16 BN | **−7.5 @ 0.797/0.834** | **−7.8 @ 0.816/0.784** | **−7.3 @ 0.831/0.826** | inside |
| thin r20-w16 | −19.3 @ 0.604/0.645 | −17.1 @ 0.647/0.619 | −13.6 @ 0.770/0.655 | miss |
| thin r56-w15 | −15.0 @ 0.694/0.600 | −17.8 @ 0.682/0.491 | −17.6 @ 0.682/0.497 | miss |
| RepVGG-A0 | −12.1 @ 0.571/0.446 | −10.6 @ 0.686/0.540 | −12.6 @ 0.570/0.449 | miss |
| ShuffleNet-v2×1 | **−3.9 @ 0.833/0.823** (job **20289197**) | **−3.4 @ 0.819/0.823** (job **20307286**) | **−4.3 @ 0.837/0.823** (job **20307395**) | inside three seeds. s44 log also reports masked effective-params 0.815 — quote **0.837** structural. |

ShuffleNet-v2×1 first C9 jobs crashed (`groups=116` vs input 104) — **not Slurm**. Dummy-forward after every structural prune; restore and mask. Three-seed TEST: s42 **−3.9 @ 0.833/0.823** (72.6% → 68.7%); s43 **−3.4 @ 0.819/0.823** (72.6% → 69.2%); s44 **−4.3 @ 0.837/0.823** (72.6% → 68.3%; masked effective-params 0.815). Quote structural param_ratio. Do not claim every grouped layer was resized. C10 ShuffleNet TEST still stands.

Read with C6: VGG is the C100 family that recovers from structured cuts; residuals miss under C9’s Adam-40 recipe. C9 is dataset transfer of the *frozen C10 agent*, not a new C100 policy. SGD-recipe A/B is §25 (one seed): r56-w15 enters τ; r20-w16 still misses. Do not start a second C100 DRL.

---

## 22. Digit-MNIST LeNet held-out TEST (17–18 Aug) — three-seed LOCKED

Never in `database_offline_train.json` (Fashion-MNIST is; digit MNIST is not). Frozen 10-net actor. Tiny LeNet (`lenet_mnist_sublinear_97.75.pt`). SPECTRA-measured origin **96.3%** (filename 97.75%).

| Job | Seed | TEST Δacc (pp) | params / FLOPs | Status |
|---|---|---|---|---|
| 20276582 | 42 | **+2.8** | 0.883 / 0.900 | LOCKED |
| 20276583 | 43 | **+2.7** | 0.733 / 0.820 | LOCKED |
| 20276584 | 44 | **+2.9** | 0.867 / 0.960 | LOCKED |

Honest caveat: toy 1-channel net, modest param cut. It is a held-out **dataset** cell (NEON analog), not an ImageNet substitute. Do not quote `eval_train`.

---

## 23. SVHN r20-w8 held-out **width** TEST (17 Aug) — three-seed LOCKED

**Not** C9-style dataset transfer. SVHN is already in `database_offline_train.json` (r20-w16 + VGG-11 BN). This checkpoint is a **new width** (8), pretrained 17 Aug job **20276585** (`resnet20-width8_svhn_thin-res-net_96.30_0.069_10.42.pt`, SPECTRA origin **96.3%**). Frozen 10-net actor. Do not present in-catalog r20-w16 / VGG-11 SVHN as held-out TEST.

| Job | Seed | TEST Δacc (pp) | params / FLOPs | vs τ=10 |
|---|---|---|---|---|
| 20276586 | 42 | **−2.0** | 0.652 / 0.612 | inside |
| 20276587 | 43 | **−1.5** | 0.710 / 0.667 | inside |
| 20276588 | 44 | **−2.0** | 0.638 / 0.641 | inside |

Similar-family width transfer on a train-mix dataset. Complements C1 (which is CIFAR-10-only) and C9 (held-out CIFAR-100).

---

## 24. Param floor 0.80 walk (no FLOP floor) — LOCKED miss on r56-w4

Same frozen 10-net actors as C4. Floor **0.80 params**, look-ahead off, no FLOP floor. Jobs **20289103** (s42) / **20289105** (s43) / **20317564** (s44, **COMPLETED** 17:02).

| Net | s42 TEST | s43 TEST | s44 TEST | vs τ=10 |
|---|---|---|---|---|
| thin r20-w2 | **−4.3 @ 0.600/0.741** | **−2.0 @ 0.800/0.780** | **−5.0 @ 0.600/0.760** (overshot 0.80 like s42) | inside |
| thin r56-w4 | **−21.7 @ 0.815/0.543** | **−25.7 @ 0.796/0.537** | **−22.5 @ 0.796/0.537** | miss three seeds |

A ~20% param cut is **not** between C4’s cliff and the FLOP-floor win. r56-w4 still falls off on three seeds. Easy r20 stays inside τ. Do not quote eval_train (s44 r56 eval_train was −1.3 @ 0.722/0.494 vs TEST **−22.5 @ 0.796/0.537**).

ImageNet **20289097 FAILED** 02:28: `build_transform` never Resize/CenterCrops ImageNet JPEGs (collate `[3,489,379]` vs `[3,333,500]`). Not TEST.

---

## 25. C100 residual SGD recipe A/B — PRELIM three-seed (r56 seed-sensitive)

Same frozen 10-net actors. Catalog `configs/input_offline_c100_residuals.json` (thin r20-w16 + r56-w15 only). Skip-train. FT **80-ep SGD + cosine + MixUp + AutoAugment** (C6 VGG recipe, shortened from 160 ep). Param floor 0.70, no FLOP floor. Jobs **20289099** / **20307289** / **20307396** COMPLETED.

**Quote TEST.** Sizes are **not** matched to C9 Adam-40 or across SGD seeds. Do not overwrite §21.

| Net | C9 Adam-40 s42 | SGD s42 (20289099) | SGD s43 (20307289) | SGD s44 (20307396) |
|---|---|---|---|---|
| thin r20-w16 | −19.3 @ 0.604/0.645 | **−12.2 @ 0.687/0.648** miss | **−10.1 @ 0.691/0.675** miss | **−8.8 @ 0.698/0.613** inside (73.0% → 64.2%) |
| thin r56-w15 | −15.0 @ 0.694/0.600 | **−9.1 @ 0.612/0.442** inside | **−9.4 @ 0.672/0.450** inside (78.4% → 69.0%) | **−10.8 @ 0.571/0.360** miss (78.4% → 67.6%) |

r56-w15 is **not** three-seed inside τ. s42/s43 inside at 0.61–0.67 params; s44 missed at a smaller net. r20 is seed-sensitive (only s44 inside, milder cut). Do not claim “C100 residuals are solved.” Do not quote eval_train. Do not overwrite §21.

FLOP floor 0.70 on the same SGD recipe — **COMPLETED three-seed**. Operating point, not matched-size. Do not overwrite §21. s43 r56 **misses τ at 95% params**.

| Net | s42 (20353538) | s43 (20353581) | s44 (20359432) |
|---|---|---|---|
| thin r20-w16 | **−5.7 @ 0.813/0.709** inside | **−7.0 @ 0.881/0.702** inside | **−7.7 @ 0.835/0.707** inside |
| thin r56-w15 | **−4.7 @ 0.871/0.702** inside | **−10.4 @ 0.950/0.702** miss | **−6.4 @ 0.918/0.702** inside |

**20359432** ran the **full** C100 eval catalog (`input_offline_c100.json`, 13h 11m), not residuals-only. Extra s44 SGD+FLOP-floor TEST (not C9 Adam-40): VGG-16 BN **+0.5 @ 0.817/0.800**; ShuffleNet-v2×1 **−0.8 @ 0.824/0.789**; RepVGG-A0 **−1.1 @ 0.825/0.734**. One seed; do not three-seed-lock. Do not overwrite §21.

FLOP floor + prefer Δparams/ΔFLOPs, same SGD recipe — **COMPLETED three-seed** (**20360212 / 213 / 214**). Residuals-only catalog. Same 0.703/0.872 r56 point as C10-thin prefer.

| Net | s42 (20360212) | s43 (20360213) | s44 (20360214) |
|---|---|---|---|
| thin r20-w16 | **−8.1 @ 0.716/0.879** inside | **−7.9 @ 0.716/0.879** inside | **−10.4 @ 0.716/0.879** miss |
| thin r56-w15 | **−8.2 @ 0.703/0.872** inside | **−4.5 @ 0.703/0.872** inside | **−8.5 @ 0.703/0.872** inside |

r56-w15 is three-seed inside τ at this prefer point. r20-w16 is not (s44 miss). Still not “C100 residuals are solved.”

---

## 26. 24-net s43 post-train similar-pool TEST (20202693) — PRELIM

Job **20202693** COMPLETED 18 Aug 12:12 (status 0). Profile `offline_wide`: train on `database_offline_wide.json`, then eval `input_offline_similar.json` in the **same** job. Those similar checkpoints are **not** in the 24-net train catalog. Skip akamaster r32.

This is **not** skinny ResNet-56 w4 (that is **20201263**). Dedicated skip-train similar job **20201260** (24-net **s42** actor): r20-w16 **−5.9 @ 0.695/0.611** vs 10-net s42 **−5.4 @ 0.603/0.639**; r56-w10 **−12.6 @ 0.634/0.410** vs 10-net s42 **−9.2 @ 0.658/0.488** — 10-net s42 was inside τ; 24-net s42 **misses**, at a smaller net (not size-matched). r44 **−4.5 @ 0.682/0.542** vs 10-net s42 **−4.3 @ 0.632/0.519**. VGG-19 **−2.8 @ 0.817/0.771**. MobileNet-v2×0.75 **−1.7 @ 0.668/0.651** vs 10-net s42 **−2.5 @ 0.689/0.630**. Skip r32 (broken origin_acc; TEST line exists and is discarded).

| Net | 10-net s43 (§3, 20163257) | 24-net s43 (20202693) | 24-net s42 skip-train (20201260) | vs τ=10 |
|---|---|---|---|---|
| ResNet-20 w16 | −4.5 @ 0.673/0.696 | **−4.5 @ 0.673/0.716** | **−5.9 @ 0.695/0.611** | inside |
| ResNet-56 w10 | −12.4 @ 0.664/0.433 | **−12.3 @ 0.661/0.422** | **−12.6 @ 0.634/0.410** | miss. Fair s42 compare: 10-net **−9.2 @ 0.658/0.488** (inside) |
| ResNet-44 | −4.2 @ 0.700/0.586 | **−5.1 @ 0.667/0.509** | **−4.5 @ 0.682/0.542** | inside |
| VGG-19 BN | −2.6 @ 0.788/0.802 | **−2.6 @ 0.898/0.882** | **−2.8 @ 0.817/0.771** | inside |
| MobileNet-v2×0.75 | −2.1 @ 0.697/0.651 | **−2.4 @ 0.672/0.590** | **−1.7 @ 0.668/0.651** | inside |
| DenseNet-100 | −2.1 @ 0.805/0.833 | **−2.4 @ 0.793/0.814** | **−1.7 @ 0.835/0.831** | inside |

24-net s42/s43 did **not** pull similar r56-w10 inside τ. Do not quote as the skinny-w4 result. **20201260 COMPLETED** 21:08; DenseNet TEST is **−1.7 @ 0.835/0.831** (do not quote eval_train +0.0 @ 0.843). 24-net s44 same-job eval (**20204215**): r20-w16 **−5.6 @ 0.680/0.653** (10-net s44 **−5.6 @ 0.669/0.634**); r56-w10 **−8.6 @ 0.616/0.538** inside τ (10-net s44 **−13.0 @ 0.604/0.397** — not size-matched; more FLOPs kept); r44 **−3.3 @ 0.693/0.576**; VGG-19 BN **−2.6 @ 0.752/0.801** (10-net s44 **−2.7 @ 0.814/0.796**); MobileNet-v2×0.75 **−2.2 @ 0.721/0.630** (10-net s44 **−1.6 @ 0.706/0.684**). Skip r32. DenseNet-100 **−2.4 @ 0.798/0.813** (10-net s44 **−2.1 @ 0.834/0.851**; 24-net s43 **−2.4 @ 0.793/0.814**). **20204215 COMPLETED** 01:09. Seed 44 inside on r56-w10 does not overwrite s42/s43 misses. Skinny eval **20201263 COMPLETED** 22:30: r56-w4 **−25.0 @ 0.704/0.499** (miss; worse than 10-net −15.9 @ 0.704/0.550); easy r20-w2 **−5.7 @ 0.600/0.748**.

---

## 27. 24-net unlike-family (20201265) — PRELIM COMPLETED (RepVGG only)

Job **20201265** COMPLETED 19 Aug ~01:19 (status 0). 24-net s42 actor, skip-train unlike pool. This is **not** C8 (skinny r56-w4 is **20201263**, LOCKED miss).

| Net | 10-net s42 (§4) | 24-net s42 (20201265) | vs τ=10 |
|---|---|---|---|
| RepVGG-A0 | −4.8 @ 0.681/0.565 | **−4.9 @ 0.671/0.545** | inside |
| RepVGG-A1 | −4.7 @ 0.650/0.521 | **−3.9 @ 0.662/0.552** | inside |
| ShuffleNet-v2×1 / ×1.5 | −1.1 / −2.4 | **no eval_test FINAL** (grouping / `step.finetune` fail) | — |

24-net unlike RepVGG matches the 10-net agent (A1 slightly milder FLOP cut: 0.552 vs 0.521). ShuffleNet on this 24-net eval did not yield TEST — same grouping hole as earlier C10 greedy ShuffleNet, not a DRL miss. Do not quote eval_train +0.0 on A0/A1. Do not quote the job-mean **−0.02 pp**. Do not treat this as a 24-net skinny-w4 result.

---

## 28. Unlike-family FLOP floor 0.70 — LOCKED three-seed

Frozen 10-net actors, look-ahead on, `eval_offline_novel`. Jobs **20353537 / 20353579 / 20359431** COMPLETED. Default unlike (§4) was already inside τ. These points keep more weights / more FLOPs. Not a new transfer win.

| Net | s42 (20353537) | s43 (20353579) | s44 (20359431) | vs τ=10 |
|---|---|---|---|---|
| ShuffleNet-v2×1 | −1.9 @ 0.809/0.826 | **−1.8 @ 0.872/0.843** | **−1.5 @ 0.887/0.868** | inside |
| ShuffleNet-v2×1.5 | −2.0 @ 0.821/0.799 | **−2.3 @ 0.836/0.801** | **−2.8 @ 0.856/0.846** | inside |
| RepVGG-A0 | −3.9 @ 0.792/0.702 | **−4.6 @ 0.879/0.748** | **−4.6 @ 0.858/0.754** | inside |
| RepVGG-A1 | −4.3 @ 0.888/0.735 | **−4.2 @ 0.887/0.735** | **−4.2 @ 0.850/0.738** | inside |

Default unlike sizes for comparison: ×1 ~0.80/0.83; ×1.5 ~0.82/0.80; A0 0.681/0.565; A1 0.650/0.521.

---

## 29. Similar-family FLOP floor 0.70 — PRELIM (r56-w10 s43 miss)

Jobs **20360209** / **20360210** / **20360211** (s42/s43/s44 **COMPLETED**). Frozen 10-net skip-train, `eval_offline_similar`. Skip akamaster r32 (broken origin_acc; jsonl TEST exists and is discarded).

| Net | s42 (20360209) | s43 (20360210) | s44 (20360211) | default §3 | vs τ=10 |
|---|---|---|---|---|---|
| ResNet-20 w16 | **−4.4 @ 0.768/0.707** | **−4.5 @ 0.717/0.738** | **−5.3 @ 0.787/0.717** | −5.4 / −4.5 / −5.6 | inside |
| ResNet-56 w10 | **−5.9 @ 0.902/0.702** | **−14.3 @ 0.946/0.700** | **−5.6 @ 0.911/0.702** | −9.2 / −12.4 / −13.0 | s43 miss at 95% params |
| ResNet-44 | **−3.7 @ 0.893/0.702** | **−3.2 @ 0.829/0.704** | **−3.6 @ 0.921/0.702** | −4.3 / −4.2 / −4.3 | inside |
| VGG-19 BN | **−2.7 @ 0.767/0.755** | **−3.0 @ 0.807/0.876** | **−2.1 @ 0.821/0.815** | −2.7 / −2.6 / −2.7 | inside |
| MobileNet-v2×0.75 | **−2.1 @ 0.777/0.701** | **−2.1 @ 0.806/0.704** | **−2.1 @ 0.719/0.704** | −2.5 / −2.1 / −1.6 | inside |
| DenseNet-100 | **−2.2 @ 0.837/0.833** | **−2.4 @ 0.822/0.827** | **−2.2 @ 0.780/0.797** | −2.0 / −2.1 / −2.1 | three-seed inside |

FLOP floor **does not** three-seed-rescue similar r56-w10: seed 43 still misses at 94.6% weights / 70% FLOPs. Seeds 42/44 move inside τ because the net is larger (~90% weights), not because the default 0.60–0.66-param miss is fixed. Operating point. DenseNet three-seed is inside τ at a milder cut than default.

---

## 30. Unlike-family FLOP 0.70 + prefer Δparams/ΔFLOPs — LOCKED three-seed

Frozen 10-net skip-train, `eval_offline_novel`, FLOP floor 0.70, `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`. Jobs **20381788 / 20381798 / 20381799** COMPLETED. Default unlike (§4) was already inside τ. Compare to FLOP-floor-only §28. Same param/FLOP point on all three seeds.

| Net | s42 (20381788) | s43 (20381798) | s44 (20381799) | vs τ=10 |
|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.5 @ 0.871/0.944** | **−1.1 @ 0.871/0.944** | **−1.5 @ 0.871/0.944** | inside |
| ShuffleNet-v2×1.5 | **−2.1 @ 0.879/0.950** | **−2.1 @ 0.879/0.950** | **−1.9 @ 0.879/0.950** | inside |
| RepVGG-A0 | **−4.4 @ 0.715/0.756** | **−3.7 @ 0.715/0.756** | **−4.0 @ 0.715/0.756** | inside |
| RepVGG-A1 | **−3.5 @ 0.705/0.753** | **−3.5 @ 0.705/0.753** | **−4.1 @ 0.705/0.753** | inside |

RepVGG prefer keeps fewer weights than FLOP-floor-only §28 (~0.79–0.89) and more FLOPs than default unlike (A0 0.681/0.565). Operating point, not a new unlike-family transfer win.

---

## 31. C100 DRL actor → held-out residuals — PRELIM one seed

Job **20353582 COMPLETED** 6 h 30 m. Frozen actor from C100 DRL train **20307403** (VGG-11/16 + ShuffleNet, SGD recipe). Catalog thin r20-w16 + r56-w15 only. Skip-train. **Quote TEST.** Do not quote 20307403 train returns. Do not overwrite frozen-C10-agent §21.

| Net | C100-DRL s42 (20353582) | frozen C10 Adam-40 s42 (§21) | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | **−8.3 @ 0.673/0.627** | −19.3 @ 0.604/0.645 | DRL inside (milder param cut than §21) |
| thin r56-w15 | **−8.4 @ 0.662/0.469** | −15.0 @ 0.694/0.600 | DRL inside (fewer FLOPs than §21) |

One seed. Sizes are **not** matched to §21. Not “C100 residuals are solved.” Need s43/s44 actors before locking. τ=5 C10-thin **20382193** started after this job.

---

## 32. C10-thin eval τ=5 (frozen 10-net actor) — PRELIM two seeds

Same frozen actors as C4. Skip-train `input_c10_thin.json`. Eval τ=5 instead of default τ=10. **Quote TEST.** Do not quote job-mean −0.08 / −0.05 pp.

| Net | τ=10 DRL s42 (§5) | τ=5 s42 (20382193 COMPLETED) | τ=5 s43 (20382194 COMPLETED) | vs τ=10 / vs τ=5 |
|---|---|---|---|---|
| thin r20-w2 | −4.2 @ 0.600/0.760 | **−4.4 @ 0.600/0.748** | **−4.4 @ 0.600/0.780** | Easy-net tie at 60% params. Inside τ=10; **misses eval τ=5**. |
| thin r56-w4 | −15.9 @ 0.704/0.550 | **−23.9 @ 0.667/0.481** | **−20.1 @ 0.593/0.475** | Cliff. Unmatched greedy size. Misses both τ=10 and τ=5. |

Tightening the **eval** budget does not replay the frozen τ=10 policy as a milder pruner. Two seeds landed at the unmatched-greedy operating point. Seed 44 **20382195 RUNNING**. Do not lock. Do not treat as a C4 fix. Do not expand τ=5.

---

## 33. Similar-family FLOP 0.70 + prefer Δparams/ΔFLOPs — PRELIM (jobs still running)

Frozen 10-net skip-train, `eval_offline_similar`, FLOP floor 0.70, `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`. Skip akamaster r32. Sized `eval_test` with `param_ratio`. Jobs **20381785–787** still **RUNNING**.

| Net | s43 (20381786) | s44 (20381787) | default §3 | FLOP-floor-only §29 | vs τ=10 |
|---|---|---|---|---|---|
| ResNet-20 w16 | **−3.5 @ 0.717/0.880** | **−3.8 @ 0.717/0.880** | −5.4 / −4.5 / −5.6 | −4.4 / −4.5 / −5.3 | inside |
| ResNet-56 w10 | **−4.0 @ 0.702/0.872** | **−4.4 @ 0.702/0.872** | −9.2 / −12.4 / −13.0 | −5.9 / **−14.3 @ 0.946/0.700** / −5.6 | s43/s44 inside |
| ResNet-44 | **−2.6 @ 0.702/0.872** | **−2.6 @ 0.702/0.872** | −4.3 / −4.2 / −4.3 | −3.7 / −3.2 / −3.6 | s43/s44 inside |
| VGG-19 BN | **−2.6 @ 0.837/0.923** | — | −2.7 / −2.6 / −2.7 | −2.7 / −3.0 / −2.1 | s43 inside |
| MobileNet-v2×0.75 | **−2.0 @ 0.767/0.912** | — | −2.5 / −2.1 / −1.6 | −2.1 / −2.1 / −2.1 | s43 inside |

Same prefer point as C10-thin r56-w4 (**−8.9 / −8.0 / −8.1 @ 0.704/0.872**). Seeds 43 and 44 put similar r56-w10 **and** r44 inside τ at **the same** 70% weights / 87% FLOPs; FLOP-floor-only seed 43 kept 95% weights and still missed on r56-w10. MobileNet s43 **−2.0 @ 0.767/0.912** (0.938→0.918). **Do not lock** until s42 TEST exists. s42 **20381785** still DenseNet **eval_train**. s43 DenseNet **eval_test**. s44 VGG-19 **eval_test** (r32 skipped).

