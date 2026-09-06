# SPECTRA results ledger

**As of:** 6 Sep 2026 12:25 IDT. **5 R + 37 PD.** Leap `c08d513` + overlay. Unlike FLOP look-ahead s43 **20412395 COMPLETED**: ×1.5 **−2.6 @ 0.771/0.701** two-seed **−2.5 / −2.6** same size (quote structural; skip masked 0.724). Catalog COMPLETED PRELIM. Freed GPU → matched-VGG **20884671 R** (~24 m, 24G). Child **20412396** still PD (`QOSMaxGRESPerUser`). FLOP-greedy s44 **20715871 R**: MobileNet `eval_test` (not FINAL). Shaped **20967060 R** (`structural_shaped`). C100 recoverable s43 **20884673 R** — do not quote train. Unlike look-ahead s44 **20382198 R** (no TEST yet). A/B nice 100.  
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
| C11 | Prefer Δparams/ΔFLOPs under FLOP floor 0.70 puts **similar** r56-w10 **inside τ=10** at 70% weights / 87% FLOPs (same frozen 10-net actor). FLOP-floor-only does not three-seed-rescue this net. | LOCKED | §33 **−3.8 / −4.0 / −4.4 @ 0.702/0.872**. vs §29 s43 **−14.3 @ 0.946/0.700**. |
| C12 | Frozen C10-trained agent on **ImageNet** MobileNet-v2 (truncated JPEG loader): two-seed inside τ=10 unmatched sizes. Not a SOTA ImageNet fight. No ImageNet DRL train. | **PRELIM** two-seed unmatched | §41 s43 **20360208 −4.6 @ 0.823/0.729** (0.719→0.673); s44 **20382192 −5.1 @ 0.772/0.652** (0.719→0.668). s42 TIMEOUT. |

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

### 2.3 Similar thin ResNet-56 w10 · CIFAR-10 — second NEON-style frontier

Same frozen 10-net actors. Prefer is the lever; FLOP-floor-only is not a three-seed rescue.

| Point | Jobs | TEST Δacc (s42 / s43 / s44) | params / FLOPs | vs τ=10 |
|---|---|---|---|---|
| DRL default (param floor 0.70) | 20158274 / 20163257 / 20164515 | **−9.2 / −12.4 / −13.0** | ~0.66 / 0.43 | miss |
| DRL FLOP floor 0.70 | 20360209 / 210 / 211 | **−5.9 / −14.3 / −5.6** | ~0.90 / 0.70; s43 **0.946/0.700** | not three-seed inside |
| DRL prefer Δparams/ΔFLOPs under FLOP floor 0.70 | **20381785 / 786 / 787** | **−3.8 / −4.0 / −4.4** | **0.702 / 0.872** | **LOCKED three-seed inside** |

Look-ahead greedy s42/s43/s44 **ALL COMPLETED** PRELIM: r56-w10 **three-seed cliff −22.4 / −20.1 / −22.2 @ 0.702/0.376**. r44 three-seed **−8.0 / −9.0 / −8.0 @ 0.703/0.391** inside τ; VGG-19 **−3.5 / −3.2 / −2.5 @ 0.703/0.669**; MobileNet three-seed **−3.2 / −3.3 / −3.0 @ 0.708/0.511** inside; DenseNet three-seed **−2.4 / −2.6 / −2.6 @ 0.701/0.679**. FLOP-floor look-ahead r56 three-seed **−8.3 / −9.2 / −8.6 @ 0.952/0.702** inside PRELIM (§39); r44 three-seed **−4.2 / −4.4 / −4.4 @ 0.947/0.702** inside; VGG three-seed **−3.6 / −2.7 / −2.8 @ 0.804/0.701** inside. s44 now MobileNet `eval_test`. Similar mild s44 r56 **−13.9 @ 0.661/0.421** miss; s43 r56 **−12.8**; r44 three-seed **−4.7 / −4.1 / −4.2 @ 0.699/0.542** inside; VGG three-seed **−3.3 / −2.6 / −3.5 @ 0.811/0.819** inside; MobileNet three-seed **−2.3 / −2.1 / −2.3 @ 0.689/0.666** inside PRELIM (§42). Similar mild s42 catalog **COMPLETED** PRELIM (§42): DenseNet three-seed **−2.4 / −2.1 / −2.3 @ 0.823/0.828** inside. FLOP-floor mild s42 catalog **COMPLETED** PRELIM (§43): DenseNet **−2.2 @ 0.823/0.828** inside (floor did not bind); r56 **−11.6 miss**. s43 r56 **−9.9 @ 0.946/0.701** inside at the same size (two-seed split; do not lock). s43 r44 **−3.6 @ 0.905/0.703** two-seed with s42 **−3.9** same size. Unlike mild s42 ShuffleNet×1 **−1.6 @ 0.857/0.835** structural inside PRELIM (§45). Unlike-random s42 ShuffleNet×1 **−1.6 @ 0.801/0.753** structural inside PRELIM (§49); same Δacc as mild at a harder cut. Similar-random s43 r56 **−14.9 @ 0.628/0.368**; s44 **−16.6 @ 0.690/0.388** two-seed miss unmatched PRELIM (§46); s42 first-pass r20 **−6.4 @ 0.688/0.588** inside; older random **−17.2 @ 0.622/0.357**; not a look-ahead cliff. Prefer-greedy s42 **20715876 COMPLETED** catalog PRELIM (§48): r20 **−3.9 @ 0.717/0.880**; r56 **−4.4 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872**; VGG **−2.9 @ 0.837/0.923**; MobileNet **−2.0 @ 0.767/0.912**; DenseNet **−2.1 @ 0.870/0.951** (0.949→0.928) size-matched near-tie vs DRL prefer **−2.0**. FLOP-floor greedy s42 **20715868** PRELIM (§51): r20 **−4.8 @ 0.908/0.701**; r56 **−9.2 @ 0.952/0.702** inside size-matched to look-ahead **−8.3 / −9.2 / −8.6**; r44 **−4.6 @ 0.947/0.702**; VGG **−3.1 @ 0.804/0.701**; MobileNet **−3.1 @ 0.933/0.700**. Floor stopped unconstrained greedy **−23.1**; greedy ties look-ahead once the floor binds. Similar-random s43 **20382188 COMPLETED** DenseNet **−2.3 @ 0.735/0.754**. Do not lock.

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

**Similar-family FLOP 0.70 + prefer — LOCKED three-seed including DenseNet** (s42/s43/s44 **COMPLETED**). DenseNet **−2.0 / −1.9 / −2.7 @ 0.870/0.951** (s42 0.949→0.929). Table §33. afterok C100 prefer s42 **20381800 COMPLETED** (RepVGG three-seed **−13.0 / −12.5 / −13.0 @ 0.719/0.756** miss; VGG/ShuffleNet three-seed inside). Child FLOP-only **20382180 COMPLETED**.

**Similar look-ahead greedy — PRELIM** (s42/s43/s44 **20382177 / 178 / 179 ALL COMPLETED**). r20 three-seed **−7.4 / −7.0 / −7.3 @ 0.713/0.525**. r56 three-seed **−22.4 / −20.1 / −22.2 @ 0.702/0.376** cliff. r44 three-seed **−8.0 / −9.0 / −8.0 @ 0.703/0.391** inside τ. VGG three-seed **−3.5 / −3.2 / −2.5 @ 0.703/0.669**. MobileNet three-seed **−3.2 / −3.3 / −3.0 @ 0.708/0.511** (s42 0.938→0.906) inside. DenseNet three-seed **−2.4 / −2.6 / −2.6 @ 0.701/0.679** (s42 0.949→0.925). Catalogs COMPLETED. Do not lock. Table §34.

**Similar FLOP-floor look-ahead — PRELIM** (s42 **20412391 COMPLETED**; s43 **20412392 COMPLETED** DenseNet **−2.1 @ 0.798/0.700** two-seed **−2.6 / −2.1**; VGG three-seed **−3.6 / −2.7 / −2.8 @ 0.804/0.701**; r44 three-seed **−4.2 / −4.4 / −4.4 @ 0.947/0.702**; r56 three-seed **−8.3 / −9.2 / −8.6 @ 0.952/0.702** inside). FLOP-floor-only s43 r56 was **−14.3 miss**. s44 r20 **−5.2 @ 0.908/0.701** three-seed **−4.6 / −5.9 / −5.2**; MobileNet three-seed **−3.3 / −3.1 / −3.2 @ 0.933/0.700**. Now DenseNet `eval_test`. Do not lock. Table §39.

**Similar FLOP-floor mild — PRELIM** (s42 **20412538 COMPLETED**; s43 **20412540 COMPLETED**). s42 catalog: r20 **−5.6 @ 0.801/0.706** inside; r56 **−11.6 @ 0.946/0.701** miss; r44 **−3.9 @ 0.905/0.703** inside; VGG **−2.8 @ 0.811/0.819** inside (floor did not bind); MobileNet **−2.4 @ 0.791/0.703** inside (floor bound); DenseNet **−2.2 @ 0.823/0.828** (0.949→0.927) inside (floor did not bind vs unconstrained mild **−2.1 / −2.3** same size). s43 catalog COMPLETED: r20 **−5.1 @ 0.801/0.706** two-seed inside same size; r56 **−9.9 @ 0.946/0.701** (0.959→0.860) **inside τ** at the same size as s42 miss (two-seed split); r44 **−3.6 @ 0.905/0.703** (0.935→0.899) two-seed inside same size as s42 **−3.9**; VGG **−3.0 @ 0.811/0.819** (0.934→0.904) two-seed **−2.8 / −3.0** same size inside (floor did not bind); MobileNet **−2.3 @ 0.791/0.703** (0.938→0.915) two-seed **−2.4 / −2.3** same size inside (floor bound); DenseNet **−2.1 @ 0.823/0.828** (0.949→0.928) two-seed **−2.2 / −2.1** same size inside (floor did not bind). Skip r32. Child spoof **20884670 PD QOS**. Table §43.

**Similar random — PRELIM** (s43 **20382188 COMPLETED** restart; s42 **20382187 COMPLETED**; s44 **20382189 COMPLETED** restart). Keep first-pass s43/s44 r20/r56/r44/VGG. s44 restart DenseNet **−2.5 @ 0.740/0.745** (0.949→0.924). Skip r32. Child unlike look-ahead **20382198 RUNNING**. Table §46.

**Similar FLOP+prefer greedy (L1, no look-ahead) — PRELIM** (s42 **20715876 COMPLETED**). r20 **−3.9 @ 0.717/0.880**; r56 **−4.4 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872** (0.935→0.909); VGG **−2.9 @ 0.837/0.923** (0.934→0.905); MobileNet **−2.0 @ 0.767/0.912** (0.938→0.918); DenseNet **−2.1 @ 0.870/0.951** (0.949→0.928) — all size-matched to DRL prefer. r32 skip. Catalog COMPLETED one seed. Child **20715877 PD QOS**. Do not quote eval_train. Table §48.

**Similar FLOP-floor greedy (L1, no look-ahead, no prefer) — PRELIM** (s42 **20715868 COMPLETED**; s43 **20715870 COMPLETED**; s44 **20715871 RUNNING**). Three-seed same size: r20 **−4.8 / −5.7 / −5.1 @ 0.908/0.701**; r56-w10 **−9.2 / −9.3 / −9.0 @ 0.952/0.702** inside; r44 **−4.6 / −5.1 / −4.1 @ 0.947/0.702**; VGG **−3.1 / −2.7 / −2.5 @ 0.804/0.701**. MN/DenseNet still two-seed. L1 greedy does not use the actor — Δacc spreads are fine-tune noise. Skip r32. Now MobileNet `eval_test` (not FINAL). Table §51.

**Unlike mild — PRELIM** (s42 **20412380 COMPLETED**). Catalog all four unlike nets **inside τ** one seed. ShuffleNet-v2×1 **−1.6 @ 0.857/0.835** (quote structural); ×1.5 **−2.6 @ 0.849/0.828** (quote structural; do not quote masked 0.818); RepVGG-A0 **−5.3 @ 0.680/0.548** (0.943→0.890); A1 **−4.2 @ 0.663/0.539** (0.944→0.902). Look-ahead is worse Δacc on both RepVGGs. Table §45.

**Unlike random — PRELIM** (s42 **20412385 COMPLETED**). Catalog all four unlike nets **inside τ** one seed. ShuffleNet-v2×1 **−1.6 @ 0.801/0.753** (quote structural; do not quote masked 0.764). RepVGG-A0 **−6.2 @ 0.659/0.503** (0.943→0.881). RepVGG-A1 **−5.3 @ 0.649/0.508** (0.944→0.891). ShuffleNet-v2×1.5 **−2.2 @ 0.794/0.761** (0.932→0.910) (quote structural; do not quote masked 0.755). Child Wave Q **20412555 PD QOS**. Table §49.

**Similar mild — PRELIM** (s44 **20382186** r20 **−6.8 @ 0.669/0.649** inside; r56 **−13.9 @ 0.661/0.421** miss; r44 **−4.2**; VGG **−3.5**; MobileNet **−2.3 @ 0.689/0.666** (0.938→0.915) inside same size as 20202691 **−1.7**. s42 **20382184 COMPLETED** r20 **−5.9** three-seed **−5.9 / −5.7 / −6.8** same size inside; r56 **−14.0** three-seed **−14.0 / −12.8 / −13.9** miss; r44 **−4.7 @ 0.699/0.542** (0.935→0.888) three-seed **−4.7 / −4.1 / −4.2** same size inside; VGG **−3.3 @ 0.811/0.819** (0.934→0.901) three-seed **−3.3 / −2.6 / −3.5** same size inside; MobileNet **−2.3 @ 0.689/0.666** (0.938→0.915) three-seed **−2.3 / −2.1 / −2.3** same size inside; DenseNet **−2.4 @ 0.823/0.828** (0.949→0.925) three-seed **−2.4 / −2.1 / −2.3** same size inside. Table §42.

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

**Unlike look-ahead greedy — PRELIM one seed** (s42 **20382196 COMPLETED**). ShuffleNet-v2×1 **−2.2 @ 0.723/0.682** inside; ShuffleNet-v2×1.5 **−2.5 @ 0.710/0.675** inside; quote **structural** keep. RepVGG-A0 **−7.2 @ 0.709/0.577** inside; RepVGG-A1 **−6.3 @ 0.710/0.574** inside. Catalog COMPLETED. Full table §44.

**Unlike mild — PRELIM one seed** (s42 **20412380 COMPLETED**). ShuffleNet-v2×1 **−1.6 @ 0.857/0.835**; ×1.5 **−2.6 @ 0.849/0.828** (quote structural); RepVGG-A0 **−5.3 @ 0.680/0.548**; A1 **−4.2 @ 0.663/0.539**. All four inside τ. Table §45.

**Unlike random — PRELIM one seed** (s42 **20412385 COMPLETED**). ShuffleNet-v2×1 **−1.6 @ 0.801/0.753** inside (quote structural; do not quote masked 0.764). RepVGG-A0 **−6.2 @ 0.659/0.503** inside. RepVGG-A1 **−5.3 @ 0.649/0.508** inside. ShuffleNet-v2×1.5 **−2.2 @ 0.794/0.761** inside (quote structural; do not quote masked 0.755). Catalog COMPLETED. Table §49.

**Unlike FLOP-floor look-ahead — PRELIM** (s42 **20412394 COMPLETED**; s43 **20412395 COMPLETED**). Two-seed same size: ShuffleNet×1 **−1.7 / −2.0 @ 0.801/0.716**; A0 **−6.9 / −6.6 @ 0.847/0.701**; A1 **−6.7 / −5.7 @ 0.857/0.702**; ×1.5 **−2.5 / −2.6 @ 0.771/0.701**. Quote structural keep; skip masked 0.724. Table §47.

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
| **20382193** | Frozen 10-net s42, eval τ=5 | **−4.4 @ 0.600/0.748** | **−23.9 @ 0.667/0.481** | PRELIM. Easy-net ~tie. Hard net unmatched greedy cliff. §32. |
| **20382194** | Frozen 10-net s43, eval τ=5 | **−4.4 @ 0.600/0.780** | **−20.1 @ 0.593/0.475** | PRELIM. Easy-net tie. Hard net cliff (smaller than s42). §32. |
| **20382195** | Frozen 10-net s44, eval τ=5 | **−2.5 @ 0.600/0.794** | **−21.0 @ 0.685/0.490** | **COMPLETED.** Easy net milder. Hard net cliff (same 0.685/0.490 as L2/SVD s44). §32. |

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
| **20289097 / 20307394 / 20308031 / 20318168** | ImageNet MobileNet-v2 s42 lineage | **20318168 TIMEOUT** 2 d 0 h. **No TEST.** Do not quote 82.8% or train-loader. afterany started s43 **20360208**. 7-day retry **20715875** PD afterok **20412393** (`rtx_4090`, nice=10000). |
| **20353581** | C100 residual FLOP s43 | **COMPLETED.** r20 **−7.0 @ 0.881/0.702** inside; r56 **−10.4 @ 0.950/0.702** miss. |
| **20357696** | SVD ranking s44 | **COMPLETED.** r20 **−4.0 @ 0.600/0.794**; r56 **−23.7 @ 0.685/0.490**. Three-seed SVD miss. |
| **20353579 / 20359431** | Unlike FLOP s43 / s44 | **COMPLETED.** §28. |
| **20359432** | C100 FLOP s44 (full C100 catalog) | **COMPLETED** 13h 11m. Residuals + VGG **+0.5** / ShuffleNet **−0.8** / RepVGG-A0 **−1.1**. §25. |
| **20353582** | C100 DRL **20307403** actor → held-out residuals | **COMPLETED** 6 h 30 m. r20-w16 **−8.3 @ 0.673/0.627**; r56-w15 **−8.4 @ 0.662/0.469**. One seed, both inside τ. Do not overwrite §21. §31. |
| **20360208** | ImageNet MobileNet-v2 s43 | **COMPLETED** 4 d 10 h 25 m (ended 25 Aug 01:57 cluster). TEST **−4.6 @ 0.823/0.729** (0.719→0.673) inside τ PRELIM. Truncated JPEG. Do not quote 82.8% or train-loader. afterany s44 **20382192 COMPLETED** **−5.1 @ 0.772/0.652**. §41. |
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
| **20381785 / 786 / 787** | Similar-family FLOP 0.70 + prefer Δparams/ΔFLOPs s42/s43/s44 | **ALL COMPLETED.** Full catalog **LOCKED three-seed** including DenseNet **−2.0 / −1.9 / −2.7 @ 0.870/0.951**. §33. afterok C100 **20381800 COMPLETED**; **801 / 802 COMPLETED**. |
| **20381788 / 798 / 799** | Unlike-family FLOP 0.70 + prefer s42 / s43 / s44 | **COMPLETED three-seed.** §30. Same size. All four nets inside τ. |
| **20381800 / 801 / 802** | C100 C9 Adam-40 FLOP 0.70 + prefer s42/s43/s44 | **ALL COMPLETED** (s42 14 h 33 m, ended 24 Aug 02:51). r20 three-seed **−14.5 / −13.8 / −12.7 @ 0.716/0.879** miss. r56 three-seed **−12.0 / −11.4 / −10.5 @ 0.703/0.872** miss same size. VGG three-seed **−7.6 / −7.7 / −7.3 @ 0.838/0.931** inside same size. ShuffleNet three-seed **−3.5 / −4.1 / −4.6 @ 0.873/0.944** inside same size. RepVGG three-seed **−13.0 / −12.5 / −13.0 @ 0.719/0.756** (s42 0.753→0.623) miss same size. **LOCKED mixed.** Child **20382180 COMPLETED**. Do not overwrite §21. §35. |
| **20382177 / 178 / 179** | Similar look-ahead greedy s42/s43/s44 | **ALL COMPLETED.** s42 **20382177** 6 d 18 h 14 m (ended 26 Aug 22:01 cluster). DenseNet TEST **−2.4 @ 0.701/0.679** (0.949→0.925) three-seed **−2.4 / −2.6 / −2.6** same size inside. Catalogs PRELIM: r20 **−7.4 / −7.0 / −7.3**; r56 **−22.4 / −20.1 / −22.2** cliff; r44 **−8.0 / −9.0 / −8.0**; VGG **−3.5 / −3.2 / −2.5**; MobileNet **−3.2 / −3.3 / −3.0**. Skip r32. §34. |
| **20382180 / 181 / 182** | C100 Adam-40 FLOP-floor only (no prefer) | **ALL COMPLETED** (s42 1 d 10 h 42 m, ended 25 Aug 13:34 cluster). s42 ShuffleNet **−3.4 @ 0.881/0.858** (0.726→0.692) inside unmatched vs **−3.9 / −4.0**; RepVGG **−10.2 @ 0.886/0.761** (0.753→0.651) miss (s43/s44 inside only at ~95% params). afterok look-ahead **20412388 COMPLETED**. §36. |
| **20382184 / 185 / 186** | Similar mild s42/s43/s44 | s44 **20382186 COMPLETED** 5 d 16 h 39 m (ended 28 Aug 21:04 cluster). DenseNet TEST **−2.3 @ 0.823/0.828** (0.949→0.926) two-seed **−2.1 / −2.3** same size inside. Catalog COMPLETED PRELIM: r20 **−6.8**; r56 **−13.9** miss; r44 **−4.2**; VGG **−3.5**; MobileNet **−2.3**. r32 skipped. afterok **20382189 RUNNING**. s43 **20382185 SCANCEL** 30 Aug 06:12 cluster (zombie wrap after **Run finished** 28 Aug 19:19; DenseNet **−2.1** already TESTed; child **20382188** had been released). Freed GPU filled by **20715868**. **Epilog cleared ~09:11 IDT 1 Sep** — job **gone from squeue**. s42 **20382184 COMPLETED** 5 d 22 h 23 m (ended 31 Aug 21:49 cluster). DenseNet TEST **−2.4 @ 0.823/0.828** (0.949→0.925) three-seed **−2.4 / −2.1 / −2.3** same size inside. Catalog COMPLETED PRELIM: r20 **−5.9**; r56 **−14.0** miss; r44 **−4.7**; VGG **−3.3**; MobileNet **−2.3**; DenseNet **−2.4**. r32 skipped (broken origin_acc — do not quote). afterok **20382187 RUNNING**. §42. |
| **20382187 / 188 / 189** | Similar random s42/s43/s44 | s43 **20382188 COMPLETED** 1 d 1 h 40 m (ended 2 Sep 12:24 cluster, `Restarts=2`). Restart DenseNet TEST **−2.3 @ 0.735/0.754** (0.949→0.926) inside unmatched vs older random **−2.2 @ 0.754/0.747**. Catalog COMPLETED PRELIM (restart). Keep first-pass r20/r56/r44/VGG. Restart also: r20 **−6.9**; r56 **−17.6** miss; r44 **−6.9**; VGG **−2.6**; MobileNet **−2.4**. r32 skip. s42 **20382187 COMPLETED** 3 d 0 h 31 m (ended 4 Sep 09:02 cluster). DenseNet TEST **−2.3 @ 0.730/0.755** (0.949→0.926) inside unmatched vs s43 restart **−2.3 @ 0.735/0.754**. Catalog COMPLETED PRELIM first-pass: r20 **−6.4**; r56 **−18.2** miss; r44 **−6.2**; VGG **−3.5**; MobileNet **−2.7**; DenseNet **−2.3**. r32 skip. Child C100 recoverable s43 **20884673 PD QOS**. s44 **20382189 COMPLETED** 2 d 15 h 32 m (ended 5 Sep 21:34 cluster). Restart DenseNet TEST **−2.5 @ 0.740/0.745** (0.949→0.924) inside unmatched. Keep first-pass table. Child unlike look-ahead **20382198 RUNNING**. Child C100 **20884673 RUNNING**. §46. |
| **20382192** | ImageNet MobileNet-v2 s44 | **COMPLETED** 5 d 3 h 52 m (ended 30 Aug 05:49 cluster). TEST **−5.1 @ 0.772/0.652** (0.719→0.668) inside τ unmatched vs s43 **−4.6 @ 0.823/0.729**. Two-seed PRELIM. Do not quote 82.8% or train-loader. afterany released unlike FLOP-floor look-ahead **20412394 RUNNING**. §41. |
| **20382193 / 194 / 195** | C10-thin τ=5 s42/s43/s44 | All **COMPLETED**. TEST §32. r56 **−23.9 / −20.1 / −21.0** cliffs. afterok **20412391** started. |
| **20382196 / 197 / 198** | Unlike look-ahead greedy s42/s43/s44 | **20382196 COMPLETED** 2 d 14 h 19 m (ended 29 Aug 12:49 cluster). ShuffleNet-v2×1.5 TEST **−2.5 @ 0.710/0.675** (0.932→0.907) inside (quote structural 0.710; do not quote masked 0.665). Catalog COMPLETED PRELIM: ShuffleNet×1 **−2.2 @ 0.723/0.682**; RepVGG-A0 **−7.2 @ 0.709/0.577**; RepVGG-A1 **−6.3 @ 0.710/0.574**; ShuffleNet×1.5 **−2.5**. All four inside τ one seed. afterok unlike mild **20412380 COMPLETED** catalog PRELIM. s43 **20382197 PD** (`QOSMaxGRESPerUser`; parent **20382188 COMPLETED**). s44 **20382198** afterok on **20382189**. §44. |
| **20412380 / 382 / 384** | Unlike mild s42/s43/s44 | **20412380 COMPLETED** 1 d 1 h 18 m (ended 30 Aug 14:08 cluster). Catalog COMPLETED PRELIM: ShuffleNet-v2×1 **−1.6 @ 0.857/0.835** (0.924→0.908) inside (quote structural; do not quote masked 0.831); RepVGG-A0 **−5.3 @ 0.680/0.548** (0.943→0.890) inside; RepVGG-A1 **−4.2 @ 0.663/0.539** (0.944→0.902) inside; ShuffleNet-v2×1.5 **−2.6 @ 0.849/0.828** (0.932→0.906) inside (quote structural; do not quote masked 0.818). All four inside τ one seed. Do not quote wrap job-mean **−0.02 pp**. Child unlike-random **20412385 COMPLETED**. Wave I. s43/s44 still afterok. §45. |
| **20412385 / 386 / 387** | Unlike random s42/s43/s44 | s42 **20412385 COMPLETED** 2 d 1 h 11 m (ended 3 Sep 09:42 cluster, `dt-2080-07`). ShuffleNet-v2×1.5 TEST **−2.2 @ 0.794/0.761** (0.932→0.910) inside (quote structural; do not quote masked 0.755). Catalog COMPLETED PRELIM: ShuffleNet×1 **−1.6 @ 0.801/0.753**; RepVGG-A0 **−6.2 @ 0.659/0.503**; RepVGG-A1 **−5.3 @ 0.649/0.508**; ×1.5 **−2.2**. All four inside τ one seed. Child Wave Q **20412555 PD** (`QOSMaxGRESPerUser`; GRES went to **20412394**). s43/s44 still afterok unlike mild. §49. |
| **20412388 / 389 / 390** | C100 Adam-40 look-ahead greedy s42/s43/s44 | **ALL COMPLETED** (s42 11 h 47 m, ended 26 Aug 01:21 cluster). s42 RepVGG **−11.4 @ 0.709/0.577** (0.753→0.639) miss three-seed **−11.4 / −11.7 / −12.1** same size. Catalogs COMPLETED PRELIM: r20 miss; r56 cliff; VGG/ShuffleNet inside; RepVGG miss. afterok mild **20412530 / 531 / 532 COMPLETED**. Wave K. §37. |
| **20412391 / 392 / 393** | Similar FLOP-floor look-ahead serial s42→s43→s44 | **20412391 COMPLETED** 4 d 16 h 45 m (ended 26 Aug 01:23 cluster). s42 catalog COMPLETED PRELIM. **20412392 COMPLETED** 3 d 14 h 08 m (ended 27 Aug 21:11 cluster). DenseNet TEST **−2.1 @ 0.798/0.700** (0.949→0.928) two-seed **−2.6 / −2.1** same size inside. Catalog PRELIM: r20 **−5.9 @ 0.908/0.701**; r56 **−9.2 @ 0.952/0.702** (FLOP-floor-only s43 was **−14.3 miss**); r44 **−4.4 @ 0.947/0.702**; VGG **−2.7 @ 0.804/0.701**; MobileNet **−3.1 @ 0.933/0.700**. r32 skipped. **20412393 COMPLETED** 3 d 14 h 32 m (ended 31 Aug 11:43 cluster). s44 DenseNet TEST **−2.8 @ 0.798/0.700** (0.949→0.921) three-seed **−2.6 / −2.1 / −2.8** same size inside. Catalog COMPLETED PRELIM: r20 **−5.2 @ 0.908/0.701** three-seed **−4.6 / −5.9 / −5.2**; r56 **−8.6 @ 0.952/0.702** three-seed **−8.3 / −9.2 / −8.6**; r44 **−4.4 @ 0.947/0.702** three-seed **−4.2 / −4.4 / −4.4**; VGG **−2.8 @ 0.804/0.701** three-seed **−3.6 / −2.7 / −2.8**; MobileNet **−3.2 @ 0.933/0.700** (0.938→0.906) three-seed **−3.3 / −3.1 / −3.2**. r32 skipped. Children ImageNet s42 **20715875** + prefer-greedy **20715876** PD (`QOSMaxGRESPerUser`; CG **20382185** still holds a GRES). Wave O already on **20412540**. Do not jump Wave Q. §39. |
| **20412394 / 395 / 396** | Unlike FLOP-floor look-ahead serial | **20412394 COMPLETED** 2 d 12 h 21 m (ended 5 Sep 22:04 cluster). s42 catalog COMPLETED PRELIM all four unlike nets inside τ. Keep first-pass: ShuffleNet×1 **−1.7 @ 0.801/0.716**; RepVGG-A0 **−6.9 @ 0.847/0.701**; A1 **−6.7 @ 0.857/0.702**; ×1.5 **−2.5 @ 0.771/0.701** (quote structural; skip masked 0.724). Child **20412395 COMPLETED** 11 h 17 m (ended 6 Sep 12:01 cluster). s43 catalog COMPLETED PRELIM: ×1 **−2.0 @ 0.801/0.716**; A0 **−6.6 @ 0.847/0.701**; A1 **−5.7 @ 0.857/0.702**; ×1.5 **−2.6 @ 0.771/0.701** (0.932→0.906) — all **same size** as s42. Two-seed ×1.5 **−2.5 / −2.6**. FLAGS neon prefer=0. Freed GPU → **20884671** (age beat nice-0 child). Child **20412396 PD** (`QOSMaxGRESPerUser`). §47. |
| **20412530 / 531 / 532 → 20412533 / 534 / 536** | C100 Adam-40 mild then random s42/s43/s44 | **20412530 / 531 / 532 COMPLETED** (s42 12 h 15 m, ended 26 Aug 13:37 cluster). Mild catalogs COMPLETED PRELIM: r20 three-seed miss; r56 three-seed miss (not a look-ahead cliff); VGG three-seed **−8.0 / −8.3 / −8.0 @ 0.811/0.822** inside; ShuffleNet three-seed **−4.0 / −4.4 / −4.3 @ 0.860/0.835** inside (quote structural 0.860); RepVGG three-seed **−11.8 / −11.5 / −11.8 @ 0.684/0.548** miss. afterok random **20412533 COMPLETED** 2 d 9 h 19 m (ended 28 Aug 22:57 cluster). Catalog COMPLETED PRELIM: r20 **−18.0 @ 0.676/0.579** miss unmatched; r56 **−26.2 @ 0.628/0.349** miss unmatched three-seed **−26.2 / −26.4 / −30.3**; VGG **−9.3 @ 0.698/0.718** inside unmatched three-seed **−9.3 / −8.5 / −8.5**; ShuffleNet **−4.2 @ 0.785/0.747** inside unmatched three-seed **−4.2 / −5.5 / −4.4** (quote structural 0.785; do not quote masked 0.760); RepVGG-A0 **−13.4 @ 0.662/0.498** (0.753→0.619) miss unmatched three-seed **−13.4 / −12.8 / −12.6**. Designed leaf — do not attach Wave O. Idle GPU filled by **20382188**, not Wave Q. **20412534** s43 **COMPLETED** catalog PRELIM: r20 **−17.0** miss; r56 **−26.4** miss; VGG **−8.5** inside; ShuffleNet **−5.5 @ 0.741/0.732** inside; RepVGG **−12.8 @ 0.677/0.513** miss. **20412536** s44 **COMPLETED** catalog PRELIM: r20 **−17.3** miss unmatched; r56 **−30.3** miss unmatched; VGG **−8.5 @ 0.714/0.695** inside unmatched; ShuffleNet **−4.4 @ 0.762/0.741** inside unmatched (quote structural 0.762); RepVGG **−12.6 @ 0.664/0.498** (0.753→0.627) miss unmatched vs s43 **−12.8**. §38 / §40. |
| **20412538 / 540 / 542 → 20412545 / 546 / 548** | Similar FLOP-floor mild then random serial | **20412538 COMPLETED** 3 d 10 h 54 m (ended 30 Aug 04:51 cluster). Catalog COMPLETED PRELIM: r20 **−5.6 @ 0.801/0.706** inside; r56 **−11.6 @ 0.946/0.701** miss; r44 **−3.9 @ 0.905/0.703** inside; VGG **−2.8 @ 0.811/0.819** inside (floor did not bind); MobileNet **−2.4 @ 0.791/0.703** inside (floor bound); DenseNet **−2.2 @ 0.823/0.828** (0.949→0.927) inside (floor did not bind). r32 skipped. **20412540 COMPLETED** 2 d 18 h 58 m (ended 4 Sep 04:22 cluster). s43 catalog COMPLETED PRELIM: r20 **−5.1**; r56 **−9.9** inside (s42 miss same size); r44 **−3.6**; VGG **−3.0**; MobileNet **−2.3**; DenseNet **−2.1 @ 0.823/0.828** (0.949→0.928) two-seed **−2.2 / −2.1** same size. r32 skipped. Child spoof **20884670 PD QOS**. s44 **20412542** afterok **20884672**. Wave O. §43. |
| **20412549–551 → 20412552–554** | Unlike FLOP-floor mild then random serial | **PENDING** afterok unlike FLOP-floor look-ahead s44. Wave P. |
| **20412555 / 556 / 557** | Similar FLOP+prefer look-ahead greedy s42/s43/s44 | **20412555 PD** (`QOSMaxGRESPerUser`; parent **20412385 COMPLETED**; dep released). Wave Q. Prefer-point heuristic. Do not jump. Do not submit duplicates. |
| **20715868 / 870 / 871 → 20715872 / 873 / 874** | Similar then unlike FLOP-floor greedy (L1, no look-ahead) serial | **20715868 COMPLETED** 2 d 1 h 36 m (ended 4 Sep 14:00 cluster). s42 catalog COMPLETED PRELIM: r20 **−4.8**; r56 **−9.2** inside; r44 **−4.6**; VGG **−3.1**; MobileNet **−3.1**; DenseNet **−2.2 @ 0.798/0.700**. **20715870 COMPLETED** 22 h 27 m (ended 5 Sep 15:38 cluster). s43 catalog COMPLETED PRELIM: r20 **−5.7**; r56 **−9.3** inside; r44 **−5.1**; VGG **−2.7**; MobileNet **−3.0**; DenseNet **−2.2 @ 0.798/0.700** (0.949→0.927) — **same size** as s42. Skip r32. Child **20715871 RUNNING** (~13 h 48 m). s44 TESTs: r20 **−5.1**; r56 **−9.0**; r44 **−4.1**; VGG **−2.5 @ 0.804/0.701** — three-seed same size **−3.1 / −2.7 / −2.5**. Now MobileNet `eval_test` (not FINAL). Wave R. §51. |
| **20715875** | ImageNet MobileNet-v2 s42 retry | **PENDING** (`QOSMaxGRESPerUser`; parent **20412393 COMPLETED**). Needs `rtx_4090` (Nice=10000). The ~08:32 GRES went to 2080-safe jobs, not this one. No ImageNet DRL. |
| **20715876 / 877 / 878** | Similar FLOP+prefer greedy (L1, no look-ahead) | **20715876 COMPLETED** 1 d 21 h 30 m (ended 3 Sep 06:02 cluster, `dt-2080-18`). DenseNet TEST **−2.1 @ 0.870/0.951** (0.949→0.928) inside, size-matched near-tie vs DRL prefer **−2.0**. Catalog COMPLETED PRELIM: r20 **−3.9**; r56 **−4.4**; r44 **−2.6**; VGG **−2.9**; MobileNet **−2.0**; DenseNet **−2.1**. r32 skipped. Child **20715877 PD** (`QOSMaxGRESPerUser`; GRES went to **20382189**). Then s44 **20715878**. §48. |
| **20715879** | Unlike FLOP+prefer look-ahead s42 | **PENDING** afterok C100 recoverable s44 **20884675** (rechain 3 Sep; was afterok **20382187**). |

**PC cadence (4 Sep – freeze 15 Sep):** 6 Sep 00:34: PC **open** (VPN back after SSH out 20:45–00:31). **5 R.** Shaped **20967060 RUNNING**. FLOP-greedy s44 **20715871 R**. Unlike look-ahead s44 **20382198 R**. **20884671** still PD (nice 50). Wave Q niced — do not jump. Commute ~08:15: afterok survives the laptop.

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

## 32. C10-thin eval τ=5 (frozen 10-net actor) — PRELIM

Same frozen actors as C4. Skip-train `input_c10_thin.json`. Eval τ=5 instead of default τ=10. **Quote TEST.** Do not quote job-mean −0.08 / −0.05 pp. Do not quote eval_train.

| Net | τ=10 DRL s42 (§5) | τ=5 s42 (20382193) | τ=5 s43 (20382194) | τ=5 s44 (20382195) |
|---|---|---|---|---|
| thin r20-w2 | −4.2 @ 0.600/0.760 | **−4.4 @ 0.600/0.748** | **−4.4 @ 0.600/0.780** | **−2.5 @ 0.600/0.794** |
| thin r56-w4 | −15.9 @ 0.704/0.550 | **−23.9 @ 0.667/0.481** | **−20.1 @ 0.593/0.475** | **−21.0 @ 0.685/0.490** |

Easy net stays at 60% params. Seeds 42/43 **−4.4** miss the eval τ=5 budget (inside τ=10). Seed 44 **−2.5** is inside τ=5 (0.648→0.623). Hard net **three-seed cliff** at unmatched greedy / L2-s44 sizes (s44 **−21.0 @ 0.685/0.490**, 0.888→0.678 — same size as L2/SVD s44 **−23.4 / −23.7**). Tightening eval τ does not replay the frozen τ=10 policy as a milder pruner on the skinny ResNet. Jobs **COMPLETED**. **Do not lock** as a matched-size τ=10 comparison. Do not expand τ=5.

---

## 33. Similar-family FLOP 0.70 + prefer Δparams/ΔFLOPs — LOCKED three-seed including DenseNet

Frozen 10-net skip-train, `eval_offline_similar`, FLOP floor 0.70, `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`. Skip akamaster r32. Sized `eval_test` with `param_ratio`. **20381785 / 786 / 787 ALL COMPLETED.**

| Net | s42 (20381785) | s43 (20381786) | s44 (20381787) | default §3 | FLOP-floor-only §29 | vs τ=10 |
|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−3.8 @ 0.717/0.880** | **−3.5 @ 0.717/0.880** | **−3.8 @ 0.717/0.880** | −5.4 / −4.5 / −5.6 | −4.4 / −4.5 / −5.3 | **LOCKED** three-seed inside |
| ResNet-56 w10 | **−3.8 @ 0.702/0.872** | **−4.0 @ 0.702/0.872** | **−4.4 @ 0.702/0.872** | −9.2 / −12.4 / −13.0 | −5.9 / **−14.3 @ 0.946/0.700** / −5.6 | **LOCKED** three-seed inside |
| ResNet-44 | **−2.3 @ 0.702/0.872** | **−2.6 @ 0.702/0.872** | **−2.6 @ 0.702/0.872** | −4.3 / −4.2 / −4.3 | −3.7 / −3.2 / −3.6 | **LOCKED** three-seed inside |
| VGG-19 BN | **−2.2 @ 0.837/0.923** | **−2.6 @ 0.837/0.923** | **−2.4 @ 0.837/0.923** | −2.7 / −2.6 / −2.7 | −2.7 / −3.0 / −2.1 | **LOCKED** three-seed inside |
| MobileNet-v2×0.75 | **−1.9 @ 0.767/0.912** | **−2.0 @ 0.767/0.912** | **−2.2 @ 0.767/0.912** | −2.5 / −2.1 / −1.6 | −2.1 / −2.1 / −2.1 | **LOCKED** three-seed inside |
| DenseNet-100 | **−2.0 @ 0.870/0.951** | **−1.9 @ 0.870/0.951** | **−2.7 @ 0.870/0.951** | −2.0 / −2.1 / −2.1 | −2.2 / −2.4 / −2.2 | **LOCKED** three-seed inside |

Same prefer point as C10-thin r56-w4 (**−8.9 / −8.0 / −8.1 @ 0.704/0.872**). Easy r20 **−3.8 / −3.5 / −3.8 @ 0.717/0.880**. Hard similar r56-w10 **−3.8 / −4.0 / −4.4 @ 0.702/0.872**. r44 **−2.3 / −2.6 / −2.6 @ 0.702/0.872**. VGG-19 **−2.2 / −2.6 / −2.4 @ 0.837/0.923**. MobileNet **−1.9 / −2.0 / −2.2 @ 0.767/0.912** (s42 0.938→0.919). DenseNet **−2.0 / −1.9 / −2.7 @ 0.870/0.951** (s42 0.949→0.929) — **same size**, **LOCKED three-seed**. FLOP-floor-only seed 43 kept 95% weights and still missed on r56-w10. Prefer is the lever. Skip r32. afterok C100 prefer s42 **20381800 COMPLETED**; **801 / 802 COMPLETED**. Child FLOP-only **20382180 COMPLETED**.

---

## 34. Similar-family look-ahead greedy — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, `SPECTRA_EVAL_LOOKAHEAD=1`. Skip akamaster r32. Sized `eval_test` with `param_ratio`. Jobs **20382177 COMPLETED** (6 d 18 h 14 m, ended 26 Aug 22:01 cluster) / **20382178 COMPLETED** (5 d 8 h 15 m, ended 25 Aug 12:19 cluster) / **20382179 COMPLETED** (2 d 17 h 56 m, ended 23 Aug 04:26 cluster).

| Net | s42 (20382177) | s43 (20382178) | s44 (20382179) | DRL prefer s44 §33 | DRL default s44 §3 | Greedy 20202684 |
|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−7.4 @ 0.713/0.525** (0.950→0.876) | **−7.0 @ 0.713/0.525** (0.950→0.880) | **−7.3 @ 0.713/0.525** | **−3.8 @ 0.717/0.880** | −5.6 @ 0.669/0.634 | −7.7 @ 0.640/0.494 |
| ResNet-56 w10 | **−22.4 @ 0.702/0.376** (0.959→0.735) | **−20.1 @ 0.702/0.376** (0.959→0.758) | **−22.2 @ 0.702/0.376** | **−4.4 @ 0.702/0.872** | −13.0 @ 0.604/0.397 | −23.1 @ 0.607/0.336 |
| ResNet-44 | **−8.0 @ 0.703/0.391** (0.935→0.855) | **−9.0 @ 0.703/0.391** (0.935→0.845) | **−8.0 @ 0.703/0.391** | **−2.6 @ 0.702/0.872** | −4.3 @ 0.635/0.582 | −9.0 @ 0.614/0.353 |
| VGG-19 BN | **−3.5 @ 0.703/0.669** (0.934→0.899) | **−3.2 @ 0.703/0.669** (0.934→0.902) | **−2.5 @ 0.703/0.669** | **−2.4 @ 0.837/0.923** | −2.7 @ 0.814/0.796 | −3.0 @ 0.669/0.661 |
| MobileNet-v2×0.75 | **−3.2 @ 0.708/0.511** (0.938→0.906) | **−3.3 @ 0.708/0.511** (0.938→0.905) | **−3.0 @ 0.708/0.511** | **−2.2 @ 0.767/0.912** | −1.6 @ 0.706/0.684 | −3.5 @ 0.662/0.494 |
| DenseNet-100 | **−2.4 @ 0.701/0.679** (0.949→0.925) | **−2.6 @ 0.701/0.679** (0.949→0.923) | **−2.6 @ 0.701/0.679** (0.949→0.923) | **−2.7 @ 0.870/0.951** | −2.1 @ 0.834/0.851 | — |

r20 three-seed **same size** **−7.4 / −7.0 / −7.3 @ 0.713/0.525** (s42 0.950→0.876) inside τ. r56 three-seed **same size** **−22.4 / −20.1 / −22.2 @ 0.702/0.376** **cliffs** (tracks greedy −23.1). r44 three-seed **same size** **−8.0 / −9.0 / −8.0 @ 0.703/0.391** (s42 0.935→0.855) stays **inside τ** and tracks greedy Δacc (−9.0) at a larger net. Look-ahead keeps ~70% params. Easy r20 stays **inside τ** at 52% FLOPs (prefer 88%). Hard r56 cuts FLOPs to 38% vs prefer 87%. Easy VGG three-seed **same size** **−3.5 / −3.2 / −2.5 @ 0.703/0.669** (s42 0.934→0.899) inside τ. MobileNet three-seed **same size** **−3.2 / −3.3 / −3.0 @ 0.708/0.511** (s42 0.938→0.906) inside τ at a harder FLOP cut than prefer. DenseNet three-seed **same size** **−2.4 / −2.6 / −2.6 @ 0.701/0.679** (s42 0.949→0.925) **inside τ**. Catalogs COMPLETED. FLOP-floor look-ahead s42 r56 **−8.3 @ 0.952/0.702** inside (§39) — the floor stopped this cliff. FLOP-floor look-ahead s42 MobileNet **−3.3 @ 0.933/0.700** inside vs unconstrained three-seed **−3.2 / −3.3 / −3.0 @ 0.708/0.511**. Mild s44 r20 **−6.8 @ 0.669/0.649** (§42). **Do not lock.**

---

## 35. C100 C9 Adam-40 FLOP 0.70 + prefer — LOCKED mixed (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_c100`, FLOP floor 0.70, `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`, Adam-40 C9 recipe. Same actors as §21. **Do not overwrite §21.** Quote `eval_test` only. Jobs **20381800 COMPLETED** (14 h 33 m, ended 24 Aug 02:51 cluster) / **20381801 COMPLETED** / **20381802 COMPLETED**. Child FLOP-only **20382180 COMPLETED**.

| Net | C9 default s43 §21 | prefer s42 (20381800) | prefer s43 (20381801) | prefer s44 (20381802) | SGD prefer s43 §25 | vs τ=10 |
|---|---|---|---|---|---|---|
| thin r20-w16 | −17.1 @ 0.647/0.619 miss | **−14.5 @ 0.716/0.879** (0.730→0.585) | **−13.8 @ 0.716/0.880** (0.730→0.592) | **−12.7 @ 0.716/0.880** (0.730→0.603) | **−7.9 @ 0.716/0.879** inside | three-seed miss |
| thin r56-w15 | −17.8 @ 0.682/0.491 miss | **−12.0 @ 0.703/0.872** (0.784→0.664) | **−11.4 @ 0.703/0.872** (0.784→0.670) | **−10.5 @ 0.703/0.872** (0.784→0.679) | **−4.5 @ 0.703/0.872** inside | three-seed miss |
| VGG-16 BN | **−7.8 @ 0.816/0.784** inside | **−7.6 @ 0.838/0.931** (0.740→0.664) | **−7.7 @ 0.838/0.931** (0.740→0.663) | **−7.3 @ 0.838/0.931** (0.740→0.667) | (SGD table is residuals-only) | three-seed inside |
| ShuffleNet-v2×1 | **−3.4 @ 0.819/0.823** inside | **−3.5 @ 0.873/0.944** (0.726→0.691) | **−4.1 @ 0.873/0.944** (0.726→0.685) | **−4.6 @ 0.873/0.944** (0.726→0.680) | (SGD table is residuals-only) | three-seed inside |
| RepVGG-A0 | **−10.6 @ 0.686/0.540** miss | **−13.0 @ 0.719/0.756** (0.753→0.623) | **−12.5 @ 0.719/0.756** (0.753→0.628) | **−13.0 @ 0.719/0.756** (0.753→0.623) | (SGD table is residuals-only) | three-seed miss |

r20 three-seed **same size** still **misses τ** (−14.5 / −13.8 / −12.7). r56 three-seed **same size** **−12.0 / −11.4 / −10.5 @ 0.703/0.872** also **misses τ** (beats default −17.8; SGD prefer on the same net is **−4.5 inside**). VGG three-seed **same size** **−7.6 / −7.7 / −7.3 @ 0.838/0.931** **inside τ**. ShuffleNet three-seed **same size** **−3.5 / −4.1 / −4.6 @ 0.873/0.944** **inside τ** (milder FLOPs than default −3.4 @ 0.819/0.823). RepVGG three-seed **same size** **−13.0 / −12.5 / −13.0 @ 0.719/0.756** (s42 0.753→0.623) **misses τ**. Prefer is not an Adam-40 residual/RepVGG rescue. Same family split as §21: VGG and ShuffleNet recover, thin residuals and RepVGG miss. Recipe remains the residual lever. FLOP-only control is §36. **LOCKED mixed.** Do not quote mid-FT or train-loader. Do not overwrite §21.

---

## 36. C100 C9 Adam-40 FLOP floor 0.70 only (no prefer) — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_c100`, FLOP floor 0.70, **no** `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP`. Adam-40 C9 recipe. Same actors as §21. Control for §35 prefer. **Do not overwrite §21.** Quote `eval_test` FINAL only. Jobs **20382180 COMPLETED** (1 d 10 h 42 m, ended 25 Aug 13:34 cluster) / **20382181 COMPLETED** / **20382182 COMPLETED** (7 h 00 m, ended 22 Aug 15:40 cluster). Sizes **not** matched across seeds. afterok look-ahead **20412388 COMPLETED**.

| Net | C9 default s43 §21 | FLOP-only s42 (20382180) | FLOP-only s43 (20382181) | FLOP-only s44 (20382182) | vs τ=10 |
|---|---|---|---|---|---|
| thin r20-w16 | −17.1 @ 0.647/0.619 miss | **−11.8 @ 0.827/0.700** (0.730→0.612) | **−12.1 @ 0.781/0.701** (0.730→0.609) | **−10.6 @ 0.831/0.707** (0.730→0.624) | three-seed miss unmatched |
| thin r56-w15 | −17.8 @ 0.682/0.491 miss | **−10.4 @ 0.910/0.702** (0.784→0.680) | **−11.7 @ 0.898/0.703** (0.784→0.667) | **−12.0 @ 0.950/0.703** (0.784→0.664) | three-seed miss unmatched |
| VGG-16 BN | **−7.8 @ 0.816/0.784** inside | **−8.4 @ 0.908/0.869** (0.740→0.656) | **−8.4 @ 0.909/0.848** (0.740→0.656) | **−7.7 @ 0.817/0.852** (0.740→0.663) | three-seed inside unmatched |
| ShuffleNet-v2×1 | **−3.4 @ 0.819/0.823** inside | **−3.4 @ 0.881/0.858** (0.726→0.692) | **−3.9 @ 0.849/0.824** (0.726→0.687) | **−4.0 @ 0.897/0.852** (0.726→0.686) | three-seed inside unmatched |
| RepVGG-A0 | −10.6 @ 0.686/0.540 miss | **−10.2 @ 0.886/0.761** (0.753→0.651) | **−8.2 @ 0.950/0.788** (0.753→0.671) | **−7.4 @ 0.956/0.801** (0.753→0.679) | not three-seed inside |

Quote ShuffleNet **structural** 0.881 (s42 masked 0.859). s42 r20 **−11.8 @ 0.827/0.700** (0.730→0.612) **misses τ**. Tracks two-seed **−12.1 / −10.6** (unmatched sizes: 82.7% / 78.1% / 83.1% params at ~70% FLOPs). s42 r56 **−10.4 @ 0.910/0.702** (0.784→0.680) **misses τ**. Three-seed unmatched **−10.4 / −11.7 / −12.0** (91.0% / 89.8% / 95.0% params at ~70% FLOPs). s42 VGG **−8.4 @ 0.908/0.869** (0.740→0.656) **inside τ**. Three-seed unmatched **−8.4 / −8.4 / −7.7** (90.8% / 90.9% / 81.7% params). s42 ShuffleNet **−3.4 @ 0.881/0.858** (0.726→0.692) **inside τ**. Three-seed unmatched **−3.4 / −3.9 / −4.0**. Hitting the FLOP floor without prefer does **not** rescue Adam-40 residuals. Prefer r56 was **−12.0 / −11.4 / −10.5 @ 0.703/0.872** — also a miss, at fewer weights / more FLOPs. VGG stays **inside** at both this point and prefer (§35, **−7.6 / −7.7 / −7.3 @ 0.838/0.931**). RepVGG s43/s44 were **inside only** at a **tiny param cut** (~95% weights / 79–80% FLOPs); s42 cut to **88.6%/76.1%** and **missed** (−10.2). Prefer **−13.0 / −12.5 / −13.0 @ 0.719/0.756** still misses. Default s44 RepVGG was **−12.6 @ 0.570/0.449**. Prefer is a different operating point, not a residual/RepVGG rescue under Adam-40. Recipe remains the residual lever. Same family split as §21 on VGG/ShuffleNet; RepVGG is not a FLOP-only three-seed rescue. Look-ahead control is §37. **Do not lock.** Do not overwrite §21.

---

## 37. C100 C9 Adam-40 look-ahead greedy — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_c100`, `SPECTRA_EVAL_LOOKAHEAD=1`, Adam-40 C9 recipe. Same actors as §21. Heuristic at the default stop, not FLOP+prefer. **Do not overwrite §21.** Quote `eval_test` FINAL only. Jobs **20412388 COMPLETED** (11 h 47 m, ended 26 Aug 01:21 cluster) / **20412389 COMPLETED** (11 h 41 m, ended 23 Aug 00:25 cluster) / **20412390 COMPLETED** (11 h 41 m, ended 23 Aug 03:21 cluster). afterok mild **20412530 / 531 / 532 COMPLETED**.

| Net | C9 default s43 §21 | look-ahead s42 (20412388) | look-ahead s43 (20412389) | look-ahead s44 (20412390) | prefer s43 §35 | vs τ=10 |
|---|---|---|---|---|---|---|
| thin r20-w16 | −17.1 @ 0.647/0.619 miss | **−18.1 @ 0.716/0.525** (0.730→0.549) | **−17.3 @ 0.716/0.525** (0.730→0.557) | **−17.9 @ 0.716/0.525** (0.730→0.551) | **−13.8 @ 0.716/0.880** miss | three-seed miss same size |
| thin r56-w15 | −17.8 @ 0.682/0.491 miss | **−34.6 @ 0.701/0.357** (0.784→0.438) | **−32.1 @ 0.701/0.357** (0.784→0.463) | **−33.2 @ 0.701/0.357** (0.784→0.452) | **−11.4 @ 0.703/0.872** miss | three-seed cliff same size |
| VGG-16 BN | **−7.8 @ 0.816/0.784** inside | **−8.8 @ 0.701/0.672** (0.740→0.652) | **−8.6 @ 0.701/0.672** (0.740→0.654) | **−9.0 @ 0.701/0.672** (0.740→0.650) | **−7.7 @ 0.838/0.931** inside | three-seed inside same size |
| ShuffleNet-v2×1 | **−3.4 @ 0.819/0.823** inside | **−6.2 @ 0.736/0.682** (0.726→0.664) | **−6.4 @ 0.736/0.682** (0.726→0.662) | **−5.9 @ 0.736/0.682** (0.726→0.667) | **−4.1 @ 0.873/0.944** inside | three-seed inside same size |
| RepVGG-A0 | −10.6 @ 0.686/0.540 miss | **−11.4 @ 0.709/0.577** (0.753→0.639) | **−11.7 @ 0.709/0.577** (0.753→0.636) | **−12.1 @ 0.709/0.577** (0.753→0.632) | **−12.5 @ 0.719/0.756** miss | three-seed miss same size |

Quote ShuffleNet **structural** 0.736 (s42 masked 0.705; s43 masked 0.704; s44 masked 0.706). r20 three-seed **same size** **−18.1 / −17.3 / −17.9 @ 0.716/0.525** (s42 0.730→0.549) **misses τ**. r56 three-seed **same size** **−34.6 / −32.1 / −33.2 @ 0.701/0.357** (s42 0.784→0.438) **cliffs**. VGG three-seed **same size** **−8.8 / −8.6 / −9.0 @ 0.701/0.672** (s42 0.740→0.652) **inside τ** at 70%/67%. ShuffleNet three-seed **same size** **−6.2 / −6.4 / −5.9 @ 0.736/0.682** (s42 0.726→0.664) **inside τ** at a harder cut than prefer. RepVGG three-seed **same size** **−11.4 / −11.7 / −12.1 @ 0.709/0.577** (s42 0.753→0.639) still **misses** (Δacc ≈ prefer, fewer FLOPs). Same family split as §21: VGG/ShuffleNet recover, residuals and RepVGG do not. Look-ahead is **not** an Adam-40 residual/RepVGG rescue. Catalogs COMPLETED. Mild control is §38. **Do not lock.** Do not overwrite §21.

---

## 38. C100 C9 Adam-40 mild — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_c100`, same-loop **mild** rate-picker at the default stop (not FLOP+prefer). Adam-40 C9 recipe. Same actors as §21. **Do not overwrite §21.** Quote `eval_test` FINAL only. Jobs **20412530 COMPLETED** (12 h 15 m, ended 26 Aug 13:37 cluster) / **20412531 COMPLETED** / **20412532 COMPLETED** (12 h 03 m, ended 23 Aug 15:24 cluster). afterok random **20412533 / 534 / 536 COMPLETED** (designed leaves — do not attach Wave O).

| Net | C9 default s43 §21 | mild s42 (20412530) | mild s43 (20412531) | mild s44 (20412532) | look-ahead s43 §37 | prefer s43 §35 | vs τ=10 |
|---|---|---|---|---|---|---|---|
| thin r20-w16 | −17.1 @ 0.647/0.619 miss | **−17.0 @ 0.669/0.649** (0.730→0.560) | **−16.0 @ 0.669/0.649** (0.730→0.570) | **−16.1 @ 0.669/0.649** (0.730→0.569) | −17.3 @ 0.716/0.525 miss | **−13.8 @ 0.716/0.880** miss | three-seed miss same size |
| thin r56-w15 | −17.8 @ 0.682/0.491 miss | **−18.2 @ 0.689/0.499** (0.784→0.602) | **−16.8 @ 0.689/0.499** (0.784→0.616) | **−17.7 @ 0.689/0.499** (0.784→0.607) | −32.1 @ 0.701/0.357 cliff | **−11.4 @ 0.703/0.872** miss | three-seed miss same size |
| VGG-16 BN | **−7.8 @ 0.816/0.784** inside | **−8.0 @ 0.811/0.822** (0.740→0.660) | **−8.3 @ 0.811/0.822** (0.740→0.657) | **−8.0 @ 0.811/0.822** (0.740→0.660) | **−8.6 @ 0.701/0.672** inside | **−7.7 @ 0.838/0.931** inside | three-seed inside same size |
| ShuffleNet-v2×1 | **−3.4 @ 0.819/0.823** inside | **−4.0 @ 0.860/0.835** (0.726→0.686) | **−4.4 @ 0.860/0.835** (0.726→0.682) | **−4.3 @ 0.860/0.835** (0.726→0.683) | **−6.4 @ 0.736/0.682** inside | **−4.1 @ 0.873/0.944** inside | three-seed inside same size |
| RepVGG-A0 | −10.6 @ 0.686/0.540 miss | **−11.8 @ 0.684/0.548** (0.753→0.635) | **−11.5 @ 0.684/0.548** (0.753→0.638) | **−11.8 @ 0.684/0.548** (0.753→0.635) | **−11.7 @ 0.709/0.577** miss | **−12.5 @ 0.719/0.756** miss | three-seed miss same size |

Quote ShuffleNet **structural** 0.860 (masked 0.837). r20 three-seed **same size** **−17.0 / −16.0 / −16.1 @ 0.669/0.649** (s42 0.730→0.560) **misses τ**. r56 three-seed **same size** **−18.2 / −16.8 / −17.7 @ 0.689/0.499** (s42 0.784→0.602) **misses τ** — tracks default, **not** a look-ahead-style cliff (−32). VGG three-seed **same size** **−8.0 / −8.3 / −8.0 @ 0.811/0.822** (s42 0.740→0.660) **inside τ**. ShuffleNet three-seed **same size** **−4.0 / −4.4 / −4.3 @ 0.860/0.835** (s42 0.726→0.686) **inside τ**. RepVGG three-seed **same size** **−11.8 / −11.5 / −11.8 @ 0.684/0.548** (s42 0.753→0.635) **misses**. Same family split as §21. Not an Adam-40 residual/RepVGG rescue. Catalogs COMPLETED. afterok random **20412533 COMPLETED** — designed leaf, do not attach Wave O. **Do not lock.** Do not overwrite §21.

---

## 39. Similar-family FLOP floor 0.70 + look-ahead greedy — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, FLOP floor 0.70, look-ahead greedy (no prefer). Serial s42→s43→s44. Quote `eval_test` FINAL only. Skip akamaster r32. Job **20412391 COMPLETED** (4 d 16 h 45 m, ended 26 Aug 01:23 cluster). **20412392 COMPLETED** (3 d 14 h 08 m, ended 27 Aug 21:11 cluster). **20412393 COMPLETED** (3 d 14 h 32 m, ended 31 Aug 11:43 cluster).

| Net | s42 (20412391) | s43 (20412392) | s44 (20412393) | FLOP-floor only §29 | look-ahead §34 | vs τ=10 |
|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−4.6 @ 0.908/0.701** (0.950→0.904) | **−5.9 @ 0.908/0.701** (0.950→0.891) | **−5.2 @ 0.908/0.701** (0.950→0.898) | s42 **−4.4 @ 0.768/0.707**; s43 **−4.5 @ 0.717/0.738** | **−7.4 @ 0.713/0.525** | three-seed inside same size |
| ResNet-56 w10 | **−8.3 @ 0.952/0.702** (0.959→0.876) | **−9.2 @ 0.952/0.702** (0.959→0.867) | **−8.6 @ 0.952/0.702** (0.959→0.873) | s42 **−5.9** inside; s43 **−14.3 @ 0.946/0.700** miss | **−22.4 @ 0.702/0.376** cliff | three-seed inside same size |
| ResNet-44 | **−4.2 @ 0.947/0.702** (0.935→0.893) | **−4.4 @ 0.947/0.702** (0.935→0.891) | **−4.4 @ 0.947/0.702** (0.935→0.891) | — | **−8.0 @ 0.703/0.391** | three-seed inside same size |
| VGG-19 BN | **−3.6 @ 0.804/0.701** (0.934→0.898) | **−2.7 @ 0.804/0.701** (0.934→0.907) | **−2.8 @ 0.804/0.701** (0.934→0.906) | **−2.7 @ 0.767/0.755** | **−3.5 / −3.2 / −2.5 @ 0.703/0.669** | three-seed inside same size |
| MobileNet-v2×0.75 | **−3.3 @ 0.933/0.700** (0.938→0.905) | **−3.1 @ 0.933/0.700** (0.938→0.907) | **−3.2 @ 0.933/0.700** (0.938→0.906) | **−2.1 @ 0.777/0.701** | **−3.2 / −3.3 / −3.0 @ 0.708/0.511** | three-seed inside same size |
| DenseNet-100 | **−2.6 @ 0.798/0.700** (0.949→0.923) | **−2.1 @ 0.798/0.700** (0.949→0.928) | **−2.8 @ 0.798/0.700** (0.949→0.921) | **−2.2 @ 0.837/0.833** | three-seed **−2.4 / −2.6 / −2.6 @ 0.701/0.679** | three-seed inside same size |

s44 DenseNet **same size** as s42/s43 **−2.6 / −2.1 / −2.8 @ 0.798/0.700** (s44 0.949→0.921) **inside τ** (PASS1 `acc 0.949 -> 0.921 (-0.028) | params x0.798 | FLOPs x0.700`). Floor **did** bind vs unconstrained look-ahead **−2.4 / −2.6 / −2.6 @ 0.701/0.679**. s44 slightly worse Δacc than s42/s43; still the easy net. FLOP-floor-only was **−2.2 @ 0.837/0.833**. MobileNet three-seed **−3.3 / −3.1 / −3.2 @ 0.933/0.700** (s44 0.938→0.906) **inside τ** same size (PASS1 `acc 0.938 -> 0.906 (-0.032) | params x0.933 | FLOPs x0.700`). Floor **did** bind vs unconstrained look-ahead **−3.2 / −3.3 / −3.0 @ 0.708/0.511**. VGG three-seed **−3.6 / −2.7 / −2.8 @ 0.804/0.701** (s44 0.934→0.906) **inside τ** same size (unconstrained look-ahead **−3.5 / −3.2 / −2.5 @ 0.703/0.669**; FLOP-floor mild s42 was **−2.8 @ 0.811/0.819** — floor did not bind on mild). r44 three-seed **−4.2 / −4.4 / −4.4 @ 0.947/0.702** (s44 0.935→0.891) **inside τ** same size (unconstrained look-ahead **−8.0 @ 0.703/0.391**). r56 three-seed **−8.3 / −9.2 / −8.6 @ 0.952/0.702** (s44 0.959→0.873) **inside τ** same size (FLOP-floor-only s43 was **−14.3 miss**; unconstrained look-ahead **cliffed** at 38% FLOPs). r20 three-seed **−4.6 / −5.9 / −5.2 @ 0.908/0.701** (s44 0.950→0.898) **inside τ** same size. r32 skipped — do not quote. s42/s43/s44 catalogs **COMPLETED PRELIM**. Children **20715875** / **20715876** PD (`QOSMaxGRESPerUser`). **Do not lock.** Do not quote wrap job-mean or eval_train.

---

## 40. C100 C9 Adam-40 random — PRELIM (s42/s43/s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_c100`, same-loop **random** rate-picker at the default stop (not FLOP+prefer). Adam-40 C9 recipe. Same actors as §21. **Do not overwrite §21.** Quote `eval_test` FINAL only. Jobs **20412533 COMPLETED** (2 d 9 h 19 m, ended 28 Aug 22:57 cluster) / **20412534 COMPLETED** (11 h 43 m, ended 24 Aug 00:22 cluster) / **20412536 COMPLETED** (11 h 32 m, ended 24 Aug 02:56 cluster). Wave N leaves — do not attach Wave O.

| Net | C9 default s43 §21 | random s42 (20412533) | random s43 (20412534) | random s44 (20412536) | mild s43 §38 | look-ahead s43 §37 | vs τ=10 |
|---|---|---|---|---|---|---|---|
| thin r20-w16 | −17.1 @ 0.647/0.619 miss | **−18.0 @ 0.676/0.579** (0.730→0.550) | **−17.0 @ 0.687/0.551** (0.730→0.560) | **−17.3 @ 0.640/0.616** (0.730→0.557) | **−16.0 @ 0.669/0.649** miss | −17.3 @ 0.716/0.525 miss | three-seed miss unmatched |
| thin r56-w15 | −17.8 @ 0.682/0.491 miss | **−26.2 @ 0.628/0.349** (0.784→0.522) | **−26.4 @ 0.633/0.370** (0.784→0.520) | **−30.3 @ 0.580/0.347** (0.784→0.481) | **−16.8 @ 0.689/0.499** miss | −32.1 @ 0.701/0.357 cliff | three-seed miss unmatched |
| VGG-16 BN | **−7.8 @ 0.816/0.784** inside | **−9.3 @ 0.698/0.718** (0.740→0.647) | **−8.5 @ 0.728/0.766** (0.740→0.655) | **−8.5 @ 0.714/0.695** (0.740→0.655) | **−8.3 @ 0.811/0.822** inside | **−8.6 @ 0.701/0.672** inside | three-seed inside unmatched |
| ShuffleNet-v2×1 | **−3.4 @ 0.819/0.823** inside | **−4.2 @ 0.785/0.747** (0.726→0.684) | **−5.5 @ 0.741/0.732** (0.726→0.671) | **−4.4 @ 0.762/0.741** (0.726→0.682) | **−4.4 @ 0.860/0.835** inside | **−6.4 @ 0.736/0.682** inside | three-seed inside unmatched |
| RepVGG-A0 | −10.6 @ 0.686/0.540 miss | **−13.4 @ 0.662/0.498** (0.753→0.619) | **−12.8 @ 0.677/0.513** (0.753→0.625) | **−12.6 @ 0.664/0.498** (0.753→0.627) | **−11.5 @ 0.684/0.548** miss | **−11.7 @ 0.709/0.577** miss | three-seed miss unmatched |

Quote ShuffleNet **structural** sizes (s42 0.785 masked 0.760; s43 0.741 masked 0.715; s44 0.762 masked 0.734). s42/s43/s44 catalogs **COMPLETED**. Same family split as default / mild / look-ahead: VGG and ShuffleNet **inside τ**; residuals and RepVGG miss. Random r20 three-seed **−18.0 / −17.0 / −17.3** unmatched, all **miss τ**, tracks default **−17.1**. Random r56 three-seed **−26.2 / −26.4 / −30.3** unmatched, all **miss τ**, all **worse than default −17.8** — not a look-ahead-style cliff (−32). VGG three-seed unmatched **−9.3 / −8.5 / −8.5** (s42 0.740→0.647), all **inside τ**. ShuffleNet three-seed unmatched **−4.2 / −5.5 / −4.4** (s42 0.726→0.684), all **inside τ**. RepVGG three-seed unmatched **−13.4 / −12.8 / −12.6** (s42 0.753→0.619), all **miss τ**. Random is **not** an Adam-40 residual/RepVGG rescue. Mild catalogs COMPLETED PRELIM (§38). Designed leaves — do not attach Wave O. Idle GPU from **20412533** went to similar-random s43 **20382188**, not Wave Q. **Do not lock.** Do not overwrite §21.

---

## 41. Frozen 10-net → ImageNet MobileNet-v2 (truncated JPEG) — PRELIM two-seed unmatched

Frozen 10-net skip-train actors **s43 `job20163257`** / **s44 `job20164515`**. Eval catalog: torchvision MobileNet-v2 ImageNet-1k, **truncated-JPEG** loader. **No ImageNet DRL train.** Quote `eval_test` FINAL / `pass 1/1` only. Do **not** quote 82.8%, train-loader, mid-FT, or `best_loss`. This is Gilad’s frozen-transfer probe, **not** a home-court SOTA fight vs DepGraph / OCS / SACP on ImageNet.

Job **20360208 COMPLETED** 4 d 10 h 25 m (ended 25 Aug 01:57 cluster). s42 **20318168 TIMEOUT** — no TEST. s44 **20382192 COMPLETED** 5 d 3 h 52 m (ended 30 Aug 05:49 cluster). afterany released unlike FLOP-floor look-ahead **20412394 RUNNING** (Wave M).

| Net | s42 (20318168) | s43 (20360208) | s44 (20382192) | vs τ=10 |
|---|---|---|---|---|
| MobileNet-v2 ImageNet | TIMEOUT, no TEST | **−4.6 @ 0.823/0.729** (0.719→0.673) | **−5.1 @ 0.772/0.652** (0.719→0.668) | two-seed inside unmatched |

Origin **71.9%** is this job’s unpruned `eval_test` on the truncated-JPEG loader (checkpoint name 71.88). Do not mix it with a standard ImageNet val number from another paper. s44 PASS1 `acc 0.719 -> 0.668 (-0.051) | params x0.772 | FLOPs x0.652`. s44 cut more than s43 (77%/65% vs 82%/73%) and lost 0.5 pp more. Both **inside τ**. Layer-wise 3-ep FT is the SPECTRA eval recipe, not DRL training. Two seeds, unmatched sizes. s42 TIMEOUT. **Do not lock.** Do not start ImageNet DRL.

---

## 42. Similar-family mild — PRELIM (s42/s44 catalogs COMPLETED; s43 SCANCEL wrap)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **mild** rate-picker at the default stop. Skip akamaster r32. Quote `eval_test` FINAL only. Jobs **20382184 COMPLETED** 5 d 22 h 23 m (ended 31 Aug 21:49 cluster) / **20382185 SCANCEL** 30 Aug 06:12 cluster (zombie wrap after Run finished 28 Aug 19:19; DenseNet **−2.1** already TESTed) / **20382186 COMPLETED** (5 d 16 h 39 m, ended 28 Aug 21:04 cluster).

| Net | s42 (20382184) | s43 (20382185) | s44 (20382186) | look-ahead §34 | DRL default s44 §3 | Mild 20202691 | vs τ=10 |
|---|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−5.9 @ 0.669/0.649** (0.950→0.891) | **−5.7 @ 0.669/0.649** (0.950→0.893) | **−6.8 @ 0.669/0.649** (0.950→0.882) | **−7.3 @ 0.713/0.525** | −5.6 @ 0.669/0.634 | **−5.5 @ 0.669/0.649** | three-seed inside |
| ResNet-56 w10 | **−14.0 @ 0.661/0.421** (0.959→0.819) | **−12.8 @ 0.661/0.421** (0.959→0.831) | **−13.9 @ 0.661/0.421** (0.959→0.820) | **−22.2 @ 0.702/0.376** | −13.0 @ 0.604/0.397 | **−12.6 @ 0.661/0.421** | three-seed miss |
| ResNet-44 | **−4.7 @ 0.699/0.542** (0.935→0.888) | **−4.1 @ 0.699/0.542** (0.935→0.894) | **−4.2 @ 0.699/0.542** (0.935→0.893) | **−8.0 @ 0.703/0.391** | −4.3 @ 0.635/0.582 | **−4.3 @ 0.699/0.542** | three-seed inside |
| VGG-19 BN | **−3.3 @ 0.811/0.819** (0.934→0.901) | **−2.6 @ 0.811/0.819** (0.934→0.908) | **−3.5 @ 0.811/0.819** (0.934→0.899) | **−2.5 @ 0.703/0.669** | −2.7 @ 0.814/0.796 | **−2.4 @ 0.811/0.819** | three-seed inside |
| MobileNet-v2×0.75 | **−2.3 @ 0.689/0.666** (0.938→0.915) | **−2.1 @ 0.689/0.666** (0.938→0.917) | **−2.3 @ 0.689/0.666** (0.938→0.915) | **−3.0 @ 0.708/0.511** | −1.6 @ 0.706/0.684 | **−1.7 @ 0.689/0.666** | three-seed inside |
| DenseNet-100 | **−2.4 @ 0.823/0.828** (0.949→0.925) | **−2.1 @ 0.823/0.828** (0.949→0.928) | **−2.3 @ 0.823/0.828** (0.949→0.926) | **−2.6 @ 0.701/0.679** | −2.1 @ 0.834/0.851 | — | three-seed inside same size |

s44 r20 **same size** as unmatched one-seed mild 20202691 (0.669/0.649). Three-seed **−5.9 / −5.7 / −6.8** **inside τ** (s42 0.950→0.891). s43 closer to the older mild (−5.5); s42 tracks default DRL (−5.6 @ 0.669/0.634); s44 slightly worse; milder than look-ahead three-seed **−7.4 / −7.0 / −7.3 @ 0.713/0.525**. s44 r56 **same size** as 20202691 (0.661/0.421). Three-seed **−14.0 / −12.8 / −13.9** all **miss τ** (plus unmatched 20202691 **−12.6**; s42 0.959→0.819). Tracks DRL default s44 **−13.0** at a slightly smaller DRL net. Far milder than look-ahead cliff **−22.2 @ 0.702/0.376** and greedy **−23.1 @ 0.607/0.336**. Mild is **not** a τ rescue on the hard similar net. s44 r44 **same size** as 20202691 **−4.3 @ 0.699/0.542**. Three-seed **−4.7 / −4.1 / −4.2** **inside τ** (s42 0.935→0.888). s42 slightly worse Δacc than s43/s44; tracks older mild **−4.3**; milder than look-ahead **−8.0 @ 0.703/0.391**. s44 VGG **same size** as 20202691 **−2.4 @ 0.811/0.819**. Three-seed **−3.3 / −2.6 / −3.5** **inside τ** (s42 0.934→0.901). s43 closer to the older mild; s42 slightly worse Δacc than s43; s44 worst of the three, still inside. Floor did **not** bind (same 81%/82% as FLOP-floor mild s42 **−2.8**). s44 MobileNet **same size** as 20202691 **−1.7 @ 0.689/0.666**. Three-seed **−2.3 / −2.1 / −2.3** (s42 0.938→0.915) **inside τ** (s43 closer to the older mild; s42/s44 match; milder FLOP cut than look-ahead **−3.0 @ 0.708/0.511**; FLOP-floor mild s42 was **−2.4 @ 0.791/0.703** — floor bound). s44 DenseNet **same size** as s43 **−2.1 / −2.3 @ 0.823/0.828** (s44 0.949→0.926) **inside τ**. s42 DenseNet **−2.4 @ 0.823/0.828** (0.949→0.925) three-seed **−2.4 / −2.1 / −2.3** same size **inside τ** (PASS1 `acc 0.949 -> 0.925 (-0.024) | params x0.823 | FLOPs x0.828`, cluster 21:48:40). Floor did **not** bind (FLOP-floor mild s42 **−2.2** same size). Look-ahead was **−2.6 @ 0.701/0.679**; default DRL s44 **−2.1 @ 0.834/0.851**. Mild keeps more params than look-ahead at the same Δacc as default. s42 and s44 catalogs **COMPLETED PRELIM**. afterok similar-random **20382187** PD (`QOSMaxGRESPerUser`). s43 last-net TEST landed, job still wrapping. r32 skipped — do not quote. Do not quote wrap job-mean. **Do not lock.**

---

## 43. Similar-family FLOP-floor mild — PRELIM (s42 and s43 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **mild** rate-picker at the **70% FLOP floor** (same stop as FLOP-floor DRL / FLOP-floor look-ahead). Skip akamaster r32. Quote `eval_test` FINAL only. Job **20412538 COMPLETED** 3 d 10 h 54 m (ended 30 Aug 04:51 cluster). s43 **20412540 COMPLETED** 2 d 18 h 58 m (ended 4 Sep 04:22 cluster, `dt-2080-13`). Scientific parent FLOP-floor look-ahead s42 **20412391 COMPLETED**. Wave O. Child spoof **20884670 PD QOS**; FLOP-mild s44 **20412542** still afterok arm-A residual **20884672**.

| Net | FLOP-floor mild s42 (20412538) | FLOP-floor mild s43 (20412540) | unconstrained mild s43 §42 | FLOP-floor look-ahead s43 §39 | vs τ=10 |
|---|---|---|---|---|---|
| ResNet-20 w16 | **−5.6 @ 0.801/0.706** (0.950→0.894) | **−5.1 @ 0.801/0.706** (0.950→0.899) | **−5.7 @ 0.669/0.649** inside | **−5.9 @ 0.908/0.701** inside | two-seed inside same size |
| ResNet-56 w10 | **−11.6 @ 0.946/0.701** (0.959→0.843) | **−9.9 @ 0.946/0.701** (0.959→0.860) | **−12.8 @ 0.661/0.421** miss | **−9.2 @ 0.952/0.702** inside | two-seed split same size: s42 miss / s43 inside |
| ResNet-44 | **−3.9 @ 0.905/0.703** (0.935→0.896) | **−3.6 @ 0.905/0.703** (0.935→0.899) | **−4.1 @ 0.699/0.542** inside | **−4.4 @ 0.947/0.702** inside | two-seed inside same size |
| VGG-19 BN | **−2.8 @ 0.811/0.819** (0.934→0.906) | **−3.0 @ 0.811/0.819** (0.934→0.904) | **−2.6 @ 0.811/0.819** inside | **−2.7 @ 0.804/0.701** inside | two-seed inside same size |
| MobileNet-v2×0.75 | **−2.4 @ 0.791/0.703** (0.938→0.914) | **−2.3 @ 0.791/0.703** (0.938→0.915) | **−2.1 @ 0.689/0.666** inside | **−3.1 @ 0.933/0.700** inside | two-seed inside same size |
| DenseNet-100 | **−2.2 @ 0.823/0.828** (0.949→0.927) | **−2.1 @ 0.823/0.828** (0.949→0.928) | **−2.1 @ 0.823/0.828** inside | **−2.1 @ 0.798/0.700** inside | two-seed inside same size |

s42 r20 **inside τ** at 80% params / 71% FLOPs. s43 r20 **inside τ** at the **same size** (PASS1 `acc 0.950 -> 0.899 (-0.051) | params x0.801 | FLOPs x0.706`, cluster 20:37:31). Two-seed **−5.6 / −5.1 @ 0.801/0.706**. Floor **did** bind vs unconstrained mild **−5.7 @ 0.669/0.649**. s42 r56 **miss τ** at 95% params / 70% FLOPs. s43 r56 **inside τ** at the **same size** (PASS1 `acc 0.959 -> 0.860 (-0.099) | params x0.946 | FLOPs x0.701`, cluster 22:43:04). Two-seed **−11.6 / −9.9 @ 0.946/0.701** — split, not a three-seed rescue. Floor **did** bind vs unconstrained mild s43 **−12.8 @ 0.661/0.421** miss. Near FLOP-floor look-ahead s43 **−9.2 @ 0.952/0.702** inside. DRL FLOP-floor-only s43 was **−14.3 @ 0.946/0.700** miss at this size; prefer is the lever (**−3.8 / −4.0 / −4.4 @ 0.702/0.872**). s43 r44 **inside τ** at the **same size** as s42 (PASS1 `acc 0.935 -> 0.899 (-0.036) | params x0.905 | FLOPs x0.703`, cluster 00:03:14). Two-seed **−3.9 / −3.6 @ 0.905/0.703**. Floor **did** bind vs unconstrained mild s43 **−4.1 @ 0.699/0.542**. FLOP-floor look-ahead s43 **−4.4 @ 0.947/0.702** kept more params. Skip r32 (PASS1 01:04 — broken origin_acc). s43 VGG **inside τ** at the **same size** as s42 (PASS1 `acc 0.934 -> 0.904 (-0.030) | params x0.811 | FLOPs x0.819`, cluster 02:12:30). Two-seed **−2.8 / −3.0 @ 0.811/0.819**. Floor **did not** bind vs unconstrained mild s43 **−2.6 @ 0.811/0.819**. FLOP-floor look-ahead s43 **−2.7 @ 0.804/0.701**. s43 MobileNet **inside τ** at the **same size** as s42 (PASS1 `acc 0.938 -> 0.915 (-0.023) | params x0.791 | FLOPs x0.703`, cluster 06:59:11). Two-seed **−2.4 / −2.3 @ 0.791/0.703**. Floor **did** bind vs unconstrained mild s43 **−2.1 @ 0.689/0.666**. FLOP-floor look-ahead s43 **−3.1 @ 0.933/0.700**. s43 DenseNet **inside τ** at the **same size** as s42 (PASS1 `acc 0.949 -> 0.928 (-0.021) | params x0.823 | FLOPs x0.828`, cluster 04:21:00). Two-seed **−2.2 / −2.1 @ 0.823/0.828**. Floor **did not** bind vs unconstrained mild s43 **−2.1 @ 0.823/0.828**. FLOP-floor look-ahead s43 **−2.1 @ 0.798/0.700**. Catalog **COMPLETED PRELIM**. Skip r32. **Do not lock.**

---

## 44. Unlike-family look-ahead greedy — PRELIM (s42 catalog COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead greedy** rate-picker (no extra FLOP floor). Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20382196 COMPLETED** 2 d 14 h 19 m (ended 29 Aug 12:49 cluster). Scientific parent similar look-ahead catalogs COMPLETED. Wave H. s43/s44 still afterok on similar-random. afterok unlike mild **20412380 COMPLETED**.

| Net | look-ahead s42 (20382196) | default unlike s42 §4 | FLOP-floor unlike s42 §28 | vs τ=10 |
|---|---|---|---|---|
| ShuffleNet-v2×1 | **−2.2 @ 0.723/0.682** (0.924→0.902) | **−1.1 @ 0.800/0.825** | **−1.9 @ 0.809/0.826** | one-seed inside |
| ShuffleNet-v2×1.5 | **−2.5 @ 0.710/0.675** (0.932→0.907) | **−2.4 @ 0.818/0.801** | **−2.0 @ 0.821/0.799** | one-seed inside |
| RepVGG-A0 | **−7.2 @ 0.709/0.577** (0.943→0.871) | **−4.8 @ 0.681/0.565** | **−3.9 @ 0.792/0.702** | one-seed inside |
| RepVGG-A1 | **−6.3 @ 0.710/0.574** (0.944→0.881) | **−4.7 @ 0.650/0.521** | **−4.3 @ 0.888/0.735** | one-seed inside |

s42 ShuffleNet-v2×1 **inside τ** at a harder structural cut than default unlike (72%/68% vs 80%/83%). PASS1 also printed `effective-params x0.688` (masked zeros) — **do not quote** that as the size. Same-loop greedy historically crashed on ShuffleNet grouping; look-ahead produced a TEST. s42 ShuffleNet-v2×1.5 **inside τ** at **−2.5 @ 0.710/0.675** (PASS1 `acc 0.932 -> 0.907 (-0.025) | params x0.710 | FLOPs x0.675`). PASS1 also printed `effective-params x0.665` — **do not quote**. Default unlike **−2.4 @ 0.818/0.801** already inside; FLOP-floor unlike **−2.0 @ 0.821/0.799**. Similar Δacc at a harder structural cut. s42 RepVGG-A0 **inside τ** at **−7.2 @ 0.709/0.577** (0.943→0.871). Default unlike was **−4.8 @ 0.681/0.565** (already inside); FLOP-floor unlike **−3.9 @ 0.792/0.702**. s42 RepVGG-A1 **inside τ** at **−6.3 @ 0.710/0.574**. Default unlike A1 **−4.7 @ 0.650/0.521** (already inside); FLOP-floor unlike **−4.3 @ 0.888/0.735**. Look-ahead is **worse Δacc** at a similar size on both RepVGGs — not a new transfer win. All four unlike nets **inside τ** one seed. Catalog **COMPLETED PRELIM**. afterok unlike mild **20412380 COMPLETED** catalog PRELIM (§45). **Do not lock.**

---

## 45. Unlike-family mild — PRELIM (s42 catalog COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **mild** rate-picker at the default stop. Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20412380 COMPLETED** 1 d 1 h 18 m (ended 30 Aug 14:08 cluster). Scientific parent unlike look-ahead s42 **20382196 COMPLETED**. Wave I. s43/s44 still afterok. Child unlike-random **20412385** PD on QOS (CG zombie **20382185**). Do **not** quote the wrap job-mean **−0.02 pp** or `eval_train`.

| Net | mild s42 (20412380) | look-ahead s42 §44 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.6 @ 0.857/0.835** (0.924→0.908) | **−2.2 @ 0.723/0.682** | **−1.1 @ 0.800/0.825** | one-seed inside |
| RepVGG-A0 | **−5.3 @ 0.680/0.548** (0.943→0.890) | **−7.2 @ 0.709/0.577** | **−4.8 @ 0.681/0.565** | one-seed inside |
| RepVGG-A1 | **−4.2 @ 0.663/0.539** (0.944→0.902) | **−6.3 @ 0.710/0.574** | **−4.7 @ 0.650/0.521** | one-seed inside |
| ShuffleNet-v2×1.5 | **−2.6 @ 0.849/0.828** (0.932→0.906) | **−2.5 @ 0.710/0.675** | **−2.4 @ 0.818/0.801** | one-seed inside |

s42 ShuffleNet-v2×1 **inside τ** at **−1.6 @ 0.857/0.835**. PASS1 also printed `effective-params x0.831` — **do not quote**. Look-ahead cut harder (**−2.2 @ 0.723/0.682**); default unlike **−1.1 @ 0.800/0.825** already inside. Unmatched sizes. s42 RepVGG-A0 **inside τ** at **−5.3 @ 0.680/0.548** (0.943→0.890) — almost the default unlike size (0.681/0.565, **−4.8**); look-ahead is **worse Δacc** at a similar size (**−7.2 @ 0.709/0.577**). s42 RepVGG-A1 **inside τ** at **−4.2 @ 0.663/0.539** (0.944→0.902) vs default **−4.7 @ 0.650/0.521** and look-ahead **−6.3 @ 0.710/0.574**. s42 ShuffleNet-v2×1.5 **inside τ** at **−2.6 @ 0.849/0.828** (0.932→0.906). PASS1 also printed `effective-params x0.818` — **do not quote**. Default unlike **−2.4 @ 0.818/0.801**; look-ahead **−2.5 @ 0.710/0.675**. All four unlike nets **inside τ** one seed. Catalog **COMPLETED PRELIM**. **Do not lock.**

---

## 46. Similar-family random — PRELIM (s42 / s43 / s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **random** rate-picker at the default stop (not FLOP+prefer). Skip akamaster r32. Quote `eval_test` FINAL only. Job **20382188 COMPLETED** (`Restarts=2`). **20382187 COMPLETED** 3 d 0 h 31 m (ended 4 Sep 09:02 cluster, `dt-2080-08`). **20382189 COMPLETED** 2 d 15 h 32 m (ended 5 Sep 21:34 cluster, `Restarts=2`). Wave E. Do not overwrite unmatched one-seed random **20202692** in §6.

| Net | random s42 (20382187) | random s43 (20382188) | random s44 (20382189) | older random 20202692 §6 | mild s43 §42 | look-ahead s43 §34 | vs τ=10 |
|---|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−6.4 @ 0.688/0.588** (0.950→0.886) | **−6.7 @ 0.691/0.606** (0.950→0.883) | **−7.4 @ 0.680/0.570** (0.950→0.876) | **−6.0 @ 0.688/0.603** | **−5.7 @ 0.669/0.649** | **−7.0 @ 0.713/0.525** | three-seed inside unmatched |
| ResNet-56 w10 | **−18.2 @ 0.577/0.347** (0.959→0.777) | **−14.9 @ 0.628/0.368** (0.959→0.810) | **−16.6 @ 0.690/0.388** (0.959→0.793) | **−17.2 @ 0.622/0.357** | **−12.8 @ 0.661/0.421** miss | **−20.1 @ 0.702/0.376** cliff | three-seed miss unmatched |
| ResNet-44 | **−6.2 @ 0.590/0.415** (0.935→0.873) | **−6.2 @ 0.589/0.437** (0.935→0.873) | **−7.8 @ 0.648/0.403** (0.935→0.857) | **−6.2 @ 0.589/0.408** | **−4.1 @ 0.699/0.542** | **−9.0 @ 0.703/0.391** | three-seed inside unmatched |
| VGG-19 BN | **−3.5 @ 0.756/0.754** (0.934→0.899) | **−3.1 @ 0.755/0.744** (0.934→0.903) | **−3.1 @ 0.741/0.719** (0.934→0.903) | **−2.6 @ 0.714/0.716** | **−2.6 @ 0.811/0.819** | **−3.2 @ 0.703/0.669** | three-seed inside unmatched |
| MobileNet-v2×0.75 | **−2.7 @ 0.697/0.554** (0.938→0.911) | **−2.4 @ 0.692/0.581** (0.937→0.913) restart | **−2.9 @ 0.694/0.615** (0.938→0.909) restart | **−2.4 @ 0.698/0.583** | **−2.1 @ 0.689/0.666** | **−3.3 @ 0.708/0.511** | three-seed inside unmatched |
| DenseNet-100 | **−2.3 @ 0.730/0.755** (0.949→0.926) | **−2.3 @ 0.735/0.754** (0.949→0.926) restart | **−2.5 @ 0.740/0.745** (0.949→0.924) restart | **−2.2 @ 0.754/0.747** | **−2.1 @ 0.823/0.828** | **−2.6 @ 0.701/0.679** | three-seed inside unmatched |

s42 **20382187** first-pass r20 **inside τ** at 69% params / 59% FLOPs (PASS1 `acc 0.950 -> 0.886 (-0.064) | params x0.688 | FLOPs x0.588`, cluster 22:34:30). Unmatched vs first-pass s43 **−6.7 @ 0.691/0.606** / s44 **−7.4 @ 0.680/0.570**; near older random **−6.0 @ 0.688/0.603**. Do not replace first-pass s43/s44 rows. s42 first-pass r56 **miss τ** at 58% params / 35% FLOPs (PASS1 `acc 0.959 -> 0.777 (-0.182) | params x0.577 | FLOPs x0.347`, cluster 02:14:05). Harder cut than first-pass s43 **−14.9 @ 0.628/0.368** / s44 **−16.6 @ 0.690/0.388**; near older random **−17.2 @ 0.622/0.357** and s43 restart **−17.6 @ 0.622/0.356**. Three-seed miss unmatched **−18.2 / −14.9 / −16.6**. Not a look-ahead cliff (**−20.1 @ 0.702/0.376**) and **not** a τ rescue. s42 first-pass r44 **inside τ** at 59% params / 42% FLOPs (PASS1 `acc 0.935 -> 0.873 (-0.062) | params x0.590 | FLOPs x0.415`, cluster 04:28:28). Same Δacc as first-pass s43 **−6.2 @ 0.589/0.437**; s44 **−7.8 @ 0.648/0.403**. Three-seed inside unmatched **−6.2 / −6.2 / −7.8**. Near older random **−6.2 @ 0.589/0.408**. Skipped r32 (do not quote). s42 first-pass VGG **inside τ** at 76% params / 75% FLOPs (PASS1 `acc 0.934 -> 0.899 (-0.035) | params x0.756 | FLOPs x0.754`, cluster 07:05:33). Unmatched vs first-pass s43 **−3.1 @ 0.755/0.744** / s44 **−3.1 @ 0.741/0.719** (0.4 pp worse Δacc; params nearly the same as s43). Three-seed unmatched **−3.5 / −3.1 / −3.1**. Near look-ahead s43 **−3.2 @ 0.703/0.669**. s42 first-pass MobileNet **inside τ** at 70% params / 55% FLOPs (PASS1 `acc 0.938 -> 0.911 (-0.027) | params x0.697 | FLOPs x0.554`, cluster ~12:20). Unmatched vs s43 restart **−2.4 @ 0.692/0.581** (0.3 pp worse Δacc; fewer FLOPs). Near older random **−2.4 @ 0.698/0.583**. Milder Δacc than look-ahead s43 **−3.3 @ 0.708/0.511**. s42 first-pass DenseNet **inside τ** at 73% params / 76% FLOPs (PASS1 `acc 0.949 -> 0.926 (-0.023) | params x0.730 | FLOPs x0.755`, cluster 09:01:49). Same Δacc as s43 restart **−2.3 @ 0.735/0.754**; near older random **−2.2 @ 0.754/0.747**. Mild s43 **−2.1 @ 0.823/0.828** kept more params. Look-ahead s43 **−2.6 @ 0.701/0.679**. Catalog **COMPLETED PRELIM** (first pass). Skip r32. s43 **20382188 COMPLETED** 1 d 1 h 40 m (ended 2 Sep 12:24 cluster). Restart DenseNet **inside τ** at 74% params / 75% FLOPs (PASS1 `acc 0.949 -> 0.926 (-0.023) | params x0.735 | FLOPs x0.754`, cluster 12:23:58). Near older random **−2.2 @ 0.754/0.747**. Mild s43 **−2.1 @ 0.823/0.828** kept more params. Look-ahead s43 **−2.6 @ 0.701/0.679**. Table r20/r56/r44/VGG for s43/s44 is the **first pass** (keep those rows). Restart (`Restarts=2`) re-TESTed r20 **−6.9 @ 0.673/0.538**; r56 **−17.6 @ 0.622/0.356** miss; r44 **−6.9 @ 0.637/0.433**; VGG **−2.6 @ 0.700/0.731**; first MobileNet **−2.4 @ 0.692/0.581**; DenseNet **−2.3 @ 0.735/0.754**. r32 skip (do not quote). Catalog COMPLETED PRELIM (restart). s44 **20382189 COMPLETED** 2 d 15 h 32 m (ended 5 Sep 21:34 cluster). Restart DenseNet **inside τ** at 74% params / 75% FLOPs (PASS1 `acc 0.949 -> 0.924 (-0.025) | params x0.740 | FLOPs x0.745`, cluster 21:33:27). Keep first-pass table rows for r20/r56/r44/VGG. Restart also: r20 **−7.2 @ 0.665/0.517**; r56 **−15.3 @ 0.628/0.369** miss; r44 **−6.7 @ 0.646/0.414**; VGG **−2.7 @ 0.750/0.764**; MobileNet **−2.9 @ 0.694/0.615**; DenseNet **−2.5 @ 0.740/0.745**. r32 skip. Catalog COMPLETED PRELIM (restart). Child unlike look-ahead **20382198 RUNNING**. Child C100 recoverable s43 **20884673 RUNNING**. Random is **not** a look-ahead cliff and **not** a τ rescue on hard similar r56. **Do not lock.**

---

## 47. Unlike-family FLOP floor 0.70 + look-ahead greedy — PRELIM (s42 + s43 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead greedy** under eval FLOP floor 0.70. Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20412394 COMPLETED** 2 d 12 h 21 m (ended 5 Sep 22:04 cluster, `Restarts=2`). Wave M. Scientific parent ImageNet s44 **20382192 COMPLETED**. Child **20412395 COMPLETED** 11 h 17 m (ended 6 Sep 12:01 cluster, `ise-pheno-02`). FLAGS `SPECTRA_REWARD_MODE=neon` `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0`. Child **20412396 PD** (`QOSMaxGRESPerUser`; GRES went to **20884671**).

| Net | FLOP-floor look-ahead s42 / s43 (20412394 / 20412395) | unconstrained look-ahead s42 §44 | FLOP-floor unlike DRL s42 §28 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.7 / −2.0 @ 0.801/0.716** (s42 0.924→0.907; s43 0.924→0.904) | **−2.2 @ 0.723/0.682** | **−1.9 @ 0.809/0.826** | **−1.1 @ 0.800/0.825** | two-seed inside |
| RepVGG-A0 | **−6.9 / −6.6 @ 0.847/0.701** (s42 0.943→0.874; s43 0.943→0.877) | **−7.2 @ 0.709/0.577** | **−3.9 @ 0.792/0.702** | **−4.8 @ 0.681/0.565** | two-seed inside |
| RepVGG-A1 | **−6.7 / −5.7 @ 0.857/0.702** (s42 0.944→0.877; s43 0.944→0.887) | **−6.3 @ 0.710/0.574** | **−4.3 @ 0.888/0.735** | **−4.7 @ 0.650/0.521** | two-seed inside |
| ShuffleNet-v2×1.5 | **−2.5 / −2.6 @ 0.771/0.701** (s42 0.932→0.907; s43 0.932→0.906) | **−2.5 @ 0.710/0.675** | **−2.0 @ 0.821/0.799** | **−2.4 @ 0.818/0.801** | two-seed inside |

s42 ShuffleNet-v2×1 **inside τ** at 80% params / 72% FLOPs (PASS1 `acc 0.924 -> 0.907 (-0.017) | params x0.801 | FLOPs x0.716`, cluster 22:49:07). PASS1 also printed `effective-params x0.763` — **do not quote**. Floor **did** bind vs unconstrained look-ahead **−2.2 @ 0.723/0.682**. s43 **20412395** ShuffleNet-v2×1 TEST **−2.0 @ 0.801/0.716** (PASS1 `acc 0.924 -> 0.904 (-0.020) | params x0.801 | FLOPs x0.716`). **Same size** as s42 first-pass **−1.7** and matches s42 restart **−2.0**. PASS1 also printed `effective-params x0.763` — **do not quote**. Look-ahead is deterministic; 0.3 pp vs first-pass is fine-tune noise (§54), not a seed effect. Two-seed **−1.7 / −2.0 @ 0.801/0.716**. s42 RepVGG-A0 **inside τ** at 85% params / 70% FLOPs (PASS1 `acc 0.943 -> 0.874 (-0.069) | params x0.847 | FLOPs x0.701`, cluster 02:50:24). Floor **did** bind vs unconstrained look-ahead **−7.2 @ 0.709/0.577** (kept more params and FLOPs; similar Δacc). FLOP-floor DRL unlike **−3.9 @ 0.792/0.702** is better Δacc at the same FLOP point. Default unlike already **−4.8 @ 0.681/0.565**. s42 RepVGG-A1 **inside τ** at 86% params / 70% FLOPs (PASS1 `acc 0.944 -> 0.877 (-0.067) | params x0.857 | FLOPs x0.702`, cluster 08:21:50). Floor **did** bind vs unconstrained look-ahead **−6.3 @ 0.710/0.574** (kept more params and FLOPs; similar Δacc). FLOP-floor DRL unlike **−4.3 @ 0.888/0.735** is better Δacc. Default unlike already **−4.7 @ 0.650/0.521**. Not a new transfer win. Keep first-pass table. Restart (`Restarts=2`) re-TESTed ShuffleNet-v2×1 **−2.0 @ 0.801/0.716** (0.924→0.904) — **same size** as first-pass **−1.7**; 0.3 pp is inside §54 resampling noise. Restart also printed `effective-params x0.762` — **do not quote**. Restart re-TESTed RepVGG-A0 **−6.9 @ 0.847/0.701** (0.943→0.874) — **same size and Δacc** as first-pass (look-ahead is deterministic). Restart re-TESTed RepVGG-A1 **−6.6 @ 0.857/0.702** (0.944→0.878) — **same size** as first-pass **−6.7** (0.1 pp). Now ShuffleNet-v2×1.5 **inside τ** at 77% params / 70% FLOPs (PASS1 `acc 0.932 -> 0.907 (-0.025) | params x0.771 | FLOPs x0.701`, cluster 22:04). Quote **structural** keep. PASS1 also printed `effective-params x0.724` — **do not quote**. Floor **did** bind vs unconstrained look-ahead **−2.5 @ 0.710/0.675** (same Δacc, more params kept). FLOP-floor unlike DRL **−2.0 @ 0.821/0.799**; default unlike **−2.4 @ 0.818/0.801**. s42 catalog **COMPLETED PRELIM** all four unlike nets inside τ one seed. s43 **20412395** RepVGG-A0 TEST **−6.6 @ 0.847/0.701** (PASS1 `acc 0.943 -> 0.877 (-0.066)`, cluster 08:54). **Same size** as s42 **−6.9**; 0.3 pp is fine-tune noise. Two-seed **−6.9 / −6.6**. s43 RepVGG-A1 TEST **−5.7 @ 0.857/0.702** (PASS1 `acc 0.944 -> 0.887 (-0.057)`, cluster 10:06). **Same size** as s42 **−6.7**; 1.0 pp is fine-tune noise (look-ahead does not sample the actor). Two-seed **−6.7 / −5.7**. DRL FLOP-floor unlike is still better Δacc on both RepVGGs. s43 ShuffleNet-v2×1.5 TEST **−2.6 @ 0.771/0.701** (PASS1 `acc 0.932 -> 0.906 (-0.026) | params x0.771 | FLOPs x0.701`, cluster 12:01). **Same size** as s42 **−2.5**; 0.1 pp is fine-tune noise. Quote **structural** keep. PASS1 also printed `effective-params x0.724` — **do not quote**. Two-seed **−2.5 / −2.6 @ 0.771/0.701**. Floor **did** bind vs unconstrained look-ahead **−2.5 @ 0.710/0.675** (same Δacc, more params kept). s43 catalog **COMPLETED PRELIM** all four unlike nets inside τ two seeds. Child **20412396 PD**. **Do not lock.**

---

## 48. Similar-family FLOP 0.70 + prefer greedy (L1, no look-ahead) — PRELIM (s42 catalog COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **L1 greedy** (`SPECTRA_EVAL_POLICY=l1`, `SPECTRA_EVAL_LOOKAHEAD=0`) under FLOP floor 0.70 **and** prefer Δparams/ΔFLOPs. Quote `eval_test` FINAL only. Skip akamaster r32. Job **20715876 COMPLETED** 1 d 21 h 30 m (ended 3 Sep 06:02 cluster on `dt-2080-18`; parent **20412393 COMPLETED**). Wave S. Child **20715877 PD** (`QOSMaxGRESPerUser`; GRES went to **20382189**). Do not quote eval_train (this job prunes the whole catalog on the train loader first, then TESTs).

| Net | prefer-greedy s42 (20715876) | DRL prefer s42 §33 | look-ahead s42 §34 | unconstrained greedy 20202684 | vs τ=10 |
|---|---|---|---|---|---|
| ResNet-20 w16 | **−3.9 @ 0.717/0.880** (0.950→0.911) | **−3.8 @ 0.717/0.880** | **−7.4 @ 0.713/0.525** | **−7.7 @ 0.640/0.494** | one-seed inside, size-matched to DRL |
| ResNet-56 w10 | **−4.4 @ 0.702/0.872** (0.959→0.915) | **−3.8 @ 0.702/0.872** | **−22.4 @ 0.702/0.376** cliff | **−23.1 @ 0.607/0.336** | one-seed inside, size-matched to DRL |
| ResNet-44 | **−2.6 @ 0.702/0.872** (0.935→0.909) | **−2.3 @ 0.702/0.872** | **−8.0 @ 0.703/0.391** | **−9.0 @ 0.614/0.353** | one-seed inside, size-matched to DRL |
| VGG-19 BN | **−2.9 @ 0.837/0.923** (0.934→0.905) | **−2.2 @ 0.837/0.923** | **−3.5 @ 0.703/0.669** | **−3.0 @ 0.669/0.661** | one-seed inside, size-matched to DRL |
| MobileNet-v2×0.75 | **−2.0 @ 0.767/0.912** (0.938→0.918) | **−1.9 @ 0.767/0.912** | **−3.2 @ 0.708/0.511** | **−3.5 @ 0.662/0.494** | one-seed inside, size-matched to DRL |
| DenseNet-100 | **−2.1 @ 0.870/0.951** (0.949→0.928) | **−2.0 @ 0.870/0.951** | **−2.4 @ 0.701/0.679** | **−2.3 @ 0.700/0.679** | one-seed inside, size-matched to DRL |

s42 r20 **inside τ** at 72%/88% (cluster 08:56:29). r56 **inside τ** at 70%/87% (cluster 10:11:49) — prefer stopped the greedy cliff vs look-ahead **−22.4 @ 0.702/0.376**. r44 **inside τ** at 70%/87% (PASS1 `acc 0.935 -> 0.909 (-0.026) | params x0.702 | FLOPs x0.872`, cluster 11:08:09) — **same size** as DRL prefer **−2.3 / −2.6 / −2.6**; 0.3 pp worse than DRL s42. VGG **inside τ** at 84%/92% (PASS1 `acc 0.934 -> 0.905 (-0.029) | params x0.837 | FLOPs x0.923`, cluster 12:27:51) — **same size** as DRL prefer **−2.2 / −2.6 / −2.4**; 0.7 pp worse than DRL s42. MobileNet **inside τ** at 77%/91% (PASS1 `acc 0.938 -> 0.918 (-0.020) | params x0.767 | FLOPs x0.912`, cluster 14:11:14) — **same size** as DRL prefer **−1.9 / −2.0 / −2.2**; near-tie with DRL s42. DenseNet **inside τ** at 87%/95% (PASS1 `acc 0.949 -> 0.928 (-0.021) | params x0.870 | FLOPs x0.951`, cluster 06:01:11) — **same size** as DRL prefer **−2.0 / −1.9 / −2.7**; 0.1 pp worse than DRL s42 (near-tie). Unconstrained greedy **−2.3 @ 0.700/0.679** is a harder cut, not size-matched. r32 skip. Catalog COMPLETED one seed. Prefer sizes greedy to the DRL point on every similar net; DRL keeps a small Δacc edge on VGG/r56/r44 vs this one greedy seed; DenseNet and MobileNet are near-ties. Child **20715877** waits QOS. **Do not lock.**

---

## 49. Unlike-family random — PRELIM (s42 catalog COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **random** rate-picker at the default stop (not FLOP+prefer). Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20412385 COMPLETED** 2 d 1 h 11 m (ended 3 Sep 09:42 cluster on `dt-2080-07`). Wave J. Parent unlike mild **20412380 COMPLETED**. Child Wave Q **20412555 PD** (`QOSMaxGRESPerUser`; GRES went to **20412394**). Do not jump Q.

| Net | random s42 (20412385) | look-ahead s42 §44 | mild s42 §45 | FLOP-floor look-ahead s42 §47 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.6 @ 0.801/0.753** (0.924→0.908) | **−2.2 @ 0.723/0.682** | **−1.6 @ 0.857/0.835** | **−1.7 @ 0.801/0.716** | **−1.1 @ 0.800/0.825** | one-seed inside |
| RepVGG-A0 | **−6.2 @ 0.659/0.503** (0.943→0.881) | **−7.2 @ 0.709/0.577** | **−5.3 @ 0.680/0.548** | **−6.9 @ 0.847/0.701** | **−4.8 @ 0.681/0.565** | one-seed inside |
| RepVGG-A1 | **−5.3 @ 0.649/0.508** (0.944→0.891) | **−6.3 @ 0.710/0.574** | **−4.2 @ 0.663/0.539** | **−6.7 @ 0.857/0.702** | **−4.7 @ 0.650/0.521** | one-seed inside |
| ShuffleNet-v2×1.5 | **−2.2 @ 0.794/0.761** (0.932→0.910) | **−2.5 @ 0.710/0.675** | **−2.6 @ 0.849/0.828** | **−2.5 @ 0.771/0.701** | **−2.4 @ 0.818/0.801** | one-seed inside |

s42 ShuffleNet-v2×1 **inside τ** at 80% params / 75% FLOPs (PASS1 `acc 0.924 -> 0.908 (-0.016) | params x0.801 | FLOPs x0.753`, cluster 20:13:54). PASS1 also printed `effective-params x0.764` — **do not quote**. Same Δacc as mild **−1.6** at a harder cut. s42 RepVGG-A0 **inside τ** at 66% params / 50% FLOPs (PASS1 `acc 0.943 -> 0.881 (-0.062) | params x0.659 | FLOPs x0.503`, cluster 22:23:49). Harder cut than default **−4.8 @ 0.681/0.565** and mild **−5.3 @ 0.680/0.548**; milder Δacc than look-ahead **−7.2 @ 0.709/0.577**. Same-loop greedy was **−7.1 @ 0.654/0.484**. s42 RepVGG-A1 **inside τ** at 65% params / 51% FLOPs (PASS1 `acc 0.944 -> 0.891 (-0.053) | params x0.649 | FLOPs x0.508`, cluster 00:39:51). Nearly the default unlike size (**−4.7 @ 0.650/0.521**); 0.6 pp worse Δacc. Mild **−4.2 @ 0.663/0.539** is milder Δacc. Look-ahead **−6.3 @ 0.710/0.574** and greedy **−6.4 @ 0.639/0.477** are worse. s42 ShuffleNet-v2×1.5 **inside τ** at 79% params / 76% FLOPs (PASS1 `acc 0.932 -> 0.910 (-0.022) | params x0.794 | FLOPs x0.761`, cluster 09:41:39). PASS1 also printed `effective-params x0.755` — **do not quote**. Mild **−2.6 @ 0.849/0.828**; look-ahead **−2.5 @ 0.710/0.675**; default unlike **−2.4 @ 0.818/0.801**; FLOP-floor unlike DRL **−2.0 @ 0.821/0.799**. All four unlike nets **inside τ** one seed. Not a new transfer win — default unlike already inside. Catalog **COMPLETED PRELIM**. Child Wave Q waits QOS. **Do not lock.**

---

## 50. C100 RC — classifier-width spoof + matched-VGG train (queued 3 Sep 12:20 IDT)

Gilad 3 Sep: same family / similar size works on C10 and misses on C100. **Do not** fold unrecovered residuals into the 10-net C10 catalog (20202760 already failed). **Do not** quote train returns.

| Arm | What | Parent GPU | Job |
|---|---|---|---|
| A1 spoof | Frozen 10-net s42, C9 catalog, Adam-40, `SPECTRA_SPOOF_NUM_CLASSES=10` (tokens only). Compare to §21. | afterok **20412540** | **20884670 COMPLETED** |
| A2 matched VGG | Train VGG-16 BN C10 + VGG-16 BN C100 (`database_c10_c100_matched_vgg.json`), rates 1.0/0.9/0.8, SGD-80. | afterok A1 | **20884671 RUNNING** (~24 m, 24G, `cs-1080-02`). FLAGS `neon` seed 42. **Do not quote train.** |
| A3 residual eval | Skip-train `input_offline_c100_residuals.json`, SGD-80, A2 actor. | afterok A2 | **20884672** PD |
| B1 recoverable s43 | Clone **20307403** recipe, seed 43, VGG+ShuffleNet only. | afterok **20382187** | **20884673 RUNNING** — **do not quote train** |
| B2 residual eval s43 | Skip-train residuals, SGD-80. Compare to §31 unmatched s42. | afterok B1 | **20884674** PD |
| B3 recoverable s44 | Seed 44. | afterok B2 | **20884675** PD |

Displaced: **20412542** (mild s44) waits after A3; **20715879** (unlike prefer look-ahead) waits after B3. ImageNet **20715875** and Wave Q **20412555** untouched. Encoder has no dataset id; class count is last-Linear `out_features`. ImageNet 1000-way already transferred (C12) — spoof is the C100-specific test of that coordinate, not a claim that outputs explain ImageNet.

---

## 51. Similar-family FLOP floor 0.70 greedy (L1, no look-ahead) — PRELIM (s42 + s43 COMPLETED; s44 r20/r56/r44 TEST)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **L1 greedy** (`SPECTRA_EVAL_POLICY=l1`, `SPECTRA_EVAL_LOOKAHEAD=0`) under eval FLOP floor 0.70 (**no prefer**). Quote `eval_test` FINAL only. Skip akamaster r32. Job **20715868 COMPLETED** 2 d 1 h 36 m (ended 4 Sep 14:00 cluster). Child **20715870 COMPLETED** 22 h 27 m (ended 5 Sep 15:38 cluster). Child **20715871 RUNNING** (~13 h, `ise-pheno-09`). Wave R.

| Net | FLOP-floor greedy s42 / s43 / s44 | FLOP-floor look-ahead §39 | FLOP-floor mild s42 §43 | DRL FLOP-floor s42 §29 | unconstrained greedy 20202684 | vs τ=10 |
|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−4.8 / −5.7 / −5.1 @ 0.908/0.701** | **−4.6 / −5.9 / −5.2 @ 0.908/0.701** | **−5.6 @ 0.801/0.706** | **−4.4 @ 0.768/0.707** | **−7.7 @ 0.640/0.494** | three-seed inside, size-matched to look-ahead |
| ResNet-56 w10 | **−9.2 / −9.3 / −9.0 @ 0.952/0.702** | **−8.3 / −9.2 / −8.6 @ 0.952/0.702** | **−11.6 @ 0.946/0.701** miss | **−5.9 @ 0.902/0.702** | **−23.1 @ 0.607/0.336** cliff | three-seed inside, size-matched to look-ahead |
| ResNet-44 | **−4.6 / −5.1 / −4.1 @ 0.947/0.702** | **−4.2 / −4.4 / −4.4 @ 0.947/0.702** | **−3.9 @ 0.905/0.703** | **−3.7 @ 0.893/0.702** | **−9.0 @ 0.614/0.353** | three-seed inside, size-matched to look-ahead |
| VGG-19 BN | **−3.1 / −2.7 / −2.5 @ 0.804/0.701** | **−3.6 / −2.7 / −2.8 @ 0.804/0.701** | **−2.8 @ 0.811/0.819** | **−2.7 @ 0.767/0.755** | **−3.0 @ 0.669/0.661** | three-seed inside, size-matched to look-ahead |
| MobileNet-v2×0.75 | **−3.1 / −3.0 @ 0.933/0.700** | **−3.3 / −3.1 / −3.2 @ 0.933/0.700** | **−2.4 @ 0.791/0.703** | **−2.1 @ 0.777/0.701** | **−3.5 @ 0.662/0.494** | two-seed inside, size-matched to look-ahead |
| DenseNet-100 | **−2.2 / −2.2 @ 0.798/0.700** | **−2.6 / −2.1 / −2.8 @ 0.798/0.700** | **−2.2 @ 0.823/0.828** | **−2.2 @ 0.837/0.833** | **−2.3 @ 0.700/0.679** | two-seed inside, size-matched to look-ahead |

s42 r20 **inside τ** at 91%/70% (0.950→0.902). s43 r20 **−5.7** (0.950→0.893) **same size**. s44 **20715871** r20 **−5.1 @ 0.908/0.701** (PASS1 `acc 0.950 -> 0.899 (-0.051)`, cluster 09:12). Three-seed **−4.8 / −5.7 / −5.1 @ 0.908/0.701**. Floor **did** bind vs unconstrained greedy **−7.7 @ 0.640/0.494**. s42 r56 **inside τ** at 95%/70% (0.959→0.867). s43 r56 **−9.3** (0.959→0.866) **same size**. s44 r56 **−9.0 @ 0.952/0.702** (PASS1 `acc 0.959 -> 0.869 (-0.090)`, cluster 09:48). Three-seed **−9.2 / −9.3 / −9.0 @ 0.952/0.702** — still **inside τ**, size-matched to look-ahead three-seed **−8.3 / −9.2 / −8.6**. Unconstrained greedy **cliffed** at **−23.1**. DRL FLOP-floor s42 **−5.9 @ 0.902/0.702** is still better Δacc at a slightly smaller net. Prefer remains the lever for ~70% params. s42/s43/s44 r44 **−4.6 / −5.1 / −4.1** (s44 PASS1 `acc 0.935 -> 0.894 (-0.041)`, cluster 10:18) same size as look-ahead **−4.2 / −4.4 / −4.4**. s42/s43/s44 VGG **−3.1 / −2.7 / −2.5 @ 0.804/0.701** (s44 PASS1 `acc 0.934 -> 0.909 (-0.025)`, cluster 11:09) **same size** as look-ahead **−3.6 / −2.7 / −2.8**. r32 skipped. s42/s43 MobileNet **−3.1 / −3.0** (s43 PASS1 `acc 0.937 -> 0.907`) same size as look-ahead **−3.3 / −3.1 / −3.2**. s42/s43 DenseNet **−2.2 / −2.2 @ 0.798/0.700** (both 0.949→0.927) **same size** as FLOP-floor look-ahead **−2.6 / −2.1 / −2.8**. L1 greedy does not use the actor — Δacc spreads at identical size are fine-tune noise, not seed effects. s42 and s43 catalogs **COMPLETED PRELIM**. s44 now MobileNet `eval_test` (not FINAL). Child **20715872 PD**. **Do not lock.**

Read against §48: under FLOP floor **plus prefer**, greedy sizes to the DRL point and DRL keeps a small Δacc edge. Under the FLOP floor **alone**, greedy still **ties** look-ahead once the floor binds (two-seed greedy vs look-ahead three-seed, same sizes). The floor, not the look-ahead, is what stops the greedy cliff.

---

## 52. C100 root cause — the reward band (3 Sep, Gilad push)

**Gilad 3 Sep:** why does C10 train and C100 not, at the *same family and similar size*? Three deliverables: (1) train on C100, (2) if that fails, name the root cause — class count or agent "heaviness", (3) make sure it does not bite later.

### 52.1 The mechanism (code, not a new experiment)

`compute_reward` (`src/utils.py`) is the NEON trichotomy on `delta_acc` in **percentage points against the unpruned baseline**, with **τ = 10 pp absolute** and **no normalization** by class count, baseline accuracy, or task difficulty:

| Condition | Reward |
|---|---|
| Δacc < −τ | **−reduction³** |
| Δacc > 0 | +reduction³ |
| −τ ≤ Δacc ≤ 0 | +reduction |

Only the middle arm pays for a *cut that costs something*. If a dataset never lands in that band, every non-identity action scores −reduction³, and because the penalty is **cubic in the size of the cut**, the argmax policy is provably *"always pick rate 1.0"*. That is not a learning failure — the agent is correctly optimizing a degenerate objective. It also explains §7.3 exactly: the only C100 cells ever inside τ were at **≥98.5% params**, i.e. no real cut.

**§21 is the confirmation, already on the books.** Under the same frozen actor and the same Adam-40 recipe, the C100 families split precisely along the band boundary:

| C100 net (§21) | Δacc s42/s43/s44 | Band | Outcome |
|---|---|---|---|
| VGG-16 BN | −7.5 / −7.8 / −7.3 | **in budget** | transfers |
| ShuffleNet-v2×1 | −3.9 / −3.4 / −4.3 | **in budget** | transfers |
| thin r20-w16 | −19.3 / −17.1 / −13.6 | over budget | misses |
| thin r56-w15 | −15.0 / −17.8 / −17.6 | over budget | misses |
| RepVGG-A0 | −12.1 / −10.6 / −12.6 | over budget | misses |

The same mechanism covers the **C10** exception: skinny r56-w4 sits at −15.9 / −16.2 / −17.2, also outside the band, and is the one C10 net that misses. So "C100 is hard" and "skinny ResNets are hard" are **one** phenomenon, not two — which is a stronger paper claim than either alone.

It also re-reads the **24-net regression** (§17, r56-w4 −25.0 vs 10-net −15.9): adding nets whose every action is over budget injects constant −reduction³ into training and pushes the policy toward "never prune". Catalog size was never the lever; **band health** was.

**Consequence for the two candidate root causes Gilad named:**
- *Number of outputs* — predicts failure tracks the **dataset**. Tested by the spoof arm (**20884670**, tokens 100→10) and the matched-VGG arm (**20884671**). ImageNet MobileNet (1000-way) already transfers inside τ (§41), which is prior evidence against it.
- *Task heaviness* — predicts failure tracks the **band**, i.e. the outcome, regardless of dataset. Tested directly by §52.2.

### 52.2 Reward-band telemetry (new code, in the leap overlay)

- `utils.reward_branch()` / `utils.trace_reward()` write `reward_trace.jsonl` per step (net, dataset, rate, Δacc, nominal/realized reduction, τ, **branch**, reward) when `SPECTRA_REWARD_TRACE=1`. Off by default; `NetworkEnv.step()` is the single call site.
- `scripts/reward_band_report.py` prints the branch histogram per dataset and per net, excluding the identity rate, and names any dataset whose band is empty.
- Tests: `tests/test_reward_modes.py::test_reward_branch_labels_the_trichotomy`, `::test_reward_trace_records_branch_per_step`, `::test_reward_trace_is_off_by_default`. **23 passed on leap.**

`configs/input_reward_band_diag.json` crosses **dataset × known outcome** on four nets so the two hypotheses give different answers:

| Net | Dataset | Known §21/§5 outcome |
|---|---|---|
| VGG-16 BN | C10 | inside τ |
| VGG-16 BN | C100 | inside τ |
| thin r56-w4 | C10 | **misses** |
| thin r20-w16 | C100 | **misses** |

VGG-16 BN is the *same architecture at both class counts*, which is exactly the comparison Gilad asked for. **Read:** if empty bands track the **outcome** column, the root cause is recoverability under the FT recipe and the class count is exonerated. If they track the **dataset** column, the 100-way head is implicated and the spoof arm becomes the headline.

### 52.3 GPU carve (3 Sep 18:5x IDT)

QOS cap is 5 concurrent GPUs and all five were C10 heuristic baselines. Carve applied **without cancelling anything** (Ido's call): the three dependency-free C10 tail jobs **20382197 / 20715877 / 20412555** were niced 0 → 5000, so C100 now wins the next free GPU. Reversible with `scontrol update jobid=X nice=0`. **20412540** and **20382187** were deliberately left alone — they are the afterok parents of arms A and B, both are on their final net (DenseNet), and cancelling them would strand those arms with `DependencyNeverSatisfied`.

| Job | Arm | Gate |
|---|---|---|
| **20900185** | `diag_reward_band` s42, Adam-40, 1-day wall | **FAILED** 58 s (4 Sep 04:23). Frozen 10-net actor is 3 rates / encoder dim 48; profile asked for 5 rates / dim 52. No band report. |
| **20900187** | `c100_recoverable_drl_fine` s42 — fine ladder 1.0/0.99/0.98/0.96/0.94/0.90 + SGD-80 in loop | **COMPLETED** 1 d 20 h 21 m (ended 6 Sep 00:44 cluster). Train-from-scratch — **do not quote** in-loop `eval_test`. No afterok child (by design). Freed GPU → **20412395**. |
| **20903951** | same, `SPECTRA_REWARD_MODE=structural_shaped` | **CANCELLED** 04:23 (`afterok` parent 20900185 FAILED). |
| **20930175** | `diag_reward_band` retry, frozen menu **1.0 / 0.9 / 0.8** | **COMPLETED** 2 h 10 m (ended 4 Sep 16:10). Report: C100 band **empty** (42/42 over-budget). §55. |
| **20930177** | shaping arm (export dropped `structural_shaped`) | **CANCELLED** 5 Sep 01:22 still-PD. Replaced by **20967060**. |
| **20967060** | `c100_recoverable_drl_fine_shaped` — log FLAGS **has** `SPECTRA_REWARD_MODE=structural_shaped` | **RUNNING** `ise-6000-04` (~28 min, started when **20715870** COMPLETED). Train — do not quote. No afterok child (by design). |
| **20884670–672** | Arm A: spoof → matched VGG DRL → residual eval | **20884670 COMPLETED** 8 h 9 m (ended 4 Sep 17:11). TEST §55. Child **20884671 PD** (nice 50, behind shaped). |
| **20884673–675** | Arm B: recoverable C100 DRL s43 → residual eval → s44 | **20884673 RUNNING** `cs-1080-01` (~9 h, train ep 22 — do not quote). Child **20884674 PD** afterok. |

Three levers, each of which can re-open the band: **finer rate ladder** (some action lands inside τ), **SGD-80 + aug in the loop** (the cut becomes recoverable), **graded over-budget reward** (rank cuts even when all bust τ). The shaping arm is gated on the diagnostic so we do not change the reward before confirming the band is actually empty.

**Frozen-eval constraint (4 Sep):** skip-train jobs that load `job20158274` must keep `--compression_rates 1.0 0.9 0.8`. Extra rates rebuild the actor head and the encoder's action-cost slots (`48 → 52`) and fail `load_state_dict`. The fine ladder stays on **train** jobs (`20900187`). Spoof **20884670** already uses the 3-rate menu.

### 52.4 Not to be repeated

Do **not** fold unrecovered C100 residuals into the C10 train catalog — **20202760** and the mixed-catalog RL **20158277** already failed, and §52.1 now explains why. Do not read a DRL train return as a result. Quote `eval_test` FINAL only.

---

## 53. Train-catalog expansion — gated design (not yet submitted)

Gated on §52 producing a C100 agent that prunes. Two hard constraints from what is already LOCKED.

**Constraint 1 — admit on band health, not on count.** 3-net → 10-net helped; 10-net → 24-net *hurt* (§17). §52.1 says the discriminator is whether a candidate's actions land inside τ under the training recipe. So screen every candidate with `reward_band_report.py` first and admit only nets with a non-empty in-budget band. This turns "marginally increase the training set" into a measurable admission rule instead of a guess.

**Constraint 2 — family purity is a finite resource.** Every family put into train is spent as an *unlike-family* test family. The unlike axis is currently only {ShuffleNet-v2, RepVGG}; moving either into train collapses it to one family and weakens the generalizability claim far more than the extra train cell helps. **Therefore expand using families already in train (VGG / ResNet / DenseNet / MobileNet) at the new class count, and keep ShuffleNet-v2 and RepVGG held out — even though C100 ShuffleNet is one of the two families that currently transfers.**

Proposed 10 → 13, subject to the screen:

| Add | Why | Costs a test family? |
|---|---|---|
| VGG-16 BN C100 | LOCKED recoverable (§7.1) and inside τ (§21); VGG already in train on C10 | no |
| DenseNet-40 C100 | new dataset for an in-train family; screen first (§7.2 cut was tiny) | no |
| MobileNet-v2×1 C100 | same; screen first (val DROP in §7.2 — likely screens out) | no |

Resulting evaluation design, which is what the paper claims generalize:

| Axis | Set | Familiar? |
|---|---|---|
| Same family, unseen width/depth/source, C10 | `input_offline_similar.json` | familiar family, new instance |
| Same family, unseen instance, C100 | thin r20-w16 / r56-w15 residuals | familiar family, new dataset |
| **Unlike family, C10** | ShuffleNet-v2 ×1/×1.5, RepVGG-A0/A1 | wholly different |
| **Unlike family, C100** | ShuffleNet-v2×1, RepVGG-A0 | wholly different **and** new dataset |
| Held-out dataset, frozen | ImageNet MobileNet-v2, digit-MNIST LeNet | new dataset |

The bottom-right cell — **unlike family on the unseen dataset** — is the strongest single claim available and is currently a miss (§21 RepVGG-A0 −10.6 to −12.6). If §52 re-opens the band, that cell is the headline generalizability result. Do not grow the C10 side of the catalog; §17 already settled that.

---

## 54. Eval-path defects found in the 3-week audit (4 Sep) — provenance correction

Opus 5 code audit of the DRL agent. Two defects change how **every** TEST row in this ledger was produced. Neither is a science result; both are provenance. Fixes are in the leap overlay, default **off**, and under A/B (jobs below).

### 54.1 The frozen policy was evaluated stochastically

`evaluate_model` called `action_dist.sample()`, and the actor/critic were never put in `eval()`, so the state encoder's dropout (p=0.1) was live at TEST. The reported agent is therefore a *sample* from the policy, not the policy.

**Proof, from logs already on disk (no new GPU time).** `eval_train` and `eval_test` both reset from the pristine checkpoint and encode state from the *same* `train_loader`; only the accuracy loader differs. A deterministic policy must take the identical action sequence and land on the identical parameter count. Across all actor-policy jobs: **123 identical, 154 different**. On the three frozen 10-net actors (r32 skipped):

| Actor | Nets differing | Worst kept-param spread, same actor, same net |
|---|---|---|
| s42 `20158274` | 6 / 6 | VGG-19 **0.772 vs 0.879** (10.7 pp) |
| s43 `20163257` | 6 / 6 | r44 **0.632 vs 0.700** (6.8 pp) |
| s44 `20164515` | 6 / 6 | r44 **0.690 vs 0.635** (5.4 pp) |

Median spread ≈3.2 pp of kept params; worst 10.7 pp. Seed-to-seed "three-seed" spreads of 1–3 pp are therefore **within single-actor resampling noise** and cannot be read as seed effects. This also plausibly contributes to the r56 cliffs: one aggressive rate sampled on a narrow layer is unrecoverable.

### 54.2 The `prefer Δparams/ΔFLOPs` arms contain no agent

`fortify.action_preferring_param_per_flop` discards its `action` argument and returns an argmax over Δparams/ΔFLOPs computed from the environment alone. Whenever `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1` *and* a FLOP floor are set, the actor's output is never used.

Confirmed on disk: **20381785 / 786 / 787** (seeds 42/43/44, actors `20158274` / `20163257` / `20164515`) return **byte-identical** parameter counts on all 7 similar-family nets (0.669 / 1.049 / 0.195 / 0.329 / 0.464 / 0.236 / 17.221 M). Same for unlike **20381788 / 798 / 799** and C100 **20381800 / 801 / 802**.

**Consequence — do not quote §30, §33, §35 (and prefer-greedy §48) as three-seed DRL.** They are one *deterministic heuristic* run reported three times. They remain valid as a **same-loop heuristic** Pareto operating point (and a strong one), and should be relabelled as such: "Δparams/ΔFLOPs greedy under a FLOP floor", alongside greedy / mild / random / look-ahead. The draft sentence "Prefer is the lever" is about a heuristic, not about SPECTRA.

### 54.3 Code changes (overlay, all default-off)

| Switch | Default | What it does |
|---|---|---|
| `SPECTRA_EVAL_DETERMINISTIC=1` | off | argmax over the masked policy + actor/critic `eval()` (no dropout at TEST) |
| `SPECTRA_REWARD_MODE=structural_band` | off | NEON trichotomy, but the over-budget arm is `-(-Δacc - τ)³` — graded by **accuracy overshoot** instead of cut size |
| `SPECTRA_REWARD_SCALE=cbrt` | off | monotone `cbrt` of the step reward. Per-step ordering is exactly NEON's; magnitude falls from ~1e5 to ~1e2 so the Smooth-L1 critic (β=100) can actually regress the return |
| `SPECTRA_PREVIEW_CACHE=0` | cache **on** | memoizes `param_ratio` / `flops_ratio` / `preview_ratios` / input shape within a step. Pure speedup: eval asked for the same deepcopy-and-prune and the same hooked FLOP forward 5–7× per step, and probed FLOPs on every identity-padded step (DenseNet ≈596 steps/net) |

Rationale for `structural_band` is §52.1: under `-reduction³` every over-budget cut is punished in proportion to its **size**, so the only signal is "cut less" — never "cut somewhere else". On a net whose band is empty the argmax is provably "never prune" and the episode teaches nothing about layer selection. Grading by overshoot keeps the trichotomy and keeps in-budget strictly better than any violation. **154 tests pass on leap.**

### 54.4 A/B queued 4 Sep ~11:45 IDT (nice 100 — behind `20930175` / `20884673`, ahead of the nice-5000 tail)

First submit **20944078–085** was cancelled before any GPU: `--export` dropped the switches. Resubmit uses named profiles (`eval_c10_thin_det`, `offline_train_cbrt`, `offline_train_band_cbrt`) so sbatch pins the flag even if `--export` is incomplete.

Chain A — determinism, frozen s42, **no retrain**. Control is §17 `20189046` (r20-w2 −4.2 @ 0.600/0.760; r56-w4 −15.9 @ 0.704/0.550), same actor `20158274`.

| Job | Arm |
|---|---|
| **20945567** | `eval_c10_thin` **sampled** — in-code control for 20189046 |
| **20945568** | `eval_c10_thin_det` **argmax** (`afterany` 20945567) |
| **20945570** | similar-family argmax (`afterany` 20945568) |
| **20945572** | C100 argmax (`afterany` 20945570) |

Chain B — reward retrain, 10-net catalog, seed 42, evaluated with argmax.

| Job | Arm | Isolates |
|---|---|---|
| **20945574** → **20945576** | `neon` + `cbrt` | optimization conditioning only |
| **20945744** → **20945749** | `structural_band` + `cbrt` | conditioning **+** the over-budget grading fix |

**Read.** If argmax collapses to identity everywhere (params ≈1.0, Δacc ≈0), that is not a null result — it is direct confirmation of §52.1: the quoted compressions were produced by exploration noise, not by a learned schedule. If argmax prunes and Δacc improves at matched size, every table in this ledger should be re-run deterministically before the 15 Sep freeze.

**Not yet done:** re-running the paper catalogs. Gated on Chain A. Overlay is now the working tree (commit this section with the switches). These jobs need `scripts/spectra.sbatch` profiles and `src/*` on the leap until they start.

### 54.5 Replan (6 Sep 12:33 IDT) — sanity first, then retake

Ido: the §54 defects are paper-critical. Status at the ask:

| Defect | Code | GPU TEST | Conclusion so far |
|---|---|---|---|
| Frozen policy sampled + dropout live | `SPECTRA_EVAL_DETERMINISTIC=1` (default **off**) | Chain A **20945567–572 still PD** (nice 100). No argmax TEST yet. | **Not tested.** Quoted DRL TESTs remain samples. Do not relabel them as the policy until Chain A returns. |
| Prefer arms contain no agent | No code “fix”: the knob is a heuristic. Relabel §30 / §33 / §35 / §48. | Confirmed on disk (byte-identical sizes across seeds). | **Provenance fixed in the draft.** Do not re-run prefer as DRL. |
| Reward band empty on C100 | Telemetry `reward_band_report.py`; `structural_band` / `structural_shaped` / `cbrt` | Diag **20930175 COMPLETED**: C100 **42/42 over-budget**. Shaped train **20967060 R** (no TEST). Chain B **20945574–749 still PD**. | Histogram insight: band tracks **dataset recoverability**, not class-count (spoof §55.1). Shaping/band **not yet TESTed**. |

**GPU order from this stamp (QOS 5):** keep current R jobs. Next free GPU: Chain A **20945567** (sampled thin control) then **20945568** (argmax). Demote remaining C10 heuristic afterok children so they do not beat the sanity chain. Do **not** scancel C100 trains. Do **not** retake the paper catalogs until argmax on thin r20/r56 says whether the quoted compressions survive without sampling.

**How wide to re-run, after Chain A:**

1. If argmax **collapses to identity** (params ≈1, Δacc ≈0): the quoted DRL tables are exploration noise. Paper claim becomes “sampled frozen policy”; do **not** mass-rerun catalogs. Retrain (Chain B) is the next science, not another sampled eval.
2. If argmax **prunes and Δacc holds at matched size**: retake **DRL** catalogs only (similar / unlike / thin / C9 frozen) with `SPECTRA_EVAL_DETERMINISTIC=1`, three seeds. Heuristics (greedy / mild / random / look-ahead / prefer) are unchanged — they never sampled the actor.
3. If argmax **prunes but sizes/Δacc move**: replace the DRL cells that moved; leave heuristics.

**FPGM / BN-scale (6 Sep):** implemented as `SPECTRA_FILTER_IMPORTANCE=fpgm|bn_scale`, default still **`l1`**. Same-loop ranking, not a new agent, not a third-party import (He et al. CVPR 2019 / Liu et al. ICCV 2017 formulas in `src/pruning.py`). Do **not** change the frozen 10-net ranker. Queued **21040934** (`eval_c10_thin_fpgm`, `afterok:20945568`, nice 50) then **21040935** (`bn_scale`, `afterok:21040934`). Fold into SPECTRA DRL *train* only if those evals beat L1 enough to justify a new actor before 15 Sep — unlikely; keep them on the Pareto as ranking A/Bs.

---

## 55. C100 spoof TEST + reward-band diag (4 Sep COMPLETED)

### 55.1 Spoof arm A1 — **20884670 COMPLETED** (8 h 9 m, ended 4 Sep 17:11 cluster)

Frozen 10-net s42, C9 catalog, Adam-40, `SPECTRA_SPOOF_NUM_CLASSES=10` (encoder tokens only; the net is still 100-way). Compare to §21 s42. Quote `eval_test` FINAL only.

| Net | Spoof s42 (20884670) | §21 s42 (no spoof) | vs τ=10 |
|---|---|---|---|
| VGG-16 BN | **−7.3 @ 0.893/0.926** (0.740→0.667) | **−7.5 @ 0.797/0.834** | inside both |
| ShuffleNet-v2×1 | **−4.2 @ 0.778/0.794** (0.726→0.684) | **−3.9 @ 0.833/0.823** | inside both (quote structural) |
| thin r20-w16 | **−20.4 @ 0.615/0.625** (0.730→0.526) | **−19.3 @ 0.604/0.645** | miss both |
| thin r56-w15 | **−18.2 @ 0.645/0.511** (0.784→0.602) | **−15.0 @ 0.694/0.600** | miss both |
| RepVGG-A0 | **−11.5 @ 0.668/0.522** (0.753→0.638) | **−12.1 @ 0.571/0.446** | miss both |

**Read.** Spoofing the class-count token 100→10 did **not** change the family split. VGG and ShuffleNet still transfer; residuals and RepVGG still miss. Size differences vs §21 are inside §54 resampling noise — do not call them a spoof effect. The encoder seeing `out_features=10` is **not** why C100 residuals fail. Child matched-VGG DRL **20884671 PD QOS** (prio 197).

### 55.2 Reward-band diag — **20930175 COMPLETED** (2 h 10 m, ended 4 Sep 16:10)

`python scripts/reward_band_report.py runs/job20930175` (216 non-identity steps):

| Slice | n | in-budget | over-budget | median Δacc |
|---|---|---|---|---|
| cifar-10 | 56 | 30 (53.6%) | 26 (46.4%) | −9.31 pp |
| cifar-100 | 42 | **0 (0%)** | **42 (100%)** | −25.73 pp |
| VGG-16 BN C10 | 14 | 14 (100%) | 0 | −7.00 pp |
| VGG-16 BN C100 | 19 | **0** | **19 (100%)** | −28.36 pp |
| r56-w4 C10 | 42 | 16 (38.1%) | 26 (61.9%) | −11.47 pp |
| r20-w16 C100 | 23 | **0** | **23 (100%)** | −19.90 pp |

**EMPTY BAND: cifar-100.** Every non-identity cut on C100 scored `−reduction³`. TEST finals on the same job: VGG-16 C10 **−3.2 @ 0.845/0.844** inside; VGG-16 C100 **−7.5 @ 0.802/0.828** inside (final net recovered; the *step* band was still empty); r56-w4 C10 **−26.5 @ 0.667/0.485** miss; r20-w16 C100 **−16.1 @ 0.719/0.582** miss.

**Read with §52.1 and §55.1.** Same architecture (VGG-16) is 100% in-budget on C10 and 100% over-budget on C100 *at the step*. That tracks the **dataset**, not the architecture. Spoof (§55.1) then shows it is **not** the class-count token — it is C100 recoverability under Adam-40 (the FT recipe / task). Shaping arm **20967060 RUNNING** (`structural_shaped` confirmed in the log FLAGS). Do not quote 20900187 / 20884673 / 20967060 train returns.

---
