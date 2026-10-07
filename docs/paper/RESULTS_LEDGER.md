# SPECTRA results ledger

**As of:** 18 Sep 2026 00:09 IDT. svd thin TRAJ **21433272 COMPLETED §97** (cloned 2-pass mild keep). G1 CLOSED **§96**. neonraw TRAJ next. V4 freeze **ep0083 / 0.262**. Last TESTs §77–§97. Do not quote wrap / 0.639.  
**Backfill:** every important result since git `ecefe78` (8 Aug 2026, “Transfer to new PC”) through the 10-net leap. Later jobs only *extend* these tables.  
**Protocol (current defaults):** τ = 10 pp; identity-pad 0.70 params unless `SPECTRA_EVAL_TRAJECTORY=1` (floor-hold then continue; quote val-best on **val**); rates 1.0 / 0.9 / 0.8 unless noted; full-net FT 40 ep / patience 10 on C10; NEON reward; small Transformer encoder; Fortify on.  
**Quote TEST only.** Skip akamaster ResNet-32.  
**10-net train catalog:** `configs/database_offline_train.json` (C10 + SVHN + Fashion-MNIST; no C100; **no** r20-w2 / r56-w4).  
**Held-out C10:** similar `input_offline_similar.json`; unlike `input_offline_novel.json`; thin `input_c10_thin.json`.

### Quoting rules (do not regress)

- `param_ratio` / `flops_ratio` = fraction **kept** (rebuilt shapes, not masked zeros).
- `eval_test` vs `eval_train` in logs is the **CNN data loader** (CIFAR test vs train images), **not** “held-out architecture vs in-catalog.” On r56-w4 job 20189046, eval_train is −1.7 pp at 0.667 params while TEST is −15.9 pp at 0.704 — that is fine-tune looking healthy on train images, not “the agent trained on this net.”
- Do **not** quote overnight-matrix “Eval Δacc” or train-step “within −10 %” as paper TEST. Those mixes are in §10 and §13–15 for archaeology only.
- Do **not** describe early C100 failure as “C100 was missing from the 10-net train set.” Early tests were recovery probes and mixed-catalog RL. Frozen 10-net → C100 numbers remain §21. **Ido 17 Sep 14:45:** C9 as *C10→C100 transfer* was **never** the intended paper claim. The claim is NEON-style diverse **train** (architectures × datasets, including C100) preparing a still-more-diverse **TEST**. Recaption §21; do not sell it as dataset-transfer of a C10-only agent.
- A probe cell `within_budget=True` at ≥98% params is **not** a 2–5% cut.
- **Paper finalization (Ido 6 Oct 09:47) — 10k companion, do not drop.** Live recipe stays protocol **P** (5k TEST half) for every agent decision, freeze TEST, and same-loop heuristic. Align to SOTA convention **only** with a companion **10k** column (or labelled `crossfit_readout.py` both-halves) beside literature rows, plus a **different-FT** caption. Do **not** switch the live walk/train recipe to full-10k val/TEST. Do not mix 5k P TEST with published 10k numbers in one cell without that caption. At paper freeze: every SOTA-facing table (DepGraph / OCS / HRank / Slimming / SPA) carries the 10k companion; the 5k P half remains the audit / agent-decision column.

---

## 1. Claims that the paper can already make

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | One offline agent prunes **similar-family** C10 nets (new width/source) inside τ=10, except skinny-deep ResNet-56. | LOCKED sampled; **argmax s42 r56-w10 misses** | §3 three seeds. Argmax **20945570 r56-w10 −12.8 @ 0.661** §57 |
| C2 | Same agent prunes **unlike-family** C10 nets (ShuffleNet, RepVGG; never in train) inside τ=10. | LOCKED | §4 seeds 42 / 43 / 44 |
| C3 | Easy thin ResNet-20 w2 is a size-matched **tie** vs greedy (~−4 pp at 60% params). | LOCKED | §5 |
| C4 | Skinny-deep ResNet-56 w4 **misses** τ=10 at matched DRL size across **three** seeds. | LOCKED sampled; **argmax s42 is the cliff** | §5 s42 −15.9, s43 −16.2, s44 −17.2 @ 0.704/0.550. Argmax **20945568 −25.2 @ 0.667/0.465** |
| C5 | On hard ResNets, DRL beats greedy: similar r56-w10 ~10 pp at matched params; held-out r56-w4 ~9 pp vs look-ahead greedy that kept **more** params. | LOCKED sampled; **under review** | Argmax r56-w4 **−25.2 @ 0.667** ties greedy cliff. Argmax similar r56-w10 **−12.8 @ 0.661** miss (ties mild; still beats look-ahead **−22.4**). Do not overwrite sampled **−9.2**. §57 |
| C6 | C10 is a recoverable FT environment. C100 is **not**, except VGG-11 BN under the 160-ep SGD recipe. | LOCKED | §7. Probe **20204214 COMPLETED**. Residuals/DenseNet/MobileNet are tiny cuts or val DROP. |
| C7 | Encoder capacity (BERT / wider / set) did not fix r56-w4. Catalog diversity **3-net → 10-net** moved it to −15.9; **10-net → 24-net did not** (C8 miss). | LOCKED | §16 encoder ~−24 pp; §17; C8 **−25.0 @ 0.704/0.499** |
| C8 | 24-net train catalog moves r56-w4 further. | **LOCKED miss** | **20201263 COMPLETED** 22:30. r56-w4 **−25.0 @ 0.704/0.499** vs 10-net s42 **−15.9 @ 0.704/0.550**. Worse, same params, fewer FLOPs. Easy r20-w2 **−5.7 @ 0.600/0.748**. |
| C9 | Frozen **C10-only** actor measured on CIFAR-100 (VGG/ShuffleNet inside τ; residuals/RepVGG miss). **Not the intended paper claim** (Ido 17 Sep 14:45): that was never C10→C100 transfer. Intended: diverse train, more diverse TEST. Keep the §21 numbers as a C10-only-actor table. | LOCKED mixed; **recaption** | §21; argmax §58. Do not overwrite numbers. Do not caption as the thesis transfer cell. |
| C10 | Eval-only FLOP floor 0.70 puts held-out r56-w4 **inside τ=10** (same frozen 10-net actor). | LOCKED | §5 s42 **−8.9 @ 0.907/0.702**; s43 **−9.2 @ 0.926/0.703**; s44 **−9.6 @ 0.907/0.703**. Not the 0.704-param operating point of C4. |
| C11 | Prefer Δparams/ΔFLOPs under FLOP floor 0.70 puts **similar** r56-w10 **inside τ=10** at 70% weights / 87% FLOPs (same frozen 10-net actor). FLOP-floor-only does not three-seed-rescue this net. | LOCKED | §33 **−3.8 / −4.0 / −4.4 @ 0.702/0.872**. vs §29 s43 **−14.3 @ 0.946/0.700**. |
| C12 | Frozen C10-trained agent on **ImageNet** MobileNet-v2 (truncated JPEG loader): two-seed inside τ=10 unmatched sizes. Not a SOTA ImageNet fight. No ImageNet DRL train. | **PRELIM** two-seed unmatched | §41 s43 **20360208 −4.6 @ 0.823/0.729** (0.719→0.673); s44 **20382192 −5.1 @ 0.772/0.652** (0.719→0.668). s42 TIMEOUT. |

---

## 2. Headline Pareto (C10 TEST)

NEON Figure 5 grammar (Gilad 18 Aug): plot **compression vs TEST Δacc** for DRL operating points **and** same-loop heuristics. Coverage matrix is a different artifact (family × dataset transfer). See `GILAD_DIRECTIVES_18AUG.md`. Maintain by appending TEST rows; do not rebuild from chat.

**Current paper-facing frontier is §2.4** (protocol P). §§2.1–2.3 are the legacy-val 10-net actor (do not mix). 10k companion + different-FT caption at freeze (quoting rules). Ops plot: `spectra-pareto-6oct.canvas.tsx`.

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

### 2.4 Protocol P Pareto (current recipe; Ido 6 Oct 09:47) — PRELIM; do not mix with §2.1–2.3

§2.1–2.3 are the **legacy-val** 10-net frontier (memorized val; prefer is a heuristic). This subsection is the **paper-facing** frontier under protocol P + crop+flip. **5k** = P TEST half (agent / same-loop). **10k** = literature companion (`crossfit` / both halves). Caption different-FT on every published star. **v10 actor: no TEST yet** (ep0015/ep0031 NEVER TEST).

**DepGraph ResNet-56 CIFAR-10** (home cell L1; FLOPs kept):

| Series | Point | FLOPs / params | Δacc | Protocol | Ledger |
|---|---|---|---|---|---|
| DepGraph published | 2.11× | 0.474 / 0.470 | **+0.24** | their sparse+FT, 10k | quote |
| DepGraph published | 2.57× | 0.389 / 0.382 | **+0.11** | their sparse+FT, 10k | quote |
| OCSPruner Table 5 pretrained | — | 0.388 / 0.423 | **−0.51** | their one-cycle, 10k | draft §4.1 |
| N3 mild 5-pass + 100-ep | flop 0.60 | 0.599 / 0.638 | 10k **−0.03** | P 10k companion | §157 |
| N3 | 2.11× | 0.463 / 0.470 | 10k **−0.46** (walk 5k **−0.22**) | P; M4 | §157 |
| N3 | 2.57× | 0.380 / 0.382 | 10k **−1.63** (walk 5k **−1.32**) | P | §157 |
| Scratch-B 200-ep | 2.11× | 0.463 / 0.470 | 10k **−0.16** | Liu et al. architecture | §159 |
| Mild-landed 6-pass 100-ep | flop 0.60 | 0.599 / 0.638 | 5k **−0.1** | v10 heuristic pin | **§217** |
| τ-off 10-pass τ=30 | 2.11× | 0.463 / 0.470 | 10k **−0.94** (walk 5k **−0.76**) | PATH-SAME vs N3; not a new star | **§220** |

**Thin r56-w4 CIFAR-10** (thesis WIN cell; params kept; 5k P; M1 bar **1.0 pp**, census ≥ 2):

| Series | Point | params | 5k TEST | Ledger |
|---|---|---|---|---|
| Mild 2-pass | size 0.80 / `val_best` | 0.795 / 0.622 | **−2.6 / −4.5** | §152 (21729557) |
| Greedy 2-pass | first | 0.743 | **−5.7** | §173 |
| Stage-4 DRL ep0179 | size 0.80 / 0.60 / `val_best` | 0.743 / 0.600 / 0.389 | **−4.0 / −5.0 / −7.6** | **§218** (census 0.8 only; M1 does not fire) |
| Mild-landed 6-pass | κ 0.8 / 0.6 | 0.799 / 0.600 | **−2.1 / −5.1** | §211 / §212 |
| v10 DRL ep0127 | κ 0.8 / 0.6 | 0.799 / 0.600 | **−2.88 / −5.28** | **§237 / §248** (census 0.9 only; **M1-v10 FLAT**) |
| Greedy-landed 6-pass | κ 0.8 / 0.6 | 0.788 / 0.600 | **−2.4 / −4.7** | §214 / §216 |
| Random-landed 6-pass | κ 0.6 | **0.564** (gap 0.036) | **−5.1** | §215 (not equal-size) |

**VGG-16 C10** (L2; FLOPs kept; param mismatch vs HRank/OCS):

| Series | FLOPs / params | Δacc | Protocol | Ledger |
|---|---|---|---|---|
| HRank | 0.465 / **0.171** | **−0.53** | published 10k | quote; params not matched |
| OCSPruner | 0.212 / **0.137** | **−0.44** | published 10k | quote |
| Mild 10-pass + 100-ep | 0.464 / 0.444 | 10k **−0.25** | P companion | §165 |
| Mild 10-pass | 0.211 / 0.187 | 10k **−2.02** | P companion | §165 |
| Mild-landed 6-pass | 0.593 / 0.623 | 5k **−0.0** | v10 pin | **§213** |

Do **not** lock. Do not put Stage-4 §218 on the same-loop WIN line (constant 0.8). Do not plot v10 probes. Ops canvas (6 Oct): `spectra-pareto-6oct.canvas.tsx`. At paper freeze, promote this table and the 10k companion; leave §2.1–2.3 as provenance of the legacy actor.

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

**Similar FLOP-floor greedy (L1, no look-ahead, no prefer) — PRELIM** (s42 **20715868** / s43 **20715870** / s44 **20715871 ALL COMPLETED**). Three-seed same size: r20 **−4.8 / −5.7 / −5.1 @ 0.908/0.701**; r56-w10 **−9.2 / −9.3 / −9.0 @ 0.952/0.702** inside; r44 **−4.6 / −5.1 / −4.1 @ 0.947/0.702**; VGG **−3.1 / −2.7 / −2.5 @ 0.804/0.701**; MobileNet **−3.1 / −3.0 / −2.8 @ 0.933/0.700**; DenseNet **−2.2 / −2.2 / −2.3 @ 0.798/0.700**. L1 greedy does not use the actor — Δacc spreads are fine-tune noise. Skip r32. Table §51.

**Unlike mild — PRELIM** (s42 **20412380 COMPLETED**). Catalog all four unlike nets **inside τ** one seed. ShuffleNet-v2×1 **−1.6 @ 0.857/0.835** (quote structural); ×1.5 **−2.6 @ 0.849/0.828** (quote structural; do not quote masked 0.818); RepVGG-A0 **−5.3 @ 0.680/0.548** (0.943→0.890); A1 **−4.2 @ 0.663/0.539** (0.944→0.902). Look-ahead is worse Δacc on both RepVGGs. Table §45.

**Unlike random — PRELIM** (s42 **20412385 COMPLETED**). Catalog all four unlike nets **inside τ** one seed. ShuffleNet-v2×1 **−1.6 @ 0.801/0.753** (quote structural; do not quote masked 0.764). RepVGG-A0 **−6.2 @ 0.659/0.503** (0.943→0.881). RepVGG-A1 **−5.3 @ 0.649/0.508** (0.944→0.891). ShuffleNet-v2×1.5 **−2.2 @ 0.794/0.761** (0.932→0.910) (quote structural; do not quote masked 0.755). Child Wave Q **20412555 COMPLETED** §66. Table §49.

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

**Unlike look-ahead greedy — PRELIM** (s42/s43/s44 **20382196 / 197 / 198 ALL COMPLETED**). ShuffleNet-v2×1 three-seed **−2.2 / −1.9 / −1.8 @ 0.723/0.682**; ×1.5 three-seed **−2.5 / −2.4 / −2.4 @ 0.710/0.675**; RepVGG-A0 three-seed **−7.2 / −7.2 / −6.4 @ 0.709/0.577**; RepVGG-A1 three-seed **−6.3 / −6.1 / −6.3 @ 0.710/0.574** same size. Quote **structural** keep. Heuristic. Full table §44.

**Unlike mild — PRELIM one seed** (s42 **20412380 COMPLETED**). ShuffleNet-v2×1 **−1.6 @ 0.857/0.835**; ×1.5 **−2.6 @ 0.849/0.828** (quote structural); RepVGG-A0 **−5.3 @ 0.680/0.548**; A1 **−4.2 @ 0.663/0.539**. All four inside τ. Table §45.

**Unlike random — PRELIM one seed** (s42 **20412385 COMPLETED**). ShuffleNet-v2×1 **−1.6 @ 0.801/0.753** inside (quote structural; do not quote masked 0.764). RepVGG-A0 **−6.2 @ 0.659/0.503** inside. RepVGG-A1 **−5.3 @ 0.649/0.508** inside. ShuffleNet-v2×1.5 **−2.2 @ 0.794/0.761** inside (quote structural; do not quote masked 0.755). Catalog COMPLETED. Table §49.

**Unlike FLOP-floor look-ahead — PRELIM three-seed** (s42 **20412394**; s43 **20412395**; s44 **20412396 COMPLETED** §65). Same size: ShuffleNet×1 **−1.7 / −2.0 / −2.2 @ 0.801/0.716**; A0 **−6.9 / −6.6 / −7.1 @ 0.847/0.701**; A1 **−6.7 / −5.7 / −6.2 @ 0.857/0.702**; ×1.5 **−2.5 / −2.6 / −2.4 @ 0.771/0.701**. Quote structural keep; skip masked 0.725. Heuristic. Table §47.

**Unlike FLOP+prefer look-ahead — PRELIM one seed** (s42 **20715879 COMPLETED**). ShuffleNet-v2×1 **−1.0 @ 0.871/0.944**; ×1.5 **−1.8 @ 0.879/0.950**; RepVGG-A0 **−3.0 @ 0.715/0.756**; A1 **−2.5 @ 0.705/0.753**. **Same sizes** as §30 prefer (no look-ahead). Heuristic, not DRL. Catalog COMPLETED. Table §62.

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
| **20945567** | sampled control (same actor, no det) | **−3.9 @ 0.600/0.748** | **−25.4 @ 0.667/0.482** | **PRELIM** — resampling vs 20189046; not the policy |
| **20945568** | argmax + `eval()` (`SPECTRA_EVAL_DETERMINISTIC=1`) | **−4.4 @ 0.600/0.741** | **−25.2 @ 0.667/0.465** | **PRELIM** — not identity. Path 3 vs locked 0.704/0.550. Matches sampled 567 cliff |
| **20945576** | Chain B neon+cbrt argmax (actor **20945574**, not frozen 10-net) | **+0.0 @ 1.000/1.000** | **+0.0 @ 1.000/1.000** | **PRELIM identity** §63. Do not overwrite Path 3 |
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
| **20382196 / 197 / 198** | Unlike look-ahead greedy s42/s43/s44 | **ALL COMPLETED.** s42 **20382196** 2 d 14 h 19 m (ended 29 Aug 12:49). s44 **20382198** 1 d 21 h 29 m (ended 7 Sep 19:33). s43 **20382197 COMPLETED** 11 h 41 m (ended 9 Sep 20:53 cluster). ShuffleNet×1 three-seed **−2.2 / −1.9 / −1.8 @ 0.723/0.682**; ×1.5 **−2.5 / −2.4 / −2.4 @ 0.710/0.675** (quote structural; skip masked 0.666); A0 **−7.2 / −7.2 / −6.4 @ 0.709/0.577**; A1 **−6.3 / −6.1 / −6.3 @ 0.710/0.574**. Heuristic. Child **20412382** PD QOS. §44. |
| **20412380 / 382 / 384** | Unlike mild s42/s43/s44 | **20412380 COMPLETED** 1 d 1 h 18 m (ended 30 Aug 14:08 cluster). Catalog COMPLETED PRELIM: ShuffleNet-v2×1 **−1.6 @ 0.857/0.835** (0.924→0.908) inside (quote structural; do not quote masked 0.831); RepVGG-A0 **−5.3 @ 0.680/0.548** (0.943→0.890) inside; RepVGG-A1 **−4.2 @ 0.663/0.539** (0.944→0.902) inside; ShuffleNet-v2×1.5 **−2.6 @ 0.849/0.828** (0.932→0.906) inside (quote structural; do not quote masked 0.818). All four inside τ one seed. Do not quote wrap job-mean **−0.02 pp**. Child unlike-random **20412385 COMPLETED**. Wave I. s43 **20412382** still afterok. s44 **20412384 R** (`cs-pheno-04`). §45. |
| **20412385 / 386 / 387** | Unlike random s42/s43/s44 | s42 **20412385 COMPLETED** 2 d 1 h 11 m (ended 3 Sep 09:42 cluster, `dt-2080-07`). ShuffleNet-v2×1.5 TEST **−2.2 @ 0.794/0.761** (0.932→0.910) inside (quote structural; do not quote masked 0.755). Catalog COMPLETED PRELIM: ShuffleNet×1 **−1.6 @ 0.801/0.753**; RepVGG-A0 **−6.2 @ 0.659/0.503**; RepVGG-A1 **−5.3 @ 0.649/0.508**; ×1.5 **−2.2**. All four inside τ one seed. Child Wave Q **20412555 PD** (`QOSMaxGRESPerUser`; GRES went to **20412394**). s43/s44 still afterok unlike mild. §49. |
| **20412388 / 389 / 390** | C100 Adam-40 look-ahead greedy s42/s43/s44 | **ALL COMPLETED** (s42 11 h 47 m, ended 26 Aug 01:21 cluster). s42 RepVGG **−11.4 @ 0.709/0.577** (0.753→0.639) miss three-seed **−11.4 / −11.7 / −12.1** same size. Catalogs COMPLETED PRELIM: r20 miss; r56 cliff; VGG/ShuffleNet inside; RepVGG miss. afterok mild **20412530 / 531 / 532 COMPLETED**. Wave K. §37. |
| **20412391 / 392 / 393** | Similar FLOP-floor look-ahead serial s42→s43→s44 | **20412391 COMPLETED** 4 d 16 h 45 m (ended 26 Aug 01:23 cluster). s42 catalog COMPLETED PRELIM. **20412392 COMPLETED** 3 d 14 h 08 m (ended 27 Aug 21:11 cluster). DenseNet TEST **−2.1 @ 0.798/0.700** (0.949→0.928) two-seed **−2.6 / −2.1** same size inside. Catalog PRELIM: r20 **−5.9 @ 0.908/0.701**; r56 **−9.2 @ 0.952/0.702** (FLOP-floor-only s43 was **−14.3 miss**); r44 **−4.4 @ 0.947/0.702**; VGG **−2.7 @ 0.804/0.701**; MobileNet **−3.1 @ 0.933/0.700**. r32 skipped. **20412393 COMPLETED** 3 d 14 h 32 m (ended 31 Aug 11:43 cluster). s44 DenseNet TEST **−2.8 @ 0.798/0.700** (0.949→0.921) three-seed **−2.6 / −2.1 / −2.8** same size inside. Catalog COMPLETED PRELIM: r20 **−5.2 @ 0.908/0.701** three-seed **−4.6 / −5.9 / −5.2**; r56 **−8.6 @ 0.952/0.702** three-seed **−8.3 / −9.2 / −8.6**; r44 **−4.4 @ 0.947/0.702** three-seed **−4.2 / −4.4 / −4.4**; VGG **−2.8 @ 0.804/0.701** three-seed **−3.6 / −2.7 / −2.8**; MobileNet **−3.2 @ 0.933/0.700** (0.938→0.906) three-seed **−3.3 / −3.1 / −3.2**. r32 skipped. Children ImageNet s42 **20715875** + prefer-greedy **20715876** PD (`QOSMaxGRESPerUser`; CG **20382185** still holds a GRES). Wave O already on **20412540**. Do not jump Wave Q. §39. |
| **20412394 / 395 / 396** | Unlike FLOP-floor look-ahead serial | **20412394 COMPLETED** 2 d 12 h 21 m (ended 5 Sep 22:04 cluster). s42 catalog COMPLETED PRELIM all four unlike nets inside τ. Keep first-pass: ShuffleNet×1 **−1.7 @ 0.801/0.716**; RepVGG-A0 **−6.9 @ 0.847/0.701**; A1 **−6.7 @ 0.857/0.702**; ×1.5 **−2.5 @ 0.771/0.701** (quote structural; skip masked 0.724). Child **20412395 COMPLETED** 11 h 17 m (ended 6 Sep 12:01 cluster). s43 catalog COMPLETED PRELIM: ×1 **−2.0 @ 0.801/0.716**; A0 **−6.6 @ 0.847/0.701**; A1 **−5.7 @ 0.857/0.702**; ×1.5 **−2.6 @ 0.771/0.701** (0.932→0.906) — all **same size** as s42. Two-seed ×1.5 **−2.5 / −2.6**. FLAGS neon prefer=0. Freed GPU → **20884671** (age beat nice-0 child). Child **20412396 COMPLETED** 2 d 11 h 47 m (ended 10 Sep 19:27 cluster). s44 catalog COMPLETED PRELIM all four unlike nets inside τ **same size** as s42/s43: ×1 **−2.2 @ 0.801/0.716**; A0 **−7.1 @ 0.847/0.701**; A1 **−6.2 @ 0.857/0.702**; ×1.5 **−2.4 @ 0.771/0.701**. Three-seed. Heuristic. Child **20412549** still **JobHeldUser** — do not release. §47 / §65. |
| **20412530 / 531 / 532 → 20412533 / 534 / 536** | C100 Adam-40 mild then random s42/s43/s44 | **20412530 / 531 / 532 COMPLETED** (s42 12 h 15 m, ended 26 Aug 13:37 cluster). Mild catalogs COMPLETED PRELIM: r20 three-seed miss; r56 three-seed miss (not a look-ahead cliff); VGG three-seed **−8.0 / −8.3 / −8.0 @ 0.811/0.822** inside; ShuffleNet three-seed **−4.0 / −4.4 / −4.3 @ 0.860/0.835** inside (quote structural 0.860); RepVGG three-seed **−11.8 / −11.5 / −11.8 @ 0.684/0.548** miss. afterok random **20412533 COMPLETED** 2 d 9 h 19 m (ended 28 Aug 22:57 cluster). Catalog COMPLETED PRELIM: r20 **−18.0 @ 0.676/0.579** miss unmatched; r56 **−26.2 @ 0.628/0.349** miss unmatched three-seed **−26.2 / −26.4 / −30.3**; VGG **−9.3 @ 0.698/0.718** inside unmatched three-seed **−9.3 / −8.5 / −8.5**; ShuffleNet **−4.2 @ 0.785/0.747** inside unmatched three-seed **−4.2 / −5.5 / −4.4** (quote structural 0.785; do not quote masked 0.760); RepVGG-A0 **−13.4 @ 0.662/0.498** (0.753→0.619) miss unmatched three-seed **−13.4 / −12.8 / −12.6**. Designed leaf — do not attach Wave O. Idle GPU filled by **20382188**, not Wave Q. **20412534** s43 **COMPLETED** catalog PRELIM: r20 **−17.0** miss; r56 **−26.4** miss; VGG **−8.5** inside; ShuffleNet **−5.5 @ 0.741/0.732** inside; RepVGG **−12.8 @ 0.677/0.513** miss. **20412536** s44 **COMPLETED** catalog PRELIM: r20 **−17.3** miss unmatched; r56 **−30.3** miss unmatched; VGG **−8.5 @ 0.714/0.695** inside unmatched; ShuffleNet **−4.4 @ 0.762/0.741** inside unmatched (quote structural 0.762); RepVGG **−12.6 @ 0.664/0.498** (0.753→0.627) miss unmatched vs s43 **−12.8**. §38 / §40. |
| **20412538 / 540 / 542 → 20412545 / 546 / 548** | Similar FLOP-floor mild then random serial | **20412538 COMPLETED** 3 d 10 h 54 m (ended 30 Aug 04:51 cluster). Catalog COMPLETED PRELIM: r20 **−5.6 @ 0.801/0.706** inside; r56 **−11.6 @ 0.946/0.701** miss; r44 **−3.9 @ 0.905/0.703** inside; VGG **−2.8 @ 0.811/0.819** inside (floor did not bind); MobileNet **−2.4 @ 0.791/0.703** inside (floor bound); DenseNet **−2.2 @ 0.823/0.828** (0.949→0.927) inside (floor did not bind). r32 skipped. **20412540 COMPLETED** 2 d 18 h 58 m (ended 4 Sep 04:22 cluster). s43 catalog COMPLETED PRELIM: r20 **−5.1**; r56 **−9.9** inside (s42 miss same size); r44 **−3.6**; VGG **−3.0**; MobileNet **−2.3**; DenseNet **−2.1 @ 0.823/0.828** (0.949→0.928) two-seed **−2.2 / −2.1** same size. r32 skipped. Child spoof **20884670 PD QOS**. s44 **20412542 CANCELLED** 10 Sep 03:24 (22 h 23 m; no TEST). Child **20412545** still afterok. Wave O. §43. |
| **20412549–551 → 20412552–554** | Unlike FLOP-floor mild then random serial | **PENDING** afterok unlike FLOP-floor look-ahead s44. Wave P. |
| **20412555 / 556 / 557** | Similar FLOP+prefer look-ahead greedy s42/s43/s44 | **20412555 COMPLETED** 2-00:02:43 (ended 11 Sep 17:02 cluster, `cs-1080-01`, exit 0). Catalog COMPLETED PRELIM §66. Heuristic, not DRL. kids=NONE. `20412556` still JobHeldUser `afterok:20412386` — do not release. Do not attach a child. |
| **20715868 / 870 / 871 → 20715872 / 873 / 874** | Similar then unlike FLOP-floor greedy (L1, no look-ahead) serial | **20715868 COMPLETED** 2 d 1 h 36 m (ended 4 Sep 14:00 cluster). s42 catalog COMPLETED PRELIM: r20 **−4.8**; r56 **−9.2** inside; r44 **−4.6**; VGG **−3.1**; MobileNet **−3.1**; DenseNet **−2.2 @ 0.798/0.700**. **20715870 COMPLETED** 22 h 27 m (ended 5 Sep 15:38 cluster). s43 catalog COMPLETED PRELIM: r20 **−5.7**; r56 **−9.3** inside; r44 **−5.1**; VGG **−2.7**; MobileNet **−3.0**; DenseNet **−2.2 @ 0.798/0.700**. **20715871 COMPLETED** 22 h 46 m (ended 6 Sep 20:20 cluster). s44 DenseNet **−2.3 @ 0.798/0.700** (0.949→0.926) three-seed **−2.2 / −2.2 / −2.3** same size. Skip r32. Child **20715872 COMPLETED** 1 d 20 h 22 m (ended 10 Sep 08:12 cluster) — unlike FLOP-floor greedy s42 §64. Wave R. §51. |
| **20715875** | ImageNet MobileNet-v2 s42 retry | **CANCELLED** 10 Sep 02:13 (5 h 19 m; no TEST). Do not quote train-loader. Frozen ImageNet serial **21166873–876** waits `afterany:21168774`. |
| **20715876 / 877 / 878** | Similar FLOP+prefer greedy (L1, no look-ahead) | **20715876 COMPLETED** 1 d 21 h 30 m (ended 3 Sep 06:02 cluster, `dt-2080-18`). DenseNet TEST **−2.1 @ 0.870/0.951** (0.949→0.928) inside, size-matched near-tie vs DRL prefer **−2.0**. Catalog COMPLETED PRELIM: r20 **−3.9**; r56 **−4.4**; r44 **−2.6**; VGG **−2.9**; MobileNet **−2.0**; DenseNet **−2.1**. r32 skipped. Child **20715877 PD** (`QOSMaxGRESPerUser`; GRES went to **20382189**). Then s44 **20715878**. §48. |
| **20715879** | Unlike FLOP+prefer look-ahead s42 | **COMPLETED** 1 h 59 m (ended 9 Sep 09:08). Catalog COMPLETED PRELIM §62. Heuristic, not DRL. |
| **20945576** | Chain B neon+cbrt thin eval | **COMPLETED** 3 m 22 s (ended 9 Sep 09:12). Identity §63. Actor `job20945574`. |
| **20382197** | Unlike look-ahead greedy s43 | **COMPLETED** 11 h 41 m (ended 9 Sep 20:53 cluster). FLAGS prefer=0 neon. Catalog §44. Child **20412382** still JobHeldUser. |
| **20945744** | Chain B band train `structural_band`+`cbrt` | **CANCELLED** 10 Sep 04:15 (5-step; invalid). Replaced 12 Sep by full-net 128-step **21194543** `afterok:21168557` → thin argmax **21194544**. **Not TEST.** C100 residual DRL still left. |
| **21168773 → 21184512 → 21194534 / 94535** | Prefer full-net continue then thin argmax (prefer knob **off**) | **21168773 COMPLETED** 36h at **69/130** (best ep 31, 73.23). Continue **21184512 COMPLETED** 13 Sep 06:03 (132 eps, patience 100/100, best still 73.23 — **not TEST**). Snapshot thin **21229256 COMPLETED** §68. Cont2 **21194534** PD `afterok:84512`. |
| **21168838 → 21184514 → 21194536 / 94537** | NEON cubes continue then thin argmax | **21168838 COMPLETED** 36h at **79/130** (best ep 61, 75.04). Continue **21184514 COMPLETED** 2 d 16 m (ended 13 Sep 17:49). **Not TEST.** |
| **21168840 → 21184407 → 21194538 / 94539** | Prefer + 0.70 train floor continue | **21168840 COMPLETED** 36h at **87/130** (best ep 15, 46.37). Continue **21184407 CANCELLED** 13 Sep 18:26 (ep 104, entropy 0.9887 = uniform; Ido, slot → v2c). **Not TEST.** |
| **21168844 → 21184409 → 21194540 / 94542** | F1 unified then continue | Parent **COMPLETED** 11 Sep 20:30 skip-eval. Continue **21184409 CANCELLED** 13 Sep 11:15 (ckpt collapsed 71.37→32.94). Cont2/eval cancelled. **Not TEST.** |
| **21194543 → 21194544** | Full-net `structural_band`+`cbrt` 128-step | **21194543 CANCELLED** 13 Sep 11:15 (ep 10/130 after 27 h; would miss 17 Sep). Kid cancelled. **Not TEST.** |
| **21166873** | ImageNet r50 truncated-JPEG s42 | **CANCELLED** 13 Sep 11:15 (no `pass 1/1`; two-seed MobileNet PRELIM already exists). Kids cancelled. |
| **21168557** | C100 unlike-extra argmax s42 (ShuffleNet×1.5 + RepVGG-A1) | **COMPLETED** 13.9 h (ended 12 Sep 07:20 cluster). Catalog COMPLETED PRELIM §67. Frozen `job20158274`, det=1, neon. Freed GPU → band **21194543**. |
| **21229256** | Prefer snapshot C10-thin argmax (`latest_best` from 84512, prefer knob **off**) | **COMPLETED** 1 h 20 m (ended 13 Sep 07:26 cluster, `cs-pheno-01`, exit 0). Catalog COMPLETED PRELIM §68 **CLIFF**. r20 **−3.5 @ 0.600/0.753**; r56 **−23.8 @ 0.667/0.457**. Dispatch **21230428 COMPLETED** 5 s classified **WIN** (min-keep is r20, not r56). Snap PD catalogs **scancelled** 13 Sep for trajectory TESTs. |
| **21229257** | Cubes snapshot C10-thin argmax (`latest_best` from 84514, prefer knob **off**) | **COMPLETED** 45 m (ended 13 Sep 08:11 cluster, exit 0). Catalog COMPLETED PRELIM §69. r20 **−4.3 @ 0.600/0.741**; r56 **−21.9 @ 0.722/0.524**. |
| **21230664** | Prefer snapshot similar-family argmax (`eval_offline_similar_det`) | **COMPLETED** 15 h 42 m (ended 13 Sep 23:54 cluster). Catalog COMPLETED PRELIM §70. DenseNet **−2.3 @ 0.697/0.686**. Skip r32. Do not quote wrap **+0.05**. Freed GPU → **21238729**. |
| **21233223 / 226 / 227** | C10-thin **trajectory** TESTs Path 3 / prefer / cubes | Path 3 **21233223 COMPLETED** §72. Prefer **21233226 COMPLETED** §71 invalid identity. Cubes **21233227 COMPLETED** §73 log1p. Quote `[eval] TRAJ`. |
| **21233371** | Prefer traj redo with matching std | **COMPLETED** §74. Std n=721. r20 + r56 TRAJ. |
| **21235566** | Cubes traj redo with matching std | **COMPLETED** 1 h 3 m (ended 13 Sep 16:22). r20 + r56 TRAJ §75. Std n=721. |
| **21237066** | v2 smoke gate `smoke_v2` | **COMPLETED** 22 m (ended 13 Sep 18:16). Integration only. `afterok` fired v2a/b. Do not quote wrap. |
| **21237253 / 254 / 255** | v2 train A/B/C (`offline_train_v2a/b/c`) | **A 21237253 COMPLETED** 1 d 22 h 51 m (ended 15 Sep 17:10). Patience 256 ep, PPO 64, best still **ep0155 / 0.312** (Δ vs TESTed +0.027 — do not TEST that snap). **B 21237254 COMPLETED** 15:06 patience 116 ep, best snap still ep0015 **0.287**. **C 21237255 COMPLETED** 21:02 — **do not TEST C**. |
| **21252195 / 199 / 250** | similar TRAJ v2b-ep0015 / mild-once / l1-once | **21252195 COMPLETED §92** (2 d 6 h 8 m, ended 16 Sep 17:24). mild **21252199** still R DenseNet. l1 **21260250 CANCELLED** 09:53. Skip r32. |
| **21252197** | unlike TRAJ v2b-ep0015 | **COMPLETED** 13 h 56 m (ended 15 Sep 01:13). Catalog COMPLETED PRELIM **§83**. |
| **21275601** | unlike TRAJ mild-once | **COMPLETED** 20 h 1 m (ended 15 Sep 17:32). Catalog COMPLETED PRELIM **§86**. |
| **21298586** | unlike TRAJ l1-once | **COMPLETED** 3 h 27 m (ended 15 Sep 12:45). Catalog COMPLETED PRELIM **§84**. |
| **21315161** | thin TRAJ mild-plain (H0, no group-once) | **COMPLETED** 2 h 15 m (ended 15 Sep 15:23). Catalog COMPLETED PRELIM **§85**. |
| **21325687** | thin TRAJ l1-plain (no group-once) | **COMPLETED** 8 h 46 m (ended 16 Sep 00:35). Catalog COMPLETED PRELIM **§88**. Twin of §85. |
| **21363176 / 532 / 533** | thin / similar / unlike TRAJ v2a-ep0155 | **21363176 COMPLETED §90**. **21363533 COMPLETED §89**. **21363532 CANCELLED** 09:53 (v3 GPU; DenseNet incomplete — do not ledger). |
| **21378931** | C100 TRAJ v2a-ep0155 | **CANCELLED** 09:53 (v3 GPU). r20+r56+VGG identity PRELIM; catalog incomplete. Do not overwrite §21 / §87. |
| **21385158 / 159 / 160 / 161** | v3 last-train fpgm / svd / bnscale / fpgm-neonraw | **All four R** 09:53. 24-net wide, groupcost, rewind, min 250 ep. Cold. Do not TEST until first snap. |
| **21237620** | First v2a snapshot TRAJ | **COMPLETED** 5 h 7 m (ended 14 Sep 00:18). Catalog COMPLETED PRELIM §77. r20 **+0.7 @ 0.746**; r56 val-best **−6.1 @ 0.923/0.769**. Caption: snap ep0003 still near-uniform. |
| **21237621 / 622 / 623** | Group-once TRAJ controls | **21237621 COMPLETED** §77. **21237622 COMPLETED** §81. **21237623 COMPLETED** 1 h 32 m (ended 14 Sep 04:32) §82. r20 **−1.4 @ 0.746**; r56 val-best **−7.0 @ 0.923**. |
| **21238729 / 39023 / 38730** | Later v2 snapshot TRAJs | **21238729 COMPLETED** §78. **39023 COMPLETED** 5 h 11 m (ended 14 Sep 05:35) §79. r20 **+0.4 @ 0.746**; r56 val-best **−6.5 @ 0.923**. **38730 COMPLETED** 5 h 11 m (ended 14 Sep 05:30) §80. r20 **−1.2 @ 0.606**; r56 val-best **−7.1 @ 0.879**. |

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

## 21. C10-only actor on CIFAR-100 (was claim C9) — LOCKED mixed; **recaption 17 Sep**

**Ido 17 Sep 14:45:** this table is **not** the intended paper claim. Gilad/Ido never wanted “C10 train → C100 TEST” as the transfer story. Intended: extensive diverse **training** (architectures × datasets, C100 in the train pool once recoverability allows) so **TEST** can be even more diverse. Keep these numbers as what a *C10-only* frozen actor did on C100. Do not overwrite.

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

Read with C6: VGG is the C100 family that recovers from structured cuts; residuals miss under this table’s Adam-40 recipe. These numbers are a **C10-only-actor measurement** on C100, not a new C100 policy and not the intended transfer claim (Ido 17 Sep). SGD-recipe A/B is §25 (one seed): r56-w15 enters τ; r20-w16 still misses. Do not start a second C100 DRL while v3/V4 are R.

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

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **mild** rate-picker at the **70% FLOP floor** (same stop as FLOP-floor DRL / FLOP-floor look-ahead). Skip akamaster r32. Quote `eval_test` FINAL only. Job **20412538 COMPLETED** 3 d 10 h 54 m (ended 30 Aug 04:51 cluster). s43 **20412540 COMPLETED** 2 d 18 h 58 m (ended 4 Sep 04:22 cluster, `dt-2080-13`). Scientific parent FLOP-floor look-ahead s42 **20412391 COMPLETED**. Wave O. Child spoof **20884670 PD QOS**; FLOP-mild s44 **20412542 R** (`ise-pheno-01`). Child **20412545** PD afterok. The **20884675** GPU went here, not to **20715879**.

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

## 44. Unlike-family look-ahead greedy — PRELIM (s42 / s43 / s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead greedy** rate-picker (no extra FLOP floor). Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20382196 COMPLETED** 2 d 14 h 19 m (ended 29 Aug 12:49 cluster). Scientific parent similar look-ahead catalogs COMPLETED. Wave H. s43 **20382197 COMPLETED** 11 h 41 m (ended 9 Sep 20:53 cluster). s44 **20382198 COMPLETED** 1 d 21 h 29 m (ended 7 Sep 19:33 cluster). afterok unlike mild **20412382** PD QOS; **20412384 R**. Heuristic — Δacc spread at identical size is FT noise, not a DRL seed.

| Net | look-ahead s42 (20382196) | look-ahead s43 (20382197) | look-ahead s44 (20382198) | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−2.2 @ 0.723/0.682** (0.924→0.902) | **−1.8 @ 0.723/0.682** (0.924→0.906) | **−1.9 @ 0.723/0.682** (0.924→0.905) | **−1.1 @ 0.800/0.825** | three-seed inside, **same size** |
| ShuffleNet-v2×1.5 | **−2.5 @ 0.710/0.675** (0.932→0.907) | **−2.4 @ 0.710/0.675** (0.932→0.908) | **−2.4 @ 0.710/0.675** (0.932→0.908) | **−2.4 @ 0.818/0.801** | three-seed inside, **same size** |
| RepVGG-A0 | **−7.2 @ 0.709/0.577** (0.943→0.871) | **−6.4 @ 0.709/0.577** (0.943→0.879) | **−7.2 @ 0.709/0.577** (0.943→0.871) | **−4.8 @ 0.681/0.565** | three-seed inside, **same size** |
| RepVGG-A1 | **−6.3 @ 0.710/0.574** (0.944→0.881) | **−6.3 @ 0.710/0.574** (0.944→0.881) | **−6.1 @ 0.710/0.574** (0.944→0.883) | **−4.7 @ 0.650/0.521** | three-seed inside, **same size** |

s43 **20382197** ShuffleNet-v2×1 **inside τ** at **−1.8 @ 0.723/0.682** (PASS1 `acc 0.924 -> 0.906 (-0.018) | params x0.723 | FLOPs x0.682`). **Same size** as s42/s44. Three-seed **−2.2 / −1.9 / −1.8 @ 0.723/0.682**. PASS1 also printed `effective-params x0.688` — **do not quote**. s43 RepVGG-A0 **inside τ** at **−6.4 @ 0.709/0.577** (PASS1 `acc 0.943 -> 0.879 (-0.064) | params x0.709 | FLOPs x0.577`). **Same size** as s42/s44 **−7.2**; 0.8 pp milder is FT noise. Three-seed **−7.2 / −7.2 / −6.4 @ 0.709/0.577**. s43 RepVGG-A1 **inside τ** at **−6.3 @ 0.710/0.574** (PASS1 `acc 0.944 -> 0.881 (-0.063) | params x0.710 | FLOPs x0.574`). **Same size and same Δacc** as s42. Three-seed **−6.3 / −6.1 / −6.3 @ 0.710/0.574**. Look-ahead is still worse Δacc than default unlike A1 **−4.7 @ 0.650/0.521**. s43 ShuffleNet-v2×1.5 **inside τ** at **−2.4 @ 0.710/0.675** (PASS1 `acc 0.932 -> 0.908 (-0.024) | params x0.710 | FLOPs x0.675`). **Same size** as s42/s44. Three-seed **−2.5 / −2.4 / −2.4 @ 0.710/0.675**. PASS1 also printed `effective-params x0.666` — **do not quote**. s43 catalog **COMPLETED PRELIM**. Child **20412382** PD QOS. Freed GRES → ImageNet **20715875 R** (`rtx_4090`), not band train **20945744**.

s44 **20382198** ShuffleNet-v2×1 **inside τ** at **−1.9 @ 0.723/0.682** (PASS1 `acc 0.924 -> 0.905 (-0.019) | params x0.723 | FLOPs x0.682`, cluster 04:12). **Same size** as s42 **−2.2**. PASS1 also printed `effective-params x0.688` — **do not quote**. s44 RepVGG-A0 **inside τ** at **−7.2 @ 0.709/0.577** (PASS1 `acc 0.943 -> 0.871 (-0.072) | params x0.709 | FLOPs x0.577`, cluster 07:14). **Same size and same Δacc** as s42. s44 RepVGG-A1 **inside τ** at **−6.1 @ 0.710/0.574** (PASS1 `acc 0.944 -> 0.883 (-0.061) | params x0.710 | FLOPs x0.574`, cluster 11:16). **Same size** as s42 **−6.3**. s44 ShuffleNet-v2×1.5 **inside τ** at **−2.4 @ 0.710/0.675** (PASS1 `acc 0.932 -> 0.908 (-0.024) | params x0.710 | FLOPs x0.675`, cluster 19:33). PASS1 also printed `effective-params x0.666` — **do not quote**. s44 catalog **COMPLETED PRELIM**.

s42 ShuffleNet-v2×1 **inside τ** at a harder structural cut than default unlike (72%/68% vs 80%/83%). PASS1 also printed `effective-params x0.688` (masked zeros) — **do not quote** that as the size. Same-loop greedy historically crashed on ShuffleNet grouping; look-ahead produced a TEST. s42 ShuffleNet-v2×1.5 **inside τ** at **−2.5 @ 0.710/0.675** (PASS1 `acc 0.932 -> 0.907 (-0.025) | params x0.710 | FLOPs x0.675`). PASS1 also printed `effective-params x0.665` — **do not quote**. Default unlike **−2.4 @ 0.818/0.801** already inside; FLOP-floor unlike **−2.0 @ 0.821/0.799**. Similar Δacc at a harder structural cut. s42 RepVGG-A0 **inside τ** at **−7.2 @ 0.709/0.577** (0.943→0.871). Default unlike was **−4.8 @ 0.681/0.565** (already inside); FLOP-floor unlike **−3.9 @ 0.792/0.702**. s42 RepVGG-A1 **inside τ** at **−6.3 @ 0.710/0.574**. Default unlike A1 **−4.7 @ 0.650/0.521** (already inside); FLOP-floor unlike **−4.3 @ 0.888/0.735**. Look-ahead is **worse Δacc** at a similar size on both RepVGGs — not a new transfer win. All four unlike nets **inside τ** three seeds. Catalogs **COMPLETED PRELIM**. afterok unlike mild **20412380 COMPLETED** catalog PRELIM (§45). **Do not lock.**

---

## 45. Unlike-family mild — PRELIM (s42 catalog COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **mild** rate-picker at the default stop. Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20412380 COMPLETED** 1 d 1 h 18 m (ended 30 Aug 14:08 cluster). Scientific parent unlike look-ahead s42 **20382196 COMPLETED**. Wave I. s43 **20412382** PD QOS (parent **20382197 COMPLETED**). s44 **20412384 R** (`cs-pheno-04`, started 9 Sep 04:35). Child unlike-random **20412387** PD afterok **20412384**. Do **not** quote the wrap job-mean **−0.02 pp** or `eval_train`.

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

## 47. Unlike-family FLOP floor 0.70 + look-ahead greedy — PRELIM (s42 + s43 + s44 catalogs COMPLETED)

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead greedy** under eval FLOP floor 0.70. Quote `eval_test` FINAL only. Quote ShuffleNet **structural** keep, not masked `effective-params`. Job **20412394 COMPLETED** 2 d 12 h 21 m (ended 5 Sep 22:04 cluster, `Restarts=2`). Wave M. Scientific parent ImageNet s44 **20382192 COMPLETED**. Child **20412395 COMPLETED** 11 h 17 m (ended 6 Sep 12:01 cluster, `ise-pheno-02`). FLAGS `SPECTRA_REWARD_MODE=neon` `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0`. Child **20412396 COMPLETED** 2 d 11 h 47 m (ended 10 Sep 19:27 cluster). FLAGS `SPECTRA_EVAL_LOOKAHEAD=1` `SPECTRA_EVAL_POLICY=l1` `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0` `SPECTRA_REWARD_MODE=neon`. **Heuristic, not DRL.** Child **20412549** HELD. §65.

| Net | FLOP-floor look-ahead s42 / s43 / s44 | unconstrained look-ahead s42 §44 | FLOP-floor unlike DRL s42 §28 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.7 / −2.0 / −2.2 @ 0.801/0.716** | **−2.2 @ 0.723/0.682** | **−1.9 @ 0.809/0.826** | **−1.1 @ 0.800/0.825** | three-seed inside |
| RepVGG-A0 | **−6.9 / −6.6 / −7.1 @ 0.847/0.701** | **−7.2 @ 0.709/0.577** | **−3.9 @ 0.792/0.702** | **−4.8 @ 0.681/0.565** | three-seed inside |
| RepVGG-A1 | **−6.7 / −5.7 / −6.2 @ 0.857/0.702** | **−6.3 @ 0.710/0.574** | **−4.3 @ 0.888/0.735** | **−4.7 @ 0.650/0.521** | three-seed inside |
| ShuffleNet-v2×1.5 | **−2.5 / −2.6 / −2.4 @ 0.771/0.701** | **−2.5 @ 0.710/0.675** | **−2.0 @ 0.821/0.799** | **−2.4 @ 0.818/0.801** | three-seed inside |

s42 ShuffleNet-v2×1 **inside τ** at 80% params / 72% FLOPs (PASS1 `acc 0.924 -> 0.907 (-0.017) | params x0.801 | FLOPs x0.716`, cluster 22:49:07). PASS1 also printed `effective-params x0.763` — **do not quote**. Floor **did** bind vs unconstrained look-ahead **−2.2 @ 0.723/0.682**. s43 **20412395** ShuffleNet-v2×1 TEST **−2.0 @ 0.801/0.716** (PASS1 `acc 0.924 -> 0.904 (-0.020) | params x0.801 | FLOPs x0.716`). **Same size** as s42 first-pass **−1.7** and matches s42 restart **−2.0**. PASS1 also printed `effective-params x0.763` — **do not quote**. Look-ahead is deterministic; 0.3 pp vs first-pass is fine-tune noise (§54), not a seed effect. Two-seed **−1.7 / −2.0 @ 0.801/0.716**. s42 RepVGG-A0 **inside τ** at 85% params / 70% FLOPs (PASS1 `acc 0.943 -> 0.874 (-0.069) | params x0.847 | FLOPs x0.701`, cluster 02:50:24). Floor **did** bind vs unconstrained look-ahead **−7.2 @ 0.709/0.577** (kept more params and FLOPs; similar Δacc). FLOP-floor DRL unlike **−3.9 @ 0.792/0.702** is better Δacc at the same FLOP point. Default unlike already **−4.8 @ 0.681/0.565**. s42 RepVGG-A1 **inside τ** at 86% params / 70% FLOPs (PASS1 `acc 0.944 -> 0.877 (-0.067) | params x0.857 | FLOPs x0.702`, cluster 08:21:50). Floor **did** bind vs unconstrained look-ahead **−6.3 @ 0.710/0.574** (kept more params and FLOPs; similar Δacc). FLOP-floor DRL unlike **−4.3 @ 0.888/0.735** is better Δacc. Default unlike already **−4.7 @ 0.650/0.521**. Not a new transfer win. Keep first-pass table. Restart (`Restarts=2`) re-TESTed ShuffleNet-v2×1 **−2.0 @ 0.801/0.716** (0.924→0.904) — **same size** as first-pass **−1.7**; 0.3 pp is inside §54 resampling noise. Restart also printed `effective-params x0.762` — **do not quote**. Restart re-TESTed RepVGG-A0 **−6.9 @ 0.847/0.701** (0.943→0.874) — **same size and Δacc** as first-pass (look-ahead is deterministic). Restart re-TESTed RepVGG-A1 **−6.6 @ 0.857/0.702** (0.944→0.878) — **same size** as first-pass **−6.7** (0.1 pp). Now ShuffleNet-v2×1.5 **inside τ** at 77% params / 70% FLOPs (PASS1 `acc 0.932 -> 0.907 (-0.025) | params x0.771 | FLOPs x0.701`, cluster 22:04). Quote **structural** keep. PASS1 also printed `effective-params x0.724` — **do not quote**. Floor **did** bind vs unconstrained look-ahead **−2.5 @ 0.710/0.675** (same Δacc, more params kept). FLOP-floor unlike DRL **−2.0 @ 0.821/0.799**; default unlike **−2.4 @ 0.818/0.801**. s42 catalog **COMPLETED PRELIM** all four unlike nets inside τ one seed. s43 **20412395** RepVGG-A0 TEST **−6.6 @ 0.847/0.701** (PASS1 `acc 0.943 -> 0.877 (-0.066)`, cluster 08:54). **Same size** as s42 **−6.9**; 0.3 pp is fine-tune noise. Two-seed **−6.9 / −6.6**. s43 RepVGG-A1 TEST **−5.7 @ 0.857/0.702** (PASS1 `acc 0.944 -> 0.887 (-0.057)`, cluster 10:06). **Same size** as s42 **−6.7**; 1.0 pp is fine-tune noise (look-ahead does not sample the actor). Two-seed **−6.7 / −5.7**. DRL FLOP-floor unlike is still better Δacc on both RepVGGs. s43 ShuffleNet-v2×1.5 TEST **−2.6 @ 0.771/0.701** (PASS1 `acc 0.932 -> 0.906 (-0.026) | params x0.771 | FLOPs x0.701`, cluster 12:01). **Same size** as s42 **−2.5**; 0.1 pp is fine-tune noise. Quote **structural** keep. PASS1 also printed `effective-params x0.724` — **do not quote**. Two-seed **−2.5 / −2.6 @ 0.771/0.701**. Floor **did** bind vs unconstrained look-ahead **−2.5 @ 0.710/0.675** (same Δacc, more params kept). s43 catalog **COMPLETED PRELIM** all four unlike nets inside τ two seeds. s44 **20412396 COMPLETED** §65: all four **same size**, three-seed inside τ. Look-ahead is deterministic; 0.2–0.5 pp vs s42 is fine-tune noise (§54). Child **20412549** still **JobHeldUser** — do not `scontrol release`. **Do not lock.**

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
| A2 matched VGG | Train VGG-16 BN C10 + VGG-16 BN C100 (`database_c10_c100_matched_vgg.json`), rates 1.0/0.9/0.8, SGD-80. | afterok A1 | **20884671 COMPLETED** 2 d 0 h 30 m (ended 8 Sep 12:31 cluster). FLAGS `neon` seed 42. **Do not quote train** / in-loop `eval_test`. |
| A3 residual eval | Skip-train `input_offline_c100_residuals.json`, SGD-80, A2 actor. | afterok A2 | **20884672 COMPLETED** 16 h 3 m (ended 9 Sep 04:35 cluster). Actor `job20884671`. FLAGS `neon` **no det=1** (sampled, like §56). Both residual PASS1s — §61. r20 **miss**; r56 **miss**. Child **20412542 R**. |
| B1 recoverable s43 | Clone **20307403** recipe, seed 43, VGG+ShuffleNet only. | afterok **20382187** | **20884673 COMPLETED** 2 d 3 h 00 m (ended 6 Sep 19:11). **Do not quote train.** Self-eval residuals were sampled — use B2. |
| B2 residual eval s43 | Skip-train residuals, SGD-80. Compare to §31 unmatched s42. | afterok B1 | **20884674 COMPLETED** 9 h 29 m (ended 7 Sep 05:50). FLAGS `neon` **no det=1** (sampled). Both residual PASS1s — §56. r20 miss; r56 inside unmatched. |
| B3 recoverable s44 | Seed 44. | afterok B2 | **20884675 COMPLETED** 1 d 23 h 11 m (ended 9 Sep 05:01 cluster). SGD-80 train — **do not quote train** / in-loop `eval_test`. Child **20715879 R**. |

Displaced: **20412542** (mild s44) **R**; **20715879** (unlike prefer look-ahead) **R**. ImageNet **20715875** and Wave Q **20412555** untouched. Encoder has no dataset id; class count is last-Linear `out_features`. ImageNet 1000-way already transferred (C12) — spoof is the C100-specific test of that coordinate, not a claim that outputs explain ImageNet.

---

## 51. Similar-family FLOP floor 0.70 greedy (L1, no look-ahead) — PRELIM (s42 / s43 / s44 COMPLETED)

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **L1 greedy** (`SPECTRA_EVAL_POLICY=l1`, `SPECTRA_EVAL_LOOKAHEAD=0`) under eval FLOP floor 0.70 (**no prefer**). Quote `eval_test` FINAL only. Skip akamaster r32. Job **20715868 COMPLETED** 2 d 1 h 36 m (ended 4 Sep 14:00 cluster). Child **20715870 COMPLETED** 22 h 27 m (ended 5 Sep 15:38 cluster). Child **20715871 COMPLETED** 22 h 46 m (ended 6 Sep 20:20 cluster). Wave R. Child **20715872 COMPLETED** unlike FLOP-floor greedy s42 (§64). s43/s44 **20715873 / 874** still JobHeldUser.

| Net | FLOP-floor greedy s42 / s43 / s44 | FLOP-floor look-ahead §39 | FLOP-floor mild s42 §43 | DRL FLOP-floor s42 §29 | unconstrained greedy 20202684 | vs τ=10 |
|---|---|---|---|---|---|---|
| ResNet-20 w16 | **−4.8 / −5.7 / −5.1 @ 0.908/0.701** | **−4.6 / −5.9 / −5.2 @ 0.908/0.701** | **−5.6 @ 0.801/0.706** | **−4.4 @ 0.768/0.707** | **−7.7 @ 0.640/0.494** | three-seed inside, size-matched to look-ahead |
| ResNet-56 w10 | **−9.2 / −9.3 / −9.0 @ 0.952/0.702** | **−8.3 / −9.2 / −8.6 @ 0.952/0.702** | **−11.6 @ 0.946/0.701** miss | **−5.9 @ 0.902/0.702** | **−23.1 @ 0.607/0.336** cliff | three-seed inside, size-matched to look-ahead |
| ResNet-44 | **−4.6 / −5.1 / −4.1 @ 0.947/0.702** | **−4.2 / −4.4 / −4.4 @ 0.947/0.702** | **−3.9 @ 0.905/0.703** | **−3.7 @ 0.893/0.702** | **−9.0 @ 0.614/0.353** | three-seed inside, size-matched to look-ahead |
| VGG-19 BN | **−3.1 / −2.7 / −2.5 @ 0.804/0.701** | **−3.6 / −2.7 / −2.8 @ 0.804/0.701** | **−2.8 @ 0.811/0.819** | **−2.7 @ 0.767/0.755** | **−3.0 @ 0.669/0.661** | three-seed inside, size-matched to look-ahead |
| MobileNet-v2×0.75 | **−3.1 / −3.0 / −2.8 @ 0.933/0.700** | **−3.3 / −3.1 / −3.2 @ 0.933/0.700** | **−2.4 @ 0.791/0.703** | **−2.1 @ 0.777/0.701** | **−3.5 @ 0.662/0.494** | three-seed inside, size-matched to look-ahead |
| DenseNet-100 | **−2.2 / −2.2 / −2.3 @ 0.798/0.700** | **−2.6 / −2.1 / −2.8 @ 0.798/0.700** | **−2.2 @ 0.823/0.828** | **−2.2 @ 0.837/0.833** | **−2.3 @ 0.700/0.679** | three-seed inside, size-matched to look-ahead |

s42 r20 **inside τ** at 91%/70% (0.950→0.902). s43 r20 **−5.7** (0.950→0.893) **same size**. s44 **20715871** r20 **−5.1 @ 0.908/0.701** (PASS1 `acc 0.950 -> 0.899 (-0.051)`, cluster 09:12). Three-seed **−4.8 / −5.7 / −5.1 @ 0.908/0.701**. Floor **did** bind vs unconstrained greedy **−7.7 @ 0.640/0.494**. s42 r56 **inside τ** at 95%/70% (0.959→0.867). s43 r56 **−9.3** (0.959→0.866) **same size**. s44 r56 **−9.0 @ 0.952/0.702** (PASS1 `acc 0.959 -> 0.869 (-0.090)`, cluster 09:48). Three-seed **−9.2 / −9.3 / −9.0 @ 0.952/0.702** — still **inside τ**, size-matched to look-ahead three-seed **−8.3 / −9.2 / −8.6**. Unconstrained greedy **cliffed** at **−23.1**. DRL FLOP-floor s42 **−5.9 @ 0.902/0.702** is still better Δacc at a slightly smaller net. Prefer remains the lever for ~70% params. s42/s43/s44 r44 **−4.6 / −5.1 / −4.1** (s44 PASS1 `acc 0.935 -> 0.894 (-0.041)`, cluster 10:18) same size as look-ahead **−4.2 / −4.4 / −4.4**. s42/s43/s44 VGG **−3.1 / −2.7 / −2.5 @ 0.804/0.701** (s44 PASS1 `acc 0.934 -> 0.909 (-0.025)`, cluster 11:09) **same size** as look-ahead **−3.6 / −2.7 / −2.8**. r32 skipped. s42/s43/s44 MobileNet **−3.1 / −3.0 / −2.8 @ 0.933/0.700** (s44 PASS1 `acc 0.937 -> 0.909 (-0.028)`, cluster 13:00) **same size** as look-ahead **−3.3 / −3.1 / −3.2**. s42/s43/s44 DenseNet **−2.2 / −2.2 / −2.3 @ 0.798/0.700** (s44 PASS1 `acc 0.949 -> 0.926 (-0.023)`, cluster 20:20) **same size** as FLOP-floor look-ahead **−2.6 / −2.1 / −2.8**. L1 greedy does not use the actor — Δacc spreads at identical size are fine-tune noise, not seed effects. s42/s43/s44 catalogs **COMPLETED PRELIM**. Child **20715872 COMPLETED** unlike FLOP-floor greedy s42 (§64). s43/s44 **20715873 / 874** still JobHeldUser. **Do not lock.**

Read against §48: under FLOP floor **plus prefer**, greedy sizes to the DRL point and DRL keeps a small Δacc edge. Under the FLOP floor **alone**, greedy still **ties** look-ahead once the floor binds (three-seed, same sizes). The floor, not the look-ahead, is what stops the greedy cliff.

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
| **20967060** | `c100_recoverable_drl_fine_shaped` — log FLAGS **has** `SPECTRA_REWARD_MODE=structural_shaped` | **COMPLETED** 1 d 20 h 6 m (ended 8 Sep 11:50 cluster). Stopped at 231 episodes (`reward_not_improving`). Actor/critic written. Train / in-loop `eval_test` — **do not quote**. kids=NONE by design — do not attach. Freed GPU → unlike FLOP-greedy s42 **20715872**. |
| **20884670–672** | Arm A: spoof → matched VGG DRL → residual eval | **20884670 COMPLETED**. **20884671 COMPLETED** 2 d 0 h (ended 8 Sep 12:31) — train, do not quote. Child **20884672 COMPLETED** 16 h 3 m (ended 9 Sep 04:35) residual eval (sampled; no det=1). Both residuals **miss** §61. |
| **20884673–675** | Arm B: recoverable C100 DRL s43 → residual eval → s44 | **20884673 COMPLETED**. **20884674 COMPLETED** — residual eval sampled §56. **20884675 COMPLETED** 1 d 23 h 11 m (ended 9 Sep 05:01) — train, do not quote. Child **20715879 R**. |

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
| `SPECTRA_SKIP_EVAL_TRAIN=1` | off in code; **on** for skip-train sbatch profiles | Skip the duplicated `eval_train` prune+FT walk. `eval_test` is unchanged. ~2× remaining eval wall-clock. |
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
| **20945574** → **20945576** | `neon` + `cbrt` | optimization conditioning only. **20945574 COMPLETED** train. **20945576 COMPLETED** 3 m 22 s (ended 9 Sep 09:12): r20 and r56 **+0.0 @ 1.000/1.000 identity** §63. Actor `job20945574`, FLAGS det=1 neon. |
| **20945744** → **20945749** | `structural_band` + `cbrt` | conditioning **+** the over-budget grading fix. **20945744 CANCELLED** 10 Sep 04:15 (5-step train; scientifically invalid). C100 residual DRL left. Overnight C10 full-net retrains (not TEST): **21168773 / 838 / 840 / 844**. |

**Read (4 Sep, frozen-actor fork).** If *frozen* argmax collapses to identity everywhere (params ≈1.0, Δacc ≈0), that is not a null result — it is direct confirmation of §52.1: the quoted compressions were produced by exploration noise, not by a learned schedule. **Chain A resolved this fork:** frozen s42 argmax **20945568** is **not** identity (Path 3). Do not retroactively call locked tables exploration noise.

**Read (9 Sep, Chain B Arm A).** neon+cbrt retrain argmax **20945576 is identity** on C10-thin. That is a negative for *cbrt-only* conditioning, not a replay of the frozen-actor fork. Band **20945744 CANCELLED** 10 Sep 04:15 (5-step). Overnight C10 cubes retry is **21168838** (full-net, not TEST until thin child). C100 residual DRL left.

**Not yet done:** re-running the paper catalogs. Gated on Chain A. Overlay is now the working tree (commit this section with the switches). These jobs need `scripts/spectra.sbatch` profiles and `src/*` on the leap until they start.

### 54.5 Replan (6 Sep 12:33 IDT) — sanity first, then retake

Ido: the §54 defects are paper-critical. Status at the ask:

| Defect | Code | GPU TEST | Conclusion so far |
|---|---|---|---|
| Frozen policy sampled + dropout live | `SPECTRA_EVAL_DETERMINISTIC=1` (default **off**) | Sampled **20945567 COMPLETED**. Thin argmax **20945568 COMPLETED**: r20 **−4.4**; r56-w4 **−25.2 @ 0.667**. Similar argmax **20945570 COMPLETED**: DenseNet **−2.1 @ 0.801**. C100 argmax **20945572 COMPLETED**: VGG **−8.3** inside; ShuffleNet **−6.6** inside; residuals **−16.2 / −17.9** miss; RepVGG **−11.3 @ 0.685** miss. | **Path 3** on hard C10 ResNets. C9 mixed split is policy-grade. Do not overwrite locked sampled rows. |
| Prefer arms contain no agent | No code “fix”: the knob is a heuristic. Relabel §30 / §33 / §35 / §48. | Confirmed on disk (byte-identical sizes across seeds). | **Provenance fixed in the draft.** Do not re-run prefer as DRL. |
| Reward band empty on C100 | Telemetry `reward_band_report.py`; `structural_band` / `structural_shaped` / `cbrt` | Diag **20930175 COMPLETED**: C100 **42/42 over-budget**. Shaped train **20967060 COMPLETED** (no TEST; do not quote). Chain B **20945574 COMPLETED** (`neon`+`cbrt`) — train, do not quote. Eval **20945576 COMPLETED identity** §63. Band train **20945744 CANCELLED** 10 Sep 04:15 (5-step). | Histogram insight: band tracks **dataset recoverability**, not class-count (spoof §55.1). cbrt-only did **not** yield a pruning argmax on C10-thin. Band arm **not TESTed** (train was invalid). C100 residual DRL left. |

**GPU order from this stamp (QOS 5):** keep current R jobs. Next free GPU: Chain A **20945567** (sampled thin control) then **20945568** (argmax). Demote remaining C10 heuristic afterok children so they do not beat the sanity chain. Do **not** scancel C100 trains. Do **not** retake the paper catalogs until argmax on thin r20/r56 says whether the quoted compressions survive without sampling.

**How wide to re-run, after Chain A:**

1. If argmax **collapses to identity** (params ≈1, Δacc ≈0): the quoted DRL tables are exploration noise. Paper claim becomes “sampled frozen policy”; do **not** mass-rerun catalogs. Retrain (Chain B) is the next science, not another sampled eval.
2. If argmax **prunes and Δacc holds at matched size**: retake **DRL** catalogs only (similar / unlike / thin / C9 frozen) with `SPECTRA_EVAL_DETERMINISTIC=1`, three seeds. Heuristics (greedy / mild / random / look-ahead / prefer) are unchanged — they never sampled the actor.
3. If argmax **prunes but sizes/Δacc move**: replace the DRL cells that moved; leave heuristics.

**FPGM / BN-scale (6 Sep):** implemented as `SPECTRA_FILTER_IMPORTANCE=fpgm|bn_scale`, default still **`l1`**. Same-loop ranking, not a new agent, not a third-party import (He et al. CVPR 2019 / Liu et al. ICCV 2017 formulas in `src/pruning.py`). Do **not** change the frozen 10-net ranker. **21040934 COMPLETED** (`eval_c10_thin_fpgm`, FLAGS `det=1` + `fpgm`) — §59. **21040935 COMPLETED** 2 h 14 m (ended 8 Sep 07:41 cluster) (`bn_scale`) r20 **−3.8**; r56 **−27.1 @ 0.667/0.465** vs L1 **−25.2** same size — **worse** — §60. Fold into SPECTRA DRL *train* only if those evals beat L1 enough to justify a new actor before 15 Sep — FPGM r56 is **−23.4 vs L1 −25.2** (mildest cliff); BN-scale is **worse**. **Do not retrain.** Keep them on the Pareto as ranking A/Bs.

**6 Sep 15:43 — Chain A sampled catalog TESTed.** **20945567 COMPLETED**. FLAGS: `eval_c10_thin`, `neon`, actor `job20158274`, **no** `SPECTRA_EVAL_DETERMINISTIC`. r20-w2 **−3.9 @ 0.600/0.748** (0.648→0.609) vs `20189046` **−4.2 @ 0.600/0.760**. r56-w4 **−25.4 @ 0.667/0.482** (0.888→0.634) vs **−15.9 @ 0.704/0.550**. Same frozen actor: easy net matches; hard net sampled a smaller, worse net. Do **not** quote wrap means or eval_train r56 **−4.8 @ 0.685/0.492**.

**6 Sep 21:28 — Chain A argmax catalog TESTed.** **20945568 COMPLETED** 1 h 49 m (ended 21:00 cluster). FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `neon`, profile `eval_c10_thin_det`, actor `job20158274`. r20-w2 **−4.4 @ 0.600/0.741** (0.648→0.604). r56-w4 **−25.2 @ 0.667/0.465** (0.888→0.636). Vs sampled **20945567 −25.4 @ 0.667/0.482**. Vs locked **20189046 −15.9 @ 0.704/0.550**. **Identity fork is dead.** **Path 3:** the argmax policy prunes the hard net to the greedy-cliff size (0.667 params), not the milder sampled 0.704. Do **not** overwrite locked −15.9 — quote it as a *sample*. Do not quote `eval_train`. Similar-argmax **20945570 COMPLETED** DenseNet **−2.1 @ 0.801/0.803**. C100 argmax **20945572 R** first PASS1 r20 **−16.2 @ 0.669/0.649** miss §58. Chain B **20945574 R** (`neon`+`cbrt`; eval **20945576** PD). FPGM **21040934 PD**.

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

**Read.** Spoofing the class-count token 100→10 did **not** change the family split. VGG and ShuffleNet still transfer; residuals and RepVGG still miss. Size differences vs §21 are inside §54 resampling noise — do not call them a spoof effect. The encoder seeing `out_features=10` is **not** why C100 residuals fail. Child matched-VGG DRL **20884671 COMPLETED** (train — do not quote). Residual eval **20884672 COMPLETED** (sampled; no det=1): both residuals **miss** §61.

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

**Read with §52.1 and §55.1.** Same architecture (VGG-16) is 100% in-budget on C10 and 100% over-budget on C100 *at the step*. That tracks the **dataset**, not the architecture. Spoof (§55.1) then shows it is **not** the class-count token — it is C100 recoverability under Adam-40 (the FT recipe / task). Shaping arm **20967060 COMPLETED** (`structural_shaped` confirmed in the log FLAGS). Train — do not quote. kids=NONE. Do not quote 20900187 / 20884673 / 20967060 train returns.

---

## 56. C100 residual eval s43 (**20884674**) — PRELIM sampled, not policy

Skip-train SGD-80 residual catalog, actor from recoverable train **20884673**. FLAGS `SPECTRA_REWARD_MODE=neon` only — **`SPECTRA_EVAL_DETERMINISTIC` unset** (default sample + live dropout). Compare to §31 s42 **20353582** only as a size-unmatched *draw*, not a second seed of the policy. Do not overwrite §31. Do not quote 20884673 train / self-eval.

| Net | s43 sampled 20884674 | §31 s42 20353582 | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | **−16.4 @ 0.597/0.611** (0.730→0.566) | **−8.3 @ 0.673/0.627** | miss; unmatched size; **sampled** |
| thin r56-w15 | **−7.5 @ 0.623/0.495** (0.784→0.709) | **−8.4 @ 0.662/0.469** | inside τ; unmatched (harder param cut); **sampled** |

Two-net residual catalog **COMPLETED** 9 h 29 m (ended 7 Sep 05:50 cluster). Child **20884675 COMPLETED** (s44 recoverable train — do not quote). Attach a **det=1** overlay before treating s43 as paper-grade. Do not mix the r20 miss into the C10 train catalog. Do not quote as a second seed of §31.

---

## 57. Similar-family argmax s42 (**20945570**) — PRELIM det=1

Frozen actor `job20158274`, skip-train `eval_offline_similar`, FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`**. Compare to locked §3 s42 only as a *sample*. Do **not** overwrite §3. Skip r32. Do not quote `eval_train`.

| Net | argmax 20945570 | §3 s42 sampled | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | **−5.1 @ 0.669/0.649** (0.950→0.899) | **−5.4 @ 0.603/0.639** | inside; unmatched (argmax kept more params). Near s44 **−5.6 @ 0.669/0.634** |
| thin r56-w10 | **−12.8 @ 0.661/0.421** (0.959→0.831) | **−9.2 @ 0.658/0.488** | **miss**; near-matched params, fewer FLOPs. Ties mild **−13.9 / −12.8 / −14.0 @ 0.661/0.421**. Still beats look-ahead cliff **−22.4 @ 0.702/0.376** |
| ResNet-44 | **−4.5 @ 0.699/0.542** (0.935→0.890) | **−4.3 @ 0.632/0.519** | inside; unmatched (argmax kept more params). Ties mild **−4.7 / −4.1 / −4.2 @ 0.699/0.542**. Beats look-ahead **−8.0 @ 0.703/0.391** |
| VGG-19 BN | **−2.7 @ 0.699/0.738** (0.934→0.907) | **−2.7 @ 0.879/0.882** | inside; unmatched (argmax kept **fewer** params, same Δacc). Better Δacc than look-ahead **−3.5 @ 0.703/0.669** at near-matched params |
| MobileNet-v2×0.75 | **−2.4 @ 0.688/0.587** (0.937→0.913) | **−2.5 @ 0.689/0.630** | inside; near-matched params, fewer FLOPs. Ties sampled s42 and mild **−2.3 / −2.1 / −2.3 @ 0.689/0.666**. Beats look-ahead **−3.2 @ 0.708/0.511** |
| DenseNet-100 | **−2.1 @ 0.801/0.803** (0.949→0.928) | **−2.0 @ 0.847/0.849** | inside; unmatched (argmax kept fewer params). Near sampled Δacc. Ties FLOP-greedy **−2.2 @ 0.798/0.700**. Beats look-ahead **−2.4 @ 0.701/0.679** at more params kept |

Job **20945570 COMPLETED** 1 d 0 h 06 m (ended 7 Sep 21:06 cluster). FLAGS **det=1**. Skip r32. Catalog COMPLETED PRELIM. Child **20945572 COMPLETED** (C100 argmax catalog in §58). Do not overwrite §3.

**Read.** Path 3 on the similar hard net only: locked s42 **−9.2** inside τ was a milder *sample*. Argmax r56-w10 **misses** at the mild-heuristic size. Easy r44 / VGG / MobileNet / DenseNet hold. C5 “DRL beats greedy by ~10 pp” was sampled −9.2 vs greedy −23.1 at unmatched sizes — do not retell that as the policy.

---

## 58. C100 argmax s42 (**20945572**) — PRELIM det=1, catalog COMPLETED

Frozen actor `job20158274`, skip-train `eval_offline_c100`, FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `neon`. Compare to locked §21 s42 only as a *sample*. Do **not** overwrite §21. Skip r32. Do not quote `eval_train`. Quote ShuffleNet **structural** keep. Job **COMPLETED** 6 h 34 m (ended 8 Sep 03:40 cluster). kids=NONE — Chain A end; do not attach a child.

| Net | argmax 20945572 | §21 s42 sampled | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | **−16.2 @ 0.669/0.649** (0.730→0.568) | **−19.3 @ 0.604/0.645** | **miss**; unmatched (argmax kept **more** params). Ties C9 mild **−17.0 / −16.0 / −16.1 @ 0.669/0.649** |
| thin r56-w15 | **−17.9 @ 0.689/0.499** (0.784→0.605) | **−15.0 @ 0.694/0.600** | **miss**; near-matched params, fewer FLOPs. Ties C9 mild **−18.2 / −16.8 / −17.7 @ 0.689/0.499**. Still beats look-ahead cliff **−34.6 @ 0.701/0.357** |
| VGG-16 BN | **−8.3 @ 0.769/0.771** (0.740→0.657) | **−7.5 @ 0.797/0.834** | **inside**; unmatched (argmax kept fewer params). Near mild **−8.0 / −8.3 / −8.0 @ 0.811/0.822**. Beats look-ahead **−8.8 @ 0.701/0.672** at more params kept |
| ShuffleNet-v2×1 | **−6.6 @ 0.726/0.718** (0.726→0.660) | **−3.9 @ 0.833/0.823** | **inside**; unmatched (argmax kept fewer params). Quote **structural** 0.726 (do not quote masked 0.695). Near look-ahead **−6.2 @ 0.736/0.682**. Milder Δacc than look-ahead at similar size; worse than sampled −3.9 at 83% params |
| RepVGG-A0 | **−11.3 @ 0.685/0.556** (0.753→0.640) | **−12.1 @ 0.571/0.446** | **miss**; unmatched (argmax kept **more** params). Ties C9 mild **−11.8 / −11.5 / −11.8 @ 0.684/0.548**. Near look-ahead **−11.4 @ 0.709/0.577** |

Catalog **COMPLETED PRELIM**. kids=NONE — do not attach.

**Read.** C9 mixed split is **policy-grade**: thin residuals and RepVGG miss; **VGG and ShuffleNet stay inside τ**. Locked §21 s42 ShuffleNet **−3.9 @ 0.833** and RepVGG **−12.1 @ 0.571** were milder/harder *samples*. Argmax RepVGG still misses at the mild size (~68.5% params). Do not overwrite §21. Do not mix residual misses into the C10 train catalog. Chain A is done. Chain B Arm A **20945576 COMPLETED identity** §63. Band **20945744 CANCELLED** (5-step). Overnight C10 retrains are not TEST. C100 residual DRL left.

---

## 59. FPGM ranking A/B thin argmax s42 (**21040934**) — PRELIM det=1, catalog COMPLETED

Frozen actor `job20158274`, skip-train `eval_c10_thin`, FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `SPECTRA_FILTER_IMPORTANCE=fpgm` + `neon`. Compare to L1 argmax **20945568** at matched size. Same actor, same catalog, only the filter ranker changes. Do **not** overwrite §5 / §54.5. Skip r32. Do not quote `eval_train`.

| Net | FPGM 21040934 | L1 argmax 20945568 | vs τ=10 |
|---|---|---|---|
| thin r20-w2 | **−4.8 @ 0.600/0.741** (0.648→0.600) | **−4.4 @ 0.600/0.741** (0.648→0.604) | **inside**; **same size**. 0.4 pp is FT noise. Does **not** beat L1 |
| thin r56-w4 | **−23.4 @ 0.667/0.465** (0.888→0.654) | **−25.2 @ 0.667/0.465** (0.888→0.636) | **miss** / cliff; **same size**. 1.8 pp milder than L1. Still ~greedy cliff. Does **not** rescue Path 3 |

Job **COMPLETED** 1 h 47 m (ended 8 Sep 05:27 cluster). Child **21040935** BN-scale R. kids of 21040935 = NONE (ranking A/B leaf).

**Read.** FPGM does **not** beat L1 enough to retrain the frozen 10-net actor. Easy net ties. Hard net stays a 0.667-param cliff (locked sampled **−15.9 @ 0.704** is still a milder *sample*). 1.8 pp vs L1 is inside the cliff band, not a ranking rescue. Keep L1. Quote FPGM as a Pareto ranking A/B. BN-scale first cell is §60. Do not start a new DRL train on FPGM.

---

## 60. BN-scale ranking A/B thin argmax s42 (**21040935**) — PRELIM det=1, catalog COMPLETED

Frozen actor `job20158274`, skip-train `eval_c10_thin`, FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `SPECTRA_FILTER_IMPORTANCE=bn_scale` + `neon`. Compare to L1 argmax **20945568** and FPGM **21040934** at matched size. Same actor, same catalog, only the filter ranker changes. Do **not** overwrite §5 / §54.5 / §59. Skip r32. Do not quote `eval_train`. kids=NONE — ranking A/B leaf; do not attach a child.

| Net | BN-scale 21040935 | L1 argmax 20945568 | FPGM 21040934 | vs τ=10 |
|---|---|---|---|---|
| thin r20-w2 | **−3.8 @ 0.600/0.741** (0.648→0.610) | **−4.4 @ 0.600/0.741** | **−4.8 @ 0.600/0.741** | **inside**; **same size**. 0.6 pp vs L1 is FT noise |
| thin r56-w4 | **−27.1 @ 0.667/0.465** (0.888→0.617) | **−25.2 @ 0.667/0.465** | **−23.4 @ 0.667/0.465** | **miss** / cliff; **same size**. **Worse** than L1 by 1.9 pp. Does **not** rescue Path 3 |

Job **COMPLETED** 2 h 14 m (ended 8 Sep 07:41 cluster). Freed GPU → unlike FLOP-floor look-ahead s44 **20412396 R** (heuristic tail; do not jump Wave Q).

**Read.** Ranking A/B pair is closed: easy net all tie at 60% params; hard net stays a 0.667-param cliff. FPGM **−23.4** is the mildest cliff; BN-scale **−27.1** is the worst. **Keep L1. Do not retrain.** Quote both as Pareto ranking A/Bs. Do not quote 20884671 in-train C100 r20.

---

## 61. C100 residual eval matched-VGG s42 (**20884672**) — PRELIM sampled, catalog COMPLETED

Skip-train SGD-80 residual catalog, actor from matched-VGG train **20884671**. FLAGS `SPECTRA_REWARD_MODE=neon` only — **`SPECTRA_EVAL_DETERMINISTIC` unset** (sampled + live dropout, like §56). Compare to §31 s42 **20353582** and §56 s43 **20884674** as size-unmatched *draws*, not extra seeds of those policies. Do **not** overwrite §21 / C9 / §31. Do not quote 20884671 train / in-loop `eval_test`. Do not quote this job’s `eval_train` wraps.

| Net | s42 sampled 20884672 | §31 s42 20353582 | §56 s43 sampled 20884674 | vs τ=10 |
|---|---|---|---|---|
| thin r20-w16 | **−12.0 @ 0.612/0.598** (0.730→0.610) | **−8.3 @ 0.673/0.627** | **−16.4 @ 0.597/0.611** | **miss**; unmatched; **sampled** |
| thin r56-w15 | **−10.4 @ 0.697/0.402** (0.784→0.680) | **−8.4 @ 0.662/0.469** | **−7.5 @ 0.623/0.495** | **miss**; unmatched (more params, fewer FLOPs); **sampled** |

r20 PASS1 `acc 0.730 -> 0.610 (-0.120) | params x0.612 | FLOPs x0.598` (cluster 8 Sep 20:23:56). r56 PASS1 `acc 0.784 -> 0.680 (-0.104) | params x0.697 | FLOPs x0.402` (cluster 9 Sep 04:34:19). Job **COMPLETED** 16 h 3 m (ended 9 Sep 04:35 cluster). Child **20412542 R** (started after **20884675** freed a GPU). The 84672 GPU went to unlike mild s44 **20412384 R**.

**Read.** Matched-VGG residual eval **both residuals miss τ**. r20 milder Δacc than §56 s43 **−16.4** at a similar size (0.612 vs 0.597); worse than §31 **−8.3**, which kept more params (0.673). r56 **−10.4** just over τ, worse than §31 **−8.4** and §56 **−7.5**, at more params kept (0.697 vs 0.662 / 0.623) but a harder FLOP cut (0.402 vs 0.469 / 0.495). Unmatched sizes. Sampled like §56 — do not treat as a second seed of §31. Matched-VGG train did **not** rescue held-out residuals. Do not mix these misses into the C10 train catalog. Do not overwrite C9.

---

## 62. Unlike-family FLOP 0.70 + prefer look-ahead s42 (**20715879**) — PRELIM heuristic, catalog COMPLETED

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead** (`SPECTRA_EVAL_LOOKAHEAD=1`) under FLOP floor 0.70 **and** prefer Δparams/ΔFLOPs (`SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`, `SPECTRA_EVAL_POLICY=l1`). Actor `job20158274`. FLAGS `SPECTRA_REWARD_MODE=neon` — **heuristic, not DRL** (§54.2). Quote `eval_test` FINAL / `pass 1/1` only. Quote ShuffleNet **structural** keep. Skip r32. Job **COMPLETED** 1 h 59 m (ended 9 Sep 09:08 cluster, `cs-pheno-03`). kids=NONE.

| Net | prefer-LA s42 (20715879) | prefer no-LA s42 §30 | FLOP-floor LA s42 §47 | unconstrained LA s42 §44 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.0 @ 0.871/0.944** (0.924→0.914) | **−1.5 @ 0.871/0.944** | **−1.7 @ 0.801/0.716** | **−2.2 @ 0.723/0.682** | one-seed inside; **same size as §30** |
| RepVGG-A0 | **−3.0 @ 0.715/0.756** (0.943→0.913) | **−4.4 @ 0.715/0.756** | **−6.9 @ 0.847/0.701** | **−7.2 @ 0.709/0.577** | one-seed inside; **same size as §30** |
| RepVGG-A1 | **−2.5 @ 0.705/0.753** (0.944→0.919) | **−3.5 @ 0.705/0.753** | **−6.7 @ 0.857/0.702** | **−6.3 @ 0.710/0.574** | one-seed inside; **same size as §30** |
| ShuffleNet-v2×1.5 | **−1.8 @ 0.879/0.950** (0.932→0.914) | **−2.1 @ 0.879/0.950** | **−2.5 @ 0.771/0.701** | **−2.5 @ 0.710/0.675** | one-seed inside; **same size as §30** |

ShuffleNet×1 PASS1 `acc 0.924 -> 0.914 (-0.010) | params x0.871 | FLOPs x0.944` (08:26:56). RepVGG-A0 PASS1 `acc 0.943 -> 0.913 (-0.030) | params x0.715 | FLOPs x0.756` (08:34:02). RepVGG-A1 PASS1 `acc 0.944 -> 0.919 (-0.025) | params x0.705 | FLOPs x0.753` (08:43:24). ShuffleNet×1.5 PASS1 `acc 0.932 -> 0.914 (-0.018) | params x0.879 | FLOPs x0.950` (09:07:30). Quote **structural** keep. **Same param/FLOP points** as locked prefer-without-look-ahead §30 on all four unlike nets. Look-ahead did **not** change the prefer operating point. Δacc is 0.3–1.4 pp milder than §30 s42 (fine-tune noise on a deterministic heuristic, not a seed effect). Prefer still beats unconstrained look-ahead on both RepVGGs. Default unlike already inside τ. Not a new transfer win. Do not quote wrap job-mean **−0.01 pp**. Catalog **COMPLETED PRELIM**. **Do not lock.**

---

## 63. Chain B Arm A thin eval neon+cbrt s42 (**20945576**) — PRELIM identity, catalog COMPLETED

Skip-train C10-thin catalog, actor from neon+cbrt retrain **20945574** (`runs/job20945574/.../latest_best_actor.pt`). FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `SPECTRA_REWARD_MODE=neon`. This is **not** the frozen 10-net actor (`job20158274`). Quote `eval_test` FINAL / `pass 1/1` only. Job **COMPLETED** 3 m 22 s (ended 9 Sep 09:12 cluster). Child none. Freed GPU → unlike look-ahead s43 **20382197**. Band train **20945744 CANCELLED** 10 Sep 04:15 (5-step; invalid).

| Net | neon+cbrt argmax 20945576 | frozen s42 argmax 20945568 §54.5 | locked sampled s42 20189046 | vs τ=10 |
|---|---|---|---|---|
| thin r20-w2 | **+0.0 @ 1.000/1.000** (0.648→0.648) | **−4.4 @ 0.600/0.741** | **−4.2 @ 0.600/0.760** | identity — no cut |
| thin r56-w4 | **+0.0 @ 1.000/1.000** (0.888→0.888) | **−25.2 @ 0.667/0.465** | **−15.9 @ 0.704/0.550** | identity — no cut |

r20 PASS1 `acc 0.648 -> 0.648 (+0.000) | params x1.000 | FLOPs x1.000` (09:10:20). r56 PASS1 `acc 0.888 -> 0.888 (+0.000) | params x1.000 | FLOPs x1.000` (09:11:02). Catalog **COMPLETED** (thin is two nets). 3-minute wall time matches identity (no prune, no fine-tune).

**Read.** Frozen-actor Path 3 is **unchanged**: argmax **20945568** still prunes. neon+cbrt retrain produced an **identity argmax** on the same thin catalog. cbrt-only optimization conditioning did not yield a pruning policy. Do **not** overwrite locked −15.9 / Path 3 / C9. Do not mass-rerun catalogs on this result. Band train **20945744** was **CANCELLED** 10 Sep 04:15 (5-step train; scientifically invalid TEST of that reward). Overnight C10 retrains (full-net, not TEST): prefer **21168773**, neon cubes **21168838**, prefer-floor **21168840**, F1 **21168844**. Do not quote 20945574 train / post-train similar PASS1s.

---

## 64. Unlike-family FLOP floor 0.70 greedy (L1, no look-ahead) s42 (**20715872**) — PRELIM heuristic, catalog COMPLETED

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **L1 greedy** (`SPECTRA_EVAL_POLICY=l1`, `SPECTRA_EVAL_LOOKAHEAD=0`) under eval FLOP floor 0.70 (**no prefer**). Actor unused. FLAGS `SPECTRA_REWARD_MODE=neon` — **heuristic, not DRL**. Quote `eval_test` FINAL / `pass 1/1` only. Quote ShuffleNet **structural** keep. Skip r32. Job **COMPLETED** 1 d 20 h 22 m (ended 10 Sep 08:12 cluster). Scientific parent similar FLOP-floor greedy catalogs **COMPLETED** §51. Child **20715873** (s43) still **JobHeldUser** — do not `scontrol release`. Freed GRES → F1 full-net train **21168844 R**.

| Net | FLOP-floor greedy s42 (20715872) | FLOP-floor look-ahead s42 §47 | unconstrained look-ahead s42 §44 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−2.0 @ 0.801/0.716** (0.924→0.904) | **−1.7 @ 0.801/0.716** | **−2.2 @ 0.723/0.682** | **−1.1 @ 0.800/0.825** | one-seed inside; **same size as §47** |
| RepVGG-A0 | **−6.7 @ 0.847/0.701** (0.943→0.876) | **−6.9 @ 0.847/0.701** | **−7.2 @ 0.709/0.577** | **−4.8 @ 0.681/0.565** | one-seed inside; **same size as §47** |
| RepVGG-A1 | **−6.5 @ 0.857/0.702** (0.944→0.879) | **−6.7 @ 0.857/0.702** | **−6.3 @ 0.710/0.574** | **−4.7 @ 0.650/0.521** | one-seed inside; **same size as §47** |
| ShuffleNet-v2×1.5 | **−2.4 @ 0.771/0.701** (0.932→0.908) | **−2.5 @ 0.771/0.701** | **−2.5 @ 0.710/0.675** | **−2.4 @ 0.818/0.801** | one-seed inside; **same size as §47** |

ShuffleNet×1 PASS1 `acc 0.924 -> 0.904 (-0.020) | params x0.801 | FLOPs x0.716` — also printed `effective-params x0.762` (**do not quote**). RepVGG-A0 PASS1 `acc 0.943 -> 0.876 (-0.067) | params x0.847 | FLOPs x0.701` (9 Sep 20:08). RepVGG-A1 PASS1 `acc 0.944 -> 0.879 (-0.065) | params x0.857 | FLOPs x0.702` (10 Sep 00:05). ShuffleNet×1.5 PASS1 `acc 0.932 -> 0.908 (-0.024) | params x0.771 | FLOPs x0.701` — also printed `effective-params x0.724` (**do not quote**). **Same param/FLOP points** as FLOP-floor look-ahead §47 on all four nets. Greedy ≈ look-ahead (0.1–0.3 pp); the floor, not look-ahead, sets the unlike greedy operating point — same read as similar-family §51. Default unlike already inside τ at a harder RepVGG cut. Do not quote wrap job-mean **−0.02 pp**. Catalog **COMPLETED PRELIM**. **Do not lock.**

---

## 65. Unlike-family FLOP floor 0.70 + look-ahead greedy s44 (**20412396**) — PRELIM heuristic, catalog COMPLETED

Frozen 10-net skip-train, `eval_offline_novel`, same-loop **look-ahead greedy** under eval FLOP floor 0.70. Actor unused. FLAGS `SPECTRA_EVAL_LOOKAHEAD=1` `SPECTRA_EVAL_POLICY=l1` `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0` `SPECTRA_REWARD_MODE=neon`. **Heuristic, not DRL.** Quote `eval_test` FINAL / `pass 1/1` only. Quote ShuffleNet **structural** keep. Skip r32. Job **COMPLETED** 2 d 11 h 47 m (ended 10 Sep 19:27 cluster, exit 0). Scientific parents s42/s43 **20412394 / 395 COMPLETED** §47. Child **20412549** (`c10-unlike-flop70-mild-s42`) still **JobHeldUser** with dependency fulfilled — do not `scontrol release`. Do not fill the freed GPU to chase 6/8.

| Net | FLOP-floor look-ahead s44 (20412396) | s42 / s43 §47 | unconstrained look-ahead s42 §44 | default unlike s42 §4 | vs τ=10 |
|---|---|---|---|---|---|
| ShuffleNet-v2×1 | **−2.2 @ 0.801/0.716** (0.924→0.902) | **−1.7 / −2.0 @ 0.801/0.716** | **−2.2 @ 0.723/0.682** | **−1.1 @ 0.800/0.825** | three-seed inside; **same size** |
| RepVGG-A0 | **−7.1 @ 0.847/0.701** (0.943→0.872) | **−6.9 / −6.6 @ 0.847/0.701** | **−7.2 @ 0.709/0.577** | **−4.8 @ 0.681/0.565** | three-seed inside; **same size** |
| RepVGG-A1 | **−6.2 @ 0.857/0.702** (0.944→0.882) | **−6.7 / −5.7 @ 0.857/0.702** | **−6.3 @ 0.710/0.574** | **−4.7 @ 0.650/0.521** | three-seed inside; **same size** |
| ShuffleNet-v2×1.5 | **−2.4 @ 0.771/0.701** (0.932→0.908) | **−2.5 / −2.6 @ 0.771/0.701** | **−2.5 @ 0.710/0.675** | **−2.4 @ 0.818/0.801** | three-seed inside; **same size** |

ShuffleNet×1 PASS1 `acc 0.924 -> 0.902 (-0.022) | params x0.801 | FLOPs x0.716` — also printed `effective-params x0.763` (**do not quote**). RepVGG-A0 PASS1 `acc 0.943 -> 0.872 (-0.071) | params x0.847 | FLOPs x0.701`. RepVGG-A1 PASS1 `acc 0.944 -> 0.882 (-0.062) | params x0.857 | FLOPs x0.702`. ShuffleNet×1.5 PASS1 `acc 0.932 -> 0.908 (-0.024) | params x0.771 | FLOPs x0.701` — also printed `effective-params x0.725` (**do not quote**). Three-seed **same size**: ×1 **−1.7 / −2.0 / −2.2**; A0 **−6.9 / −6.6 / −7.1**; A1 **−6.7 / −5.7 / −6.2**; ×1.5 **−2.5 / −2.6 / −2.4**. Look-ahead is deterministic; 0.2–0.5 pp vs s42 is fine-tune noise (§54). FLOP-floor DRL unlike is still better Δacc on both RepVGGs. Greedy s42 §64 is the same operating point (0.1–0.3 pp). Default unlike already inside τ at a harder RepVGG cut. Catalog **COMPLETED PRELIM**. **Do not lock.**

---

## 66. Similar-family FLOP 0.70 + prefer look-ahead greedy s42 (**20412555**) — PRELIM heuristic, catalog COMPLETED

Frozen 10-net skip-train, `eval_offline_similar`, same-loop **look-ahead greedy** (`SPECTRA_EVAL_LOOKAHEAD=1` `SPECTRA_EVAL_POLICY=l1`) under FLOP floor 0.70 **and** prefer Δparams/ΔFLOPs (`SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1`). Actor `job20158274` unused. FLAGS `SPECTRA_REWARD_MODE=neon` seed 42. **Heuristic, not DRL.** Quote `eval_test` FINAL / `pass 1/1` only. Skip akamaster r32. Do not quote `eval_train` (train-loader wrap near +0.00). Job **COMPLETED** 2-00:02:43 (ended 11 Sep 17:02 cluster on `cs-1080-01`, exit 0). Wave Q. Scientific parent unlike-random **20412385 COMPLETED** §49. kids=NONE. Child **20412556** still JobHeldUser `afterok:20412386` — do not release.

| Net | prefer-LA s42 (20412555) | prefer-greedy s42 §48 | DRL prefer s42 §33 | FLOP-floor LA s42 §39 | vs τ=10 |
|---|---|---|---|---|---|
| ResNet-20 w16 | **−3.4 @ 0.717/0.880** (0.950→0.916) | **−3.9 @ 0.717/0.880** | **−3.8 @ 0.717/0.880** | **−4.6 @ 0.908/0.701** | one-seed inside; **same size as §33/§48** |
| ResNet-56 w10 | **−4.1 @ 0.702/0.872** (0.959→0.918) | **−4.4 @ 0.702/0.872** | **−3.8 @ 0.702/0.872** | **−8.3 @ 0.952/0.702** | one-seed inside; **same size as §33/§48** |
| ResNet-44 | **−2.6 @ 0.702/0.872** (0.935→0.909) | **−2.6 @ 0.702/0.872** | **−2.3 @ 0.702/0.872** | **−4.2 @ 0.947/0.702** | one-seed inside; **same size as §33/§48** |
| VGG-19 BN | **−2.4 @ 0.837/0.923** (0.934→0.910) | **−2.9 @ 0.837/0.923** | **−2.2 @ 0.837/0.923** | **−3.6 @ 0.804/0.701** | one-seed inside; **same size as §33/§48** |
| MobileNet-v2×0.75 | **−1.9 @ 0.767/0.912** (0.938→0.919) | **−2.0 @ 0.767/0.912** | **−1.9 @ 0.767/0.912** | **−3.3 @ 0.933/0.700** | one-seed inside; **same size as §33/§48** |
| DenseNet-100 | **−1.8 @ 0.870/0.951** (0.949→0.931) | **−2.1 @ 0.870/0.951** | **−2.0 @ 0.870/0.951** | **−2.6 @ 0.798/0.700** | one-seed inside; **same size as §33/§48** |

Prefer **did** bind the operating point — same param/FLOP sizes as prefer-greedy §48 and DRL prefer §33, not the FLOP-floor-only sizes of §39. Look-ahead is a small Δacc edge vs prefer-greedy at that point (r20 0.5 pp; r56 0.3 pp; VGG 0.5 pp; DenseNet 0.3 pp; r44 tie; MobileNet 0.1 pp). DRL prefer still 0.3 pp better on r56 and r44. All six **inside τ**. r20 PASS1 `acc 0.950 -> 0.916 (-0.034) | params x0.717 | FLOPs x0.880` (cluster 10 Sep 17:00). r56 PASS1 `acc 0.959 -> 0.918 (-0.041) | params x0.702 | FLOPs x0.872` (17:41). r44 PASS1 `acc 0.935 -> 0.909 (-0.026) | params x0.702 | FLOPs x0.872` (18:22). VGG PASS1 `acc 0.934 -> 0.910 (-0.024) | params x0.837 | FLOPs x0.923` (19:36). MobileNet PASS1 `acc 0.938 -> 0.919 (-0.019) | params x0.767 | FLOPs x0.912` (21:47). DenseNet PASS1 `acc 0.949 -> 0.931 (-0.018) | params x0.870 | FLOPs x0.951` (11 Sep 17:01). Skip r32. Catalog **COMPLETED PRELIM**. One seed. **Do not lock.**

---

## 67. C100 unlike-extra argmax s42 (**21168557**) — PRELIM, catalog COMPLETED

Frozen 10-net skip-train, `eval_offline_c100`, input `configs/input_offline_c100_unlike_extra.json` (ShuffleNet-v2×1.5 + RepVGG-A1 only). Actor `job20158274`. FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`** + `neon` seed 42, `SKIP_EVAL_TRAIN=1`. Prefer off. Quote `eval_test` FINAL / `pass 1/1` only. Quote ShuffleNet **structural** keep. Do not quote wrap job-mean **−0.09 pp**. Do **not** overwrite §21 / Path 3 C9. Job **COMPLETED** 13.9 h (ended 12 Sep 07:20 cluster). Freed GPU → band train **21194543**.

| Net | argmax 21168557 | vs τ=10 |
|---|---|---|
| ShuffleNet-v2×1.5 | **−5.9 @ 0.758/0.738** (0.742→0.683) | **inside**. Quote structural 0.758. PASS1 also printed `effective-params x0.716` — **do not quote** |
| RepVGG-A1 | **−12.7 @ 0.667/0.547** (0.764→0.637) | **miss**. Extra unlike net; C9 A0 already misses under Adam-40 |

Catalog **COMPLETED PRELIM**. Two designed nets. Frozen Path 3 actor, not a new DRL train. **Do not lock.**

---

## 68. Prefer snapshot C10-thin argmax (**21229256**) — PRELIM **CLIFF**, catalog COMPLETED

Actor `runs/snapshots/20260913T0025/prefer_job21184512/latest_best_*.pt` + matching standardizer (frozen copy of continue **21184512** `latest_best`, last save 11 Sep 17:33 — no new best in the continue). Skip-train `eval_c10_thin_det`. FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`**, **`SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0`**, `SPECTRA_STATE_ALIGN=next`, `SPECTRA_SKIP_EVAL_TRAIN=1`, seed 42. Quote `eval_test` FINAL / `pass 1/1` only. Skip r32. Do **not** quote wrap job-mean **−0.14 pp** or train return 73.23. Compare to frozen Path 3 argmax **20945568** at matched size. Job **COMPLETED** 1 h 20 m (ended 13 Sep 07:26 cluster, `cs-pheno-01`, exit 0).

| Net | snapshot 21229256 | Path 3 argmax 20945568 | vs τ=10 |
|---|---|---|---|
| thin r20-w2 | **−3.5 @ 0.600/0.753** (0.648→0.613) | **−4.4 @ 0.600/0.741** | **inside**; matched params, unmatched FLOPs (snapshot kept more). 0.9 pp kinder Δacc than Path 3 |
| thin r56-w4 | **−23.8 @ 0.667/0.457** (0.888→0.650) | **−25.2 @ 0.667/0.465** | **miss / CLIFF**; matched params, unmatched FLOPs (snapshot kept fewer). 1.4 pp kinder than Path 3, still ≤ −20 pp at 0.667 |

**Read.** Class **CLIFF**, not identity, not a thesis WIN. The prefer-retrain argmax still drives the hard net to the greedy-cliff size (66.7% params). Easy net is slightly kinder than Path 3 at the same keep. Overnight dispatch **21230428 COMPLETED** 5 s labeled **WIN** because the classifier uses *min* keep (r20 0.600, −3.5 pp) not r56 — so it queued spectrum **plus** FLOP-0.70 and skipped ep99. Scientific class remains CLIFF. Prefer similar **21230664 R**. **Do not lock.** Do not overwrite locked −15.9.

---

## 69. Cubes snapshot C10-thin argmax (**21229257**) — PRELIM, catalog COMPLETED

Actor `runs/snapshots/20260913T0025/cubes_job21184514/latest_best_*.pt` + matching standardizer (frozen copy of continue **21184514** `latest_best`). Skip-train `eval_c10_thin_det`. FLAGS **`SPECTRA_EVAL_DETERMINISTIC=1`**, **`SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0`**, `SPECTRA_STATE_ALIGN=next`, `SPECTRA_SKIP_EVAL_TRAIN=1`, seed 42. Quote `eval_test` FINAL / `pass 1/1` only. Skip r32. Do **not** quote wrap job-mean **−0.13 pp** or train return 75.04. Compare to frozen Path 3 argmax **20945568**. Job **COMPLETED** 45 m (ended 13 Sep 08:11 cluster, exit 0).

| Net | snapshot 21229257 | Path 3 argmax 20945568 | vs τ=10 |
|---|---|---|---|
| thin r20-w2 | **−4.3 @ 0.600/0.741** (0.648→0.605) | **−4.4 @ 0.600/0.741** | **inside**; matched params and FLOPs. Tie vs Path 3 |
| thin r56-w4 | **−21.9 @ 0.722/0.524** (0.888→0.669) | **−25.2 @ 0.667/0.465** | **miss**; unmatched (snapshot kept more params). Still ≤ −20 pp. Not the 0.667 cliff size |

**Read.** Not identity. r20 is a size-matched tie with Path 3. r56 is a *milder keep* (72.2% vs 66.7%) with a still-failing Δacc — do not call this a matched-size cliff fix. Dispatch **21230429** labeled **WIN** and queued similar+FLOP-0.70+unlike+C100 (**21230675–678** PD QOS). Spectrum TESTs decide whether good/mid cells moved. **Do not lock.**

---

## 70. Prefer snapshot similar-family argmax (**21230664**) — PRELIM, catalog COMPLETED

Same actor as §68 (prefer `latest_best`, knob **off**, det=1). Skip-train `eval_offline_similar_det`. Quote `eval_test` FINAL / `pass 1/1` only. Skip r32. Compare to Path 3 similar **20945570**. Job **COMPLETED** 15 h 42 m (ended 13 Sep 23:54 cluster, exit 0). Do not quote wrap **+0.05 pp**.

| Net | snapshot 21230664 | Path 3 argmax 20945570 | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | **−5.2 @ 0.669/0.614** (0.950→0.898) | **−5.1 @ 0.669/0.649** | **inside**; matched params, unmatched FLOPs (snapshot kept fewer). 0.1 pp — tie |
| thin r56-w10 | **−22.7 @ 0.613/0.344** (0.959→0.732) | **−12.8 @ 0.661/0.421** miss | **miss**; unmatched (snapshot kept fewer params and FLOPs). Worse Δacc than Path 3 |
| ResNet-44 | **−8.1 @ 0.620/0.356** (0.935→0.854) | **−4.5 @ 0.699/0.542** inside | **inside** τ; unmatched (snapshot kept fewer). Worse Δacc than Path 3. Near look-ahead greedy **−8.0 @ 0.703/0.391** at a smaller keep |
| VGG-19 BN | **−3.0 @ 0.679/0.706** (0.934→0.904) | **−2.7 @ 0.699/0.738** inside | **inside** τ; unmatched (snapshot kept fewer). 0.3 pp worse |
| MobileNet-v2×0.75 | **−2.1 @ 0.691/0.590** (0.937→0.916) | **−2.4 @ 0.688/0.587** inside | **inside**; near-matched params and FLOPs. 0.3 pp kinder — tie |
| DenseNet-100 | **−2.3 @ 0.697/0.686** (0.949→0.926) | **−2.1 @ 0.801/0.803** inside | **inside**; unmatched (snapshot kept fewer). 0.2 pp worse at a smaller keep. Near FLOP-greedy **−2.2 @ 0.798/0.700** |

**Read.** Catalog COMPLETED. Easy similar cell ties Path 3. Hard similar cell is a **regression**: smaller keep than Path 3 and a deeper miss (−22.7 vs −12.8). r44 stays inside τ but is unmatched and **3.6 pp worse** than Path 3 −4.5, at 0.620 vs 0.699 params. VGG is inside τ, unmatched, **0.3 pp worse** than Path 3 −2.7 at a smaller keep. MobileNet is a **near-matched tie** vs Path 3 −2.4. DenseNet is inside τ, unmatched, **0.2 pp worse** than Path 3 −2.1 at 0.697 vs 0.801 params. Do not treat unmatched size as a matched-size comparison. Skip r32 (this job's r32 origin printed 0.101 — skip, do not quote). Do not quote wrap **+0.05**. Freed GPU → v2a-ep0015 **21238729**. **Do not lock.**

---

## 71. Prefer snapshot C10-thin trajectory (**21233226**) — PRELIM identity, **invalid protocol**

Skip-train `eval_c10_thin_traj`. FLAGS **`SPECTRA_EVAL_TRAJECTORY=1`**, det=1, prefer off, `STATE_ALIGN=next`, lookahead=0. Actor freeze `runs/snapshots/20260913T1105/prefer_job21184512/` (same 11 Sep 17:33 `latest_best` as §68). Job **COMPLETED** 4 m 12 s (ended 13 Sep 11:19 cluster, exit 0). Quote `[eval] TRAJ` only.

| Net | TRAJ floor_hold / val_best / terminal |
|---|---|
| thin r20-w2 | **+0.0 @ 1.000/1.000** (0.648→0.648), step=−1 hold / step=20 terminal. `floor_cross` NONE |
| thin r56-w4 | **+0.0 @ 1.000/1.000** (0.888→0.888), step=−1 hold / step=56 terminal. `floor_cross` NONE |

**Read. Do not quote as a prefer-policy TRAJ.** T1105 copies had actor+critic only — no `standardizer.pt`. Log: `FeatureStandardizer: eval-only with no cache at .../20260913T1105/standardizer.pt; log1p fallback (train z-score / eval log1p mismatch)`. §68 / §70 used the matching z-score cache (`.../20260913T0025/prefer_job21184512/standardizer.pt`, n=721) and **pruned**. This identity is the log1p mismatch, not Chain B. Cubes traj **21233227** r20 is now in §73 (also log1p; pruning). Path 3 **21233223** r20 is §72 (log1p). Redo prefer **21233371 R** with matching std. Do not mix with identity-pad §68. **Do not lock.**

---

## 72. Path 3 C10-thin trajectory (**21233223**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj`. Actor `job20158274`, `STATE_ALIGN=prev`, det=1, `TRAJECTORY=1`. Log1p fallback (`job20158274/standardizer.pt` missing). Quote `[eval] TRAJ` (env shape `param_ratio`). Do not quote wrap **−0.16 pp**. Do not mix with identity-pad **20945568**. Job **COMPLETED** 7 h 36 m (ended 13 Sep 18:51 cluster, exit 0). Caption: argmax-of-bias ≡ mild (audit F4).

| Net / point | TRAJ (shape params) | vs pad Path 3 **20945568** / cubes+std §75 |
|---|---|---|
| r20 floor_hold step=15 | **−3.0 @ 0.755/0.772** (0.648→0.618), val **−3.61 pp** | pad r20 **−4.4 @ 0.600/0.741** — unmatched; hold is a larger keep |
| r20 floor_cross step=16 | **−4.8 @ 0.673/0.740** (0.648→0.600), val **−5.53 pp** | near pad Δacc; TRAJ params are the shape counter |
| r20 val_best = terminal step=20 | **−8.1 @ 0.478/0.663** (0.648→0.567), val **−9.04 pp** | past 0.70; val still inside τ. pass 1/1 params **x0.400** |
| r56 val_best step=7 | **−6.6 @ 0.969/0.802** (0.888→0.823), val **−9.81 pp** | cubes+std val-best **−6.8 @ 0.967** — same almost-no-cut hold |
| r56 floor_hold step=39 | **−21.1 @ 0.708/0.485** (0.888→0.677), val **−23.96 pp** | val **over** τ. Pad Path 3 **−25.2 @ 0.667** was this cliff, not an operating point |
| r56 floor_cross step=40 | **−20.6 @ 0.663/0.465** (0.888→0.682), val **−24.35 pp** | val **over** τ |
| r56 terminal step=56 | **−24.2 @ 0.274/0.299** (0.888→0.646), val **−27.97 pp** | val **over** τ. pass 1/1 **−24.2 @ 0.278/0.299** |

**Read.** Unconstrained Path 3 on the easy net still prunes (val-best **−8.1 at 0.478**, val inside τ). On the hard net val-best is **−6.6 at 96.9% params** — the walk leaves τ as soon as it cuts. Do not pick floor-hold / terminal. The locked pad **−15.9 / −25.2 at ~0.67** is F8 (identity-pad after the cliff), not this policy's `val_best`. Log1p vs a z-score Path 3 replay. **Do not lock.**

---

## 73. Cubes C10-thin trajectory (**21233227**) — PRELIM, catalog COMPLETED, **log1p**

Skip-train `eval_c10_thin_traj`. Freeze `20260913T1105/cubes_job21184514/` (ckpt 13 Sep 10:37), `STATE_ALIGN=next`, det=1, `TRAJECTORY=1`. **Log1p fallback** (no std at submit). Job **COMPLETED** 2 h 31 m (ended 13 Sep 13:46 cluster, exit 0). Quote `[eval] TRAJ`. pass 1/1 param counter disagrees (r20 **x0.600** vs TRAJ **x0.698**; r56 **x0.907** vs TRAJ **x0.894**). Do not quote wrap **−0.06 pp**. Do not treat as a fair z-score compare to pad **§69**. Redo **21235566** pins the matching std.

| Net / point | TRAJ (shape params) | vs pad cubes **21229257** §69 |
|---|---|---|
| r20 floor_hold step=11 | **−0.2 @ 0.865/0.826** (0.648→0.646), val **−0.74 pp** | pad **−4.3 @ 0.600/0.741** — unmatched |
| r20 val_best = floor_cross = terminal | **−3.5 @ 0.698/0.760** (0.648→0.613), val **−4.02 pp** | pass 1/1 **−3.5 @ 0.600/0.760** |
| r56 val_best step=19 | **−7.0 @ 0.965/0.792** (0.888→0.818), val **−9.82 pp** | pad **−21.9 @ 0.722/0.524** miss — unmatched; traj never crossed 0.70 (`floor_cross` NONE) |
| r56 floor_hold = terminal step=56 | **−8.8 @ 0.894/0.678** (0.888→0.800), val **−11.65 pp** | val **over** τ — do not pick as val-best. pass 1/1 **−8.8 @ 0.907/0.678** |

**Read.** Catalog COMPLETED. Not identity. Hard net never reached the 0.70 floor; val-best is **−7.0 at 96.5% params** (inside τ on val, almost no compression). Terminal test **−8.8** is inside τ but val **−11.65** is over — quote val-best. Log1p vs §69 z-score. **Do not lock.** Do not call this a cliff fix vs pad −21.9.

---

## 74. Prefer C10-thin trajectory (**21233371**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj`. Same T1105 prefer actor as invalid §71, but **`SPECTRA_STANDARDIZER_PATH` pinned** (`standardizer.pt` n=721, same cache as §68). FLAGS det=1, prefer off, `STATE_ALIGN=next`, `TRAJECTORY=1`. Job **COMPLETED** 7 h 23 m (ended 13 Sep 18:58 cluster, exit 0). Quote `[eval] TRAJ`. Do not quote wrap **−0.22 pp**. Compare to Path 3 traj **§72**, not to pad §68.

| Net / point | TRAJ (shape params) | vs Path 3 traj **21233223** §72 |
|---|---|---|
| r20 floor_hold step=15 | **−2.9 @ 0.763/0.785** (0.648→0.619), val **−2.69 pp** | Path 3 hold **−3.0 @ 0.755/0.772** — near-tie |
| r20 floor_cross step=16 | **−3.6 @ 0.682/0.753** (0.648→0.612), val **−4.56 pp** | Path 3 cross **−4.8 @ 0.673/0.740** |
| r20 val_best = terminal step=20 | **−7.5 @ 0.470/0.670** (0.648→0.573), val **−7.88 pp** | Path 3 **−8.1 @ 0.478/0.663** — same deep walk |
| r56 val_best step=15 | **−7.4 @ 0.966/0.784** (0.888→0.814), val **−9.99 pp** | Path 3 **−6.6 @ 0.969** — same almost-no-cut hold |
| r56 floor_hold step=38 | **−22.6 @ 0.789/0.511** (0.888→0.662), val **−25.85 pp** | val **over** τ |
| r56 floor_cross step=39 | **−22.0 @ 0.654/0.454** (0.888→0.668), val **−25.55 pp** | val **over** τ |
| r56 terminal step=56 | **−35.8 @ 0.142/0.235** (0.888→0.530), val **−39.43 pp** | val **over** τ. pass 1/1 **−35.8 @ 0.148/0.235** |

**Read.** Matching-std prefer traj is **not identity** (that was §71 log1p). On both nets it clones Path 3's unconstrained curve. Hard-net val-best is **−7.4 at 96.6% params**. Retrain did not yield a different schedule. Uniform-policy prefer actor (audit F3). **Do not lock.**

---

## 75. Cubes C10-thin trajectory with matching std (**21235566**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj`. Same T1105 cubes actor as log1p **§73** (ckpt 13 Sep 10:37), **`SPECTRA_STANDARDIZER_PATH` pinned** (`n=721`). FLAGS det=1, prefer off, `STATE_ALIGN=next`, `TRAJECTORY=1`. Job **COMPLETED** 1 h 3 m (ended 13 Sep 16:22 cluster, exit 0). Quote `[eval] TRAJ`. Do not quote wrap **−0.18 pp**. Compare to Path 3 traj **§72** and log1p cubes **§73**, not to pad §69.

| Net / point | TRAJ (shape params) | vs Path 3 §72 / log1p §73 |
|---|---|---|
| r20 floor_hold step=14 | **−1.5 @ 0.824/0.809** (0.648→0.633), val **−2.60 pp** | Path 3 hold **−3.0 @ 0.755**; log1p hold **−0.2 @ 0.865** |
| r20 val_best step=19 | **−8.6 @ 0.442/0.658** (0.648→0.562), val **−9.01 pp** | Path 3 **−8.1 @ 0.478** — same deep walk. log1p was **−3.5 @ 0.698** |
| r20 terminal step=20 | **−17.0 @ 0.378/0.633**, val **−17.85 pp** | val **over** τ — do not pick |
| r56 val_best step=17 | **−6.8 @ 0.967/0.793** (0.888→0.820), val **−8.96 pp** | Path 3 §72 **−6.6 @ 0.969**. log1p **−7.0 @ 0.965** |
| r56 floor_hold step=31 | **−18.1 @ 0.834/0.584**, val **−21.54 pp** | val **over** τ |
| r56 terminal step=56 | **−19.1 @ 0.685/0.520**, val **−22.35 pp** | val **over** τ. pass 1/1 **−19.1 @ 0.685/0.520** |

**Read.** Matching std fixes the easy-net walk (r20 clones Path 3). It does **not** rescue the hard net: r56 val-best is still **−6.8 at 96.7% params**. Do not pick terminal. Uniform-policy cubes actor (audit F3). **Do not lock.**

---

## 76. Provenance audit of the DRL loop (13 Sep) — corrects how every DRL row was produced

Full audit with `file:line` cites, patch series and experiment card: `docs/AUDIT_13SEP_OVERHAUL.md`. No new GPU time; evidence is `runs/job*/manifest.json`, `runs/job*/events/rank0.jsonl` and two CPU-partition probe jobs (**21236549**, **21236614**). Nothing here overwrites a locked row; it captions them.

| Finding | Disk evidence | What it changes |
|---|---|---|
| Frozen 10-net actors trained on **5-step** episodes (`rollout_limit 5`, `passes 1`): rows 0–4 only, row 0 forced identity. s42 115 ep, s43 358 ep, s44 300 ep, all `steps: 5`. | `runs/job20158274/manifest.json`; `events/rank0.jsonl` `train_config` | TEST walks 21 / 57 rows; rows ≥ 5 never carried a policy gradient. |
| **s42 `job20158274` never left warm-up** (`warmup_len 200`, stopped at ep 114). Warm-up actions are uniform random; the 14 Aug code still back-propagated them into the actor (off-policy). `latest_best` = warm-up **ep 34**. | same events file; `git show a8fe748:src/A2C_Agent_Reinforce.py` | Every "frozen s42" row (§3–§5, §17 `20189046`, Chain A `20945567/568/570/572`, traj `21233223`) is an actor with zero on-policy episodes. |
| **All trained policies are uniform over the legal actions** (entropy at the legal-set maximum to 5 digits: 0.87889 = 0.8·ln 3 for s42/s43/s44; 1.046292 = (20/21)·ln 3 for prefer/cubes). Checkpoint probe: max−min action prob 0.4 % (s42), 0.35 % (s43), 0.7 % (s44), ~2 % (prefer), 0.7 % (cubes). | `entropy` in `episode` records; probe **21236549** | `det=1` argmax is the head's bias vector (s42: `[0.015, 0.045, −0.004]` → rate 0.9). No learned schedule exists to transfer. |
| **Path 3 argmax ≡ mild**: in `20945568` every non-identity action on r20-w2 / r56-w4 is **0.9**; identities are mask-forced (stem, width ≤ 2, pad). | `runs/job20945568/events/rank0.jsonl` `step` records | DRL-vs-mild rows on thin nets compare mild with itself. |
| **Cliff mechanism = group multiplicity.** Stage-2 stream of r56-w4 cut 8→7→6→5→4→3→2 on six consecutive owning rows (`conv2` ×5 + shortcut), val Δacc −10 → −27 pp; conv1s lost one channel each. Reproduced exactly (36 110 params) by `tests/test_action_space_representability.py`. A 0.95 rung rounds onto 0.9/0.8 on widths 4/8/16. | same step records | Lever = cut each coupled group once per pass (`SPECTRA_GROUP_ONCE_PER_PASS`, default off), not a finer ladder. |
| **`pass k/K` param ratios quantised** by `round(n/1e6, 3)`: r20-w2 (4 556 params) `x0.600` ⇒ [0.55, 0.77). True Path 3 pad points: r20-w2 **0.673**, r56-w4 **0.663**. | `src/NetworkEnv.py` (old `compute_and_log_results`); fixed in the 13 Sep patch | Caption every r20-w2 `0.600` "size-matched" claim (§16, §17, §63, §68, §69). TRAJ counter was always exact. |
| Prefer / cubes parents **21168773 / 21168838** were **cold** (`CONTINUE_TRAIN=0`, no checkpoint path), not warm-starts of 20158274. | their `SPECTRA_*` env in the slurm logs | Ledger narrative correction only. |

**13 Sep 17:53 IDT — v2 campaign fired (audit Part II).** Gate **21237066** (smoke_v2, integration) → arms afterok: **21237253** offline_train_v2a (PPO + accuracy-slack/progress/kept-ratio state + group-once + cold zero-init head, 3 rates, structural+cbrt, train-FT 12 ep), **21237254** offline_train_v2b (A + (rate, ranking) actions 1.0 | 0.9/0.8 × {l1, fpgm}), **21237255** offline_train_v2c (A with the original nominal-rate NEON reward, raw). Checkpoints carry policy_config.json + standardizer.pt; TEST with eval_c10_thin_traj (contract auto-pinned) against the group-once heuristics `baseline_c10_{mild,l1}_traj_gonce`. Go/no-go and win criteria: docs/AUDIT_13SEP_OVERHAUL.md §9; ops instructions: docs/PROMPT_OPS_GROK_V2.md. 208 unit tests pass (CPU job 21237051). 18:25 IDT: Ido cancelled **21184407** (prefer-floor continue; ep 104, entropy 0.9887 = 0.9·ln 3, uniform — low ROI) to free the QOS slot; **21237255** (v2c) started at 18:26 on ise-6000-06 (rtx_6000). All three arms R by 18:27.

**Rule from this section.** Quote frozen-actor rows as "argmax-of-bias rate picker over the generic structured-pruning environment (≡ mild on thin nets)". The genericity claim rests on the environment, the same-loop heuristics and the τ-band TEST protocol. Next TEST: the five-arm group-once card in the audit (`eval_c10_thin_traj_gonce` vs `21233223`; `baseline_c10_mild_traj[_gonce]`; `baseline_c10_l1_traj_gonce`). Do not retrain from any 20158274-family checkpoint.

---

## 77. Group-once C10-thin: v2a ep0003 (**21237620**) vs mild-once (**21237621**) — PRELIM, catalogs COMPLETED

v2a: skip-train `eval_c10_thin_traj` of **21237253** `snapshots/ep0003` (PPO update 1, `batch_score=0.198`). `[policy_config]` pinned `next` / group-once / slack / budget. Caption: **still near-uniform** at freeze. H1: `baseline_c10_mild_traj_gonce` **21237621**. Quote `[eval] TRAJ`. Do not quote wrap. Jobs **COMPLETED** 14 Sep 00:18 / 00:24 cluster.

| Arm / point | TRAJ (shape params) | vs Path 3 §72 (plain walk) |
|---|---|---|
| v2a r20 floor_hold = val_best = terminal step=19 | **+0.7 @ 0.746/0.806** (0.648→0.655), val **−0.42 pp** | Path 3 val-best **−8.1 @ 0.478**. `floor_cross` **NONE** |
| mild-once r20 same points step=19 | **+0.4 @ 0.746/0.806** (0.648→0.652), val **−0.49 pp** | **same keep** as v2a. `floor_cross` **NONE** |
| v2a r56 val_best step=38 | **−6.1 @ 0.923/0.769** (0.888→0.827), val **−8.59 pp** | Path 3 val-best **−6.6 @ 0.969**. Same almost-no-cut hold |
| mild-once r56 val_best step=38 | **−7.1 @ 0.923/0.769** (0.888→0.817), val **−9.11 pp** | **same keep** as v2a. 1.0 pp worse Δacc |
| v2a r56 floor_hold = terminal step=55 | **−8.2 @ 0.757/0.698** (0.888→0.806), val **−11.27 pp** | val **over** τ — do not pick. `floor_cross` **NONE** |
| mild-once r56 floor_hold = terminal step=55 | **−9.1 @ 0.757/0.698** (0.888→0.797), val **−11.99 pp** | val **over** τ — do not pick. `floor_cross` **NONE** |

**Read.** Catalogs COMPLETED. A1 ≈ H1 on both nets at identical keep. Easy net: 0.3 pp, 0.746 params. Hard net val-best: **1.0 pp kinder** at **0.923** — below the ≥2 pp win bar; still almost no compression vs greedy-once 0.639 (not TESTed yet; **21237622** PD). Floor-hold val is over τ at 0.757 on both arms — do not pick it. The lever vs Path 3 −8.1 @ 0.478 is **group-once**, not the ep0003 actor. Later snap r20 **21238729 −0.4 @ 0.746** is the same keep, slightly worse. Do not call this a DRL win. Do not quote wrap. **Do not lock.** Do not edit the draft until Ido sees this catalog.

---

## 78. v2a ep0015 (**21238729**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21237253** `snapshots/ep0015` (`batch_score=0.252`). `[policy_config]` pinned. Quote `[eval] TRAJ`. Job **COMPLETED** 1 h 32 m (ended 14 Sep 01:26 cluster, exit 0). Do not quote wrap **−0.05 pp**. Freed GPU → l1-once **21237622**.

| Arm / point | TRAJ (shape params) | vs ep0003 §77 / mild-once |
|---|---|---|
| r20 floor_hold = val_best = terminal step=19 | **−0.4 @ 0.746/0.806** (0.648→0.644), val **−0.47 pp** | ep0003 **+0.7 @ 0.746**; mild-once **+0.4 @ 0.746**. **same keep**. `floor_cross` **NONE** |
| r56 val_best step=38 | **−6.7 @ 0.923/0.769** (0.888→0.821), val **−9.45 pp** | ep0003 **−6.1 @ 0.923**; mild-once **−7.1 @ 0.923**. **same keep** |
| r56 floor_hold = terminal step=55 | **−8.7 @ 0.757/0.698** (0.888→0.801), val **−11.69 pp** | val **over** τ — do not pick. `floor_cross` **NONE** |

**Read.** Catalog COMPLETED. Later A snapshot is still the group-once plateau: r20 0.746, r56 val-best 0.923. 0.6 pp worse than ep0003 on r56, 0.4 pp kinder than mild-once — inside noise, not a 2 pp win. Train `batch_score` 0.252 did not buy a different walk. **Do not lock.**


---

## 79. v2a ep0019 (**21239023**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21237253** `snapshots/ep0019` (`batch_score=0.285`). `[policy_config]` pinned `next` / group-once / slack / budget. Quote `[eval] TRAJ`. Job **COMPLETED** 5 h 11 m (ended 14 Sep 05:35 cluster, exit 0). Do not quote wrap **−0.04 pp**. Do not pick r56 floor-hold (val over τ).

| Arm / point | TRAJ (shape params) | vs ep0003 §77 / mild-once |
|---|---|---|
| r20 floor_hold = val_best = terminal step=19 | **+0.4 @ 0.746/0.806** (0.648→0.652), val **−0.56 pp** | ep0003 **+0.7 @ 0.746**; mild-once **+0.4 @ 0.746**. **same keep**. `floor_cross` **NONE** |
| r56 val_best step=38 | **−6.5 @ 0.923/0.769** (0.888→0.823), val **−9.79 pp** | ep0003 **−6.1 @ 0.923**; mild-once **−7.1 @ 0.923**. **same keep** |
| r56 floor_hold = terminal step=55 | **−8.1 @ 0.757/0.698** (0.888→0.807), val **−11.52 pp** | val **over** τ — do not pick. `floor_cross` **NONE** |

**Read.** Catalog COMPLETED. Best A train-score snap is still the group-once plateau (r20 0.746, r56 val-best 0.923). r56 is 0.4 pp worse than ep0003, 0.6 pp kinder than mild-once — not a 2 pp win. Train `batch_score` 0.285 did not buy a different walk. **Do not lock.**

---

## 80. v2b ep0015 (**21238730**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21237254** `snapshots/ep0015` (`batch_score=0.287`). `[policy_config]` pinned 5 actions `1.0 0.9 0.8 0.9 0.8` / `none l1 l1 fpgm fpgm`. Quote `[eval] TRAJ`. Job **COMPLETED** 5 h 11 m (ended 14 Sep 05:30 cluster, exit 0). Do not quote wrap **−0.05 pp**. Do not pick r56 floor-hold / cross / terminal (val over τ). Do not quote terminal **0.639** as an operating point.

| Arm / point | TRAJ (shape params) | vs mild-once §77 / l1-once §81 / v2a |
|---|---|---|
| r20 floor_hold step=15 | **−0.9 @ 0.702/0.789** (0.648→0.639), val **−2.61 pp** | l1-once **−5.2 @ 0.702** — **same keep**, 4.3 pp kinder |
| r20 floor_cross step=17 | **−2.1 @ 0.654/0.770** (0.648→0.626), val **−4.03 pp** | l1-once **−3.7 @ 0.654** — **same keep**, 1.6 pp kinder |
| r20 val_best = terminal step=19 | **−1.2 @ 0.606/0.750** (0.648→0.636), val **−2.30 pp** | l1-once **−3.1 @ 0.606** — **same keep**, 1.9 pp kinder. mild-once **+0.4 @ 0.746** unmatched keep |
| r56 val_best step=38 | **−7.1 @ 0.879/0.702** (0.888→0.818), val **−9.87 pp** | mild-once **−7.1 @ 0.923** — **equal Δacc, less keep**. l1-once **−5.9 @ 0.973** unmatched. v2a-ep0019 **−6.5 @ 0.923** |
| r56 floor_hold step=45 | **−9.1 @ 0.704/0.627** (0.888→0.797), val **−11.39 pp** | val **over** τ — do not pick. l1-once **−10.4 @ 0.704** also over τ |
| r56 floor_cross step=47 | **−9.7 @ 0.691/0.622** (0.888→0.791), val **−11.85 pp** | val **over** τ — do not pick |
| r56 terminal step=56 | **−9.7 @ 0.639/0.599** (0.888→0.791), val **−12.71 pp** | greedy-once bar 0.639; val **over** τ — **not** an operating point |

**Read.** Catalog COMPLETED. First v2 TEST off the group-once plateau on **both** nets. r56 val-best meets the audit keep-at-equal-Δacc bar vs mild-once (**−7.1 @ 0.879** vs **−7.1 @ 0.923**). Val **−9.87** is inside τ by 0.13 pp — do not treat as comfortable. Does **not** reach 0.639 inside τ (same failure as l1-once). r20 vs l1-once at 0.606 is 1.9 pp kinder, shy of 2 pp. vs mild-once r20 is unmatched keep. **Do not lock.** Do not call this a draft-ready DRL win.

---

## 81. L1 group-once (**21237622**) — PRELIM, catalog COMPLETED

Skip-train `baseline_c10_l1_traj_gonce`. Quote `[eval] TRAJ`. Same walk as the v2 actors. Job **COMPLETED** 1 h 34 m (ended 14 Sep 02:59 cluster, exit 0). Freed GPU → Path 3+once **21237623**. Do not quote wrap. Do not pick r56 terminal (val over τ).

| Arm / point | TRAJ (shape params) | vs v2b §80 / mild-once §77 / v2a §77 |
|---|---|---|
| r20 floor_hold step=15 | **−5.2 @ 0.702/0.789** (0.648→0.596), val **−5.56 pp** | v2b hold **−0.9 @ 0.702** — **same keep**, 4.3 pp kinder |
| r20 floor_cross step=17 | **−3.7 @ 0.654/0.770** (0.648→0.611), val **−3.54 pp** | v2b cross **−2.1 @ 0.654** — **same keep**, 1.6 pp kinder |
| r20 val_best = terminal step=19 | **−3.1 @ 0.606/0.750** (0.648→0.617), val **−4.10 pp** | v2b **−1.2 @ 0.606** — **same keep**, 1.9 pp kinder |
| r56 val_best step=19 | **−5.9 @ 0.973/0.842** (0.888→0.829), val **−8.70 pp** | v2a **−6.1 @ 0.923**; mild **−7.1 @ 0.923**. Almost no cut |
| r56 floor_hold step=45 | **−10.4 @ 0.704/0.627** (0.888→0.784), val **−13.39 pp** | val **over** τ — do not pick |
| r56 floor_cross step=47 | **−10.4 @ 0.691/0.622** (0.888→0.784), val **−14.03 pp** | val **over** τ — do not pick |
| r56 terminal step=56 | **−11.9 @ 0.639/0.599** (0.888→0.769), val **−14.49 pp** | the 0.639 greedy-once bar; val **over** τ — **not** an operating point |

**Read.** Catalog COMPLETED. r20: v2b is 1.9 pp kinder at equal 0.606 — not a 2 pp win. r56: l1-once **does** reach 0.639, but only after val leaves τ. The quoted operating point is val-best **−5.9 at 97.3% params**. v2b §80 also reaches 0.639 only with val over τ; its quoted point is **−7.1 @ 0.879**. **Do not lock.**

---

## 82. Path 3 + group-once (**21237623**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj_gonce`. Frozen Path 3 actor `job20158274`, group-once on. Quote `[eval] TRAJ`. Job **COMPLETED** 1 h 32 m (ended 14 Sep 04:32 cluster, exit 0). Do not quote wrap. Do not pick r56 floor-hold (val over τ).

| Arm / point | TRAJ (shape params) | vs mild-once §77 / v2a §77 |
|---|---|---|
| r20 floor_hold = val_best = terminal step=19 | **−1.4 @ 0.746/0.806** (0.648→0.634), val **−1.36 pp** | mild-once **+0.4 @ 0.746**; v2a-ep0003 **+0.7 @ 0.746**. **same keep**. `floor_cross` **NONE** |
| r56 val_best step=38 | **−7.0 @ 0.923/0.769** (0.888→0.818), val **−9.22 pp** | mild-once **−7.1 @ 0.923**; v2a-ep0003 **−6.1 @ 0.923**. **same keep** |
| r56 floor_hold = terminal step=55 | **−8.5 @ 0.757/0.698** (0.888→0.803), val **−11.44 pp** | val **over** τ — do not pick. `floor_cross` **NONE**. mild-once **−9.1 @ 0.757** also over τ |

**Read.** Catalog COMPLETED. Same group-once plateau as mild-once and v2a (r20 0.746, r56 val-best 0.923). r20: **1.8 pp worse** than mild-once at equal keep. r56 val-best: **0.1 pp kinder** than mild-once — noise, not a 2 pp win. Audit F4 (argmax ≡ mild) is a same-keep family, not an identity on Δacc. Freed GPU left idle for Ido at 07:20 (similar-family v2 vs wait for a snap that clears +0.03). **Do not lock.**

---

## 83. Unlike-family TRAJ of v2b-ep0015 (**21252197**) — PRELIM, catalog COMPLETED

Skip-train `eval_offline_novel` of **21237254** `snapshots/ep0015`. `[policy_config]` pinned 5-action B menu. Quote `[eval] TRAJ val_best`. Job **COMPLETED** 13 h 56 m (ended 15 Sep 01:13 cluster, exit 0). Mild-unlike control **§86**. L1-unlike **§84**. Skip r32 (none in this catalog). Do not quote wrap.

| Net | v2b val_best | mild-unlike `21275601` (§86) | l1-unlike `21298586` (§84) |
|---|---|---|---|
| ShuffleNet-v2×1 | **−1.9 @ 0.776/0.788**, val **−7.87** (no floor_cross) | **−1.7 @ 0.906/0.901**, val **−7.48** | **−1.9 @ 0.776/0.788**, val **−7.35** |
| RepVGG-A0 | **−4.4 @ 0.643/0.642**, val **−8.95** | **−4.3 @ 0.811/0.810**, val **−8.57** (no floor_cross) | **−3.9 @ 0.643/0.642**, val **−7.84** |
| RepVGG-A1 | **−4.3 @ 0.641/0.640**, val **−8.81** | **−3.4 @ 0.808/0.809**, val **−8.33** (no floor_cross) | **−3.2 @ 0.641/0.640**, val **−7.21** |
| ShuffleNet-v2×1.5 | **−2.4 @ 0.783/0.793**, val **−8.29** | **−2.2 @ 0.907/0.902**, val **−7.99** | **−2.2 @ 0.783/0.793**, val **−7.14** |

**Read.** Catalog COMPLETED for the actor. Vs mild: unmatched keeps (v2b ~0.64–0.78 vs mild ~0.81–0.91). Vs l1-once: **same keep on all four nets**. ShuffleNet TESTs match; RepVGG l1 is kinder at equal keep (A0 0.5 pp, A1 1.1 pp). Ranking did not beat L1 on unlike. Val inside τ. **Not** a Gilad equal-keep win. Cheap abort still **no**. Do not lock.

---

## 84. Unlike-family TRAJ of l1-once (**21298586**) — PRELIM, catalog COMPLETED

Skip-train `baseline_c10_l1_traj_gonce` on `input_offline_novel.json`. Quote `[eval] TRAJ val_best`. Job **COMPLETED** 3 h 27 m (ended 15 Sep 12:45 cluster, exit 0). Same walk as the v2 actors (group-once, TRAJ, det=1). Do not quote wrap **−0.03 pp**. ShuffleNet×1.5 PASS1 also printed `effective-params x0.731` — **do not quote**. Skip r32. Freed GPU → H0 plain-walk mild thin **21315161**.

| Net | l1-once val_best | v2b-ep0015 §83 | vs v2b at equal keep |
|---|---|---|---|
| ShuffleNet-v2×1 | **−1.9 @ 0.776/0.788**, val **−7.35** (no floor_cross) | **−1.9 @ 0.776/0.788**, val **−7.87** | same TEST, same size |
| RepVGG-A0 | **−3.9 @ 0.643/0.642**, val **−7.84** | **−4.4 @ 0.643/0.642**, val **−8.95** | l1 **0.5 pp kinder** |
| RepVGG-A1 | **−3.2 @ 0.641/0.640**, val **−7.21** | **−4.3 @ 0.641/0.640**, val **−8.81** | l1 **1.1 pp kinder** |
| ShuffleNet-v2×1.5 | **−2.2 @ 0.783/0.793**, val **−7.14** (no floor_cross) | **−2.4 @ 0.783/0.793**, val **−8.29** | l1 0.2 pp kinder |

**Read.** Catalog COMPLETED PRELIM. On unlike, v2b-ep0015 **walked to the greedy-once size** on every net. ShuffleNet TESTs are a match. On both RepVGGs, L1 is kinder at that same keep (shy of Gilad's 2 pp bar, and the *heuristic* is the kinder one). Frozen-actor caption: this is not a ranking-action transfer win vs L1. Mild-unlike (`21275601`) still unmatched-keep. Cheap abort still **no**. Do not lock. Do not overwrite §4 / §83.

---

## 85. Plain-walk mild TRAJ H0 (**21315161**) — PRELIM, catalog COMPLETED

Skip-train `baseline_c10_mild_traj` on `input_c10_thin.json` (no `GROUP_ONCE`). Quote `[eval] TRAJ val_best`. Job **COMPLETED** 2 h 15 m (ended 15 Sep 15:23 cluster, exit 0). Audit F4 twin of unconstrained Path 3. Do not quote wrap **−0.18 pp**. Do not quote r20 terminal (val **−10.41** over τ) or r56 floor-hold / terminal (val **−27 / −29**). Skip r32. Freed GPU → l1-plain thin **21325687**.

| Net | H0 val_best | mild-once §77 | Path 3 unconstrained §72 |
|---|---|---|---|
| r20-w2 | **−5.3 @ 0.552/0.692**, val **−5.97** | **+0.4 @ 0.746/0.806** (unmatched keep) | **−8.1 @ 0.478** (unmatched keep) |
| r56-w4 | **−7.5 @ 0.964/0.776**, val **−9.67** | **−7.1 @ 0.923/0.769** (almost-no-cut) | **−6.6 @ 0.969** (almost-no-cut) |

**Read.** Catalog COMPLETED PRELIM. Group-once is the lever that parked r20 at 0.746: without it, mild walks to **0.552**. Hard-net val_best still refuses to compress (**0.964**), same as unconstrained Path 3. Do not pick r56 floor_hold **−23.3 @ 0.708** (val over τ). Not a DRL row. **Do not lock.** Do not overwrite §72 / §77.

---

## 86. Unlike-family TRAJ of mild-once (**21275601**) — PRELIM, catalog COMPLETED

Skip-train `baseline_c10_mild_traj_gonce` on `input_offline_novel.json`. Quote `[eval] TRAJ val_best`. Job **COMPLETED** 20 h 1 m (ended 15 Sep 17:32 cluster, exit 0). Same walk as the v2 actors (group-once, TRAJ, det=1). Do not quote wrap **−0.03 pp**. ShuffleNet×1.5 PASS1 also printed `effective-params x0.873` — **do not quote**. Skip r32. Unlike trio now complete (v2b §83, l1 §84, mild this section). Freed GPU left idle: C100 v2b already R; ImageNet TRAJ not wired as a 40-ep TRAJ profile — do not start `eval_imagenet_short`.

| Net | mild-once val_best | v2b-ep0015 §83 | l1-once §84 |
|---|---|---|---|
| ShuffleNet-v2×1 | **−1.7 @ 0.906/0.901**, val **−7.48** (no floor_cross) | **−1.9 @ 0.776/0.788** | **−1.9 @ 0.776/0.788** |
| RepVGG-A0 | **−4.3 @ 0.811/0.810**, val **−8.57** (no floor_cross) | **−4.4 @ 0.643/0.642** | **−3.9 @ 0.643/0.642** |
| RepVGG-A1 | **−3.4 @ 0.808/0.809**, val **−8.33** (no floor_cross) | **−4.3 @ 0.641/0.640** | **−3.2 @ 0.641/0.640** |
| ShuffleNet-v2×1.5 | **−2.2 @ 0.907/0.902**, val **−7.99** (no floor_cross) | **−2.4 @ 0.783/0.793** | **−2.2 @ 0.783/0.793** |

**Read.** Catalog COMPLETED PRELIM. Mild parks unlike at ~0.81–0.91 keep; v2b and l1 walk to ~0.64–0.78. Keeps unmatched — not a Gilad equal-keep test. At the *greedy* size, l1 is the kinder RepVGG row, not the actor. Cheap abort still **no**. Do not lock. Do not overwrite §4 / §83 / §84.

---

## 87. C100 TRAJ of v2b-ep0015 (**21337730**) — PRELIM, catalog COMPLETED, no in-band TEST

Skip-train `eval_c100` of **21237254** `snapshots/ep0015` on `input_offline_c100.json`. `[policy_config]` pinned 5-action B menu. Quote `[eval] TRAJ val_best` only. Job **COMPLETED** 4 h 19 m (ended 15 Sep 21:45 cluster, exit 0). Do **not** quote wrap **−0.10 pp**, terminals, or floor_hold (all val over τ). Do not overwrite §21 (claim C9) or §7 recoverability. Not a C100 DRL train. Freed GPU left idle: ranked TRAJ queue is C100-last; do not start `eval_imagenet_short`. Do not TEST A ep0155. Do not TEST C.

| Net | TRAJ val_best |
|---|---|
| r20-w16 C100 | **identity** (step −1, x1.000 / x1.000, val +0.00). Cuts val over τ. |
| r56-w15 C100 | **identity** (step −1, x1.000 / x1.000, val +0.00). Cuts val over τ. |
| VGG-16 BN C100 | **identity** (step −1, x1.000 / x1.000, val +0.00). Cuts val over τ. |
| ShuffleNet-v2×1 C100 | **identity** (step −1, x1.000 / x1.000, val +0.00). Cuts val over τ. |
| RepVGG-A0 C100 | **identity** (step −1, x1.000 / x1.000, val +0.00). Cuts val over τ. |

**Read.** Catalog COMPLETED PRELIM. Frozen C10 v2b-ep0015 found **no in-band cut** on any of the five C100 nets (τ = 10 pp on val). Identity is the selected TRAJ point, not a compression TEST. Matches the standing C100-miss pattern (residuals unrecovered; this actor also refused VGG/ShuffleNet/RepVGG inside τ). Do **not** call this a C9 overwrite. G1 still OPEN. Cheap abort still **no**. Do not lock.

---

## 88. Plain-walk l1 TRAJ (**21325687**) — PRELIM, catalog COMPLETED

Skip-train `baseline_c10_l1_traj` on `input_c10_thin.json` (no `GROUP_ONCE`). Quote `[eval] TRAJ val_best`. Job **COMPLETED** 8 h 46 m (ended 16 Sep 00:35 cluster, exit 0). Twin of H0 mild-plain **§85**. Do not quote wrap **−0.23 pp**. Do not quote r56 floor_hold / terminal (val **−25 / −38** over τ). Skip r32. Freed GPU left idle: ranked TRAJ queue is C100-last; do not start ImageNet. Do not TEST A ep0155.

| Net | l1-plain val_best | H0 mild-plain §85 | mild-once §77 |
|---|---|---|---|
| r20-w2 | **−6.5 @ 0.433/0.645**, val **−6.88** | **−5.3 @ 0.552/0.692**, val **−5.97** | **+0.4 @ 0.746/0.806** (group-once) |
| r56-w4 | **−5.5 @ 0.984/0.897**, val **−8.59** | **−7.5 @ 0.964/0.776**, val **−9.67** | **−7.1 @ 0.923/0.769** (almost-no-cut) |

**Read.** Catalog COMPLETED PRELIM. Without group-once, l1 walks r20 to **0.433** (deeper than mild-plain 0.552; both far past the group-once 0.746 plateau). Hard-net val_best still refuses to compress (**0.984**), same family as H0 **0.964** and mild-once **0.923**. Do not pick r56 floor_hold **−22.2 @ 0.789** (val over τ). Not a DRL row. **Do not lock.** Do not overwrite §72 / §77 / §85.

---

## 89. Unlike-family TRAJ of v2a-ep0155 (**21363533**) — PRELIM, catalog COMPLETED

Skip-train `eval_offline_novel` of **21237253** `snapshots/ep0155`. `[policy_config]` pinned 3-rate L1 A menu (`STATE_ALIGN=next`, `GROUP_ONCE=1`). Quote `[eval] TRAJ val_best`. Job **COMPLETED** 3 h 36 m (ended 16 Sep 05:08 cluster, exit 0). Caption: **cloned mild keep** (~0.81–0.91), not the v2b/l1 greedy size. Do not quote wrap **−0.03 pp**. ShuffleNet×1.5 PASS1 also printed `effective-params x0.872` — **do not quote**. Skip r32. Freed GPU → A-ep0155 C100 TRAJ fill.

| Net | A-ep0155 val_best | mild-unlike §86 | v2b-ep0015 §83 |
|---|---|---|---|
| ShuffleNet-v2×1 | **−1.8 @ 0.906/0.901**, val **−7.15** (no floor_cross) | **−1.7 @ 0.906/0.901**, val **−7.48** | **−1.9 @ 0.776/0.788** |
| RepVGG-A0 | **−3.3 @ 0.811/0.810**, val **−7.36** (no floor_cross) | **−4.3 @ 0.811/0.810**, val **−8.57** | **−4.4 @ 0.643/0.642** |
| RepVGG-A1 | **−2.9 @ 0.808/0.809**, val **−7.23** (no floor_cross) | **−3.4 @ 0.808/0.809**, val **−8.33** | **−4.3 @ 0.641/0.640** |
| ShuffleNet-v2×1.5 | **−2.1 @ 0.907/0.902**, val **−7.55** (no floor_cross) | **−2.2 @ 0.907/0.902**, val **−7.99** | **−2.4 @ 0.783/0.793** |

**Read.** Catalog COMPLETED PRELIM. A-ep0155 parked unlike at the **mild** keep on all four nets. Vs mild at equal keep: ShuffleNet is a match; RepVGG-A0 is **1.0 pp kinder**, A1 **0.5 pp kinder** (shy of Gilad's 2 pp bar). Vs v2b: unmatched keep — not a ranking-transfer comparison. Cheap abort still **no**. Do not lock. Do not overwrite §4 / §83 / §86.

---

## 90. C10-thin TRAJ of v2a-ep0155 (**21363176**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21237253** `snapshots/ep0155` (train best **0.312**). `[policy_config]` pinned 3-rate L1 A menu. Quote `[eval] TRAJ val_best`. Job **COMPLETED** 6 h 17 m (ended 16 Sep 07:20 cluster, exit 0). Caption: **cloned mild keep**. Do not quote wrap **−0.04 pp**. Do not pick r56 floor_hold / terminal (val **−12.61** over τ). Skip r32. Freed GPU left idle: ranked TRAJ queue is C100-last and C100 A is already R; do not start ImageNet. Do not TEST C.

| Net | A-ep0155 val_best | mild-once §77 | v2b-ep0015 §80 |
|---|---|---|---|
| r20-w2 | **+0.7 @ 0.746/0.806**, val **+0.24** (no floor_cross) | **+0.4 @ 0.746/0.806** | **−1.2 @ 0.606** |
| r56-w4 | **−7.0 @ 0.930/0.772**, val **−9.40** (no floor_cross) | **−7.1 @ 0.923/0.769** | **−7.1 @ 0.879** |

**Read.** Catalog COMPLETED PRELIM. Raising A's `batch_score` to **0.312** did not leave the mild walk: r20 is the same **0.746** plateau as ep0003. Hard-net val_best is still almost no cut (**0.930**, same family as mild **0.923**). Not a keep-learning win vs TESTed early A snaps. Cheap abort still **no**. Do not lock. Do not overwrite §77–§79.






## 91. v3 last train fired (16 Sep 09:55 IDT) — four arms, cold, 24-net catalog

Fable 5.1 MAX sitting (thesis-mission overview chat). Design + evidence: `docs/AUDIT_13SEP_OVERHAUL.md` **Part III (§10–§15)**; ops notes: `docs/PROMPT_FABLE_V3.md` **§11**. Overlay on the leap tree 09:53 (backup `/home/paretsky/scratch_audit/leap_backup_20260916T0953/`); **223 unit tests pass** (CPU job 21385027). Per Ido §10.5: **21378931** (A C100), **21363532** (A similar), **21260250** (l1 similar DenseNet) cancelled 09:54; **21252195** (v2b similar DenseNet) and **21252199** (mild similar DenseNet) kept.

**Diagnosis that drove v3 (from the train traces, not TESTs):** B was over budget on **81 / 3 208** training steps (2.5 %), A on **78 / 7 114** (1.1 %) — the policy never saw the τ-band edge, so "identity when slack ≈ 0" had no signal; at TEST the skinny nets spend τ within 8–12 % of cut and the actor keeps cutting (B r56 terminal 0.639 val over τ). Plus: no per-layer group cost in the state (H2), and patience on the 4-net `batch_score` max (H3: B died at 116, A at 256).

| Job | Profile | Menu `1.0 | 0.9/0.8 × {l1, RANK}` | Reward | GPU |
|---|---|---|---|---|
| **21385158** | `offline_train_v3_fpgm` | fpgm | `structural`+`cbrt` | `ise-4090-07` |
| **21385159** | `offline_train_v3_svd` | svd | `structural`+`cbrt` | `ise-4090-06` |
| **21385160** | `offline_train_v3_bnscale` | bn_scale | `structural`+`cbrt` | `ise-4090-06` |
| **21385161** | `offline_train_v3_fpgm_neonraw` | fpgm (projected winner) | `neon`+`raw` (original NEON, no cube-root) | `ise-4090-09` |

Common: PPO 4×4, agent lr 3e-4, entropy 0.01 → 0.005 (300 ep), zero-init head, dropout 0, `STATE_ALIGN=next`, slack + budget + **group-cost** state (`SPECTRA_STATE_GROUPCOST=1`, +4 token cols), group-once, **`--passes 2`** (band-edge exposure; TEST replays 2 via `policy_config`), rollout 128, train FT 12/4 (TEST 40), **deterministic probe** every 12 episodes on `resnet56-width6` + `resnet20-width10` as the selection/patience score, snapshot baseline 0.05, min lifetime 250, patience 150, **rewind** to the elite after 50 stale probe episodes (max 3, entropy 0.02 for 30 ep, Adam reset), cold, **24-net `database_offline_wide.json`** (superset of the 10; zero overlap with any hold-out), 6-day fuse, no in-job eval. All flags default-off; v2 profiles byte-identical.

**Projection recorded before submit:** FPGM > BN-scale > SVD. Neon-raw is expected to be scale-fragile like v2c (in-band 10/20 invisible next to ±1 000/±8 000 after return scaling); if `ret_scale` ≫ 500 / `pmax` → 1 / `ev` ≤ 0 by update 5, kill it and submit the fallback `offline_train_v3_fpgm_structraw` (NEON trichotomy on the realised cut, no cube-root). Quote v3 actors only from `[eval] TRAJ val_best` of `eval_c10_thin_traj` on a frozen snapshot (`[policy_config]` must show 5 actions, group-cost, `passes: 1 -> 2`); fair controls are the **2-pass** group-once heuristics (`SPECTRA_EVAL_PASSES=2`). Skip-train recommendation: **no**. Do not TEST C. Do not edit the draft.

**12:04 IDT — V4-1 enqueued (audit §16).** `offline_train_v4_factored` **21394377** (nice 0): the v3 recipe with a **factored rate × ranking policy** (`SPECTRA_FACTORED_HEAD=1`, `--ranking_menu l1 fpgm bn_scale svd taylor`; ranking head inactive on identity; Taylor = first-order |w·∇w| on one train batch bound before each cut). `offline_train_v4_factored_tau6` **21394378** (nice 10, later raised to 50 so 2-pass heuristics outrank it): same + `SPECTRA_TRAIN_TAU=6`. Overlay 12:04 (backup `/home/paretsky/scratch_audit/leap_backup_20260916T1204_v4/`); **234 tests pass** (CPU job 21392550). Flag-gated: v2/v3 actors and profiles unchanged. Ops: `docs/PROMPT_OPS_V3_V4_HANDOFF.md`.

---

## 92. Similar-family TRAJ of v2b-ep0015 (**21252195**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21237254** `snapshots/ep0015` on `input_offline_similar.json`. Quote `[eval] TRAJ val_best`. Job **COMPLETED** 2 d 6 h 8 m (ended 16 Sep 17:24 cluster, exit 0). `[policy_config]` pinned 5 actions `1.0 0.9 0.8 0.9 0.8` / `none l1 l1 fpgm fpgm`. Quote TRAJ. Skip r32. Do not quote wrap. Do not quote DenseNet wrap **+0.08**. Val inside τ on every finished cell.

DenseNet-100 last cell (this wake):

| Net | v2b-ep0015 val_best | mild-once `21252199` | l1-once `21260250` |
|---|---|---|---|
| DenseNet-100 | **−2.5 @ 0.662/0.673**, val **−6.38** | still R (step ~52) | CANCELLED 09:53 (incomplete) |

Rest of the catalog (Fable §8.2 PRELIM, skip r32): r20-w16 **−5.3 @ 0.644/0.655**; r56-w10 **−6.1 @ 0.642/0.642** (val −9.76, tight τ); r44 **−4.0 @ 0.643/0.654**; VGG-19 BN **−3.2 @ 0.642/0.657**; MobileNet-v2×0.75 **−2.0 @ 0.650/0.657**.

**Read.** Actor similar catalog COMPLETED PRELIM. DenseNet is **not** the connectivity blow-up: −2.5 at 0.662, val −6.38, kinder Δacc than the ResNet/VGG similar cells at similar keep. Hard cell remains r56-w10 (tight τ, 0.7 pp worse than mild at unmatched keep). Cheap abort still **no** until mild DenseNet lands. G1 still OPEN (`21252199` R). Do not lock. Do not overwrite §3.

**17:56 IDT — 2-pass heuristic controls submitted** (handoff §4.1; V4-1 already R). `SPECTRA_EVAL_PASSES=2 SPECTRA_NICE=5`: **21413236** `mild-once-p2` R `cs-1080-05` (`[eval] … passes=2`); **21413237** `l1-once-p2` PD `QOSMaxGRESPerUser`. tau6 **21394378** nice raised 10→50 so it stays behind l1-p2.

---

## 93. 2-pass group-once mild thin (**21413236**) — PRELIM, catalog COMPLETED

Fair control for v3/V4 (`SPECTRA_EVAL_PASSES=2`, group-once, `det=1` TRAJ). Job **COMPLETED** 13 h 28 m (ended 17 Sep 07:24 cluster, exit 0). Quote `[eval] TRAJ val_best`. Do not quote wrap. Do not pick r56 terminals over τ.

| Arm / point | TRAJ (shape params) | vs 1-pass mild-once §77 |
|---|---|---|
| r20 val_best = terminal step=40 | **−3.4 @ 0.536/0.655** (0.648→0.614), val **−4.27 pp** | 1-pass **+0.4 @ 0.746**. Extra cut from the second pass, as designed. |
| r56 val_best step=38 | **−6.6 @ 0.923/0.769** (0.888→0.822), val **−9.32 pp** | 1-pass **−7.1 @ 0.923**. **Same keep.** 0.5 pp kinder. |

**Read.** Catalog COMPLETED. Two passes buy a smaller easy net (0.536 vs 0.746) and do **not** move the hard-net val_best keep (still 0.923). That is the H1 plateau with an extra pass on r20 only. Fair v3/V4 TESTs quote against these keeps, not against 1-pass mild. **Do not lock.** Do not edit the draft.

---

## 94. 2-pass group-once L1 thin (**21413237**) — PRELIM, catalog COMPLETED

Fair control for v3/V4 (`SPECTRA_EVAL_PASSES=2`, group-once, `det=1` TRAJ). Job **COMPLETED** 3 h 3 m (ended 17 Sep 10:27 cluster, exit 0). Quote `[eval] TRAJ val_best`. Do not quote wrap. Val-best r56 is inside τ by 0.08 pp — do not treat as comfortable.

| Arm / point | TRAJ (shape params) | vs 2-pass mild §93 / 1-pass l1 §81 |
|---|---|---|
| r20 val_best = terminal step=40 | **−7.3 @ 0.417/0.608** (0.648→0.575), val **−7.93 pp** | 2-pass mild **−3.4 @ 0.536**. Extra cut, 3.9 pp worse TEST. 1-pass l1 **−3.1 @ 0.606**. |
| r56 val_best step=32 | **−7.8 @ 0.898/0.720** (0.888→0.810), val **−9.92 pp** | 2-pass mild **−6.6 @ 0.923**. Slightly smaller, 1.2 pp worse TEST. 1-pass l1 **−5.9 @ 0.973**. |

**Read.** Catalog COMPLETED PRELIM. Two-pass L1 walks the easy net to **0.417** (deeper than 2-pass mild 0.536) and the hard net to **0.898** (mild stayed 0.923). Both val_bests are inside τ. This is the equal-keep yardstick for v3/V4 TESTs on thin, together with mild §93. **Do not lock.** Do not edit the draft.

---

## 95. First v3 TRAJ — fpgm ep0011 thin (**21428727**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21385158** `snapshots/ep0011` (probe 0.262). Job **COMPLETED** 3 h 5 m (ended 17 Sep 13:45 cluster, exit 0). `[policy_config]` pinned 5-action fpgm, `passes: 2`, group-cost, `align=next`, `det=1`. Quote `[eval] TRAJ val_best`. Do not quote wrap. Fair yardstick = 2-pass mild §93 and 2-pass L1 §94.

| Net | v3-fpgm ep0011 | 2-pass mild §93 | 2-pass L1 §94 |
|---|---|---|---|
| r20-w2 | **−5.1 @ 0.536/0.655** (0.648→0.597), val **−6.17** | **−3.4 @ 0.536/0.655** | −7.3 @ 0.417/0.608 |
| r56-w4 | **−6.8 @ 0.923/0.769** (0.888→0.820), val **−9.13** | **−6.6 @ 0.923/0.769** | −7.8 @ 0.898/0.720 |

**Read.** Catalog COMPLETED PRELIM. **Equal keep vs 2-pass mild on both nets.** r20 is **1.7 pp worse**; r56 is **0.2 pp worse** (not a 2 pp win). This snap cloned the mild keep. Not a Gilad family win. Cheap abort still **no**. Next TRAJ: svd ep0011 **21433272 R** 18:36 (FLAGS `passes=2`, groupcost, svd menu). V4 froze **ep0083 / 0.262** (probe 84; beat ep0071 0.241). Do **not** lock. Do not edit the draft until Ido says.

---

## 96. mild-once similar catalog — DenseNet lands, G1 closes (**21252199**) — PRELIM

Skip-train `traj-mild-gonce-similar` COMPLETED 17 Sep 18:35 cluster (3 d 7 h 19 m, exit 0). Group-once mild, **1-pass** (keep ~0.81–0.82). Quote `[eval] TRAJ val_best`. Skip r32. Other six nets were already in Fable §8.2; DenseNet was the walker.

| Net | mild-once `21252199` val_best | v2b-ep0015 `21252195` §92 |
|---|---|---|
| r20-w16 | **−4.7 @ 0.819/0.802**, val −8.25 | −5.3 @ 0.644/0.655 |
| r56-w10 | **−5.4 @ 0.811/0.811**, val −8.64 | −6.1 @ 0.642/0.642 (tight τ) |
| r44 | **−3.0 @ 0.819/0.803**, val −8.23 | −4.0 @ 0.643/0.654 |
| VGG-19 BN | **−3.2 @ 0.811/0.819**, val −8.38 | −3.2 @ 0.642/0.657 |
| MobileNet-v2×0.75 | **−2.6 @ 0.816/0.825**, val −7.23 | −2.0 @ 0.650/0.657 |
| DenseNet-100 | **−1.7 @ 0.822/0.828** (0.949→0.932), val **−5.61** | **−2.5 @ 0.662/0.673**, val −6.38 |

**Read.** Catalog COMPLETED PRELIM (skip r32). DenseNet is **not** a connectivity blow-up on mild either: −1.7 at 0.822, val −5.61. v2b cuts more (0.662) at 0.8 pp worse TEST — not a 2 pp Gilad win at equal keep. Hard cell remains r56-w10. **G1 CLOSED** for the v2b+mild pair (l1 similar DenseNet was sacrificed 16 Sep 09:53). Cheap abort still **no**. Do not lock. Do not edit the draft.

---

## 97. v3-svd ep0011 thin TRAJ (**21433272**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21385159** `snapshots/ep0011` (probe 0.262). Job **COMPLETED** 5 h 15 m (ended 17 Sep 23:51 cluster, exit 0). FLAGS at start: `passes=2`, groupcost, svd menu, `align=next`, `det=1`. Quote `[eval] TRAJ val_best`. Do not quote wrap. Fair yardstick = 2-pass mild §93, 2-pass L1 §94, v3-fpgm §95.

| Net | v3-svd ep0011 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−5.0 @ 0.536/0.655** (0.648→0.598), val **−5.44** | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.7 @ 0.923/0.769** (0.888→0.821), val **−9.44** | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **Equal keep vs 2-pass mild and vs fpgm on both nets.** r20 is **1.6 pp worse** than mild; r56 is **0.1 pp worse**. svd ≡ fpgm walk (0.1 pp kinder both nets — noise). Cloned mild keep again. Not a Gilad family win. Cheap abort still **no**. Next TRAJ: neonraw ep0023 **21442936 R** 00:11 (FLAGS `passes=2`, groupcost, fpgm menu, snap ep0023). Do **not** lock. Do not edit the draft until Ido says.

---

## 98. P2 gate — reward-band census of the live v3 / V4 / v2b traces (18 Sep 02:00, Fable V5 sitting) — not a TEST

`reward_trace.jsonl` of every train with `SPECTRA_REWARD_TRACE=1`, non-identity steps only (rate < 1). Read on leap at `c08d513`; v3/V4 still R (rows are a census, not final).

| Job | Arm | non-identity steps | gain (Δacc > 0) | in-budget | over-budget |
|---|---|---|---|---|---|
| 21237254 | v2b (10-net, 1 pass, 12/4) | 1956 | **0 (0.0 %)** | 1917 (98.0 %) | 39 (2.0 %) |
| 21385158 | v3-fpgm (24-net, 2 pass) | 2968 | **0** | 2435 (82.0 %) | 533 (18.0 %) |
| 21385159 | v3-svd | 2973 | **0** | 2394 (80.5 %) | 579 (19.5 %) |
| 21385160 | v3-bnscale | 2937 | **0** | 2299 (78.3 %) | 638 (21.7 %) |
| 21385161 | v3-neonraw | 2940 | **0** | 2581 (87.8 %) | 359 (12.2 %) |
| 21394377 | V4-factored | 3814 | **0** | 3063 (80.3 %) | 751 (19.7 %) |
| 21394378 | V4-tau6 | 1065 | **0** | 916 (86.0 %) | 149 (14.0 %) |

**Addendum 21 Sep 17:50 (Fable V6 sitting).** Same census on the two trains that did not exist on 18 Sep: **ft40 `21443408`** (train FT **40/10**) — 2649 non-identity steps, gain **0**, in-budget 83.7 %, over-budget 16.3 %, max Δacc **−0.80 pp**; **in-band-linear `21459737`** — 3477 non-identity steps, gain **0**, in-budget 80.8 %, over-budget 19.2 %, max Δacc **−0.99 pp**. The empty gain arm is **not** the 12/4 cost cut (40/10 never overshoots the origin either) and not the cube-root. P2 stays a no-op on this catalog.

**Read.** (1) The **gain arm never fires** — 0 of ~17 700 non-identity steps across seven trains. After prune + 12-epoch Adam the post-FT val accuracy never exceeded the origin on these 90–96 % C10 nets. **P2 (`×2` on Δacc > 0) is a no-op on this loop; no GPU** (PROMPT_FABLE_V5 P2 gate: empty arm → skip). It only becomes non-vacuous on nets with room above origin (P5-B3's ~70 % C100 candidates) — re-run this census on that catalog before revisiting P2. (2) The trichotomy the critic actually sees is a **dichotomy**: `+ρ^{1/3}` in-band vs `−ρ` over-budget under `cbrt`. (3) The second pass moved **12–22 %** of non-identity steps over budget (v2b 1-pass: 2 %): band-edge exposure arrived, mostly as `−ρ` penalties. Not a TEST row; do not quote in the draft.

---

## 99. v3-neonraw ep0023 thin TRAJ (**21442936**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21385161** `snapshots/ep0023` (probe 0.262, raw cubes). Job **COMPLETED** 5 h 21 m (ended 18 Sep 05:33 cluster, exit 0; landed in the VPN gap). `[policy_config]` pinned 5-action fpgm, `passes: 2`, group-cost, `align=next`, `det=1`. Quote `[eval] TRAJ val_best`. Do not quote wrap. Fair yardstick = 2-pass mild §93, fpgm §95, svd §97.

| Net | v3-neonraw ep0023 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−4.1 @ 0.536/0.655** (0.648→0.607), val **−4.83** | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−7.4 @ 0.757/0.698** (0.888→0.814), val **−9.66** | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. r20 **equal keep vs 2-pass mild** (0.7 pp worse than mild, 1.0 pp kinder than fpgm). r56 did **not** clone mild keep: 0.757 vs mild/fpgm **0.923**, in-band, **0.8 pp worse** than mild at a deeper cut. Not a Gilad 2 pp win. Raw cubes are not a silent crash, and they are not a ranking-menu sentence either. Cheap abort still **no**. Do **not** lock. Do not edit the draft until Ido says.

---

## 100. P8 thin mild C-G (**21443376**) — PRELIM, catalog COMPLETED, no-agent

No-agent mild 2-pass group-once walk, recipe **C-G** (NEON layer replacement, group trained to a val plateau). Scratch tree `/home/paretsky/scratch_audit/tree`. Job **COMPLETED** 2 h 41 m (ended 18 Sep 08:14 cluster, exit 0; started when neonraw freed `ise-pheno-05`). FLAGS: `ft_recipe=C-G refresh_all=1`, policy=mild, `passes=2`, group-once, `align=prev`. 76 layer-replacement steps, 0 Tracebacks. Quote `[eval] TRAJ val_best`. Do not quote wrap / terminal. Control = 2-pass mild recipe A §93.

| Net | P8 C-G `21443376` | 2-pass mild A §93 |
|---|---|---|
| r20-w2 | **−0.9 @ 0.988/0.980** (0.648→0.639), val **−0.89** | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−0.1 @ 0.999/0.991** (0.888→0.887), val **+0.00** | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. Caption: **not DRL**. C-G `val_best` is an **empty band** (kept ≥ 0.98) on both nets — the walk did not recover a 2–5 % in-band cut, so the selected point is near identity. Decision table (`docs/V5_P8_RECOVERY_PROBE.md` §A): this is the **C-G ≪ A** row. C-G+ catalog is **§101**. Group-budget from 76 `[C-G group] Early stopping` lines: median **26.5** epochs (min 7, max 60, mean 27.9) — proposed `SPECTRA_FT_REINIT_EPOCHS≈28` if a P8 DRL cell is ever opened. Do **not** lock. Do not edit the draft.

---

## 101. P8 thin mild C-G+ (**21443377**) — PRELIM, catalog COMPLETED, no-agent

No-agent mild 2-pass group-once walk, recipe **C-G+** (C-G + 0.1× lr full-net polish). Scratch tree. Job **COMPLETED** 3 h 45 m (ended 18 Sep 11:59 cluster, exit 0). FLAGS: `ft_recipe=C-G+ refresh_all=1`, 76 layer replacements, 0 Tracebacks. Quote `[eval] TRAJ val_best`. Do not quote wrap / terminal. Control = §93; C-G = §100.

| Net | P8 C-G+ `21443377` | P8 C-G §100 | 2-pass mild A §93 |
|---|---|---|---|
| r20-w2 | **−10.3 @ 0.884/0.861** (0.648→0.545), val **−10.00** | −0.9 @ 0.988/0.980 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−0.2 @ 0.999/0.991** (0.888→0.886), val **−0.80** | −0.1 @ 0.999/0.991 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **Not DRL.** Polish helps r20 a little (in-band at τ, keep 0.884 vs A's 0.536) and does **not** open an r56 band (identity keep). Decision table on the thin pair: **C-G+ ≪ A**. Catalog L C-G+ `21443380` is the remaining third net — do not close "No P8 DRL GPU" until it catalogs, but the thin evidence already says NEON layer replacement does not recover the mild walk. Do **not** lock.

---

## 102. V4-factored ep0083 thin TRAJ (**21447387**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21394377** `snapshots/ep0083` (probe 0.262; the only snap that beat its first freeze). Job **COMPLETED** 5 h 18 m (ended 18 Sep 17:18 cluster, exit 0). `[policy_config]` pinned factored head, `passes: 2`, group-cost, `align=next`, `det=1`. Quote `[eval] TRAJ val_best`. Fair yardstick = 2-pass mild §93, fpgm §95.

| Net | V4 ep0083 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−3.7 @ 0.536/0.655** (0.648→0.610), val **−5.13** | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.9 @ 0.923/0.769** (0.888→0.820), val **−9.04** | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **Equal keep vs 2-pass mild on both nets.** r20 is **0.3 pp worse** than mild (kindest actor TEST so far at this keep); r56 is **0.3 pp worse**. Cloned mild keep. The live-loop snap that beat first freeze did **not** falsify "first freeze ≡ mild." Not a Gilad family win. Cheap abort still **no**. Do **not** lock. Do not edit the draft until Ido says.

---

## 103. P8 Catalog L ResNet-56 mild A (**21443378**) — PRELIM, catalog COMPLETED, no-agent

No-agent mild 2-pass group-once walk, recipe **A**, Catalog L chenyaofo ResNet-56 C10 (origin 94.37 %). Scratch tree. Job **COMPLETED** 3 h 12 m (ended 18 Sep 20:31 cluster, exit 0). FLAGS: `ft_recipe=A refresh_all=0`. Quote `[eval] TRAJ val_best`.

| Net | Catalog L mild A |
|---|---|
| chenyaofo r56 | **−3.9 @ 0.661/0.662** (0.943→0.903), val **−8.19** |

**Read.** Catalog COMPLETED PRELIM. **Not DRL.** In-band real cut (kept 0.661). This is the A control for Catalog L C-G §104 and C-G+ §106. Do **not** lock.

---

## 104. P8 Catalog L ResNet-56 mild C-G (**21443379**) — PRELIM, catalog COMPLETED, no-agent

Same walk as §103, recipe **C-G**. Job **COMPLETED** 1 h 42 m (ended 18 Sep 22:13 cluster, exit 0). FLAGS: `ft_recipe=C-G refresh_all=1`, 60 replacements, 0 Tracebacks.

| Net | Catalog L C-G | Catalog L A §103 |
|---|---|---|
| chenyaofo r56 | **−0.5 @ 0.999/0.995** (0.943→0.938), val **−0.03** | **−3.9 @ 0.661/0.662** |

**Read.** Catalog COMPLETED PRELIM. **Not DRL.** Empty band again. C-G ≪ A on the committee-slide net too. C-G+ catalog is **§106**. Do **not** lock.

---

## 105. P5-B3 C100 gate (**21443381**) — FAILED, not a TEST

Scratch-tree job **FAILED** 31 s (ended 18 Sep 17:19 cluster, exit 1). `FileNotFoundError`: `/home/paretsky/scratch_audit/tree/configs/v5_p5b3_c100_candidates_input.json` missing, then `ValueError: Invalid input`. Zero evaluations. Do **not** quote. File placed 19 Sep 01:23; gate resubmitted **21459732** — **COMPLETED** 20 Sep 03:09, **0 admits**. Emit **21459733 FAILED** exit 2. V6 DRL **21459734/35 CANCELLED**. Outcome: **§109**.

---

## 106. P8 Catalog L ResNet-56 mild C-G+ (**21443380**) — PRELIM, catalog COMPLETED, no-agent

Same walk as §103, recipe **C-G+**. Job **COMPLETED** 2 h 51 m (ended 19 Sep 01:04 cluster, exit 0). FLAGS: `ft_recipe=C-G+ refresh_all=1`, 60 replacements, 0 Tracebacks. Quote `[eval] TRAJ val_best`. Do not quote wrap / terminal (terminal −24.4 @ 0.661 is over τ).

| Net | Catalog L C-G+ | Catalog L C-G §104 | Catalog L A §103 |
|---|---|---|---|
| chenyaofo r56 | **−1.0 @ 0.999/0.995** (0.943→0.932), val **−0.23** | −0.5 @ 0.999/0.995 | **−3.9 @ 0.661/0.662** |

**Read.** Catalog COMPLETED PRELIM. **Not DRL.** Empty band. Polish did **not** open a 2–5 % in-band cut on the committee-slide net. Thin pair §101 + this row: **C-G+ ≪ A on all three nets.** Decision table (`docs/V5_P8_RECOVERY_PROBE.md` §A): NEON layer replacement looks dense-DNN-specific; SPECTRA's recoverable CNN recipe remains A. Producers-scope ablation (oral reading) queued separately. Do **not** lock. Do not edit the draft.

---

## 107. v3-bnscale ep0011 thin TRAJ (**21447388**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21385160** `snapshots/ep0011` (first freeze; later snaps ep0095 / ep0107 exist and were **not** TESTed). Job **COMPLETED** 5 h 20 m (ended 19 Sep 06:23 cluster, exit 0). `[policy_config]` pinned 5-action bn_scale, `passes: 2`, group-cost, `align=next`, `det=1`. Quote `[eval] TRAJ val_best`. Fair yardstick = 2-pass mild §93, fpgm §95, V4 §102.

| Net | v3-bnscale ep0011 | V4 ep0083 §102 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−3.6 @ 0.536/0.655** (0.648→0.612), val **−4.31** | −3.7 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.7 @ 0.923/0.769** (0.888→0.822), val **−9.07** | −6.9 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **Equal keep vs 2-pass mild on both nets.** Kindest ranking-menu TEST so far (r20 0.2 pp worse than mild, 0.1 pp kinder than V4). Still cloned mild keep. Later freeze ep0107 / 0.262 is untested — do not auto-queue. Do **not** lock.

---

## 108. P8 thin mild C-G producers-scope (**21459742**) — PRELIM, catalog COMPLETED, no-agent

No-agent mild 2-pass group-once walk, recipe **C-G** with `SPECTRA_FT_REINIT_SCOPE=producers` (fresh producers + group norms only; consumers sliced, not replaced; no polish). Scratch tree. Job **COMPLETED** 2 h 26 m (ended 20 Sep 05:35 cluster, exit 0, `cs-pheno-03`). FLAGS: `ft_recipe=cg`, `FT_REINIT_THEN_POLISH=0`, 60/6 group, policy=mild, `passes=2`, 76 replacements, 0 Tracebacks. Quote `[eval] TRAJ val_best`. Do not quote wrap / `pass 1/1` / terminals. Control = C-G all-group §100; mild A §93.

| Net | P8 C-G producers `21459742` | P8 C-G §100 | 2-pass mild A §93 |
|---|---|---|---|
| r20-w2 | **−0.7 @ 0.988/0.980** (0.648→0.641), val **−0.69** | −0.9 @ 0.988/0.980 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **+0.0 @ 0.999/0.991** (0.888→0.888), val **+0.08** | −0.1 @ 0.999/0.991 | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **Not DRL.** Empty band, same keep as full-group C-G. Restricting replacement to producers does **not** open a 2–5 % in-band cut. Oral “maybe only producers matter” is not a recoverability win on this walk. Decision table: producers ≈ C-G ≪ A. Do **not** start C-G DRL from this row. Do **not** lock. Do not edit the draft.

---

## 109. P5-B3 C100 gate resubmit (**21459732**) — PRELIM, 0 admits, not a mixed-train GO

Mild 2-pass recipe-A walk under the **train** FT (12/4) on the five C100 candidates. Job **COMPLETED** 3 h 41 m (ended 20 Sep 03:09 cluster, exit 0, `cs-pheno-03` rtx_3090). Emit **21459733 FAILED** 05:35 exit **2** (`only 0 CIFAR-100 net(s) admitted (need 2)`). V6 C-G / C-G+ DRL **21459734/35 CANCELLED** by afterok. Quote `[eval] TRAJ val_best`. Skip r32. Do not quote identity wraps as TEST.

| Net | TRAJ val_best | Admit (kept ≤ 0.98 and val Δacc ≥ −10) |
|---|---|---|
| vgg11_bn C100 | identity `x1.000` | reject |
| vgg13_bn C100 | identity `x1.000` | reject |
| mobilenet-v2×1 C100 | identity `x1.000` | reject |
| densenet40 C100 | **−2.6 @ 0.989/0.971**, val **−9.40** | reject (kept 0.989) |
| chenyaofo r32 C100 | skip / identity `x1.000` | reject |

**Read.** **Zero C100 admits.** P5-B3 mixed catalog collapses. Do **not** mix unrecovered C100 into the actor. Morning sitting: P5-B2 fallback (keep one SVHN in train; hold Fashion-MNIST + ImageNet), not a new DRL train today. Do **not** lock. Do not edit the draft.

---

## 110. V4-factored train (**21394377**) COMPLETED — freeze already TESTed §102; not a new TEST

Parent train **COMPLETED** 11:48 IDT 20 Sep (exit 0, 3 d 23 h 44 m, `ise-6000-06`). `min_episodes=250`; last DONE ep249. Last live probe ep240 = **0.262** (matched freeze). Rewind **3/3** already used at ep240. **Four snaps**, last freeze still `snapshots/ep0083` (score 0.262) — thin TRAJ of that snap is **§102** (`21447387`, 18 Sep). No later snap beat 0.262. Do **not** re-TEST ep0083. Do **not** fill the freed GPU with a ranking-menu train, C-G DRL, or a second linear. In-band-linear **21459737** remains the live A/B. Do **not** lock. Do not edit the draft.

---

## 111. In-band-linear ep0083 thin TRAJ (**21512868**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21459737** `snapshots/ep0083` (probe 0.2618; first non-zero after probes 12–72 = 0). Submitted from scratch `tree_v6_inband` (not leap). Job **COMPLETED** 3 h 2 m (ended 21 Sep 08:24 cluster, exit 0, `cs-pheno-08` rtx_3090). `[policy_config]` pinned 2-pass, group-once, 5-action fpgm, `ft_recipe=A`, `align=next`, `det=1`, `cbrt_cubes` train FLAGS. Quote `[eval] TRAJ val_best`. Fair yardstick = 2-pass mild §93, V4 same-ep §102, neonraw §99.

| Net | in-band ep0083 `21512868` | V4 ep0083 §102 | neonraw §99 | 2-pass mild §93 |
|---|---|---|---|---|
| r20-w2 | **−3.5 @ 0.536/0.655** (0.648→0.613), val **−4.50** | −3.7 @ 0.536/0.655 | −4.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−7.1 @ 0.756/0.691** (0.888→0.818), val **−9.65** | −6.9 @ 0.923/0.769 | **−7.4 @ 0.757/0.698** | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. **r20 equal keep vs mild.** r56 **did not** clone 0.923 keep — **0.756**, same depth as neonraw §99, **0.3 pp kinder** than neonraw at that keep, **0.5 pp worse** than mild at a much smaller size. Val −9.65 is inside τ=10 (selected step 58). “First freeze ≡ mild keep” is **falsified on r56**, not on r20. Not a Gilad 2 pp family win. Linear-in-band is not a full mild clone and not a new SOTA point. Do **not** lock. Do not fill the freed GPU with C-G DRL. Fable §7.8 2a/2b is mixed — Ido pastes. Do not edit the draft.

---

## 112. ft40 ep0059 thin TRAJ (**21535193**) — PRELIM, catalog COMPLETED

Skip-train `eval_c10_thin_traj` of **21443408** `snapshots/ep0059` (probe **0.2679**). Submitted from scratch `tree` (ft40’s tree). Job **COMPLETED** 3 h 1 m (val_best logged 21 Sep 20:24, exit 0, `cs-pheno-05`). Pins: 2-pass, group-once, 5-action fpgm, `ft_recipe=A`, train FT **40/10**, eval FT 40, `align=next`, `det=1`, `cbrt`. Quote `[eval] TRAJ val_best`. Fair yardstick = 2-pass mild §93, v3-fpgm 12/4 §95, in-band ep0083 §111.

| Net | ft40 ep0059 `21535193` | in-band §111 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|---|
| r20-w2 | **−4.2 @ 0.536/0.655** (0.648→0.606), val **−5.65**, step 40 | −3.5 @ 0.536/0.655 | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−7.1 @ 0.923/0.769** (0.888→0.817), val **−9.35**, step 38 | **−7.1 @ 0.756/0.691** | **−6.8 @ 0.923/0.769** | **−6.6 @ 0.923/0.769** |

**Read.** Catalog COMPLETED PRELIM. r56 keep is **0.923**, the mild / fpgm keep. Δacc is **0.5 pp worse** than mild and **0.3 pp worse** than fpgm at that keep. The V7 40/10 rule (deeper than 0.923 inside the band, or ≥ 1 pp kinder at equal keep) **did not fire**. Do **not** resubmit `21536396/97/98` with 40/10. r20 keep is 0.536, the same 2-pass keep as every other arm; that row is not the comparison. The walk does continue past the band (floor-hold step 91 is 0.702 at val −15.06); the selected point does not. Probe 0.2679 did not buy a different operating point. Later train probe 72 = 0.2622, below this freeze. Do **not** lock. Do not edit the draft.

---

## 113. v3-svd train (**21385159**) COMPLETED — freeze already TESTed §97; not a new TEST

Parent train **COMPLETED** 21:59 IDT 21 Sep (exit 0, 5 d 12 h 5 m). Last DONE ep249. Last probe ep240 = **0.2406** (below the freeze). One snap, `snapshots/ep0011` (score 0.2622) — thin TRAJ is **§97**. No later snap. Tracebacks 0. Do **not** re-TEST. The freed GPU was taken by V7 3-pass L1 **21536387**. Do **not** submit a replacement. Do not edit the draft.

---

## 114. 3-pass group-once mild thin (**21536384**) — PRELIM, catalog COMPLETED

No-agent `baseline_c10_mild_traj_gonce` with `SPECTRA_EVAL_PASSES=3`, recipe A, Adam lr=0.001, from `tree_v7`. Job **COMPLETED** 4 h 2 m (ended 22 Sep 00:27, exit 0, `cs-pheno-05`). Tracebacks 0. Quote `[eval] TRAJ val_best`. Fair yardstick = 2-pass mild §93 and in-band §111 (the 0.756 r56 row this control exists to match).

| Net | 3-pass mild `21536384` | 2-pass mild §93 | in-band §111 |
|---|---|---|---|
| r20-w2 | **−7.4 @ 0.417/0.608** (0.648→0.574), val **−7.75**, step 61 | −3.4 @ 0.536/0.655 | −3.5 @ 0.536/0.655 |
| r56-w4 | **−6.9 @ 0.923/0.769** (0.888→0.819), val **−9.49**, step 38 | **−6.6 @ 0.923/0.769** | **−7.1 @ 0.756/0.691** |

**Read.** Catalog COMPLETED PRELIM. On r56 the third pass does **not** move the selected keep: still **0.923**, 0.3 pp worse than 2-pass mild. The walk does go past that point (floor-hold step 91 is 0.702 at val −14.27, outside τ). It does **not** reach ~0.75 kept inside the band, so this heuristic does not erase the §111 learned-schedule gap. 3-pass L1 **21536387** is still walking r56; the sentence waits on that row. Do **not** lock. Do not edit the draft.

---

## 115. v3-fpgm train (**21385158**) COMPLETED — freeze already TESTed §95; not a new TEST

Parent train **COMPLETED** 02:45 IDT 22 Sep (exit 0, 5 d 16 h 51 m). Last DONE ep249. Last probe ep240 = **0.2104** (below the freeze). One snap, `snapshots/ep0011` (score 0.2622) — thin TRAJ is **§95**. No later snap. Tracebacks 0. Do **not** re-TEST. The freed GPU was taken by the SGD re-gate **21536389** (`optim=sgd lr=0.01` confirmed). Do **not** submit a replacement. Do not edit the draft.

---

## 116. v3-bnscale train (**21385160**) COMPLETED — freeze already at the 0.262 ceiling; not a new TEST

Parent train **COMPLETED** 05:05 IDT 22 Sep (exit 0, 5 d 19 h 12 m). Last DONE ep259. Last probe ep252 = **0.2622**. Last snap `snapshots/ep0107` (18 Sep, score 0.2622); earlier snap ep0095 was 0.2406. No snap beat the ceiling that the other v3 arms already TESTed as a mild clone. Do **not** auto-TEST ep0107. Do not edit the draft.

---

## 117. CIFAR-100 re-gate, Adam 1e-4 (**21536388**) — PRELIM, catalog COMPLETED

No-agent 2-pass mild, recipe A, train budget 12/4, `optim=adam lr=0.0001` confirmed, from `tree_v7`. Job **COMPLETED** 5 h 26 m (ended 22 Sep 05:53, exit 0). Tracebacks 0. Admit = `val_best` kept ≤ 0.98 and val Δacc ≥ −10. Quote `[eval] TRAJ val_best`.

| Net | val_best | kept | val Δacc | admit |
|---|---|---|---|---|
| r20-w13 | −2.9 @ 0.986/0.936 | 0.986 | −7.96 | no |
| r56-w9 | −1.6 @ 0.999/0.996 | 0.999 | −2.97 | no |
| r32 | −3.3 @ 0.984/0.912 | 0.984 | −9.60 | no |
| VGG-11 | **−4.8 @ 0.722/0.694** | 0.722 | −9.83 | **yes** |
| VGG-13 | **−4.9 @ 0.805/0.768** | 0.805 | −8.38 | **yes** |
| MobileNet-v2×0.5 | −2.5 @ 0.984/0.906 | 0.984 | −9.72 | no |
| MobileNet-v2×1 | **−2.4 @ 0.801/0.750** | 0.801 | −9.85 | **yes** |
| DenseNet-40 | **−4.1 @ 0.944/0.875** | 0.944 | −9.85 | **yes** |

**Read.** **4/8 admits.** The three ResNets and MobileNet-v2×0.5 stay above 0.98 kept. VGG, MobileNet-v2×1, and DenseNet-40 cut inside the band. The count meets the V7 threshold. The arm still has to pass the CIFAR-10 thin control (§118) before any catalog switch. Do **not** lock. Do not edit the draft.

---

## 118. Thin control, Adam 1e-4, 12/4 (**21536390**) — PRELIM, catalog COMPLETED

No-agent 2-pass mild on the thin pair, `optim=adam lr=0.0001` confirmed. Job **COMPLETED** 58 min (ended 22 Sep 06:04, exit 0). Tracebacks 0. Pass = within ~0.5 pp of the Adam 1e-3 reference §120 at equal keep. Fail = > 1 pp worse, or a shallower keep.

| Net | Adam 1e-4 `21536390` | Adam 1e-3 ref §120 |
|---|---|---|
| r20-w2 | **−8.3 @ 0.655/0.702**, val **−9.11** | −5.3 @ 0.536/0.655, val −6.23 |
| r56-w4 | **−8.0 @ 0.930/0.772**, val **−9.79** | −6.5 @ 0.933/0.776, val −9.03 |

**Read.** **Fail.** r20 is 3.0 pp worse and shallower (0.655 vs 0.536). r56 is 1.5 pp worse at the same keep. Adam 1e-4 does not get to be the V7 training recipe. Combined with §117, the 4/8 CIFAR-100 admits do **not** unlock `database_offline_v7_diverse_admitted.json`. Leave `21536396/97/98` on the p5b2 catalog at 12/4. Do **not** lock. Do not edit the draft.

---

## 119. Thin control, SGD 0.01, 12/4 (**21536391**) — PRELIM, catalog COMPLETED

No-agent 2-pass mild, `optim=sgd lr=0.01` confirmed. Job **COMPLETED** 57 min (ended 22 Sep 06:50, exit 0). Tracebacks 0. Same pass rule as §118, against §120.

| Net | SGD 0.01 `21536391` | Adam 1e-3 ref §120 |
|---|---|---|
| r20-w2 | **−8.1 @ 0.560/0.664**, val **−8.62** | −5.3 @ 0.536/0.655 |
| r56-w4 | **−7.8 @ 0.930/0.772**, val **−9.96** | −6.5 @ 0.933/0.776 |

**Read.** **Fail.** r20 is 2.8 pp worse. r56 is 1.3 pp worse at the same keep. SGD 0.01 is not the V7 training recipe either. Do **not** lock. Do not edit the draft.

---

## 120. Thin reference, Adam 1e-3, 12/4 (**21536392**) — PRELIM, catalog COMPLETED

No-agent 2-pass mild, `optim=adam lr=0.001` confirmed. This is today’s train recipe on the thin pair at 12/4. Job **COMPLETED** 58 min (ended 22 Sep 07:01, exit 0). Tracebacks 0.

| Net | Adam 1e-3 12/4 `21536392` |
|---|---|
| r20-w2 | **−5.3 @ 0.536/0.655** (0.648→0.595), val **−6.23**, step 40 |
| r56-w4 | **−6.5 @ 0.933/0.776** (0.888→0.823), val **−9.03**, step 34 |

**Read.** The reference keep on r56 is 0.933, the same neighborhood as 2-pass mild §93 (0.923) under the shorter 12/4 budget. r20 keep is 0.536. This is the yardstick §118 and §119 failed. Do **not** lock. Do not edit the draft.

---

## 121. CIFAR-100 re-gate, SGD 0.01 (**21536389**) — PRELIM, catalog COMPLETED

No-agent 2-pass mild, 12/4, `optim=sgd lr=0.01` confirmed. Job **COMPLETED** 5 h 24 m (ended 22 Sep 08:08, exit 0). Same admit rule as §117.

| Net | val_best | kept | val Δacc | admit |
|---|---|---|---|---|
| r20-w13 | −4.1 @ 0.992/0.954 | 0.992 | −9.66 | no |
| r56-w9 | −2.8 @ 0.999/0.996 | 0.999 | −5.64 | no |
| r32 | −2.7 @ 0.990/0.939 | 0.990 | −8.76 | no |
| VGG-11 | **−0.9 @ 0.659/0.680** | 0.659 | −7.64 | **yes** |
| VGG-13 | **−1.7 @ 0.658/0.686** | 0.658 | −5.55 | **yes** |
| MobileNet-v2×0.5 | −1.4 @ 1.000/0.989 | 1.000 | −9.34 | no |
| MobileNet-v2×1 | −1.8 @ 0.988/0.932 | 0.988 | −8.97 | no |
| DenseNet-40 | −3.8 @ 0.981/0.952 | 0.981 | −9.26 | no |

**Read.** **2/8 admits**, both VGGs, at a deeper keep than Adam 1e-4. ResNets, both MobileNets, and DenseNet-40 stay above 0.98 kept. Below the 4/8 threshold, and the thin control §119 failed. Do **not** emit the diverse catalog. Do **not** lock. Do not edit the draft.

---

## 122. 3-pass group-once L1 thin (**21536387**) — PRELIM, catalog COMPLETED

No-agent `baseline` L1, `SPECTRA_EVAL_PASSES=3`, recipe A, Adam lr=0.001, group-once, `align=prev`, `det=0`, from `tree_v7`. Job **COMPLETED** 12 h 59 m (ended 22 Sep 10:58, exit 0, `ise-pheno-05`). Tracebacks 0. Quote `[eval] TRAJ val_best`. Fair yardstick = 3-pass mild §114 and in-band §111 (the 0.756 r56 row).

| Net | 3-pass L1 `21536387` | 3-pass mild §114 | in-band §111 |
|---|---|---|---|
| r20-w2 | **−9.3 @ 0.319/0.569** (0.648→0.555), val **−9.18**, step 61 | −7.4 @ 0.417/0.608 | −3.5 @ 0.536/0.655 |
| r56-w4 | **−7.6 @ 0.914/0.748** (0.888→0.812), val **−9.94**, step 24 | **−6.9 @ 0.923/0.769** | **−7.1 @ 0.756/0.691** |

**Read.** Catalog COMPLETED PRELIM. On r56 the selected keep is **0.914**, the same shallow neighborhood as 3-pass mild (0.923). It does **not** reach ~0.75 kept inside the band. Floor-hold step 45 is 0.704 at val −13.56, outside τ, so the walk leaves the band before ~0.80. Both 3-pass heuristics therefore leave the §111 / §123 0.756 in-band point as a schedule they do not select. r20 at 0.319 is the 3-pass walk, not the policy comparison. The learned-schedule sentence stands. Do **not** lock. Do not edit the draft.

---

## 123. In-band ep0095 thin TRAJ + counterfactual (**21536395**) — PRELIM, catalog COMPLETED

Skip-train of **21459737** `snapshots/ep0095` with `SPECTRA_EVAL_COUNTERFACTUAL=1`, from `tree_v7`. Job **COMPLETED** 3 h 1 m (ended 22 Sep 11:10, exit 0, `cs-pheno-07`). Tracebacks 0. Quote `[eval] TRAJ val_best` and the `[cf]` fractions. `[cf]` lines are loguru-prefixed. Fair yardstick = in-band ep0083 §111.

| Net | ep0095 `21536395` | ep0083 §111 | cf |
|---|---|---|---|
| r20-w2 | **−3.8 @ 0.536/0.655** (0.648→0.610), val **−4.48**, step 40 | −3.5 @ 0.536/0.655, val −4.50 | 42 steps, content_used **38.1%**, state_used **38.1%** |
| r56-w4 | **−7.1 @ 0.756/0.691** (0.888→0.817), val **−9.68**, step 58 | **−7.1 @ 0.756/0.691**, val −9.65 | 114 steps, content_used **52.6%**, state_used **52.6%** |

**Read.** Catalog COMPLETED PRELIM. ep0095 reproduces ep0083: same r56 keep and the same −7.1 pp, r20 0.3 pp worse at the same 0.536 keep. `state_used` is 38% on r20 and 53% on r56, both above the 20% line, so the encoder is read on both nets. Do not park the representation cell. Floor-hold on r56 is 0.702 at val −13.75, outside τ; the quoted point is the in-band 0.756. Freed GPUs were taken by area **21536396** and PPO-8 **21536397**, both still `SPECTRA_TRAIN_FT_EPOCHS=12`. Factored **21536398** stays pending. Do **not** lock. Do not edit the draft.

---

## 124. Catalog L twins, 2-pass mild (**21536393**) — PRELIM, catalog COMPLETED

No-agent mild, 2-pass, group-once, recipe A, from `tree_v7`. Nets: chenyaofo ResNet-56 C10, VGG-16 C10, VGG-19 C100. Job **COMPLETED** 4 h 33 m (ended 22 Sep 11:24, exit 0, `cs-pheno-05`). Tracebacks 0. Quote `[eval] TRAJ val_best`. r56 yardstick is §103.

| Net | origin | mild `21536393` | val Δacc |
|---|---|---|---|
| r56 C10 | 0.943 | **−3.3 @ 0.661/0.662** (0.943→0.909), step 112 | −8.18 |
| VGG-16 C10 | 0.936 | **−3.5 @ 0.657/0.678** (0.936→0.902), step 29 | −8.20 |
| VGG-19 C100 | 0.739 | **0.0 @ 1.000/1.000** (0.739→0.739), step −1 | +0.00 |

**Read.** Catalog COMPLETED PRELIM. r56 reproduces §103’s keep (0.661) and is 0.6 pp kinder (−3.3 vs −3.9). VGG-16 lands in the same keep band. VGG-19 C100’s selected point is the **unpruned** net: floor-hold step 30 is 0.705 at val −30.45 and the terminal is 0.657 at val −30.91, both outside τ. The end-of-walk summary’s pass 2/2 line matches that terminal and is not the quoted point. Do **not** lock. Do not edit the draft.

---

## 125. Catalog L twins, 2-pass L1 (**21536394**) — PRELIM, catalog COMPLETED

No-agent L1, 2-pass, group-once, recipe A, from `tree_v7`. Same three nets. Job **COMPLETED** 4 h 16 m (ended 22 Sep 11:18, exit 0, `cs-pheno-08`). Tracebacks 0. Quote `[eval] TRAJ val_best`.

| Net | origin | L1 `21536394` | val Δacc |
|---|---|---|---|
| r56 C10 | 0.943 | **−5.1 @ 0.415/0.413** (0.943→0.891), step 112 | −9.49 |
| VGG-16 C10 | 0.936 | **−3.5 @ 0.411/0.442** (0.936→0.901), step 29 | −9.22 |
| VGG-19 C100 | 0.739 | **0.0 @ 1.000/1.000** (0.739→0.739), step −1 | +0.00 |

**Read.** Catalog COMPLETED PRELIM. On the two CIFAR-10 nets L1 selects a deeper keep than mild (§124): r56 0.415 vs 0.661, VGG-16 0.411 vs 0.657. VGG-16’s test drop stays −3.5 pp at that deeper keep. VGG-19 C100 again selects the unpruned net. Floor-hold step 13 is 0.711 at val −30.57; the terminal is 0.412 at val −33.59. Both are outside τ. Do **not** lock. Do not edit the draft.

---

## 126. A-LSQ recovery, no agent (thin **21703433**, Catalog L r56 twin **21703434**) — PRELIM, catalog COMPLETED

Fable V8 sitting (27 Sep). Recipe **A-LSQ** = keep survivors + closed-form least-squares refit of every consumer of the cut channels (He, Zhang & Sun 2017; ridge 1e-5; 2 × 32 calibration images), BN re-estimated, then the usual whole-net fine-tune (40/10). 2-pass group-once mild, from `tree_v8`. Both jobs COMPLETED (2 h 59 m `cs-pheno-10`; 6 h 20 m `ise-pheno-03`), tracebacks 0. Quote `[eval] TRAJ val_best`.

| Net | A-LSQ | recipe A control | Δ vs A |
|---|---|---|---|
| r20-w2 | **−4.1 @ 0.536/0.655** (0.648→0.607), val −4.53, step 40 | −3.4 @ 0.536/0.655 (§93) | 0.7 pp worse |
| r56-w4 | **−6.2 @ 0.923/0.769** (0.888→0.827), val −9.04, step 38 | −6.6 @ 0.923/0.769 (§93) | 0.4 pp kinder |
| r56 C10 twin (chenyaofo 94.37) | **−4.3 @ 0.661/0.662** (0.943→0.900), val −9.06, step 112 | −3.3 @ 0.661/0.662 (§124) | 1.0 pp worse |

**Read.** Same walk, same keeps as the control on all three nets (the refit does not change which point the 10-pp rule selects). Accuracy at equal size is split: kinder on r56-w4, worse on r20-w2 and on the full-width twin. The V8 pass rule was "≥ A on **both** thin nets" → **A-LSQ does not pass**; it stays a default-off recipe, not a training-loop change. BN-recal alone (§128) explains none of the r56-w4 gain (−6.9 there), so the 0.4 pp is the refit, and it does not generalise. Do **not** lock. Do not edit the draft.

---

## 127. C-PCA replacement, no agent (thin **21703435**, Catalog L r56 twin **21703436**) — PRELIM, catalog COMPLETED

Recipe **C-PCA** = the cut group is *replaced* by a layer of the new width whose filters are the principal directions of the old group's activations; consumers rotated to match; one shared basis per residual stream; group norms reset; depthwise groups skipped; BN re-estimated; same 40/10 fine-tune. 2-pass mild, `tree_v8`. COMPLETED (5 h 15 m `ise-pheno-05`; 2 h 41 m `cs-pheno-06`), tracebacks 0.

| Net | C-PCA | recipe A control |
|---|---|---|
| r20-w2 | **−6.0 @ 0.536/0.655** (0.648→0.588), val −5.44, step 40 | −3.4 @ 0.536 (§93) |
| r56-w4 | **−7.7 @ 0.975/0.845** (0.888→0.811), val −9.84, step 19 | −6.6 @ 0.923 (§93) |
| r56 C10 twin | **−5.2 @ 0.946/0.858** (0.943→0.891), val −9.57, step 38 | −3.3 @ 0.661 (§124) |

**Read.** Worse than A at equal or shallower size on every net: 2.6 pp worse on r20-w2 at the same keep, and on both ResNet-56s the walk leaves the band so early that the selected point is nearly the full net (0.975 / 0.946 kept). A generated layer built from the old activations recovers *less* than the surviving filters do — the third failure of "generate a new layer" on residual CNNs after C-G / C-G+ (§§100–108). Pass rule was "within ~1 pp of A at equal size" → **C-PCA does not pass**. Cross off layer replacement (random or informed) as the CNN recovery. Do **not** lock.

---

## 128. BN re-estimation alone on recipe A (thin **21703437**) — PRELIM, catalog COMPLETED, no-agent

Control for §126: recipe A + `SPECTRA_FT_BN_RECAL=1` only. 2-pass mild, `tree_v8`, COMPLETED 5 h 15 m (`ise-pheno-05`), tracebacks 0.

| Net | A + BN-recal | A (§93) |
|---|---|---|
| r20-w2 | **−4.1 @ 0.536/0.655** (0.648→0.607), val −4.85, step 40 | −3.4 @ 0.536 |
| r56-w4 | **−6.9 @ 0.923/0.769** (0.888→0.819), val −9.65, step 38 | −6.6 @ 0.923 |

**Read.** No gain on either net (0.3–0.7 pp worse, inside resampling noise of the fine-tune). BN re-estimation is a captioned internal detail, not a paper claim (Ido decision 4, 28 Sep).

---

## 129. One-recipe schedule gate — AdamW + warm-up + cosine (thin ctl **21703438**, C100 gate **21703439**) — PRELIM, catalog COMPLETED, no-agent

`SPECTRA_FT_OPTIM=adamw SPECTRA_FT_SCHEDULE=warmcos` (wd 5e-4, 1 warm-up epoch, cosine to 1e-5) inside the same 12/4 fine-tune, 2-pass mild, recipe A. Pass rule: thin within 0.5 pp of Adam 1e-3 (§120) **and** ≥ 4/8 C100 admits. COMPLETED (47 m `cs-pheno-06`; 8 h 52 m `ise-pheno-03`), tracebacks 0.

| Net | warmcos (12/4) | Adam 1e-3 12/4 reference (§120) |
|---|---|---|
| r20-w2 | **−7.1 @ 0.536/0.655** (0.648→0.577), val −7.36 | −5.3 @ 0.536 |
| r56-w4 | **−7.8 @ 0.832/0.730** (0.888→0.810), val −9.79 | −6.5 @ 0.933 |
| C100 candidates (8) | **0 / 8 admitted**: 6 select the unpruned net (step −1), r20-w13 −1.1 @ 0.999, DenseNet-40 −1.7 @ 0.989 | Adam 1e-3: 0/8 (§109) |

**Read.** Fails both halves: the thin control is 1.8 pp worse on r20-w2 at the same keep and 1.3 pp worse on r56-w4 at a deeper-but-out-of-band-selected keep, and it admits nothing on CIFAR-100. The schedule is not the C100 lever. Cross off.

---

## 130. One-recipe schedule gate — RAdam (thin ctl **21703440**, C100 gate **21703441**) — PRELIM, catalog COMPLETED, no-agent

`SPECTRA_FT_OPTIM=radam SPECTRA_FT_WARMUP_EPOCHS=0`, otherwise as §129. COMPLETED (43 m `cs-pheno-06`; 2 h 57 m `cs-pheno-06`), tracebacks 0.

| Net | RAdam (12/4) | Adam 1e-3 12/4 reference (§120) |
|---|---|---|
| r20-w2 | **−5.6 @ 0.655/0.702** (0.648→0.592), val −6.57, step 35 | −5.3 @ 0.536 |
| r56-w4 | **−8.2 @ 0.832/0.730** (0.888→0.806), val −9.85 | −6.5 @ 0.933 |
| C100 candidates | **2 / 8 admitted**: VGG-11 **−5.0 @ 0.814/0.823**, VGG-13 **−6.7 @ 0.843/0.831**; the other six stay ≥ 0.986 kept | 0/8 |

**Read.** Fails the thin half (shallower and 0.3 pp worse on r20-w2, 1.7 pp worse on r56-w4) and admits only the two plain VGGs on CIFAR-100. Same picture as Adam 1e-4 (§117/§118): whatever helps CIFAR-100 hurts the CIFAR-10 control **inside 12 epochs**. Cross off RAdam. Next lever is the *budget*, not the optimizer: Adam, patience 4, cap 40 (cap binds only where 12 epochs were truncating) — thin control + C100 gate at 1e-3 (`21715233`/`21715234`) and at 1e-4 (`21715235`/`21715236`), submitted 28 Sep 01:00.

---

## 131. Catalog L anchor — DepGraph's own ResNet-56 checkpoint, same-loop mild and L1 (**21703466** / **21703467**) — PRELIM, catalog COMPLETED, no-agent

`resnet56_cifar10_dep_graph_93.53.pth` loads strict-clean into `resnet_chenyaofo.resnet56` (CPU probe `21703461`: 93.43 % on 3 000 test images; our TEST loader reads it as 0.932). Recipe A, 2-pass group-once, 40/10, `tree_v8`. COMPLETED (2 h 10 m `cs-pheno-06`; 3 h 11 m `ise-pheno-05`), tracebacks 0.

| Walk | DepGraph anchor | chenyaofo twin |
|---|---|---|
| mild | **−3.1 @ 0.661/0.662** (0.932→0.901), val −8.56, step 112 | −3.3 @ 0.661/0.662 (§124) |
| L1 | **−3.7 @ 0.575/0.482** (0.932→0.895), val −9.28, step 95 | −5.1 @ 0.415/0.413 (§125) |

**Read.** The anchor behaves like the twin under mild (same keep, 0.2 pp kinder) and is kinder under L1 (−3.7 at 0.575 vs −5.1 at 0.415; the walk stops one pass shallower). These are the SPECTRA-loop τ-matched rows for the DepGraph ResNet-56 CIFAR-10 cell; the size-matched row (FLOPs ≈ 0.39, DepGraph 2.57×) is not run yet. DepGraph's published point on this checkpoint is 93.53 → 93.64 (+0.11) at 2.57× after sparsity learning + fine-tune on the target. Do **not** lock. Do not edit the draft.

---

## 132. Cap-40 recipe gate, Adam 1e-3 (thin **21715233** COMPLETED; C100 **21715234** still R) — PRELIM, no-agent

Adam 1e-3, patience 4, **cap 40** (budget, not LR), recipe A, 2-pass mild, `tree_v8`. FLAGS: `optim=adam lr=0.001 epochs=40 patience=4`. Thin **COMPLETED** 2 h 12 m (ended 28 Sep 03:04, `cs-pheno-06`, exit 0). Tracebacks 0. Pair rule: thin within ~0.5 pp of §120 at equal keep **and** ≥ 4/8 C100 admits (kept ≤ 0.98, val Δacc ≥ −10). Quote `[eval] TRAJ val_best`. Headline numbers are TEST Δacc, same as §120.

| Net | cap-40 Adam 1e-3 | Adam 1e-3 12/4 ref §120 |
|---|---|---|
| r20-w2 | **−4.5 @ 0.536/0.655** (0.648→0.603), val **−5.26**, step 40 | −5.3 @ 0.536/0.655, val −6.23 |
| r56-w4 | **−7.2 @ 0.923/0.769** (0.888→0.816), val **−9.64**, step 38 | −6.5 @ 0.933/0.776, val −9.03 |

C100 gate **21715234** still R (~1 h 45 m at 09:45, `cs-1080-05`). First row only: r20-w13 **0.0 @ 1.000/1.000**, step −1, val +0.00 → not admitted.

**Read.** **Thin fail → arm out.** r20 is 0.8 pp kinder at the same keep (pass). r56 is 0.7 pp worse than §120 and a notch deeper (0.923 vs 0.933). The pair rule needs both nets. CIFAR-100 remaining rows cannot save the arm. Do **not** emit `database_offline_v7_diverse_admitted.json`. Do **not** lock. Do not edit the draft.

---

## 133. Cap-40 recipe gate, Adam 1e-4 (thin **21715235** COMPLETED; C100 **21715236** still R) — PRELIM, no-agent

Same walk as §132 at `optim=adam lr=0.0001`. Thin **COMPLETED** 2 h 47 m (ended 28 Sep 05:52, exit 0). Tracebacks 0. Same pair rule.

| Net | cap-40 Adam 1e-4 | Adam 1e-3 12/4 ref §120 |
|---|---|---|
| r20-w2 | **−8.6 @ 0.536/0.655** (0.648→0.562), val **−8.75**, step 40 | −5.3 @ 0.536/0.655, val −6.23 |
| r56-w4 | **−7.8 @ 0.937/0.784** (0.888→0.810), val **−9.97**, step 30 | −6.5 @ 0.933/0.776, val −9.03 |

C100 gate **21715236** still R (~1 h 46 m at 09:45, `ise-pheno-05`). First net r20-w13 **−3.7 @ 0.993/0.962**, val −8.45 (second pass val −9.72 at the same keep) → not admitted (kept > 0.98).

**Read.** **Thin fail → arm out.** r20 is 3.3 pp worse at the same keep as §120; r56 is 1.3 pp worse. Forty epochs does not repair the 12-epoch 1e-4 thin failure (§118). Combined with §132: **neither cap-40 arm passes**. Recipe stays Adam 1e-3 12/4; CIFAR-100 stays test-only unless Ido answers Gilad note Q4 otherwise. Catalog not emitted. Do **not** lock. Do not edit the draft.

---

## 134. Factored-head train (**21536398**) COMPLETED — freeze ep0167 / 0.0608; no TEST until GO

Parent train **COMPLETED** 11:37 IDT 28 Sep (exit 0, 6 d 0 h 19 m, `ise-cpu256-06`). Stop: `since_improvement=148/150`, `min_episodes=250`, `rewinds=3`, wall `519436s/518400s`. Last DONE ep315. Last probe ep312 = **0.0377** (r56 0.029 / r20 0.047), under the freeze. One snap: `snapshots/ep0167` (25 Sep 19:34, area **0.0608**). `policy_config` pins `factored_head=true`, ranking menu l1/fpgm/bn_scale/svd/taylor, `ft_recipe=A`, in-band linear (`cbrt_cubes`), area probe, 12/4, P5-B2 catalog. Tracebacks 0.

**Read.** This is the two-decision-head cell against the area train `21536396` (freeze ep83 / 0.0586). Area scores are not TESTs. Verdict is a skinny-r56 TRAJ of `ep0167` vs that area freeze at equal keep — **TEST only on Ido GO**. Do not auto-queue. Do not start a second factored train. Group-token `21716380` took the GPU (`ise-6000-04`, R since ~13:59). Do **not** lock. Do not edit the draft.

---

## 135. Budget + STOP train (**21715228**) COMPLETED — no snapshot (best area 0.0273 < 0.05 baseline)

Parent train **COMPLETED** 13:58 IDT 28 Sep (exit 0, 13 h 7 m, `ise-6000-04`). Stop: `reward_not_improving=True`, `since_improvement=164/150`, `min_episodes=250`, `rewinds=3`. Last DONE ep250. Peak probe ep84 = **0.0273** (both halves ~0.027); last probe ep240 = **0.0239**. `SPECTRA_SNAPSHOT_BASELINE=0.05` → **no `snapshots/` freeze** (score never cleared the bar). FLAGS confirmed: `compression_rates=[1.0, 0.01, 0.02, 0.04, -1.0]`, `SPECTRA_ACTION_MENU=budget`, `cbrt_cubes`, area probe, P5-B2, STOP scale 100. Tracebacks 0. Never quote `21703443`.

**Read.** Probe area is about half the clean-catalog area freeze (0.0586). STOP was not a learned depth choice (rewinds exhausted; no freeze). A TRAJ of `latest_best` vs `21536396` at equal keep is the kill cell — **TEST only on Ido GO**. Recommendation: do not TEST a sub-baseline actor; the training probe already lost. Do not start a second Budget train. Do **not** lock. Do not edit the draft.

---

## 136. Area-train freeze TRAJ (**21725471** COMPLETED) — clean-catalog `ep0083` vs 2-pass mild §93 — PRELIM

Ido GO A. Skip-train `eval_c10_thin_traj` of **21536396** `snapshots/ep0083` (area probe 0.0586). Job **COMPLETED** 2 h 59 m (ended 28 Sep 23:10 cluster, exit 0, `cs-pheno-09` 3090 = **batch 256**). Pin: det=1, 2-pass, TEST 40/10, look-ahead 0, group-once from `policy_config`. Quote `[eval] TRAJ val_best`. Do not quote wrap. Fair yardstick = 2-pass mild §93 (GTX 1080 = **batch 64**) / 2-pass L1 §94. Twin factored TRAJ **21725472** COMPLETED 01:58 (§137). Protocol: **legacy** (train-split val).

| Net | area ep0083 **21725471** | 2-pass mild §93 | 2-pass L1 §94 |
|---|---|---|---|
| r20-w2 | **−5.1 @ 0.536/0.655** (0.648→0.597), val **−6.39** | **−3.4 @ 0.536/0.655** | −7.3 @ 0.417/0.608 |
| r56-w4 | **−6.8 @ 0.923/0.769** (0.888→0.820), val **−8.94** | **−6.6 @ 0.923/0.769** | −7.8 @ 0.898/0.720 |

**Read.** Same keep as mild on both nets. r56 is the 90 %-rule clone (mild staircase step 38). r20 at identical widths sits inside mild's own batch-256 seed spread (§138: −4.0 / −5.3 / −5.6) — do **not** call the area head worse than mild on r20. Geometry on r56-w4 is 0.9 on every legal row through step 57. Factored r56 (§137) picked the next stair, not a different policy. Stay on the **area-train stack**. Group-token stays **held**. Do **not** lock. Do not edit the draft.

---

## 137. Factored-head freeze TRAJ (**21725472** COMPLETED) — `ep0167` vs area §136 / mild §93 — PRELIM

Ido GO A. Skip-train `eval_c10_thin_traj` of **21536398** `snapshots/ep0167` (area probe 0.0608). Job **COMPLETED** 5 h 48 m (ended 29 Sep 01:58 cluster, exit 0, `ise-pheno-05` 2080 Ti = **batch 128**). Same pin as §136. Quote `[eval] TRAJ val_best`. Protocol: **legacy** (train-split val, batch 128).

| Net | factored ep0167 **21725472** | area §136 (batch 256) | 2-pass mild §93 (batch 64) |
|---|---|---|---|
| r20-w2 | **−3.7 @ 0.536/0.655** (0.648→0.611), val **−4.02**, step 40 | −5.1 @ 0.536/0.655 | −3.4 @ 0.536/0.655 |
| r56-w4 | **−7.1 @ 0.832/0.730** (0.888→0.817), val **−9.76**, step 39 | −6.8 @ 0.923/0.769, step 38 | −6.6 @ 0.923/0.769, step 38 |

**Read.** Compression-rate census: every legal r56-w4 row is **0.9** through step 57 (same as area, same as N0). Step 39 keep **0.832 / 0.730** is mild's documented next stair after step 38's 0.923, with val −9.76 sitting on −τ. That is the band-edge lottery, not a ranking-head effect. **Do not promote the factored head.** Stay on the area stack. r20 at equal keep is kinder than area and close to §93 — confounded with batch 128 vs 256 vs 64 (§138). Do **not** lock. Do not edit the draft.

---

## 138. N0 — 2-pass mild seed × batch-256 thin — PRELIM

No-agent mild, det TRAJ, 2-pass, recipe A, `tree_v9`. Protocol: **legacy** (train-split val), **batch 256** pinned or landed. Yardstick §93 is the same walk at seed 42 on a GTX 1080 (**batch 64**). Quote `[eval] TRAJ val_best`.

| Job | Seed / GPU | r20-w2 step 40 (0.536/0.655) | r56-w4 val_best |
|---|---|---|---|
| §93 **21413236** | 42 / 1080 batch **64** | **−3.4** (0.648→0.614), val −4.27 | **−6.6 @ 0.923/0.769**, step 38, val −9.32 |
| **21726098** COMPLETED 02:09, pheno-06 3090 | 43 / batch **256** | **−4.0** (0.648→0.608), val −4.85 | **−6.4 @ 0.923/0.769**, step 38, val −9.13 |
| **21726099** COMPLETED 02:08, pheno-08 3090 | 44 / batch **256** | **−5.3** (0.648→0.595), val −5.52 | **−6.4 @ 0.923/0.769**, step 38, val −8.90 |
| **21726342** COMPLETED 06:48, ise-1080-01, batch **pinned 256** | 42 / batch **256** | **−5.6** (0.648→0.592), val −6.29 | **−6.7 @ 0.923/0.769**, step 38, val −9.22 |

**Read.** All three batch-256 seeds stay at **0.923** on r56-w4. The N0 kill (any seed ≤ 0.83 ⇒ actor-vs-mild r56 is band-edge noise) does **not** fire. Geometry is still 0.9 throughout (§137). Batch-256 r20 at identical widths: **−4.0 / −5.3 / −5.6**. §93's −3.4 (batch 64) is outside that cluster. GO A area's −5.1 sits inside it. Do **not** lock. Do not edit the draft.

---

## 139. N4 — 3-pass mild + rollback, legacy protocol (**21726100** COMPLETED) — PRELIM

`SPECTRA_EVAL_ROLLBACK=1`, 3 passes, seed 42, `tree_v9`, `cs-pheno-09` 3090 = **batch 256**. COMPLETED 3 h 15 m, 29 Sep 02:25, exit 0. Quote `[eval] TRAJ val_best`. Diagnostic, no kill. Protocol: **legacy**. P twin is **§145**.

| Net | val_best | Rollbacks |
|---|---|---|
| r20-w2 | **−7.0 @ 0.417/0.608** (0.648→0.578), val **−7.51**, step 61 | none in the printed TRAJ |
| r56-w4 | **−8.4 @ 0.834/0.678** (0.888→0.804), val **−10.00**, step 150 | **23** undos: first at step **39** (10 layers locked), then 49, 51, 55, 58, **59** (10 layers), 66, 70, 74, **77** (10 layers), 79, 81, 89, 91, 102, 104, 110, 133, 142, 144, 152, 155, 157 |

**Read.** Not "step 39 only". Many stage-3 internals roll back too, so the sitting's N4 fork says the binding problem is **recovery** (F arms), not only the first stream cut. Selected keep 0.834 at val exactly −10 is the lottery again. Re-read under P when **21726338** prints. Do **not** lock. Do not edit the draft.

---

## 140. VGG-11 C100 canary — P vs legacy, train FT 12/4 — PRELIM

Same net `vgg11_bn_cifar100_chenyaofo_70.78`, 2-pass mild, `tree_v9b`. Admitted = kept ≤ 0.98 with val ≥ −10. Quote `[eval] TRAJ val_best`. **No `final_ft` lines** (pickle crash, §141). Do not mix the TEST columns: P TEST is the **5k half**.

| Job | Protocol | val_best | terminal (not the quote) |
|---|---|---|---|
| **21726336** COMPLETED 02:22 | **P** (clean val, batch 256, final FT intended 100) | **−8.4 @ 0.659/0.680** (0.714→0.630), val **−7.30**, step 19 | same point |
| **21726339** COMPLETED 02:36 | **legacy** (train-split val, batch 256, no final FT) | **0.0 @ 1.000/1.000**, step −1, val +0.00 | TEST **−9.6 @ 0.659/0.680**, val **−36.11** |

Size-matched, quoted even if val left τ: P `size_param0.90` −9.4 @ 0.889 val −8.46; P `size_param0.80` −8.4 @ 0.787 val −7.54. Legacy at the same keeps: TEST −10.2 / −9.2 with val **−35.1 / −34.9**.

**Read.** **Admitted under P, not under legacy.** Same terminal keep (0.659); TEST is similar (−8.4 vs −9.6); val is −7.3 vs −36.1. The C100 train-pool block on this canary was the memorized val. Do **not** rewrite claim C6 from this one net. Do **not** emit the diverse catalog. No origin-controlled final-FT gain (crash). Do **not** lock. Do not edit the draft.

---

## 141. V9b provenance — memorized val, batch lottery, final-FT pickle (29 Sep) — not a TEST row

Opus 5.5 sitting 28 Sep 20:39–23:45. Three measurement defects, confirmed from finished logs (`PROMPT_FABLE_NEXT_SITTING.md` §10). Fixes live in **`tree_v9b` only**, all default **off**. Do not re-grade pre-V9b rows.

1. **Memorized val.** Legacy val is carved from the CIFAR train split the zoo nets were trained on. Unpruned val vs TEST: R56·C10 1.000/0.943, VGG-16 C10 1.000/0.936, VGG-19 C100 0.999/0.739, r56-w4 0.926/0.888, r20-w2 0.658/0.648. §124 VGG-19 C100 sat at TEST −8.8 with val −30.9 (`val_best` = unpruned). Every train's reward read this val.
2. **Batch lottery.** Fine-tune batch follows the GPU (1080 64, 2080 128, 3090/4090 256, rtx_6000 384). §93 ran at 64; most later rows at 256. Pin `SPECTRA_BATCH_SIZE=256` in new cells.
3. **`val_best` lottery** at a flat band edge (r56-w4 hovers near −10 for ~30 steps). Size points + `scripts/traj_readout.py` are the second readout.

**Smoke 21726334** (never ledger as TEST): split banner `n_train=50000, n_val=5000, n_test=5000`, header `val_from_test=0.5 batch=256 size_points=param:0.9 final_ft=1+origin`, `size_param0.90` printed, then `PicklingError: Can't pickle thin_res_net.ResNet` inside `_run_final_ft` `torch.save`. Exit 0, so afterok children **started**. `traj_models/*.pt` are ~1 KB stubs. Same pickle on P canary (`vgg_chenyaofo.VGG`). Walk `val_best` is usable; `final_ft` is not. Do **not** patch `tree_v9b`. Do **not** scancel the children.

**P twins 21726337 COMPLETED 08:28.** Unpruned val vs TEST on chenyaofo R56·C10 **0.942 / 0.943**. Walk `val_best` on all three nets; VGG-19 C100 **off unpruned**. See **§142**. Same pickle on every P cell after the walk (`resnet_chenyaofo`, `vgg_chenyaofo`, `thin_res_net`, `vgg_depgraph`). Do **not** patch `tree_v9b`.

---

## 142. P twins — Catalog L chenyaofo R56·C10 / VGG-16·C10 / VGG-19·C100 (**21726337** COMPLETED) — PRELIM

Protocol **P**: clean val, batch 256, final FT intended 100 (pickle-crashed; no `final_ft` / origin). 2-pass mild, `tree_v9b`, COMPLETED 6 h 6 m, 29 Sep 08:28, `cs-pheno-06`, exit 0. TEST is the **5k half**. Quote `[eval] TRAJ val_best`. Yardstick = legacy §124 (`21536393`, batch 256, train-split val, 10k TEST).

| Net | P **21726337** | legacy §124 | unpruned val vs TEST (P) |
|---|---|---|---|
| R56·C10 | **−2.8 @ 0.661/0.662** (0.943→0.914), val **−3.22**, step 112 | −3.3 @ 0.661/0.662, val −8.18 | **0.942 / 0.943** |
| VGG-16·C10 | **−2.8 @ 0.657/0.678** (0.937→0.910), val **−2.78**, step 29 | −3.5 @ 0.657/0.678, val −8.20 | TEST 0.937; val tracks TEST |
| VGG-19·C100 | **−6.7 @ 0.657/0.673** (0.745→0.678), val **−6.44**, step 35 | **0.0 @ 1.000/1.000** (unpruned) | TEST 0.745; val tracks TEST |

Size-matched, quoted even if val left τ: R56 `size_param0.80` −2.7 @ 0.794; `0.70` −2.7 @ 0.694. VGG-16 `0.80` −2.9 @ 0.796; `0.70` −2.8 @ 0.698. VGG-19 `0.80` −6.4 @ 0.796 val −4.74; `0.70` −6.8 @ 0.688 val −6.32.

**Read. Twins GO.** Unpruned val agrees with TEST within ~1.5 pp (not the §124 5.7–26 pp gaps). VGG-19 C100 `val_best` is **off unpruned** at the same keep §124's terminal sat at with val −30.9. Adopt **P as the walk TEST protocol**. Do not quote `final_ft`. Do **not** rewrite C6 from this plus the VGG-11 canary alone. Do **not** lock. Do not edit the draft.

---

## 143. P thin vs N0 s42-b256 — val effect, same seed and batch — PRELIM

P **21726335** COMPLETED 07:54, `ise-1080-01`, 5 h 46 m (pickle after each net). Legacy **21726342** §138. Both seed 42, batch 256, 2-pass mild. P TEST is the **5k half** (unpruned r20 0.649 / r56 0.890 vs legacy 0.648 / 0.888).

| Net | P **21726335** | legacy N0 s42-b256 **21726342** |
|---|---|---|
| r20-w2 step 40 | **−3.7 @ 0.536/0.655** (0.649→0.611), val **−1.42** | **−5.6 @ 0.536/0.655**, val **−6.29** |
| r56-w4 | **−10.1 @ 0.739/0.585** (0.890→0.788), val **−9.52**, step 76 | **−6.7 @ 0.923/0.769**, val −9.22, step 38 |

Size-matched: P r20 `size_param0.80` +0.1 @ 0.774 val +1.16; `0.60` −4.2 @ 0.584. P r56 `size_param0.80` −7.6 @ 0.795 val −7.18; `0.60` NONE.

**Read.** Same seed, same batch, same widths on r20: clean val is **1.9 pp kinder** on TEST and **4.9 pp kinder** on val. On r56-w4, P selects **0.739** (in-band on val) vs legacy **0.923**; TEST −10.1 is 0.1 pp past τ on the 5k half. That is the predicted val effect (r56 gap 3.8 pp under legacy). Do **not** lock. Do not edit the draft.

---

## 144. P N4 — 3-pass mild + rollback under clean val (**21726338** COMPLETED) — PRELIM

`SPECTRA_EVAL_ROLLBACK=1`, 3 passes, seed 42, P flags, `tree_v9b`, `cs-pheno-08`, COMPLETED 4 h 40 m, 29 Sep 07:16. Quote `val_best`. Compare legacy §139.

| Net | P **21726338** | legacy §139 |
|---|---|---|
| r20-w2 | **−7.9 @ 0.417/0.608** (0.649→0.569), val **−5.92**, step 61 | −7.0 @ 0.417/0.608, val −7.51 |
| r56-w4 | **−11.0 @ 0.691/0.537** (0.890→0.780), val **−9.56**, step 150 | −8.4 @ 0.834/0.678, val −10.00, step 150 |

Rollbacks on r56-w4: **18** undos, first at step **77** (10 layers), then 81, **96** (10 layers), 102, 108, 110, 133, 136, 140, 142, 144, 146, 152, 155, 157, 161, 163, 169. `size_param0.80` −7.2 @ 0.795 val −8.06; `0.60` NONE.

**Read.** Still many internals, not step-39-only — recovery fork stands. Clean val **delays** the first undo (77 vs legacy 39) and the selected keep is deeper (0.691 vs 0.834). TEST −11.0 is past τ on the 5k half; val stayed in band. Do **not** lock. Do not edit the draft.

---

## 145. DepGraph VGG-19 C100 under P (**21726341** COMPLETED) — PRELIM, quote-only vs literature

DepGraph's own `vgg19_cifar100_dep_graph_73.5.pth`, 3-pass mild, P, `tree_v9b`, COMPLETED 2 h 14 m, 29 Sep 09:30, `cs-pheno-08`. TEST is the **5k half** (unpruned 0.740). No `final_ft`. Twin under P is §142. DepGraph published (Fang et al. CVPR 2023): **73.50 → 70.39 (−3.11) at 8.92×** after sparse learning + target fine-tune. Quote beside; **never "beats".**

| Readout | SPECTRA P mild | Caption |
|---|---|---|
| `val_best` | **−7.9 @ 0.534/0.550** (0.740→0.660), val **−7.18**, step 47 | τ-matched walk; off unpruned |
| `size_param0.70` | **−6.3 @ 0.684/0.686** (0.740→0.677), val **−5.92**, step 29 | size-matched, quoted even if val left τ |
| `size_param0.50` | NONE | 3-pass mild did not reach 0.50; not DepGraph's 8.92× (~0.11 keep) |

**Read.** Same story as the zoo twin: clean val lets VGG-19 C100 `val_best` leave 1.000. Not a SOTA comparison (different FT, 5k TEST, no sparse learning, pickle killed the 100-ep finish). DepGraph R56 P is **§146**. Do **not** lock. Do not edit the draft.

---

## 146. DepGraph R56·C10 under P (**21726340** COMPLETED) — PRELIM, quote-only vs literature

DepGraph's own `resnet56_cifar10_dep_graph_93.53.pth`, 5-pass mild, P, `tree_v9b`, COMPLETED 5 h 58 m, 29 Sep 12:46, `cs-pheno-09`. TEST is the **5k half** (unpruned 0.934). No `final_ft` (pickle `resnet_chenyaofo.CifarResNet`). Legacy 2-pass same checkpoint: §131 **−3.1 @ 0.661/0.662**. DepGraph published: **93.53 → 93.64 (+0.11) at 2.57× FLOPs** (FLOPs kept ≈ 0.39) after sparse learning + target fine-tune. Quote beside; **never "beats".**

| Readout | SPECTRA P mild 5-pass | Caption |
|---|---|---|
| `val_best` | **−4.0 @ 0.356/0.369** (0.934→0.894), val **−4.04**, step 283 | τ-matched walk; 5-pass ceiling, not §131's 2-pass keep |
| `size_flop0.60` | **−3.1 @ 0.638/0.599** (0.934→0.902), val **−2.56**, step 136 | size-matched |
| `size_flop0.39` | **−3.9 @ 0.382/0.380** (0.934→0.894), val **−4.04**, step 267 | **DepGraph 2.57× size**; quote next to +0.11 |

**Read.** At their FLOP point we are **−3.9** on the 5k half vs their **+0.11** after a different protocol. Caption the gap; do not call it a loss of the agent (this row is no-agent mild). Walk numbers usable; `final_ft` missing. V9b GPU queue is now empty. Do **not** lock. Do not edit the draft.

---

## 147. Zero-GPU readout of the finished P walks (29 Sep) — not a TEST row

Offline, login node, `tree_v9c` readers on the `tree_v9b` walks 21726335 / 36 / 37 / 40 / 41 (rollback walk 21726338 excluded: its walk reads val). Source: `PROMPT_FABLE_NEXT_SITTING.md` §13.1. No new run and no selection on a reported half.

**Census (`crossfit_readout.py`).** **0 of 343** cut points on six full-width nets under P have val Δacc > 0 (R56 twin, VGG-16, VGG-19 twin, DepGraph R56, DepGraph VGG-19, VGG-11 C100 canary). The best is DepGraph R56 at −0.72 pp after its first cut. Thin r20-w2 does gain: 8 of 18 under P (4 of 18 legacy), max +3.76. Published light cuts of over-parameterised CIFAR nets recover to ≥ 0 under SGD + crop/flip (DepGraph R56 2.11× +0.24; Network Slimming VGG-19 +0.14). So the "no accuracy increase" fact belongs to the walk fine-tune (Adam 1e-3, no augmentation), not to the reward and not only to memorized val.

**Cross-fit, P, 10k (both halves).** The mild walk reads neither half, so the τ rule runs twice with the halves swapped (mean shown; both fold points in the log), and size points use both halves. A two-fold estimate of the rule, not one network's accuracy.

| Net | τ = 10 rule, 10k | Size points, 10k | 5k row |
|---|---|---|---|
| R56 twin | −3.03 @ 0.661 | — | §142 |
| VGG-16 | −2.78 @ 0.657 | — | §142 |
| VGG-19 twin | −6.55 @ 0.657 | — | §142 |
| r20-w2 | −2.57 @ 0.536 | — | §143; halves disagree by 2.3 pp |
| r56-w4 | −9.86 | params 0.80: −7.37 | §143; folds pick 0.739 / 0.743 |
| DepGraph R56 | −4.02 @ 0.356 | FLOPs 0.47: −3.32; FLOPs 0.39: −3.98 | §146; DepGraph +0.24 / +0.11 |
| DepGraph VGG-19 | −7.55 @ 0.534 | params 0.70: −6.09 | §145 |
| VGG-11 C100 canary | −7.86 @ 0.659 | — | §140 |

**Read.** Walk numbers only, no final FT. Quote as "P, 10k (cross-fit)". Not valid for rollback walks or accuracy-reading policies. Do not lock.

---

## 148. Crop+flip in the walk fine-tune, C100 gate (**21729554** COMPLETED vs **21729552** TIMEOUT) — PRELIM, 8 of 8 nets; gate adopt rule met; **C100 catalog emitted**

`tree_v9b`, P, gate profile (τ = 10, 2-pass mild, walk FT 12/4), both on 1080s. Arm adds `SPECTRA_FT_AUG=1` (RandomCrop+Flip on train images only). TEST is the **5k half**. Mild holds the same widths at the same step, so the size columns are paired (keep shown once). First read (r20-w13 only, 29 Sep 17:01) is the first row. Rows for nets 1–6 stamped 30 Sep 02:13; nets 7–8 and the admit lines 30 Sep 11:45. 21729554 COMPLETED in 15 h 05 m (`cs-1080-01`). 21729552 hit its 16 h wall (`cs-1080-05`) at mbv2x1 step ~102, before that net's TRAJ rows, and never started densenet40.

| Net (unpruned) | size 0.90: no aug → aug | size 0.80: no aug → aug | `val_best` no aug | `val_best` aug | Paired val (n, mean, better) | Admit no aug / aug |
|---|---|---|---|---|---|---|
| r20-w13 (0.700) | −10.9 → −5.2 (**+5.7**) @ 0.861 | −13.1 → −6.9 (**+6.2**) @ 0.787 | −9.2 @ 0.926, val −9.54 | −8.4 @ 0.662, val −8.84 | 24, **+5.48**, 96 % | yes / yes |
| r56-w9 (0.733) | −13.7 → −7.7 (**+6.0**) @ 0.860 | −12.2 → −9.5 (**+2.7**) @ 0.792 | −9.0 @ 0.941, val −8.46 | −9.5 @ 0.647, val −9.70 | 60, **+4.94**, 98 % | yes / yes |
| resnet32 (0.706) | −7.8 → −5.4 (**+2.4**) @ 0.873 | −8.1 → −4.6 (**+3.5**) @ 0.793 | −9.5 @ 0.663, val −8.42 | −5.3 @ 0.663, val −4.90 | 36, **+2.48**, 97 % | yes / yes |
| vgg11_bn (0.714) | −9.6 → −5.6 (**+4.0**) @ 0.889 | −7.3 → −6.4 (**+0.9**) @ 0.787 | −9.4 @ 0.659, val −8.78 | −4.3 @ 0.659, val −4.98 | 18, **+2.07**, 100 % | yes / yes |
| vgg13_bn (0.751) | −7.1 → −5.1 (**+2.0**) @ 0.887 | −9.7 → −6.4 (**+3.3**) @ 0.798 | −9.3 @ 0.661, val −8.54 | −4.8 @ 0.658, val −4.58 | 22, **+3.39**, 100 % | yes / yes |
| mbv2x0.5 (0.711) | −2.6 → −0.8 (**+1.8**) @ 0.868 | −1.9 → −0.8 (**+1.1**) @ 0.798 | −2.9 @ 0.692, val −2.78 | −2.0 @ 0.692, val −2.16 | 50, +0.92, 92 % (CONTINUE) | yes / yes |
| mbv2x1 (0.747) | not finished → **−1.3** @ 0.889 | not finished → **−2.1** @ 0.789 | not finished (wall) | −0.4 @ 0.671, val −1.18 | 49, **+1.58**, 94 % | not finished / yes |
| densenet40 (0.703) | not started → **−3.9** @ 0.900 | not started → **−4.7** @ 0.798 | not started | −6.0 @ 0.696, val −6.14 | — | not started / yes |

**Paired read (`paired_steps.py`, val).** `ADOPT?` on 6 of 7 paired nets; mbv2x0.5 CONTINUE. First net at 16 cuts read +4.60, final +5.48.

**Admit lines (rule: TRAJ `val_best` kept ≤ 0.98 and val Δ ≥ −10).**
- *Aug gate 21729554:* **8 / 8 admitted**. Kept 0.647–0.696 (every net reaches the deepest 2-pass mild point); val Δ −1.18 (mbv2x1) to −9.70 (r56-w9). The two thin residual nets sit near the edge: r56-w9 −9.70, r20-w13 −8.84.
- *No-aug gate 21729552:* **6 / 6 finished admitted**. mbv2x1 was cut by the wall and densenet40 never started: "not finished", never "not admitted". r20-w13 and r56-w9 admit only at a shallow `val_best` (0.926 and 0.941 kept); every later point is below −10.
- *Legacy gate (§109, train-split val, Adam 1e-3 12/4):* 0 / 8.

**Emit (Ido GO 30 Sep 11:08).** The C100 recipe is the Stage-4 train's (P + crop+flip, 12/4), so the aug gate is the admitting gate.
- `configs/v7_c100_gate.json`: all 8 rows `admitted`, `probe_job` 21729554, `val_best_*`, and a `no_aug_gate` column.
- `python scripts/build_v5_catalog.py --emit-admitted --intended configs/database_offline_v7_diverse.json --gate configs/v7_c100_gate.json --out configs/database_offline_v7_diverse_admitted.json --min-c100 8` wrote **16 nets** (the 8-net C10 core + 8 C100). `--check-admitted` ok; `tests/test_v5_catalog.py` 16/16, with a new check that the committed file equals the builder's output.
- No job reads this file yet. N8 (`docs/N8_DIVERSE_TRAIN_ROADMAP.md`) needs a profile that accepts it.

**Read.** At the same widths crop+flip is kinder at **12 of 12** size points, **+0.9 to +6.2 pp TEST, mean +3.3**. Both arms admit **6/6** (TRAJ `val_best` kept ≤ 0.98, val Δ ≥ −10). The pre-registered gate rule (admits ≥ no-aug and kinder TEST at equal keep on ≥ 5/8) is **met on 6 nets**, whatever nets 7–8 show. Four of six nets stay inside τ to the deepest 2-pass mild point (~0.66 keep) in **both** arms: under clean val, C100 recovery at the live recipe is a gate pass (Q4 evidence "yes"; emitted 30 Sep, above), and crop+flip halves the drop there (resnet32 −9.5 → −5.3, VGG-11 −9.4 → −4.3, VGG-13 −9.3 → −4.8). One seed per arm: the smallest gaps (VGG-11 size 0.80 +0.9, mbv2x0.5 +1.1) sit inside the 5k-half noise. Still a walk-recipe result, not a training-recipe pass: the thin-C10 rule is §150. Do **not** rewrite C6 as "C100 solved". Do not lock. Do not edit the draft.

---

## 149. 100-epoch SGD final fine-tune, DepGraph VGG-19 C100 (**21729551** COMPLETED) — PRELIM; O2 adopt rule met on this cell

`tree_v9b`, P, 3-pass mild (τ = 10, walk FT 40/10), then `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1`: SGD lr 0.01, momentum 0.9, wd 5e-4, cosine, crop+flip, batch 128, no KD, from the inherited weights. COMPLETED 8 h 06 m, 30 Sep 00:27, `cs-1080-05`; 27–38 min per candidate. TEST = **5k half** (unpruned 0.740); 10k = both halves, size points only (`val_best` is val-selected). Nothing saved (`SAVE_TRAJ_MODELS` unset on v9b). Honest gain = (final − walk) − (origin final − origin walk), `final_ft_readout.py`.

| Point | Keep params / FLOPs | Walk | Final FT | Honest gain | 10k final |
|---|---|---|---|---|---|
| `val_best` step 47 | 0.534 / 0.550 | −7.82 | **−3.64** | **+4.08** | n/a |
| `size_param0.70` step 29 | 0.684 / 0.686 | −7.08 | **−2.52** | **+4.46** | **−2.39** |
| `size_param0.60` step 42 | 0.599 / 0.590 | −8.90 | **−3.28** | **+5.52** | **−3.04** |
| origin step −1 | 1.000 | 0 | +0.10 | — | +0.56 |

**Re-walk vs §145** (same checkpoint and flags; `cs-1080-05` vs `cs-pheno-08`). `val_best` step 47 at the same keep: TEST −7.8 vs −7.9, val −7.02 vs −7.18. `size_param0.70` step 29: TEST −7.1 vs −6.3, val −6.30 vs −5.92. One re-walk moves a size point by up to **0.8 pp** TEST at equal widths (SKU / cuDNN nondeterminism). Caption walk gaps below ~1 pp as inside that noise.

**Cross-fit of this walk (10k).** τ = 10: −7.42 @ 0.534 (both folds step 47). τ = 5: −4.83 (folds pick 0.822 and 0.986). Census 0/46 cut points with val Δ > 0 (max −3.50).

**DepGraph** (Fang et al. CVPR 2023, their checkpoint): 73.50 → 70.39 (**−3.11**) at **8.92×** params (keep ≈ 0.11) after sparse learning + fine-tune. We are −2.52 at 0.684 keep and −3.64 at 0.534, i.e. far less compression. Quote beside; **never "beats"**.

**Read.** Honest gain **+4.1 to +5.5 pp** at every point, ≥ 2 pp: the pre-registered O2 rule is met on this C100 cell. The C10 cells (21730500 / 01 / 06) are still PD. Bar-3 rows (the SOTA comparison) use the final FT, captioned "after a 100-epoch SGD final fine-tune; origin +0.10". Bar 2 (same loop, same size vs heuristics) stays on the walk recipe. It does not change what the agent learns: the reward still reads the walk fine-tune. Do **not** lock. Do not edit the draft.

---

## 150. Thin pair at the train fine-tune 12/4 under P; crop+flip as the training recipe (**21729555** vs **21729556**, both COMPLETED) — PRELIM; training rule PASSED

`tree_v9b`, P, 2-pass mild, walk FT **12/4** (the train recipe), seed 42, `cs-pheno-06`. 555 COMPLETED 01:45 (1 h 11 m). 556 COMPLETED 03:12 (1 h 26 m). TEST = 5k half; 10k = both halves / cross-fit. Pre-registered rule (sitting §13.4): crop+flip becomes the training recipe if the C100 gate rule holds (§148: met) and r56-w4 is not > 0.5 pp worse at equal keep. r20-w2 is a 2 pp guard only (seed SD ≈ 0.85 pp; the halves differ by up to 2.3 pp).

| Net (unpruned) | Point | P 12/4 (555) | P + crop+flip 12/4 (556) | Aug − P, TEST | 10k |
|---|---|---|---|---|---|
| r56-w4 (0.890) | `size_param0.80`, step 47, 0.795 | −8.2, val −8.52 | **−5.9**, val −5.34 | **+2.3** | −5.59 vs −8.35 (+2.8) |
| r56-w4 | `val_best` | −10.6 @ 0.741, step 70, val −9.42 | **−5.1 @ 0.622**, step 112, val −5.30 | deeper and kinder | τ10 −5.21 @ 0.622 vs −10.30 @ 0.741 |
| r56-w4 | `size_param0.60` | NONE | NONE | — | — |
| r20-w2 (0.649) | `val_best`, step 40, 0.536 | −5.3, val −4.42 | −6.6, val −5.18 | −1.3 | τ10 −5.87 vs −4.82 |
| r20-w2 | `size_param0.80`, step 17, 0.774 | −0.1, val +1.30 | −3.2, val −0.96 | **−3.1** | −2.08 vs +0.64 |
| r20-w2 | `size_param0.60`, step 36, 0.584 | −5.2, val −3.04 | −6.2, val −4.78 | −1.0 | −5.46 vs −4.11 |

**Paired read (val).**
- r56-w4: 60 cuts, **+5.03 pp**, better on 100 %; deepest pair step 112, −5.30 vs −12.92.
- r20-w2: 16 cuts, −1.92, better on 0 % (the generic KILL label; r20 is a guard, not a decider).
- Recipe length, info only (555 vs P thin 40/10 21726335): r56-w4 −1.03 pp over 60 pairs, r20 −0.93. The train FT is harsher than TEST's.

**Census (val Δ > 0).** P 12/4: r20 8/18 (2 on both halves), r56-w4 0/62. Aug 12/4: r20 3/18 (0 on both), r56-w4 0/62 (max −1.88).

**Read.**
- **Training rule PASSED.** On r56-w4, crop+flip is +2.3 pp TEST at equal keep (10k +2.8). The τ rule stays inside the band to the end of the 2-pass walk: 0.622 at −5.1, against 0.741 at −10.6.
- **The r20 guard holds on the mean:** −1.8 pp TEST over the three equal-keep points, paired val −1.92. One point is at −3.1. Crop+flip hurts the 5k-parameter, 64.8 % r20-w2, a net that underfits (NetAug, Cai et al. ICLR 2022). r20-w2 is a hold-out diagnostic, not a train net.
- **Crop+flip becomes the Stage-4 train recipe (§151).**
- **Still no cut gains on r56-w4 with aug (0/62),** so the cubic reward's positive branch stays unvisited there.

Do not lock. Do not edit the draft.

---

## 151. Stage-4 train released (**21737123**) — the area train under clean val + crop+flip

Ido GO 30 Sep 01:56: Stage 4 = one train, the area train under P, plus crop+flip if it passes the training rule.
- *Submitted.* From `tree_v9c` (frozen, CPU pytest 367/367): `offline_train_v6_inband_p5b2` + `SPECTRA_PROBE_SCORE=area` + **P** (`SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256`) + **`SPECTRA_FT_AUG=1`**. Nice 0; the picker's default card (`rtx_6000`; the control ran on `ise-6000-09`, RTX 6000 Ada); 7-day scheduler limit, 6-day runtime fuse.
- *Sequence.* The P-only arm **21737095** was submitted at ~02:10 and held. The P+aug arm was submitted held at ~02:35. When §150 passed at 03:11, 21737123 was released and 21737095 cancelled; it never started.
- *Control.* Area train **21536396** (`tree_v7`, legacy val, adaptive batch; freeze `ep0083` ≡ mild's 90 % rule, §136).
- *Unchanged from the control.* P5-B2 catalog (C10 + SVHN), in-band linear reward (structural + cbrt_cubes), 5-action menu (1.0 | 0.9/0.8 × l1/fpgm), probes r56-w6 + r20-w10 every 12 episodes, 12/4 train FT, PPO 4×4, lr 3e-4, entropy 0.01, min 250 episodes / patience 150 / 3 rewinds, snapshot baseline 0.05.
- *Aug scope.* CIFAR only (RandomCrop pad 4 + flip on train images). The SVHN net stays unaugmented.
- *Pre-flight.* On this path the `tree_v7` → `tree_v9c` sbatch diff is added profiles, the `PROBE_SET` case (default = the same thin probes) and resume logic. Catalog byte-identical; `bash -n` ok.

**Why these changes.**
- Every train so far scored its reward on memorized train-split val (§141). Unpruned val read ≈ 1.000, so every cut looked like a large drop and the mildest policy scored best. Under P the reward sees the drop TEST sees.
- Crop+flip gives the agent the recovery the zoo nets were trained with (§148, §150).
- One train with both changes is the more decisive single experiment: if it cannot leave mild, the weaker P-only recipe will not. Attribution (P alone) is O40, after a success only.

**Pre-registered reads.**
1. `policy_config.json` differs from 21536396's only in the P keys (`VAL_FROM_TEST`, `VAL_TEST_FRACTION`, `SPLIT_SEED`, `BATCH_SIZE`) and run id / stamp. `SPECTRA_FT_AUG` is not a `policy_config` key on `tree_v9c`; it is recorded in the log's `SPECTRA_* env` header and the job name.
2. The log shows `Val from test` for cifar-10 **and** svhn, and `FT aug on` for cifar-10 only.
3. Probe area is on a new scale: never compare it with 0.0586.
4. The verdict is a TRAJ of a freeze, on Ido's GO, against 2-pass mild at equal keep on the thin pair **under the same walk recipe** (P + crop+flip, 40/10: 21729557). P thin 21726335 is the no-aug audit column. The coverage set follows.
5. Mild-clone read at that TEST: a 0.9 rate on ≥ 95 % of legal r56-w4 rows means "mild clone under P+aug" (the §136 read).

**Start (R 30 Sep 03:14, `ise-cpu256-32`, RTX 6000 Ada).** Reads 1 and 2 pass.
- The `SPECTRA_* env` header has `VAL_FROM_TEST '1'`, `BATCH_SIZE '256'`, `FT_AUG '1'`, `PROBE_SCORE 'area'`.
- `Val from test on cifar-10: n_train=50000 … n_val=5000, n_test=5000`; `on svhn: n_train=73257 … n_val=13016, n_test=13016`.
- `FT aug on cifar-10: RandomCrop+Flip on train only`, and no aug line for svhn.
- `policy_config.json` vs 21536396: only `created`, `SPECTRA_RUN_ID`, `SPECTRA_BATCH_SIZE` (unset → 256) and `SPECTRA_VAL_FROM_TEST` (unset → 1). The fraction and split seed are at their defaults and are not recorded.
- *Pace.* Episode 0 took 993 s vs the control's 422 s (the same 24 steps); one walk-FT epoch 6.7 s vs 2.9 s.
  - Causes, all protocol: P's batch 256 against the control's adaptive 384 (RTX 6000: 64 × 6), the whole 50k split, and crop+flip (+18 % per epoch; 21729556 vs 21729555 on the same node at batch 256).
  - ~~At ~2.3×, the 6-day runtime fuse stops training near episode ~160~~ (corrected 30 Sep 11:45, below).

**Fuse and resume (30 Sep 11:45).**
- *Pace.* 12 episodes by the update-3 line (10:58): median 1,296 s, mean 2,315 s per episode (20–114 steps each), plus a probe every 12 episodes.
- *Fuse.* The 518,400 s fuse counts from 03:14:51, so it fires **~6 Oct 03:15**, near **episode ~200**. The earlier "~10 Oct" and "~160" were wrong.
- *Resume.* Ido GO 11:08: let the train run its course past the fuse. Resume **21767188** (`v9c-paug-area-train-r1`, same profile, env and tree) is chained `afterok:21737123`, nice 0, any RTX 6000 / 4090.
  - It restores weights, both optimisers, the episode index, the standardizer and the governor's best probe score and since-improvement count (`load_train_resume`); only the rewind count resets.
  - So the stop rule runs on unchanged: episode ≥ 250 **and** 150 episodes since the best probe. The resume's own fuse is 6 days.
- *Requeue.* The cluster requeues by default (`JobRequeue=1`, `PreemptMode=REQUEUE`). A requeue of 21737123 under its own id would run the sbatch "always cold" block, which deletes its `train_resume.pt`. Set `Requeue=0` on 21737123 and 21767188. Bundle backed up to `~/spectra_backups/job21737123_20260930`; the ops heartbeat keeps one copy a day.

Do **not** lock.

---

## 152. Crop+flip in the TEST walk: the three twins (**21729553**, **21737104** vs **21726337**) and the thin guard (**21729557** vs **21726335**) — PRELIM; decision (d) met, twins 3/3

`tree_v9b`, P, 2-pass mild (τ = 10, walk FT 40/10), `SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7`. The arm adds `SPECTRA_FT_AUG=1`; control = the P twins **21726337**. Net: `resnet56_cifar10_chenyaofo_94.37` (unpruned TEST half 0.943). Mild takes the same step at the same widths in both arms, so every row compares equal architectures. TEST = the 5k half; 10k = both halves, size points and cross-fit only. Arm on `cs-1080-05`; control on another card: re-walk noise is up to 0.8 pp (§149).

| Point | Step | Params / FLOPs kept | P TEST | P + aug TEST | Aug − P | 10k: P → aug |
|---|---|---|---|---|---|---|
| size 0.80 | 77 | 0.794 / 0.737 | −2.68 | **−0.16** | **+2.52** | −2.85 → **−0.37** |
| size 0.70 | 102 | 0.694 / 0.676 | −2.70 | **−0.50** | **+2.20** | −3.04 → **−0.63** |
| `floor_hold` | 100 | 0.701 / 0.679 | −3.2 | −0.3 | +2.9 | — |
| `val_best` | 112 | 0.661 / 0.662 | −2.84 | **−0.06** | **+2.78** | cross-fit τ10: −3.03 → **−0.51** |

- *Paired val.* 60 cuts, arm − control +2.08 pp, arm better on 98 % (`ADOPT?`: candidate only). Last step 112: −0.96 vs −3.22.
- *Census (aug).* 62 cut points: val Δ > 0 on **0**, TEST Δ > 0 on 6, both on 0; max val Δ −0.12 pp. Control: 0 / 0 / 0, max val −1.42. The cubic's positive branch is still not reached on val, but aug moves the best cut from −1.42 to −0.12 (O42 / N10 stays untriggered).
- *Reading.* On a full-width C10 net, crop+flip in the walk FT removes almost all of the walk's accuracy cost at 1.26–1.51× parameter compression. This is the same-loop (bar-2) recipe, **without** the 100-ep final FT. The literature rows use a long final FT, and DepGraph R56 is +0.24 at 2.11× (§147). Do not call this a match: the compression is lower and the recipes differ.
- *Stopped.* Scancelled 03:49 after the R56 rows (pre-registered, ops §9.3). Its VGG-16 was at step 1 and could not finish before the 08:21 wall. **21737104** re-walks both VGG twins on `tree_v9c`.
- *TEST-walk rule (O1, way-ahead decision d).* This covers R56 only. Still pending: the VGG twins (21737104) and the thin guard at 40/10 (21729557).

**VGG-16 twin (30 Sep 11:45; 21737104 R on `tree_v9c`, `cs-4090-07`, VGG-19 C100 walking).** Same design and control (21726337), `vgg16_bn_cifar10_chenyaofo_94.16` (unpruned TEST half 0.937). The arm ran on a 4090 and the control on another card.

| Point | Step | Params kept | P TEST | P + aug TEST | Aug − P |
|---|---|---|---|---|---|
| size 0.80 | 20 | 0.796 | −2.9 | **−0.3** | **+2.6** |
| size 0.70 | 25 | 0.698 | −2.8 | **−0.3** | **+2.5** |
| `val_best` | 29 | 0.657 | −2.8 | **−0.5** | **+2.3** |

- *Paired val.* 28 cuts, +2.44 pp, arm better on 100 %. The VGG-19 C100 twin: 7 cuts so far, +3.21 pp, 100 %.
- *TEST-walk rule (d), 11:45.* **Two of three twins** are ≥ 1 pp kinder at equal keep (R56 +2.2 to +2.8; VGG-16 +2.3 to +2.6). The twin half of the rule is met.
- *Thin guard (21729557 vs 21726335, 40/10).*
  - r20-w2: aug is 1.3 / 0.7 / 1.6 pp worse at size 0.80 / 0.60 / `val_best`, all at equal keep. That is inside its 2 pp guard.
  - r56-w4 is still walking: paired val +4.99 over 54 cuts, better on 100 %. Its rows decide (d): not > 0.5 pp worse than 21726335 at equal keep = met.

**Thin guard, r56-w4 (30 Sep 11:55; 21729557 COMPLETED 4 h 15 m, `ise-4090-20`, 0 Tracebacks).**
- *The net.* `resnet56-width4_cifar10_thin-res-net_88.80`, unpruned TEST half 0.890.
- *The comparison.* TEST FT 40/10 against the P thin walk 21726335; same steps, same widths.

| Point | Step (P / aug) | Params kept (P / aug) | P TEST | P + aug TEST | Aug − P |
|---|---|---|---|---|---|
| size 0.80 | 47 / 47 | 0.795 / 0.795 | −7.6 | **−2.6** | **+5.0** |
| size 0.60 | NONE | — | — | — | — |
| `val_best` | 76 / 112 | 0.739 / **0.622** | −10.1 (val −9.52) | **−4.5** (val −4.34) | deeper by 0.117 keep **and** 5.6 pp kinder |

- *Against the training FT (§150).* The same walk with crop+flip at 12/4 (21729556) was −5.9 at 0.795; at 40/10 it is −2.6. With crop+flip, the longer TEST FT recovers another 3.3 pp.
- *(d) verdict.* The thin guard holds with a wide margin: r56-w4 is 5.0 pp kinder at equal keep, and r20-w2 is inside its 2 pp. With the twins at 2/3, **decision (d) is met** (runbook M3, 11:55).
- *Control.* 21729557 is now final, and it is the mild control for every Stage-4 freeze TEST (runbook §10.3 item 1).

**VGG-19 C100 twin (30 Sep 12:50; 21737104 COMPLETED 2 h 48 m, `cs-4090-07`, 0 Tracebacks).** Same design and control (21726337), `vgg19_bn_cifar100_chenyaofo_73.87` (unpruned TEST half 0.745).

| Point | Step | Params / FLOPs kept | P TEST | P + aug TEST | Aug − P |
|---|---|---|---|---|---|
| size 0.80 | 24 | 0.796 / 0.745 | −6.4 | **−2.6** | **+3.8** |
| size 0.70 | 31 | 0.688 / 0.679 | −6.8 | **−1.9** | **+4.9** |
| `floor_hold` | 30 | 0.705 / 0.682 | −6.5 | −2.2 | +4.3 |
| `val_best` | 35 | 0.657 / 0.673 | −6.7 | **−2.5** | **+4.2** |

- *Paired val.* 34 cuts, +4.39 pp, arm better on 100 %. Last step 35: −1.24 vs −6.44.
- *Twins, final.* **Three of three** are ≥ 1 pp kinder at equal keep: R56 C10 +2.2 to +2.8, VGG-16 C10 +2.3 to +2.6, VGG-19 C100 +3.8 to +4.9. The effect is largest on C100, where the no-aug walk FT loses 6–7 pp.
- *Reading.* Walk rows only, mild, no agent, no 100-epoch final FT: a same-loop recipe fix, not a policy result. On C100 it also bounds how much of the old walk cost was the recipe: at 0.66–0.80 kept the twin's TEST cost falls from ~6.5 pp to ~2.2 pp.
- *Conversion (Ido GO 12:34).* 21730506 (no-aug walk + final FT, R56 + VGG-16 C10) was cancelled while PD at 12:47 and resubmitted as **21809595** `v9c-aug-ft100-twins-c10`: the same line + `SPECTRA_FT_AUG=1`, nice 40, `tree_v9c`. Its walk must reproduce 21737104 (VGG-16) and 21729553 (R56) by step, ≈ 0.

Do **not** lock.

---

## 153. 100-epoch SGD final fine-tune, DepGraph ResNet-56 C10 (**21730500** COMPLETED) — PRELIM; O2 honest gain HOLD

`tree_v9c`, P, 5-pass mild (τ = 10, walk FT 40/10, no aug), `SPECTRA_EVAL_SIZE_POINTS=flop:0.6,0.47,0.39` (0.47 and 0.39 are DepGraph's 2.11× and 2.57×). Then the §149 final FT: SGD 0.01, momentum 0.9, wd 5e-4, cosine, crop+flip, batch 128, 100 epochs, no KD, from the inherited weights, plus the origin control. `SAVE_TRAJ_MODELS=1`: the candidates are saved in `runs/job21730500/traj_models` (N1, N2 and scratch 21730516 start from them). COMPLETED 6 h 07 m, 30 Sep ~09:57, `ise-4090-18`. Checkpoint `resnet56_cifar10_dep_graph_93.53.pth`; TEST = the **5k half** (unpruned 0.934); 10k = both halves. Readers: `readers_s30/scripts/final_ft_readout.py` (ORIGIN-HURT fix) and `crossfit_readout.py`.

| Point | Step | Params / FLOPs kept | Walk (5k) | Final FT (5k) | Honest gain | 10k: walk → final | DepGraph published (10k) |
|---|---|---|---|---|---|---|---|
| size_flop0.60 | 136 | 0.638 / 0.599 | −2.50 | **−0.90** | +1.18 HOLD | −2.67 → **−1.06** | — |
| size_flop0.47 | 210 | 0.470 / 0.463 | −3.66 | **−1.44** | +1.80 HOLD | −3.81 → **−1.52** | **+0.24** at 2.11× |
| size_flop0.39 | 267 | 0.382 / 0.380 | −4.34 | **−2.18** | +1.74 HOLD | −4.60 → **−2.11** | **+0.11** at 2.57× |
| `val_best` | 283 | 0.356 / 0.369 | −3.88 | −1.94 | +1.52 HOLD | cross-fit τ10 walk −4.25; final n/a (val-selected) | — |
| origin | — | 1 / 1 | 0 | +0.42 | origin change +0.42 | +0.49 | — |

- *Re-walk determinism* vs 21726340 (same checkpoint and walk, `tree_v9b`): 150 paired cuts, mean −0.04 pp. ≈ 0, as required.
- *Census.* 152 cut points, val Δ > 0 on 0 (max −0.84): no-aug walk.
- *Reading.*
  - The final FT recovers **+1.2 to +1.8 pp beyond what it gives the unpruned net**. That is real, but under the 2 pp ADOPT line (HOLD). The VGG-19 C100 cell (§149) was +4.1 to +5.5.
  - At DepGraph's FLOPs points the 10k gap is **1.8 pp at 2.11× and 2.2 pp at 2.57×**. DepGraph learns sparsity and fine-tunes on this checkpoint. This row is a no-agent mild walk with the no-aug walk FT, then a 100-epoch SGD.
  - Quote beside DepGraph, never "beats" or "matches". Whether the crop+flip walk closes part of the gap is **N3 = 21767189** (this line + `SPECTRA_FT_AUG=1`).
- *Triggered.* N1 (KD, **21767190**) and N2 (AutoAugment, **21767192**) run from these saves: the pre-registered condition was honest ≥ 0.5 pp and no ORIGIN-HURT, and the origin change is +0.42.

Do **not** lock.

---

## 154. 100-epoch SGD final fine-tune, P thin, no-aug walk (**21730501** COMPLETED) — PRELIM; split: r20 CROSS-OFF, r56-w4 ADOPT

`tree_v9c`, P, 2-pass mild, walk FT 40/10 **no aug**, then the §149 final FT + origin, `SAVE_TRAJ_MODELS=1`. COMPLETED 5 h 01 m, 30 Sep 13:22, `ise-4090-21`. TEST = 5k half. Reader: `final_ft_readout.py`. Re-walk vs §143: paired mean −0.06 pp (r20) / −0.51 pp (r56-w4).

| Net | Point | Keep | Walk (5k) | Final (5k) | Honest | 10k final | Verdict |
|---|---|---|---|---|---|---|---|
| r20-w2 (0.649) | size 0.80 / 0.60 / `val_best` | 0.774 / 0.584 / 0.536 | +0.42 / −3.64 / −3.50 | +0.10 / −3.94 / −4.54 | **−3.78 / −3.76 / −4.50** | +0.99 / −3.43 / n/a | **CROSS-OFF** |
| r20-w2 | origin | 1 | 0 | **+3.46** | origin +3.46 | +3.83 | origin ran away |
| r56-w4 (0.890) | size 0.80 / `val_best` | 0.795 / 0.756 | −8.28 / −8.46 | −2.72 / −2.64 | **+5.26 / +5.52** | −2.85 / n/a | **ADOPT** |
| r56-w4 | origin | 1 | 0 | +0.30 | +0.30 | +0.25 | healthy |

**Read.** On the 5k-param r20-w2 the unpruned net gains **+3.5 pp** from the long SGD; the pruned points do not, so honest is −3.8 to −4.5 (CROSS-OFF). On r56-w4 the origin moves +0.30 and the pruned points recover **+5.3 to +5.5 pp** honest (ADOPT ≥ 2). Do not average the two nets into one cell verdict. Scratch-B **21730507** starts from these saves. Do **not** lock. Do not edit the draft.

---

## 155. N4 — crop+flip walk + 100-ep final FT, DepGraph VGG-19 C100 (**21737105** COMPLETED) — PRELIM; bar-3 CROSS-OFF vs §149

`tree_v9c`, P + `SPECTRA_FT_AUG=1`, 3-pass mild, size param 0.70 / 0.60, then the same final FT + origin as §149. COMPLETED 2 h 43 m, 30 Sep 15:28, `cs-4090-07`. Pair: no-aug final-FT cell **21729551** (§149). Checkpoint `vgg19_cifar100_dep_graph_73.5.pth`. TEST = 5k half (unpruned 0.740).

| Point | Keep | Walk (5k) | Final (5k) | Honest | 10k final | §149 10k final |
|---|---|---|---|---|---|---|
| size 0.70 step 29 | 0.684 / 0.686 | −2.24 | −2.24 | **−0.50 CROSS-OFF** | **−1.62** | −2.39 |
| size 0.60 step 42 | 0.599 / 0.590 | −2.52 | −2.94 | **−0.92 CROSS-OFF** | **−2.97** | −3.04 |
| `val_best` step 47 | 0.534 / 0.550 | −2.16 | −2.60 | **−0.94 CROSS-OFF** | n/a | n/a |
| origin | 1 | 0 | +0.50 | +0.50 | +0.96 | +0.56 |

Paired **walk** vs 21729551: 45 cuts, mean val **+4.48 pp**, better on 98 %. Census: 46 cuts, **val Δ > 0 on 2** (max +0.28); TEST Δ > 0 on 0. Cross-fit 10k τ10 mean **−2.11** @ 0.534.

**Read.** The aug walk is much kinder. The 100-ep finish **does not add** ≥ 0.5 pp honest (CROSS-OFF). 10k at 0.684 is −1.62 vs §149 −2.39 (**+0.77 pp**, under the 1 pp N4 adopt line). Bar-3 VGG-19 stays on §149's no-aug-walk + final FT unless a later recipe moves it. Quote DepGraph −3.11 at 8.92× beside; **never "beats"**. **M6** letter: two full-width cuts with val Δ > 0 (tiny). Do **not** lock. Do not edit the draft.

---

## 156. C-G NEON-rule under P — KILL, scancelled (**21730509** twins, **21730514** thin)

`tree_v9c`, NEON-literal C-G redraw, train-loss stop (patience 10, cap 100), P. Big-effect kill (runbook §10.3): ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse. **Scancelled 30 Sep 19:12** while R (`ise-4090-18` / `cs-4090-07`). Never a TEST of C-G working.

| Arm | vs | Pairs | Mean arm−control val | Arm better | Last step |
|---|---|---|---|---|---|
| 21730509 R56 | 21726337 | 60 | **−30.72 pp** | 2 % | 112: −40.40 vs −3.22 |
| 21730509 VGG-16 | 21726337 | 25 | **−11.58 pp** | 0 % | 26: −14.02 vs −2.56 |
| 21730514 r20-w2 | 21726335 | 16 | **−27.84 pp** | 0 % | 40: −23.88 vs −1.42 |
| 21730514 r56-w4 | 21726335 | 28 | **−54.00 pp** | 4 % | 51: −58.40 vs −7.96 |

Slots went to **21809595** and **21814029**.

**Read.** Under clean val, C-G with NEON's stop is not an in-band competitor. Do not resubmit. Do **not** lock.

---

## 157. N3 — crop+flip walk + 100-ep final FT, DepGraph ResNet-56 C10 (**21767189** COMPLETED) — PRELIM; **M4 fired** at 2.11×; final FT CROSS-OFF

`tree_v9c`, P + `SPECTRA_FT_AUG=1`, 5-pass mild, size `flop:0.60,0.47,0.39`, then the same 100-ep SGD final FT + origin as §153. COMPLETED 10 h 27 m, 30 Sep ~22:40, `ise-4090-19`. Pair: no-aug final-FT cell **21730500** (§153). Checkpoint `resnet56_cifar10_dep_graph_93.53.pth`. TEST = 5k half (unpruned 0.934). Reader: `final_ft_readout.py`. Census: 152 cuts, **val Δ > 0 on 70** (max +0.86); TEST Δ > 0 on 53.

| Point | Params / FLOPs | Walk (5k) | Final (5k) | Honest | 10k final | §153 10k final | DepGraph |
|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | **+0.08** | +0.04 | **−0.40 CROSS-OFF** | **−0.03** | −1.06 | — |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | **−0.22** | −0.44 | **−0.58 CROSS-OFF** | **−0.46** | −1.52 | **+0.24** |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | **−1.32** | −1.34 | **−0.38 CROSS-OFF** | **−1.63** | −2.11 | **+0.11** |
| `val_best` | 0.356 / 0.369 | −1.12 | −0.96 | −0.20 CROSS-OFF | n/a | n/a | — |
| origin | 1 | 0 | +0.36 | +0.36 | +0.60 | +0.49 | — |

Paired walk vs 21730500: 150 cuts, mean val **+2.68 pp**, better on 99 %.

**Read.** The crop+flip **walk** is the lever. 10k at DepGraph's 2.11× is **−0.46** vs their **+0.24** (**0.70 pp** behind) → **M4**. At 2.57×, 10k **−1.63** vs +0.11 (1.74 pp behind). The 100-ep finish does not add honest gain (CROSS-OFF), same pattern as N4 §155. Bar-3 ResNet-56 rows use the **crop+flip walk**; caption the final FT as protocol alignment that did not move this cell. Quote beside DepGraph; **never "beats"**. Do **not** lock. Do not edit the draft.

---

## 158. Scratch-B, P thin from 21730501 saves (**21730507** COMPLETED) — PRELIM; CROSS-OFF / ORIGIN-HURT

`tree_v9c`, from-saved, `SPECTRA_EVAL_FINAL_FT_SCRATCH`, 200-ep SGD 0.1 + crop+flip. COMPLETED 2 h 43 m, `ise-4090-14`. Control = inherit final FT §154.

| Net | Point | Keep | Walk (inherit) | Scratch final | Honest | Origin scratch | Verdict |
|---|---|---|---|---|---|---|---|
| r20-w2 | size 0.80 / 0.60 / `val_best` | 0.774 / 0.584 / 0.536 | +0.42 / −3.64 / −3.50 | +0.74 / −4.44 / −5.48 | **−4.78 / −5.90 / −7.08** | **+5.10** | **CROSS-OFF** |
| r56-w4 | size 0.80 / `val_best` | 0.795 / 0.756 | −8.28 / −8.46 | −3.20 / −3.50 | +6.24 / +6.12 printed | **−1.16** | **ORIGIN-HURT** |

**Read.** Liu et al. scratch-B does not recover the thin pair: the 5k-param origin runs away (+5.1), and the r56-w4 origin **loses** 1.16 pp so the printed honest is invalid. Do not use scratch-B as a thin-net architecture metric. Do **not** lock.

---

## 159. Scratch-B, DepGraph ResNet-56 from 21730500 saves (**21730516** COMPLETED) — PRELIM; ADOPT at 2.11× / 2.57×

Same recipe as §158, from the no-aug walk's saved architectures. COMPLETED 1 h 54 m, `ise-6000-07`. Origin change **+1.20** (healthy).

| Point | Params / FLOPs | Inherit walk (5k) | Scratch final (5k) | Honest | 10k scratch |
|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | −2.50 | +0.08 | +1.38 HOLD | −0.26 |
| size_flop0.47 | 0.470 / 0.463 | −3.66 | **+0.20** | **+2.66 ADOPT** | **−0.16** |
| size_flop0.39 | 0.382 / 0.380 | −4.34 | −0.50 | **+2.64 ADOPT** | **−0.56** |
| `val_best` | 0.356 / 0.369 | −3.88 | −0.98 | +1.70 HOLD | n/a |

**Read.** On DepGraph's ResNet-56, training the pruned **architecture** from scratch for 200 ep matches or beats inherit+long-FT, and 10k **−0.16 at 2.11×** is 0.40 pp from DepGraph's +0.24. This is Liu et al.'s network-level "fresh weights", not C-G. N3's crop+flip **walk** (10k −0.40 at the same size) is in the same band without throwing the weights away. Do **not** lock.

---

## 160. N1 — KD in the 100-ep final FT, from 21730500 saves (**21767190** COMPLETED) — PRELIM; mixed vs §153; does not beat N3's walk

`tree_v9c`, `SPECTRA_FT_KD=1` + `EVAL_FINAL_FT_KD=1`, from-saved, no new walk. COMPLETED 1 h 20 m, `ise-6000-07`. Origin +0.38 (healthy). M5 bar: ≥ +0.5 pp over §153's plain final FT.

| Point | N1 final (5k) | §153 final (5k) | Δ vs plain FT | 10k N1 | 10k §153 |
|---|---|---|---|---|---|
| size_flop0.60 | **−0.28** | −0.90 | **+0.62** | −0.18 | −1.06 |
| size_flop0.47 | −1.38 | −1.44 | +0.06 | −1.45 | −1.52 |
| size_flop0.39 | −1.92 | −2.18 | +0.26 | −1.66 | −2.11 |
| origin | +0.38 | +0.42 | — | +0.44 | +0.49 |

**Read.** KD helps the **easy** size (0.60) by 0.62 pp and is noise at DepGraph's two FLOPs points. It does **not** beat N3's crop+flip walk (10k −0.46 at 2.11× vs N1 −1.45). Do not fire M5 as a global adopt. Do **not** lock.

---

## 161. N2 — AutoAugment in the 100-ep final FT, from 21730500 saves (**21767192** COMPLETED) — PRELIM; CROSS-OFF / HOLD; origin +1.24

Same from-saved walk as N1, `SPECTRA_FT_AUTOAUG=1`. COMPLETED 1 h 34 m, `cs-4090-08`. Origin **+1.24** (the recipe trains the unpruned net harder). Honest +0.36 to +0.88. 10k at 2.11× **−1.71** (worse than §153 −1.52). **Not M5.** Do **not** lock.

---

## 162. N2-streams — 3-pass mild, residual streams protected, under P (**21729558** COMPLETED) — PRELIM; split

`tree_v9b`, P, no crop+flip, `SPECTRA_PROTECT_STREAMS=1`, 3 passes, vs P thin **21726335** by params. COMPLETED 3 h 37 m, `ise-6000-07`.

| Net | `val_best` TEST | size 0.80 TEST | vs §152 aug thin at ~0.80 |
|---|---|---|---|
| r20-w2 | **−0.20 @ 0.646** (10k mean +0.41) | **+1.14 @ 0.798** | kinder; shallower keep |
| r56-w4 | −7.64 @ 0.719 | **−6.74 @ 0.800** | worse than aug thin **−2.6 @ 0.795** |

Paired by params vs 21726335: r20 mean **+1.50 pp** (ADOPT?); r56-w4 mean **+2.10 pp** vs **no-aug** P. Against the live **aug** TEST walk, r56-w4 is not deeper in band (0.719 vs 0.622) and is worse at equal ~0.80. Do not replace the TEST walk. Do **not** lock.

---

## 163. C-G+ under P + crop+flip, thin pair — KILL, scancelled (**21938280**)

`tree_v9d`, recipe **C-G+** (layer replacement + 0.1× polish), P + crop+flip, 2-pass mild, vs aug thin 40/10 **21729557**. Big-effect kill (runbook §10.3): ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse. **Scancelled 1 Oct 01:45** while R (`ise-4090-15`), still on r20-w2 (r56-w4 never started). Never a TEST of C-G+ working under P.

| Arm | vs | Pairs | Mean arm−control val | Arm better | Last step |
|---|---|---|---|---|---|
| 21938280 r20-w2 | 21729557 | 5 | **−20.82 pp** | 0 % | 14: −21.44 vs +2.36 |

**Read.** Crop+flip does not rescue NEON layer replacement: C-G+ on the same mild walk is ~21 pp worse than recipe A at five paired cuts. Close C-G+ under P, same as C-G §156. Do not resubmit. Do **not** lock.

---

## 164. Crop+flip walk + 100-ep final FT, zoo twins C10 (**21809595** COMPLETED) — PRELIM; walk ≈ §152; long FT CROSS-OFF

`tree_v9c`, P + `SPECTRA_FT_AUG=1`, 2-pass mild, size `param:0.8,0.7`, then 100-ep SGD final FT + origin. COMPLETED 6 h 34 m, 1 Oct 01:46, `cs-4090-07`. Conversion of cancelled 21730506 (Ido GO 12:34). Walk must ≈ 0 vs R56 **21729553** and VGG-16 **21737104**. TEST = 5k half. Reader: `final_ft_readout.py`.

| Net | Point | Keep | Walk (5k) | Final (5k) | Honest | 10k final | §152 walk |
|---|---|---|---|---|---|---|---|
| R56 zoo | size 0.80 | 0.794 / 0.737 | −0.40 | −0.38 | **+0.02 CROSS-OFF** | **−0.39** | −0.16 |
| R56 zoo | size 0.70 | 0.694 / 0.676 | −0.36 | −0.62 | **−0.26 CROSS-OFF** | **−0.76** | −0.50 |
| R56 zoo | `val_best` | 0.661 / 0.662 | −0.22 | −0.20 | **+0.02 CROSS-OFF** | n/a | −0.06 |
| R56 zoo | origin | 1 | 0 | +0.00 | +0.00 | −0.01 | — |
| VGG-16 zoo | size 0.80 | 0.796 / 0.747 | −0.10 | −0.10 | **−0.88 CROSS-OFF** | **+0.00** | −0.3 |
| VGG-16 zoo | size 0.70 | 0.698 / 0.685 | −0.70 | −0.20 | **−0.38 CROSS-OFF** | **+0.01** | −0.3 |
| VGG-16 zoo | `val_best` | 0.657 / 0.678 | −0.40 | +0.04 | **−0.44 CROSS-OFF** | n/a | −0.5 |
| VGG-16 zoo | origin | 1 | 0 | +0.88 | +0.88 | +0.55 | — |

**Read.** Re-walk is inside noise of the crop+flip twins. The 100-ep finish adds nothing on zoo ResNet-56 (honest ~0) and is CROSS-OFF on VGG-16 because the origin gains +0.88. Same pattern as N3/N4: the walk is the lever. Bar-3 zoo C10 rows stay on the crop+flip **walk**. Do **not** lock.

---

## 165. 10-pass mild crop+flip + 100-ep FT, VGG-16 C10 to HRank / OCS FLOPs (**21814029** COMPLETED) — PRELIM; long FT CROSS-OFF

`tree_v9c`, P + `SPECTRA_FT_AUG=1`, **10-pass** mild (default L1 ranking; job name `l2-vgg16` is the literature size, not `FILTER_IMPORTANCE=l2`), size `flop:0.465,0.212`, then 100-ep SGD + origin. COMPLETED 6 h 48 m, 1 Oct 02:00, `ise-4090-18`. Checkpoint `vgg16_bn_cifar10_chenyaofo_94.16`. TEST = 5k half. Reader: `final_ft_readout.py`.

| Point | Step | Params / FLOPs | Walk (5k) | Final (5k) | Honest | 10k final | Published (own base) |
|---|---|---|---|---|---|---|---|
| size_flop0.47 (HRank 46.5 %) | 56 | 0.444 / 0.464 | −0.60 | −0.44 | **−0.74 CROSS-OFF** | **−0.25** | HRank **−0.53** at 46.5 % FLOPs / **17.1 % params** |
| size_flop0.21 (OCS 21.2 %) | 122 | 0.187 / 0.211 | −2.50 | −2.36 | **−0.76 CROSS-OFF** | **−2.02** | OCS pretrained **−0.44** at 21.2 % FLOPs / **13.7 % params** |
| `val_best` | 149 | 0.123 / 0.154 | −2.36 | −2.32 | **−0.86 CROSS-OFF** | n/a | — |
| origin | — | 1 | 0 | +0.90 | +0.90 | +0.67 | — |

**Read.** Mild L1 at HRank's FLOPs keeps **44 % params** vs their 17 % — not a param-matched row. At OCS's FLOPs we keep 19 % params vs 14 %, 10k **−2.02** vs their **−0.44**. Never "beats". The 100-ep finish is CROSS-OFF (origin +0.90). Quote 10k walk-adjacent finals beside HRank/OCS; caption the param mismatch. Do **not** lock.

---

## 166. Adam 1e-4 vs Adam 1e-3, thin C10, P + crop+flip 12/4 (**21938281** COMPLETED) — PRELIM; CROSS-OFF as train FT

`tree_v9d`, no-agent mild, recipe A, P + `SPECTRA_FT_AUG=1`, 12/4, `SPECTRA_FT_LR=1e-4`. COMPLETED 1 h 19 m, 1 Oct 02:34, `ise-4090-16`. Control = aug thin 12/4 **21729556** (§150, Adam 1e-3). Pair rule (G2 sitting): within 0.5 pp of 556 at equal keep. TEST = 5k half. Do not quote terminal over τ.

| Net | Point | Keep | Adam 1e-4 TEST | §150 1e-3 TEST | Δ vs 556 |
|---|---|---|---|---|---|
| r56-w4 | size 0.80 | 0.795 | **−2.8** | −5.9 | **+3.1** |
| r56-w4 | `val_best` | 0.622 | −5.6 | −5.1 | −0.5 |
| r20-w2 | size 0.80 | 0.774 | **−8.5** | −3.2 | **−5.3** |
| r20-w2 | size 0.60 | 0.584 | **−20.7** | −6.2 | **−14.5** |
| r20-w2 | `val_best` | 0.736 / 0.536 | −9.0 | −6.6 | unmatched keep |

Paired val vs 556: r20 16 cuts mean **−7.41 pp**, 0 % better (KILL label); r56 60 cuts **+0.36 pp**, 47 % better.

**Read.** Lower Adam lr is kinder on skinny ResNet-56 at 80 % kept and a wash at `val_best`, but it **destroys** r20-w2 at equal keep. Fails the 0.5 pp thin rule. Cap-40 stays crossed. Live train stays Adam **1e-3**. Do **not** lock.

---

## 167. SGD 0.01 vs Adam 1e-3, thin C10, P + crop+flip 12/4 (**21938282** COMPLETED) — PRELIM; CROSS-OFF as train FT

Same walk as §166, `SPECTRA_FT_OPTIM=sgd SPECTRA_FT_SGD_LR=0.01` (momentum, wd 5e-4). COMPLETED 1 h 18 m, 1 Oct 02:32, `ise-4090-19`. Control **21729556**.

| Net | Point | Keep | SGD 0.01 TEST | §150 Adam 1e-3 | Δ vs 556 |
|---|---|---|---|---|---|
| r56-w4 | size 0.80 | 0.795 | −6.6 | −5.9 | **−0.7** |
| r56-w4 | `val_best` | 0.622 | −7.3 | −5.1 | **−2.2** |
| r20-w2 | size 0.80 | 0.774 | −3.0 | −3.2 | +0.2 |
| r20-w2 | size 0.60 | 0.584 | −9.0 | −6.2 | **−2.8** |
| r20-w2 | `val_best` | 0.655 / 0.536 | −5.4 | −6.6 | unmatched keep |

Paired val vs 556: r20 mean **−3.87 pp**; r56 60 cuts **−1.37 pp**, 28 % better.

**Read.** SGD 0.01 fails the r56-w4 0.5 pp rule (−0.7 / −2.2). Crop+flip does not make SGD a train-FT replacement. Do **not** lock.

---

## 168. Adam 1e-4, C100 tight-2, P + crop+flip 12/4 (**21938283** COMPLETED) — PRELIM; kinder than §148 on both nets; not a train switch

`tree_v9d`, r20-w13 + r56-w9 (the two §148 nets that sat nearest τ). COMPLETED 1 h 28 m, 1 Oct 02:43, `ise-4090-20`. Control = aug C100 gate **21729554** (Adam 1e-3). TEST = 5k half.

| Net | Point | Keep | Adam 1e-4 TEST | §148 1e-3 TEST | Δ vs 554 |
|---|---|---|---|---|---|
| r20-w13 | size 0.90 | 0.861 | −3.9 | −5.2 | **+1.3** |
| r20-w13 | size 0.80 | 0.787 | −4.8 | −6.9 | **+2.1** |
| r20-w13 | `val_best` | 0.662 | −7.8 | −8.4 | **+0.6** |
| r56-w9 | size 0.90 | 0.860 | −6.6 | −7.7 | **+1.1** |
| r56-w9 | size 0.80 | 0.792 | −5.8 | −9.5 | **+3.7** |
| r56-w9 | `val_best` | 0.642 / 0.647 | −8.7 | −9.5 | ~+0.8 |

**Read.** On the two hardest C100 admitters, Adam 1e-4 is kinder than 1e-3 at every equal-keep size point. That does **not** clear the G2 pair rule: thin C10 §166 failed. SGD C100 t2 **21938284** still R. Do not retune the live train. Do **not** lock.

---

## 169. O26 memorization census: unpruned val at episode reset vs the TEST accuracy in the checkpoint name — zero GPU, not a TEST row

`scripts/memorization_census.py` (G2 sitting, 1 Oct 03:25) reads each run's `episode_reset` events (`baseline_acc` = the unpruned net on the env's val split) and compares the median with the accuracy encoded in the checkpoint name. MEMORIZED = val ≥ 0.995 or val − TEST ≥ +3 pp.

| Run | Val split | Nets | MEMORIZED | val − TEST | Detail |
|---|---|---|---|---|---|
| v3 train **21385158** | legacy (train-side) | 24 | **24 / 24** | **+3.35 to +7.19 pp** | 11 nets read val ≥ 0.999 (VGG-13, R56-w12, R56-w14 read 1.0000). The 9 nets below the 0.995 val bar (narrow thin ResNets, VGG-11 F-MNIST) are flagged on the gap alone: val 0.939–0.994, +3.4 to +5.6 pp |
| Stage-4 train **21737123** (P) | half of the test split | 10 | **0 / 10** | **−0.63 to +0.08 pp** | split noise; val sits slightly below the full-test number on 8 / 10 |

**Read.** This quantifies §141 across a full train catalog. Every legacy train scored every cut against a val split that read 3–7 pp above TEST, so a cut that only cost memorization looked like an accuracy loss. Under P, the reward and gates see the TEST-level accuracy on every net. This supports re-opening only the cells whose kill used legacy val (G2 charge); it does not by itself re-open any cell. The checkpoint-name accuracy is the pretraining's best test epoch, so it can sit a little above the last epoch.

---

## 170. SGD 0.01 vs Adam 1e-3, C100 tight-2, P + crop+flip 12/4 (**21938284** COMPLETED) — PRELIM; harsher on 3 of 4 equal-keep points; CROSS-OFF as train FT

`tree_v9d`, r20-w13 + r56-w9 (the §168 pair). COMPLETED 1 h 26 m, 1 Oct 03:26. Control = aug C100 gate **21729554** (Adam 1e-3); the Adam 1e-4 column is §168. TEST = 5k half.

| Net | Point | Keep | SGD 0.01 TEST | 554 Adam 1e-3 TEST | Δ vs 554 | §168 Adam 1e-4 |
|---|---|---|---|---|---|---|
| r20-w13 | size 0.90 | 0.861 | −7.2 | −5.2 | **−2.0** | −3.9 |
| r20-w13 | size 0.80 | 0.787 | −8.4 | −6.9 | **−1.5** | −4.8 |
| r20-w13 | `val_best` | 0.762 (554: 0.662) | −8.5 | −8.4 | different keep | −7.8 @ 0.662 |
| r56-w9 | size 0.90 | 0.860 | −12.6 | −7.7 | **−4.9** | −6.6 |
| r56-w9 | size 0.80 | 0.792 | −9.2 | −9.5 | +0.3 (noise) | −5.8 |
| r56-w9 | `val_best` | 0.757 (554: 0.642) | −10.8 | −9.5 | different keep | −8.7 @ 0.642 / 0.647 |

**Read.** SGD 0.01 is harsher than Adam 1e-3 at three of the four equal-keep points (−1.5 to −4.9 pp); the fourth is inside the ~0.8 pp re-walk noise. With the thin failure (§167), SGD 0.01 is CROSS-OFF as the train FT. Neither A3 arm passed the pair rule (Adam 1e-4 fails thin, §166), so cap-40 stays crossed. Do **not** lock.

---

## 171. F1 cosine vs plateau, thin C10, P + crop+flip 12/4 (**21938286** COMPLETED) — PRELIM; r56 kinder, r20 deep keep fails; CROSS-OFF as train FT

`tree_v9d`, `SPECTRA_FT_COSINE=1` (Adam 1e-3, 12/4, cosine instead of plateau). COMPLETED 1 h 21 m, 1 Oct 03:55, `ise-4090-16`. Control = aug thin 12/4 **21729556** (§150, plateau). Pair rule (G2 A5): within 0.5 pp on r20 **and** kinder on r56-w4. TEST = 5k half.

| Net | Point | Keep | Cosine TEST | §150 plateau | Δ vs 556 |
|---|---|---|---|---|---|
| r56-w4 | size 0.80 | 0.795 | **−2.6** | −5.9 | **+3.3** |
| r56-w4 | `val_best` | 0.622 | **−4.1** | −5.1 | **+1.0** |
| r20-w2 | size 0.80 | 0.774 | −2.8 | −3.2 | +0.4 |
| r20-w2 | size 0.60 | 0.584 | **−10.4** | −6.2 | **−4.2** |
| r20-w2 | `val_best` | 0.536 | −9.2 | −6.6 | **−2.6** |

Paired val vs 556: r20 16 cuts mean −0.69 pp; r56 60 cuts **+1.49 pp**, better on 93 % (`ADOPT?` on val only).

**Read.** Cosine is clearly kinder on skinny ResNet-56 at the train FT, and a wash at r20 80 % kept, but it **fails** r20 at 60 % kept by 4.2 pp. That misses the 0.5 pp r20 half of the pair rule (and the 2 pp guard). Do not put cosine into Stage-4. F2 is **§172**. Do **not** lock.

---

## 172. F2 group-first-4 vs full-net FT, thin C10, P + crop+flip 12/4 (**21938287** COMPLETED) — PRELIM; r56 kinder, r20 0.60 fails; CROSS-OFF as train FT

`tree_v9d`, `SPECTRA_FT_GROUP_FIRST_EPOCHS=4` (4 epochs on the edited group, then the usual 12/4 full-net). COMPLETED 1 h 40 m, 1 Oct 04:24, `ise-4090-20`. Control **21729556**. Same pair rule as §171. TEST = 5k half. FLAGS: `group_first=4`.

| Net | Point | Keep | Group-first TEST | §150 full-net | Δ vs 556 |
|---|---|---|---|---|---|
| r56-w4 | size 0.80 | 0.795 | **−3.6** | −5.9 | **+2.3** |
| r56-w4 | `val_best` | 0.622 | **−4.2** | −5.1 | **+0.9** |
| r20-w2 | size 0.80 | 0.774 | −2.6 | −3.2 | +0.6 |
| r20-w2 | size 0.60 | 0.584 | **−8.2** | −6.2 | **−2.0** |
| r20-w2 | `val_best` | 0.536 | −6.8 | −6.6 | −0.2 |

Paired val vs 556: r20 mean −0.79 pp; r56 60 cuts **−0.03 pp**, 50 % better (val wash; TEST still kinder at equal keep).

**Read.** Same split as cosine: skinny ResNet-56 likes the extra group recovery; skinny ResNet-20 at 60 % kept fails the 0.5 pp rule by 2.0 pp (on the 2 pp guard line). Do not put group-first into Stage-4. A5 both arms CROSS-OFF as train FT. Do **not** lock.

---

## 173. Greedy (L1 profile) vs mild, thin C10, P + crop+flip 40/10 (**21938279** COMPLETED) — PRELIM; not size-matched; CROSS-OFF as the bar-2 walk

`tree_v9d`, `SPECTRA_EVAL_POLICY` / profile `l1` (Gilad "greedy"), 2-pass, 40/10, P + aug. COMPLETED 4 h 10 m, 1 Oct 05:25, `cs-4090-08`. Control = mild **21729557** (§152). Greedy does not take the same widths at the same step — do not read `paired_steps.py` as equal-architecture. TEST = 5k half.

| Net | Point (first keep ≤ target) | Greedy keep / TEST | Mild keep / TEST |
|---|---|---|---|
| r20-w2 | size ~0.80 | **0.702 / −5.1** | 0.774 / −1.2 |
| r20-w2 | size ~0.60 | 0.595 / **−4.0** | 0.584 / −4.9 |
| r20-w2 | `val_best` | **0.417 / −8.2** | 0.536 / −5.3 |
| r56-w4 | size ~0.80 | **0.743 / −5.7** | 0.795 / −2.6 |
| r56-w4 | size ~0.60 | 0.600 / −5.9 | mild 2-pass stops at 0.622 (NONE) |
| r56-w4 | `val_best` | **0.389 / −7.8** | 0.622 / −4.5 |

**Read.** Greedy overshoots the named size points. At the one near-equal keep (r20 ~0.59) it is 0.9 pp kinder; everywhere else it is a smaller net with a larger TEST drop. It is not a milder bar-2 walk. Keep mild. Random r20 3-draw mean is **§174**; r56 3-draw mean is **§175**. Do **not** lock.

---

## 174. Random walk, r20-w2, three seeds, P + crop+flip 40/10 (**21938894 / 21938896 / 21938930** COMPLETED) — PRELIM; mean ≈ mild; not a better bar-2

`tree_v9d`, random policy, 2-pass, 40/10, P + aug. Seeds 42 / 43 / 44. Sitting rule: quote the **mean**, never the best draw. Control = mild **21729557**. TEST = 5k half. Keeps differ by draw — not an equal-architecture pair.

| Point | s42 keep / TEST | s43 | s44 | Mean TEST | Mild 557 |
|---|---|---|---|---|---|
| size ~0.80 | 0.782 / −0.9 | 0.774 / −0.7 | 0.774 / −1.5 | **−1.0** | −1.2 @ 0.774 |
| size ~0.60 | 0.599 / −3.6 | 0.560 / −5.1 | 0.584 / −4.1 | **−4.3** | −4.9 @ 0.584 |
| `val_best` | 0.471 / −6.7 | 0.511 / −5.2 | 0.536 / −4.5 | **−5.5** | −5.3 @ 0.536 |

r56-w4 three-seed mean is **§175**.

**Read.** On skinny ResNet-20, random's three-seed mean sits next to mild. It is the stochastic baseline, not a better walk. Do **not** lock.

---

## 175. Random walk, r56-w4, three seeds, P + crop+flip 40/10 (**21938285 / 21938895 / 21938929** COMPLETED) — PRELIM; mean harsher than mild; not a better bar-2

`tree_v9d`, random policy, 2-pass, 40/10, P + aug. Seeds 42 / 43 / 44. Sitting rule: quote the **mean**, never the best draw. Control = mild **21729557** (§152 / §173). TEST = 5k half. Keeps differ by draw — not an equal-architecture pair. s44 COMPLETED 08:11.

| Point | s42 keep / TEST | s43 | s44 | Mean TEST | Mild 557 |
|---|---|---|---|---|---|
| size ~0.80 | 0.770 / −4.5 | 0.758 / −6.0 | 0.796 / −4.8 | **−5.1** | −2.6 @ 0.795 |
| size ~0.60 | 0.510 / −7.7 | 0.491 / −8.0 | 0.590 / −6.0 | **−7.2** | 2-pass mild stops at 0.622 |
| `val_best` | 0.450 / −6.8 | 0.431 / −6.9 | 0.525 / −5.7 | **−6.5** | −4.5 @ 0.622 |

**Read.** On skinny ResNet-56, random's three-seed mean is 2.5 pp harsher than mild at ~80 % kept, and 2.0 pp harsher at the quoted point, while keeping less. With §174, random is the stochastic baseline, not a replacement walk. Keep mild as bar 2. Do **not** lock.

---

## 176. C-G, P + crop+flip, thin pair (**21940176 / 21940177** CANCELLED 09:20) — PRELIM; KILL; same deficit as §156

`tree_v9d`, NEON-literal C-G (train-loss stop, patience 10, cap 100) on the mild walk, 40/10, P + aug. Control = mild keep-the-survivors **21729557**. Paired val, same step = same widths. Pre-authorized kill: ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940176** | r20-w2 | 8 | **−27.2 pp** | 0/8 | arm −23.8 vs control −0.0 |
| **21940177** | r56-w4 | 7 | **−56.8 pp** | 1/7 | arm −68.5 vs control −1.6 |

Scancel 09:20 (CG). Full-width C-G **21940183 / 21940184** still PD — construction not crossed until ≥ 3 of 4 nets.

**Read.** Crop+flip does not close C-G on the diagnostic pair. Matches §156 (clean, no aug) and §163 (C-G+). Do **not** lock the construction off the thin pair alone.

---

## 177. C-G producers-only, P + crop+flip, thin pair (**21940178 / 21940179** CANCELLED 09:20) — PRELIM; KILL

`tree_v9d`, redraw the pruned layer only (same NEON-literal stop), 40/10, P + aug. Control **21729557**. Same kill rule.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940178** | r20-w2 | 9 | **−16.9 pp** | 0/9 | arm −19.4 vs control +0.3 |
| **21940179** | r56-w4 | 8 | **−57.2 pp** | 1/8 | arm −67.7 vs control −1.7 |

Scancel 09:20 (CG). Full-width **21940186 / 21940187** still PD.

**Read.** Producers-only is the same failure mode as full group redraw, under P + aug. Do **not** lock the construction off the thin pair alone.

---

## 178. C-PCA, P + crop+flip, r20-w2 (**21940180** COMPLETED) — PRELIM; harsher than mild at equal keep

`tree_v9d`, principal-direction replacement, 2-pass, 40/10, P + aug. COMPLETED 45 m, 1 Oct 09:19. Control **21729557**. Same keep at the named points (recovery recipe on the mild walk). TEST = 5k half. TB=0.

| Point | Keep | C-PCA TEST | Mild 557 | Δ vs mild |
|---|---|---|---|---|
| size 0.80 | 0.774 | **−5.9** | −1.2 | **−4.7** |
| size 0.60 | 0.584 | **−9.3** | −4.9 | **−4.4** |
| `val_best` | 0.536 | **−7.3** | −5.3 | **−2.0** |

Paired val at 08:52 (6 cuts): mean −1.4 pp, better on 50% — not a kill; the TEST still loses at every equal-keep point. r56-w4 / twins still PD (**21940181 / 88 / 89**). Re-open needs ≥ 3 of 4 nets within 0.5 pp or kinder, including one full-width.

**Read.** On skinny ResNet-20, C-PCA under P + aug is a worse recovery than keep-the-survivors. One net; construction still open. Do **not** lock.

---

## 179. C-PCA, P + crop+flip, r56-w4 (**21940181** COMPLETED) — PRELIM; harsher than mild at equal keep

`tree_v9d`, principal-direction replacement, 2-pass, 40/10, P + aug. COMPLETED 3 h 34 m, 1 Oct 12:55. Control **21729557**. Same keep at the named points. TEST = 5k half. TB=0. Size 0.60: NONE (2-pass mild also stops at 0.622).

| Point | Keep | C-PCA TEST | Mild 557 | Δ vs mild |
|---|---|---|---|---|
| size 0.80 | 0.795 | **−3.9** | −2.6 | **−1.3** |
| `val_best` | 0.622 | **−6.2** | −4.5 | **−1.7** |

Paired val through the morning never hit the kill (58 pairs, mean −1.38 pp). TEST still loses at both equal-keep points. With §178, C-PCA is worse on both thin nets. Twins **21940188 / 89** still PD. Cross-off needs ≥ 3 of 4.

**Read.** C-PCA under P + aug is not keep-the-survivors on the diagnostic pair. Construction still open until a full-width net. Do **not** lock.

---

## 180. C-G+, P + crop+flip, r56-w4 (**21940182** CANCELLED 13:50) — PRELIM; KILL; same deficit as §163

`tree_v9d`, NEON-literal C-G then 0.1× whole-network polish, 40/10, P + aug. Control = mild **21729557**. Paired val. Pre-authorized kill: ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940182** | r56-w4 | 10 | **−11.6 pp** | 1/10 | arm −7.3 vs control −1.7 |

Scancel 13:50. r20-w2 already **§163** KILL. Full-width **21940191 / 21940192** still PD — construction not crossed until ≥ 3 of 4 nets.

**Read.** Crop+flip plus the 0.1× polish does not close C-G+ on skinny ResNet-56. The diagnostic pair is now both KILL under P + aug. Do **not** lock the construction off the thin pair alone.

---

## 181. C-G, P + crop+flip, full-width ResNet-56 (**21940183** CANCELLED 18:22) — PRELIM; KILL; construction 3/4

`tree_v9d`, NEON-literal C-G (train-loss stop, patience 10, cap 100), 40/10, P + aug, `ft_recipe=C-G`. Control = mild twin **21809595**. Paired val. Pre-authorized kill: ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940183** | R56 C10 | 8 | **−34.4 pp** | 1/8 | arm −41.0 vs control −0.4 |

Scancel 18:22 (CG). Thin pair already **§176** KILL. VGG-16 **21940184** still PD at write time. C-G is now killed or worse on **3 of 4** nets → **CROSS-OFF** the construction (queue rule).

**Read.** Crop+flip does not close group redraw on full-width ResNet-56 either. Same failure as the thin pair and as §156 (clean, no aug).

---

## 182. C-G, P + crop+flip, VGG-16 (**21940184** CANCELLED 19:22) — PRELIM; KILL; construction 4/4

`tree_v9d`, same recipe as §181. Control = mild twin **21809595**. Same kill rule.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940184** | VGG-16 C10 | 14 | **−6.9 pp** | 0/14 | arm −6.0 vs control +0.2 |

Scancel 19:22 (CG). With §176 and §181 this is **4 of 4** nets. C-G under P + crop+flip is closed.

**Read.** Milder than the ResNet-56 collapse (−7 pp vs −34 pp) and still a clear miss vs keep-the-survivors. Do **not** start another C-G walk.

---

## 183. C-G producers-only, P + crop+flip, full-width ResNet-56 (**21940186** CANCELLED 21:22) — PRELIM; KILL; construction 3/4

`tree_v9d`, redraw the pruned layer only (NEON-literal stop), 40/10, P + aug. Control = mild twin **21809595**. Same kill rule.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940186** | R56 C10 | 20 | **−4.3 pp** | 1/20 | arm −8.7 vs control +0.1 |

Scancel 21:22 (CG). Thin pair already **§177** KILL. VGG-16 **21940187** still PD at write time. Producers-only is now killed on **3 of 4** nets → **CROSS-OFF**.

**Read.** Milder than full group redraw on the same net (§181 −34 pp) and still a miss vs keep-the-survivors. Crop+flip does not close the “pruned layer only” construction on full-width ResNet-56.

---

## 184. C-G producers-only, P + crop+flip, VGG-16 (**21940187** CANCELLED 22:22) — PRELIM; KILL; construction 4/4

`tree_v9d`, same recipe as §183. Control = mild twin **21809595**. Same kill rule.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940187** | VGG-16 C10 | 5 | **−7.9 pp** | 1/5 | arm −5.5 vs control −0.1 |

Scancel 22:22 (CG). With §177 and §183 this is **4 of 4** nets. Producers-only under P + crop+flip is closed.

**Read.** Hits the kill line faster than full-width ResNet-56 (§183 needed 20 pairs). Do **not** start another producers-only walk.

---

## 185. C-G+, P + crop+flip, full-width ResNet-56 (**21940191** CANCELLED 23:53) — PRELIM; KILL; construction 3/4

`tree_v9d`, NEON-literal C-G then 0.1× whole-network polish, 40/10, P + aug, `ft_recipe=C-G+`. Control = mild twin **21809595**. Paired val. Pre-authorized kill: ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940191** | R56 C10 | 9 | **−7.02 pp** | 1/9 | arm −5.58 vs control −0.18 |

Scancel 23:53 (CG). Thin pair already **§163** / **§180** KILL. VGG-16 **21940192** started on the C-PCA VGG-16 slot — leave it for its own kill rule. C-G+ is now killed on **3 of 4** nets → **CROSS-OFF** the construction (queue rule). Do **not** resubmit 21940191.

**Read.** The 0.1× polish does not close group redraw on full-width ResNet-56. Milder than full C-G on the same net (§181 −34 pp) and still a miss vs keep-the-survivors.

---

## 186. C-PCA, P + crop+flip, VGG-16 (**21940189** COMPLETED) — PRELIM; harsher than mild at equal keep; construction 3/4

`tree_v9d`, principal-direction replacement, 2-pass, 40/10, P + aug. COMPLETED 1 h 17 m, 1 Oct 23:53, `ise-4090-04`. Control = mild twin walk **21809595** (§164). Same keep at the named points. TEST = 5k half. TB=0. Quote `[eval] TRAJ` size points and `val_best` (traj_readout). Do not quote floor / terminal / `pass 1/1`.

| Point | Keep | C-PCA TEST | Mild 595 walk | Δ vs mild |
|---|---|---|---|---|
| size 0.80 | 0.796 / 0.747 | **−2.8** | −0.10 | **−2.7** |
| size 0.70 | 0.698 / 0.685 | **−1.5** | −0.70 | **−0.8** |
| `val_best` | 0.657 / 0.678 | **−1.5** | −0.40 | **−1.1** |

Paired val never hit the kill (28 pairs, mean −1.47 pp, 0 % better). TEST still loses at every equal-keep point. With §178 and §179 this is **3 of 4** nets worse than keep-the-survivors → **CROSS-OFF**. Full-width R56 **21940188** still R (CONTINUE on paired val) — leave it; one remaining net cannot re-open.

**Read.** C-PCA under P + aug is not a collapse, and it is not keep-the-survivors on VGG-16 either. Do **not** start another C-PCA walk.

---

## 187. C-G+, P + crop+flip, VGG-16 (**21940192** CANCELLED 00:54) — PRELIM; KILL; construction 4/4

`tree_v9d`, same recipe as §185 (`ft_recipe=C-G+`, 0.1× polish). Control = mild twin **21809595**. Same kill rule.

| Job | Net | Pairs | Mean arm−control val | Better | Last step |
|---|---|---|---|---|---|
| **21940192** | VGG-16 C10 | 14 | **−4.39 pp** | 0/14 | arm −4.72 vs control +0.18 |

Scancel 00:54 (CG). With §163, §180 and §185 this is **4 of 4** nets. C-G+ under P + crop+flip is closed. Do **not** resubmit 21940191 or 21940192.

**Read.** Milder than C-G on the same net (§182 −6.9 pp) and still a clear miss vs keep-the-survivors. The 0.1× polish does not rescue group redraw on VGG-16 either.

---

## 188. S0 selection headroom — lever measurement, never a TEST row (`21945105` / `06` / `07` COMPLETED)

`tree_v9d`, protocol P (clean val = 5k half, TEST = the other 5k half), crop+flip FT, keep 0.8 and 0.6, budgets 0 / bn / 1 / 3 / 10 / 40. Smoke `21944622` COMPLETED 12:58 (plumbing only). Cells: ResNet-56 C10 **21945105** (4.3 h, nap=4), VGG-16 C10 **21945106** (3.2 h, nap=0), VGG-19 C100 **21945107** (3.7 h, nap=0). Exit 0, TB=0. Decision on **val**. TEST sits beside each `[sel]` row and was not used to pick a criterion. Full Kendall / `[lever]` / `[overlap]` in `docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §8.

Keep 0.6 L1 (seed 0), val with TEST beside it, params ~0.36:

| Cell | budget=0 L1 val / TEST | budget=1 L1 val / TEST | budget=40 L1 val / TEST |
|---|---|---|---|
| r56 | −79.84 / −81.28 | −12.78 / −12.04 | −2.60 / −2.86 |
| vgg16 | −77.30 / −77.30 | −12.24 / −11.84 | −2.12 / −2.48 |
| vgg19 | −71.50 / −72.70 | −17.34 / −17.50 | −6.02 / −6.12 |

**M8** (runbook §10.4): on ≥ 2 of 3 cells, keep 0.6 `best_minus_l1_pp` or `ablation_minus_l1_pp` ≥ max(0.5, 2σ) at budget 40 or ≤ 3. **Fires on 3/3.** r56 at budget 0 (hrank +4.78); vgg16 at 0 (act +17.1) and 1 (fpgm +1.72 ≥ 1.70); vgg19 at 0 (svd +1.34), 1 (l2 +0.52) and **40** (l2 +1.03 ≥ 0.95). Not M8-neg. Not “only ≤ 3”.

**Read.** Which filters survive is a lever under our own fine-tune, including after 40 epochs on VGG-19 C100. S1 (learned NAP-F scorer, zero GPU) is sitting work. Do **not** start S1–S3 from ops. Do **not** quote these as a method TEST row. Do **not** lock.

---

## 189. FT proxy fidelity — six jobs COMPLETED; ceiling uninformative; never a TRAJ TEST (`21941343–48`)

`tree_v9d`, `SPECTRA_EVAL_PROXY_FIDELITY`, mild TRAJ under P + crop+flip 40/10 until keep ≤ 0.9 / 0.7, then the battery (val proxies `none` / `bn` / `12x4` / `40x10`; final TEST seeds 0 and 1). All six **COMPLETED**, TB=0. Last: **21941348** pf-mbv2-k70 9.5 h, 01:29. Never ledger these walks' TRAJ rows (truncated). Readout: `python scripts/proxy_fidelity_readout.py runs/job21941343 … runs/job21941348` (9 ranked `crit`/`where` sets).

**Zero-GPU look (1 Oct, inherit finals on disk):** with crop+flip, 40/10 walk TEST is within 0.5 pp of the 100-ep final at 16/16 size points. Without it, the final adds +1.6 to +5.8 pp. The final never reorders with-vs-without crop+flip on DepGraph R56 and VGG-19 (7/7).

**GPU battery (registered calls, written before the run):**

| Call | Result |
|---|---|
| Ceiling = mean ρ(final s0, final s1) | **+0.41** over 9 sets |
| Below 0.5 → uninformative | **yes.** Do not read any proxy |
| 12x4 valid (ρ ≥ 0.60 and median regret ≤ 0.5 pp) | not read (ceiling) |
| 40x10 valid, or ρ(40x10)−ρ(12x4) ≥ 0.2 | not read (ceiling) |
| Release held 40/10 train **21940321** | **no** — stays held |
| Depth penalty (12x4 vs final, printed only) | 1.09× (bar was 1.5; ignore under a dead ceiling) |

Printed but not used: bn / none / 12x4 / 40x10 mean ρ −0.07 / +0.16 / +0.35 / −0.07. Candidate spread is 0.24–1.55 pp; the two final seeds often disagree (MobileNet-V2 0.7 `crit` ceiling **−0.70**).

**Read.** These cuts are too small for the 100-ep final to rank them stably. Next cell (sitting): **widen the cuts** before any SGD-proxy A/B. 12/4 stays the in-loop recipe. Do **not** release 21940321. Do **not** lock.

---

## 190. C-PCA, P + crop+flip, zoo ResNet-56 (**21940188** COMPLETED) — PRELIM; harsher than mild at equal keep; construction **4/4**

`tree_v9d`, principal-direction replacement, 2-pass, 40/10, P + aug. COMPLETED 3 h 39 m, 2 Oct 02:01, TB=0. Control = mild twin walk **21809595** (§164). Same keep at the named points. TEST = 5k half. Quote `[eval] TRAJ` size points and `val_best`. Do not quote floor / terminal / `pass 1/1`.

| Point | Keep | C-PCA TEST | Mild 595 walk | Δ vs mild |
|---|---|---|---|---|
| size 0.80 | 0.794 / 0.737 | **−2.1** | −0.40 | **−1.7** |
| size 0.70 | 0.694 / 0.676 | **−2.4** | −0.40 | **−2.0** |
| `val_best` | 0.661 / 0.662 | **−2.7** | −0.20 | **−2.5** |

Paired val never hit the kill (60 pairs, mean −1.23 pp, 10 % better). TEST still loses at every equal-keep point. With §178, §179 and §186 this is **4 of 4** nets worse than keep-the-survivors → **CROSS-OFF**.

**Read.** C-PCA is milder than C-G and still not the recovery. Do **not** start another C-PCA walk. Do **not** lock.

---

## 191. S1 learned NAP-F selection scorer — zero GPU, probe section, never a TEST row

`scripts/selection_scorer_s1.py` over the §188 run dirs (sitting 2 Oct, Ido's delegated GO). Leave one network out. The learner is picked by inner cross-fit on the two training nets, never on the held-out one. Metric: width-weighted within-group Kendall τ against the single-channel oracle. The hand τ reproduced §188's printed values on 27/27 entries before any fit.

| Held-out net | Learned τ | Best hand on that net | Hand pick from the other two nets |
|---|---|---|---|
| DepGraph R56 C10 | **+0.572** | L2 +0.240 | Taylor +0.121 |
| chenyaofo VGG-16-BN C10 | **+0.657** | Taylor +0.417 | L2 +0.267 |
| DepGraph VGG-19 C100 | **+0.643** | L2 +0.170 | Taylor +0.123 |

**G1 PASS 3/3** (bar: +0.05 on 2 of 3). NAPv2 gradient statistics alone give +0.552 / +0.624 / +0.625; hand criteria alone give +0.210 / +0.270 / +0.181. Exported scorer `tree_v9d/runs/selection_scorer_s1/nap_f_model.pkl` (md5 `2a3bf48db614`, LONO τ 0.632).

**M8 re-read with the selection effect** (max over 9 named criteria vs pooled fine-tune noise). At keep 0.6 and trained budgets 1 / 3 / 10 / 40, p-null is 0.10–0.95; at budget 40 it is 0.91 / 0.95 / 0.49. So §188's M8 fire rests on budget 0 / BN and is noise-level once fine-tuned. The oracle's own masks are below L1 at 40 on all three cells (−0.55 / −0.30 / −0.71).

**Read.** A transferable selection *ranking* exists (S1). Whether it *recovers* better is S2 (**21982334** / **21982335**, registered calls in `docs/SITTING_GPU_QUEUE.md`), with a FAIL-at-40 prior. Do **not** quote as a method row. Do **not** lock. Full tables: design §8 "S1 results". S2 landed as **§192 HARM**.

---

## 192. S2 nap_f vs L1 recovery — a lever measurement, never a TEST row

`scripts/selection_probe_s2.py --readout` on `tree_v9d` run dirs `s2_mbv2_21982334` (COMPLETED 2 Oct 22:51) and `s2_r56c100_21982335` (COMPLETED 3 Oct 02:09). Scorer md5 `2a3bf48db614`. Keep 0.6, five shared fine-tune seeds, protocol P + crop+flip. Jobs `21982334` / `21982335`.

| Cell | *H_BN* | *H_1* | *H_3* | *H_40* (σ_ft) | vs G2 |
|---|---|---|---|---|---|
| MBV2 ×0.5 C10 | +2.80 (SE 0.07) | +1.83 (SE 0.54) | −0.25 | **+0.21** (0.72) | PASS already out (+0.21 < 1.44); BN would be cheap-FT on this cell alone |
| chenyaofo R56 C100 | −0.54 (SE 0.03) | +2.86 (SE 1.24) | −0.43 | **−0.87** (0.86) | *H_40* < −σ_ft |

**G2 call: HARM.** Cheap-FT budgets passing on every cell: none. Keep L1. Do **not** start S3. Kendall vs the single-channel oracle: MBV2 nap_f 0.254 < L1 0.286; R56-C100 nap_f 0.423 > L1 0.292. Oracle − L1 at 40: +0.57 / −1.03. Full table: design §8 "S2 result". Do **not** quote as a method row. Do **not** lock.

---

## 193. Stage-4 freeze TEST ep0095 (**21990060**) vs mild 21729557 — PRELIM; M1 does not fire

Skip-train `eval_c10_thin_traj` of **21737123** `snapshots/ep0095` (probe 0.286; first freeze after PPO-20). `tree_v9c`, P + crop+flip, 40/10, 2-pass, det=1, seed 42. COMPLETED 4 h 10 m, 3 Oct 15:40, `cs-4090-01`, TB 0. Control = mild **21729557** (§152), same walk recipe. TEST = 5k P half. Unpruned: r20-w2 0.649, r56-w4 0.890.

| Point | Actor 21990060 TEST (keep) | Mild 21729557 TEST (keep) |
|---|---|---|
| r20 size 0.80 | **−5.2 @ 0.702** | −1.2 @ 0.774 |
| r20 size 0.60 | −4.7 @ 0.595 | −4.9 @ 0.584 |
| r20 `val_best` | **−7.7 @ 0.417** | −5.3 @ 0.536 |
| r56 size 0.80 | **−4.5 @ 0.743** | −2.6 @ 0.795 |
| r56 size 0.60 | −5.3 @ 0.600 | NONE (mild ends 0.622) |
| r56 `val_best` | **−7.1 @ 0.389** | −4.5 @ 0.622 |

**Read.** Not a mild clone (r56 `val_best` keep 0.389 vs mild 0.622). Not M1: on both nets the named size / `val_best` points are more than 0.5 pp worse than mild, except r20 size 0.60 (≈ equal keep, 0.2 pp kinder). The first honest-protocol actor TEST is **deeper and harsher** than mild, not kinder. M1-neg needs a second freeze TEST; C2 ep0083 **22056144** is that TEST (in flight). Do **not** lock. Do not start N8.

**Equal-keep addendum (sitting, 4 Oct ~02:15; same TEST numbers).** The table above sets points at different keeps side by side, but M1 is defined at equal keep. Below, the agent's walk is read against the three mild walks of this recipe (21729557, RW43 21990185, D5-bis 21990184), interpolated linearly in keep (`scripts/_tmp_oct4_m1read.sh`):
- *r56-w4* (19 shared keeps, 0.743–0.622): kinder than the mild mean at **16**, by +0.3 to +1.2 pp over keep 0.72–0.62 (mild spread 0.2–0.8). It is worse at its first cut (**−1.2 @ 0.743**, mild spread 0.05) and at one transient step (−2.5 @ 0.628; +0.0 at the next step). It then continues to keep 0.389, which mild never reaches.
- *r20-w2* (7 shared keeps, 0.702–0.536): worse at 5. At the first cut it is **−1.4 @ 0.702** (mild spread 0.55). At keep 0.552–0.536 it is −1.0 to −1.4, inside mild's 1.8–2.2 pp spread.

The verdict stands: at equal keep against 21729557, the first size point is worse by more than 0.5 pp on both nets (r20 −1.7, r56 −1.2). "Harsher" holds for the first cut and for the keeps mild never reaches. Through keep 0.72–0.62, r56 is kinder on this one walk. FR43 **22059502** (§199) re-walked this actor with seed 43: widths match; **R56 kinder band replicates**; r20 first-point deficit does not. M1 on ep0095 stays this section’s.

---

## 194. H0 hold-out bars, mild, P, hold-out crop(+flip), 40/10, 2 passes, seed 42 — PRELIM TEST

Jobs **21986700** SVHN (COMPLETED 3 Oct 18:27, 16.5 h, `cs-4090-10`) and **21986701** Fashion-MNIST (COMPLETED 19:30, 17.3 h, `cs-4090-10`). `tree_v9d`, profile `baseline_c10_mild_traj_gonce`, `SPECTRA_FT_AUG_HOLDOUT=1`, `SPECTRA_VAL_FROM_TEST=1`. Origin TEST on the P half inside 0.2 pp of the checkpoint names. Size points 0.8 / 0.6 were **not** printed (resubmit did not pin `SPECTRA_EVAL_SIZE_POINTS`). Quote `[eval] TRAJ` TEST (`acc`). Never mix with 10k. Never put these nets in a training catalog.

| Net | Origin TEST | `val_best` TEST (keep) | `floor_hold` TEST (keep) |
|---|---|---|---|
| SVHN DenseNet-40 | 0.968 | **−0.8 @ 0.687** | −0.8 @ 0.701 |
| SVHN MBV2 ×0.5 | 0.969 | **−0.5 @ 0.672** | −0.5 @ 0.726 |
| SVHN RepVGG-A0 | 0.966 | **−0.3 @ 0.659** | −0.4 @ 0.802 |
| SVHN ShuffleNetV2 ×1 | 0.967 | **−0.3 @ 0.786** | −0.3 @ 0.786 |
| FMNIST DenseNet-40 | 0.953 | **−0.6 @ 0.687** | −0.9 @ 0.701 |
| FMNIST MBV2 ×0.5 | 0.949 | **−0.1 @ 0.672** | −0.1 @ 0.726 |
| FMNIST RepVGG-A0 | 0.947 | **+0.3 @ 0.659** | +0.2 @ 0.802 |
| FMNIST ShuffleNetV2 ×1 | 0.945 | **+0.2 @ 0.786** | +0.2 @ 0.786 |

Do **not** lock.

---

## 195. Wider proxy fidelity (`where`, keep ≤ 0.6 / 0.36) — probe, never TRAJ TEST rows

`proxy_fidelity_readout.py --sets where` on `21970086/87/88/89` (89 COMPLETED 3 Oct 11:30, 13.9 h, 9 candidates, not wall-truncated). Ranked `where` sets with n ≥ 3: **2 of 6** (MBV2 keep 0.6 n=9 ceiling **+0.69**; r56-w4 keep 0.6 n=9 ceiling **+0.74**). Mean ceiling **+0.71** (bar 0.60). All four proxies **not valid** (bn / none / 12x4 / 40x10 mean ρ +0.48 / +0.45 / +0.39 / +0.32). **12x4 NOT validated.** ρ(40x10) − ρ(12x4) = **−0.07** → do **not** ping to release 21940321. Neither budget valid → next cell is SGD-proxy variants on the saved finals (sitting). Ceiling ≥ 0.5, so this is not the “stop the pf line” call. Do **not** quote as TEST. **21940321 stays held.**

---

## 196. D5-bis and RW43 re-walks of mild 21729557 — probe + D5 call EQUIVALENT; RW43 is a seed-43 mild TEST

D5-bis **21990184** (`tree_v9d`, `SPECTRA_FT_AUG_GPU=1`, seed 42, 2-pass) COMPLETED 3 Oct 18:28, 2.8 h, `cs-4090-01`, GPU-aug banner present, TB 0. RW43 **21990185** (`tree_v9b`, seed 43, loader aug) COMPLETED 22:38, 4.2 h, `cs-4090-08`, TB 0. Same mild widths as 21729557 at the named steps. Five registered TEST points (`acc`, origin 0.649 / 0.890):

| Point | 21729557 s42 | D5-bis GPU | RW43 s43 | \|Δ\| D5 vs 42 | \|Δ\| 43 vs 42 |
|---|---|---|---|---|---|
| r20 size 0.80 | 0.637 (−1.2) | 0.634 | 0.634 | 0.3 pp | 0.3 pp |
| r20 size 0.60 | 0.600 (−4.9) | 0.607 | 0.588 | 0.7 | **1.2** |
| r20 terminal | 0.596 (−5.3) | 0.607 | 0.585 | **1.1** | **1.1** |
| r56 size 0.80 | 0.864 (−2.6) | 0.868 | 0.860 | 0.4 | 0.4 |
| r56 terminal | 0.844 (−4.6) | 0.837 | 0.843 | 0.7 | 0.1 |

**D5 call: EQUIVALENT ⇒ ADOPT for new cells.** Exactly one D5-bis point in (1.0, 2.0] (r20 terminal 1.1 pp); RW43 is also > 1.0 pp from s42 at that point → re-walk noise, not DIVERGE. Never into a live train, a resume, or a freeze TEST. s/epoch (FT ÷ epochs run): r20 **1.47**, r56 **3.73** vs control 4.29 / 5.24 on another 4090.

**RW43:** largest |ΔTEST| vs s42 = **1.2 pp**. M1’s 0.5 pp “no worse than mild” margin is inside re-walk noise; do not change the bar. Seed-43 rows are a real mild TEST; the bar may be quoted as the 42/43 mean **beside** the single walk, never instead of it. Do **not** lock the D5-bis columns as a SPECTRA method row.

*Per-walk noise (sitting addendum, 4 Oct; three walks, D5-bis counted since EQUIVALENT):* TEST SD ≈ **0.15 pp** at r20 size 0.80, **0.9–1.1 pp** at r20 keep ≤ 0.6, **≈ 0.4 pp** on r56 at both points. Two single walks differ by √2 × that.

---

## 197. Budget+STOP freeze TEST ep0131 (**22059501**) vs mild 21729557 — PRELIM; M1 waits on C2

Skip-train `eval_c10_thin_traj` of **21940311** `snapshots/ep0131` (probe 0.1339; first freeze after PPO-20). `tree_v9d`, P + crop+flip, 40/10, 2-pass, det=1, seed 42, `SPECTRA_TIME_DECIDE=1`. COMPLETED 1 h 52 m, 4 Oct 03:53, `ise-4090-03`, TB 0, exit 0. Budget menu pinned (`SPECTRA_ACTION_MENU` → `budget`). Control = mild **21729557** (§152). TEST = 5k P half. Unpruned TRAJ origin 0.649 / 0.890. **Not STOP-EARLY:** both nets have a size 0.80 point (r20 ended keep 0.618; r56 ended 0.399). r20 size 0.60 is NONE because the walk stopped at 0.618.

| Point | Actor 22059501 TEST (keep) | Mild 21729557 TEST (keep) |
|---|---|---|
| r20 size 0.80 | **−2.7 @ 0.792** | −1.2 @ 0.774 |
| r20 size 0.60 | NONE (ends 0.618) | −4.9 @ 0.584 |
| r20 `val_best` | −4.5 @ 0.618 | −5.3 @ 0.536 |
| r56 size 0.80 | **−3.8 @ 0.779** | −2.6 @ 0.795 |
| r56 size 0.60 | −8.3 @ 0.586 | NONE (mild ends 0.622) |
| r56 `val_best` | **−7.3 @ 0.399** | −4.5 @ 0.622 |

Equal keep vs the three-walk mild mean (`_tmp_oct4_m1read.sh`): r20 first cut **−1.39 @ 0.792** (5 of 7 shared keeps > 0.5 pp worse; 0 kinder ≥ 0.5). r56 first cut **−1.27 @ 0.779** (1 of 6 shared keeps > 0.5 pp worse; 0 kinder ≥ 0.5); then continues past mild’s floor to 0.399. Not a 0.9 mild clone (budget menu; 22 cuts / 114 r56 steps). **M1-neg** with §193 and §198 (4 Oct 06:16). Do **not** lock.

---

## 198. C2 NEON-raw freeze TEST ep0083 (**22056144**) vs mild 21729557 — PRELIM; **M1-neg**

Skip-train `eval_c10_thin_traj` of **21938810** `snapshots/ep0083` (probe 0.2893; first freeze after PPO-20). `tree_v9d`, P + crop+flip, 40/10, 2-pass, det=1, seed 42, `SPECTRA_TIME_DECIDE=1`. COMPLETED 4 h 12 m, 4 Oct 05:47, `ise-4090-21`, TB 0, exit 0. Control = mild **21729557** (§152). TEST = 5k P half. Unpruned TRAJ origin 0.649 / 0.890.

| Point | Actor 22056144 TEST (keep) | Mild 21729557 TEST (keep) |
|---|---|---|
| r20 size 0.80 | **−4.8 @ 0.702** | −1.2 @ 0.774 |
| r20 size 0.60 | −3.9 @ 0.595 | −4.9 @ 0.584 |
| r20 `val_best` | **−7.6 @ 0.417** | −5.3 @ 0.536 |
| r56 size 0.80 | **−4.2 @ 0.743** | −2.6 @ 0.795 |
| r56 size 0.60 | −5.6 @ 0.600 | NONE (mild ends 0.622) |
| r56 `val_best` | **−7.2 @ 0.389** | −4.5 @ 0.622 |

Equal keep vs the three-walk mild mean: r20 first cut **−1.02 @ 0.702** (3 of 7 shared keeps > 0.5 pp worse; 2 kinder ≥ 0.5 at 0.61–0.60). r56 first cut **−0.90 @ 0.743**; then kinder ≥ 0.5 at **14 of 19** shared keeps in 0.73–0.62 (peak +1.48 @ 0.665). Not a 0.9 mild clone (60 cuts / 114 r56 steps). Decide: r20 **8.2 ms**, r56 **3.0 ms**.

**M1 does not fire** on C2 or on Budget §197 (first cut > 0.5 pp worse on both nets). **M1-neg fires:** Stage-4 **21990060** plus C2 (and Budget) are each > 0.5 pp worse than mild on both nets at the first cut. Quote the margin with RW43 noise (§196: up to 1.2 pp at r20 keep ≤ 0.6; ≈ 0.4 pp on r56). Do **not** start N8. Do **not** lock.

---

## 199. FR43 — Stage-4 ep0095 freeze TEST re-walked seed 43 (**22059502**) — PRELIM; no call

Skip-train `eval_c10_thin_traj` of **21737123** `snapshots/ep0095`, seed 43, `tree_v9c`, P + crop+flip, 40/10, 2-pass. COMPLETED 4 h 18 m, 4 Oct 06:20, `ise-4090-03`, TB 0. Pair with seed-42 TEST **21990060** (§193). TEST = 5k P half. Unpruned TRAJ origin 0.649 / 0.890. M1 on ep0095 stays 21990060’s.

| Point | 21990060 s42 TEST (keep) | FR43 s43 TEST (keep) | \|ΔTEST\| |
|---|---|---|---|
| r20 size 0.80 | −5.2 @ 0.702 | −4.2 @ 0.702 | 1.0 |
| r20 size 0.60 | −4.7 @ 0.595 | −3.7 @ 0.595 | 1.0 |
| r20 `val_best` | −7.7 @ 0.417 | −7.7 @ 0.417 | 0.0 |
| r56 size 0.80 | −4.5 @ 0.743 | −6.2 @ 0.743 | 1.7 |
| r56 size 0.60 | −5.3 @ 0.600 | −5.0 @ 0.600 | 0.3 |
| r56 `val_best` | −7.1 @ 0.389 | −6.8 @ 0.389 | 0.3 |

- **Stability.** Traj-point keeps match **17/17 (r20)** and **61/61 (r56)** (\|Δkeep\| < 0.001). First differing step: none. The frozen actor’s widths survive seed 43. *(4 Oct: trivially, since the actor plays one action at every decision — §200. Not robustness evidence.)*
- **Noise.** Mean \|ΔTEST\| at shared widths: r20 **0.53 pp**, r56 **0.52 pp**. Largest: r20 **1.56** @ 0.519; r56 **3.18** @ 0.985 (near origin). Named-point largest is r56 size 0.80 **1.7 pp**.
- **Replication.** Two-walk agent mean vs three-walk mild: **R56 kinder band replicates** (14/17 keeps in 0.72–0.62 with gap ≥ +0.5). R20 first-point deficit vs mild at 0.702 is −1.42 (s42) and −0.42 (s43) — **does not replicate** (s43 inside mild spread 0.55).

No call. Do **not** lock.

---

## 200. Constant-policy census + O38b reward replay — the M1-neg mechanism (zero GPU; probe section, never a TEST row)

4 Oct sitting (Opus 5.5, opened 11:41 on ops' hand-over: "diagnose before any new train"). Read-only on the login node. Sources:
- *Walks:* the `step` events of the TEST walks (`events/rank0.jsonl`) and the five trains' sbatch profiles.
- *Replay:* `scripts/reward_replay.py` (O38, 1 Oct) through the live `compute_reward`, val only.
- *Sitting scripts:* `_tmp_s4oct_census*.sh`, `_tmp_s4oct_rewardflags.sh`, `_tmp_s4oct_replay*.sh` (not committed).

Full write-up: `docs/paper/GILAD_1OCT_POINTS_REPORT.md` Part III.

**Census: one action at every decision.** "Free" means the env did not force identity.

| Walk (TEST job) | r20 free decisions | r56 free decisions | Forced identity (r20 / r56) |
|---|---|---|---|
| mild 21729557 | 16 at 0.9 | 60 at 0.9 | 26 / 54 |
| Stage-4 ep0095 s42 **21990060** | 16 at **0.8** | 60 at **0.8** | 26 / 54 |
| Stage-4 ep0095 s43 FR43 **22059502** | 16 at 0.8 | 60 at 0.8 | 26 / 54 |
| C2 ep0083 **22056144** | 16 at 0.8 | 60 at 0.8 | 26 / 54 |
| Stage-4 ep0131 **22124693** (R) | 16 at 0.8 | 12 / 12 so far at 0.8 | 26 / 10 so far |
| Budget+STOP ep0131 **22059501** | the largest budget (4 % of origin) at every cut | same | never STOP |

The 0.8 decisions realize 13 / 51 cuts; at the rest, 0.8 rounds to the same width. Mild's 0.9 realizes 12 / 52. **M1-neg (§198) compares two fixed schedules: uniform 0.8 vs uniform 0.9.** §193 / §198 / §199 still stand as TEST numbers of that comparison.

**Why: the reward pays size inside a 10 pp band.**
- All five trains use `SPECTRA_REWARD_MODE=structural`, τ = `--allowed_acc_reduction` 10, no `SPECTRA_TRAIN_TAU`. Scales: `cbrt_cubes` (Stage-4, Budget, factored), `cbrt_miss` (C1), `raw` (C2); Budget adds STOP = 100 × `episode_inband_area`.
- Per step, with Δ cumulative against the origin's val accuracy: in band (−10 ≤ Δ ≤ 0) the reward is +ρ (the realized % param cut) **whatever Δ is**; gain +ρ or +ρ³; miss −ρ or −ρ³.
- The thin walks end 4–5 pp below origin on val, so the band never binds. Episodes have a fixed number of decisions. Hence "the largest cut at every decision" maximizes the return.

| Net | Walks (val replay) | Return over the walk: live (C1 = C2) | Return to equal depth | Val Δ at that depth (pp) |
|---|---|---|---|---|
| R56-w4 | mild × 3 (21729557, RW43 21990185, D5-bis 21990184) | 126.6 (126.6), to keep 0.622 | **126.0** @ 0.626 | −4.94 / −4.76 / −5.12 |
| R56-w4 | 0.8 × 3 (21990060, 22059502, 22056144) | **270.5** (270.5), to keep 0.389 | **124.9** @ 0.628 | −7.98 / −4.64 / −5.70 |
| R20-w2 | mild × 3 | 102.9 (1,566–2,267), to keep 0.522 | 91.5 @ 0.587 | −4.36 / −4.82 / −3.52 |
| R20-w2 | 0.8 × 3 | **144.2** (**8,255–8,544**), to keep 0.413 | 70.5 @ 0.587 | −2.94 / −3.22 / −3.34 |

At equal depth the live reward is blind to accuracy: on R56-w4 it pays 126.0 vs 124.9 across a 3 pp val spread. Over the walk it pays the 0.8 schedule about twice as much. C1 / C2's cubic gain arm makes it starker on R20-w2.

**Reward pre-check by replay (candidates; whole walk | equal depth; mean of 3 walks each):**

| Shape | R20-w2 mild vs 0.8 | R56-w4 mild vs 0.8 |
|---|---|---|
| live (`structural` / `cbrt_cubes`, τ 10) | 102.9 vs **144.2** \| **91.5** vs 70.5 | 126.6 vs **270.5** \| **126.0** vs 124.9 |
| live with τ 5 | **102.9** vs 99.7 \| **91.5** vs 70.5 | 118.1 vs **165.4** \| **117.9** vs 108.8 |
| F1 `structural_unified` (in band ρ · (τ + Δ) / τ), τ 10 | 90.6 vs **107.2** \| **83.7** vs 66.8 | 84.8 vs **154.3** \| 84.5 vs **87.7** |
| F1, τ 5 | **78.3** vs 69.3 \| **76.0** vs 63.1 | **43.0** vs −16.0 \| 43.0 vs **50.5** |
| F1, τ 3 | **66.1** vs 17.9 \| **70.3** vs 58.9 | **−81.4** vs −360.2 \| −72.0 vs **3.5** |

F1 with τ 5 (config only: `SPECTRA_REWARD_MODE=structural_unified SPECTRA_TRAIN_TAU=5`) is the mildest shape that stops paying the 0.8 schedule more over the walk on both nets. At equal depth on R56-w4 it still pays the 0.8 walks more (50.5 vs 43.0), although their val there is lower on average (−6.11 vs −4.94 pp). The slack-weighted sum pays early cuts, not the accuracy reached, so F1 τ 5 does **not** price accuracy at equal size. *(Corrected 4 Oct ~13:30: an earlier line read this as agreeing with TEST; the replay is on val.)* Val replay on six walks is a pre-check, not a train result.

**Consequences.**
- M1-neg is a reward-design result: PPO found the reward's optimum.
- §199's stability is trivial for a constant policy.
- A new action menu alone cannot help: the agent picks its largest entry.
- N10 (cubic reward) is covered by C1 / C2, which collapsed the same way.
- N8 waits for (i) a reward that passes this replay check and (ii) A0 headroom on at least one family.
- A0 thin **22127527 COMPLETED** — **§201 HEADROOM** on both keeps at budget 40. dg **22127528** R; cy **22127529** PD. Cross-net A0 call waits.
- No train without Ido's GO.

---

## 201. A0 allocation headroom, thin r56-w4 (**22127527**) — probe, never a TEST row

`scripts/allocation_probe.py` on ResNet-56-w4 C10 (`tree_v9d`). COMPLETED 1 h 28 m, 4 Oct 13:40, `cs-pheno-03` (GTX 1080), TB 0, exit 0. Protocol P, recipe A 40/10, crop+flip on the GPU. Uniform × 3 FT seeds; sens / sens2 / anti; random × 4. Matched to uniform's realized params (tolerance 0.003). Start checks green (`aug=1 aug_gpu=1`, GPU-loader banner, `Sensitivity at keep 0.5: 30 groups`). Log `runs/slurm_logs/alloc_22127527.out`.

**Call = budget 40 only** (bar = max(0.5, 2 × uniform val-Δ SD)). Allocations > 0.02 params from uniform are unmatched and not counted. Quote params-matched FLOPs: they differ.

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | sens2 val/TEST | Call |
|---|---|---|---|---|---|
| 0.6 | −8.22 (0.69) / −8.63 | 1.39 | **+1.62 / +1.95** (FLOPs 0.519) | **+1.68 / +2.25** (0.466) | **HEADROOM** |
| 0.35 | −17.81 (1.00) / −18.19 | 2.00 | **+7.81 / +7.69** | **+8.33 / +7.93** | **HEADROOM** |

Anti and three of four random draws are at or below uniform at budget 40. Budgets 0 / bn are proxy orderings, never a call.

**Per-net:** A0-HEADROOM on thin r56-w4 (both keeps). A reward that prices accuracy at equal size has something to learn on this net. **22127528 dg-r56 COMPLETED — §204 HEADROOM both keeps.** **22127529 cy-vgg16 COMPLETED — §205 HEADROOM both keeps.** **Cross-net A0-HEADROOM 3/3.** One-shot cut + 40-ep recovery, not the iterative walk. Do **not** quote as TEST. Do **not** lock.

---

## 202. Stage-4 freeze TEST ep0131 (**22124693**) vs mild 21729557 — PRELIM; M1 does not fire (M1-neg stands)

Skip-train `eval_c10_thin_traj` of **21737123** `snapshots/ep0131` (probe 0.2863; newest freeze since 21990060). `tree_v9c`, P + crop+flip, 40/10, 2-pass, det=1, seed 42. No `TIME_DECIDE`. COMPLETED 4 h 18 m, 4 Oct 15:07, `cs-4090-01`, TB 0, exit 0. Control = mild **21729557** (§152). TEST = 5k P half. Unpruned TRAJ origin 0.649 / 0.890. Same named keeps as ep0095 (constant 0.8).

| Point | Actor 22124693 TEST (keep) | Mild 21729557 TEST (keep) | ep0095 21990060 |
|---|---|---|---|
| r20 size 0.80 | **−4.8 @ 0.702** | −1.2 @ 0.774 | −5.2 @ 0.702 |
| r20 size 0.60 | −4.6 @ 0.595 | −4.9 @ 0.584 | −4.7 @ 0.595 |
| r20 `val_best` | **−8.7 @ 0.417** | −5.3 @ 0.536 | −7.7 @ 0.417 |
| r56 size 0.80 | **−3.9 @ 0.743** | −2.6 @ 0.795 | −4.5 @ 0.743 |
| r56 size 0.60 | −6.2 @ 0.600 | NONE (mild ends 0.622) | −5.3 @ 0.600 |
| r56 `val_best` | **−7.3 @ 0.389** | −4.5 @ 0.622 | −7.1 @ 0.389 |

Equal keep vs the three-walk mild mean (`_tmp_oct4_m1read.sh`): r20 first cut **−1.04 @ 0.702** (5 of 7 shared keeps > 0.5 pp worse; 0 kinder ≥ 0.5). r56 first cut **−0.62 @ 0.743** (2 of 19 shared keeps > 0.5 pp worse; **16 kinder ≥ 0.5** in 0.72–0.62). Not a mild clone (`val_best` keep 0.389 vs 0.622). **M1 does not fire:** the first equal-keep cut is still more than 0.5 pp worse than mild on both nets. **M1-neg stands.** Do **not** lock. Do not start N8.

---

## 203. FW — 12/4 + GPU crop+flip mild walk to DepGraph R56 sizes (**22127216**) — PRELIM probe; **SLOWER**; never an agent row

N3's line with `SPECTRA_FT_AUG_GPU=1 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4`, 5-pass, `flop:0.6,0.47,0.39`, `tree_v9d`, RTX 4090 `ise-4090-03`. COMPLETED 3 h 14 m, 4 Oct 15:10, TB 0, exit 0. Widths match N3 **21767189**: steps 136 / 210 / 267, params 0.638 / 0.470 / 0.382, FLOPs 0.599 / 0.463 / 0.380. Origin TEST 0.934. 10k = both CIFAR halves. Cost = walk to the point + that point's 100-ep final (`cost_readout.py 22127216 21767189`). DepGraph's whole pipeline is **85.1 min** on the same GPU model (21943448).

| Point | Params / FLOPs | Walk 5k | Final 5k | Honest | 10k final | N3 10k | DepGraph | K=1 cost (walk+final) | N3 cost |
|---|---|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | −0.60 | −0.34 | −0.06 CROSS-OFF | **−0.24** | −0.03 | — | 61.8 + 13.0 = **74.8 min** | 264.8 + 15.5 = 280.2 |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | −1.64 | −1.76 | −0.44 CROSS-OFF | **−1.24** | **−0.46** | **+0.24** | 95.0 + 13.4 = **108.4 min** | 405.6 + 15.8 = 421.4 |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.90 | −1.84 | −0.26 CROSS-OFF | **−1.25** | −1.63 | +0.11 | 120.4 + 13.3 = 133.6 min | 518.1 + 15.9 = 534.0 |
| `val_best` | 0.356 / 0.369 | −1.96 | −1.94 | −0.30 CROSS-OFF | n/a | n/a | — | 127.3 + 13.0 = 140.3 min | 546.6 + 15.6 = 562.2 |
| origin | 1.000 / 1.000 | 0.00 | +0.32 | — | +0.48 | — | — | 13.4 min | 15.9 |

**Call at 2.11×: SLOWER.** K=1 cost **108.4 min > 85.1**. 10k **−1.24** is 0.78 pp behind N3's −0.46 and 1.48 pp behind DepGraph's +0.24. The walk itself is **4.3×** faster than N3 to the same step (95.0 vs 405.6 min) and still **1.27×** DepGraph at one target. Peak alloc 1.56 GB; 439.2 Wh to 2.11×. Final FT CROSS-OFF (same as N3). K* vs DepGraph, measured 12/4 W: **1.3** at the 2.11× point (95.0 / (85.1 − 13.4)); **1.8** to keep 0.36 (127.3 / (85.1 − 13.0)). The projected 12/4 column in EFFICIENCY §7 used W ≈ 165 min (K* 2.4); replace that projection with these numbers. Do **not** quote as a SPECTRA-agent row. Do **not** lock.

---

## 204. A0 allocation headroom, DepGraph ResNet-56 C10 (**22127528**) — probe, never a TEST row

`scripts/allocation_probe.py` on DepGraph's ResNet-56 C10 (`tree_v9d`). COMPLETED 3 h 21 m, 4 Oct 17:01, `cs-pheno-03` (GTX 1080), TB 0, exit 0. Same recipe as §201: P, 40/10, GPU crop+flip. Uniform × 3 FT seeds; sens / sens2 / anti; random × 4. Matched to uniform's realized params. Call = budget 40 only. Log `runs/slurm_logs/alloc_22127528.out`.

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | sens2 val/TEST | Call |
|---|---|---|---|---|---|
| 0.6 | −2.36 (0.18) / −2.63 | 0.50 | **+0.60 / +0.61** (FLOPs 0.503 vs 0.596) | −0.42 / +0.03 (0.458) | **HEADROOM** |
| 0.35 | −4.08 (0.47) / −4.03 | 0.94 | **+2.24 / +1.99** (FLOPs 0.277 vs 0.343) | **+1.50 / +1.33** (0.281) | **HEADROOM** |

Keep 0.6 is tight: random0 **+0.76 / +0.81** also clears the 0.50 bar; sens just clears. Keep 0.35 is the lever (sens +2.24 val). Anti is below uniform at budget 40. Budgets 0 / bn are proxy orderings, never a call.

**Per-net:** A0-HEADROOM on DepGraph R56 C10 (both keeps). Together with thin r56-w4 §201, allocation is a lever on both ResNet-56 cells. VGG-16 **22127529 COMPLETED — §205 HEADROOM** both keeps. **Cross-net A0-HEADROOM 3/3.** Do **not** quote as TEST. Do **not** lock. Do not start a train from this.

---

## 205. A0 allocation headroom, chenyaofo VGG-16 C10 (**22127529**) — probe, never a TEST row; **cross-net A0-HEADROOM 3/3**

`scripts/allocation_probe.py` on chenyaofo VGG-16 C10 (`tree_v9d`). COMPLETED 2 h 21 m, 4 Oct 17:28, `cs-pheno-09` (GTX 1080), TB 0, exit 0. Same recipe as §201. Call = budget 40 only. Log `runs/slurm_logs/alloc_22127529.out`.

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | sens2 val/TEST | Call |
|---|---|---|---|---|---|
| 0.6 | −2.44 (0.54) / −2.23 | 1.09 | +0.32 / +0.57 (FLOPs 0.818 vs 0.601) | +0.58 / −0.07 (0.821) | **HEADROOM** (random1 **+1.76 / +1.33**) |
| 0.35 | −2.61 (0.24) / −2.72 | 0.50 | +0.25 / −0.04 (FLOPs 0.729 vs 0.354) | **+0.57 / +0.72** (0.713) | **HEADROOM** |

Keep 0.6: the sensitivity rule does **not** clear the bar; a random allocation does. Keep 0.35: sens2 just clears (0.57 ≥ 0.50); matched params, unmatched FLOPs (0.71 vs 0.35). Anti is below uniform at budget 40.

**Per-net:** A0-HEADROOM on VGG-16 C10 (both keeps).

**Cross-net (budget 40, both keeps):** thin r56-w4 §201, DepGraph R56 §204, VGG-16 this section — **A0-HEADROOM 3/3**. Allocation is a lever on these cells. Ido GO 19:23: next train is fixed-target (sitting builds a new tree). Do **not** quote as TEST. Do **not** lock. Do **not** start that train from ops.

---

## 206. Factored freeze TEST ep0083 (**22132735**) vs mild 21729557 — PRELIM; Taylor vs L1; not M1

Skip-train `eval_c10_thin_traj` of **21940316** `snapshots/ep0083` (probe 0.2947; first freeze after PPO-20). `tree_v9d`, P + crop+flip, 40/10, 2-pass, det=1, seed 42, `SPECTRA_TIME_DECIDE=1`. COMPLETED 4 h 14 m, 4 Oct 19:34, `ise-4090-21`, TB 0, exit 0. `SPECTRA_FACTORED_HEAD` pin. Constant policy **(keep 0.8, Taylor)** at every free decision; same named keeps as Stage-4 ep0095 / ep0131 (L1 at 0.8). Control = mild **21729557**. TEST = 5k P half. Origin 0.649 / 0.890. The train was scancelled 19:26 (Ido); this TEST had already started.

| Point | Actor 22132735 TEST (keep) | Mild 21729557 TEST (keep) | Stage-4 ep0131 22124693 |
|---|---|---|---|
| r20 size 0.80 | **−3.4 @ 0.702** | −1.2 @ 0.774 | −4.8 @ 0.702 |
| r20 size 0.60 | −3.1 @ 0.595 | −4.9 @ 0.584 | −4.6 @ 0.595 |
| r20 `val_best` | **−7.3 @ 0.417** | −5.3 @ 0.536 | −8.7 @ 0.417 |
| r56 size 0.80 | **−3.0 @ 0.743** | −2.6 @ 0.795 | −3.9 @ 0.743 |
| r56 size 0.60 | −5.6 @ 0.600 | NONE (mild ends 0.622) | −6.2 @ 0.600 |
| r56 `val_best` | **−8.0 @ 0.389** | −4.5 @ 0.622 | −7.3 @ 0.389 |

Equal keep vs the three-walk mild mean: r20 first cut **+0.38 @ 0.702** (3 of 7 shared keeps > 0.5 pp worse; 2 kinder ≥ 0.5). r56 first cut **+0.30 @ 0.743** (2 of 19 worse ≥ 0.5; 9 kinder ≥ 0.5). **Not M1-neg** on the first-cut bar. **Not M1:** the first-cut gaps are inside RW43 noise (≤ 1.2 pp) and r56 `val_best` is harsher than Stage-4.

**Taylor vs L1** at shared keeps (same widths): vs ep0095 **21990060** mean TEST **+0.43 pp** on r20 (n=11) and **−0.17 pp** on r56 (n=39). FR43 |s42−s43| means **0.68 / 0.55**. The ranking-menu mean is inside re-walk noise. First cut is kinder than L1 (r20 +1.80, r56 +1.52 at keep 0.702 / 0.743); `val_best` on r56 is harsher (−8.0 vs −7.1 / −7.3). Adopt-vs-Stage-4 (> 1 pp on **both** nets) does **not** fire (r20 +1.4, r56 +0.9 at the first named keep). Decide **5.4 / 3.6 ms**. Do **not** lock. Do not start N8. Do not resubmit the cancelled factored train.

---

## 207. A0b allocation headroom, thin ResNet-20-w2 C10 (**22155641**) — probe, never a TEST row

`scripts/allocation_probe.py` on thin r20-w2 C10 (`tree_v9d`). COMPLETED 40 min, 4 Oct 20:11, `cs-pheno-11` (GTX 1080), TB 0, exit 0. Call = budget 40 only. Match params. Keeps 0.8 / 0.6 / 0.35. Log `runs/slurm_logs/alloc_22155641.out`. Keep 0.8 uniform realizes params ~0.70 (coarse groups).

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | Call |
|---|---|---|---|---|
| 0.8 | −6.19 (0.35) / −7.70 | 0.70 | +0.73 / +0.52 | **HEADROOM** (random1 **+1.93 / +2.86**) |
| 0.6 | −9.09 (0.78) / −10.52 | 1.55 | −2.97 / −2.62 | **FLAT** (best random1 +0.17 / +1.22) |
| 0.35 | −17.08 (0.86) / −18.33 | 1.72 | −0.40 / −1.43 | **HEADROOM** (random1 **+2.24 / +3.11**) |

Keep 0.8: sens just clears the bar; a random draw is the winner. Keep 0.6: nothing clears 1.55. Keep 0.35: random, not sensitivity.

**Per-net:** A0b-HEADROOM on thin r20-w2 at keep 0.8 and 0.35; **FLAT at 0.6**. Registered consequence: R20-w2 stays in the v10 M1 read (the drop rule needed FLAT/HARM at **both** 0.8 and 0.6). Do **not** quote as TEST. Do **not** lock. No train action from ops.

---

## 208. A0b allocation headroom, thin ResNet-56-w4 C10 keep 0.8 (**22155642**) — probe, never a TEST row

`scripts/allocation_probe.py` on thin r56-w4 C10 (`tree_v9d`), keep **0.8** only. COMPLETED 32 min, 4 Oct 20:04, `ise-pheno-01` (GTX 1080), TB 0, exit 0. Call = budget 40. Match params. Log `runs/slurm_logs/alloc_22155642.out`. Keep 0.8 realizes params ~0.84 (coarse groups). Anti unmatched.

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | Call |
|---|---|---|---|---|
| 0.8 | −4.69 (0.67) / −4.19 | 1.33 | **+2.63 / +2.01** (params 0.842) | **HEADROOM** (random0 +2.53 / +2.25) |

**Per-net:** A0b-HEADROOM on thin r56-w4 at keep 0.8. Together with §207, keep 0.8 has headroom on a thin net ⇒ v10 M1's first-cut (κ = 0.8) read **stands**. Do **not** quote as TEST. Do **not** lock. No train action from ops.

---

## 209. A0b allocation headroom, chenyaofo VGG-16 C10 equal FLOPs (**22155643**) — probe, never a TEST row

`scripts/allocation_probe.py` on chenyaofo VGG-16 C10 (`tree_v9d`), `--match flops`. COMPLETED 2 h 16 m, 4 Oct 21:48, `ise-pheno-01` (GTX 1080), TB 0, exit 0. Call = budget 40. Keeps 0.6 / 0.35 of FLOPs. Log `runs/slurm_logs/alloc_22155643.out`. Matching: first non-uniform FLOPs within 0.02 of uniform (0.601 / 0.601 and 0.352 / 0.351). Sensitivity at those FLOPs keeps far fewer params (0.180 vs 0.599 at keep 0.6; 0.078 vs 0.348 at 0.35).

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | Call |
|---|---|---|---|---|
| 0.6 | −2.25 (0.27) / −2.33 | 0.54 | +0.21 / +0.25 (params 0.180 vs 0.599) | **HEADROOM** (random1 **+1.27 / +1.17**) |
| 0.35 | −2.65 (0.24) / −2.91 | 0.50 | −0.31 / −0.47 (params 0.078 vs 0.348) | **FLAT** (best random0 +0.37 / +0.01) |

Keep 0.6: sensitivity does **not** clear 0.54; a random allocation does. Keep 0.35: nothing clears 0.50. Compare §205 (params-matched): both keeps were HEADROOM, and the 0.35 winner kept twice uniform's FLOPs.

**Per-net:** A0b equal-FLOPs HEADROOM at keep 0.6, **FLAT at 0.35**. Registered consequence: FLOPs targets belong at the 0.6 operating point; equal-FLOPs 0.35 on VGG is not a lever. No train action from ops. Do **not** quote as TEST. Do **not** lock.

---

## 210. A0b allocation headroom, chenyaofo ResNet-56 CIFAR-100 (**22155644**) — probe, never a TEST row

`scripts/allocation_probe.py` on chenyaofo ResNet-56 C100 (`tree_v9d`). COMPLETED 3 h 24 m, 4 Oct 22:55, `ise-pheno-04` (GTX 1080), TB 0, exit 0. Call = budget 40. Match params. Keeps 0.6 / 0.35. Log `runs/slurm_logs/alloc_22155644.out`. Matching: first non-uniform params within 0.02 of uniform (0.595 / 0.592 and 0.354 / 0.354). Sensitivity unmatched on FLOPs (0.385 vs 0.588 at keep 0.6; 0.237 vs 0.343 at 0.35).

| Keep | Uniform val Δ (SD) / TEST Δ | Bar | sens val/TEST | Call |
|---|---|---|---|---|
| 0.6 | −6.91 (0.79) / −7.19 | 1.58 | −1.61 / −2.41 | **FLAT** (best random2 +0.01 / −1.13) |
| 0.35 | −10.39 (1.46) / −10.89 | 2.92 | +1.71 / +0.95 | **FLAT** (sens does not clear 2.92) |

BN-only budgets show HEADROOM (not a call). After 40-ep recovery nothing clears the bar.

**Per-net:** A0b **FLAT** on R56-C100 at both keeps. Registered consequence: **the first fixed-target train stays C10**; C100 nets do not join that catalog. No train action from ops. Do **not** quote as TEST. Do **not** lock.

---

## 211. v10 mild-landed control κ = 0.8 (**22156061**) — PRELIM control; not an agent row

`baseline_c10_mild_traj_gonce` + `SPECTRA_FIXED_TARGET=1`, `tree_v10`, thin pair, P, loader crop+flip, walk 40/10, 6 passes, floor off, `SIZE_MATCH=SIZE_POINTS=param:0.8`, final FT 100 from the origin, seed 42. COMPLETED 2 h 36 m, 4 Oct 22:53, `ise-4090-16`, TB 0, exit 0. Walk plays 0.9 and lands by bisection. Control for every v10 freeze TEST at κ = 0.8. κ = 0.6 control is **§212**.

| Net | Landed params (κ 0.800) | Walk TEST | Final-FT TEST (size_param0.80 = `val_best`) | FLOPs |
|---|---|---|---|---|
| r20-w2 | **0.774** (gap 0.026) | −1.1 @ 0.774 | **−0.4 @ 0.774** | 0.818 |
| r56-w4 | **0.799** (gap 0.001) | −2.2 @ 0.799 | **−2.1 @ 0.799** | 0.716 |

No `TRAJ … param:0.8 … NONE`. Origin final-FT TEST r20 +3.2 / r56 +0.3. **Flag:** r20 landed keep differs from κ by 0.026 (channel granularity; registered). Quote this landed size beside every actor Δ; do not re-pick a point. Do **not** lock. Never resubmit per freeze.

---

## 212. v10 mild-landed control κ = 0.6 (**22156062**) — PRELIM control; not an agent row

Same recipe as §211 at `param:0.6`. COMPLETED 5 h 57 m, 5 Oct 02:14, `ise-4090-21`, TB 0, exit 0. The registered M1-v10 WIN cell is r56 at this κ.

| Net | Landed params (κ 0.600) | Walk TEST | Final-FT TEST (size_param0.60 = `val_best`) | FLOPs |
|---|---|---|---|---|
| r20-w2 | **0.584** (gap 0.016) | −4.7 @ 0.584 | **−2.9 @ 0.584** | 0.674 |
| r56-w4 | **0.600** (gap 0.000) | −6.0 @ 0.600 | **−5.1 @ 0.600** | 0.453 |

No `TRAJ … param:0.6 … NONE`. Origin final-FT TEST r20 +3.3 / r56 +0.1. r20 land is within the 0.02 matching bar (unlike §211). Quote this landed size beside every actor Δ; do not re-pick a point. Do **not** lock. Never resubmit per freeze.

---

## 213. v10 FLOPs mild keep 0.6, chenyaofo VGG-16 C10 (**22228975**) — PRELIM heuristic; not an agent row

Ido GO 08:06 Pareto counterpart. `baseline_c10_mild_traj_gonce`, **no** `SPECTRA_FIXED_TARGET` (v10 landing is params-only), `SIZE_MATCH=SIZE_POINTS=flop:0.6`, `tree_v10`, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, final FT 100 from the origin, seed 42. COMPLETED 1 h 51 m, 5 Oct 10:02, `ise-4090-02`, TB 0, exit 0. Walk ends at the first point with FLOPs ≤ 0.6 (step 37). `TRAJ floor_cross … NONE` is expected (floor off). DepGraph R56 twin **22228976** still R.

| Net | FLOPs (target 0.600) | params | Walk TEST | Final-FT TEST (`size_flop0.60` = `val_best`) |
|---|---|---|---|---|
| VGG-16 BN C10 | **0.593** (gap 0.007) | 0.623 | −0.0 @ 0.593 | **−0.0 @ 0.593** |

No `TRAJ … flop:0.6 … NONE`. Origin final-FT TEST **+0.8**. FLOPs gap 0.007 is inside the 0.02 matching bar — do not re-pick a point. Do **not** quote in-walk val −0.38 as TEST. Do **not** lock. Never resubmit.

---

## 214. v10 greedy-landed control κ = 0.8 (**22228972**) — PRELIM heuristic; not an agent row

Ido GO 08:06 Pareto counterpart. `baseline_c10_l1_traj_gonce` + `SPECTRA_FIXED_TARGET=1` (Gilad greedy = L1), `tree_v10`, thin pair, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, `SIZE_MATCH=SIZE_POINTS=param:0.8`, final FT 100 from the origin, seed 42. COMPLETED 2 h 25 m, 5 Oct 10:37, `ise-4090-21`, TB 0, exit 0. Menu is baseline 1.0/0.9/0.8 (no overlay of the actor's 0.7/0.6). Mild-landed κ 0.8 control remains **§211**. κ = 0.6 greedy twin **22228973** still R.

| Net | Landed params (κ 0.800) | Walk TEST | Final-FT TEST (size_param0.80 = `val_best`) | FLOPs |
|---|---|---|---|---|
| r20-w2 | **0.782** (gap 0.018) | −1.2 @ 0.782 | **−0.2 @ 0.782** | 0.821 |
| r56-w4 | **0.788** (gap 0.012) | −3.2 @ 0.788 | **−2.4 @ 0.788** | 0.663 |

No `TRAJ … param:0.8 … NONE`. Origin final-FT TEST r20 +3.8 / r56 +0.3. Both landed gaps inside the 0.02 matching bar. Beside mild-landed §211 (r20 **−0.4 @ 0.774**, r56 **−2.1 @ 0.799**): not a lock, not an actor Δ. Do **not** lock. Never resubmit.

---

## 215. v10 random-landed κ = 0.6, r56-w4 only (**22228974**) — PRELIM heuristic; not an agent row

Ido GO 08:06 Pareto counterpart (WIN cell, one seed). `baseline_c10_random` + TRAJ/gonce overlay + `SPECTRA_FIXED_TARGET=1`, `tree_v10`, `input_c10_thin_r56w4.json`, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, `SIZE_MATCH=SIZE_POINTS=param:0.6`, final FT 100 from the origin, seed 42. COMPLETED 3 h 37 m, 5 Oct 11:49, `ise-4090-20`, TB 0, exit 0. Mild-landed κ 0.6 r56 remains **§212**. Greedy κ 0.6 twin **22228973** still R.

| Net | Landed params (κ 0.600) | Walk TEST | Final-FT TEST (size_param0.60 = `val_best`) | FLOPs |
|---|---|---|---|---|
| r56-w4 | **0.564** (gap 0.036) | −6.3 @ 0.564 | **−5.1 @ 0.564** | 0.436 |

No `TRAJ … param:0.6 … NONE`. Origin final-FT TEST **+0.3**. **Flag:** landed keep differs from κ by 0.036 (above the 0.02 matching bar). Quote 0.564 beside any Δ; do not re-pick a point; do not call this equal-size vs mild §212 (**−5.1 @ 0.600**). Do **not** quote in-walk val −6.80 as TEST. Do **not** lock. Never resubmit.

---

## 216. v10 greedy-landed control κ = 0.6 (**22228973**) — PRELIM heuristic; not an agent row

Ido GO 08:06 Pareto counterpart. Same recipe as **§214** at `param:0.6`. COMPLETED 3 h 44 m, 5 Oct 11:56, `ise-4090-20`, TB 0, exit 0. The registered M1-v10 WIN cell is r56 at this κ. Mild-landed control remains **§212**. Random-landed r56 is **§215** (not equal-size).

| Net | Landed params (κ 0.600) | Walk TEST | Final-FT TEST (size_param0.60 = `val_best`) | FLOPs |
|---|---|---|---|---|
| r20-w2 | **0.595** (gap 0.005) | −4.0 @ 0.595 | **−2.4 @ 0.595** | 0.734 |
| r56-w4 | **0.600** (gap 0.000) | −4.3 @ 0.600 | **−4.7 @ 0.600** | 0.453 |

No `TRAJ … param:0.6 … NONE`. Origin final-FT TEST r20 +3.5 / r56 +0.5. Both landed gaps inside the 0.02 matching bar. Beside mild-landed §212 (r20 **−2.9 @ 0.584**, r56 **−5.1 @ 0.600**): greedy r56 is **+0.4 pp** at equal keep 0.600 — not a lock, not an actor Δ. Do **not** lock. Never resubmit.

---

## 217. v10 FLOPs mild keep 0.6, DepGraph ResNet-56 C10 (**22228976**) — PRELIM heuristic; not an agent row

Ido GO 08:06 Pareto counterpart. Twin of **§213**. `baseline_c10_mild_traj_gonce`, **no** `SPECTRA_FIXED_TARGET` (v10 landing is params-only), `SIZE_MATCH=SIZE_POINTS=flop:0.6`, `tree_v10`, `input_catalog_l_depgraph_r56.json`, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, final FT 100 from the origin, seed 42. COMPLETED 4 h 54 m, 5 Oct 13:06, `ise-4090-01`, TB 0, exit 0. Walk ends at the first point with FLOPs ≤ 0.6 (step 136). `TRAJ floor_cross … NONE` is expected (floor off). Quote `[eval] TRAJ final_ft size_flop0.60` (= `val_best`); do **not** quote in-walk val +0.10 as TEST.

| Net | FLOPs (target 0.600) | params | Walk TEST | Final-FT TEST (`size_flop0.60` = `val_best`) |
|---|---|---|---|---|
| DepGraph R56 C10 | **0.599** (gap 0.001) | 0.638 | −0.1 @ 0.599 | **−0.1 @ 0.599** |

No `TRAJ … flop:0.6 … NONE`. Origin final-FT TEST **+0.2**. FLOPs gap 0.001 is inside the 0.02 matching bar — do not re-pick a point. This is the v10-recipe FLOPs heuristic star, not a 10k N3 / DepGraph-published comparison. Do **not** lock. Never resubmit.

---

## 218. Stage-4 freeze TEST ep0179 (**22260374**) vs mild 21729557 — PRELIM; M1 does not fire (M1-neg stands; 1.0 pp bar)

Skip-train `eval_c10_thin_traj` of **21737123** `snapshots/ep0179` (probe 0.2888; newest freeze since 22124693 / ep0131). `tree_v9c`, P + crop+flip, 40/10, 2-pass, det=1, seed 42. No `TIME_DECIDE`, no `FT_AUG_GPU`. COMPLETED 4 h 20 m, 6 Oct 02:46, `ise-4090-11`, TB 0, exit 0. Control = mild **21729557** (§152). TEST = 5k P half. Unpruned TRAJ origin 0.649 / 0.890. Same named keeps as ep0095 / ep0131 (constant 0.8).

| Point | Actor 22260374 TEST (keep) | Mild 21729557 TEST (keep) | ep0131 22124693 |
|---|---|---|---|
| r20 size 0.80 | **−5.1 @ 0.702** | −1.2 @ 0.774 | −4.8 @ 0.702 |
| r20 size 0.60 | −4.1 @ 0.595 | −4.9 @ 0.584 | −4.6 @ 0.595 |
| r20 `val_best` | **−7.0 @ 0.417** | −5.3 @ 0.536 | −8.7 @ 0.417 |
| r56 size 0.80 | **−4.0 @ 0.743** | −2.6 @ 0.795 | −3.9 @ 0.743 |
| r56 size 0.60 | −5.0 @ 0.600 | NONE (mild ends 0.622) | −6.2 @ 0.600 |
| r56 `val_best` | **−7.6 @ 0.389** | −4.5 @ 0.622 | −7.3 @ 0.389 |
| r56 `floor_hold` | −3.5 @ 0.704 | — | — |

**Census (r56-w4):** 60 cuts at **0.8**, 54 identity at 1.0. Distinct prune actions = **1**. r20: 16 at 0.8, 26 identity. **M1 cannot fire** (Ido 6 Oct 01:32: census ≥ 2 distinct actions). Not a mild clone on keep (`val_best` 0.389 vs mild 0.622).

**Equal keep** vs the three-walk mild mean (`_tmp_oct4_m1read.sh`; bar now **1.0 pp**):
- *r56-w4* (19 shared keeps): first cut **−0.74 @ 0.743** (inside 1.0 pp). One kinder ≥ 1.0 (**+1.21 @ 0.717**). One worse ≤ −1.0 (**−2.31 @ 0.628**).
- *r20-w2* (7 shared): first cut **−1.36 @ 0.702** — disaster guard only; not a ≳ 3 pp cliff; does not veto.

**M1 does not fire.** Historical M1-neg (4 Oct, 0.5 pp / both nets) **stands**. New bar: first r56 cut is inside 1.0 pp, but the policy is still constant 0.8 and one shared keep is −2.3 pp. Do **not** start N8. Do **not** lock. Never quote probe 0.2888.

---

## 219. S0 keep 0.35 on DepGraph R56 — L1 vs Taylor vs nap_f vs random vs anti-L1 (**22288374**) — probe, never a TEST row

`tree_v9d`, keep **0.35**, budgets **0 and 40**, scorer md5 `2a3bf48db614`. COMPLETED 1 h 10 m, 6 Oct 02:47, `ise-pheno-09`, TB 0. Per-group keep 0.35 ⇒ **params 0.121 / FLOPs 0.129**. Protocol P + crop+flip. Read jsonl at keep=0.35 (do not use the default 0.6 readout).

| Budget | L1 val (sd, n=3) | nap_f − L1 | taylor − L1 | random − L1 | anti-L1 − L1 |
|---|---|---|---|---|---|
| 0 | −82.82 (0.00) | **+0.06** | −0.46 | −0.26 | −0.68 |
| 40 | −8.03 (0.12) | **−0.42** | −5.74 | −2.09 | −2.46 |

Lever budget 40: best named = nap_f, `best_minus_l1_pp` **−0.42**. L1 still beats random (+2.09) and anti-L1 (+2.33). Kendall vs ablation: nap_f **0.628** (descriptor); vs L1 0.225. Jaccard nap_f vs L1 0.31.

**Call.** 40-ep does **not** beat L1. nap_f *H_40* < −σ_ft (**HARM** vs L1). Budget 0 is ~−83 pp on every mask. Ranking at high sparsity is dead under our FT. Keep L1. Do **not** start S3. Write the paper sentence and stop the ranking ladder. Remaining NAP idea = cheap proxy for stopping a walk (pf line already hurt). Design paste: `FILTER_SELECTION_NAP_DESIGN.md` "S0 keep 0.35". Do **not** quote as TEST. Do **not** lock.

---

## 220. τ-off mild — 10-pass, τ=30, no param floor, DepGraph ResNet-56 C10 (**22288423**) — PRELIM; PATH-SAME vs N3; 100-ep CROSS-OFF; do not train this recipe

Ido GO 6 Oct 01:32. N3 line (`tree_v9d`) + `allowed_acc_reduction=30` + **10 passes** + `MIN_PARAM=0` + `EVAL_ROLLBACK=0` + P + loader crop+flip never `FT_AUG_GPU` + walk 40/10 + `flop:0.6,0.47,0.39` + 100-ep origin SGD final FT. Pair: N3 **21767189** (§157). COMPLETED 19 h 28 m, 6 Oct 21:06, `ise-4090-08`, TB 0, exit 0. Reader: `final_ft_readout.py` + `crossfit_readout.py`. TEST = 5k P half (unpruned 0.934). Origin change **+0.86** (10k **+0.73**).

**PATH-SAME widths vs N3:** steps **136 / 210 / 267**, params 0.638 / 0.470 / 0.382, FLOPs 0.599 / 0.463 / 0.380.

| Point | Params / FLOPs | Walk (5k) | Final (5k) | Honest | 10k final | N3 walk 5k | N3 10k | DepGraph |
|---|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | **+0.18** | −0.20 | **−1.24 CROSS-OFF** | **−0.16** | +0.08 | −0.03 | — |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | **−0.76** | −0.90 | **−1.00 CROSS-OFF** | **−0.94** | −0.22 | **−0.46** | **+0.24** |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | **−1.22** | −1.34 | **−0.98 CROSS-OFF** | **−1.52** | −1.32 | −1.63 | **+0.11** |
| `val_best` | 0.123 / 0.121 | −5.42 | −5.02 | −0.46 CROSS-OFF | n/a | −1.12 @ 0.356 | n/a | — |
| origin | 1 | 0 | +0.86 | +0.86 | +0.73 | 0 | +0.60 | — |

`TRAJ floor_cross … NONE` (floor off). `floor_hold` = `val_best` keep 0.123 — do **not** quote as the DepGraph comparison. Census: 301 cuts, val Δ>0 on 71 (max +0.96).

**Read.** τ=30 + more passes + no floor did **not** change the architectures at DepGraph's sizes. Extra passes only walked **past** them to keep **0.123** (walk **−5.42**). τ did not stop the mild walk. 100-ep origin FT is **CROSS-OFF** at every size point (same pattern as N3). 2.11× walk 5k **−0.76** vs N3 **−0.22** is **0.54 pp** at identical widths (`det=1`, different 4090) — inside the 1.2 pp re-walk band (RW43); not a τ effect. 10k at 2.11× **−0.94** vs N3 **−0.46** / DepGraph **+0.24**. Paper DepGraph rows stay **N3**. Do **not** put τ-off into a DRL train. Do **not** lock. Never resubmit.

---

## 221. L3a: Le & Hua large-LR final FT (cosine from lr 0.1) on §212's saved thin candidates (**22340235**) — PRELIM; thin pair **CROSS-OFF** (raw)

Sitting 7 Oct, under Ido's prompt (`docs/PROMPT_FABLE_OCT7_SITTING.md` B1). `tree_v10`, no src change.
- *Recipe.* From-saved: `SPECTRA_EVAL_FINAL_FT_FROM=tree_v10/runs/job22156062/traj_models`, `SPECTRA_EVAL_FINAL_FT_LR=0.1`. Otherwise §212's final FT: SGD m 0.9, wd 5e-4, per-epoch cosine, 100 epochs, batch 128, loader crop+flip. P (5k TEST half), seed 42, deterministic, origin control.
- *Run.* COMPLETED 48 m, 7 Oct 03:11, `ise-4090-19`, exit 0.
- *Reader and call.* `final_ft_readout.py`. The call, registered before the read, is in queue "Sitting 7 Oct", Lead 3. ADOPT needs honest Δ ≥ +0.5 pp and raw final_new ≥ final_old. The thin rule reads r56-w4 at κ 0.6; N3's 2.11× (22340234) gates the schedule.

| Net | Point (params / FLOPs) | Walk | Final, lr 0.01 (§212) | Final, lr 0.1 | Origin change, 0.01 → 0.1 | Honest, 0.01 → 0.1 | Δ honest | 10k final, lr 0.1 |
|---|---|---|---|---|---|---|---|---|
| r20-w2 | size 0.60 (0.584 / 0.674) | −4.68 | −2.86 | **−1.82** | +3.32 → +4.38 | −1.50 → −1.52 | **−0.02** | −1.14 |
| r56-w4 | size 0.60 (0.600 / 0.453) | −5.96 | −5.06 | **−5.22** | +0.12 → **−0.64** | +0.78 → +1.38 (ORIGIN-HURT) | +0.60, inflated | −5.16 |

**Read.**
- *r20-w2.* lr 0.1 lifts the pruned net by +1.04 pp raw and the undertrained origin by +1.06. No pruned-specific gain.
- *r56-w4.* It costs the converged origin 0.64 pp and the pruned net 0.16 pp raw. The +0.60 honest Δ is the origin's loss subtracted; the reader flags ORIGIN-HURT, so read the raw value.
- *Call.* The thin rule fails on raw (final_new < final_old): **CROSS-OFF on the thin pair** for cosine from 0.1. The 100-ep lr 0.01 recipe stays the thin caption.
- The gating read is N3 at 2.11× (22340234, still R). Do not lock. Never an agent row.

---

## 222. L3b: 1-cycle large-LR final FT (warmup 30 epochs to lr 0.1, then cosine) on §212's saved thin candidates (**22340388**) — PRELIM; thin pair **CROSS-OFF**

Same cell as §221, with the Le & Hua 1-cycle shape.
- *Recipe.* `SPECTRA_EVAL_FINAL_FT_LR=0.1 SPECTRA_EVAL_FINAL_FT_SCHEDULE=warmcos SPECTRA_EVAL_FINAL_FT_WARMUP=30`: per-batch linear warmup over 30 epochs, then cosine to 1e-5; 100 epochs; the rest as §221.
- *Run.* `tree_v10h`. COMPLETED 48 m, 7 Oct 03:24, `ise-4090-20`, exit 0. Reader `final_ft_readout.py`; call as §221.

| Net | Point (params / FLOPs) | Walk | Final, lr 0.01 (§212) | Final, 1-cycle | Origin change, 0.01 → 1-cycle | Honest, 0.01 → 1-cycle | Δ honest | 10k final, 1-cycle |
|---|---|---|---|---|---|---|---|---|
| r20-w2 | size 0.60 (0.584 / 0.674) | −4.68 | −2.86 | **−2.60** | +3.32 → +4.58 | −1.50 → −2.50 | **−1.00** | −1.85 |
| r56-w4 | size 0.60 (0.600 / 0.453) | −5.96 | −5.06 | **−6.12** | +0.12 → **−1.38** | +0.78 → +1.22 (ORIGIN-HURT) | +0.44, inflated | −6.06 |

**Read.** Worse than cosine from 0.1 (§221) on both nets.
- The warmup to a large LR costs the converged r56-w4 origin 1.38 pp, and the pruned net 1.06 pp raw.
- On r20-w2 the undertrained origin gains 1.26 pp more than under lr 0.01, but the pruned net gains only 0.26.
- **CROSS-OFF on the thin pair** for the 1-cycle schedule. N3 gating **22340387 COMPLETED §224**: ADOPT by the registered rule at 2.11×, by 0.06 pp and origin-driven; robustness reads pending. Do not lock. Never an agent row.

---

## 223. L3a: cosine-from-lr-0.1 final FT on N3's saved DepGraph R56 candidates (**22340234**) — PRELIM; **CROSS-OFF** at 2.11× (the gating point)

The gating cell of Lead 3 (`docs/PROMPT_FABLE_OCT7_SITTING.md` B1 + B2).
- *Recipe.* `tree_v10`, from-saved `SPECTRA_EVAL_FINAL_FT_FROM=tree_v9c/runs/job21767189/traj_models` (all four N3 candidates), `SPECTRA_EVAL_FINAL_FT_LR=0.1`. Otherwise N3's final FT (§157): SGD m 0.9, wd 5e-4, per-epoch cosine, 100 epochs, batch 128, loader crop+flip. P, seed 42, deterministic, origin control.
- *Run.* COMPLETED 1 h 22 m, 7 Oct 03:45, `ise-4090-18`, exit 0.
- *Reader and call.* `final_ft_readout.py`; call as §221. ADOPT needs honest Δ ≥ +0.5 pp at `size_flop0.47` and raw final_new ≥ final_old there. TEST = 5k half (unpruned 0.934).

| Point | Params / FLOPs | Walk (5k) | Final 5k, lr 0.01 (§157) | Final 5k, lr 0.1 | Honest, 0.01 → 0.1 | Δ honest | 10k final, 0.01 → 0.1 | DepGraph (10k) |
|---|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | +0.08 | +0.04 | −0.26 | −0.40 → −0.96 | −0.56 | −0.03 → +0.04 | — |
| **size_flop0.47 (2.11×)** | 0.470 / 0.463 | −0.22 | −0.44 | **−0.32** | −0.58 → −0.72 | **−0.14** | −0.46 → **−0.36** | +0.24 |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.32 | −1.34 | **−0.62** | −0.38 → +0.08 | +0.46 | −1.63 → **−0.37** | +0.11 |
| `val_best` | 0.356 / 0.369 | −1.12 | −0.96 | −0.90 | −0.20 → −0.40 | −0.20 | n/a | — |
| origin | 1 | 0 | +0.36 | **+0.62** | — | — | +0.60 → +0.64 | — |

**Read.**
- *2.11×.* The large LR lifts the pruned net by +0.12 pp raw and the origin by +0.26 more than lr 0.01 does. Honest Δ −0.14: **CROSS-OFF** under the registered rule. The 100-ep lr 0.01 final FT stays the paper caption.
- *2.57× (not gating).* The deepest saved point gains +0.72 pp raw on 5k and +1.26 on 10k. Honest Δ +0.46 is just under the bar. That is the shape Le & Hua report: large-LR retraining helps more at higher sparsity. On 10k it narrows the 2.57× gap to DepGraph's +0.11 from 1.74 pp to 0.48. One FT seed; never "beats".
- *Consequence.* No schedule change for the paper rows. If a deeper cell (keep ≤ 0.4) is ever reported, a pre-registered lr 0.1 re-finalisation of *all* compared rows at that size is the one open use. Sitting's wave-6 cell **22341051** (cosine on τ-off saves, keep 0.123) is that registered deep read — leave it; do not invent a twin. The 1-cycle arm **22340387 COMPLETED §224**.
- Do not lock. Never an agent row.

---

## 224. L3b: 1-cycle large-LR final FT (warmup 30 epochs to lr 0.1, then cosine) on N3's saved DepGraph R56 candidates (**22340387**) — PRELIM; **ADOPT by the registered rule** at 2.11×, by 0.06 pp and origin-driven; the paper caption waits on the robustness reads

Pair of §223. `tree_v10h`, `SPECTRA_EVAL_FINAL_FT_SCHEDULE=warmcos SPECTRA_EVAL_FINAL_FT_WARMUP=30`, from-saved N3 `job21767189/traj_models`. Otherwise the §223 recipe. Reader `final_ft_readout.py` on the run dir (not the `.out`). COMPLETED 1 h 22 m, 7 Oct 03:56, `ise-4090-15`, exit 0. TEST = 5k half (unpruned 0.9336). ADOPT still needs honest Δ ≥ +0.5 pp **and** raw final_new ≥ final_old at `size_flop0.47`.

| Point | Params / FLOPs | Walk (5k) | Final 5k, lr 0.01 (§157) | Final 5k, 1-cycle | Honest, 0.01 → 1-cycle | Δ honest | 10k final, 0.01 → 1-cycle | DepGraph (10k) |
|---|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | +0.08 | +0.04 | **−0.04** | −0.40 → −0.12 | +0.28 | −0.03 → **+0.07** | — |
| **size_flop0.47 (2.11×)** | 0.470 / 0.463 | −0.22 | −0.44 | **−0.24** | −0.58 → −0.02 | **+0.56** | −0.46 → **−0.44** | +0.24 |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.32 | −1.34 | **−1.16** | −0.38 → +0.16 | +0.54 | −1.63 → **−1.30** | +0.11 |
| `val_best` | 0.356 / 0.369 | −1.12 | −0.96 | **−1.06** | −0.20 → +0.06 | +0.26 | n/a | — |
| origin | 1 | 0 | +0.36 | **+0.00** | — | — | +0.60 → **+0.07** | — |

**Read.**
- *2.11×.* Raw **+0.20 pp** (−0.24 vs −0.44). Honest Δ **+0.56** is mostly the origin **not** lifting (+0.00 vs §157 +0.36), not a 0.5 pp pruned-net win. Thin pair already CROSS-OFF (§222).
- *Call (sitting correction, 04:10).* The rule registered before the read (queue "Sitting 7 Oct", Lead 3) is honest Δ ≥ +0.5 **and** raw final_new ≥ final_old. Both hold (+0.56; raw +0.20 ≥ 0), so it fires **ADOPT**. The prompt's own rule (honest ≥ +0.5 vs the 100-ep at the same widths) agrees. The first version of this entry read the raw condition as also needing +0.5. The point of substance stands: the margin is 0.06 pp (three TEST images), +0.36 of the +0.56 is the origin control, and the identical lr 0.01 final FT moved this same origin by +0.42 (§153), +0.36 (§157) and +0.86 (§220). On 10k the 2.11× point moves −0.46 → −0.44.
- *Robustness reads (registered 04:10, before submit; queue Lead 3, wave 7).* A from-saved lr 0.01 control on N3's candidates gives the paired reference: same code path and RNG state, only the schedule differs. A 1-cycle replicate on τ-off's PATH-SAME candidates is read against §220. A schedule enters the paper caption only if it passes the Lead 3 rule in all three reads (this one, paired, replicate), and then every compared row is re-finalised with it.
- *2.57×.* Raw +0.18 / 10k +0.33. Does **not** replicate cosine-from-0.1's +0.72 raw / 10k +1.26 at this size (§223). Wave-6 **22341051** tests that cosine trend on τ-off saves; do not attach 1-cycle to it. The 1-cycle run on the same saves (**22341280**, wave 7) is a separate cell with a different job: it re-reads this entry's 2.11× ADOPT, not the deep trend.
- Paper caption stays **100-ep lr 0.01** until the robustness reads land. Do not lock. Never an agent row.

---

## 225. Greedy 5-rate menu (1.0/0.9/0.8/0.7/0.6) landed κ 0.6 thin (**22340233**) — PRELIM; r56 **FLAT** vs 3-rate §216; r20 overshoot

Sitting 7 Oct B3 ladder (`docs/PROMPT_FABLE_OCT7_SITTING.md`; queue call HELP ≥ +1.0 / HURT ≤ −1.0 vs §216 on r56-w4). `tree_v10`, profile `baseline_c10_l1_traj_gonce`, `--compression_rates 1.0 0.9 0.8 0.7 0.6`, `FIXED_TARGET=1`, `SIZE_MATCH=param:0.6`, 6 passes, P, loader crop+flip, walk 40/10, 100-ep origin final FT lr 0.01, seed 42, det=1. COMPLETED 2 h 20 m, 7 Oct 04:42, `ise-4090-20`, exit 0. Reader `final_ft_readout.py` on the run dir. Pair **22340232** (4-rate, no 0.6) read in §226 — the ladder is closed there.

| Net | Landed params (κ 0.600) | Walk 5k | Final 5k (`size_param0.60` = `val_best`) | FLOPs | vs §216 3-rate |
|---|---|---|---|---|---|
| r20-w2 | **0.538** (gap 0.062) | −8.28 | **−4.84 @ 0.538** | 0.657 | not equal-size (guard) |
| r56-w4 | **0.594** (gap 0.006) | −7.02 | **−5.06 @ 0.594** | 0.436 | **−0.36 pp** vs −4.7 @ 0.600 |

Origin final-FT TEST r20 **+3.40** / r56 **+0.24**. r56 gap 0.006 is inside the 0.02 matching bar. r20 gap 0.062 is the overshoot sitting flagged at 02:51: one 0.6 step on a 2/4/8-wide net.

**Read.** r56 is inside the ±1.0 pp bar: **neither HELP nor HURT**. Adding 0.6 to the *greedy* menu at this κ does not beat 3-rate §216. Do **not** overlay 0.6 onto the paper greedy counterpart. Actor-menu 0.6 stays open until a v10 freeze TEST. r20 is the disaster guard only. Do not lock. Never an agent row.

---

## 226. Step-size ladder at landed κ 0.6, thin pair, both arms: greedy 4-rate (**22340232**) and 5-rate (**22340233**, §225) — PRELIM; **FLAT** (r56-w4 best +0.52 vs §216); step size is a cost lever; mild and greedy-3 land on the same r56-w4 architecture

Sitting 7 Oct, B3' (the prompt's menu A/B, corrected; queue "Sitting 7 Oct"). Closes the ladder that §225 opened with greedy-5 alone. §225's −0.36 used §216 rounded to −4.7; the reader's −4.68 gives −0.38.
- *Recipe.* `tree_v10`, sbatch only: §216's greedy walk (profile `baseline_c10_l1_traj_gonce`, `FIXED_TARGET=1`, `param:0.6`, 6 passes, P, loader crop+flip, 40/10, 100-ep final FT + origin, seed 42, deterministic), with the menu widened through `SPECTRA_EXTRA_ARGS="--compression_rates …"` to 1.0/0.9/0.8/0.7 (greedy-4) and 1.0/…/0.6 (greedy-5). Greedy plays the strongest legal cut, so the step is 0.7 / 0.6.
- *Run.* 22340232 COMPLETED 2 h 44 m, 7 Oct 05:07, `ise-4090-19`; 22340233 COMPLETED 2 h 20 m, 04:42, `ise-4090-20`; exit 0, `floor_cross NONE` (floor off).
- *Reader.* `final_ft_readout.py`; the landed point is `val_best` = `size_param0.60` (same step). Architectures are compared from the saved `traj_models` (out-channels of every conv).

| Net | Walk (cut per step) | Step at landing | Params / FLOPs | Channels kept, thirds of the conv list | Walk Δ | Final (5k) | Origin change | Final − §216 |
|---|---|---|---|---|---|---|---|---|
| r56-w4 | mild §212 (0.9) | 136 | 0.600 / 0.453 | 38 / 102 / 247 | −5.96 | −5.06 | +0.12 | −0.38 |
| r56-w4 | greedy-3 §216 (0.8) | 79 | 0.600 / 0.453 | 38 / 102 / 247 | −4.32 | −4.68 | +0.48 | — |
| r56-w4 | **greedy-4 (0.7)** | 45 | 0.600 / **0.582** | 57 / 114 / 236 | −4.02 | **−4.16** | +0.38 | **+0.52** |
| r56-w4 | **greedy-5 (0.6)** | 39 | 0.594 / 0.436 | 38 / 95 / 248 | −7.02 | **−5.06** | +0.24 | **−0.38** |
| r20-w2 | mild §212 | 36 | 0.584 / 0.674 | 14 / 14 / 44 | −4.68 | −2.86 | +3.32 | −0.44 |
| r20-w2 | greedy-3 §216 | 28 | 0.595 / 0.734 | 14 / 20 / 42 | −3.98 | −2.42 | +3.50 | — |
| r20-w2 | greedy-4 | 28 | 0.595 / 0.734 | 14 / 20 / 42 | −4.24 | −2.32 | +3.48 | +0.10 |
| r20-w2 | greedy-5 | 15 | **0.538** / 0.657 | 14 / 14 / 41 | −8.28 | −4.84 | +3.40 | not equal-size |

**Read.**
- *Call (r56-w4).* The better arm, greedy-4, is +0.52 pp over §216, under the +1.0 bar: **FLAT**. Greedy-5 is −0.38. At κ 0.6 the wider menu is a cost lever: 39 decisions with greedy-5, against 45 (greedy-4), 79 (greedy-3) and 136 (mild). Walk time is about 94 / 116 / 177 / 302 min in the same order (job wall minus final-FT minutes; different 4090 nodes and loads).
- *Same architecture, different path.* On r56-w4, mild §212 and greedy-3 §216 landed on the identical architecture (all 57 conv widths equal) by different paths. Their 0.38 pp gap is path and fine-tune noise, not architecture. On r20-w2, greedy-3 and greedy-4 also share one architecture, 0.10 pp apart. These are the first same-architecture noise reads for the thin cells.
- *FLOPs.* Greedy-4 kept FLOPs 0.582 at equal params, against 0.453. It kept more channels in the first third of the network (57 vs 38) and fewer in the last (236 vs 247). Its +0.52 comes with 13 pp more FLOPs kept: not an equal-cost gain.
- *For the v10 read (not a call).* HURT did not fire: playing 0.7 / 0.6 does not cost accuracy at κ 0.6. But greedy-4 sits +0.90 over mild §212, just 0.10 under the v10 WIN bar (+1.0), with FLOPs 0.582. An actor that clears the bar while keeping FLOPs near 0.58 has shown no more than the 0.7 step does, so v10's FLOPs are quoted beside its Δ.
- *Guard net.* r20-w2 greedy-5 overshoots κ (0.538): one 0.6 step on 2/4/8-wide groups. No disaster at equal size.
- Do not lock. Never an agent row.

---

## 227. Allocation-following walk, A0's sens rule, landed κ 0.8, thin pair (**22340393**) — PRELIM; r56-w4 **+0.82 over mild §211**; lever **WEAK** vs uniform §229

Sitting 7 Oct, allocation walk (queue "Sitting 7 Oct", allocation section).
- *Recipe.* `tree_v10h`, profile `baseline_c10_alloc_traj_gonce`, `SPECTRA_ALLOC_KIND=sens` (α 0.5), the v10 5-rate menu, `FIXED_TARGET=1`, `param:0.8`, 6 passes, P, loader crop+flip, walk 40/10, 100-ep final FT + origin, seed 42, deterministic.
- *Run.* COMPLETED 2 h 36 m, 7 Oct 05:12, `ise-4090-21`, exit 0, no stall fallback.
- *Plan lines.* r56-w4: plan keeps x0.781 (target x0.780) over 30 groups; group keeps min 0.19, median 0.67, max 1.00. r20-w2: x0.774 over 12 groups, min 0.23, median 0.87. On r20-w2 every group reached its target with params still at x0.827, above κ, so the walk finished with strongest legal cuts (logged).

| Net | Params / FLOPs | Walk Δ | Final (5k) | Origin change | vs mild §211 | vs greedy §214 |
|---|---|---|---|---|---|---|
| r56-w4 | 0.800 / 0.696 | −1.30 | **−1.30** | +0.46 | **+0.82** (−2.12 @ 0.799 / 0.716) | +1.08 (−2.38 @ 0.788 / 0.663) |
| r20-w2 | 0.799 / 0.880 | +0.16 | **+1.30** | +3.74 | +1.74; flagged, params 0.799 vs 0.774 | +1.50 (−0.20 @ 0.782 / 0.821) |

**Read.**
- *r56-w4.* +0.82 pp over mild at equal params, with slightly fewer FLOPs kept (0.696 vs 0.716). The registered lever (sens − uniform) is **§229**: **WEAK** (+0.96). The bar rule is set at κ 0.6. Same-architecture noise on this net is 0.38 pp (§226).
- *r20-w2.* +1.30 is above the unpruned origin's TEST, but the origin control gains +3.74 under the same final FT (an undertrained 5k-param net): final-FT gain, not pruning gain (honest −2.60). The sens plan cut a few late, parameter-heavy groups hard and left the rest near full, so FLOPs stay at 0.880. Guard net only.
- Seed-43 twins 22341283 / 84 (wave 8) PD. Do not lock. Never an agent row.

---

## 228. L3a-deep: cosine-from-lr-0.1 final FT on τ-off's saved DepGraph R56 candidates (**22341051**) — PRELIM; **TREND** (keep 0.123 Δ honest +2.26; 2.57× +0.70); at 2.11× the replicate passes (+1.10) where §223 failed (−0.14)

Sitting 7 Oct, Lead 3, wave 6 (registered 03:55, before submit).
- *Recipe.* `tree_v10`, from-saved `SPECTRA_EVAL_FINAL_FT_FROM=tree_v9d/runs/job22288423/traj_models` (all four τ-off candidates: PATH-SAME widths as N3, different inherited weights), `SPECTRA_EVAL_FINAL_FT_LR=0.1`. Otherwise §223's recipe: SGD m 0.9, wd 5e-4, per-epoch cosine, 100 epochs, batch 128, loader crop+flip; P, seed 42, deterministic, origin control.
- *Run.* COMPLETED 1 h 21 m, 7 Oct 05:18, `cs-4090-08`, exit 0.
- *Reader and calls.* `final_ft_readout.py` on this run and on 22288423 (§220, lr 0.01, same candidates). TREND needs Δ honest ≥ +1.0 at `val_best` (keep 0.123) and ≥ +0.3 at 2.57×. The 2.11× row is also cosine's replicate read under the Lead 3 rule (wave 7), reported only, because cosine failed its first read (§223).

| Point | Params / FLOPs | Walk (5k) | Final 5k, lr 0.01 (§220) | Final 5k, lr 0.1 | Honest, 0.01 → 0.1 | Δ honest | 10k final, 0.01 → 0.1 | N3 under lr 0.1 (§223): Δ honest / 10k | DepGraph (10k) |
|---|---|---|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | +0.18 | −0.20 | +0.38 | −1.24 → −0.54 | +0.70 | −0.16 → +0.05 | −0.56 / +0.04 | — |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | −0.76 | −0.90 | **+0.08** | −1.00 → +0.10 | **+1.10** | −0.94 → **+0.01** | −0.14 / −0.36 | +0.24 |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.22 | −1.34 | **−0.76** | −0.98 → −0.28 | **+0.70** | −1.52 → **−0.36** | +0.46 / −0.37 | +0.11 |
| `val_best` (keep 0.123) | 0.123 / 0.121 | −5.42 | −5.02 | **−2.88** | −0.46 → +1.80 | **+2.26** | n/a | — | — |
| origin | 1 | 0 | +0.86 | +0.74 | — | — | +0.73 → +0.82 | origin +0.62 | — |

**Read.**
- *Call: **TREND**.* Keep 0.123: Δ honest +2.26 (raw +2.14). 2.57×: +0.70. Large-LR retraining helps more the deeper the cut. At 2.57× it lands at the same place from both walks: 10k −0.37 (N3) and −0.36 (τ-off), against −1.63 and −1.52 under lr 0.01.
- *2.11× (replicate read, reported).* Passes the Lead 3 rule here (honest Δ +1.10, raw +0.98), where N3's candidates failed it (−0.14 / +0.12). On 10k the two walks move +0.10 and +0.95. At 2.11× the effect depends on the inherited weights; at 2.57× it does not.
- *Origin.* The lr 0.1 origin control replicates: +0.74 here, +0.62 in §223. Under lr 0.01 the same origin moved +0.36 / +0.86 / +0.42 (§157 / §220 / §153).
- *DepGraph.* At 2.57× on 10k, lr 0.1 sits about 0.47 pp behind DepGraph's +0.11 from both walks, against about 1.6 under lr 0.01. At 2.11×: −0.36 / +0.01 against +0.24. One final-FT seed per walk; never "beats".
- *Consequence.* The caption stays lr 0.01: cosine failed its first read, and the wave 7 rule needs all three. The registered TREND caption note applies, worded to the data: large-LR retraining (Le & Hua 2021) helps consistently at deep sparsity (2.57× and beyond), and at 2.11× it varies with the walk. A paper row that uses it re-finalises every compared row at that size, pre-registered. Do not lock. Never an agent row.

---

## 229. Allocation-following walk, uniform control κ 0.8 + lever vs §227 (**22340394**) — PRELIM; r56 **WEAK** (+0.96, 0.04 under SURVIVES)

Pair of **§227**. `tree_v10h`, `SPECTRA_ALLOC_KIND=uniform`, otherwise the §227 recipe (`param:0.8`, 6 passes, P, 100-ep origin FT). COMPLETED 2 h 43 m, 7 Oct 05:53, `ise-4090-17`, exit 0. Reader `final_ft_readout.py`. Call (queue): SURVIVES ≥ +1.0 / ABSORBED ≤ +0.3 / **WEAK** in between, on r56-w4 (sens − uniform). Seed-43 twins still PD.

| Net | Uniform **22340394** params / FLOPs | Uniform 5k | Sens §227 5k | Sens − uniform | vs mild §211 |
|---|---|---|---|---|---|
| r56-w4 | 0.799 / 0.716 | **−2.26** | **−1.30** @ 0.800 / 0.696 | **+0.96** | uniform −0.16; sens +0.82 |
| r20-w2 | 0.774 / 0.818 | **+0.02** | **+1.30** @ 0.799 / 0.880 | +1.28; params gap 0.025 | guard |

**Read.** r56 lever **+0.96 pp** is **WEAK** (0.04 under SURVIVES). Params match (0.800 vs 0.799). Sens keeps fewer FLOPs (0.696 vs 0.716). Two-seed mean (22341283 / 84) still pending. Do not lock. Never an agent row.

---

## 230. Allocation-following walk, sens vs uniform at landed κ 0.6, thin pair (**22340391 / 22340392**) — PRELIM; r56 **WEAK** (+0.54); bar vs mild **+2.30**

Sitting 7 Oct allocation walk at the v10 WIN κ. `tree_v10h`, `FIXED_TARGET=1`, `param:0.6`, 6 passes, P, 100-ep origin FT. Sens **22340391** COMPLETED 3 h 9 m, 05:43, `ise-4090-21`. Uniform **22340392** COMPLETED 3 h 13 m, 05:48, `ise-4090-21`. Reader `final_ft_readout.py`. Lever: SURVIVES ≥ +1.0 / ABSORBED ≤ +0.3 / WEAK in between. Bar: sens − mild-landed §212 ≥ +1.0 on r56-w4.

| Net | Sens **91** params / FLOPs | Sens 5k | Uniform **92** params / FLOPs | Uniform 5k | Sens − uniform | vs mild §212 |
|---|---|---|---|---|---|---|
| r56-w4 | 0.600 / 0.572 | **−2.80** | 0.599 / 0.582 | **−3.34** | **+0.54** | sens **+2.30**; uniform +1.76 |
| r20-w2 | 0.595 / 0.800 | **−2.92** | 0.581 / 0.741 | **−2.92** | 0.00; params gap 0.014 | guard |

**Read.**
- *Lever.* **WEAK** (+0.54). A0's sensitivity plan beats uniform at this κ, but not by the +1.0 SURVIVES bar. Seed-43 twins 22341281 / 82 still PD.
- *Bar vs mild.* Sens **−2.80 @ 0.600** vs mild **−5.1 @ 0.600** is **+2.30 pp** at equal params — clears the +1.0 bar. That is a no-agent allocation walk, not an actor. Quote FLOPs beside it (0.572 vs mild 0.453).
- Do not lock. Never an agent row.

---

## 231. Allocation arms by architecture (zero GPU; the saved candidates of §212 / §216 / §226 / §227 / §229 / §230) — PRELIM read; r56-w4's final Δ follows its **residual width**; mild and uniform land on the **same architecture** at κ 0.8; wave 9 registered

Sitting 7 Oct. Conv widths from each run's saved `val_best` candidate (`traj_models/*.json`, `arch`); no GPU. *Residual* = the width of a stage's coupled stream (stem or downsample plus every block's second conv); *inner* = each block's first conv. r56-w4's origin widths are 4 / 8 / 16 per stage.

| κ | Arm (job) | Final 5k | Params / FLOPs | Residual s1 / s2 / s3 | Inner median s1 / s2 / s3 |
|---|---|---|---|---|---|
| 0.6 | sens (22340391) | **−2.80** | 0.600 / 0.572 | **4 / 8 / 16** | 2 / 3 / 11 |
| 0.6 | uniform (22340392) | −3.34 | 0.599 / 0.582 | 3 / 6 / 12 | 3 / 6 / 13 |
| 0.6 | greedy-4 (22340232) | −4.16 | 0.600 / 0.582 | 3 / 6 / 11 | 3 / 6 / 16 |
| 0.6 | greedy-3 (22228973) | −4.68 | 0.600 / 0.453 | 2 / 5 / 13 | 2 / 6 / 13 |
| 0.6 | mild (22156062) | −5.06 | 0.600 / 0.453 | 2 / 5 / 13 | 2 / 6 / 13 |
| 0.6 | greedy-5 (22340233) | −5.06 | 0.594 / 0.436 | 2 / 5 / 11 | 2 / 5 / 16 |
| 0.8 | sens (22340393) | **−1.30** | 0.800 / 0.696 | **4 / 8 / 16** | 2 / 5 / 16 |
| 0.8 | mild (22156061) | −2.12 | 0.799 / 0.716 | 3 / 7 / 14 | 3 / 7 / 15 |
| 0.8 | uniform (22340394) | −2.26 | 0.799 / 0.716 | 3 / 7 / 14 | 3 / 7 / 15 |
| 0.8 | greedy-3 (22228972) | −2.38 | 0.788 / 0.663 | 3 / 6 / 14 | 3 / 6 / 16 |

**Read.**
- *Order.* At κ 0.6 the arms rank by residual width: full (sens) above 3 / 6 (uniform, greedy-4) above 2 / 5 (mild, greedy-3, greedy-5). At κ 0.8 sens is again the only arm with full residual streams. Inner widths do not order the arms: greedy-4 leaves the late inner convs full and finishes 0.82 below uniform. This is observational (the arms also differ in walk path), so wave 9 is the controlled read.
- *Same architecture.* At κ 0.8 mild and uniform land on identical widths, conv for conv, on both nets. Their final TEST differs by 0.14 pp on r56-w4 and 0.46 on r20-w2 (different walk paths and inherited weights, one final FT each). With §226's mild = greedy-3 at κ 0.6 (0.38 pp), r56-w4's same-architecture spread is 0.14–0.38 pp. At κ 0.8 the allocation comparison is in effect sens against a uniform cut.
- *Matched FLOPs.* Sens, uniform and greedy-4 keep FLOPs 0.57–0.58 and finish −2.80 / −3.34 / −4.16. Mild and greedy-3 remove more FLOPs (0.453) by halving stage 1, residual included.
- *Ties.* Sens's r56-w4 final TEST equals its walk TEST to the image at both κ (0.8618; 0.8768), while val moved (+0.36 / −0.24 pp) and the final-FT loss rose and settled. Checked: the runner scores TEST and val on the same fine-tuned copy with a fresh pass (no cache), and `tree_v10h` differs from `tree_v10` only in `alloc_walk.py` and the schedule getters. A coincidence, quoted as measured.
- *Margins at two decimals* (§211 / §212 are quoted to one decimal): κ 0.6 sens − mild **+2.26**, uniform − mild +1.72 (§230 has +2.30 / +1.76 from −5.1); κ 0.8 uniform − mild −0.14 (§229: −0.16). No call changes.
- *For the v10 read (registered bar rule, κ 0.6).* The bar fired on seed 42, so a v10 WIN at κ 0.6 is quoted as "learned allocation at heuristic level". "Beyond heuristic" needs v10 ≥ sens + 0.5 = **−2.30** on r56-w4 at the same landed keep (on the two-seed sens mean once wave 8 lands). At κ 0.8, sens −1.30 and uniform −2.26 go beside v10. Also read v10's landed residual widths against sens's: does the actor keep the residual streams?
- *Registered next: wave 9* (queue, 06:15, before submit). `SPECTRA_ALLOC_KIND=inner` holds every residual stream at full width and cuts the inner convs uniformly (the rule of Li et al. 2017, PFEC), on a new tree `tree_v10i` (`tree_v10h` + this kind only; 10 alloc tests green). Jobs: κ 0.6 **22341865** / **22341867** (seeds 42 / 43), κ 0.8 **22341866** / **22341870** (undershoot 0.04: the 0.02 plan keeps x0.801, above κ), κ 0.35 **22341871**. Call on r56-w4, gap = sens − inner on matched seeds: **STRUCTURAL** ≤ +0.3 at both κ, **SENS-ADDS** ≥ +0.5 at both κ, else **PARTIAL** (κ 0.35: ≤ +0.5 / ≥ +2.0).
- Do not lock. Never an agent row.

---

## 232. L3-ctrl: the paper's lr 0.01 final FT re-run from N3's saved DepGraph R56 candidates (**22341277**) — PRELIM; noise floor at 2.11× **0.02 pp raw / 0.20 honest** (under the 0.3 caveat); paired: **1-cycle passes** (+0.76, raw +0.22), cosine fails (+0.06)

Sitting 7 Oct, Lead 3, wave 7 (registered 04:10, before submit).
- *Recipe.* `tree_v10`, sbatch only, from-saved `SPECTRA_EVAL_FINAL_FT_FROM=tree_v9c/runs/job21767189/traj_models` (N3's four candidates) with the default final FT: SGD 0.01, m 0.9, wd 5e-4, per-epoch cosine, 100 epochs, batch 128, loader crop+flip; P, seed 42, deterministic, origin control. Same code path and RNG state as 22340234 (§223) and 22340387 (§224); only the schedule differs.
- *Run.* COMPLETED 1 h 24 m, 7 Oct 06:07, `ise-4090-15`, exit 0. Reader `final_ft_readout.py` on all four runs.

Each cell: final 5k / honest / 10k final.

| Point | Params / FLOPs | Walk (5k) | §157, lr 0.01 in the walk | **L3-ctrl**, lr 0.01 from saved | Cosine from 0.1 (§223) | 1-cycle (§224) |
|---|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | +0.08 | +0.04 / −0.40 / −0.03 | −0.06 / −0.68 / −0.03 | −0.26 / −0.96 / +0.04 | −0.04 / −0.12 / +0.07 |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | −0.22 | −0.44 / −0.58 / −0.46 | **−0.46 / −0.78 / −0.62** | −0.32 / −0.72 / −0.36 | **−0.24 / −0.02 / −0.44** |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.32 | −1.34 / −0.38 / −1.63 | −1.24 / −0.46 / −1.33 | −0.62 / +0.08 / −0.37 | −1.16 / +0.16 / −1.30 |
| `val_best` | 0.356 / 0.369 | −1.12 | −0.96 / −0.20 / n/a | −1.26 / −0.68 / n/a | −0.90 / −0.40 / n/a | −1.06 / +0.06 / n/a |
| origin change (5k / 10k) | 1 | 0 | +0.36 / +0.60 | +0.54 / +0.66 | +0.62 / +0.64 | +0.00 / +0.07 |

**Read.**
- *Noise floor (registered, reported).* |L3-ctrl − §157| on 5k final: 0.10 / **0.02** / 0.10 / 0.30 (size 0.60 / 2.11× / 2.57× / `val_best`); origin change 0.18; 10k 0.00 / 0.16 / 0.30. Honest at 2.11×: 0.20, mostly the origin. That is under 0.3 at 2.11×, so by the registered rule no noise caveat goes beside the single-run Lead 3 calls. The same recipe on the same saved weights still moves 0.1–0.3 pp from run to run, and the lr 0.01 origin control has now moved +0.42 / +0.36 / +0.86 / +0.54 in four runs (§153 / §157 / §220 / here).
- *Paired call, 1-cycle: **passes**.* Honest Δ −0.02 − (−0.78) = **+0.76** ≥ +0.5, and raw −0.24 ≥ −0.46 (+0.22). Two of its three registered reads pass (§224, paired). The replicate, L3b-rep 22341280 on τ-off's candidates against §220, is R.
- *Paired, cosine from 0.1 (reported only; it failed its first read).* Honest Δ +0.06, raw +0.14: fails again at 2.11×. At 2.57× it helps (raw +0.62, 10k +0.96), as in the TREND of §228.
- *What 1-cycle's margin is made of.* At 2.11× its raw gain is +0.20 / +0.22 on 5k and +0.02 / +0.18 on 10k (vs §157 / L3-ctrl), inside the run-to-run spread above. Most of the honest margin is the origin control: lr 0.01 lifts the unpruned origin by +0.36 to +0.86, 1-cycle by +0.00. The registered rule counts that. If the replicate passes too, the caption note must say the gain is measured against a schedule that also improves the unpruned net.
- Do not lock. Never an agent row.

---

## 233. Allocation-following walk, uniform control, DepGraph ResNet-56 C10 landed params 0.47 (**22340524**) — PRELIM; lever waits on **22340523**

Pair of **22340523** (still R). Sitting 7 Oct, `tree_v10h`, `SPECTRA_ALLOC_KIND=uniform`, `SIZE_MATCH=param:0.47`, 6 passes, P, 100-ep origin FT. COMPLETED 2 h 30 m, 7 Oct ~06:13, exit 0. Reader `final_ft_readout.py` on the run dir. Call (queue): SURVIVES ≥ +0.5 / ABSORBED ≤ +0.15 on r56 (sens − uniform) once 523 lands.

| Point | Params / FLOPs | Walk (5k) | Final 5k | Honest | 10k |
|---|---|---|---|---|---|
| `val_best` = `size_param0.47` | 0.465 / 0.472 | −0.56 | **−0.74** | −0.64 CROSS-OFF | n/a (val-selected) |
| origin | 1.000 / 1.000 | 0 | +0.46 | — | +0.64 |

**Read.** Uniform lands at **−0.74 @ 0.465 / 0.472**. N3 mild at the nearby 2.11× point is −0.44 @ 0.470 / 0.463 (§157; L3-ctrl −0.46, §232). Different walk, not a registered comparison. 10k is n/a because the size point is `val_best`. Do not quote a lever. Never an agent row.

---

## 234. DepGraph's own allocation beside ours (zero GPU; the pruned module trees DepGraph's benchmark printed in h2h **21943448**; the DG pair's plans; N3's saved candidates) — PRELIM read; DepGraph keeps the **early residual streams near full**; wave 10 (architecture transplant) registered

Sitting 7 Oct ~06:50. Source: `tree_v9d/runs/h2h_depgraph/job_21943448/*.log`, DepGraph's official Torch-Pruning pipeline on our 4090 from the same released checkpoints (2 Oct). DepGraph's widths are parsed with `scripts/widths_from_module_print.py`; ours are read with `scripts/arch_widths_readout.py`. Our counter reproduces DepGraph's sizes on its own trees exactly: R56 params 0.5044 / FLOPs 0.4735 (its log: 50.44 % / 47.37 %, 2.11×); VGG-19 C100 0.0608 / 0.1104 (6.08 % / 11.08 %; our h2h run landed at **9.02×**, the paper's row is 8.84×).

DepGraph R56 C10 (origin 16 / 32 / 64 per stage). Residual width / inner conv widths min–median–max:

| Allocation | Params / FLOPs | Stage 1 | Stage 2 | Stage 3 | Final, 10k |
|---|---|---|---|---|---|
| DepGraph 2.11× (its pipeline, h2h) | 0.504 / 0.474 | **13** / 4–8–11 | **31** / 7–12–28 | 42 / **34–54–62** | **+0.24** (93.77, last epoch) |
| N3 mild, `size_flop0.47` (§157) | 0.470 / 0.463 | 11 / 11–11–11 | 21 / 21–21–21 | 42 / 42–47–47 | −0.46 |
| Uniform alloc walk 22340524 (§233) | 0.465 / 0.472 | 11 / 11–11–11 | 22 / 22–22–22 | 44 / 27–45–45 | n/a (val-selected) |
| Sens alloc plan 22340523 (R) | plan 0.447 | **16** / 3–6–9 | **32** / 3–5–32 | **64** / 12–30–64 | pending |

VGG-19 C100 at 9.02× (h2h; origin 64, 64 | 128, 128 | 256 ×4 | 512 ×4 | 512 ×4): **4**, 30 | 41, 108 | 102, 163, 67, 237 | 265, 33, 29, 15 | 19, 24, 20, 26. Final 70.53 (last epoch), **−2.97** against its 73.50. DepGraph's "Best Acc" lines are selected on the test set and are not quoted.

**Read.**
- *Three allocations.* DepGraph's allocation is neither uniform nor residual-full. It keeps the stage 1–2 residual streams near full (13 / 16, 31 / 32) and cuts their inner convs hard and unevenly (4–11 of 16, 7–28 of 32). In stage 3 it does the opposite: the residual drops to 42 / 64 while the inner convs stay wide (median 54 of 64). N3's mild walk and the uniform walk are both close to a uniform 2/3 cut, residual included. The sens plan holds every stream full and cuts the inner convs deeper.
- *Prior evidence.* §162 (N2-streams: mild with `SPECTRA_PROTECT_STREAMS=1`, no crop+flip) was **+2.10 pp** paired by params against its matched no-aug control on r56-w4. Same direction as §231; never re-run under crop+flip, which wave 9 now does.
- *The gap to explain.* At 2.11× N3 sits 0.70 pp (10k) under DepGraph's own pipeline on the same checkpoint. DepGraph differs in allocation, in ranking (group-L2 after sparsity learning), in a sparsity-learning pre-training stage, and in its fine-tune. The transplant moves the allocation alone.
- *Registered next: wave 10* (queue, 06:50, before submit). `SPECTRA_ALLOC_KIND=widths` gives every group the width DepGraph's printed tree names for it. It runs on a new tree `tree_v10j` (`tree_v10i` + this kind; 12 alloc tests green), with our L1 ranking, walk recovery and final FT. R56 uses the 9-rate menu (1.0 0.95 … 0.6), which reaches 0.507 / 0.479 with 6 of 30 groups one channel wide; the 5-rate menu would miss 14 by up to 3. VGG-19 C100 uses the 12-rate menu down to 0.3 and `SPECTRA_STEM_ROWS=0`, because DepGraph cuts the first conv 64 → 4, which the stem rule would forbid. It reaches 0.0605 / 0.109. Call on R56 (10k, size point): lift = transplant − (−0.54) − 0.15, where −0.54 is N3's two lr-0.01 final FTs at 2.11× (§157 −0.46, §232 −0.62) and 0.15 is a size credit (FLOPs 0.479 vs 0.463 at N3's own ~9.6 pp per unit FLOPs between its 2.11× and 2.57× final-FT points). **ALLOCATION** if lift ≥ +0.5 (≥ 2/3 of the 0.78 pp to DepGraph's +0.24); **NOT-ALLOCATION** if ≤ +0.2; **PARTIAL** between. VGG-19 C100 (10k), reported against DepGraph's −2.97: **MATCH** if ≥ −3.47.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 235. Which epoch the final fine-tune keeps (zero GPU; census of 191 final FTs in the `tree_v9b`–`tree_v10h` logs; τ-off reads 22341051 / 22341280) — PRELIM; after a crop+flip walk the default **keeps epoch 1** on most lr-0.01 DepGraph R56 points and on every 1-cycle run; **1-cycle VOID**; wave 11 (`select=last`) registered

- *Mechanism.* `ClassificationHandler.train_model` keeps the lowest-train-loss epoch and restores it after the last one (`select=train_loss`). The final FT uses the same rule: patience is epochs + 1, so it runs all 100 epochs, then restores. A net recovered by a crop+flip walk often starts below the train loss that 100 SGD epochs with weight decay 5e-4 end at. The restore then brings back epoch 1: the walk plus one epoch. Origins at lr 0.01 / 0.1 keep a late epoch, so "honest" compares one pruned epoch with 100 origin epochs. Selection is on the train loss, never on test, so these TEST numbers are legitimate; the label "100-epoch final FT" is not.
- *Why the start sits below the endpoint (N3's log, 07:55).* The walk's per-step fine-tune is **Adam lr 1e-3, plateau schedule, weight decay 0**, 40 epochs, patience 10 (150 calls: 116 ran all 40, 34 stopped early). Late in the walk it ends at train loss 0.0004–0.0008 (lr decayed to 2e-5–6e-5). The final FT is SGD lr 0.01 with weight decay 5e-4, whose loss settles near 0.006–0.008 (§236's epoch-100 losses). So after a long walk no epoch of the final FT can beat epoch 1 on train loss. The origin starts at 0.023 and descends (0.0026 at epoch 100). A no-aug walk sees a different train distribution from the crop+flip final FT and starts above it, which is why §153 kept late epochs.
- *Census.* The log prints epochs 1, 5, 10, …; EARLY = the lowest printed loss is at epoch 1 or 5 and `best_loss` matches it. Pruned rows:

| Final FT | Net | EARLY / all | Where |
|---|---|---|---|
| lr 0.01 cosine (paper caption) | DepGraph R56 C10 | **14 / 30** | N3 21767189 **4/4** (§157), τ-off 22288423 3/4 (§220; `val_best` late), L3-ctrl 22341277 4/4 (§232), §203 1/4, §217 1/1, §233 1/1; §153 (no-aug walk) and 21767190 / 92 0/4 |
| lr 0.01 cosine | chenyaofo R56 C10 | **3 / 3** | twins 21809595 (§164) |
| lr 0.01 cosine | DepGraph VGG-19 C100 | 3 / 6 | all three in N4 21737105 (§155); §149 21729551 late (epochs ~90–95) |
| lr 0.01 cosine | thin pair | 15 / 48 | 21729550, 21730499, 22155997 and a smoke run; every landed-κ row since §211 kept a late epoch |
| lr 0.01 cosine | VGG-16 C10 | 0 / 7 | — |
| cosine from lr 0.1 | DepGraph R56 C10 | **0 / 12** | §223, §228 (epochs ~95–100) |
| 1-cycle (warmup 30 to lr 0.1, cosine) | DepGraph R56 C10 | **8 / 8**, both origins too | §224 22340387, L3b-rep 22341280 |

- *1-cycle: **VOID**.* All three Lead 3 reads kept epoch 1 on every pruned point **and** on the origin, one epoch at ≤ lr 0.0033 of warmup. The L3b replicate 22341280 (τ-off's candidates, 7 Oct 06:39, exit 0) passes the registered rule numerically at 2.11×: honest −0.16 vs §220's −1.00 (Δ +0.84), raw −0.86 ≥ −0.90, 10k −0.73 vs −0.94; origin change +0.06. Like §224 (+0.56) and §232's paired read (+0.76), it compares no fine-tune against lr 0.01's genuine 100-epoch origin lift. Withdrawn: the sitting's 04:10 correction ("§224 fires ADOPT") is right as arithmetic and wrong as evidence. 1-cycle does not enter the caption on these runs.
- *lr 0.01 rows.* §157's M4 row (2.11× 10k −0.46), §220 and §232 are walk + 1 epoch. "Long FT CROSS-OFF" on those rows (§155, §157, §164, §220) measures one pruned epoch against the origin's 100. §155's CROSS-OFF vs §149 compares epoch 1 with epoch ~95, so it is confounded too.
- *Cosine from lr 0.1 is the only genuine long FT on these points.* N3's walk: 2.11× 10k **−0.36**, 2.57× **−0.37** (§223). τ-off's walk: **+0.01** / **−0.36** (§228, 22341051). §228's 2.57× TREND compares a genuine FT with walk + 1 epoch. Learning rate and selection are confounded until wave 11 lands.
- *Not affected:* VGG-16 rows, the lr 0.1 rows, §149, §153 (no-aug walk), landed-κ thin rows (§211–§231).
- *Registered: wave 11* (queue, 07:10, before submit). New tree `tree_v10k` = `tree_v10j` + default-off `SPECTRA_EVAL_FINAL_FT_SELECT=last`: the final FT keeps its last epoch (`train_model(keep_last=True)`; the runner passes it only when the flag is set; 58 staged tests green, 3 new). From-saved re-FTs with the paper recipe otherwise unchanged (lr 0.01 cosine, 100 epochs, P, seed 42, origin control). The training trajectory is the train-loss run's own; only the kept epoch changes. Cells: N3 21767189, τ-off 22288423, N4 21737105, twins 21809595 (VGG-16 = negative control, already late), 1-cycle-last on N3 and τ-off, and the DepGraph allocation rows (§233; 22340523 and wave 10 by `afterok`).
- *Call (10k; 2.11× = `size_flop0.47`, 2.57× = `size_flop0.39`).* Δsel = select=last − the same walk's train-loss final (N3 vs §157, τ-off vs §220). **REQUOTE** if Δsel ≥ +0.3 on both walks at 2.11× or at 2.57×: the M4 row and every early-epoch lr-0.01 row are re-finalised with `select=last` before quoting. **STANDS** if Δsel ≤ −0.3 on both walks at both points: keep the numbers, caption "walk + 1 epoch (train-loss selection)". **NEUTRAL** otherwise: keep the numbers; the caption discloses the selection.
- *Reported, not gating.* Honest, now 100 epochs against 100. lr 0.01-last against cosine-0.1: within 0.3 at 2.57× on both walks means §228's TREND was the selection, not the learning rate. N4 against §149 on equal epochs re-reads §155. Twins: R56 Δsel, with VGG-16 |Δsel| > 0.5 meaning the R56 read is noise-limited. 1-cycle-last by the Lead 3 rule against lr 0.01-last (honest Δ ≥ +0.5 and raw ≥, at 2.11×, both walks). Allocation rows under `select=last` beside their lr 0.01 rows.
- *Noise.* §232's 0.02 pp (2.11×, 5k) compares epoch 1 with epoch 1. Endpoint noise after 100 epochs is unmeasured. A Δsel within 0.1 of a bar is "unresolved", not a call.
- *Registered: wave 11b* (queue, 07:50, before submit). (1) **Endpoint noise:** N3 select=last again with `SPECTRA_SEED=43`. The split is set by `SPECTRA_SPLIT_SEED` (0) alone, so only data order and augmentation change. Read: \|s43 − s42\| at each point and on the origin change, 5k and 10k. If it reaches 0.3 (10k) at a gating point, the wave 11 call there is "unresolved" unless both walks clear the bar by more than that noise. (2) **Learning rate at the endpoint, other architectures:** cosine from lr 0.1 with select=last on N4 (VGG-19 C100) and on the zoo twins. Read: lr 0.1-last − lr 0.01-last per point. Reported: if it is ≥ +0.3 on N4 at both size points and on DepGraph R56 at 2.57×, "a large-LR endpoint FT helps across architectures" (Le & Hua); otherwise the TREND stays a DepGraph-R56 observation.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 236. Allocation-following walk, A0's sens rule, DepGraph ResNet-56 C10 landed params 0.47 (**22340523**) — PRELIM; lever **WEAK** (+0.40 vs uniform §233); every residual stream kept full, FLOPs 0.398 vs 0.472

Pair of §233 (uniform 22340524). Sitting 7 Oct, `tree_v10h`, `SPECTRA_ALLOC_KIND=sens` (α 0.5), `SIZE_MATCH=param:0.47`, 6 passes, P, origin control. COMPLETED 4 h 13 m, 7 Oct 07:36, `ise-4090-12`, exit 0, no fallback. Plan line: `sens alpha=0.5 plan keeps x0.447 … over 30 groups; group keep min 0.10 median 0.36 max 1.00`. Reader `final_ft_readout.py`; widths `arch_widths_readout.py`.

| Arm | Params / FLOPs | Residual / inner min–median–max, stages 1 · 2 · 3 | Walk (5k) | Final 5k | Honest | 10k |
|---|---|---|---|---|---|---|
| **sens** `val_best` = `size_param0.47` | 0.469 / **0.398** | **16** / 4–6–9 · **32** / 7–7–32 · **64** / 16–30–64 | −0.32 | **−0.34** | −0.62 | n/a (val-selected) |
| uniform §233 | 0.465 / 0.472 | 11 / 11 · 22 / 22 · 44 / 27–45–45 | −0.56 | −0.74 | −0.64 | n/a |
| origin (sens run / uniform run) | 1 | 16 · 32 · 64 | 0 | +0.60 / +0.46 | — | +0.56 / +0.64 |

**Read.**
- *Call (registered, queue).* sens − uniform at the size point, 5k final: −0.34 − (−0.74) = **+0.40** → **WEAK**, between ABSORBED (≤ +0.15) and SURVIVES (≥ +0.5). That is the same direction as the thin pair (+0.82 / +0.96 / +0.54, §227 / §229 / §230). Honest is flat (−0.62 vs −0.64), so the gain is in the walk (+0.24 at walk TEST) more than in the final FT.
- *Kept epochs (§235).* Both arms' final FTs kept **epoch 1**: sens loss 0.00195 at epoch 1 and 0.00822 at epoch 100; uniform 0.00424 and 0.00624. Both origins kept ~epoch 100 (0.023 → 0.0026). The walk drives the pruned net's train loss about ten times below the unpruned checkpoint's. The two arms are compared like-for-like (walk + 1 epoch each); wave 11 **22342665 / 22342666** re-read them at the endpoint.
- *At equal params, sens keeps 16 % fewer FLOPs* (0.398 vs 0.472, i.e. 2.51× vs 2.12×), because it cuts inner convs instead of streams. On the FLOPs axis it sits beside N3's 2.57× point (FLOPs 0.380): −0.34 against N3's −1.34 (§157) / −1.24 (§232), both walk + 1 epoch. Different walks; the 0.018 FLOPs gap is worth ~0.17 pp at N3's slope. Visible, not a call.
- No 10k: the size point is `val_best`. The DepGraph comparison waits on the transplant (wave 10) and wave 11.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 237. First v10 freeze TEST, ep0127, landed κ 0.8 (**22341736**) — PRELIM; vs mild §211 r56 **−0.78**; residual **3 / 7 / 14** (mild/uniform, not sens); M1-v10 waits on **22341737**

Skip-train `eval_c10_thin_traj` of **22156116** `snapshots/ep0127` (first freeze after PPO-20 with `vs_mild ≥ +0.5`; the probe is a gate, never a result). `tree_v10`, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, `FIXED_TARGET=1`, `STATE_SENS=1`, det=1, `SIZE_MATCH=param:0.8`, 100-ep origin FT, seed 42, menu 1.0/0.9/0.8/0.7/0.6 all L1. COMPLETED 2 h 44 m, 7 Oct ~08:26, `cs-4090-10`, TB 0, exit 0. Start-check green (`policy=actor`, `fixed_target=1`, `state_sens=1`, 6-pass). Reader `final_ft_readout.py`. Control = mild-landed **§211**. Pair κ 0.6 **22341737** still R — **do not call M1-v10**.

| Net | Actor params / FLOPs | Walk | Final 5k | vs mild §211 | vs uniform §229 | vs sens §227 |
|---|---|---|---|---|---|---|
| r20-w2 | 0.774 / 0.818 | −0.32 | **+0.54** | **+0.94** (−0.4) | — | — |
| r56-w4 | 0.799 / 0.716 | −2.52 | **−2.88** | **−0.78** (−2.1) | −0.62 (−2.26) | −1.58 (−1.30) |
| origin r20 / r56 | 1 | 0 | +3.32 / +0.28 | — | — | — |

**Census (r56-w4):** 25 cuts at **0.9**, 22 identity at 1.0, 1 landing ~1.0. Distinct prune actions = **1**. r20: 6 at 0.9, 11 identity, 1 landing. Menu 0.8 / 0.7 / 0.6 **never played**. Not a Stage-4 0.8 clone; it is the mild path (0.9 + skip) at this κ.

**Residual widths (r56-w4 `val_best`):** **3 / 7 / 14** — the mild/uniform architecture (§231), not sens 4 / 8 / 16. Inner convs are uniform 3 / 7 then 14–16. Same landed params and FLOPs as mild/uniform (0.799 / 0.716).

**Read.** M1-v10 WIN / NEG / FLAT is registered on r56 κ **0.6**. This arm is the κ 0.8 quote-beside. Actor is 0.78 pp behind mild and 1.58 behind sens at equal params; residual streams are cut, so it did not learn the A0 plan here. r20 +0.94 is the disaster guard, not a veto. 10k n/a (`val_best`). Do not lock. Never quote the probe.

---

## 238. Allocation-following walk, uniform control, landed κ 0.35, thin pair (**22340637**) — PRELIM; r56-w4 **−7.90 @ 0.349**; lever waits on **22340636**

Pair of 22340636 (sens, still R). Sitting 7 Oct wave 4, `tree_v10h`, `SPECTRA_ALLOC_KIND=uniform`, `SIZE_MATCH=param:0.35`, 5-rate menu, P, origin control. COMPLETED 3 h 26 m, 7 Oct 08:38, `ise-4090-21`, exit 0, no fallback. Call (queue): SURVIVES ≥ +2.0 / ABSORBED ≤ +0.5 on r56-w4 (sens − uniform). Bar: mild-landed κ 0.35 **22340796** (R).

| Net / point | Params / FLOPs | Residual / inner, stages 1 · 2 · 3 | Walk (5k) | Final 5k | Honest | 10k |
|---|---|---|---|---|---|---|
| **r56-w4** `val_best` = `size_param0.35` | 0.349 / 0.331 | 2 / 2 · 5 / 5 · 9 / 9–10–10 (origin 4 · 8 · 16) | −7.60 | **−7.90** | −0.90 | n/a (val-selected) |
| r20-w2 `size_param0.35` | 0.331 / 0.574 | 2 / 2 · 2 / 2 · 4 / 3–5–5 (origin 2 · 4 · 8) | −10.60 | **−9.80** | −2.56 | −9.82 |
| r20-w2 `val_best` | 0.417 / 0.608 | 2 / 2 · 2 / 2 · 5 / 5 | −8.64 | −7.26 | −1.98 | n/a |
| origin r56-w4 / r20-w2 | 1 | — | 0 | +0.60 / +3.36 | — | +0.45 / +3.67 |

**Read.**
- Uniform cuts r56-w4's residual streams to 0.5–0.6 of origin (2 / 4, 5 / 8, 9 / 16), as it cuts them at κ 0.6 / 0.8 (§229 / §230, §231). r20-w2's plan stops at params 0.417 (`every group at its target … strongest legal cut`), so its κ 0.35 point is a later step.
- *Kept epochs (§235).* Every final FT here kept a **late** epoch (train loss falls to epoch 100: r56-w4 0.517 → 0.490, r20-w2 1.245 → 1.188), as on every landed-κ thin row. Honest is 100 epochs against 100.
- Do not quote a lever until 22340636 lands. Never an agent row.

---

## 239. Wave 11 select=last: paper recipe keeping the last epoch, N3's saved DepGraph R56 candidates (**22342659**) — PRELIM; N3 Δsel 10k **+0.15 / +0.43** at 2.11× / 2.57×; call waits on **22342660**

Sitting 7 Oct wave 11, `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from-saved N3 `job21767189/traj_models`, otherwise the paper recipe (SGD 0.01, cosine, 100 epochs, P, seed 42, origin). COMPLETED 1 h 24 m, 7 Oct 08:59, `cs-4090-08`, exit 0, TB 0. Log prints `keep=last`. Reader `final_ft_readout.py`. Pair: τ-off **22342660** still R. Call (queue, 10k): Δsel = this final − §157's train-loss final. **REQUOTE** if Δsel ≥ +0.3 on both walks at 2.11× or at 2.57×; **STANDS** if ≤ −0.3 at both points on both walks; **NEUTRAL** otherwise.

| Point | Params / FLOPs | Walk (5k) | §157 5k / 10k | **select=last** 5k / 10k | Δsel 10k |
|---|---|---|---|---|---|
| size_flop0.60 | 0.638 / 0.599 | +0.08 | +0.04 / −0.03 | +0.22 / **+0.17** | **+0.20** |
| size_flop0.47 (2.11×) | 0.470 / 0.463 | −0.22 | −0.44 / **−0.46** | −0.30 / **−0.31** | **+0.15** |
| size_flop0.39 (2.57×) | 0.382 / 0.380 | −1.32 | −1.34 / **−1.63** | −1.18 / **−1.20** | **+0.43** |
| `val_best` | 0.356 / 0.369 | −1.12 | −0.96 / n/a | −0.90 / n/a | n/a |
| origin | 1 | 0 | +0.36 / +0.60 | +0.34 / +0.42 | −0.18 |

**Read.**
- *This walk.* 2.57× Δsel **+0.43** clears +0.3; 2.11× **+0.15** does not. REQUOTE needs the τ-off walk at the same point. STANDS cannot fire from these signs. Do not change the M4 caption on one walk.
- *M4 if this number were used.* 2.11× 10k would move −0.46 → −0.31, still 0.55 behind DepGraph +0.24.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 240. Wave 11 call: select=last on τ-off's saved DepGraph R56 candidates (**22342660**), beside N3 (§239) — PRELIM; **NEUTRAL** at 2.11×, 2.57× **unresolved**; the M4 numbers stay, disclosed; cosine-0.1's lead at 2.57× is **not** the selection

Sitting 7 Oct wave 11, `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from-saved τ-off `tree_v9d/runs/job22288423/traj_models` (PATH-SAME widths as N3, §220), otherwise the paper recipe (SGD 0.01, cosine, 100 epochs, P, seed 42, origin). COMPLETED 1 h 25 m, 7 Oct 09:53, `cs-4090-10`, exit 0, TB 0. The call was read at 09:20 from the two gating points, finished by 09:01; the rest landed by 09:53 and changes nothing. Start check green on all five final FTs: env `last`, `select=last`, "kept the last epoch", `keep=last`. Reader `final_ft_readout.py`.

| Point | §220 train-loss 5k / 10k | **select=last** 5k / 10k | Δsel 10k | Honest last / §220 | cosine 0.1 §228, 10k |
|---|---|---|---|---|---|
| size_flop0.60 | −0.20 / −0.16 | −0.18 / −0.06 | +0.10 | −0.64 / −1.24 | +0.05 |
| size_flop0.47 (2.11×) | −0.90 / −0.94 | −0.60 / **−0.54** | **+0.40** | −0.12 / −1.00 | +0.01 |
| size_flop0.39 (2.57×) | −1.34 / −1.52 | −1.40 / **−1.31** | **+0.21** | −0.46 / −0.98 | −0.36 |
| `val_best` (keep 0.123) | −5.02 / n/a | −5.06 / n/a | n/a | +0.08 / −0.46 | n/a |
| origin | +0.86 / +0.73 | **+0.28** / +0.64 | −0.09 | — | +0.82 |

| Point | N3 Δsel (§239) | τ-off Δsel | Call (queue, registered 07:10) |
|---|---|---|---|
| 2.11× | +0.15 | +0.40 | **NEUTRAL**: N3 is 0.15 short of +0.3, outside the 0.1 band |
| 2.57× | +0.43 | +0.21 | **unresolved**: τ-off is 0.09 short, inside the band |

**Read.**
- *Call.* REQUOTE needs both walks ≥ +0.3 at one point, and no point has that. STANDS needs ≤ −0.3 everywhere, and every Δsel is positive. So the M4 numbers stay, with the disclosure "walk + 1 epoch (§235); a 100-epoch lr-0.01 endpoint adds +0.15 to +0.43 at 10k on two walks". 22342767 (N3 select=last, seed 43) can make 2.11× unresolved as well, if N3 moves ≥ 0.3 there; it cannot make REQUOTE fire. *(10:30, §244: it moved 0.05 at 2.11× and 0.14 at 2.57×, so the rule did not fire and the call stands.)*
- *§228's TREND is not the selection.* lr 0.01-last against cosine-0.1 at 2.57×: N3 −1.20 vs −0.37 (0.83 apart), τ-off −1.31 vs −0.36 (0.95). Both gaps exceed 0.3, the registered bar. At 2.11× they are level on N3 (−0.31 vs −0.36) and 0.55 apart on τ-off (−0.54 vs +0.01). A genuine lr-0.01 endpoint recovers at most 0.43 of those gaps. Cosine from 0.1 leads on merit at 2.57× on both walks.
- *Honest, 100 epochs against 100.* N3 (§239): −0.42 at 2.11×, −0.20 at 2.57×, −0.20 at keep 0.60, −0.12 at `val_best`. τ-off: −0.12, −0.46, −0.64, +0.08. At lr 0.01 the pruned nets gain less over their walk than the origin gains from 100 epochs.
- *Origin noise.* The identical lr-0.01 FT of the same unpruned origin gave +0.28 here and +0.86 in §220's run (5k). Only the RNG state differs: a fresh process here, after the walk there. At 10k they are +0.64 / +0.73. So at 5k one origin FT moves ~0.6 pp, and every honest number carries it. Earlier runs of this origin: +0.42 / +0.36 / +0.86 (§153 / §157 / §220). Honest at 5k is noise-limited at about ±0.5; gate on 10k raw where a call allows it.
- *Slide line (tracker §6), revised 09:30.* It now shows both genuine recipes, the lr-0.01 endpoint and cosine-0.1, so no recipe is picked on TEST. Wave 12 (**22343160 / 65**, seed 43) measures cosine-0.1's seed noise; 22342767 measures lr-0.01-last's.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 241. Allocation-following walk, sens α 1.0 ("sens2"), landed κ 0.6, thin pair (**22340638**) — PRELIM; dose-response **flat**: r56-w4 **−2.80 @ 0.600**, identical to α 0.5 (§230); r56-w4's residual streams full under both

Sitting 7 Oct wave 4, `tree_v10h`, `SPECTRA_ALLOC_KIND=sens`, `SPECTRA_ALLOC_ALPHA=1.0` (α 0.5 is §230's sens), `SIZE_MATCH=param:0.6`, 6 passes, P, origin control. COMPLETED 3 h 32 m, 7 Oct 09:26, `ise-4090-02`, exit 0, TB 0, no fallback. Registered as dose-response, reported only. Reader `final_ft_readout.py` + `arch_widths_readout.py`.

| Net | Arm | Params / FLOPs | Steps | Residual s1 / s2 / s3 | Inner median s1 / s2 / s3 | Final 5k | Honest |
|---|---|---|---|---|---|---|---|
| **r56-w4** | **sens2 α 1.0** | 0.600 / 0.552 | 226 | **4 / 8 / 16** | 2 / 2 / 9 | **−2.80** | −0.44 |
| r56-w4 | sens α 0.5 (§230) | 0.600 / 0.572 | 169 | **4 / 8 / 16** | 2 / 3 / 11 | −2.80 | — |
| r56-w4 | uniform (§230) | 0.599 / 0.582 | 96 | 3 / 6 / 12 | 3 / 6 / 13 | −3.34 | — |
| r20-w2 | sens2 α 1.0 | **0.551** / 0.783 | 39 | 2 / 4 / 4 | 2 / 4 / 8 | −6.12 | −2.16 |
| r20-w2 | sens α 0.5 (§230) | 0.595 / 0.800 | 40 | 2 / 4 / 5 | 2 / 4 / 5 | −2.92 | — |
| origin r56-w4 / r20-w2 | sens2 run | 1 | — | — | — | +0.86 / +3.20 | — |

**Read.**
- *Dose-response.* Doubling α concentrates the inner cuts (stage-2 / 3 medians 2 / 9 against 3 / 11) and saves FLOPs (0.552 against 0.572). It keeps the residual streams full as α 0.5 does, and lands on the same TEST to two decimals. The lever does not grow with the sensitivity dose at κ 0.6. That fits the structural reading (§231: full residual streams), which wave 9's `inner` arm (22341865 / 67) tests directly.
- *r20-w2 is not at equal size.* The plan stopped at params 0.655 (`every group at its target … strongest legal cut`). Its strongest legal cut overshot to 0.551, so −6.12 is 0.044 params below α 0.5's point. That is a guard row, not a lever read; group keep min 0.10 under α 1.0.
- *Kept epochs (§235).* Late on every final FT (r56-w4 `val_best` loss 0.279 at epoch 1, 0.269 at 100, best 0.268). Honest is 100 epochs against 100.
- *Seed caveat (added with §242).* One seed per α. α 0.5 moved 0.72 pp between seeds 42 and 43 (§242), so "identical to two decimals" is inside seed noise. "Flat" means that no dose effect is visible beyond ±0.7 pp.
- Do not lock. Never an agent row.

---

## 242. Allocation-following walk, sens α 0.5, landed κ 0.6, thin pair, **seed 43** (**22341281**) — PRELIM; r56-w4 **−2.08 @ 0.597** (seed 42: −2.80, §230): seed spread **0.72 pp** on one rule and nearly one architecture; lever and bar wait on **22341282 / 22341278**

Sitting 7 Oct wave 8, `tree_v10h`, `SPECTRA_ALLOC_KIND=sens` (α 0.5), `SIZE_MATCH=param:0.6`, `SPECTRA_SEED=43` (data order and crop / flip; the val / TEST split is `SPECTRA_SPLIT_SEED`'s, unchanged), 6 passes, P, origin control. COMPLETED 3 h 11 m, 7 Oct 09:51, `ise-4090-02`, exit 0, TB 0, no fallback. Call (queue): lever (sens − uniform) and bar (sens − mild-landed) on the **two-seed mean**. Pairs: uniform s43 **22341282** (R from 09:50), mild-landed κ 0.6 s43 **22341278** (R).

| Net | Seed | Params / FLOPs | Steps | Residual s1 / s2 / s3 | Inner min / median / max, s3 | Walk 5k | Final 5k | Honest |
|---|---|---|---|---|---|---|---|---|
| **r56-w4** | **43** | 0.597 / 0.568 | 169 | **4 / 8 / 16** | 5 / 10 / 16 | −2.74 | **−2.08** | +0.66 |
| r56-w4 | 42 (§230) | 0.600 / 0.572 | 169 | **4 / 8 / 16** | 4 / 11 / 16 | — | −2.80 | — |
| r20-w2 | 43 | 0.595 / 0.800 | 40 | 2 / 4 / 5 | 4 / 6 / 8 | −4.90 | −3.46 | −2.00 |
| r20-w2 | 42 (§230) | 0.595 / 0.800 | 40 | 2 / 4 / 5 | 5 / 5 / 8 | — | −2.92 | — |
| origin r56-w4 / r20-w2 | 43 | 1 | — | — | — | 0 | +0.00 / +3.44 | — |

**Read.**
- *Seed spread.* The same rule lands r56-w4 0.72 pp apart across seeds, on nearly the same architecture: the residual streams are full in both runs, and stage 1–2 inner widths match. At fixed architecture, one walk plus one final FT moves by up to ~0.7 pp. Mild and greedy-3 were 0.38 apart on one architecture (§226).
- *What that does to the single-seed calls.* The WEAK levers (+0.54 here at κ 0.6, +0.96 at κ 0.8, +0.40 on DepGraph R56 §236) and the flat dose-response (§241) all sit inside that spread. They stay PRELIM until the two-seed means. The bar over mild at κ 0.6 (+2.26, §231) is three times the spread, so it does not depend on one seed. Its seed-43 read is 22341278.
- *Two-seed sens mean, r56-w4:* −2.44 @ ~0.60. The lever needs uniform s43. Under §231's rule for the v10 read, "beyond heuristic" at κ 0.6 becomes v10 ≥ −2.44 + 0.5 = **−1.94**.
- *Kept epochs (§235).* Late on every final FT (r56-w4 `val_best` 0.283 at epoch 1, 0.275 at 100, best 0.270; origin 0.299 → 0.188). The r56 origin gained 0.00 pp from 100 epochs at this seed; §241's seed-42 run gained +0.86. Honest therefore moves with the origin's seed as well.
- Do not lock. Never an agent row.

---

## 243. Allocation-following walk, sens α 0.5, landed κ 0.35, thin pair (**22340636**) — PRELIM; lever **WEAK** (+1.90 vs uniform §238, 0.10 short of SURVIVES); r56-w4 **−6.00 @ 0.338 / 0.409**; r20-w2 guard **−3.26**

Sitting 7 Oct wave 4, `tree_v10h`, `SPECTRA_ALLOC_KIND=sens` (α 0.5), `SIZE_MATCH=param:0.35`, 5-rate menu, 6 passes, P, origin control, seed 42. COMPLETED 5 h 2 m, 7 Oct 10:10, `ise-4090-19`, exit 0, TB 0, no fallback. Call (queue, registered 02:57): on r56-w4, sens − uniform **SURVIVES** ≥ +2.0 / **ABSORBED** ≤ +0.5 / **WEAK** between; r20-w2 reported. Bar: mild-landed κ 0.35 **22340796** (R).

| Net / point | Arm | Params / FLOPs | Steps | Residual s1 / s2 / s3 | Inner median s1 / s2 / s3 | Walk 5k | Final 5k | Honest | 10k |
|---|---|---|---|---|---|---|---|---|---|
| **r56-w4** `val_best` = `size_param0.35` | **sens** | 0.338 / 0.409 | 284 | **4 / 8 / 15** | 2 / 2 / 5 | −8.42 | **−6.00** | +1.96 | n/a (val-selected) |
| r56-w4 `size_param0.35` | uniform (§238) | 0.349 / 0.331 | 102 | 2 / 5 / 9 | 2 / 5 / 10 | −7.60 | −7.90 | −0.90 | n/a |
| r20-w2 `size_param0.35` | sens | 0.340 / 0.631 | 60 | 2 / 3 / 3 | 2 / 3 / 5 | −14.08 | **−13.06** | −2.22 | −12.55 |
| r20-w2 `size_param0.35` | uniform (§238) | 0.331 / 0.574 | 41 | 2 / 2 / 4 | 2 / 2 / 5 | −10.60 | −9.80 | −2.56 | −9.82 |
| r20-w2 `val_best` | sens | 0.384 / 0.648 | 59 | 2 / 3 / 4 | 2 / 3 / 5 | −10.66 | −8.34 | −0.92 | n/a |
| origin r56-w4 / r20-w2 | sens run | 1 | — | — | — | 0 | +0.46 / +3.24 | — | +0.43 / +3.65 |

**Read.**
- *Lever, r56-w4: **WEAK**, +1.90* (−6.00 vs −7.90). Sens lands 0.011 *below* uniform's params, so it gets no size credit. It keeps 24 % more FLOPs (0.409 vs 0.331): the near-full residual streams cost FLOPs at this depth. At κ 0.6 it kept slightly fewer (§230). The margin to SURVIVES (0.10) is far inside the 0.72 pp seed spread of §242. Wave 14 (seed 43) puts the call on a two-seed mean.
- *Against A0.* A0's one-shot probe found its largest lever here: r56-w4 keep 0.35, +7.81 / +7.69 val / TEST (§201; a probe, never a TEST row). Under the walk protocol (per-step recovery plus a 100-epoch final FT) about +1.9 survives, roughly a quarter. At κ 0.6 the ratio is the same: A0's probe lever was +1.95 TEST (§201), and the walk's is +0.54 (§230). Iterative recovery absorbs most of what the one-shot cut shows.
- *r20-w2 guard: sens hurts.* The plan stops at params 0.401 (`every group at its target … strongest legal cut`), and the walk reaches 0.35 by strongest legal cuts. It lands the stage-2 / 3 residual streams at 3 / 3 (of 4 / 8), against uniform's 2 / 4, and −3.26 below uniform at near-equal params. That matches §207, where random beat sens on r20-w2 at 0.35. On this narrow net the plan's floor binds before κ, so off-plan cuts decide the point.
- *Kept epochs (§235).* Late on every final FT (r56-w4 `val_best` loss 0.448 at epoch 1, 0.428 at 100, best 0.428). Honest is 100 epochs against 100. r56's origin moved +0.46 here (+0.60 in §238's run).
- Do not lock. Never an agent row.

---

## 244. Wave 11b endpoint noise: N3 select=last at seed 43 (**22342767**) against seed 42 (§239) — PRELIM; \|s43 − s42\| at 10k **0.14 / 0.05** at 2.57× / 2.11× (< 0.3): the wave 11 call stands (§240)

Sitting 7 Oct wave 11b, `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, `SPECTRA_SEED=43` (data order and crop / flip; same split), from-saved N3 `job21767189/traj_models`, otherwise identical to 22342659. COMPLETED 1 h 26 m, 7 Oct 10:26, `cs-4090-08`, exit 0, TB 0. Start check green on all five final FTs (`select=last`, "kept the last epoch", `keep=last`). Rule (queue, registered 07:50): \|s43 − s42\| ≥ 0.3 at 10k at a gating point makes the wave 11 call there "unresolved" unless both walks clear its bar by more.

| Point | Seed 42 (§239) 5k / 10k | **Seed 43** 5k / 10k | s43 − s42, 5k / 10k | Honest s43 / s42 | Δsel vs §157 at 10k, s43 (s42) |
|---|---|---|---|---|---|
| size_flop0.60 | +0.22 / +0.17 | −0.10 / +0.10 | −0.32 / −0.07 | −0.66 / −0.20 | +0.13 (+0.20) |
| size_flop0.47 (2.11×) | −0.30 / −0.31 | −0.40 / **−0.36** | −0.10 / **−0.05** | −0.66 / −0.42 | +0.10 (+0.15) |
| size_flop0.39 (2.57×) | −1.18 / −1.20 | −0.96 / **−1.06** | +0.22 / **+0.14** | −0.12 / −0.20 | +0.57 (+0.43) |
| `val_best` | −0.90 / n/a | −0.84 / n/a | +0.06 / n/a | −0.20 / −0.12 | n/a |
| origin | +0.34 / +0.42 | +0.48 / +0.42 | +0.14 / 0.00 | — | — |

**Read.**
- *Rule: not fired.* At 10k the endpoint moves 0.14 at 2.57× and 0.05 at 2.11×, both under 0.3. The wave 11 call stands as read in §240: **NEUTRAL** at 2.11×, now resolved on N3's side (two-seed Δsel +0.10 / +0.15). 2.57× stays **unresolved**, because τ-off sits inside the band (+0.21); N3's two-seed Δsel there is +0.50.
- *Where the seed spread lives.* One final FT, re-seeded on the same walk, moves ≤ 0.32 at 5k and ≤ 0.14 at 10k (origin 0.14 / 0.00). ~~Most of the seed spread is therefore the walk.~~ **Corrected 12:05 (§247):** on the thin pair §242's 0.72 is in the final FT. The two sens walks end within 0.06 of each other (−2.80 / −2.74), and the final FT gains +0.00 against +0.66. On DepGraph R56 a re-seeded final FT moves ≤ 0.32 at 5k; on the thin r56-w4 it can move 0.6–0.7.
- *Honest at 5k is the noisiest column* (−0.66 against −0.42 at 2.11×; −0.66 against −0.20 at 0.60), because it inherits the origin's 5k move (+0.14) on top of the point's own. Gate on 10k raw, as the wave 11 call does.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 245. Wave 11 N4: select=last re-FT, DepGraph VGG-19 C100 (**22342661**) — PRELIM; endpoint **+0.69 / +1.34** over the train-loss pick at 10k; on equal epochs the aug walk clears the N4 adopt line vs §149 (**+1.46 / +1.41**); bar-3 VGG-19 stays §149 under the NEUTRAL selection call

Sitting 7 Oct wave 11, `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from-saved N4 `job21737105/traj_models`, paper recipe otherwise (SGD lr 0.01, cosine, 100 epochs, P, seed 42, origin control). COMPLETED 36 min, 7 Oct 10:29, `cs-4090-04`, exit 0, TB 0. Start check green on all four final FTs (env 1, `select=last` 4, kept last 4, `keep=last` 4). Checkpoint `vgg19_cifar100_dep_graph_73.5.pth`; TEST = 5k half (unpruned 0.740). Registered read (wave 11, reported, not gating): N4 against §149 on equal epochs, re-reading §155; the N4 adopt line is "≥ 1 pp after final FT" against 21729551 (way-ahead §3, 30 Sep).

| Point | Keep params / FLOPs | Walk (5k) | **Last** (5k) | Honest | **10k last** | §155 10k (epoch 1) | §149 10k (no-aug walk, late) | Last − §149, 10k |
|---|---|---|---|---|---|---|---|---|
| size 0.70 step 29 | 0.684 / 0.686 | −2.24 | **−1.34** | +0.48 | **−0.93** | −1.62 | −2.39 | **+1.46** |
| size 0.60 step 42 | 0.599 / 0.590 | −2.52 | **−1.44** | +0.66 | **−1.63** | −2.97 | −3.04 | **+1.41** |
| `val_best` step 47 | 0.534 / 0.550 | −2.16 | **−1.74** | +0.00 | n/a | n/a | n/a | 5k +1.90 |
| origin | 1 | 0 | +0.42 | — | +0.61 | +0.96 | +0.56 | — |

Kept epoch: every pruned final FT ends at train loss 0.0089–0.0099 against a best of 0.0027–0.0049, which is where §155 restored to (epoch 1 or 5, §235). The origin's best and last are adjacent (0.0104 / 0.0109).

**Read.**
- *Selection.* The endpoint beats the train-loss pick by **+0.69 / +1.34** at 10k (5k +0.90 / +1.50; `val_best` +0.86), against +0.10 to +0.57 on DepGraph R56 (§239, §244). Off R56 the restore costs much more: here the lowest-train-loss epoch is the most overfit state the Adam walk left (§235's mechanism), and C100 has more room to recover.
- *§155 re-read on equal epochs.* Both rows are genuine ~100-epoch final FTs now (§149 kept epochs ~90–95 at near-zero cosine lr). The crop+flip walk lands **+1.46 / +1.41** above §149 at 10k, clearing the 1 pp N4 adopt line by 0.41–0.46 at both size points (5k: +1.18 / +1.84). §155's "+0.77, under the line" was the selection.
- *Honest is now 100 epochs against 100:* +0.48 / +0.66 / 0.00. The finish adds about half a point beyond what it gives the unpruned net; §155's CROSS-OFF (−0.50 / −0.92 / −0.94) was one epoch against 100.
- *Noise.* This is a from-saved re-FT; §155's final FTs ran inside the walk, so the trajectories are not identical. The origin, which kept a late epoch in both, moved −0.08 (5k) / −0.35 (10k); every Δsel at 10k is above that. One run each.
- *What it moves.* Nothing in the quoted rows yet. The wave 11 call is NEUTRAL (§240): the paper keeps train-loss selection with the caption disclosing it. Under that selection §155 does not clear the line, so bar-3 VGG-19 stays on **§149** (−2.39 / −3.04 at 10k). Quoting N4-last as the row would change the selection for one net after reading its TEST. N4-last is reported beside it as the endpoint sensitivity row. If the paper adopts endpoint selection for every final FT on §235's mechanism (tracker Q7), bar-3 VGG-19 moves to N4-last by the registered 1 pp rule.
- DepGraph's own VGG-19 C100: 73.50 → 70.39 (**−3.11**) at **8.92×** params (keep ≈ 0.11); we are at keep 0.68 / 0.60, far less compression. Quote beside; **never "beats"**.
- 22342768 (cosine from lr 0.1, select=last, same candidates) is the "across architectures" read against these rows.
- Do not lock. Never an agent row.

---

## 246. Wave 12: the slide line's cosine-from-0.1 final FTs at seed 43 (**22343160** N3, **22343165** τ-off; both COMPLETED) — PRELIM; every \|d\| < 0.3 at 2.11× / 2.57× (max **0.27**): the line keeps "one run each" and adds the seed bound

Sitting 7 Oct wave 12, `tree_v10` (sbatch only), §223's / §228's recipe (SGD from lr 0.1, cosine, 100 epochs, origin control) from the same saved candidates, with `SPECTRA_SEED=43` (data order and crop / flip; same split). 22343160 COMPLETED 1 h 22 m, 7 Oct 10:48, `ise-4090-02`, exit 0, TB 0. 22343165 COMPLETED 1 h 24 m, 11:32, `cs-4090-10`, exit 0, TB 0. Its 2.57× and 2.11× rows were read at 11:00, before the job ended, and did not change. Start check green (seed 43, `final_ft from`, recipe lr 0.1 cosine). Every final FT kept a late epoch (best loss at or within 0.0006 of epoch 100's), so these are genuine endpoints with no `select=last` needed. Rule (queue, registered 09:15): d = s43 − s42 at 10k per walk and gating point. If every \|d\| < 0.3, the slide line keeps "one run each" and adds "a second final-FT seed moves each point by ≤ max \|d\|"; otherwise it quotes the two-seed mean and range.

| Walk, point | Seed 42 5k / 10k | **Seed 43** 5k / 10k | d, 5k / **10k** | Honest s43 / s42 |
|---|---|---|---|---|
| N3, 2.57× (`size_flop0.39`) | −0.62 / −0.37 (§223) | −0.64 / −0.64 | −0.02 / **−0.27** | +0.20 / +0.08 |
| N3, 2.11× (`size_flop0.47`) | −0.32 / −0.36 | −0.36 / −0.11 | −0.04 / **+0.25** | −0.62 / −0.72 |
| N3, `size_flop0.60` | −0.26 / +0.04 | +0.44 / +0.41 | +0.70 / +0.37 | −0.12 / −0.96 |
| N3, `val_best` | −0.90 / n/a | −0.84 / n/a | +0.06 / n/a | −0.20 / −0.40 |
| N3, origin | +0.62 / +0.64 | +0.48 / +0.63 | −0.14 / −0.01 | — |
| τ-off, 2.57× | −0.76 / −0.36 (§228) | −0.40 / −0.10 | +0.36 / **+0.26** | −0.18 / −0.28 |
| τ-off, 2.11× | +0.08 / +0.01 | −0.24 / −0.15 | −0.32 / **−0.16** | −0.48 / +0.10 |
| τ-off, `size_flop0.60` | +0.38 / +0.05 | +0.34 / +0.44 | −0.04 / +0.39 | −0.84 / −0.54 |
| τ-off, `val_best` (keep 0.123) | −2.88 / n/a | −2.70 / n/a | +0.18 / n/a | +1.72 / +1.80 |
| τ-off, origin | +0.74 / +0.82 | +1.00 / +1.09 | +0.26 / +0.27 | — |

**Read.**
- *Rule: every \|d\| < 0.3* at the gating points (0.27 / 0.25 on N3, 0.26 / 0.16 on τ-off, 10k). The slide line keeps "one run each" and adds "a second final-FT seed moves each point by ≤ 0.27" (cosine 0.1, both walks; lr 0.01-last on N3 ≤ 0.14, §244). The bound is close to the rule's 0.3, so it is a bound on two runs, not a precision claim.
- *Not gating:* the largest move is at 0.60 on N3 (+0.70 at 5k, +0.37 at 10k). The cosine-from-0.1 endpoint is noisier than lr 0.01's (§244: ≤ 0.32 at 5k, ≤ 0.14 at 10k), plausibly because its first ~37 epochs run at lr ≥ 0.07 and move the weights far from the walk's (untested).
- *Walk gap at 2.11×.* The registered necessary condition holds: seed 42's 0.37 (N3 −0.36, τ-off +0.01) exceeds that point's larger \|d\| (0.25). But at seed 43 the gap reverses (N3 −0.11, τ-off −0.15), and the two-seed means are −0.24 / −0.07, a gap of 0.17. A walk effect at 2.11× is **not** established.
- *Large lr at 2.57× on two seeds.* Cosine 0.1 minus lr 0.01-last at 10k is +0.83 / +0.42 on N3 (seed 42 / 43, the latter against §244) and +0.95 / +1.21 on τ-off (both against §240's seed-42 lr 0.01-last). It is positive on all four pairs, and it already meets the R56 half of wave 11b's "across architectures" read (≥ +0.3 at 2.57×). 22342768 supplies the N4 half.
- *Two-seed cosine-0.1 means at 10k:* N3 −0.24 (2.11×) / −0.51 (2.57×); τ-off −0.07 / −0.23. The slide line keeps its single runs by the rule; the means are reported, never a pick.
- *τ-off's remaining rows (11:32).* Its 0.60 point moves +0.39 at 10k, like N3's +0.37: non-gating, and the largest moves on both walks are at 0.60. The origin moves +0.26 / +0.27, so honest at 2.11× drops to −0.48 from +0.10. Honest is the noisiest column again: it carries the origin's move. At the deep point (keep 0.123, where §228's TREND sits) honest repeats: +1.72 against +1.80.
- Do not lock. Never an agent row. Never call DepGraph a beat.

---

## 247. Wave 8 at κ 0.6: mild-landed at seed 43 (**22341278**) — PRELIM; same architecture as §212; two-seed bar sens − mild **+2.54** (≥ +1.0, clears, as seed 42 did); lever waits on 22341282; on the thin r56-w4 either the walk or the final FT can carry a ~0.7 pp seed spread

`tree_v10`, §212's recipe (3-rate baseline menu, 6 passes, landed `param:0.6`, P, loader crop+flip, walk 40/10, 100-ep final FT + origin, deterministic) with `SPECTRA_SEED=43`. COMPLETED 5 h 15 m, 7 Oct 11:29, `ise-4090-12`, exit 0, TB 0, no fallback; read at 12:10, after the VPN gap. Rule (wave 8, registered 04:10): at r56-w4 the bar (sens − mild-landed ≥ +1.0 at κ 0.6) and the lever (sens − uniform: SURVIVES ≥ +1.0, ABSORBED ≤ +0.3, WEAK between) are read on the two-seed mean. Where the seed-42 call and the two-seed call disagree, the two-seed call stands.

| r56-w4, κ 0.6 | Seed 42: walk / final (5k) | Seed 43: walk / final | Two-seed final | Params / FLOPs | Residual s1 / s2 / s3 |
|---|---|---|---|---|---|
| Mild (§212 / **22341278**) | −5.96 / −5.06 | −5.20 / **−4.90** | **−4.98** | 0.600 / 0.453, both seeds | 2 / 5 / 13, both seeds |
| Sens α 0.5 (§230 / §242) | −2.80 / −2.80 | −2.74 / −2.08 | **−2.44** | 0.600 / 0.572; 0.597 / 0.568 | 4 / 8 / 16, both seeds |
| Uniform (§230 / 22341282 R) | −4.18 / −3.34 | pending | pending | 0.599 / 0.582 | 3 / 6 / 12 |

r20-w2 (reported): mild −2.86 / −3.18 at 0.584 (FLOPs 0.674); sens −2.92 / −3.46 at 0.595 (FLOPs 0.800). Mild's origin: r56 +0.12 / +0.12, r20 +3.32 / +3.44.

**Read.**
- *Bar: clears on two seeds.* Sens − mild = −2.44 − (−4.98) = **+2.54** (seed 42 +2.26, seed 43 +2.82). The seed-42 call stands. A non-learned allocation beats the standard heuristic by 2.5 pp at equal params, while keeping more FLOPs (0.57 against 0.45).
- *Where a seed spread sits.* Mild's two walks land on the same widths (stage min / median / max 2/2/2, 5/6/6, 13/13/13, and identical params / FLOPs). The walks end 0.76 apart and the final FTs 0.16 apart. Sens is the reverse: walks 0.06 apart, final FTs 0.72 apart (gain +0.00 against +0.66). So on the thin r56-w4, either piece can carry a ~0.7 pp seed spread, and the final FT can widen it or close it. Single-seed gaps under ~0.8 pp on this net are noise. This corrects §244's "mostly the walk".
- *Lever.* Waits on 22341282 (uniform, seed 43). Seed 42 alone: WEAK (+0.54).
- *For ops' v10 calls (not decided here).* Mild's two-seed mean is −4.98, 0.08 above §212's −5.06. The beyond-heuristic line (sens two-seed mean + 0.5) stays at ≥ −1.94.
- Do not lock. Never an agent row.

---

## 248. First v10 freeze TEST, ep0127, landed κ 0.6 (**22341737**) — PRELIM; **M1-v10 FLAT**: r56 **−5.28 @ 0.600** vs mild §212 **−5.1 = −0.18**; residual **2 / 5 / 13** (mild clone); census 0.9 only

Skip-train `eval_c10_thin_traj` of **22156116** `snapshots/ep0127` (first freeze after PPO-20 with `vs_mild ≥ +0.5`; the probe is a gate, never a result). Pair of κ 0.8 **§237**. `tree_v10`, P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, floor off, `FIXED_TARGET=1`, `STATE_SENS=1`, det=1, `SIZE_MATCH=param:0.6`, 100-ep origin FT, seed 42, menu 1.0/0.9/0.8/0.7/0.6 all L1. COMPLETED 5 h 4 m, 7 Oct 10:51, `ise-4090-03`, TB 0, exit 0. Start-check green (`policy=actor`, `fixed_target=1`, `state_sens=1`, 6-pass, no `FT_AUG_GPU`). Reader `final_ft_readout.py`. Control = mild-landed **§212**. Ops live-read 12:04 after VPN returned.

| Net | Actor params / FLOPs | Walk | Final 5k | Honest | vs mild §212 | vs sens §230 | vs uniform §230 |
|---|---|---|---|---|---|---|---|
| r20-w2 | 0.584 / 0.674 | −4.66 | **−2.48** | −1.42 | **+0.42** (−2.9) | — | — |
| **r56-w4** | 0.600 / 0.453 | −5.86 | **−5.28** | +0.10 | **−0.18** (−5.1) | −2.48 (−2.80) | −1.94 (−3.34) |
| origin r20 / r56 | 1 | 0 | +3.60 / +0.48 | — | — | — | — |

**Census (r56-w4):** 62 cuts at **0.9**, 74 identity at 1.0, 1 landing ~1.0. Distinct prune actions = **1**. r20: 13 at 0.9, 23 identity, 1 landing. Menu 0.8 / 0.7 / 0.6 **never played**. Same mild path as κ 0.8 (§237).

**Residual widths (r56-w4 `val_best`):** **2 / 5 / 13** — identical to mild / greedy-3 at κ 0.6 (§231), not sens 4 / 8 / 16. Inner medians 2 / 6 / 13. Same landed params and FLOPs as mild (0.600 / 0.453).

**Read.**
- *M1-v10: **FLAT**.* Registered: WIN Δ ≥ +1.0 on r56 κ 0.6 vs mild, no cell ≤ −1.0; NEG Δ ≤ −1.0 on r56 κ 0.6. Actor −5.28 vs mild −5.1 = **−0.18** (vs sitting's two-seed mild mean −4.98, §247: **−0.30**). κ 0.8 was −0.78 (§237). Neither WIN nor NEG. Census ≥ 2 also fails (0.9 only), as on Stage-4. r20 +0.42 is the disaster guard, not a veto.
- *Allocation.* The frozen actor did not keep residual streams full. It walked 0.9 + skip to the mild architecture, 2.48 pp behind sens and 3.34 behind the two-seed beyond-heur bar (−1.94). Same-architecture noise vs mild is 0.14–0.38 pp (§231); −0.18 sits inside it.
- *Never quote the probe* (`vs_mild=+0.505`). 10k n/a (`val_best`). Do not lock. Do not start N8 from this FLAT.

## 249. Wave 10 transplant: DepGraph's own R56 C10 widths at 2.11×, walked and fine-tuned by our pipeline (**22342029**) — PRELIM; registered lift **+0.25 → PARTIAL** (10k −0.14 vs N3 −0.54, less the 0.15 size credit); the two halves disagree (TEST −0.42, val +0.92); the final FT kept epoch 1

Sitting 7 Oct, wave 10 (registered before submit). `tree_v10j`, `SPECTRA_ALLOC_KIND=widths` from `configs/widths_depgraph_r56_c10_2.11x.json` (the widths of DepGraph's released 2.11× model; the plan keeps x0.504 over 30 groups against a target of x0.508; group keep min 0.22, median 0.67, max 0.97). L1 ranking, P, loader crop+flip, walk 40/10, landed `param:0.508`, 100-ep lr 0.01 final FT + origin, deterministic, seed 42. COMPLETED 3 h 31 m, 7 Oct ~12:10, `cs-4090-07`, exit 0, TB 0, no fallback, every group named. Checkpoint `resnet56_cifar10_dep_graph_93.53.pth`, the same unpruned DepGraph R56 as N3. TEST = 5k half (unpruned 0.9336, val half 0.9304). Reader `final_ft_readout.py`. The walk has one candidate, so `val_best` and the size point are the same model ("not fine-tuned twice"), and the reader prints "10k n/a" from the label alone. The 10k below is the reader's `full_test_dacc` on that row. Nothing chose the point on val: the widths plan fixed it, and the walk's per-step FT selects on train loss (`ClassificationHandler.train_model`; only P8 selects on val).

| Row | Params / FLOPs | Walk 5k | Final 5k | Honest | 10k |
|---|---|---|---|---|---|
| **Transplant** | 0.508 / 0.480 | −0.56 | **−0.72** | −0.66 | **−0.14** |
| N3 2.11× (§157 / §232, lr 0.01) | 0.470 / 0.463 | −0.22 | −0.44 / −0.46 | — | −0.46 / −0.62 (mean **−0.54**) |
| Origin (this run) | 1 / 1 | 0 | +0.50 | — | +0.58 |
| DepGraph's own 2.11× model (head-to-head) | — | — | — | — | +0.24 |

| Half | Transplant walk / final | N3 walk / final (§157; §232) | Final lift after the 0.15 credit |
|---|---|---|---|
| TEST 5k | −0.56 / −0.72 | −0.22 / −0.44; −0.46 | **−0.42** |
| val 5k | +0.52 / +0.44 | −0.58 / −0.48; −0.78 | **+0.92** |
| 10k (registered) | −0.02 / −0.14 | −0.40 / −0.46; −0.62 | **+0.25** |

**Widths (the copy).** Residual streams 13 / 32 / 42 (of 16 / 32 / 64). Block-inner convs min / median / max: stage 1 4 / 8 / 11, stage 2 7 / 12 / 28, stage 3 34 / 54 / 61. DepGraph keeps stage 2's residual stream full and stage 3's inner convs nearly full. It cuts the inner convs of stages 1–2 hard.

**Read.**
- *Call: **PARTIAL**.* Registered: lift = transplant − (−0.54) − 0.15 at 10k on the size point. ALLOCATION if ≥ +0.5, NOT-ALLOCATION if ≤ +0.2. The lift is −0.14 + 0.54 − 0.15 = **+0.25**, 0.05 above the NOT-ALLOCATION line and inside the 2.11× noise floor (0.20 honest, §232). The walk alone gives the same answer (+0.23). On one run, DepGraph's widths explain at most about a third of the 0.78 pp between N3 and DepGraph's own model, and possibly none of it. The rest is in what DepGraph does that we do not: its sparsity-regularised training and its own fine-tune.
- *The two halves disagree.* On the TEST half the copy is 0.27 behind N3 (−0.42 after the credit). On the val half it is 1.07 ahead (+0.92). N3's halves agree within 0.04–0.36. Nothing selected on either half, so this is per-half noise on one point, and the 10k average is the most precise number available. That is why the call is registered on 10k. PARTIAL is therefore not distinguishable from NOT-ALLOCATION on one run.
- *The final FT kept **epoch 1*** (best train loss 0.00155 at e1 against 0.00619 at e100). This is the §235 pattern, and N3's lr 0.01 rows did the same, so the comparison is like for like: walk + 1 epoch on both sides. The honest gain −0.66 is CROSS-OFF, because the origin gains +0.50 while the pruned point gains nothing. The genuine-endpoint re-read is wave 11's **22342667** (select=last, `afterok` 22342029, now released), reported beside.
- *Size.* The copy keeps 0.038 more params and 0.017 more FLOPs than N3. The registered credit is on FLOPs; a params-based credit would be larger and the lift smaller.
- The VGG-19 C100 transplant **22342030** is PD (MATCH if ≥ −3.47). This is a diagnostic on DepGraph's architecture, not a SPECTRA row, and never a beat.

## 250. Wave 11 twins: select=last re-FT of the zoo twins, R56 + VGG-16 C10 (**22342662**) — PRELIM; R56 endpoint **+0.36 / +0.40** at 10k (sizes 0.70 / 0.80); the VGG-16 negative control moves ≤ 0.46 at 5k and ≤ 0.30 at 10k (under the 0.5 rule), so the R56 read stands, a little above the control's own spread

Sitting 7 Oct, wave 11 (registered before submit). `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from the saved candidates of **21809595** (§164; chenyaofo R56 C10 94.37 and VGG-16-BN C10 94.16), the paper's lr 0.01 100-ep final FT + origin, P, deterministic, seed 42. COMPLETED 1 h 45 m, 7 Oct ~12:05, `cs-4090-08`, exit 0, TB 0. Start check: select=last on all 8 final FTs, and every one "finished all 100 epochs; kept the last epoch". Reader `final_ft_readout.py` against 21809595.

| Net / point | Params | Final 5k last / §164 / Δsel | 10k last / §164 / Δsel |
|---|---|---|---|
| R56 size 0.70 | 0.694 | −0.48 / −0.62 / +0.14 | −0.40 / −0.76 / **+0.36** |
| R56 size 0.80 | 0.794 | −0.08 / −0.38 / +0.30 | +0.01 / −0.39 / **+0.40** |
| R56 `val_best` | 0.661 | −0.04 / −0.20 / +0.16 | n/a |
| R56 origin | 1 | −0.04 / +0.00 / −0.04 | +0.13 / −0.01 / +0.14 |
| VGG-16 size 0.70 | 0.698 | −0.22 / −0.20 / −0.02 | −0.09 / +0.01 / −0.10 |
| VGG-16 size 0.80 | 0.796 | +0.36 / −0.10 / +0.46 | +0.30 / +0.00 / +0.30 |
| VGG-16 `val_best` | 0.657 | −0.18 / +0.04 / −0.22 | n/a |
| VGG-16 origin | 1 | +0.78 / +0.88 / −0.10 | +0.81 / +0.55 / +0.26 |

**Read.**
- *Rule (wave 11):* VGG-16 already kept late epochs, so it is the negative control. If its |Δsel| exceeds 0.5, the R56 read is noise-limited. Its largest |Δsel| is **0.46** (size 0.80, 5k) and 0.30 at 10k. The rule does not fire.
- *R56:* the true endpoint adds **+0.36 / +0.40** at 10k on a second ResNet-56 checkpoint. DepGraph R56 gave +0.15 to +0.43 (§239, §240), so the ResNet-56 pattern holds. VGG-19 C100 is larger (+0.69 / +1.34, §245).
- *Caveat beside it:* VGG-16's selection barely changed, so its Δsel is mostly re-run noise of one 100-ep final FT. That noise reaches +0.30 at 10k, close to R56's effect. One run gives "about +0.4 at 10k on a re-run spread of up to ~0.3". This does not touch the NEUTRAL call (§240), which is on DepGraph R56 and has its own seed repeat (§244).

## 251. Wave 11b: cosine from lr 0.1, select=last, N4 VGG-19 C100 (**22342768**) — PRELIM; lr 0.1-last − lr 0.01-last **+1.66 / +1.41** at 10k (sizes 0.60 / 0.70); "helps across architectures" **MET** (with DepGraph R56 at 2.57×, §246); about half of it is the unpruned origin improving too (+0.87 at 10k)

Sitting 7 Oct, wave 11b (registered 07:50, before submit). `tree_v10k`, from N4's saved candidates (`tree_v9c/runs/job21737105/traj_models`), final FT SGD from lr 0.1, cosine, 100 epochs, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, origin control, P, deterministic, seed 42. COMPLETED 37 m, 7 Oct 12:47, `cs-4090-07`, exit 0, TB 0. Start check: select=last and keep=last on all 4 final FTs, each with best loss = epoch 100's. Reader `final_ft_readout.py` against **22342661** (lr 0.01-last, §245) and **21737105** (N4, train-loss selection, §155). Checkpoint `vgg19_cifar100_dep_graph_73.5.pth`.

| Point | Params / FLOPs | Walk 5k | Final 5k | Honest | 10k | 10k lr 0.01-last (§245) | d 10k | d 5k |
|---|---|---|---|---|---|---|---|---|
| size 0.60 | 0.599 / 0.590 | −2.52 | **+0.22** | +1.36 | **+0.03** | −1.63 | **+1.66** | +1.66 |
| size 0.70 | 0.684 / 0.686 | −2.24 | **−0.16** | +0.70 | **+0.48** | −0.93 | **+1.41** | +1.18 |
| `val_best` | 0.534 / 0.550 | −2.16 | −0.12 | +0.66 | n/a | — | — | +1.62 |
| origin | 1 / 1 | 0 | +1.38 | — | +1.48 | +0.61 | +0.87 | +0.96 |

**Read.**
- *Call (wave 11b):* "helps across architectures" needs lr 0.1-last − lr 0.01-last ≥ +0.3 on N4 at both size points and on DepGraph R56 at 2.57×. N4: **+1.66 / +1.41** at 10k (+1.66 / +1.18 at 5k). DepGraph R56 at 2.57×: +0.83 / +0.42 (N3, seeds 42 / 43) and +0.95 / +1.21 (τ-off), §246. **MET.** The twins (22342769) are reported beside when they land; they do not gate.
- *Half of it is the baseline.* The same FT lifts the unpruned VGG-19 C100 by +1.48 at 10k, against +0.61 at lr 0.01. The released 73.5 checkpoint is therefore not converged for this recipe. Origin-corrected, the lead is +0.70 / +0.22 at 5k (honest). Quote the honest column beside raw Δacc for any cosine-0.1 row.
- VGG-19 C100 at 0.60 of its params ends at +0.03 at 10k against the released baseline. That is 1.7× in FLOPs, far from DepGraph's 8.92× row (§245), so it is not a comparison with DepGraph and never a beat.

## 252. Q7 val-half check: cosine-0.1 against lr 0.01-last on the val half alone (zero GPU; the rows of §239, §240, §244, §245, §246, §251) — PRELIM; **VAL-AGREES**: d_val **+1.66 / +1.64** (N4) and **+1.10 / +1.26** (DepGraph R56 2.57×, N3 / τ-off), honest +0.86 to +1.36; 2.11× level

Sitting 7 Oct, registered 12:49 before reading (queue "Q7 val-half check"). Under P the final FT selects on train loss or keeps the last epoch, and the walk FT selects on train loss (`ClassificationHandler.train_model`). So no final-FT recipe has read the val half, and it is an independent 5k replicate for choosing between recipes. TEST had already been read, so this is a replication, not a first look. d_val = (val_final − val_origin) under cosine-0.1 minus the same under lr 0.01-last, per point, from each row's `eval_traj_final_ft` event. Honest d_val subtracts the origin's own d_val.

| Walk / point | cos-0.1 job | lr 0.01-last job | d_val | Honest d_val | TEST d (5k) |
|---|---|---|---|---|---|
| N4 size 0.60 | 22342768 | 22342661 | **+1.66** | +0.88 | +1.66 |
| N4 size 0.70 | 22342768 | 22342661 | **+1.64** | +0.86 | +1.18 |
| N3 2.57× | 22340234 | 22342659 | **+1.10** | +0.94 | +0.56 |
| τ-off 2.57× | 22341051 | 22342660 | **+1.26** | +1.36 | +0.64 |
| N3 2.11× | 22340234 | 22342659 | −0.08 | −0.24 | −0.02 |
| τ-off 2.11× | 22341051 | 22342660 | +0.42 | +0.52 | +0.68 |
| N3 2.57×, seed 43 | 22343160 | 22342767 | +0.52 | +0.10 | +0.32 |
| N3 2.11×, seed 43 | 22343160 | 22342767 | +0.46 | +0.04 | +0.04 |
| origin N4 / N3 / τ-off / N3 s43 | — | — | +0.78 / +0.16 / −0.10 / +0.42 | — | +0.96 / +0.28 / +0.46 / +0.00 |

**Read.**
- *Call: **VAL-AGREES**.* Every gating point (N4 0.60 / 0.70 and DepGraph R56 2.57× on both walks, seed 42) has d_val ≥ +0.3, and the honest d_val is ≥ +0.86 at all four. Cosine-0.1's lead at deep compression and on VGG-19 C100 replicates on data no recipe has read. Q7's "adopt cosine-0.1" option is therefore val-chosen, not test-chosen.
- *Where it does not help.* At 2.11× it is level (N3 −0.08, τ-off +0.42 on val; −0.02 / +0.68 on TEST). Seed 43 halves the 2.57× lead on N3 (+0.52 raw, +0.10 honest). The claim is "cosine-0.1 helps at deep compression and on VGG-19 C100", not "everywhere".
- The 10k values in §246 are the means of these two halves (N3 2.57×: (+1.10 + 0.56) / 2 = +0.83).
- Not switched here: the paper's final-FT recipe is Gilad's Q7. Adopting it means re-running each paper row's final FT from its saved candidates (P rows set `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`; older rows would need a re-walk). From-saved re-runs took 0.6–1.8 GPU-h per job here. The lr 0.01 rows are kept beside.

## 253. Wave 11: 1-cycle-last (warm-up 30, peak lr 0.1, keep the last epoch) on N3 / τ-off's saved DepGraph R56 candidates (**22342663 / 64**) — PRELIM; Lead 3 rule on genuine endpoints **FAILS on both walks** (honest Δ at 2.11× **−0.68 / −0.18**, under +0.5); raw is level with cosine-0.1; its raw gain is the origin rising

Sitting 7 Oct, wave 11 (registered before submit). `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, warmcos schedule (30 warm-up epochs, peak 0.1, 100 epochs), from-saved N3 (`tree_v9c/runs/job21767189`) and τ-off (`tree_v9d/runs/job22288423`) candidates, origin control, P, deterministic, seed 42. 22342663 COMPLETED 1 h 21 m, 7 Oct 12:50, `ise-4090-04`. 22342664 COMPLETED 1 h 22 m, 12:54, `cs-4090-10`. Both exit 0, TB 0. Start check: select=last on all 5 final FTs of each, and each kept its last epoch. Reader `_tmp_s7oct_selread.sh` pairs (from `final_ft_readout.py`) against lr 0.01-last (**22342659 / 60**, §239 / §240), 1-cycle with train-loss selection (**22340387 / 22341280**, §224 / §235) and cosine-0.1 (**22340234 / 22341051**, §223 / §228).

| Walk / point | 1-cycle-last final 5k | lr 0.01-last | Raw Δ | Honest 1-cycle / lr 0.01 / Δ | 10k 1-cycle / lr 0.01 / Δ | 10k vs cosine-0.1 |
|---|---|---|---|---|---|---|
| N3 2.11× | −0.26 | −0.30 | +0.04 | −1.10 / −0.42 / **−0.68** | −0.08 / −0.31 / +0.23 | +0.28 |
| N3 2.57× | −0.80 | −1.18 | +0.38 | −0.54 / −0.20 / −0.34 | −0.45 / −1.20 / +0.75 | −0.08 |
| N3 origin | +1.06 | +0.34 | +0.72 | — | +0.98 / +0.42 / +0.56 | +0.34 |
| τ-off 2.11× | −0.04 | −0.60 | +0.56 | −0.30 / −0.12 / **−0.18** | −0.07 / −0.54 / +0.47 | −0.08 |
| τ-off 2.57× | −0.50 | −1.40 | +0.90 | −0.30 / −0.46 / +0.16 | −0.26 / −1.31 / +1.05 | +0.10 |
| τ-off origin | +1.02 | +0.28 | +0.74 | — | +0.93 / +0.64 / +0.29 | +0.11 |

**Read.**
- *Lead 3 rule at 2.11×, both walks (honest Δ ≥ +0.5 and raw final ≥ the reference's):* raw holds on both (+0.04, +0.56), but honest Δ is **−0.68** (N3) and **−0.18** (τ-off). **FAILS on both walks.** §235's VOID becomes a fail on genuine endpoints, and the caption does not move to 1-cycle.
- *Why:* with the endpoint kept, 1-cycle's peak lr 0.1 does what cosine-0.1 does to the pruned points (10k within ±0.3 of cosine-0.1 at every point). It lifts the unpruned origin more, though (+1.06 / +1.02 at 5k against cosine-0.1's +0.62 / +0.74), so its honest gain is lower. Cosine-0.1 (§252) remains the only large-lr candidate for Q7. 1-cycle adds nothing beside it.

## 254. Wave 8 at κ 0.6: uniform allocation at seed 43 (**22341282**) — PRELIM; two-seed lever sens − uniform **+1.09 → SURVIVES** (seed 42 +0.54, seed 43 +1.64), 0.09 above the line; uniform already holds +1.45 of sens's +2.54 over mild

`tree_v10h`, `SPECTRA_ALLOC_KIND=uniform`, §230's recipe (`param:0.6`, 6 passes, P, loader crop+flip, walk 40/10, 100-ep final FT + origin, deterministic) with `SPECTRA_SEED=43`. COMPLETED 3 h 5 m, 7 Oct 12:55, `ise-4090-02`, exit 0, TB 0, no fallback. `[alloc]` uniform plan keeps x0.554 on r20-w2 (every group 0.75). Reader `final_ft_readout.py`. Rule (wave 8, registered 04:10): at r56-w4 the lever (sens − uniform: SURVIVES ≥ +1.0, ABSORBED ≤ +0.3, WEAK between) is read on the two-seed mean, and where the seed-42 call and the two-seed call disagree, the two-seed call stands.

| r56-w4, κ 0.6 | Seed 42 | Seed 43 | Mean | Residual widths (s42 / s43) |
|---|---|---|---|---|
| Sens α 0.5 (§230 / §242) | −2.80 | −2.08 | **−2.44** | 4 / 8 / 16 both |
| Uniform (§230 / this) | −3.34 | **−3.72** | **−3.53** | s42 not read here; s43 3 / 6 / 12 |
| Mild-landed (§212 / §247) | −5.06 | −4.90 | **−4.98** | 2 / 5 / 13 both |
| **Lever sens − uniform** | +0.54 | +1.64 | **+1.09** | |
| Uniform − mild | +1.72 | +1.18 | +1.45 | |

Uniform seed 43: r56-w4 walk −4.88, final −3.72 @ params 0.599 / FLOPs 0.582, honest +0.64, inner medians 3 / 6 / 13. r20-w2 guard: −2.70 @ 0.581 (sens seed 43: −3.46).

**Read.**
- *Lever at κ 0.6: **SURVIVES**.* The two-seed mean is +1.09, so the seed-42 call (WEAK, §230) is replaced. It clears by 0.09 on a per-seed spread of 1.1. "Survives" here means about +1 pp, not a precise number.
- *Decomposition of the bar.* Sens beats mild by +2.54 on two seeds (§247). Uniform already supplies +1.45 of it: it keeps residual streams 3 / 6 / 12 where mild keeps 2 / 5 / 13. Sens adds +1.09 and keeps them full (4 / 8 / 16). So about 60 % of the non-learned allocation's lead over mild comes from spreading the cut evenly, and about 40 % from what sens does beyond that. Whether that 40 % is the full residual streams is wave 9's question (`inner`, the residual-full rule alone; 22341865 R, the rest PD).
- κ 0.8's two-seed lever waits on 22341283 / 84 (both R).

## 255. Wave 8 at κ 0.8: seed-43 sens / uniform / mild-landed (**22341283 / 22341284 / 22341279**) — PRELIM; two-seed lever sens − uniform **+1.13 → SURVIVES** (seed 42 +0.96, seed 43 +1.30; 10k +1.25 on both); mild and uniform land on one architecture on both seeds, yet their 5k finals differ by 0.14 / 0.92

`tree_v10h` (sens α 0.5 and uniform, 5-rate menu) and `tree_v10` (mild-landed, §211's recipe), `param:0.8`, 6 passes, P, crop+flip walk 40/10, 100-ep final FT + origin, deterministic, `SPECTRA_SEED=43`. Sens **22341283** COMPLETED 2 h 28 m, 13:15, `ise-4090-02`; uniform **22341284** COMPLETED 2 h 40 m, 13:31, `cs-4090-07`; mild **22341279** COMPLETED 2 h 40 m, 13:08, `ise-4090-05`. All exit 0, TB 0, no fallback. Reader `final_ft_readout.py`; 10k from its `full_test_dacc` at the landed point (the size point is the `val_best` model). Rule (wave 8, registered 04:10): at r56-w4 the lever (SURVIVES ≥ +1.0, ABSORBED ≤ +0.3, WEAK between) is read on the two-seed mean, and the two-seed call replaces seed 42's (§229, WEAK +0.96). The κ 0.8 bar is reported only.

| r56-w4, κ 0.8 | 5k s42 | 5k s43 | 5k mean | 10k s42 | 10k s43 | 10k mean | Residual (both seeds) |
|---|---|---|---|---|---|---|---|
| Sens α 0.5 (§227 / this) | −1.30 | −1.32 | **−1.31** | −1.17 | −1.18 | −1.18 | 4 / 8 / 16 |
| Uniform (§229 / this) | −2.26 | −2.62 | **−2.44** | −2.42 | −2.43 | −2.43 | 3 / 7 / 14 |
| Mild-landed (§211 / this) | −2.12 | −1.70 | **−1.91** | −2.21 | −2.09 | −2.15 | 3 / 7 / 14 |
| **Lever sens − uniform** | +0.96 | +1.30 | **+1.13** | +1.25 | +1.25 | +1.25 | |
| Bar sens − mild (reported) | +0.82 | +0.38 | +0.60 | +1.04 | +0.91 | +0.98 | |

Seed 43 at r56-w4: sens walk −1.58, final −1.32 @ params 0.798 / FLOPs 0.708 (step 150; inner 2/2/4, 2/5/8, 6/14/16; seed 42 had 5/16/16 in stage 3), honest +0.14. Uniform walk −2.66, final −2.62 @ 0.799 / 0.716, honest −0.44. Mild walk −2.20, final −1.70 @ 0.799 / 0.716, honest +0.28. r20-w2 guard: sens +0.54 @ 0.799, uniform +0.14 @ 0.774, mild +0.92 @ 0.774.

**Read.**
- *Lever at κ 0.8: **SURVIVES**.* The two-seed mean is +1.13 on the 5k half, replacing §229's WEAK; on 10k it is +1.25 on both seeds. With §254 (+1.09 at κ 0.6), the allocation lever survives at both of v10's probe keeps on two seeds, under lr 0.01. Wave 19 re-reads both under cosine-0.1.
- *Walk noise at a fixed architecture.* Mild and uniform end in the identical r56-w4 at κ 0.8 on both seeds (3/7/14, inner 3/3/3, 7/7/7, 14/15/16, step 47; §231). Their finals still differ by 0.14 (s42) and 0.92 (s43) on the 5k half, and by 0.21 and 0.34 on 10k. That is the spread of one walk plus one final FT at a fixed architecture. Single-seed 5k calls against a 1 pp bar are fragile at this level; 10k halves it.
- So the lever (sens − uniform) and the bar (sens − mild) measure the same architecture difference here. They read +1.13 and +0.60 on the 5k half, and the gap between them is that walk noise; at 10k they are +1.25 and +0.98.
- Sens's two seeds land 0.02 apart on the 5k half, and their val halves tie exactly (−1.04). At the 0.02 pp granularity of 5,000 images that is a coincidence, not a cache: they are two different models (steps 169 / 150, different stage-3 widths), and their TEST halves differ.
- *Beside v10 (ops' §237):* the κ 0.8 freeze TEST was −2.88 on r56-w4 with residual 3/7/14, mild's architecture. The non-learned allocation sits at −1.31 (two-seed, 5k).

## 256. Wave 11b twins: cosine from lr 0.1, select=last, on the zoo twins' saved candidates (**22342769**) against lr 0.01-last (§250) — PRELIM; **level** at the shallow keeps (10k |d| ≤ 0.31; val half ≤ 0.22 at the pruned points); with §246 and §251, the large-lr gain grows with compression depth

`tree_v10k`, `SPECTRA_EVAL_FINAL_FT_FROM=tree_v9c/runs/job21809595/traj_models`, `SPECTRA_EVAL_FINAL_FT_LR=0.1`, `SELECT=last`, P, seed 42. COMPLETED 1 h 41 m, 13:51, `cs-4090-08`, exit 0, TB 0; all 8 final FTs kept the last epoch. Reported beside §251 (wave 11b registration: "lr 0.1-last − lr 0.01-last, reported"; the "helps across architectures" call was already MET there).

| Net / point | Params | Cosine-0.1 5k / 10k | lr 0.01-last 5k / 10k (§250) | d 5k (honest) | d 10k | d val half (honest) |
|---|---|---|---|---|---|---|
| Zoo R56, size 0.70 | 0.694 | −0.28 / −0.19 | −0.48 / −0.40 | +0.20 (−0.02) | +0.21 | +0.22 (+0.54) |
| Zoo R56, size 0.80 | 0.794 | +0.20 / +0.09 | −0.08 / +0.01 | +0.28 (+0.06) | +0.08 | −0.12 (+0.20) |
| Zoo R56, origin | 1.000 | +0.18 / +0.08 | −0.04 / +0.13 | +0.22 | −0.05 | −0.32 |
| VGG-16, size 0.70 | 0.698 | +0.38 / +0.22 | −0.22 / −0.09 | +0.60 (+1.10) | +0.31 | +0.02 (−0.06) |
| VGG-16, size 0.80 | 0.796 | +0.46 / +0.46 | +0.36 / +0.30 | +0.10 (+0.60) | +0.16 | +0.22 (+0.14) |
| VGG-16, origin | 1.000 | +0.28 / +0.60 | +0.78 / +0.81 | −0.50 | −0.21 | +0.08 |

**Read.**
- *Level on the twins.* At keeps 0.7 / 0.8 on CIFAR-10 the large-lr fine-tune neither helps nor hurts: every 10k difference is within ±0.31, and at the pruned points the val half, which no final FT reads, is within ±0.22. Honest 5k gains on VGG-16 (+0.6 / +1.1) come from its origin dropping under cosine-0.1 (−0.50 at 5k), not from the pruned points rising.
- *Depth dependence.* Cosine-0.1 over lr 0.01-last is about 0 here and at DepGraph R56 2.11× (§246), about +0.9 at R56 2.57× (§223, §228, §246) and +1.4 to +1.7 on VGG-19 C100 at 0.60 / 0.70 (§251). The larger the accuracy the walk removed, the more the large-lr fine-tune recovers, as Le & Hua (2021) report. For Q7 this means adopting cosine-0.1 moves the deep rows and leaves the shallow ones where they are.

## 257. Wave 11: select=last re-FT of the DepGraph R56 uniform allocation's saved candidates (**22342665**) against its train-loss final (22340524, §233) — PRELIM, reported; **+0.35** at 10k at the landed point (+0.22 at 5k, +0.48 on the val half), inside wave 11's +0.15 to +0.43; the origin moves −0.39 the other way

`tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, lr 0.01 (default), P, seed 42, from `tree_v10h/runs/job22340524/traj_models`. COMPLETED 36 m, 14:07, `ise-4090-01`, exit 0, TB 0; both final FTs kept the last epoch. The landed point is step 113 (params 0.465 / FLOPs 0.472). The parent labels that step `val_best`, so the readout withholds its 10k; the step was chosen by size, not val, so its 10k is computed directly here (`full_test_dacc`).

| Row | select=last 5k / val / 10k | train-loss final (§233) 5k / val / 10k | Δsel 5k / val / 10k |
|---|---|---|---|
| Landed, params 0.465 | −0.52 / −0.30 / −0.41 | −0.74 / −0.78 / −0.76 | **+0.22 / +0.48 / +0.35** |
| Origin | +0.16 / +0.34 / +0.25 | +0.46 / +0.82 / +0.64 | −0.30 / −0.48 / −0.39 |

**Read.** Keeping the true endpoint lifts the pruned DepGraph uniform row by about a third of a point and lowers the unpruned origin by about as much, so the honest difference is +0.74 at 10k. That is the same pattern as N3 / τ-off (§239, §240: +0.15 to +0.43 at 10k, NEUTRAL) and the zoo R56 twin (§250: +0.36 / +0.40). It does not reopen wave 11's call. Reported beside §233 and wave 18's cosine-0.1 re-read of the same candidates (22374248, PD).

## 258. Wave 15: N4's endpoint final FT at seed 43 (**22344788**) against seed 42 (**22342661**, §245) — PRELIM; the equal-epoch re-read **holds on two final-FT seeds**: last − §149 at 10k **+1.53 / +1.22** (seed 42 +1.46 / +1.41); a second seed moves each pruned point by ≤ 0.30

`tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, 22342661's recipe exactly (SGD lr 0.01, cosine, 100 epochs, P, origin control, from N4's saved `job21737105/traj_models`) with `SPECTRA_SEED=43` (data order and crop / flip; same split, candidates and code). COMPLETED 38 m, 7 Oct 14:29, `ise-4090-01`, exit 0, TB 0. Start check green: seed 43 in the env line, `select=last` on all 4 recipe lines, all 4 final FTs kept the last epoch (train loss 0.0088–0.0097 pruned, 0.0105 origin), `keep=last` on all 4 TRAJ final-FT lines. Rule (queue wave 15, registered 10:50, before submit): at 10k and both size points, if both seeds' last − §149 is ≥ +1.0, the re-read holds on two final-FT seeds; otherwise "met on one seed only" at each point that misses. Never keep the better seed.

| Point | Keep params / FLOPs | Seed 43 last 5k / 10k | Seed 42 last 5k / 10k (§245) | d = s43 − s42, 5k / 10k | Honest s43 / s42 | §149 10k | **Last − §149, 10k: s43 / s42** | Δsel vs §155, 10k: s43 / s42 |
|---|---|---|---|---|---|---|---|---|
| size 0.70 step 29 | 0.684 / 0.686 | −1.22 / −0.86 | −1.34 / −0.93 | +0.12 / +0.07 | +0.72 / +0.48 | −2.39 | **+1.53 / +1.46** | +0.76 / +0.69 |
| size 0.60 step 42 | 0.599 / 0.590 | −1.74 / −1.82 | −1.44 / −1.63 | −0.30 / −0.19 | +0.48 / +0.66 | −3.04 | **+1.22 / +1.41** | +1.15 / +1.34 |
| `val_best` step 47 | 0.534 / 0.550 | −1.88 / n/a | −1.74 / n/a | −0.14 / n/a | −0.02 / 0.00 | n/a | 5k +1.76 / +1.90 | 5k +0.72 / +0.86 |
| origin | 1 | +0.30 / +0.38 | +0.42 / +0.61 | −0.12 / −0.23 | — | +0.56 | — | — |

**Read.**
- *Call: HOLDS on two seeds.* Every seed and size point clears the 1 pp N4 adopt line at 10k. The smallest margin is seed 43 at size 0.60, 0.22 above the line. Two-seed means are **+1.50 / +1.32**.
- *Seed noise.* A second final-FT seed moves a pruned point by at most 0.30 (5k, size 0.60) and the origin by 0.23 (10k). That is well inside the 1.2–1.5 pp margin over §149.
- *Selection.* The endpoint's gain over §155's train-loss pick repeats: +0.76 / +1.15 at 10k against seed 42's +0.69 / +1.34. The two-seed means are +0.73 / +1.25.
- *Honest at seed 43:* +0.72 / +0.48 / −0.02 (seed 42: +0.48 / +0.66 / 0.00).
- *What it moves.* Nothing in the quoted rows, as registered: wave 15 does not change the bar-3 row, and tracker Q7 decides the selection rule. Under train-loss selection bar-3 VGG-19 stays on §149. If Q7 adopts `select=last`, N4-last's row is the two-seed mean (10k −0.90 / −1.73; +1.50 / +1.32 over §149), never the better seed.
- DepGraph's own VGG-19 C100 is −3.11 at 8.92× params (keep ≈ 0.11). We keep 0.68 / 0.60, far less compression. Quote it beside; **never "beats"**.
- Do not lock. Never an agent row.

## 259. Waves 16–17 at κ 0.6: the allocation lever under v10's walk FT 12/4 (**22371882 / 22371892** seed 42, **22372632 / 22372633** seed 43) — PRELIM; two-seed sens − mild on the reward's own view **+4.05 → VISIBLE** (seed 42 +3.48, seed 43 +4.62; 40/10 +2.33, ×1.74): v10's FLAT is a learning failure, not a budget that hid the lever; after the 100-epoch final FT the lever is **+2.29**, as at 40/10 (+2.54)

Sitting 7 Oct, wave 16 (registered 12:10, submitted 12:15) and wave 17 (registered 12:30, submitted 12:31). Sens α 0.5 in `tree_v10h` (§230's recipe, 5-rate menu) against mild-landed in `tree_v10` (§212's recipe), thin pair, landed κ 0.6. The only change from 40/10 is `SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4`: 6 passes, P, loader crop+flip, 100-epoch final FT + origin and deterministic eval stay. 22371882 COMPLETED 14:26 and 22372632 COMPLETED 14:32 (1 h 36 m each, exit 0). 22371892 COMPLETED 15:06 (2 h 11 m, exit 0) and 22372633 COMPLETED 15:19 (2 h 9 m, exit 0). Start check green on all four: env `NUM_EPOCHS` 12 / `FINETUNE_PATIENCE` 4 with the right seed, walk FT lines `Epoch …/12`, no Traceback, no fallback, and the sens jobs print their `[alloc]` plan line. Call (registered): d = sens − mild on r56-w4's return (`fixed target: episode ends … return`, the in-walk val Δacc at the landed point), read on the two-seed mean. **VISIBLE** ≥ +1.0, **HIDDEN** ≤ +0.3, PARTIAL between; where seed 42 and the two-seed read disagree, the two-seed read stands.

| r56-w4 at κ 0.6 | Landed params, 12/4 | **Return 12/4** | Return 40/10 (job) | 12/4 − 40/10 |
|---|---|---|---|---|
| Sens, seed 42 | 0.5971 | −4.36 | −3.30 (22340391) | −1.06 |
| Mild, seed 42 | 0.5997 | −7.84 | −5.86 (22156062) | −1.98 |
| **d, seed 42** | | **+3.48** | +2.56 | ×1.36 |
| Sens, seed 43 | 0.5998 | −3.10 | −2.78 (22341281) | −0.32 |
| Mild, seed 43 | 0.5997 | −7.72 | −4.88 (22341278) | −2.84 |
| **d, seed 43** | | **+4.62** | +2.10 | ×2.20 |
| **d, two-seed mean** | | **+4.05** | +2.33 | **×1.74** |

TEST on the 5k half at the landed point (walk / final-FT, 100 epochs). Sens at 12/4: seed 42 −4.28 / **−2.40** (honest +1.56), seed 43 −3.14 / **−2.68** (honest −0.06). Sens at 40/10: −2.80 / −2.80 and −2.74 / −2.08. Mild at 40/10: −5.96 / −5.06 and −5.20 / −4.90. Mild at 12/4: seed 42 −7.52 / **−4.72** (honest +2.52), seed 43 −7.72 / **−4.94** (honest +2.18), both on the 40/10 architecture exactly (residual 2 / 5 / 13). Final-FT d: seed 42 **+2.32** (40/10 +2.26), seed 43 **+2.26** (40/10 +2.82); two-seed **+2.29**, against §247's +2.54.

Guard, r20-w2 returns (sens at 0.5948, mild at 0.5838): seed 42 −5.00 vs −7.16 (d +2.16), seed 43 −6.60 vs −7.12 (d +0.52). At 40/10 both seeds give −3.82 vs −3.90 (d +0.08). Final-FT TEST at 12/4: sens −3.96 / −3.82, mild −3.54 / −4.22, so after the final FT r20's d is −0.42 / +0.40 (two-seed −0.01; 40/10 −0.06 / −0.28).

**Read.**
- *Call: VISIBLE on two seeds.* At v10's own walk budget, its reward puts the sens allocation about 4 pp above mild at r56-w4 κ 0.6, and the two seeds agree (+3.48, +4.62). The lever was in v10's reward, larger than at the TEST budget. So the FLAT M1-v10 (§248: the actor plays 0.9 or skip and lands on mild's architecture) is a learning failure (exploration, credit assignment or representation), not a recipe that hid the lever. The HIDDEN branch's remedy (a longer walk FT or a different reward) is not indicated by this read.
- *Why the lever grows at 12/4.* A shorter walk FT costs mild 2.0–2.8 pp of return and sens 0.3–1.1. Mild's landed r56-w4 thins the residual streams (2 / 5 / 13 at both budgets, §248 and 22371892), while sens keeps them full (4 / 8 / 16 at 12/4); the thin streams recover more slowly in 12 epochs. The guard shows the same thing: level at 40/10 (+0.08 on both seeds), sens ahead at 12/4 (+2.16 / +0.52). Part of the 12/4 lever is therefore recoverability under a short walk FT. It still points the same way as the 40/10 lever the TEST rewards (+2.33, §247's +2.54 after final FT).
- *Final FT at 12/4.* Sens's 100-epoch finals (−2.40 / −2.68) and mild's (−4.72 / −4.94) sit where the 40/10 walks' finals do (−2.80 / −2.08 and −5.06 / −4.90). After the final FT the two-seed lever is +2.29 at 12/4 against +2.54 at 40/10 (§247). The extra size of the 12/4 return (×1.74 on two seeds) is recovery speed under a short walk FT, and 100 epochs erase it. The guard agrees: r20-w2's +2.16 / +0.52 in the return become −0.42 / +0.40 after the final FT. So v10's reward overstates the TEST lever by about 1.8× (+4.05 against +2.29) and still points the same way.
- *Visible, not a call.* At this keep the final-FT TEST barely depends on the walk budget. Two-seed means are sens −2.54 at 12/4 against −2.44 at 40/10, and mild −4.83 against −4.98. That is one keep on one net. A5's proxy-fidelity read (§195: 12/4 does not keep the agent's candidate ranking) asks a different question and stands.
- κ 0.8 (**22372634 / 35**, PD) gets the same bars on seed 42 and is reported beside. If it disagrees, the write-up says at which keep the budget hides the lever.
- Not a train and not a v10 TEST; ops' M1-v10 FLAT (§248) stands. Never quote the in-walk returns as TEST. Do not lock. Never an agent row.

## 260. Mild-landed κ 0.35 control, thin pair (**22340796**) — PRELIM, reported; r56-w4 **−10.34 @ 0.348 / FLOPs 0.271**; the κ 0.35 bar: sens **+4.34**, uniform **+2.44** above mild (5k), at 1.51× / 1.22× mild's FLOPs; r20-w2 guard: sens −3.36, uniform −0.10

Sitting 7 Oct wave 5 (registered 03:20, before submit), `tree_v10`, §211 / §212's recipe at `SIZE_MATCH = SIZE_POINTS = param:0.35`: 3-rate baseline menu, 6 passes, P, loader crop+flip, 100-epoch final FT + origin, seed 42. COMPLETED 8 h 45 m, 7 Oct 14:53, `ise-4090-02`, exit 0, TB 0, no fallback. Every final FT kept a late epoch (best train loss within 0.005 of epoch 100's), so these are genuine 100-epoch finals. Registered use: the standard-heuristic bar for the κ 0.35 allocation walks and the thin Pareto's deep point, reported beside the κ 0.35 pair. No separate call.

| Net / point | Arm | Params / FLOPs | Residual s1 / s2 / s3 | Walk 5k | **Final 5k** | Honest | 10k | Final − mild |
|---|---|---|---|---|---|---|---|---|
| **r56-w4** `size_param0.35` step 277 | **mild** | 0.348 / **0.271** | 2 / 3 / 10 | −11.04 | **−10.34** | +0.16 | −10.40 | — |
| r56-w4 `size_param0.35` | sens (§243) | 0.338 / 0.409 | 4 / 8 / 15 | −8.42 | −6.00 | +1.96 | n/a | **+4.34** |
| r56-w4 `size_param0.35` | uniform (§238) | 0.349 / 0.331 | 2 / 5 / 9 | −7.60 | −7.90 | −0.90 | n/a | **+2.44** |
| r56-w4 `val_best` step 266 | mild | 0.396 / 0.291 | 2 / 3 / 11 | −9.14 | −9.50 | −0.90 | n/a | — |
| r20-w2 `size_param0.35` = `val_best` step 80 | mild | 0.335 / 0.575 | 2 / 2 / 4 | −11.34 | **−9.70** | −1.04 | n/a | — |
| r20-w2 `size_param0.35` | sens (§243) | 0.340 / 0.631 | 2 / 3 / 3 | −14.08 | −13.06 | −2.22 | −12.55 | −3.36 |
| r20-w2 `size_param0.35` | uniform (§238) | 0.331 / 0.574 | 2 / 2 / 4 | −10.60 | −9.80 | −2.56 | −9.82 | −0.10 |
| origin r56-w4 / r20-w2 | mild run | 1 | 4 / 8 / 16 · 2 / 4 / 8 | 0 | +0.54 / +2.68 | — | +0.44 / +3.38 | — |

**Read.**
- *The κ 0.35 bar.* On r56-w4 at equal params, an allocation that keeps the residual streams wide recovers far better than mild's. Sens is +4.34 above mild and uniform +2.44, on one seed each; the r56-w4 seed spread is ~0.7 pp (§242). The ordering matches κ 0.6 (§247, §254: sens +2.54 over mild on two seeds, of which uniform takes about +1.45). At κ 0.35 both gaps are larger.
- *The FLOPs price.* Mild halves the stage-1 residual stream as uniform does and cuts stage 2 hardest (3 of 8; uniform 5, sens 8). Those stages run at the highest resolution, so it keeps the fewest FLOPs: 0.271, against uniform 0.331 and sens 0.409. On a params-only axis sens dominates. On a FLOPs axis it buys its +4.34 with 1.51× mild's FLOPs. The thin Pareto quotes both axes.
- *Guard.* r20-w2 goes the other way for sens (−3.36), as §243 found against uniform: its plan's floor binds before κ. Uniform and mild are level there (−0.10).
- Walk − final: mild's final FT adds +0.70 on r56-w4 (sens +2.42, uniform −0.30). The lever is in what the architecture recovers to, not in the walk endpoint.
- Wave 19's cosine-0.1 re-read of this run (22374696, `afterok` met at 14:53) and the seed-43 κ 0.35 pair (22344456 / 57) are still queued. §243's WEAK call waits on the second seed.
- Do not lock. Never an agent row.

## 261. Wave 11: select=last re-FT of the DepGraph R56 sens allocation's saved candidates (**22342666**) against its train-loss final (22340523, §236) — PRELIM, reported; the endpoint moves sens **−0.34** at 5k (val +0.20, 10k −0.07); with §257, sens − uniform on genuine endpoints is **−0.16** at 5k / +0.26 val / **+0.05** at 10k, against §236's +0.40 / +0.54 / +0.47 under the epoch-1 restore

`tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, lr 0.01 (default), P, seed 42, from `tree_v10h/runs/job22340523/traj_models`. COMPLETED 34 m, 15:04, `ise-4090-01`, exit 0, TB 0. Start check green (env 1, `select=last` 2, kept the last epoch 2, `keep=last` 2). The pruned final FT ends at train loss 0.00834 against a best of 0.00175; §236's restore had kept epoch 1 (0.00195). The origin ends at 0.00242 against 0.00230. The landed point is step 165 (params 0.469 / FLOPs 0.398). The parent labels it `val_best` but it was chosen by size, so its 10k is computed directly (`full_test_dacc`), as in §257. Registered (wave 11): allocation rows reported beside their lr 0.01 rows; no call.

| Row | select=last 5k / val / 10k | train-loss final (§236) 5k / val / 10k | Δsel 5k / val / 10k |
|---|---|---|---|
| Landed sens, params 0.469 | −0.68 / −0.04 / −0.36 | −0.34 / −0.24 / −0.29 | **−0.34 / +0.20 / −0.07** |
| Origin | +0.48 / +0.54 / +0.51 | +0.60 / +0.52 / +0.56 | −0.12 / +0.02 / −0.05 |

| sens − uniform at the landed points (params 0.469 vs 0.465) | 5k | val | 10k |
|---|---|---|---|
| select=last (22342666 − 22342665, §257) | **−0.16** | +0.26 | **+0.05** |
| train-loss pick (§236 − §233) | +0.40 | +0.54 | +0.47 |

**Read.**
- *Sens gains nothing from the endpoint.* Its 100th epoch is level with its epoch-1 restore (−0.34 at 5k, −0.07 at 10k, +0.20 on val). The uniform twin gained +0.22 / +0.35 (§257), and the other wave 11 R56 rows +0.10 to +0.57.
- *The DepGraph R56 lever is level on genuine endpoints.* Sens − uniform is −0.16 at 5k and +0.05 at 10k, with the val half at +0.26; all three reads were +0.40 to +0.54 under the restore. Read against §236's bars (5k; reported, not a call), −0.16 sits in ABSORBED (≤ +0.15). §236 already put its gain in the walk (+0.24 at walk TEST, honest flat), and 100 genuine epochs remove it.
- *The FLOPs saving stays.* At equal params and now equal accuracy, sens keeps 16 % fewer FLOPs (0.398 vs 0.472; 2.51× vs 2.12×), because it cuts inner convs instead of residual streams. On the FLOPs axis it is still the better architecture; on the accuracy axis at equal params it is not.
- *What it does not touch.* Every landed-κ thin final kept a late epoch (§235, §238, §260), so the thin-pair levers (§254, §255, §259, §260) are on genuine endpoints already. Only the DepGraph R56 row changes.
- One run per arm. Wave 13's seed-43 walks (22344275 / 76, PD) use the train-loss pick. Wave 18's cosine-0.1-last re-read of both arms (22374230 / 48, PD) is the next read of this lever on genuine endpoints, under the fine-tune Q7 recommends.
- Never call DepGraph a beat. Do not lock. Never an agent row.

## 262. Wave 18 (a): the DepGraph R56 transplant re-fine-tuned under cosine from lr 0.1, keep last (**22374229**, from 22342029's saved candidates) — PRELIM; registered lift_cos **+0.87 → ALLOCATION** (10k **+0.69** against N3's cosine-0.1 two-seed −0.24, less the 0.06 size credit); both halves clear the bar (5k +1.32, val +0.41); under a genuine fine-tune DepGraph's widths close its 2.11× lead over N3

Sitting 7 Oct wave 18 (registered 13:04, before submit). `tree_v10k`, wave 11b's recipe (SGD lr 0.1, cosine, wd 5e-4, 100 epochs, `select=last`, origin control, P, seed 42), from `tree_v10j/runs/job22342029/traj_models`. That is wave 10's transplant walk: DepGraph's released 2.11× widths, L1 ranking, walk 40/10, landed `param:0.508`. COMPLETED 35 m, 7 Oct 15:40, `ise-4090-01`, exit 0, TB 0. Start check green: `final_ft from` names job22342029's `traj_models`, both recipe lines read `optim=sgd lr=0.1 cosine=1`, both final FTs kept the last epoch (pruned loss 0.01405, origin 0.00725, each equal to its best), `keep=last` 2. Checkpoint `resnet56_cifar10_dep_graph_93.53.pth`, as N3's. The size point is step 150 (params 0.508 / FLOPs 0.480). The widths plan fixed it and nothing chose it on val, so its 10k is quoted (`full_test_dacc`). Call (registered): lift_cos = T_cos − (−0.24) − 0.06 at 10k on the size point. The −0.24 is N3's two-seed cosine-0.1 mean at 2.11× (§223 / §246: −0.36 / −0.11) and the 0.06 the size credit. **ALLOCATION** ≥ +0.32 (2/3 of the 0.48 pp gap to DepGraph), **NOT-ALLOCATION** ≤ +0.20, PARTIAL between. Reported: T_cos against DepGraph's +0.24 directly, both halves, honest.

| Row | Params / FLOPs | Walk 5k | Final 5k / val / **10k** | Honest | Lift after the 0.06 credit, 5k / val / **10k** |
|---|---|---|---|---|---|
| **Transplant, cosine-0.1-last** | 0.508 / 0.480 | −0.56 | +1.04 / +0.34 / **+0.69** | +0.62 | +1.32 / +0.41 / **+0.87** |
| N3 2.11× cosine-0.1, two-seed mean (§223 / §246) | 0.470 / 0.463 | −0.22 | −0.34 / −0.13 / −0.24 | −0.67 | — |
| Transplant, lr 0.01 (§249; kept epoch 1) | 0.508 / 0.480 | −0.56 | −0.72 / +0.44 / −0.14 | −0.66 | lr 0.01 call +0.25, PARTIAL |
| Origin, this run (N3's cosine-0.1 runs, 10k) | 1 | 0 | +0.98 / +1.06 / **+1.02** (+0.64 / +0.63) | — | — |
| DepGraph's own 2.11× model (head-to-head) | — | — | 10k **+0.24** | — | — |

N3's val half is derived as 2 × 10k − 5k (the halves are equal; seeds 42 / 43: −0.40 / +0.14).

**Read.**
- *Call: **ALLOCATION**, +0.87,* 0.55 above the bar. Both halves clear it (5k +1.32, val +0.41), unlike §249's split halves (−0.42 / +0.92). Under the genuine 100-epoch fine-tune Q7 recommends, DepGraph's widths walked by our pipeline land 0.93 pp above our own N3 walk at 10k (+0.69 against −0.24), for +0.017 FLOPs.
- *Against each run's own origin.* This run's origin gained +1.02 at 10k, against +0.64 / +0.63 in N3's two runs, so the same unpruned checkpoint under the same recipe moves about 0.4 between runs. Measured against its own retrained origin, the transplant is −0.33 and N3 −0.87 (two-seed). That lift after the credit is +0.48, still ALLOCATION.
- *Against DepGraph's own model.* T_cos is +0.69 at 10k against DepGraph's +0.24, +0.45 raw. Our origin under the same fine-tune gains +1.02, so this mixes DepGraph's widths with our longer, stronger recovery. It is not a like-for-like model comparison: **never "beats"**. What it shows is that, given DepGraph's widths, our walk and fine-tune reach DepGraph's own accuracy at 2.11×. Its lead over N3 is where it cuts.
- *What it reverses.* §249's lr 0.01 read (PARTIAL +0.25: "the gap is in DepGraph's training") was walk + 1 epoch, because the final FT kept epoch 1 on both sides (§235). On genuine endpoints the widths explain the gap. The mechanism is the thin pair's: DepGraph's copy keeps the residual streams at 13 / 32 / 42 of 16 / 32 / 64, while N3's mild walk cuts them to about 2/3 (§234).
- *For the agent.* The 2.11× gap to DepGraph is an allocation gap, which is what a frozen agent is meant to learn. v10 has not learned it yet (FLAT §248; its reward saw the lever, §259).
- One walk and one fine-tune seed. Queued: the seed-43 transplant walk (22374250, lr 0.01, wave 18 c), the lr 0.01 endpoint re-read (22342667), and the VGG-19 C100 transplant (22342030, R) with its cosine-0.1 re-read.
- Do not lock. Never an agent row.

## 263. Wave 9: residual-full allocation walk (`inner`), κ 0.6, seed 42 (**22341865**) — PRELIM, provisional (the call is two-seed at κ 0.6 and 0.8); r56-w4 **−2.40 @ 0.595 / FLOPs 0.580**; sens − inner **−0.40** (STRUCTURAL side); inner − uniform **+0.94**; r20-w2 inner −0.78 (sens / uniform −2.92)

Sitting 7 Oct wave 9 (registered 06:15, before submit). `tree_v10i`, `SPECTRA_ALLOC_KIND=inner`, undershoot 0.02, `SIZE_MATCH = SIZE_POINTS = param:0.6`, 5-rate menu, landed, 6 passes, P, loader crop+flip, walk 40/10, 100-epoch final FT + origin, deterministic, seed 42. COMPLETED 3 h 7 m, 7 Oct 15:53, `cs-4090-07`, exit 0, TB 0. Start check green. The env shows kind `inner`, the alloc lines say "3 coupled groups held at full width", and r56-w4 lands with every residual stream full (4 / 8 / 16) and inner convs 2 / 5 / 9–10, which is the planned 2 / 5 / 9. As registered, r20-w2's plan keeps x0.621, above κ, so it finished by the logged strongest-cut path (`every group at its target … strongest legal cut from here`, at x0.646); it is reported only. Every final FT kept a late epoch (best loss within 0.005 of epoch 100's). Call (registered): gap = sens − inner on r56-w4, same seeds. At κ 0.6 and κ 0.8 on the two-seed mean, **STRUCTURAL** if gap ≤ +0.3 at both, **SENS-ADDS** if ≥ +0.5 at both, PARTIAL otherwise. Seed 42 alone is provisional.

| Net, κ 0.6, seed 42 | Arm | Params / FLOPs | Residual s1 / s2 / s3 | Inner s1 / s2 / s3 | Walk 5k | **Final 5k** | Honest | Sens − arm |
|---|---|---|---|---|---|---|---|---|
| **r56-w4** | **inner** | 0.595 / 0.580 | 4 / 8 / 16 | 2 / 5 / 9–10 | −2.92 | **−2.40** | +0.46 | **−0.40** |
| r56-w4 | sens (§230) | 0.600 / 0.572 | 4 / 8 / 16 | 2–4 / 2–8 / 4–16 | −2.80 | −2.80 | −0.50 | — |
| r56-w4 | uniform (§230) | 0.599 / 0.582 | — | — | — | −3.34 | — | +0.54 |
| r56-w4 | mild (§212) | 0.600 / 0.453 | 2 / 5 / 13 | — | −5.96 | −5.06 | +0.78 | +2.26 |
| r56-w4 | v10 ep0127 TEST (§248) | 0.600 | — | — | — | −5.28 | — | — |
| r20-w2 | inner | 0.582 / 0.722 | 2 / 4 / 8 | 2 / 2 / 3–5 | −2.68 | **−0.78** | −1.32 | −2.14 |
| r20-w2 | sens (§230) | 0.595 / 0.800 | 2 / 4 / 5 | 2 / 2–4 / 5–8 | −4.74 | −2.92 | −1.46 | — |
| r20-w2 | uniform (§230) | 0.581 / 0.741 | — | — | — | −2.92 | — | 0.00 |
| origin r56-w4 / r20-w2 | inner run | 1 | — | — | 0 | +0.06 / +3.22 | — | — |

**Read (provisional).**
- *Gap on seed 42: −0.40, the STRUCTURAL side.* Holding the residual streams full and giving the remaining groups one uniform keep matches or beats the sensitivity plan on r56-w4. Both keep the streams at 4 / 8 / 16. Sens spreads the inner convs unevenly (2–16), while `inner` keeps them even (2 / 5 / 9–10), and the final is 0.40 better at 0.005 lower params and 0.008 more FLOPs. The seed spread on this net is ~0.7 pp (§242), so seed 42 alone does not decide; 22341867 (κ 0.6, seed 43) and 22341866 / 70 (κ 0.8) are R.
- *Inner − uniform +0.94; inner − mild +2.66; +2.88 over v10's ep0127 TEST.* The residual-full rule alone carries the whole non-learned lever over the uniform cut on this seed (sens − uniform was +0.54 here, +1.09 on two seeds, §254).
- *Guard, r20-w2.* `inner` keeps r20's stage-3 stream full (8), where sens cut it to 5, and its walk ends 2.1 pp better (−2.68 against −4.74). After the final FT it is +2.14 over both sens and uniform. Honest is close (−1.32 against −1.46), so the difference is in the walk's endpoint, and the final FT carries it. The plan above κ makes this a strongest-cut walk: reported, not gating.
- Wave 20's cosine-0.1-last re-read of this run (22376020) has its `afterok` met. Do not lock. Never an agent row.

## 264. Wave 19 at κ 0.6, seed 42: the allocation lever under cosine-0.1-last (**22374680** sens / **22374681** uniform, re-fine-tuned from 22340391 / 92) — PRELIM, provisional (the call is the two-seed mean); lever_cos **+1.48** at 5k (lr 0.01 +0.54) and **+1.56** at 10k (+0.84): the SURVIVES side on seed 42; the stronger fine-tune lifts sens 0.12 and costs uniform 0.82

Sitting 7 Oct wave 19 (registered 13:14, before submit). `tree_v10k`, cosine from lr 0.1, keep last, 100 epochs, origin control, P, from the κ 0.6 thin walks' saved candidates; final-FT seed 42, the walk's (verified in the env and submit lines). 22374680 COMPLETED 50 m, 15:56, `ise-4090-06`; 22374681 COMPLETED 49 m, 16:07, `cs-4090-10`; both exit 0, TB 0, every final FT kept the last epoch. The walk is the lr 0.01 cell's own, so only the final FT differs. Call (registered): r56-w4 5k at the landed point, two-seed mean of lever_cos = sens − uniform. **SURVIVES** ≥ +1.0, **ABSORBED** ≤ +0.3, WEAK between. Reported: lever_cos − 1.09, the two-seed bar_cos = sens − mild against +2.54, 10k, honest, and r20-w2.

| r56-w4, κ 0.6, seed 42 | Params | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / 10k (§230) | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / 10k |
|---|---|---|---|---|---|---|
| Sens | 0.600 | −2.68 / −2.40 / **−2.54** | −0.68 | −2.80 / −2.87 | +0.50 | +0.12 / +0.33 |
| Uniform | 0.599 | −4.16 / −4.04 / **−4.10** | −0.78 | −3.34 / −3.71 | +0.36 | −0.82 / −0.39 |
| **Lever, sens − uniform** | | **+1.48** / +1.64 / **+1.56** | | +0.54 / +0.84 | | +0.94 / +0.72 |

Guard, r20-w2 (sens at 0.595, uniform at 0.581), cosine 5k / 10k: sens −2.50 / −1.70, uniform −2.14 / −1.54, so sens − uniform is −0.36 / −0.16 (lr 0.01: 0.00 / −0.10). Both r20 origins gain +4.5 to +5.1 at 5k under cosine-0.1, against +3.3 to +3.4 at lr 0.01.

**Read (provisional).**
- *Seed 42: the SURVIVES side.* Under the fine-tune Q7 recommends, sens − uniform is +1.48 at 5k, above the +1.0 bar. It agrees on the val half (+1.64) and at 10k (+1.56). The lever grows rather than shrinks (+0.94 over lr 0.01 at 5k). That is the "capacity" branch of wave 19's question: a stronger fine-tune does not repair what the uniform cut removed from the residual streams. lever_cos − 1.09 (the two-seed lr 0.01 lever, §254) is +0.39 on this seed.
- *Why it grows.* Uniform loses 0.82 at 5k under cosine-0.1 while sens gains 0.12. Both thin r56-w4 origins lose accuracy under the stronger schedule (−0.68 / −0.78 at 5k), so lr 0.1 cosine is not a free gain on this small net. Honest, which subtracts the origin's change, therefore rises for every arm here. The registered call reads raw Δ at the landed point, so the origin's drop does not enter it.
- *Guard.* r20-w2 is level, with sens 0.36 behind at 5k and 0.16 at 10k, while uniform lands 0.014 lower in params. Its undertrained origins gain 4.5–5.1 pp under the stronger fine-tune.
- Seed 43's pair (22374686 / 87) and both mild arms (22374685 / 88; bar_cos) are R. The call is their two-seed mean. Do not lock. Never an agent row.

## 265. Wave 9: residual-full allocation walk (`inner`), κ 0.6, seed 43 (**22341867**) — PRELIM; two-seed sens − inner **+0.18** at 5k (val −0.74, 10k −0.28): κ 0.6 is on the **STRUCTURAL** side, and the call waits on κ 0.8; inner − uniform **+0.91**, so the residual-full rule carries most of sens's +1.09 lever

Sitting 7 Oct wave 9 (registered 06:15, before submit). §263's recipe with `SPECTRA_SEED=43` (verified in the env). COMPLETED 3 h 2 m, 7 Oct 16:18, `ise-4090-04`, exit 0, TB 0, no fallback. r56-w4 lands on seed 42's architecture exactly: params 0.595 / FLOPs 0.580, every residual stream full (4 / 8 / 16), inner convs 2 / 5 / 9–10. r20-w2 again finished by the logged strongest-cut path (plan x0.621 above κ) and is reported only. Every final FT kept a late epoch (best loss within 0.005 of epoch 100's). The 10k is the reader's `full_test_dacc` at the landed point, which is the `val_best` model, as in §255. Call (registered): gap = sens − inner on r56-w4, same seeds, two-seed mean at κ 0.6 and κ 0.8. **STRUCTURAL** if ≤ +0.3 at both, **SENS-ADDS** if ≥ +0.5 at both, PARTIAL otherwise.

| r56-w4, κ 0.6 | Seed 42: 5k / val / 10k | Seed 43: 5k / val / 10k | **Two-seed** 5k / val / 10k |
|---|---|---|---|
| Sens α 0.5 (§230 / §242) | −2.80 / −2.94 / −2.87 | −2.08 / −2.82 / −2.45 | −2.44 / −2.88 / −2.66 |
| **Inner** (§263 / this) | −2.40 / −2.12 / −2.26 | **−2.84** / −2.16 / −2.50 | **−2.62** / −2.14 / −2.38 |
| Uniform (§230 / §254) | −3.34 / −4.08 / −3.71 | −3.72 / −3.92 / −3.82 | −3.53 / −4.00 / −3.77 |
| Mild-landed (§212 / §247), 5k | −5.06 | −4.90 | −4.98 |
| **Gap sens − inner** | −0.40 / −0.82 / −0.61 | +0.76 / −0.66 / +0.05 | **+0.18** / −0.74 / −0.28 |
| Inner − uniform | +0.94 / +1.96 / +1.45 | +0.88 / +1.76 / +1.32 | **+0.91** / +1.86 / +1.39 |

Seed 43 inner, r56-w4: walk −2.60, final −2.84 (gain −0.24), honest −0.76; its origin gained +0.52 at 5k (+0.38 at 10k), against +0.06 in the seed-42 run. r20-w2 origin +3.38.

Guard, r20-w2 (reported; strongest-cut path), two-seed 5k / 10k: inner **−0.98** / −0.35 at params 0.582 / FLOPs 0.722; sens −3.19 / −2.84 at 0.595 / 0.800; uniform −2.81 / −2.42 at 0.581 / 0.741. Seed 43 alone: inner −1.18, sens −3.46, uniform −2.70.

**Read.**
- *κ 0.6 on two seeds: the STRUCTURAL side.* Sens − inner is +0.18 at 5k, 0.12 under the +0.3 line, on a per-seed spread of 1.16 (−0.40 / +0.76). The val half and 10k favour inner (−0.74 / −0.28). Seed 43's +0.76 at 5k comes from the halves splitting in opposite directions: sens s43 is 0.74 better on 5k than on val, and inner s43 is 0.68 worse. Inner's two seeds land on one architecture and sit 0.04 apart on val, 0.44 on 5k. The registered call needs κ 0.8 as well (22341866 / 70, R).
- *Decomposition of the bar (5k, two seeds).* Sens beats mild by +2.54 (§247). The even cut (uniform − mild) gives +1.45, holding the residual streams full (inner − uniform) +0.91, and the sensitivity measurement beyond that (sens − inner) +0.18. At 10k inner − uniform is +1.39, above sens − uniform (+1.1). That answers §254's question at κ 0.6: what sens adds over uniform is the full residual streams, within noise. The rule is PFEC's (Li et al. 2017, wave 9's registration), not a SPECTRA finding.
- *Guard.* On r20-w2 `inner` keeps the stage-3 stream full (8) where sens cuts it to 5. It ends 2.21 above sens at 5k on two seeds, with fewer params and FLOPs. The plan above κ makes this a strongest-cut walk: reported, not gating.
- *Beside v10 (reported).* The two-seed inner mean (−2.62) is 2.66 above the v10 ep0127 TEST at κ 0.6 (−5.28, §248).
- Wave 20's cosine-0.1-last re-read of this run (22376022) has its `afterok` met. κ 0.8 (22341866 / 70) and κ 0.35 (22341871) are R. Do not lock. Never an agent row.

## 266. Wave 19 at κ 0.6, seed 42: mild-landed under cosine-0.1-last (**22374685**, re-fine-tuned from 22156062) — PRELIM, reported; bar_cos sens − mild **+2.92** at 5k (lr 0.01 +2.26) and **+2.71** at 10k (+2.33); mild changes −0.54 at 5k and −0.05 at 10k under the stronger fine-tune

Sitting 7 Oct wave 19 (registered 13:14, before submit). `tree_v10k`, §264's recipe (cosine from lr 0.1, keep last, 100 epochs, origin control, P), from the saved candidates of §212's mild-landed κ 0.6 walk (`tree_v10/runs/job22156062`); final-FT seed 42 (verified, §264). COMPLETED 50 m, 16:31, `ise-4090-01`, exit 0, TB 0, kept the last epoch. Registered: the two-seed bar_cos = sens − mild is reported against +2.54 (§247), not called.

| r56-w4, κ 0.6, seed 42 | Params / FLOPs | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / val / 10k | cos − lr 0.01, 5k / val / 10k |
|---|---|---|---|---|---|
| Mild-landed (§212) | 0.600 / 0.453 | −5.60 / −4.90 / **−5.25** | −0.76 | −5.06 / −5.34 / −5.20 | −0.54 / +0.44 / −0.05 |
| Sens (§264) | 0.600 / 0.572 | −2.68 / −2.40 / **−2.54** | −0.68 | −2.80 / −2.94 / −2.87 | +0.12 / +0.54 / +0.33 |
| Uniform (§264) | 0.599 / 0.582 | −4.16 / −4.04 / **−4.10** | −0.78 | −3.34 / −4.08 / −3.71 | −0.82 / +0.04 / −0.39 |
| **Bar, sens − mild** | | **+2.92** / +2.50 / **+2.71** | | +2.26 / +2.40 / +2.33 | +0.66 / +0.10 / +0.38 |
| Lever, sens − uniform (§264) | | +1.48 / +1.64 / +1.56 | | +0.54 / +1.14 / +0.84 | +0.94 / +0.50 / +0.72 |

Mild's lr 0.01 val is 2 × 10k − 5k (±0.01 from rounding). Guard, r20-w2 (mild at 0.584 / FLOPs 0.674), cosine 5k / 10k: mild −2.84 / −1.87, so sens − mild is +0.34 / +0.17 (lr 0.01: −0.06 / −0.32). Mild's r20 origin gains +4.56 at 5k under cosine-0.1 (lr 0.01 +3.32).

**Read (provisional).**
- *Seed-42 bar_cos +2.92 at 5k, +0.66 above its lr 0.01 read.* On the 5k half the stronger fine-tune costs both arms that cut the residual streams (uniform −0.82, mild −0.54) and lifts sens (+0.12). On the val half all three gain or hold (sens +0.54, mild +0.44, uniform +0.04), so the bar barely moves there (+0.10) while the lever still grows (+0.50). The three thin r56-w4 origins lose 0.68–0.78 at 5k under cosine-0.1.
- *On seed 42 the stronger fine-tune keeps or widens both gaps on every view; it closes neither.* The two-seed bar_cos against +2.54 and the lever_cos call wait on 22374686 / 87 / 88 (R). Do not lock. Never an agent row.

## 267. Wave 10 transplant: DepGraph's own VGG-19 C100 widths at params 0.061 / FLOPs 0.109, walked and fine-tuned by our pipeline (**22342030**) — PRELIM, reported; 10k **−7.43**, below the MATCH bar (≥ −3.47; DepGraph's own −2.97), but the final FT kept **epoch 1** and lost 1.56 at 5k against the walk; the genuine-endpoint reads are 22342668 / 22374249

Sitting 7 Oct, wave 10 (registered 06:50, before submit). `tree_v10j`, `SPECTRA_ALLOC_KIND=widths` from `configs/widths_depgraph_vgg19_c100_8.84x.json`, undershoot 0, `SPECTRA_STEM_ROWS=0`, catalog `input_catalog_l_depgraph_vgg19_c100.json`, landed `param:0.061`, L1 ranking, P, loader crop+flip, walk 40/10, 100-ep lr 0.01 final FT + origin, deterministic, seed 42. COMPLETED 1 h 52 m, 7 Oct 16:44, `ise-4090-02`, exit 0, TB 0, no fallback. Start check green: `widths of widths_depgraph_vgg19_c100_8.84x.json plan keeps x0.061 … over 16 groups`, with no "not named" suffix. Checkpoint `vgg19_cifar100_dep_graph_73.5.pth` (5k 0.7396, val 0.7304). As in §249, the walk has one landed candidate (step 46), so `val_best` and the size point are one model. The 10k is `full_test_dacc` on that row, and nothing chose it on val: the widths plan fixed it. Read (registered): 10k against DepGraph's own −2.97 at 9.02× (paper −3.11 at 8.84×), **MATCH** if ≥ −3.47; reported, not gating.

| Row | Params / FLOPs | Walk 5k / val / 10k | Final 5k / val / **10k** | Honest |
|---|---|---|---|---|
| **Transplant** | 0.0610 / 0.1094 | −6.64 / −5.90 / −6.27 | −8.20 / −6.66 / **−7.43** | −1.98 |
| Origin (this run) | 1 / 1 | 0 | +0.42 / +0.80 / +0.61 | — |
| DepGraph's own model (h2h 21943448) | exact copy 0.0608 / 0.1104 | — | 10k **−2.97** | — |

**Widths (the copy).** Convs 4 / 30 / 38 / 109 / 102 / 166 / 66 / 243 / 256 / 34 / 30 / 14 / 18 / 23 / 25 / 46, of 64 / 64 / 128 / 128 / 256 ×4 / 512 ×8. DepGraph cuts the stem conv to 4 of 64, keeps 26–95 % of the 256-wide convs and half of the first 512-wide one, and cuts the last seven to 14–46 channels (3–9 %). Params sit mostly in those late convs and FLOPs in the early ones, so the copy keeps FLOPs 0.109 at params 0.061.

**Read.**
- *Registered read: below the MATCH bar.* At 10k the copy is −7.43, 3.96 under the −3.47 bar and 4.46 under DepGraph's own model on the same architecture. Reported, not gating.
- *The final FT kept epoch 1* (train loss 0.0349 at e1, 0.0519 at e50, 0.0389 at e100), which is §235's pattern. The kept model is the walk plus one lr 0.01 epoch, and that epoch cost 1.56 at 5k (0.76 on val). The origin control kept a late epoch and gained +0.61 at 10k, so honest is −1.98, CROSS-OFF. Even the walk endpoint (10k −6.27) is 2.80 under the bar.
- *What decides it.* On R56 the same epoch-1 pattern read PARTIAL, and the cosine-0.1-last re-read reversed it to ALLOCATION, +0.83 at 10k over the lr 0.01 final (§249 → §262). Wave 18 (d) **22374249** (cosine-0.1-last) and wave 11's **22342668** (select=last) re-fine-tune this run's saved candidate, and both have their `afterok` met. Do not quote −7.43 as what our pipeline reaches at DepGraph's widths. Never "beats" either way.
- Do not lock. Never an agent row.

## 268. Wave 19 at κ 0.6, seed 43: sens / uniform under cosine-0.1-last (**22374686 / 22374687**, re-fine-tuned from 22341281 / 82) — PRELIM; two-seed lever_cos **+1.44 → SURVIVES** at 5k (seed 42 +1.48, seed 43 +1.40), and +1.44 on val and at 10k; lr 0.01 two-seed +1.09, so the lever holds under the stronger fine-tune and its seed spread falls from 1.10 to 0.08

Sitting 7 Oct wave 19 (registered 13:14, before submit). §264's recipe at final-FT seed 43, the walk's (verified: `seed=43`, `SPECTRA_SEED': '43'` in the env, `SPECTRA_SEED=43` in the submit line). 22374686 COMPLETED 52 m, `cs-4090-07`; 22374687 COMPLETED 50 m, `ise-4090-06`; both ~16:45, exit 0, TB 0, and every final FT kept the last epoch. Call (registered): r56-w4 5k at the landed point, two-seed mean of lever_cos = sens − uniform. **SURVIVES** ≥ +1.0, **ABSORBED** ≤ +0.3, WEAK between. Reported: lever_cos − 1.09, the two-seed bar_cos against +2.54, 10k, honest, and r20-w2.

| r56-w4, κ 0.6 | Seed 42, cos: 5k / val / 10k (§264) | Seed 43, cos: 5k / val / 10k | **Two-seed, cos** | Two-seed, lr 0.01 (§254, §265) | cos − lr 0.01, two-seed |
|---|---|---|---|---|---|
| Sens (0.600 / 0.597) | −2.68 / −2.40 / −2.54 | −3.06 / −2.70 / −2.88 | −2.87 / −2.55 / −2.71 | −2.44 / −2.88 / −2.66 | −0.43 / +0.33 / −0.05 |
| Uniform (0.599) | −4.16 / −4.04 / −4.10 | −4.46 / −3.94 / −4.20 | −4.31 / −3.99 / −4.15 | −3.53 / −4.00 / −3.77 | −0.78 / +0.01 / −0.38 |
| **Lever, sens − uniform** | +1.48 / +1.64 / +1.56 | +1.40 / +1.24 / +1.32 | **+1.44 / +1.44 / +1.44** | +1.09 / +1.12 / +1.11 | +0.35 / +0.32 / +0.33 |

Seed 43's thin r56-w4 origins lose 0.44 / 0.54 at 5k under cosine-0.1 (lr 0.01: +0.00 / +0.52). Honest, two-seed under cosine: sens +0.46, uniform +0.88 (lr 0.01: +0.08 / +0.56). Guard, r20-w2 (sens 0.595, uniform 0.581), seed 43 under cosine: sens −2.92 / −2.19, uniform −1.92 / −1.18 (5k / 10k); two-seed sens − uniform **−0.68 / −0.59** (lr 0.01: −0.38 / −0.43).

**Read.**
- *Call: **SURVIVES**.* The two-seed lever under cosine-0.1-last is +1.44 at 5k, 0.44 above the bar, and the val half and 10k give the same +1.44. lever_cos − 1.09 = +0.35. A stronger fine-tune does not absorb the lever, which is the capacity branch of wave 19's question, now on two seeds.
- *The seeds agree.* Seed 42 gives +1.48 and seed 43 +1.40 under cosine-0.1, against +0.54 / +1.64 at lr 0.01. Seed 43's sens loses 0.98 at 5k but gains 0.12 on val: its lr 0.01 5k read was the high one (§265).
- *Where the lever sits.* At the walk's end the two-seed lever is +1.76 (seed 42 +1.38, seed 43 +2.14). The lr 0.01 fine-tune recovers more of uniform's loss and leaves +1.09; cosine-0.1 leaves +1.44. The thin r56 origins of all four sens / uniform runs lose 0.44–0.78 at 5k under cosine-0.1, so it is not a free gain on this net.
- *Guard.* On r20-w2 uniform is ahead of sens under both fine-tunes, by 0.68 at 5k under cosine-0.1, at 0.014 fewer params. Reported only, as registered.
- *For Q7 and the paper.* If Q7 adopts cosine-0.1, the paper's κ 0.6 lever is +1.44 on two seeds, with both halves agreeing. Of that, wave 9 attributes most to holding the residual streams full at lr 0.01 (§265); wave 20 re-reads `inner` under cosine-0.1. The bar_cos (sens − mild) waits on 22374688 (R). Do not lock. Never an agent row.

## 269. Wave 19 at κ 0.6, seed 43: mild-landed under cosine-0.1-last (**22374688**, re-fine-tuned from 22341278) — PRELIM, reported; two-seed bar_cos sens − mild **+2.40** at 5k (lr 0.01 +2.54) and +2.26 at 10k (+2.43): the bar holds within 0.2 under the stronger fine-tune, and its split shifts toward the lever over uniform

Sitting 7 Oct wave 19 (registered 13:14, before submit). §264's recipe, from the saved candidates of §247's mild-landed κ 0.6 seed-43 walk (`tree_v10/runs/job22341278`); final-FT seed 43, the walk's (verified with the pair, §268). COMPLETED 49 m, 16:57, `ise-4090-10`, exit 0, TB 0, kept the last epoch. Registered: the two-seed bar_cos = sens − mild is reported against +2.54 (§247), not called. Wave 19's κ 0.6 cells are now complete (§264, §266, §268, this).

| r56-w4, κ 0.6 | Seed 42, cos: 5k / val / 10k (§266) | Seed 43, cos: 5k / val / 10k | **Two-seed, cos** | Two-seed, lr 0.01 | cos − lr 0.01, two-seed |
|---|---|---|---|---|---|
| Mild-landed (0.600 / 0.453) | −5.60 / −4.90 / −5.25 | −4.94 / −4.42 / −4.68 | −5.27 / −4.66 / −4.97 | −4.98 / −5.19 / −5.09 | −0.29 / +0.53 / +0.12 |
| Sens (§268) | −2.68 / −2.40 / −2.54 | −3.06 / −2.70 / −2.88 | −2.87 / −2.55 / −2.71 | −2.44 / −2.88 / −2.66 | −0.43 / +0.33 / −0.05 |
| **Bar, sens − mild** | +2.92 / +2.50 / +2.71 | +1.88 / +1.72 / +1.80 | **+2.40** / +2.11 / +2.26 | +2.54 / +2.31 / +2.43 | −0.14 / −0.20 / −0.17 |

Mild's lr 0.01 val halves are 2 × 10k − 5k (±0.01 from rounding). Seed 43's mild origin loses 0.94 at 5k under cosine-0.1 (lr 0.01 +0.12). Guard, r20-w2 (mild 0.584, sens 0.595): seed 43 under cosine, mild −2.24 / −1.62 (5k / 10k), origin +5.00; two-seed sens − mild −0.17 / −0.20 (lr 0.01: −0.17 / −0.43).

**Read.**
- *Bar_cos on two seeds: +2.40 at 5k, against +2.54 at lr 0.01 (reported, not called).* It holds within 0.2 on every view (−0.14 / −0.20 / −0.17). The seeds disagree more than at lr 0.01: +2.92 / +1.88, against +2.26 / +2.82.
- *The split shifts.* Under cosine-0.1 the bar is uniform − mild **+0.96** plus sens − uniform **+1.44** (§268), against +1.45 plus +1.09 at lr 0.01. The stronger fine-tune helps mild a little on val and 10k (+0.53 / +0.12) and costs uniform (−0.78 at 5k, −0.38 at 10k). At 10k it moves no arm by more than 0.4. What the even cut buys over mild shrinks, and what sens buys over uniform grows.
- *Origins.* All six thin r56-w4 origins in wave 19's κ 0.6 cells lose 0.44–0.94 at 5k under cosine-0.1.
- *For ops' v10 calls (not decided here).* The beyond-heuristic line stays sens's two-seed mean + 0.5 (§242): −1.94 at lr 0.01. The same rule on a like-for-like cosine-0.1 re-FT of a v10 point would be −2.37.
- Do not lock. Never an agent row.

## 270. Wave 9: residual-full allocation walk (`inner`), κ 0.8, seed 42 (**22341866**) — PRELIM, provisional (the call is two-seed at κ 0.6 and 0.8); r56-w4 **−1.60 @ 0.797 / FLOPs 0.775**; sens − inner **+0.30** at 5k (val +0.08, 10k +0.19), on the STRUCTURAL line; sens matches it with 10 % fewer FLOPs (0.696)

Sitting 7 Oct wave 9 (registered 06:15, before submit). §263's recipe at `param:0.8` with undershoot 0.04 (registered: the 0.02 plan keeps x0.801, above κ), seed 42 (verified in the env). COMPLETED 2 h 53 m, 7 Oct 17:00, `ise-4090-01`, exit 0, TB 0, no fallback. Start check green: both `[alloc]` lines end "3 coupled groups held at full width", and the r56-w4 plan keeps x0.755, as registered. r56-w4 lands at step 95 with every residual stream full (4 / 8 / 16) and inner convs 3 / 6 / 12–13 (planned 3 / 6 / 12). r20-w2's plan keeps x0.754, below κ, so this walk needs no strongest-cut path; it lands at 0.779. Every final FT kept a late epoch (best loss within 0.008 of epoch 100's). The 10k is as in §265. Call (registered): sens − inner on r56-w4, two-seed mean at κ 0.6 and κ 0.8; **STRUCTURAL** if ≤ +0.3 at both, **SENS-ADDS** if ≥ +0.5 at both, PARTIAL otherwise.

| r56-w4, κ 0.8, seed 42 | Params / FLOPs | Residual s1 / s2 / s3 | Inner s1 / s2 / s3 (min / median / max) | Walk 5k | Final 5k / val / **10k** | Sens − arm, 5k / val / 10k |
|---|---|---|---|---|---|---|
| **Inner** | 0.797 / 0.775 | 4 / 8 / 16 | 3 / 6 / 12–13 | −1.72 | **−1.60** / −1.12 / **−1.36** | **+0.30** / +0.08 / +0.19 |
| Sens α 0.5 (§227) | 0.800 / 0.696 | 4 / 8 / 16 | 2 / 2 / 4 · 2 / 5 / 8 · 5 / 16 / 16 | — | −1.30 / −1.04 / −1.17 | — |
| Uniform (§229) | 0.799 / 0.716 | — | — | — | −2.26 / −2.58 / −2.42 | +0.96 / +1.54 / +1.25 |
| Mild-landed (§211) | 0.799 / 0.716 | — | — | — | −2.12 / −2.30 / −2.21 | +0.82 / +1.26 / +1.04 |

Inner's origin gained +0.34 at 5k (+0.30 at 10k); its honest is −0.22. Guard, r20-w2 (reported), 5k / val / 10k: inner **+1.28** / +2.90 / +2.09 at 0.779 / FLOPs 0.856; sens +1.30 / +1.86 / +1.58 at 0.799 / 0.880; uniform +0.02 / +1.28 / +0.65 and mild −0.44 / +1.38 / +0.47, both at 0.774 / 0.818. r20's undertrained origin gains +3.48 at 5k, so every r20 Δ is positive.

**Read (provisional).**
- *Gap on seed 42: +0.30, on the STRUCTURAL line.* The val half (+0.08) and 10k (+0.19) sit under it. Seed 43 (22341870) has finished its walk on the same architecture (params 0.797 / FLOPs 0.775, inner 3 / 6 / 12–13) and is in its final FT.
- *What the call can still be (arithmetic from the registration).* κ 0.6 is +0.18 on two seeds (§265), so SENS-ADDS (≥ +0.5 at both κ) is out. With sens seed 43 at −1.32 (§255), the call is **STRUCTURAL** if seed 43's inner lands at ≥ −1.62 at 5k, and **PARTIAL** otherwise.
- *FLOPs.* At equal params sens keeps 10 % fewer FLOPs than `inner` (0.696 against 0.775). Sens cuts the high-resolution stage-1 / stage-2 inner convs hardest (medians 2 / 5) and keeps stage 3's nearly full (median 16); `inner` cuts every inner conv evenly. At κ 0.6 the two kept the same FLOPs (0.572 / 0.580). So at κ 0.8 what the sensitivity measurement adds beyond the residual rule shows up in FLOPs rather than accuracy. The call is on accuracy, so this is a caption, not a re-call.
- *Inner − uniform +0.66 at 5k, +1.06 at 10k.* On this seed the residual rule carries most of the κ 0.8 lever (sens − uniform +0.96 / +1.25).
- *Guard.* On r20-w2 `inner` is level with sens at 5k (0.02 behind) and 0.51 ahead at 10k, at 0.020 fewer params and FLOPs 0.856 against 0.880. Reported only.
- Wave 20's cosine-0.1-last re-read of this run (22376021) has its `afterok` met. Do not lock. Never an agent row.

## 271. Wave 9: residual-full allocation walk (`inner`), κ 0.8, seed 43 (**22341870**) — PRELIM; two-seed sens − inner **+0.06** at κ 0.8 (val +0.16, 10k +0.11) and **+0.18** at κ 0.6 (§265): wave 9 calls **STRUCTURAL**. Holding the residual streams full and cutting the rest evenly matches the sensitivity plan's accuracy at both keeps; at κ 0.8 sens keeps 9 % fewer FLOPs

Sitting 7 Oct wave 9 (registered 06:15, before submit). §270's recipe at `SPECTRA_SEED=43` (verified in the env). COMPLETED 2 h 44 m, 7 Oct 17:11, `ise-4090-11`, exit 0, TB 0, no fallback, no strongest-cut line. Start check green: both plans as in §270, each ending "3 coupled groups held at full width". r56-w4 lands on seed 42's architecture: step 95, params 0.797 / FLOPs 0.775, residual 4 / 8 / 16, inner 3 / 6 / 12–13. Walk −1.36, final −1.14 (gain +0.22), honest −0.24; the origin gained +0.46 at 5k (+0.43 at 10k). Every final FT kept a late epoch (best loss within 0.006 of epoch 100's). The 10k is as in §265. Call (registered 06:15): sens − inner on r56-w4, two-seed mean at κ 0.6 and κ 0.8. **STRUCTURAL** if ≤ +0.3 at both, **SENS-ADDS** if ≥ +0.5 at both, PARTIAL otherwise.

| r56-w4, κ 0.8 | Seed 42: 5k / val / 10k | Seed 43: 5k / val / 10k | **Two-seed** 5k / val / 10k | FLOPs (s42 / s43) |
|---|---|---|---|---|
| Sens α 0.5 (§227 / §255) | −1.30 / −1.04 / −1.17 | −1.32 / −1.04 / −1.18 | −1.31 / −1.04 / −1.18 | 0.696 / 0.708 |
| **Inner** (§270 / this) | −1.60 / −1.12 / −1.36 | **−1.14** / −1.28 / −1.21 | **−1.37** / −1.20 / −1.29 | 0.775 / 0.775 |
| Uniform (§229 / §255) | −2.26 / −2.58 / −2.42 | −2.62 / −2.24 / −2.43 | −2.44 / −2.41 / −2.43 | 0.716 / 0.716 |
| Mild-landed (§211 / §255) | −2.12 / −2.30 / −2.21 | −1.70 / −2.48 / −2.09 | −1.91 / −2.39 / −2.15 | 0.716 / 0.716 |
| **Gap sens − inner** | +0.30 / +0.08 / +0.19 | −0.18 / +0.24 / +0.03 | **+0.06** / +0.16 / +0.11 | |
| Inner − uniform | +0.66 / +1.46 / +1.06 | +1.48 / +0.96 / +1.22 | **+1.07** / +1.21 / +1.14 | |

| Two-seed, r56-w4 | κ 0.6 (§265): 5k / val / 10k | κ 0.8 (this): 5k / val / 10k |
|---|---|---|
| **Sens − inner (the call)** | **+0.18** / −0.74 / −0.28 | **+0.06** / +0.16 / +0.11 |
| Inner − uniform | +0.91 / +1.86 / +1.39 | +1.07 / +1.21 / +1.14 |
| Sens − uniform (the lever, §254 / §255) | +1.09 / +1.12 / +1.11 | +1.13 / +1.37 / +1.25 |
| FLOPs kept, inner / sens | 0.580 / 0.570 | 0.775 / 0.702 |

Guard, r20-w2 (reported): at κ 0.8 `inner` lands at 0.779 / FLOPs 0.856 on both seeds. Seed 43: inner +1.90 / +3.38 / +2.64, sens +0.54 / +1.28 / +0.91. Two-seed inner − sens is +0.67 at 5k and +1.12 at 10k.

**Read.**
- *Call: **STRUCTURAL**.* On two seeds, sens − inner is +0.18 at κ 0.6 and +0.06 at κ 0.8 at 5k, both under +0.3. The val half and 10k agree at both keeps: at most +0.16, and inner is ahead at κ 0.6. On the thin ResNets the accuracy lever of the non-learned sensitivity plan is the residual rule. Holding the streams full and cutting every other group evenly gives +0.91 / +1.07 of sens's +1.09 / +1.13 over uniform (83 % / 95 %). That answers §254's question at both keeps. The rule is PFEC's (Li et al. 2017), so the paper quotes it as known structure, not as a SPECTRA finding.
- *What the measurement still buys: FLOPs at κ 0.8.* There sens keeps FLOPs 0.702 against inner's 0.775, 9 % fewer at the same accuracy, because it cuts the high-resolution stage-1 / stage-2 inner convs hardest (§270). At κ 0.6 the two keep the same FLOPs. Caption both axes.
- *For the agent and the baselines (not decided here).* `inner` reaches sens's accuracy with no sensitivity measurement, so at κ 0.6–0.8 it is an equally strong non-learned baseline and belongs beside sens in the same-loop table, with its FLOPs. The accuracy-relevant part of the measured sensitivity in v10's state is, on these nets, whether a group is a residual stream.
- *Guard.* On r20-w2 `inner` is ahead of sens on two seeds at κ 0.8 (+0.67 at 5k), as it was at κ 0.6. Reported only, as registered.
- *Still open in wave 9:* κ 0.35 (22341871, seed 42, R), with its own bars (STRUCTURAL ≤ +0.5, SENS-ADDS ≥ +2.0). Wave 20's cosine-0.1-last re-reads 22376020–23 now have their `afterok` met; 22376024 waits on 22341871. Do not lock. Never an agent row.

## 272. Wave 11: the DepGraph R56 transplant re-fine-tuned by the paper recipe, keep last (**22342667**, from 22342029's saved candidates) — PRELIM, reported; 10k **+0.25** (Δsel +0.39 over the epoch-1 restore), level with DepGraph's own +0.24; against N3's keep-last −0.31 the lift is **+0.41** after wave 10's 0.15 credit

Sitting 7 Oct wave 11 (registered 07:10, before submit). `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from-saved `tree_v10j/runs/job22342029/traj_models`, the paper recipe (SGD lr 0.01, cosine, wd 5e-4, 100 epochs, origin control, P), seed 42. COMPLETED 35 m, 7 Oct 17:22, `ise-4090-12`, exit 0, TB 0. Start check green: `final_ft from` names 22342029, env `select=last`, and both final FTs print "kept the last epoch" (the size point's lowest train loss was 0.00115, against 0.00610 kept). Registered (wave 11): allocation and transplant rows are reported beside their lr 0.01 rows, not called. The 10k is `full_test_dacc` at the size point.

| DepGraph R56 transplant, params 0.508 / FLOPs 0.480 | Final FT | 5k / val / **10k** | Origin 5k / val / 10k |
|---|---|---|---|
| Epoch-1 restore (§249, 22342029) | lr 0.01, lowest train loss (epoch 1) | −0.72 / +0.44 / −0.14 | +0.50 / +0.66 / +0.58 |
| **Keep last (this)** | lr 0.01, epoch 100 | −0.14 / +0.64 / **+0.25** | +0.38 / +0.42 / +0.40 |
| Keep last (§262, 22374229) | cosine from 0.1, epoch 100 | +1.04 / +0.34 / +0.69 | +0.98 / +1.06 / +1.02 |

| References at 2.11× (params 0.470 / FLOPs 0.463), 10k | |
|---|---|
| N3, lr 0.01 keep last (§239) | −0.31 (5k −0.30, val −0.32) |
| N3, lr 0.01 train-loss finals (§157 / §232) | −0.46 / −0.62, mean −0.54 |
| N3, cosine-0.1 two-seed (§246) | −0.24 |
| DepGraph's own 2.11× model (h2h 21943448) | **+0.24** |

| Transplant lift at 10k, like-for-like endpoints | Lift |
|---|---|
| Epoch-1 restore: −0.14 − (−0.54) − 0.15 (§249, registered) | +0.25, PARTIAL |
| **Keep last, lr 0.01: +0.25 − (−0.31) − 0.15 (this, reported)** | **+0.41** |
| Keep last, cosine-0.1: +0.69 − (−0.24) − 0.06 (§262, registered) | +0.87, ALLOCATION |

N3-last's val half is 2 × 10k − 5k. §239's keep-last slope between 2.11× and 2.57× (10.7 pp per unit FLOPs) would make the size credit 0.18 and the lift +0.38. Against each run's own retrained origin, the keep-last lift is +0.43: the transplant sits 0.15 under its origin and N3-last 0.73 under its own.

**Read.**
- *Level with DepGraph's own model.* On a genuine lr 0.01 endpoint, the transplant is +0.25 at 10k against DepGraph's +0.24. Under cosine-0.1 it is +0.69, but our unpruned origin also gains +1.02 there. Given DepGraph's widths, our walk and either genuine fine-tune reach DepGraph's accuracy. The 12:25 framing (§249) was an epoch-1 read. Never "beats".
- *Lift +0.41, reported.* It sits inside wave 10's PARTIAL band (+0.2 to +0.5). Those bars were set against the epoch-1 reference, where N3 was 0.78 behind DepGraph. On keep-last endpoints N3 is 0.55 behind, and the transplant closes 0.41 of it.
- *Δsel +0.39 at 10k.* The epoch-1 restore cost the transplant 0.39 and N3 only 0.15 at 2.11× (§239), so §249's PARTIAL was partly that artefact.
- *The halves still disagree.* After the credit, the lift is +0.01 on the 5k half and +0.81 on val. Under cosine-0.1 the halves reversed (+1.32 / +0.41, §262). On this single point the per-half spread is 0.8–1.3 pp, which is why the reads are on 10k.
- The VGG twin (22342668, keep last) and the seed-43 transplant walk (22374250) are queued. Do not lock. Never an agent row.

## 273. Wave 19 at κ 0.35, seed 42: sens / uniform under cosine-0.1-last (**22374689 / 22374690**, re-fine-tuned from 22340636 / 37) — PRELIM, provisional (the call is two-seed); lever_cos **+1.72** at 5k (lr 0.01 +1.90) and +1.45 at 10k (+1.85): the WEAK side of §243's bars, as at lr 0.01; at κ 0.35 the stronger fine-tune trims the lever rather than growing it

Sitting 7 Oct wave 19 (registered 13:14, before submit). §264's recipe, from the saved candidates of the κ 0.35 thin walks (`tree_v10h/runs/job22340636` / `37`). Final-FT seed 42, the walk's (verified: `seed=42`, `SPECTRA_SEED': '42'` in the env, `SPECTRA_SEED=42` in the submit line). 22374689 COMPLETED 57 m, 18:07, `ise-4090-11`; 22374690 COMPLETED 56 m, 18:18, `ise-4090-21`; both exit 0, TB 0, and every final FT kept the last epoch. Call (registered): r56-w4 5k at the landed point with §243's bars. **SURVIVES** ≥ +2.0, **ABSORBED** ≤ +0.5, WEAK between; seed 42 first, two-seed when the seed-43 pair lands.

| r56-w4, κ 0.35, seed 42 | Params / FLOPs | Residual s1 / s2 / s3 | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / val / 10k (§243 / §238) | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / val / 10k |
|---|---|---|---|---|---|---|---|
| Sens | 0.338 / 0.409 | 4 / 8 / 15 | −6.20 / −6.46 / **−6.33** | −1.20 | −6.00 / −6.36 / −6.18 | +0.46 | −0.20 / −0.10 / −0.15 |
| Uniform | 0.349 / 0.331 | 2 / 5 / 9 | −7.92 / −7.64 / **−7.78** | −0.80 | −7.90 / −8.16 / −8.03 | +0.60 | −0.02 / +0.52 / +0.25 |
| **Lever, sens − uniform** | | | **+1.72** / +1.18 / **+1.45** | | +1.90 / +1.80 / +1.85 | | −0.18 / −0.62 / −0.40 |

The lr 0.01 val halves are 2 × 10k − 5k (±0.01 from rounding). Guard, r20-w2 (sens 0.340, uniform 0.331), cosine 5k / 10k: sens −11.98 / −11.60, uniform −8.78 / −8.31, so sens − uniform is −3.20 / −3.29 (lr 0.01: −3.26 / −2.73). Both r20 origins gain about +4.9 at 5k under cosine-0.1 (lr 0.01 +3.2 / +3.4).

**Read (provisional).**
- *Seed 42: the WEAK side, as at lr 0.01.* lever_cos is +1.72 at 5k, 0.28 short of SURVIVES and well above ABSORBED. The val half (+1.18) and 10k (+1.45) are lower. Unlike κ 0.6 (+0.94 over lr 0.01 at 5k, §264), the stronger fine-tune trims the lever here, by 0.18 / 0.62 / 0.40: uniform gains on val and 10k (+0.52 / +0.25), and sens does not (−0.10 / −0.15).
- *FLOPs.* At κ 0.35 sens keeps FLOPs 0.409 against uniform's 0.331, 24 % more at 0.011 fewer params. At this keep its lever is bought with FLOPs, as §260's caption says.
- *Origins.* Both thin r56-w4 origins lose more under cosine-0.1 here (−1.20 / −0.80 at 5k) than in the κ 0.6 cells. The call reads raw Δ at the landed point.
- *Guard.* On r20-w2 uniform is 3.2 ahead of sens at κ 0.35 under both fine-tunes. Reported only.
- The seed-43 κ 0.35 walks (22344456 / 57) are R, and their cosine re-reads follow by `afterok`. The mild-landed κ 0.35 re-read gives bar_cos. Do not lock. Never an agent row.

## 274. Wave 17 at κ 0.8, seed 42: the allocation lever under v10's walk FT 12/4 (**22372634 / 22372635**) — PRELIM, reported beside §259; sens − mild on the reward's own view **+1.58 → VISIBLE** (40/10 +2.14, ×0.74): at κ 0.8 the reward also sees the lever, but the short budget shrinks it instead of growing it; after the 100-epoch final FT the lever is **+1.36** (40/10 +0.82)

Sitting 7 Oct, wave 17 (registered 12:30, submitted 12:31). §259's recipe at landed κ 0.8: sens α 0.5 in `tree_v10h` (§227's recipe) against mild-landed in `tree_v10` (§211's), thin pair, seed 42. The only change from 40/10 is `SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4`. 22372634 COMPLETED 1 h 20 m, 18:17, `ise-4090-18`; 22372635 COMPLETED 1 h 22 m, 18:22, `ise-4090-21`; both exit 0. The start check is green on both: env `NUM_EPOCHS` 12 / `FINETUNE_PATIENCE` 4 / seed 42, walk FT lines `Epoch …/12`, no Traceback, no fallback, and the sens job prints its `[alloc]` plan line. Every final FT in these two jobs and in the 40/10 references kept a late epoch (90–100), so all are genuine 100-epoch finals. Call (registered): §259's bars on seed 42, reported beside. d = sens − mild on r56-w4's return (`fixed target: episode ends … return`); **VISIBLE** ≥ +1.0, **HIDDEN** ≤ +0.3, PARTIAL between.

| r56-w4 at κ 0.8, seed 42 | Landed params / FLOPs, 12/4 | Residual s1 / s2 / s3; inner median | **Return 12/4** | Return 40/10 (job) | 12/4 − 40/10 |
|---|---|---|---|---|---|
| Sens | 0.7996 / 0.717 | 4 / 8 / 16; 2 / 5 / 14 | −2.84 | −0.80 (22340393) | −2.04 |
| Mild | 0.7994 / 0.716 | 3 / 7 / 14; 3 / 7 / 15 | −4.42 | −2.94 (22156061) | −1.48 |
| **d** | | | **+1.58** | +2.14 | ×0.74 |

TEST on the 5k half at the landed point (walk / final FT, 100 epochs). Sens at 12/4: −2.54 / **−0.82** (honest +1.12, origin +0.60). Mild at 12/4: −4.50 / **−2.18** (honest +2.04, origin +0.28), on the 40/10 architecture exactly (step 47, identical widths). At 40/10 (§227 / §211): sens −1.30 / −1.30 (honest −0.46), mild −2.22 / −2.12. Sens's 40/10 final kept epoch 95: its walk had already converged this net (train loss 0.197 at final-FT epoch 1, 0.189 when kept), so §227's zero gain is real, not a restored epoch. Final-FT d: **+1.36** at 12/4 against **+0.82** at 40/10.

Guard, r20-w2 (sens at 0.799, mild at 0.774; unequal size, flagged as in §227): the return d is +0.60 at 12/4 (−0.58 vs −1.18) and +1.58 at 40/10 (+2.04 vs +0.46). Final-FT TEST at 12/4: sens +0.52, mild −0.32 (d +0.84); at 40/10, +1.30 vs −0.44 (d +1.74). Both r20 origins gain +3.6 to +4.0 under the same final FT, so these finals are final-FT gains, not pruning gains (honest −1.90 / −0.16 at 12/4).

**Read.**
- *Call: VISIBLE at κ 0.8, seed 42.* At v10's own walk budget, its reward puts sens 1.58 pp above mild on r56-w4 at κ 0.8, at equal params and equal FLOPs (0.717 vs 0.716). With §259 (+4.05 on two seeds at κ 0.6) the lever is in v10's reward at both of its probe keeps. κ 0.8 does not disagree with κ 0.6, so neither keep shows the budget hiding the lever, and the M1-v10 FLAT (§248) reads as a learning failure at both.
- *At κ 0.8 the short budget shrinks the lever.* 12/4 costs sens 2.04 pp of return and mild 1.48. At κ 0.6 the order was reversed (mild lost 2.0–2.8, sens 0.3–1.1), which §259 traced to mild's thin residual streams (2 / 5 / 13) recovering slowly. At κ 0.8 mild thins the streams less (3 / 7 / 14), and that premium is gone: the lever is ×0.74 of 40/10's, against ×1.36 at κ 0.6 on the same seed.
- *Final FT.* Mild's 12/4 final (−2.18) matches its 40/10 final (−2.12) on the same architecture: 100 epochs erase the walk budget, as at κ 0.6. Sens's 12/4 walk lands on a slightly different architecture from its 40/10 walk (step 112 against 169; FLOPs 0.717 against 0.696; the same full residual streams, more kept in s2 and less in s3). Its final is 0.48 better (−0.82 against −1.30), close to this net's same-architecture noise (0.38, §226). So the final-FT lever is +1.36 at 12/4 and +0.82 at 40/10. On one seed, read that as the same sign at both budgets, not as a budget effect.
- *For v10.* On this seed the reward's lever (+1.58) is close to the final-FT lever at the same budget (+1.36). At κ 0.6 the reward overstated it about 1.8× (§259). It points the same way at both keeps.
- Seed 42 only, as registered. Not a train and not a v10 TEST; ops' M1-v10 FLAT (§248) stands. Never quote the in-walk returns as TEST. Do not lock. Never an agent row.

## 275. Wave 18 (b): DepGraph R56 sens / uniform allocation re-fine-tuned under cosine-0.1-last (**22374230 / 22374248**, from 22340523 / 24's saved candidates) — PRELIM, reported; lever_cos **−0.12** at 5k (val +0.10, 10k −0.01) → **ABSORBED**, as at lr 0.01 keep-last (§261); both arms gain about +0.7 and reach 10k **+0.26 / +0.27**, level with DepGraph's own 2.11× model (+0.24); within our pipeline DepGraph's widths (§262, +0.69) do not separate from a uniform cut on one seed

Sitting 7 Oct wave 18 (registered 13:04, before submit). `tree_v10k`, wave 11b's recipe (SGD lr 0.1, cosine, wd 5e-4, 100 epochs, `select=last`, origin control, P, seed 42), from the saved candidates in `tree_v10h/runs/job22340523` / `job22340524`. 22374230 COMPLETED 34 m, 18:41, `ise-4090-11`; 22374248 COMPLETED 35 m, 18:52, `ise-4090-12`; both exit 0, TB 0, no fallback. Start check green on both: `final_ft from` names the parent's `traj_models`, env `select=last`, both recipe lines read `optim=sgd lr=0.1 cosine=1 … select=last`, both final FTs kept the last epoch, `keep=last` 2, seed 42. The size point (`size_param0.47`: sens step 165, uniform step 113) was fixed by size, not chosen on val, so its 10k is quoted (`full_test_dacc`). Call (registered): sens − uniform at the landed point, 5k, §236's bars: **SURVIVES** ≥ +0.5, **ABSORBED** ≤ +0.15, WEAK between; reported.

| DepGraph R56 C10 at params 0.47, seed 42 | Params / FLOPs | Residual (of 16 / 32 / 64) | cosine-0.1-last 5k / val / **10k** | Origin (cosine) 5k / val / 10k | lr 0.01 keep last 5k / val / 10k (§261 / §257) | lr 0.01 epoch-1 restore (§236 / §233) |
|---|---|---|---|---|---|---|
| Sens | 0.469 / 0.398 | 16 / 32 / 64 | +0.06 / +0.46 / **+0.26** | +0.70 / +1.04 / +0.87 | −0.68 / −0.04 / −0.36 | −0.34 / −0.24 / −0.29 |
| Uniform | 0.465 / 0.472 | 11 / 22 / 44 | +0.18 / +0.36 / **+0.27** | +0.60 / +0.78 / +0.69 | −0.52 / −0.30 / −0.41 | −0.74 / −0.78 / −0.76 |
| **Lever, sens − uniform** | | | **−0.12** / +0.10 / **−0.01** | | −0.16 / +0.26 / +0.05 | +0.40 / +0.54 / +0.47 |

References at 10k under the same cosine-0.1 fine-tune (§262): the transplant of DepGraph's widths **+0.69** at 0.508 / 0.480 (origin +1.02); N3's two-seed mean −0.24 at 0.470 / 0.463 (origins +0.64 / +0.63); DepGraph's own 2.11× model **+0.24** (head-to-head 21943448).

**Read.**
- *Call: ABSORBED (reported).* Sens − uniform is −0.12 at 5k, under the +0.15 line, and the val half (+0.10) and 10k (−0.01) agree. The stronger fine-tune lifts both arms by about +0.7 (sens +0.74 / +0.62 at 5k / 10k, uniform +0.70 / +0.68) and leaves the lever at zero, as lr 0.01 did on genuine endpoints (§261: −0.16 / +0.05). §236's +0.40 was the epoch-1 restore.
- *FLOPs.* At equal params and equal accuracy sens keeps 16 % fewer FLOPs (0.398 against 0.472: 2.51× against 2.12×). On DepGraph R56 that is the sens rule's only gain, and it holds under both fine-tunes.
- *Beside the transplant, same pipeline and fine-tune (reported).* At 10k the transplant gets +0.69 at params 0.508; uniform gets +0.27 and sens +0.26 at 0.47. After a 0.07 size credit for the transplant's extra 0.04 params (§262's slope) it leads uniform by +0.35 and sens by +0.37. Against each run's own retrained origin (+1.02 / +0.69 / +0.87) the leads are +0.09 and +0.28. Origins move about 0.4 between runs of one recipe (§262), so on one seed DepGraph's widths do not separate from a uniform cut in our pipeline.
- *Beside DepGraph's own model and N3 (reported).* Both allocations reach DepGraph's own 2.11× level (+0.24) at 0.47 params, uniform at slightly fewer FLOPs (0.472 against 0.480). As in §262 this mixes the allocation with our stronger recovery: **never "beats"**. Both sit about 0.5 above N3's cosine mean (−0.24), but N3 is another walk pipeline (§157: `tree_v9c`, 5-pass mild, a FLOPs size point), and at lr 0.01 keep-last uniform was 0.10 below it (−0.41 against −0.31). The uniform − N3 gap is not an allocation measurement.
- *What it changes in §262's read.* The registered call (lift_cos +0.87 over N3 → ALLOCATION) stands as a measurement. Its mechanism line does not. The uniform walk cuts the residual streams to 11 / 22 / 44, close to N3's 2/3 cut (§234), and still reaches DepGraph's level under cosine. So "its lead over N3 is where it cuts" and "the gap is an allocation gap the agent must learn" (§262) are not supported on one seed. What holds is that our walk and fine-tune reach DepGraph's own 10k accuracy at its size or smaller, starting from DepGraph's widths, from a uniform cut, or from the sens plan.
- One walk and one fine-tune seed per arm. The seed-43 transplant walk (22374250, lr 0.01) is R. Do not lock. Never an agent row. Never call DepGraph a beat.

## 276. Wave 9 at κ 0.35, seed 42: residual-full allocation walk (`inner`, **22341871**) — PRELIM; sens − inner **+0.12** at 5k (val −1.28, 10k −0.58) → **STRUCTURAL** (bar ≤ +0.5); with §271 the residual rule carries the non-learned lever at κ 0.35, 0.6 and 0.8 (94 % of sens − uniform here); at κ 0.35 inner and sens keep the same FLOPs

Sitting 7 Oct wave 9 (registered 06:15, before submit). §263's recipe at `SIZE_MATCH = SIZE_POINTS = param:0.35`: `tree_v10i`, `SPECTRA_ALLOC_KIND=inner`, undershoot 0.02, 5-rate menu, landed, 6 passes, P, loader crop+flip, walk 40/10, 100-epoch final FT + origin, deterministic, seed 42. COMPLETED 4 h 35 m, 7 Oct ~19:08, `ise-4090-14`, exit 0, TB 0, no fallback. Start check green: env kind `inner`, the alloc lines end "3 coupled groups held at full width", and r56-w4 lands with every residual stream full (4 / 8 / 16) and the inner convs at one width per stage (2 / 3 / 5). r20-w2 reached every group target at x0.381, above κ, so it finished by the logged strongest-cut path; reported only. Every final FT kept a late epoch (best loss within 0.003 of epoch 100's). The size point was fixed by size, so its 10k is quoted (`full_test_dacc`). Call (registered 06:15): gap = sens − inner on r56-w4 at 5k, seed 42 against seed 42 (22340636). **STRUCTURAL** ≤ +0.5, **SENS-ADDS** ≥ +2.0, PARTIAL between.

| r56-w4, κ 0.35, seed 42 | Params / FLOPs | Residual s1 / s2 / s3 | Inner median s1 / s2 / s3 | Walk 5k | **Final 5k** / val / 10k | Honest | Sens − arm, 5k / val / 10k |
|---|---|---|---|---|---|---|---|
| Sens α 0.5 (§243) | 0.338 / 0.409 | 4 / 8 / 15 | 2 / 2 / 5 | −8.42 | **−6.00** / −6.36 / −6.18 | +1.96 | — |
| **Inner** (this) | 0.346 / 0.412 | **4 / 8 / 16** | 2 / 3 / 5 | −6.16 | **−6.12** / −5.08 / −5.60 | −0.42 | **+0.12** / −1.28 / −0.58 |
| Uniform (§238) | 0.349 / 0.331 | 2 / 5 / 9 | 2 / 5 / 10 | −7.60 | −7.90 / −8.16 / −8.03 | −0.90 | +1.90 / +1.80 / +1.85 |
| Mild-landed (§260) | 0.348 / 0.271 | 2 / 3 / 10 | — | −11.04 | −10.34 / −10.46 / −10.40 | +0.16 | +4.34 / +4.10 / +4.22 |

| Decomposition, r56-w4 κ 0.35, seed 42 | 5k / val / 10k |
|---|---|
| Uniform − mild | +2.44 / +2.30 / +2.37 |
| Inner − uniform (the residual rule) | +1.78 / +3.08 / +2.43 |
| **Sens − inner (the call)** | **+0.12** / −1.28 / −0.58 |
| Sens − uniform (the lever, §243) | +1.90 / +1.80 / +1.85 |

Guard, r20-w2 (reported), 5k / val / 10k: inner (0.332 / FLOPs 0.623) −7.54 / −7.20 / −7.37; sens's size point (0.340 / 0.631) −13.06 / −12.04 / −12.55; uniform's (0.331 / 0.574) −9.80 / −9.84 / −9.82; mild (0.335 / 0.575) −9.70 / −8.16 / −8.93.

**Read.**
- *Call: **STRUCTURAL** at κ 0.35, seed 42.* Sens − inner is +0.12 at 5k, under +0.5, and inner is ahead on the val half (−1.28) and at 10k (−0.58). Inner keeps 0.008 more params, about 0.1 pp at sens's slope between κ 0.35 and 0.6; with that credit the gap is +0.22, still STRUCTURAL. With §271's two-seed STRUCTURAL at κ 0.6 and 0.8, the residual rule carries the non-learned lever at all three keeps. Here it gives +1.78 of sens's +1.90 over uniform at 5k (94 %; 83 % / 95 % at κ 0.6 / 0.8).
- *FLOPs.* At κ 0.35 inner and sens keep the same FLOPs (0.412 against 0.409) and nearly the same architecture (inner medians 2 / 3 / 5 against 2 / 2 / 5; sens's s3 stream at 15 of 16). The 9 % FLOPs saving sens bought at κ 0.8 (§271) appears at neither κ 0.35 nor κ 0.6.
- *Walk and final FT.* At the walk's TEST inner leads sens by 2.26 (−6.16 against −8.42), and the final FT closes it (sens +2.42, inner +0.04). Inner's pruned r56-w4 barely trains in the final FT (train loss 0.428 → 0.421, against the origin's 0.303 → 0.183), so its final is the walk's. The cosine-0.1-last re-read of this cell (22376024, queue row 72) now has its `afterok` met.
- *Guard.* On r20-w2 inner lands at −7.54, 2.26 above uniform and 5.52 above sens's size point, as at κ 0.6 / 0.8. Reported only.
- *For the paper (not decided here).* As §271 says, the rule is PFEC's (Li et al. 2017): quote it as known structure, and put `inner` beside sens in the same-loop table with its FLOPs. Seed 42 only at κ 0.35, as registered. Do not lock. Never an agent row.

## 277. Wave 19 at κ 0.35, seed 42: mild-landed under cosine-0.1-last (**22374696**, re-fine-tuned from 22340796) — PRELIM, reported; bar_cos (sens − mild) **+3.42** at 5k (lr 0.01 +4.34) and +3.27 at 10k (+4.22); the stronger fine-tune repairs part of mild's cut (+0.72 / +0.80), so uniform − mild shrinks from +2.44 to +1.70 while the lever moves less

Sitting 7 Oct wave 19 (registered 13:14, before submit). §264's recipe, from the saved candidates of §260's mild-landed κ 0.35 walk (`tree_v10/runs/job22340796`). Final-FT seed 42, the walk's (verified in the env and the submit line). COMPLETED 1 h 8 m, 19:30, `ise-4090-19`, exit 0, TB 0, and every final FT kept the last epoch. Registered: reported beside §273's lever, as bar_cos = sens − mild against §260's +4.34. No call.

| r56-w4, κ 0.35, seed 42 | Params / FLOPs | Residual s1 / s2 / s3 | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / val / 10k | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / val / 10k |
|---|---|---|---|---|---|---|---|
| Sens (§273) | 0.338 / 0.409 | 4 / 8 / 15 | −6.20 / −6.46 / −6.33 | −1.20 | −6.00 / −6.36 / −6.18 | +0.46 | −0.20 / −0.10 / −0.15 |
| Uniform (§273) | 0.349 / 0.331 | 2 / 5 / 9 | −7.92 / −7.64 / −7.78 | −0.80 | −7.90 / −8.16 / −8.03 | +0.60 | −0.02 / +0.52 / +0.25 |
| **Mild-landed** (this) | 0.348 / 0.271 | 2 / 3 / 10 | **−9.62** / −9.58 / **−9.60** | −0.36 | −10.34 / −10.46 / −10.40 | +0.54 | **+0.72** / +0.88 / **+0.80** |
| **Bar, sens − mild** | | | **+3.42** / +3.12 / **+3.27** | | +4.34 / +4.10 / +4.22 | | −0.92 / −0.98 / −0.95 |
| Uniform − mild | | | +1.70 / +1.94 / +1.82 | | +2.44 / +2.30 / +2.37 | | −0.74 / −0.36 / −0.55 |
| Lever, sens − uniform (§273) | | | +1.72 / +1.18 / +1.45 | | +1.90 / +1.80 / +1.85 | | −0.18 / −0.62 / −0.40 |

Guard, r20-w2 (mild 0.335), cosine 5k / 10k: mild −10.00 / −8.96 (lr 0.01 −9.70 / −8.93), so under cosine mild sits 1.98 / 2.64 above sens's size point and 1.22 / 0.65 below uniform's.

**Read.**
- *Reported: bar_cos +3.42 at 5k (+3.27 at 10k), against +4.34 / +4.22 at lr 0.01.* Under the fine-tune Q7 recommends, sens stays well above the standard heuristic, but the margin shrinks by about 0.9 at every view. Almost all of the shrink is mild recovering: +0.72 at 5k and +0.80 at 10k, the largest cosine gain of the three arms (uniform −0.02 / +0.25, sens −0.20 / −0.15).
- *Where it comes from.* Uniform − mild drops from +2.44 to +1.70 at 5k, while the lever (sens − uniform) moves less (−0.18 at 5k, −0.40 at 10k; §273). So at κ 0.35 the stronger fine-tune repairs part of what mild's walk cut. At κ 0.6 the two-seed bar barely moved (+2.40 against +2.54 at 5k, §269).
- *FLOPs.* At the same params mild keeps FLOPs 0.271, against uniform's 0.331 (1.22×) and sens's 0.409 (1.51×). Both margins are bought with FLOPs at this keep, as §260's caption says.
- *Guard.* On r20-w2 mild sits above sens's size point and below uniform's under cosine, as at lr 0.01. Reported only.
- Seed 42 only. There is no seed-43 mild-landed κ 0.35 walk, so bar_cos at κ 0.35 stays one-seed. The seed-43 sens / uniform walks (22344456 / 57) are R, and their cosine re-reads follow by `afterok`. Do not lock. Never an agent row.

## 278. Wave 20, VGG-19 C100 at landed params 0.6, seed 42: sens / uniform (**22375982 / 22375983**) — PRELIM, provisional (the call is two-seed); lr 0.01 as walked, sens − uniform **+2.76** at 5k (val +2.72, 10k +2.74): the SENS-MATTERS side on a plain chain, but bought with **36 % more FLOPs** (0.749 against 0.551); every pruned final FT kept epoch 1

Sitting 7 Oct wave 20 (registered 13:43, before submit). `tree_v10h`, wave 3's recipe (sens α 0.5 / uniform, 5-rate menu, landed `param:0.6`, 6 passes, P, loader crop+flip, walk 40/10, 100-epoch final FT at lr 0.01 + origin, deterministic), on the VGG-19 C100 DepGraph checkpoint (`vgg19_cifar100_dep_graph_73.5.pth`, `input_catalog_l_depgraph_vgg19_c100.json`), seed 42. 22375982 COMPLETED 35 m, 19:16, `ise-4090-11`; 22375983 COMPLETED 1 h 13 m, 20:05, `cs-4090-01`; both exit 0, TB 0, no fallback. Start check green: env kind, catalog, `param:0.6` and seed; walk lines `Epoch …/40`; both plans print 16 groups. Uniform's plan reached every group target at x0.642, above κ, and finished by the logged strongest-cut path. Both pruned final FTs kept **epoch 1** (§235's VGG pattern; train loss 0.0052 / 0.0110 there), so "lr 0.01 as walked" is walk + 1 epoch on both arms; both origins kept a late epoch. The size point was fixed by size, so its 10k is quoted. Call (registered 13:43): VGG-19 C100, 5k at the landed point, two-seed mean, lr 0.01 as walked. Sens − uniform **SENS-MATTERS** ≥ +1.0 / **NONE** ≤ +0.3 / WEAK between; seed 42 alone is provisional; a lever bought with ≥ 10 % more FLOPs kept is captioned that way.

| VGG-19 C100, params 0.600, seed 42 | FLOPs | Conv widths 1–8 · 9–16 (of 64 64 128 128 256 ×4 · 512 ×8) | In-walk return | Walk 5k | **lr 0.01 as walked** 5k / val / 10k | Origin 5k / 10k | Honest |
|---|---|---|---|---|---|---|---|
| Sens (22375982) | 0.749 | 64 64 128 128 256 256 256 256 · 512 358 227 307 307 307 512 512 | −1.02 | −0.72 | **−1.60** / −1.10 / −1.35 | +0.12 / +0.52 | −1.00 |
| Uniform (22375983) | 0.551 | 64 31 61 61 186 205 205 205 · 410 ×7, 246 | −2.98 | −3.58 | **−4.36** / −3.82 / −4.09 | +0.36 / +0.68 | −1.14 |
| **Sens − uniform** | **1.36×** | | +1.96 | +2.86 | **+2.76** / +2.72 / **+2.74** | | +0.14 |

Seed 43's sens (22375985, COMPLETED 37 m, 19:44, `ise-4090-06`, exit 0) lands on seed 42's architecture exactly (step 26, FLOPs 0.749): return +0.34, walk −0.66, lr 0.01 as walked −1.46 / −0.28 / −0.87, origin +0.36 / +0.78, kept epoch 1. Reference, cross-pipeline and reported only: N4's 3-pass mild walk (§155, `tree_v9c`) at params 0.599 / FLOPs 0.590 got walk −2.52, final −2.94 at 5k (10k −2.97), and at 0.684 / 0.686 got −2.24 / −2.24 (10k −1.62).

**Read (provisional).**
- *Seed 42: the SENS-MATTERS side.* At equal params sens − uniform is +2.76 at 5k, and the val half (+2.72) and 10k (+2.74) agree. The lever is in the walk (+2.86 at the walk's TEST); both arms lose about 0.8 to the epoch-1 restore (honest +0.14). Seed 43's sens lands on the same architecture with a similar final (−1.46; 10k −0.87), so the two-seed call turns on seed 43's uniform (22375986, R).
- *Captioned: bought with FLOPs.* Sens keeps every layer up to conv 9 at full width and cuts only conv 10–14, the late 512-wide layers that hold the params, so at params 0.600 it keeps FLOPs 0.749. Uniform cuts every layer after conv 1, the early ones too (conv 2–4 to 31 / 61 / 61 after its strongest-cut finish), and keeps 0.551. That is 1.36× the FLOPs, far past the registered 10 % caption line. On a plain chain equal params and equal FLOPs are different comparisons, and this one is at equal params.
- *Against a mild point with more params (reported, cross-pipeline).* N4's mild walk kept params 0.684 / FLOPs 0.686 and reached −2.24 at 5k (§155). Sens reaches −1.60 at 0.600 / 0.749, with fewer params and 9 % more FLOPs. N4 ran in another pipeline (`tree_v9c`, 3-pass), so this only says sens is not merely a FLOPs-rich point. The in-pipeline mild-landed cells (22375994 / 95, R) give the registered sens − mild.
- *Structure.* VGG-19 has no residual streams, so wave 9's rule does not apply here (`inner` is uniform), and the lever is per-layer sensitivity. The plan it finds matches PFEC's VGG-16 sensitivity analysis (Li et al. 2017): the late 512-wide layers are the insensitive ones.
- *Fine-tune.* Every pruned final FT kept epoch 1, so these rows are walk + 1 epoch. The cosine-0.1-last re-reads (seed 42: 22376019 / 13 / 11, queue row 71) give genuine endpoints, reported beside the call; if Q7 adopts cosine-0.1, the cosine version is the paper's.
- Do not lock. Never an agent row.

---


## 279. Wave 20, VGG-19 C100 at landed params 0.6, seed 43: uniform (**22375986**) completes the two-seed call — PRELIM; two-seed sens − uniform **+2.77** at 5k (seeds +2.76 / +2.78; val +3.44, 10k +3.11) → **SENS-MATTERS** (≥ +1.0), captioned **bought with 1.36× FLOPs** (0.749 against 0.551 on both seeds); established at equal params only

Sitting 7 Oct wave 20 (a); recipe, start check and call as §278. 22375986 COMPLETED 1 h 16 m, 20:29, `ise-4090-15`, exit 0, TB 0, no fallback. Start check green: env `uniform`, the VGG catalog, `param:0.6`, seed 43; walk lines `Epoch …/40`; 16 groups at keep 0.76, every group target reached at x0.642, finished by the logged strongest-cut path. It lands at step 36 on seed 42's uniform architecture exactly (FLOPs 0.551), as seed 43's sens landed on seed 42's (§278). Its pruned final FT kept epoch 1 (train loss 0.0110), its origin epoch 95. Each arm has one architecture across the two seeds, so the seed spread below is fine-tune noise (walk FT, final FT, origin), not allocation noise.

| VGG-19 C100, params 0.600 | FLOPs | In-walk return | Walk 5k | **lr 0.01 as walked** 5k / val / 10k | Origin 5k / 10k | Honest |
|---|---|---|---|---|---|---|
| Sens s42 (22375982) | 0.749 | −1.02 | −0.72 | **−1.60** / −1.10 / −1.35 | +0.12 / +0.52 | −1.00 |
| Sens s43 (22375985) | 0.749 | +0.34 | −0.66 | **−1.46** / −0.28 / −0.87 | +0.36 / +0.78 | −1.16 |
| Uniform s42 (22375983) | 0.551 | −2.98 | −3.58 | **−4.36** / −3.82 / −4.09 | +0.36 / +0.68 | −1.14 |
| Uniform s43 (22375986) | 0.551 | −3.26 | −3.70 | **−4.24** / −4.44 / −4.34 | +0.16 / +0.33 | −0.70 |
| Sens − uniform, s42 · s43 | 1.36× | +1.96 · +3.60 | +2.86 · +3.04 | +2.76 / +2.72 / +2.74 · +2.78 / +4.16 / +3.47 | | +0.14 · −0.46 |
| **Two-seed mean** | | +2.78 | +2.95 | **+2.77** / +3.44 / **+3.11** | | −0.16 |

Seed spread |s43 − s42|: sens 0.14 at 5k (val 0.82, 10k 0.48), uniform 0.12 (val 0.62, 10k 0.25); the lever 0.02 at 5k (val 1.44, 10k 0.73). The 5k half, where the call is read, is the quiet one on both arms.

**Read.**
- *Call: SENS-MATTERS, at equal params.* The two-seed 5k mean is +2.77 against the +1.0 bar, and each seed alone clears it by 1.7. The val half and 10k agree on both seeds. VGG-19 has no residual streams (`inner` is uniform, wave 20's CPU check), so here per-layer sensitivity carries the lever, on a second family and dataset. On the thin ResNets the residual rule carries it (wave 9 STRUCTURAL at all three keeps, §271, §276); this chain has no such rule.
- *Caption: bought with FLOPs (registered).* On both seeds sens keeps 0.749 of the FLOPs against uniform's 0.551: 1.36×, far past the 10 % caption line. Sens keeps convs 1–9 full and cuts the late 512-wide layers, where the params sit and the FLOPs do not. The call holds at equal params; equal FLOPs is a different comparison and is not in this wave. The nearest evidence is a probe on another net: A0b on VGG-16 C10 (§209, 40-epoch recovery, never TEST) found at equal FLOPs (keep 0.6) that the sensitivity rule did not clear its bar (+0.25 against 0.54), while a random draw did (+1.17). The slide line is therefore "at equal params on a plain chain, the sensitivity plan beats an even cut by ~2.8 pp while keeping 36 % more FLOPs", never "allocation beats uniform" without the caption.
- *Where it sits.* In the walk: walk lever +2.95, final +2.77, honest −0.16 (two-seed). All four pruned final FTs restored epoch 1, so this fine-tune neither adds nor removes the lever. The return lever (+1.96 / +3.60) is in-walk, never TEST.
- *Fine-tune.* As §278, these rows are walk + 1 epoch. The cosine-0.1-last re-reads (queue row 71; seed 42 22376019 / 13 / 11, seed 43 22376010 / 14 / 22375996, PD) give genuine endpoints and are reported beside this call; if Q7 adopts cosine-0.1, the cosine version is the paper's.
- *Still to come.* Sens − mild (reported) on 22375994 / 95, R. Cross-pipeline reference only: N4's mild at params 0.684 / FLOPs 0.686 got −2.24 at 5k (§155), against sens's −1.60 / −1.46 at 0.600 / 0.749.
- *Literature.* DepGraph's own VGG-19 C100 row (73.50 → 70.39, −3.11 at 8.92× params, keep ≈ 0.11; §145) is a far harsher cut. At params 0.6 it is not a comparison point, and never a beat.
- *Open, not registered.* Whether the plain-chain lever survives at equal FLOPs needs a FLOPs-matched cell (sens landed at FLOPs 0.551, or uniform at 0.749). The fixed-target walk ends on `param:<keep>` only (`NetworkEnv`), so that is new code in a new tree: next sitting's design, not tonight's queue.
- Do not lock. Never an agent row.

---

## 280. Wave 13, DepGraph R56 C10 at landed params 0.47, seed 43: sens / uniform (**22344275 / 22344276**) — PRELIM; two-seed sens − uniform **+0.78** at 5k (seeds +0.40 / +1.16) → **SURVIVES** (≥ +0.5) as registered, but the precise 10k is **+0.48** on both seeds (+0.47 / +0.49), on the bar; every pruned row is walk + 1 epoch, and sens keeps 10–16 % fewer FLOPs

Sitting 7 Oct wave 13 (registered 10:05, before submit): the seed-43 repeat of §233 / §236 (`tree_v10h`, sens α 0.5 / uniform, `param:0.47`, the DepGraph R56 C10 catalog, 6 passes, P, loader crop+flip, walk 40/10, 100-epoch final FT at lr 0.01 + origin, deterministic), `SPECTRA_SEED=43`. 22344276 COMPLETED 2 h 43 m, 19:12, `ise-4090-15`; 22344275 COMPLETED 4 h 26 m, 20:44, `ise-4090-10`; both exit 0, TB 0, no fallback. Start check green: env kind, catalog, `param:0.47`, seed 43; walk lines `Epoch …/40`; both plans print 30 groups. Uniform lands on seed 42's architecture exactly (step 113, params 0.465 / FLOPs 0.472; plan at x0.490, then the logged strongest-cut finish, as §233). Sens does not: its plan moves slightly (median group keep 0.38 against 0.36) and the walk lands at step 201 (seed 42: 165) with FLOPs 0.425 (0.398). Both seeds keep every residual stream full (16 / 32 / 64); the inner convs differ (min–median–max s1 2–6–14 against 4–6–9, s2 4–7–32 against 7–7–32, s3 14–27–64 against 16–30–64). All four pruned final FTs kept **epoch 1** (seed 43 train loss 0.00159 / 0.00417) and all four origins a late epoch (90–100), so every pruned row is walk + 1 epoch (§235). The landed point is fixed by size, so its 10k is computed directly, as in §261. Call (queue row 88): §236's bars on the two-seed mean of sens − uniform, 5k at the landed point: **SURVIVES** ≥ +0.5 / **ABSORBED** ≤ +0.15 / WEAK between; per-arm seed spread beside it.

| DepGraph R56 C10, landed params 0.47 | Params / FLOPs | In-walk return | Walk 5k | **lr 0.01 final** 5k / val / 10k | Origin 5k / val / 10k | Honest |
|---|---|---|---|---|---|---|
| Sens s42 (22340523, §236) | 0.469 / 0.398 | −0.36 | −0.32 | **−0.34** / −0.24 / −0.29 | +0.60 / +0.52 / +0.56 | −0.62 |
| Sens s43 (22344275) | 0.469 / 0.425 | −0.94 | +0.16 | **+0.06** / −0.64 / −0.29 | +0.24 / +0.62 / +0.43 | −0.34 |
| Uniform s42 (22340524, §233) | 0.465 / 0.472 | −0.84 | −0.56 | **−0.74** / −0.78 / −0.76 | +0.46 / +0.82 / +0.64 | −0.64 |
| Uniform s43 (22344276) | 0.465 / 0.472 | −0.16 | −0.82 | **−1.10** / −0.46 / −0.78 | +0.40 / +0.48 / +0.44 | −0.68 |
| Sens − uniform, s42 · s43 | | +0.48 · −0.78 | +0.24 · +0.98 | +0.40 / +0.54 / +0.47 · +1.16 / −0.18 / +0.49 | | +0.02 · +0.34 |
| **Two-seed mean** | | −0.15 | +0.61 | **+0.78** / +0.18 / **+0.48** | | +0.18 |

Seed spread |s43 − s42|: sens 0.40 at 5k (val 0.40, 10k 0.00), uniform 0.36 (val 0.32, 10k 0.02); the lever 0.76 at 5k (val 0.72, 10k 0.02). Across seeds both arms trade accuracy between the two halves and hold the 10k within 0.02. The per-arm 5k spread is about half of §242's 0.72 on the thin pair.

**Read.**
- *Call: SURVIVES as registered, on the bar at 10k.* The two-seed 5k mean is +0.78 against +0.5. It leans on seed 43's 5k half (+1.16): seed 42 alone was WEAK (+0.40, §236), and seed 43's val half reverses the sign (−0.18). The 10k has twice the images and is quiet across seeds (per-arm spread ≤ 0.02); it gives +0.47 / +0.49, on the +0.5 line. Quote both: "+0.78 at 5k (SURVIVES), +0.48 at 10k". Never pick the half.
- *What it measures.* Every pruned row is walk + 1 epoch, so this is the walk's allocation effect: walk lever +0.61, honest +0.18 (two-seed). Under seed 42's genuine endpoints the lever was absorbed: select=last −0.16 at 5k (+0.05 at 10k, §261), cosine-0.1-last −0.12 (−0.01, §275). Seed 43's cosine re-reads (22376025 / 26, queue row 73) are unblocked now and give the two-seed endpoint read beside this call.
- *FLOPs: saved, not bought.* At equal params sens keeps 10–16 % fewer FLOPs than uniform (0.425 / 0.398 against 0.472; 2.35× / 2.51× against 2.12×), because it cuts inner convs and keeps the streams. No caption needed.
- *Against DepGraph.* These rows are walk + 1 epoch. The like-for-like comparison is §275's cosine read (both arms level with DepGraph's own +0.24 at 2.11×, seed 42); never a beat.
- Do not lock. Never an agent row.

---

## 281. Wave 19 at κ 0.8, seed 43: sens / uniform / mild-landed under cosine-0.1-last (**22374703 / 22374704 / 22374788**, re-fine-tuned from 22341283 / 84 / 22341279) — PRELIM, reported; lever_cos **+0.78** at 5k (lr 0.01 +1.30) and **+1.17** at 10k (+1.25); bar_cos +0.38 (+0.38) and +0.75 at 10k (+0.91); cosine-0.1 lowers every r56-w4 arm and its unpruned origin, as it lowers that origin at every keep in wave 19

Sitting 7 Oct wave 19 (registered 13:14, before submit). §264's recipe (`tree_v10k`, cosine from lr 0.1, select=last, 100 epochs + origin), from the saved candidates of the κ 0.8 seed-43 thin walks (§255: `tree_v10h/runs/job22341283` / `84`, `tree_v10/runs/job22341279`). Final-FT seed 43, the walk's (verified on all three: `seed=43`, `SPECTRA_SEED': '43'` in the env, `SPECTRA_SEED=43` in the submit line). 22374788 COMPLETED 40 m, 20:58, `ise-6000-05`; 22374703 COMPLETED 44 m, 21:01, `ise-cpu256-11` (its one RTX 6000); 22374704 COMPLETED 38 m, 21:08, `cs-6000-02`. All exit 0, TB 0; each ran 4 final FTs (landed point and origin, two nets), all at lr 0.1 cosine with select=last, and all 4 kept the last epoch. Registered (wave 19): κ 0.8 is reported, no bars.

| r56-w4, κ 0.8, seed 43 | Params / FLOPs | Residual s1 / s2 / s3 | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / val / 10k (§255) | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / val / 10k |
|---|---|---|---|---|---|---|---|
| Sens | 0.798 / 0.708 | 4 / 8 / 16 | −2.10 / −1.42 / **−1.76** | −0.50 | −1.32 / −1.04 / −1.18 | +0.12 | −0.78 / −0.38 / −0.58 |
| Uniform | 0.799 / 0.716 | 3 / 7 / 14 | −2.88 / −2.98 / **−2.93** | −0.68 | −2.62 / −2.24 / −2.43 | +0.48 | −0.26 / −0.74 / −0.50 |
| Mild-landed | 0.799 / 0.716 | 3 / 7 / 14 | −2.48 / −2.54 / **−2.51** | −0.82 | −1.70 / −2.48 / −2.09 | +0.22 | −0.78 / −0.06 / −0.42 |
| **Lever, sens − uniform** | | | **+0.78** / +1.56 / **+1.17** | | +1.30 / +1.20 / +1.25 | | −0.52 / +0.36 / −0.08 |
| Bar, sens − mild | | | +0.38 / +1.12 / +0.75 | | +0.38 / +1.44 / +0.91 | | 0.00 / −0.32 / −0.16 |

The lr 0.01 val halves are 2 × 10k − 5k (±0.01 from rounding). Guard, r20-w2 (sens 0.799, uniform and mild 0.774), cosine 5k / 10k: sens +1.32 / +2.02, uniform +0.86 / +1.14, mild +0.42 / +1.20, so sens − uniform is +0.46 / +0.88 (lr 0.01: +0.40 / +0.17). All three r20 origins gain +4.3 to +5.0 at 5k under cosine-0.1 (lr 0.01 +3.2 to +3.5).

**Read (reported).**
- *Lever: 10k level, 5k trimmed.* lever_cos is +1.17 at 10k against +1.25 at lr 0.01, and +0.78 at 5k against +1.30; the val half rises (+1.56 against +1.20). The bar is unchanged at 5k (+0.38) and 0.16 lower at 10k. At κ 0.8 the stronger fine-tune neither absorbs the lever nor grows it.
- *Cosine-0.1 hurts the thin r56-w4, here as at κ 0.6.* Every arm loses (5k −0.26 to −0.78, 10k −0.42 to −0.58), and so does the unpruned origin (5k −0.50 / −0.68 / −0.82, against +0.12 / +0.48 / +0.22 at lr 0.01). Seed 42's sens re-read (22374700) does the same: 5k −2.00, 10k −1.43, origin −0.80 (lr 0.01 −1.30 / −1.17). Across wave 19 the r56-w4 origin loses under cosine at every keep (5k −0.36 to −1.20), while the r20-w2 origins gain +4.2 to +5.3. Q7's cosine recommendation rests on full-width nets (N4 VGG-19 and DepGraph R56, §251, §252); on the thin r56-w4 it is a cost, not a repair, so the lever under it is a difference of losses.
- *Noise floor.* The three origin controls are one unpruned net, one recipe and one seed on three GPUs. Their 5k spread is 0.32 under cosine and 0.36 under lr 0.01. Uniform and mild land on the same r56-w4 here (§255), so their gap (−0.40 at 5k, −0.42 at 10k under cosine) is fine-tune noise on a fixed architecture, the same size.
- *FLOPs.* Sens keeps FLOPs 0.708 against 0.716; at κ 0.8 the lever is not bought with FLOPs.
- *Seed 42.* Sens done (above); the uniform and mild re-reads 22385251 / 52 (resubmitted after the 20:15 preemption, queue row 77) are PD. The two-seed κ 0.8 means follow when they land.
- Do not lock. Never an agent row.

---

## 282. Wave 14, thin pair landed κ 0.35, seed 43: sens / uniform (**22344456 / 22344457**) — PRELIM; two-seed lever sens − uniform on r56-w4 **+1.64** at 5k (seeds +1.90 / +1.38) → **WEAK** on §243's bars (SURVIVES ≥ +2.0 / ABSORBED ≤ +0.5); the 10k is **+1.98** (+1.85 / +2.11), 0.02 under the bar; both arms land on seed 42's architectures, and sens keeps 24 % more FLOPs

Sitting 7 Oct wave 14 (registered 10:20, before submit): the seed-43 repeat of §238 / §243 (`tree_v10h`, sens α 0.5 / uniform, `param:0.35`, thin catalog `input_c10_thin.json`, 6 passes, P, loader crop+flip, walk 40/10, 100-epoch final FT at lr 0.01 + origin, deterministic), `SPECTRA_SEED=43`. 22344457 COMPLETED 3 h 25 m, 20:12, `ise-4090-02`; 22344456 COMPLETED 4 h 53 m, 21:39, `ise-4090-02`; both exit 0, TB 0, no fallback. Start check green: env kind, thin catalog, `param:0.35`, seed 43; walk lines `Epoch …/40`; the plans print 12 / 30 groups (r20-w2 / r56-w4). On r56-w4 both arms land on seed 42's architectures exactly: sens at step 284, params 0.338 / FLOPs 0.409, residual 4 / 8 / 15 (inner 2–2–3, 2–2–7, 2–5–16); uniform at step 102, 0.349 / 0.331, residual 2 / 5 / 9. So do the r20-w2 landings (sens step 60, uniform 41). Unlike the DepGraph and VGG rows, the thin pruned final FTs keep late epochs (r56-w4 100 / 95 at seed 43; r20-w2 95–100), so these are genuine lr 0.01 endpoints. On r56-w4 the landed model is the `val_best` candidate, fine-tuned once, so its row is that one, as in §243 and wave 19's reader. Call (wave 14): §243's bars on the two-seed mean of sens − uniform on r56-w4, 5k at the landed point.

| r56-w4, landed κ 0.35 | Params / FLOPs | Walk 5k | **lr 0.01 final** 5k / val / 10k | Origin 5k / val / 10k |
|---|---|---|---|---|
| Sens s42 (22340636, §243) | 0.338 / 0.409 | −8.42 | **−6.00** / −6.36 / −6.18 | +0.46 / +0.40 / +0.43 |
| Sens s43 (22344456) | 0.338 / 0.409 | −5.96 | **−6.34** / −5.46 / −5.90 | +0.12 / +0.18 / +0.15 |
| Uniform s42 (22340637, §238) | 0.349 / 0.331 | −7.60 | **−7.90** / −8.16 / −8.03 | +0.60 / +0.30 / +0.45 |
| Uniform s43 (22344457) | 0.349 / 0.331 | −8.00 | **−7.72** / −8.30 / −8.01 | +0.20 / +0.48 / +0.34 |
| Sens − uniform, s42 · s43 | 1.24× FLOPs | −0.82 · +2.04 | +1.90 / +1.80 / +1.85 · +1.38 / +2.84 / +2.11 | |
| **Two-seed mean** | | +0.61 | **+1.64** / +2.32 / **+1.98** | |

Seed spread |s43 − s42|: sens 0.34 at 5k (val 0.90, 10k 0.28), uniform 0.18 (val 0.14, 10k 0.02); the lever 0.52 at 5k (val 1.04, 10k 0.26). Guard, r20-w2 (sens 0.340 / FLOPs 0.631, uniform 0.331 / 0.574), 5k / 10k: seed 42 sens −13.06 / −12.55, uniform −9.80 / −9.82; seed 43 sens −13.28 / −13.34, uniform −9.40 / −9.12; two-seed sens − uniform −3.57 / −3.48.

**Read.**
- *Call: WEAK.* The two-seed 5k mean is +1.64, between ABSORBED (+0.5) and SURVIVES (+2.0). Seed 42 alone was +1.90 (§243); seed 43 adds +1.38. The val half (+2.32) and 10k (+1.98) are higher, and the 10k sits 0.02 under the bar. On the registered half the call is WEAK; quote it with the 10k beside it.
- *Genuine endpoints, unlike §280.* The thin pruned FTs kept epochs 95–100, so this lever is what lr 0.01's full fine-tune leaves. The lever at the walk's TEST swings across seeds (−0.82 / +2.04); the final FT ends both at +1.4 to +1.9.
- *Bought with FLOPs.* At κ 0.35 sens keeps FLOPs 0.409 against 0.331 (1.24×) at 0.011 fewer params (§273's caption). On seed 42 the decomposition was uniform − mild +2.44, inner − uniform +1.78, sens − inner +0.12 (§260, §276): at κ 0.35 the non-learned lever is the residual rule, and sens adds nothing measurable over it.
- *Guard.* On r20-w2 uniform is 3.5 ahead of sens on both seeds. Reported only.
- *Cosine.* The seed-43 cosine re-reads (22374697 / 98, queue row 69) are unblocked; with §273 they give the two-seed lever_cos.
- Do not lock. Never an agent row.

---

## 283. Wave 20, VGG-19 C100 at landed params 0.6: mild-landed, seeds 42 / 43 (**22375994 / 22375995**) — PRELIM, reported; two-seed sens − mild **+1.09** at 5k (+0.94 / +1.24; val +1.91, 10k +1.50) at 1.27× mild's FLOPs; mild sits **+1.68** above uniform, so sens − uniform (+2.77, §279) is mostly the heuristic's margin over uniform, and sens adds about +1.1 over the heuristic

Sitting 7 Oct wave 20 (a), registered 13:43: the mild-landed cells (`tree_v10`, §211's recipe: the mild heuristic walk ended at `param:0.6`; the same VGG catalog, P, loader crop+flip, walk 40/10, 100-epoch final FT at lr 0.01 + origin, deterministic). 22375994 (seed 42) COMPLETED 2 h 09 m, 21:26, `ise-4090-11`; 22375995 (seed 43) COMPLETED 2 h 07 m, 21:38, `ise-4090-02`; both exit 0, TB 0, no fallback. Start check green: VGG catalog, `param:0.6`, the seed, walk lines `Epoch …/40`. Both seeds land on one architecture (step 42; params 0.600 / FLOPs 0.591; convs `64 47 94 94 186 186 186 186 · 374 374 376 415 415 415 415 415`). Both pruned final FTs kept **epoch 1** (train loss 0.0055 / 0.0054), the origins epoch 100 / 95. Registered: sens − mild is reported beside the call (§279).

| VGG-19 C100, params 0.600 | FLOPs | In-walk return | Walk 5k | **lr 0.01 as walked** 5k / val / 10k | Origin 5k / 10k | Honest |
|---|---|---|---|---|---|---|
| Mild s42 (22375994) | 0.591 | −2.32 | −2.34 | **−2.54** / −2.84 / −2.69 | +0.46 / +0.66 | −0.66 |
| Mild s43 (22375995) | 0.591 | −1.90 | −2.54 | **−2.70** / −2.36 / −2.53 | +0.92 / +0.94 | −1.08 |
| Sens − mild, s42 · s43 | 1.27× | +1.30 · +2.24 | +1.62 · +1.88 | +0.94 / +1.74 / +1.34 · +1.24 / +2.08 / +1.66 | | −0.34 · −0.08 |
| **Sens − mild, two-seed** | | +1.77 | +1.75 | **+1.09** / +1.91 / **+1.50** | | −0.21 |
| Uniform − mild, two-seed | 0.93× | | −1.20 | **−1.68** / −1.53 / −1.61 | | |

Mild's seed spread |s43 − s42|: 0.16 at 5k (val 0.48, 10k 0.16).

**Read (reported).**
- *Sens beats the standard heuristic by +1.09 at 5k on two seeds* (10k +1.50), at 1.27× mild's FLOPs (0.749 against 0.591). Mild is the heuristic the agent is benchmarked against; on VGG-19 C100 at equal params the sensitivity plan sits about 1.1 above it, with the val half (+1.91) and 10k higher.
- *The gap splits.* Sens − uniform +2.77 (§279) is mild − uniform +1.68 plus sens − mild +1.09. Mild and uniform differ mainly where uniform's strongest-cut finish cut: convs 2–4 (47 / 94 / 94 against 31 / 61 / 61) and conv 16 (415 against 246).
- *Cross-pipeline check.* N4's 3-pass mild at 0.599 / 0.590 (§155) got walk −2.52 and final −2.94 at 5k (10k −2.97). The in-pipeline mild at 0.600 / 0.591 lands within 0.4 of it (5k −2.54 / −2.70, 10k −2.69 / −2.53), so v10's mild reproduces the other pipeline's point.
- *Fine-tune.* All six VGG arms are walk + 1 epoch; the cosine re-reads (queue row 71) give the endpoints.
- *Literature.* As §279: DepGraph's own VGG-19 C100 row is at 8.92×, not a comparison point at params 0.6.
- Do not lock. Never an agent row; mild is the heuristic row.

---

## 284. Wave 18 (a): the DepGraph R56 transplant at seed 43 (**22374250**) — PRELIM; two-seed lift **+0.35** at 10k (seeds +0.25 / +0.44) → **PARTIAL** on wave 10's bars (ALLOCATION ≥ +0.5 / NOT ≤ +0.2); seed 43 lands on the same widths and its halves agree better (5k −0.08, val +0.18); both seeds are walk + 1 epoch, like N3's reference rows

Sitting 7 Oct wave 18 (registered 13:04, before submit): the seed-43 repeat of §249 (`tree_v10j`, `SPECTRA_ALLOC_KIND=widths` from `configs/widths_depgraph_r56_c10_2.11x.json`, landed `param:0.508`, the DepGraph R56 C10 catalog, L1, P, loader crop+flip, walk 40/10, 100-epoch lr 0.01 final FT + origin, deterministic), `SPECTRA_SEED=43`. COMPLETED 3 h 37 m, 21:55, `ise-4090-18`, exit 0, TB 0, no fallback. Start check green: env `widths` and the widths file, `param:0.508`, seed 43; walk lines `Epoch …/40`; plan x0.504 over 30 groups (min 0.22), as seed 42. It lands at step 150 on DepGraph's widths exactly (params 0.508 / FLOPs 0.480; residual 13 / 32 / 42; inner 4–8–11, 7–12–28, 34–54–61). The pruned final FT kept **epoch 1** (train loss 0.00097) and the origin epoch 90, so both seeds are walk + 1 epoch, like N3's lr 0.01 reference rows (§249). The widths plan fixed the size point, so its 10k is computed directly (`full_test_dacc`), as in §249. Call (queue row 66, wave 10's): lift = T − (−0.54) − 0.15 at 10k on the two-seed mean; **ALLOCATION** ≥ +0.5, **NOT-ALLOCATION** ≤ +0.2, PARTIAL between.

| Row | Walk 5k | Final 5k / val / **10k** | Origin 5k / val / 10k | Honest | Lift after the 0.15 credit, 5k / val / **10k** |
|---|---|---|---|---|---|
| Transplant s42 (22342029, §249), params 0.508 / FLOPs 0.480 | −0.56 | −0.72 / +0.44 / **−0.14** | +0.50 / +0.66 / +0.58 | −0.66 | −0.42 / +0.92 / **+0.25** |
| Transplant s43 (22374250), same widths | +0.10 | −0.08 / +0.18 / **+0.05** | +0.42 / +0.62 / +0.52 | −0.60 | +0.22 / +0.66 / **+0.44** |
| **Two-seed mean** | −0.23 | −0.40 / +0.31 / **−0.045** | | −0.63 | −0.10 / +0.79 / **+0.345** |
| N3 2.11× reference (§157 / §232, lr 0.01), 0.470 / 0.463 | −0.22 | means −0.45 / −0.63 / **−0.54** | | | |
| DepGraph's own 2.11× model (head-to-head) | | 10k **+0.24** | | | |

The transplant's seed spread |s43 − s42|: 0.64 at 5k (val 0.26, 10k 0.19). Its halves disagree by 1.16 on seed 42 and by 0.26 on seed 43.

**Read.**
- *Call: PARTIAL.* The two-seed lift is +0.345 at 10k, between NOT-ALLOCATION (+0.2) and ALLOCATION (+0.5); seed 43 alone (+0.44) is PARTIAL too. After the size credit, DepGraph's widths walked by our pipeline explain about 0.35 of the 0.78 pp between N3 and DepGraph's own model (about 45 %) under this fine-tune. The registered 10k is also the quiet read: the 5k lift swings −0.42 / +0.22 across seeds, 10k +0.25 / +0.44.
- *Like for like.* Both transplant seeds and N3's two reference rows restore epoch 1, so the comparison is walk + 1 epoch on both sides. Under the genuine endpoint, seed 42's cosine-0.1-last re-read made the transplant ALLOCATION (+0.87, §262), but a uniform cut reaches the same level under that fine-tune (§275). The seed-43 cosine re-read (22376027, queue row 73) is unblocked; wave 18's two-seed lift under cosine is reported beside this call.
- *Against DepGraph.* The transplant is DepGraph's own architecture. At lr 0.01 its two-seed 10k is −0.045 against DepGraph's +0.24, so 0.29 is left for what DepGraph does beyond widths (its sparsity training and its own fine-tune) or for fine-tune noise. Never a beat, and not "matches" at this fine-tune.
- Do not lock. Never an agent row.

---

## 285. Wave 20 (b) at κ 0.6, seed 42: wave 9's `inner` cell under cosine-0.1-last (**22376020**, re-fine-tuned from 22341865) — PRELIM, reported, provisional (two-seed with 22376022); sens − inner **−0.24** at 5k (lr 0.01 −0.40) and +0.13 at 10k (−0.61), still the STRUCTURAL side; inner − uniform **+1.72** at 5k (lr 0.01 +0.94)

Sitting 7 Oct wave 20 (registered 13:43, before submit), part (b). `tree_v10k` (wave 19's recipe: cosine from lr 0.1, keep last, 100 epochs, origin control, P), from the saved candidates of wave 9's κ 0.6 `inner` walk (`tree_v10i/runs/job22341865/traj_models`); final-FT seed 42 (verified in the env and the submit line). COMPLETED 40 m, 23:23, `ise-6000-09`, exit 0, TB 0, kept the last epoch. The walk is wave 9's own, so only the final FT differs. Registered: wave 9's sens − inner under cosine-0.1, reported beside its lr 0.01 call (STRUCTURAL at all three keeps, §271, §276); not called. Sens and uniform under cosine are §264's.

| r56-w4, κ 0.6, seed 42 | Params | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / 10k | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / 10k |
|---|---|---|---|---|---|---|
| Inner (this run; §263) | 0.595 | −2.44 / −2.90 / **−2.67** | −0.06 | −2.40 / −2.26 | +0.06 | −0.04 / −0.41 |
| Sens (§264) | 0.600 | −2.68 / −2.40 / **−2.54** | −0.68 | −2.80 / −2.87 | +0.50 | +0.12 / +0.33 |
| Uniform (§264) | 0.599 | −4.16 / −4.04 / **−4.10** | −0.78 | −3.34 / −3.71 | +0.36 | −0.82 / −0.39 |
| **Gap, sens − inner** | | **−0.24** / +0.50 / **+0.13** | | −0.40 / −0.61 | | +0.16 / +0.74 |
| Inner − uniform | | +1.72 / +1.14 / +1.43 | | +0.94 / +1.45 | | +0.78 / −0.02 |

Guard, r20-w2 (inner at 0.582, by wave 9's logged strongest-cut path), cosine 5k / 10k: inner −1.06 / +0.12 (lr 0.01 −0.78 / −0.24), against sens −2.50 / −1.70 and uniform −2.14 / −1.54 at 0.595 / 0.581. Reported only; the r20 origin gains +4.20 at 5k under cosine-0.1.

**Read (provisional).**
- *Still STRUCTURAL on seed 42.* Under the stronger fine-tune the residual-full rule still carries the lever at κ 0.6: sens is 0.24 below inner at 5k and 0.13 above it at 10k, both inside wave 9's STRUCTURAL band (≤ +0.3). The 10k gap moves +0.74 toward sens because cosine costs inner 0.41 at 10k and gains sens 0.33.
- *The structure is the lever.* Inner − uniform grows from +0.94 to +1.72 at 5k (10k unchanged, +1.43 against +1.45), because cosine costs uniform 0.82 at 5k and inner 0.04.
- *Origin control.* This run's origin changes −0.06 at 5k under cosine against −0.68 / −0.78 in §264's two runs, with the same recipe and seed. The origin control therefore moves up to 0.72 at 5k between runs, more than §281's 0.32. Honest numbers carry that noise; the gap compares final points and does not use the origin.
- *Pending.* The seed-43 re-read 22376022 is running; κ 0.8 (22376021 / 23) and κ 0.35 (22376024) are queued. Do not lock. Never an agent row.

---

## 286. Wave 11: the DepGraph VGG-19 C100 transplant re-fine-tuned by the paper recipe, keep last (**22342668**, from 22342030's saved candidates) — PRELIM, reported; 10k **−5.85** (+1.58 over the epoch-1 restore, +0.42 over the walk), still 2.38 under the MATCH bar (≥ −3.47) and 2.88 under DepGraph's own −2.97 on the same architecture

Sitting 7 Oct wave 11 (registered 07:10, before submit). `tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, from-saved `tree_v10j/runs/job22342030/traj_models`, the paper recipe (SGD lr 0.01, cosine, wd 5e-4, 100 epochs, origin control, P), seed 42. COMPLETED 18 m, 23:42, `ise-4090-12`, exit 0, TB 0. Start check green: `final_ft from` names 22342030, env `select=last`, and both final FTs print "kept the last epoch" (the size point's lowest train loss was 0.03288, against 0.03858 kept). Registered (§267): reported beside the lr 0.01 row against MATCH ≥ −3.47 at 10k; not called. The 10k is `full_test_dacc` at the size point (step 46, the walk's one landed candidate).

| VGG-19 C100 transplant, params 0.061 / FLOPs 0.109 | Final FT | 5k / val / **10k** | Origin 5k / val / 10k |
|---|---|---|---|
| Walk endpoint (§267) | none | −6.64 / −5.90 / −6.27 | — |
| Epoch-1 restore (§267, 22342030) | lr 0.01, lowest train loss (epoch 1) | −8.20 / −6.66 / −7.43 | +0.42 / +0.80 / +0.61 |
| **Keep last (this)** | lr 0.01 cosine, epoch 100 | −6.06 / −5.64 / **−5.85** | +0.30 / +1.48 / +0.89 |
| DepGraph's own model (h2h 21943448), exact copy 0.0608 / 0.1104 | DepGraph's | 10k **−2.97** | — |

**Read.**
- *Registered read: still below MATCH.* At the genuine endpoint the copy is −5.85 at 10k, 2.38 under the −3.47 bar and 2.88 under DepGraph's own model on the same architecture. Reported, not gating.
- *Selection.* Keeping the last epoch recovers +1.58 at 10k over the epoch-1 restore (+2.14 at 5k, +1.02 on val). The fine-tune adds +0.42 at 10k over the walk itself. The origin control gains +0.89 at 10k, so honest is −0.47 at 10k (+0.28 at 5k).
- *Against R56.* On R56 at 2.11× the same re-read brought the transplant level with DepGraph (§272, +0.25 against +0.24). At 9× on VGG-19 C100 it does not. What DepGraph does beyond the widths (which filters, its sparsity training, its own fine-tune) is worth about 2.9 pp at this compression under the paper recipe. The cosine-0.1-last re-read 22374249 is running and is reported beside.
- Do not lock. Never an agent row. Never a DepGraph beat.

---

## 287. Wave 21 at seed 42: MobileNetV2 ×0.5 C10 landed at params 0.6, sens / uniform / inner (**22376484 / 85 / 86**) — PRELIM, provisional (the call is two-seed; mild 22376487 still running); sens − uniform **+0.10** at 5k (val +0.76, 10k +0.43), the NONE side; inner − uniform **−0.72**; every arm ends within 0.5 of the unpruned net, so κ 0.6 is a light cut for this net

Sitting 7 Oct wave 21 (registered 14:06, before submit). MobileNetV2 ×0.5 C10 (chenyaofo 92.99 %, `configs/input_pf_mbv2x05.json`, `database_c10_thin.json`), landed `param:0.6`, L1, P, loader crop+flip, walk 40/10, 100-epoch lr 0.01 final FT (lowest train loss) + origin, deterministic, seed 42. Sens / uniform in `tree_v10h` (wave 3's recipe), `inner` in `tree_v10i` (wave 9's). 22376484 COMPLETED 2 h 9 m, 22:42, `ise-6000-08`; 22376485 2 h 57 m, 23:42, `ise-4090-07`; 22376486 1 h 46 m, 22:43, `ise-6000-04`; all exit 0, TB 0, no fallback. Start check green: env kind, the MBV2 catalog, `param:0.6` and seed 42. The `[alloc]` plans over 25 groups keep x0.578 (sens: group keep min 0.28, median 1.00), x0.580 (uniform: every group 0.75) and x0.579 (inner: min / median 0.67, "5 coupled groups held at full width"). Inner reached its plan at x0.616 and finished by the logged strongest-cut path, as wave 9's r20-w2 did. Every final FT kept a late epoch (95–100), so these are genuine endpoints. Call (registered): the two-seed mean of sens − uniform at 5k, **SENS-MATTERS** ≥ +1.0 / **NONE** ≤ +0.3 / WEAK between; seed 42 alone is provisional; a lever bought with ≥ 10 % more FLOPs is captioned. pf-w's mild walk (§195) is a probe and is not quoted.

| MBV2 ×0.5 C10, κ 0.6, seed 42 | Params / FLOPs | Step | Walk 5k | Final 5k / val / **10k** | Origin 5k / val / 10k | Honest |
|---|---|---|---|---|---|---|
| Sens (22376484) | 0.600 / 0.714 | 149 | +0.32 | **+0.34** / +0.84 / +0.59 | +0.16 / +0.54 / +0.35 | −0.14 |
| Uniform (22376485) | 0.600 / 0.591 | 97 | +0.08 | **+0.24** / +0.08 / +0.16 | +0.44 / +0.88 / +0.66 | −0.28 |
| Inner (22376486) | 0.600 / 0.508 | 124 | −0.58 | **−0.48** / −0.40 / −0.44 | +0.70 / +0.84 / +0.77 | −0.60 |
| **Lever, sens − uniform** | 1.21× FLOPs | | +0.24 | **+0.10** / +0.76 / +0.43 | | |
| Inner − uniform | 0.86× FLOPs | | −0.66 | −0.72 / −0.48 / −0.60 | | |
| Sens − inner | 1.41× FLOPs | | +0.90 | +0.82 / +1.24 / +1.03 | | |
| Sens s43 (22376488), beside | 0.600 / 0.706 | 100 | +0.36 | +0.70 / +0.80 / +0.75 | +0.42 / +0.60 / +0.51 | −0.08 |

**Read (provisional).**
- *Seed 42: the NONE side.* Sens − uniform is +0.10 at 5k, under the +0.3 line, though +0.76 on val and +0.43 at 10k: the halves disagree by 0.66. Sens keeps 21 % more FLOPs (0.714 against 0.591), so even a lever would carry that caption. Seed 43's sens is +0.70; its uniform (22376490) is running.
- *Residual-full is not the rule here.* `inner` holds the five residual streams full and cuts the rest evenly. On MobileNetV2 it lands 0.72 below uniform at 5k (0.60 at 10k), with the fewest FLOPs (0.508). On the thin ResNets the same rule is +0.9 to +1.7 above uniform (§263, §285).
- *A light cut for this net.* Every arm ends within 0.5 of the unpruned net at 5k, and sens and uniform end above it. The origin controls gain +0.16 to +0.70 under the same fine-tune. At κ 0.6 MobileNetV2 ×0.5 on CIFAR-10 sits where allocation barely matters, so a NONE here would say little about heavier cuts. Honest −0.14 / −0.28 / −0.60 is inside the origin's run-to-run noise (§285).
- *Sens seeds differ in architecture.* Sens lands at step 149 (FLOPs 0.714) on seed 42 and at step 100 (0.706) on seed 43.
- Mild (22376487) is running; sens − mild is reported with the two-seed section. Do not lock. Never an agent row.

---

## 288. Wave 20 (b) at κ 0.6, seed 43: wave 9's `inner` cell under cosine-0.1-last (**22376022**, re-fine-tuned from 22341867) — PRELIM, reported; two-seed sens − inner **−0.28** at 5k (lr 0.01 +0.18) and +0.16 at 10k (−0.28), so κ 0.6 stays on the STRUCTURAL side under cosine; inner − uniform **+1.72** at 5k on both seeds (lr 0.01 +0.91)

Sitting 7 Oct wave 20 (b), as §285. `tree_v10k`, from `tree_v10i/runs/job22341867/traj_models`; final-FT seed 43 (verified in the env and the submit line). COMPLETED 50 m, 23:59, `ise-4090-10`, exit 0, TB 0, kept the last epoch. Sens and uniform under cosine at seed 43 are §268's (22374686 / 87). Reported beside wave 9's call; not called.

| r56-w4, κ 0.6, seed 43 | Params | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / 10k | Origin 5k (lr 0.01) | cos − lr 0.01, 5k / 10k |
|---|---|---|---|---|---|---|
| Inner (this run; §265) | 0.595 | −2.74 / −3.40 / **−3.07** | −0.54 | −2.84 / −2.50 | +0.52 | +0.10 / −0.57 |
| Sens (§268) | 0.597 | −3.06 / −2.70 / **−2.88** | −0.44 | −2.08 / −2.45 | +0.00 | −0.98 / −0.43 |
| Uniform (§268) | 0.599 | −4.46 / −3.94 / **−4.20** | −0.54 | −3.72 / −3.82 | +0.52 | −0.74 / −0.38 |
| **Gap, sens − inner** | | **−0.32** / +0.70 / **+0.19** | | +0.76 / +0.05 | | −1.08 / +0.14 |
| Inner − uniform | | +1.72 / +0.54 / +1.13 | | +0.88 / +1.32 | | +0.84 / −0.19 |

| r56-w4, κ 0.6, two-seed mean | cosine-0.1-last 5k / val / **10k** | lr 0.01 5k / 10k |
|---|---|---|
| **Gap, sens − inner** (cosine seeds −0.24 / −0.32 at 5k) | **−0.28** / +0.60 / **+0.16** | +0.18 / −0.28 (§265) |
| Inner − uniform | **+1.72** / +0.84 / +1.28 | +0.91 / +1.39 |
| Lever, sens − uniform (§268) | +1.44 / +1.44 / +1.44 | +1.09 |

Guard, r20-w2, seed 43 (inner at 0.582), cosine 5k / 10k: inner −0.68 / +0.55 (lr 0.01 −1.18 / −0.46), against sens −2.92 / −2.19 and uniform −1.92 / −1.18. Reported only.

**Read.**
- *STRUCTURAL under cosine too, at κ 0.6.* The two-seed sens − inner is −0.28 at 5k, inside wave 9's STRUCTURAL band (≤ +0.3); +0.16 at 10k, +0.60 on val. Under lr 0.01 it was +0.18 (§265). The seed spread of the gap falls from 1.16 (lr 0.01: −0.40 / +0.76) to 0.08.
- *The lever is the structure.* Inner − uniform is +1.72 at 5k on both seeds (lr 0.01 +0.91; 10k +1.28 against +1.39). The two-seed lever_cos (+1.44, §268) is the residual-full structure less 0.28: sens adds nothing at 5k under the stronger fine-tune.
- *Origins agree on seed 43.* The three seed-43 cosine runs' origins change −0.44 to −0.54 at 5k, so §285's 0.72 origin spread was a seed-42 case.
- *Pending.* κ 0.8 (22376021 / 23) is running and κ 0.35 (22376024) is queued; wave 9's two-κ framing is read under cosine when κ 0.8 lands. Do not lock. Never an agent row.

---

## 289. Wave 18: the DepGraph VGG-19 C100 transplant re-fine-tuned under cosine from lr 0.1, keep last (**22374249**, from 22342030's saved candidates) — PRELIM, reported; 10k **−2.72**, above the MATCH bar (≥ −3.47) and level with DepGraph's own −2.97 on the same architecture; +3.13 over the paper recipe's keep-last (§286), so §286's 2.9 pp gap was the fine-tune

Sitting 7 Oct wave 18 (registered 13:04, before submit). `tree_v10k`, wave 19's recipe (SGD lr 0.1, cosine, wd 5e-4, 100 epochs, keep last, origin control, P), from-saved `tree_v10j/runs/job22342030/traj_models`, seed 42. COMPLETED 18 m, 23:59, `ise-4090-12`, exit 0, TB 0, no restarts. Start check green: `final_ft from` names 22342030, env `select=last`, recipe `lr=0.1 cosine=1`, and both final FTs print "kept the last epoch". The pruned model's last train loss is 0.277, against 0.039 under lr 0.01 (§286). Registered (queue row 67): reported beside wave 10's MATCH bar (≥ −3.47 at 10k); not called. The 10k is `full_test_dacc` at the size point (step 46, the walk's one landed candidate).

| VGG-19 C100 transplant, params 0.061 / FLOPs 0.109 | Final FT | 5k / val / **10k** | Origin 5k / val / 10k |
|---|---|---|---|
| Walk endpoint (§267) | none | −6.64 / −5.90 / −6.27 | — |
| Epoch-1 restore (§267, 22342030) | lr 0.01, lowest train loss (epoch 1) | −8.20 / −6.66 / −7.43 | +0.42 / +0.80 / +0.61 |
| Keep last (§286, 22342668) | lr 0.01 cosine, epoch 100 | −6.06 / −5.64 / −5.85 | +0.30 / +1.48 / +0.89 |
| **Keep last (this)** | cosine from lr 0.1, epoch 100 | −2.84 / −2.60 / **−2.72** | +0.50 / +1.24 / +0.87 |
| DepGraph's own model (h2h 21943448), exact copy 0.0608 / 0.1104 | DepGraph's | 10k **−2.97** | — |

**Read.**
- *Registered read: MATCH.* At 10k the copy is −2.72, 0.75 above the −3.47 bar and 0.25 above DepGraph's own model on the same architecture. 0.25 is inside the origin control's run-to-run spread (up to 0.72 at 5k, §285), so the wording is "level with DepGraph", never a beat. Reported, not gating.
- *The fine-tune was the gap.* The stronger fine-tune adds +3.13 at 10k over the paper recipe's keep-last and +3.55 over the walk. The origin gains +0.87, the same as under lr 0.01 (+0.89), so the extra is on the pruned model: honest +2.68 at 10k (+3.30 at 5k). §286's "about 2.9 pp beyond the widths" was therefore the fine-tune. Given DepGraph's widths and a cosine-0.1 fine-tune, our pipeline reaches DepGraph's accuracy at 9× on VGG-19 C100, as it did at 2.11× on R56 (§262, +0.69 against +0.24).
- *What it does not say.* No SPECTRA allocation was run at params 0.061, so this shows the architecture DepGraph found is reachable by our walk and fine-tune, not that our allocation would find it. One seed.
- Do not lock. Never an agent row. Never a DepGraph beat.

---

## 290. Wave 20 (b) at κ 0.8, seeds 42 and 43: wave 9's `inner` cells under cosine-0.1-last (**22376021 / 22376023**, re-fine-tuned from 22341866 / 70) — PRELIM, reported; two-seed sens − inner **−0.08** at 5k (lr 0.01 +0.06) and −0.125 at 10k (+0.11); with κ 0.6's −0.28 (§288), wave 9's STRUCTURAL framing holds at both keeps under cosine

Sitting 7 Oct wave 20 (b), as §285. `tree_v10k`, from `tree_v10i/runs/job22341866/traj_models` and `job22341870/traj_models`; final-FT seeds 42 / 43 (verified in the env and the submit lines). Both COMPLETED 49 m, 00:31, `ise-4090-07` / `ise-4090-12`, exit 0, TB 0, kept the last epoch. Sens under cosine is wave 19's (22374700 s42, COMPLETED 20:32; 22374703 s43, §281) and uniform s43 is 22374704 (§281). Uniform s42 (22385251) is still queued, so inner − uniform is seed 43 only. Reported beside wave 9's call (STRUCTURAL, §271); not called.

| r56-w4, κ 0.8 | Params / FLOPs | cosine-0.1-last 5k / val / **10k** | Origin 5k (cos) | lr 0.01 5k / 10k | cos − lr 0.01, 5k / 10k |
|---|---|---|---|---|---|
| s42 inner (this run; §270) | 0.797 / 0.775 | −2.08 / −1.02 / **−1.55** | −0.96 | −1.60 / −1.36 | −0.48 / −0.19 |
| s42 sens (22374700) | 0.800 / 0.696 | −2.00 / −0.86 / **−1.43** | −0.80 | −1.30 / −1.17 | −0.70 / −0.26 |
| **s42 gap, sens − inner** | | **+0.08** / +0.16 / **+0.12** | | +0.30 / +0.19 | −0.22 / −0.07 |
| s43 inner (this run; §271) | 0.797 / 0.775 | −1.86 / −0.92 / **−1.39** | −0.76 | −1.14 / −1.21 | −0.72 / −0.18 |
| s43 sens (22374703, §281) | 0.798 / 0.708 | −2.10 / −1.42 / **−1.76** | −0.50 | −1.32 / −1.18 | −0.78 / −0.58 |
| s43 uniform (22374704, §281) | 0.799 | −2.88 / −2.98 / **−2.93** | −0.68 | −2.62 / −2.43 | −0.26 / −0.50 |
| **s43 gap, sens − inner** | | **−0.24** / −0.50 / **−0.37** | | −0.18 / +0.03 | −0.06 / −0.40 |
| s43 inner − uniform | | +1.02 / +2.06 / +1.54 | | +1.48 / +1.22 | −0.46 / +0.32 |

| r56-w4, two-seed mean of sens − inner | cosine-0.1-last 5k / val / **10k** | lr 0.01 5k / val / 10k |
|---|---|---|
| **κ 0.8** (this section) | **−0.08** / −0.17 / **−0.125** | +0.06 / +0.16 / +0.11 (§271) |
| κ 0.6 (§288) | −0.28 / +0.60 / +0.16 | +0.18 / −0.74 / −0.28 (§265) |

Guard, r20-w2 (inner at 0.779, sens at 0.799), cosine 5k / 10k: s42 inner +1.64 / +2.84 against sens +1.40 / +2.05; s43 inner +2.82 / +3.59 against sens +1.32 / +2.02 and uniform +0.86 / +1.14 (at 0.774). Reported only.

**Read.**
- *STRUCTURAL at both keeps under cosine.* The two-seed sens − inner is −0.08 at κ 0.8 and −0.28 at κ 0.6, both inside wave 9's STRUCTURAL band (≤ +0.3), as under lr 0.01 (+0.06 / +0.18, §271). Wave 9's registered call stands, and the fine-tune Q7 recommends does not change it.
- *Cosine lowers every r56-w4 arm at κ 0.8.* Inner loses 0.48 / 0.72 at 5k, sens 0.70 / 0.78, uniform (s43) 0.26, and the origins lose 0.50–0.96 (§281's pattern). On this thin net the light cut gains nothing from lr 0.1.
- *Inner − uniform, seed 43:* +1.02 at 5k (lr 0.01 +1.48) and +1.54 at 10k. Seed 42 waits for uniform 22385251.
- *FLOPs.* At κ 0.8 sens matches inner with 9–10 % fewer FLOPs (0.696 / 0.708 against 0.775), as at lr 0.01: the re-reads keep the walks' architectures.
- κ 0.35 (22376024) is running. Do not lock. Never an agent row.

---

## 291. Wave 21 call, MobileNetV2 ×0.5 C10 at params 0.6: uniform seed 43 (**22376490**) completes the pair — PRELIM; two-seed sens − uniform **+0.67** at 5k (seeds +0.10 / +1.24; val +1.06, 10k +0.87) → **WEAK** as registered, captioned 1.2× FLOPs; seed 43's margin is uniform's epoch-1 restore (−0.86 against its own walk), and on walk endpoints the lever is +0.14

Sitting 7 Oct wave 21 (registered 14:06, before submit), as §287. 22376490 (`tree_v10h`, uniform, seed 43) COMPLETED 2 h 58 m, 00:37, `ise-4090-02`, exit 0, TB 0, no fallback. Start check green: env `uniform`, the MBV2 catalog, `param:0.6`, seed 43; the plan keeps x0.580 over 25 groups (every group 0.75), as on seed 42. It lands at step 97 on seed 42's architecture (params 0.600 / FLOPs 0.591). Its pruned final FT's best train loss (0.00549) is epoch 1's printed loss, so the default select restored **epoch 1** (walk + 1 epoch). The other three arms kept late epochs: uniform s42 epoch 95 (best 0.00581), and both sens arms below epoch 100's printed loss (best 0.00590 / 0.00493, at unprinted epochs). Call (registered): the two-seed mean of sens − uniform at 5k, **SENS-MATTERS** ≥ +1.0 / **NONE** ≤ +0.3 / WEAK between; a lever bought with ≥ 10 % more FLOPs is captioned.

| MBV2 ×0.5 C10, κ 0.6 | Params / FLOPs | Step | Final FT kept | Walk 5k | Final 5k / val / **10k** | Origin 5k / val / 10k | Honest |
|---|---|---|---|---|---|---|---|
| Sens s42 (22376484) | 0.600 / 0.714 | 149 | late | +0.32 | +0.34 / +0.84 / +0.59 | +0.16 / +0.54 / +0.35 | −0.14 |
| Uniform s42 (22376485) | 0.600 / 0.591 | 97 | epoch 95 | +0.08 | +0.24 / +0.08 / +0.16 | +0.44 / +0.88 / +0.66 | −0.28 |
| Sens s43 (22376488) | 0.600 / 0.706 | 100 | late | +0.36 | +0.70 / +0.80 / +0.75 | +0.42 / +0.60 / +0.51 | −0.08 |
| **Uniform s43 (this run)** | 0.600 / 0.591 | 97 | **epoch 1** | +0.32 | **−0.54** / −0.56 / −0.55 | +0.64 / +0.80 / +0.72 | −1.50 |

| Lever, sens − uniform | Walk 5k | Final 5k / val / **10k** | FLOPs, sens / uniform |
|---|---|---|---|
| Seed 42 | +0.24 | +0.10 / +0.76 / +0.43 | 1.21× |
| Seed 43 | +0.04 | +1.24 / +1.36 / +1.30 | 1.19× |
| **Two-seed mean** | +0.14 | **+0.67** / +1.06 / **+0.865** | 1.2× |

**Read.**
- *Call: **WEAK**, captioned "bought with 1.2× FLOPs".* +0.67 at 5k is between NONE (+0.3) and SENS-MATTERS (+1.0).
- *Seed 43's margin is a fine-tune artefact.* Uniform s43's walk endpoint is +0.32, level with sens s43's +0.36. Its final FT restored epoch 1 and lost 0.86 at 5k, while every other arm kept a late epoch and gained 0.02–0.34. On walk endpoints the lever is +0.24 / +0.04 (two-seed +0.14, the NONE side). Had uniform s43 kept a late epoch, the call would likely read NONE; that is a counterfactual and is not quoted.
- *The like-for-like read.* Wave 21's cosine-0.1-last re-reads (22376493 / 95 / 96 / 97 and 22376498 / 99 / 22376500 / 01) keep the last epoch on every arm. They are reported beside this call; if Q7 adopts cosine-0.1, they are the paper's numbers.
- *A light cut.* Every genuine endpoint ends within 0.7 of the unpruned net at 5k, three of the four above it (§287).
- Inner s43 (22376491) and mild (22376487 / 92) are running; sens − mild and inner − uniform are reported with them. Do not lock. Never an agent row.

---

## 292. Wave 20 (c): thin κ 0.6 sens / uniform at seed 44 (**22375992 / 22375993**) — PRELIM, reported; three-seed lever sens − uniform **+0.88** at 5k (+0.54 / +1.64 / +0.46), +1.11 on val, +1.00 at 10k (0.997): the WEAK band of wave 8's bars at 5k, so wave 8's two-seed SURVIVES (§254) stands as registered, and the paper should quote the three-seed number

`tree_v10h`, §254's recipe with `SPECTRA_SEED=44` (seed verified in the env); `SPECTRA_ALLOC_KIND=sens` (α 0.5) / `uniform`. Sens **22375992** COMPLETED 2 h 34 m, 7 Oct 23:40, `cs-6000-02`; uniform **22375993** COMPLETED 3 h 8 m, 8 Oct 01:03, `ise-4090-10`; both exit 0, TB 0, no fallback, start check green (`input_c10_thin.json`, `param:0.6`; the sens plan spans 30 groups on r56-w4, uniform keeps every group at 0.75). Reader `final_ft_readout.py`, run once over all six wave 8 / wave 20 (c) cells, so seeds 42 and 43 now carry val and 10k beside the 5k of §230 / §242 / §254. Registration (wave 20 (c), 13:43): the three-seed lever is reported beside wave 8's two-seed call, which stands.

| r56-w4, κ 0.6, landed ~0.60 | Seed 42 (§230) | Seed 43 (§242 / §254) | Seed 44 (this) | Three-seed mean |
|---|---|---|---|---|
| Sens, final 5k | −2.80 | −2.08 | **−3.10** | **−2.66** |
| Uniform, final 5k | −3.34 | −3.72 | **−3.56** | **−3.54** |
| **Lever, 5k** | +0.54 | +1.64 | **+0.46** | **+0.88** |
| Lever, val | +1.14 | +1.10 | +1.10 | +1.11 |
| Lever, 10k | +0.84 | +1.37 | +0.78 | +1.00 (0.997) |
| Lever at the walk endpoint, 5k | +1.38 | +2.14 | +0.86 | +1.46 |
| Origin control, 5k (sens run / uniform run) | +0.50 / +0.36 | +0.00 / +0.52 | +0.18 / +0.58 | — |

Seed 44: sens r56-w4 walk −3.30 → final −3.10 (val −2.88, 10k −2.99) @ params 0.597 / FLOPs 0.575, step 169, residual streams 4 / 8 / 16 (full), inner medians 2 / 3 / 10, kept epoch 95; uniform walk −4.16 → final −3.56 (val −3.98, 10k −3.77) @ 0.599 / 0.582, step 96, residual 3 / 6 / 12, inner medians 3 / 6 / 13, kept epoch 100. Honest (final − walk − origin, 5k) +0.02 on both. Uniform lands on one architecture on all three seeds; sens keeps the residual streams full on all three, with stage-3 inner medians 11 / 10 / 10.

**Read.**
- *Three-seed lever: +0.88 at 5k.* On wave 8's bars (SURVIVES ≥ +1.0 / ABSORBED ≤ +0.3) that is the WEAK band, 0.12 under the line. As registered, wave 8's two-seed call (SURVIVES, +1.09, §254) stands; this section sits beside it and does not replace it. §254 already read SURVIVES as "about +1 pp, not a precise number". The paper should quote the three-seed figure: +0.88 at 5k (per seed +0.46 to +1.64, standard error 0.38) and +1.00 at 10k.
- *Where the spread sits.* Seed 43 is the high seed. On the val half the lever is flat (+1.14 / +1.10 / +1.10); the 5k half carries the spread. Across seeds the sens arm moves 1.02 at 5k (−2.08 to −3.10) and uniform 0.38. Two same-recipe origin controls of one seed differ by up to 0.52 at 5k, so a single 5k read moves by about 0.5 with nothing changed.
- *The fine-tune narrows it.* At the walk endpoint the three-seed lever is +1.46. The 100-epoch final FT gives uniform more back (+0.84 / +1.16 / +0.60 over its walk; sens +0.00 / +0.66 / +0.20), which closes about 0.6 of it.
- *Under cosine.* Wave 19's two-seed lever_cos at κ 0.6 is +1.44 with a seed spread of 0.08 (§268). This seed's cosine re-reads (22375997 / 22376009, PD) make that read three-seed; if Q7 adopts cosine-0.1, it is the paper's number.
- *Guard r20-w2.* Three-seed sens − uniform −0.35 at 5k (0.00 / −0.76 / −0.28), with sens keeping 0.014 more params and 0.06 more FLOPs. On r20-w2 both plans stop above the target with every group at its target (sens x0.635, uniform x0.606), and its origin gains +3.4 to +3.8 from the fine-tune, so it stays a guard, not a lever read.
- Do not lock. Never an agent row.

---

## 293. Wave 21, MobileNetV2 ×0.5 C10 at params 0.6: `inner` at seed 43 (**22376491**) — PRELIM, reported; the residual-full rule is **not** the lever here: two-seed inner − uniform **−0.06** at 5k (−0.72 / +0.60, and seed 43's +0.60 is uniform's epoch-1 restore), **−0.62** on walk endpoints, at FLOPs 0.51 vs 0.59; sens − inner **+0.73** at 5k (+0.82 / +0.64; 10k +0.89) with both arms on late epochs, bought with 1.4× inner's FLOPs

`tree_v10i`, `SPECTRA_ALLOC_KIND=inner`, §287's seed-42 recipe with `SPECTRA_SEED=43` (verified in the env). COMPLETED 2 h 34 m, 8 Oct 01:16, `ise-6000-08`, exit 0, TB 0, no fallback. Start check green: the inner plan keeps x0.579 over 25 groups (group keep min / median 0.67, max 1.00; "5 coupled groups held at full width"). As on seed 42, every group reaches its target at params x0.616, so the walk finishes to 0.600 with the strongest legal cut. Reader `final_ft_readout.py` over all six lr 0.01 cells of wave 21. Registration (wave 21, 14:06): inner − uniform and sens − inner are reported ("is the residual-full rule the lever here too"); no bar.

| MBV2 ×0.5, params 0.600, 5k | Seed 42 | Seed 43 | Mean | FLOPs (s42 / s43) | Kept epoch (s42 / s43) |
|---|---|---|---|---|---|
| Sens (§287 / §291) | +0.34 | +0.70 | **+0.52** | 0.714 / 0.706 | 100 / 100 |
| Uniform (§287 / §291) | +0.24 | −0.54 | −0.15 | 0.591 / 0.591 | 95 / **1** |
| Inner (§287 / this) | −0.48 | **+0.06** | **−0.21** | 0.508 / 0.508 | 100 / 100 |
| **Inner − uniform** | −0.72 | +0.60 | **−0.06** | | |
| Inner − uniform, walk endpoints | −0.66 | −0.58 | **−0.62** | | |
| **Sens − inner** | +0.82 | +0.64 | **+0.73** | | |
| Sens − inner, walk endpoints | +0.90 | +0.62 | +0.76 | | |

Val / 10k, two-seed: inner − uniform +0.01 / −0.025; sens − inner +1.05 / +0.89. Inner seed 43: walk −0.26 → final +0.06 (val −0.06, 10k +0.00) @ 0.600 / 0.508, step 124, honest −0.24, origin control +0.56. On seed 42 the three same-recipe origin controls read +0.16 / +0.44 / +0.70 at 5k, so a single 5k read carries about ±0.5.

**Read.**
- *The residual-full rule is not the lever on MBV2.* Where both arms kept a late epoch (seed 42), inner is 0.72 under uniform; on walk endpoints it is 0.66 / 0.58 under on both seeds. Seed 43's +0.60 at the final is uniform's restore of epoch 1 (−0.86 against its own walk, §291), not inner gaining. Holding MBV2's 5 narrow residual groups full and cutting the rest evenly does no better than an even cut, and keeps the fewest FLOPs (0.51). On the thin ResNets the same rule carries the whole κ 0.6 lever (inner − uniform +0.91 at lr 0.01 and +1.72 under cosine, §288). It is a ResNet property, not a rule for every family with skip connections.
- *Sens over inner: +0.73, captioned.* Both arms kept a late epoch on both seeds, so this is like-for-like, and the walk endpoints agree (+0.76). Sens keeps 1.4× inner's FLOPs (0.71 vs 0.51) at equal params: its plan keeps most groups whole (median group keep 1.00) and cuts a few hard (min 0.28 / 0.34). At equal params the three arms' accuracy follows the FLOPs they keep (sens 0.71 > uniform 0.59 > inner 0.51). Whether sens adds anything at equal FLOPs needs a FLOPs landing target (new tree, next sitting), as on VGG (queue row 70).
- *Like-for-like.* The cosine re-reads (22376493–501, keep last on every arm) are queued; if Q7 adopts cosine-0.1, they are the paper's numbers. Mild (22376487 / 92) is running; sens − mild is reported with it.
- Do not lock. Never an agent row.

---

