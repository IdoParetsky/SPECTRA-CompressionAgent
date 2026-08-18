# SPECTRA results ledger

**As of:** 18 Aug 2026 11:30 IDT. Idle GPUs filled. ImageNet resize retry **20307394** RUNNING. C100 ShuffleNet s43/s44 **20307286 / 20307395**. C100 residual SGD s43 **20307289**. FLOP-floor prefer-Δparams/ΔFLOPs s42 **20307291**. C100 recoverable DRL **20307403** queued. 24-net similar **20201260** still eval_train.  
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
| C7 | Encoder capacity (BERT / wider / set) did not fix r56-w4. Catalog diversity (10-net vs 3-net) is the remaining lever. | LOCKED | §16 encoder ~−24 pp; §17 10-net moved it to −15.9 |
| C8 | 24-net train catalog moves r56-w4 further. | TBD | s42 in-catalog eval done; held-out **20201260** still eval_train |
| C9 | Frozen C10-trained agent on held-out **CIFAR-100**: VGG-16 BN and ShuffleNet-v2×1 (s42) inside τ; thin ResNets and RepVGG-A0 miss. | LOCKED mixed | §21; ShuffleNet **20289197** −3.9 @ 0.833/0.823 |
| C10 | Eval-only FLOP floor 0.70 puts held-out r56-w4 **inside τ=10** (same frozen 10-net actor). | LOCKED | §5 s42 **−8.9 @ 0.907/0.702**; s43 **−9.2 @ 0.926/0.703**; s44 **−9.6 @ 0.907/0.703**. Not the 0.704-param operating point of C4. |

---

## 2. Headline Pareto (C10 TEST)

| Job | Agent | Net | Δacc (pp) | params / FLOPs | Status |
|---|---|---|---|---|---|
| 20066579 | 2-net C10-ResNet train, structural reward | thin r20-w2 (held out) | −3.1 | 0.60 / 0.79 | LOCKED |
| 20189046 | 10-net s42 | thin r20-w2 | −4.2 | 0.600 / 0.760 | LOCKED |
| 20189050 | 10-net s44 | thin r20-w2 | −4.6 | 0.600 / 0.760 | LOCKED |

Do **not** quote overnight-matrix **−1.2 pp** for 20066579. That was not TEST. TEST is **−3.1**.

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

Do **not** quote unmatched always-0.8 −24 as the only r56-w4 greedy. Look-ahead greedy (param floor 0.70) is **−24.9 @ 0.722/0.477** — more params than DRL 0.704, fewer FLOPs than DRL 0.550, and still the cliff. DRL’s −15.9 is not “milder because it kept more weights.”

**FLOP floor (same actors as 20189046/048/050):** r56-w4 TEST is **inside τ=10** on three seeds: s42 **−8.9 @ 0.907/0.702**, s43 **−9.2 @ 0.926/0.703**, s44 **−9.6 @ 0.907/0.703**. Cost vs param-floor DRL: ~91–93% params vs 0.704, ~70% FLOPs vs 0.550. Easy r20-w2: s42 **−2.1 @ 0.600/0.773**; s43 **−3.7 @ 0.600/0.763**; s44 **−4.0 @ 0.800/0.799** (FLOP floor bound earlier on this seed — not the 0.60-param tie). Do **not** quote eval_train r56-w4 **+7.4**. C4 stays LOCKED at 0.704/0.550.

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
| **20201235** | 24-net DRL s42 | **COMPLETED** 04:29. In-catalog DenseNet-100 −2.1 @ 0.793/0.825. **Not C8 held-out.** |
| 20202693 | 24-net DRL s43 | in-catalog **eval_test** DenseNet. Do not quote. |
| 20204215 | 24-net DRL s44 | train **stopped ep 463/720** (runtime_limit). Eval next. |
| **20201260** | 24-net similar held-out | RUNNING **eval_train** ~4h (DenseNet). C8 TEST is this job’s eval_test. |
| 20201263 / 65 | 24-net thin / unlike afterok | PENDING (Dependency) |
| **20289097** | ImageNet MobileNet-v2 3-ep s42 | **FAILED** 02:28 (no Resize). Retry **20307394** RUNNING (Resize/CenterCrop in `a7987fd`). First retry **20307283** CUDA-dead on cs-4090-09. |
| **20289103 / 20289105** | C10-thin param floor 0.80 | **COMPLETED**. §24. r56-w4 still cliffs. |
| **20289099** | C100 residual SGD 80-ep s42 | **COMPLETED** 08:51. §25. r56-w15 **−9.1 @ 0.612/0.442** inside τ; r20-w16 **−12.2** still miss. |
| **20289197** | C100 ShuffleNet dummy-forward s42 | **COMPLETED**. TEST **−3.9 @ 0.833/0.823**. Inside τ. §21. |
| **20307286 / 20307395** | C100 ShuffleNet s43 / s44 | RUNNING. Dummy-forward already in leap. |
| **20307289** | C100 residual SGD 80-ep s43 | RUNNING. s44 **20307396** PENDING QOS. |
| **20307291** | FLOP floor 0.70 + prefer Δparams/ΔFLOPs s42 | RUNNING. s43 **20307296** PENDING QOS. |
| **20307403** | CIFAR-100 DRL on VGG-11/16 + ShuffleNet (SGD recipe) | PENDING QOS. Held-out eval = thin residuals. |

**20189049 is done.** Overlay of `e985d5e` onto `/home/paretsky/SPECTRA-CompressionAgent` is now allowed. New jobs use the leap tree after `git pull` (ShuffleNet rollback is not in SPECTRA-night).

**C100 next lever (not more C10 catalog diversity):** Frozen CIFAR-10 agent + Adam-40 misses on thin residuals. SGD-80 one-seed puts r56-w15 inside τ (§25). Next: lock SGD on seeds 43/44 (**20307289 / 20307396**), then train a CIFAR-100 agent only on families that recover (VGG-11/16 + ShuffleNet, **20307403**). Do not mix unrecovered residuals into that train catalog. Do not restart encoder/BERT A/Bs.

**Floors:** They are eval-time stop rules on how far a *held-out network* may be pruned. They do not use test labels to train the agent, and they do not change the test images. They **do** limit how small that network gets (param floor 0.70 → ~30% cut; FLOP floor 0.70 → ~9% param cut on skinny ResNet-56). Prefer-Δparams/ΔFLOPs under a FLOP floor is now on (**20307291**).

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
| ShuffleNet-v2×1 | **−3.9 @ 0.833/0.823** (job **20289197**) | — | — | inside (s42; 39% mask fallback) |

ShuffleNet-v2×1 first C9 jobs crashed (`groups=116` vs input 104) — **not Slurm**. Dummy-forward after every structural prune; restore and mask. Rerun **20289197** TEST **−3.9 @ 0.833/0.823** (72.6% → 68.7%). Five rollbacks; 39% actions masked. Quote Δacc; do not claim every grouped layer was resized. C10 ShuffleNet TEST still stands.

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

Same frozen 10-net actors as C4. Floor **0.80 params**, look-ahead off, no FLOP floor. Jobs **20289103** (s42) / **20289105** (s43).

| Net | s42 TEST | s43 TEST | vs τ=10 |
|---|---|---|---|
| thin r20-w2 | **−4.3 @ 0.600/0.741** | **−2.0 @ 0.800/0.780** | inside (s42 overshot the 0.80 floor) |
| thin r56-w4 | **−21.7 @ 0.815/0.543** | **−25.7 @ 0.796/0.537** | miss |

A ~20% param cut is **not** between C4’s cliff and the FLOP-floor win. r56-w4 still falls off. Easy r20 stays inside τ. Do not quote eval_train (s42 r56 eval_train was +0.5 pp).

ImageNet **20289097 FAILED** 02:28: `build_transform` never Resize/CenterCrops ImageNet JPEGs (collate `[3,489,379]` vs `[3,333,500]`). Not TEST.

---

## 25. C100 residual SGD recipe A/B (frozen s42) — PRELIM one-seed TEST

Same frozen 10-net actor as C9 s42 (`job20158274`). Catalog `configs/input_offline_c100_residuals.json` (thin r20-w16 + r56-w15 only). Skip-train. FT **80-ep SGD + cosine + MixUp + AutoAugment** (C6 VGG recipe, shortened from 160 ep). Job **20289099** COMPLETED 18 Aug 08:51, elapsed 6h 27m. Param floor 0.70, no FLOP floor.

**Quote TEST.** Sizes are **not** matched to C9 Adam-40. One seed. Do not overwrite §21.

| Net | C9 Adam-40 s42 (§21) | SGD-80 s42 (20289099) | vs τ=10 |
|---|---|---|---|
| thin r20-w16 | −19.3 @ 0.604/0.645 | **−12.2 @ 0.687/0.648** (72.98% → 60.8%) | still miss (milder param cut than C9) |
| thin r56-w15 | −15.0 @ 0.694/0.600 | **−9.1 @ 0.612/0.442** (78.4% → 69.3%) | **inside** (smaller net than C9) |

r56-w15 is the clean recipe win: better Δacc at **fewer** params and FLOPs. r20-w16 improved vs Adam but kept more params (0.687 vs 0.604), so it is not a matched-size A/B. Do not claim “C100 residuals are solved.” Do not quote eval_train. Three-seed SGD is the next lock if Gilad wants it; a second C100 DRL is still the wrong spend.
