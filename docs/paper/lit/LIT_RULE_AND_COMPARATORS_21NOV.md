# The threshold rule's lineage, the "simple baselines" evidence, and the comparators a reviewer expects (21 Nov 2026)

Compiled 21 Nov 2026 by the spectra-literature subagent. Read-only scan; this is the only file written.

**How the sources were read.** arXiv PDFs (export.arxiv.org) were converted with `pdftotext -layout`; ar5iv / arXiv HTML tables were flattened row by row; venue fields came from the arXiv API (`comment`, `journal_ref`) or the venue page; library code was read at its release tags (NNI `v1.9`; PaddleSlim `release/1.0.0`–`1.2.0` and `develop`); documentation pages were fetched directly.

**Labels.** [V] read at the source this session (table, section or file named). [R] recorded by an earlier verified scan in the repo (file named), not re-read today. [c] my arithmetic. [H] my interpretation. [UNVERIFIED] not checked at a source.

**Conventions.** FLOPs and params are fractions *kept* (kept = 1 − printed reduction, or pruned count / base count [c]). Δ = pruned − base, each paper on its own base, recipe and test split. ↓ = reduction as printed. "T3" = Table 3. Every published number is beside our rows, never a beat.

---

## 0. Read first (decision-relevant)

1. **Li et al. 2017 did not run a threshold rule.** They measured one sensitivity curve per layer (each layer pruned alone, accuracy read on validation) and then *chose the ratios by hand*: one ratio per stage, a smaller ratio or none for sensitive layers [V §3.2, §4.1–4.2]. The single-global-threshold form comes from the tools. PaddleSlim's `get_ratios_by_loss` gives every parameter the largest ratio whose relative accuracy loss is at most one threshold, and its `SensitivePruner` documents a binary search on that threshold to hit a FLOPs target [V code]. NNI v1.9's `SensitivityPruner` uses one accuracy-drop threshold, but then rescales the ratios proportionally each round instead of searching the threshold [V code].
2. **Our rule is PaddleSlim's documented design run once on coupled channel groups.** It uses a calibration loss-rise curve made monotone, and τ is bisected to the FLOPs or params target [H]. A detail: in PaddleSlim 1.0.0–1.2.0 the shipped bisection loop never runs, because both bounds start at 0 (`min_loss = 0.`, `max_loss = 0.`). Its tutorial tells users to tune the threshold by hand, with an example value of 0.01 [V code, tutorial].
3. **The most recent paper whose core is a threshold allocation for structured CNN pruning is FAIR-Pruner** (arXiv 2508.02291, v3 20 May 2026, "Submitted to IEEE TPAMI", not accepted).
   - Rule: one shared tolerance level sets each layer's pruning depth, the largest depth whose "rank conflict" stays at or under the tolerance. It is search-free [V Fig. 2 caption, abstract].
   - Points: ResNet-56 CIFAR-10 93.93 → 93.64 at 0.462 FLOPs; ResNet-50 ImageNet 76.88 → 75.29 at 0.472 [V T I].
   - Claims: it says it "match[es] or slightly exceed[s]" HRank and ITPruner. It never mentions DepGraph or CHIP [V §IV-B; grep].
4. **The most recent peer-reviewed allocation paper of any model type is LSA (ICLR 2026).** It scores each layer by its minimal linear reconstruction error under a probe that removes 50 % of the layer's weights, then allocates non-uniform sparsity. The setting is LLMs, unstructured; it claims to beat OWL and DLP at 70 % sparsity [V ICLR page]. The probe is one point, like our weaker rule's half-width probe [H]. A later preprint (June 2026, an EMNLP submission) adds an "absorption" correction on top of OWL and AlphaPruning [V abstract].
5. **The standing critique is that per-layer sensitivities do not add up.**
   - AMC on Li et al.'s rule: it "assumes that errors of different pruned layers can be summed up linearly, which does not stand according to our experiments" [V AMC §4.2].
   - EvoPress (ICML 2025) refutes "error monotonicity" for LLM compression [V §1].
   - SPDY adds a local search on top of a dynamic program over per-layer error scores [V abstract].
   - [H] Our result, an agent level with the rule, bounds what that non-additivity is worth at about 0.45 FLOPs after a full recovery. It does not contradict them.
6. **Why no recent paper claims "a simple rule beats DepGraph":**
   - Once recipes are equalised, the old baselines already sit near the top. L1-norm pruning with a better fine-tune schedule "match[es] or beat[s] many filter pruning papers published in recent top-tier venues" on ResNet-50 [V Wang 2023 T1]. Random allocations come within 0.5 % of architecture-optimising methods on ImageNet [V Li 2022, F3]. Networks simply scaled and trained with modern recipes "clearly and utterly outperform" the literature [V ThinResNet abstract].
   - Cross-recipe comparisons of quoted numbers cannot carry such a claim [V Blalock; PruningBench pitfalls 1–3].
   - Strong methods already embed a hand allocation. GReg and TPP print per-stage ratios and skip layers "following the setting of L1 (pruned-B) Li et al. 2017" [V GReg T6, TPP T7]. [H] So the rule is a component of those methods, not a rival to them.
7. **Missing comparators a reviewer will ask for, most likely first:**
   1. AMC;
   2. a random-allocation baseline;
   3. ResRep;
   4. EagleEye, with MetaPruning, AutoSlim and DMCP;
   5. ThiNet and He et al.'s Channel Pruning;
   6. ABCPruner;
   7. global magnitude and LAMP as same-loop rules;
   8. Isomorphic Pruning and HALP at ImageNet ResNet-50.

   FAIR-Pruner is the newest threshold rule (§4).
8. **AMC's own points on our cells:**
   - ResNet-56 CIFAR-10: **92.8 → 91.9 (−0.9) at 50 % FLOPs**. The same table has uniform 89.8 and the hand "deep" policy 91.5 [V AMC T2].
   - MobileNetV2 ImageNet: **71.8 → 70.8 (−1.0) at 70 % FLOPs** [V AMC T3; base in the T3 caption].
   - ResNet-50 ImageNet: no channel-pruning point. AMC's ResNet-50 result is fine-grained (5× fewer weights, 76.13 → 76.11) [V §4.2].
9. **Draft reference gaps** [V grep of `SPECTRA_draft.md`]:
   - Blalock et al. is used in the body (line 901) but missing from the reference list.
   - RL-Pruner is in the cost table (line 945) but has no reference.
   - Uncited: ThiNet (only a 2022 derivative, [26]), He et al.'s Channel Pruning, ResRep, ABCPruner, DMCP, Isomorphic Pruning, HAP, Distiller, OWL, AlphaPruning, EvoPress, LSA, FAIR-Pruner, Wang et al. 2023, PruningBench, ThinResNet.

---

## 1. Lineage of sensitivity and threshold layer allocation

### 1.1 The origin: Li et al., *Pruning Filters for Efficient ConvNets* (ICLR 2017, arXiv:1608.08710) [V]

- **What was measured (§3.2):** "we prune each layer independently and evaluate the resulting pruned network's accuracy on the validation set". Fig. 2(b) gives the curves without retraining and Fig. 2(c) with per-layer retraining. ResNet-56/110 are in Fig. 6 and ResNet-34 in Fig. 7.
- **How ratios were chosen (§3.2):** "We empirically determine the number of filters to prune for each layer based on their sensitivity to pruning … To avoid introducing layer-wise meta-parameters, we use the same pruning ratio for all layers in the same stage. For layers that are sensitive to pruning, we prune a smaller percentage of these layers or completely skip pruning them." **The allocation was chosen by hand, per stage.**
- **The resulting plans:**
  - VGG-16 C10: "With 50% of the filters being pruned in layer 1 and from 8 to 13, we achieve 34% FLOP reduction" (§4.1, T2).
  - ResNet-56-pruned-B "skips more layers (16, 18, 20, 34, 38, 54) and prunes layers with p1=60%, p2=30% and p3=10%".
  - ResNet-110-pruned-B "skips layers 36, 38, 74 and prunes with p1=50%, p2=40% and p3=30%" (§4.2).
- **Recipe:** constant LR 0.001, 40 retraining epochs on CIFAR-10 and 20 on ImageNet (§4). T1's caption reads: "The best test/validation accuracy during the retraining process is reported." The epoch is selected on test.
- **Kin:** He et al.'s Channel Pruning (ICCV 2017) also ran a single-layer analysis (Fig. 4, "Single layer performance analysis under different speed-up ratios (without fine-tuning)") [V]. AMC then describes CP's VGG-16 ratios as "carefully tuned by human experts (conv5 are skipped, sparsity ratio for conv4 and remaining layers is 1 : 1.5)" [V AMC §4.2].

### 1.2 The tools that turned it into a rule [V]

| Tool | What it measures | How per-layer ratios are set | Threshold → budget | Source read |
|---|---|---|---|---|
| Distiller (Intel) | Each layer's accuracy on the test set after a one-shot prune at several sparsity levels; filter-wise uses mean L1 per filter | **By the user** from the curves. "Sensitivity analysis is an empirical method, and we still have to spend time to hone in on the exact multiplier value." Its *Sensitivity Pruner* is Han et al.'s element-wise threshold λ = s·σ_l | none | docs `pruning.html`, `algo_pruning.html` |
| NNI v1.9 `SensitivityPruner` | `SensitivityAnalysis` prunes one layer at a time at `np.arange(0.1, 1.0, 0.1)` and records validation accuracy | Each round, `_max_prune_ratio` takes each layer's ratio at the accuracy-drop threshold (`acc_drop_threshold`, default 0.05). `normalize` then rescales all ratios proportionally so that the round removes `sparsity_per_iter` (default 0.1) of the weights, capped per layer. The model is fine-tuned between rounds | **Proportional rescale, no threshold search**. The docs link Han et al. 2015 (arXiv:1506.02626) | docs; `src/sdk/pynni/nni/compression/torch/pruning/sensitivity_pruner.py` @v1.9 |
| PaddleSlim | `sensitivity()` prunes each conv parameter alone at 0.1 … 0.9; loss = (baseline − pruned) / baseline | `get_ratios_by_loss(sens, loss)` gives "the max ratio of each parameter" whose loss is ≤ `loss`, linearly interpolated between grid points | `SensitivePruner.get_ratios_by_sensitive`: "Search a group of ratios for pruning target flops" by **binary search on `loss`** (stop within 0.015 of target, 20 tries). In releases 1.0.0–1.2.0 both bounds start at 0, so the loop never runs. The tutorial says to tune `loss` by hand (example 0.01) | `paddleslim/prune/sensitive.py` (develop); `sensitive_pruner.py` @release/1.0.0–1.2.0; zh tutorial §8.1 |

[V code] A detail of NNI's `_max_prune_ratio`: it sets `last_ratio = ratio` before testing the threshold. So it returns the first ratio that *breaks* the threshold, although the docstring promises the last one under it [H: off by one grid step].

### 1.3 Methods whose core is a per-layer sensitivity or threshold allocation

| Method (venue, date) | Models; granularity | Allocation rule | Compared against | Claim vs DepGraph / HRank / CHIP-class |
|---|---|---|---|---|
| NetAdapt (ECCV 2018; 1804.03230) | MobileNet V1/V2 ImageNet; filters | Each iteration tries every layer: remove filters until that layer saves the iteration's measured latency step, short fine-tune, keep the best of the K proposals; long fine-tune at the end [V §3.2, Fig. 2] | MorphNet, ADC, width multipliers [V T1, T3] | none (predates them) |
| AMC (ECCV 2018; 1802.03494) | Plain-20, ResNet-56 C10; VGG-16, MobileNet V1/V2 ImageNet; channels | DDPG agent replaces the heuristic ratios; 100 + 300 episodes [V §3] | uniform / shallow / deep handcrafted; FP (= Li 2017), RNP, SPP, CP [V T2–T3] | none (predates them) |
| EigenDamage (ICML 2019; 1905.05934) | VGG-19, ResNet-32 C10/C100, Tiny-ImageNet; structures in a Kronecker eigenbasis | Hessian-based (OBS-style) saliency in the eigenbasis, pruned one pass or iteratively [V abstract] | C-OBD/OBS, Kron-OBD/OBS, Slimming [V T1] | none |
| LAMP (ICLR 2021; 2010.07611) | VGG, ResNet, EfficientNet; **weights** | Rescaled magnitude score plus **one global threshold**. "layerwise sparsities are mostly selected algorithm-by-algorithm, often resorting to handcrafted heuristics or an extensive hyperparameter search" [V abstract] | global magnitude, uniform, uniform+, Erdős–Rényi kernel [V §5] | none |
| HAP (WACV 2022, pp. 3880–3891; 2101.08940) | VGG-16, ResNet-56, PreResNet29 C10, ResNet-50 ImageNet; channels | Relative Hessian trace as sensitivity: insensitive parts pruned, moderately sensitive ones replaced by a low-rank "implant" [V abstract, §1] | HRank, GAL, SSS, VarP, L1, CP, AMC, FPGM, LFPC [V T1–T2] | **beats HRank**: "up to 2% higher accuracy with fewer parameters and FLOPs" [V §1]; no CHIP or DepGraph |
| SOSP (ICLR 2022; 2110.11395) | VGG, ResNet-32/56, DenseNet-40 C10/C100, ResNet-50 and MobileNetV2 ImageNet; feature maps | Second-order saliency over all structures at once, so the allocation falls out of one global ranking [V abstract] | first-order global methods, HRank, AMC and others [V T5–T8] | "match or beat numerous SOTA"; no CHIP or DepGraph |
| SPDY (ICML 2022; 2201.13096) | ImageNet CNNs, BERT; weights | Dynamic programming over per-layer error scores to a speed-up target, plus a local search on the score weights [V abstract] | uniform, AMC-style and others | none |
| OWL (ICML 2024; 2310.05175) | LLMs; weights | Sparsity S_i ∝ 1 − D_i (D = the layer's outlier ratio), held in [S − λ, S + λ] around the target S [V §4] | uniform Wanda / SparseGPT; beats them "by 61.22 and 6.80 perplexity at … 70%" [V abstract] | n/a |
| AlphaPruning (NeurIPS 2024; 2410.10912) | LLMs, plus ConvNeXt and ViT; weights | The heavy-tail exponent of each layer's weight spectrum sets its sparsity [V abstract, §5] | uniform, OWL; "LLaMA-7B to 80% sparsity" [V abstract] | n/a |
| DSA (NeurIPS 2024; DOI 10.52202/079017-4487) | LLaMA 1/2/3, Mistral, OPT; weights | Evolutionary search over allocation *functions* (pre-process, reduction, transform, post-process) [V poster abstract] | Wanda, SparseGPT, uniform | n/a |
| Chen & Wang (TNNLS 2024; 2408.16772; DOI 10.1109/TNNLS.2024.3365194) | ResNet-56 C10, ResNet-50 ImageNet; channels | "Information concentration" (rank fused with entropy) presets each layer's ratio; Shapley values pick channels [V abstract] | [UNVERIFIED] | [UNVERIFIED] |
| EvoPress (ICML 2025; 2410.14649) | Llama, Mistral, Phi; depth, sparsity, quantization | Evolutionary search over per-layer levels. Current methods "implicitly assum[e] that layers contribute independently" and "error monotonicity does not hold" [V abstract, §1] | sum-of-error allocations (OWL among them) | n/a |
| **FAIR-Pruner** (arXiv 2508.02291; v1 4 Aug 2025, v3 20 May 2026; TPAMI submission) | VGG-16, AlexNet, ResNet-56, DenseNet-40, ResNet-50, ConvNeXt, DeiT-B, a Qwen MoE; channels, heads, experts | **One shared tolerance level**; each layer gets "the largest feasible depth whose conflict is no greater than the tolerance level". Conflict is the overlap of a removal ranking (Wasserstein separability) with a protection ranking (Taylor sensitivity) [V Fig. 2 caption, abstract]. For literature comparisons it sweeps the level rather than bisecting it [V §IV-B]. Its theory gives conditional optimality over uniform at equal budget and shows that ranks alone "cannot unconditionally dominate uniform pruning" [V §III] | uniform within the paper; quoted Li 2017, GAL, Random, TAS, AMC, DECORE, LASSO, APS, ITPruner, MFP, HRank, AutoSlim, MetaPruning, Taylor [V T I] | **parity with HRank and ITPruner** ("matching or slightly exceeding" at ResNet-50) [V §IV-B]; no DepGraph or CHIP anywhere [V grep] |
| **LSA** (ICLR 2026 poster) | LLaMA3-8B, Qwen2.5-7B, Qwen3-8B; weights (70 % unstructured) | Layer score = minimal linear reconstruction error "assuming 50% of its least important weights are removed"; non-uniform at layer, block or projection level [V ICLR page]. The score-to-ratio mapping is [UNVERIFIED]; the repo has an OWL-like `--Lamda` flag [V README] | Uniform, OWL, DLP, each with SparseGPT / Wanda [V README] | n/a |
| *Beyond Layer Importance …* (arXiv 2606.15161, 13 Jun 2026; EMNLP submission) | LLMs | Absorption-aware correction added to OWL / AlphaPruning [V abstract] | OWL, AlphaPruning | n/a |

**Sensitivity measured inside methods (not allocation papers).**
- FPGM T4: "'SA' means 'sensitivity analysis'". On VGG-16 CIFAR-10 FPGM reaches 93.54 with Li et al.'s sensitivity allocation (34.2 % FLOPs ↓) and 93.23 without it (35.9 % ↓) [V]. [c] That is +0.31 for the hand sensitivity allocation.
- GReg (T6) and TPP (T7) print hand per-stage ratios [V]:
  - ResNet-56 C10 at 2.55×: [0, 0.75, 0.75, 0.32];
  - VGG-19 C100 at 8.84×: [0:0, 1-15:0.70];
  - ResNet-50 ImageNet at 2.31×: [0, 0.60, 0.60, 0.60, 0.21];

  with layers skipped "following the setting of L1 (pruned-B) Li et al. 2017" [V GReg T6 note].

### 1.4 Critiques of additive per-layer sensitivity [V unless marked]

- **AMC (2018), §4.2:** "FP [31] [= Li et al. 2017] proposed a sensitive analysis scheme … Layers with lower sensitivity are pruned more aggressively. Such method assumes that errors of different pruned layers can be summed up linearly, which does not stand according to our experiments." On SPP: "conducted based on one single layer, failing to take the correlations between layers into consideration."
- **SPDY (2022):** a dynamic program "given a set of layer-wise error scores", then "a local search procedure" to correct those scores.
- **EvoPress (2025):** "there are instances where compressed models with lower sums of per-layer errors can perform worse" (§1).
- **Isomorphic Pruning (ECCV 2024), §4.2:** "The conventional global pruning, employing a naive global ranking threshold, faces the potential risks of over-pruning certain layers due to the biased importance distribution."
- **PruningBench, §5.2:** global and protected-global pruning both beat local (uniform) pruning at 2×. At 4× and 8× plain global pruning causes "layer collapse"; protected global pruning (at least 10 % kept per group) wins. Its text cites sensitivity analysis as the classic fix for local pruning.

### 1.5 Where our rule sits [H]

- Same family as PaddleSlim: per-structure curves, one global threshold, the threshold searched to the target. It differs in four ways:
  - coupled channel groups, not single conv weights;
  - a calibration loss rise, not accuracy;
  - monotone curves;
  - one cut and one recovery, not iterative rounds.
- Closest modern cousin: FAIR-Pruner (one tolerance level, per-layer depth, search-free). FAIR-Pruner ranks by a conflict score, where our rule thresholds a measured loss.
- Under the AMC and EvoPress critique, the rule should be beatable by a policy that models interactions. Our learned agent, which blends several such rules, lands level with it. [H] Read together with Li 2022 and PruningBench, this supports "in this regime the additivity error is small after a full recovery".

---

## 2. "Simple baselines match sophisticated pruning"

| Paper (venue, ID) | (i) allocation vs recovery budget | (ii) value of per-network search | (iii) evaluation practice |
|---|---|---|---|
| Liu et al., *Rethinking the Value of Network Pruning* (ICLR 2019; 1810.05270) | For predefined-architecture methods, train the target from scratch instead [V abstract]. Fine-tuning ResNet-56-B for 40/80/160 epochs gives 92.68/92.67/92.76 [R LIT_ALLOCATION_PROXIES_13OCT, Liu T16] | "the pruned architecture itself, rather than a set of inherited 'important' weights, is more crucial"; "pruning can be useful as an architecture search paradigm" [V abstract]. Slimming's allocation beats uniform on VGG-16 C10, not on PreResNet-164 [R 13 Oct, Figs. 3, 5] | "the need for more careful baseline evaluations in future research on structured pruning" [V abstract] |
| Blalock et al., *What is the State of Neural Network Pruning?* (MLSys 2020; 2003.03033) | — | "pruning all layers uniformly tends to perform worse than intelligently allocating parameters to different layers … or pruning globally" [V §3.2]. Pruning beats random "at least for large amounts of pruning … this does not always hold for small amounts" [V §3.2] | 81 papers: "the community suffers from a lack of standardized benchmarks and metrics" [V abstract]. Most papers use at most three operating points [V §4]. In ShrinkBench, method rankings flip between CIFAR-VGG and ResNet-56 [V §7] |
| Le & Hua, *Network Pruning That Matters* (ICLR 2021; 2105.03193) | With the allocation fixed, random channels plus a large-LR schedule give 92.40 ± 0.16 against 92.32 on ResNet-56 [R 13 Oct, T5] | — | Retraining schedule is a confound [R] |
| Li et al., *Revisiting Random Channel Pruning* (CVPR 2022; 2205.05676) | F4: "Fine-tuning epochs has a strong influence … High-performing pruned networks usually comes with prolonged fine-tuning epochs" [V §1]. 25 → 100 epochs: −0.75 error on ResNet-50 [R 13 Oct, Fig. 4c] | F2: random per-layer ratios find "a pruned model with comparable or even superior performances". F3: "less than 0.5% on ImageNet" from architecture-optimising methods [V §1] | F1: under random ratios, recent criteria are "just comparable with the simple L1 and L2"; "surprisingly no clear winners" [V abstract, §1] |
| Wang et al., *Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning* (arXiv 2301.05219, v2 21 Feb 2023; no venue in the arXiv record) | A larger fine-tuning LR "delivers significantly better performance than 0.001" [V §1]. L1-norm pruning (2017) re-run with step LR 0.01 and a 90-epoch fine-tune: 75.24 at 2.31× and 74.77 at 2.56× on ResNet-50, level with or above GReg-1/2 (75.16/75.36), CC (75.59 at 2.12×), HRank (74.98 at 1.78×) and LFPC (74.46 at 2.55×) [V T1] | Gains often come from the recipe rather than the method [V] | "Fairness principle: The performance advantage of a pruning paper should originate from the pruning method stage, not pretraining or finetuning" [V §3.1]. Its fine-tune table spans 20 epochs at fixed LR 0.001 (L1-norm) to 180 epochs cosine (ResRep); HRank uses 30 × #layers [V] |
| ThinResNet (arXiv 2309.12854, 2023; no venue) | Simply scaled networks trained with current methods "clearly and utterly outperform all the literature we compared to" [V abstract] | Uniform scaling with a modern recipe is the bar to beat [V] | The standard benchmarks rest on outdated training practice [V] |
| Gupta et al., *Is Complexity Required …? Global Magnitude Pruning* (arXiv 2209.14624 v3, Jan 2024; venue [UNVERIFIED]) | — | Global magnitude "performs better than all SOTA techniques at all sparsity ratios" for sparsity vs accuracy; some methods win on FLOPs at low sparsity [V abstract]. Unstructured | — |
| PruningBench / *A Comprehensive Study of Structural Pruning for Vision Models* (arXiv 2406.12315, v7 21 Jan 2025; no venue) | One fixed fine-tune for all: 100 epochs on CIFAR-100, 90 on ImageNet [V App.] | Protected global > global ≈ > local at 2×; protected global best at 4–8× [V §5.2] | "no single method consistently outperforms the others"; plain magnitude L1/L2 sits in the top 5 in most rankings [V §5.1]. Pitfalls: (1) limited comparisons with SOTA, (2) inconsistent settings (pretrained models, local vs global, regularisers), (3) params and FLOPs not controlled [V §1] |

**Same-pipeline facts from the method papers** [V]:
- **GReg T1** (uniform ratio): L1 one-shot against GReg-1 is 92.97 vs 93.06 at 1.99× on ResNet-56 C10, and 66.05 vs 67.35 at 8.84× on VGG-19 C100. Regularisation matters only when the cut is deep.
- **TPP T1:** L1 one-shot 93.51 against TPP 93.46 at 1.99× on ResNet-56 C10 (LR 1e-2); TPP wins only at 11–19×.
- **ResRep T3** (ResNet-50 ImageNet, about 55 % ↓): a uniformly shrunk network trained from scratch reaches 74.39, norm-pruned and fine-tuned 74.66, ResRep variants 75.05–75.57, and ResRep itself 76.15 (T1).
- **SOSP T10:** shuffling which structures are pruned within each layer, at the same per-layer counts, shows "the dominant factor is the architecture and not the specific values of the weights".
- **EagleEye T2:** search cost for 1,000 candidates is ThiNet ≈ 8,000 GPU-h, NetAdapt 864, Li 2017-style filter pruning ≈ 20,000, EagleEye 25.
- **LeGR §4.2:** "LeGR is 7× and 5× faster than AMC and MorphNet". It searches its ranking once and re-uses it at every FLOPs count, where AMC and MorphNet search per count.
- **Gains of allocation over uniform:** AMC T2 shows +2.1 over uniform and +0.4 over the best hand policy at 50 % on ResNet-56. MetaPruning T1 shows MobileNetV1 0.25× at 57.2 vs 50.6. DepGraph T2 shows learned over uniform sparsity on ResNet-56 C100 of +0.89 / +1.68 / +1.61 / +0.69 at 1.5× / 3× / 6× / 12× [R LIT_SEALED_WEB_20NOV §1a].

### 2.1 Why no recent paper claims a simple rule beats DepGraph and its peers [H, built on the [V] facts above]

1. **The baselines are already near the top once recipes match.** Wang 2023, Le & Hua and Li 2022 each show a 2017-era baseline that, with a modern fine-tune, matches the top-venue rows. A paper that says "a rule beats DepGraph" would inherit this objection: the win would be credited to the recipe, not the rule.
2. **The incentive is to propose criteria or search, not allocation rules.** The rule is old (Li 2017, PaddleSlim, NNI) and already used inside SOTA pipelines. GReg and TPP, for instance, print hand ratios following Li's pruned-B. A rule paper has no novelty claim unless it adds theory (FAIR-Pruner) or another domain (OWL, LSA for LLMs).
3. **The rule papers that exist compare against uniform, not against DepGraph.** NetAdapt, LAMP, OWL, AlphaPruning, LSA and FAIR-Pruner all show gains over uniform or over other allocations in their own pipeline, and quote older per-network methods (HRank, ITPruner) beside them. None claims DepGraph- or CHIP-level superiority [V greps].
4. **DepGraph's advantage is not only allocation.** Its pipeline is grouping plus 30 sparse-learning epochs plus 90 of fine-tuning [R draft §5.1c]. Its own T2 credits learned layer sparsity with +0.7 to +1.7 over uniform on ResNet-56 C100 [R]. A rule versus DepGraph mixes three variables. Only a same-loop comparison isolates allocation, which is what our tables do.
5. **What remains of per-network search.**
   - Search is worth about a point or more against uniform at tight budgets or on compact nets (AMC, MetaPruning, DepGraph T2).
   - Against a good rule after a full recovery it is worth tenths: +0.31 (EagleEye vs L1-uniform on MobileNetV1 at 26.5 M FLOPs), +0.4 (AMC vs hand "deep") [R 13 Oct], and less than 0.5 % for random search on ImageNet [V Li 2022].
   - Its costs run from tens to thousands of GPU-hours per network (EagleEye T2).
   - That is the gap a frozen, transferable policy can fill: parity at near-zero per-network cost.

---

## 3. Canonical comparator coverage

**Cells:**
- CIFAR-10: ResNet-56, ResNet-110, VGG-16(-BN);
- CIFAR-100: VGG-19, ResNet-56;
- Tiny-ImageNet: VGG-16-BN, ResNet-18;
- ImageNet: ResNet-50;
- a possible cell: MobileNetV2 on CIFAR-10 or ImageNet.

"In tables" means the method already appears in the draft's Table 1, 1b or 1c sub-tables. Draft reference numbers are given where cited. Each point is each paper's own pipeline and base. Where a column is absent, a dash is used.

### 3.1 The table

| Method (venue; arXiv) | In draft? | Cells with a point | Verified point(s): base → pruned (Δ) @ FLOPs kept / params kept; recovery; source |
|---|---|---|---|
| L1-norm / Li et al. (ICLR 2017; 1608.08710) | [18]; in Table 1b via TPP's run | R56, R110, V16 C10 | R56-B 93.04 → 93.06 (+0.02) @ 0.724 / 0.863; R110-B 93.53 → 93.30 (−0.23) @ 0.614 / 0.676; V16-A 93.25 → 93.40 (+0.15) @ 0.658 / 0.360 [V T1, from errors]; 40 epochs at LR 0.001, best test/val epoch |
| Network Slimming (ICCV 2017; 1708.06519) | [106]; not in tables | V19 C100, V19 C10 (VGGNet with 16 conv layers, 20.04 M params) | C100 73.26 → 73.48 (+0.22) @ 0.629 / 0.249; C10 93.66 → 93.80 @ 0.490 / 0.115 [V T1]; sparsity training then a 160-epoch fine-tune [V §4.3] |
| ThiNet (ICCV 2017; 1707.06342) | **no** (only a 2022 derivative, [26]) | R50-IN | ThiNet-50 72.88 → 71.01 (−1.87) @ 0.442 / 0.484; ThiNet-70 72.04 (−0.84) @ 0.632 / 0.663 [V T3]. Its base is 72.88, not 76.15: reprints restate it (MetaPruning T3: ThiNet-50 74.7 at 2.1 G; ResRep T1: 75.30 → 72.03) [V]; fine-tune [UNVERIFIED] |
| Channel Pruning, He et al. (ICCV 2017; 1707.06168) | **no** | R56 C10; R50-IN (top-5 only) | R56 92.8 → 91.8 (−1.0) @ 0.50 [V T8]; R50 2×: +1.4 top-5 error on a 92.2 base [V T6] |
| SFP (IJCAI 2018; 1808.06866) | [162]; in Table 1b via TAS/LFPC | R56, R110 C10; R50-IN | R56 (pretrained) 93.59 → 93.35 (−0.24) @ 0.474; R110 93.68 → 93.86 (+0.18) @ 0.592 [V T1]; R50 76.15 → 74.61 (−1.54) @ 0.582 [V T2] |
| FPGM (CVPR 2019; 1811.00250) | [145]; in tables | R56, R110, V16 C10; R50-IN | R56 (pretrained) 93.59 → 93.49 (−0.10) @ 0.474; R110 93.68 → 93.74 (+0.06; printed −0.16) @ 0.477 [V T1]; R50 76.15 → 75.59 (−0.56) @ 0.578 and 74.83 (−1.32) @ 0.465 [V T2] |
| **AMC** (ECCV 2018; 1802.03494) | [79]; **not in tables** | R56 C10; MBV2-IN | R56 92.8 → 91.9 (−0.9) @ 0.50 [V T2]; MBV2 71.8 → 70.8 (−1.0) @ 0.70 [V T3]; fine-tune length [UNVERIFIED]. Full list in §3.2 |
| NetAdapt (ECCV 2018; 1804.03230) | [116]; no | MBV2-IN (latency only) | MobileNetV2 70.9 at the latency of the 0.75× model (69.8) [V T3]; MobileNetV1 69.1 at 284.3 MMACs [V T1] |
| MetaPruning (ICCV 2019; 1903.10258) | [80]; no | R50-IN, MBV2-IN | R50, trained from scratch, base 76.6: 75.4 at 2.0 G (≈0.488 [c]), 73.4 at 1.0 G [V T3]; MBV2 71.2 at 217 M [V T4] |
| AutoSlim (arXiv tech report; 1903.11728) | [112]; no | R50-IN, MBV2 C10/IN | R50 76.1 → 75.6 at 2.0 G (≈0.488) [V T1, from errors 23.9 → 24.4]; MBV2 C10 93.0 at 59 M vs the 1.0× model's 91.9 at 88 M (0.67 [c]) [V T2]; slimmable training from scratch [V DMCP T6 note] |
| LeGR (CVPR 2020; 1904.12368) | [120]; in Table 1c | R56 C10; R50-IN; MBV2-IN; R56 C100 as a figure | R56 93.9 → 93.7 ± 0.2 @ 0.47 [V T1]; R50 76.1 → 75.3 (−0.8) @ 0.47 and 75.7 @ 0.58 [V App. C T2]; MBV2 71.8 → 71.4 @ 0.70 [V]; 400 fine-tune epochs on C10, 60 on IN [V §4.1] |
| ABCPruner (IJCAI 2020; 2001.08565) | **no** | R56, R110, V16 C10; R50-IN | R56 93.26 → 93.23 (−0.03) @ 0.459 / 0.458; R110 93.50 → 93.58 @ 0.350 / 0.326; V16 93.02 → 93.08 @ 0.263 / 0.113 [V T1]; R50 76.01 → 73.86 (−2.15) @ 0.457 / 0.460 [V T2]; 12 search + 90 fine-tune epochs [V T3] |
| EagleEye (ECCV 2020; 2007.02491) | [117]; no | R56 C10; R50-IN | R56 93.26 → 94.66 (+1.40) @ 0.496 [V T3; the README labels this row "ResNet-50"]; R50 76.4 at 2.0 G (≈0.488), 77.1 at 3.0 G, 74.2 at 1.0 G [V T4; base not printed in T4]; pruned R50 checkpoints released [V README] |
| DMCP (CVPR 2020; 2005.03354) | **no** | R50-IN, MBV2-IN | R50 76.6 → 76.2 (−0.4) at 2.2 G (≈0.537); MBV2 72.3 → 72.2 (−0.1) at 211 M (0.70) [V T6]; pruned nets trained 100 epochs (ResNet) and 250 (MBV2) [V §4] |
| GNN-RL (ICML 2022; 2102.03214) | [125]; in Table 1c | R56, R110 C10; R50-IN | R56 93.49 (+0.10) @ 0.46; R110 94.31 (+0.63) @ 0.48 [V T1]; R50 74.28 (−1.82) @ 0.47 [V T2]; 100 fine-tune epochs on C10 [V §4]. Its AMC rows are a hazard (§3.4) |
| HRank (CVPR 2020; 2002.10179) | [164]; in tables | R56, R110, V16 C10; R50-IN | R56 93.26 → 93.17 (−0.09) @ 0.500 / 0.576; R110 93.50 → 93.36 @ 0.418 / 0.408 [V T3]; V16 93.96 → 93.43 @ 0.465 / 0.171 [V T1]; R50 76.15 → 74.98 @ 0.562 / 0.633 [V T5] |
| CHIP (NeurIPS 2021; 2110.13981) | [165]; in tables | R56, R110, V16 C10; R50-IN | R56 93.26 → 94.16 (+0.90) @ 0.526 / 0.572; R110 93.50 → 94.44 (+0.94) @ 0.479 / 0.517; V16 93.96 → 93.86 (−0.10) @ 0.419 / 0.184 [V T1]; R50 76.15 → 76.15 (0.00) @ 0.513 / 0.558 and 76.30 @ 0.552 [V T2] |
| ResRep (ICCV 2021; 2007.03260) | **no** | R56, R110 C10; R50-IN | R56 93.71 → 93.71 ± 0.02 @ 0.471; R110 94.64 → 94.62 ± 0.04 @ 0.418 [V T2]; R50 76.15 → 76.15 ± 0.01 @ 0.455 and 75.97 @ 0.439 [V T1]; 180-epoch ImageNet fine-tune [V §4.3] |
| GReg (ICLR 2021; 2012.09243) | [158]; in Table 1b | R56 C10; V19 C100; R50-IN | R56 93.36 → 93.36 (GReg-2) @ 0.392 [V T2]; V19 74.02 → 67.75 @ 0.113 (printed drop 6.47; the accuracies give 6.27) [V]; R50 76.13 → 75.36 (GReg-2) @ 0.433 [V T3]; hand per-stage ratios [V T6] |
| DepGraph (CVPR 2023; 2301.12900) | [12]; in tables | R56 C10; V19 C100; R50-IN; MBV2-IN | R56 93.53 → 93.77 (+0.24) at 2.11× and 93.64 (+0.11) at 2.57×; V19 73.50 → 70.39 at 8.92× [V T1]; R50 76.15 → 75.83 (−0.32) @ 0.482; MBV2 71.87 → 68.46 @ 0.455 [V T3]; no params column |
| OCSPruner (WACV 2026; 2501.13439) | [93]; in tables | R56, R110, V16 C10; V19 C100; R50-IN | R56 (pretrained) 94.01 → 93.50 @ 0.388 / 0.423 [R 20 Nov §1d]; R50 76.29 → 75.95 [V T3] @ 0.517 [R draft]; training cycle 300 epochs on CIFAR, 130 on ImageNet [V T10] |
| SOSP (ICLR 2022; 2110.11395) | [160]; in Table 1b | R56 C10; VGG C100; MBV2-IN | R56 SOSP-H 93.27 @ 0.432 / 0.388 (printed base 93.88; its gaps imply 93.78) [V T12]; VGG-Net C100 73.45 → 72.94 @ 0.484 [V T11]; MBV2 71.04 (gap 0.84) @ 0.71 [V T8] |
| TPP (ICLR 2023; 2207.12534) | [159]; in Table 1b | V19 C100; R50-IN; R56 C10 (study) | R50 76.13 → 75.60 (−0.53) @ 0.433 [V T2]; R56 93.78 → 93.46 at 1.99× against L1's 93.51 [V T1]; V19 −6.24 @ 0.113 [R draft Table 1b] |
| Isomorphic Pruning (ECCV 2024; 2407.04616) | **no** | R50-IN, MBV2-IN | R50 76.13 → 75.91 (−0.22) @ 0.499 / 0.589; MBV2 71.88 → 68.91 @ 0.469 [V T1]; no CIFAR rows |
| HALP (NeurIPS 2022; 2210.06659) | [114]; no | R50-IN, MBV2-IN | R50 76.2 → 76.5 (+0.3) at 2.0 G (≈0.488), 74.3 at 1.0 G [V T1]; MBV2 72.2 at 249 M, 70.4 at 183 M [V T1]; 90-epoch fine-tune [V App.] |
| HAP (WACV 2022; 2101.08940) | **no** | R56, V16 C10 | R56 (HAP + implant) 93.88 → 93.55 (−0.33) @ 0.407 [V T2]; V16 93.88 → 93.66 @ 0.297 / 0.101 [V T1] |
| FAIR-Pruner (arXiv 2508.02291 v3) | **no** | R56 C10; R50-IN | R56 93.93 → 93.64 (−0.29) @ 0.462; R50 76.88 → 75.29 (−1.59) @ 0.472 [V T I]; 300-epoch fine-tune on C10 [V App. A] |
| Others already in tables | LFPC [161], TAS [66], DECORE [167], GAL [166], EigenDamage [157], Jiang et al. [163] | — | [R] the draft's §5.1 to §5.1c |
| Other 2024–26 methods, cited but not in tables | ATO [82], SPA [81], HESSO [83] | ATO: R56, MBV2 C10 | ATO R56 93.50 → 93.74 @ 0.450 and MBV2 94.45 → 94.78 @ 0.542 [R 20 Nov §2.6, §2.14] |

### 3.2 AMC in detail

**AMC's own points** [V, arXiv 1802.03494 v4]:

| Network, data | Budget | Result | Same table | Where |
|---|---|---|---|---|
| ResNet-56 C10 (base 92.8) | 50 % FLOPs, channels | **91.9** after fine-tune; 90.2 val / 90.1 test before | uniform 89.8, hand "deep" 91.5 (before: 87.5 and 88.4) | T2 |
| Plain-20 C10 (base 90.5) | 50 % FLOPs | 90.2 after fine-tune (86.4 val before) | uniform 89.7, shallow 89.2, deep 88.3 | T2 |
| ResNet-50 C10 (base 93.53) | 60 % params, fine-grained, accuracy-guaranteed reward | 93.64 val / 93.55 test, no fine-tune | — | T2 |
| VGG-16 ImageNet | 20 % FLOPs (5×) | **−1.4**, against CP −1.7, SPP −2.3, RNP −3.58, FP (Li) −14.6 | — | T3 |
| MobileNet V1 ImageNet (base 70.6) | 50 % / 40 % FLOPs | −0.4 / −1.7 in T3; T4: 285 MMACs (0.50 [c]) gives **70.5 top-1** (−0.1), and 272 M (50 % latency) gives 70.2 | uniform 0.75-224: −2.5 at 56 %; 0.75-192: −3.7 at 41 % | T3, T4 |
| MobileNet V2 ImageNet (base 71.8) | 70 % FLOPs | **−1.0** (→ 70.8) | uniform 0.75-224: −2.0 | T3 |
| ResNet-50 ImageNet (base 76.13) | 5× fewer weights, fine-grained, 4 stages of 30-epoch fine-tunes | 76.11 top-1 | hand ("human expert") 3.4× | §4.2, Fig. 4 |

Notes on the table:
- The VGG-16 numbers are on CP's scale: CP T1 reports "Increase of top-5 error (1-view, baseline 89.9%)" with 1.7 for CP at 5× and 14.6 for its own re-implementation of Li et al. AMC's caption, however, quotes the top-1 base 70.5 [V both].
- AMC's MobileNet change at 50 % FLOPs is −0.4 in T3 but −0.1 in T4 [V]: an internal inconsistency.
- Search: "the RL can finish searching within 1 hour on a single GeForce GTX TITAN Xp GPU" (CIFAR), with 100 exploration and 300 exploitation episodes [V §3, §4.1].

**Re-reported by others** [V, each in the paper named]:
- DepGraph T1: ResNet-56 92.80 → 91.90 (−0.90) at 2.00×.
- LeGR T1: ResNet-56 92.8 → 91.9 at 62.7 MFLOPs (50 %). The row has no ∗ (∗ marks LeGR's own implementation), so it quotes AMC.
- LeGR App. C T2: MobileNetV2 71.8 → 70.8 at 70 %.
- ResRep T2 and HAP T2: ResNet-56 92.8 → 91.9 at 50.
- MetaPruning T4: MobileNetV1 285 M 70.5; MobileNetV2 220 M 70.8.
- DMCP T6: MobileNetV2 211 M 70.8 (−1.0).
- AutoSlim T1: "AMC-MobileNet v2" 211 M, 29.2 error; "AMC-MobileNet v1" 285 M, 29.5 error.
- HALP T1: MobileNetV1 285 M 70.5.
- EagleEye §4: adaptive-BN evaluation inside AMC lifts MobileNetV1 "(reported in AMC) to 70.7%".
- SOSP T7 compares SOSP with AMC on Plain-20 C10.
- FAIR-Pruner T I: 91.90 with "Δ −2.03", computed against FAIR's own 93.93 base, not AMC's 92.8.

**Re-run by others.**
- LeGR re-implemented AMC on CIFAR-100 ResNet-56 and MobileNetV2 and on Bird-200, with τ̂ = 200 fine-tune steps before each accuracy reading ("we do the same for AMC … for a fair comparison") and 400 explored architectures. The results appear as curves only: Fig. 3a (mean ± SD of three trials) and Fig. 5. "AMC and MorphNet have higher variances when pruned more aggressively" [V §4.2].
- LeGR's repository ships `amc.py` at top level [V repo listing]. The README does not document its arguments [V].
- **Our own run** used LeGR's `amc.py` on ResNet-56 CIFAR-10: prune_away 53 % FLOPs, 200 search steps, a 60-epoch fine-tune, final test top-1 92.78 from a base of about 93.36.
  - Timing in ledger §419: search 1:25:26, fine-tune 0:23:39, total 1:49:05 [R ledger §419 heading and table lines].
  - The accuracy figure is from the brief [R]. It is ours, a re-run, **never a published AMC number**.

### 3.3 Coverage by cell (published points that fall in our tables' FLOPs ranges)

| Cell | Already in our tables | Published, verified above, not yet in our tables |
|---|---|---|
| ResNet-56 C10 (0.39, 0.48) | DepGraph, OCSPruner, the draft's other rows [R] | AMC 0.50; ResRep 0.471; LeGR 0.47; ABCPruner 0.459; EagleEye 0.496; GNN-RL 0.46; HAP 0.407; FAIR-Pruner 0.462; SFP / FPGM 0.474; CP 0.50; HRank 0.500; CHIP 0.526 |
| ResNet-110 C10 (0.48, 0.40) | FPGM, CHIP, OCSPruner, LFPC, HRank, DECORE | ResRep 0.418; GNN-RL 0.48; SFP 0.592; ABCPruner 0.350 |
| VGG-16 C10 (0.465, 0.34) | HRank, CHIP | ABCPruner 0.263; HAP 0.297; Li 2017 0.658; FPGM with sensitivity analysis 0.658 |
| VGG-19 C100 (0.112, 0.50) | DepGraph, OCSPruner, TPP, GReg, EigenDamage, SOSP, L1 | Slimming 0.629 (between our points) |
| ResNet-56 C100 (0.48) | Jiang et al., LFPC, TAS, FPGM / SFP | LeGR (figure only); SOSP T5 rows (base not extracted) |
| Tiny-ImageNet, VGG-16-BN and ResNet-18 at 64 px | Jiang et al. (another VGG-16) | none found among the 37 texts read today. EigenDamage T2 is VGG-19 on Tiny-ImageNet, a different network [V caption] [H: coverage claim limited to these texts] |
| ResNet-50 IN (0.48, 0.56) | CHIP, OCSPruner, DepGraph, LeGR, FPGM, GNN-RL, HRank | ResRep 0.455; Isomorphic 0.499; EagleEye, MetaPruning, AutoSlim and HALP at about 0.488; DMCP about 0.537; GReg-2 / TPP 0.433; ABCPruner 0.457; ThiNet-50 0.442 (on a 72.88 base); FAIR-Pruner 0.472 |
| MobileNetV2 IN (possible) | — | AMC, LeGR, DMCP 0.70; MetaPruning 217 M; HALP 249 / 183 M; SOSP 0.71; DepGraph 0.455; Isomorphic 0.469 |
| MobileNetV2 C10 (possible) | — | AutoSlim 59 M; DCP, DMC, SCOP, NPPM, ATO [R 20 Nov §2.14] |

### 3.4 Reprint and reading hazards found today [V]

- **GNN-RL T1, AMC on ResNet-56:** "AMC 90.20 (−3.19)" is AMC's *pre-fine-tune* validation accuracy (AMC T2: 90.2 val before fine-tune), with Δ taken from GNN-RL's own 93.39 base [c]. Its "ResNet-20" AMC row (86.40) and Uniform row (84.00) are AMC's *Plain-20* pre-fine-tune values (86.4, 84.0).
- **GNN-RL T2, AMC on VGG-16 ImageNet:** "AMC 69.10 (−1.40)" turns a top-5-error change into a top-1 accuracy (70.5 − 1.4).
- **FAIR-Pruner T I:** every competitor's Δ is recomputed against FAIR's own base (93.93 on ResNet-56; 76.88 on ResNet-50), so its quoted drops for others are not their published drops.
- **AMC:** MobileNet −0.4 (T3) against −0.1 (T4).
- **SOSP:** printed ResNet-56 base 93.88, gaps implying 93.78 (T12).
- **EagleEye:** its README labels the CIFAR-10 94.66 row "ResNet-50"; paper T3 says ResNet-56.
- **Isomorphic Pruning T1** gives DepGraph's ResNet-50 20.60 M params, a number DepGraph's own T3 does not print. (Both papers share the first author.)
- **ThiNet's ResNet-50 base is 72.88.** Reprints rebase it, and they differ from one another (MetaPruning 74.7 at 2.1 G; ResRep 75.30 → 72.03).
- **Test-set epoch selection:** Li 2017's T1 reports "the best test/validation accuracy during the retraining process".

---

## 4. Gaps: canonical methods missing from our tables, ranked by how likely a reviewer asks for each [H ranking; facts as tagged above]

1. **AMC.** It is the canonical DRL pruner and the closest neighbour of a DRL thesis. Its point is on our ResNet-56 C10 cell (92.8 → 91.9 at 0.50, within 10 % of DepGraph's 2.11× FLOPs at 0.474).
   - *Cheapest:* quote it in Table 1's 2.11× sub-table and caption the 92.8 base (AMC T2).
   - Add our re-run of LeGR's `amc.py` as a separate row labelled "our run of LeGR's implementation, 60-epoch fine-tune, not published" (ledger §419).
   - If the MobileNetV2 ImageNet cell is added, quote 71.8 → 70.8 at 0.70.
2. **A random-allocation baseline** (Li 2022; Blalock). This is a baseline rather than a method, but pruning reviewers now expect it.
   - *Cheapest:* run it in our loop: N random keep vectors bisected to the budget, picked on validation by adaptive BN, then one recovery. The cuts are cheap; only the picked plan recovers.
3. **ResRep.** It appears in most ResNet-56 and ResNet-50 tables, including DepGraph's T1.
   - *Cheapest:* quote R56 0.471 (93.71 → 93.71), R110 0.418 and R50 0.455 (76.15 → 76.15), and add the reference.
4. **The allocation-search family: EagleEye, MetaPruning, AutoSlim, DMCP.** These are SPECTRA's direct per-network competitors.
   - *Cheapest:* quote the ResNet-50 points near 0.49 and EagleEye's ResNet-56 C10 point (0.496).
   - Next cheapest: transplant EagleEye's released pruned ResNet-50 widths (checkpoints in its README) into our loop. That gives a same-recipe allocation comparison at one recovery [H: widths are read from the checkpoint shapes].
   - Add references for DMCP; the others are already cited.
5. **ThiNet and He et al.'s Channel Pruning.** These are the classic channel-selection baselines; both are uncited.
   - *Cheapest:* quote CP's ResNet-56 0.50 point (92.8 → 91.8) and ThiNet's ResNet-50 points with their 72.88 base, and add both references.
6. **ABCPruner.** It is an allocation search with points on four of our cells.
   - *Cheapest:* quote R56 0.459, R110 0.350, V16 0.263 and R50 0.457.
7. **Global magnitude and LAMP as same-loop rules.** These are the field's default allocations: Torch-Pruning's global pruning, PruningBench's protected global and LAMP. PruningBench finds that global allocation beats local.
   - *Cheapest:* run them in our loop. The code exists behind a default-off flag, under the draft's decision L.
8. **Isomorphic Pruning (ECCV 2024) and HALP.** These are recent ResNet-50 and MobileNetV2 ImageNet anchors; Isomorphic Pruning comes from DepGraph's first author.
   - *Cheapest:* quote Isomorphic Pruning at 0.499 (76.13 → 75.91) and HALP at 0.488 (76.2 → 76.5); add the Isomorphic reference.
9. **FAIR-Pruner.** It is the newest threshold-style allocation, and a reviewer who knows it will ask how our rule differs.
   - *Cheapest:* quote R56 0.462 and R50 0.472 (both on their own bases).
   - Stronger: run its released pip package's tolerance allocation as one more same-loop rule [H: possible because the code is released, per its abstract and §IV].
10. **HAP and NetAdapt.** HAP is second-order sensitivity allocation and is uncited; NetAdapt is cited.
    - *Cheapest:* quote HAP's ResNet-56 point (0.407) and VGG-16 point (0.297). Quote NetAdapt only if a MobileNet ImageNet cell is added.
11. **Literature allocations transplanted at zero search cost.** Li 2017 pruned-B, GReg's T6 ratios and TPP's T7 ratios are printed per stage; AMC's appear only in figures. They are a clean test of "hand rules from the literature" in our loop.
    - Li 2017: R56 at 60/30/10 % with layers 16/18/20/34/38/54 skipped; R110 at 50/40/30 % with 36/38/74 skipped.
    - GReg and TPP: R56 at 2.55× [0, 0.75, 0.75, 0.32]; R50 at 2.31× [0, 0.60, 0.60, 0.60, 0.21]; VGG-19 at 8.84× [0:0, 1-15:0.70].

**Reference fixes for the draft:**
- Add Blalock et al. (used in the body) and RL-Pruner (in the cost table).
- Add ThiNet, CP, ResRep, ABCPruner, DMCP and Isomorphic Pruning if they are quoted.
- Add Distiller beside NNI [171] and PaddleSlim [172], since those tools are the origin of the threshold form.
- Add Wang et al. 2023, PruningBench and ThinResNet if §2.2.6's "how much allocation matters" paragraph cites the recipe-confound evidence.

---

## References (all opened this session unless marked [R])

**Rule lineage and tools**
- Li, Kadav, Durdanovic, Samet, Graf. *Pruning Filters for Efficient ConvNets.* ICLR 2017. https://arxiv.org/abs/1608.08710
- Distiller documentation (Intel AI Lab): sensitivity analysis, https://intellabs.github.io/distiller/pruning.html; pruning algorithms, https://intellabs.github.io/distiller/algo_pruning.html
- Microsoft NNI v1.9. *Pruner* docs, https://nni.readthedocs.io/en/v1.9/Compression/Pruner.html; *Sensitivity analysis*, https://nni.readthedocs.io/en/v1.9/Compression/CompressionUtils.html; source, https://github.com/microsoft/nni/blob/v1.9/src/sdk/pynni/nni/compression/torch/pruning/sensitivity_pruner.py
- PaddleSlim. `sensitive.py`, https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py; `sensitive_pruner.py`, https://github.com/PaddlePaddle/PaddleSlim/blob/release/1.2.0/paddleslim/prune/sensitive_pruner.py; tutorial, https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/pruning/static/image_classification_sensitivity_analysis_tutorial.md; API, https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/prune_api.rst
- Han, Pool, Tran, Dally. *Learning both Weights and Connections.* NeurIPS 2015. arXiv:1506.02626 (as linked by the NNI docs; the abstract page itself was not opened)
- Yang et al. *NetAdapt.* ECCV 2018. https://arxiv.org/abs/1804.03230
- He, Lin, Liu, Wang, Li, Han. *AMC.* ECCV 2018. https://arxiv.org/abs/1802.03494
- He, Zhang, Sun. *Channel Pruning for Accelerating Very Deep Neural Networks.* ICCV 2017. https://arxiv.org/abs/1707.06168
- Wang, Grosse, Fidler, Zhang. *EigenDamage.* ICML 2019. https://arxiv.org/abs/1905.05934
- Lee, Park, Mo, Ahn, Shin. *Layer-adaptive Sparsity for the Magnitude-based Pruning (LAMP).* ICLR 2021. https://arxiv.org/abs/2010.07611
- Yu, Yao, Gholami, Dong, Kim, Mahoney, Keutzer. *Hessian-Aware Pruning and Optimal Neural Implant.* WACV 2022, pp. 3880–3891. https://arxiv.org/abs/2101.08940; https://openaccess.thecvf.com/content/WACV2022/html/Yu_Hessian-Aware_Pruning_and_Optimal_Neural_Implant_WACV_2022_paper.html
- Nonnenmacher, Pfeil, Steinwart, Reeb. *SOSP.* ICLR 2022. https://arxiv.org/abs/2110.11395
- Frantar, Alistarh. *SPDY.* ICML 2022. https://arxiv.org/abs/2201.13096
- Yin et al. *Outlier Weighed Layerwise Sparsity (OWL).* ICML 2024. https://arxiv.org/abs/2310.05175
- Lu, Zhou, Liu, Wang, Mahoney, Yang. *AlphaPruning.* NeurIPS 2024. https://arxiv.org/abs/2410.10912
- Li, Dong et al. *Discovering Sparsity Allocation for Layer-wise Pruning of LLMs (DSA).* NeurIPS 2024. https://neurips.cc/virtual/2024/poster/93422 (DOI 10.52202/079017-4487, as in draft [132])
- Chen, Wang. *An Effective Information Theoretic Framework for Channel Pruning.* TNNLS 2024 (DOI 10.1109/TNNLS.2024.3365194, per the arXiv page). https://arxiv.org/abs/2408.16772
- Sieberling, Kuznedelev, Kurtic, Alistarh. *EvoPress.* ICML 2025. https://arxiv.org/abs/2410.14649
- Lin, Hussien, Yu, Jing, Ming, Nguyen, Cheriet. *FAIR-Pruner.* arXiv:2508.02291 v3 (20 May 2026), submitted to IEEE TPAMI. https://arxiv.org/abs/2508.02291
- Yang, Deng, Chen, Zhou, Cheng. *LSA: Layer-wise Sparsity Allocation for LLM Pruning Based on Minimal Linear Reconstruction Error.* ICLR 2026. https://iclr.cc/virtual/2026/poster/10006617; code, https://github.com/BeiYazi0/LSA
- Jing et al. *Beyond Layer Importance in Layer-wise Sparsity.* arXiv:2606.15161 (13 Jun 2026; EMNLP submission). https://arxiv.org/abs/2606.15161
- Wang, Kindratenko. *RL-Pruner.* arXiv:2411.06463 (10 Nov 2024). https://arxiv.org/abs/2411.06463

**Simple baselines and evaluation**
- Liu, Sun, Zhou, Huang, Darrell. *Rethinking the Value of Network Pruning.* ICLR 2019. https://arxiv.org/abs/1810.05270
- Blalock, Gonzalez Ortiz, Frankle, Guttag. *What is the State of Neural Network Pruning?* MLSys 2020. https://arxiv.org/abs/2003.03033
- Le, Hua. *Network Pruning That Matters: A Case Study on Retraining Variants.* ICLR 2021. https://arxiv.org/abs/2105.03193 [R]
- Li, Adamczewski, Li, Gu, Timofte, Van Gool. *Revisiting Random Channel Pruning for Neural Network Compression.* CVPR 2022. https://arxiv.org/abs/2205.05676
- Wang, Qin, Bai, Fu. *Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning.* arXiv:2301.05219 (v2, 21 Feb 2023). https://arxiv.org/abs/2301.05219
- Tessier, Boukli Hacene, Gripon. *ThinResNet: A New Baseline for Structured Convolutional Networks Pruning.* arXiv:2309.12854 (2023). https://arxiv.org/abs/2309.12854
- Gupta et al. *Is Complexity Required for Neural Network Pruning? A Case Study on Global Magnitude Pruning.* arXiv:2209.14624 v3 (2024). https://arxiv.org/abs/2209.14624
- Li, Li, Xue, Fang, Zhou, Feng, Wang, Song, Song. *A Comprehensive Study of Structural Pruning for Vision Models* (PruningBench). arXiv:2406.12315 v7 (21 Jan 2025). https://arxiv.org/abs/2406.12315

**Comparators**
- Liu et al. *Network Slimming.* ICCV 2017. https://arxiv.org/abs/1708.06519
- Luo, Wu, Lin. *ThiNet.* ICCV 2017. https://arxiv.org/abs/1707.06342
- He, Kang, Dong, Fu, Yang. *Soft Filter Pruning.* IJCAI 2018. https://arxiv.org/abs/1808.06866
- He, Liu, Wang, Hu, Yang. *FPGM.* CVPR 2019. https://arxiv.org/abs/1811.00250
- Liu et al. *MetaPruning.* ICCV 2019. https://arxiv.org/abs/1903.10258
- Yu, Huang. *AutoSlim* (tech report). https://arxiv.org/abs/1903.11728
- Chin, Ding, Zhang, Marculescu. *LeGR.* CVPR 2020. https://arxiv.org/abs/1904.12368; code, https://github.com/cmu-enyac/LeGR
- Lin et al. *Channel Pruning via Automatic Structure Search (ABCPruner).* IJCAI 2020. https://arxiv.org/abs/2001.08565
- Li, Wu, Su, Wang. *EagleEye.* ECCV 2020. https://arxiv.org/abs/2007.02491; code, https://github.com/anonymous47823493/EagleEye
- Guo et al. *DMCP: Differentiable Markov Channel Pruning.* CVPR 2020. https://arxiv.org/abs/2005.03354
- Yu, Mazaheri, Jannesari. *Topology-Aware Network Pruning (GNN-RL).* ICML 2022. https://arxiv.org/abs/2102.03214
- Lin et al. *HRank.* CVPR 2020. https://arxiv.org/abs/2002.10179
- Sui et al. *CHIP.* NeurIPS 2021. https://arxiv.org/abs/2110.13981
- Ding et al. *ResRep.* ICCV 2021. https://arxiv.org/abs/2007.03260
- Wang, Qin, Zhang, Fu. *Neural Pruning via Growing Regularization (GReg).* ICLR 2021. https://arxiv.org/abs/2012.09243
- Wang, Fu. *Trainability Preserving Neural Pruning (TPP).* ICLR 2023. https://arxiv.org/abs/2207.12534
- Fang, Ma, Song, Mi, Wang. *DepGraph.* CVPR 2023. https://arxiv.org/abs/2301.12900
- Ghimire, Kil, Jeong, Park, Kim. *One-Cycle Structured Pruning via Stability-Driven Subnetwork Search (OCSPruner).* WACV 2026 [R draft [93]]. https://arxiv.org/abs/2501.13439
- Fang, Ma, Mi, Wang. *Isomorphic Pruning for Vision Models.* ECCV 2024 (DOI 10.1007/978-3-031-73404-5_14). https://arxiv.org/abs/2407.04616
- Shen et al. *Structural Pruning via Latency-Saliency Knapsack (HALP).* NeurIPS 2022. https://arxiv.org/abs/2210.06659

Repo sources [R]: `docs/paper/LIT_ALLOCATION_PROXIES_13OCT.md`, `docs/paper/LIT_EVAL_PRACTICES_20OCT.md`, `docs/paper/lit/LIT_SEALED_WEB_20NOV.md`, `docs/paper/SPECTRA_draft.md` §2, §4.1–4.1b, §5.1–5.1c and the reference list, and the heading and table lines of ledger §419.
