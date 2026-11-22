# Agent versus rule: mixed granularity, rule portfolios, look-ahead, deep cuts and training-free rules (22 Nov 2026)

Compiled 22 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**How sources were read.**
- arXiv abstract pages were fetched with curl and parsed for title, authors, date, comments, journal reference and abstract. The arXiv API returned HTTP 503 all session, so every arXiv record here comes from the abstract page.
- Full texts: PDFs from arXiv, CVF open access, IEEE open access and the A*STAR repository were converted with `pdftotext -layout` and read as text. Two-column tables often come out with shifted columns. A value is quoted only where its row could be aligned (by row counts, by the prose, or by a second mention), and the problem is noted where it limited reading.
- One NeurIPS proceedings abstract page; two IEEE Access records through the Semantic Scholar API; 17 web searches for leads (§9).
- Facts already verified by earlier scans in this folder are cited, not re-read: [R: file §].

**Labels.** Every reference in §10 carries one of these.
- "Read at the source": [V full] = full text read this session (partial reads name the sections); [V abs] = abstract page or API record only.
- "Reported": [R] = seen second-hand (a search snippet, another paper's reprint, or an earlier repo scan, named).
- [c] my arithmetic. [H] my interpretation. [U] unresolved.
- (brief) = project facts from the brief, not re-checked. [P §N] = project evidence by ledger section as recorded in `LIT_ALTERNATIVES_21NOV.md` §1 (TEST 5k under the project's protocol, with val and 10k where that file gives them).
- "Not found" means not found by this session's searches (§9), not proof of absence. Bracketed numbers point to §10.

**Conventions.** FLOPs kept = fraction remaining (kept = 1/speed-up, or 1 − printed reduction [c]). Δ = pruned − base, each paper on its own base and recipe. No outside number here is a SPECTRA TEST result, and none is a "beat".

**Relation to earlier scans [R].** Not repeated here:
- proxy validity and surrogates (`LIT_DB_AVENUES_21NOV.md` DB2, DB5);
- the rule's lineage, AMC and the comparator list (`LIT_RULE_AND_COMPARATORS_21NOV.md`);
- the recovery-procedure and criterion-selection directions (`LIT_ALTERNATIVES_21NOV.md` C2, C3);
- DepGraph's and OCSPruner's verified table rows (`LIT_SEALED_WEB_20NOV.md` §1).

This note adds the mixed-granularity, rule-portfolio, look-ahead and deep-cut literature, and checks it against the agent-versus-rule question.

---

## 1. Summary

Every ingredient of "an agent that chooses per layer among granularities and rules, with a brief simulation" exists in the literature, but only fitted, searched or trained per target network, and the one same-pipeline test of a learned per-layer selector against a hand rule found no gain. Mixed granularity in one optimiser is established: channels with blocks [4], filters against layers [3], mixed N:M [9], and pruning with quantisation [17, 18, 25]. Its reported gains over a single granularity are large mainly at high ratios and on latency or bit-operations: +1.4 top-1 at 28 % more speed than a channel-only latency pruner on ResNet-50 ImageNet [4], and +2.1 for layer-wise over uniform N:M at equal size [9]. The only frozen per-layer technique selector found was trained and tested on LeNets [26]. Per-layer choice among criteria gains +0.05 to +0.32 over the best single criterion in the same pipeline [30, 32]. An RL mapper of per-layer pruning schemes, after 3–9 days of search, ended level with a training-free rule ("negligible" difference) [13]. Cheap evaluation is the weak link:
- cut-time scores rank outcomes at Kendall 0.01–0.19 [46];
- BN recalibration restores a Pearson 0.68–0.82 on one network [45];
- short fine-tune proxies are assumed rather than measured;
- no model-based or world-model pruner was found.

At about 9× on VGG-19 CIFAR-100, nothing in this scan beats DepGraph by more than a few tenths, and what reaches it changes the training, not the allocation: sparse learning (+2.8 to +3.2) [53, 57], pruning inside one training run [54], smooth decay (+0.18) [57]. Allocation-only methods with fixed ratios stay 2.6 points or more below [c]. Training-free rules with second-order, reconstruction, cost-normalised and isomorphic signals exist; reconstruction is worth points without recovery [68] and tenths after a long one. The literature's explanation for "agent = rule" matches ours. After a long recovery, layer ratios matter while criteria and inherited weights do not, and learned allocators beat uniform by points but tuned rules only by tenths. Real headroom has been shown only:
- with short or no recovery;
- at deep cuts on compact networks;
- under latency or hardware targets;
- in mixed spaces.

Ranked directions (§8):
1. a recovery-aware threshold rule;
2. a mixed-granularity space, tested first with a mixed-unit rule, then with the DRL agent and a short-recovery pick;
3. the rule inside sparse learning at deep cuts (P7 is testing the first step);
4. multi-fidelity selection among a portfolio of plans.

**At a glance.**

| Question | Short answer | Closest work | Left open for us [H] |
|---|---|---|---|
| Q1 mixed granularity | Done per network. Gains over one granularity are largest at high ratios and on latency or BOPs. No frozen cross-family policy | MDP [4]; LayerPrune [3]; DominoSearch [9]; Gong et al. [13]; APQ [17] | A frozen cross-family policy over channel groups, blocks (and N:M), judged at matched FLOPs under one long recovery against a mixed-unit rule |
| Q2 rule portfolios | Done per network. +0.05 to +0.32 over the best single criterion in the same pipeline; an RL scheme selector tied a training-free rule | Gong et al. [13]; Blending [30]; LAASP [32]; LFPC [29] | Only the frozen cross-family selector; per-network selection does not repay its cost under long recovery |
| Q3 simulation | Cut-only proxies fail (Kendall ≤ 0.19); BN recalibration repairs them on one network; short-fine-tune proxies assumed, not measured; no world-model pruner found | EagleEye [45]; Oracle pruning [46]; RL-MCTS [42] | A proxy ladder and successive-halving selection for width plans across families |
| Q4 deep cuts | The training procedure, not the allocation: sparse learning, pruning inside training from scratch, smooth decay, regrowth | DepGraph [53]; OCSPruner [54]; DPM [57]; CHEX [39]; OTOv2 [20] | The rule inside sparse learning (P7 tests the first step) |
| Q5 training-free rules | Many exist. Reconstruction and repair matter by points without recovery, little after a long one | OSSCAR [68]; Isomorphic [72]; Relative Repairability [74]; LADP [70] | A structured, recovery-aware threshold rule validated leave-one-family-out; a mixed-unit threshold rule |
| Q6 collapse to rules | Under long recovery, ratios matter and criteria do not. Learned allocators beat uniform by points and tuned rules by tenths | Li 2022 [48]; Gong et al. [13]; random tickets [83]; PruningBench [59] | The regimes with shown headroom: no or short recovery, deep cuts on compact nets, latency targets, mixed spaces |

---

## 2. Q1. Multi-granularity and hybrid pruning in one decision-maker

**Verdict: established, per network. A frozen, cross-family version was not found.**
- One optimiser or policy deciding several granularities together exists for:
  - channels with blocks: MDP [4];
  - filters against layers or depth: LayerPrune [3], DBP [8], TAS [2], SSL [1];
  - activation removal and layer merging: DepthShrinker [5], [6], LayerMerge [7];
  - mixed N:M per layer: DominoSearch [9];
  - channels with weights: hybrid pruning [10];
  - per-layer weight-pattern schemes: Gong et al. [13], PatDNN [12], SWP [11];
  - channels with low rank: CC [15];
  - pruning with quantisation: APQ [17], DJPQ [18], Bayesian Bits [19], GETA [22], FITCompress [24], Balaskas et al. [25].
- Each is fitted, searched or trained per target. The one "generalist" per-layer technique selector was trained on two LeNets and tested on a third [26].

| Work | Venue, ID | Units decided together | Reported gain over a single granularity (matched on what) | Scope | Label |
|---|---|---|---|---|---|
| SSL [1] | NIPS 2016; arXiv:1608.03665 | filters, channels, filter shapes, depth (group-lasso regulariser) | Depth regularisation cuts ResNet-20 on CIFAR-10 to 18 layers while accuracy rises 91.25 → 92.60. AlexNet conv speed-ups 5.1× (CPU) and 3.1× (GPU), "about twice" those of non-structured sparsity | per network, in training | [V abs] |
| TAS [2] | NeurIPS 2019; arXiv:1905.09717 | channel and layer counts as learned distributions; knowledge distillation to the pruned net | no number in the abstract | per network | [V abs] |
| LayerPrune [3] | arXiv:2007.05667 | layers against filters, same criterion | VGG-19-BN on CIFAR-100 (base 73.11), 1080Ti, batch 8, Table 1. At about equal latency reduction, 2 layers removed give 73.60 at 17.34 % against Slimming's 72.32 at 16.84 % (+1.28 [c]); 73.60 at 17.12 % against Taylor's 72.61 at 15.87 % (+0.99 [c]). 5 layers removed: 74.80 at 39.84 %. Weight-norm filter pruning *raised* latency (−2.04 %). Random layer-pruned models beat random filter-pruned ones by 7.09 % on average at matched latency (trained from scratch, 164 epochs) | per network, one-shot layer ranking | [V full] |
| MDP [4] | CVPR 2025; arXiv:2504.02168 (v1 arXiv:2406.12079) | channels and whole blocks (CNNs); heads, query/key, value, embeddings (ViTs); one MINLP with a latency lookup table | ResNet-50 ImageNet, TITAN V, batch 256, two runs: 70.0 at ×5.16 against HALP 68.6 at ×4.02; 72.8 at ×4.13 against SMCP 72.7 at ×3.71; 74.8 at ×3.06 against LayerPrune 74.3 at ×1.79. Ablation: multi-granularity beats channel-only "particularly at high pruning ratios" (Fig. 4, figure only) | per network (solver), then fine-tune | [V full] |
| DepthShrinker [5] | ICML 2022; arXiv:2206.00843 | removes activations and merges blocks of compact nets | +3.06 % accuracy and 1.53× throughput on a V100 over MetaPruning | per network | [V abs] |
| Depth compression by DP [6]; LayerMerge [7] | ICML 2023, arXiv:2301.12187; ICML 2024, arXiv:2406.12837 | activation removal and conv merging; joint layer pruning and merging | MobileNetV2-1.0 ImageNet: 1.41× speed-up with +0.11 points [6]. LayerMerge "consistently outperforms existing depth compression and layer pruning methods" [7] | per network (dynamic programming) | [V abs] |
| DBP [8] | arXiv:1912.10178 | blocks (Conv-BN-ReLU) by feature discrimination | argues for layers over filters because layers cannot run in parallel; no number read | per network | [V abs] |
| DominoSearch [9] | NeurIPS 2021 (no arXiv record found) | layer-wise mixed N:M | ResNet-18 ImageNet: +2.1 top-1 over uniform N:M at equal size (2.1 M params, 87.5 % sparsity); +1.3 at equal 227 M FLOPs. ResNet-50: 76.7 at 5.0 M params | per network | [V abs] |
| Hybrid pruning [10] | arXiv:1811.00482 | coarse channels plus fine-grained weights, with a fast sensitivity test | "significantly better" on ResNet-50 ImageNet; no number in the abstract | per network | [V abs] |
| SWP [11]; PatDNN [12] | NeurIPS 2020, arXiv:2009.14410; ASPLOS 2020, arXiv:2001.00138 | filter "stripes" (shape within a filter); kernel patterns plus connectivity | SWP: state-of-the-art pruning ratio "without obvious accuracy drop" (no matched figure) | per network | [V abs] |
| **Gong et al. [13]** | ACM TODAES (doi:10.1145/3495532 printed in the PDF); arXiv:2111.11581 | per-layer pruning regularity (pattern, block-punched, hybrid) and block size | Against PatDNN's single pattern scheme (Table 4, parameter rate of the conv layers), rule / RL against PatDNN: ResNet-50 C10 11.51× / 11.88× against 1.57×; VGG-16 C10 12.38× / 12.50× against 8.0×; ResNet-50 ImageNet 4.37× / 4.41× against 1.56×. Rule and RL stay within 0.3 points of the base accuracy; PatDNN gains 1.0 on ResNet-50 C10 at its 1.57× | per network. The rule is training-free for "each layer of any given DNN" | [V full] |
| AutoCompress [14] | arXiv:1907.03141 | a combination of structured schemes with ADMM; heuristic search "replacing the prior deep reinforcement learning technique" | "up to 33× in pruning rate (120× reduction in the actual parameter count)" over prior automatic compression at the same accuracy | per network | [V abs] |
| Collaborative compression [15] | CVPR 2021; arXiv:2105.11228 | channel pruning with tensor decomposition | ResNet-50 ImageNet: 52.9 % FLOPs ↓ at −0.56 top-1; no single-technique figure in the abstract | per network | [V abs] |
| HAQ [16] | CVPR 2019; arXiv:1811.08886 | per-layer bit-width (quantisation only) | latency down by 1.4–1.95× and energy by 1.9× against fixed 8-bit, at "negligible loss" | per network; RL with a hardware simulator | [V abs] |
| APQ [17] | CVPR 2020; arXiv:2006.08509 | architecture, pruning and quantisation jointly | +2.3 % ImageNet accuracy over the separate ProxylessNAS + AMC + HAQ; 2× / 1.3× lower latency / energy than MobileNetV2 + HAQ at the same accuracy | per target; predictor-guided search in a once-for-all supernet | [V abs] |
| DJPQ [18] | ECCV 2020; arXiv:2007.10463 | structured pruning and mixed precision in one differentiable loss | 53× fewer BOPs on ResNet-18 and 43× on MobileNetV2 at floating-point accuracy; "outperforms" the two-stage approach in accuracy and BOPs | per network | [V abs] |
| Bayesian Bits [19]; FITCompress [24] | arXiv:2005.07093; arXiv:2302.07612 | pruning as a 0-bit option with mixed precision; unstructured pruning with mixed precision by Fisher path planning | "better trade-off" than static bit widths [19]; "superior" trade-off [24]; no numbers in the abstracts | per network | [V abs] |
| OTOv2 [20]; OTOv3 [21]; GETA [22]; HESSO [23] | ICLR 2023, arXiv:2303.06862; arXiv:2312.09411; arXiv:2502.16638; arXiv:2409.09085 | groups found automatically for any architecture. OTOv3 adds operator "erasing"; GETA adds quantisation-aware training | architecture-agnostic code. OTOv2's ResNet-50 ImageNet ladder is in §5 | per network, one training from scratch | [V full] OTOv2; [V abs] others |
| Balaskas et al. [25] | IEEE TETC (doi:10.1109/TETC.2023.3346944); arXiv:2312.15322 | per-layer choice among fine- and coarse-grained pruning algorithms, plus per-layer bits | 39 % average energy reduction at 1.7 % average accuracy loss, with no retraining. Motivation (MobileNetV2 CIFAR-10, 1 % loss bound): fine-grained "Level" reaches 50 % sparsity, coarse "L1-Ranked" only 10 %. No ablation against the best single technique was found; compared with NSGA-II at equal evaluations | per network, 1,100 episodes | [V full: §3, §5] |
| Generalist agent [26] | IEEE Access 2024; doi:10.1109/ACCESS.2024.3385857 | per-layer choice among ten AdaDeep compressors (SVD, depthwise replacement, sparse random connections, and others) | trained on LeNet for Fashion-MNIST and KMNIST, applied to LeNet on MNIST: under 15 % of the size at under 2.5 % loss. The authors enumerated all 119 action sequences for the three compressed layers | **frozen across datasets, one architecture** | [V full: abstract, action space, results] |
| AMC [28] | ECCV 2018; arXiv:1802.03494 | channels, and fine-grained weights in separate experiments | not one joint decision | per network | [R: LIT_RULE_AND_COMPARATORS §3.2] |

**Reported gains over a single granularity, by what they are matched on.**
- *Matched latency:* depth beats width at high ratios. MDP gains +1.4 top-1 at 28 % more speed than channel-only HALP [4]. LayerPrune gains +1.0 to +1.3 points at about equal latency reduction on VGG-19-BN CIFAR-100, and 7.09 % on average for random models [3]. Filter pruning can even raise latency at small batch sizes [3].
- *Matched size or FLOPs:* mixed N:M beats uniform N:M by +2.1 (size) and +1.3 (FLOPs) on ResNet-18 ImageNet [9].
- *Joint against sequential:* +2.3 for APQ over the separate pipeline [17]; DJPQ wins on both axes, no number in the abstract [18].
- *Not found:* channel against block (or layer) pruning at matched FLOPs on CIFAR-scale families under a long recovery. MDP's and LayerPrune's wins are latency wins, and MDP's granularity ablation is a figure.

**Frozen, generic policy?** No.
- The generalist agent [26] is the only frozen per-layer technique selector found. It transfers across datasets inside one LeNet architecture and decides three layers; its companion paper reuses the agent for per-layer SVD ranks on unseen models [27].
- OTOv2/v3, GETA, HESSO and MDP are generic as code but train or solve per network.
- Gong et al.'s rule-based mapper is a training-free *rule* applied to "any given DNN" [13]; it is generic as a rule, not learned.

**Left open [H].**
- A policy trained across families and applied frozen, whose per-layer action includes the unit (channel-group width, block removal, N:M), judged at matched FLOPs under one long recovery against a mixed-unit rule, not a channel-only rule.
- The FLOPs-matched question itself: do blocks or N:M add anything over channel groups after 100 epochs? The literature answers it for latency (yes, at high ratios) but not for FLOPs.

---

## 3. Q2. Fusion of pruning criteria and rule portfolios

**Verdict: done per network. A frozen cross-family selector was not found. Gains over the best single rule are small.**

| Work | Venue, ID | What is chosen, and how | Against the best single criterion or rule in the same pipeline | Cost | Label |
|---|---|---|---|---|---|
| LFPC [29] | CVPR 2020 (CVF) | criterion per layer (ℓ1, ℓ2, geometric median; adversarial variants as a check), by a differentiable Gumbel-softmax sampler on validation loss | ResNet-56 CIFAR-100 (Table 4): LFPC 70.83 (drop 0.58) at 51.6 % FLOPs ↓, against FPGM 69.66 (1.75) and SFP 68.79 (2.61) at 52.6 %. All three rows imply the same base, 71.41 [c], so +1.17 over FPGM [c]. Whether those rows are the authors' re-runs is not stated in the text read [U] | per network | [V full] |
| Blending criteria [30] | ICANN 2021 [venue R]; arXiv:2107.05033 | per-layer weighted blend of criteria (11 in the setup text, 12 rows in Table 1), clustered by rank similarity, by evolutionary search | CIFAR-100, mean of three runs (Table 1). VGG-16 (base 72.22): 71.68 against 71.45 for the best single criterion, a BN-scale variant (+0.23). ResNet-56 (base 69.50): 69.82 against FPGM's 69.50 (+0.32). The pruning ratio was not found in the text read | population 20 × 50 iterations, each fitness after a 3-epoch fine-tune (CIFAR-100) | [V full] |
| Composite saliency, "myopic oracle" [31] | IEEE SSCI 2020; arXiv:2004.03376 | at each step, every criterion proposes a channel; the one whose removal costs least on measured loss is pruned | No retraining at all; LeNet-5, a CIFAR-10 net, ResNet-20, NIN, AlexNet. "In most cases our method finds better selections than even the best individual pruning saliency" (abstract). Table III (weights removed at a 5-point drop) is too scrambled to quote cells | k cut-only evaluations per step | [V full] |
| LAASP [32] | Image Vis. Comput. 136 (2023) 104745; arXiv:2506.20152 | criterion (ℓ1, ℓ2, Euclidean, cosine) and layer at each iteration, by loss on a training subset; pruning while training | ResNet-32 CIFAR-10 at 53 % FLOPs ↓ (Table 4): all four 92.64 ± 0.09, similarity pair 92.59 ± 0.15, magnitude pair 92.35 ± 0.14. That is +0.05 over the best pair, within one SD | one training | [V full] |
| MFP [34] | IEEE TNNLS (per arXiv comment); arXiv:1904.03961 | adaptively selects norm- or distance-based criteria during iterative pruning via a "meta-attribute" | ResNet-50 ImageNet: over 50 % FLOPs ↓ at 0.44 % top-5 loss; no single-criterion comparison in the abstract | per network | [V abs] |
| EvoFC [33] | Neurocomputing 2024 [venue R]; author manuscript read | criterion and ratio per layer, by genetic algorithm, from random initialisation | compared with other methods (SSNIP, SRAND, ThiNet, GAL), not with its own best single criterion, in the text read | population search | [V full: abstract, §4] |
| Per-layer ratio and algorithm search [40] | Electronics 13(4):688 (MDPI) | ratio and pruning algorithm per layer | not read (HTTP 403) | — | [R: search snippet] |
| CPMC [35] | IJCNN 2021; arXiv:2011.03240 | one fused score: filter weights, next-layer dependent weights, FLOPs and params, globally normalised | "outperforms" (abstract truncated) | per network | [V abs] |
| CWDA [36] | NeurIPS 2021; arXiv:2004.11627 | analysis, not a method | Common criteria rank filters "almost identical[ly]", giving "similar pruned structures" | — | [V abs] |
| Evolved transferable criteria [37] | GECCO 2022; arXiv:2110.10876 | a closed-form channel criterion evolved by genetic programming on small datasets | transfers to harder datasets; a criterion, not a per-layer selector | GP search once | [V abs] |
| AutoMC [38] | ICDE 2024 [venue R]; arXiv:2201.09884 | a sequence of compression methods and settings, from a knowledge graph plus progressive search | "satisfying compression schemes within short time"; no rule baseline in the abstract | per target | [V abs] |
| **Gong et al. [13]** | see §2 | RL (per network) against a training-free rule, over per-layer pruning schemes | "the search-based method only shows a slightly better performance than the rule-based method"; for MobileNetV2 CIFAR-10 "the performance difference ... is negligible" (7.53× against 7.54×). RL search took 3 days (CIFAR-10) and 9 days (ImageNet) on 5 GPU servers. The RL reward was the accuracy after a one-shot magnitude cut and 2 retraining epochs | — | [V full] |
| CHEX criterion study [39] | CVPR 2022; arXiv:2203.15794 | ResNet-50 ImageNet at 1 GFLOPs (4×), Table 3 | Pretrain-prune-finetune: CSS 74.6, magnitude 73.8, BN 73.1 (+0.8, +1.5). Inside CHEX's prune-and-regrow: 76.0 / 75.7 / 75.3 (+0.3, +0.7) | — | [V full] |
| Li et al. 2022 [48] | CVPR 2022 | random per-layer ratios; criterion compared | under random ratios, recent criteria are "just comparable" with ℓ1 and ℓ2 | — | [R: LIT_RULE_AND_COMPARATORS §2] |
| Pruner-Zero, OptiShear [41] | ICML 2024; arXiv:2502.10735 | evolved LLM pruning metrics | LLM-only | — | [R: LIT_ALTERNATIVES refs. 48–49] |
| In-house [P §398] | — | random channels against ℓ1 channels at the agent's widths | +0.13 and +0.23, with a run-to-run SD of 0.40–0.45 per cell difference | — | [R: LIT_ALTERNATIVES §1.1] |

**Is "an agent that chooses among rules and granularities per layer" already done?** Yes, per network, in three forms:
- RL choosing a per-layer pruning scheme and block size [13];
- RL choosing a per-layer pruning algorithm and bit-width [25] (and HiReLC, for pruning with quantisation [R: LIT_DB_AVENUES ref. 38]);
- a genetic or evolutionary search choosing a per-layer criterion and ratio [30, 33].

**How it fared against the best single rule.**
- Gong et al. compare RL with a well-designed rule head to head. RL gains 0.01× to 0.37× in compression rate [c] at equal or near-equal accuracy, after days of search: 11.88× against 11.51× on ResNet-50 CIFAR-10, 4.41× against 4.37× on ResNet-50 ImageNet [13].
- Portfolios of criteria gain +0.05 [32], +0.23 and +0.32 [30], and +1.17 in LFPC's own table [29].
- The criterion matters more in a pretrain-prune-finetune pipeline at a deep ImageNet cut (+0.8 to +1.5), and the gap shrinks when training can regrow channels (+0.3 to +0.7) [39].
- [H] Gong et al. is the closest published analogue of SPECTRA's finding: a learned per-layer selector, compared fairly with a well-designed rule, gains almost nothing.

**Algorithm selection or hyper-heuristics.** Not found under those names. The nearest are EvoFC [33] and the per-layer ratio-and-algorithm search [40], both population searches per network.

**Left open [H].** Only the frozen cross-family selector. Under a long recovery the evidence predicts a null for criteria (CWDA [36]; [48]; [P §398]). Granularity is the axis with published headroom (§2).

---

## 4. Q3. Simulation and look-ahead before committing

**Verdict.**
- Cut-only signals rank post-recovery outcomes badly. BN recalibration repairs them on the one network measured.
- Short fine-tune proxies are used everywhere but their fidelity is almost never measured.
- "Propose several plans, simulate briefly, pick one" exists in many forms, per network, and gains tenths over a good rule.
- No model-based RL or world model for compression was found.

### 4.1 Measured fidelity of cheap evaluations

| Work | Proxy → target | Units ranked | Measured agreement | Label |
|---|---|---|---|---|
| Is Oracle Pruning the True Oracle? [46] | loss before retraining → after retraining (120 epochs on CIFAR, 30 on ImageNet) | masks at fixed layer ratios | Kendall τ 0.02 (ResNet-56 C10), 0.01 (VGG-19 C100), 0.19 (ResNet-18 ImageNet), against its own bar of 0.2. "Only 10 % retraining" correlates highly with full retraining (figure only) | [R: LIT_DB_AVENUES §2] |
| EagleEye [45] | cut-only and adaptive-BN accuracy → fine-tuned accuracy | width plans of MobileNetV1 on ImageNet (random ratios, ℓ1 filters), four FLOPs settings | Pearson −0.04 to 0.25 (cut-only) against 0.68–0.82 (adaptive BN; 1/30 of the training set, 100 iterations) | [R: LIT_DB_AVENUES §2] |
| Blending [30] | cut-only and 2-epoch accuracy → best fine-tuned accuracy | blended-criterion candidates (VGG-16 CIFAR-100, ResNet-34 ImageNet) | cut-only "does not have a strong linear relationship"; after 2 epochs R² "improve[s] significantly" (Fig. 3; values only in the figure) | [V full] |
| AMC [28] | cut-only → fine-tuned | seven CIFAR-10 policies; ImageNet | rank order kept on CIFAR-10 (Table 2); on ImageNet, cut-only accuracy "cannot provide useful supervision" | [R: LIT_DB_AVENUES §2] |
| NetAdapt [47] | short-term fine-tune (10k–40k iterations) | per-iteration proposals | needed: without it accuracy "rapidly drops to nearly zero"; no correlation reported | [R: LIT_DB_AVENUES §2] |
| Gong et al. [13] | one-shot magnitude cut + 2 epochs → final accuracy | per-layer scheme mappings | assumed ("can be used to predict the final"), not measured | [V full] |
| Zero-cost NAS proxies [50] | zero-cost scores, or reduced training → trained accuracy | architectures trained from scratch | best zero-cost proxy Spearman 0.82 against 0.61 for EcoNAS reduced training (NAS-Bench-201) | [R: LIT_DB_AVENUES §2] |
| TraceNAS [51] | zero-shot gradient-trace proxy → accuracy after recovery | depth- and width-pruned LLMs | Spearman 0.94, Kendall 0.82 (search snippet); the abstract gives no values | [V abs; values R] |
| In-house (brief) | cut-only accuracy → 100-epoch outcome; seed against seed | width plans | τ 0.15 against the outcome; two seeds' plan rankings agree at τ 0.54 | (brief) |

### 4.2 "Propose several plans, evaluate briefly, keep one"

| Method | Candidates | Brief evaluation | Reported gain, and against what | Label |
|---|---|---|---|---|
| NetAdapt [47] | K per-layer proposals per iteration | short-term fine-tune | a latency-driven loop; against width multipliers | [R: LIT_RULE_AND_COMPARATORS §1.3] |
| Random channel pruning [48] | 100 random ratio vectors | top 5 fine-tuned for 5 epochs | "comparable or even superior"; within 0.5 % of architecture-search methods on ImageNet | [R: LIT_RULE_AND_COMPARATORS §2] |
| EagleEye [45] | random ratio vectors with ℓ1 filters | adaptive BN | +0.31 over ℓ1-uniform at MobileNetV1 26.5 M FLOPs | [R: LIT_RULE_AND_COMPARATORS §2.1, from the 13 Oct scan] |
| Blending [30] | evolutionary blends of criteria | 3-epoch fine-tune | +0.23 and +0.32 over the best single criterion | [V full] |
| Composite saliency [31] | one channel per criterion, per step | cut-only measured loss | better than the best single saliency "in most cases" (no retraining) | [V full] |
| RL-MCTS [42] | filter sequences, MCTS look-ahead (gains level off beyond 10 simulations) | accuracy drop after the cut; 30-epoch fine-tune per segment | ResNet-50 ImageNet, own base 77.34: −0.54 at about 46 % FLOPs ↓, against −4.05 for random search with the same number of executions. ResNet-56 C10 93.20 → 93.56 at 55 % ↓ | [V full] |
| Gong et al. RL [13] | per-layer scheme mappings | 2-epoch retraining | level with a training-free rule | [V full] |
| BOCR [44] | Bayesian optimisation over per-layer ratios, with clustering and rollback | — | faster convergence than plain BO; no rule baseline in the abstract | [V abs] |
| In-house [P §443] | seven candidate plans per network | one-epoch check | TEST 5k +0.25 [+0.07, +0.43] at FLOPs 0.45 (val +0.15, 10k +0.20); −1.95 (val −1.94) at 0.20, through one network whose picked plan never trained | [R: LIT_ALTERNATIVES §1.1] |

### 4.3 Look-ahead criteria, model-based RL, surrogates, zero-cost proxies
- **Look-ahead criteria.** LAP extends magnitude pruning from one layer to the distortion over neighbouring layers. It "consistently outperforms magnitude-based pruning ... particularly in the high-sparsity regime" (unstructured) [43, V abs]. It is a criterion, not a simulation.
- **Model-based RL or world models for compression:** not found. The search returned work on compressing RL models, not RL that models the pruning process (§9).
- **Learned surrogates in the loop.**
  - APQ transfers an fp32 accuracy predictor to a quantisation-aware one [17, V abs].
  - SeerNet, HiReLC and Fine-Pruning fit per-target predictors [52].
  - In-house, a gradient-boosting surrogate fitted to 8,340 recovered outcomes of 104 networks failed to rank, predict or select on held-out families (brief; commit 307665f, "NO-SURROGATE").
- **Zero-cost proxies applied to CNN width plans:** not found. TraceNAS [51] does it for LLM depth and width.

**Left open [H].**
- (i) The proxy ladder (cut-only, BN, 1, 5, 20 epochs) for width plans across families against 100-epoch TEST, with selection regret beside τ. This is DB2's open item [R: LIT_DB_AVENUES §2].
- (ii) Successive halving or other multi-fidelity selection among pruning plans: not found.
- (iii) A frozen agent that proposes plans and picks among them by a short recovery: not found.

---

## 5. Q4. What beats DepGraph and OCSPruner at deep cuts (2023–2026)?

**Verdict.** At about 9× on VGG-19 CIFAR-100, nothing in this scan beats DepGraph's full method by more than a few tenths. What reaches it changes the training procedure, not the allocation. On ImageNet ResNet-50 at a quarter to a third of the FLOPs, the best numbers come from long from-scratch schedules with regrowth, and from learned soft masks.

### 5.1 VGG-19 on CIFAR-100, about 11 % of FLOPs kept (8.8–9.1×)

| Method (venue) | Base → pruned | Δ | FLOPs kept | Allocation | Procedure | Label |
|---|---|---|---|---|---|---|
| ℓ1 one-shot, hand ratios [0:0, 1-15:0.70] (as re-run in DTP's Table 1) | 74.02 → 66.05 ± 0.04 | −7.97 [c] | 0.113 (8.84×) | hand, uniform per layer | cut, then fine-tune | [V full, DTP T1] |
| MagnitudeL2, PruningBench pipeline [59] | 73.87 → 64.96 | −8.91 | 0.125 (8×) [c] | the benchmark's scheme [U] | 100-epoch fine-tune | [V full, T5] |
| GReg-1 / GReg-2 [55] (ICLR 2021) | 74.02 → 67.55 / 67.75; DTP's re-run of GReg-1: 67.35 ± 0.15 | −6.47 / −6.27 [c] | 0.113 | hand | growing regularisation before the cut | [R: LIT_SEALED_WEB §1(a)]; re-run [V full, DTP T1] |
| DTP [56] (ICCV 2023) | 74.02 → 67.74 ± 0.13 | −6.28 [c] | 0.113 | hand (the same ratios) | learned soft masks by differentiable optimal transport | [V full, T1] |
| DepGraph without SL [53] (CVPR 2023) | 73.50 → 67.60 | −5.90 [c] (printed −5.44) | 0.113 (8.87×) | learned global (group norm) | cut, then fine-tune | [R: LIT_SEALED_WEB §1(a)] |
| Metanetwork meta-pruning [58] (2025 preprint) | 73.65 → 68.68 ± 0.87 (3 seeds: 68.65, 67.63, 69.75) | −4.97 [c] | 0.111 (8.98×) | a metanetwork rewrites the weights; group pruning follows | meta-trained on 8 VGG-19 "data models" | [V full, App. D.3, T11–12] |
| OCSPruner [54], pretrained start (WACV 2026) | 73.58 → 69.98 | −3.60 [c] | 0.112 | stability-driven subnetwork search | pruning inside one training cycle | [R: LIT_SEALED_WEB §1(d)] |
| OCSPruner, from scratch | 74.24 → 70.47 | −3.77 | 0.112 | same | one cycle from scratch, mean of 3 | [R: LIT_SEALED_WEB §1(d)] |
| **DepGraph with SL** [53] | 73.50 → 70.39 | −3.11 | 0.112 (8.92×) | learned global | sparse learning, then cut and fine-tune | [R: LIT_SEALED_WEB §1(a)] |
| DepGraph group pruner, re-run by DPM [57] | base not in T5 → 70.74 with SL; 67.58 without | SL adds +3.16 [c] | 0.110 / 0.114 | learned global | as DepGraph | [V full, DPM T5] |
| DPM on DepGraph [57] (2024 preprint) | → 70.92 with SL (+0.18); 67.76 without (+0.18) | — | 0.109 / 0.121 | learned global | smooth multi-step decay to zero, plus gradient-based "self-rectifying" | [V full, DPM T5] |
| SPECTRA rule widths in DepGraph's code (queue P6) | — | rule − DepGraph without SL = +0.47 (all seeds positive). Rule widths with SL − DepGraph with SL = −0.36 at 10k | ≈ 0.11 | threshold rule | as DepGraph | (brief); [R: commit a8815a4 message] |

### 5.2 ResNet-50 on ImageNet, about 0.15–0.33 of FLOPs kept

| Method | Base → pruned | FLOPs kept | Mechanism | Label |
|---|---|---|---|---|
| Taylor-FO (DTP's T4 reprint) | 76.18 → 71.69 | 0.328 (3.05×) [c] | iterative Taylor pruning | [V full, reprint] |
| GReg-1 / GReg-2 (DTP's T4 reprint) | 76.13 → 73.75 / 73.90 | 0.327 (3.06×) [c] | hand per-stage ratios [0, 0.68, 0.68, 0.68, 0.50] (DTP T6), growing regularisation | [V full, reprint] |
| DTP [56] | 76.13 → 74.26 (−1.87) | 0.327 | the same hand ratios, learned soft masks; 90 epochs of pruning, then 90 of fine-tuning | [V full] |
| MetaPruning; DMCP; EagleEye-1G; CafeNet (CHEX's T1 reprints) | 1.0 G → 73.4; 1.1 G → 74.1; 1.0 G → 74.2; 1.0 G → 75.3 | 0.24–0.27 [c] (of 4.1 G) | searched or trained allocation; EagleEye picks by adaptive BN | [V full, reprints] |
| HALP [60] | 76.2 → 74.3 at 1.0 G | 0.24 [c] | latency-saliency knapsack, iterative | [R: LIT_RULE_AND_COMPARATORS §3.1] |
| **CHEX** [39] | 77.8 (own base) → 76.0 at 1.0 G | 0.24 [c] | prune and regrow throughout 250 epochs from scratch; CSS criterion | [V full] |
| OTOv2 [20] | 76.1 → 74.3 / 72.2 / 70.1 (averages; best 72.3 and 70.3) | 0.287 / 0.200 / 0.145 | DHSPG structured-sparsity optimiser, one training from scratch, no fine-tune | [V full, Fig. 6 table] |
| MDP [4] | 70.0 at ×5.16 latency speed-up, against HALP 68.6 at ×4.02 | latency, not FLOPs | channels and blocks, one MINLP | [V full] |

### 5.3 Mechanisms behind the deep-cut gains
1. **Sparse learning co-adapted with the cut.**
   - DepGraph: 70.39 with SL against 67.60 without (+2.79 [c]); DPM's re-run: +3.16 [c].
   - In-house, applying SL after fixing the rule's widths stays 0.36 behind DepGraph's SL at 10k (brief; commit a8815a4).
2. **Regularised or soft-mask training at fixed hand ratios.**
   - +1.30 (GReg-1) and +1.69 (DTP) over one-shot ℓ1 at 8.84× [c, DTP T1].
   - Still about 2.6 below DepGraph with SL [c: 70.39 − 67.74], on a base 0.52 higher (74.02 against 73.50).
3. **Pruning inside training from scratch.**
   - OCSPruner reaches 70.47 [54]; OTOv2 [20] and CHEX [39] do the same on ImageNet. The pruned network gets the whole schedule: 300 epochs per cycle on CIFAR for OCSPruner [R: LIT_RULE_AND_COMPARATORS §3.1], 250 for CHEX.
   - CHEX's ablation (T2) credits +0.9 to periodic prune-and-regrow and +0.8 to dynamic re-allocation of widths across layers [c: 73.9 → 74.8 → 75.6 → 76.0].
4. **Smooth, multi-step decay:** DPM adds +0.18 on DepGraph, with and without SL [57].
5. **Distillation and re-parameterisation.**
   - TAS transfers by distillation [2]; knapsack pruning distils from inner layers [62]; prune-then-distill uses the pruned teacher [61]; ResRep re-parameterises [63].
   - No study found isolates their share at a deep cut. In-house, untuned distillation added +0.00 val [P §444].
6. **Depth and blocks:** MDP's margin over HALP grows with the ratio, on latency [4].
7. **Recipe confounds.** The modern recipe alone adds +4.3 to unpruned ResNet-50 [64]. CHEX's 77.8 base and 250 epochs, and OCSPruner's 300-epoch cycles, make cross-paper deep-cut comparisons recipe-bound.

### 5.4 Left open [H]
- No allocation-only method in this scan reaches DepGraph with SL at 9×, so the remaining gap is a procedure gap. Allocation is worth +0.47 in DepGraph's own code (brief).
- A training-free allocation that co-adapts with sparse learning (measure, then sparse-learn, then re-measure) was not found as such. GReg uses hand ratios, and DepGraph its global norm criterion.
- P7 (the rule re-measured on each seed's post-SL network, in DepGraph's code) is registered and has cells submitted [R: commits 260c82a, 72631dd].

---

## 6. Q5. Better training-free allocation rules

| Rule | Venue | Unit | Signal | Recovery assumed | Reported result | Label |
|---|---|---|---|---|---|---|
| Sensitivity threshold (Li 2017; PaddleSlim; NNI) [77] | ICLR 2017; tools | conv filters or parameters | cut-only validation accuracy per layer; one threshold (PaddleSlim bisects it to a FLOPs target) | fine-tune | the lineage of our rule | [R: LIT_RULE_AND_COMPARATORS §1] |
| FAIR-Pruner | arXiv:2508.02291 (TPAMI submission) | channels, heads, experts | rank conflict at most one shared tolerance | 300-epoch fine-tune on CIFAR-10 | parity with HRank and ITPruner | [R: LIT_RULE_AND_COMPARATORS §1.3] |
| LAMP [77]; ERK [78]; uniform+ | ICLR 2021; ICML 2020 | weights | rescaled magnitude with one global threshold; Erdős–Rényi-kernel densities | fine-tune; sparse training | LAMP beats global, uniform and ERK at high sparsity | [R: LIT_RULE_AND_COMPARATORS §1.3; LIT_ALTERNATIVES ref. 70] |
| Global and protected global [59] | arXiv:2406.12315 | coupled groups | one global ranking; at least 10 % kept per group | 100-epoch fine-tune | comparable at 2×; protected global wins at 4× and 8×, where plain global shows "layer collapse" (§5.2) | [V full] |
| Isomorphic pruning [72] | ECCV 2024 [venue R] | groups, ranked within each isomorphic sub-structure type | the same criterion, ranked by type | fine-tune | "surpasses several pruning baselines" on ImageNet ViTs and CNNs | [V abs] |
| CPMC [35] | IJCNN 2021 | channels | filter + next-layer dependent weights + FLOPs + params, globally normalised | fine-tune | "outperforms" (abstract truncated) | [V abs] |
| Group Fisher [65] | ICML 2021 | coupled channels | Fisher importance, normalised by each channel's memory reduction | pruning during fine-tuning [U: schedule not read] | inference speed-up "without sacrificing accuracy" | [V abs] |
| SOSP; HAP; EigenDamage | ICLR 2022; WACV 2022; ICML 2019 | structures | second-order (Hessian or Fisher) saliency | fine-tune | match or beat the methods of their time | [R: LIT_RULE_AND_COMPARATORS §1.3] |
| HALP [60]; Knapsack pruning [62] | NeurIPS 2022; arXiv:2002.08258 | channels | importance against cost, solved as a knapsack | fine-tune (with inner distillation in [62]) | HALP 74.3 at 1.0 G on ResNet-50 | [R]; [V abs] |
| Layer-compensated pruning [71] | arXiv:1810.00518 | filters | global ranking with a learned per-layer compensation (meta-learning) | fine-tune | gap 0.9 % → 0.7 %, meta-learning 1 h → 7 min | [V abs] |
| LADP [70] | ICCV 2023 [venue R] | weights | per-layer output distortion, shown to be additive, then dynamic programming | fine-tune | up to +1.0 (ResNet-32 C10) and +4.6 (ResNet-50 ImageNet) over other allocations | [V abs] |
| OBC [66]; CHITA [67] | NeurIPS 2022; arXiv:2302.14623 | weights, N:M, bits | exact OBS; combinatorial local quadratic | none (post-training) | large one-shot gains (no CNN-channel figure read) | [V abs] |
| **OSSCAR** [68] | arXiv:2403.12983 | channels | layer-wise reconstruction with combinatorial local search | none | ResNet-50 ImageNet (base 77.01) at 1.7×: 50.16 ± 0.59, against ThiNet 41.08, Lasso 41.68, and below 1 for magnitude, CHIP and FPGM. MobileNetV1 at 1.7×: 63.03 against 59.51 (ThiNet). Ten runs each | [V full, T1] |
| SNOWS [69] | arXiv:2411.18376 | any mask | global non-linear reconstruction by Hessian-free optimisation | none | state of the art on one-shot ResNet and ViT benchmarks | [V abs] |
| OBSPA (SPA) | arXiv:2403.18955 | groups | OBS without fine-tuning | none | loses 0.80–3.73 points at 1.5–1.7× on CIFAR ResNet-50 and VGG-19 | [R: LIT_ALTERNATIVES §1.2] |
| Neuron merging [73] | NeurIPS 2020 | neurons, filters | data-free compensation by cosine similarity | none | VGG-16 C10 93.16 with 64 % of parameters removed, no fine-tuning | [V abs] |
| **Relative Repairability** [74] | arXiv:2605.25508 (May 2026) | weights | allocate by the share of local damage left after channelwise variance-matching repair (unlabelled calibration data) | label-free repair, no fine-tune | "not a universally dominant allocation rule". Beats ERK across a central transition band and LAMP near its upper part (ResNet-18 C100) | [V abs] |
| Allocation and recoverability [76]; ASR [75] | arXiv:2605.21972; arXiv:2605.21426 | weights | ERK against LAMP under one repair; channelwise repair | label-free | The preferred allocation "varies with architecture, dataset difficulty, and sparsity level". ASR: 55.6 against 41.0 (layer-wise repair) and 28.0 (BN only) on ResNet-50 C10 at 90 % sparsity | [V abs] |

**Plausible "improved rule" novelties for SPECTRA [H].**
1. **A recovery-aware sensitivity signal.** Measure each coupled group's loss rise after a cheap repair (BN statistics re-estimated on a calibration split, or channelwise variance matching), then apply the same single threshold.
   - Closest: EagleEye (BN repair for whole-plan evaluation, one network) [45], and Relative Repairability (repair-conditioned allocation; unstructured, label-free, no long recovery) [74].
   - Not found: the structured, coupled-group version, evaluated under a long recovery across families.
2. **A mixed-unit threshold rule.** Put channel groups and removable blocks (and N:M) on one cost-normalised loss-rise scale.
   - Closest: MDP (a per-network MINLP over Taylor importance and a latency table) [4]; LayerPrune (layers only) [3]; CPMC and Group Fisher (cost-normalised channel ranking) [35, 65].
   - Not found: at matched FLOPs, under a long recovery.
3. **A rule whose single setting is validated leave-one-family-out.**
   - FAIR-Pruner and PaddleSlim sweep or bisect per network; no cross-family default was validated in the texts read.
   - Our rule bisects τ to the budget per network: a one-scalar search, not training.

**Already claimed; not a novelty for us:** second-order or Fisher saliency, cost normalisation, isomorphic grouping, protected floors, dynamic programming over additive distortion, one-shot reconstruction, data-free compensation.

**Expected size [H].** Reconstruction-type signals move one-shot accuracy by points [68], but post-recovery accuracy by tenths ([48]; [P §398]). A recovery-aware signal targets exactly the cut-only proxy's failure (τ 0.15 in-house; Kendall ≤ 0.19 elsewhere [46]). Its gain is most likely at the deep budget.

---

## 7. Q6. Why learned pruners collapse to rules

### 7.1 The evidence

| Finding | Source | Label |
|---|---|---|
| The pruned architecture matters more than inherited weights. For ResNet and DenseNet, Slimming's allocation is "not significantly more efficient" than uniform | Liu et al. 2019 [80] | [R: LIT_DB_AVENUES §4] |
| Random per-layer ratios find "comparable or even superior" models; under random ratios criteria are "just comparable" with ℓ1/ℓ2; within 0.5 % of search methods on ImageNet | Li et al. 2022 [48] | [R: LIT_RULE_AND_COMPARATORS §2] |
| Random channel-wise profiles are "as good as metric-based ones". Some layer-wise profiles are "measurably better than common baselines", and the top random layer-wise profiles of one dataset rank high on others | Out-of-the-box [82] | [V abs] |
| Randomly changing which weights are kept within each layer, at fixed per-layer counts, "does not affect the final performance"; data-independent layer ratios with random pruning match "initial tickets" | Su et al. [83] | [V abs] |
| Random pruning at initialisation with pre-chosen layer-wise ratios can match the dense network in sparse training | Liu et al. 2022 [84] | [V abs] |
| Pre-training the large model is "not necessary for obtaining the target pruned structure"; pruning from random weights reaches "similar or even higher accuracy under the same computation budgets" | Pruning from Scratch [85] | [V abs] |
| Pruning-at-initialisation scores reduce to per-layer fractions | Frankle et al. [86] | [R: LIT_DB_AVENUES §2] |
| Pruning beats random "at least for large amounts of pruning"; method rankings flip between CIFAR-VGG and ResNet-56 | Blalock et al. [81] | [R: LIT_RULE_AND_COMPARATORS §2] |
| "No single method consistently outperforms the others"; plain magnitude ℓ1/ℓ2 sits in the top 5 in most rankings | PruningBench [59] | [R: LIT_RULE_AND_COMPARATORS §2] |
| The recipe beats the method: ℓ1 with a better fine-tune matches top-venue rows; large-LR random pruning is competitive | Wang et al. 2023; Le & Hua [87] | [R: LIT_RULE_AND_COMPARATORS §2] |
| Common criteria rank filters almost identically | CWDA [36] | [V abs] |
| Learned against a hand rule in the same pipeline: AMC +0.4 over the hand "deep" policy (ResNet-56, 50 % FLOPs); on VGG-16 ImageNet at 5×, AMC −1.4 against CP's hand-tuned −1.7 (top-5 error increase); RL per-layer scheme mapping ≈ a rule | AMC [28]; Gong et al. [13] | [R: LIT_RULE_AND_COMPARATORS §3.2]; [V full] |
| AMC re-run by LeGR: "higher variances when pruned more aggressively" | LeGR [89] | [R: LIT_RULE_AND_COMPARATORS §3.2] |
| Counterexample: RL-MCTS beats random search at equal executions by 3.5 points [c] on ResNet-50 ImageNet at about 46 % FLOPs ↓. But its random search samples filter sequences, not layer ratios with a criterion | [42] | [V full] |
| In-house: eleven agents level with or below the threshold rule (brief); scratch weights = inherited [P §394]; random = ℓ1 channels [P §398]; 28 judges cannot separate close plans [P §432]; the surrogate does not transfer across families (brief) | — | (brief); [R: LIT_ALTERNATIVES §1.1] |

### 7.2 Conditions shown to give a learned or searched allocator real headroom

| Condition | Evidence | Measured against | Label |
|---|---|---|---|
| No or short recovery | OSSCAR: the method is worth about 9 points one-shot (ResNet-50 at 1.7×: 50.16 against 41.08) [68]. EvoPress [79]: searched allocation 28.76 against OWL 48.07 and uniform 85.84 Wiki2 perplexity (Llama-3-8B, 70 %). AMC: cut-only rewards fail on ImageNet [28] | heuristic criteria; OWL; uniform | [V full]; [R: LIT_ALTERNATIVES §1.2]; [R] |
| Deep cuts and compact networks | DepGraph T2: learned over uniform sparsity +0.69 to +1.68 (ResNet-56 C100, 1.5–12×). MetaPruning [88]: MobileNetV1 0.25× 57.2 against 50.6. Plain global collapses layers at 4–8× [59]. DominoSearch: +2.1 at 87.5 % [9]. Random plan spaces diverge above about 90 % FLOPs removed [49] | uniform, plain global | [R: LIT_SEALED_WEB §1(a)]; [R: LIT_RULE_AND_COMPARATORS §2]; [V full]; [V abs]; [R: LIT_DB_AVENUES §1] |
| Latency and hardware targets | Filter pruning can raise latency, and layer removal wins at matched latency [3]. MDP +1.4 at 28 % more speed than HALP [4]. HAQ 1.4–1.95× lower latency than fixed 8-bit [16] | channel-only pruning; fixed bit width | [V full]; [V abs] |
| Mixed or joint spaces | APQ +2.3 over the separate pipeline [17]; DJPQ over two-stage [18]; DominoSearch [9] | sequential or uniform pipelines | [V abs] |
| **Against a tuned rule, in any regime** | tenths: AMC +0.4; EagleEye +0.31; Blending +0.23 / +0.32; Gong et al. ≈ 0; SPECTRA 0 (brief) | a tuned or sensitivity rule | as above |

### 7.3 Reading for SPECTRA [H]
- Published headroom is almost always measured against uniform or a sequential pipeline. Against a tuned rule it is tenths, and SPECTRA's agents sit at zero against the strongest rule. Our result agrees with this literature; it does not contradict it.
- The regimes where a learned allocator has shown points of headroom are the ones SPECTRA's main protocol excludes:
  - no or short recovery;
  - latency targets;
  - mixed granularity;
  - deep cuts on compact nets.

  A thesis that wants an agent win has to enter one of them deliberately, with the rule-side comparator registered in the same space.

---

## 8. Ranked directions

Ranked by expected value for the thesis, given the evidence above. Run costs use the ResNet-56 fine-tune time recorded for ledger §419: 0:23:39 for 60 epochs [R: LIT_RULE_AND_COMPARATORS §3.2]. That is about 0.66 GPU-h per 100-epoch run [c]; the GPU type is not restated there, and larger networks cost more. Every bar is to be registered in `docs/SITTING_GPU_QUEUE.md` before any job, and results quoted on TEST 5k with val, 10k and honest beside.

### D1. A recovery-aware threshold rule (the improved training-free rule)
- **Idea.** Keep the one-threshold design and its bisection to the budget, but measure each coupled group's loss rise after a cheap repair (BN statistics re-estimated on a calibration split, optionally channelwise variance matching), with a protected per-group floor.
- **Why it could beat the current rule.**
  - The rule's weak point is its signal. Cut-only accuracy ranks plans at τ 0.15 against the 100-epoch outcome (brief), and cut-time scores elsewhere reach Kendall 0.01–0.19 [46].
  - On one network, BN recalibration lifts the correlation of plan scores with fine-tuned accuracy from −0.04–0.25 to 0.68–0.82 [45].
  - Repair-conditioned allocation beats ERK and LAMP in a high-sparsity transition band [74]. Repair or reconstruction is worth points when recovery is short [68, 75].
- **Closest prior; novelty left.**
  - Closest: EagleEye [45]; Relative Repairability [74]; the PaddleSlim/NNI cut-only rules [R]; FAIR-Pruner [R].
  - Left: a structured, coupled-group, repair-aware threshold rule, evaluated under one long recovery across about ten families, with one setting validated leave-one-family-out.
- **First test (days).**
  1. No training: recompute the per-group curves with BN recalibration on the 100+ networks (forward passes only), and diff the keep vectors against today's rule at FLOPs 0.45 and 0.20. If the plans barely move, stop here.
  2. If they move, run both plans one-shot with the G2 100-epoch recovery on 24 networks spread across families, × 2 budgets × 2 seeds: 192 runs, of the order of 130 GPU-h at the ResNet-56 cost [c].
- **Risk.**
  - Medium. After a long recovery, differences among sensible plans sit near seed noise (seeds agree at τ 0.54; brief), so a gain may show only at the deep budget.
  - The repair is itself a choice: calibration split, size, number of iterations (EagleEye used 1/30 of the training set and 100 iterations [R]).
  - This rule is also the comparator D2 needs, so it pays off even if it ties.

### D2. A mixed-granularity agent with a short-recovery pick (the DRL candidate)
- **Idea.** Extend the plan-as-action bandit so that each layer's action picks a unit and a level (channel-group width, whole-block removal where the shortcut allows, 2:4 N:M on eligible convolutions, a kernel or stripe pattern) and, optionally, the criterion. The agent proposes K plans per network, checks them with a short recovery (BN-only or one epoch), and commits to the best.
- **Why it could beat the threshold rule.**
  - A channel-only rule cannot express depth or N:M. The literature shows granularity headroom at high ratios: MDP +1.4 top-1 at 28 % more speed than channel-only HALP [4]; layer pruning +1.0 to +1.3 at equal latency reduction, and +7.09 % for random models [3]; mixed N:M +2.1 (equal size) and +1.3 (equal FLOPs) over uniform N:M [9].
  - A larger, interacting action space weakens the additivity that a threshold rule relies on (the AMC and EvoPress critiques [R: LIT_RULE_AND_COMPARATORS §1.4]), leaving a learner more to exploit.
- **Closest prior; novelty left.**
  - Closest: Gong et al. (RL against a rule over per-layer schemes, per network: a tie) [13]; Balaskas et al. (RL over per-layer pruning algorithm and bits, per network, no retraining) [25]; the generalist agent (frozen across datasets, LeNet only) [26]; MDP (per-network MINLP) [4].
  - Left: a policy trained across families and applied frozen over a mixed unit set, judged at matched FLOPs under one recovery against a mixed-unit rule.
- **First test (days).**
  1. Headroom before learning. Build the mixed-unit rule: D1's curves plus block-removal (and N:M) curves on one cost-normalised scale.
  2. Compare it with the channel-only rule on the families that have removable blocks (ResNets, MobileNetV2): about 12 networks × FLOPs 0.20 and 0.45 × 2 seeds × 2 arms = 96 runs, about 65 GPU-h [c].
  3. Register the FLOPs convention for N:M beforehand (for example, 2:4 counts as half that layer's MACs [H]) and report measured latency separately.
  4. Only if the mixed rule beats the channel rule, train the agent on the mixed space. Give its short-recovery pick a registered guard against untrainable picks; the one-epoch pick lost 1.95 at FLOPs 0.20 [P §443].
- **Risk.**
  - High. Under FLOPs (not latency), block removal may not beat channel thinning; the published wins are latency wins [3, 4].
  - Gong et al.'s RL matched a rule even in a mixed space [13].
  - N:M saves FLOPs only on paper unless the kernels run sparse. Whether the stack can measure 2:4 speed on the RTX 4090 is [U].
  - Agent training is the expensive step; gate it on step 2.

### D3. The rule inside sparse learning at deep cuts
- **Idea.** Let the rule co-adapt with sparse learning: measure its curves on the network after (or during) sparse learning and cut from there. If one pass is not enough, alternate measure, sparse-learn and re-measure for two or three rounds.
- **Why it could tie or beat DepGraph at 9×.**
  - The deep-cut gap is a procedure gap. Sparse learning adds +2.79 in DepGraph and +3.16 in DPM's re-run [53, 57].
  - The rule's widths beat DepGraph's allocation without SL (+0.47), but trail by 0.36 at 10k when SL is added afterwards (brief; commit a8815a4).
  - CHEX credits +0.8 to re-allocating widths during training [39].
- **Closest prior; novelty left.**
  - Closest: DepGraph SL (global norm criterion) [53]; GReg (hand ratios) [55]; DPM [57]; OCSPruner [54]; CHEX [39].
  - Left: a training-free sensitivity rule as the allocator inside sparse learning. The novelty is small, but it would close the 9× gap without a learned allocation.
- **First test.** Already registered: P7, the rule re-measured on each seed's post-SL network and cut and fine-tuned in DepGraph's code (about 2.5 GPU-h; bars against DepGraph SL written before any job; cells 24393719-21) [R: commits 260c82a, 72631dd]. Read P7 first. If it ties or trails, the next cell is a two-round alternation at the same VGG-19 CIFAR-100 point, with 3 seeds.
- **Risk.** Low cost and low novelty. A win would be a recipe result, not an agent result.

### D4. Multi-fidelity selection among a portfolio of plans (the "simulate briefly" part, measured first)
- **Idea.** For each network, generate a small portfolio of plans (the threshold rule at two or three settings, D1's rule, a protected-global rule, agent samples). Screen them by successive halving on BN-only, then 1, 5 and 20 epochs, and give only the survivor the 100-epoch recovery.
- **Why it could help.**
  - Within a network, plans differ for real: the best beats a random one by about 1.2 points, and seeds agree at τ 0.54 (brief).
  - BN-repaired and briefly trained scores track recovered accuracy better than cut-only scores [45, 46, 30].
  - Successive halving is the standard way to spend a budget on noisy cheap signals.
- **Closest prior; novelty left.**
  - Closest: Li et al. (100 random plans, top 5 at 5 epochs) [48]; EagleEye [45]; NetAdapt [47]; Blending (3-epoch fitness) [30]; Gong et al.'s 2-epoch RL reward [13].
  - Left: successive halving for pruning-plan selection was not found (§9). The width-plan proxy ladder is DB2's open item [R: LIT_DB_AVENUES §2].
- **First test.**
  1. No GPU: from logged runs, compute Kendall τ and selection regret of the one-epoch scores against 100-epoch TEST per network. `LIT_ALTERNATIVES_21NOV.md` §0 says such one-epoch logs exist [R].
  2. Then 20 networks × 8 plans through the BN, 1-epoch and 5-epoch rungs, with 100 epochs for the top 2 and for the rule (about 60 full runs plus the short rungs).
- **Risk.** Medium. Seed noise caps the attainable τ (0.54 between seeds). The in-house one-epoch pick lost 1.95 at FLOPs 0.20 [P §443]. Any gain at FLOPs 0.45 is likely tenths.

**Why this order [H].**
- D1 is the cheapest, has the most direct evidence, and builds the comparator D2 needs.
- D2 carries the thesis's identity and the highest risk; its first step is a rule-only headroom test.
- D3 is already running and closes a known gap.
- D4 is a component that serves D1 and D2, and it overlaps DB2.

---

## 9. Search log and limits

- **Searches.** 17 web searches:
  - per-layer method selection by RL;
  - hyper-heuristics for pruning;
  - criterion ensembles;
  - MCTS look-ahead;
  - short-fine-tune rank correlation (two);
  - zero-cost proxies for pruning;
  - model-based RL for compression;
  - deep-cut VGG-19 CIFAR-100 results (two);
  - DominoSearch;
  - RL-MCTS results;
  - RL against random search;
  - recovery-aware sensitivity;
  - successive halving for subnetwork selection;
  - venue look-ups (two).
- **Reads.**
  - 69 arXiv abstract records (curl; the arXiv API returned HTTP 503).
  - 20 PDFs converted with `pdftotext`: MDP, LayerPrune, Gong et al., Balaskas et al., LFPC, Blending, composite saliency, LAASP, RL-MCTS, ICCV 2021 exploration-estimation (skimmed, not used), DPM, OTOv2, CHEX, DTP, PruningBench, the generalist agent, EvoFC, the metanetwork paper, OSSCAR, LADP.
  - One NeurIPS abstract page; two Semantic Scholar records.
  - Failed: Torch-Pruning `reproduce/README.md` (HTTP 404), MDPI Electronics (HTTP 403).
- **Layout limits.** Two-column extraction scrambled MDP's Table 1, LFPC's Table 2, PruningBench's leaderboards, the composite-saliency Table III and LADP's tables. From those, only prose figures, rows that could be aligned, or legible companion tables (PruningBench Table 5, LFPC Table 4) are quoted.
- **Not opened:**
  - TraceNAS, ATO, HESSO, GETA, OTOv3, SWP, TAS, DBP and FITCompress (abstract only);
  - the NCA ensemble paper and the MDPI paper (snippets only);
  - DFPC;
  - EagleEye, Oracle pruning, NetAdapt, Li 2022, Network Pruning Spaces and AMC, whose facts come from earlier verified scans [R].
- **Coverage.** IEEE, Elsevier, Springer and MDPI venues without open copies are under-covered. "Not found" is a statement about these searches.
- **Venues** come from the arXiv comment or journal reference, the PDF's own header, or the proceedings page, except where marked [venue R]: Blending (Springer listing via search), EvoFC and AutoMC (earlier repo scan), LADP (CVF listing via search), Isomorphic pruning (earlier repo scan).
- **Corrections.** My first guess for CHEX's ID (2112.02184) opened an unrelated paper; the ID in §10 is the one its abstract page confirms. DominoSearch has no arXiv record in these searches.
- **Project facts** are taken from the brief and from the commit messages in this session's git log (a8815a4, 307665f, 260c82a, 72631dd). The ledger and the queue were not read.

---

## 10. References

Labels: [V full] or [V abs] = read at the source; [R] = reported (second-hand, source named).

**Mixed granularity and joint compression**
1. Wen, Wu, Wang, Chen, Li. Learning Structured Sparsity in Deep Neural Networks (SSL). NIPS 2016. arXiv:1608.03665. [V abs]
2. Dong, Yang. Network Pruning via Transformable Architecture Search (TAS). NeurIPS 2019. arXiv:1905.09717. [V abs]
3. Elkerdawy et al. (5 authors). To Filter Prune, or to Layer Prune, That Is The Question (LayerPrune). arXiv:2007.05667; no venue on the arXiv record or in the PDF. [V full]
4. Sun, Lakshmanan, Shen, Lan, et al. (6 authors). MDP: Multidimensional Vision Model Pruning with Latency Constraint. CVPR 2025. arXiv:2504.02168 (earlier version arXiv:2406.12079). [V full]
5. Fu et al. (8 authors). DepthShrinker: A New Compression Paradigm Towards Boosting Real-Hardware Efficiency of Compact Neural Networks. ICML 2022. arXiv:2206.00843. [V abs]
6. Kim, Jeong, Lee, Song. Efficient Latency-Aware CNN Depth Compression via Two-Stage Dynamic Programming. ICML 2023. arXiv:2301.12187. [V abs]
7. Kim, El Halabi, Ji, Song. LayerMerge: Neural Network Depth Compression through Layer Pruning and Merging. ICML 2024. arXiv:2406.12837. [V abs]
8. Wang et al. (6 authors). DBP: Discrimination Based Block-Level Pruning for Deep Model Acceleration. arXiv:1912.10178. [V abs]
9. Sun, Zhou, Stuijk, Wijnhoven, Nelson, Li, Corporaal. DominoSearch: Find Layer-Wise Fine-Grained N:M Sparse Schemes from Dense Neural Networks. NeurIPS 2021 (proceedings abstract page; no arXiv record found). [V abs]
10. Xu, Park, Brick. Hybrid Pruning: Thinner Sparse Networks for Fast Inference on Edge Devices. arXiv:1811.00482. [V abs]
11. Meng et al. (7 authors). Pruning Filter in Filter (SWP). NeurIPS 2020. arXiv:2009.14410. [V abs]
12. Niu et al. (8 authors). PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning. ASPLOS 2020. arXiv:2001.00138. [V abs]
13. Gong, Yuan, Zhan, Niu, Li, Zhao, Cai, Liu, Ren, Lin, Tang, Wang. Automatic Mapping of the Best-Suited DNN Pruning Schemes for Real-Time Mobile Acceleration. ACM Transactions on Design Automation of Electronic Systems, doi:10.1145/3495532 (as printed in the arXiv PDF; issue year not verified). arXiv:2111.11581. [V full]
14. Liu, Ma, Xu, Wang, et al. (6 authors). AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates. arXiv:1907.03141; no venue on the record. [V abs]
15. Li, Lin, Liu, Ye, et al. (10 authors). Towards Compact CNNs via Collaborative Compression. CVPR 2021. arXiv:2105.11228. [V abs]
16. Wang, Liu, Lin, Lin, Han. HAQ: Hardware-Aware Automated Quantization with Mixed Precision. CVPR 2019. arXiv:1811.08886. [V abs]
17. Wang, Wang, Cai, Lin, et al. (6 authors). APQ: Joint Search for Network Architecture, Pruning and Quantization Policy. CVPR 2020. arXiv:2006.08509. [V abs]
18. Wang, Lu, Blankevoort. Differentiable Joint Pruning and Quantization for Hardware Efficiency (DJPQ). ECCV 2020. arXiv:2007.10463. [V abs]
19. van Baalen, Louizos, Nagel, Amjad, et al. (7 authors). Bayesian Bits: Unifying Quantization and Pruning. arXiv:2005.07093; no venue on the record. [V abs]
20. Chen, Liang, Ding, Zhu, et al. (5 authors). OTOv2: Automatic, Generic, User-Friendly. ICLR 2023. arXiv:2303.06862. [V full]
21. Chen et al. (7 authors). OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators. arXiv:2312.09411. [V abs]
22. Qu, Aponte, Banbury, Robinson, et al. (8 authors). Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression (GETA). arXiv:2502.16638. [V abs]
23. Chen, Qu, Aponte, Banbury, et al. (10 authors). HESSO: Towards Automatic Efficient and User Friendly Any Neural Network Training and Pruning. arXiv:2409.09085. [V abs]
24. Zandonati, Bucagu, Pol, Pierini, et al. (6 authors). Towards Optimal Compression: Joint Pruning and Quantization (FITCompress). arXiv:2302.07612. [V abs]
25. Balaskas, Karatzas, Sad, Siozios, Anagnostopoulos, Zervakis, Henkel. Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization. IEEE Transactions on Emerging Topics in Computing, doi:10.1109/TETC.2023.3346944 (accepted, per the PDF). arXiv:2312.15322. [V full: §3, §5]
26. Gonzalez-Sahagun, Conant-Pablos, Ortíz-Bayliss, Cruz-Duarte. A Generalist Reinforcement Learning Agent for Compressing Convolutional Neural Networks. IEEE Access 2024. doi:10.1109/ACCESS.2024.3385857. [V full: abstract, action space, results]
27. Gonzalez-Sahagun, Conant-Pablos, Ortíz-Bayliss, Cruz-Duarte. A Generalist Reinforcement Learning Agent for Compressing Multiple Convolutional Networks Using Singular Value Decomposition. IEEE Access 2024. doi:10.1109/ACCESS.2024.3457863. [V abs, Semantic Scholar record]
28. He, Lin, Liu, Wang, Li, Han. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018. arXiv:1802.03494. [R: LIT_RULE_AND_COMPARATORS_21NOV §3.2; LIT_DB_AVENUES_21NOV §2]

**Criteria and rule portfolios**

29. He, Ding, Liu, Zhu, Zhang, Yang. Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration (LFPC). CVPR 2020, CVF open access. [V full]
30. He, Huang, Liang, Liang, Yang. Blending Pruning Criteria for Convolutional Neural Networks. ICANN 2021, LNCS 12894, doi:10.1007/978-3-030-86380-7_1 [venue R: Springer listing via search]. arXiv:2107.05033. [V full]
31. Persand, Anderson, Gregg. Composition of Saliency Metrics for Channel Pruning with a Myopic Oracle. IEEE SSCI 2020. arXiv:2004.03376. [V full]
32. Ghimire, Lee, Kim. Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural Network Acceleration (LAASP). Image and Vision Computing 136 (2023) 104745, doi:10.1016/j.imavis.2023.104745. arXiv:2506.20152. [V full]
33. Chen, Liu, Hu, Lin, Gong, Chen, Peng, Geng. Evolving Filter Criteria for Randomly Initialized Network Pruning in Image Classification (EvoFC). Neurocomputing 2024, doi:10.1016/j.neucom.2024.127872 [venue R: LIT_ALTERNATIVES_21NOV ref. 112]. Author manuscript, A*STAR OAR. [V full: abstract, §4]
34. He, Liu, Zhu, Yang. Filter Pruning by Switching to Neighboring CNNs with Good Attributes (MFP). IEEE TNNLS (accepted, per the arXiv comment). arXiv:1904.03961. [V abs]
35. Yan, Guo, Li, Yang, et al. (5 authors). Channel Pruning via Multi-Criteria based on Weight Dependency (CPMC). IJCNN 2021. arXiv:2011.03240. [V abs]
36. Huang, Shao, Wang, Lin, Luo. Convolution-Weight-Distribution Assumption: Rethinking the Criteria of Channel Pruning. NeurIPS 2021. arXiv:2004.11627. [V abs]
37. Liu, Kung, Wentzlaff. Evolving Transferable Neural Pruning Functions. GECCO 2022. arXiv:2110.10876. [V abs]
38. Wang, Wang, Shi. AutoMC: Automated Model Compression based on Domain Knowledge and Progressive Search Strategy. ICDE 2024 [venue R: LIT_ALTERNATIVES_21NOV ref. 75]. arXiv:2201.09884. [V abs]
39. Hou, Qin, Sun, Ma, et al. (10 authors). CHEX: CHannel EXploration for CNN Model Compression. CVPR 2022. arXiv:2203.15794. [V full]
40. A hybrid search for the per-layer pruning ratio and algorithm. Electronics 13(4):688 (MDPI). [R: search snippet; the page returned HTTP 403]
41. Dong et al. Pruner-Zero (ICML 2024; arXiv:2406.02924); Liu, He, Wu, Song. OptiShear (arXiv:2502.10735). [R: LIT_ALTERNATIVES_21NOV refs. 48–49]

**Simulation, look-ahead and proxies**

42. Wang, Li. Channel Pruning via Lookahead Search Guided Reinforcement Learning (RL-MCTS). WACV 2022, CVF open access (pp. 2029–2040 [R: proceedings listing via search]). [V full]
43. Park, Lee, Mo, Shin. Lookahead: A Far-Sighted Alternative of Magnitude-based Pruning (LAP). ICLR 2020. arXiv:2002.04809. [V abs]
44. Fan, Mu, Zhang. Bayesian Optimization with Clustering and Rollback for CNN Auto Pruning (BOCR). ECCV 2022. arXiv:2109.10591. [V abs]
45. Li, Wu, Su, Wang, Lin. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV 2020. arXiv:2007.02491. [R: LIT_DB_AVENUES_21NOV §2, read in full there]
46. Feng, Tao, Wang. Is Oracle Pruning the True Oracle? TMLR 2026. arXiv:2412.00143. [R: LIT_DB_AVENUES_21NOV §2]
47. Yang et al. NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications. ECCV 2018. arXiv:1804.03230. [R: LIT_DB_AVENUES_21NOV §2; LIT_RULE_AND_COMPARATORS_21NOV §1.3]
48. Li, Adamczewski, Li, Gu, Timofte, Van Gool. Revisiting Random Channel Pruning for Neural Network Compression. CVPR 2022. arXiv:2205.05676. [R: LIT_DB_AVENUES_21NOV §1; LIT_RULE_AND_COMPARATORS_21NOV §2]
49. He, Yang, Song, Pu, Hu, Jiang, Zhang, Ding. Network Pruning Spaces. arXiv:2304.09453. [R: LIT_DB_AVENUES_21NOV §1]
50. Abdelfattah, Mehrotra, Dudziak, Lane. Zero-Cost Proxies for Lightweight NAS. ICLR 2021. arXiv:2101.08134. [R: LIT_DB_AVENUES_21NOV §2]
51. Malettira, Nagaraj, Roy, Negi, et al. (5 authors). TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation. arXiv:2602.02891. [V abs; correlation values R: search snippet]
52. Per-target surrogates: SeerNet, HiReLC, Fine-Pruning. [R: LIT_DB_AVENUES_21NOV §5, refs. 36–38]

**Deep cuts**

53. Fang, Ma, Song, Mi, Wang. DepGraph: Towards Any Structural Pruning. CVPR 2023. arXiv:2301.12900. [R: LIT_SEALED_WEB_20NOV §1(a)]
54. Ghimire, Kil, Jeong, Park, Kim. One-Cycle Structured Pruning via Stability-Driven Subnetwork Search (OCSPruner). WACV 2026. arXiv:2501.13439. [R: LIT_SEALED_WEB_20NOV §1(d)]
55. Wang, Qin, Zhang, Fu. Neural Pruning via Growing Regularization (GReg). ICLR 2021. arXiv:2012.09243. [R: LIT_SEALED_WEB_20NOV §1(a), (c); its 8.84× and 3.06× rows reprinted or re-run in ref. 56, V full]
56. Li, van Gemert, Hoefler, Moons, et al. (6 authors). Differentiable Transportation Pruning (DTP). ICCV 2023. arXiv:2307.08483. [V full]
57. Yang, Gao, Li, Li, et al. (6 authors). Decay Pruning Method: Smooth Pruning With a Self-Rectifying Procedure (DPM). arXiv:2406.03879. [V full]
58. Liu, Wang, Zhang. Meta Pruning via Graph Metanetworks: A Universal Meta Learning Framework for Network Pruning. arXiv:2506.12041 (PDF header "Under Review"). [V full: App. D.3]
59. Li, Li, Xue, Fang, et al. A Comprehensive Study of Structural Pruning for Vision Models (PruningBench). arXiv:2406.12315. [V full: Table 5, §5.2; R: LIT_DB_AVENUES_21NOV ref. 5]
60. Shen, Yin, Molchanov, Mao, Liu, Alvarez. Structural Pruning via Latency-Saliency Knapsack (HALP). NeurIPS 2022. arXiv:2210.06659. [R: LIT_RULE_AND_COMPARATORS_21NOV §3.1]
61. Park, No. Prune Your Model Before Distill It. ECCV 2022. arXiv:2109.14960. [V abs]
62. Aflalo, Noy, Lin, Friedman, et al. (5 authors). Knapsack Pruning with Inner Distillation. arXiv:2002.08258. [V abs]
63. Ding et al. ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting. ICCV 2021. arXiv:2007.03260. [R: LIT_RULE_AND_COMPARATORS_21NOV §3.1]
64. Tessier, Boukli Hacene, Gripon. ThinResNet: A New Baseline for Structured Convolutional Networks Pruning. arXiv:2309.12854. [R: LIT_DB_AVENUES_21NOV §4, §6]

**Training-free allocation rules**

65. Liu, Zhang, Kuang, Zhou, et al. (10 authors). Group Fisher Pruning for Practical Network Compression. ICML 2021. arXiv:2108.00708. [V abs]
66. Frantar, Singh, Alistarh. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning. NeurIPS 2022. arXiv:2208.11580. [V abs]
67. Benbaki, Chen, Meng, Hazimeh, et al. (7 authors). Fast as CHITA: Neural Network Pruning with Combinatorial Optimization. arXiv:2302.14623. [V abs]
68. Meng, Ibrahim, Behdin, Hazimeh, et al. (6 authors). OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization. arXiv:2403.12983; no venue on the record or in the text read. [V full: Table 1]
69. Lucas, Mazumder. Preserving Deep Representations in One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework (SNOWS). arXiv:2411.18376. [V abs]
70. Xu, Wang, Geng, Wu, Li, Lin (the arXiv version adds Jie Lin). Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks (LADP). ICCV 2023, pp. 17447–17457 [venue R: CVF listing via search]. arXiv:2308.10438. [V abs]
71. Chin, Zhang, Marculescu. Layer-compensated Pruning for Resource-constrained Convolutional Neural Networks. arXiv:1810.00518. [V abs]
72. Fang, Ma, Mi, Wang. Isomorphic Pruning for Vision Models. ECCV 2024 [venue R: LIT_RULE_AND_COMPARATORS_21NOV §1.4]. arXiv:2407.04616. [V abs]
73. Kim, Kim, Park, Jeon. Neuron Merging: Compensating for Pruned Neurons. NeurIPS 2020. arXiv:2010.13160. [V abs]
74. Zhan, He, Hu, Chen. Relative Repairability: A Calibration-Based Diagnostic for High-Sparsity Post-Pruning Allocation. arXiv:2605.25508. [V abs]
75. Zhan, Chen, Hu. Adaptive Signal Resuscitation: Channel-wise Post-Pruning Repair for Sparse Vision Networks (ASR). arXiv:2605.21426. [V abs]
76. Zhan, Hu, He. How Sparsity Allocation Shapes Label-Free Post-Pruning Recoverability. arXiv:2605.21972. [V abs]
77. Rules verified in earlier scans, not re-read: Li et al. 2017 (ICLR 2017, arXiv:1608.08710); the PaddleSlim and NNI sensitivity pruners; LAMP (ICLR 2021, arXiv:2010.07611); SOSP (ICLR 2022, arXiv:2110.11395); HAP (WACV 2022, arXiv:2101.08940); EigenDamage (ICML 2019, arXiv:1905.05934); FAIR-Pruner (arXiv:2508.02291); SPA/OBSPA (arXiv:2403.18955). [R: LIT_RULE_AND_COMPARATORS_21NOV §1; LIT_ALTERNATIVES_21NOV ref. 32]
78. Evci, Gale, Menick, Castro, Elsen. Rigging the Lottery (RigL; ERK densities). ICML 2020. arXiv:1911.11134. [R: LIT_ALTERNATIVES_21NOV ref. 70]
79. Sieberling, Kuznedelev, Kurtic, Alistarh. EvoPress: Accurate Dynamic Model Compression via Evolutionary Search. arXiv:2410.14649. [R: LIT_ALTERNATIVES_21NOV §1.2]

**Why learned allocators collapse to rules**

80. Liu, Sun, Zhou, Huang, Darrell. Rethinking the Value of Network Pruning. ICLR 2019. arXiv:1810.05270. [R: LIT_DB_AVENUES_21NOV §4]
81. Blalock, Gonzalez Ortiz, Frankle, Guttag. What is the State of Neural Network Pruning? MLSys 2020. arXiv:2003.03033. [R: LIT_RULE_AND_COMPARATORS_21NOV §2]
82. Venkatesan, Swaminathan, Zhou, Luo. Out-of-the-box Channel Pruned Networks. arXiv:2004.14584. [V abs]
83. Su, Chen, Cai, Wu, et al. (7 authors). Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot. NeurIPS 2020. arXiv:2009.11094. [V abs]
84. Liu, Chen, Chen, Shen, et al. (7 authors). The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training. ICLR 2022. arXiv:2202.02643. [V abs]
85. Wang, Zhang, Xie, Zhou, et al. (7 authors). Pruning from Scratch. arXiv:1909.12579; no venue on the record. [V abs]
86. Frankle, Dziugaite, Roy, Carbin. Pruning Neural Networks at Initialization: Why are We Missing the Mark? ICLR 2021. arXiv:2009.08576. [R: LIT_DB_AVENUES_21NOV §2]
87. Wang, Qin, Bai, Fu. Why is the State of Neural Network Pruning so Confusing? arXiv:2301.05219. Le, Hua. Network Pruning That Matters: A Case Study on Retraining Variants. ICLR 2021. arXiv:2105.03193. [R: LIT_RULE_AND_COMPARATORS_21NOV §2]
88. Liu, Mu, Zhang, Guo, Yang, Cheng, Sun. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. ICCV 2019. arXiv:1903.10258. [R: LIT_RULE_AND_COMPARATORS_21NOV §2]
89. Chin, Ding, Zhang, Marculescu. Towards Efficient Model Compression via Learned Global Ranking (LeGR). CVPR 2020. arXiv:1904.12368. [R: LIT_RULE_AND_COMPARATORS_21NOV §3.2]
