# Data-backed avenues DB1-DB7: novelty check with verified citations (21 Nov 2026)

Compiled 21 Nov 2026 (IST) by the literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**How sources were checked.** arXiv abstract pages and the arXiv API (titles, authors, comments, journal refs); full texts through arXiv HTML or ar5iv, read by a page-summarising model; venue pages through search results where the arXiv record names no venue. A reference number in brackets points to §9.

**Labels.** [V] read at the source this session ("full" = full text, "abs" = abstract page or API record only). [R] seen reported elsewhere (a search snippet, another paper, or an earlier repo scan), not opened. [H] my interpretation. Every verdict and every "what would still be new" list is my synthesis of the cited items [H]. "Not found" means not found in this session's searches (§8). Statements about the project are taken from the brief and marked (brief); they were not re-checked against the ledger.

**Relation to earlier scans [R].** `LIT_ALTERNATIVES_21NOV.md` (same folder) ranks a level predictor (its C1, close to DB3 here) and an evaluation paper (its C6, close to DB2 and DB7). This note adds the outcome benchmark (DB1), prune-or-pick (DB4) and recovered-outcome rewards (DB5), and checks each avenue's closest prior directly.

---

## 0. Verdicts at a glance

| Avenue | Verdict | Closest prior | Still new [H] |
|---|---|---|---|
| **DB1** surrogate or tabular benchmark for structured-pruning allocation | **Open** | NATS-Bench size space: widths, but trained from scratch on one skeleton [3]; PruningBench: 645 pruned models in leaderboards, no per-plan table or surrogate [5]; Network Pruning Spaces: random width plans of one network, unreleased [7] | The first released table of width plan × recovered accuracy across families, multi-fidelity, with a surrogate judged by selection regret |
| **DB2** proxy validity at scale | **Partly done** | Oracle-pruning study (TMLR 2026): about 37K models, Kendall τ of 0.01-0.19 between cut-time and retrained scores on three CNN settings, but masks at fixed layer ratios [12]; EagleEye: BN-recalibrated vs cut-only on one network's width plans [10]; PruneShift: rank agreement near one with maximal regret, in theory [13] | Width plans, not masks; a ladder of proxies (cut, BN, 1, 20 epochs, data-free) against 100-epoch honest TEST across families and ratios; the case where proxies favour the learned agent and recovery erases the edge (brief) |
| **DB3** predicting prunability | **Partly done in pieces; the cross-family predictor is open** | Lost in Pruning: prune potential defined and measured, not predicted [20]; Rosenfeld et al.: within-family scaling law, unstructured [21]; Pruning Laws for LLMs: post-pruning, no recovery [22]; heavy-tail metrics [23-25] | A predictor of the deepest structured cut within X points after a fixed recovery, from cheap measurements, tested leave-one-family-out |
| **DB4** prune or pick | **Partly done** | Crowley et al.: reduced networks beat pruned-and-tuned at matched params (WRN, DenseNet on CIFAR-10; ResNet on ImageNet) [29]; ThinResNet [30]; Blalock Fig. 1 [4] | A matched-FLOPs map across many families and datasets, siblings under one recipe, seeds, honest deltas, and each route's GPU cost |
| **DB5** allocator trained on recovered outcomes | **Partly done per target; the generic version is open, with low expected value** | Fine-Pruning: GP over post-fine-tune error [36]; SeerNet: per-backbone predictor fitted to about 800 trained compressed models [37]; HiReLC: MLP surrogate of post-fine-tune accuracy as the reward [38]; N2N: reward after 5 epochs [39] | A frozen cross-family allocator trained through a surrogate of 100-epoch outcomes; the brief's own findings predict a flat target at mild ratios |
| **DB7** the fine-tune confound | **Partly done** | Jin et al.: dense extended-training control, unstructured pruning [45]; Wang et al. 2023: the LR schedule as hidden factor [43]; Le & Hua [41]; Renda et al. [42]; Blalock [4] | The unpruned network under the same recovery, for structured pruning, across families and published recipes; an audit of how much reported Δacc it absorbs |

---

## 1. DB1. A surrogate or tabular benchmark for structured-pruning allocation

**Verdict: open.** NAS has tabular benchmarks [1-3], surrogate benchmarks [8] and learning-curve surrogates [9]. Pruning has benchmarks of methods (leaderboards) [4, 5] and per-network studies of random width plans [6, 7]. No released table of (network, width plan) → recovered accuracy, and no surrogate over width plans across networks, was found.

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| NATS-Bench, size space [3] | IEEE TPAMI 2021; arXiv:2009.00437 | 8^5 = 32,768 channel configurations (5 layers, each from {8, ..., 64}) of one fixed skeleton; trained on CIFAR-10, CIFAR-100 and ImageNet16-120 under 12- and 90-epoch budgets; per-epoch losses and accuracies, params, FLOPs, latency and trained weights released; "trained multiple times" [V full] | No pruning: every configuration is trained from scratch; one skeleton; no inherited weights, recovery or cross-network surrogate [V full] |
| PruningBench [5] | arXiv:2406.12315 (v7, Jan 2025), preprint | 16 methods (15 names: BNScale is both regulariser and criterion); ResNet-18/50 and VGG-19 on CIFAR-100, ResNets and ViT-Small on ImageNet, YOLOv8 on COCO; speed-ups 2/4/8× (CIFAR) and 2/3/4×; 100-epoch fine-tune on CIFAR-100; 13 leaderboards with 645 pruned models; an online platform [V full] | No per-plan outcome table and no surrogate; only post-fine-tune accuracy reported, no pre-vs-post analysis; code "will be made publicly available" (future tense) [V full] |
| Network Pruning Spaces [7] | arXiv:2304.09453 (Apr 2023), preprint | Random width plans: n = 100 subnetworks of ResNet-50 on CIFAR-10, each fine-tuned 50 epochs; on ImageNet 300 candidates × 5 epochs, top 5 retrained 100 epochs. At medium ratios about 60 % of sampled plans beat the uniform plan and the best show no drop; above about 90 % FLOPs removed the spaces diverge [V full] | One network per study; no release stated; the conjectured limit function is not fitted [V full] |
| ShrinkBench, in Blalock et al. [4] | MLSys 2020; arXiv:2003.03033 | Open-source library (github.com/jjgo/shrinkbench) with magnitude, gradient-magnitude and random baselines; the paper reports more than 800 pruned networks [V full] | No released outcome dataset is described; no surrogate [V full] |
| Revisiting Random Channel Pruning [6] | CVPR 2022; arXiv:2205.05676 | Random per-layer ratios (100 samples, least-squares layer update, top 5 fine-tuned 5 epochs), proposed as "a strong baseline to benchmark other channel pruning methods"; VGG-16, ResNet-18/20/50/56, MobileNetV2 [V full] | No per-plan table released (code promised in the arXiv comment); no surrogate [V full] |

**Also checked.**
- NAS-Bench-101: 423k unique architectures and over 5 million trained models (ICML 2019; arXiv:1902.09635) [1, V abs]. NAS-Bench-201 (ICLR 2020; arXiv:2001.00326) [2, V API].
- Surrogate NAS benchmarks for spaces of up to 10^21 architectures (arXiv:2008.09777) [8, V abs]. NAS-Bench-x11, surrogates of full learning curves (NeurIPS 2021; arXiv:2111.03602) [9, V abs].
- GENNAPE: a performance predictor pretrained on NAS-Bench-101 and transferred to NAS-Bench-201/301 and the MobileNet and ResNet families (AAAI 2023; arXiv:2211.17226) [54, V abs]. Architectures trained from scratch, not pruned.
- Model Zoos: 27 zoos, 50,360 unique models and their sparsified counterparts (NeurIPS 2022 Datasets and Benchmarks; arXiv:2209.14764) [26, V abs]. Sparsified Model Zoo Twins: magnitude pruning and variational dropout over model zoos, released at modelzoos.cc (ICLR 2023 workshop; arXiv:2304.13718) [27, V abs].
- Once-for-All's accuracy predictor: 16K sub-networks of one supernet, RMSE 0.21 % (ICLR 2020; arXiv:1908.09791) [34, V full].

**What would still be new [H].**
- The first released table of (pretrained network, width plan, recovery length) → TEST accuracy for structured pruning: 100+ networks, about ten families, five datasets (brief), with the origin's own gain under the same recovery (the honest delta) and seeds.
- Multi-fidelity rows (cut, BN-recalibrated, 1, 20 and 100 epochs): the pruning analogue of NAS-Bench-x11 [9], and the data DB2 needs.
- A surrogate conditioned on the network graph and the plan, scored leave-one-family-out and by selection regret near the optimiser's picks. Rank correlation alone is not enough: PruneShift shows that rank agreement can approach one while normalised selection regret stays maximal [13, V abs].

**Hazards [H].**
- The brief's finding that allocation barely matters after a long recovery at mild ratios means the mild end of the table will be flat within seed noise. Publish the noise floor per cell and weight the release toward deep cuts.
- Plans from our own allocators cluster. A surrogate also needs space-filling random plans, as in [6, 7].
- One recovery recipe: results may not carry to sparsity-training recipes, the lever the brief names for deep cuts.

---

## 2. DB2. Proxy validity at scale

**Verdict: partly done.** Cut-time scores are a poor proxy for which channels to remove at fixed widths, at scale [12], and for one network's width plans without BN repair [10]. BN recalibration restores the correlation on that one network [10]. Short retraining is asserted to be a good proxy, qualitatively [12], or assumed [6, 7, 16]. For width plans, the evidence is one network for BN recalibration [10] and seven CIFAR-10 policies whose cut-time ranks survived fine-tuning [11]. No study ranks width plans across families and ratios against a long recovery with a ladder of proxies.

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| Is Oracle Pruning the True Oracle? [12] | TMLR 2026; arXiv:2412.00143 (v3, 25 Oct 2026) | About 37K models (LeNet5 to ResNets, VGG, ViT, an MLLM). Kendall τ between pre- and post-retraining performance: 0.02 (ResNet-56, CIFAR-10), 0.01 (VGG-19, CIFAR-100), 0.19 (ResNet-18, ImageNet), against its own validity bar of 0.2. Retraining 120 epochs (CIFAR) and 30 (ImageNet). "Only 10% retraining" correlates highly with full retraining (App. E, figure only, no τ) [V full] | Per-layer ratios are fixed per setting (uniform 50 % by default, Table A2), so it ranks masks, not width plans; no BN recalibration [V full] |
| EagleEye [10] | ECCV 2020; arXiv:2007.02491 | Width plans: random layer-wise rates with L1 filter choice (Sec. 3.4), MobileNet V1 on ImageNet, four FLOPs settings. Pearson ρ with fine-tuned accuracy: 0.68-0.82 for the adaptive-BN score, -0.04 to 0.25 for cut-only (Table 1). BN uses 1/30 of the training set, 100 iterations [V full] | One network. The number of candidates and the fine-tune length behind Table 1 are not stated (a 15-epoch fine-tune appears only in a figure caption). No short-fine-tune proxy is measured [V full] |
| AMC [11] | ECCV 2018; arXiv:1802.03494 | Reward = accuracy without fine-tuning, called "a good approximation for fine-tuned accuracy" (Sec. 3.2); seven CIFAR-10 policies (Plain-20, ResNet-56) keep their rank order after fine-tuning (Table 2). On ImageNet, accuracy after one-shot pruning without retraining "cannot provide useful supervision" (Sec. 4.2) [V full] | No correlation coefficient; no study across networks or ratios [V full] |
| PruneShift [13] | arXiv:2608.29765 (30 Aug 2026), preprint | Separates overall predictive fit, fit near the selector's picks, and decision quality; shows that Spearman and Kendall agreement can approach one while normalised selection regret stays maximal; experiments on TextbookQA, Natural Questions, QQP and an OSSCAR study on OPT-125M [V abs] | No CNN experiment named in the abstract; no proxy ladder [V abs] |
| NAS zero-cost proxies: Abdelfattah et al. [14]; NAS-Bench-Suite-Zero [15] | ICLR 2021, arXiv:2101.08134; NeurIPS 2022 D&B, arXiv:2210.03230 | Best zero-cost proxy: Spearman 0.82 against 0.61 for the EcoNAS reduced-training proxy on NAS-Bench-201 [14]; 13 proxies over 28 tasks with unified code and data [15] [V abs] | Architectures trained from scratch; no pruning, inherited weights or recovery [V abs] |

**Also checked.**
- NetAdapt (ECCV 2018; arXiv:1804.03230) [16, V full]: a short-term fine-tune of 10k-40k iterations inside the loop; without it accuracy "rapidly drops to nearly zero"; the long-term fine-tune adds 3.4 points on average (Fig. 9); no short-vs-long correlation analysis.
- Li et al. 2022 [6, V full]: "Directly evaluating the pruned network is not reliable" (Sec. 4.2); no correlation test.
- Frankle et al. (ICLR 2021; arXiv:2009.08576) [55, V abs]: pruning-at-initialisation scores survive random shuffling within layers, so their per-weight choices reduce to a per-layer fraction, an allocation.
- EcoNAS (CVPR 2020; arXiv:2001.01233) [17, V abs]: some reduced-training proxies preserve rankings better than others. White et al. (NeurIPS 2021; arXiv:2104.01177) [18, V abs]: 31 NAS performance predictors compared.
- LLM-KICK (ICLR 2024; arXiv:2310.01382) [19, V abs]: perplexity misses capability loss in compressed LLMs.

**What would still be new [H].**
- The unit is the width plan, not the mask: [12] holds layer ratios fixed, and [10] uses one network.
- A proxy ladder (cut-only, BN-recalibrated, 1-epoch, 20-epoch, data-free) on the same plans, against 100-epoch honest TEST, resolved by family and by ratio, with seed noise floors.
- Decision metrics beside τ: selection regret and top-k recall at the budget [13].
- The Goodhart case (brief): a learned agent that wins on cheap proxies and loses its edge after 100 epochs. That is a measured warning for every pruner rewarded at cut time (AMC-style [11]; RL-Pruner [40]).

---

## 3. DB3. Predicting prunability

**Verdict: partly done in pieces; the predictor itself is open.**
- The quantity is defined and measured [20].
- Within-family scaling laws exist for unstructured pruning [21]; for LLMs without recovery they transfer zero-shot to unseen models [22].
- Heavy-tail theory links SGD to compressibility [24], and heavy-tail metrics set LLM layer sparsities [25].
- Not found: a prediction of how far a CNN can be structurally pruned for at most X points after a fixed recovery, from cheap measurements, on unseen families.

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| Lost in Pruning [20] | MLSys 2021; arXiv:2103.03014 | "Prune potential" = the largest prune ratio within δ = 0.5 % of the unpruned test accuracy, measured by iterative prune-and-retrain with filter (FT, PFP) and weight (WT, SiPP) pruning, on CIFAR-10 networks and ImageNet ResNets; it falls to about 0 % under some corruptions; filter pruning has the lower potential [V full] | Measured per network by the full pipeline; no predictor from cheap measurements; no unseen-architecture test [V full] |
| On the Predictability of Pruning Across Scales [21] | ICML 2021 (DBLP key in the Semantic Scholar record); arXiv:2006.10621 | A functional form for the error of iteratively magnitude-pruned networks across width, depth and density, through one invariant, on CIFAR and ImageNet ResNets [V abs] | Unstructured; coefficients fitted per architecture and task; no cross-family prediction [V abs] |
| Pruning Laws for Large Language Models [22] | EMNLP 2026; arXiv:2504.04342 | Laws linking post-pruning performance to the unpruned model and the fraction removed; ten LLMs (1.3-30B) and a 20B MoE; unstructured, width and depth pruning; eight tasks; extrapolation error below 7 %; zero- and one-shot transfer to unseen models [V abs] | No recovery (post-pruning only); LLMs only [V abs] |
| Network Pruning Spaces [7] | arXiv:2304.09453 | Conjectures that each regime's performance limit "can be predicted by a functional form" of an optimal FLOPs-to-parameter ratio (about 1.0 in most regimes) [V full] | No fitted limit function; one network [V full] |
| Heavy-tail line: Barsbey et al. [24]; Martin, Peng, Mahoney [23]; AlphaPruning [25] | NeurIPS 2021, arXiv:2106.03795; Nature Communications 2021, arXiv:2002.06716; NeurIPS 2024, arXiv:2410.10912 | Heavy-tailed SGD plus overparametrisation gives ℓp-compressibility, in theory [24]; power-law metrics rank pretrained models without data [23]; HT-SR shape metrics set LLM layer sparsities, LLaMA-7B at 80 % with "reasonable perplexity" [25] [V abs] | None predicts how deep a network can be cut after recovery; the abstract of [25] covers LLMs only [V abs] |

**Also checked.** PQ Index, a compressibility measure that adapts pruning (ICLR 2023; arXiv:2302.05601) [28, V abs]. Kuhn et al., prunability as a predictor of generalisation, the reverse direction (arXiv:2103.06002) [31, V abs]. ℓq soft sparsity as a compressibility measure (arXiv:2206.05604) [32, V abs].

**What would still be new [H].**
- The target: prune potential after a fixed long recovery, for structured pruning, at a FLOPs or params budget, with honest deltas and a noise floor.
- Inputs that cost seconds to minutes: per-layer HT-SR α, PQI, cut-only and BN-recalibrated sensitivity curves, a 1-epoch probe, parameter and FLOP counts, dataset.
- Evaluation leave-one-family-out and leave-one-dataset-out, against a per-family Rosenfeld-style fit [21] and a family-median baseline.
- Risk: about 100 networks in about ten families is a small sample for a cross-family regressor. Report the prediction's error bars, not only a correlation.

---

## 4. DB4. Prune or pick

**Verdict: partly done.**
- At matched parameters, smaller networks trained from scratch beat pruned-and-tuned ones on WRN and DenseNet (CIFAR-10), and they run faster than pruned ResNet-34 on ImageNet [29].
- Width-scaled ResNets under a modern recipe beat published pruning numbers [30], and pruning "generally does not help as much as switching to a better architecture" [4].
- The other way: large-sparse beats small-dense in unstructured pruning [33], and in LLMs a pruned larger model beats same-size models trained from scratch [35, 46].
- Not found: a matched-FLOPs map over many families, with siblings under one recipe.

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| A Closer Look at Structured Pruning for Neural Network Compression, Crowley et al. [29] | arXiv:1810.04622 (v3, Jun 2019, retitled), preprint | Reduced networks (depth, width, bottleneck) against ℓ1- and Fisher-pruned ones, matched on params. WRN-40-2 on CIFAR-10 at 0.5M params: test error 9.14 (ℓ1) and 7.41 (Fisher) against 6.31-6.55 for reduced networks (Table 1). Pruned ResNet-34 against ResNet-18 and ResNet-9 on ImageNet; the reduced networks are faster on CPU and GPU (Table 2) [V full] | Matched on params, with MACs "for completeness"; two criteria; three families; a 2018 recipe [V full] |
| ThinResNet [30] | arXiv:2309.12854 (Sep 2023), preprint | Uniformly width-scaled ResNet-20/56 (CIFAR-10) and ResNet-50 (ImageNet) under the torchvision recipe "clearly and utterly outperform" about 20 and 18 published pruning results; for example ThinResNet-56 at 47 % of operations reaches 96.25 %, against a pruned reference's 93.26 % at 50 % (Tables 1-2) [V full] | The references are copied from papers with older recipes, so method and recipe are confounded; no matched budget; no pruning run under the modern recipe [V full] |
| Blalock et al. [4] | MLSys 2020; arXiv:2003.03033 | Fig. 1 (ImageNet): pruned models "sometimes outperform the original architecture, but rarely outperform a better architecture" [V full] | Numbers normalised from papers; no pruned EfficientNet; no controlled sibling runs [V full] |
| Rethinking the Value of Network Pruning [44] | ICLR 2019; arXiv:1810.05270 | Network Slimming against uniform channel scaling, both trained from scratch: for VGG the pruned pattern "can always beat the uniform ones"; for ResNet and DenseNet it is "not significantly more efficient" (Sec. 5) [V full] | Uniform scaling of the same depth, not a different-depth sibling; unpruned ResNet-20/32/56 appear beside pruned ResNet-56 but are not compared at matched cost (Table 10) [V full] |
| To prune, or not to prune [33] | arXiv:1710.01878 (2017); no venue on the record | Large-sparse models "consistently outperform small-dense models" at equal memory, for CNNs and LSTMs [V abs] | Unstructured sparsity; no FLOPs or latency match [V abs] |

**Also checked.** Sheared LLaMA: LLaMA2-7B pruned to 1.3B and 2.7B beats same-size open models at 3 % of the from-scratch compute (arXiv:2310.06694) [35, V abs]. Minitron: pruning plus distillation, up to 40× fewer training tokens per model than training from scratch (arXiv:2407.14679) [46, V abs]. Li et al. 2022: at similar FLOPs, plans with more parameters tend to have lower error [6, V full].

**What would still be new [H].**
- A matched-FLOPs (params beside) prune-or-pick map across the families that have siblings: ResNet depths, WRN widths and depths, MobileNetV2 widths, VGG variants; on CIFAR-10/100, SVHN, Fashion-MNIST and Tiny-ImageNet (brief).
- Siblings and pruned networks under one recipe and seeds, with honest deltas and each route's GPU cost: a pretrained sibling costs nothing to pick.
- The answer as a function of the ratio. The CNN evidence [29, 30] and the LLM evidence [35, 46] point in opposite directions; a controlled CNN map would locate the crossover.
- Caveat: the database's siblings were trained under their own recipes. The comparison is clean only once each sibling is checked under the recipe of its larger origin, or retrained.

---

## 5. DB5. A learned allocator trained on recovered outcomes

**Verdict: partly done per target; a generic version is open, with low expected value.** Each of the following exists, fitted to one target network: short-training rewards of 2-5 epochs [39, 47]; a GP over post-fine-tune error [36]; a per-backbone predictor fitted to about 800 fully trained compressed models [37]; an MLP surrogate of post-fine-tune accuracy used as the reward [38]. Not found: an allocator trained on long-recovery outcomes across families and then frozen.

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| HiReLC [38] | arXiv:2606.26002 (24 Jun 2026), preprint | Hierarchical RL for pruning plus quantisation. Once the buffer holds at least 3 entries, an MLP surrogate fitted to "ground-truth post-fine-tuning accuracy" replaces a logit-MSE proxy for reward shaping; warm-up with 100 random configurations fine-tuned one epoch each (ResNet-18 and MobileNetV2, Tiny-ImageNet); surrogate MAE 15.20 % and 10.37 % (Table 4) [V full, first 100k of 141k characters] | Per target network; no rank correlation; label fine-tune length unclear; no transfer [V full] |
| SeerNet [37] | IJCV (arXiv comment); arXiv:2304.06393 | A predictor of compressed-model accuracy (pruning plus quantisation) fitted to about 800 policies per backbone and dataset (50 per round × 16 rounds), each "trained exhaustively"; the policy is optimised by gradient on the predictor; 0.43-0.58 GPU-h per CIFAR-10 label [V full, first 100k of 134k characters] | Per backbone; other architectures left to future work (Sec. 5); MSE only [V full] |
| Fine-Pruning [36] | BMVC 2017 (oral); arXiv:1707.09102 | Gaussian-process Bayesian optimisation over pruning hyper-parameters; objective = error after pruning and fine-tuning minus λ·sparsity; up to 50 candidates per iteration [V full] | Weight-level pruning; AlexNet only; per target [V full] |
| N2N Learning [39] | ICLR 2018 (iclr.cc listing); arXiv:1709.06030 | Reward after training the student 5 epochs by distillation ("5 epochs was sufficient to compare performance", Sec. 4.2); policies transferred ResNet-18 ↔ ResNet-34 and VGG-11 → VGG-19 (Table 5) [V full] | Short training, not a long recovery; transfer only within a family, compared over the first 10 policy iterations [V full] |
| AMC [11] | ECCV 2018; arXiv:1802.03494 | Cut-time reward; the authors speculate that "if reward were fine-tuned accuracy, the agent should compress more aggressively" (Sec. 4.1) [V full] | Not tested [V full] |

**Also checked.**
- ABCPruner: fitness after "two epochs" of training (IJCAI 2020; arXiv:2001.08565) [47, V full].
- Jointly training and pruning: the reward is the accuracy of a network whose weights keep training (CVPR 2024; arXiv:2403.19490) [48, V abs].
- Mu et al.: history from other networks reused by retrieval; XGBoost and matrix-factorisation predictors judged "non-practical due to the huge data requirement" (IEEE TCAD, doi:10.1109/TCAD.2023.3319594; arXiv:2107.08815) [49, V full].
- RL-Pruner: the reward includes test accuracy and appears to be scored before post-training (arXiv:2411.06463) [40, V full].
- Once-for-All's predictor inside one supernet: RMSE 0.21 % [34, V full]. MetaPruning: candidates scored with PruningNet weights, no fine-tuning (ICCV 2019; arXiv:1903.10258) [50, V abs].
- A generalist RL compression agent trained on two LeNets (IEEE Access 2024; doi:10.1109/ACCESS.2024.3385857) [51, V abs].

**What would still be new [H].** One policy trained across families against a surrogate of 100-epoch honest outcomes (DB1's surrogate), then frozen and scored on unseen families under the same recovery.

**Why it ranks last [H].**
- The brief's own findings predict a flat target: at mild ratios allocation barely matters after 100 epochs, and at deep cuts the gap is the recipe.
- To help, a surrogate must rank plans within seed noise. The per-target surrogates report MAE of 10-15 points [38]; the one predictor at 0.21 % RMSE works inside a single supernet [34].
- The agent's value would rest on the deep-cut cells, where DB1 must first show a spread above noise.

---

## 6. DB7. The fine-tune confound

**Verdict: partly done.**
- The confound is named [4]. The fine-tune learning rate moves results by tens of points [43]. Large-LR retraining makes random pruning competitive [41]. LR rewinding often ends above the unpruned accuracy [42]. None of these four runs the unpruned network under the same retraining, in the text read.
- Jin et al. run exactly that control, "extended dense training", and find it matches or beats pruning without label noise, but for unstructured magnitude pruning and through a generalisation lens [45].

| Work | Venue, ID | Did | Did not |
|---|---|---|---|
| Pruning's Effect on Generalization Through the Lens of Training and Regularization [45] | NeurIPS 2022; arXiv:2210.13738 | The dense network trained with the pruning algorithm's epochs and LR schedule "matches or exceeds" the optimally sparse model without label noise. Test error, dense control vs LR rewinding: 29.0 vs 29.9 (ResNet-32, CIFAR-100), 7.6 vs 7.7 (ResNet-20, CIFAR-10), 6.0 vs 5.9 (VGG-16, CIFAR-10), 23.4 vs 23.4 (ResNet-50, ImageNet) (App. C, Table 2) [V full, first 100k of 134k characters] | Individual weights removed (unstructured); no structured pruning or post-hoc fine-tune recipes; framed as generalisation, not as an audit of compression claims [V full] |
| Why is the State of Neural Network Pruning so Confusing? [43] | arXiv:2301.05219 (2023), preprint | Comparison setups S1-S4; the fine-tune LR schedule is the factor most papers ignore. ResNet-34 on ImageNet-100 at 95 % pruning, P90F30: 75.20 at LR 1e-2 against 29.89 at 1e-4. Recommends setup S3.2 or stricter (Sec. 6) [V full] | No dense control under the same fine-tune: pruned settings reach 85.27-85.48 against a single dense figure of 84.56 (Table 4), and the dense model is not tested under the same LR [V full] |
| Network Pruning That Matters [41] | ICLR 2021; arXiv:2105.03193 | Large-LR retraining (LRW, SLR, CLR) beats fine-tuning; "Random Pruning + CLR could surpass" sophisticated criteria (Sec. 4.2), against numbers taken from the original papers [V full] | No dense network retrained with the same schedule; the closest is a from-scratch dense schedule check (App. G) [V full] |
| Comparing Rewinding and Fine-tuning in Neural Network Pruning [42] | ICLR 2020; arXiv:2003.02389 | LR rewinding "typically results in higher accuracy than the unpruned network"; structured pruning shows the same trends; ResNet-20/56/110 on CIFAR-10, ResNet-34/50 on ImageNet [V full] | No dense retraining control [V full] |
| What is the State of Neural Network Pruning? [4] | MLSys 2020; arXiv:2003.03033 | Lists "Pruning and fine-tuning schedule" among the confounders (Sec. 4.5); asks for a control, "usually the original model before pruning" (Sec. 6) [V full] | The control asked for is the unpruned model before fine-tuning, not the unpruned model fine-tuned the same way [V full] |

**Also checked.**
- ThinResNet [30, V full]: the modern recipe alone takes unpruned ResNet-50 from 76.15 % to 80.46 % top-1 (Tables 1 and 3), a gain of 4.3 points with no pruning.
- Liu et al.'s Scratch-B trains the pruned architecture on a matched budget, not the dense one [44, V full].
- "When to Prune?": an equal-epoch control for a network pruned during training from scratch, no dense baseline (CVPR 2022; arXiv:2110.12007) [52, V full].
- Zimmer et al., retraining budgets (ICLR 2023; arXiv:2111.00843) [53, R: `LIT_ALTERNATIVES_21NOV.md` ref. 7].

**What would still be new [H].**
- The structured-pruning audit: for each family and each published recovery recipe (and ours), the unpruned network's own gain under that recipe, and the share of reported Δacc it absorbs, by ratio.
- Published "Δacc ≥ 0" rows at mild ratios re-scored under the honest delta.
- The honest delta as a reporting standard: [4] and [43] propose comparison setups but not this control.
- Cheap and near-certain to give a result, since the harness already measures the origin's gain (brief).

---

## 7. Ranking: novelty remaining × community value [H]

1. **DB1 (benchmark and surrogate).** Highest novelty remaining: no pruning analogue of the NAS benchmarks was found, and those benchmarks were widely adopted. It also supplies the data for DB2, DB3, DB4 and DB7. Its value depends on publishing noise floors and the deep-cut cells.
2. **DB2 (proxy validity).** Cut-only failure is now known at scale for masks [12], but the width-plan ladder (BN, 1, 20 epochs) against 100-epoch truth across families is not. The answer bears directly on every pruner rewarded at cut time, and the learned-agent Goodhart case is a distinctive result.
3. **DB7 (fine-tune confound).** The dense control is absent from the four main structured-pruning critiques [4, 41-43] and present only for unstructured pruning [45]. High value to the community, cheap and certain.
4. **DB3 (prunability).** The cross-family, post-recovery predictor is open. Its value is moderate, limited by the small number of networks for a cross-family fit.
5. **DB4 (prune or pick).** Valuable to practitioners, but the core answer exists at matched params [29] and under modern recipes [30]. What is left is the matched-FLOPs, many-family, one-recipe map.
6. **DB5 (allocator on recovered outcomes).** Per-target surrogates exist [36-38]. The generic version is new, but the brief's findings predict little to learn at mild ratios after a long recovery.

---

## 8. Search log and limits

- **Searches.** 21 web searches: benchmarks and surrogates for pruning; tabular pruning benchmarks; sparsity-allocation datasets; model-zoo datasets; proxy correlations in pruning, including short fine-tune and BN proxies (two); prunability prediction; learned rewards and surrogates; predictors across architectures; fine-tune controls (two); zero-cost proxies for widths; PruneShift; Network Pruning Spaces; LeGR; Fine-Pruning; the EcoNAS and NAS-Bench-x11 IDs (two); venue look-ups (three).
- **Reads.** 84 fetch calls, about 70 successful page reads: arXiv abstract pages; one arXiv API batch of 17 records (comments and journal refs); arXiv HTML and ar5iv full texts; three Semantic Scholar records (five more were rate-limited, HTTP 429). dblp refused all six automated requests.
- **Reading model.** Full texts came through a page-summarising model, so numbers quoted from them rest on that single reading; they were not cross-checked against PDFs. Five long pages were read only to 100k characters, as marked: HiReLC, SeerNet, Jin et al., Wang et al. (missing its last 3k characters) and Lost in Pruning.
- **Coverage.** IEEE, Elsevier and Springer venues without arXiv copies are under-covered. "Not found" is a statement about these searches, not proof of absence.
- **Venues** come from arXiv comments or journal refs, except AMC (CVF open-access listing, via search), Barsbey et al. (NeurIPS 2021 proceedings, via search), N2N (iclr.cc listing, via search), Rosenfeld et al. (DBLP key in the Semantic Scholar record), "When to Prune?" (CVF open-access URL, via search) and Mu et al. (Semantic Scholar record). Zhu & Gupta, Wang et al. 2023, Sheared LLaMA, Minitron, PruningBench, Network Pruning Spaces, ThinResNet, Crowley et al., PruneShift and HiReLC carry no venue on their arXiv records and are cited as preprints.
- **ID corrections.** My first guesses for the EcoNAS and NAS-Bench-x11 IDs opened unrelated papers. The IDs in §9 are the ones their abstract pages confirm.

---

## 9. References

**Benchmarks and surrogates**
1. Ying, Klein, Real, Christiansen, Murphy, Hutter. NAS-Bench-101: Towards Reproducible Neural Architecture Search. ICML 2019. arXiv:1902.09635. [V abs]
2. Dong, Yang. NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search. ICLR 2020. arXiv:2001.00326. [V API]
3. Dong, Liu, Musial, Gabrys. NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size. IEEE TPAMI 2021. arXiv:2009.00437. [V full]
4. Blalock, Gonzalez Ortiz, Frankle, Guttag. What is the State of Neural Network Pruning? MLSys 2020. arXiv:2003.03033. [V full]
5. Li, Li, Xue, Fang, Zhou, Feng, Wang, Song, Song. A Comprehensive Study of Structural Pruning for Vision Models (PruningBench). arXiv:2406.12315 (v7). [V full]
6. Li et al. (6 authors). Revisiting Random Channel Pruning for Neural Network Compression. CVPR 2022. arXiv:2205.05676. [V full]
7. He, Yang, Song, Pu, Hu, Jiang, Zhang, Ding. Network Pruning Spaces. arXiv:2304.09453. [V full]
8. Zela, Siems, Zimmer, Lukasik, Keuper, Hutter. Surrogate NAS Benchmarks: Going Beyond the Limited Search Spaces of Tabular NAS Benchmarks. arXiv:2008.09777 (v4, 2022). [V abs]
9. Yan, White, Savani, Hutter. NAS-Bench-x11 and the Power of Learning Curves. NeurIPS 2021. arXiv:2111.03602. [V abs]
26. Schürholt, Taskiran, Knyazev, Giró-i-Nieto, Borth. Model Zoos: A Dataset of Diverse Populations of Neural Network Models. NeurIPS 2022 Datasets and Benchmarks. arXiv:2209.14764. [V abs]
27. Honegger, Schürholt, Borth. Sparsified Model Zoo Twins. ICLR 2023 Workshop on Sparsity in Neural Networks. arXiv:2304.13718. [V abs]
34. Cai, Gan, Wang, Zhang, Han. Once-for-All: Train One Network and Specialize it for Efficient Deployment. ICLR 2020. arXiv:1908.09791. [V full]
54. Mills et al. GENNAPE: Towards Generalized Neural Architecture Performance Estimators. AAAI 2023. arXiv:2211.17226. [V abs]

**Proxies**
10. Li, Wu, Su, Wang, Lin. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV 2020. arXiv:2007.02491. [V full]
11. He, Lin, Liu, Wang, Li, Han. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018. arXiv:1802.03494. [V full]
12. Feng, Tao, Wang. Is Oracle Pruning the True Oracle? TMLR 2026. arXiv:2412.00143 (v3). [V full]
13. Ye, Zhang. PruneShift: A Framework for Evaluating Decision Reliability in Structured Pruning. arXiv:2608.29765. [V abs]
14. Abdelfattah, Mehrotra, Dudziak, Lane. Zero-Cost Proxies for Lightweight NAS. ICLR 2021. arXiv:2101.08134. [V abs]
15. Krishnakumar, White, Zela, Tu, Safari, Hutter. NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies. NeurIPS 2022 Datasets and Benchmarks. arXiv:2210.03230. [V abs]
16. Yang et al. NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications. ECCV 2018. arXiv:1804.03230. [V full]
17. Zhou, Zhou, Zhang, Loy, Yi, Zhang, Ouyang. EcoNAS: Finding Proxies for Economical Neural Architecture Search. CVPR 2020. arXiv:2001.01233. [V abs]
18. White, Zela, Ru, Liu, Hutter. How Powerful are Performance Predictors in Neural Architecture Search? NeurIPS 2021. arXiv:2104.01177. [V abs]
19. Jaiswal, Gan, Du, Zhang, Wang, Yang. Compressing LLMs: The Truth is Rarely Pure and Never Simple (LLM-KICK). ICLR 2024. arXiv:2310.01382. [V abs]
55. Frankle, Dziugaite, Roy, Carbin. Pruning Neural Networks at Initialization: Why are We Missing the Mark? ICLR 2021. arXiv:2009.08576. [V abs]

**Prunability**
20. Liebenwein et al. (5 authors). Lost in Pruning: The Effects of Pruning Neural Networks beyond Test Accuracy. MLSys 2021. arXiv:2103.03014. [V full, first 100k of 135k characters]
21. Rosenfeld, Frankle, Carbin, Shavit. On the Predictability of Pruning Across Scales. ICML 2021. arXiv:2006.10621. [V abs; venue from the Semantic Scholar record]
22. Sengupta, Chaudhary, Chakraborty. Pruning Laws for Large Language Models. EMNLP 2026. arXiv:2504.04342. [V abs]
23. Martin, Peng, Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications 2021. doi:10.1038/s41467-021-24025-8; arXiv:2002.06716. [V abs]
24. Barsbey, Sefidgaran, Erdogdu, Richard, Şimşekli. Heavy Tails in SGD and Compressibility of Overparametrized Neural Networks. NeurIPS 2021. arXiv:2106.03795. [V abs; venue via the proceedings listing]
25. Lu, Zhou, Liu, Wang, Mahoney, Yang. AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved Layer-wise Pruning of Large Language Models. NeurIPS 2024. arXiv:2410.10912. [V abs]
28. Diao, Wang, Zhan, Yang, Ding, Tarokh. Pruning Deep Neural Networks from a Sparsity Perspective (PQ Index). ICLR 2023. arXiv:2302.05601. [V abs]
31. Kuhn, Lyle, Gomez, Rothfuss, Gal. Robustness to Pruning Predicts Generalization in Deep Neural Networks. arXiv:2103.06002. [V abs]
32. Yang, Wang, Ding, Yang. A Theoretical Understanding of Neural Network Compression from Sparse Linear Approximation. arXiv:2206.05604. [V abs]

**Prune or pick**
29. Crowley, Turner, Storkey, O'Boyle. A Closer Look at Structured Pruning for Neural Network Compression. arXiv:1810.04622 (v3). [V full]
30. Tessier, Boukli Hacene, Gripon. ThinResNet: A New Baseline for Structured Convolutional Networks Pruning. arXiv:2309.12854. [V full]
33. Zhu, Gupta. To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv:1710.01878. [V abs]
35. Xia, Gao, Zeng, Chen. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. arXiv:2310.06694. [V abs]
46. Muralidharan et al. Compact Language Models via Pruning and Knowledge Distillation (Minitron). arXiv:2407.14679. [V abs]

**Learned rewards and surrogates**
36. Tung, Muralidharan, Mori. Fine-Pruning: Joint Fine-Tuning and Compression of a Convolutional Network with Bayesian Optimization. BMVC 2017. arXiv:1707.09102. [V full]
37. Wang, Lu, Xiao, Liu, Zhou. Learning Accurate Performance Predictors for Ultrafast Automated Model Compression (SeerNet). IJCV. arXiv:2304.06393. [V full, first 100k of 134k characters]
38. Baghdadi, Belhamidi, Belhadj, Boulmerka, Farhi. Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization. arXiv:2606.26002. [V full, first 100k of 141k characters]
39. Ashok, Rhinehart, Beainy, Kitani. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. ICLR 2018. arXiv:1709.06030. [V full; venue via the iclr.cc listing]
40. Wang, Kindratenko. RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration. arXiv:2411.06463. [V full]
47. Lin, Ji, Zhang, Zhang, Wu, Tian. Channel Pruning via Automatic Structure Search (ABCPruner). IJCAI 2020. arXiv:2001.08565. [V full]
48. Ganjdanesh, Gao, Huang. Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment. CVPR 2024. arXiv:2403.19490. [V abs]
49. Mu, Wang, Zhu, Yang, Lin, Zhang. Boosting the Convergence of Reinforcement Learning-Based Auto-Pruning Using Historical Data. IEEE TCAD. doi:10.1109/TCAD.2023.3319594; arXiv:2107.08815. [V full]
50. Liu, Mu, Zhang, Guo, Yang, Cheng, Sun. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. ICCV 2019. arXiv:1903.10258. [V abs]
51. Gonzalez-Sahagun, Conant-Pablos, Ortíz-Bayliss, Cruz-Duarte. A Generalist Reinforcement Learning Agent for Compressing Convolutional Neural Networks. IEEE Access 2024. doi:10.1109/ACCESS.2024.3385857. [V abs, Semantic Scholar record]

**Fine-tune confound**
41. Le, Hua. Network Pruning That Matters: A Case Study on Retraining Variants. ICLR 2021. arXiv:2105.03193. [V full]
42. Renda, Frankle, Carbin. Comparing Rewinding and Fine-tuning in Neural Network Pruning. ICLR 2020. arXiv:2003.02389. [V full]
43. Wang, Qin, Bai, Fu. Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning. arXiv:2301.05219. [V full, missing the last 3k of 103k characters]
44. Liu, Sun, Zhou, Huang, Darrell. Rethinking the Value of Network Pruning. ICLR 2019. arXiv:1810.05270. [V full]
45. Jin, Carbin, Roy, Frankle, Dziugaite. Pruning's Effect on Generalization Through the Lens of Training and Regularization. NeurIPS 2022. arXiv:2210.13738. [V full, first 100k of 134k characters]
52. Shen et al. When to Prune? A Policy towards Early Structural Pruning. CVPR 2022. arXiv:2110.12007. [V full; authors not re-read]
53. Zimmer, Spiegel, Pokutta. How I Learned to Stop Worrying and Love Retraining. ICLR 2023. arXiv:2111.00843. [R: `LIT_ALTERNATIVES_21NOV.md` ref. 7]
