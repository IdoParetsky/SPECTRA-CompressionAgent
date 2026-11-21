# Pruning that takes an accuracy-drop tolerance as its input (21 Nov 2026)

The spectra-literature subagent compiled this on 21 Nov 2026 for SPECTRA's accuracy-limit mode (ledger §423). The scan was read-only, and this is the only file written.

**How the sources were read.** PDFs (arXiv, NeurIPS proceedings) were converted with `pdftotext -layout` and read as text. Documentation pages, source files, arXiv HTML pages and PMC pages were read through the web-fetch page reader, which relays quotes; those carry **(f)**. The NEON PDF (`docs/NEON___Information_Sciences_No_highlights (4).pdf`) was read only at the pages cited; page numbers are the preprint's "Page N of 22" footers, and "l." is its line numbering.

**Labels.**
- [V]: read at the source this session.
- (f): read through the web-fetch reader.
- [R]: seen reported elsewhere (a search snippet, the brief, another paper) and not read at the source.
- [c]: my arithmetic.
- [H]: my interpretation.
- [U]: unresolved.

---

## Verdict

1. **Yes.** Several CNN pruning papers and one widely used tool take an accuracy-drop tolerance and return the most compressed model that meets it, so the input itself is not new. NEON's sentence that it is "the first pruning approach to offer this capability" (p. 4, ll. 157–158) does not hold against its own reference [13], Huang et al. (WACV 2018). That paper puts a user-set accuracy-drop bound *b* into an RL filter-pruning reward on VGG-16 and ResNet-18 [V].
2. **Closest tool: OpenVINO NNCF's accuracy-aware `adaptive_compression_level` loop** (since NNCF v2.0). In: the maximal relative or absolute accuracy degradation. Search: the filter-pruning (or sparsity) rate moves in steps that halve at each change of direction, with fine-tuning at each level. Out: "the model with the highest compression rate encountered, which satisfies the accuracy drop criterion" [V(f)].
3. **Other tools.** PaddleSlim has a per-layer version (`get_ratios_by_loss`). Intel Neural Compressor, AIMET and NNCF's post-training quantization offer an accuracy criterion for quantization only. NNI, Torch-Pruning and TF-MOT take budgets only [V(f)].
4. **Closest papers:** Condensa (IEEE Micro 2020; constrained Bayesian optimisation for "the highest sparsity s_acc that loses at most ε accuracy") [V]; Zhao et al. (arXiv 2201.10520; accuracy-guaranteed adaptive structured pruning) [V]; MetaML-Pro (ACM TRETS; binary search for the highest pruning rate within α_p) [V(f)]. With a statistical guarantee: Pareto Testing (BERT) [V] and Alvarez 2024 (unstructured) [V(f)].
5. **None of them serves the tolerance with a frozen, architecture-generic policy.** Every CNN method above either searches per network (bisection, Bayesian optimisation, rollback, each with fine-tuning) or trains an agent per network. NEON is the only generic one: dense networks on tabular data, and one trained configuration per tolerance [V p. 11, H].
6. **SPECTRA §423 uses the NNCF / MetaML-Pro mechanism** (five 20-epoch bisection probes per network), and its agent came out SAME as the rule. Novelty risk for the mode alone is high (see the last section).

---

## Table

Group A: the user's input is an accuracy tolerance (papers).

| # | Work | Venue / ID | User input | Constraint type | Mechanism | Structured? | Networks shown (key numbers) | Generic, no per-network search? | Label |
|---|---|---|---|---|---|---|---|---|---|
| A1 | NEON (Hirsch & Katz) | Inf. Sci. 2022, DOI 10.1016/j.ins.2022.07.134 | C, the "permissible" reduction in performance (p. 8, l. 302). Configurations C ∈ {0, 1, 5, 50} % (p. 11, ll. 380–382) | Soft, reward term (Eq. 4): R = c_t if −C ≤ acc_d ≤ 0; −c_t³ if acc_d < −C; c_t³ if acc_d > 0 | REINFORCE agent; four passes over the layers; action = layer size ratio {1, 0.9, 0.8, 0.7, 0.6} (p. 8) | Yes (neurons of dense layers; "we focus on structured pruning", p. 3, l. 127) | Random dense nets on 28 tabular datasets, 5-fold over datasets. Within the bound on 82–100 % of datasets (p. 2) | Yes, for unseen nets and datasets (offline training). But C is fixed per trained configuration [H] | [V] |
| A2 | Huang, Zhou, You, Neumann, "Learning to prune filters in convolutional neural networks" | WACV 2018, pp. 709–718 (NEON ref [13]); arXiv 1801.07365 | Drop bound *b*, "a hyper-parameter used to control the trade-off"; b ∈ {0.5, 1, 2, 4} | Soft, reward term: R = ψ·φ (Eq. 1), ψ = (b − (p* − p̂))/b (Eq. 2), φ = log(N_l / C(A_l)) (Eq. 4). ψ "contributes negatively" once the drop exceeds b; computed on a hold-out val set | One REINFORCE agent per layer, trained on the target net, layers low to high with a fine-tune after each (Alg. 1–2) | Yes (filters) | VGG-16 CIFAR-10: test drop 0.6 / 1.1 / 1.9 / 3.4 at b = 0.5 / 1 / 2 / 4, prune ratio 83.3–92.8 % (Table 2). ResNet-18: 0.3 / 1.0 / 1.7 / 2.9 (Table 3). Also FCN-32s VOC, SegNet CamVid (b = 2) | No | [V] |
| A3 | AMC, "accuracy-guaranteed" protocol (He et al.) | ECCV 2018 (NEON ref [14]); arXiv 1802.03494 | None; the goal is "the smallest model with no loss of accuracy" (§1) | Reward shaping: R_FLOPs = −Error·log(FLOPs), R_Param = −Error·log(#Param) (Eqs. 5–6); val accuracy without fine-tuning | DDPG agent per network | Shown only fine-grained (unstructured) | ResNet-50 CIFAR-10: 60 % params, val 93.64 / test 93.55 against 93.53 (Table 2). The ImageNet "no loss" ResNet-50 runs used fixed densities [50, 35, 25, 20] % per stage, i.e. budgets (§4.2) | No | [V] |
| A4 | Condensa (Joseph, Muralidharan, Garg, Garland, Gopalakrishnan) | IEEE Micro 40(5), 2020, pp. 17–25; arXiv 1911.02497 | ε, "Accuracy threshold" (`eps=0.02` in the code listing), plus an objective such as throughput | Hard, on the search: level set L = Acc(w) − ε | Two-stage constrained BO. DR-UCB finds s_acc, "the highest sparsity … that loses at most ε" (it assumes accuracy falls monotonically with sparsity). UCB/LCB then optimises the objective in (0, s_acc). Each sample is recovered with L-C; 9:1 val split on CIFAR-10; at most 10 samples per search | Both (filter pruning; unstructured P+Q) | VGG19-BN, ResNet-56 CIFAR-10; VGG16-BN ImageNet; LSTM WikiText-2; level set "2% below baseline in all experiments". VGG-19 filter: sparsity 0.79, 2.59× throughput, +0.36 % top-1 | A tool for any model, but a per-network search | [V] |
| A5 | Zhao, Jain, Zhao, "Adaptive Activation-based Structured Pruning" (accuracy-guaranteed policy) | arXiv 2201.10520 (v3, Mar 2023); overlaps AAP, arXiv 2303.08595 | "Target Accuracy Loss" (Alg. 2 input), for "the most resource-efficient model given the acceptable accuracy loss" | Hard: each round is accepted or rejected | Per round: raise the attention threshold T by λ (start 0.005), train, evaluate. On an unacceptable round, roll back to the last acceptable one and halve λ. Stop when model size changes < 0.1 % for several rounds | Yes (filters) | CIFAR-10 ResNet-56/50, VGG-16/19, MobileNetV2; ImageNet ResNet-50; Tiny-ImageNet VGG-19. ResNet-56: 79.11 % params removed "without any accuracy drop" | Per-network search | [V]. Split used for the loop's accuracy [U] |
| A6 | MetaML-Pro auto-pruning (Que, Coutinho, Guo, Fan, Luk) | ACM TRETS (accepted, per arXiv comments); arXiv 2502.05850 | α_p (`tolerate_acc_loss`); β_p, the stop resolution | Hard (Eq. 1): maximise the pruning rate subject to accuracy loss ≤ α_p | Binary search on the rate, 1 + log₂(1/β_p) steps; pruning during training; FPGA/HLS design flow | [U] (the text says it zeroes weights, and one section says filters) | Jet-DNN 93.8 %; LSTM seq-MNIST 71.9 % at a 2 % tolerance; ResNet-9 SVHN; VGG-7; GNN | Per-network search | [V(f)] |
| A7 | Tian, Yamagiwa, Wada, "Heuristic Method for Minimizing Model Size of CNN by Combining Multiple Pruning Techniques" | Sensors 22(15):5874, 2022, DOI 10.3390/s22155874 | Absolute target accuracy (85–95 %), step width, margin | Hard, stopping rule | Network slimming until accuracy falls below the target, pull back by a margin, then deep-compression weight pruning until below the target; repeat with smaller margins and keep the smallest | Both (channels, then weights) | CIFAR-10 VGG-19, ResNet-110, DenseNet-40/121/202; VGG up to 13× smaller; 0.5 h against 7.5 h brute force | Per-network search | [V(f)] |
| A8 | Vu-Van, Du Thanh, Ho, Vu, "Pruning Deep Convolutional Neural Network Using Conditional Mutual Information" | arXiv 2411.18578 | δ, "the targeted maximum reduction in accuracy" (a_f − a_p ≤ δ) | Hard, per layer | Per layer: K = 3 Scree-test cutoffs; keep the most pruned one that meets δ, evaluated on training data without fine-tuning; one 100-epoch retrain at the end | Yes (filters) | VGG-16 CIFAR-10 only: 36.15 % of filters removed, 94.00 → 93.68 | Per network | [V(f)] |
| A9 | PuRL (Gupta et al.) | ICML 2020 AutoML workshop; arXiv 2007.04756 | Target accuracy T_A **and** target sparsity T_P | Saturating penalty R = −β(max(1 − A/T_A, 0) + max(1 − P/T_P, 0)) (Eq. 2); A is "the test accuracy" | RL per network, with dense rewards | No (per-layer magnitude threshold) | ResNet-50 CIFAR-10 (T_A 95 %, T_P 60 %); ResNet-50 ImageNet; MobileNetV2 | No | [V] |

Group B: tools.

| # | Tool | Documentation | User input | Constraint type | Mechanism | Pruning covered? | Shown | Generic, no per-network search? | Label |
|---|---|---|---|---|---|---|---|---|---|
| B1 | OpenVINO NNCF, accuracy-aware training, mode `adaptive_compression_level` | `docs/accuracy_aware_model_training/AdaptiveCompressionLevelTraining.md` at tags v2.1.0 and v2.7.0 (github.com/openvinotoolkit/nncf). The v2.0 release note reads: "Accuracy-aware training available for filter pruning and sparsity" | `maximal_relative_accuracy_degradation` (%, default 1.0) or `maximal_absolute_accuracy_degradation` | Hard, applied at each level | Step search on one scalar compression rate inside one training run: initial step 0.1, × 0.5 at each change of direction, `patience_epochs` (3) per level, stop at `minimal_compression_rate_step` 0.025. Returns the most compressed model that meets the criterion. One adaptive algorithm at a time (filter pruning or magnitude sparsity, optionally with quantization) | **Yes, filter pruning** (and magnitude sparsity) | ModelZoo v2.7.0: ResNet-50 ImageNet, 52.5 % filter pruning, 75.23 (drop 0.93); ResNet-18, 60 %, 69.2 (−0.6 in the drop column) | Any model NNCF can prune; a per-network search | [V(f)]. The current develop docs list only unstructured `nncf.prune(ratio=…)` and no accuracy-aware mode [V(f)] |
| B2 | PaddleSlim `get_ratios_by_loss(sensitivities, loss)` | `paddleslim/prune/sensitive.py` (develop, github.com/PaddlePaddle/PaddleSlim) | `loss`, "The threshold of accuracy loss"; relative, (baseline − pruned)/baseline | Per-layer bound | "Get the max ratio of each parameter. The loss of accuracy must be less than given `loss` when the single parameter was pruned by the max ratio." A one-shot sensitivity scan per layer, no fine-tuning, linear interpolation between scan points | Yes (conv filters, `l1_norm`) | – | A per-network scan; the joint drop is not bounded | [V(f)] |
| B3 | Intel Neural Compressor | Tuning strategies (v2.0); pruning (v2.1), intel.github.io/neural-compressor | `AccuracyCriterion`: `criterion` relative/absolute, `tolerable_loss` (0.01 in the example) | Stop rule for quantization tuning: "till a quantization configuration meets the pre-defined accuracy criterion" | Pruning takes `target_sparsity`, with channel-wise, block and N:M patterns; the pruning page has no accuracy criterion | **Quantization only** | – | – | [V(f)] |
| B4 | Qualcomm AIMET AutoQuant | User guide 1.21.0 (quic.github.io/aimet-pages) | "the amount of accuracy drop that can be tolerated" | Stops once met | BN folding, cross-layer equalisation, AdaRound; the page never mentions pruning | **Quantization only** | – | – | [V(f)] |
| B5 | Microsoft NNI | SensitivityPruner source (v1.9); pruner list (v2.10); AMC pruner source (v2.10) | Target sparsity in `config_list`. `acc_drop_threshold` (default 0.05) only shapes the per-layer proportions. AMC: `total_sparsity`, `target` 'flops' or 'params' | Budget | The SensitivityPruner loop runs `while cur_ratio > target_ratio`; the AMC reward is "score − 1"; no v2.10 pruner takes an accuracy target | Yes (L1/L2 filter and others) | – | – | [V(f)] |
| B6 | Torch-Pruning | README (master, github.com/VainF/Torch-Pruning) | `pruning_ratio`, `global_pruning`, `pruning_ratio_dict` | Budget | No accuracy-target option is documented | Yes | – | – | [V(f)] |
| B7 | TensorFlow Model Optimization | Pruning comprehensive guide (tensorflow.org/model_optimization) | The schedule's target sparsity (`ConstantSparsity`, `PolynomialDecay`), `block_size` | Budget | "Consider the pruning schedule to be a hyperparameter" | Weights and blocks | – | – | [V(f)] |
| B8 | Intel Distiller, AMC | github.com/IntelLabs/distiller (HTTP 404 on 21 Nov 2026; the docs pages 404 too) | `--amc-protocol` "mac-constrained" or "accuracy-guaranteed" ("punish-agent" also appears in a README diff) | – | In issue #64 a maintainer reportedly found the paper's accuracy-guaranteed reward "not so good", with solutions that push every layer to maximum density | Channel pruning | – | – | [R] (search snippets only) |

Group C: tolerance with a statistical guarantee (risk control).

| # | Work | Venue / ID | User input | Constraint type | Mechanism | Structured? | Networks shown | Generic, no per-network search? | Label |
|---|---|---|---|---|---|---|---|---|---|
| C1 | Pareto Testing (Laufer-Goldshtein, Fisch, Barzilay, Jaakkola) | ICLR 2023 [R]; arXiv 2210.07913 | Risk bound α on the relative accuracy reduction against the full model, Q_acc = E[1{f(X;τ0)=Y}] − E[1{f(X;τ)=Y}] (Eq. 7), and δ; minimise the relative cost (Eq. 6) or FLOPs | Hard; holds with probability ≥ 1 − δ | Multi-objective search for a Pareto front, then multiple hypothesis testing on a calibration split (Learn-then-Test family) | Head pruning (static, structured), token pruning, early exit; a grid of 6,480 configurations | BERT-base on IMDB, AG News, QNLI, QQP, MNLI (e.g. QNLI at α = 0.05) | Per-model calibration, no retraining [H] | [V] |
| C2 | Alvarez, "Confident magnitude-based neural network pruning" | arXiv 2408.04759 | Tolerance α on the risk against the full model (1 − accuracy, disagreement, or 1 − IoU), and δ | Hard; holds with probability ≥ 1 − δ | Fixed-sequence testing over λ_j = j/100 (global magnitude ratio), keeping the largest λ that controls the risk | No (unstructured, one-shot) | MNIST MLP (118,282 params): α 0.03 → λ 0.65–0.72. U-Net PolypGen: α 0.05 → λ 0.06 | Per-model calibration | [V(f)] |
| C3 | Learn then Test (Angelopoulos, Bates, Candès, Jordan, Lei) | arXiv 2110.01052 | The framework: "reframe the risk-control problem as multiple hypothesis testing" | – | – | – | – | – | [V(f)] (abstract only) |

Group D: near misses (the tolerance is only an evaluation measure, the method is the budget dual, or "tolerance" means something else).

| # | Work | Venue / ID | User input | Constraint type | Mechanism | Structured? | Networks shown | Note | Label |
|---|---|---|---|---|---|---|---|---|---|
| D1 | Provable Filter Pruning (Liebenwein, Baykal, Lang, Feldman, Rus) | ICLR 2020 [R]; arXiv 1911.07412 | In practice, "the desired pruning ratio (PR) and failure probability δ". In theory, an (ε, δ) output-error guarantee | Budget | Binary search over ε to fit the sampling budget, i.e. the dual. Reported PR is the highest "when the pruned model's test accuracy is within 0.5% of the original", over 10–20 target ratios | Yes (filters) | CIFAR-10 VGG16, ResNet20/56/110, DenseNet22, WRN16-8; ImageNet ResNet18/50/101 | Tolerance used for reporting only, on test | [V] |
| D2 | Lost in Pruning, "prune potential" (Liebenwein, Baykal, Carter, Gifford, Rus) | 4th MLSys 2021; arXiv 2103.03014 | None: an evaluation measure | P(θ, D) = the largest prune ratio whose expected loss increase is ≤ δ (Def. 1, Eq. 2); δ = 0.5 % test error (§5.1) | Sweeps of PRUNERETRAIN cycles | WT and SiPP (unstructured), FT and PFP (structured) | CIFAR-10 ResNet18/56/110, WRN16-8, DenseNet22, VGG16; ImageNet ResNet18/101; VOC DeeplabV3 | The closest "maximal compression at bounded loss" definition, measured on test | [V] |
| D3 | FAIR-Pruner (Lin et al.) | arXiv 2508.02291 (v3, 20 May 2026; "Submitted to IEEE TPAMI") | α, a "Tolerance of Difference": the overlap between the removal set (lowest U-Scores) and the protection set (highest R-Scores) (Eqs. 1–2) | Per-layer depth rule; a budget is met by sweeping α | Search-free | Yes | ResNet-56, DenseNet-40 CIFAR-10; ResNet-50 ImageNet; ConvNeXt CIFAR-100; DeiT-B; Qwen1.5-MoE | **Not an accuracy tolerance** (as the brief suspected) | [V(f)] |
| D4 | NetAdapt (Yang et al.) | ECCV 2018; arXiv 1804.03230 | A resource budget (latency, energy) | Budget | Simplifies the net "until the resource budget is met while maximizing the accuracy" | Yes | – | The dual | [V(f)] (abstract) |
| D5 | Li, Chen, Liu, Sun, Zhang, "Towards Optimal Filter Pruning with Balanced Performance and Pruning Speed" | arXiv 2010.06821 | A per-layer loss-variation threshold θ (a loss, not an accuracy), or a global rate γ | Per-layer bound; an outer search on θ hits γ within a tolerance ε (the dual) | Binary search per layer for the most filters within θ; one fine-tune at the end | Yes | VGG16-BN CIFAR-10 at θ = 0.12: FLOPs −70.29 %, params −87.72 %, top-1 "almost the same" | Per-layer bound | [V] |
| D6 | Oh, Kim, Baik, Hong, Lee, "Batch Normalization Tells You Which Filter is Important" | arXiv 2112.01155 | A per-layer degradation target δ_i, "given as a hyper-parameter" | Per layer; the target is matched, not bounded | Five-step bisection per layer on validation accuracy | Yes | CIFAR-10, ImageNet | How δ_i is set for a whole network [U] | [V] |
| D7 | Out-of-the-box Channel Pruned Networks (Venkatesan, Swaminathan, Zhou, Luo) | arXiv 2004.14584 | Expected compression C_e (a budget). A_e is "the accuracy of the original unpruned model", used as a normaliser | Gaussian reward around C_e (Eq. 10) | RL search for layer-wise profiles that transfer across datasets (same architecture) | Yes (channels) | CIFAR-10/100, ImageNet | Transfer across datasets, not across architectures; budget | [V] |
| D8 | CACP, Conditional Automated Channel Pruning (Liu et al.) | arXiv 2009.09724; DOI 10.1109/LSP.2021.3088323 | Compression rate, as an input to "a conditional model" | Budget-conditioned: one model for all rates | – | Yes (channels) | – | Amortised over budgets, per network; no accuracy-conditioned analogue was found | [V(f)] (abstract) |
| D9 | DeepSZ (Jin, Di, Liang, Tian, Tao, Cappello) | HPDC 2019, DOI 10.1145/3307681.3326608; arXiv 1901.09124 | A "user-set accuracy constraint" | Hard (estimated) | Per-layer error bounds of an error-bounded lossy compressor, applied to a pruned net, chosen "to maximize the compression ratio under the user-set accuracy constraint" | Not pruning (lossy weight encoding after pruning) | AlexNet 46×, VGG-16 116× on ImageNet; LeNet-300-100 57×, LeNet-5 56× on MNIST; ≤ 0.3 % loss | The same "maximal compression at a tolerance" contract, for weight encoding | [V(f)] (abstract) |
| D10 | Optimal Brain Surgeon (Hassibi & Stork) | NIPS 1992 (NeurIPS 5, pp. 164–171) | None | Stopping rule | Delete the least-salient weight while its error increase "is much smaller than E" (the training error); "(Other stopping criteria can be used too.)" | No | XOR, MONK's, NETtalk | The historical stop-at-error rule | [V] |

---

## Mechanisms (brief item 3)

- **A search over one scalar compression level per network, with fine-tuning:**
  - NNCF: adaptive steps that halve at each change of direction, within one training run.
  - MetaML-Pro: binary search.
  - Zhao et al.: increasing steps, with rollback and halving.
  - Tian et al.: increase until failure, then a margin pull-back.
  - Condensa: Bayesian optimisation on a level set, at most 10 samples.
  - SPECTRA §423: bisection over κ ∈ [0.10, 0.90] with five 20-epoch probes, each restarted from the origin, then one 100-epoch final (ledger §423 heading).
- **The constraint inside training, as a reward term:**
  - NEON (C).
  - Huang et al. (b).
  - PuRL (a target accuracy plus a target sparsity).
  - AMC accuracy-guaranteed (no tolerance at all; only a shaped reward toward no loss).
- **A per-layer bound from single-layer sensitivities:** PaddleSlim (relative accuracy loss), Li et al. 2020 (loss variation), Oh et al. (matched per-layer degradation), Vu-Van et al. (per-layer δ). None of these bounds the joint drop. AMC's critique that per-layer errors do not sum linearly is recorded in `docs/paper/lit/LIT_RULE_AND_COMPARATORS_21NOV.md`, item 5 [R].
- **Statistical risk control:** Pareto Testing and Alvarez (Learn-then-Test). They give a finite-sample guarantee on the test-time risk, but they calibrate over a fixed grid of configurations and retrain nothing.

## Generic across architectures with no per-network search? (brief item 4)

- **NEON, on dense nets only.** Its agent is trained offline on random nets and applied to unseen datasets without further training. The tolerance, however, is fixed for each trained configuration: NEON 0, 1, 5 and 50 are four configurations (p. 11). The state is the feature maps of the network and of the analysed layer (Fig. 3 caption, p. 9), so C is not a policy input [H].
- **Architecture-agnostic code is not the same as no per-network search.** NNCF, Condensa and PaddleSlim run on any model, but each run is a per-network search with fine-tuning or a per-network sensitivity scan.
- **Risk-control methods** calibrate per model over a configuration grid; they need a calibration split but no training.
- **I found no work that conditions a frozen policy on τ.** CACP conditions on the budget, for one network [H: this rests on the searches listed below, so it is an absence of evidence, not proof].
- **SPECTRA §423 is a per-network search:** five 20-epoch probes and one 100-epoch final per network and τ. The allocation inside each probe comes from a frozen cross-architecture agent (T6-fs) or the rule (sens_cost).

## NEON itself (brief item 5)

- **The problem statement** (p. 1, ll. 44–48): "users have no easy way to define their preferences in the compression/performance-reduction trade-offs"; "existing approaches require the user to test multiple hyperparameter configurations to reach the desired outcome."
- **Prior work, as NEON frames it.** Table 1 (p. 4) has a column "Define Compression-Accuracy Tradeoff". It is × for basic pruning, global ranking pruning and existing DRL-based pruning agents, and ✓ only for NEON. The text (p. 4, ll. 155–158) reads: "to the best of our knowledge, NEON is the first pruning approach to offer this capability."
- **The motivating example** (p. 8, ll. 296–301): "compress as much as possible, but don't hurt performance by more than 3%". NEON adds that "Existing DRL-based pruning approaches … either require the manual configuration of hyperparameters that balance accuracy reduction and compression or only optimize their DRL agent for one of these goals. To date, no existing study proposes a method for automatically balancing and optimizing these two goals and their trade-offs."
- **Its definition of the tolerance** (Eq. 4, p. 8; symbols on pp. 8–9, ll. 302–304):
  - R(acc_d, c_t) = −c_t³ if acc_d < −C; c_t if −C ≤ acc_d ≤ 0; c_t³ if acc_d > 0.
  - C is "the 'permissible' reduction in performance (as defined by the user)".
  - acc_d = accuracy_current − accuracy_original.
  - c_t = (1 − |N_pruned| / |N_org|) · 100.
  - The text calls the penalty "exponential" (p. 9, l. 307). The extracted formula is cubic [H, read from the extracted glyphs].
- **Adherence.** NEON stays within the user's bound on "82%-100% of datasets, depending on NEON's configuration" (p. 2, ll. 71–72). At C = 5 and 50 no dataset exceeded the bound; at C = 0 and 1, 5 and 3 of 28 datasets did (p. 16, ll. 510–513).
- **What NEON leaves out.** It describes Huang et al. [13] only as using "a pruning agent to determine which filters need to be eliminated" (p. 4, l. 174), and AMC [14] as using "four pruning and re-training stages" (p. 4, l. 176) and as a single-network agent (p. 11). It never mentions Huang's drop bound b or AMC's accuracy-guaranteed protocol; a text search for "guarant" finds nothing [V].

## What would be new for SPECTRA (candid novelty risk)

**Not new; do not claim it:**
1. **An accuracy tolerance as the user's input to CNN pruning.** Prior art: Huang 2018 (b), Condensa (ε), NNCF (maximal degradation), Zhao et al. (target accuracy loss), MetaML-Pro (α_p) and PaddleSlim (per-layer loss). SPECTRA must not repeat NEON's "first" sentence: a reviewer who knows NNCF or Huang et al. will flag it, and Huang et al. is in NEON's own reference list.
2. **Bisection over a scalar compression level with short fine-tunes.** This is NNCF's adaptive step search and MetaML-Pro's binary search. A coincidence that shows how close it is: §423's five probes over [0.10, 0.90] resolve κ to 0.8 / 2⁵ = 0.025 [c], the same as NNCF's default `minimal_compression_rate_step` of 0.025 (on its own rate scale).
3. **Reporting the maximal compression within a tolerance.** PFP and Lost in Pruning already do this ("commensurate accuracy", 0.5 % on test).

**Possibly new (each needs evidence that does not exist yet):**
- **(a) The combination.** A tolerance mode whose allocation at each probe comes from a frozen, cross-architecture structured-pruning agent with no per-network training, evaluated for TEST adherence across families and seeds against the same-loop rule. No CNN paper in this scan reports adherence rates across nets × tolerances × seeds; NEON reports the dense analogue (82–100 % of datasets). The risk: §423 found KAPPA-P SAME at every τ (−0.014, −0.024, −0.014). The agent reaches the same size as the rule, so the mode contributes usability and an honest protocol, not an agent win.
- **(b) A tolerance-conditioned frozen policy.** τ would be an input, and κ and the plan would come out in one shot with no probes. That is NEON's promise without NEON's one configuration per C. The scan found none (CACP conditions on the budget; NEON fixes C) [H]. The risk: it needs a new train, and predicting the recovered accuracy per network is exactly what the probes measure. If it misses τ on TEST more often than bisection does, it is worse than bisection.
- **(c) A calibrated TEST guarantee for structured CNN pruning with recovery.** Run §423's κ candidates as a fixed-sequence test (Learn-then-Test) on a held-out calibration split, so that the chosen κ meets τ with probability ≥ 1 − δ. Pareto Testing (BERT configurations, no retraining) and Alvarez (unstructured, one-shot) are the precedents; I found none for structured CNN pruning with fine-tuning [H].
  - Motivation: per the §423 heading, the finals at the chosen κ meet τ on TEST 5k in 17 / 19 / 21 of 27 searches for the agent and 12 / 18 / 23 for the rule. That is 6–10 misses in 27 for the agent [c].
  - Risk: a calibration split of CIFAR's size gives wide bounds, and a δ-guarantee at τ = −0.5 may push κ towards 1.

**Units and splits differ, so state SPECTRA's exactly:**
- NNCF's default is relative (%), and Condensa's ε is absolute top-1 (2 %).
- PaddleSlim's loss is relative. Huang's b is in accuracy points [H].
- NEON's C is in % of accuracy.
- PuRL's reward uses test accuracy, and PFP and Lost in Pruning measure on test. Huang and Condensa use validation; for Zhao et al. and Tian et al. the split is [U].
- SPECTRA's τ is in absolute val points after a 20-epoch probe, its final runs 100 epochs, and adherence is read on TEST 5k.

**Bottom line [H].** The novelty risk of "an accuracy-limit mode" alone is high; (b) and (c) carry medium risk, each conditional on new runs. Frame §423 as a NEON-lineage usability feature with an honest adherence protocol, and cite NNCF, Condensa, Huang et al. 2018 and Zhao et al. as prior art.

---

## Gaps and what was not verified

- **Distiller:** the repository, its docs pages and issue #64 returned 404, and the Illinois GitLab mirror refused the connection. The row is [R] only.
- **Venues [R]:** ICLR 2023 for Pareto Testing (ICLR virtual poster page, from a search result) and ICLR 2020 for PFP (from the brief). Neither arXiv page lists a venue.
- **Unresolved [U]:**
  - whether MetaML-Pro's pruning is structured;
  - which split Zhao et al. and Tian et al. use for the loop's accuracy;
  - how Oh et al. set δ_i for a whole network.
- **Not checked** (so not listed): Vitis AI Optimizer, NVIDIA ModelOpt/TAO, Neural Magic SparseML, Sony MCT, Apple coremltools, and NNCF's TensorFlow docs.
- **Searches run (21 Nov), as web queries:**
  - NNCF adaptive compression level;
  - INC accuracy_criterion with pruning;
  - NNI SensitivityPruner;
  - Distiller amc-protocol;
  - PaddleSlim get_ratios_by_loss;
  - structured pruning under a user accuracy-drop constraint (2023–2025);
  - binary search on the pruning ratio under an accuracy-drop threshold;
  - risk control / Learn-then-Test for pruning;
  - DeepSZ;
  - AIMET AutoQuant;
  - RL pruning with an accuracy threshold;
  - accuracy budget against a FLOPs budget;
  - policies conditioned on a target accuracy.
- **Related repo file:** `docs/paper/lit/LIT_RULE_AND_COMPARATORS_21NOV.md` (the rule's PaddleSlim lineage and FAIR-Pruner).
