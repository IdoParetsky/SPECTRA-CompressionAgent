# Paper cards for Gilad: what SPECTRA is compared with, what it is closest to, and how pruning is judged (22 Nov 2026)

Compiled 22 Nov 2026 (IST) by the literature subagent from `docs/paper/lit/*.md`, `docs/paper/SPECTRA_draft.md`, `PAPER_SKELETON.md` and `GILAD_DIRECTIVES_18AUG.md`. The web was used only for links, venues and titles, marked (web). Read-only except this file.

## How to read
1. One card per paper: what it does, what it reports (WIN / TIE / LOSS against its own or others' comparators), its limits, and where SPECTRA stands. Group 1: published methods in the draft's tables. Group 2: work closest to SPECTRA's approach. Group 3: evaluation practice.
2. Labels kept from the lit scans: [V] read at the source by one of our scans; [R] reported second-hand (a reprint, the repo's own transcription, or a search snippet); [c] arithmetic in a scan; [H] interpretation; [U] not established.
3. "Against SPECTRA" quotes only the draft's tables, by ledger section (§N), and every SPECTRA number is PRELIM. TEST 5k = our held-out half of the test set; val = the other half; 10k = the full test set; honest = our change minus the unpruned network's own change under the same recovery.
4. Published rows come from their own base network, recipe and epochs; ours are one cut plus one fixed 100-epoch recovery (90 epochs on ImageNet). They stand side by side, never as a beat. A published CIFAR row is compared with our 10k change.
5. FLOPs kept and params kept are fractions remaining (2× = 0.50). C10 / C100 = CIFAR-10 / CIFAR-100.
6. "Our frozen agent": trained once on a catalog, applied with no per-network search. "Our rule": the sensitivity rule fixed in advance for the budget (the FLOPs rule at equal FLOPs, the cost-aware rule at equal params). "The threshold rule": one global sensitivity threshold. "The even cut": one keep fraction for every group.
7. "Level" means the registered call found the pair within ±0.3 points, or not significant. Tables 1 and 1b use five seeds, the development reads quoted here (§382, §394, §396, §398, §423, §424, §440–§442) three, and ImageNet one (no call).
8. Times are minutes on one RTX 4090 (draft §5.6, §419): SPECTRA takes 7.7–8.5 min per ResNet-56 (decision 3.3–3.7 min; 20.9–21.4 min with a stock data loader, the like-for-like row).
9. "No shared cell": the draft has no SPECTRA row on that paper's network and dataset near its operating point. "Not in the draft's tables": such a cell exists, but the paper's row is not quoted there.
10. Gong et al. and the papers that cite it are carded in `LIT_GONG_CITERS_22NOV.md` (in progress); this file gives a one-line pointer.

## Index

| # | Paper | Venue, year | Kind | Verdict against SPECTRA (one line) |
|---|---|---|---|---|
| 1 | DepGraph | CVPR 2023 | comparator | Same ResNet-56: our agent is level with our rule, beside its +0.11 at 2.57×; at 8.9× it loses 3.1, our one-cut arms about 7 (§408, §418) |
| 2 | OCSPruner | WACV 2026 | comparator | Beside us at 2.57× (its −0.51 on its own base); cheaper than us only when no trained base exists (§408, §419) |
| 3 | HRank | CVPR 2020 | comparator | Level-with-rule row beside its VGG-16 point; our ImageNet arms sit above its row; its pipeline takes 344 min (§408, §430, §419) |
| 4 | CHIP | NeurIPS 2021 | comparator | Highest published ImageNet rows; our arms 0.4–0.9 below at 2.08×, likely recovery, untested (§430) |
| 5 | FPGM | CVPR 2019 | comparator | Beside us on ResNet-110 at 0.48; our ImageNet arms sit above its rows; 2.1× our stock-loader time (§408, §430, §419) |
| 6 | SFP | IJCAI 2018 | comparator | Our ResNet-56 C100 10k change (−1.28) sits above its −2.61 as LFPC ran it (§418) |
| 7 | LFPC | CVPR 2020 | comparator | Its −0.58 on ResNet-56 C100 sits above our −1.28 at 10k (§418) |
| 8 | DECORE | CVPR 2022 | comparator (RL) | ResNet-110 near 0.4 FLOPs: both at base level; its row is unverified at the source (§408) |
| 9 | LeGR | CVPR 2020 | comparator | Our ImageNet arms "sit with LeGR's rows"; its pipeline takes 167 min (§430, §419) |
| 10 | TPP | ICLR 2023 | comparator | VGG-19 C100: our arms near its −6.24 at 8.9×, at base level where it loses 0.83 at 2× (§418); rows unverified |
| 11 | GReg | ICLR 2021 | comparator | The same VGG-19 C100 picture as TPP: −6.27 at 8.9×, −0.94 at 2× (§418) |
| 12 | EigenDamage | ICML 2019 | comparator | Our 8.9× arms sit above its −8.16 (§418) |
| 13 | SOSP | ICLR 2022 | comparator | At 2× our arms keep their accuracy where it loses 0.51; its 8.9× row differs in FLOPs by more than 10 % (§418) |
| 14 | Li et al. 2017 | ICLR 2017 | comparator (origin of our rules) | Its single-threshold form is our strongest rule; our rule-blending agent is level with it (§440, §441) |
| 15 | Jiang et al. | IJCAI 2022 | comparator | Its ResNet-56 C100 rows sit above our −1.28 at 10k; its Tiny-ImageNet cell is not a SOTA cell; rows unverified (§418) |
| 16 | TAS | NeurIPS 2019 | comparator | Its −0.93, with distillation, sits above our −1.28 on ResNet-56 C100 at 10k (§418) |
| 17 | LAMP | ICLR 2021 | comparator (rule in our loop) | Our structured adaptation collapses on residual networks (−6.63 below our FLOPs rule; §442) |
| 18 | SPDY | ICML 2022 | comparator (form in our loop) | Our implementation of its knapsack form is below our rule (−3.60, −2.93; §424) |
| 19 | MetaPruning | ICCV 2019 | comparator (feature table) | Not in our tables; a per-network search (§6) |
| 20 | SPA | arXiv 2024 | comparator (feature table) | No shared cell; the grouping layer we leave to it and to DepGraph (§1) |
| 21 | NEON | Inf. Sci. 2022 | approach (predecessor) | No shared cell; our frozen agent is its idea for CNNs; its tolerance knob is our accuracy-limit mode, which stops at our rule's size (§423) |
| 22 | AMC | ECCV 2018 | approach (also cost table) | Timed only: 109 min per ResNet-56 against our 7.7–8.5; its accuracy row is missing from our tables (§419) |
| 23 | GNN-RL | ICML 2022 | approach (also Table 1c, cost table) | Our ImageNet arms beside its 74.28; its search had not finished after 17 h (§430, §419) |
| 24 | RL-Pruner | arXiv 2024 | approach (also cost table) | Timed only: 271 min against our 7.7–8.5 (§419) |
| 25 | Generalist agent | IEEE Access 2024 | approach | No shared cell; the only frozen per-layer selector found, on LeNet only |
| 26 | Balaskas et al. | IEEE TETC 2024 | approach | No shared cell |
| 27 | MDP | CVPR 2025 | approach | No shared cell (latency targets) |
| 28 | LayerPrune | ACCV 2020 | approach | No shared cell (latency targets) |
| 29 | Metanetwork pruning | arXiv 2025 | approach | Not in our tables; the closest frozen CNN transfer, within one family |
| 30 | Gong et al. | ACM TODAES 2022 | approach | Pointer only: `LIT_GONG_CITERS_22NOV.md` |
| 31 | DominoSearch | NeurIPS 2021 | approach | No shared cell |
| 32 | APQ | CVPR 2020 | approach | No shared cell |
| 33 | Blending criteria | ICANN 2021 | approach | No shared cell |
| 34 | LAASP | Image Vis. Comput. 2023 | approach | No shared cell |
| 35 | EagleEye | ECCV 2020 | approach | Its adaptive BN is our batch-norm reward, weak at deep budgets (§375) |
| 36 | RL-MCTS | WACV 2022 | approach | No shared cell |
| 37 | NetAdapt | ECCV 2018 | approach | No shared cell |
| 38 | DPM | arXiv 2024 | approach (deep cuts) | Not in our tables (the same 8.9× cell as DepGraph) |
| 39 | CHEX | CVPR 2022 | approach (deep cuts) | No shared cell |
| 40 | OTOv2 | ICLR 2023 | approach (generic code) | No shared cell |
| 41 | OSSCAR | ICML 2024 | approach (training-free rule) | No shared cell (no recovery) |
| 42 | Isomorphic Pruning | ECCV 2024 | approach (training-free rule) | Not in our tables (the ImageNet 0.48 cell) |
| 43 | Relative Repairability | arXiv 2026 | approach (training-free rule) | No shared cell |
| 44 | LADP | ICCV 2023 | approach (training-free rule) | No shared cell |
| 45 | FAIR-Pruner | arXiv 2025 | approach (threshold rule) | Not in our tables; the closest modern cousin of our threshold rule |
| 46 | CPMC | IJCNN 2021 | approach (training-free rule) | No shared cell |
| 47 | Group Fisher | ICML 2021 | approach (training-free rule) | No shared cell; a precedent of our cost-aware rule (§4.1) |
| 48 | Blalock et al. | MLSys 2020 | evaluation | Our random-channel control: the channels chosen matter little (§398) |
| 49 | Liu et al. 2019 | ICLR 2019 | evaluation | Our scratch control: scratch equals inherited at equal compute (§394) |
| 50 | Li et al. 2022 | CVPR 2022 | evaluation | Our agent ends above our implementation of its best-of-100 (+0.51, +0.45; §396) |
| 51 | PruningBench | arXiv 2024 | evaluation | No row; not cited in the draft |
| 52 | Wang et al. 2023 | arXiv 2023 | evaluation | No shared cell; its fairness principle is our one-recovery protocol |
| 53 | Jin et al. | NeurIPS 2022 | evaluation | Its dense control is our honest change (+0.78 control gain on ImageNet; §430) |
| 54 | Le & Hua | ICLR 2021 | evaluation | Our recovery follows it (§4.2) |
| 55 | Oracle pruning | TMLR 2026 | evaluation | Our proxy study agrees: cut-time scores rank plans weakly (§375) |
| 56 | Random tickets | NeurIPS 2020 | evaluation | Agrees with our channels-matter-little control (§398) |

---

## Group 1. Published methods in the draft's tables (20 cards)

AMC, GNN-RL and RL-Pruner are also in the draft's tables (the cost table; GNN-RL in Table 1c too); their cards are in group 2.

### DepGraph (CVPR 2023): DepGraph: Towards Any Structural Pruning
- Link: https://arxiv.org/abs/2301.12900
- Kind: comparator we benchmark against
- What it does: Builds a dependency graph so that coupled layers (residual sums, concatenations) are cut together as groups, and ranks groups by a norm after a sparse-learning stage, on CNNs, RNNs, GNNs and Transformers. It solves one given model at a time; there is no transferred policy (draft §2.2.4).
- Results: WIN at the top of its own Table 1: ResNet-56 C10 93.53 → 93.77 (+0.24) at 2.11× and → 93.64 (+0.11) at 2.57×; VGG-19 C100 73.50 → 70.39 (−3.11) at 8.92×, against GReg-2 (−6.27) and EigenDamage (−8.16) [V]. TIE without its sparse learning at 2.11× (93.46, −0.07) [V]. LOSS without sparse learning at 8.87× (67.60: −5.90 by its accuracies, printed −5.44), so +2.79 of its deep cut is the training procedure [V; c]; ResNet-50 ImageNet 76.15 → 75.83 at FLOPs 0.482 [V].
- Limits: It prints no params column for any CIFAR model, so no parameter count can be attributed to it (§447) [V], and its official script keeps the best test-set epoch [V: LIT_ALTERNATIVES §1.4]. Its pipeline adds sparse learning (30 epochs on ImageNet) before a 90-epoch fine-tune (draft §5.1c), so grouping, allocation and training are mixed in its rows [H].
- Against SPECTRA: On DepGraph's own ResNet-56 at 2.57× (FLOPs 0.39), our frozen FLOPs agent ends at 93.38 ± 0.23 TEST 5k from an origin of 93.36, level with our rule at 93.24 (+0.15; val +0.11, 10k +0.13, honest +0.21; Table 1, §408), beside DepGraph's published 93.53 → 93.64. On its VGG-19 C100 at 8.9× our arms lose about 7 points (agent 66.62 ± 1.45 TEST 5k from 73.96; 10k −7.26) where its pipeline loses 3.11 (Table 1b, §418), and that pipeline takes 85 min per ResNet-56 against our 7.7–8.5 (§419).

### OCSPruner (WACV 2026): One-Cycle Structured Pruning via Stability-Driven Subnetwork Search
- Link: https://arxiv.org/abs/2501.13439 (WACV 2026 confirmed on the conference's poster page, web)
- Kind: comparator we benchmark against
- What it does: Trains, prunes and fine-tunes in one training cycle: early in training it removes the groups with the lowest L2 norm under a growing penalty, by a binary-searched global threshold, once the chosen subnetwork stays stable across epochs [R: GILAD_DIRECTIVES §5]. The cycle is 300 epochs on CIFAR and 130 on ImageNet [V].
- Results: WIN at 8.9× on VGG-19 C100: 73.58 → 69.98 (−3.60; printed −3.64) from a pretrained start and 74.24 → 70.47 (−3.77) from scratch, well above GReg-2's −6.27 at the same point [V; c]. TIE on ResNet-110 C10: 94.36 → 94.30 (−0.06) at FLOPs 0.464 [V]. LOSS against DepGraph with sparse learning at 8.9× (−3.11), and on ResNet-56 at 2.57×, where it drops 0.51 (94.01 → 93.50 at FLOPs 0.388, params 0.423) against DepGraph's +0.11, each on its own base [V; c].
- Limits: Every row starts from its own base and its own long cycle, so its rows are recipe-bound. Its FLOPs and params columns are percent remaining, not percent removed [V].
- Against SPECTRA: At the shared 2.57× point on DepGraph's ResNet-56, our frozen FLOPs agent is level with our rule (93.38 ± 0.23 against 93.24 TEST 5k; Table 1, §408), and our arms there keep more params than OCSPruner (0.46 against its 0.42; three-seed sub-table, §382). When no trained base exists, its single cycle is the cheaper route on ResNet-56: 71 min against about 82 min for "train a base, then SPECTRA" (an estimate; 95 with a stock loader; §419).

### HRank (CVPR 2020): HRank: Filter Pruning using High-Rank Feature Map
- Link: https://arxiv.org/abs/2002.10179
- Kind: comparator we benchmark against
- What it does: Scores filters by the rank of the feature maps they produce and removes the low-rank ones layer by layer, retraining after each stage (55 stages on ResNet-56 in our timing run, draft §5.6). It works per network.
- Results: [V, its Tables 1–5] ResNet-56 C10 93.26 → 93.17 (−0.09) at FLOPs 0.500 / params 0.576; ResNet-110 93.50 → 93.36 at 0.418; VGG-16 C10 93.96 → 93.43 (−0.53) at 0.465 / 0.171, and → 92.34 at 0.347; ResNet-50 ImageNet 76.15 → 74.98 at 0.562. TIE: FAIR-Pruner reports parity with it [V]. LOSS: HAP reports "up to 2% higher accuracy with fewer parameters and FLOPs" [V], and L1 with a modern fine-tune reaches 74.77 at 2.56× on ResNet-50 (Wang et al. 2023), against HRank's 74.98 at only 1.78× [V].
- Limits: Reprints carry its ResNet-56 result as 92.17 (DepGraph Table 1, NPPM, ATO), a typo for 93.17 [V]. Its staged pipeline is slow: 344 min on ResNet-56 and 66 min on VGG-16 on our GPU (§419, §425).
- Against SPECTRA: On VGG-16 C10 at its FLOPs 0.465, our frozen FLOPs agent ends at 93.92 ± 0.17 TEST 5k from 93.74, level with our rule (+0.04; val −0.07, 10k −0.01, honest −0.12; Table 1, §408), beside its 93.96 → 93.43 from a different base. On ImageNet ResNet-50 at 0.56, our three arms (75.66–75.89 top-1, one seed; Table 1c, §430) sit above its 74.98 in the draft's reading, beside it and never a beat.

### CHIP (NeurIPS 2021): CHIP: CHannel Independence-based Pruning for Compact Neural Networks
- Link: https://arxiv.org/abs/2110.13981
- Kind: comparator we benchmark against
- What it does: Scores each channel by how independent its feature map is of the other channels' maps, removes the least independent channels, and fine-tunes for 300 epochs on CIFAR (draft §5.6). It works per network.
- Results: WIN above its own unpruned base on ResNet-56 C10, 93.26 → 94.16 (+0.90) at FLOPs 0.526 / params 0.572, and on ResNet-110, 93.50 → 94.44 (+0.94) at 0.479 / 0.517 [V]. TIE on ResNet-50 ImageNet, 76.15 → 76.15 at 0.513 and 76.30 at 0.552, and on VGG-16 C10, 93.96 → 93.86 (−0.10) at 0.419 / 0.184 [V]. LOSS with no fine-tune at all: at 1.7× on ResNet-50 it falls below 1 % top-1 in OSSCAR's one-shot table [V].
- Limits: Its gains above the base come with a 300-epoch fine-tune, so method and recipe are mixed, the critique of Wang et al. 2023 [H]. It is slow: 101 min on ResNet-56 and 118 min on VGG-16 on our GPU, mostly fine-tune (§419, §425).
- Against SPECTRA: On VGG-16 C10 at its FLOPs 0.34, our frozen FLOPs agent ends at 93.23 ± 0.21 TEST 5k from 93.74, level with our rule (−0.24; val −0.04, 10k −0.14, honest −0.44; Table 1, §408), beside its 93.96 → 93.72 at 0.334. On ImageNet ResNet-50 our three arms sit "0.4–0.9 below DepGraph's, OCSPruner's and CHIP's at 2.08× and 0.1–0.6 below OCSPruner's and CHIP's at 1.79×" (one seed; Table 1c, §430), a gap the draft reads as more likely recovery (CHIP fine-tunes 180 epochs on ImageNet) than allocation, untested.

### FPGM (CVPR 2019): Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
- Link: https://arxiv.org/abs/1811.00250
- Kind: comparator we benchmark against
- What it does: In each layer, removes the filters nearest the layer's geometric median (the title's criterion), with soft pruning during training (200 epochs in our timing run, draft §5.6). Its FLOPs count convolutions only [V].
- Results: [V, its Tables 1–2] ResNet-56 C10 (pretrained) 93.59 → 93.49 (−0.10) at FLOPs 0.474; ResNet-110 93.68 → 93.74 (+0.06; printed −0.16) at 0.477; ResNet-50 ImageNet 76.15 → 75.59 at 0.578 and 74.83 at 0.465. WIN with Li et al.'s hand sensitivity allocation on VGG-16 C10: 93.54 against 93.23 without it (+0.31, at 34.2 % and 35.9 % FLOPs ↓) [V; c]. LOSS on ResNet-56 C100 in LFPC's table (69.66 against LFPC's 70.83) and to Blending by 0.32 [V].
- Limits: DepGraph reprints it as 93.59 → 93.26 at "1.70×", a speed-up with no counterpart in FPGM, which prints 52.6 % FLOPs ↓ [V]. It works per network.
- Against SPECTRA: On ResNet-110 C10 at its FLOPs 0.48, our frozen FLOPs agent ends at 93.91 ± 0.39 TEST 5k from 93.84, level with our rule (−0.04; val +0.22, 10k +0.09, honest +0.04; Table 1, §408), beside its 93.68 → 93.74. On ImageNet ResNet-50 our arms sit above both its rows (one seed; Table 1c, §430), and its pipeline takes 44 min on ResNet-56, 2.1× our stock-loader time (§419).

### SFP (IJCAI 2018): Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
- Link: https://arxiv.org/abs/1808.06866
- Kind: comparator we benchmark against
- What it does: Sets low-norm filters to zero during training but keeps updating them ("soft" pruning), so that a pruned filter can recover before the final cut [H: from its title; its L2 ranking is one of our ranking A/Bs, GILAD_DIRECTIVES §3].
- Results: [V, its Tables 1–2] ResNet-56 C10 (pretrained) 93.59 → 93.35 (−0.24) at FLOPs 0.474; ResNet-110 93.68 → 93.86 (+0.18) at 0.592; ResNet-50 ImageNet 76.15 → 74.61 (−1.54) at 0.582. LOSS on ResNet-56 C100 as run in LFPC's table: 68.79 (−2.61), below FPGM (−1.75) and LFPC (−0.58) [V].
- Limits: Two of its own ResNet-32 rows are internally inconsistent (90.63 with a printed drop of 0.00; 90.08 with 0.55), and reprints change them [V].
- Against SPECTRA: On a public (chenyaofo) ResNet-56 C100 at FLOPs 0.48, our frozen FLOPs agent ends at 71.86 ± 0.39 TEST 5k from 73.18, level with our rule (+0.08; val −0.15, 10k −0.04, honest +0.38; Table 1b, §418). The agent's 10k change (−1.28) lies above FPGM's (−1.75) and SFP's (−2.61) as TAS and LFPC ran them, and below LFPC (−0.58), TAS (−0.93) and Jiang et al. (0.00, −0.64), beside them and never a beat.

### LFPC (CVPR 2020): Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration
- Link: https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.pdf (no arXiv record found; the draft's DOI 10.1109/CVPR42600.2020.00208 resolves to IEEE Xplore, title not shown to our fetch)
- Kind: comparator we benchmark against
- What it does: Learns which criterion each layer uses (ℓ1, ℓ2 or geometric median) with a differentiable Gumbel-softmax sampler on validation loss, then prunes and fine-tunes, per network [V].
- Results: WIN on ResNet-56 C100 (its Table 4): 70.83 (drop 0.58) at FLOPs 0.484, against FPGM 69.66 and SFP 68.79, +1.17 over FPGM on an implied common base of 71.41 [V; c]; whether those rows are re-runs is [U]. Other rows [V]: ResNet-56 C10 (pretrained) 93.59 → 93.24 at 0.471; ResNet-110 93.68 → 93.07 (pretrained) and 93.79 (from scratch) at 0.397. LOSS: none in our scans.
- Limits: In the same pipeline, other criterion portfolios gain only +0.05 to +0.32 over the best single criterion (LAASP, Blending) [V]. It works per network.
- Against SPECTRA: On ResNet-56 C100 at FLOPs 0.48, our frozen FLOPs agent is level with our rule (71.86 ± 0.39 against 71.78 ± 0.15 TEST 5k from 73.18; Table 1b, §418), and the agent's 10k change (−1.28) sits below LFPC's −0.58. On ResNet-110 C10 at FLOPs 0.40, our agent ends at 93.80 ± 0.21 TEST 5k from 93.84, level with our rule (+0.01; Table 1, §408), beside LFPC's 93.68 → 93.07.

### DECORE (CVPR 2022): DECORE: Deep Compression with Reinforcement Learning
- Link: https://arxiv.org/abs/2106.06091 (web; CVPR 2022 per the CVF open-access listing and draft ref [167])
- Kind: comparator we benchmark against (also reinforcement learning)
- What it does: By its title, compression by reinforcement learning; our scans did not read its mechanism [U]. The repo records a joint schedule of about 300 epochs [R: repo].
- Results: TIE on ResNet-110 C10 (DECORE-300): 93.50 → 93.50 at FLOPs 0.382 / params 0.352 [R: the repo's transcription in ledger §392; not re-verified at the source, LIT_SEALED_WEB §3].
- Limits: Its one row in our tables is not verified at the source. It works per network.
- Against SPECTRA: On ResNet-110 C10 at FLOPs 0.40, our frozen FLOPs agent ends at 93.80 ± 0.21 TEST 5k from 93.84, level with our rule (+0.01; val −0.01, 10k +0.00, honest −0.10; Table 1, §408), beside its 93.50 → 93.50. Allocation barely matters at that point: our rule is only +0.14 above the even cut.

### LeGR (CVPR 2020): Towards Efficient Model Compression via Learned Global Ranking
- Link: https://arxiv.org/abs/1904.12368
- Kind: comparator we benchmark against
- What it does: Learns one global filter ranking per network by an evolutionary search and reuses it at every FLOPs target, where AMC and MorphNet search per target [V §4.2; draft §5.6]. It fine-tunes for 400 epochs on CIFAR-10 and 60 on ImageNet [V].
- Results: [V] ResNet-56 C10 93.9 → 93.7 ± 0.2 at FLOPs 0.47 (94.1 at 0.70); ResNet-50 ImageNet 76.1 → 75.3 at 0.47 and 75.7 at 0.58; MobileNetV2 ImageNet 71.8 → 71.4 at 0.70. WIN on search cost: "LeGR is 7× and 5× faster than AMC and MorphNet" [V]. Its re-run of AMC finds "higher variances when pruned more aggressively" [V].
- Limits: Per network, with long fine-tunes; its CIFAR-100 results are curves only [V].
- Against SPECTRA: On ImageNet ResNet-50 our three arms "sit with LeGR's rows" (agent 75.24, rule 75.33, even cut 75.42 top-1 at FLOPs 0.48; 75.73, 75.89, 75.66 at 0.56; one seed; Table 1c, §430), beside its 75.3 and 75.7. Its pipeline takes 167 min on ResNet-56 (10 min of search, 157 of fine-tune) against our 7.7–8.5 (§419).

### TPP (ICLR 2023): Trainability Preserving Neural Pruning
- Link: https://arxiv.org/abs/2207.12534
- Kind: comparator we benchmark against
- What it does: Adds a regulariser meant to keep the network trainable while filters are removed (by its title [H]), at hand-set per-stage ratios that skip layers "following the setting of L1 (pruned-B) Li et al. 2017" [V, its Table 7].
- Results: TIE at a mild cut: on ResNet-56 C10 at 1.99×, L1 one-shot 93.51 against TPP 93.46 [V]; WIN only at 11–19× [V]. ResNet-50 ImageNet 76.13 → 75.60 at FLOPs 0.433 [V]. VGG-19 C100: −6.24 at 0.113 and −0.83 at 0.508 [R: repo; not re-verified at the source].
- Limits: The allocation is set by hand. Its two VGG-19 C100 rows in the draft are untraced to a source reading in our scans (LIT_SEALED_WEB §3).
- Against SPECTRA: On VGG-19 C100 at 2×, our arms keep their accuracy (10k +0.16 agent, +0.41 rule, −0.15 even cut; agent − rule −0.25 TEST 5k, level; Table 1b, §418), beside its −0.83. At 8.9× our arms (10k −7.26, −6.73, −7.38) sit near its −6.24.

### GReg (ICLR 2021): Neural Pruning via Growing Regularization
- Link: https://arxiv.org/abs/2012.09243
- Kind: comparator we benchmark against
- What it does: Grows an L2 penalty on the filters to be removed before cutting them, at hand-set per-stage ratios (for example [0, 0.75, 0.75, 0.32] on ResNet-56 at 2.55×) [V, its Table 6].
- Results: WIN at deep cuts against uniform L1 one-shot on VGG-19 C100: 67.35 against 66.05 at 8.84×, 63.55 against 61.59 at 17.3×, 57.09 against 51.36 at 44× [V]. TIE at a mild cut: ResNet-56 C10 at 1.99×, 93.06 against 92.97 [V]. Headline rows [V]: VGG-19 C100 74.02 → 67.75 (GReg-2, −6.27; printed 6.47) at 8.84×; ResNet-56 C10 93.36 → 93.36 at 2.55×; ResNet-50 ImageNet 76.13 → 75.36 at FLOPs 0.433.
- Limits: Its printed VGG-19 C100 drops are 0.20 larger than its own accuracies give [V]. The allocation is set by hand.
- Against SPECTRA: On VGG-19 C100 at 8.9×, our arms lose about 7 points at 10k (agent −7.26, rule −6.73, even cut −7.38), near its −6.27; at 2× they keep their accuracy (10k −0.15 to +0.41), beside GReg-1's −0.94 at 0.521 (Table 1b, §418).

### EigenDamage (ICML 2019): EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis
- Link: https://arxiv.org/abs/1905.05934
- Kind: comparator we benchmark against
- What it does: Computes a second-order (Hessian-based) saliency in a Kronecker-factored eigenbasis and prunes structures there, in one pass or iteratively [V abstract].
- Results: WIN against Kron-OBD and Kron-OBS on VGG-19 C100: 73.34 → 65.18 (−8.16) at 8.80×, against 60.70 and 60.66 at 5.7–6.1× [V, as reprinted in GReg Table 2 and DepGraph Table 1]. Its own table, VGG-19 C10 (mean of 5 runs): 94.17 → 94.15 at FLOPs 0.719, down to 92.29 at 0.135 [V].
- Limits: Our scans read its −8.16 only through reprints, not its own table. Its ResNet-32 is widened ×4 [V].
- Against SPECTRA: On VGG-19 C100 at 8.9×, our arms (10k −6.73 to −7.38) sit above its −8.16 at 0.114 (Table 1b, §418), beside it and never a beat.

### SOSP (ICLR 2022): SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning
- Link: https://arxiv.org/abs/2110.11395
- Kind: comparator we benchmark against
- What it does: Computes a second-order saliency over all structures at once, so each layer's share falls out of one global ranking [V abstract].
- Results: [V] VGG-Net C100 73.45 → 72.94 (−0.51) at FLOPs 0.484; ResNet-56 C10 (SOSP-H) 93.27 at 0.432 / params 0.388; MobileNetV2 ImageNet 71.04 (gap 0.84) at 0.71. Its Table 10 shuffles which structures go inside each layer at fixed counts and finds "the dominant factor is the architecture and not the specific values of the weights" [V]. VGG-19 C100: −9.21 at 0.137, at its final epoch [R: repo; not re-verified].
- Limits: It claims to "match or beat numerous SOTA" without CHIP or DepGraph in its tables [V]. Its ResNet-56 base prints as 93.88, while its gaps imply 93.78 [V].
- Against SPECTRA: On VGG-19 C100 at 2×, our arms keep their accuracy (10k +0.16, +0.41, −0.15), beside its −0.51; at 8.9× ours (10k −6.73 to −7.38) sit above its −9.21, whose FLOPs (0.137) differ from ours by more than 10 % (Table 1b, §418).

### Li et al. 2017 (ICLR 2017): Pruning Filters for Efficient ConvNets
- Link: https://arxiv.org/abs/1608.08710
- Kind: comparator we benchmark against (the origin of our rules)
- What it does: Ranks filters by L1 norm and measures one sensitivity curve per layer (each layer pruned alone), then sets one ratio per stage by hand, skipping sensitive layers [V §3.2]. The single-global-threshold form comes from tools (PaddleSlim, NNI), not from the paper [V code].
- Results: [V, its Table 1] ResNet-56-B 93.04 → 93.06 at FLOPs 0.724; ResNet-110-B 93.53 → 93.30 at 0.614; VGG-16-A 93.25 → 93.40 at 0.658. TIE with a modern fine-tune: re-run with LR 0.01 for 90 epochs, L1 matches top-venue methods on ResNet-50 (75.24 at 2.31×; Wang et al. 2023) [V]. LOSS as others ran it: −1.01 at 0.508 on VGG-19 C100 in TPP's table [R], and −14.6 top-5 at 5× on VGG-16 ImageNet in Channel Pruning's re-implementation [V].
- Limits: It reports the best test/validation epoch during retraining [V]. AMC objects that it "assumes that errors of different pruned layers can be summed up linearly" [V].
- Against SPECTRA: L1 is our ranking inside each group, and our threshold rule (its single-threshold form) sits +0.64 [+0.26, +1.06] TEST 5k above our FLOPs rule on the transfer program's 13 held-out CIFAR-100 networks at FLOPs 0.45 (val +0.90, 10k +0.77; §440). Our frozen rule-blending agent is level with the threshold rule there (+0.00 [−0.32, +0.44]; val −0.06, 10k −0.03; §441).

### Jiang et al. (IJCAI 2022): On the Channel Pruning using Graph Convolution Network for Convolutional Neural Network Acceleration
- Link: https://doi.org/10.24963/ijcai.2022/431 (web: resolves to IJCAI-22, pp. 3107–3113)
- Kind: comparator we benchmark against
- What it does: Uses a graph convolution network over the CNN to decide channel pruning; the repo records it as a "GCN agent" with 30 epochs of fine-tuning after each pruned layer [R: repo].
- Results: [R: the repo's transcription; not re-verified at the source] ResNet-56 C100 72.86 → 72.86 (0.00) at FLOPs 0.512 and → 72.22 (−0.64) at 0.478; on Tiny-ImageNet, with a VGG-16 that has 4096-wide FC layers, −1.04 at 0.621 and −1.55 at 0.479.
- Limits: Untraced to a source reading in our scans. Its Tiny-ImageNet network differs from ours, so that cell is not a SOTA cell (draft claims guard).
- Against SPECTRA: On ResNet-56 C100 at FLOPs 0.48, our frozen FLOPs agent is level with our rule (71.86 ± 0.39 TEST 5k from 73.18; Table 1b, §418), and the agent's 10k change (−1.28) sits below Jiang et al.'s 0.00 and −0.64. On VGG-16-BN Tiny-ImageNet the even cut beats both our rule and our agent at both points (§418), which the draft treats as coverage, not a SOTA cell.

### TAS (NeurIPS 2019): Network Pruning via Transformable Architecture Search
- Link: https://arxiv.org/abs/1905.09717
- Kind: comparator we benchmark against
- What it does: Searches each layer's channel count (and the depth) as a learned distribution, then trains the pruned network by knowledge distillation from the unpruned one [V abstract].
- Results: [V] ResNet-56 C100 72.25 (drop 0.93) at FLOPs 0.487, with distillation. In its own ablation on ResNet-32 C100 at about 40 % FLOPs cut, TAS reaches 69.70 against uniform 69.34 and random search 69.14 from unpruned weights, while keeping about 3 % more FLOPs; distillation adds about a point [R: repo].
- Limits: Its reprints of SFP and FPGM on ResNet-56 C10 carry drops that do not match the originals, and its SFP ResNet-110 row has no source in SFP [V].
- Against SPECTRA: On ResNet-56 C100 at FLOPs 0.48, our frozen FLOPs agent's 10k change (−1.28; level with our rule at 71.86 ± 0.39 TEST 5k from 73.18; Table 1b, §418) sits below TAS's −0.93. TAS distils during its recovery; ours does not, and the draft requires saying so beside its row.

### LAMP (ICLR 2021): Layer-adaptive Sparsity for the Magnitude-based Pruning
- Link: https://arxiv.org/abs/2010.07611
- Kind: comparator we benchmark against (run in our loop as a rule)
- What it does: Rescales weight magnitudes within each layer so that one global threshold sets every layer's sparsity, for weight (unstructured) pruning [V abstract].
- Results: WIN against global magnitude, uniform, uniform+ and Erdős–Rényi-kernel allocations, most at high sparsity [R: LIT_RULE_AND_COMPARATORS §1.3]. In PruningBench's VGG-19 C100 8× table: 73.87 → 69.91 (−3.96) [R: repo].
- Limits: It is built for weights, not coupled channel groups. Its abstract notes that layer sparsities are otherwise chosen "algorithm-by-algorithm", by heuristics or search [V].
- Against SPECTRA: Our structured one-shot adaptation of LAMP (not its authors' pipeline) falls far below our FLOPs rule, −6.63 [−12.53, −1.13] TEST 5k at FLOPs 0.45 (val −6.32, 10k −6.48) and −7.57 at 0.20, collapsing on the residual networks (§442; reported, never a call). The global magnitude ranking, run the same way, is +0.37 [−0.65, +1.30] above our FLOPs rule at 0.45 and −0.35 below the threshold rule (§442).

### SPDY (ICML 2022): SPDY: Accurate Pruning with Speedup Guarantees
- Link: https://arxiv.org/abs/2201.13096
- Kind: comparator we benchmark against (its allocation form, run in our loop)
- What it does: Allocates per-layer sparsity to a speed-up target by dynamic programming over per-layer error scores, then corrects those scores by a local search, for weight pruning of ImageNet CNNs and BERT [V abstract].
- Results: It compares against uniform and AMC-style allocations [R: LIT_RULE_AND_COMPARATORS §1.3]; per the draft, its programme over per-layer losses falls below uniform in unstructured pruning of BERT [R: draft §5.9f, §6]. No numbers in our scans.
- Limits: Weight pruning, with no CNN channel row in our scans. Its local search exists because per-layer errors do not add up, the point EvoPress makes for LLMs [V; H].
- Against SPECTRA: Our implementation of its knapsack form (damage curves with one multiplier, also the form of Chen et al.'s FLOPs utilization ratio [154]), not SPDY's pipeline, is below our cost-aware rule at 25 % and 15 % params: −3.60 [−8.85, +0.16] and −2.93 [−6.53, −0.13] TEST 5k (val −3.49, −2.94; §424), losing 10–21 points on the two fragile networks. The same curves with one threshold (Li et al.'s rule) are level with our rule (+0.23, +0.43; §424).

### MetaPruning (ICCV 2019): MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning
- Link: https://arxiv.org/abs/1903.10258
- Kind: comparator we benchmark against (named in the draft's feature table, §2.2.3)
- What it does: Learns a PruningNet for the channel configurations of a target network family (draft §2.2.5), then searches configurations scored with the PruningNet's weights, with no fine-tuning per candidate [V abstract, LIT_DB_AVENUES §5].
- Results: [V] WIN on MobileNetV1 0.25×: 57.2 against uniform's 50.6. ResNet-50 ImageNet (trained from scratch, base 76.6): 75.4 at 2.0 G (about 0.49 FLOPs kept) and 73.4 at 1.0 G; MobileNetV2 71.2 at 217 M.
- Limits: It needs a PruningNet per network family, rebuilt for each new target (draft §2.2.6). Its ResNet-50 rows are trained from scratch, so they mix allocation and recipe [H].
- Against SPECTRA: Not in the draft's accuracy tables; the draft's §6 groups it with AMC and random best-of-N as per-network search. Its ResNet-50 point near FLOPs 0.49 is one of the verified points the comparator scan suggests quoting (LIT_RULE_AND_COMPARATORS §4).

### SPA (arXiv 2024): Structurally Prune Anything: Any Architecture, Any Framework, Any Time
- Link: https://arxiv.org/abs/2403.18955 (web: still a preprint under review; no venue found)
- Kind: comparator we benchmark against (named in the draft's feature table, §2.2.3)
- What it does: Builds coupled-channel groups from ONNX graphs for any architecture and framework and ranks them by group-level importance. Its OBSPA variant prunes after training, with no fine-tuning [V full; draft §2.2.4].
- Results: LOSS without fine-tuning: OBSPA loses 0.80–3.73 points at 1.5–1.7× on CIFAR ResNet-50 and VGG-19 [V]. It positions itself beyond DepGraph and OTO-v2 (draft §2.2.4).
- Limits: Its generality is the tool's (grouping), not a transferred policy's. "Any time" means any training stage, not an unseen dataset (draft §2.2.4).
- Against SPECTRA: No shared cell. The draft leaves the grouping layer to DepGraph and SPA and claims a transferred allocation policy instead (§1, §2.2.4).

**Not carded.**
- Named only in the draft's proposal-era feature table (§2.2.3): AFP [19], GDP [20], DeepPruningES [30], FPAC [37] and Amelio et al.'s multi-layer ResNet compression [34]. GAL [166] appears only as the VGG-16 base lineage, and Chen et al.'s Network Adjustment [154] only as a precedent in §5.9f. Our scans hold no numbers for them, and the draft has no shared cell.
- Comparators a reviewer will expect but the draft's tables lack, with verified points in `LIT_RULE_AND_COMPARATORS_21NOV.md` §3–§4: AMC's accuracy rows, ResRep, ABCPruner, ThiNet, He et al.'s Channel Pruning, HALP and HAP. EagleEye, Isomorphic Pruning and FAIR-Pruner are carded in group 2.

---

## Group 2. Closest to SPECTRA in approach (26 cards and one pointer)

### NEON (Information Sciences 2022): Multi-objective pruning of dense neural networks using deep reinforcement learning
- Link: https://doi.org/10.1016/j.ins.2022.07.134
- Kind: close to SPECTRA in approach (its predecessor)
- What it does: A policy-gradient (REINFORCE) agent walks a dense network layer by layer, choosing each layer's kept size from {1, 0.9, 0.8, 0.7, 0.6} over four passes, rewarded for compression within the user's permissible accuracy loss C. It is trained offline on random dense networks over 28 tabular datasets and applied to unseen datasets [V, LIT_ACCURACY_TOLERANCE].
- Results: WIN: its configurations (C = 0, 1, 5, 50 %) form a Pareto front against L1, look-ahead pruning, AMC, ADMM and a random agent [R: GILAD_DIRECTIVES §2]. It stays within the user's bound on 82–100 % of datasets: at C = 5 and 50 no dataset exceeded it, and at C = 0 and 1, 5 and 3 of 28 did [V p. 2, p. 16].
- Limits: Dense networks on tabular data only, and one trained configuration per C, which is not an input of the policy [V; H]. Its claim to be "the first pruning approach to offer this capability" does not hold against its own reference, Huang et al. 2018, which bounds the accuracy drop in an RL reward [V].
- Against SPECTRA: No shared cell. SPECTRA keeps NEON's frozen generic agent and replaces its walk, per-step fine-tune, actor-critic training and mixed reward (draft §6); the preference knob returns as our accuracy-limit mode, where our agent stops at the same size as our rule (chosen size minus the rule's −0.014, −0.024 and −0.014 at limits of −0.5, −1.0 and −2.0 val points; §423).

### AMC (ECCV 2018): AMC: AutoML for Model Compression and Acceleration on Mobile Devices
- Link: https://arxiv.org/abs/1802.03494
- Kind: close to SPECTRA in approach (also timed in our cost table)
- What it does: A DDPG agent walks one given network layer by layer and sets each layer's compression under a resource budget, rewarded by the cut network's accuracy before any fine-tune. It searches per network, with 100 exploration and 300 exploitation episodes [V §3].
- Results: WIN on ResNet-56 C10 at 50 % FLOPs: 92.8 → 91.9 (−0.9), +2.1 over uniform (89.8) and +0.4 over the hand "deep" policy (91.5) [V, its Table 2; c]; on VGG-16 ImageNet at 5×, −1.4 against Channel Pruning's hand-tuned −1.7 (top-5 error increase) [V]; MobileNetV2 ImageNet 71.8 → 70.8 at 0.70 [V]. LOSS on ImageNet, where accuracy before retraining "cannot provide useful supervision" [V], and in LeGR's re-run, with "higher variances when pruned more aggressively" [V].
- Limits: A search per target network. Reprints misquote it (GNN-RL lists its pre-fine-tune 90.20 as its result), and its own MobileNet change reads −0.4 in its Table 3 but −0.1 in Table 4 [V].
- Against SPECTRA: Timing only: LeGR's AMC implementation, run by us on ResNet-56, takes 109 min (85 of search, 24 of a 60-epoch fine-tune) against our 7.7–8.5 (§419). Its accuracy row is not in the draft's tables, the first gap a reviewer will name (LIT_RULE_AND_COMPARATORS §4).

### GNN-RL (ICML 2022): Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning
- Link: https://arxiv.org/abs/2102.03214
- Kind: close to SPECTRA in approach (also in Table 1c and the cost table)
- What it does: Encodes the network as a multi-stage graph with a graph neural network and trains an RL agent to set per-layer compression for that network. In a transfer run, the encoder trained on ResNet-56 was frozen and the policy head retrained on ResNet-44 by continued RL search [R: repo].
- Results: [V] ResNet-56 C10 93.49 (+0.10) at FLOPs 0.46; ResNet-110 94.31 (+0.63) at 0.48; ResNet-50 ImageNet 74.28 (−1.82) at 0.47; 100 fine-tune epochs on CIFAR-10. Its ResNet-44 transfer run: 93.10 → 93.23 at 0.49 [R: repo].
- Limits: Its AMC rows quote AMC's pre-fine-tune accuracy and turn a top-5 change into top-1 [V]. Its search has no stop rule (draft §5.6).
- Against SPECTRA: On ImageNet ResNet-50 at FLOPs 0.48, our three arms (75.24–75.42 top-1, one seed; Table 1c, §430) sit beside its 74.28, never a beat. Its search had not finished after 17 h on our GPU (10,390 of 15,000 episodes; its best came 9.3 min in), against our 7.7–8.5 min per network (§419).

### RL-Pruner (arXiv 2024): RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration
- Link: https://arxiv.org/abs/2411.06463 (web: no venue found)
- Kind: close to SPECTRA in approach (also timed in our cost table)
- What it does: Learns one CNN's layer-wise pruning distribution by RL over many small pruning steps, rewarded by the cut network's accuracy before recovery (apparently on the test set), with optional distillation between steps [V full].
- Results: Five CIFAR-100 networks; "the entire pruning process takes several hours" on one RTX 4090D [V]. Its accuracy numbers were not extracted in our scans [U].
- Limits: It works per network. Its reward is read before recovery, apparently on test [V].
- Against SPECTRA: No accuracy row in the draft; its whole run on ResNet-56 took 271 min on our GPU against our 7.7–8.5 (§419). The draft's reference list lacks it (LIT_RULE_AND_COMPARATORS §0).

### Generalist agent (IEEE Access 2024): A Generalist Reinforcement Learning Agent for Compressing Convolutional Neural Networks
- Link: https://doi.org/10.1109/ACCESS.2024.3385857
- Kind: close to SPECTRA in approach
- What it does: An RL agent picks, per layer, one of ten AdaDeep compression techniques (SVD, depthwise replacement, sparse random connections and others). Trained on LeNet for Fashion-MNIST and KMNIST, it is applied frozen to LeNet on MNIST [V full].
- Results: WIN: under 15 % of the size at under 2.5 % accuracy loss on MNIST [V]; the authors also enumerated all 119 action sequences for the three compressed layers [V]. A companion paper reuses the agent for per-layer SVD ranks on unseen models (doi:10.1109/ACCESS.2024.3457863) [V abstract].
- Limits: One small architecture and three layers. It transfers across datasets, not across architectures [V].
- Against SPECTRA: No shared cell. It is the only frozen per-layer selector found; SPECTRA's frozen agent is read across families and datasets instead (draft §2.2.6, refs [129], [130]).

### Balaskas et al. (IEEE TETC 2024): Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization
- Link: https://arxiv.org/abs/2312.15322 (doi:10.1109/TETC.2023.3346944; volume 12, 2024, per OpenAlex, web)
- Kind: close to SPECTRA in approach
- What it does: An RL agent chooses, per layer, among fine- and coarse-grained pruning algorithms and a bit-width, with no retraining, per network (1,100 episodes) [V full, §3 and §5].
- Results: 39 % average energy reduction at 1.7 % average accuracy loss [V]. On MobileNetV2 C10 at a 1 % loss bound, fine-grained "Level" pruning reaches 50 % sparsity against 10 % for coarse "L1-Ranked" [V]. It compares with NSGA-II at equal evaluations; no ablation against the best single technique was found [V].
- Limits: Per network, with no recovery, and centred on energy. It has no single-rule baseline.
- Against SPECTRA: No shared cell.

### MDP (CVPR 2025): MDP: Multidimensional Vision Model Pruning with Latency Constraint
- Link: https://arxiv.org/abs/2504.02168
- Kind: close to SPECTRA in approach
- What it does: One mixed-integer nonlinear program decides channels and whole blocks together (heads and embeddings for ViTs) under a latency lookup table, per network, then fine-tunes [V full].
- Results: WIN on ResNet-50 ImageNet (TITAN V, batch 256): 70.0 at a ×5.16 latency speed-up against HALP's 68.6 at ×4.02; 72.8 at ×4.13 against SMCP's 72.7 at ×3.71; 74.8 at ×3.06 against LayerPrune's 74.3 at ×1.79 [V]. Mixing granularities beats channel-only pruning "particularly at high pruning ratios" (a figure only) [V].
- Limits: These are latency wins, not FLOPs wins. Each network needs its own solver run.
- Against SPECTRA: No shared cell. Our budgets are FLOPs and params, and on our GPU fewer FLOPs are not faster for the small CIFAR ResNets (§395).

### LayerPrune (ACCV 2020): To Filter Prune, or to Layer Prune, That Is The Question
- Link: https://arxiv.org/abs/2007.05667 (ACCV 2020 per the CVF open-access listing, web)
- Kind: close to SPECTRA in approach
- What it does: Ranks whole layers in one shot with an existing filter criterion, removes the least important layers and fine-tunes, for latency rather than FLOPs [V full].
- Results: WIN on VGG-19-BN C100 (base 73.11) at about equal latency reduction: 73.60 at 17.34 % against Slimming's 72.32 at 16.84 % (+1.28) and Taylor's 72.61 (+0.99) [V; c]; random layer-pruned models beat random filter-pruned ones by 7.09 % at matched latency [V]. LOSS for filter pruning: weight-norm filter pruning raised latency (−2.04 %) [V].
- Limits: Latency on one GPU at batch 8. It works per network.
- Against SPECTRA: No shared cell. It agrees with our finding that fewer FLOPs need not be faster: DepGraph's ResNet-56 runs at 0.93–1.07× with 37–61 % fewer FLOPs (§395).

### Metanetwork pruning (arXiv 2025): Meta Pruning via Graph Metanetworks: A Universal Meta Learning Framework for Network Pruning
- Link: https://arxiv.org/abs/2506.12041 (web: v3 of 15 Dec 2025; no venue found)
- Kind: close to SPECTRA in approach
- What it does: A graph network, meta-trained on a set of networks, rewrites a network's weights so that a fixed group-norm criterion prunes it better; group pruning and fine-tuning follow on each target [V].
- Results: WIN on VGG-19 C100 at 8.98×: 73.65 → 68.68 ± 0.87 over three seeds [V, App. D.3]. Across architectures on CIFAR-10: ResNet-56 with a metanetwork from ResNet-110 reaches 92.81 against 92.08 without one, and ResNet-110 from ResNet-56 93.04 against 92.40 [R], 0.34–0.59 below a same-architecture metanetwork (draft §2.2.6).
- Limits: It edits weights rather than allocating, transfers within one family, and sets speed-ups by hand [R].
- Against SPECTRA: Not in the draft's tables; it is the closest frozen transfer for CNNs that the draft cites ([124], §2.2.6). Its VGG-19 C100 row falls in Table 1b's 8.9× cell (§418; see the DepGraph card).

**Gong et al. (ACM TODAES 2022): Automatic Mapping of the Best-Suited DNN Pruning Schemes for Real-Time Mobile Acceleration** (https://arxiv.org/abs/2111.11581; TODAES 27(5), 2022, per dblp, web). Pointer only: its RL search over per-layer pruning schemes ends level with a training-free rule after 3–9 days of search [V]; its card and its citers are in `LIT_GONG_CITERS_22NOV.md` (in progress).

### DominoSearch (NeurIPS 2021): DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense neural networks
- Link: https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html (no arXiv record; page checked, web)
- Kind: close to SPECTRA in approach
- What it does: Searches a per-layer mixed N:M sparsity scheme from a pretrained dense network, per network [V abstract].
- Results: WIN on ResNet-18 ImageNet: +2.1 top-1 over uniform N:M at equal size (2.1 M params, 87.5 % sparsity) and +1.3 at an equal 227 M FLOPs; ResNet-50 reaches 76.7 at 5.0 M params [V abstract].
- Limits: N:M sparsity saves FLOPs only on hardware that runs sparse kernels [H]. It is compared with uniform N:M, not with a tuned rule.
- Against SPECTRA: No shared cell.

### APQ (CVPR 2020): APQ: Joint Search for Network Architecture, Pruning and Quantization Policy
- Link: https://arxiv.org/abs/2006.08509
- Kind: close to SPECTRA in approach
- What it does: Searches architecture, pruning and quantization jointly, with an accuracy predictor inside a once-for-all supernet, per target [V abstract].
- Results: WIN: +2.3 % ImageNet accuracy over the separate ProxylessNAS + AMC + HAQ pipeline, and 2× lower latency and 1.3× lower energy than MobileNetV2 + HAQ at the same accuracy [V abstract].
- Limits: Its gain is joint over sequential search, not over a tuned rule. It works per target.
- Against SPECTRA: No shared cell.

### Blending criteria (ICANN 2021): Blending Pruning Criteria for Convolutional Neural Networks
- Link: https://arxiv.org/abs/2107.05033
- Kind: close to SPECTRA in approach
- What it does: Blends several filter criteria per layer, clusters them by rank similarity, and tunes the blend by evolutionary search with a 3-epoch fine-tune as the fitness, per network [V full].
- Results: WIN by tenths on CIFAR-100, mean of three runs: VGG-16 71.68 against 71.45 for the best single criterion (+0.23); ResNet-56 69.82 against FPGM's 69.50 (+0.32) [V]. Two fine-tune epochs rank candidates much better than the cut alone (a figure only) [V].
- Limits: Its pruning ratio was not found in the text read, and its venue is [R]. It works per network.
- Against SPECTRA: No shared cell. After our recovery the channels chosen matter little: L1 − random channels is +0.13 and +0.23 TEST 5k at the agent's widths (§398).

### LAASP (Image and Vision Computing 2023): Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural Network Acceleration
- Link: https://arxiv.org/abs/2506.20152 (doi:10.1016/j.imavis.2023.104745)
- Kind: close to SPECTRA in approach
- What it does: At each iteration picks a criterion (ℓ1, ℓ2, Euclidean or cosine) and a layer by the loss on a training subset, pruning while training [V full].
- Results: TIE on ResNet-32 C10 at 53 % FLOPs ↓: all four criteria 92.64 ± 0.09 against the best pair's 92.59 ± 0.15 (+0.05, within one SD) and the magnitude pair's 92.35 [V].
- Limits: It works per network. Its gain is inside run-to-run noise.
- Against SPECTRA: No shared cell.

### EagleEye (ECCV 2020): EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning
- Link: https://arxiv.org/abs/2007.02491
- Kind: close to SPECTRA in approach
- What it does: Samples random per-layer width plans (L1 filters inside each layer), scores each by its accuracy after re-estimating batch-norm statistics with frozen weights ("adaptive BN"), and fine-tunes the best, per network [V full, LIT_DB_AVENUES §2].
- Results: WIN as a proxy: correlation with fine-tuned accuracy 0.68–0.82 against −0.04 to 0.25 for the cut alone (MobileNetV1 ImageNet) [V], and +0.31 over L1-uniform at 26.5 M FLOPs [R]. ResNet-56 C10 93.26 → 94.66 at FLOPs 0.496 [V]; ResNet-50 ImageNet 76.4 at 2.0 G [V]; 25 GPU-h for 1,000 candidates against 864 for NetAdapt [V].
- Limits: Its correlation study covers one network [V]. It is a per-network search.
- Against SPECTRA: Not in the draft's accuracy tables. Its adaptive-BN score is our batch-norm reward (§2.2.6), which ranks our plans only weakly at params 0.35 (+0.45, and backwards on ResNet-20 ×8 C100) where one fine-tune epoch is valid (§375).

### RL-MCTS (WACV 2022): Channel Pruning via Lookahead Search Guided Reinforcement Learning
- Link: https://openaccess.thecvf.com/content/WACV2022/html/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.html (no arXiv or DOI found; this URL comes from a search result and returned HTTP 403 to our fetch)
- Kind: close to SPECTRA in approach
- What it does: An RL policy chooses which filters to prune, guided by Monte Carlo tree search look-ahead (gains level off beyond 10 simulations), with a 30-epoch fine-tune per segment, per network [V full].
- Results: WIN on ResNet-50 ImageNet (its own base 77.34): −0.54 at about 46 % FLOPs ↓, against −4.05 for random search with the same number of executions [V]; ResNet-56 C10 93.20 → 93.56 at 55 % ↓ [V].
- Limits: Its random search samples filter sequences, not layer ratios with a criterion [V]. It is a per-target search (draft §2.1).
- Against SPECTRA: No shared cell.

### NetAdapt (ECCV 2018): NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications
- Link: https://arxiv.org/abs/1804.03230
- Kind: close to SPECTRA in approach
- What it does: Each iteration tries every layer, removes filters until that layer saves the iteration's measured latency step, gives each proposal a short fine-tune and keeps the best; a long fine-tune ends the run [V §3.2].
- Results: [V] MobileNetV1 69.1 at 284.3 MMACs; MobileNetV2 70.9 at the latency of the 0.75× model (69.8). Without the short fine-tune, accuracy "rapidly drops to nearly zero" [V]. EagleEye puts its search at 864 GPU-h for 1,000 candidates [V].
- Limits: Per network and per budget, on one device's latency.
- Against SPECTRA: No shared cell.

### DPM (arXiv 2024): Decay Pruning Method: Smooth Pruning With a Self-Rectifying Procedure
- Link: https://arxiv.org/abs/2406.03879 (web: a preprint; no venue found)
- Kind: close to SPECTRA in approach (a deep-cut procedure)
- What it does: Replaces the one-step cut with a smooth multi-step decay of the pruned structures to zero, plus a gradient-based "self-rectifying" step, on top of DepGraph [V full].
- Results: WIN by tenths on DepGraph's VGG-19 C100 at about 9×: 70.92 against DepGraph's re-run 70.74 with sparse learning, and 67.76 against 67.58 without (+0.18 both) [V]. Its re-run credits sparse learning with +3.16 [V; c].
- Limits: One network. It changes the procedure, not the allocation.
- Against SPECTRA: Not in the draft's tables; same 8.9× cell as Table 1b (§418; see the DepGraph card).

### CHEX (CVPR 2022): CHEX: CHannel EXploration for CNN Model Compression
- Link: https://arxiv.org/abs/2203.15794
- Kind: close to SPECTRA in approach (a deep-cut procedure)
- What it does: Prunes and regrows channels throughout a 250-epoch training from scratch, re-allocating widths across layers as it goes [V full].
- Results: WIN on ResNet-50 ImageNet: 77.8 (its own base) → 76.0 at 1.0 G FLOPs, about 0.24 kept [V; c]; its ablation credits +0.9 to prune-and-regrow and +0.8 to re-allocating widths [V; c]. At 1 G the criterion matters in a pretrain-prune-finetune pipeline (CSS 74.6, magnitude 73.8, BN 73.1) and less inside its regrowth (76.0, 75.7, 75.3) [V].
- Limits: Its own 77.8 base and 250 epochs make its rows recipe-bound [V].
- Against SPECTRA: No shared cell (our ImageNet points keep 0.48 and 0.56 of the FLOPs).

### OTOv2 (ICLR 2023): OTOv2: Automatic, Generic, User-Friendly
- Link: https://arxiv.org/abs/2303.06862
- Kind: close to SPECTRA in approach (generic as code)
- What it does: Finds removable groups automatically for any architecture and trains once from scratch with a structured-sparsity optimiser (DHSPG), with no separate fine-tune [V full].
- Results: [V] ResNet-50 ImageNet 76.1 → 74.3, 72.2 and 70.1 (averages; best runs 72.3 and 70.3) at FLOPs 0.287, 0.200 and 0.145.
- Limits: It is generic as code, but each network is trained or solved anew [V].
- Against SPECTRA: No shared cell.

### OSSCAR (ICML 2024): OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization
- Link: https://arxiv.org/abs/2403.12983 (ICML 2024, PMLR 235, per search results, web)
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: One-shot structured pruning with no fine-tuning: a layer-wise reconstruction objective solved by combinatorial local search [V full].
- Results: WIN on ResNet-50 ImageNet (base 77.01) at 1.7× with no recovery: 50.16 ± 0.59 against ThiNet's 41.08 and Lasso's 41.68, while magnitude, CHIP and FPGM fall below 1; MobileNetV1 at 1.7×: 63.03 against 59.51 [V, its Table 1].
- Limits: Its win is without recovery. Reconstruction is worth points one-shot but tenths after a long recovery [H, LIT_AGENT_VS_RULE §6].
- Against SPECTRA: No shared cell: every SPECTRA row is recovered, for 100 epochs on CIFAR and 90 on ImageNet.

### Isomorphic Pruning (ECCV 2024): Isomorphic Pruning for Vision Models
- Link: https://arxiv.org/abs/2407.04616
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: Ranks prunable groups only against groups of the same sub-structure type, to avoid the over-pruning that "a naive global ranking threshold" risks [V abstract; V §4.2].
- Results: [V, its Table 1] ResNet-50 ImageNet 76.13 → 75.91 (−0.22) at FLOPs 0.499 / params 0.589; MobileNetV2 71.88 → 68.91 at 0.469; no CIFAR rows.
- Limits: It works per network. Its Table 1 gives DepGraph's ResNet-50 20.60 M params, a number DepGraph's own Table 3 does not print [V].
- Against SPECTRA: Not in the draft's tables; the nearest draft cell is ImageNet ResNet-50 at FLOPs 0.48 (Table 1c, §430, one seed).

### Relative Repairability (arXiv 2026): Relative Repairability: A Calibration-Based Diagnostic for High-Sparsity Post-Pruning Allocation
- Link: https://arxiv.org/abs/2605.25508
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: Allocates sparsity by how much of each layer's damage survives a label-free channelwise variance-matching repair on calibration data, with no fine-tune [V abstract].
- Results: TIE overall, "not a universally dominant allocation rule"; WIN over ERK across a central transition band and over LAMP near its upper part (ResNet-18 C100) [V abstract].
- Limits: Weights (unstructured) and no long recovery. A preprint.
- Against SPECTRA: No shared cell. It is the closest prior for the recovery-aware threshold rule ranked first in LIT_AGENT_VS_RULE §8.

### LADP (ICCV 2023): Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks
- Link: https://arxiv.org/abs/2308.10438
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: Shows that per-layer output distortion adds up across layers and allocates per-layer sparsity by dynamic programming [V abstract].
- Results: WIN: up to +1.0 (ResNet-32 C10) and +4.6 (ResNet-50 ImageNet) over other allocations [V abstract].
- Limits: Weights, not channels. Its venue is [R].
- Against SPECTRA: No shared cell.

### FAIR-Pruner (arXiv 2025): FAIR-Pruner: A Flexible Framework for Automatic Layer-Wise Pruning via Tolerance of Difference
- Link: https://arxiv.org/abs/2508.02291 (web: v3 of 20 May 2026, "Submitted to IEEE TPAMI")
- Kind: close to SPECTRA in approach (a threshold rule)
- What it does: One shared tolerance level sets each layer's pruning depth: the deepest cut whose "rank conflict" (the overlap of a removal ranking by Wasserstein separability with a protection ranking by Taylor sensitivity) stays within the tolerance. It is search-free [V].
- Results: TIE: it "match[es] or slightly exceed[s]" HRank and ITPruner [V]; ResNet-56 C10 93.93 → 93.64 at FLOPs 0.462, ResNet-50 ImageNet 76.88 → 75.29 at 0.472 [V].
- Limits: It never mentions DepGraph or CHIP and recomputes every competitor's drop against its own base [V]. For literature comparisons it sweeps the tolerance rather than bisecting it to a budget [V].
- Against SPECTRA: Not in the draft's tables. It is the newest threshold-style allocation and the closest modern cousin of our threshold rule, which thresholds a measured loss rise instead of a rank conflict (LIT_RULE_AND_COMPARATORS §1.5).

### CPMC (IJCNN 2021): Channel Pruning via Multi-Criteria based on Weight Dependency
- Link: https://arxiv.org/abs/2011.03240
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: Builds one fused channel score from filter weights, the next layer's dependent weights, FLOPs and params, normalised globally [V abstract].
- Results: "outperforms" (the abstract read was truncated); no numbers in our scans [V abstract].
- Limits: Not read beyond the abstract.
- Against SPECTRA: No shared cell. Like our cost-aware rule, it weighs importance against cost (draft §2.2.6).

### Group Fisher (ICML 2021): Group Fisher Pruning for Practical Network Compression
- Link: https://arxiv.org/abs/2108.00708
- Kind: close to SPECTRA in approach (a training-free rule)
- What it does: Computes Fisher importance for coupled channels and divides it by each channel's memory saving [V abstract].
- Results: An inference speed-up "without sacrificing accuracy" [V abstract]; no numbers in our scans.
- Limits: Its pruning schedule was not read [U].
- Against SPECTRA: No shared cell. The draft cites its loss-per-cost ratio as a precedent of our cost-aware rule, our rule at equal params (§2.2.6, §4.1).

**Also close, no card** (from the same scans; the draft's §2.2.6 cites most of them):
- Huang et al. (WACV 2018), https://arxiv.org/abs/1801.07365: one REINFORCE agent per layer on the target network, with a user's accuracy-drop bound in the reward [V].
- N2N learning (ICLR 2018), https://arxiv.org/abs/1709.06030: policy-gradient compression rewarded after 5 epochs of distillation; policies transferred within families (ResNet-18 ↔ ResNet-34, VGG-11 → VGG-19) [V].
- Out-of-the-box channel pruned networks (arXiv 2020), https://arxiv.org/abs/2004.14584: RL-searched layer-wise profiles reused across datasets on the same architecture [V].
- CACP (IEEE SPL 2021), https://doi.org/10.1109/LSP.2021.3088323: one AMC-style model conditioned on the compression rate, per network [V abstract].
- Mu et al. (IEEE TCAD 2024), https://arxiv.org/abs/2107.08815: RL auto-pruning warm-started from other networks' history [V].
- Balemans et al. (arXiv 2025), https://arxiv.org/abs/2509.10526: per-model graph RL that names per-model policies "a fundamental limitation" [R].
- AgenticPruner (arXiv 2026), https://arxiv.org/abs/2601.12272: a general LLM proposes per-layer ratios in a propose–evaluate loop per target [R].
- DSA (NeurIPS 2024), https://doi.org/10.52202/079017-4487: an allocation function searched once on one LLM and reused across LLM families [R].
- HiReLC (arXiv 2026), https://arxiv.org/abs/2606.26002: hierarchical RL for pruning plus quantization, with an MLP surrogate of post-fine-tune accuracy as the reward, per target [V].

---

## Group 3. Evaluation practice (9 cards)

### Blalock et al. (MLSys 2020): What is the State of Neural Network Pruning?
- Link: https://arxiv.org/abs/2003.03033
- Kind: evaluation practice
- What it does: A meta-analysis of 81 pruning papers, with ShrinkBench, an open library with magnitude, gradient and random baselines [V].
- Results: LOSS for the field: "the community suffers from a lack of standardized benchmarks and metrics", most papers use at most three operating points, and method rankings flip between CIFAR-VGG and ResNet-56 [V]. WIN for allocation: pruning beats random "at least for large amounts of pruning", and pruning all layers uniformly "tends to perform worse than intelligently allocating parameters to different layers … or pruning globally" [V].
- Limits: Mostly unstructured pruning. Its control is the unpruned model before fine-tuning, not the unpruned model fine-tuned the same way [V].
- Against SPECTRA: Our test of its random baseline at the agent's own widths: L1 − random channels is +0.13 [−0.13, +0.34] TEST 5k at 45 % params and +0.23 [−0.08, +0.59] at 25 % (val +0.02, +0.24), so the channels chosen matter little (§398). The draft cites it in §5.6, but its reference list lacks it (LIT_RULE_AND_COMPARATORS §0).

### Liu et al. 2019 (ICLR 2019): Rethinking the Value of Network Pruning
- Link: https://arxiv.org/abs/1810.05270
- Kind: evaluation practice
- What it does: Trains pruned architectures from scratch, for equal epochs (Scratch-E) and for equal compute (Scratch-B), against the same architectures fine-tuned from inherited weights [V].
- Results: TIE scratch against inherited: "the pruned architecture itself, rather than a set of inherited 'important' weights, is more crucial" [V]. Network Slimming's learned allocation beats uniform on VGG but is "not significantly more efficient" on ResNet and DenseNet [V]. Fine-tuning ResNet-56-B for 40, 80 or 160 epochs gives 92.68, 92.67 and 92.76 [R].
- Limits: Uniform scaling at the same depth only, with 2018 recipes [V].
- Against SPECTRA: On DepGraph's ResNet-56, our agent's architectures trained from scratch with Scratch-B's compute match their inherited weights (inherited − scratch −0.23 and −0.03 TEST 5k), and Scratch-E's 200 epochs fall behind at params 0.38 (+0.39 for the inherited weights; val +0.61, 10k +0.50) (§394).

### Li et al. 2022 (CVPR 2022): Revisiting Random Channel Pruning for Neural Network Compression
- Link: https://arxiv.org/abs/2205.05676
- Kind: evaluation practice
- What it does: Samples 100 random per-layer ratio vectors, fine-tunes the top 5 for 5 epochs, and proposes the best as a baseline for channel pruning; it also compares criteria under random ratios [V].
- Results: TIE random against searched: random ratios find "comparable or even superior" models, within 0.5 % of architecture-optimising methods on ImageNet, and under random ratios recent criteria are "just comparable" with L1 and L2 [V]. Longer fine-tunes matter: going from 25 to 100 epochs lowers ResNet-50's error by 0.75 [R].
- Limits: It works per network, and no per-plan results are released [V].
- Against SPECTRA: Against our implementation of its best-of-N (100 random plans per network in our action space, scored by our 32-step fine-tune reward), our frozen agent ends +0.51 [+0.16, +0.91] TEST 5k above it at 45 % params (val +0.50, 10k +0.51, honest +0.47; 8 of 9 networks) and +0.45 [−0.61, +1.31] at 25 % (6 of 9), while the search is level with our cost-aware rule (−0.09, −0.29) (§396). Never "beats Li et al.": this is our implementation of their baseline.

### PruningBench (arXiv 2024): A Comprehensive Study of Structural Pruning for Vision Models
- Link: https://arxiv.org/abs/2406.12315 (web: v7 of 21 Jan 2025; no venue found)
- Kind: evaluation practice
- What it does: Benchmarks 16 structural pruning methods under one fixed fine-tune (100 epochs on CIFAR-100, 90 on ImageNet) on ResNet-18/50, VGG-19, ViT-Small and YOLOv8: 645 pruned models [V full].
- Results: TIE among methods: "no single method consistently outperforms the others", and plain L1 or L2 magnitude sits in the top 5 of most rankings [V]. WIN for protected global pruning (at least 10 % kept per group) at 4–8×, where plain global pruning suffers "layer collapse" [V]; sparsity regularisers help in only 57.30 % of cases [V].
- Limits: Five models, no learned allocator, no held-out architecture, unstated checkpoint selection and no variance [V].
- Against SPECTRA: No row in the draft, which does not cite it; our global-ranking and LAMP adaptations run in our own loop instead (§442).

### Wang et al. 2023 (arXiv 2023): Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning
- Link: https://arxiv.org/abs/2301.05219
- Kind: evaluation practice
- What it does: Audits how pruning papers set up their comparisons and shows that the fine-tune learning-rate schedule is the factor most of them ignore [V full].
- Results: TIE, old baseline against new methods: L1-norm pruning re-run with step LR 0.01 and a 90-epoch fine-tune reaches 75.24 at 2.31× and 74.77 at 2.56× on ResNet-50, level with or above GReg (75.16, 75.36), CC (75.59 at 2.12×), HRank (74.98 at 1.78×) and LFPC (74.46 at 2.55×) [V]. The learning rate alone moves a 95 %-pruned ResNet-34 on ImageNet-100 from 29.89 to 75.20 [V].
- Limits: No dense control under the same fine-tune [V]. No venue on its arXiv record.
- Against SPECTRA: No shared cell. Its "fairness principle" (a method's advantage must come from the pruning stage, not from pretraining or the fine-tune) is what our protocol of one recovery for every arm enforces (§4.2).

### Jin et al. (NeurIPS 2022): Pruning's Effect on Generalization Through the Lens of Training and Regularization
- Link: https://arxiv.org/abs/2210.13738
- Kind: evaluation practice
- What it does: Trains the dense network for the pruning algorithm's epochs and learning-rate schedule as a control, to separate the effect of pruning from that of extra training [V full].
- Results: TIE: the dense control "matches or exceeds" the optimally sparse model without label noise; test error, dense against learning-rate rewinding: 29.0 vs 29.9 (ResNet-32 C100), 7.6 vs 7.7 (ResNet-20 C10), 6.0 vs 5.9 (VGG-16 C10), 23.4 vs 23.4 (ResNet-50 ImageNet) [V].
- Limits: Unstructured magnitude pruning, framed as generalisation rather than as an audit of compression claims [V].
- Against SPECTRA: Our "honest" change is this control for structured pruning: on ImageNet the unpruned ResNet-50 gains +0.78 [+0.52, +1.04] under the same 90 epochs, so honest is −1.48 to −1.66 at 2.08× (Table 1c, §430). On CIFAR our recovery lifts the unpruned ResNet-56 by +0.64 to +1.38 (§382).

### Le & Hua (ICLR 2021): Network Pruning That Matters: A Case Study on Retraining Variants
- Link: https://arxiv.org/abs/2105.03193
- Kind: evaluation practice
- What it does: Compares retraining schedules after pruning: fine-tuning, learning-rate rewinding, and cyclic or large learning rates [V full].
- Results: WIN for large-LR retraining over plain fine-tuning, and "Random Pruning + CLR could surpass" sophisticated criteria [V]; with the allocation fixed, random channels plus a large-LR schedule give 92.40 ± 0.16 against 92.32 on ResNet-56 [R].
- Limits: No dense network is retrained with the same schedule [V].
- Against SPECTRA: No shared cell. Our recovery follows it: learning rate 0.1, cosine, 100 epochs, last epoch kept, for every arm (§4.2).

### Oracle pruning (TMLR 2026): Is Oracle Pruning the True Oracle?
- Link: https://arxiv.org/abs/2412.00143
- Kind: evaluation practice
- What it does: Tests, over about 37K models, whether the loss right after pruning predicts the accuracy after retraining (120 epochs on CIFAR, 30 on ImageNet) [V full].
- Results: LOSS for cut-time scores: Kendall τ 0.02 (ResNet-56 C10), 0.01 (VGG-19 C100) and 0.19 (ResNet-18 ImageNet), against its own bar of 0.2 [V]. "Only 10 % retraining" correlates highly with full retraining (a figure only) [V].
- Limits: It ranks masks at fixed layer ratios, not width plans, and uses no batch-norm recalibration [V].
- Against SPECTRA: Our proxy study agrees for width plans: at params 0.35 the raw cut ranks plans weakly (+0.41, and −0.84 on ResNet-20 ×8 C100), while one fine-tune epoch is valid (+0.60) (§375).

### Random tickets (NeurIPS 2020): Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot
- Link: https://arxiv.org/abs/2009.11094
- Kind: evaluation practice
- What it does: Runs sanity checks on methods that prune at initialisation [V abstract].
- Results: TIE, random against chosen weights: changing which weights are kept inside each layer, at fixed per-layer counts, "does not affect the final performance", and data-independent layer ratios with random pruning match "initial tickets" [V abstract].
- Limits: Unstructured pruning, at initialisation.
- Against SPECTRA: It agrees with our control that the widths, not the channels kept, carry the result (L1 − random channels +0.13 and +0.23 TEST 5k; §398).

**Also relevant, no card** (same scans):
- Renda et al. (ICLR 2020), https://arxiv.org/abs/2003.02389: learning-rate rewinding often ends above the unpruned network [V].
- Zimmer et al. (ICLR 2023), https://arxiv.org/abs/2111.00843: the retraining budget and schedule set the size of a pruning lever [V].
- ThinResNet (arXiv 2023), https://arxiv.org/abs/2309.12854: width-scaled ResNets under a modern recipe "clearly and utterly outperform" the pruning results it compares with [V].
- Crowley et al. (arXiv 2018), https://arxiv.org/abs/1810.04622: at matched params, reduced networks beat pruned-and-tuned ones on WRN and DenseNet on C10 [V].
- Pruning from Scratch (AAAI 2020) and EKG (ECCV 2022), the draft's [110] and [111]: sensible allocations end within 0.15–0.31 points of each other under one recovery (draft §2.2.6) [R: repo].
