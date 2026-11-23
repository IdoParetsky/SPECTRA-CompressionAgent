# Short recovery after structured pruning on ImageNet: prior art, odds and a recipe (23 Nov 2026)

Compiled 23 Nov 2026 (IST) by the literature subagent, before any GPU study of short recovery. Read-only everywhere except this file.

**Labels.** [V] read at the source in this scan (arXiv abstract page, ar5iv HTML, arXiv HTML, a proceedings PDF, or a repository page). [R] second-hand: our earlier scans (`docs/paper/lit/*.md`, `docs/paper/EFFICIENCY_AND_TRANSFER.md`), another paper's reprint, or a search snippet. [c] my arithmetic. [H] my interpretation. [U] not established.

**Our own evidence** appears only as context, by ledger section (§N), never as literature. On ImageNet, TEST is top-1 on the ImageNet validation set and "val" is 10,000 images held out of the training split (§427).

**Words.** "Recovery" is the fine-tune after channels are cut. "2x" means about half the FLOPs kept (FLOPs kept 0.48 to 0.52). "Plan" is the per-layer width allocation. "Our frozen agent" was trained once on CIFAR-style networks and is applied with no per-network search. "Our rule" is the sensitivity rule; "the even cut" keeps one fraction in every group. "Honest" is our change minus the unpruned network's own change under the same recovery.

---

## 1. Verdict in plain English

1. Short recoveries are not new. The first ImageNet channel-pruning papers fine-tuned for only 10 to 25 epochs (He et al. 2017: 10; Li et al. 2017: 20; Molchanov et al. 2019: 25 in all), and later audits showed that those short runs left accuracy behind.
2. Only one paper measures the same kind of pruned ResNet-50 after a short and a longer recovery. Li et al. (CVPR 2022) found that, at about 2x fewer FLOPs, 25 epochs end 0.63 top-1 below 75 epochs. A separate ablation puts 25 epochs 0.75 below 100, at a FLOPs setting the paper does not state.
3. The closest modern study (Zimmer et al., ICLR 2023) recovers uniformly cut ResNet-50s for 2, 5 and 10 epochs. It shows that the learning-rate schedule matters most when recovery is short: 72.3 against 71.2 after 10 epochs at a 30 % filter cut. It runs no 90-epoch anchor, and its 10-epoch numbers sit several points below the 75 to 76 that 90-epoch pipelines report at a similar size.
4. So "10 to 20 epochs within about 0.5 point of the 90-epoch result at 2x" has not been shown. The nearest published case is 50 epochs plus distillation from the unpruned network, landing at the level of 90- to 180-epoch methods (76.2 from 76.15 at about 2x, in a preprint).
5. The per-layer plan clearly matters more when recovery is absent or tiny. In language models, in few-shot CNN pruning and in our own one-epoch CIFAR read, plan choices are worth whole points; after long recovery they shrink to tenths. Nobody has measured where 10 to 20 ImageNet epochs fall between those two ends.
6. Short runs rank candidate pruned networks moderately well: rank correlations of about 0.5 to 0.7 where anyone measured them. No study forecasts the final ImageNet accuracy of a pruned network from a short run with a stated error.
7. What would be new for SPECTRA: an ImageNet map of accuracy against recovery length for a frozen allocator that is never searched per network, beside a rule and an even cut, each with the unpruned network recovered the same way. A registered early estimate of the 90-epoch result, with its error, would also be new.
8. Odds. Landing within 0.5 of the 90-epoch result is low at 10 epochs and low to medium at 20; it is better in honest terms, which subtract the unpruned network's own gain. The plan mattering more under short recovery is medium, and likelier at deeper cuts than at 2x. A short run predicting the 90-epoch result well enough to choose a budget is medium.
9. Recipe. Recalibrate the batch-norm statistics after the cut, then fine-tune with the 90-epoch recipe's optimizer and learning rate (0.01), decayed linearly to zero over the budget. Add a distillation arm, and run the unpruned control and the even cut at every budget.

---

## 2. Answers to questions 1 to 5

### Q1. Prior art on short or budgeted fine-tuning after pruning (ImageNet first)

**Short answer.** Recovery length has been varied often, but almost never on the same ImageNet plan from short to long. The only same-size ImageNet ResNet-50 comparison found is Li et al. 2022: 25 against 75 epochs at FLOPs ratio 50.72 %, a 0.63-point gap. An ablation in the same paper gives 0.75 from 25 to 100 epochs at an unstated FLOPs setting. Zimmer et al. 2023 is the only structured ImageNet study with 2- to 10-epoch recoveries under modern schedules, and it has no long anchor. Everything else is either CIFAR, unstructured, mild cuts, or a single short length with no ablation.

| Study | Setting | Recovery lengths | Result (numbers) | Gap to a long recovery | Label |
|---|---|---|---|---|---|
| Li et al., CVPR 2022 (Revisiting random channel pruning) | ResNet-50 ImageNet, base 76.15 (error 23.85); random per-layer ratios, 100 sampled, top 5 screened by a 5-epoch fine-tune | 25, 75, 120 epochs (Table 3); 25 vs 100 (Fig. 4c) | Top-1 error 25.85 / 25.22 / 24.87, i.e. 74.15 / 74.78 / 75.13 [c], at FLOPs ratio 50.72 / 50.72 / 48.99 %. Sec. 6.3: from 25 to 100 epochs "the Top-1 and Top-5 error of ResNet-50 drops by 0.75% and 0.4%" (Fig. 4c; its FLOPs setting is not stated, and the only 70 % mention in the appendix belongs to Fig. 6); ResNet-18: 0.97 and 0.62 (supplement) | At about 2x, 25 epochs end 0.63 below 75 (Table 3; the equal FLOPs ratio suggests the same plan [H]). The 25-to-120 rows differ by 0.98 but also in FLOPs, so they are not a pure epoch contrast | [V; c] |
| Zimmer, Spiegel, Pokutta, ICLR 2023 | ResNet-50 ImageNet, dense 76.17 ± 0.03 (Table 3); one-shot L2 filter pruning, "a uniform distribution of sparsity among the layers" (App. A.1); six retraining schedules | 2, 5, 10 epochs (2.22, 5.55, 11.11 % of 90; Table 18) | 30 % of filters: plain fine-tuning (FT) 67.28 / 69.87 / 71.23; ALLR 69.11 / 71.23 / 72.31. 40 %: FT 60.87 / 65.58 / 68.11; ALLR 67.52 / 70.12 / 71.32. 50 %: FT 48.85 / 58.28 / 62.93; ALLR 65.04 / 68.32 / 69.98. ALLR is best in all nine cells, and learning-rate rewinding equals FT in every one | None: no 90-epoch run of the same cut. The curve still rises at 10 epochs (ALLR +1.08 from 5 to 10 at 30 %) [c] | [V; c] |
| Zimmer et al., unstructured analogue | ResNet-50 ImageNet, global magnitude, one shot (Tables 1, 8) | 2, 5, 10, 20 epochs | 70 % sparsity: ALLR 73.69 / 74.37 / 74.89 / 75.27; 90 %: 69.56 / 71.19 / 71.99 / 71.76 | At 70 %, 20 epochs end 0.35 below BIMP's 75.62 in its Table 2, a different, longer method [c; H] | [V; c] |
| Le & Hua, ICLR 2021 (poster) | ImageNet ResNet-34 with Li et al.'s variants A and B (15.5 % and 24.2 % of FLOPs pruned, Li et al. 2017 Table 1); ResNet-50 with Taylor pruning | 20 epochs (ResNet-34, Table 2), 25 (ResNet-50, Table 8), 45 (figures) | ResNet-34 A / B at 20 epochs: FT 72.96 / 72.50, one-cycle 73.44 / 73.14. ResNet-50 at 25 epochs, 56 / 72 / 81 % of neurons kept: FT (Taylor's published) 71.69 / 74.50 / 75.48, one-cycle 72.51 / 75.22 / 75.67. With peak LR 0.1, "CLR achieve lower performance than fine-tuning for low epochs" on ImageNet and passes it above 45 epochs; for short ImageNet budgets they lowered the peak to 0.01 (Sec. 3.4) | At 1.2-1.3x, 20 epochs of one-cycle end within about 0.2 of the torchvision base (73.44 / 73.14 against 73.31 in Liu et al.; Le & Hua's own base not checked) [c; U]. No long anchor at 2x | [V; c] |
| Liu et al., ICLR 2019 (Rethinking) | CIFAR-10 L1 pruning (App. D, Table 16); ImageNet: 20-epoch fine-tunes against training the pruned architecture from scratch | CIFAR: 40, 80, 160 epochs; ImageNet: 20 (their standard, "90/20 epochs") | CIFAR flat: ResNet-56-B 92.68 / 92.67 / 92.76; VGG-16-A 93.40 / 93.45 / 93.45. ImageNet L1 ResNet-34-A / B (base 73.31): fine-tuned 72.56 / 72.29, Scratch-B 73.03 / 72.91. Released short-fine-tune ResNet-50s, change from each base: ThiNet 50 % −4.13 (fine-tuned) against −2.23 (Scratch-B); He et al. 2x −3.25 against −1.07 | ImageNet: 20 epochs end 0.47-0.62 below full training of the same mild-cut architecture [c]; the released short-fine-tune ResNet-50s end about 2 points below [c, bases differ] | [V; c] |
| Renda, Frankle, Carbin, ICLR 2020 | Structured L1 pruning with Li et al.'s rates: ResNet-56-B (CIFAR-10), ResNet-34-A (ImageNet) | a sweep of retraining epochs (Fig. 4) | "Structured pruning exhibits the same trends as unstructured pruning"; both rewinding methods beat fine-tuning across retraining times. Fine-tuning uses the final training rate (0.0004 on ImageNet). No structured numbers in the text | Figures only | [V] |
| Molchanov et al., CVPR 2019 (Taylor) | ResNet-50 ImageNet, base error 23.82 | "pruning and fine-tuning run for 25 epochs total" | 72 % of neurons: 2.25 GFLOPs (0.55 kept [c]), error 25.50 (top-1 74.50); 81 %: 2.66 GFLOPs (0.65), 75.48; 56 %: 1.34 (0.33), 71.69. Best validation epoch reported | None | [V; c] |
| He, Zhang, Sun, ICCV 2017 (Channel Pruning) | VGG-16 and ResNet-50 ImageNet | 10 epochs at LR 1e-5 ("less than 1/10 iterations of training from scratch") | ResNet-50 2x: top-5 error +8.0 (plain) and +4.0 ("enhanced") before, +1.4 after fine-tuning (Table 6); VGG-16 at 2x / 4x / 5x: +2.7 / +7.9 / +22.0 before, 0 / 1.0 / 1.7 after (Table 1) | See Liu et al. row | [V] |
| Li et al., ICLR 2017 (L1 filters) | ImageNet ResNet-34-A / B / C | "a constant learning rate 0.001 and retrain ... 20 epochs for ImageNet", "one-fourth of the original training epochs" | Error 27.44 / 27.83 / 27.52 against 26.77 at 15.5 / 24.2 / 7.5 % of FLOPs pruned (Table 1) | See Liu et al. row | [V] |
| Knapsack pruning with inner distillation (Aflalo et al., arXiv 2020, no venue on the record) | ResNet-50 ImageNet, torchvision base 76.15; knapsack channel choice | 50 epochs with distillation from the parent's inner layers (IKD) | With / without IKD: 76.60 / 76.17 at prune ratio 42.56 %; 76.21 / 75.94 at 50.80 % (2.03 GFLOPs) (Table 1) | 50 epochs land at the level of 90- to 180-epoch methods (DepGraph 75.83 at 2.08x) [c; H] | [V; c; the no-IKD arm's epochs U] |
| Budgeted training (Li, Yumer, Ramanan, ICLR 2020) | Training from scratch, not pruning: ResNet-18 ImageNet, 100-epoch full budget | 1, 5, 10, 25, 50, 100 % | Linear decay: 30.63 / 57.26 / 62.32 / 66.34 / 68.18 / 69.33; step (budget-aware conversion): 20.39 / 51.94 / 59.51 / 65.58 / 67.96 / 69.34 (Table 3) | 10 %: 7.0 below full; 25 %: 3.0; 50 %: 1.15 (linear) [c] | [V; c] |
| FPGM, CVPR 2019 | VGG CIFAR-10 (Table 3) | 40 against 160 fine-tune epochs | FPGM 93.24 → 94.00; L1 (PFEC) 93.22 → 93.28 | Longer recovery helps one method by 0.76 and not the other | [V] |
| Is Oracle Pruning the True Oracle? (Feng, Tao, Wang; TMLR 2026 per the arXiv comment) | Masks at a 0.5 pruning ratio; LeNet-5, ResNet-56, ResNet-18 (App. E) | 10 % against 100 % of retraining (ImageNet ResNet-18 retrains 30 epochs, Table A1) | "the model performance after full retraining is highly correlated with the performance with only 10% retraining" (Fig. A6, no coefficient) | Not a gap; a rank statement | [V; whether App. E's ResNet-18 is ImageNet: U] |

**Other 2022 to 2026 work on little or no recovery** (details in Q3 to Q5):
- Few-shot recovery with the unpruned network as teacher: MiR (CVPR 2022) and PRACTISE (CVPR 2023), 2,000 iterations at batch 64 on 50 to 1,000 images [V].
- One-shot structured pruning with no recovery: OSSCAR (ICML 2024 [R]); at 1.7x on ResNet-50 it keeps 50.16 from 77.01 [V].
- Post-training allocation without retraining: FCPTS (AAAI 2024; weight sparsity) [V] and DSA (NeurIPS 2024; language models) [V].
- Language models: light retraining reorders candidates (Minitron, 2024) [V], and training-aware selection (DarwinLM, 2025) [V].
- A 2026 study of pruning vision-language models recovers "using just 5% of the original data" while "retaining over 95% of the original performance" (IJCV 2026) [V abs].
- One shot against iterative pruning (Janusz et al., arXiv 2025, revised 2026): "one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios" [V abs].
- A reproduction of Renda et al. found that learning-rate rewinding "can worsen pruning results on large architectures" (Wide ResNets, CIFAR; Mikler 2021) [V abs].

### Q2. Published ImageNet ResNet-50 structured pruning: results and recovery epochs

FLOPs kept are fractions remaining; reductions printed by a paper are converted [c]. "Extra training" counts what follows a trained base (or the whole run, if from scratch). Rows from different bases and recipes stand side by side, never as a ranking.

| Method (venue) | Base → pruned top-1 | FLOPs kept (speed-up) | Extra training | Short-recovery ablation | Label |
|---|---|---|---|---|---|
| DepGraph (CVPR 2023) | 76.15 → 75.83 | 1.99 of 4.13 GMACs = 0.482 (2.08x) [c] | 30 sparse-learning + 90 fine-tune epochs (Torch-Pruning ImageNet script); the paper gives no epochs | None on ImageNet; on CIFAR only "w/o SL" (93.46 against 93.77) | [V Table 3; recipe R: EFFICIENCY §4.2] |
| OCSPruner (WACV 2026 [R]) | From scratch, base 76.29 → 76.00 / 75.95 / 75.49 / 74.72 | 0.570 / 0.517 / 0.430 / 0.331 | One 130-epoch cycle from scratch (cosine from LR 0.1, batch 256); the 75.95 row prunes at epoch 36. 27 h 6 min against 37 h 55 min for baseline training | None | [V Tables 3, 4, 8, 10] |
| CHIP (NeurIPS 2021) | 76.15 → 76.30 / 76.15 / 75.26 / 73.30 | 0.552 / 0.513 / 0.372 / 0.233 [c] | 180 fine-tune epochs, LR 0.1, batch 256 (momentum printed as 0.99) | None | [V Table 2, Sec. 4.1] |
| HRank (CVPR 2020) | 76.15 → 74.98 / 71.98 / 69.10 | 0.562 / 0.379 / 0.240 [c] (2.30 / 1.55 / 0.98 of 4.09 GFLOPs) | "For each layer, we retrain the network for 30 epochs after pruning" (per block on ResNet-50), LR 0.01 ÷10 every 10 epochs; total not stated (Li et al. 2022's table lists 480) | Footnote only: "Better accuracy can be observed when training more epochs" | [V Table 5, Sec. 4.1; total R] |
| FPGM (CVPR 2019) | From pretrained: 76.15 → 75.59 / 74.83 | 0.578 / 0.465 [c] | Prunes at the end of every training epoch; ImageNet epochs not stated (Li et al. 2022's table: 90) | CIFAR VGG only (40 against 160 epochs, above) | [V Table 2; epochs R] |
| EagleEye (ECCV 2020) | 76.4 at 2.0 G (77.1 at 3.0 G, 74.2 at 1.0 G); the repo's 2 G checkpoint reads 76.366 at 2.057 GFLOPs | about 0.50 [c] | Fine-tune schedule only in the supplement (not read). Search: 1,000 candidates, fine-tunes the top 2 (Sec. 4.4) | None; its proxy study is in Q5 | [V Table 4, repo; base 77.2 R: HALP App. C; schedule U] |
| ThiNet (ICCV 2017) | Caffe base 72.88 → 72.04 / 71.01 / 68.42 (ThiNet-70 / 50 / 30) | 0.632 / 0.442 / 0.285 [c] (of 7.72 GFLOPs) | "fine-tuned in one epoch with fixed learning rate 10⁻⁴" after each step, then "9 epochs fine-tuning" at the end; total not stated (Li et al. 2022's table: 87) | None | [V Table 3, Sec. 4.3; total R] |
| Channel Pruning, He et al. (ICCV 2017) | Top-5 error +1.4 at "2x" (base top-5 92.2); top-1 72.30 at 2.73 GFLOPs in HRank's reprint | 0.67 by HRank's count [c] | 10 epochs at LR 1e-5 | Before fine-tuning +4.0 ("enhanced"; +8.0 plain), after +1.4 (top-5) | [V Table 6; top-1 R: HRank Table 5] |
| ResRep (ICCV 2021) | 76.15 → 76.15 / 75.97 / 75.30 | 0.455 / 0.439 / 0.379 [c] | 180 epochs, cosine from LR 0.01, batch 256 | Dense control: the base fine-tuned 180 more epochs reads 76.19; prune-then-fine-tune for 180 epochs reads 74.66 at 0.455 (Table 3) | [V Tables 1, 3] |
| GReg (ICLR 2021) | 76.13 → 75.16 (GReg-1) / 75.36 (GReg-2) at 2.31x; 76.27 at 1.49x; 74.93 at 2.56x | 0.433 / 0.671 / 0.391 [c] | A regularisation phase at LR 1e-3, batch 64 (K_s = 40k stabilisation iterations plus penalty growth), then 90 fine-tune epochs at LR 0.01 dropping at 30, 60, 75 | None | [V Table 3, Tables 5, 8] |
| SOSP (ICLR 2022) | 76.15 → 75.85 / 75.21 / 74.39 | 0.51 / 0.44 / 0.36 under CCP's MAC count [c] | "batch size 256, 128 epochs, and cosine" for ImageNet ResNets; whether that is the fine-tune is not stated | Accuracy before fine-tuning on CIFAR only | [V Table 2; U] |
| MetaPruning (ICCV 2019) | From scratch, base 76.6: 75.4 at 2.0 G (uniform 0.75x: 74.8 at 2.3 G) | about 0.49 [c] | PruningNet trained for about a quarter of normal epochs; search with batch-norm statistics recomputed on 20,000 images and no fine-tuning; the winner is trained from scratch (Li et al. 2022's table: 160 epochs in all) | None | [V Table 3, Sec. 4.1; total R] |
| AutoSlim (arXiv 2019) | From scratch: 75.6 at 2.0 G, 76.0 at 3.0 G, 74.0 at 1.0 G | about 0.49 | Slimmable training, then from scratch (Li et al. 2022's table: 150) | None | [V via HALP Table 1; R] |
| HALP (NeurIPS 2022) | NVIDIA recipe, base 76.2 → 76.5 (HALP-55 %) at 2.0 G; 77.2 at 3.1 G; 74.3 at 1.0 G | about 0.49 [c] | 30 pruning steps every 320 minibatches, then 90 fine-tune epochs (8-epoch warm-up, cosine); selection costs 30 CPU-minutes against EagleEye's 30 GPU-hours | None | [V Table 1, App. A, Table 3] |
| LeGR (CVPR 2020 [R]) | 76.1 → 75.3 / 75.7 / 76.2 | 0.47 / 0.58 / 0.73 | 60 fine-tune epochs, LR 0.01 ÷10 at 30 and 45; candidates scored after 200 fine-tune steps | Proxy length 0 / 50 / 200 / 500 steps on CIFAR-100 (Sec. 5.2) | [V Table 2, Sec. 5.2] |
| Taylor-FO (CVPR 2019) | 76.18 → 75.48 / 74.50 / 71.69 | 0.65 / 0.55 / 0.33 [c] | 25 epochs in all, LR 0.001 or 0.01 ÷10 every 10 epochs, best validation epoch | None | [V Table 3, Secs. 4.2.2, 6.2] |
| Random channel pruning (CVPR 2022) | 76.15 → 74.15 / 74.78 / 75.13 [c] | 0.507 / 0.507 / 0.490 | 25 / 75 / 120 epochs, plus a 5-epoch screen of the top 5 | **Yes, the only one**: +0.63 from 25 to 75 epochs at 0.507 [c]; 0.75 from 25 to 100 epochs at an unstated FLOPs setting (Fig. 4c) | [V Table 3, Sec. 6.3] |
| Knapsack + IKD (arXiv 2020) | 76.15 → 76.21 (with IKD) / 75.94 (without) | about 0.49 (prune ratio 50.80 %) | 50 epochs | IKD on and off | [V Table 1] |
| Uniform one-shot L2 + ALLR (ICLR 2023) | 76.17 → 72.31 at 30 % of filters (10 epochs) | not reported (about 0.49 if inputs shrink with outputs [c; H]) | 2 / 5 / 10 epochs | Yes, but no long anchor | [V Table 18] |
| GAL-0.5 / AutoPruner / DCP (via Li et al. 2022's Table 3) | 71.95 / 74.76 / 74.95 [c] | 0.57 / 0.49 / 0.445 | 30 / 32 / 60 epochs | None | [R: Li et al. 2022 Table 3] |

**Reading [H].**
- At about 2x, the rows at 75.8 to 76.3 all use 90 or more epochs (DepGraph, CHIP, ResRep, HALP), or 50 epochs plus inner distillation (Knapsack).
- The rows with 25 to 32 epochs (Taylor, random pruning, AutoPruner) sit at 74.2 to 74.8, and GAL's 30-epoch row lower. These differ in method as well as epochs, so the 1- to 2-point spread is an upper bound on what recovery length costs.
- Our same-loop rows sit beside them (context): 75.24 / 75.33 / 75.42 TEST top-1 at FLOPs 0.48 for the agent, the rule and the even cut, after 90 epochs of DepGraph's repository flow. That flow is SGD with momentum 0.9 at LR 0.01, ÷10 at epochs 30 and 60. The same 90 epochs lift the unpruned network to 76.90, so the honest changes are −1.66 / −1.57 / −1.48 (§427, §430).
- Published dense controls disagree. ResRep's base gains +0.04 from 180 more epochs, ours +0.77 from 90 [V; c]. A short-recovery study therefore needs its own dense control at every budget [H].

### Q3. What speeds recovery, and how much it buys on ImageNet

| Lever | Best ImageNet evidence | How little training | Size of effect | Label |
|---|---|---|---|---|
| **Learning-rate schedule that decays to zero over the budget** | Budgeted training: "we suggest linear schedule should be the 'go-to' budget-aware schedule"; "the key for a robust budgeted-schedule is to decay smoothly to zero" | 1 to 50 % of a 100-epoch run (from scratch) | Linear over step: +10.2 at 1 %, +2.8 at 10 %, +0.76 at 25 %, +0.2 at 50 % (ResNet-18) | [V Table 3, Sec. 4.1; c] |
| **ALLR** (adaptive linear restart): linear decay from d·η₁, where d = max(pruning-induced weight change, budget fraction) | Zimmer et al.: structured ResNet-50, best in all nine cells | 2, 5, 10 epochs | Over plain fine-tuning: +1.83 / +1.36 / +1.08 at 30 % of filters; +16.19 / +10.04 / +7.05 at 50 % | [V Table 18, Sec. 2.1; c] |
| **One-cycle or large-LR restart** (Le & Hua) | ResNet-34 at 20 epochs; ResNet-50 (Taylor plans) at 25 | 20 to 25 epochs | +0.48 / +0.64 (ResNet-34 A / B); +0.82 / +0.72 / +0.19 (ResNet-50 at 56 / 72 / 81 % of neurons) over FT. With peak 0.1, worse than FT below 45 epochs; peak 0.01 for short budgets | [V Tables 2, 8, Secs. 3.1, 3.4; c] |
| **Learning-rate rewinding** (Renda et al.) | Beats fine-tuning across retraining times, structured included | full sweep | After a step schedule, a short rewind replays the final tiny rate: in Zimmer's structured ImageNet cells it equals fine-tuning at 2 to 10 epochs. Under our 90-epoch flow (0.01, ÷10 at 30 and 60), any rewind of up to 30 epochs runs at 1e-4 | [V; c] |
| **Larger fine-tune rate** (Wang et al. 2023) | "a larger LR accelerates the optimization process, making the higher performance observed earlier" | equal epochs | ImageNet-100 ResNet-34 at 95 % (pruning type as in their Table 4): 77.81 at 1e-2 against 70.53 at 1e-3 (60 epochs); their L1 ResNet-50 at 2.31x reaches 75.24 with 90 epochs from 1e-2, against the original recipe's 20 epochs at 1e-3 | [V Tables 1, 3, 4] |
| **Distillation from the unpruned network** | Knapsack IKD on ResNet-50 | 50 epochs | +0.43 at prune ratio 42.56 % and +0.27 at 50.80 % (torchvision base); about +1.1 with a stronger base | [V Table 1; c] |
| Distillation, schedule length | Beyer et al. (CVPR 2022 [R]), ResNet-50 student | 30 to 9,600 epochs | Gain over labels +1.41 at 90 epochs, +2.22 at 300, +3.39 at 1,200 [c]; "knowledge distillation requires an atypically large number of epochs". A pretrained student "improves short training runs": 77.52 against 73.89 at 30 epochs | [V Table 3, Fig. 5] |
| Distillation, language models | Minitron: pruned + distilled against conventionally retrained, equal compute | 100B against 150B tokens | MMLU 37.81 against 24.57; HellaSwag 51.04 against 50.85; retraining on "<3%" of the original data | [V Table 11, abstract] |
| **Batch-norm recalibration** (adaptive BN) | EagleEye on MobileNetV1 ImageNet; MetaPruning recomputes BN statistics on 20,000 images before scoring | forward passes only | Rank agreement with fine-tuned accuracy restored (Q5); no accuracy gain reported by itself | [V] |
| **Layer-wise reconstruction** (least squares, combinatorial) | Channel Pruning; ThiNet; OSSCAR | 0 epochs | Channel Pruning VGG-16, top-5 error before fine-tuning: +2.7 / +7.9 / +22.0 at 2x / 4x / 5x. ThiNet's least squares: +2.5 before fine-tuning (CUB, VGG-16). OSSCAR on ResNet-50 at 1.7x: 50.16 against ThiNet 41.08 and magnitude 0.37 (base 77.01; 500 calibration images) | [V] |
| **Few-shot recovery with teacher features** | MiR (CVPR 2022): ResNet-34, uniform L1, 31.3 % of FLOPs pruned | 2,000 iterations at batch 64 (about 0.1 epoch [c]) on 50 to 3,000 images | 64.1 to 68.4 against 39.0 to 61.0 for plain fine-tuning (base 73.3). With 10,000 images and 16,000 iterations (about 0.8 epoch [c]): 71.8 at about 24 % of FLOPs pruned | [V Table 1, Sec. 5.2] |
| Few-shot, choosing what to drop | PRACTISE (CVPR 2023) | same | ResNet-34 at 16 % latency cut: 72.5 with 500 images (base 73.31). ResNet-50 at 21 % latency cut: 74.58 (500 images), 75.14 (5,000), 75.24 ("All", apparently the full set [U]); base 76.1 | [V Tables 2, 6, 7] |
| **Data-free or tiny-data recovery** | DeepInversion: ResNet-50 at 20 % of filters (−37 % FLOPs), 30 epochs of distillation on 0.1 M unlabeled images (about 2.3 ImageNet epochs [c]) | about 2.3 epoch-equivalents | 74.9 with real images, 73.3 with synthesized ones (base 76.1); Taylor with full data: 75.5 at 1.5x | [V Tables 4, 5] |
| Data-free, others | Lazarevich et al. (layer-wise calibration): "~1.5% top@1 accuracy drop for a ResNet50 on ImageNet at 50% sparsity rate", data-free (weight sparsity). DFPC (ICLR 2023): CIFAR numbers only in our sources | 0 epochs | — | [V abs; R] |

**Reading [H].**
- At short budgets the schedule is the largest lever: 1 to 16 points between plain fine-tuning and a budget-aware restart in Zimmer's structured table.
- Distillation is the next lever, worth tenths at 2x on a torchvision base.
- Reconstruction and recalibration matter most at 0 epochs and fade as epochs accumulate.

### Q4. Does the per-layer plan matter more when recovery is short?

**Short answer.** Yes, wherever recovery is absent or tiny, and the gap shrinks with recovery wherever it was measured. Nobody has measured the gap at 5 to 20 ImageNet epochs, and no study on CNNs or language models maps it across several recovery lengths.

| Regime | Evidence | Plan or structure gap with no or little recovery | Gap after more recovery | Label |
|---|---|---|---|---|
| CNN post-training, weight sparsity (ResNet-50 ImageNet) | FCPTS (learned allocation, per network) against magnitude allocation (POT, L2 norm) | 50 / 70 / 80 % sparsity: 77.43 / 75.38 / 71.26 against 76.83 / 68.74 / 17.03 (dense 77.89): +0.60 / +6.64 / +54.2 [c] | not run | [V Table 3; c] |
| Language models, structured, before and after training | DarwinLM (searched non-uniform) against ShearedLLaMA, Llama-2-7B, average accuracy | Before training: 57.2 against 51.0 (+6.2); size-matched against uniform: 54.2 against 38.4 | After 10B tokens: 62.8 against 61.9 (+0.9); against ShearedLLaMA's 50B tokens: 62.8 against 62.6 (+0.2) [c] | [V Tables 1, 3, 7] |
| Language models, rank order during light retraining | Minitron: "relative rankings undergo significant changes up to ∼ 300 steps, and then stabilize"; "a flip in the ranking prior to 200 steps"; width pruning "outperforms depth, but only after retraining" | the best candidate before retraining is not the best after | ranking stable after about 300 of 400 steps (1.8B tokens) | [V Sec. 4.3, Fig. 6, Table 1] |
| Few-shot CNN (ResNet-34 ImageNet) | PRACTISE: dropping blocks against cutting filters at matched latency; which blocks to drop (its recoverability score) against random | Blocks over filters: +27.5 / +25.8 / +16.8 / +12.0 points at 50 / 100 / 500 / 1,000 images. Recoverability over random: +1.90 (1 block) to +5.88 (5 blocks) at 500 images [c] | not run | [V Tables 2, 8; c] |
| CNN, long recovery | Random against searched allocations within 0.5 % on ImageNet (Li et al. 2022) [R: card]. Network Slimming's learned allocation not significantly better than uniform on ResNet and DenseNet (Liu et al. 2019) [R: card]. A stronger retraining schedule "compresses the results by equalizing performance" across allocation methods, at one budget each (Zimmer et al., App. C.2) [V] | — | tenths | [R; V] |
| CNN from scratch | MetaPruning against uniform width scaling: 75.4 at 2.0 G against 74.8 at 2.3 G; 73.4 at 1.0 G against 72.0 at 1.1 G (FLOPs not matched) | — | +0.6 and +1.4, larger at the deeper cut | [V Table 3; c] |
| Our context (not literature) | CIFAR-100, 11 networks, validation only, exploratory: the frozen agent leads the rule after 1 epoch at FLOPs 0.20 (mean +1.42, 7 of 11) but not at 0.45 (median −0.51); the lead is gone after 100 epochs. ImageNet ResNet-50 after 90 epochs: agent, rule and even cut level at 2.08x and 1.79x (§427, §428, §430) | — | — | (brief; ledger) |

**[H]** The two ends are clear: whole points with no recovery, tenths after a long one. The middle is not mapped.
- Ten ImageNet epochs are 11 % of the 90-epoch recipe [c].
- In the DarwinLM case, 10B tokens, about 0.5 % of Llama-2's pretraining tokens (the 2T figure is my recollection [U]), already shrank a +6.2 gap to +0.9.
- If the same holds for CNNs, the plan gap at 2x after 10 to 20 epochs is likely tenths, and a deeper cut (FLOPs about 0.25 to 0.30) is where a short-recovery lever has room. This matches our CIFAR read: an edge at 0.20, none at 0.45.

**The Think Tank claim, checked.** "Every learned allocator in this regime so far searches per network: RL-Pruner, AMC, FCPTS."
- **Confirmed for CNNs** in this scan:
  - AMC searches each network [V].
  - RL-Pruner learns one CNN's distribution [R: card, read in full].
  - FCPTS learns "per-layer sparsity allocation ... in minutes" for each network, with 10k calibration images and about 30 minutes for ResNet-18. It uses weight sparsity and no end-to-end retraining [V].
  - MetaPruning trains a PruningNet per network family and searches per target [V]. The graph-metanetwork pruner is meta-trained once but still prunes and fine-tunes each target for 60 to 100+ epochs [R: EFFICIENCY §4.1].
- **Correction for language models.** DSA (NeurIPS 2024) searched an allocation function once, on LLaMA-1-7B with Wanda and WikiText2 validation perplexity, "in only 0.5 day on a 1x NVIDIA GPU H800 server". It then "directly transfer[s] this discovered allocation function to different tasks and scenarios without conducting additional searches" (LLaMA-1/2/3, Mistral, OPT and LLaVA's language model), with no retraining [V §5, App. C.1].
- DSA is a formula over per-layer importance statistics, found by search, not a policy that reads the network. It works on weight sparsity in language models.
- Corrected statement: no frozen learned allocator for structured CNN pruning in the low-retraining regime was found; in language models, one searched allocation function transfers without a new search (DSA) [V; H].

### Q5. Early prediction of the final accuracy

| Work | Proxy → target | Units ranked | Agreement measured | Label |
|---|---|---|---|---|
| EagleEye (ECCV 2020) | Accuracy after batch-norm recalibration (weights frozen) → fine-tuned accuracy; vanilla cut as control | Random width plans, MobileNetV1 ImageNet, four FLOPs settings | Adaptive BN: Pearson 0.683 to 0.819, Spearman 0.644 to 0.850, Kendall 0.458 to 0.679; vanilla: Kendall −0.003 to 0.267 (Table 1). Their own study fine-tunes 15 epochs (Fig. 3). "it is very time-consuming to do the training-based evaluation for even single-epoch fine-tuning" | [V] |
| Is Oracle Pruning the True Oracle? (TMLR 2026) | Loss right after the cut → retrained accuracy; 10 % retraining → full retraining | Masks at a 0.5 ratio | Cut-only: Kendall 0.19 on ResNet-18 ImageNet (160 models, 30-epoch retraining). 10 % retraining: "highly correlated" (Fig. A6, no coefficient) | [V] |
| LeGR (CVPR 2020) | Accuracy after τ̂ = 200 fine-tune steps → final | Learned filter rankings | "diminishing returns in increasing τ̂"; at low FLOPs "the validation accuracy after fine-tuning a few steps might not be representative" (CIFAR-100, qualitative) | [V Sec. 5.2] |
| Random channel pruning (CVPR 2022) | 5-epoch fine-tune of the top 5 of 100 plans → choice | Width plans, ImageNet | Used, not measured | [V] |
| Minitron (2024) | Light retraining (400 steps, 1.8B tokens) → longer retraining | Pruned language-model candidates | Rankings change for about 300 steps, then stabilise; one flips before 200 | [V] |
| DarwinLM (2025) | Short training at each selection stage (10K to 200K tokens) → after full training | Offspring architectures | "the best offspring after full finetuning is successfully identified" (Fig. 2, no coefficient) | [V] |
| Blending criteria (ICANN 2021) | Cut-only and 2-epoch accuracy → best fine-tuned accuracy, ResNet-34 ImageNet among others | Blended-criterion candidates | 2-epoch R² "improve[s] significantly" over the cut (figure only) | [R: LIT_AGENT_VS_RULE §4.1] |
| Gong et al. (TODAES 2022) | 2-epoch retraining → final | Per-layer scheme mappings | Assumed, not measured | [R: LIT_GONG_CITERS H3] |
| Learning-curve extrapolation | Domhan et al. (IJCAI 2015): a probabilistic model of the early curve stops poor runs, about 2x faster hyperparameter search on CIFAR-10, CIFAR-100 and MNIST [V search]. Baker et al. (arXiv 2017): frequentist regression on partial curves, "up to 6x" speed-up [V abs]. LC-PFN (NeurIPS 2023 [R]): 20,000 real curves (LCBench, NAS-Bench-201, Taskset, PD1), 2-6x early-stopping speed-ups "on 45 of these datasets" [V abs] | Training runs | None evaluated on pruned networks or ImageNet fine-tunes [H: not found]; whether PD1 holds ImageNet curves is unverified [U] | [V; R; U] |
| Budgeted training (ICLR 2020) | Final accuracy as a smooth function of the budget under a decay-to-zero schedule | One architecture | 30.63 → 69.33 from 1 % to 100 % (above); gradients vanish "at the end of each allowed budget" | [V] |
| Our context (not literature) | Gentle 2-epoch fine-tune → full 100-epoch run | Candidate plans, CIFAR-100, 21 networks | Kendall 0.52 against a 0.55 noise ceiling (§452) | (ledger) |

**How well on ImageNet [H].**
- Ranking is moderately reliable where measured: Kendall about 0.46 to 0.68 with adaptive BN on MobileNetV1, and "highly correlated" after 10 % of retraining.
- Cut-only scores are near useless (Kendall 0.19 on ResNet-18).
- No study reports the error of forecasting the final ImageNet accuracy of a pruned network from a short run.
- A short run with a budget-aware schedule is also not the first part of the long run. Its accuracy is the endpoint of a different schedule, so the forecast must model the budget-to-accuracy curve, not extrapolate one run's epochs.

---

## 3. The verdict in detail (question 6)

### (a) Has "short recovery after structured pruning on ImageNet, close to the full result" been shown?
- **Short recovery: yes, since 2017.** Channel Pruning used 10 epochs, L1 filters 20, Taylor 25, GAL 30 and AutoPruner 32 (Q1, Q2) [V; R].
- **Close to the full result at 2x: not shown at 10 to 20 epochs.** The evidence:
  - The same-size ImageNet gap at about 2x: 0.63 at 25 against 75 epochs, and 0.75 at 25 against 100 at an unstated FLOPs setting (Li et al. 2022) [V; c].
  - Liu et al. 2019 show 20-epoch fine-tunes 0.5 to 0.6 below full training at mild cuts, and the released short-fine-tune ResNet-50s about 2 points below [V; c].
  - Zimmer et al.'s 10-epoch structured numbers are still climbing [V].
- **Close at mild cuts: yes.** At 1.2x to 1.3x, 20 epochs of one-cycle end within about 0.2 of the base (Le & Hua), and few-shot teacher mimicking lands within 0.8 to 1.5 points (MiR, PRACTISE) [V; c].
- **Close with longer-but-not-full recovery: yes, once.** Fifty epochs plus inner distillation gave 76.21 at about 2x from 76.15 (Knapsack, preprint) [V].

### (b) What would still be new for SPECTRA [H]
1. **A recovery-budget map on ImageNet with dense controls.** Accuracy against recovery length (batch-norm only, 5, 10, 20, 90 epochs) for the agent, the rule and the even cut at two or three budgets. Each cell would carry the unpruned network recovered the same way.
   - Zimmer compares allocations at one budget only, and Li et al. vary epochs for one random-plan family.
   - No paper found runs the dense control under a short budget. Jin et al. (NeurIPS 2022) ran it for long unstructured schedules [R: card].
2. **A frozen cross-architecture allocator read across recovery budgets**, with no per-network search and never trained on ImageNet. The only transferred allocator found is DSA's formula for language-model weight sparsity without retraining [V].
3. **A registered early estimate** of the 90-epoch result from short runs, with its error measured on held-out arms. None was found for pruned ImageNet networks.
4. **The cost row.** On our GPU, 90 epochs take 19 to 28 GPU-hours, so 12.7 to 18.7 GPU-minutes per epoch [c]. Recovery would then cost 1.1 to 1.6 GPU-hours at 5 epochs, 2.1 to 3.1 at 10 and 4.2 to 6.2 at 20 [c], beside a 44- to 107-minute decision.

### (c) Chance of success

| Outcome | Chance [H] | Main evidence | What moves it |
|---|---|---|---|
| (i) 10 to 20 epochs within about 0.5 top-1 of 90 epochs at 2x on ResNet-50 | **Low at 10 epochs; low to medium at 20; medium at 20 with distillation, and medium in honest terms** | Same-size gap 0.63 at 25 against 75 epochs at about 2x, and 0.75 at 25 against 100 (Li et al. 2022). Structured 2-to-10-epoch curves still rising (Zimmer). Our 90-epoch flow lifts even the unpruned network by +0.77 (§430), and a short run can buy only part of that. If the unpruned network gains less under the short budget, the honest gap shrinks with it | A decay-to-zero schedule (+0.8 to +2.8 against step decay at 10 to 25 % budgets, from scratch); distillation (+0.27 to +0.43 at 50 epochs); batch-norm recalibration first |
| (ii) The per-layer plan matters more under short recovery | **Medium**: high that the spread between plans is at least as large as after 90 epochs; low that it exceeds about 0.3 at 2x after 10 to 20 epochs; medium at a deeper cut or at 5 epochs | Every low-recovery regime shows point-sized plan effects (FCPTS, DarwinLM, PRACTISE; our CIFAR 1-epoch read at 0.20). The gap shrinks fast with recovery (DarwinLM +6.2 → +0.9); our CIFAR edge was absent at 0.45 even after 1 epoch | Add a deeper FLOPs point (about 0.25 to 0.30); run two seeds at 5 and 10 epochs |
| (iii) A short run predicts the 90-epoch result well enough to pick a budget | **Medium**: medium-high for ranking plans or budgets; low for the absolute level within ±0.2 without anchors | Kendall 0.46 to 0.68 for adaptive BN on ImageNet MobileNetV1; "highly correlated" after 10 % retraining (Oracle pruning); our CIFAR τ 0.52 against a 0.55 ceiling. ImageNet budget-to-accuracy curves are smooth and saturating under linear decay | Use 5, 10 and 20 epochs on the same plans with the existing 90-epoch anchors (§427 to §430) to calibrate; register the predictor and its error bar before reading |

### (d) Recommendations for the experiment
1. **Schedule** [H, from V evidence].
   - Keep the 90-epoch flow's optimizer, augmentation and evaluation: SGD with momentum 0.9, weight decay 1e-4, batch 256, RandomResizedCrop and flip, mixed precision, last epoch kept (§427).
   - Keep its peak learning rate of 0.01 for comparability, and replace the step schedule with a linear decay to zero over the budget (Budgeted training's "go-to"; ALLR is also linear). Use a short warm-up of about a quarter epoch; that figure is a guess, with no source behind it.
   - Before epoch 1, recalibrate batch-norm statistics on a few thousand training images; this is also the 0-epoch point of the map.
   - Pilot at 5 epochs on one plan, choosing the peak by val only: 0.01 against 0.03. ALLR's budget term alone gives at least 0.1 × 10/90 ≈ 0.011 at 10 epochs, and more for heavier damage [c]. Then fix the peak for every arm and budget.
   - Do not use a constant small rate, such as the classic 0.001 of Li et al. 2017. A constant-rate fine-tune trails ALLR by 1.1 to 16 points at 2 to 10 epochs in Zimmer's structured table [V; the exact constant there: U].
   - Do not use learning-rate rewinding of the 90-epoch schedule: any rewind of up to 30 epochs replays its final rate, 1e-4 [c].
2. **Distillation.**
   - Register a distillation factor at 10 epochs (and at 20 if affordable): the unpruned ResNet-50's logits beside the label loss. The temperature (1 to 4) is a guess with no source; Knapsack's inner-layer feature distillation is a costlier option.
   - Expected gain: +0.2 to +0.5 [H, from Knapsack +0.27 / +0.43 at 50 epochs]. The cost is one extra teacher forward per step, an estimated 25 to 40 % more time [H].
   - Report distillation rows apart from label-only rows, and give them their own dense control (self-distillation).
3. **Comparators a reviewer will demand.**
   - The same plans after 90 epochs (they exist: §427 to §430).
   - The unpruned network under each short budget (the honest change).
   - The even cut and our rule at every budget.
   - One deeper FLOPs point (about 0.25 to 0.30, near HALP-30 %, EagleEye-1G and ThiNet-30) where a plan lever can show.
   - A schedule control on one plan: constant 0.001 against linear decay. Le & Hua, Zimmer and Wang et al. all show that short-budget results move with the schedule.
   - Seeds: two at 5 and 10 epochs for the agent and the rule (cheap). Budgeted training's ResNet-18 spread at 10 % of the budget is 0.04 (linear) to 0.21 (step) points [V].
   - Literature rows beside ours, never as a beat:
     - Short budgets: Taylor, 25 epochs (74.50 at 0.55; 75.22 with one-cycle); random plans, 25 epochs (74.15 at 0.51); Zimmer's ALLR, 10 epochs (72.31 at a uniform 30 % filter cut); Channel Pruning, 10 epochs; Knapsack, 50 epochs (76.21 / 75.94); LeGR, 60 epochs (75.3 at 0.47).
     - Full budgets: DepGraph 75.83, GReg-2 75.36, CHIP 76.15, ResRep 76.15, HALP 76.5 from 76.2.
     - The 0-epoch end: OSSCAR, and PRACTISE at its mild cuts.
4. **Early estimate.**
   - Before reading, register the predictor and the pass bar (for example, absolute error at most 0.3 on held-out arms):
     - an offset model: 90-epoch TEST ≈ short-budget val + an offset fitted on the existing anchors per FLOPs point;
     - a two-parameter saturating fit over 5, 10 and 20 epochs.
   - Score it leave-one-arm-out on val, then read TEST once. Never pick a budget or a plan on TEST.

---

## 4. References

Ordered by first use. Venues come from the arXiv comment field, the proceedings page or the PDF where read; otherwise [R].

1. Li, Yumer, Ramanan. Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints. ICLR 2020. https://arxiv.org/abs/1905.04753 ; ar5iv HTML. [V: abstract, Sec. 3 and 4, Table 3, BAC definition]
2. Li et al. Revisiting Random Channel Pruning for Neural Network Compression. CVPR 2022. https://arxiv.org/abs/2205.05676 [V: Tables 1, 3, Fig. 4c text, supplement statement]
3. Zimmer, Spiegel, Pokutta. How I Learned to Stop Worrying and Love Retraining. ICLR 2023 (arXiv comment: "ICLR2023 camera-ready"). https://arxiv.org/abs/2111.00843 [V: Sec. 2.1, 3.1, 3.3, App. A.1, B.1 (Tables 1, 8, 9, 17 to 21), C.1, C.2 text; figures not read]
4. Le, Hua. Network Pruning That Matters: A Case Study on Retraining Variants. ICLR 2021 (poster). https://arxiv.org/abs/2105.03193 [V: Secs. 3.1 to 3.4, 4.2, Tables 2, 5, 8, 10]
5. Li, Kadav, Durdanovic, Samet, Graf. Pruning Filters for Efficient ConvNets. ICLR 2017. https://arxiv.org/abs/1608.08710 [V: Table 1, Sec. 4 retraining schedule]
6. Liu et al. Rethinking the Value of Network Pruning. ICLR 2019. https://arxiv.org/abs/1810.05270 [V: Sec. 3, Tables 1 to 4, App. D Table 16]
7. Renda, Frankle, Carbin. Comparing Rewinding and Fine-tuning in Neural Network Pruning. ICLR 2020 [R venue]. https://arxiv.org/abs/2003.02389 [V: Secs. 2.2, 2.3, 4, App. E text; figures not digitised]
8. Molchanov et al. Importance Estimation for Neural Network Pruning. CVPR 2019 [R venue]. https://arxiv.org/abs/1906.10771 [V: Table 3, Secs. 4.2.2, 6.2]
9. He, Zhang, Sun. Channel Pruning for Accelerating Very Deep Neural Networks. ICCV 2017 [R venue]. https://arxiv.org/abs/1707.06168 [V: Tables 1, 6, Sec. 4.1]
10. ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression. ICCV 2017 [R venue]. https://arxiv.org/abs/1707.06342 [V: Table 3, Secs. 3.1, 3.2.3, 4.1 to 4.3]
11. Aflalo, Noy, Lin, Friedman, Zelnik(-Manor). Knapsack Pruning with Inner Distillation. arXiv 2002.08258 (v3, June 2020; no venue on the record). https://arxiv.org/abs/2002.08258 [V: Tables 1, 3]
12. Feng, Tao, Wang. Is Oracle Pruning the True Oracle? TMLR 2026 (arXiv comment). https://arxiv.org/abs/2412.00143 [V: abstract, Sec. 5, Table A1, App. E, Fig. A6 caption]
13. FPGM: Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration. CVPR 2019 [R venue]. https://arxiv.org/abs/1811.00250 [V: Tables 2, 3, Sec. 4.1]
14. Wang et al. Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning. arXiv 2301.05219. [V: Tables 1, 3, 4, 5 text]
15. OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization. ICML 2024 [R venue]. https://arxiv.org/abs/2403.12983 [V: Table 1, Sec. 5.1, App. B.1.1]
16. FCPTS: Gong et al. Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes. AAAI 2024 (AAAI proceedings 38(11), per search [R]). https://arxiv.org/abs/2405.05808 ; HTML v1. [V: Table 3, method text]
17. Li, Dong, Tang, Liu, Wang, Luo, Xue, Liu, Chu, Guo. Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models (DSA). NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf [V: Sec. 5, App. C.1, Sec. 7.3, text extracted from the PDF]
18. Muralidharan, Sreenivas, Joshi, Chochowski, Patwary, Shoeybi, Catanzaro, Kautz, Molchanov. Compact Language Models via Pruning and Knowledge Distillation (Minitron). arXiv 2407.14679 (v2 Nov 2024; venue not on the record [U]). [V: Secs. 2.3, 4.2, 4.3, Tables 1, 11, Fig. 6 caption]
19. DarwinLM: Evolutionary Structured Pruning of Large Language Models. arXiv 2502.07780 (COLM 2026 per our earlier scan [R]). [V: Secs. 3.5, 4.1, Tables 1 to 4, 7]
20. Huang, Thede, Mancini, Xu, Akata. Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency. IJCV 2026 (arXiv 2604.24380). [V: abstract]
21. Janusz, Wojnar, Li, Benini, Adamczewski. One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression. arXiv 2508.13836 (v2 June 2026). [V: abstract]
22. Mikler. Reproducibility Study: Comparing Rewinding and Fine-tuning in Neural Network Pruning. arXiv 2109.09670. [V: abstract]
23. DepGraph: Towards Any Structural Pruning. CVPR 2023 [R venue]. https://arxiv.org/abs/2301.12900 [V: Tables 1, 3, Sec. 4.1]; ImageNet recipe [R: `docs/paper/EFFICIENCY_AND_TRANSFER.md` §4.2, from the Torch-Pruning script]
24. Ghimire, Kil, Jeong, Park, Kim. One-Cycle Structured Pruning via Stability-Driven Subnetwork Search (OCSPruner). arXiv 2501.13439 (WACV 2026 per our card [R]). [V: Tables 3, 4, 8, 10]
25. CHIP: CHannel Independence-based Pruning for Compact Neural Networks. NeurIPS 2021 [R venue]. https://arxiv.org/abs/2110.13981 [V: Table 2, Sec. 4.1]
26. HRank: Filter Pruning using High-Rank Feature Map. CVPR 2020 [R venue]. https://arxiv.org/abs/2002.10179 [V: Table 5, Sec. 4.1, footnote 2]
27. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV 2020 [R venue]. https://arxiv.org/abs/2007.02491 [V: Tables 1, 4, Secs. 3.1, 3.2, 4.1, 4.4]; repository https://github.com/anonymous47823493/EagleEye [V: README]
28. ResRep (Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting). ICCV 2021 [R venue]. https://arxiv.org/abs/2007.03260 [V: Tables 1, 3, Sec. 4.2]
29. GReg: Neural Pruning via Growing Regularization. ICLR 2021 [R venue]. https://arxiv.org/abs/2012.09243 [V: Tables 3, 5, 6, 8]
30. SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning. ICLR 2022 [R venue]. https://arxiv.org/abs/2110.11395 [V: Table 2, Sec. 3, App. A.1 Table 3]
31. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. ICCV 2019 [R venue]. https://arxiv.org/abs/1903.10258 [V: Table 3, Sec. 4.1]
32. AutoSlim: Towards One-Shot Architecture Search for Channel Numbers. arXiv 1903.11728. [R: numbers via HALP Table 1; epochs via Li et al. 2022 Table 3]
33. HALP: Hardware-Aware Latency Pruning. NeurIPS 2022 [R venue]. https://arxiv.org/abs/2110.10811 [V: Table 1, Table 3, App. A, App. C]
34. LeGR: Towards Efficient Model Compression via Learned Global Ranking. CVPR 2020 [R venue]. https://arxiv.org/abs/1904.12368 [V: Table 2, Sec. 5.2, ImageNet schedule]
35. Beyer et al. Knowledge Distillation: A Good Teacher is Patient and Consistent. CVPR 2022 [R venue]. https://arxiv.org/abs/2106.05237 [V: Table 3, Fig. 5 caption, Secs. 1, 3.2, 3.5]
36. Wang, Liu, Ma, Yong, Chai, Wu. Compressing Models with Few Samples: Mimicking then Replacing (MiR). CVPR 2022 (CVF open access, per search). https://arxiv.org/abs/2201.02620 [V: Tables 1, 2, 5, Sec. 5.2]
37. Wang, Wu. Practical Network Acceleration with Tiny Sets (PRACTISE). CVPR 2023 (CVF open access, pp. 20331-20340, per search). https://arxiv.org/abs/2202.07861 [V: Tables 1 to 8, 10, Eq. 1 to 3]
38. Yin et al. Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion. CVPR 2020 [R venue]. https://arxiv.org/abs/1912.08795 [V: Sec. 4.3, Tables 4, 5]
39. Luo, Wu. Neural Network Pruning with Residual-Connections and Limited-Data (CURL). CVPR 2020 [R venue]. https://arxiv.org/abs/1911.08114 [V: Table 1, Sec. 4.1: ImageNet uses 100 fine-tune epochs, no distillation; distillation only in its small-data experiments]
40. Lazarevich, Kozlov, Malinin. Post-training Deep Neural Network Pruning via Layer-wise Calibration. arXiv 2104.15023 (ICCV 2021 Workshops per the CVF search result [R]). [V: abstract]
41. Narshana, Murti, Bhattacharyya. DFPC: Data Flow Driven Pruning of Coupled Channels Without Data. ICLR 2023. [R: search snippet; CIFAR numbers only]
42. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018 [R venue]. https://arxiv.org/abs/1802.03494 [V: Sec. 4.2; its "cannot provide useful supervision" concerns fine-grained (weight) pruning at high compression]
43. Domhan, Springenberg, Hutter. Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves. IJCAI 2015. https://www.ijcai.org/Abstract/15/487 [V: search summary of the proceedings page]
44. Baker, Gupta, Raskar, Naik. Accelerating Neural Architecture Search using Performance Prediction. arXiv 1705.10823 ("Submitted to ICLR 2018"). [V: abstract]
45. Adriaensen, Rakotoarison, Müller, Hutter. Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks (LC-PFN). NeurIPS 2023 [R: proceedings link in search]. https://arxiv.org/abs/2310.20447 [V: abstract]
46. Gong et al. (ACM TODAES 2022): 2-epoch retraining reward, assumed predictive. [R: `docs/paper/lit/LIT_GONG_CITERS_22NOV.md` H3]
47. Blending Pruning Criteria for Convolutional Neural Networks. ICANN 2021. [R: `docs/paper/lit/LIT_AGENT_VS_RULE_22NOV.md` §4.1]
48. RL-Pruner (arXiv 2411.06463), Jin et al. (NeurIPS 2022, arXiv 2210.13738), graph-metanetwork pruning (arXiv 2506.12041), Network Slimming as re-run by Liu et al. 2019. [R: `docs/paper/lit/LIT_CARDS_FOR_GILAD_22NOV.md`; `docs/paper/EFFICIENCY_AND_TRANSFER.md` §4.1]

**Not opened or not resolved in this scan.**
- EagleEye's supplement (its fine-tune schedule) and DepGraph's appendix: not read.
- Renda et al.'s and Zimmer et al.'s figures (accuracy against retraining epochs): not digitised.
- Oracle pruning's Fig. A6 coefficients: not printed in the text.
- PD1's task list: not checked.
- The CVF PDF of CURL returned HTTP 403; the arXiv version was read instead.
- The Torch-Pruning benchmarks page returned 404.
- Searches: 16 web searches; about 42 pages fetched, and two PDFs converted to text locally (DSA, plus a wrong-ID PDF discarded).
