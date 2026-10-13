<!-- 13 Oct 2026, Claude Code science sitting: literature scan by the spectra-literature subagent (Opus), reviewed in the main tab before use. [V] = read in the source by that agent; the main tab spot-checked NEON's numbers only. -->
# Allocation, proxies and transfer: a literature check for SPECTRA (13 Oct 2026)

Labels: [V] read in the source this session (PDF text, arXiv page or publisher record); [H] my interpretation; [U] not stated or not verified. All accuracies are each paper's own test numbers under its own protocol, not our TEST protocol. "C10/C100" = CIFAR-10/100; "FT" = fine-tune. Reference numbers [n] are listed at the end.

## Q1. Does per-layer allocation matter after a long fine-tune?

Controlled comparisons: same recovery, different allocation. All rows are [V] unless marked.

| Source (where) | Net / data / budget | Comparison | Reported result |
|---|---|---|---|
| AMC [3], Table 2 | RN-56 / C10 / 50% FLOPs; base 92.8 | AMC vs uniform vs "deep" heuristic, after FT (FT length [U]) | 91.9 vs 89.8 vs 91.5: **+2.1 over uniform, +0.4 over the best heuristic** |
| EKG [5], Table 2 | RN-56 / C10 / 50% FLOPs cut; base 93.84 | 3 search-phase variants, same plain FT | 93.54±0.09 … 93.85±0.10: **spread 0.31**; KD-FT adds +0.24 (94.09) |
| EKG [5], Table 1 | RN-56 / C10 (budget [U]) | sub-nets from L1 / FPGM / GBN / EKG scores, same FT | 93.57 / 93.57 / 93.70 / 93.85: **spread 0.28** (C100: 0.39) |
| Pruning from Scratch [6], Table 1 | VGG-16 / C10 / 50% FLOPs (Fig. 2); RN-56 (budget [U]) | 8 allocations: one from random weights (structurally "not similar", Fig. 2) and seven from pre-training checkpoints; trained from scratch, mean of 5 runs | VGG-16 93.58–93.83 (**0.25**); RN-56 92.90–93.05 (**0.15**) |
| Li et al. 2022 [2], Table 2 | RN-56 / C10 / 50–52% FLOPs; VGG / C10 / 48–51% | best-of-100 random allocations per channel criterion (6 criteria), 50-epoch FT | RN-56 error 6.39–7.12 (base 5.58); VGG 5.99–6.51 (base 5.67) |
| TAS [7], Table 1 | RN-32 / C100 / ~40% FLOPs cut | uniform ratio vs random search vs TAS (TAS has ~3% more FLOPs) | from unpruned weights: 69.34 / 69.14 / 69.70 (**TAS +0.36**); from scratch +0.77; with KD +1.01 |
| DepGraph [8], Table 2 | RN-56 / C100 at 1.5× / 3× / 6× / 12× | uniform vs learned layer sparsity (FT recipe [U]) | +0.89 / **+1.68** / +1.61 / +0.69 (e.g. 68.70→70.38 at 3×); random importance: +0.58 / −0.36 at 1.5× / 3× |
| EagleEye [4], Table 3 and supp. §3 | MobileNetV1 / C10 at 26.5M / 12.1M / 3.3M FLOPs | EagleEye vs uniform-L1, same 30-epoch FT | 91.89 vs 91.58 (**+0.31**); 91.44 vs 90.40 (+1.04); 88.01 vs 85.81 (+2.20) |
| MetaPruning [9], Tables 1–3 | ImageNet, nets trained from scratch | searched vs uniform width (the MBv1 baselines come from the MobileNet paper) | MBv1 0.75×: 70.9 vs 68.4; 0.25×: 57.2 vs 50.6; RN-50 1.0G vs 1.1G: 73.4 vs 72.0 |

**Liu et al. 2019 [1], §5, Figs. 3 and 5** [V]
- On VGG-16/C10, the Network Slimming allocations reach uniform's accuracy "using 5× fewer parameters" (Fig. 3).
- On PreResNet-164 they are "not significantly more efficient" than uniform (Fig. 5), and the pruned ResNet patterns are near-uniform (Table 18).
- Both results are scratch-trained and given as figures, with no pp numbers.

**Liu et al. 2019 [1], Table 1 and App. D** [V]
- Table 1's models are not at 50% FLOPs: RN-56-A, RN-56-B and VGG-16-A cut 10.4%, 27.6% and 34.2% of FLOPs (Li et al. 2017, Table 1).
- Fine-tuning RN-56-B for 40, 80 or 160 epochs gives 92.68, 92.67 and 92.76 (Table 16).
- The seed std over 5 runs is 0.12–0.26 pp.

**Other context** [V]
- **Li et al. 2022 [2]:** random search comes within "less than 0.5%" of architecture-optimizing methods on ImageNet (F3). Nearby configurations score alike (Fig. 1a). Going from 25 to 100 FT epochs lowers ResNet-50 error by 0.75 (Fig. 4c).
- **Venkatesan et al. [18]:** hand-designed profiles show "no significant differences ... for most compression factors" (Fig. 3).
- **Le & Hua [10]:** with the allocation fixed, random channels plus a large-LR schedule give 92.40±0.16 vs 92.32 for ResNet-56 (Table 5).

[H] At ~50% FLOPs on CIFAR ResNets, sensible allocations end within ~0.1–0.4 pp of one another after full recovery, about one to two seed std. Gaps of 1 pp or more appear against uniform, at tight budgets, on C100 or on compact nets. VGG is the exception at moderate budgets.

## Q2. Proxies that rank candidates before the fine-tune

**EagleEye [4], main Table 1** [V]
- Setting: MobileNetV1/ImageNet, with candidates drawn as random layer-wise rate vectors uniformly in [0, R] (§3.4). BN statistics come from 100 iterations on 1/30 of the training set.
- Spearman ρ, adaptive BN vs vanilla:

| FLOPs kept | Adaptive BN | Vanilla |
|---|---|---|
| not fixed | 0.850 | 0.025 |
| 75% | 0.829 | −0.030 |
| 62.5% | 0.644 | 0.395 |
| 50% | 0.803 | 0.127 |

**EagleEye [4], supplementary Table 1** [V]. Spearman for adaptive BN, with vanilla in brackets:
- MobileNetV1/C10: 0.829 (0.511) at 50% FLOPs.
- **ResNet-50/C10: 0.519 (0.450).**
- MobileNetV2/ImageNet: 0.564 (−0.216).

The supplement swaps the X1/X2 labels relative to the main text; its text says adaptive BN is the stronger in every case. The candidate count and the FT length behind the target accuracy are not stated [U].

**Widely varied or near-equivalent candidates?** All of EagleEye's are widely varied: random rates, and the "not fixed" row also mixes FLOPs [V]. I found no paper that measures adaptive-BN fidelity among near-equivalent candidates [U].

**Fidelity degrading among close or best candidates**
- **EKG [5], Fig. 2** [V]: 50 random RN-56/C10 sub-nets, with validation loss computed using batch statistics. The correlation with post-FT loss is "not so high", with "high variance in the low potential loss region of the most interest". The PCC value appears only in the figure [U].
- **LeGR [12], Fig. 7** (RN-56/C100) [V]: a fine-tune of τ̂ steps is the proxy (τ̂ = 200 for both LeGR and AMC). In "low-FLOP-count regimes, the validation accuracy after fine-tuning a few steps might not be representative."
- **Laurent et al. [14], §6.1** (unstructured pruning) [V]: Spearman between the loss the cut preserves and the post-FT gap is 0.27 (VGG11/C10) and 0.20 (PreActResNet18).
- **NAS analogue, Abdelfattah et al. [13], §5** [V]: synflow's Spearman over all architectures vs the top 10% is 0.76 vs 0.42 (NAS-Bench-201) and 0.37 vs 0.14 (NAS-Bench-101). Among ~25 edit-distance-1 neighbours it picks the best model 32–35% of the time, against 4% for a random pick (Table 5).

**Proxies used as rewards or selectors** [V]
- AMC's reward is the raw cut, which EagleEye calls vanilla evaluation. Its in-paper support is the unchanged order of 3–4 policies in Table 2.
- Li et al. 2022 select by least-squares feature reconstruction, then a 5-epoch FT of the top 5. They report no fidelity number.

[H] Our ρ ≈ 0.0–0.6 among near-prior plans is what this predicts: the high published correlations come from wide candidate spreads.

## Q3. Learned policies that transfer without per-target search

**Per-target methods** [V]
- **AMC [3]** searches each model anew: 100 exploration plus 300 exploitation episodes, about 1 h per CIFAR model. Its "Generalization Ability" section transfers the compressed VGG-16 to detection, not the agent.
- **MetaPruning [9]** trains a PruningNet per network, searches, then retrains the winner.
- **PuRL [17]** reuses its hyper-parameters across ResNet-50, MobileNetV2 and EfficientNet-B2, but runs its agent on each target (per-target [H]).

**Warm starts within one family** [V]
- **N2N [15], Table 5:** a policy pretrained on one teacher warm-starts another. Reward over the first 10 updates, pretrained vs scratch: 0.81 vs 0.50 (ResNet-18→34); 0.52 vs −0.07 (VGG-11→19).
- **GNN-RL [16], §4.3:** the encoder is trained on RN-56 and frozen, and the head is retrained on RN-44. Its learning curve is "similar" to direct search; it reaches 93.23% at a 51% FLOPs cut vs 93.10% unpruned.

**Zero-shot transfer** [V]
- **Venkatesan et al. [18], arXiv preprint:** one RL policy is trained on 8 ResNet-20-16 models (C10 and C100). Its profiles transfer to unseen models, TinyImageNet and ImageNet, within the same architecture only. On ImageNet it loses 5% at 3.35×, against 3.7% at 3.17× for per-target DARC.
- **Graph metanetwork [19], Table 5 (C10):**

| Target | Own-arch metanetwork | Cross-arch metanetwork | None |
|---|---|---|---|
| RN-56 (3.0×) | 93.40 | 92.81 (from RN-110) | 92.08 |
| RN-110 (4.0×) | 93.38 | 93.04 (from RN-56) | 92.40 |

  Crossing architectures costs 0.34–0.59 pp against an in-domain metanetwork and gains 0.64–0.73 over none. The metanetwork reshapes weights; it does not allocate.
- **Transferred allocations (no agent):**
  - Liu et al. Fig. 6: VGG-16/C10 patterns applied to VGG-19/C100 are "slightly worse" than direct pruning but "significantly better than uniform".
  - Width Transfer [20]: widths transfer across width, depth and resolution; DMCP on MobileNetV2 loses 0.4% but still beats uniform.
- **NEON [21]:** dense nets. Trained offline on random architectures and datasets, then applied without retraining: ×24.59 vs ×13.26 compression and +0.5% accuracy over 28 datasets (abstract).

[H] I found no verified CNN work that applies one policy zero-shot across architecture families with a same-recovery comparison against per-target search. The nearest results stay within one family, and either cost 0.3–1.3 pp or need a warm-start search.

## Implications for SPECTRA

- **[H] Headline.** "Matches the best rule while transferring" is defensible and honest.
  - At 50% FLOPs, sensible allocations end within 0.15–0.31 pp of each other, about one seed std [5, 6], and random search rivals published methods [2].
  - Every reused policy stays within one family and pays 0.3–1.3 pp [18, 19] or needs a warm start [15, 16].
  - Cross-family parity with no per-target search would be new. Pair it with search cost and coverage; do not claim "learns a better allocation" at 50% FLOPs.
- **[H] The ≥ 0.3 pp bar.** Beating a *good* prior by ≥ 0.3 pp after a long FT has thin precedent at ~50% FLOPs on CIFAR ResNets.
  - The best controlled gains are +0.31 to +0.4 [3, 4, 5, 7].
  - Gains of 1–7 pp are all against uniform, mostly at tight budgets, on C100 or on compact nets.
  - Our +0.2 pp hindsight ceiling fits. Register such a claim only with ≥ 3 seeds per arm, and look for it at ≤ 35% FLOPs, on C100, VGG or MobileNet.
- **[H] RL reward proxy.** Adaptive-BN accuracy has the best support.
  - Spearman is 0.64–0.85 on MobileNetV1/ImageNet (0.56 on MobileNetV2) and 0.83 at 50% on C10. That beats the raw cut (−0.22 to 0.69 across the same settings) and loss-preservation (0.20–0.27).
  - But it was measured only on widely varied candidates, reaches only 0.52 on ResNet-50/C10, and fidelity breaks near the top (EKG; NAS top-10%: 0.14–0.42).
  - Use it for shaping, or as a filter that rejects bad plans. To rank near-prior plans, use a multi-seed longer recovery, or a reward that asks only for parity at lower cost.
- **[H] Comparators.** Add "random search best-of-N with the same proxy" [2, 18] and "uniform" to the same-loop panel. Report the recovery recipe beside allocation results, since it moves accuracy more: +0.75 pp for 25→100 epochs [2], KD +0.24 [5], and the LR schedule [10].

## Sources (verified: title, authors, venue, id)

1. Rethinking the Value of Network Pruning. Z. Liu, M. Sun, T. Zhou, G. Huang, T. Darrell. ICLR 2019. arXiv:1810.05270.
2. Revisiting Random Channel Pruning for Neural Network Compression. Y. Li, K. Adamczewski, W. Li, S. Gu, R. Timofte, L. Van Gool. CVPR 2022. arXiv:2205.05676.
3. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, S. Han. ECCV 2018. doi:10.1007/978-3-030-01234-2_48; arXiv:1802.03494.
4. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. B. Li, B. Wu, J. Su, G. Wang, L. Lin. ECCV 2020. arXiv:2007.02491, plus the ECCV supplementary.
5. Ensemble Knowledge Guided Sub-network Search and Fine-tuning for Filter Pruning. S. Lee, B. C. Song. ECCV 2022. arXiv:2203.02651.
6. Pruning from Scratch. Y. Wang, X. Zhang, L. Xie, J. Zhou, H. Su, B. Zhang, X. Hu. AAAI 2020. doi:10.1609/aaai.v34i07.6910.
7. Network Pruning via Transformable Architecture Search. X. Dong, Y. Yang. NeurIPS 2019. arXiv:1905.09717.
8. DepGraph: Towards Any Structural Pruning. G. Fang, X. Ma, M. Song, M. B. Mi, X. Wang. CVPR 2023. doi:10.1109/CVPR52729.2023.01544.
9. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. Z. Liu, H. Mu, X. Zhang, Z. Guo, X. Yang, K.-T. Cheng, J. Sun. ICCV 2019. arXiv:1903.10258.
10. Network Pruning That Matters: A Case Study on Retraining Variants. D. H. Le, B.-S. Hua. ICLR 2021. arXiv:2105.03193.
11. Pruning Filters for Efficient ConvNets. H. Li, A. Kadav, I. Durdanovic, H. Samet, H. P. Graf. ICLR 2017. arXiv:1608.08710.
12. Towards Efficient Model Compression via Learned Global Ranking. T.-W. Chin, R. Ding, C. Zhang, D. Marculescu. CVPR 2020. arXiv:1904.12368.
13. Zero-Cost Proxies for Lightweight NAS. M. S. Abdelfattah, A. Mehrotra, Ł. Dudziak, N. D. Lane. ICLR 2021. arXiv:2101.08134.
14. Revisiting Loss Modelling for Unstructured Pruning. C. Laurent, C. Ballas, T. George, N. Ballas, P. Vincent. arXiv:2006.12279 (2020; venue [U]).
15. N2N Learning: Network to Network Compression via Policy Gradient RL. A. Ashok, N. Rhinehart, F. Beainy, K. M. Kitani. ICLR 2018. arXiv:1709.06030.
16. Topology-Aware Network Pruning using Multi-stage Graph Embedding and RL. S. Yu, A. Mazaheri, A. Jannesari. ICML 2022 (PMLR 162). arXiv:2102.03214.
17. Learning to Prune Deep Neural Networks via Reinforcement Learning (PuRL). M. Gupta, S. Aravindan, A. Kalisz, V. Chandrasekhar, L. Jie. ICML 2020 AutoML Workshop. arXiv:2007.04756.
18. Out-of-the-box Channel Pruned Networks. R. Venkatesan, G. Swaminathan, X. Zhou, A. Luo. arXiv:2004.14584 (preprint "under review at ECCV 2020"; no later venue found).
19. Meta Pruning via Graph Metanetworks: A Universal Meta Learning Framework for Network Pruning. Y. Liu, X. Wang, M. Zhang. arXiv:2506.12041 (2025, under review).
20. Width Transfer: On the (In)variance of Width Optimization. T.-W. Chin, D. Marculescu, A. S. Morcos. CVPR Workshops 2021. arXiv:2104.13255.
21. Multi-objective pruning of dense neural networks using deep reinforcement learning (NEON). L. Hirsch, G. Katz. Information Sciences 610:381–400 (2022). doi:10.1016/j.ins.2022.07.134.
