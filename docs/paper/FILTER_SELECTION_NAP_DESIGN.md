# Which filters to cut: a second agent, NAP2 as decision support, and robustness vs verification

**Opened:** 1 Oct 2026 (Gilad's second group of notes, relayed by Ido at 11:54). **Status board:** `docs/paper/GILAD_OCT8_TRACKER.md`.
**Code:** `scripts/selection_probe.py` (+ `.sbatch`, `tests/test_selection_probe.py`, 9/9 pass on the login node). **Jobs:**
- Smoke `21944622`: COMPLETED in 3.3 min.
- Cells, with the walk's crop+flip fine-tune: `21945105` (ResNet-56 C10), `21945106` (VGG-16 C10), `21945107` (VGG-19 C100).
- The first cell submits, `21944623–25`, had no crop+flip; they were cancelled while pending.
**NAPv2:** Michael Bohadana's repo, from the 20 Aug zip (GitHub still refuses our token, 403). Copied read-only to `scratch_audit/third_party/NAPv2` on the cluster.

Every claim about another paper below was checked against its primary source by the 1 Oct literature pass. Cells that the source does not settle say UNVERIFIED.

---

## 0. The answers, in one page

1. **How does the SOTA we benchmark against choose filters?** Every method makes two decisions: *how many* channels each layer or group keeps (allocation) and *which* ones (selection). §2 has both for 49 published methods. Three patterns stand out:
   - Most "selection" is regularization followed by a magnitude read-out. DepGraph, GReg, ResRep, OCSPruner, Slimming and Polarization all train the doomed set toward zero, then cut by norm.
   - Allocation is often not learned at all: it is uniform or hand-set in HRank, GReg, CHIP, Li et al., FPGM, SFP and C-SGD.
   - The few learned pruners (AMC, AGMC, GNN-RL, RL-Pruner) learn allocation only. They select by magnitude or Taylor.

   SPECTRA is in the same position: its agent learns allocation, and selection is SPECTRA's L1 group vote (each producer's L1, normalised by its maximum, summed over the coupled group).
2. **Is the dual-MDP / two-decision head (keep-rate × criterion) backed by literature?** Yes, so do not present it as new.
   - LFPC (CVPR 2020) learns a criterion per layer. MFP and "Blending pruning criteria" switch or blend criteria.
   - The closest precedent is Balaskas et al. (IEEE TETC 2024). A DDPG agent sets each layer's ratio and a Rainbow DQN picks that layer's pruning algorithm.
   - The action-space design is standard RL: action branching (Tavakoli et al. 2018) and parameterized actions (P-DQN, H-PPO).
   - What none of them do is SPECTRA's setting: one policy, trained offline across families and datasets, run frozen on unseen networks, acting on coupled groups. That setting is the novelty.
   - It also explains why our head found nothing (§137). Every criterion on its menu was a norm, and norm criteria rank filters almost identically (Spearman above 0.9 in most layers, Huang et al. NeurIPS 2021).
3. **A second DRL agent that decides which filters to prune?** Learned filter selectors exist, but all are trained per target: Huang et al. 2018 (REINFORCE), DECORE 2022 (an agent per channel) and Chen et al. 2020.
   - The only transferable selection methods are an evolved closed-form score (GECCO 2022) and a weight-editing metanetwork (2025, under review).
   - A frozen, transferable selection agent would therefore be new.
   - Whether it can help at all depends on a number nobody has measured in our regime: how much accuracy *which filters* buys at a fixed allocation under SPECTRA's own fine-tune. That is S0 (§8). It runs now.
4. **Robustness vs verification in DRL.** Robustness is empirical: behaviour under chosen attacks or noise, which can only over-estimate safety. Verification is a proof over a whole input set; it is sound, but complete only for small networks.
   - The DRL-verification line is Guy Katz's group at HUJI (Reluplex, Marabou, whiRL, verification-driven agent selection). We found no DRL-verification papers by Gilad. Ask whether that line is the one he meant.
   - SPECTRA has three cheap hooks (§7): an action-stability certificate for the frozen actor; choosing among our three frozen seeds by their agreement; and a "selection shield" that falls back to L1 for the second agent.
5. **NAP2 as a CNN representation for supported decision-making.** NAPv2 is a **network-level** performance predictor trained on NAS-Bench-201 cells from scratch. It cannot rank filters as shipped:
   - Its per-unit statistics run along the kernel-width axis.
   - Its 100-value rows are dominated by the first one or two filters of each layer.
   - Its pretrained predictor targets a different quantity than ours.

   Its *pipeline* is reusable, and we now use it in three roles (§6):
   - **NAP-F:** NAPv2's statistics taken *per filter*, as features for a learned selector and the second agent. Built and checked against NAPv2's own code to 1e-9.
   - **An anytime predictor** of recovered accuracy from early fine-tune snapshots. It is for ranking candidates and stopping fine-tunes early (the cost axis), never as the reward.
   - **A state feature** for the agent.

   S0 already records NAPv2 feature maps over its ResNet-56 fine-tunes.
6. **What we expect.** The literature predicts that selection matters with short or no fine-tuning and at high sparsity, and fades after about 40 epochs. MFP's matched comparison: no FT 84.80 vs 77.45, 40 epochs 93.26 vs 93.22.
   - SPECTRA's reward is computed after short in-loop fine-tunes (12/4 or 40/10). So a better selector could matter *for the agent's signal and cost* even if the final accuracy converges.
   - S0 measures exactly that curve: budgets 0, BN-only, 1, 3, 10 and 40 epochs.

---

## 1. Allocation vs selection

A structured cut of a CNN is two decisions per coupled channel group *g* of width *w_g*:
- **Allocation:** *k_g*, how many channels survive.
- **Selection:** *S_g*, the subset of size *k_g* that survives.

SPECTRA's PPO actor makes the first decision, group by group, as a keep-rate from a small menu. `src/pruning.py` makes the second: `group_importance` sums each producer's filter score, normalised by its own maximum (DepGraph / SPA style), and `select_group_survivors` keeps the top *k_g*.

NEON made no selection at all. After each action it replaces the layer with a randomly initialised one of the new width and trains it to convergence (Hirsch & Katz 2022, §3 and Algorithm 1). On CNNs that recipe was **12–54 points worse** than keeping the survivors under clean validation (ledger §156). The crop+flip variant was killed at **−20.8 pp** (§163). So on CNNs, *inheriting* the surviving filters matters a great deal. Whether *which* survivors are inherited matters, at matched allocation and SPECTRA's fine-tune, is the open question S0 answers.

---

## 2. How each method chooses filters (canonical table)

"How many" = allocation, "which" = selection. **Data** is what the selection rule reads. **Per-target** means the method searches or trains again on every new network. The other tables in `docs/paper/` carry a one-line version of this column and point here.

| Method | How many (allocation) | Which (selection) | Data | Per-target or transferable | Source |
|---|---|---|---|---|---|
| **SPECTRA** | Frozen PPO agent picks each coupled group's keep-rate, in walk order | L1 group vote, one cut per action (FPGM / BN-scale / Taylor available as switches) | weights | **Transferable: one offline train, frozen** | `src/pruning.py`, `src/NetworkEnv.py` |
| Same-loop heuristics | Mild: keep 90 % of every group. L1: keep 80 % everywhere | L1 group vote | weights | none needed | `scripts/spectra.sbatch` |
| NEON (dense nets) | DRL agent per layer | **None:** the layer is re-initialised at the new width and trained | — | transferable agent | Hirsch & Katz 2022 |
| DepGraph | Learned global sparsity to a MACs target (`--global-pruning`, 400 steps); uniform also reported | Group sparse training (L2 penalty up to 16× on low-importance groups), then lowest normalised group L2 | labels (training); weights (score) | per-target | [2301.12900](https://arxiv.org/abs/2301.12900) §3.3, `reproduce/main.py` |
| OCSPruner | Global, binary-searched threshold | Lowest group score (per-layer L2/√\|w\|, group mean) under a growing L2 penalty; cut once the set is stable (Jaccard) | labels | per-target, from scratch | [2501.13439](https://arxiv.org/abs/2501.13439) §3.2–3.4 |
| HRank | Hand-set per-layer rates | Lowest mean feature-map rank (500 images) | activations | per-target | [2002.10179](https://arxiv.org/abs/2002.10179) §3.3 |
| GReg-1 / -2 | Pre-specified per-stage ratios | GReg-1: lowest L1 fixed up front, then a growing L2 penalty. GReg-2: penalty on all filters until L1 is "faithful" | labels | per-target | [2012.09243](https://arxiv.org/abs/2012.09243) §3 |
| ResRep | Global: compactor-row norms sorted across layers to a FLOPs target | Smallest compactor-row L2, with gradient resetting | labels | per-target | [2007.03260](https://arxiv.org/abs/2007.03260) §3.4 |
| FPGM | Uniform rate | Filters nearest the layer's geometric median (redundancy), soft-pruned every epoch | weights | per-target | [1811.00250](https://arxiv.org/abs/1811.00250) §3.4 |
| SFP | Uniform rate | Smallest L2, recomputed every epoch; zeroed filters keep training | weights | per-target | [1808.06866](https://arxiv.org/abs/1808.06866) §3.2 |
| Li et al. (L1) | Sensitivity analysis, then hand-set per-stage ratios | Smallest filter L1 | weights | per-target | [1608.08710](https://arxiv.org/abs/1608.08710) §3 |
| Network Slimming | Global percentile over all BN γ | Smallest γ after L1 on γ | labels | per-target | [1708.06519](https://arxiv.org/abs/1708.06519) §3 |
| Polarization | Per-layer threshold at the γ histogram's first minimum; λ searched to FLOPs | Channels in the γ peak nearest 0 after polarization training | labels | per-target | NeurIPS 2020, `resprune-expand.py` |
| C-SGD | Uniform (3/8 of every ResNet conv) | **No ranking:** clusters made identical by training; keep one per cluster | labels | per-target | [1904.03837](https://arxiv.org/abs/1904.03837) §3.4 |
| CHIP | Hand-set per-layer counts | Lowest channel independence (nuclear-norm drop when the channel is removed; 640 images) | activations | per-target | [2110.13981](https://arxiv.org/abs/2110.13981) Alg. 1 |
| ThiNet | Hand-set per-layer rate | Greedy: channels whose removal least increases next-layer reconstruction error; least-squares rescale | activations | per-target | [1707.06342](https://arxiv.org/abs/1707.06342) §3.2 |
| He et al. channel pruning | Hand-set per-layer speed-up | LASSO on channel coefficients, then least-squares refit | activations | per-target | [1707.06168](https://arxiv.org/abs/1707.06168) §3.1 |
| Taylor (Molchanov 2019) | Global ranking | Smallest first-order Taylor score | gradients | per-target | [1906.10771](https://arxiv.org/abs/1906.10771) §3.1 |
| HALP | Global knapsack under a GPU-latency table | Highest-Taylor prefix per layer | gradients + latency | per-target (net × GPU) | [2210.06659](https://arxiv.org/abs/2210.06659) §3.1 |
| PruningBench | Protected global (≥ 10 % per group), 400 steps | Benchmarks L1, L2, LAMP, FPGM, random, BN-scale, CP, HRank, ThiNet, OBD-C, OBD-Hessian, Taylor and four sparsity trainers | per criterion | a protocol | [2406.12315](https://arxiv.org/abs/2406.12315) §4 |
| SPA / OBSPA | Global sort of group-normalised scores | Any per-weight criterion summed over coupled channels (L1, SNIP, CroP, GraSP); OBSPA: OBS score + weight update, no FT | weights / gradients / 2048 calibration inputs | per-target | [2403.18955](https://arxiv.org/abs/2403.18955) §3.2 |
| AMC | DDPG agent sets each layer's ratio | Largest-magnitude input channels, then least-squares refit | calibration activations; reward on 5k val | per-target | [1802.03494](https://arxiv.org/abs/1802.03494) §4 |
| AGMC | DDPG on a GCN embedding | Smallest L2 (code; the paper does not say) | weights | per-target; encoder reused with a retrained decoder | [2011.12641](https://arxiv.org/abs/2011.12641), code |
| GNN-RL | PPO on a multi-stage GNN | Smallest L2 (code; the paper does not say) | weights | per-target; encoder reused with a retrained MLP | [2102.03214](https://arxiv.org/abs/2102.03214), code |
| RL-Pruner | Learned per-layer sparsity distribution | Smallest Taylor per output channel | gradients | per-target | [2411.06463](https://arxiv.org/abs/2411.06463) §3.3 |
| AgenticPruner | LLM agent (in-context) proposes ratios | Taylor by default, ranked inside dependency groups | gradients | per-target | [2601.12272](https://arxiv.org/abs/2601.12272) §3.3 |
| Balaskas et al. | DDPG ratio + bit-width per layer | **A Rainbow DQN picks the pruning algorithm per layer** (L1, L2, Bernoulli, reconstruction, 3 unstructured) | per algorithm | per-target | [2312.15322](https://arxiv.org/abs/2312.15322) |
| LFPC | Per-layer (details in its supplement, UNVERIFIED) | **A learned per-layer criterion** over {L1, L2, GM} (Gumbel-softmax) | weights | per-target | CVPR 2020 |
| EagleEye | Random ratio candidates, picked by adaptive-BN accuracy | Smallest L1 | weights; BN stats for picking | per-target | [2007.02491](https://arxiv.org/abs/2007.02491) §3.4 |
| NetAdapt | Greedy per-layer proposals under a latency table | Largest L2 kept | weights | per-target (net × device) | [1804.03230](https://arxiv.org/abs/1804.03230) §3.3 |
| DSA | Differentiable keep ratios | Top BN-scale magnitude | BN weights | per-target | [2004.02164](https://arxiv.org/abs/2004.02164) §5.1 |
| TAS / DMCP / MetaPruning | Searched widths | **None:** the first *k* channels, retrained from scratch | labels | per-target | [1905.09717](https://arxiv.org/abs/1905.09717), [2005.03354](https://arxiv.org/abs/2005.03354), [1903.10258](https://arxiv.org/abs/1903.10258) |
| ABCPruner | Bee-colony search over counts | **Random** filters from the pretrained net | none | per-target | [2001.08565](https://arxiv.org/abs/2001.08565) §3.3 |
| Out-of-the-box | PPO keep fractions over a queue of same-architecture nets | **Random** | none | transferable profiles, same architecture | [2004.14584](https://arxiv.org/abs/2004.14584) §4.1 |
| OTO / ATO | Emerges from group sparsity | Groups projected exactly to zero (ATO: a controller's mask) | gradients | per-target | [2107.07467](https://arxiv.org/abs/2107.07467), [2403.14729](https://arxiv.org/abs/2403.14729) |
| PaS | Emerges from learned masks | Learned binary channel mask | gradients | per-target | [2206.01198](https://arxiv.org/abs/2206.01198) §4.1 |
| TPP | Pre-specified ratios | Lowest L1 fixed at the start, regularized away | weights | per-target | [2207.12534](https://arxiv.org/abs/2207.12534) §3.2 |
| ICE-Pruning | Uniform ratio | Smallest L1 (pluggable) | weights | per-target | [2505.07411](https://arxiv.org/abs/2505.07411) §IV-C |
| Once-for-All | Evolutionary search + accuracy predictor | Largest-L1 prefix after sorting | weights | supernet reused across hardware | [1908.09791](https://arxiv.org/abs/1908.09791) §3.3 |
| Graph metanetworks | Global ranking | Smallest group L2 **after a metanetwork rewrites the weights** | weights | **transferable metanetwork** (under review) | [2506.12041](https://arxiv.org/abs/2506.12041) App. B.2.3 |
| Evolved pruning functions | Fixed ratios | **An evolved closed-form score** on class-partitioned feature-map statistics | activations + labels | **transferable function** | [2110.10876](https://arxiv.org/abs/2110.10876) §4 |
| SACP | Searched per-layer ratios (GCN similarity, then val accuracy) | UNVERIFIED (L1 named only for its GCN training graphs); final model picked on **test** | graphs | per-target | [2506.11469](https://arxiv.org/abs/2506.11469) §3 |
| GoPrune | Ratio 0.7 (global vs per-layer UNVERIFIED; demo: global L1) | Lowest normalised channel magnitude after ℓ2,p group-sparse training | labels | per-target | [2511.22120](https://arxiv.org/abs/2511.22120) §III-A |
| LAMP | Global threshold | Smallest LAMP score: w² over the sum of w² of equal-or-larger weights in the layer (unstructured in the paper; structured in PruningBench) | weights | per-target | [2010.07611](https://arxiv.org/abs/2010.07611) §3 |
| Mu et al. (IEEE TCAD 2024) | DDPG per-layer ratio, warm-started from earlier runs | LASSO reconstruction, after He et al. | activations | per-target agent, warm-started across ratios, models and datasets | [2107.08815](https://arxiv.org/abs/2107.08815) §II-A |
| N2N | REINFORCE: remove layers, then shrink each one | UNVERIFIED (each candidate is trained by distillation for 5 epochs) | — | per-teacher; pretrained policies warm-start larger teachers | [1709.06030](https://arxiv.org/abs/1709.06030) §3.2.2 |
| Huang et al. 2018 | Per layer, one layer at a time | **A REINFORCE agent keeps or drops each filter** | filter weights | per-target | [1801.07365](https://arxiv.org/abs/1801.07365) |
| DECORE | Emerges | **One REINFORCE agent per channel** | none (a logit per channel) | per-target | [2106.06091](https://arxiv.org/abs/2106.06091) |

What the column shows, beyond the cells:
- **Selection is mostly regularization.** It is not ranking, which means "same criterion, same fine-tune" comparisons across these rows are rarely like-for-like.
- **Several pipelines pick their reported epoch on the CIFAR test set.** We confirmed this in the code or paper for Torch-Pruning's reproduction, HRank, FPGM, Polarization, CHIP, the GoPrune demo, SACP and Li et al. GNN-RL, RL-Pruner, ABCPruner and ICE-Pruning use test data inside their loop.
- **Copied numbers disagree.** For example, HRank's ResNet-56 result is 93.17, not the 92.17 that DepGraph's Table 1 and SACP print. Always quote from the original paper.
- **ResNet-56 layer coverage differs.** Li et al., GReg and ResRep prune only the first conv of each block. DepGraph, OCSPruner, SPA and PruningBench (and SPECTRA) prune whole coupled groups, including the residual stream. So equal ratios do not mean equal FLOPs.

---

## 3. Does the choice of filters matter?

### 3.1 The literature, by fine-tune length

**Against, with long fine-tunes:**
- Liu et al. (ICLR 2019) find that a predefined pruned architecture trains as well from scratch: "no matter which specific channels are pruned".
- Mittal et al. (2018) find random filters as good as the criteria after fine-tuning.
- Li et al. (CVPR 2022), ResNet-56 at about 50 % FLOPs: top-1 error ranges from GM 6.39 to KL 7.12, with "no clear winners".
- PruningBench reports "no single method consistently outperforms".
- Le & Hua (ICLR 2021): random pruning with a 1-cycle schedule beats methodical criteria.

**For, with no or short fine-tunes, or high sparsity:**
- MFP Table III (VGG-16 C10, same ratios as L1) is the cleanest matched comparison: no FT **84.80 vs 77.45**, 40 epochs 93.26 vs 93.22, 160 epochs 93.76 vs 93.28.
- Huang et al. (2018): the learned selector beats L1, and the gap widens with compression.
- Blalock et al. (MLSys 2020): methods beat random "at least for large amounts of pruning".
- Molchanov et al. (ICLR 2017): with only 30 minibatch updates between removals, Taylor beats weight and activation criteria, and random has zero correlation with the oracle. Ratios are not matched (selection is global). In their 2019 follow-up, Taylor reaches Spearman above 0.93 against the oracle.

**The gap:** no matched-ratio, matched-short-fine-tune ResNet-56 comparison of data-driven criteria against L1 was found.

### 3.2 Our own evidence

| What we ran | Result | § |
|---|---|---|
| L2 and SVD (nuclear-norm) rankers in the walk | Misses | 2 |
| FPGM and BN-scale in the frozen-actor walk | Tie L1 on ResNet-20; stay in the cliff band on ResNet-56 | 59–60 |
| v2b: the agent picks (rate, ranking) | Did not beat L1 | 83 |
| V4 factored head | Not a win | 102 |
| Two-decision head under clean validation | "A band-edge lottery, not a ranking-head effect. Do not promote." | 137 |
| NEON layer replacement (no inherited filters at all) | 12–54 points worse; crop+flip variant killed at −20.8 pp | 156, 163 |

Every ranking we tested is in the magnitude family, and Huang et al. (NeurIPS 2021) show that family ranks filters nearly identically. S0's smoke measured this on DepGraph's ResNet-56: within-group Kendall τ against L1 is 0.90 for L2, 0.85 for FPGM and 0.83 for SVD (§8). **The selection lever was never measured with a data-driven criterion, an oracle or random masks at matched allocation.** The last row shows that what the network inherits matters.

### 3.3 Why SPECTRA's regime might differ from the "it washes out" papers

1. **The reward comes from a short fine-tune.** The agent's reward is read after the in-loop fine-tune (12/4 in training, 40/10 at TEST), not after 160 epochs. A cut whose survivors recover faster looks better to the agent, so selection shapes what the agent learns.
2. **CIFAR-100 recovery has been the bottleneck all month.** It only came right this week, through protocol and augmentation, not selection. S0's VGG-19 C100 cell tests whether selection moves recovery there.
3. **Residual-stream groups.** One stream cut removes a channel from every block of a stage, so the choice there should matter most. S0 cuts those groups too.

---

## 4. Precedent: the two-decision head and a filter-selection agent

### 4.1 Choosing the criterion per layer or step

| Work | What it learns | How | Scope |
|---|---|---|---|
| LFPC (He et al., CVPR 2020) | A criterion distribution per layer over {L1, L2, GM} | Gumbel-softmax, 600 epochs, validation loss + FLOPs penalty. ResNet-56: 93.59 → 93.24 at 52.9 % FLOPs removed; GM in deep layers, norms in shallow ones | per target |
| MFP ([1904.03961](https://arxiv.org/abs/1904.03961)) | One shared criterion, switched every 2 epochs among L1, L2 and two distance criteria | Greedy, nothing learned | per target |
| Blending pruning criteria ([2107.05033](https://arxiv.org/abs/2107.05033)) | A per-layer blend of L1, L2, FPGM, Fermat, BN, entropy and Taylor | Spearman clustering + evolutionary search on val | per target |
| **Balaskas et al., IEEE TETC 2024** ([2312.15322](https://arxiv.org/abs/2312.15322)) | Ratio and bit-width per layer (DDPG) **and the pruning algorithm per layer** (Rainbow DQN reading the DDPG actor's features) | RL, energy + accuracy reward, no FT | per target |

The action space has standard RL precedent:
- Several discrete factors share a torso with one head each in **action branching** (Tavakoli et al., AAAI 2018). That is our factored keep-rate × criterion head.
- A discrete choice with a continuous parameter is a **parameterized action** (Hausknecht & Stone 2016; P-DQN; H-PPO, Fan et al. IJCAI 2019).
- Holding a choice for several steps is an **option** (Sutton, Precup & Singh 1999).

**Verdict:** the two-decision head has direct precedent. Cite LFPC and Balaskas and do not claim it as a contribution. The contribution is the setting: a single factored policy over coupled groups, trained offline across families and datasets, applied frozen. Its menu only offers a real choice once it contains non-norm criteria (Taylor, activation, rank, a learned score).

### 4.2 Agents that pick filters

| Work | Observes | Action | Scope | Result |
|---|---|---|---|---|
| Huang et al., WACV 2018 ([1801.07365](https://arxiv.org/abs/1801.07365)) | The layer's filter weights | Keep or drop each filter (REINFORCE) | per target, layer by layer | VGG-16 C10 −1.9 pp vs L1 −2.4 at the same ratios |
| DECORE, CVPR 2022 ([2106.06091](https://arxiv.org/abs/2106.06091)) | Nothing; one logit per channel | Keep or drop per channel | per target, joint with ~300 epochs | ResNet-56 C10 93.26 at ~50 % FLOPs removed |
| Chen et al., NeurIPS 2020 | Input feature maps (runtime); filter features (static) | Two DRL agents set sparsity; learned predictors pick channels | per target, about 4 epochs of DRL on C10 | C10 −0.65 pp at 3.92× vs RNP −7.14 at 3.56× (network UNVERIFIED) |
| Graph metanetworks, 2025 ([2506.12041](https://arxiv.org/abs/2506.12041)) | All weights, through a GNN | Edits weights so a fixed group-L2 cut hurts less | **transferable** (C10 / C100 / SVHN, ResNet sizes) | ResNet-56 93.51 → 93.64 at 65.6 % FLOPs removed |
| Evolved pruning functions, GECCO 2022 ([2110.10876](https://arxiv.org/abs/2110.10876)) | Weights, BN, class-partitioned feature maps | An evolved closed-form score | **transferable** | ResNet-56 C100 +0.87 over LFPC |

**Verdict:** every learned filter selector is trained per target. A selection policy that observes per-filter weights, activations and training dynamics, is trained once across families and runs frozen would be new. The risk, reported by Mu et al. ([2107.08815](https://arxiv.org/abs/2107.08815)), is that learned predictors inside RL pruning generalise poorly. That is why §6 trains the selector on dense supervised labels first, and why §8 gates every step on held-out networks.

---

## 5. NAPv2, read from the code

**What it is.** It descends from Amsel & Katz's "NAP2" (ICLR 2024 submission, NAS-Bench-101, LSTM; most likely not accepted) and continues in Bohadana, Schneider & Katz, "Neural Networks Performance Prediction using Weights and Gradients Analysis" (TMLR, July 2026; NAS-Bench-201; BiGRU; ranking by Kendall τ). The README reports τ = 0.869 (LSTM, 19M parameters) and 0.882 (BiGRU, 659K) on C10. Cross-dataset τ is 0.521 raw and 0.728 with log-normalisation.

**Pipeline**, file by file:
1. `snapshot_collector.SnapshotCollector(model, interval=100, max_snapshots=23)` copies every Conv2d / Linear weight and its gradient after each `interval`-th optimizer step.
2. `stats.extract_layer_stats` computes 12 statistics per tensor, globally and along axis −1: mean, variance, median, std, max, min, covariance, skewness, kurtosis, percentiles (0/25/75/50/100), L1, L2.
3. `feature_maps.create_feature_map` lays them out as a `[65 layers, 100 values, 12 stats]` map per snapshot. Per stat, the global value comes first and the axis variants fill the rest, truncated or padded to 100. `log_normalize` = log1p(\|x\|)·sign(x).
4. `autoencoder.py`: a conv autoencoder compresses each map (weights and gradients: 256-d per step). `bigru_predictor.py` (2-layer BiGRU, attention pooling, sigmoid) maps the sequence to a score. Training truncates sequences (`--aug`), so the predictor works at any time ("anytime").
5. `predictor.NAP2Predictor.score(model, loader, steps)` partially trains a *copy* with SGD (lr 0.1, Nesterov) and scores it. `sharpness.py` adds Kalra et al.'s four sharpness features.

**Quirks that matter for CNN pruning** (found while reimplementing; tested in `tests/test_selection_probe.py`):
- **Per-unit means per kernel column, not per filter.** Axis −1 of a PyTorch conv weight `[out, in, kh, kw]` is the kernel width. With 100-value rows, the per-unit block of a 3×3 conv holds the first one or two filters only. As shipped, NAPv2 carries **no per-filter signal**.
- **Skewness and kurtosis run across filters.** On an N-D tensor, `skew` and `kurtosis` run along axis 0 (`do_flatten` is a documented no-op), i.e. across filters at each weight position. The other "global" statistics are scalars.
- **A 65-layer cap.** ResNet-56 (57 weight layers) and VGG fit; DenseNet-100 does not.
- **The domain is from-scratch NAS cells** trained with SGD at lr 0.1. A SPECTRA fine-tune starts from a converged, freshly cut network under Adam. The pretrained predictor therefore answers a different question. The ICLR paper's accuracy error, read from its Fig. 6, is roughly 12–29 points RMSE, so the evidence supports ranking, not calibrated accuracy.

**What we reuse:** the collector, the statistic definitions, log-normalisation, the map layout, and the AE/BiGRU trainers with anytime truncation and Kendall evaluation. **What we do not reuse:** the pretrained weights; `NAP2Predictor.score()` inside the prune loop (the 20 Aug reasons still hold: wrong object, and a second partial training per step); and the per-unit block as a filter signal.

The 20 Aug verdict "scanned; do not lift" (`GILAD_DIRECTIVES_18AUG.md` §6) covered putting the predictor in the loop, and that part stands. Gilad's new direction is different: rebuild the *representation* at filter level and retrain the *predictor* on our own trajectories. This document does both.

---

## 6. The mechanism: NAP-informed decision support

### 6.1 Three roles, in order of evidence needed

| Role | Decision it supports | Output | Needs |
|---|---|---|---|
| **R1. NAP-F selection features** | Which channels a cut keeps (a learned scorer, then the second agent's input) | a score per channel | S0 headroom (G0), then a scorer that transfers (G1) |
| **R2. Anytime recovery predictor** | Rank candidate cuts, or stop a fine-tune early | a rank over candidates | Beat the free proxies: BN-recalibrated accuracy (EagleEye τ 0.68), accuracy after 1 epoch, and the pf-* proxy-fidelity results |
| **R3. NAP embedding as agent state** | The keep-rate (SPECTRA's existing decision) | a state feature | A one-change train A/B; only after R1 or R2 shows signal |

R2 also serves Gilad's first note group (the cost axis). In-loop fine-tuning dominates SPECTRA's wall-clock: 3.6–9.1 h for a ResNet-56 walk on a 4090 (`EFFICIENCY_AND_TRANSFER.md` §3). A predictor that ranks recovered accuracy from the first few snapshots could cut the 40-epoch in-loop budget. That claim is testable and would be a cost win beside the published numbers.

### 6.2 The NAP-F descriptor (built; in `selection_features.npz`)

One row per channel of every group S0 can cut, computed on the unpruned network from four calibration batches of training images.

| Block | Features | Source |
|---|---|---|
| Position | group index, depth fraction, group width, number of coupled producers | `channel_groups` |
| Weight criteria | L1, L2, SVD, FPGM, BN-scale (SPECTRA's group vote) | `pruning.group_importance` |
| Data criteria | first-order Taylor; mean post-ReLU activation; 1 − APoZ; HRank's mean feature-map rank | hooks on each group's norm outputs |
| **Oracle** | Calibration-loss increase when the channel alone is removed | hooks zeroing the channel at every norm (or producer) output |
| Consumer side | L1 of each consumer's input slice (how much later layers read the channel) | `consumer_l1` |
| **NAPv2 statistics of the filter** | NAPv2's 12 statistics (16 numbers) of each producer filter, averaged over the coupled producers | `nap_filter_stats`, equal to `nap2.stats.extract_layer_stats` on the flattened filter to 1e-9 |
| **NAPv2 statistics of the gradient** | The same 16 numbers on the filter's mean calibration-loss gradient | as above |

The oracle is Molchanov et al.'s single-channel ablation. A test checks that its value equals the loss after structurally cutting that one channel with SPECTRA's own edit, on both a residual-stream group and an inner group. It ignores interactions: two redundant filters each look cheap alone. It is therefore a strong reference, not a true upper bound. S0's random masks bound the lever from the other side.

Two pieces of NAP's training-dynamics idea are not in the static table yet:
- **The same statistics over fine-tune snapshots** (the filter's movement, and how its gradient norm evolves; movement pruning is the precedent).
- **Redundancy against other filters in the group** (cosine; FPGM covers one form).

S0's ResNet-56 job records NAPv2's network-level maps over every 40-epoch fine-tune. Per-filter trajectories are a later addition, if G0 passes.

### 6.3 S1: a learned scorer, zero GPU

Fit a learning-to-rank model: features on the right, the within-group oracle rank as the label. Use gradient-boosted trees or a small MLP; scikit-learn is in the env. Evaluate leave one network out: train on two of S0's cells, test on the third. The metric is width-weighted within-group Kendall τ against the oracle, compared with every hand criterion's τ, which S0 already prints. Features are log-normalised (NAPv2's transform) and rank-normalised within each group, so scales from different networks never meet.

Why supervised first: the oracle gives one dense label per channel, about 1,100 on ResNet-56 and 5,500 on VGG-19. An RL selector gets one noisy scalar per fine-tune. Distilling the oracle first is cheap, and it shows whether a transferable signal exists before anything is trained with RL.

### 6.4 S3: the second agent (hierarchical), only on Ido's GO

**Level 1 is SPECTRA's existing actor, unchanged.** It picks the group (walk order) and the keep-rate *k_g*.

**Level 2 is the selection policy.** Each channel of the chosen group is a token: its NAP-F row plus the level-1 state embedding.
- A small permutation-equivariant set transformer (2 layers, width 64, about 50K parameters) scores every token. The cut keeps the top *k_g*.
- For policy-gradient training, the scores define a Plackett–Luce distribution. Masks are sampled by Gumbel-top-k, which gives exact log-probabilities.

**Pretraining:** the S1 scorer is the initialisation.

**RL reward**, paired: recovered val accuracy of the policy's mask minus that of the L1 mask, at the same allocation, fine-tune seed and short budget (BN-only or 1–3 epochs, whichever S0 shows to be rank-faithful). Pairing removes the allocation's variance and the fine-tune seed's, so the reward is the selection effect itself. Episodes are single groups sampled across the training catalog, offline, and the policy is frozen at TEST, exactly as level 1 is.

**Why hierarchical rather than one flat action:**
- A flat joint action over (keep-rate, subset) is combinatorial.
- Level 1 is already trained and its interface does not change.
- Each level gets its own credit: level 2's reward is immediate and paired; level 1 keeps its episode return.

This is the options / feudal pattern, with level 2 as a cooperative low-level policy.

**The selection shield** (from the verification literature, §7):
- *Preemptive:* exactly *k_g* channels, structurally legal (the existing edit guarantees this).
- *Post-posed:* if the policy's mask, after BN re-estimation, has a calibration loss worse than L1's by more than δ, use L1's mask. The calibration batch is disjoint from validation.

The shield makes the second agent no worse than L1, up to δ, on calibration data by construction. A binomial bound turns zero violations on *n* batches into a disagreement-rate bound (about 3/*n* at 95 %).

**Cost estimate:** a group-level episode is a cut plus a 1–3 epoch fine-tune, 10–40 s on a 4090 for a CIFAR ResNet. About 2,000 episodes is roughly one GPU-day.

### 6.5 What would be a breakthrough, and what would not

- **Breakthrough:** a frozen, transferable selector that improves short-fine-tune recovery at matched allocation on held-out networks. That makes the agent's reward truer and its walk cheaper (shorter in-loop FT), and no published method has it.
- **Also publishable:** a clean negative. "At SPECTRA's budget, which filters survive does not matter once allocation is fixed; allocation is the whole game." That is the first matched measurement of its kind on these cells, and it closes the second-agent question for good.
- **Not a breakthrough:** a selector that wins only at budget 0 and loses after 3 epochs, or one that wins only on the network it was fit on.

---

## 7. Robustness vs verification in DRL

| | Robustness | Verification |
|---|---|---|
| Question | How does the model behave under this attack or noise? | Does the property hold for **every** input in this set? |
| Answer | Empirical; a failed attack proves nothing | A proof (complete) or a sound bound that may say "unknown" (incomplete) |
| Scale | Any model | Complete: small nets. Bound propagation: up to CIFAR ResNets trained for it. Randomized smoothing: any model (forward passes only) |
| In DRL | Attacks on policies (Huang et al. 2017); robust training (SA-MDP, ATLA, RADIAL-RL) | Step-wise certificates (CROP, policy smoothing, CARRL); multi-step model checking (whiRL); verification-driven agent selection (Amir et al., CAV 2023) |

**Which Katz.** The DRL-verification line is Guy Katz's group at HUJI: Reluplex, Marabou, whiRL, "Verifying Generalization" (CAV 2023), and verification-driven pruning (Lahav & Katz, FMCAD 2021). Our search found no DRL-verification work by Gilad Katz (BGU). That this is what he pointed to is UNVERIFIED; ask on 8 Oct.

**Naming collision.** In verification, "NAPs" are *Neural Activation Patterns* (Geng et al., ICML 2023). They have nothing to do with NAP2, but our NAP-F activation features (APoZ, the on/off frequency per channel) are their statistical form.

**SPECTRA hooks, ranked:**
1. **An action-stability certificate for the frozen actor.** On TEST-walk states, recompute features under several seeds and add Gaussian noise at the measured feature variance. Certify the radius at which the argmax keep-rate cannot change (randomized smoothing, Cohen et al. 2019; CROP-style curves), and check it against a PGD attack on the features.
   - *Prerequisite:* deterministic evaluation, which `SPECTRA_EVAL_DETERMINISTIC` already provides. Ledger §54 found up to 10.7 pp of spread from sampling and live dropout.
   - *Cost:* forward passes only.
2. **Choosing a frozen actor by agreement.** Amir et al. keep the agents that agree over an input domain. Our analogue: agreement among the frozen seeds s42 / s43 / s44 on states from networks none of them saw. It is cheap, and it gives the paper a principled way to pick a snapshot.
3. **The selection shield for the second agent** (§6.4).
4. **Provable Filter Pruning** ([1911.07412](https://arxiv.org/abs/1911.07412), ICLR 2020) sensitivities as one more NAP-F feature. It gives a (1 ± ε) per-layer guarantee with probability 1 − δ, and was tested on ResNet-56 and VGG-16.

**Skip:**
- reward-level certificates (each sample is a multi-hour walk);
- whiRL-style multi-step verification (it needs a formal model of prune plus fine-tune);
- complete verification of ResNet-56 / VGG-16, or of a pruned net against its original.

---

## 8. Experiment ladder, gates and kill rules

Notation:
- *H_b(c)* = mean val Δ of criterion *c* minus mean val Δ of L1, at budget *b* and keep 0.6, in points.
- *σ_ft* = L1's fine-tune-seed standard deviation at that budget.
- *σ_rand* = the random masks' standard deviation.

Decisions read **validation**; test is printed beside it and is never used to pick anything.

| Stage | Question | Cost | Gate to continue | If it fails |
|---|---|---|---|---|
| **S0** (running) | How much does selection move recovered accuracy at matched allocation, as the fine-tune grows from 0 to 40 epochs? | about 2–4 GPU-h per cell | **G0:** some named criterion or the oracle has *H_b* ≥ max(0.5, 2*σ_ft*) on ≥ 2 of 3 cells. Read at *b* = 40 (the walk's budget) and at *b* ≤ 3 (cheap FT) separately | All *H_b* below the line for every *b* ≥ 1 on all cells, **and** *σ_rand* ≤ 1.5 *σ_ft*: no second agent. Write the negative result; NAP2 goes to R2 / R3 |
| **S1** (CPU) | Is there a transferable learned score? | minutes | **G1:** held-out-network τ against the oracle ≥ the best hand criterion's + 0.05 on ≥ 2 of 3 held-out nets | Keep the best hand criterion (Taylor or activation) as a ranking switch, and add it to the two-decision head's menu |
| S1b (optional, GPU) | Do joint removals differ from the single-channel oracle? | about 30 min per group | A random-mask regression ("mask datamodel") on the three ResNet-56 stream groups agrees with the oracle | Use the datamodel credit as the label instead |
| **S2** (GPU) | Does the learned score recover better? | one S0 rerun with a `nap_f` criterion | **G2:** *H_40* ≥ max(0.3, 2*σ_ft*) on held-out cells, and never worse than L1 by more than *σ_ft* | Stop at a ranking switch |
| **S3** (train, Ido's GO) | Does a frozen selection agent help the walk? | about 1 GPU-day pretrain + RL | Same-loop A/B: same frozen level-1 actor, L1 vs the selection agent, on held-out nets | Report; keep L1 |
| **R2** (mostly CPU) | Do early fine-tune snapshots rank recovered accuracy better than the free proxies? | S0's ResNet-56 maps, then the pf-* trajectories | τ against 40-epoch accuracy ≥ τ(BN-recal) + 0.1 and ≥ τ(1-epoch accuracy), on held-out masks | The cheap proxies win; adopt the best one for in-loop stopping |
| **Robustness** | How stable are the frozen actor's actions? | forward passes | none: a measurement for the paper | — |

**S0 protocol** (`scripts/selection_probe.py`):
- **Cells:** DepGraph's ResNet-56 C10 (93.53 %), chenyaofo VGG-16-BN C10 (94.16 %) and DepGraph's VGG-19 C100 (73.50 %), i.e. the literature cells L1–L3.
- **Cut:** every coupled group once, at keep 0.8 and 0.6 (FLOPs kept about 0.64 / 0.36, bracketing DepGraph's ResNet-56 points at 2.11× and 2.57×, i.e. 0.47 / 0.39), by the walk's own structural edit. Only the ranking changes, so every mask has the same shape; the probe raises if not.
- **Masks:** L1 under 3 fine-tune seeds; L2, SVD, FPGM, BN-scale, Taylor, activation, 1 − APoZ, HRank, the oracle, anti-L1 (keep the smallest-L1 channels); 5 random masks. That is 18 masks per keep-rate.
- **Recovery:** recipe A exactly (Adam 1e-3, plateau schedule, patience 10, train-loss selection), with the walk's crop+flip (`SPECTRA_FT_AUG=1`, the TEST and train recipe since 30 Sep). Budgets: 0, BN-only, 1, 3, 10 and 40 epochs. Protocol P: val is half the test split, batch 256. The data-driven scores and the oracle use four augmented training batches (1,024 images).
- **Outputs:**
  - one JSONL row per mask and budget;
  - the surviving channels of every mask;
  - the NAP-F feature table;
  - a summary row with the lever numbers, within-group τ against the oracle and L1, and the Jaccard overlap of each mask with the oracle and with L1;
  - on ResNet-56, NAPv2 maps of every 40-epoch fine-tune (one snapshot per epoch).
- **Quoting:** these are probe numbers on protocol P's 5k halves. They go into the ledger as a probe section and are **never** a method's TEST row.

**Readout:**

```
grep -E "Kendall|\[lever\]|\[overlap\]|Traceback" runs/slurm_logs/sel_<job>.out
# full rows: runs/selection_probe/sel_<net>_<job>/results/selection_probe.jsonl
```

Each `[lever]` line is one keep-rate and budget. Apply the gates as follows:
- *H_b* of the best named criterion is `best_minus_l1_pp`; the oracle's is `ablation_minus_l1_pp`.
- *σ_ft* is `l1_ft_seed_sd_pp` and *σ_rand* is `random_sd_pp`.
- At budget 0, *σ_ft* is 0, so only the 0.5-point floor applies.

**Smoke readout** (`21944622`, plumbing only; never quoted): ResNet-56 from DepGraph's checkpoint, 30 groups and 1,120 channels, unaugmented calibration and fine-tune, one seed, keep 0.8. Every mask cut identical shapes, 30/30 groups structural. The scores need no fine-tune, so the Kendall line is already a measurement on this net, from one calibration draw. Within-group τ:

| | L2 | SVD | FPGM | BN-scale | Taylor | activation | 1 − APoZ | HRank | oracle |
|---|---|---|---|---|---|---|---|---|---|
| vs L1 | 0.90 | 0.83 | 0.85 | 0.51 | 0.21 | 0.25 | 0.11 | 0.14 | 0.18 |
| vs the oracle | 0.18 | 0.16 | 0.18 | 0.12 | 0.07 | 0.11 | 0.08 | 0.07 | — |

(L1 against the oracle is also 0.18.)

What it shows:
1. **The norm family ranks filters almost exactly like L1** (τ 0.83–0.90). This is direct evidence that our four null ranking A/Bs (§3.2) compared near-copies of L1. The data-driven criteria are nearly orthogonal to L1.
2. **Nothing agrees with the single-channel oracle** (τ ≤ 0.18). One cause is visible: the calibration loss on unaugmented training images is 0.0044. The network has memorised them, so gradients and single-channel damage are both tiny and noisy, and Taylor suffers most. The cells therefore score on augmented batches.
3. **The fine-tune curve has the predicted shape.** At budget 0, HRank is 11.0 points and the oracle 3.9 points better than L1. After BN recalibration, the oracle is 12.3 points better than L1. After one epoch, all four masks are within 1.4 points, and random is best. One seed, so this is not evidence; it is the curve the cells measure with 3–5 seeds.

**NAPv2 maps contain NaNs** (332 per weight map, 624–1,006 per gradient map). They come from NAPv2's own statistics on one-element or constant slices. NAPv2 zero-fills them before its autoencoder (`np.where(np.isnan(arr), 0.0, arr)` in its tests), so any R2 analysis must do the same.

**S0 results.** Ops pastes each cell's `Within-group Kendall`, `[lever]` and `[overlap]` lines here on COMPLETED, then the M8 / M8-neg call (runbook §10.4). Pending: cells `21945105` (ResNet-56 C10), `21945106` (VGG-16 C10) and `21945107` (VGG-19 C100).

---

## 9. What we will not claim

- That the two-decision head, a learned criterion, or a filter-selection agent is new as such (§4). Only the frozen, transferable setting is.
- That NAP2 or NAPv2 predicts SPECTRA's recovered accuracy before R2 shows it on held-out masks, or that its pretrained weights apply to pruned networks.
- That any selection result beats a published number. S0 is a lever measurement on our own loop.
- Any formal guarantee for the agent or the pruned network beyond what §7 states (smoothing certificates cover the smoothed policy; the shield covers its calibration data).

## 10. Sources

- 1 Oct literature pass (five subagent reports, primary sources only): allocation and selection for 49 published methods; criterion-choice and filter-selection agents; robustness vs verification in DRL; training-dynamics predictors. Every link above is from those reports.
- NAPv2 source read: `README.md`, `docs/guide.md`, `nap2/{stats,feature_maps,snapshot_collector,autoencoder,bigru_predictor,predictor,sharpness}.py`, `nap2/training/predict_anytime.py`.
- SPECTRA: `src/pruning.py` (`group_importance`, `select_group_survivors`, `bind_taylor_scores`), `src/NetworkEnv.py` (`prune_current_model`, recipe A), `src/channel_groups.py`; ledger §2, §54, §59–60, §83, §102, §137, §156, §163.
