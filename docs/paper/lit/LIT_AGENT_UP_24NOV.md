# Making the frozen agent beat the rule: expensive rewards, latency, preference conditioning, few-shot adaptation, mixed granularity and the 2025–2026 state of the art (24 Nov 2026)

Compiled 24 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**How sources were read.**
- arXiv abstract pages were fetched with curl and parsed for title, authors, dates, comments, journal reference and abstract. The arXiv API answered this session and was used to screen the 2025–2026 listings (§9).
- Full texts: arXiv PDFs converted with `pdftotext` (`-layout`, or `-raw` where two-column tables scrambled rows). A value is quoted only where its row reads cleanly; scrambled tables are named in §9.
- Three web pages through the page reader (the Halide autoscheduler page, the TenSet proceedings page, the Google Research MLGO blog) and one Semantic Scholar API record (MCMC).
- Facts verified by earlier scans in this folder are cited, not re-read: [R: file §].

**Labels.** Every reference in §10 carries one.
- [V full] = full text read (partial reads name the sections); [V abs] = abstract page, proceedings page or API record only.
- [R] = from an earlier in-house scan or a reprint in another paper (named). [U] = unverified (reason given).
- [c] = my arithmetic. [H] = my interpretation.
- "(in-house, §N)" = a project result as stated in the brief or in the heading of ledger §N. The ledger body was not read.

**Conventions.** FLOPs kept = 1/speed-up, or 1 − printed reduction [c]. Δ = pruned − base, each paper on its own base and recipe. No outside number here is a SPECTRA result, and none is a "beat". GPU-hour estimates use the brief's costs: 0.66 GPU-h per ResNet-56 100-epoch recovery and 3.1 GPU-min per 2-epoch probe on an RTX 4090; larger networks cost more.

**Relation to earlier scans [R]. Not repeated here:**
- mixed granularity, rule portfolios, look-ahead, deep cuts and why learned pruners collapse to rules (`LIT_AGENT_VS_RULE_22NOV.md` §2–§8);
- proxy validity, surrogates and learned rewards fitted per target (`LIT_DB_AVENUES_21NOV.md` DB2, DB5);
- short recovery on ImageNet (`LIT_SHORT_RECOVERY_23NOV.md`);
- tolerance-driven pruning and NEON's mode (`LIT_ACCURACY_TOLERANCE_21NOV.md`);
- the rule's lineage and the comparator list (`LIT_RULE_AND_COMPARATORS_21NOV.md`);
- the frozen-generic prior-art check and the latency direction (`LIT_ALTERNATIVES_21NOV.md` §7, §11).

---

## 1. Summary

**Where the agent stands (in-house).** It is level with the threshold rule at FLOPs 0.45 (+0.00 [−0.32, +0.44], §441). It trails the recalibrated rule by −0.88 [−1.47, −0.27] on 9 held-out CIFAR-100 networks, −1.58 at FLOPs 0.20 (§449). The only headroom measured above the rule is the free axis:
- how many parameters to keep at a FLOPs budget: a 2-epoch pick gains +0.42 [+0.06, +0.87] through 22 % more parameters, and +0.00 [−0.18, +0.16] at equal size (§452). The recalibrated rule's own gain over the threshold rule (+0.46 [+0.02, +0.90]) also came with 17–32 % more parameters kept (§448);
- which size to choose per network: +1.0 to +1.5 utility points with six sizes on recovered curves (§457).

Nothing measured at the cut steers either. The cut ranks plans at τ 0.156 and a cut-feature predictor at 0.081, against 0.518 for a 2-epoch recovery and 0.545 between replicate runs (§454).

**What the literature adds.**
1. **Generic policies beat tuned heuristics elsewhere by single-digit percentages, and only on exact or expert signals.**
   - Compilers: 3–7 % smaller code than LLVM's `-Oz` inliner [1, 2]; +0.3 to +1.5 % datacenter throughput from a learned register-allocation eviction policy [2].
   - Integer programming: 11–16 % less solve time than SCIP's default brancher on three of four hard problem families, by imitating an expert too slow to run (strong branching) [3].
   - Where the training signal was a proxy, the wins were disputed: chip placement, where the strongest independent test, with pre-training, still favours simulated annealing [7–9]; and a learned optimiser [12, 13].
   - No published policy won with a reward that ranks outcomes as weakly as our cut (τ 0.16) [H].
2. **Frozen policies win through a few cheap target evaluations.** ChiPFormer, pre-trained offline, lost to per-circuit search on 11 of 12 unseen circuits zero-shot, and won on 10 of 12 with 300 rollouts, ten times fewer than the per-circuit baseline used [10, c]. No pruning paper measures K-shot over zero-shot use at equal size.
3. **One policy for all trade-offs is standard, but the per-network size choice is untested after recovery.**
   - Preference-conditioned single policies are routine in multi-objective RL and NAS [32–35, 40, 41]. They have reached pruning: per network with several hardware targets [42], and for one vision-language model with an explicit budget head [36].
   - No study shows a learned per-network size choice beating a fixed size after full fine-tuning on held-out networks. AMC (2018) chose sizes with a −Error·log(size) reward read at the cut, and wrote that a fine-tuned reward "should compress more aggressively" [15]; it never tested this.
4. **Latency: equal FLOPs is not equal speed, and the strongest latency allocators are lookup-table optimisers.**
   - Published plans at equal FLOPs differ by 3–18 % in measured GPU throughput [15, 26, 27, c]. FLOPs rank edge-GPU latency at Kendall 0.01–0.46 [29].
   - The strongest latency allocators are table-driven knapsacks and mixed-integer programmes [26, 66].
   - The one learned-over-rule point at equal latency is old, per network, and from the learned method's own paper: AMC over NetAdapt, +0.7 at 2 % lower phone latency [15, c].
5. **Mixed granularity under FLOPs buys little.**
   - At similar FLOPs, depth-only search trails width-only by 1.3–1.4 points, and joint depth-and-width search adds +0.2 to +0.6 over width-only (CIFAR ResNets) [56, c].
   - 2024–2025 layer-or-filter selectors report gains only past 70 % of FLOPs removed, in their own short-recovery pipelines [59, 60].
   - Depth wins on GPU latency [63–66].
6. **State of the art, 2025–2026.**
   - One recipe-matched gain over DepGraph at its home cell: ETP (preprint), VGG-19 CIFAR-100 at 9.03×, 71.30 against DepGraph's 70.39 from the same 73.50 base, under DepGraph's setup [67].
   - On ResNet-50 ImageNet at about 2×, rows of 76.6–77.0 (from bases of 76.1–76.4) carry iterative fine-tunes, 100–160 epochs, distillation or Mixup/CutMix [28, 44, 68].

**Ranked changes to the agent (§8).**
1. A size head trained on a probe-calibrated utility, conditioned on the trade-off weight (the free axis as the agent's job).
2. The strongest rule as the base policy, with learned residuals at deep cuts.
3. Test-time selection among K agent samples by 2-epoch probes at deep cuts, at equal size, with a training guard.
4. Expert iteration on the allocation (probe-picked plans distilled into the policy), only if step 3 finds signal.
5. A latency budget, judged against a lookup-table knapsack.
6. Block removal in the action, only under the latency budget.
7. Do not run: cut-time multi-fidelity rewards or offline learning from the logs.

**At a glance.**

| Question | Short answer | Closest work | Left open for SPECTRA [H] |
|---|---|---|---|
| Q1 expensive rewards | Wins elsewhere come from exact or expert signals; margins 0.3–16 %; proxy-trained wins disputed | MLGO [1, 2]; Gasse et al. [3]; ExIt [16]; ChiPFormer [10]; AlphaChip and its critics [7–9] | A frozen pruning policy trained by distilling a probe-selected expert, judged on held-out families after a long recovery |
| Q2 latency | Equal-FLOPs plans differ 3–18 % in GPU speed; table knapsacks and MINLPs lead; no frozen policy tested against them | HALP [26]; MDP [66]; AMC [15]; Gator [27]; HW-NAS-Bench [29] | A frozen allocator at equal measured latency (RTX 4090, batch 1 and 256) against a knapsack on the same table |
| Q3 preference conditioning | Standard in MORL and NAS; in pruning per network or per model; no post-recovery evidence for per-network size choice | Envelope MORL [32]; PD-MORL [33]; MODNAS [41]; HiPP-Prune [36]; MCMC [42]; AMC [15] | A λ-conditioned frozen policy that picks each unseen network's size, validated after recovery |
| Q4 few-shot adaptation | A few hundred cheap evaluations turn zero-shot losses into wins in placement; NAS predictors adapt with 10–85 samples; nothing measured in pruning | ChiPFormer [10]; HELP [47]; Multi-Predict [53]; AgenticPruner [44] | K-probe adaptation of a frozen pruning policy at equal size, with its gain over zero-shot measured |
| Q5 mixed granularity | Under FLOPs, depth adds little; under latency, depth wins | TAS [56]; layer-or-filter selection [60]; LayerMerge [63]; MDP [66] | Only inside a latency budget |
| Q6 state of the art | ETP +0.91 over DepGraph at VGG-19 CIFAR-100 9× (same base, DepGraph's setup); ResNet-50 rows above 76.5 are recipe-bound | ETP [67]; CoSeP [28]; ModHiFi [68]; AgenticPruner [44] | Competitive rows need the recipe lever, not the allocator ([R: LIT_AGENT_VS_RULE §8 D3]) |

---

## 2. Q1. Training a policy when the reward is expensive

**Verdict.**
- Outside pruning, generic learned policies beat tuned heuristics by single-digit percentages (once by 2×). In each case they were trained on an exact or expert-quality signal over many instances and judged on held-out programs or instances.
- Wins built on proxy rewards were overturned or remain disputed.
- No pruning policy is in the winning set.

### 2.1 Generic learned policies against tuned heuristics

| Domain | Policy and training signal | Tuned heuristic | Margin | Evaluation | Label |
|---|---|---|---|---|---|
| Inlining for size (MLGO) [1, 2] | Policy gradient and evolution strategies; exact reward (native size); trained on one internal package of 30k modules | LLVM `-Oz` | "up to 7 %" smaller [1]; "3 % ~ 7 %" on other software, "only slight degradation" three months later [2] | real software never seen in training | [V abs; V blog] |
| Register-allocation eviction (MLGO) [2] | RL on one internal package | LLVM's eviction heuristic | +0.3 to +1.5 % queries per second on internal datacenter applications; "persisted for months" | production | [V blog] |
| MILP branching (Gasse et al., NeurIPS 2019) [3] | Graph network imitating strong branching, an expert too slow to use: it times out on hard set-cover instances. Trained on easy instances only | SCIP's default (reliability pseudocost) | Hard instances, mean solve time: 1489.91 vs 1677.02 s (set cover), 114.16 vs 136.17 (auctions), 563.36 vs 631.50 (facility location): −11 %, −16 %, −11 % [c]. Independent set: the policy solves 29 hard instances, the default 42 | new instances, larger than in training | [V full: Table 2] |
| Halide schedules (Adams et al., SIGGRAPH 2019) [4] | Cost model trained on "hundreds of thousands of random programs", plus beam search | the previous autoscheduler; human experts | "almost twice as fast" without autotuning, "more than twice" with; the first to "significantly outperform human experts on average" | benchmark pipelines | [V abs] |
| Tensor programs (TenSet, NeurIPS 2021) [5] | Cost model pre-trained on 52 M records from 6 platforms | Ansor's per-task cost model | search time "up to 10×" shorter | search time, not schedule quality | [V abs] |
| Cluster scheduling (Decima) [6] | RL with graph encoders, trained per workload | hand-tuned heuristics | at least 21 % lower mean job completion time, up to 2× under high load | 25-node Spark cluster; workload-specific, not generic | [V abs] |
| Macro placement (AlphaChip) [7–9] | Policy pre-trained on many blocks, then fine-tuned | simulated annealing; experts | Claimed superhuman. The authors' reply objects that an earlier assessment did not pre-train and used less compute [9]. The updated assessment pre-trains (the released checkpoint, pre-trained on 20 TPU blocks, and its own slices) and still finds that annealing and human baselines "remain superior … with statistically significant differences in proxy cost and postRouteOpt PPA metrics, using substantially fewer resources" [8] | disputed; the strongest independent test favours annealing | [V full: [7] §5.1; [8] §I and §VII opening; V abs: [9]] |
| Macro placement by plain search (WireMask-BBO, NeurIPS 2023) [11] | Black-box optimisation with a greedy wire mask, no learning | RL placers | "significantly shorter HPWL by using much less time" | public benchmarks | [V abs] |
| Learned optimiser (VeLO) [12, 13] | About 4,000 TPU-months of meta-training | tuned Adam and others | An independent test finds VeLO needs "problem-specific tuning", "does not necessarily outperform" and "is not faster" | MLCommons optimiser suite | [V abs ×2] |
| Neural TSP solvers [14] | End-to-end learned | classical solvers | close on small graphs; fail to generalise to larger ones | zero-shot size transfer | [V abs] |
| Layer-wise sparsity for LLMs (FastForward Pruning, 2025) [73] | Single-step RL, a bandit over the whole allocation like SPECTRA's plan action, with the budget handled apart and a cheap-to-costly curriculum | heuristic allocations | "superior performance over strong heuristic baselines"; no number in the abstract | a search per model (LLaMA, Mistral, OPT); retraining not stated in the abstract | [V abs] |
| Pruning, per network (AMC, ECCV 2018) [15] | DDPG, reward read at the cut | hand "deep" policy | ResNet-56 at 50 % FLOPs: 91.9 vs 91.5 after fine-tuning (base 92.8; uniform 89.8) | one network | [V full: Table 2] |
| Pruning, frozen generic (in-house) | Plan-as-action bandit, reward read at the cut | threshold rule; recalibrated rule | +0.00 at FLOPs 0.45 (§441); −1.58 at 0.20 (§449) | 9 held-out networks | (in-house) |

### 2.2 Ways to train when each reward costs minutes to hours

| Approach | Evidence | What it needs | Fit to SPECTRA [H] | Label |
|---|---|---|---|---|
| Search, then distil the search into the policy (expert iteration) | ExIt [16]: tree search plans, a network generalises the plans and then guides the search; beats REINFORCE on Hex and defeats MoHex 1.0. AlphaX [17]: tree search plus an accuracy predictor for NAS; 3× and 2.8× more sample-efficient than random search and regularised evolution on NAS-Bench-101 | an expert better than the current policy | The 2-epoch probe is such an expert only where plans differ: on size (§452) and possibly at deep cuts (§458), not at equal size at FLOPs 0.45 | [V abs] |
| Imitation of an expensive expert | Gasse et al. [3]: strong-branching labels collected on small instances; the imitation runs fast and generalises to larger ones | expert labels on training instances only | Probe-picked or recovered-curve sizes (§452, §457) as labels for a size head | [V full] |
| Multi-fidelity RL with control variates | MFMCRL [18]: cheap returns used as control variates; beats RL on expensive data alone at a fixed expensive budget. MFPG [19]: unbiased, variance-reduced policy gradients; robust when the cheap data mislead | correlation ρ between cheap and expensive returns | An optimal control variate cuts variance by the factor 1 − ρ² [c]. Treating τ as a bivariate-normal ρ = sin(πτ/2) [H]: the cut (τ 0.156, ρ ≈ 0.24) cuts variance by about 6 %; the 2-epoch probe (τ 0.518, ρ ≈ 0.73) by about 53 % [c]. So a few 100-epoch anchors can debias probe rewards; the cut cannot help [H] | [V abs] |
| Offline learning from logged outcomes (counterfactual risk minimisation) | CRM/POEM [20]: propensity-weighted learning from logged bandit feedback | the probability with which each logged plan was chosen | The logged plans came from rules and deterministic agents [H; U: logging policy not checked]. The model-based alternative, a surrogate, already failed across families (τ 0.081, §454) | [V abs] |
| A few batched rounds instead of a fully online loop | Han et al. [21]: in linear contextual bandits with stochastic contexts, a pure-exploitation scheme with fewer than log T batches reaches fully online performance | a good batch schedule | Two or three rounds of probes per training cycle should lose little [H] | [V abs] |
| Learned reward model with active data acquisition | Neural predictor [22]: 20× more sample-efficient than regularised evolution on NAS-Bench-101. BRP-NAS [23]: iterative data selection. BANANAS [24]. PPF [25]: an LLM-pruning predictor scores a policy in seconds (error < 0.0011) | labels at the target fidelity | Replicate runs agree only at τ 0.545 (§454), and the probe already reaches 0.518. A reward model can make the probe cheaper, not better | [V abs] |

**Reading [H].**
- The winning cases share three things:
  - an exact or expert-quality training signal;
  - many training instances (tens of thousands of modules, thousands of MILPs);
  - evaluation on held-out instances of the same task family.

  SPECTRA has the third, a small version of the second (39 networks), and lacks the first unless probes or recovered outcomes replace the cut.
- The disputed cases trained on proxies whose link to the final metric was contested. Our cut-time reward (τ 0.156) is a proxy of that kind.
- Distillation helps only if the expert beats the policy. The probe expert beats the rule only on size (§452), so expert iteration should target the size choice first, and the allocation only at deep cuts. At deep cuts plans spread more under short recovery: 0.80 / 0.38 / 0.29 points at 3 / 9 / 27 epochs against 0.18 after 90 on ImageNet (§458).

**Left open for SPECTRA [H].** Not found: a frozen pruning policy trained by distilling a probe-selected expert and judged on held-out families after a long recovery.

---

## 3. Q2. Latency and hardware targets

**Verdict.**
- Equal FLOPs do not mean equal speed. Published plans at matched FLOPs differ by 3–18 % in measured GPU throughput, and FLOPs rank edge-GPU latency poorly.
- The strongest latency allocators are table-driven knapsacks and mixed-integer programmes.
- The only learned-over-rule point at equal latency is per network, from 2018, and from the learned method's own paper.

### 3.1 How much plans of equal FLOPs differ in measured speed

| Source | Hardware, batch | Plans compared | Difference | Label |
|---|---|---|---|---|
| HALP (NeurIPS 2022) [26] | TITAN V, batch 256 (latency table from cuDNN 7.6.5) | EagleEye against HALP at equal FLOPs, ResNet-50 ImageNet, EagleEye's base (77.2) | 2.1 G: 1471 vs 1672 img/s (+14 %), top-1 76.4 vs 76.6. 3.0 G: 1165 vs 1203 (+3 %), 77.1 vs 77.5. On Jetson TX2, Xavier and a Xeon CPU, "up to relative 1.27×" | [V full: Table 1, §4.1, appendix on the latency tables] |
| Gator (arXiv 2205.15404) [27] | GTX 1080 Ti, batch 32 | its own FLOPs-objective and latency-objective plans, ResNet-50 ImageNet | 49.91 % vs 48.27 % of FLOPs removed: 1.442× vs 1.647× (+14 % [c]), 75.19 vs 75.28. In the same table, DCP-50 against FPGM-40 %: 55.77 % vs 53.16 % removed, 1.477× vs 1.251× (+18 % [c]) | [V full: Table 2] |
| AMC (ECCV 2018) [15] | Titan XP, batch 50; Pixel-1 CPU, batch 1 | FLOPs-targeted against latency-targeted policy, MobileNetV1 | 285 vs 272 M MACs: GPU 1.43× vs 1.53× (+7 % [c]); phone 1.81× vs 1.95×; top-1 70.5 vs 70.2 | [V full: Table 4] |
| CoSeP (BMVC 2026) [28] | 4× Quadro RTX 6000; "batch" and single-image | pruned against unpruned | ResNet-50 with 2.30× fewer FLOPs: −6.32 % (batch), −8.07 % (single) wall-clock; mean over eight models −9.48 % / −6.20 %. The authors blame bandwidth-bound kernels and tensor-core tile widths | [V full: Table 6] |
| HW-NAS-Bench (ICLR 2021) [29] | Jetson TX2 (edge GPU) and five other devices | every network of the NAS-Bench-201 and FBNet spaces | Kendall(FLOPs, edge-GPU latency): 0.357 (NAS-Bench-201, CIFAR-10), 0.015 (FBNet, CIFAR-100), 0.463 (FBNet, ImageNet) | [V full: Tables 3, 4] |
| GPU tail effect (Yu et al.) [30] | server GPUs | channel counts against GPU occupancy | 11–27 % lower latency when added to pruning and NAS methods | [V abs] |
| Radu et al. (IISWC 2019) [31] | embedded GPUs (cuDNN, Arm Compute Library, TVM) | uninformed against performance-aware channel counts | pruning 12 % of channels can slow a layer 2×; performance-aware pruning gives 3× with cuDNN | [V abs] |
| In-house (§395) | RTX 4090, fp32, eager PyTorch; batch 256 and batch 1 | the published architectures of the paper's Table 1, each against its unpruned network | DepGraph's ResNet-56 gains no speed from 37–61 % fewer FLOPs (0.93–1.07×); VGG-16 gains 1.32–1.42× at 53.5 % fewer FLOPs and 1.63–1.65× at 66 %; batch-1 latency unchanged everywhere | (in-house, §395 heading); [R: LIT_ALTERNATIVES §7] |

Not found in the literature: plans of equal FLOPs compared against each other on an RTX 40-series, A100 or H100 GPU at both batch 1 and a large batch. The in-house §395 protocol already measures both batch sizes.

### 3.2 Learned allocators against latency-table rules

| Comparison | Result | Label |
|---|---|---|
| AMC (RL, per network) against NetAdapt (greedy on measured latency with short fine-tunes), MobileNetV1 on a Pixel-1 | 70.5 % at 68.3 ms against 69.8 % at 70.0 ms: +0.7 at 2 % lower latency [c]. AMC's own table; recipes differ | [V full: AMC Table 4] |
| HALP (latency table plus knapsack) against searched or learned allocators (EagleEye, MetaPruning, AutoSlim), ResNet-50 | HALP has the higher accuracy and throughput at equal FLOPs (§3.1) | [V full] |
| MDP (mixed-integer programme over channels and blocks, latency model) against HALP | +1.4 top-1 at 28 % more speed (ResNet-50) | [V abs; R: LIT_AGENT_VS_RULE ref. 4] |
| MCMC (envelope DDPG over several hardware targets, per network) | VGG-16: 80 % fewer FLOPs, 1.92× faster, +0.09; no comparison with a table-driven rule in the abstract | [V abs] |
| A frozen policy against a latency-table rule at equal latency | not found | — |

**Reading [H].**
- Cost indicators (FLOPs, parameters, measured speed) can contradict each other, and reporting one gives a partial picture [71].
- A latency target gives the free axis a measured price and turns equal-FLOPs plans into unequal products, so the room is real.
- The comparator a reviewer will demand is a knapsack on the same table. For an additive table, a knapsack is exact.
- The agent's only structural edge would be non-additive latency that a per-layer table cannot express: tail effects, fused or removed blocks.

**Left open for SPECTRA [H].** A frozen allocator judged at equal measured latency on an RTX 4090, at batch 1 and 256, against a knapsack on the same table.

---

## 4. Q3. Preference-conditioned policies and per-network choice of the compression ratio

**Verdict.**
- One policy for all trade-offs is standard in multi-objective RL and NAS.
- Two pruning works condition on preferences: one per network, one for a single vision-language model, the latter with a budget head.
- Not found: a learned per-network size choice that beats a fixed size after full fine-tuning on held-out networks. The per-network size rules that exist choose at or near the cut.

| Work | What is conditioned or chosen | Scope and signal | Label |
|---|---|---|---|
| Envelope MORL (NeurIPS 2019) [32] | One Q-network over all linear preferences; can "infer an underlying preference with very few samples" | four RL domains | [V abs] |
| PD-MORL (ICLR 2023) [33] | One universal network for the whole preference space; "up to 25 % larger hypervolume" | continuous control | [V abs] |
| Pareto hypernetworks (ICLR 2021) [34]; Pareto set learning (NeurIPS 2022) [35]; PSL-MORL (AAAI 2025) [40] | A hypernetwork maps a preference vector to a Pareto-optimal model or policy | multi-task learning; expensive black-box optimisation; MORL | [V abs] |
| MODNAS (ICLR 2025) [41] | A hypernetwork conditioned on a preference vector and on hardware features; transfers zero-shot to new devices | up to 19 devices, 3 objectives | [V abs] |
| MCMC (IEEE TNNLS 2024) [42] | One-stage envelope DDPG for pruning under several hardware targets | per network | [V abs] |
| HiPP-Prune (arXiv, Mar 2026) [36] | "a single policy invocation outputs a global pruning blueprint", split into "an overall sparsity budget and a layer-wise allocation", conditioned on a preference vector; plan-level GRPO | one vision-language model (LLaVA); no recovery | [V abs] |
| Two-stage DRL compression (ECML-PKDD 2021) [43] | A size–accuracy Pareto front of pruned and quantised models | VGG on CIFAR-10 (9× smaller), VGG-16 on ImageNet | [V abs] |
| AMC, accuracy-guaranteed mode (ECCV 2018) [15] | R = −Error · log(#Param) lets the agent "automatically find the limit of compression" | CIFAR-10 ResNet-50, fine-grained, read at the cut. "if reward were fine-tuned accuracy, the agent should compress more aggressively" | [V full: §3.4, §4.1] |
| Network Pruning Spaces (arXiv 2023) [37] | Conjecture: "an optimal FLOPs-to-parameter-bucket ratio" per network in a pruning regime | ImageNet, searched subnetworks | [V abs] |
| CoSeP (BMVC 2026) [28] | A knee detector sets each layer's retention; no ratio is given | per network, short iterative recovery | [V full] |
| BMRS (NeurIPS 2024) [45]; PQ Index (ICLR 2023) [46] | Stop pruning without thresholds (Bayesian model reduction); prune adaptively by a compressibility index | per network | [V abs] |
| NEON (Information Sciences 2022) | A generic DRL agent for dense networks; one trained configuration per tolerance | tabular data | [R: LIT_ACCURACY_TOLERANCE, Verdict 5] |
| In-house (§455, §457, §459) | Size chosen per network on recovered curves | +0.01 with two sizes; +1.0 to +1.5 with six; a choice from the cut loses 4.7–7.8. The per-halving form is recommended (§459) | (in-house) |

**Reading [H].**
- The in-house equilibrium study fills an empty cell. AMC predicted its direction in 2018 and never tested it.
- Several in-house results point the same way as Network Pruning Spaces' "optimal FLOPs-to-parameter-bucket ratio", which is the free axis at a FLOPs budget:
  - §452's gain comes through 22 % more parameters;
  - the free axis alone ranks plans at τ 0.225, above the cut's 0.156 (§454);
  - the menu agent already keeps more parameters than the prior (0.436 vs 0.387, §441).
- HiPP-Prune's split into a budget head and an allocation is the architecture the free axis needs. Its returns are read without recovery, as AMC's were.

**Left open for SPECTRA [H].** A frozen policy, conditioned on the trade-off weight λ, that chooses each unseen network's size and allocation. Judge it on recovered utility (the per-halving form) and on hypervolume against the rule's front.

---

## 5. Q4. Few-shot test-time adaptation of a frozen policy

**Verdict.**
- Where frozen policies win in placement and NAS, a handful to a few hundred cheap target evaluations do most of the work.
- In pruning, no study measures the gain of K evaluations over zero-shot use at equal size.
- In-house, the 2-epoch pick gains only through size (§452).

| Work | Zero-shot | With K target evaluations | Label |
|---|---|---|---|
| ChiPFormer (ICML 2023) [10] | Offline pre-training on 12 circuits. On 12 unseen circuits, zero-shot is worse than per-circuit RL with 3,000 rollouts (MaskPlace) on 11 [c] | 300 rollouts: better on 10 of 12; 2,000: better on all 12. Wirelength falls 5–42 % from zero-shot to 300 rollouts [c]. More offline circuits help most at zero-shot | [V full: Table 2, §4] |
| AlphaChip preprint (2020) [7] | A pre-trained policy places a new block "in less than one second" | Fine-tuning for 2–12 hours improves on zero-shot. Training from scratch is still worse after 24 hours than fine-tuning after 12. More pre-training blocks help at zero-shot and after fine-tuning | [V full: §5.1] |
| HELP (NeurIPS 2021) [47] | — | A meta-learned latency predictor on an unseen device with "as few as 10 measurement samples" | [V abs] |
| Multi-Predict (2023) [53] | — | Latency predictors transfer across search spaces "in under 85 HW measurements", 400× fewer than a meta-learning approach | [V abs] |
| MetaD2A (ICLR 2021) [48] | Generates architectures for unseen datasets in 33 GPU-seconds on average, with no training on the target | — | [V abs] |
| Few-shot BO surrogates [49]; transfer NAS with meta-learned surrogates (ICLR 2023) [50] | — | Surrogates meta-learned across tasks adapt with a few evaluations | [V abs]; [R] |
| In-context RL: Algorithm Distillation [51]; Decision-Pretrained Transformer [52] | — | The policy improves within its context, with no weight update | [V abs] |
| Moses (2022) [54]; TenSet [5] | — | Cost models adapted across devices: up to 1.53× search efficiency and 1.41× faster inference [54] | [V abs] |
| RL pruning with history transfer (IEEE TCAD) [55] | — | Reusing other networks' history speeds RL auto-pruning by 1.5–2.5× (convergence, not accuracy) | [R: LIT_DB_AVENUES ref. 49; speed-ups R: search snippet] |
| AgenticPruner (arXiv, Jan 2026) [44] | — | A general-purpose LLM proposes per-target strategies, about 5 candidates each fine-tuned for 5 epochs on 30 % of ImageNet (7.5 epoch-equivalents). The in-context agent raises the "convergence success rate" (reaching the MAC target) from 48 % to 71 % against grid search; no zero-shot accuracy baseline | [V full: §4.2, §5] |
| In-house (§443, §452) | The frozen menu agent | One-epoch pick among seven plans: −1.95 at FLOPs 0.20, through one plan that never trained (§443). 2-epoch pick: +0.42 through size, +0.00 at equal size (§452) | (in-house) |

**Reading [H].**
- ChiPFormer's pattern fits SPECTRA: the frozen policy proposes, a few cheap target evaluations choose, and those evaluations cost little against the final product.
- In SPECTRA, a probe costs 3.1 GPU-min against about 40 GPU-min for a ResNet-56 recovery [c]. Eight probes cost about 25 GPU-min, roughly 60 % of one final [c].
- Adaptation needs a guard against picks that never train (§443).
- The gain must be read at equal size, or the free axis will be credited again.

**Left open for SPECTRA [H].** Not found for pruning: K-probe adaptation of a frozen policy at equal size, with its gain over zero-shot use measured on held-out families.

---

## 6. Q5. Mixed granularity for CNNs at matched FLOPs (2024–2026)

**Verdict.**
- Under FLOPs, adding depth to width buys little. At similar FLOPs on CIFAR ResNets, depth-only search trails width-only by 1.3–1.4 points, and joint search adds +0.2 to +0.6 over width-only (TAS).
- 2024–2025 layer-or-filter methods report gains past 70 % of FLOPs removed, each in its own short-recovery pipeline.
- Depth wins on GPU latency.

| Work | Units | Matched FLOPs? | Result | Label |
|---|---|---|---|---|
| TAS (NeurIPS 2019) [56] | depth, width, both (searched, with distillation) | approximately | ResNet-20 C10: depth 90.97 at 46.2 % FLOPs removed; width 92.31 at 51.3 %; both 92.88 at 45.0 %. ResNet-32 C10: 91.48 at 41.0 %; 92.92 at 45.4 %; 93.16 at 49.4 % | [V full: Table 3] |
| Three-dimension pruning (ICML 2021) [57]; JointPruning (IEEE TIP 2021) [58] | depth, width, resolution | budget-matched | Pruning along one dimension "often leads to excessive reduction of that dimension, thus inducing a huge accuracy loss" [57]; +2.5 and +2.6 over the state of the art on compact MobileNets [58] | [V abs] |
| CKA layer pruning (ICPR 2024) [59] | layers | own points | "more than 75 % of computation" removed "while improving predictive ability" | [V abs] |
| Layer-or-filter by representation similarity (IJCNN 2025) [60] | at each step, a layer-pruned or a filter-pruned candidate, chosen by similarity to the parent after a 10-epoch fine-tune | no: each run stops when accuracy drops | The same pipeline with a coin flip in place of the choice stops earlier. ResNet-56: +0.19 at 72.67 % FLOPs removed against +0.19 at 69.64 %; ResNet-32: +0.09 at 61.99 % against +0.12 at 55.83 %. Highest ResNet-56 cut: 86.37 % | [V full: Table III, §IV] |
| Consensus layer pruning (2024) [61]; channel-then-layer pruning (ECML PKDD 2025) [62] | layers; channels then layers | own points | up to 78.80 % of FLOPs removed [61]; latency gains on a Jetson TX2 [62] | [V abs] |
| LayerMerge (ICML 2024) [63]; two-stage DP depth compression (ICML 2023) [64]; DepthShrinker (ICML 2022) [65] | layers and activations | no: matched speed-up (TensorRT) | LayerMerge "consistently outperforms existing depth compression and layer pruning methods" at a desired speed-up [63]; DepthShrinker: +3.06 accuracy and 1.53× throughput on a V100 over MetaPruning [65] | [V abs] |
| MDP (CVPR 2025) [66] | channels and blocks | no: latency | +1.4 top-1 at 28 % more speed than HALP (ResNet-50) | [V abs] |

**Reading [H].**
- The 2024–2025 FLOPs results compare against other papers' numbers under different recoveries. The informed-against-random contrast of [60] is the cleanest, and it is a stopping-point contrast, not an equal-FLOPs one.
- Under the project's FLOPs axis and 100-epoch recovery, TAS's +0.2 to +0.6 for joint search is the best guide. That joint search also ran with distillation.
- This agrees with `LIT_AGENT_VS_RULE_22NOV.md` §2: the large mixed-granularity wins are latency wins [R].

**Left open for SPECTRA [H].** Mixed granularity only inside a latency budget (§8 C6).

---

## 7. Q6. State-of-the-art update, 2025–2026, at the comparison points

**Verdict.**
- At DepGraph's own cells, one recipe-matched gain was found: ETP (preprint) at VGG-19 CIFAR-100 9×. It is a training-time regulariser, not an allocator.
- At ResNet-50 ImageNet about 2×, new rows of 76.6–77.0 come with longer or augmented recipes, from bases of 76.1–76.4.
- For MobileNetV2 and VGG-16 on CIFAR-10, the only new rows are CoSeP's.

| Cell | Method (venue) | Base → pruned | FLOPs kept | Recipe | Label |
|---|---|---|---|---|---|
| ResNet-56 C10, ~2–3× | DepGraph (CVPR 2023) | 93.53 → 93.64 | 0.389 (2.57×) [c] | as published | [R: LIT_SEALED_WEB §1(a); ETP Table 1 reprint] |
| | ETP (arXiv 2506.22015, "under review") [67] | 93.44 → 93.66 | 0.341 (2.93×) [c] | exponential "torque" regulariser; "the same experimental setup as" DepGraph | [V full: Table 1] |
| | CoSeP (BMVC 2026) [28] | 93.58 → 93.82 ± 0.09 (5 seeds) | 0.439 (2.28×) [c] | layer by layer, 2 epochs on 25 % of the data after each layer | [V full: Table 1, App. C] |
| | ATO (CVPR 2024); Torque | 93.50 → 93.74; 93.48 → 93.76 | 0.450; 0.465 [c] | as published | [R: CoSeP Table 1 reprint] |
| VGG-16 C10, ~2.5× | CoSeP [28] | 93.55 → 93.92 ± 0.07 | 0.386 (2.59×) [c] | as above | [V full: Table 1] |
| | AAP; DECORE-500 | 93.64 → 93.82; 93.96 → 94.02 | 0.389; 0.426 [c] | as published | [R: CoSeP Table 1 reprint] |
| VGG-19 C100, ~9× | DepGraph with sparse learning | 73.50 → 70.39 (without: 67.60) | 0.112 (8.92×) | sparse learning, then fine-tune | [R: LIT_SEALED_WEB §1(a); ETP Table 1 reprint] |
| | ETP [67] | 73.50 → 71.30 | 0.111 (9.03×) | DepGraph's setup | [V full: Table 1, §4.2] |
| | OCSPruner (WACV 2026), from scratch | 74.24 → 70.47 (mean of 3) | 0.112 | one training cycle with pruning inside | [R: LIT_AGENT_VS_RULE §5.1] |
| | In-house: the threshold rule inside DepGraph's code | Rule widths against DepGraph's allocation without sparse learning: +0.47; with sparse learning added after: −0.36 at 10k (BEHIND). The rule measured after DepGraph's sparse learning: LEVEL (10k −0.03 at 8.84×; also ResNet-56 C10 at 2.11×, 10k +0.21) | ≈ 0.11 | DepGraph's | [R: LIT_AGENT_VS_RULE §5.1]; (in-house, §450 heading) |
| MobileNetV2 C10, ~1.9× | CoSeP [28] | 94.48 → 94.81 ± 0.08 | 0.518 (1.93×) [c] | as above | [V full: Table 1] |
| | ATO; AAP | 94.45 → 94.78; 94.46 → 94.70 | 0.543; 0.508 [c] | as published | [R: CoSeP Table 1 reprint] |
| ResNet-50 ImageNet, ~2× | DepGraph | 76.15 → 75.83 | 0.481 (2.08×) | 30 sparse-learning + 90 fine-tune epochs (Torch-Pruning script; the paper gives no epochs) | [R: LIT_SHORT_RECOVERY Q2; ETP Table 1 reprint] |
| | OCSPruner | 76.29 → 76.00 / 75.95 | 0.570 / 0.517 | one 130-epoch cycle from scratch | [R: LIT_SHORT_RECOVERY Q2] |
| | ETP [67] | 76.15 → 75.62 | 0.435 (2.30×) [c] | DepGraph's setup | [V full: Table 1] |
| | ModHiFi (NeurIPS 2025) [68] | 76.1 → 76.70 | 0.461 (2.17×) [c] | torchvision base; 3 epochs after each pruning step, then 160 epochs at batch 512, cosine from 0.1, CutMix and Mixup | [V full: Table 1, App. C.7.2] |
| | CoSeP [28] | 76.32 → 76.98 ± 0.11 | 0.435 (2.30×) [c] | 3 epochs on 25 % of ImageNet after each layer, LR 0.001 | [V full: Table 2, App. C] |
| | AgenticPruner [44] | 76.13 → 77.04 | 1.77 G MACs, about 0.43 [c] | 100 epochs, distillation from the unpruned network, EMA, early stopping on validation. The footnote credits "modern training recipes" | [V full: §4.2, §5.1] |
| | REPrune; ATO; GETA; SMCP | 76.39 → 77.04; 76.13 → 76.59; 76.41 → 76.99; 76.20 → 76.80 | 0.439; 0.435; 0.50; 0.465 [c] | as published | [R: CoSeP Table 2 reprint] |
| | In-house (§427, §430): agent / rule / even cut | TEST 75.24 / 75.33 / 75.42 (honest −1.66 / −1.57 / −1.48) | 0.48 | 90 epochs, DepGraph's flow | [R: LIT_SHORT_RECOVERY Q2] |
| ResNet-50 ImageNet, ~3× | ModHiFi [68] | 76.1 → 73.82 | 0.273 (3.66×) [c] | as above | [V full: Table 1] |
| | OCSPruner; CHIP; DTP | 76.29 → 74.72; 76.15 → 75.26; 76.13 → 74.26 | 0.331; 0.372; 0.327 | as published | [R: LIT_SHORT_RECOVERY Q2; LIT_AGENT_VS_RULE §5.2] |

**Reading [H].**
- ETP's VGG-19 gain (+0.91 from the same base [c]) comes from a regulariser applied during training, like DepGraph's sparse learning. The allocation is not what changed.
- The ResNet-50 rows above 76.5 carry recipe changes that a reviewer of SPECTRA's same-loop rows would discount, as AgenticPruner's own footnote does.
- So competitiveness with the state of the art is decided by the recovery recipe and the regulariser (`LIT_AGENT_VS_RULE_22NOV.md` §8 D3 [R]), not by which allocator chose the widths.
- The in-house record agrees. Measured after DepGraph's sparse learning, the training-free rule is level with DepGraph at two of its home cells (§450). The next bar at VGG-19 CIFAR-100 is ETP's regulariser (+0.91 over DepGraph), a recipe change.
- Not found for 2025–2026: a new allocation-only method above DepGraph or OCSPruner at these cells.

---

## 8. Ranked changes to SPECTRA's agent

Ranked by expected value for "better than the strongest rule". Costs use the brief's figures [c]. Every first test is to be registered in `docs/SITTING_GPU_QUEUE.md` before any job, and quoted on TEST 5k with val, 10k and honest beside it.

### C1. A size head trained on a probe-calibrated utility (the free axis as the agent's job)
- **Idea.**
  - Give the policy an explicit output for the size: parameters kept at the FLOPs budget, or the size under the per-halving utility. Condition it on λ.
  - Train it by distillation from per-network labels chosen by a probe-calibrated utility: the 2-epoch probe plus a per-size offset fitted on recovered curves. Never use the cut.
- **Why it could beat the rule.**
  - It is the only place where room above the rule has been measured: +0.42 through 22 % more parameters (§452), and +1.0 to +1.5 utility for per-network sizes (§457). Even the recalibrated rule's gain came with 17–32 % more parameters (§448).
  - The rules set the size only implicitly, through their loss curves; none chooses it against a stated trade-off.
  - The probe ranks plans at τ 0.518 against a 0.545 ceiling (§454). The cut picks sizes that lose 4.7–7.8 utility (§457).
- **Closest prior; novelty left.**
  - Closest: AMC's −Error·log(size) reward, read at the cut and per network [15]; HiPP-Prune's budget head (one vision-language model, no recovery) [36]; Network Pruning Spaces' optimal ratio [37]; envelope and preference-driven MORL [32, 33].
  - Left: a frozen cross-family policy whose size choice is validated after a 100-epoch recovery on held-out families.
- **First test.**
  1. Zero to 2 GPU-h. On the six networks and six sizes of §457, run 2-epoch probes (36 × 3.1 min ≈ 1.9 GPU-h, if not logged). Fit the per-size offset leave-one-network-out, choose each network's size, and read the TEST utility regret against the recovered-curve choice and against one size for all, from existing runs. No new 100-epoch runs.
  2. Suggested bar: the probe choice keeps at least half of the +1.0 to +1.5.
  3. If it passes, label the 39-network pool (≈ 234 probes ≈ 12 GPU-h), train the head (small), and read 9 held-out networks × 2 λ × 2 arms (the agent's size against the best single size) × 2 seeds ≈ 72 recoveries ≈ 48 GPU-h at the ResNet-56 cost.
  4. Comparator: the same probe-calibrated choice applied to the rule's plans. The agent's honest claim is then the same utility with no per-network probes (six probes ≈ 19 GPU-min saved per network).
- **Risk.** Medium.
  - Short recovery penalises deep cuts more (§458), which biases probe choices toward larger sizes; the offset may not transfer across families.
  - A size win is not an equal-size win: report utility and hypervolume, with the equal-size row beside them.

### C2. The strongest rule as the base policy, residuals learned on top (stop losing at deep cuts)
- **Idea.** Make the recalibrated rule (§448) a menu component if it is not one. At FLOPs ≤ 0.25, add an imitation term toward it, so that the learned part is a residual on a strong controller.
- **Why.**
  - The deficit sits at FLOPs 0.20 (−1.58, §449), and the menu agent already leans on the threshold rule (share 0.67–0.84, §441).
  - In robotics, a residual on a good but imperfect controller "consistently and substantially improves" it [38, 39].
  - Gasse et al. beat SCIP's default by imitating a stronger expert [3].
- **Closest prior; novelty left.** Residual policy learning [38, 39]; the menu agent (§441). The novelty is small: this is a repair, not a claim.
- **First test.** Zero GPU: check the menu and the blend weights at 0.20 on the 9 held-out networks. Then 9 networks × FLOPs 0.20 × 2 arms × 3 seeds = 54 recoveries ≈ 36 GPU-h.
- **Risk.** Low cost. The ceiling is the rule's level plus tenths.

### C3. Test-time choice among K agent samples by 2-epoch probes at deep cuts, at equal size, with a training guard
- **Idea.**
  - At FLOPs ≤ 0.25, sample K plans from the frozen agent, plus the rule's plan, all held to the rule's size on both axes.
  - Probe each for 2 epochs, drop any whose training loss has not left ln(classes) (the §443 failure), and keep the best on val.
- **Why.**
  - ChiPFormer went from 1 to 10 wins in 12 with 300 cheap evaluations [10].
  - Plans spread more when recovery is short (§458).
  - §443's loss at 0.20 came from one untrainable plan; the other eight networks gave +0.27 val.
- **Closest prior; novelty left.**
  - Closest: ChiPFormer and AlphaChip fine-tuning [7, 10]; random-plan top-5 screening and EagleEye [R: LIT_AGENT_VS_RULE §4.2]; AgenticPruner's loop [44].
  - Left: K-probe choice among a frozen cross-family policy's proposals, at equal size, under a 100-epoch recovery.
- **First test.**
  1. Zero GPU: split §452's equal-size read by budget. Continue only if the 0.20 groups show at least +0.3 TEST.
  2. Then 9 held-out networks × K = 8 probes ≈ 3.7 GPU-h, plus 9 × 2 arms × 2 seeds ≈ 24 GPU-h.
- **Risk.** Medium-high: pooled over budgets, the pick gains +0.00 at equal size (§452).

### C4. Expert iteration on the allocation (gated on C3)
- **Idea.** Two or three batched rounds: sample K plans per training network at the deep budget, pick by probe as in C3, distil the picks into the policy (cross-entropy on plans or regression on widths), and repeat.
- **Why.**
  - Search-then-distil [3, 16, 17], with a few batched rounds [21].
  - The probe ranks plans 3.3× better than the cut (0.518 vs 0.156, §454 [c]).
- **Closest prior; novelty left.**
  - Closest: rewards after 5 epochs (N2N, within one family) and 2 epochs (Gong et al., per network) [R: LIT_DB_AVENUES DB5; LIT_GONG_CITERS].
  - Left: distilling probe-picked plans into a frozen cross-family policy; not found.
- **First test.** One round at FLOPs 0.20 only: 39 networks × 8 probes ≈ 16 GPU-h, plus a held-out read of 9 × 2 arms × 2 seeds ≈ 24 GPU-h.
- **Risk.** High at 0.45, where the allocation saturates (§452). Medium at 0.20.

### C5. A latency budget, judged against a lookup-table knapsack
- **Idea.** Add measured RTX 4090 latency at batch 1 and 256 as a budget input. Compare at equal measured latency against a HALP-style knapsack on the same table.
- **Why.**
  - Equal-FLOPs plans differ by 3–18 % in throughput (§3.1), and FLOPs rank GPU latency poorly [29].
  - The efficiency headline needs measured speed.
- **Closest prior; novelty left.**
  - Closest: HALP [26], MDP [66], NetAdapt and AMC's latency mode [15]; Archtree, which explores several pruned candidates with on-the-fly latency measurement [72]; HELP for device transfer [47].
  - Left: a frozen latency-conditioned allocator across families.
- **First test.**
  1. No training: time the existing agent, rule and even-cut plans of about 10 networks at both batch sizes (≈ 1–2 GPU-h [H]). Continue only if equal-FLOPs plans differ by at least 5 %.
  2. Then 12 networks × 2 latency budgets × 2 arms × 2 seeds ≈ 96 recoveries ≈ 63 GPU-h.
- **Risk.** High. The knapsack is exact for an additive table, and on the RTX 4090 DepGraph's ResNet-56 gains no speed from 37–61 % fewer FLOPs (§395).

### C6. Block removal in the action, only under the latency budget (gated on C5)
- **Idea.** Let the action remove whole residual blocks besides thinning channels.
- **Why.** Under FLOPs, depth adds +0.2 to +0.6 at best [56]. Latency is where depth wins [63–66].
- **Closest prior; novelty left.** MDP [66]; LayerMerge [63]; [R: LIT_AGENT_VS_RULE §8 D2].
- **First test.** The mixed-unit rule against the channel rule at equal latency, as in D2's step 2 (≈ 65 GPU-h [R: LIT_AGENT_VS_RULE §8 D2]).
- **Risk.** High under FLOPs, medium under latency.

### C7. Do not run: cut-time multi-fidelity rewards or offline learning from the logs
- **Why not.**
  - A control variate from the cut removes about 6 % of the variance (ρ ≈ 0.24) [c, H].
  - The logged plans carry no logging probabilities [H, U].
  - The cut-feature surrogate ranked plans at τ 0.081 across families (§454).
- **Zero-GPU check.** Compute the Pearson correlation between cut-time and 100-epoch outcomes on the logged groups. Revisit only if it exceeds 0.5.

**Not an agent change.** The gap to the state of the art at the ResNet-50 and VGG-19 cells is a recipe and regulariser gap (§7). Inside DepGraph's recipe the training-free rule is already level with DepGraph (§450); the next published bar there is ETP's regulariser [67]. That is `LIT_AGENT_VS_RULE_22NOV.md` §8 D3's route [R], not C1–C6.

---

## 9. Search log and limits

- **Web searches:** 33, all in standard mode except two in extended mode, on:
  - expert iteration in NAS;
  - TenSet; MLGO register allocation; the Halide autoscheduler;
  - the macro-placement critiques; ChiPFormer; the placement zero-shot results;
  - multi-fidelity RL; VeLO;
  - the GPU tail effect; HW-NAS-Bench; equal-FLOPs latency; learned against table-driven latency pruning; Gator;
  - preference-conditioned compression; Pareto set learning; per-network pruning ratios; knee-point selection;
  - few-shot pruning agents; transfer NAS;
  - joint depth and width pruning; layer pruning after 2024; LayerMerge; depth against width at matched FLOPs;
  - 2025–2026 results at each comparison cell (four); ETP's venue;
  - imitation learning for pruning.
- **arXiv API listings** (sorted by date and screened by title): "channel pruning" or "filter pruning" (150 newest), "structured pruning" with ImageNet (120), pruning with CIFAR-100 and VGG (60), pruning with ResNet-56 (60).
- **Reads.**
  - 75 arXiv abstract pages.
  - 13 PDFs as text: HALP, Gator, AMC, the AlphaChip preprint, Gasse et al., ChiPFormer, TAS, HW-NAS-Bench, CoSeP, ModHiFi, ETP, AgenticPruner and the layer-or-filter paper [60]; two-column tables re-extracted with `-raw`.
  - The arXiv HTML of [8] (§I and §VII's opening); three web pages; one Semantic Scholar record.
- **Scrambled tables, not used:** HALP's left ResNet-50 block and Table 2 (TensorRT); Table I of [60]; part of Gator's Table 2 (the "flops 1 / latency 1" rows).
- **Not opened:**
  - the Nature AlphaChip paper and its addendum;
  - the body of the updated macro-placement assessment [8] beyond §I and §VII's opening;
  - TNAS (listing only);
  - MDP, LayerPrune, NetAdapt, OCSPruner, DepGraph, CHIP and DTP, which come from earlier scans [R];
  - the papers reprinted in CoSeP's tables (REPrune, ATO, GETA, SMCP, AAP, DECORE, Torque).
- **Venues** come from the arXiv comment or journal reference, the PDF header, or the proceedings page. Exceptions, marked in §10: Decima, AlphaX, the neural predictor, Algorithm Distillation, DPT and the Efficiency Misnomer have no venue on the record read [venue U]; Gator, CRM, MFMCRL and ExIt have one from a search result [venue R].
- **Project facts** come from the brief and from ledger headings §395, §427, §430, §441, §443, §448, §449, §450, §452, §454, §455, §457, §458 and §459. The ledger body and the queue were not read.
- **Coverage.** IEEE, Elsevier and Springer venues without open copies are under-covered. "Not found" is a statement about these searches, not proof of absence.

---

## 10. References

Labels: [V full] or [V abs] = read at the source this session; [R] = reported (source named); [U] = unverified.

**Q1. Expensive rewards and generic policies**
1. Trofin, Qian, Brevdo, Lin, Choromanski, Li. MLGO: a Machine Learning Guided Compiler Optimizations Framework. arXiv:2101.04808. [V abs]
2. Google Research blog, "MLGO: A Machine Learning Framework for Compiler Optimization", 6 Jul 2022. research.google/blog/mlgo-a-machine-learning-framework-for-compiler-optimization/. [V full: blog page, through the page reader]
3. Gasse, Chételat, Ferroni, Charlin, Lodi. Exact Combinatorial Optimization with Graph Convolutional Neural Networks. NeurIPS 2019. arXiv:1906.01629. [V full: Tables 1–2, §5]
4. Adams, Ma, Anderson, Baghdadi, Li, Gharbi, Steiner, Johnson, Fatahalian, Durand, Ragan-Kelley. Learning to Optimize Halide with Tree Search and Random Programs. ACM TOG 38(4), SIGGRAPH 2019. halide-lang.org/papers/autoscheduler2019.html. [V abs]
5. Zheng, Liu, Shao, Chen, Gonzalez, Stoica, Haj-Ali. TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers. NeurIPS 2021 Datasets and Benchmarks. [V abs: proceedings page]
6. Mao, Schwarzkopf, Venkatakrishnan, Meng, Alizadeh. Learning Scheduling Algorithms for Data Processing Clusters (Decima). arXiv:1810.01963 [venue U]. [V abs]
7. Mirhoseini, Goldie, Yazgan, et al. (22 authors). Chip Placement with Deep Reinforcement Learning. arXiv:2004.10746 (2020 preprint; the precursor of the Nature 2021 paper [R: search results]). [V full: §5.1]
8. Cheng, Kahng, Kundu, Wang, Wang. An Updated Assessment of Reinforcement Learning for Macro Placement. arXiv:2302.11014 v3 (IEEE TCAD, per the arXiv comment). [V full: abstract, §I, §VII opening (arXiv HTML); other sections not read]
9. Goldie, Mirhoseini, Dean. That Chip Has Sailed: A Critique of Unfounded Skepticism Around AI for Chip Design. arXiv:2411.10053. [V abs]
10. Lai, Liu, Tang, Wang, Hao, Luo. ChiPFormer: Transferable Chip Placement via Offline Decision Transformer. ICML 2023 (PMLR 202). arXiv:2306.14744. [V full: Table 2, §4]
11. Shi, Xue, Song, Qian. Macro Placement by Wire-Mask-Guided Black-Box Optimization. NeurIPS 2023. arXiv:2306.16844. [V abs]
12. Metz, Harrison, Freeman, et al. VeLO: Training Versatile Learned Optimizers by Scaling Up. arXiv:2211.09760. [V abs]
13. Rezk, Antoniou, Gouk, Hospedales. Is Scaling Learned Optimizers Worth It? Evaluating The Value of VeLO's 4000 TPU Months. arXiv:2310.18191. [V abs]
14. Joshi, Cappart, Rousseau, Laurent. Learning the Travelling Salesperson Problem Requires Rethinking Generalization. CP 2021 and Constraints 2022. arXiv:2006.07054. [V abs]
15. He, Lin, Liu, Wang, Li, Han. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018. arXiv:1802.03494. [V full: §3.4, §4.1, Tables 2 and 4]
16. Anthony, Tian, Barber. Thinking Fast and Slow with Deep Learning and Tree Search (ExIt). NeurIPS 2017 [venue R: proceedings link in search]. arXiv:1705.08439. [V abs]
17. Wang, Zhao, Jinnai, Tian, Fonseca. AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search. arXiv:1903.11059 [venue U]. [V abs]
18. Khairy, Balaprakash. Multifidelity Reinforcement Learning with Control Variates. arXiv:2206.05165 (Neurocomputing 2024 [venue R: search snippet]). [V abs]
19. Liu, Neary, Gupta, Suttle, Ellis, Topcu, Fridovich-Keil. A Multi-Fidelity Control Variate Approach for Policy Gradient Estimation. arXiv:2503.05696 [venue U]. [V abs]
20. Swaminathan, Joachims. Counterfactual Risk Minimization: Learning from Logged Bandit Feedback. arXiv:1502.02362 (ICML 2015 [venue R]). [V abs]
21. Han, Zhou, Hu, Blanchet, Glynn, Ye, Zhou. Sequential Batch Learning in Finite-Action Linear Contextual Bandits. arXiv:2004.06321 ("To appear in Operations Research"). [V abs]
22. Wen, Liu, Li, Chen, Bender, Kindermans. Neural Predictor for Neural Architecture Search. arXiv:1912.00848 [venue U]. [V abs]
23. Dudziak, Chau, Abdelfattah, Lee, Kim, Lane. BRP-NAS: Prediction-based NAS using GCNs. NeurIPS 2020. arXiv:2007.08668. [V abs]
24. White, Neiswanger, Savani. BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search. AAAI 2021. arXiv:1910.11858. [V abs]
25. Ma, Cui, Qin. Beyond Manually Designed Pruning Policies with Second-Level Performance Prediction (PPF). arXiv:2508.02381. [V abs]

**Q2. Latency**
26. Shen, Yin, Molchanov, Mao, Liu, Alvarez. Structural Pruning via Latency-Saliency Knapsack (HALP). NeurIPS 2022. arXiv:2210.06659 (earlier arXiv:2110.10811). [V full: Table 1 EagleEye block, §4.1, appendix on the latency tables]
27. Passov, David, Netanyahu. Gator: Customizable Channel Pruning of Neural Networks with Gating. arXiv:2205.15404 (ICANN 2021 [venue R: search snippet]). [V full: Table 2]
28. Levin, Singer. CoSeP: Complementary Separability Pruning via Class-Separability Clustering. BMVC 2026 (arXiv comment). arXiv:2505.13225 v3. [V full: Tables 1, 2, 6, 7]
29. Li, Yu, Fu, et al. HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark. ICLR 2021. arXiv:2103.10584. [V full: Tables 3, 4]
30. Yu, Xu, Shen, et al. Towards Latency-aware DNN Optimization with GPU Runtime Analysis and Tail Effect Elimination. arXiv:2011.03897. [V abs]
31. Radu, Kaszyk, Wen, Turner, Cano, Crowley, Franke, Storkey, O'Boyle. Performance Aware Convolutional Neural Network Channel Pruning for Embedded GPUs. IISWC 2019 (arXiv comment). arXiv:2002.08697. [V abs]

**Q3. Preference conditioning and the size choice**
32. Yang, Sun, Narasimhan. A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation (envelope MORL). NeurIPS 2019. arXiv:1908.08342. [V abs]
33. Basaklar, Gumussoy, Ogras. PD-MORL: Preference-Driven Multi-Objective Reinforcement Learning Algorithm. ICLR 2023. arXiv:2208.07914. [V abs]
34. Navon, Shamsian, Chechik, Fetaya. Learning the Pareto Front with Hypernetworks. ICLR 2021. arXiv:2010.04104. [V abs]
35. Lin, Yang, Zhang, Zhang. Pareto Set Learning for Expensive Multi-Objective Optimization. NeurIPS 2022. arXiv:2210.08495. [V abs]
36. Bai, Tabia, Santos-Rodriguez. HiPP-Prune: Hierarchical Preference-Conditioned Structured Pruning for Vision-Language Models. arXiv:2603.06270. [V abs]
37. He, Yang, Song, Pu, Hu, Jiang, Zhang, Ding. Network Pruning Spaces. arXiv:2304.09453. [V abs; R: LIT_DB_AVENUES ref. 7]
38. Silver, Allen, Tenenbaum, Kaelbling. Residual Policy Learning. arXiv:1812.06298. [V abs]
39. Johannink, Bahl, Nair, et al. Residual Reinforcement Learning for Robot Control. arXiv:1812.03201. [V abs]
40. Liu, Wu, Huang, Gao, Wang, Xue, Qian. Pareto Set Learning for Multi-Objective Reinforcement Learning. AAAI 2025. arXiv:2501.06773. [V abs]
41. Sukthanker, Zela, Staffler, Dooley, Grabocka, Hutter. Multi-objective Differentiable Neural Architecture Search (MODNAS). ICLR 2025 [venue R: LIT_ALTERNATIVES ref. 97]. arXiv:2402.18213. [V abs]
42. Li, Chen, Liu, Zhu, Tian, Liu. MCMC: Multi-Constrained Model Compression via One-Stage Envelope Reinforcement Learning. IEEE TNNLS 2024. doi:10.1109/TNNLS.2024.3353763. [V abs: Semantic Scholar record]
43. Zhan, Lin, Cao. Deep Model Compression Via Two-Stage Deep Reinforcement Learning. ECML-PKDD 2021 (arXiv comment). arXiv:1912.02254. [V abs]
45. Wright, Igel, Selvan. BMRS: Bayesian Model Reduction for Structured Pruning. NeurIPS 2024. arXiv:2406.01345. [V abs]
46. Diao, Wang, Zhan, Yang, Ding, Tarokh. Pruning Deep Neural Networks from a Sparsity Perspective (PQ Index). ICLR 2023. arXiv:2302.05601. [V abs]

**Q4. Few-shot adaptation**
44. Esmat, Banisharif, Jannesari. AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search. arXiv:2601.12272. [V full: §4.2, §4.3, §5.1]
47. Lee, Lee, Chong, Hwang. HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning. NeurIPS 2021. arXiv:2106.08630. [V abs]
48. Lee, Hyung, Hwang. Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets (MetaD2A). ICLR 2021. arXiv:2107.00860. [V abs]
49. Wistuba, Grabocka. Few-Shot Bayesian Optimization with Deep Kernel Surrogates. ICLR 2021. arXiv:2101.07667. [V abs]
50. Shala, Elsken, Hutter, Grabocka. Transfer NAS with Meta-learned Bayesian Surrogates. ICLR 2023. [R: ICLR listing and AutoML.org summary via search; not opened]
51. Laskin, Wang, Oh, et al. In-context Reinforcement Learning with Algorithm Distillation. arXiv:2210.14215 [venue U]. [V abs]
52. Lee, Xie, Pacchiano, Chandak, Finn, Nachum, Brunskill. Supervised Pretraining Can Learn In-Context Reinforcement Learning (DPT). arXiv:2306.14892 [venue U]. [V abs]
53. Akhauri, Abdelfattah. Multi-Predict: Few Shot Predictors For Efficient Neural Architecture Search. arXiv:2306.02459. [V abs]
54. Zhao, Shuai, Bai, Ling, Guan, Yan, Xing. Moses: Efficient Exploitation of Cross-device Transferable Features for Tensor Program Optimization. arXiv:2201.05752. [V abs]
55. Mu et al. Boosting the Convergence of Reinforcement Learning-Based Auto-Pruning Using Historical Data. IEEE TCAD. arXiv:2107.08815. [R: LIT_DB_AVENUES ref. 49]

**Q5. Mixed granularity**
56. Dong, Yang. Network Pruning via Transformable Architecture Search (TAS). NeurIPS 2019. arXiv:1905.09717. [V full: Table 3]
57. Wang, Chen, Zhao, et al. Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework. ICML 2021. arXiv:2010.04879. [V abs]
58. Liu, Zhang, Shen, Li, Wei, Cheng, Sun. Joint Multi-Dimension Pruning via Numerical Gradient Update. IEEE TIP 2021. arXiv:2005.08931. [V abs]
59. Pons, Yamamoto, Costa, Jordao. Effective Layer Pruning Through Similarity Metric Perspective. ICPR 2024. arXiv:2405.17081. [V abs]
60. Nascimento, Pons, Costa, Jordao. Pruning Everything, Everywhere, All at Once. IJCNN 2025 (arXiv comment). arXiv:2506.04513. [V full: Table III, §IV; Table I scrambled, not used]
61. Mugnaini, Duarte, Costa, Jordao. Layer Pruning with Consensus: A Triple-Win Solution. arXiv:2411.14345. [V abs]
62. Sadaqa, Liu. Compressing CNN models for resource-constrained systems by channel and layer pruning. ECML PKDD (arXiv comment). arXiv:2509.08714. [V abs]
63. Kim, El Halabi, Ji, Song. LayerMerge: Neural Network Depth Compression through Layer Pruning and Merging. ICML 2024. arXiv:2406.12837. [V abs]
64. Kim, Jeong, Lee, Song. Efficient Latency-Aware CNN Depth Compression via Two-Stage Dynamic Programming. ICML 2023. arXiv:2301.12187. [V abs]
65. Fu, Yang, Yuan, et al. DepthShrinker. ICML 2022. arXiv:2206.00843. [V abs]
66. Sun, Lakshmanan, Shen, Lan, Chen, Alvarez. MDP: Multidimensional Vision Model Pruning with Latency Constraint. CVPR 2025. arXiv:2504.02168. [V abs; R: LIT_AGENT_VS_RULE ref. 4]

**Q6. State of the art**
67. Modi, Lim, Kuchhal, Cao, Cheng, Teo, Lin, Li. Towards Universal & Efficient Model Compression via Exponential Torque Pruning (ETP). arXiv:2506.22015 v3; no venue on the record, "Under review" per search. [V full: Table 1, §4.2]
68. Kashyap, Murti, Nayak, Narshana, Bhattacharyya. ModHiFi: Identifying High Fidelity predictive components for Model Modification. NeurIPS 2025 (spotlight). arXiv:2511.19566. [V full: Table 1, App. C.7.2]
69. Rows from earlier scans, not re-read: DepGraph (CVPR 2023), OCSPruner (WACV 2026), CHIP (NeurIPS 2021), GReg (ICLR 2021), DTP (ICCV 2023), metanetwork pruning. [R: LIT_SEALED_WEB §1; LIT_AGENT_VS_RULE §5; LIT_SHORT_RECOVERY Q2]
70. Rows reprinted in CoSeP's Tables 1–2, not opened: REPrune, ATO (CVPR 2024), GETA, SMCP, AAP, DECORE, Torque. [R: [28] reprint]

**Cited in passing**
71. Dehghani, Arnab, Beyer, Vaswani, Tay. The Efficiency Misnomer. arXiv:2110.12894 [venue U]. [V abs]
72. Reboul, Yvinec, Dapogny, Bailly. Archtree: on-the-fly tree-structured exploration for latency-aware pruning. arXiv:2311.10549. [V abs]
73. Yuan et al. FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning. arXiv:2511.18977. [V abs]
