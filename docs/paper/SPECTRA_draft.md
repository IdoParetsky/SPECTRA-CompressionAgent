# SPECTRA — paper draft (NEON format)

Working title: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Ido Paretsky · advisor Dr. Gilad Katz · Ben-Gurion University  
Predecessor: Hirsch & Katz, *Information Sciences* 2022 (NEON) [1].  
**Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md).** Every number below cites its ledger section (§N). Do not paste a job summary, a train-log value or a smoke here.

**Status: DRAFT, rewritten 14 Oct 2026 to the plan-as-action agent (10:40); updated 14 Oct 14:39 with §381, 19:42 with §382 and 21:20 with §383, and 15 Oct 01:40 with §384, 02:08 with §385, 08:05 with §386, 10:45 with §387, 13:00 with §388, 15:10 with §389 and 17:20 with §390, and 16 Oct 04:15 with §391, 05:50 with §392, 06:40 with §393–§395, 13:45 with §396, 14:40 with §397, 16:55 with §398 and 18:50 with §399, and 17 Oct 07:39 with §400, 09:47 with §401, 10:09 with §402, 19:26 with §403, and 19 Oct 07:15 with §404–§405.** The method of record is the plan-as-action agent (§3). The walk-based agent of the 16 Sep draft is history (Appendix B), and the old text, with the 18 Aug snapshot and ops' one-line pins of 6–14 Oct, is kept word for word in [SPECTRA_draft_archive_walk_era.md](SPECTRA_draft_archive_walk_era.md). Results are **PRELIM** until the final tables are locked. Submission deadline: 28 Feb (Ido, 13 Oct). Standing advisor orders: [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md).

**Who edits this file.** Science (Claude Code) owns it under Ido's standing GO of 14 Oct (CLAUDE.md, "The paper draft") and updates it after each ledger section, one reviewable commit at a time. Ops does not edit it; ops' pins are folded into the body and into the claims guard below.

**Conventions.**
- Section order follows NEON [1]: Introduction → Related Work → Approach → Evaluation → Results → Discussion → Conclusions.
- A result is a **TEST** on the 5k test half, with the val half, the full 10k test set and the honest change beside it (§4.2). For a difference between two pruned arms, honest is the difference of the arms' honest values. Each job runs its own origin control, so honest can differ from raw (§352). Selection never reads TEST.
- FLOPs is the headline axis for rows that face the literature, reviewers and the defense, with params beside them (Ido, 10 Oct). An agent row is headlined on the budget it trained on. When a pair differs by more than 10 % in FLOPs, the caption says so.
- Each budget has one prior, fixed in advance from pooled evidence and run with the agent's floors: the cost-aware sensitivity rule at equal params (§355, §370) and the floored FLOPs sensitivity rule at equal FLOPs (§357, §366). Every other same-loop rule is reported beside it.
- Tags: **LOCKED** / **PRELIM** / **TBD** / **CLAIM** ([README.md](README.md)). Citations: [1]–[78] are the August 2024 thesis proposal's; [79]–[96] come from the August 2026 survey; [97] onward come from the October 2026 scans.

**Claims guard (what this draft must not say).** These consolidate ops' pins of 6–14 Oct and the ledger's never-lists. A line leaves this list only when a registered call in the ledger fires.
- Never "beats" (a rule, DepGraph, or the state of the art) unless a registered BEATS call fires. In particular, never "the agent beats the cost-aware rule" in general, at any depth, nor "beats DepGraph". The one registered BEATS call that has fired (§382: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only) is quoted only with that scope; it does not carry over to the deep-range batch-norm agent (§385). Never "the deep-range batch-norm agent beats the rule at 45 %": its +0.2956 is under the bar (§384). Never "the 32-step agent beats the rule at 35 %": its +0.2356 is under the bar, though its 95 % range lies above zero (§387). A second, pooled BEATS call has fired (§390): the 32-step agent (T6-fs) beats the cost-aware rule at 45 % params, pooled over the nine held-out networks (+0.42, 6 of 9 positive; val, 10k and honest all above +0.3). Quote it only with that scope, with the other eight (+0.19) and the FLOPs caption beside it, and never at another depth or as a general property of the agent. A further registered call has fired against a different comparator (§396): the frozen 32-step agent beats a per-network random search, the best of 100 random plans scored by the agent's own fs32 reward, at 45 % and 25 % params pooled over the nine networks (+0.51 with 8 of 9 positive; +0.45 with 6 of 9). Quote it only with its scope. At 25 % the bootstrap range includes zero, and ResNet-20 ×8 CIFAR-100 goes the other way (−2.67). The agent keeps 1.1–2.1× the search's FLOPs. Never "beats Li et al.": this is our implementation of their baseline.
- Never "headroom" above a prior: every headroom screen so far found none (§367, §368, §372).
- Never "transfers to unseen families" in general. Family transfer is claimed only fold by fold, as §363 states it. Never "the agent transfers to VGG-19 CIFAR-100" or "the no-VGG fold transfers to VGG" (§363).
- Never "the agent conditions on the budget type" (§360). Never "advantage normalization helps" (§364). Never "a training floor or a deeper training range fixes the collapse" (§376, §377). Never "the batch-norm reward removes the collapse at 25 %": one seed still collapses there (§383). Never "a seed ensemble fixes the collapse" or "is more robust": averaging the three seeds' scores leaves it (§386). Never "unseen dataset" for the CIFAR-100 rows of an agent trained with CIFAR-100 in its catalog (T7-C, §389; T8-fs, §400). Never "a bigger catalog hurts" in general: §400 is one reward line at one instance budget, carried by two CIFAR-100 networks.
- Never a proxy's value, a train-log value, a smoke, a projection, `eval_train` or an in-walk number as a result. Never say a proxy is valid unless its registered fidelity call says VALID (§375, §379). Never the 1-epoch final as a result.
- Never mix budgets or depths in one contrast. Never drop the network that carries a result (ResNet-20 ×8 on CIFAR-100 at 35 % and 25 % params, §374, §380).
- The walk-era DepGraph comparison (§157, "M4") keeps its caption; it is not re-quoted for the plan agent (Appendix B).

**Where the evidence stands (updated 17 Oct 2026; PRELIM; TEST with val, 10k and honest beside it in §5).**

| Question | Answer so far | § |
|---|---|---|
| Does the frozen agent learn? | Yes. It beats mild and uniform on held-out networks, and it reaches, without passing, the sensitivity rule | §341, §346, §363 |
| Equal parameters, nine held-out networks | With the raw-cut reward: at the cost-aware rule at 60 % and 45 % kept, below it at 35 % and 25 %, decided by one network (ResNet-20 ×8 CIFAR-100). With the batch-norm reward on the deep training range: **one frozen agent at the rule at all four depths read**, 60 %, 45 %, 35 % and 25 %. The 32-step reward on the same range **beats the rule at 45 %** (+0.42, the first pooled beat; §390) and is at it at 60 %, 35 % and 25 %, better than batch-norm at 25 % | §370, §373, §374, §380, §383, §384, §387, §390 |
| Equal FLOPs, nine held-out networks | **Final, five deterministic seeds:** at the FLOPs rule at 60 % and 45 % kept (−0.16 at both; AT-PRIOR-F-POOL). The deficit sits on ResNet-20 ×8 CIFAR-100; the seven CIFAR-10 networks are level (−0.03, +0.04). The three-seed rows read −0.21 | §366, §371, §405 |
| Is there an allocation above the priors? | No headroom found at moderate budgets, for either budget type | §367, §368, §372 |
| What removes the deep-budget failure? | Only a recovery-aware (batch-norm) reward. On the deep training range it removes the failure at 35 % with no thin-ResNet cost, and at 25 % shrinks it from −16.8 to −2.4, with one seed still collapsing. Averaging the three seeds' scores does not remove that collapse. The 32-step reward shrinks it to −0.9, with no all-or-nothing plan. CIFAR-100 networks in the training catalog remove it at 25 % (+2.09; CATALOG-HELPS). The two together do not: one seed of the combined agent fails on that network (§400) | §376, §377, §378, §381, §383, §386, §387, §389, §400 |
| Which cheap reward is faithful? | One epoch of fine-tuning is valid at the threshold; the raw cut and batch-norm re-estimation are weak at 35 % params | §375, §379 |
| Transfer across families | The no-ResNet fold holds; the no-VGG fold fails; families no catalog holds are lever-limited | §353, §359, §363 |
| Against the literature (DepGraph's ResNet-56; VGG-16; ResNet-110) | At DepGraph's 2.11× and 2.57× points the FLOPs rule lands above DepGraph's published change, and the frozen FLOPs agent at or below it. On VGG-16 and ResNet-110 at the field's standard FLOPs points the frozen FLOPs agent is at the rule, and the allocation matters little. The recipes and base checkpoints differ, and ours also lifts the unpruned network; never a beat | §362, §365, §382, §391, §392 |
| How far does the agent go? | To 8× on DepGraph's ResNet-56: at the prior at 8× under both budgets, while the priors beat uniform by 2–3 points there. Two single-network calls fired on TEST only, within run noise | §399 |
| Has a registered beat fired? | Twice against the params prior, and once on one network at equal FLOPs (DepGraph's ResNet-56 at FLOPs 0.25, +0.40 on TEST alone, val −0.23, 10k +0.09; within run noise; §399). Narrowly, on one network: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only (val and 10k do not carry it). Pooled: the 32-step agent at 45 % params over the nine networks, +0.42 with val +0.34, 10k +0.38 and honest +0.46. One network carries part of it; the other eight average +0.19 | §382, §385, §390 |
| Does training longer help? | Not on this catalog: T6-bn's edge on the other eight is flat on val from 3,000 to 12,000 instances, and the collapse on ResNet-20 ×8 CIFAR-100 appears only from 6,000 | §388 |
| Do the two levers add up? | No. The 32-step agent's line on the 18-network catalog with CIFAR-100 is level with the 32-step agent at 60 %, 45 % and 35 %, and below it at 25 % (CATALOG-HURTS), carried by the two CIFAR-100 networks. On the seven CIFAR-10 networks every contrast lies within −0.12 to +0.22 | §400 |
| Does a per-network search match the frozen agent? | No. A random best-of-100, scored by the agent's own cheap reward on each network, ends below the frozen agent at 45 % and 25 % params (+0.51 and +0.45 for the agent), and level with the cost-aware rule. At 25 % one CIFAR-100 network goes the other way | §396 |
| Does the agent's choice of channels matter? | Little, after the 100-epoch recovery: L1 channels beat random channels at the agent's own widths by +0.13 and +0.23 (RANK-SAME). The widths carry the result | §398 |
| Do the inherited weights matter? | With Scratch-B's compute, the 32-step agent's architecture trains from scratch to the same accuracy as with its inherited weights (SCRATCH-SAME at both DepGraph points). With Scratch-E's 200 epochs, the inherited weights help at 38 % params (+0.39) | §394 |
| Are fewer FLOPs faster? | Not on DepGraph's ResNet-56 on an RTX 4090: 37–61 % fewer FLOPs run at 0.93–1.07× at batch 256. VGG-16 gains 1.32–1.65×. Batch-1 latency is unchanged | §395 |
| How reproducible is one cell? | Without deterministic algorithms, not exactly: three runs of one cell spread over 0.42 and 0.76 points on TEST. With `SPECTRA_EVAL_FT_DETERMINISTIC=1`, two runs agree exactly on five networks in four families (residual, MobileNetV2, DenseNet, VGG), at about 22–50 % more wall time | §393, §397, §401 |
| Running | The final tables: T6-fs and T4-F against their priors at five seeds with the deterministic fine-tune (FINAL); a lever screen at params 0.125 on SVHN and Fashion-MNIST (LEV-125) | queue |

---

## Abstract — DRAFT (14 Oct 2026)

Convolutional networks remain the workhorse of real-time vision, and structured pruning, which removes whole channels, is the form of compression that actually shrinks the work a GPU does [7, 11, 12]. At a given budget, much of the accuracy that survives depends on the *allocation*: how much of each layer is kept. Today allocation is solved one network at a time: by a search, by a reinforcement-learning controller trained for that network [79], or by a hand rule such as layer sensitivity [18]. SPECTRA asks whether one policy, trained offline on a catalog of networks and then frozen, can allocate a budget on networks it has never seen, as NEON did for dense networks [1].

The agent reads a pretrained CNN as a sequence of per-layer tokens and proposes a whole keep plan in one step. A Transformer scores every channel group; a decoder turns the scores into integer channel widths that meet a parameter or FLOPs budget to the nearest channel, with per-group floors and rounding that make every plan realizable. The agent is trained as a contextual bandit with REINFORCE against cheap proxies of post-fine-tune accuracy. Every plan, the agent's and the rules', is cut once and recovered with the same fixed fine-tune. The agent is compared plan for plan with same-loop allocation rules, against a prior fixed in advance for each budget type.

On nine held-out networks, the frozen agent learns allocations well above the uniform and mild rules and performs at the level of the strongest rule at 60 % and 45 % of parameters kept (§370, §373). At equal FLOPs it stays within the registered band of the strongest rule but below it on most networks (§366, §371). Screens of many alternative plans found no allocation above either prior at these budgets. At deeper budgets one held-out network exposes a reward-fidelity failure that grows with depth. A batch-norm-recalibrated reward on a deeper training range removes most of it. That one frozen agent performs at the level of the strongest rule at every parameter budget read, from 60 % down to 25 % kept, though one seed still collapses at 25 % (§383, §384). With a reward closer to recovery (32 fine-tune steps), the agent beats the strongest rule at 45 % of parameters kept, pooled over the nine networks (+0.42), and stays at its level at the other three budgets (§387, §390). Adding eight CIFAR-100 networks to that agent's training catalog does not improve it, and lowers it at 25 % (§400). A ranking study shows which cheap rewards order plans the way the final fine-tune does. Transfer across families is mixed: an agent trained without any ResNet holds on a held-out ResNet, and one trained without any VGG fails on a held-out VGG. **[PRELIM; numbers in §5 by ledger section]**

---

## 1. Introduction — DRAFT

Convolutional Neural Networks (CNNs) transformed computer vision, in classification, detection and segmentation [2], by stacking learned filters. Residual Networks [3] and DenseNets [4] made very deep stacks trainable through skip and dense connections. That accuracy costs compute and memory, and at the edge or in real time the cost is what decides deployment [5].

CNN pruning [6, 10] removes redundant components. Unstructured pruning zeroes individual weights [13–16]: it looks strong on paper but does not reduce inference cost on commodity GPUs. Structured pruning removes whole filters and channels [7, 11, 12, 18], and so shrinks the matrices that are actually multiplied. Two decisions remain once the granularity is fixed: *which* channels to remove inside a layer (the importance criterion, e.g. L1 magnitude [18]), and *how many* to remove from each layer (the allocation). Residual additions and concatenations couple layers into groups that must be cut together, so the allocation is made per channel group [12, 81].

Allocation matters. At 25 % of parameters kept, a cost-aware sensitivity allocation beats a uniform one by +0.88 points of TEST accuracy, averaged over the nine held-out networks studied here (val +1.03, 10k +0.95, honest +0.86), and by 2–3 points on three of them (§380, §5.2). Existing methods find a good allocation per network: by search or a learned controller for that network [79, 80], by a global criterion [12, 17–26], or by a hand rule [18, 24]. Each new architecture or dataset means another search.

NEON [1] showed that one preference-aware deep-RL agent, trained offline on many *dense* networks, can prune unseen dense networks without per-network retraining. SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) asks the same question for structured CNN pruning: can one frozen agent, trained offline on a catalog of CNNs, allocate a budget on networks and datasets it has never seen, at least as well as the best hand rule?

The first answer was NEON's design moved to CNNs: an agent that walks the network layer by layer, fine-tunes after each cut, and is scored on the running accuracy change. It collapsed onto the safe "keep 90 % everywhere" action, because its per-step reward could not tell its actions apart (Appendix B, ledger §331). SPECTRA therefore treats allocation as one decision per network: the agent proposes a whole plan, the plan is cut once, and a cheap proxy of post-fine-tune accuracy scores it. This turns an episodic RL problem with sparse, noisy credit into a contextual bandit with one clean reward per plan, and it shortens a training run from days to hours (§3.7).

**Contributions (as intent; status and numbers in §5):**

1. **A plan-as-action formulation of transferable allocation** for structured CNN pruning. A Transformer over per-layer tokens scores every channel group. A decoder that matches the requested parameter or FLOPs budget to the nearest channel, with per-group floors and grid rounding, makes every plan realizable. The agent is trained offline as a contextual bandit with REINFORCE and a shared baseline over K = 8 plans per network (§3).
2. **A same-loop evaluation protocol** that isolates allocation. The agent and every rule are cut once at the same budget and recovered with one fixed fine-tune. The read uses disjoint validation and test halves, an "honest" change that subtracts the unpruned network's own change under the same fine-tune, three paired seeds, and calls registered before any GPU time. One prior per budget type is fixed in advance (§4).
3. **Evidence that a frozen agent learns allocation and transfers it.** On nine held-out networks across four families and two datasets, the agent beats the uniform and mild rules and performs at the level of the strongest rule at moderate budgets, under both budget types. With a recovery-aware reward, one agent does so at every parameter budget read, 60 % to 25 % (§383, §384). Screens found no allocation above the priors at those budgets (§5.1–§5.4).
4. **A reward-fidelity study.** It measures which cheap rewards rank plans the way a 100-epoch fine-tune does, at equal parameters and at equal FLOPs. Where the training reward misranks plans, the frozen agent fails on one network at deep budgets; a recovery-aware reward removes that failure at 35 % and reduces it at 25 % (§5.6, §5.7).
5. **A transfer map** of leave-one-family-out folds and unseen-dataset cells, which states where a frozen allocation policy holds and where it fails (§5.5).

Out of scope: NEON's 28 tabular datasets; DRL training on ImageNet (frozen transfer only; Gilad, 18 Aug); vision Transformers [42]. SPECTRA does not claim a new grouping algorithm (that layer belongs to DepGraph [12] and SPA [81]), and it does not claim to beat a focused method on its own architecture and dataset.

---

## 2. Related Work — DRAFT

This section follows NEON’s related-work order [1]: reinforcement learning, neural-network pruning, then neural architecture search. CNN background and structured-CNN methods sit inside §2.2. The arena is generic *offline* DRL for structured CNN allocation at a user budget, versus same-loop allocation rules. It is not ImageNet ResNet-50 versus 2024 grouping engines.

Two kinds of generality are both real and sit on different layers. **Mechanism-level generality** — one *tool* that can prune many architectures — is occupied well by DepGraph [12] and SPA [81]. **Policy-level generality** — one *agent* that transfers across nets and datasets under a user's trade-off (NEON: an accuracy band τ; SPECTRA: a parameter or FLOPs budget) — is NEON’s idea [1], moved to CNNs. No 2023–2026 paper in the surveys closes that lane for CNNs; the closest reuse a policy within one family or one architecture (§2.2.6).

### 2.1 Reinforcement learning

“Reinforcement Learning (RL) is the problem faced by an agent that learns behavior through trial-and-error interactions with a dynamic environment” [46]. Applications range from robotics [47] and games [48] to routing [49] and dialogue [50, 51]. At step *t* the agent observes state *s_t*, picks *a_t* ∈ *A*, the environment returns *s_{t+1}* and a scalar reward [1].

Deep RL combines RL with deep networks [52]. SPECTRA, like NEON, uses an on-policy policy-gradient family. NEON’s derivation is REINFORCE [53] and its implementation is A2C [151]; the walk agent used A2C and then PPO [152]. SPECTRA's plan agent uses REINFORCE directly. Each decision is a whole plan with a single reward, which makes the problem a contextual bandit [102, 103]. The baseline is the mean reward of K plans sampled for the same network [97–100]. The policy update (proposal Eq. 1) is

*h_{t+1} = h_t + α · G_t · ∇ log p(a_t | s_t; h_t)*,

with *G_t* the return from *t*. For a one-step plan, *G* is the plan's reward minus the shared baseline of its K sibling plans. AMC [79] is the canonical *CNN* RL compressor: a DDPG controller searches layer-wise compression for a *given* MobileNet / VGG. SPECTRA’s contrast is the same as NEON vs per-task RL: the actor is trained once on a catalog and frozen at eval. Lookahead-search RL for channel pruning [57] is likewise a per-target search, not an offline multi-net agent.

### 2.2 Neural network pruning

#### 2.2.1 From dense nets to CNNs

Neural-network pruning dates to the early 1990s [6]. Optimal Brain Damage [8] and Optimal Brain Surgeon [9] used second-order criteria at high Hessian cost [10]. Modern CNNs spend most inference time in convolutions, so removing whole feature maps / filters is the practical lever [7, 10, 18].

ResNets [3] add identity skips; DenseNets [4] concatenate all preceding feature maps. Those links couple channels: pruning a filter in one layer forces aligned cuts in add / concat partners. That is why a CNN pruner cannot treat layers as independent dense maps the way NEON could.

#### 2.2.2 Structured vs unstructured

Structured pruning operates at channels, filters, or feature maps and reduces the matrices the GPU actually multiplies [7, 11, 12]. Unstructured pruning zeros individual weights [13–15] and typically wins compression *ratio* [16] without winning inference on dense kernels. SPECTRA is structured only.

#### 2.2.3 Global vs local; generalizability

Benchmarking, from the proposal, still has two axes. **Global optimality:** does the method see the whole net [17–26], or only one or two successive layers [8, 27]? **Generalizability:** can it prune a previously unseen architecture and dataset without another controller training? NEON [1] is the dense-DNN existence proof of the second axis, with a user trade-off later echoed in OCNNA [28] and in interactive plans such as CNNPruner [29]. Filter pruning for efficient ConvNets [18] is the importance ranking SPECTRA *uses inside each channel group*; the DRL policy chooses only the *allocation*, the keep fraction of each group. The same paper's per-layer sensitivity analysis [18] is the origin of the sensitivity rules SPECTRA is compared with (§4.1). Auto-balanced filter pruning [19], GDP [20], lottery tickets [21], importance estimation [22], Gate Decorator [23], layer-adaptive magnitude [24], manifold-regularized pruning [25], and ThiNet-style algorithms [26] are global or near-global *per-model* methods. Lookahead magnitude pruning [27] is a far-sighted alternative of the same ranking family — conceptually related to the walk era's look-ahead *greedy* control (Appendix B), not to the learned agent.

Proposal Table 1 still has the right axes. CONVNETS [18], AFP [19], GDP [20], DeepPruningES [30], FPAC [37], multi-layer residual compression [34], and DepGraph [12] are automatic and often global; none combine offline multi-architecture / multi-dataset training with an explicit user trade-off the way NEON (an accuracy band τ) and SPECTRA (a parameter or FLOPs budget) do. DepGraph’s and SPA’s [81] “adaptability” is *grouping* adaptability, not policy transfer. AMC [79] has a resource budget on a *given* net, not a transferred policy.

**Table 1 — feature comparison (proposal Table 1, plus 2018–2024 neighbors).** Y = yes; N = no; P = partial. “Adaptability” for DepGraph / SPA = can group many architectures, not “one frozen agent on an unseen dataset.”

| Method | Non-greedy | Global view | Adaptability | Automatic | Comp.–acc. trade-off |
|---|---|---|---|---|---|
| CONVNETS [18] | Y | Y | Y | N | N |
| AFP [19] | Y | Y | Y | Y | N |
| GDP [20] | N | Y | Y | Y | N |
| DeepPruningES [30] | Y | Y | Y | Y | P |
| FPAC [37] | N | N | Y | Y | N |
| Multi-layer ResNet compression [34] | Y | Y | N | Y | N |
| DepGraph [12] | Y | Y | Y (groups) | Y | N |
| SPA [81] | Y | Y | Y (groups) | Y | N |
| AMC [79] | Y | Y | N (per target) | Y | P (resource) |
| MetaPruning [80] | Y | Y | P (family) | Y | N |
| SPECTRA | Y | Y | Y (frozen policy) | Y | Y (user budget: params or FLOPs) |

#### 2.2.4 Mechanism-level generality: DepGraph and SPA

DepGraph [12] (CVPR 2023) models layer I/O dependencies and prunes coupled groups with a sparse-training + norm criterion, on CNNs, RNNs, GNNs, and Transformers. SPECTRA should thank it, then differentiate: skip/concat coupling and structured cuts are table stakes; they are not the thesis contribution. DepGraph *solves a given model*. It does not train an offline agent, does not take a user budget as the input of a transferred policy, and does not carry an allocation to an unseen net without re-solving.

SPA [81] (2024) attacks three practical barriers: coupled channels that differ by architecture, pruning at different training stages, and tools locked to one framework. It uses ONNX graphs and group-level importance (OBSPA is a post-training, calibration-light variant) and positions itself beyond DepGraph / OTO-v2. SPA’s “any architecture / any framework / any time” is about the *tool*. SPECTRA’s “any architecture / any dataset” is about a *transferred policy*. Using the same slogan would invite a fair referee objection. “Any time” in SPA means training stage, not “any previously unseen dataset without re-solving.”

HESSO [83] (OTO-lineage sparse optimizer) and Auto-Train-Once [82] (controller-guided prune-from-scratch, CVPR 2024) make train+prune less manual for a *given* DNN. FreePrune [84] automates across pruning granularities with training-free scores. All three are pipeline / criterion generality, not a cross-dataset DRL policy.

#### 2.2.5 2025–2026: still per-model

CNN structured pruning did not cool off. Metaheuristic channel search [85], comparative encodings for search-based CNN prune [86], learnable per-filter masks [87], differentiable attention-guided channel pruning [88], SVD-driven filter importance [89], flow-guided multi-architecture scores [90], and DepGraph-style coupling plus spectral entropy [91] all improve a criterion, a search, or a deployment pipeline for a given network. That raises the bar for “yet another pruning score.” It does not close the NEON→CNN offline-agent lane. Adjacent 2025–26 work — DualPrune [92], one-cycle structured prune [93], GCN search for channel rates [94], ℓ2,p structured sparsity [95], structured lasso + IB [96], knowledge-distillation plus structured lightweight CNNs, agentic prune/quant pipelines — sits in the same per-model bucket. RL papers with “pruning” in the title in this window are mostly domain RL, not a generic CNN offline agent; the closest DRL neighbor remains AMC [79]. Quote those papers as reference points on a Pareto plot. Do not claim SPECTRA beats them on their home architecture × dataset.

Other proposal-era CNN-pruning mechanisms still worth a clause: evolution strategy [30, 31], clustering / swarm [32], auto graph encoder-decoder [33], Transformer-related pruning under pretrain–finetune [36], FPAC [37], DETR pruning [38], global channel attention [39]. ConvNeXt [40], LLM pruning [41], ViT [42–44], and VAN [45] are out of SPECTRA’s structured-CNN loop.

MetaPruning [80] learns a PruningNet for channel configs of a target family — automatic, not an offline cross-dataset agent.

#### 2.2.6 Allocation, its cheap proxies, and transferring a pruning policy

**Allocation priors, and how much allocation matters.** Once channels are ranked inside a layer, the remaining decision is how much of each layer to keep. Closed-form priors include:
- per-layer sensitivity [18]: measure each layer's accuracy drop when it alone is pruned, then cut less where it hurts;
- ERK [107], which gives larger layers higher sparsity, and so keeps more FLOPs at equal parameters;
- LAMP's layer-adaptive magnitude scores [24].

Cost-aware allocators trade importance against what a cut saves: NetAdapt [116], constraint-aware importance [115], and HALP's latency knapsack [114]. SPECTRA's cost-aware sensitivity rule (§4.1) is the same idea in closed form.

How much allocation matters after a full recovery is contested:
- At about 50 % FLOPs on CIFAR ResNets, sensible allocations end within 0.15–0.31 points of each other under one recovery [110, 111], about one seed's spread [11].
- The best of many random allocations, chosen by a cheap selector, is a strong baseline [108].
- The retraining recipe moves accuracy more than the allocation does [108, 109].
- Network Slimming's learned allocations [106] beat uniform on VGG-16 but not on PreResNet-164 [11].
- Gains of a point or more appear against uniform cuts, at tight budgets, on CIFAR-100 or on compact networks [12, 79, 117].

SPECTRA's evaluation is built for this regime: one recovery for every arm, paired seeds, and a prior fixed in advance (§4).

**Cheap proxies for post-recovery accuracy.** An allocation search needs a cheap estimate of the accuracy after recovery. AMC [79] rewards the accuracy of the cut network directly. EagleEye [117] showed that this "vanilla" evaluation is misled by stale BatchNorm statistics [118]. Re-estimating them with the weights frozen (adaptive BN) raised the rank correlation with post-fine-tune accuracy from at most 0.40 to 0.64–0.85 on MobileNetV1 / ImageNet, and a thousand images are enough to re-estimate the statistics [119]. Short fine-tunes are the other common proxy [79, 120], although at low FLOPs a few steps may not be representative [120].

Proxy fidelity drops among near-best candidates [111, 121] and when measured at the cut [122], and reduced-training proxies differ in how well they keep the rank order [123]. SPECTRA therefore measures each proxy's rank fidelity on its own plans before training on it (§5.7). Its `bn8` reward is EagleEye's adaptive-BN evaluation used as an RL reward, as EagleEye itself did inside AMC [117].

**Transferring a pruning policy.** Most learned pruners search or train per target:
- AMC runs hundreds of episodes per model [79];
- MetaPruning trains a PruningNet per network [80];
- graph-based RL agents retrain their heads on the target [33, 125].

Others warm-start from earlier agents [126, 127], or reuse a policy only on the same architecture [128–130]. Optimised widths transfer across width, depth and resolution within a family [131].

The closest frozen transfer for CNNs applies a graph metanetwork trained on ResNet-56 to ResNet-110 and back, 0.34–0.59 points below a same-architecture metanetwork, within one family [124]. Outside CNNs:
- one allocation function searched on one LLM is reused across LLM families [132];
- an RL quantization policy transfers zero-shot between LLMs [133];
- NEON transferred a pruning agent across dense networks [1].

Budget conditioning in pruning is mostly on the level of one budget type: an AMC-style agent conditioned on the rate [137], or routers conditioned on a latency target [141]. Other methods re-optimise or re-search per budget type or constraint [112, 113, 139, 140, 142–144], and a NAS hypernetwork conditions on device and objective preferences [138]. Recent work names per-model policies as a fundamental limitation [135], and LLM-driven pipelines still loop per target [136].

SPECTRA's mechanisms have precedents:
- a single-step policy that maps per-layer scores to an exact budget, for LLM pruning [104];
- bisection to an exact budget [113];
- a plan-level allocation policy trained with a group baseline [105];
- a Transformer controller over per-layer features [134].

What SPECTRA adds is the protocol and the claim. One policy is trained once on a catalog of CNNs from four families and applied frozen to held-out networks and datasets, with no per-target search or update. It is compared under one recovery with the best same-loop rule, and the cross-family cells are reported fold by fold, including the one that fails.

### 2.3 Neural architecture search

NAS explores automatic architecture design: search space, optimization method, candidate evaluation [1]. RL was applied to NAS in [54] and followed by [55–57]; evolution [58, 59], SMBO [60], and gradient-based search [61, 62] followed. Some works use CNN pruning to help NAS [63–65]; others use NAS to help CNN pruning [66–68]; a few prune or search at initialization without data [69, 70]. SPECTRA is **not** NAS. The agent does not invent a new topology; it allocates keep fractions over an existing CNN's channel groups. Its plans resemble the width search of channel-number search [66–68, 112], but the policy is trained once and frozen rather than searched per network. Keep this subsection short, as NEON did.

---

## 3. Approach — method of record: the plan-as-action agent (14 Oct 2026)

NEON's §3 order: overview, state, action, reward, architecture and training, complexity. This section describes the code that produces every agent row of §5 (`src/plan_agent.py`, `src/plan_trainer.py`, `src/alloc_walk.py`). The walk-based agent that preceded it is described in Appendix B.

### 3.1 Problem and overview

A pretrained CNN has G prunable **channel groups**: sets of layers that must keep the same number of channels (§3.2). Given a budget type b (parameters or FLOPs) and a kept fraction κ, an allocation chooses a width w_g ≤ w⁰_g for every group, so that the network's cost under b is the fraction κ of the original. Inside each group, the channels removed are always the lowest-L1 ones [18]; the allocation decides only *how many*. The cut network is then recovered with one fixed fine-tune (§4.2), and the objective is its accuracy after that recovery.

SPECTRA learns one policy π(w | network, b, κ) offline on a catalog of networks, freezes it, and applies it to held-out networks with no per-target training, search or adaptation. At test time, a held-out network is pruned in five steps:
1. The network is read as a sequence of per-layer tokens (§3.3).
2. A Transformer scores every channel group.
3. A decoder turns the scores into integer widths at the budget (§3.4).
4. Floors and grid rounding make the plan realizable.
5. The network is cut once to that plan and recovered.

Every rule SPECTRA is compared with (§4.1) produces its plan through the same steps 3–5, so the agent and the rules differ only in where the plan comes from.

**Why one action per network.** NEON's agent walks a network layer by layer and is rewarded step by step [1]. Carried over to CNNs, that design collapsed: every walk agent through v10 converged on "keep 90 % everywhere", because after a per-step fine-tune the step reward barely separates the actions (Appendix B, §331). Here the whole allocation is a single action, scored once on the whole plan. The problem becomes a contextual bandit [102, 103]: the context is (network, budget), the action is a plan, and the reward is one number. No fine-tune runs inside the training loop.

### 3.2 Channel groups and the cut

A `torch.fx` trace recovers the layers that must share a channel dimension: producers joined by a residual addition (one group per residual stream), the segments of a concatenation, depthwise ties, and the normalisations and consumers that read them. This bookkeeping is the same class as DepGraph's [12] and SPA's [81], and SPECTRA claims no novelty for it. Cutting a group rebuilds every producer, every normalisation over that dimension and each consumer's input slice in one edit, and a dummy forward verifies the result. A group whose structure is not recognised is blocked; if a cut cannot be expressed, it falls back to a mask, earns no size credit, and is logged.

Inside a group, each producer's per-filter L1 norms are divided by their own maximum and summed across producers; the top-w_g channels survive [18]. The planned groups are the prunable groups at least two channels wide. The classifier's output dimension is never a target.

### 3.3 State

The network is read as one token per leaf module: every convolution, linear layer, normalisation, activation and pooling layer. Each token has 63 columns:

| Columns | Width | Content |
|---|---|---|
| Topology, activation and weight statistics | 38 | layer kind, kernel, width, stride, in/out features; 12 moments of the layer's activations on a fixed probe batch; 12 moments of its weights; 7 descriptors of its per-filter L1 distribution. Z-scored with a standardizer fitted on the catalog |
| Position and structure | 4 | relative depth, stem flag, coupled flag, normalised width |
| Group cost | 4 | the owning group's share of the network's parameters and of its multiply-accumulates; its number of producers relative to the largest group; cuts so far (0 at the origin) |
| **Group sensitivity** | 2 | log(s_g / median s) / 3, clipped to [−1, 1], and its percentile rank. Here s_g is the rise in calibration loss when that group alone is cut to half width (§4.1) |
| **Budget** | 2 | the target κ, written on every token |
| Budget type | 1 | 0 for parameters, 1 for FLOPs, on every token |
| Legacy slots | 12 | kept fraction and accuracy slack (fixed at the origin's values) and per-action costs (zeroed); inherited from the walk agent's state |

The agent therefore sees what the rules see: each group's measured sensitivity and its cost share. It also sees the network's structure and statistics, which no rule uses. The layout is the walk agent v10's state contract, kept unchanged. The reason was a test: with the sensitivity columns, the default encoder learns to imitate the sensitivity rule's plan on held-out networks (Spearman ρ 0.872), whereas without them every variant does only partially (§344).

### 3.4 Action: a whole plan

**Encoder and scores.** The tokens pass through a linear projection to 256 dimensions with a layer-type embedding, sinusoidal positions and a learned marker. Then come three pre-norm Transformer layers [35] (256 wide, 8 heads, feed-forward 1024, GELU, no dropout). An additive attention bias, one learned scalar, applies to every pair of tokens that share a channel group. A linear head, zero-initialised, gives one output per token. A group's score μ_g is the mean output over the layers that produce its channels, centred across groups. The encoder and head have about 2.4 M parameters. At initialisation every score is 0, so the untrained agent proposes the uniform plan.

**Decoder.** Each group keeps keep_g(β) = clip(σ(μ_g + β), k_min, 1), where σ is the logistic function and k_min = 0.1. The scalar β is found by bisection on an analytic cost model: an exact parameter count for the parameter budget, or a FLOPs model that counts convolution and linear multiply-accumulates exactly and is checked against real cuts. Bisection to a budget follows ChipNet [113], and a single-step policy that maps per-layer scores to an exact budget has been used for LLM pruning [104]. Keeps are rounded to integer widths. A polish step then moves single channels on the worst-rounded groups while that brings the cost closer to κ. The plan's cost therefore matches the budget to within the rounding of single channels. It is not guaranteed to sit at or below κ in training; at evaluation, every arm lands within 0.01 below its target (§4.2).

**Realizable plans at evaluation.** Every evaluation plan, the agent's and the rules', goes through four guards:
1. It is decoded 0.02 below the target.
2. It is held to a width floor of two channels, the narrowest width the cutting walk can produce.
3. It is rounded to widths the walk can reach exactly, from a five-rate menu {1.0, 0.9, 0.8, 0.7, 0.6}, ties to the wider. Low-priority groups step down while the plan is above target, and high-priority groups step up while it is more than 0.01 below. The priority is the agent's score, or the rule's weight.
4. A one-shot walk realizes it, with a fixed-target landing that narrows the last cut to stop at the target.

A quoted arm logs no fallback cut and no masked edit (§4.2).

**Seed ensemble (v22).** Several frozen checkpoints of one line can be decoded as one agent: their scores are averaged per group and the mean is decoded once. It is still one frozen policy with no per-target search (§386).

**The FLOPs agent: a residual on the prior.** Under a FLOPs budget, a direct plan agent learned but stayed below the FLOPs sensitivity rule (§357). The FLOPs line therefore parameterises the plan relative to that rule. Each group keeps keep_g = clip(c · w_g · e^{z′_g}, k_min, 1), where w_g is the sensitivity rule's weight (α = 0.5, §4.1) and c is bisected to the budget. The residual is z′_g = 0.3 · tanh(z_g / 0.3), a trust region that keeps every weight within e^{±0.3} of the rule's (§365). With z = 0 the agent reproduces the rule's plan exactly.

### 3.5 Reward

For a plan sampled in training, the reward is r = 100 · (acc_val(cut network after the proxy) − acc_val(origin)), in points. acc_val is top-1 accuracy on the **validation half**, the 5,000 images of the 10,000-image test split that are never used for TEST (§4.2), measured on the training network. There is no size term: the decoder fixes the budget, so a plan cannot gain by spending less. The *proxy* stands in for the 100-epoch recovery, which is far too slow to run inside training:

| Proxy | What is done to the cut network before the read | Cost per training instance (K = 8) |
|---|---|---|
| `cut` | nothing: the masked cut is read as is | 0.86 s (§346) |
| `bn8` | BatchNorm running statistics are re-estimated on 8 training batches (2,048 images), with no gradient | 1.61 s |
| `fs32` | 32 SGD steps (lr 0.01, momentum 0.9, weight decay 5·10⁻⁴) on fixed training batches, re-masking after every step | 6.9 s |
| `ft1` | one epoch of the learning handler's own fine-tune, on the real cut: the fidelity reference (§375, §379). Built as a training reward in v23 (default off), not yet trained on | about 4× `fs32` at evaluation; about 1–1.4 min per training instance |

All K plans of an instance share the same proxy batches, so their rewards differ only through the plans (common random numbers). The masked cut used in training is checked against the real structural cut at the start of every train (kept parameters within 10⁻⁶; validation accuracy within 0.05 points).

Which proxy to train on is itself measured. Each candidate proxy ranks a fixed set of real plans; that ranking is compared with the one given by the 100-epoch final fine-tune (§4.4, §5.7). The raw cut was chosen first, as the cheapest proxy valid on both voting families (§340). It later proved weak at equal parameters (§375), and the batch-norm reward removed the agent's deep-budget failure (§378). T6 trains on `bn8` and on `fs32` in parallel (§7).

### 3.6 Training

Training is REINFORCE [53] on a contextual bandit:
1. Each training instance draws a catalog network uniformly and a budget κ ~ U[κ_lo, κ_hi].
2. The policy gives the mean scores μ. Eight plans are sampled as z_k = μ + σ ε_k, with ε_k ~ N(0, I).
3. Each plan is decoded, cut once on a masked copy, and rewarded r_k.
4. The advantage of each plan is its reward minus the mean of the eight, A_k = r_k − r̄: a shared baseline in the manner of POMO [98], which acts as a control variate [101]. It equals (K − 1)/K times the leave-one-out baseline [97, 99] and is close in spirit to GRPO's group baseline [100]. The shared baseline removes the network's and the budget's difficulty from the gradient, so no critic is needed.
5. The loss is −(1/K) Σ_k A_k log π(z_k), with a Gaussian log-density around μ.

Updates use Adam (lr 3·10⁻⁴) over 4 instances per step, with gradient-norm clipping at 1.0. The exploration scale σ decays from 0.5 to 0.2 over the first 60 % of training. A train is 12,000 instances (3,000 updates), with no entropy bonus, no imitation term and no critic. The frozen policy is the **last iterate**: nothing is selected on validation, and nothing in training reads the TEST half.

**Catalog.** Ten pretrained networks from four families and two datasets:
- CIFAR-10: thin ResNet-20 at widths 8 and 10, thin ResNet-56 at width 6, ResNet-32, VGG-11-BN, VGG-13-BN, MobileNetV2 ×0.5 and ×1, DenseNet-40;
- SVHN: VGG-11-BN.

No held-out network is in the catalog (§4.3).

**Agent lines.** Each registered train changes one thing from the catalog-trained parameter agent (T1):

| Agent (code) | Budget | Change from T1 | Read in |
|---|---|---|---|
| Single-network agent (T0) | params | trained on thin ResNet-56 ×4 alone, 3,000 instances | §341 |
| **Catalog agent (T1)** | params | the reference line: κ ∈ [0.35, 0.85], raw-cut reward | §346, §370, §373, §374, §382 |
| Training floor 0.25 (T5-floor) | params | k_min 0.25 instead of 0.1 | §376 |
| Deeper range (T5-deep) | params | κ ∈ [0.25, 0.85] | §377, §380 |
| Batch-norm reward (T5-bn) | params | `bn8` reward | §378, §381, §382 |
| Deeper range + batch-norm reward (T6-bn) | params | κ ∈ [0.25, 0.85] and `bn8` | §383, §384, §385 |
| Deeper range + 32-step reward (T6-fs) | params | κ ∈ [0.25, 0.85] and `fs32` (20 h per train) | §387, §390 |
| Deeper range + batch-norm reward, 18-network catalog with CIFAR-100 (T7-C) | params | T6-bn's line on the catalog of §356 | §389 |
| Deeper range + 32-step reward, 18-network catalog with CIFAR-100 (T8-fs) | params | T6-fs's line on the catalog of §356 (21 h per train) | §400 |
| With CIFAR-100 networks (T1-C) | params | catalog plus 8 recoverable CIFAR-100 networks | §356 |
| No ResNet / no VGG (LOFO-R / LOFO-V) | params | catalog without that family (6 / 7 networks) | §354, §358, §363 |
| Direct FLOPs agent (T0-F, T0-F2) | FLOPs | trained on thin ResNet-56 ×4 under a FLOPs budget; T0-F2 adds the width floor | §350, §357 |
| Mixed-budget agent (TM) | both | each instance draws params or FLOPs, flagged in the state | §360 |
| FLOPs residual agents (T2-F → T4-F) | FLOPs | residual on the FLOPs sensitivity rule, width floor 2. T3-F adds per-instance advantage normalisation; T4-F adds the trust region | §361, §364, §365, §366, §371, §382 |

**Differences between training and evaluation**, stated so a reader can weigh them:
- The parameter trains use a floor of one channel, while every evaluation uses two. The FLOPs and mixed trains use two throughout.
- Training rewards analytic plans decoded at κ. Evaluation decodes at κ − 0.02, rounds to the walk's grid, and realizes the plan by the walk, while the state still reads κ.
- The trainer samples and scores the raw residual z, without the trust region's Jacobian.

### 3.7 Complexity

Training cost is the number of instances times K masked cuts and proxy reads, on one RTX 4090:
- the raw-cut reward: 0.86 s per instance, so the catalog agent's 12,000 instances took 175 min (§346);
- the batch-norm reward: about 5 h;
- the 32-step reward: about 23 h (projected from its smoke).

The walk agent, by comparison, spent 11–25 min per training *episode* (two passes, a 12-epoch fine-tune after every cut) and trained for days (Appendix B).

Evaluation is one cut and one 100-epoch recovery per network, budget and seed. With CUDA graphs and GPU-side augmentation, the recovery takes about 3 min on thin ResNet-56 ×4, against 15–17 min for the earlier recipe (§329). From job start to the end of recovery, one plan on DepGraph's ResNet-56 takes 7.2–8.1 min with 100 recovery epochs, against 85.1 min for DepGraph's own pipeline on the same GPU model (§334). Applying the frozen agent adds one forward pass and a bisection, which is negligible. The calibration the rules need, four batches per group, is shared with the agent's sensitivity columns.

---

## 4. Evaluation

NEON's §4 covers algorithms, setup, results and discussion. SPECTRA gives the setup here and the numbers in §5.

### 4.1 Compared methods

Every rule below shares the agent's loop:
- the same channel groups and L1 ranking;
- the same width floor, undershoot and grid rounding;
- the same one-shot walk with fixed-target landing;
- the same recovery fine-tune and seed.

The rules differ from the agent only in where the plan comes from.

**Sensitivity.** For each group g, s_g is the rise in cross-entropy over 4 training batches (batch 256) when that group alone is cut to half width by L1, with no fine-tune. This is the per-layer sensitivity analysis of Li et al. [18], measured per channel group. Small values are floored at 5 % of the median. The same s_g feeds the agent's sensitivity columns (§3.3).

| Rule | Plan | Role |
|---|---|---|
| uniform (uniform_F) | the same keep fraction for every group, bisected to the parameter (FLOPs) budget | the allocation-free baseline |
| sens (sens_F) | keep_g ∝ (s_g / median s)^α, α = 0.5, bisected on real cuts to the budget | sensitivity allocation [18] |
| **sens_cost** | keep_g ∝ (v_g / median v)^α, where v_g = s_g / c_g and c_g is the parameters (MACs) saved by the half cut: sensitivity per unit of cost saved | **the prior at equal parameters** (§355) |
| **floored sens_F** | sens under a FLOPs budget, with the width floor | **the prior at equal FLOPs** (§357) |
| inner (inner_F) | uniform, with every residual stream held whole: cut only inside blocks | the "keep the skip streams" heuristic |
| mild (mild_F) | not a plan but a walked heuristic: keep 90 % of each legal group, else the strongest legal cut; fine-tuned after each cut, then given the same final recovery | the heuristic every walk agent collapsed onto |

**One prior per budget, fixed in advance.** BEATS-PRIOR is read against one rule per budget type. The rule is chosen from pooled evidence before the agent's pool reads, never per network and never on TEST. At equal parameters it is the cost-aware sensitivity rule, which beat plain sensitivity on all three networks measured (§355). At equal FLOPs, where the cost weighting hurt (COST-HURTS, §355), it is the floored FLOPs sensitivity rule (§357). Every other rule is reported beside the prior.

**The literature.** Published methods are quoted beside SPECTRA's rows with their own fine-tune recipes, under a different-recipe caption; they are not same-loop comparators. FLOPs is the headline axis for these rows, with parameters beside them (Ido, 10 Oct). The anchor is DepGraph's ResNet-56 on CIFAR-10 [12], because its checkpoint is in the pool (§5.9). OCSPruner's ResNet-56 row [93] is the second anchor, quoted from its pretrained-start table:

> **OCSPruner ResNet-56 row (locked 27 Sep 2026).** Beside SPECTRA, quote OCSPruner [93] Table 5, the run that starts from a pretrained network, because SPECTRA does too: 38.82 % of FLOPs remaining, 42.26 % of parameters, 94.01 → 93.50, a drop of 0.51. Footnote Table 2 (from scratch): 38.88 % of FLOPs remaining, 41.42 % of parameters, 93.97 → 93.65, a drop of 0.32. The table's FLOPs column is percent remaining (their VGG text: "26.01 % of the original network FLOPs remaining"); do not read the §4.3 phrase "38.88 % reduced FLOPs" as the column definition. Do not put the lighter from-scratch row (46.93 % FLOPs remaining, drop 0.17) in the same cell. OCS's baseline is their own training run, not DepGraph's 93.53.

### 4.2 Protocol

- **Halves.** The 10,000-image test split of each dataset is divided once by a fixed permutation (seed 0) into a **validation half** (5,000 images) and a **TEST half** (5,000). Every seed and every job uses the same two halves. Training reads only the validation half, and only of the catalog networks (§3.5). Nothing reads the TEST half before the final read: no selection, no stopping and no reward.
- **Cut.** For each held-out network, budget and seed, the plan is realized in one shot (§3.4), with no fine-tune between cuts. The quoted point is the network at the target size, fixed in advance; it is never picked on TEST.
- **Recovery (G2).** The cut network and the unpruned origin each receive the same final fine-tune:
  - SGD with momentum 0.9 and weight decay 5·10⁻⁴;
  - learning rate 0.1, cosine-annealed to 0 over 100 epochs;
  - batch 256 with random crop and flip on the GPU, fp32, CUDA graphs;
  - no early stopping, and the **last epoch is kept**.

  Le & Hua [109] identify retraining at a large learning rate as the variable that decides post-pruning accuracy. G2 replaced §330's per-family recipes once it matched the plain recipe on VGG-19 CIFAR-100, the family where an earlier four-pair check had kept the plain recipe (SPEED-EQUIVALENT over 16 pairs, §343, §348). An earlier default restored the epoch with the lowest training loss. That was epoch 1 on most pruned networks, so those rows measured "cut + 1 epoch"; they are never mixed with keep-last rows (§235, §330).
- **Reads.** For a pruned arm:
  - **TEST** = 100 · (TEST accuracy after recovery − the origin's TEST accuracy before recovery), in points, on the 5k TEST half;
  - **val** = the same on the validation half;
  - **10k** = the same on all 10,000 images;
  - **honest** = TEST accuracy after recovery − the origin's TEST accuracy after the *same* recovery: the raw change minus the origin's own change. A stronger fine-tune also lifts the unpruned network, and honest removes that lift.

  For a difference between two pruned arms (agent − rule), honest is the difference of the two arms' honest values. Each job runs its own origin control, and controls of one seed differ between jobs by up to 0.7 points (§352), so honest can differ from raw. Each result is quoted as TEST, with val, 10k and honest beside it.
- **Seeds and pooling.** Three seeds, 42–44. An agent's evaluation seed is its training seed, and each rule runs at the same seed, so the contrasts are paired by seed. A network's value is the mean over seeds; a pool value is the mean over the nine networks. Uncertainty is a bootstrap over networks × seeds (4,000 resamples). The pooled interval is wide when one network dominates, and §5 always names that network.
- **Realizable arms only.** A quoted arm keeps the width floor, logs no fallback cut and no masked edit, and lands within 0.01 of its target. Rows run before this rule (10 Oct) keep their calls with a scope note (§346, §352, §357).
- **Size counting.** Parameters kept = all parameters of the rebuilt network, counted by shape, over the original's count, so masked zeros earn nothing. FLOPs kept = multiply-accumulates per 32 × 32 sample (convolutions, linear layers, normalisations and activations; pooling and additions are not counted), over the original's. When the two arms of a pair differ by more than 10 % in the other axis, the row says so.

### 4.3 Networks

- **Training catalog** (§3.6): ten networks from four families (ResNet, VGG, MobileNetV2, DenseNet) on CIFAR-10 and SVHN.
- **The nine-network held-out pool.** It covers the same four families on CIFAR-10 and CIFAR-100, and two of its networks are DepGraph's own checkpoints [12]. No pool checkpoint is in the catalog. One shares its architecture with a catalog network but was trained on CIFAR-100.

| Key | Network | Dataset | Note |
|---|---|---|---|
| thin | ResNet-56, width ×4 | CIFAR-10 | the skinny ResNet that defeated every walk agent |
| r56c | ResNet-56 (chenyaofo) | CIFAR-10 | |
| r56w8 | ResNet-56, width ×8 | CIFAR-10 | |
| mbv214 | MobileNetV2 [147] ×1.4 | CIFAR-10 | |
| dn100 | DenseNet-100 | CIFAR-10 | |
| v16 | VGG-16-BN (chenyaofo) | CIFAR-10 | a depth unseen in the catalog |
| c100r20w8 | ResNet-20, width ×8 | CIFAR-100 | a catalog architecture on an unseen dataset |
| v19 | VGG-19 (DepGraph) | CIFAR-100 | DepGraph's checkpoint |
| dgr56 | ResNet-56 (DepGraph) | CIFAR-10 | DepGraph's checkpoint; the literature anchor |

- **The five lever networks.** These are the networks where the FLOPs rule beats uniform by at least +0.5 points at FLOPs 0.6 (§366): thin ResNet-56 ×4, DepGraph's ResNet-56, VGG-19 CIFAR-100, ResNet-56 ×8 and ResNet-20 ×8 CIFAR-100. On the other four the FLOPs rule lands within 0.25 of uniform at that point, so allocation barely matters there. The headroom screens and two FLOPs calls read only the five (§367–§369).
- **Leave-one-family-out folds.**
  - No ResNet (LOFO-R): six catalog networks, namely VGG-11 and VGG-13 on CIFAR-10, VGG-11 on SVHN, MobileNetV2 ×0.5 and ×1, and DenseNet-40.
  - No VGG (LOFO-V): seven networks, namely four CIFAR ResNets, MobileNetV2 ×0.5 and ×1, and DenseNet-40.
  - Each is read on held-out networks of the family it lacks (§354, §358, §363).
- **Families no catalog holds.** ShuffleNetV2 [148] ×1 and ×1.5 and RepVGG [149] A0 and A1 on CIFAR-10 (§353); ShuffleNetV2 ×1 and RepVGG-A0 on Fashion-MNIST (§359).

### 4.4 Registered calls

Every cell is registered in `docs/SITTING_GPU_QUEUE.md` before any GPU time. A registration states the question, the comparators, the call and its thresholds, and what may not be claimed. §5 reports the calls exactly as they fired.

| Call | Rule (TEST, mean over seeds paired by seed) | Used in |
|---|---|---|
| LEARNS | agent − mild ≥ +0.5 on thin ResNet-56 ×4. For the catalog agent, also agent − uniform ≥ −0.3 on DepGraph's ResNet-56. FLOPs agents use mild_F | §341, §346, §357 |
| BEATS-PRIOR / AT-PRIOR / BELOW-PRIOR (one network) | agent − prior ≥ +0.3 / within ±0.3 / ≤ −0.3 | §341, §346, §361, §364, §365, §369 |
| BEATS-PRIOR-P-POOL (-F-POOL) | pooled agent − prior ≥ +0.3 **and** positive on at least 6 of 9 networks | §366, §370, §371, §373, §374, §376–§378, §380 |
| AT-PRIOR-POOL | pooled difference strictly between −0.3 and +0.3 (or ≥ +0.3 without the 6-of-9 majority) | same |
| BELOW-PRIOR-POOL | pooled difference ≤ −0.3 | same |
| LEARNS-MORE / SAME / LEARNS-LESS | pooled retrain − catalog agent ≥ +0.3 / within ±0.3 / ≤ −0.3 | §376, §377, §378 |
| HEADROOM (-P) | mean over the five lever networks of (best confirmed alternative − prior) ≥ +0.3, positive on at least 3 of 5 | §367, §368, §372 |
| LEVER-LIMITED (no call) | sensitivity − uniform < +0.5 in the cell: allocation barely matters, so an allocation policy cannot be tested there | §353, §359, §363 |
| HOLDS / FAILS-CLEAN (realizable arms) | HOLDS: agent − uniform ≥ −0.3 on DepGraph's ResNet-56. The no-VGG fold holds only if it reaches ≥ +0.5 on VGG-19 CIFAR-100 and ≥ −0.3 on VGG-16 | §363 |
| COST-HELPS / COST-HURTS | cost-aware − plain sensitivity ≥ +0.3 / ≤ −0.3 | §355 |
| POOL-HELPS / NEUTRAL / HURTS | catalog-with-CIFAR-100 agent − catalog agent, ±0.3 | §356 |
| SPEED-EQUIVALENT | \|mean change\| ≤ 0.30 at 10k, at ≥ 1.5× the speed | §336, §348 |

**Proxy fidelity.** A proxy is scored by how it ranks a fixed set of real plans against the ranking from the 100-epoch recovery: Spearman's ρ on the validation half, plus the share of plan pairs whose order it gets right. The early studies (§340, §349) call a proxy VALID if its mean ρ ≥ 0.60 and its median regret (the accuracy lost by taking its top plan) is ≤ 0.5 points. A family is CEILING-BOUND when the recovery's own seed-to-seed ρ is below 0.60: then no proxy can do better, and the test is uninformative. At 35 % params (§375, §379) a proxy is P35-VALID if the mean over networks of ρ is ≥ +0.6 and it orders at least two thirds of plan pairs correctly; WEAK between +0.3 and +0.6; INVALID below +0.3.

---

## 5. Results — PRELIM (every number cites its ledger section)

Proposal §4 (NEON on flattened images) is motivation, not a SPECTRA table. The walk-era results are in Appendix B.

**How to read the tables.** A row gives a difference in points of accuracy after recovery, agent − rule unless stated. It is the mean over seeds 42–44, paired by seed; the catalog agent's first reads (§346) use seeds 42–46. **TEST** is the 5k test half, and val, 10k and honest stand beside it (§4.2). For differences between two pruned arms, honest is the difference of the arms' honest values. Each job has its own origin control, so honest can differ from raw.

A pooled value is the equal-weight mean over the nine networks. Its interval is the ledger's bootstrap over networks × seeds. The ledger states the pooled TEST value. The pooled val, 10k and honest values are means of the ledger's per-network rows **[computed]**, and every pooled TEST value and positive count reproduces the ledger's. Every row is PRELIM.

### 5.1 The agent learns

| Agent, network, budget | Contrast | TEST | val | 10k | honest | FLOPs ratio | Call |
|---|---|---|---|---|---|---|---|
| Single-network agent (T0), thin ResNet-56 ×4 (its training network), params 0.6 | − mild | **+2.35** | +2.17 | +2.26 | not given | 1.58–1.72 | LEARNS (§341) |
| | − sens | +0.11 | +0.01 | +0.06 | not given | 1.24–1.37 | not BEATS-PRIOR |
| Catalog agent (T1), thin ResNet-56 ×4 (held out), params 0.6 | − mild | **+2.20** | +2.20 | +2.20 | +2.58 | 1.47–1.78 | LEARNS (§346) |
| | − sens | +0.12 | +0.02 | +0.07 | +0.29 | 1.18–1.42 | BEATS-PRIOR not fired |
| Catalog agent, DepGraph's ResNet-56 (held out), params 0.47 | − uniform | **+0.96** | +0.54 | +0.75 | +0.98 | 1.25–1.34 | LEARNS (§346) |
| | − sens | +0.28 | +0.11 | +0.20 | +0.39 | 1.37–1.58 | |
| Same, re-read with realizable arms | − uniform | **+1.10** | +0.79 | +0.94 | +1.18 | 1.38–1.43 | HOLDS (§363) |

The first plan agent, trained on one network, left the mild heuristic behind by more than two points (§341), where every walk agent had converged onto it (Appendix B). Trained on the ten-network catalog and applied frozen, the agent kept that margin on a held-out thin ResNet and beat the uniform rule on DepGraph's checkpoint (§346). The finding survived the re-read with floored, grid-rounded comparators (§363). In both cases the agent reached the level of the sensitivity rule, not above it.

The agent keeps more FLOPs than mild or uniform at equal parameters (ratios above), so these rows are parameter-budget results, not compute claims. The held-out networks of §346 are CIFAR ResNets, a family in the catalog: this is transfer to held-out networks of a seen family.

At 80 % params, reported without a call, the single-network agent is level with sensitivity (−0.13; val −0.17, 10k −0.15; §347).

### 5.2 Equal parameters: the frozen agent against the cost-aware rule

Once §355 showed that, at equal parameters, cost-aware sensitivity beats plain sensitivity on all three networks it was measured on, the prior became the cost-aware rule (§4.1). The frozen agent was then read against it on the nine-network pool at four depths.

**Table 5.2a — pooled (agent − cost-aware rule, params budget).** "Other eight" is the pool without ResNet-20 ×8 CIFAR-100, a diagnostic read after the calls.

| Kept | Agent | TEST [95 % CI] | val | 10k | honest | Positive | Call | Network that decides it | Other eight (TEST) |
|---|---|---|---|---|---|---|---|---|---|
| 60 % | catalog agent (T1) | **−0.01** [−0.21, +0.21] | +0.11 | +0.05 | +0.07 | 4 / 9 | AT-PRIOR-POOL (§370) | — | +0.01 |
| 45 % | catalog agent (T1) | **+0.18** [−0.07, +0.46] | +0.13 | +0.15 | +0.24 | 6 / 9 | AT-PRIOR-POOL (§373) | — | +0.10 |
| 35 % | catalog agent (T1) | **−0.42** [−2.06, +0.33] | −0.45 | −0.44 | −0.45 | 5 / 9 | BELOW-PRIOR-POOL (§374) | ResNet-20 ×8 CIFAR-100, −5.23 | +0.18 |
| 25 % | deeper-trained agent (T5-deep) | **−1.60** [−5.88, +0.49] | −1.56 | −1.58 | −1.49 | 5 / 9 | BELOW-PRIOR-POOL (§380) | ResNet-20 ×8 CIFAR-100, −16.84 | +0.30 |
| 60 % | deep-range batch-norm agent (T6-bn) | **+0.02** [−0.12, +0.16] | +0.12 | +0.08 | +0.03 | 6 / 9 | AT-PRIOR-POOL (§384) | — | +0.00 |
| 45 % | deep-range batch-norm agent (T6-bn) | **+0.30** [−0.05, +0.79] (exactly +0.2956) | +0.21 | +0.26 | +0.35 | 6 / 9 | AT-PRIOR-POOL (§384), 0.004 under the bar | ResNet-20 ×8 CIFAR-100, +1.89 | +0.10 |
| 35 % | deep-range batch-norm agent (T6-bn) | **+0.19** [−0.02, +0.43] | +0.06 | +0.13 | +0.23 | 6 / 9 | AT-PRIOR-POOL (§383) | — | +0.17 |
| 25 % | deep-range batch-norm agent (T6-bn) | **−0.17** [−1.11, +0.32] | −0.10 | −0.14 | −0.10 | 5 / 9 | AT-PRIOR-POOL (§383) | ResNet-20 ×8 CIFAR-100, −2.41 | +0.11 |
| 35 % | deep-range 32-step agent (T6-fs) | **+0.2356** [+0.01, +0.50] | +0.21 | +0.22 | +0.31 | 7 / 9 | AT-PRIOR-POOL (§387) | — | +0.20 |
| 25 % | deep-range 32-step agent (T6-fs) | **+0.1615** [−0.46, +0.60] | +0.22 | +0.19 | +0.22 | 6 / 9 | AT-PRIOR-POOL (§387) | ResNet-20 ×8 CIFAR-100, −0.93 | +0.30 |
| 60 % | deep-range 32-step agent (T6-fs) | **+0.0504** [−0.17, +0.28] | +0.24 | +0.14 | −0.08 | 4 / 9 | AT-PRIOR-POOL (§390) | — | +0.00 |
| 45 % | deep-range 32-step agent (T6-fs) | **+0.4193** [+0.02, +1.00] | +0.34 | +0.38 | +0.46 | 6 / 9 | **BEATS-PRIOR-P-POOL** (§390) | ResNet-20 ×8 CIFAR-100, +2.28 | +0.19 |

**Table 5.2b — per network, TEST (val / 10k / honest).**

| Network | 60 % (§370) | 45 % (§373) | 35 % (§374) | 25 % (§380) |
|---|---|---|---|---|
| ResNet-56 ×4, C10 | −0.39 (−0.08 / −0.23 / −0.34) | −0.28 (−0.03 / −0.15 / −0.06) | −0.01 (−0.30 / −0.16 / +0.17) | +0.75 (+1.06 / +0.91 / +0.69) |
| ResNet-56, C10 | +0.23 (+0.31 / +0.27 / +0.30) | +0.31 (+0.23 / +0.27 / +0.55) | +0.16 (+0.03 / +0.10 / +0.23) | +0.57 (+0.01 / +0.29 / +0.64) |
| ResNet-56 ×8, C10 | −0.05 (−0.05 / −0.05 / −0.15) | +0.57 (+0.33 / +0.45 / +0.66) | +0.42 (−0.02 / +0.20 / +0.53) | −0.05 (+0.02 / −0.02 / +0.11) |
| MobileNetV2 ×1.4, C10 | +0.04 (−0.20 / −0.08 / +0.19) | +0.31 (−0.09 / +0.11 / +0.17) | +0.67 (+0.72 / +0.69 / +0.43) | +0.93 (+0.64 / +0.78 / +1.20) |
| DenseNet-100, C10 | −0.01 (+0.23 / +0.11 / +0.10) | −0.19 (+0.15 / −0.02 / +0.09) | +0.08 (+0.15 / +0.11 / +0.15) | +0.26 (+0.23 / +0.24 / +0.06) |
| ResNet-20 ×8, C100 | −0.12 (+0.43 / +0.16 / +0.01) | +0.82 (+0.70 / +0.76 / +0.89) | **−5.23** (−4.89 / −5.06 / −4.93) | **−16.84** (−16.21 / −16.53 / −16.55) |
| VGG-19, C100 (DepGraph) | +0.56 (+0.06 / +0.31 / +0.91) | +0.02 (−0.33 / −0.15 / −0.05) | −0.01 (+0.21 / +0.10 / −0.05) | +0.21 (+0.28 / +0.24 / +0.41) |
| VGG-16-BN, C10 | −0.43 (+0.15 / −0.14 / −0.29) | −0.01 (+0.09 / +0.04 / +0.03) | +0.33 (+0.12 / +0.22 / +0.13) | −0.15 (−0.15 / −0.15 / −0.09) |
| ResNet-56, C10 (DepGraph) | +0.11 (+0.14 / +0.13 / −0.07), at 47 % | +0.07 (+0.09 / +0.08 / −0.09) | −0.19 (−0.09 / −0.14 / −0.68) | −0.09 (+0.06 / −0.01 / +0.16) |

*FLOPs caption.* At equal parameters the agent often keeps more FLOPs than the rule. The largest ratios are 1.21–1.24× at 60 % and 1.30–1.46× at 45 % (both ResNet-56 ×8), 1.26–1.48× at 35 % (ResNet-56 ×8), and up to 1.55× at 25 % (DenseNet-100).

**Reading.**
- At 60 % and 45 % the frozen agent performs at the level of the best hand-designed cost-aware rule across nine held-out networks (§370, §373). It learned that allocation from its reward alone.
- Against the cost-blind sensitivity rule, its mean advantage over the nine networks is +0.29 at 60 % (positive on six) and +0.18 at 45 % (§370, §373; reported, not calls). The registered test of going beyond the plain rule on the lever networks did not fire (+0.12, NOT-BEYOND-SENS, §370). It gains most where plain sensitivity is at or below uniform: ResNet-56, ResNet-20 ×8 CIFAR-100 and ResNet-56 ×8 (+0.52, +0.49, +0.80; §370).
- At 35 % and 25 % one network decides the call. On ResNet-20 ×8 CIFAR-100 the agent proposes an all-or-nothing plan. At 35 % (seed 43) it keeps most groups whole (median group keep 1.00) and cuts a few to about a tenth of their width (minimum 0.09), and ends 27.4 points under the origin, against 15.3 for the rule. Both plans' accuracy right after the cut is at chance (§374), so the raw-cut reward cannot tell them apart (§5.7).
- **With the batch-norm reward on the deep training range (T6-bn), one frozen agent is at the cost-aware rule at all four depths** (§383, §384; the reward and range are discussed in §5.6). At 45 % its pooled value, +0.2956, sits 0.004 under the BEATS bar, carried by ResNet-20 ×8 CIFAR-100.
- On the other eight networks the agent's mean edge over the rule rises with depth: +0.01, +0.10, +0.18, then +0.30 for the deeper-trained agent at 25 % (§370–§374, §380). This is a diagnostic read after the calls, not a call.
- Allocation matters most at the deepest budget. At 25 % the cost-aware rule beats uniform by +0.88 points over the nine networks (val +1.03, 10k +0.95, honest +0.86 [computed]), by 2.1–3.4 points on three of them, and it falls below uniform on MobileNetV2 ×1.4 and ResNet-20 ×8 CIFAR-100 (§380).

### 5.3 Equal FLOPs

**The road to the FLOPs agent.**
- **A direct agent on one network** (T0-F) failed its control: −0.42 against mild_F, and −1.49 against the FLOPs rule, at FLOPs 0.6 (val −0.59 / −1.40, 10k −0.50 / −1.45, honest −0.11 / −1.55; §350). The gap lay between what the trainer scored and what the evaluation realized (§351).
- **With the width floor in training** (T0-F2), it learned: +0.51 against mild_F (val +0.33, 10k +0.42, honest +0.83). It stayed below the floored FLOPs rule: −0.57 (val −0.25, 10k −0.41, honest −0.69; §357).
- **One agent for both budgets** (TM) failed. Its parameter and FLOPs plans ranked the groups alike (Spearman +0.96 / +0.94 on two seeds). It sat 0.52 below the catalog agent at equal params (pooled over three networks), and 0.66 below mild_F at equal FLOPs (val −0.91, 10k −0.79, honest −0.31; §360).
- **The catalog-trained FLOPs agents** therefore act as a residual on the FLOPs rule (§3.4):

| Agent (FLOPs 0.6, ResNet-56 ×4) | − floored FLOPs rule: TEST | val | 10k | honest | Call |
|---|---|---|---|---|---|
| residual (T2-F) | −1.89 | −1.21 | −1.55 | −2.01 | BELOW-PRIOR (§361) |
| + per-instance advantage normalization (T3-F) | −1.54 | −1.39 | −1.47 | −1.53 | BELOW-PRIOR; BALANCE-NEUTRAL (§364) |
| + trust region e^{±0.3} (T4-F) | **−0.47** | −0.39 | −0.43 | −0.63 | BELOW-PRIOR; BOUND-HELPS, +1.07 over T3-F (§365) |

The trust-region agent was the first catalog-trained FLOPs agent to clear the mild_F bar on a held-out network: +0.61 (val +0.19, 10k +0.40, honest +0.88; §365). It was then read on the pool.

**Table 5.3 — the nine-network pool at equal FLOPs (trust-region agent − floored FLOPs rule).** The lever is the FLOPs rule minus uniform_F (TEST) at each depth.

| Network | Lever, 60 % | Agent − rule, 60 %: TEST (val / 10k / honest) | Lever, 45 % | Agent − rule, 45 % |
|---|---|---|---|---|
| ResNet-56 ×4, C10 | +1.79 | −0.47 (−0.39 / −0.43 / −0.63) | +2.37 | +0.04 (+0.15 / +0.09 / +0.01) |
| ResNet-56, C10 (DepGraph; FLOPs 0.48 in the 60 % column) | +0.51 | −0.31 (−0.33 / −0.32 / −0.45) | +0.97 | −0.27 (+0.11 / −0.08 / −0.21) |
| VGG-19, C100 (DepGraph) | +0.77 | −0.10 (−0.19 / −0.15 / +0.03) | +1.17 | −0.09 (+0.17 / +0.04 / +0.09) |
| ResNet-56 ×8, C10 | +1.15 | −0.36 (−0.09 / −0.22 / −0.58) | +0.99 | −0.05 (−0.43 / −0.24 / +0.02) |
| ResNet-20 ×8, C100 | +0.95 | −0.72 (−0.87 / −0.79 / +0.16) | +1.92 | **−1.43** (−0.23 / −0.83 / −1.46) |
| ResNet-56, C10 | +0.05 | −0.07 (+0.17 / +0.05 / −0.27) | +0.04 | +0.01 (+0.13 / +0.07 / +0.03) |
| MobileNetV2 ×1.4, C10 | −0.23 | +0.07 (+0.06 / +0.07 / +0.06) | −1.02 | −0.05 (+0.25 / +0.10 / −0.25) |
| DenseNet-100, C10 | +0.01 | −0.15 (−0.13 / −0.14 / −0.15) | +0.17 | −0.15 (−0.14 / −0.15 / −0.19) |
| VGG-16-BN, C10 | +0.04 | +0.18 (+0.25 / +0.22 / +0.16) | +0.34 | +0.09 (+0.29 / +0.19 / +0.03) |
| **Pool** | | **−0.21** [−0.48, −0.00] (−0.17 / −0.19 / −0.19); 2 of 9; **AT-PRIOR-POOL** (§366) | | **−0.21** [−0.58, +0.03] (+0.03 / −0.09 / −0.21); 3 of 9; **AT-PRIOR-POOL** (§371) |

Every pair is matched within 1 % of FLOPs.

**Reading.**
- At both depths the frozen FLOPs agent stays within the registered band of the FLOPs rule, but below it on most networks. On the five lever networks at 60 % every difference is negative, with a mean of −0.39 (§366). At 45 %, five lever networks only, the call is BELOW-PRIOR (−0.36, §369).
- One network again weighs most: ResNet-20 ×8 CIFAR-100. Without it the other eight average −0.06 at 45 % (§371).
- The FLOPs results are, in effect, the rule's (§371): the trust region keeps the agent near the rule by design.
- At DepGraph's 2.57× point (FLOPs 0.39) on DepGraph's ResNet-56, the FLOPs agent is BELOW-PRIOR: −0.35 (val −0.31, 10k −0.33, honest −0.52; §382).
- On VGG-16 C10 at HRank's and CHIP's FLOPs points (0.465 and 0.34), the FLOPs agent is AT-PRIOR: +0.21 (val −0.07, 10k +0.07, honest +0.15) and −0.27 (val −0.25, 10k −0.26, honest −0.49). The lever is limited there: +0.04 and +0.41 (§391).
- On ResNet-110 C10 at FLOPs 0.48 and 0.40 (the ≈52 % and ≈60 % ↓ points), it is AT-PRIOR too: −0.07 (val −0.13, 10k −0.10, honest +0.11) and −0.21 (val +0.01, 10k −0.10, honest −0.35). The lever is limited again: +0.13 and +0.31 (§392). The frozen FLOPs agent has not beaten the FLOPs prior on any literature cell (§365, §382, §391, §392).
- The lever grows with depth on the lever networks, from 60 % to 45 % FLOPs. The lever-less networks stay within ±0.35 of uniform at both depths, except MobileNetV2 ×1.4 at 45 %, where uniform beats the rule by 1.02 (§366, §368, §371).

**Cost-awareness depends on the budget.** On ResNet-56 ×4 at FLOPs 0.6, the cost-aware rule is *worse* than plain sensitivity: −0.33 (val −0.35, 10k −0.34, honest −0.78; COST-HURTS, §355). With both rules floored the gap is −0.48 (§361). At equal parameters it is better on all three networks measured: +0.47, +0.39 and +0.44 (§355). This asymmetry is why each budget type has its own prior.

### 5.4 Is there an allocation above the priors?

Three screens searched for plans above each prior on the five lever networks, with no agent involved. The alternatives were tilts of the rule (α 0.25 to 2), perturbed samples around it, plain sensitivity, and inner. Each was screened on a separate seed and confirmed on seeds 42–44.

| Screen | Budget | Mean of best alternative − prior: TEST | 10k | Positive | Call |
|---|---|---|---|---|---|
| FLOPs 0.6 (HEAD-F) | FLOPs | −0.04 | not given | 1 of 5 | NO-HEADROOM (§367) |
| FLOPs 0.45 (HEAD-F2) | FLOPs | +0.04 | not given | 3 of 5 | NO-HEADROOM (§368) |
| Params 0.6 (HEAD-P) | params | +0.22 | +0.18 | 4 of 5 | NO-HEADROOM-P (§372) |

- No direction tried found headroom above either prior at these budgets. That supports "the agent matches the best rule" as the honest framing at moderate budgets, and it bounds what any allocation agent could add there.
- The parameter prior leaves more room than the FLOPs prior. The room sits on the two CIFAR-100 networks: VGG-19 +0.69 with a flatter tilt, and ResNet-20 ×8 +0.59 with inner, at 0.69–0.70× the rule's FLOPs (§372).
- The catalog agent captures most of VGG-19's room at 60 % (+0.56 over the rule, §370) but not ResNet-20 ×8's (−0.12).
- The screens also re-measured the proxies (§5.7).

### 5.5 Transfer across families and datasets

**Table 5.5 — transfer cells (params 0.6; DepGraph's ResNet-56 at 0.47).** The first seven rows use realizable arms (§363). The last three predate that rule (§353, §356, §359).

| Agent (catalog) | Network (role) | Contrast | TEST | val | 10k | honest | Call |
|---|---|---|---|---|---|---|---|
| catalog (T1) | ResNet-56 DepGraph (held out) | − uniform | +1.10 | +0.79 | +0.94 | +1.18 | T1-R56-HOLDS (§363) |
| no ResNet (LOFO-R, 6 networks) | ResNet-56 DepGraph (unseen family) | − uniform | **+0.95** | +0.63 | +0.79 | +1.19 | LOFO-R-R56-HOLDS (§363) |
| | | − cost-aware rule | −0.04 | −0.01 | −0.03 | −0.06 | reported |
| no VGG (LOFO-V, 7 networks) | VGG-19 C100 (unseen family and dataset) | − uniform | +0.65 | +1.11 | +0.88 | +0.61 | **FAILS-CLEAN** (§363), with the next row |
| | VGG-16 C10 (unseen family) | − uniform | **−0.33** | −0.01 | −0.17 | −0.34 | |
| catalog (T1) | VGG-19 C100 (unseen dataset) | − uniform | +0.76 | +1.13 | +0.94 | +1.34 | LEVER-LIMITED: sens − uniform +0.45 < +0.5 (§363) |
| catalog (T1) | VGG-16 C10 (unseen depth) | − uniform | −0.35 | +0.12 | −0.12 | −0.33 | lever −0.19 (§363) |
| catalog (T1) | RepVGG-A0 / A1, C10 (no catalog family) | − uniform | +0.23 (pooled) | — | — | — | LEVER-LIMITED: lever +0.04 (§353) |
| catalog (T1) | RepVGG-A0, Fashion-MNIST | − uniform | −0.12 | +0.21 | +0.04 | −0.16 | LEVER-LIMITED: lever −0.27 (§359) |
| with 8 CIFAR-100 networks (T1-C) | five contexts | − catalog agent | +0.02 (mean) | — | — | — | POOL-NEUTRAL (§356) |

**Reading.**
- **One fold holds.** Trained without any ResNet, the frozen agent beats uniform on DepGraph's ResNet-56 and lands at the cost-aware rule (§363). It does less well on the skinny ResNet-56 ×4: there it sits at uniform's level, about a point below sensitivity, inner and the catalog agent, so it did not learn to keep residual streams whole (§354).
- **One fold fails, in a cell where allocation barely matters.** Trained without any VGG, the agent beats uniform on VGG-19 CIFAR-100 but falls 0.33 below it on VGG-16. On VGG-16 at 60 % the sensitivity rule is itself below uniform (−0.19), and the catalog agent, which saw VGGs, is −0.35 there too (§363). The registered call stands as FAILS-CLEAN.
- **Families no catalog holds give no informative cell.** On RepVGG the sensitivity rule is level with uniform, so no allocation can show anything (§353, §359). ShuffleNetV2's stage streams can only be masked, not cut, so its rows are invalid as cuts (§353).
- **Within the pool**, the frozen agent's parameter results (§5.2) cover:
  - two CIFAR-100 networks, a dataset absent from the catalog;
  - a VGG depth the catalog lacks;
  - two of DepGraph's own checkpoints.

  The four families themselves are all in the catalog.
- **A caveat on both folds.** Each used the input standardizer fitted on the full catalog, which includes the left-out family, so that family's feature statistics shape the folds' input scaling [H]. Re-runs with fold-fitted standardizers are planned (§354, §358).
- **Adding recoverable CIFAR-100 networks to the catalog** was neutral overall (+0.02). It helped on the added dataset: +0.61 over the catalog agent on VGG-19 CIFAR-100 (val −0.37, 10k +0.12, honest +0.40; §356; reported).

### 5.6 Deep budgets and the reward

Three single-change retrains of the catalog agent asked what removes the ResNet-20 ×8 CIFAR-100 failure at 35 %. Each is read against the cost-aware rule and against the catalog agent.

| Retrain (one change) | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − catalog agent: TEST (val / 10k / honest) | Call | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|
| training floor 0.25 (T5-floor; rule also floored at 0.25) | −0.35 [−1.48, +0.26] (−0.41 / −0.38 / −0.31) | 4 / 9 | BELOW-PRIOR-POOL (§376) | +0.02 (+0.01 / +0.01 / +0.15) | SAME | −3.79 |
| training range down to 25 % (T5-deep) | −0.46 [−2.06, +0.33]† (−0.44 / −0.46 / −0.41) | 5 / 9 | BELOW-PRIOR-POOL (§377) | −0.04 (+0.01 / −0.02 / +0.03) | SAME | −5.11 |
| batch-norm reward (T5-bn) | **+0.03** [−0.38, +0.43] (−0.13 / −0.05 / +0.12) | 4 / 9 | **AT-PRIOR-POOL** (§378) | **+0.45** (+0.32 / +0.38 / +0.57), positive on 3 of 9 | **LEARNS-MORE**, carried by one network | **+1.06** (+1.09 / +1.08 / +1.27) |
| *(catalog agent, for reference)* | −0.42 | 5 / 9 | BELOW (§374) | — | — | −5.23 |

† §377 prints the same interval as §374; it may be a copy (ledger check pending).

**The batch-norm agent across its training range** (§378, §381):

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − catalog agent: TEST [CI] (val / 10k / honest) | Call |
|---|---|---|---|---|---|
| 60 % | +0.11 [−0.07, +0.29] (+0.05 / +0.08 / +0.10) | 7 / 9 | AT-PRIOR-POOL (§381) | +0.12 [−0.08, +0.31] (−0.05 / +0.03 / +0.03) | SAME |
| 45 % | +0.15 [−0.23, +0.60] (+0.13 / +0.14 / +0.12) | 5 / 9 | AT-PRIOR-POOL (§381) | −0.04 [−0.26, +0.28] (+0.00 / −0.02 / −0.12) | SAME |
| 35 % | +0.03 [−0.38, +0.43] (−0.13 / −0.05 / +0.12) | 4 / 9 | AT-PRIOR-POOL (§378) | +0.45 [−0.51, +2.39] (+0.32 / +0.38 / +0.57) | LEARNS-MORE |

**The batch-norm reward on the deep training range** (T6-bn: the deeper-range agent's line with `bn8`, §383, §384). At 60 % and 45 % the second contrast is against the shallower-range batch-norm agent, because the raw-cut deep-range agent was not read there:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − deep-range agent, raw-cut reward: TEST [CI] (val / 10k / honest) | Call | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|
| 60 % | **+0.02** [−0.12, +0.16] (+0.12 / +0.08 / +0.03) | 6 / 9 | AT-PRIOR-POOL (§384) | against the shallower-range batch-norm agent: −0.09 [−0.28, +0.11] (+0.07 / −0.01 / −0.07) | SAME | +0.20 (+0.63 / +0.42 / +0.03) |
| 45 % | **+0.2956** [−0.05, +0.79] (+0.21 / +0.26 / +0.35) | 6 / 9 | AT-PRIOR-POOL (§384), 0.004 under the bar | against the shallower-range batch-norm agent: +0.15 [−0.11, +0.47] (+0.09 / +0.12 / +0.23) | SAME | +1.89 (+1.59 / +1.74 / +1.99) |
| 35 % | **+0.19** [−0.02, +0.43] (+0.06 / +0.13 / +0.23) | 6 / 9 | AT-PRIOR-POOL | +0.65 [−0.12, +2.33] (+0.51 / +0.58 / +0.64) | REWARD-HELPS | +0.41 (+0.37 / +0.39 / +0.49) |
| 25 % | **−0.17** [−1.11, +0.32] (−0.10 / −0.14 / −0.10) | 5 / 9 | AT-PRIOR-POOL | +1.43 [−0.34, +5.37] (+1.46 / +1.44 / +1.38) | REWARD-HELPS | −2.41 (−1.59 / −2.00 / −2.45) |

**Reading.**
- Only the reward change removed the failure. With the batch-norm reward, ResNet-20 ×8 CIFAR-100 goes from −5.23 to +1.06 against the rule, positive on every seed, and its plans lose the all-or-nothing shape (§378).
- The fix costs two thin ResNets: ResNet-56 ×4 −0.96 (val −1.05, 10k −1.00, honest −0.71) and ResNet-56 ×8 −0.45 against the rule. On the other eight networks it averages −0.29 against the catalog agent (§378).
- The batch-norm fix costs no pooled ground at the moderate budgets. At 60 % and 45 % the agent is at the cost-aware rule and the same as the catalog agent (§381). It is the first parameter agent at the prior at every depth read so far, where the catalog agent fell below it at 35 %. That summarises three per-depth calls; it is not a pooled call.
- On DepGraph's ResNet-56 at params 0.38 (the literature cell, §382), the batch-norm agent's single-network call fired BEATS-PRIOR on the TEST half: +0.35, positive on every seed. The validation half (+0.02) and the full test set (+0.19) do not carry it, and the agent keeps up to 1.11× the rule's FLOPs. It is a fired call on one network, not a pooled result. The catalog agent sits at the rule there (+0.05), and so does the deep-range batch-norm agent (+0.03; val +0.15, 10k +0.09; §385).
- Its cost on the thin ResNets persists below 60 %. ResNet-56 ×4 is −0.73 at 45 % (val −0.41, 10k −0.57, honest −0.93; every seed negative) and −0.96 at 35 %, and DepGraph's ResNet-56 is −0.28 at 45 %. Its gain on ResNet-20 ×8 CIFAR-100 holds: +1.57 at 45 % (val +1.44, 10k +1.50, honest +1.32), where the cost-aware rule is itself weak (§373, §381).
- **On the deep training range, the batch-norm reward gives the first single agent at the params prior at all four depths read** (§383, §384). It is also the first at 25 %. The same line with the raw-cut reward was −1.60 there (§380).
- The reward's gain is carried by ResNet-20 ×8 CIFAR-100: +5.52 at 35 % and +14.43 at 25 % against the raw-cut agent. On the other eight networks its effect is +0.05 and −0.20.
- The collapse is removed at 35 % (+0.41 against the rule) and reduced, not removed, at 25 % (−2.41, from −16.84). At 25 % seed 42 still writes the all-or-nothing plan (−7.56).
- **A three-seed ensemble does not remove it** (§386). Averaging T6-bn's three seeds' scores and decoding once gives an agent at the rule pooled (−0.2837, AT-PRIOR-POOL, positive on 5 of 9; val −0.36, 10k −0.32, both past −0.3) and level with the single seeds (−0.1096, SAME). But ResNet-20 ×8 CIFAR-100 reads −3.61 against the rule on every recovery seed (−3.62 / −2.26 / −4.94). Averaging removes the collapsing seed's extreme plan (median keep 1.00) but keeps the groups that every member cuts to the floor (the ensemble's plan: keep min 0.09, median 0.83). [H] The failure is systematic in what this line learned about the network, not one seed's accident. The other eight are +0.13 against the rule.
- There is no thin-ResNet cost on the deep range. ResNet-56 ×4 reads +0.03, +0.31, +0.25 and +0.37 against the rule from 60 % to 25 %, where the shallower-range batch-norm agent had −0.73 at 45 % and −0.96 at 35 %. At 45 % the deep-range agent is +1.05 against that agent on this network (§384). [H] The deeper range and the reward together avoid that trade.
- DepGraph's ResNet-56 sits slightly below the rule at every depth: −0.32 at 47 % (val +0.27), then −0.22, −0.05 and −0.17 (§383, §384).
- The agent's edge on the other eight networks is +0.17 at 35 % and +0.11 at 25 %. The raw-cut agent had +0.30 at 25 %.
- A training floor moves the collapse to the floor: plans with minimum keep 0.25 still fail, at −3.79 (§376). A deeper training range leaves it unchanged (§377).
- **More instances do not help, and the collapse is learned with training** (§388). T6-bn's snapshots at 3,000, 6,000 and 9,000 instances, read beside its 12,000-instance policy, leave the other eight flat on val (+0.13, +0.12, +0.14, +0.09). The registered rule (a rise of +0.10 from 6k to 12k on TEST and val) does not fire: −0.12 and −0.04. On ResNet-20 ×8 CIFAR-100 the 3,000-instance snapshot does not collapse (+0.15, median keeps 0.69–0.75). From 6,000 instances seed 42 does (−14.86, −19.58, −7.56). [H] Longer training on a CIFAR-10-heavy catalog teaches the habit that fails on CIFAR-100, so the catalog, not the length, is the lever.
- At 25 % the agent trained on the raw-cut reward fails harder on the same network: −16.84 (§380). The failure grows with depth under that reward: −5.23 and −5.11 at 35 %, then −16.84 at 25 %.
- **Decision (Ido, 17 Oct) [D].** T6-fs is the params agent of record and T4-F the FLOPs agent. T6-bn, T7-C and T8-fs stay as ablations (below). The final tables re-read both agents at five seeds with the deterministic fine-tune (queue row FINAL).

**The 32-step reward on the deep training range** (T6-fs: the same line with `fs32`, which ranks plans closer to the final, ρ 0.58 against bn8's 0.44 in §379; §387). Read at 35 % and 25 %:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − deep-range agent, raw-cut reward: TEST [CI] | Call | − batch-norm agent (reported) | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|---|
| 35 % | **+0.2356** [+0.01, +0.50] (+0.21 / +0.22 / +0.31) | 7 / 9 | AT-PRIOR-POOL (§387) | +0.70 [−0.09, +2.25] | REWARD-HELPS | +0.04 | +0.54 (+0.67 / +0.60 / +1.18) |
| 25 % | **+0.1615** [−0.46, +0.60] (+0.22 / +0.19 / +0.22) | 6 / 9 | AT-PRIOR-POOL (§387) | +1.76 [−0.20, +5.51] | REWARD-HELPS | +0.34 | −0.93 (−0.68 / −0.80 / −1.05) |
| 60 % | **+0.0504** [−0.17, +0.28] (+0.24 / +0.14 / −0.08) | 4 / 9 | AT-PRIOR-POOL (§390) | — | — | +0.03 (SAME) | +0.43 |
| 45 % | **+0.4193** [+0.02, +1.00] (+0.34 / +0.38 / +0.46) | 6 / 9 | **BEATS-PRIOR-P-POOL** (§390) | — | — | +0.12 (SAME) | +2.28 |

- **At 35 % the pooled 95 % range lies above zero for the first time against the params prior.** It is still under the +0.3 BEATS bar, so the call is AT-PRIOR-POOL (§387).
- **At 25 % the 32-step reward gives the better agent:** +0.16 against the rule, where the batch-norm reward reads −0.17 (a reported +0.34). On the other eight networks it is +0.30, against +0.11.
- **The collapse on ResNet-20 ×8 CIFAR-100 shrinks again,** to −0.93 at 25 % (from −2.41 and −16.84). No seed writes the all-or-nothing plan. The batch-norm agent's collapsing seed reads +3.40, but another seed is still −4.70, with a plan that cuts some groups to the floor (§387).
- [H] Of the two rewards trained in parallel, the one that ranks plans more like the final gives the better agent at the tightest budget, as the fidelity account (§5.7) predicts. At 45 % it is the first agent to beat the prior pooled (§390).

**CIFAR-100 networks in the training catalog** (T7-C: T6-bn's line on the 18-network catalog of §356, T1's ten plus eight CIFAR-100 networks; §389). CIFAR-100 is now a seen dataset, so the two CIFAR-100 pool networks are held-out instances of it:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − T6-bn (the same line without CIFAR-100): TEST [CI] | Call | CIFAR-100 pair / CIFAR-10 seven, against T6-bn | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|---|
| 35 % | **+0.2778** [−0.06, +0.70] (+0.11 / +0.19 / +0.32) | 6 / 9 | AT-PRIOR-POOL (§389) | +0.08 [−0.20, +0.49] | SAME | +0.67 / −0.08 | +1.57 |
| 25 % | **+0.3422** [−0.12, +1.09] (+0.32 / +0.33 / +0.50) | 5 / 9 | AT-PRIOR-POOL (§389) | +0.52 [−0.24, +2.03] | CATALOG-HELPS | +2.13 / +0.06 | +2.09 |

- **Seeing CIFAR-100 in training removes the collapse at 25 %:** ResNet-20 ×8 CIFAR-100 reads +2.09 against the rule, against −2.41 without CIFAR-100 and −0.93 with the 32-step reward. No seed writes the all-or-nothing plan; one seed is still −1.78 (§389).
- The seven CIFAR-10 networks do not move (+0.06 at 25 %, −0.08 at 35 %). The small cost is on VGG-19 CIFAR-100 (−0.17 against the rule at 25 %).
- **Two near-misses of BEATS-PRIOR-P-POOL, on different conditions.** At 35 % the pooled +0.2778 is under the +0.3 bar, with 6 of 9 positive. At 25 % the pooled +0.3422 is above the bar on TEST, val and 10k, but only 5 of 9 networks are positive where six are required. Both calls are AT-PRIOR-POOL.
- [H] The reward lever (T6-fs) and the catalog lever (T7-C) each help at 25 %, through different networks. Their combination is the first rung of the catalog scale-up.

**The two levers together** (T8-fs: T6-fs's line, the 32-step reward on the deep range, trained on T7-C's 18-network catalog; §400). It is read at all four depths against the rule, against T6-fs (the catalog on the 32-step line) and, at 35 % and 25 %, against T7-C (the reward on the 18-network catalog):

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − T6-fs: TEST [CI] | Call | − T7-C: TEST [CI] | Call | CIFAR-100 pair / CIFAR-10 seven, against T6-fs |
|---|---|---|---|---|---|---|---|---|
| 60 % | **−0.02** [−0.17, +0.15] (+0.20 / +0.09 / −0.03) | 5 / 9 | AT-PRIOR-POOL | −0.07 [−0.29, +0.13] | SAME | — | — | −0.23 / −0.02 |
| 45 % | **+0.26** [+0.03, +0.54] (+0.27 / +0.27 / +0.20) | 8 / 9 | AT-PRIOR-POOL | −0.16 [−0.66, +0.14] | SAME | — | — | −0.75 / +0.02 |
| 35 % | **+0.09** [−0.66, +0.59] (−0.06 / +0.02 / +0.13) | 7 / 9 | AT-PRIOR-POOL | −0.14 [−1.00, +0.47] | SAME | −0.19 [−1.10, +0.25] | SAME | −0.67 / +0.01 |
| 25 % | **−0.41** [−2.27, +0.42] (−0.21 / −0.31 / −0.31) | 3 / 9 | BELOW-PRIOR-POOL | −0.57 [−2.62, +0.68] | CATALOG-HURTS | −0.75 [−2.93, +0.12] | REWARD-HURTS | −2.14 / −0.12 |

- **The levers do not add up.** The combined agent is at best level with either parent. It is no better than T6-fs at any depth, and it does not keep T6-fs's pooled beat at 45 % (+0.26 on 8 of 9 networks, under the +0.3 bar, against T6-fs's +0.42).
- **The losses sit on CIFAR-100.** On the seven CIFAR-10 networks every contrast lies within −0.12 to +0.22. At 25 % the CIFAR-100 pair reads −2.26 against the rule, −2.14 against T6-fs and −3.22 against T7-C. On the batch-norm line the same networks in the catalog helped exactly this pair at 25 % (§389), so the catalog's effect depends on the reward.
- **One frozen policy fails on one network.** Seed 42 on ResNet-20 ×8 CIFAR-100 loses 37.2 points at 25 % and 22.1 at 35 %, where every other arm and seed loses 18.0–26.2 and 12.1–16.6. That is −15.2 against the rule at 25 %, far outside run noise (§398). Its plans cut more broadly than the other seeds' (median group keep 0.69 against 0.83 and 0.75).
- [H] At a fixed 12,000 instances, going from 10 to 18 catalog networks cuts each network's share of training to about 0.56×. The catalog ladder should scale instances with the catalog before it reads breadth (`docs/CATALOG_SCALE_PLAN.md` §4).

### 5.7 Which cheap rewards rank plans like the final fine-tune?

A proxy is useful as a reward only if it orders plans the way the 100-epoch recovery does. Each measurement takes a fixed set of real plans per network, scores them with the proxy on the validation half before recovery, and compares that ranking with the recovered one.

| Study | Plans | Result | § |
|---|---|---|---|
| D-PROXY, params 0.6 | 11 per family, near the rules | The raw cut is VALID on both voting families (ρ +0.90 thin, +0.66 DepGraph's ResNet-56), so it is the reward. MobileNetV2 is CEILING-BOUND: the recovery does not rank its own plans consistently | §340 |
| D-PROXY-2, the agent's own plans | 12 per instance | CEILING-BOUND at κ 0.6 and 0.8 (ceiling ρ +0.23 / −0.22). After recovery the agent's sampled plans are indistinguishable, so the cut reward stands | §349 |
| D-PROXY-F, FLOPs 0.6 | the direct FLOPs agent's plan and three rules | All four proxies FLOPS-VALID: each ranks the agent's plan below the rule on all three seeds (cut ρ +0.87) | §351 |
| D-PROXY-3, inside the headroom screens | 6–7 per lever network | FLOPs 0.6: cut +0.47, ft1 +0.01. FLOPs 0.45: bn32 +0.39 best. Params 0.6: **ft1 +0.63**, cut +0.21 | §367, §368, §372 |
| D-PROXY-4, params 0.35 | POOL-P35's plans | ft1 **P35-VALID at the threshold** (+0.60, 55 / 71 pairs; ceiling +0.73). Cut +0.41, bn8 +0.45, bn32 +0.51, ft3 +0.41, all WEAK. On ResNet-20 ×8 C100: cut −0.84, bn8 −0.85, ft1 +0.94 | §375 |
| D-PROXY-5, step-limited fine-tunes | as D-PROXY-4, two learning rates | None VALID. fs32 at lr 0.01 is closest (+0.58, 52 / 71; +0.43 on ResNet-20 ×8 C100) at about a quarter of ft1's cost. Every lr-0.05 arm is worse. ft1 is VALID again (+0.61) | §379 |

**Reading.**
- At equal parameters and deep budgets, the cheap rewards (the raw cut and batch-norm re-estimation) rank plans weakly, and on the network where the agent fails they rank them backwards. One epoch of fine-tuning ranks them validly.
- Fidelity is a property of the plan set, not of the proxy alone. Among near-equivalent plans it falls toward the recovery's own ceiling (§349), as the literature reports for near-best candidates [111, 121].
- A proxy's ranking of four fixed plans is not a sufficient test of what it teaches. The batch-norm proxy ranks ResNet-20 ×8 C100's fixed plans backwards (§375), yet training on it removed the failure (§378).
- **Selection.** D-PROXY-4's reference ranking was the TEST-half recovery. D-PROXY-5 moved it to the validation half, so that no reward is selected on TEST (§379). The T6 rewards were chosen by Ido after D-PROXY-5's registered selection found none valid.

### 5.8 The recovery recipe

Every row from §341 on uses one recovery, G2 (§4.2). It replaced the per-family recipes of §330 through a sequence of registered checks:
- **Keep the last epoch** on every final fine-tune (§330). The old default restored epoch 1 on most pruned networks.
- **A one-shot cut** costs little against a walked one at the size point (MobileNetV2 ×0.5: −0.39 at 10k, COST-TRADE; §333).
- **75–100 epochs** of recovery suffice (§334).
- **Batch 256** helps two narrow families, and the gain is the step size, not the batch (§335, §338).
- **On VGG-19 CIFAR-100**, the one family where a four-pair check had kept the plain recipe (§343), G2 is SPEED-EQUIVALENT to it over 16 pairs: mean 10k −0.08 (TEST −0.05, val −0.12, honest +0.08; §348).

Under G2 the allocation lever survives on the thin ResNet: sensitivity − uniform +1.02 over five seeds (SURVIVES, §342). It is absorbed on MobileNetV2 ×0.5 (+0.29, §342). A G2 recovery takes about 3 min on thin ResNet-56 ×4, against 15–17 min for the plain recipe (§329).

### 5.9 The literature anchors: DepGraph's ResNet-56, VGG-16 and ResNet-110 on CIFAR-10

DepGraph's own checkpoint is in the pool, so its published rows are the natural anchors. OCSPruner reports the same network at the FLOPs of DepGraph's 2.57× row. SPECTRA's arms run one cut and the G2 recovery; the published methods use their own recipes.

**At DepGraph's 2.11× point (FLOPs 0.48 on our counter).**

| Arm | FLOPs kept | Params kept | TEST | 10k | Recovery |
|---|---|---|---|---|---|
| DepGraph, as published [12] | 0.474 (their counter); 0.480 (ours, on their released model) | 0.470 (theirs); 0.508 (ours) | — | +0.24 | their sparse training and fine-tune; their epoch is picked on the test set (§346) |
| Floored FLOPs sensitivity rule | 0.48 | 0.53–0.60 | +0.48 | +0.56 | G2 (§362) |
| Uniform_F | 0.48 | — | −0.03 | +0.03 | G2 (§362) |
| Frozen FLOPs agent (T4-F) | 0.48 | 0.96–1.01× the rule's | the rule's −0.31 | the rule's −0.32, i.e. about +0.24 [computed] | G2 (§365) |
| Frozen catalog agent (T1), at DepGraph's *params* point | about 0.61 | 0.47 | +0.50 | +0.51 | G2 (§346); about 1.2–1.3× DepGraph's FLOPs on our counter |

**At DepGraph's 2.57× point, which is also OCSPruner's (FLOPs 0.39), and at its params (0.38)** (§382, §385, §391; mean over seeds 42–44).

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| DepGraph, as published [12] | 0.389 | 0.382 | — | +0.11 | — | their recipe; epoch picked on the test set |
| OCSPruner, Table 5, pretrained start [93] | 0.388 | 0.423 | — | −0.51 (94.01 → 93.50) | — | their one-cycle schedule, own baseline |
| Floored FLOPs sensitivity rule | 0.389 | 0.463 | +0.11 (+0.36) | +0.24 | −0.70 | G2 |
| Frozen FLOPs agent (T4-F) | 0.390 | 0.462 | −0.24 (+0.05) | −0.10 | −1.22 | G2 |
| Uniform_F | 0.390 | 0.375 | −0.54 (−0.59) | −0.57 | −1.47 | G2 |
| Frozen batch-norm agent (T5-bn), at params 0.38 | 0.508 | 0.380 | +0.40 (+0.21) | +0.31 | −0.44 | G2; 1.3× DepGraph's FLOPs |
| Frozen deep-range batch-norm agent (T6-bn), at params 0.38 | 0.509 | 0.380 | +0.07 (+0.35) | +0.21 | −0.88 | G2 (§385); 1.3× DepGraph's FLOPs |
| Frozen deep-range 32-step agent (T6-fs), at params 0.38 | 0.554 | 0.379 | +0.32 (+0.54) | +0.43 | −0.62 | G2 (§391); 1.42× DepGraph's FLOPs |
| Frozen catalog agent (T1), at params 0.38 | 0.531 | 0.379 | +0.09 (+0.18) | +0.14 | −0.96 | G2 |
| Floored cost-aware rule, at params 0.38 | 0.481 | 0.380 | +0.05 (+0.19) | +0.12 | −0.80 | G2 |

**Reading.**
- **The FLOPs rule lands above DepGraph's published change at both points:** +0.56 and +0.24 at 10k, against +0.24 and +0.11. The frozen FLOPs agent lands at or below it: about +0.24 [computed] and −0.10. OCSPruner's row sits at −0.51.
- **At equal FLOPs, SPECTRA keeps more parameters.** At the 2.57× point that is 0.46, against DepGraph's 0.38 and OCSPruner's 0.42.
- **The honest caveat.** Our 100-epoch recovery lifts the unpruned network itself by +0.64 to +1.38 points (§382), so every honest value at the 2.57× point is negative. Raw changes under different recipes are not a contest: the rows sit side by side and are **not a beat**.
- **The registered same-loop calls at this point** (§382): the FLOPs agent is BELOW-PRIOR, −0.35 (val −0.31, 10k −0.33, honest −0.52). At params 0.38 the batch-norm agent's call fired BEATS-PRIOR on the TEST half only: +0.35, with val +0.02, 10k +0.19 and honest +0.36 (§5.6). The deep-range batch-norm agent is AT-PRIOR there (+0.03; val +0.15, 10k +0.09; §385), so that beat does not carry over to it. The deep-range 32-step agent is AT-PRIOR there too: +0.27, positive on every seed and every view (val +0.35, 10k +0.31, honest +0.18), at 1.10–1.23× the rule's FLOPs (§391).
- The walk era's DepGraph row (§157) is history (Appendix B).

**VGG-16-BN on CIFAR-10 at HRank's 53.5 % ↓ and CHIP's 66.6 % ↓ FLOPs points** (§391; mean over seeds 42–44). This is chenyaofo's checkpoint, one of the nine held-out pool networks, with an origin of 93.74 % on TEST 5k. The published rows start from a base of 93.96 (the GAL / HRank / CHIP lineage) and are quoted on the full test set. PRELIM.

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| HRank, as published | 0.465 | 0.171 | — | 93.96 → 93.43 (−0.53) | — | their recipe |
| CHIP, as published | 0.419 | 0.184 | — | 93.96 → 93.86 (−0.10) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.465 | 0.109 | −0.06 (+0.31) | +0.12 | −0.57 | G2 |
| Frozen FLOPs agent (T4-F) | 0.465 | 0.116 | +0.15 (+0.23) | +0.19 | −0.41 | G2 |
| Uniform_F | 0.465 | 0.528 | −0.10 (−0.08) | −0.09 | −0.73 | G2 |
| HRank, as published | 0.347 | — | — | 93.96 → 92.34 (−1.62) | — | their recipe |
| CHIP, as published | 0.334 | — | — | 93.96 → 93.72 (−0.24) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.340 | 0.063 | −0.22 (+0.07) | −0.07 | −0.49 | G2 |
| Frozen FLOPs agent (T4-F) | 0.340 | 0.071 | −0.49 (−0.18) | −0.33 | −0.97 | G2 |
| Uniform_F | 0.340 | 0.321 | −0.63 (−0.37) | −0.50 | −1.03 | G2 |

**Reading.**
- **The registered calls** (§391): T4-F − sens_F is +0.21 at 0.465 and −0.27 at 0.34, AT-PRIOR at both. The lever is limited at both points: +0.04 and +0.41.
- **Beside the published rows (setup S1, never a beat).** The base checkpoint differs (93.74 % on TEST 5k against 93.96), and our 100-epoch recovery lifts the unpruned VGG-16 by about +0.56 on TEST 5k. At HRank's point the SPECTRA arms sit within ±0.2 of their origin on the full test set, with the rules removing 88–89 % of params. HRank published −0.53 with 83 % removed.
- [H] VGG-16's deep 512-channel layers hold most of the params and few of the FLOPs, so a FLOPs budget removes about 90 % of the params almost for free.

**ResNet-110 on CIFAR-10 at the ≈52 % and ≈60 % ↓ FLOPs points** (§392; mean over seeds 42–44). This is akamaster's checkpoint, loaded through the repaired loader (§391), with an origin of 93.84 % on TEST 5k. The published rows start from their own bases and are quoted on the full test set. PRELIM.

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| FPGM-only 40 %, as published | 0.477 | — | — | 93.68 → 93.74 (+0.06) | — | their recipe |
| CHIP, as published | 0.479 | 0.517 | — | 93.50 → 94.44 (+0.94) | — | their recipe |
| OCSPruner, as published | 0.464 [H] | — | — | 94.36 → 94.30 (−0.06) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.479 | 0.584 | +0.05 (+0.58) | +0.31 | −0.58 | G2 |
| Frozen FLOPs agent (T4-F) | 0.479 | 0.572 | −0.03 (+0.45) | +0.21 | −0.47 | G2 |
| Uniform_F | 0.480 | 0.500 | −0.09 (+0.52) | +0.22 | −0.69 | G2 |
| LFPC, as published (pre-trained start) | 0.397 | — | — | 93.68 → 93.07 (−0.61) | — | their recipe |
| HRank, as published | 0.418 | — | — | 93.50 → 93.36 (−0.14) | — | their recipe |
| DECORE-300, as published | 0.382 | 0.352 | — | 93.50 → 93.50 (0.00) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.400 | 0.493 | +0.05 (+0.52) | +0.29 | −0.51 | G2 |
| Frozen FLOPs agent (T4-F) | 0.400 | 0.485 | −0.16 (+0.53) | +0.18 | −0.86 | G2 |
| Uniform_F | 0.400 | 0.380 | −0.25 (+0.35) | +0.05 | −0.61 | G2 |

**Reading.**
- **The registered calls** (§392): T4-F − sens_F is −0.07 at 0.48 and −0.21 at 0.40, AT-PRIOR at both. The lever is limited at both points: +0.13 and +0.31.
- **Beside the published rows (setup S1, never a beat).** Every SPECTRA arm ends within 0.3 of its origin on TEST at both points. Our recovery lifts the unpruned ResNet-110 by about +0.4 to +0.7, so every honest value is negative.

**Toward 8× on DepGraph's ResNet-56** (§399; the fifth and sixth operating points; single network, three seeds; PRELIM)

| Point | agent − prior | val | 10k | honest | call | prior − uniform |
|---|---|---|---|---|---|---|
| params 0.18 (5.5×) | −0.35 | +0.37 | +0.01 | −0.47 | BELOW-PRIOR (TEST only) | +2.13 |
| params 0.125 (8×) | +0.01 | −0.02 | −0.01 | −0.12 | AT-PRIOR | +3.07 |
| FLOPs 0.25 (4×) | +0.40 | −0.23 | +0.09 | +0.48 | BEATS-PRIOR (TEST only) | +1.07 |
| FLOPs 0.125 (8×) | −0.19 | +0.50 | +0.16 | −0.19 | AT-PRIOR | +1.92 |

- **Across seven params points and four FLOPs points, from 2× to 8×,** the frozen agents track the priors within ±0.35.
- **The two TEST-only calls lie within the run noise** of a single-network contrast (about ±0.23, §398).
- **Toward 8× the allocation itself matters a great deal:** the priors beat uniform cuts by 1 to 3 points.

### 5.10 Scratch training, measured speed and run-to-run noise

**Would training from scratch do as well?** (Liu et al.; §394; DepGraph's ResNet-56, the 32-step agent's architectures; mean over seeds 42–44; PRELIM)

| Point | inherit − Scratch-E (200 epochs) | inherit − Scratch-B (297–388 epochs) |
|---|---|---|
| params 0.47 (DepGraph's 2.11×) | +0.05 (val +0.30, 10k +0.18): SCRATCH-SAME | −0.23 (val −0.11, 10k −0.17): SCRATCH-SAME |
| params 0.38 (DepGraph's 2.57×) | +0.39 (val +0.61, 10k +0.50): INHERIT-HELPS | −0.03 (val −0.16, 10k −0.10): SCRATCH-SAME |

- **Scratch-B matches the inherited weights at both points.** This is Liu et al.'s result for structured pruning. The agent's contribution is the architecture (the allocation), not the weights the pruned network inherits.
- **A shorter scratch run (Scratch-E) falls behind at the tighter point.**
- **The checkpoint is a weak baseline for this recipe.** Our recipe trains the unpruned network from scratch to +0.2 to +0.8 above DepGraph's checkpoint.

**Does a per-network random search match the frozen agent?** (Li et al.'s best-of-N; §396; 100 random plans in the agent's action space, decoded with its own decoder, k_min and floor, each scored by fs32 on the val half, the best recovered with G2; nine networks, seeds 42–44; PRELIM)

| Params kept | agent − search [95 % bootstrap] | val | 10k | honest | positive | call | search − cost-aware rule |
|---|---|---|---|---|---|---|---|
| 45 % | +0.51 [+0.16, +0.91] | +0.50 | +0.51 | +0.47 | 8 of 9 | AGENT-BEATS-SEARCH | −0.09 |
| 25 % | +0.45 [−0.61, +1.31] | +0.47 | +0.46 | +0.42 | 6 of 9 | AGENT-BEATS-SEARCH | −0.29 |

- **The frozen agent, which searches nothing, beats a 100-plan per-network search that uses its own training reward.** The search costs about 2.5 min per network; the agent's plan costs one forward pass.
- **The search itself is level with the cost-aware rule.** Its cheap reward alone does not recover what the agent learned.
- **Two scope limits.** At 25 % the range includes zero: on ResNet-20 ×8 CIFAR-100 the search wins by 2.67, where the agent still collapses on two seeds. And at equal params the agent keeps 1.1–2.1× the search's FLOPs.

**Does the agent's choice of channels matter?** (Blalock et al.'s random baseline at the agent's proportions; §398; T6-fs's own widths with L1 or with random channels, both on tree_v23; nine networks, seeds 42–44; PRELIM)

| Params kept | L1 − random [95 % bootstrap] | val | 10k | honest | positive | call | random − cost-aware rule |
|---|---|---|---|---|---|---|---|
| 45 % | +0.13 [−0.13, +0.34] | +0.02 | +0.07 | +0.15 | 7 of 9 | RANK-SAME | +0.23 |
| 25 % | +0.23 [−0.08, +0.59] | +0.24 | +0.24 | +0.20 | 6 of 9 | RANK-SAME | −0.12 |

- **After the 100-epoch recovery, which channels survive matters little.** The agent's layer widths carry its result, as the scratch runs suggest too.
- **L1 still helps where it matters most:** on the thin ResNet-56 ×4 and MobileNetV2 ×1.4 at 25 % (+0.86, +0.99).

**Are fewer FLOPs faster?** (§395; RTX 4090, fp32, eager PyTorch; images per second at batch 256 and ms at batch 1, each against the unpruned network in the same job; seed 42)

| Network and arms | FLOPs kept | Speed-up at batch 256 | Speed-up at batch 1 |
|---|---|---|---|
| DepGraph ResNet-56: agents and rules | 0.39–0.63 | 0.93–0.99× | 0.94–1.03× |
| DepGraph ResNet-56: uniform cuts | 0.39–0.42 | 1.06–1.07× | 0.97–1.00× |
| VGG-16: all arms | 0.465 | 1.32–1.42× | 1.01–1.08× |
| VGG-16: all arms | 0.34 | 1.63–1.65× | 1.01–1.06× |

- **On the small CIFAR ResNet, fewer FLOPs are not faster on this GPU.** [H] The network is bound by memory traffic and kernel launches. The agents and rules keep the residual streams whole, while the uniform cuts thin them and gain 6–7 %.
- **On VGG-16 the FLOPs saving becomes speed at batch 256,** though less than one for one: 2.15× fewer FLOPs gives 1.32–1.42×, and 2.94× fewer gives 1.63–1.65×.
- Table 1's FLOPs ratios carry this table beside them. Compiled or fused runtimes are not measured.

**Run-to-run noise** (§393, §397, §401).

- **The noise.** Without deterministic GPU algorithms, the 100-epoch fine-tune is not bit-reproducible. The runner also switched on cuDNN's benchmark mode, which picks algorithms by timing. Three runs of one identical cut spread over 0.42 points on DepGraph's ResNet-56 and 0.76 on ResNet-20 ×8 CIFAR-100, a tree_v21 repeat among them. So the noise is run-to-run, not a code change.
- **Its effect on the calls.** A pooled call over 27 cells averages it down. A single-network call over three seeds carries a few tenths of it.
- **Measured at scale** (§398): over 54 pairs of the same cell on two trees, the difference has a mean of −0.05 and an SD of 0.40–0.45. That is about 0.3 per run and about 0.23 for a three-seed single-network contrast. The replication of §390 reads +0.36, against +0.42.
- **The fix.** With `SPECTRA_EVAL_FT_DETERMINISTIC=1`, two runs of the same cell agree to 4 decimals on every number (DET-EXACT). That holds on DepGraph's ResNet-56 and ResNet-20 ×8 CIFAR-100 (§397), and on MobileNetV2 ×1.4, DenseNet-100 and VGG-19 CIFAR-100 (§401), so it covers every family in the pool. PyTorch named no operation without a deterministic implementation. It costs about 22–50 % more wall time, on the RTX 4090 with this software stack, and it is proposed for the final tables.

---

## 6. Discussion — DRAFT

**Why a plan, not a walk.** NEON's walk carried over to CNNs gave the agent dozens of sequential decisions per episode, each followed by a fine-tune and scored by a per-step reward that barely separated the actions (§331). The policy that maximised that signal was "keep 90 % everywhere" (§248). Making the whole allocation one action changes three things. Credit assignment becomes exact: one plan, one reward, and K plans for the same network give a baseline that removes the network's difficulty from the gradient. The fine-tune leaves the training loop, so a train takes hours instead of days (§3.7). And the budget is met by construction, so a plan can never "win" by under-spending. The first plan agent learned on its training network (§341), and the catalog-trained agent learned on held-out ones (§346), where every walk agent through v10 had been a mild or uniform clone.

**What the agent learned: cost-awareness, where cost matters [H].** In a CIFAR ResNet each stage halves the feature map and doubles the channels, so every 3×3 convolution costs about the same FLOPs ([3], §3.3 of that paper). The same fraction of the last stage holds about 16× the parameters of the first. At equal FLOPs, then, a plain sensitivity rule is already cost-aware, and a learned allocation has little to add. At equal parameters, cost-awareness is the whole game, and it is what the frozen agent learned from its reward alone. At equal parameters it performs at the level of the cost-aware rule (§370, §373). It gains most over the cost-blind rule on exactly the networks where plain sensitivity is at or below uniform (+0.52, +0.49 and +0.80 on ResNet-56, ResNet-20 ×8 CIFAR-100 and ResNet-56 ×8; §370). The FLOPs agent sits slightly below the FLOPs rule (§366, §371). The record agrees in the other direction too: at equal FLOPs the cost-aware rule is worse than plain sensitivity, and at equal parameters it is better (§355, §361). This is a hypothesis about mechanism; the per-group plans in §370–§374 are the evidence to test it, and a figure of kept fraction by stage is planned.

**Allocation, not channels or weights.** Three controls point the same way. Random channels at the agent's own widths end level with L1 channels (§398). The agent's architecture trained from scratch with the same compute ends level with its inherited weights (§394). A per-network random search over allocations ends below the agent (§396). So what SPECTRA learns, and what matters after a long recovery, is how much of each layer to keep.

**Search versus a learned prior.** A per-network random search with the agent's own cheap reward, the baseline Li et al. found competitive, ends level with the cost-aware rule and below the frozen agent at 45 % and 25 % params (§396). What the agent adds is not search but a prior learned across networks [H]: random plans scatter where the agent's plans keep structure.

**Is there anything above the prior?** At 60 % and 45 % kept, searches over plan families around the priors found nothing above them by the bar, at either budget type (§367, §368, §372). The parameter prior leaves more room than the FLOPs prior, and that room sits on the two CIFAR-100 networks (§372). This is consistent with the literature: at about 50 % FLOPs on CIFAR ResNets, controlled comparisons find that sensible allocations end within 0.15–0.31 points of each other after a full fine-tune [110, 111], about one seed's spread [11]. Gains of a point or more are reported against uniform cuts, at tighter budgets, on CIFAR-100 or on compact networks. The first registered beat was narrow: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only (§382). The second is pooled: the 32-step agent beats the cost-aware rule at 45 % params over the nine networks, +0.42 with every view above +0.3 (§390). One network carries part of it, the CIFAR-100 ResNet where the rule itself is weak; the other eight average +0.19. SPECTRA's own room appears in the same place. At 25 % params the cost-aware rule beats uniform by +0.88 points over the nine networks (§380), and the FLOPs rule's lever grows from 60 % to 45 % FLOPs (§368). On eight of the nine networks the agent's mean edge over the rule rises with depth: +0.01, +0.10 and +0.18 from 60 % to 35 %, then +0.30 for the deeper-trained agent at 25 % (§370–§374, §380). These are diagnostics after the calls, not calls.

**Reward fidelity is the bottleneck at depth.** The training reward is a proxy for the accuracy after the final fine-tune. Measured on real plans at equal parameters, the raw-cut reward ranks plans only weakly like the final, and on the network where the agent fails it ranks them backwards (§375). A one-epoch fine-tune ranks them validly (§375, §379), but it costs too much to train on. The FLOPs agents never failed the same way. The likely reason is the trust region, which keeps every group within e^{±0.3} of the rule's weight, so an all-or-nothing plan cannot form [H]. Their worst network is the same one (−0.72 at 60 %, −1.43 at 45 %; §366, §369). The failure on ResNet-20 ×8 CIFAR-100 grows with depth under the raw-cut reward (§374, §377, §380). Neither a training floor nor a deeper training range removes it (§376, §377). A reward that re-estimates batch-norm statistics after the cut, a small step toward recovery, removes it at 35 % at a cost on the thin ResNets, and costs no pooled ground at 60 % and 45 % (§378, §381). On the deeper training range the same reward brings the agent back to the prior at 25 % and removes the thin-ResNet cost at 35 %, although one seed still collapses at 25 % (§383). Averaging the three seeds' scores does not remove that collapse: every seed cuts the same groups to the floor (§386). The lesson is the one EagleEye drew for candidate selection [117]: a pruned network's accuracy before recovery is a poor guide to its accuracy after it, and a cheap recovery step restores much of the ranking. SPECTRA adds that the same holds for *training* a policy, and that the damage concentrates at tight budgets. The 32-step reward (T6-fs) ranks plans closer to the final (§379), and it gives the better agent at the tightest budget. At 25 % it reaches +0.16 against the rule, where the batch-norm reward reads −0.17, and the collapse shrinks to −0.93, with no all-or-nothing plan (§387). Reward fidelity and the trained agent's quality move together.

**Transfer: what holds.** One frozen agent per budget type, with no per-target training or search, was trained on ten networks from four families (CIFAR-10 and SVHN) and applied to nine held-out networks from those families, CIFAR-100 among them. Where the literature reuses a pruning policy across networks, it stays within one family or needs a warm-start search [124–131]. Across families, the leave-one-family-out folds are mixed: trained without any ResNet, the agent holds on DepGraph's ResNet-56, though not on the skinny ResNet-56 ×4 (§354); trained without any VGG, it fails on VGG-16. There the sensitivity rule is itself below uniform, and the agent that saw VGGs is below uniform too (§363), so this cell says more about VGG-16 at 60 % than about the missing family. On RepVGG, a family no catalog holds, allocation itself barely matters: the sensitivity rule is level with uniform, so no allocation policy can show anything there (§353, §359); ShuffleNetV2 could not be cut cleanly (§353). That is a limit of the testbed, not evidence for or against transfer.

**Recovery is part of the measurement.** A stronger fine-tune also lifts the unpruned network, so every row reports the honest change beside the raw one, and the final fine-tune keeps its last epoch rather than a checkpoint picked on data (§330). One graphed recipe, G2, serves every family. It matched the per-family recipe on the family where they had differed (mean 10k −0.08 over 16 pairs, §348), at about a fifth of the wall time (§329). Allocation differences shrink after a long fine-tune [11, 109], so a 100-epoch recovery is the conservative setting for an allocation claim.

**Limitations.**
- CIFAR-scale networks only; ImageNet is frozen-transfer coverage, not training (Gilad, 18 Aug).
- Nine held-out networks and three seeds; pooled calls use bootstrap intervals over networks, and one network can carry a pooled result (§374, §380).
- The agent reads the same measured sensitivity as the rules (§3.3, §4.1). It saves no measurement; what it adds is a learned rather than a designed allocation. So far that allocation matches the best design at three of four parameter budgets and passes it at one (45 %, §390).
- Training and evaluation differ in floors, target and rounding (§3.6), and the FLOPs agents are held near the rule by design (§3.4).
- The FLOPs agent sits below the FLOPs rule on most networks (§366).
- The deep-budget failure is removed at 35 % and reduced, not removed, at 25 %, where one seed still collapses (§378, §381, §383). A three-seed ensemble does not remove it (§386). The 32-step reward shrinks it to −0.93, with one seed still at −4.70 (§387).
- One recovery recipe. Allocation rankings may differ under much shorter recoveries.
- Fewer FLOPs are not faster for the small ResNets on a GPU (§395). SPECTRA's budgets are FLOPs and parameters, not latency.
- The rows so far are reproducible only up to run-to-run noise of a few tenths per cell (§393, §397). Pooled calls are robust to this, single-net calls less so. The deterministic fine-tune removes it on every family in the pool (DET-EXACT, §397, §401), and is proposed for the final tables.

---

## 7. Conclusions and future work — DRAFT

SPECTRA carries NEON's idea [1] to structured CNN pruning in a form that learns. A frozen plan-as-action agent, trained offline on ten CIFAR-scale networks, allocates parameter and FLOPs budgets on nine held-out networks across four families and two datasets. It does so with no per-target training or search, and at moderate budgets it performs at the level of the strongest same-loop rule (§5.2, §5.3). The walk-based design that preceded it could not learn, because its reward could not tell its actions apart (Appendix B). Where the plan agent falls short, the cause is measurable: a training reward that ranks plans unlike the final fine-tune at tight budgets (§5.6, §5.7).

**What a submission today would claim [CLAIM, PRELIM].**
1. A frozen allocation policy that transfers to held-out networks and datasets and performs at the level of the best hand rule: at moderate budgets under both budget types, and, with a recovery-aware reward and a deeper training range, at every parameter budget read from 60 % to 25 % (§383, §384). With a reward closer to recovery it beats the best rule at 45 % of parameters, pooled over the nine networks (§390). This is new for CNNs, where reused pruning policies have stayed within one family [124–131].
2. That the agent learns: it beats the uniform and mild rules where they were measured at equal parameters, where every walk agent was a clone (§341, §346, §363).
3. A reward-fidelity account of where a learned allocation fails. A recovery-aware reward removes the failure at 35 %, reduces it at 25 %, and stays at the prior at 60 % and 45 %. A reward closer to recovery reduces it further and gives the better agent at 25 % (§375, §378, §381, §383, §387).
4. A transfer map with its negative cells stated (§363).

It would not claim to beat the best rule, any focused method on its home cell, or transfer to unseen families in general (claims guard).

**Next, in order (registered or planned).**
1. **The params agent of record.** T6-bn is at the prior at all four depths read (§383, §384). T6-fs (32 SGD steps, the cheap reward closest to the one-epoch fine-tune) beats the prior at 45 % pooled and is at it at 60 %, 35 % and 25 % (§387, §390). T8-fs, T6-fs's line on the 18-network catalog with CIFAR-100, is no better than T6-fs at any depth and below it at 25 % (§400). **Decided (Ido, 17 Oct): T6-fs, with T4-F at equal FLOPs.** The final tables are running: five seeds (42–46), every arm re-run with the deterministic fine-tune, the agent and its prior at params 0.6 / 0.45 / 0.35 / 0.25 and FLOPs 0.6 / 0.45 on the nine networks (queue row FINAL). Their five-seed calls replace the three-seed ones, whatever their direction. The remaining weak spot is ResNet-20 ×8 CIFAR-100 at 25 % (−0.93 for T6-fs against the rule). A seed ensemble does not remove it (§386). The one-epoch fine-tune reward (valid; built in v23, default off; about 1–1.4 min per training instance) stays the next reward lever.
2. **The Pareto figure (NEON Fig. 5 analog).** TEST change against params kept and against FLOPs kept, per network, for the agent and every rule at each depth. The FLOPs panel headlines literature-facing rows; the params panel holds the agent's home budget.
3. **Literature rows at matched FLOPs**, with a 10k companion column, the honest column and a different-recipe caption (§5.9). DepGraph's 2.11× and 2.57× points and OCSPruner's row are in (§362, §365, §382), VGG-16 at HRank's and CHIP's points (§391), and ResNet-110 at its two standard points (§392). Add others only where their protocol is stated.
4. **Coverage.** CIFAR-100 networks in the training catalog: neutral on T1's line (§356), CATALOG-HELPS at 25 % on the deep-range batch-norm line (§389), and CATALOG-HURTS at 25 % on the 32-step line (§400), each at a fixed 12,000 instances. The catalog scale-up with held-out splits follows (`docs/CATALOG_SCALE_PLAN.md`), with instances scaled to the catalog's size. Twenty-eight new base networks fill the empty family × dataset cells: WRN-16-4 / 28-2 and PreAct-ResNet-20 / 56 on four datasets, and VGG-BN and MobileNetV2 on SVHN and Fashion-MNIST. Each was saved at its last epoch, never the best-on-test one. All 28 cut cleanly and recover at params 0.25 (§402). Allocation matters on the CIFAR residual networks (lever +0.5 to +2.3 on six of eight) and hardly on SVHN and Fashion-MNIST at that budget (one of twenty), so a held-out dataset split needs its read budget chosen with that in mind. At 12.5 % it is no different (§403): an 8× cut costs at most about 2 points under either rule, and 0 of the 4 named Fashion-MNIST networks carry a lever at 25 % (1 of 4 at 12.5 %). By the registered rule, a Fashion-MNIST split carries no call. The unseen-dataset split moves to CIFAR-100 [D pending Ido], where the lever is large. T6-fs never saw CIFAR-100, so its CIFAR-100 rows are already unseen-dataset rows. Frozen ImageNet transfer for coverage.
5. **Write-up.** Lock the tables, then the thesis chapters. The walk era (Appendix B) becomes a short negative-result section.

**Further work.** Conditioning on a recovered sensitivity (for example, each group's batch-norm-recalibrated loss) rather than the raw half-cut sensitivity. Cross-family catalogs with more families per fold. Budget-type conditioning in one agent (it failed once, §360). A preference sweep over budgets as NEON's product knob.

---

## Appendix A — Protocol archaeology (8–16 Aug) — DRAFT

Numbers in [RESULTS_LEDGER.md](RESULTS_LEDGER.md) §§10–20. Do not promote these to §5 tables except where already LOCKED (C7, floor, Fortify).

**A.1 Inherited bugs.** Until the overhaul, `torch.nn` prune left shapes unchanged, FT was `range(0)`, DDP nested every step, and episode return was the last reward. Later compression numbers are real rebuilds (ledger §11).

**A.2 Fine-tune mode.** Layer-only FT: 0/32 recoveries (proposal default freeze). Full-net 40 ep + rates 1.0/0.9/0.8 made C10 recoverable (ledger §12). Group-aware freeze did not.

**A.3 Eval floor.** Without 0.70, held-out r20-w2 is −13 to −15 pp at 40% params; unconstrained r56-w4 is −42 pp at 17% params (20140546). With the floor, 2-net structural TEST is −3.1 @ 0.60/0.79 (20066579). Overnight **−1.2 pp is not TEST**.

**A.4 Encoder ablation.** Ledger §16. All four encoders ~−24 pp on r56-w4. Catalog diversity is the lever (§17).

**A.5 Mixed C10+C100.** 45–48% train-within −10 was C100, not the DRL recipe (ledger §14). Split datasets.

**A.6 C100 Adam probes.** 20158277: 2/34 val-OK, both ≥98.5% params. 20168590 crop+flip: 1/26 at 0.995 params. Not a 2–5% menu. VGG SGD 160-ep is the first real cut (ledger §7.1).

**A.7 Code SHA / catalogs.** Night git `e985d5e`. Train: `database_offline_train.json` (10) then `database_offline_wide.json` (24, running). Pool: `offline_pools_manifest.json` (287 mapped, not all trained).

**A.8 Proposal vs implementation (one-page).** Kept: global-generic offline DRL, CNN meta-features, Transformer sequence over layers [35], NEON τ, structured rebuild, ResNet/DenseNet/VGG/MobileNet families [3, 4, 76, 78], C10/SVHN/Fashion train. Dropped or postponed: ImageNet/Places365 in the loop; 0.7/0.6 menu; pooling as an action; skip-ratio reward; nested NEON on FC; frozen BERT default; “first any-architecture grouping.”

---

## Appendix B — From the walk agent to the plan agent (Aug–Oct 2026) — DRAFT

The thesis's first agent was NEON's design moved to CNNs: an iterative walk. This appendix records what it showed and why it was replaced, so that the thesis can tell the path honestly. The old text and every walk-era row are kept word for word in [SPECTRA_draft_archive_walk_era.md](SPECTRA_draft_archive_walk_era.md); each number there carries its ledger section.

### B.1 The walk agent

An episode walked a pretrained network's prunable layers, twice. At each layer the agent chose an action from a small menu of keep rates (1.0, 0.9 or 0.8 of the alive width) and, in later versions, a filter-ranking criterion (L1 [18], FPGM [145], BN-scale [106] or SVD). The environment cut the layer's whole channel group, fine-tuned the network and returned NEON's preference-aware trichotomy reward [1] on the cumulative validation change against a user budget τ = 10 points. The last walk agent, v10, replaced it with a fixed-target reward: the per-step change in validation accuracy toward a sampled target. The state was a sequence of per-layer tokens read by a small Transformer; the learner was A2C [151] and later PPO [152] with a critic. A held-out network was walked frozen, and the operating point quoted was the most compressed point whose validation change stayed inside τ.

### B.2 What it showed

- **Transfer inside τ, except skinny-deep ResNets.** The frozen 10-net agent kept similar and unlike CIFAR-10 families inside τ on three seeds. The skinny ResNet-56 at width 4 missed (−15.9 / −16.2 / −17.2 at 70 % params; ledger §3–§6).
- **Catalog diversity helped once.** Going from 3 to 10 training networks moved the hard net; going to 24 reversed it (§17). Encoder capacity (frozen BERT [146], wider or set encoders) did not move it (§16).
- **CIFAR-100 is a recoverability problem.** Under the 40-epoch recipe, VGG-16 BN and ShuffleNetV2 stayed inside τ while the thin ResNets and RepVGG-A0 missed (§21). A spoofed class count did not change that (§55). The root cause was an empty reward band: every real cut on those networks broke τ, so the reward taught "never prune" (§55).
- **Held-out datasets.** Digit-MNIST LeNet [150] gained accuracy on three seeds (§22). SVHN ResNet-20 at a new width stayed inside τ (§23). A frozen CIFAR-10 agent on ImageNet MobileNetV2 is a two-seed probe at unmatched sizes (§41).
- **The provenance correction (4 Sep, §54).** At evaluation the frozen policy had been sampled rather than argmaxed, with encoder dropout live. The "prefer Δparams/ΔFLOPs" arms bypassed the agent altogether, so they are deterministic heuristics and their three seeds are one run. Re-read argmax, the hard net's policy itself was the cliff (−25.2 at 66.7 % params; §54.5).
- **The literature anchor of that era.** On DepGraph's ResNet-56 CIFAR-10 at its 2.11× FLOPs point, the walk's 10k change was −0.46 against DepGraph's +0.24 (§157, "M4"). That caption stands as history; it is not re-quoted for the plan agent.
- **The collapse.** The last walk agent, v10, chose the mild keep-90 % action almost everywhere: its frozen policy tied the mild heuristic (§248, "FLAT"). §331 traced the cause to the reward. At least half of v10's per-step reward variance was read noise, and the action explained about 1 % of it. Over training, the share of 0.9 cuts rose from 0.38 to 0.73, while the sensitivity rule, whose input was in the state, beat mild by about four points on the same return. Every walk-based A/B run over such a uniform or collapsed policy is uninformative rather than failed (§332).

### B.3 What carried over

The plan agent keeps the walk era's infrastructure and its measured facts:
- the channel-group bookkeeping and rebuild (§3.2);
- L1 ranking inside each group [18];
- the finding that allocation is a real lever. On thin ResNet-56 ×4, sensitivity-shaped and residual-preserving allocations beat a uniform one by about a point at equal parameters, and beat mild by about two (§227–§277, §337, §342);
- the recovery work: keep the last epoch of every final fine-tune (§330), and a single fast recipe for every family (G2, §348).

The decisive change is the unit of decision. A walk makes dozens of small, sequential, fine-tuned decisions with a near-flat reward. A plan makes one decision per network, scored on the whole plan.

### B.4 Milestones of the plan agent (8–17 Oct 2026)

| Date | Step | Result | § |
|---|---|---|---|
| 8 Oct | Why the walk agent collapsed | its step reward was at least half read noise; the action explained about 1 % of it | §331 |
| 8 Oct | Decision: the plan-as-action agent | one decision per network, K = 8 plans, shared baseline | §332 |
| 9 Oct | Reward and state chosen by measurement | the raw cut ranks plans validly near the rules; v10's state with the sensitivity columns can imitate the sensitivity plan | §340, §344 |
| 9 Oct | Single-network agent (T0) | LEARNS: +2.35 against mild | §341 |
| 9 Oct | Catalog agent (T1) | LEARNS on held-out networks | §346 |
| 10 Oct | One recovery recipe (G2) | SPEED-EQUIVALENT on the last family | §348 |
| 10 Oct | Priors fixed per budget type | cost-aware rule at equal params; floored FLOPs rule at equal FLOPs | §355, §357 |
| 10 Oct | One agent for both budget types (TM) | FAILS | §360 |
| 10–11 Oct | FLOPs residual agents | T2-F and T3-F below the rule; the trust region (T4-F) helps by +1.07 | §361, §364, §365 |
| 11 Oct | Transfer cells re-read with realizable arms | the no-ResNet fold holds; the no-VGG fold fails | §363 |
| 11–13 Oct | Nine-network pool | at the prior at 60 % and 45 % for both budget types; below it at 35 % params | §366, §370, §371, §373, §374 |
| 12–13 Oct | Headroom screens | none found | §367, §368, §372 |
| 14 Oct | Reward fidelity at 35 % params | one-epoch fine-tune valid at the threshold; cheap rewards weak | §375, §379 |
| 14 Oct | Single-change retrains | only the batch-norm reward removes the collapse | §376, §377, §378 |
| 14 Oct | The 25 % params point | below the prior; the collapse reaches −16.84 on one network, +0.30 on the other eight | §380 |
| 14 Oct | The batch-norm agent at 60 % and 45 % params | at the prior at both, and the same as the catalog agent | §381 |
| 14 Oct | The literature cell at DepGraph's 2.57× point | the FLOPs agent below the rule; the batch-norm agent's one-network BEATS-PRIOR fired, on the TEST half only | §382 |
| 14 Oct | T6-bn: the batch-norm reward on the deep range | at the prior at 35 % and 25 %; the reward helps; one seed still collapses at 25 % | §383 |
| 15 Oct | The deep-range batch-norm agent at 60 % and 45 % | at the prior at both; with §383, at the prior at all four depths | §384 |
| 15 Oct | The deep-range batch-norm agent at DepGraph's 2.57× params point | at the rule on that network; the shallower-range agent's one-network beat does not carry over | §385 |
| 14–15 Oct | T6-fs: the 32-step reward on the deep range | at the prior at 60 %, 35 % and 25 %; beats it at 45 % pooled, the first pooled beat | §387, §390 |
| 15 Oct | T7-C: CIFAR-100 networks in the batch-norm agent's catalog | at the prior at 35 % and 25 %; CATALOG-HELPS at 25 % | §389 |
| 16–17 Oct | T8-fs: the 32-step reward and the 18-network catalog together | the levers do not add up; below T6-fs at 25 %, carried by CIFAR-100 | §400 |

---

## References

Proposal numbering **[1]–[78]** unchanged. Survey additions **[79]–[96]** (August 2026). October 2026 scans **[97]–[152]** (`LIT_ALLOCATION_PROXIES_13OCT.md`, `LIT_SCAN_9OCT_TRANSFER_BUDGET.md`; bibliographic lines verified 14 Oct; [97], [122] and [124] have open venue checks). Venue updates confirmed on 14 Oct for arXiv-only entries [10], [21], [27], [36], [41], [42], [50], [61], [65] are pending a pass.

[1] Hirsch, L., & Katz, G. (2022). Multi-objective pruning of dense neural networks using deep reinforcement learning. *Information Sciences*, 610, 381–400. https://doi.org/10.1016/j.ins.2022.07.134

[2] O’Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. arXiv:1511.08458.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

[4] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR*, 4700–4708.

[5] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

[6] Reed, R. (1993). Pruning algorithms — a survey. *IEEE Transactions on Neural Networks*, 4(5), 740–747.

[7] Anwar, S., Hwang, K., & Sung, W. (2017). Structured pruning of deep convolutional neural networks. *ACM JETC*, 13(3), 1–18.

[8] LeCun, Y., Denker, J., & Solla, S. (1989). Optimal brain damage. *NeurIPS*, 2.

[9] Hassibi, B., Stork, D. G., & Wolff, G. J. (1993). Optimal brain surgeon and general network pruning. *IEEE ICNN*, 293–299.

[10] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2016). Pruning convolutional neural networks for resource efficient inference. arXiv:1611.06440.

[11] Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). Rethinking the value of network pruning. *ICLR*. arXiv:1810.05270.

[12] Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: Towards any structural pruning. *CVPR*, 16091–16101. arXiv:2301.12900.

[13] Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. *NeurIPS*, 28.

[14] Chen, X., Zhu, J., Jiang, J., & Tsui, C. Y. (2020). Tight compression: compressing CNN model tightly through unstructured pruning and simulated annealing based permutation. *DAC*.

[15] Liao, Z., Quétu, V., Nguyen, V. T., & Tartaglione, E. (2023). Can unstructured pruning reduce the depth in deep neural networks? *ICCV Workshops*, 1402–1406.

[16] Yang, Z., & Zhang, H. (2021). Comparative analysis of structured pruning and unstructured pruning. *International Conference on Frontier Computing*, 882–889.

[17] Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2015). Compression of deep convolutional neural networks for fast and low power mobile applications. arXiv:1511.06530.

[18] Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient ConvNets. *ICLR*.

[19] Ding, X., Ding, G., Han, J., & Tang, S. (2018). Auto-balanced filter pruning for efficient convolutional neural networks. *AAAI*, 32(1).

[20] Lin, S., Ji, R., Li, Y., Wu, Y., Huang, F., & Zhang, B. (2018). Accelerating convolutional networks via global & dynamic filter pruning. *IJCAI*.

[21] Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv:1803.03635.

[22] Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance estimation for neural network pruning. *CVPR*, 11264–11272.

[23] You, Z., Yan, K., Ye, J., Ma, M., & Wang, P. (2019). Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks. *NeurIPS*, 32.

[24] Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2021). Layer-adaptive sparsity for the magnitude-based pruning. *ICLR*. arXiv:2010.07611.

[25] Tang, Y., Wang, Y., Xu, Y., Deng, Y., Xu, C., Tao, D., & Xu, C. (2021). Manifold regularized dynamic network pruning. *CVPR*, 5018–5028.

[26] Tofigh, S., Ahmad, M. O., & Swamy, M. N. S. (2022). A low-complexity modified ThiNet algorithm for pruning convolutional neural networks. *IEEE Signal Processing Letters*, 29, 1012–1016.

[27] Park, S., Lee, J., Mo, S., & Shin, J. (2020). Lookahead: A far-sighted alternative of magnitude-based pruning. arXiv:2002.04809.

[28] Balderas, L., Lastra, M., & Benítez, J. M. (2023). Optimizing convolutional neural network architecture. arXiv:2401.01361.

[29] Li, G., Wang, J., Shen, H. W., Chen, K., Shan, G., & Lu, Z. (2020). CNNPruner: Pruning convolutional neural networks with visual analytics. *IEEE TVCG*, 27(2), 1364–1373.

[30] Fernandes Jr, F. E., & Yen, G. G. (2021). Pruning deep convolutional neural networks architectures with evolution strategy. *Information Sciences*, 552, 29–47.

[31] Ferreira, G. B., de Barros, A., Ibrahim, I., & Silva, R. (2023). Surrogate-based constrained multi-objective optimization for the compression of CNNs. *ENIAC*.

[32] Chang, J., Lu, Y., Xue, P., Xu, Y., & Wei, Z. (2022). Automatic channel pruning via clustering and swarm intelligence optimization for CNN. *Applied Intelligence*, 52(15), 17751–17771.

[33] Yu, S., Mazaheri, A., & Jannesari, A. (2021). Auto graph encoder-decoder for neural network pruning. *ICCV*, 6362–6372.

[34] Amelio, A., Bonifazi, G., Cauteruccio, F., Corradini, E., Marchetti, M., Ursino, D., & Virgili, L. (2023). Representation and compression of Residual Neural Networks through a multilayer network based approach. *Expert Systems with Applications*, 215, 119391.

[35] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*, 30.

[36] Xu, D., Yen, I. E., Zhao, J., & Xiao, Z. (2021). Rethinking network pruning — under the pre-train and fine-tune paradigm. arXiv:2104.08682.

[37] Yang, H., Liang, Y., Liu, W., & Meng, F. (2023). Filter pruning via attention consistency on feature maps. *Applied Sciences*, 13(3), 1964.

[38] Sun, H., Zhang, S., Tian, X., & Zou, Y. (2024). Pruning DETR: efficient end-to-end object detection with sparse structured pruning. *Signal, Image and Video Processing*, 18(1), 129–135.

[39] Wang, Y., Guo, S., Guo, J., Zhang, J., Zhang, W., Yan, C., & Zhang, Y. (2024). Towards performance-maximizing neural network pruning via global channel attention. *Neural Networks*, 171, 104–113.

[40] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR*, 11976–11986.

[41] Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. arXiv:2306.11695.

[42] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … & Houlsby, N. (2020). An image is worth 16×16 words: Transformers for image recognition at scale. arXiv:2010.11929.

[43] Kuznedelev, D., Kurtic, E., Frantar, E., & Alistarh, D. (2023). CAP: Correlation-aware pruning for highly-accurate sparse vision models. *NeurIPS*, 36, 28805–28831.

[44] He, H., Cai, J., Liu, J., Pan, Z., Zhang, J., Tao, D., & Zhuang, B. (2024). Pruning self-attentions into convolutional layers in single path. *IEEE TPAMI*.

[45] Guo, M. H., Lu, C. Z., Liu, Z. N., Cheng, M. M., & Hu, S. M. (2023). Visual attention network. *Computational Visual Media*, 9(4), 733–752.

[46] Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). Reinforcement learning: A survey. *JAIR*, 4, 237–285.

[47] Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *IJRR*, 32(11), 1238–1274.

[48] Kaiser, Ł., Babaeizadeh, M., Miłoś, P., Osiński, B., Campbell, R. H., Czechowski, K., … & Michalewski, H. (2020). Model-based reinforcement learning for Atari. *ICLR*.

[49] Mammeri, Z. (2019). Reinforcement learning based routing in networks: Review and classification of approaches. *IEEE Access*, 7, 55916–55950.

[50] Li, J., Monroe, W., Ritter, A., Galley, M., Gao, J., & Jurafsky, D. (2016). Deep reinforcement learning for dialogue generation. arXiv:1606.01541.

[51] Wiering, M. A., & Van Otterlo, M. (2012). Reinforcement learning. *Adaptation, Learning, and Optimization*, 12.

[52] Arulkumaran, K., Deisenroth, M. P., Brundage, M., & Bharath, A. A. (2017). Deep reinforcement learning: A brief survey. *IEEE Signal Processing Magazine*, 34(6), 26–38.

[53] Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.

[54] Zoph, B., & Le, Q. V. (2016). Neural architecture search with reinforcement learning. arXiv:1611.01578.

[55] Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. *CVPR*, 8697–8710.

[56] Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). MnasNet: Platform-aware neural architecture search for mobile. *CVPR*, 2820–2828.

[57] Wang, Z., & Li, C. (2022). Channel pruning via lookahead search guided reinforcement learning. *WACV*, 2029–2040.

[58] Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). Efficient neural architecture search via parameters sharing. *ICML*, 4095–4104.

[59] Yang, Z., Wang, Y., Chen, X., Shi, B., Xu, C., Xu, C., … & Xu, C. (2020). CARS: Continuous evolution for efficient neural architecture search. *CVPR*, 1829–1838.

[60] Liu, C., Zoph, B., Neumann, M., Shlens, J., Hua, W., Li, L. J., … & Murphy, K. (2018). Progressive neural architecture search. *ECCV*, 19–34.

[61] Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable architecture search. arXiv:1806.09055.

[62] Lopes, V., Carlucci, F. M., Esperança, P. M., Singh, M., Yang, A., Gabillon, V., … & Wang, J. (2023). MANAS: Multi-agent neural architecture search. *Machine Learning*.

[63] Dai, X., Chen, D., Liu, M., Chen, Y., & Yuan, L. (2020). DA-NAS: Data adapted pruning for efficient neural architecture search. *ECCV*, 584–600.

[64] Ding, Y., Wu, Y., Huang, C., Tang, S., Wu, F., Yang, Y., … & Zhuang, Y. (2022). NAP: Neural architecture search with pruning. *Neurocomputing*, 477, 85–95.

[65] Li, Y., Zhao, P., Yuan, G., Lin, X., Wang, Y., & Chen, X. (2022). Pruning-as-search: Efficient neural architecture search via channel pruning and structural reparameterization. arXiv:2206.01198.

[66] Dong, X., & Yang, Y. (2019). Network pruning via transformable architecture search. *NeurIPS*, 32.

[67] Wei, X., Zhang, N., Liu, W., & Chen, H. (2022). NAS-based CNN channel pruning for remote sensing scene classification. *IEEE GRSL*, 19, 1–5.

[68] Lee, S., & Song, B. C. (2023). Fast filter pruning via coarse-to-fine neural architecture search and contrastive knowledge transfer. *IEEE TNNLS*.

[69] Tanaka, H., Kunin, D., Yamins, D. L., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow. *NeurIPS*, 33, 6377–6389.

[70] Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021). Neural architecture search without training. *ICML*, 7588–7598.

[71] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*, 248–255.

[72] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.

[73] Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

[74] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading digits in natural images with unsupervised feature learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*.

[75] Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2017). Places: A 10 million image database for scene recognition. *IEEE TPAMI*, 40(6), 1452–1464.

[76] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.

[77] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … & Rabinovich, A. (2015). Going deeper with convolutions. *CVPR*, 1–9.

[78] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., … & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

[79] He, Y., Lin, J., Liu, Z., Wang, H., Li, L. J., & Han, S. (2018). AMC: AutoML for model compression and acceleration on mobile devices. *ECCV*, 784–800. arXiv:1802.03494.

[80] Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K. T., & Sun, J. (2019). MetaPruning: Meta learning for automatic neural network channel pruning. *ICCV*, 3295–3304.

[81] Wang, X., Rachwan, J., Günnemann, S., & Charpentier, B. (2024). Structurally Prune Anything: Any architecture, any framework, any time. arXiv:2403.18955.

[82] Wu, X., Gao, S., Zhang, Z., Li, Z., Bao, R., Zhang, Y., Wang, X., & Huang, H. (2024). Auto-Train-Once: Controller network guided automatic network pruning from scratch. *CVPR*. https://doi.org/10.1109/cvpr52733.2024.01530

[83] Chen, T., Qu, X., Aponte, D., Banbury, C., Ko, J., Ding, T., Ma, Y., Lyapunov, V., Zharkov, I., & Liang, L. (2024). HESSO: Towards automatic efficient and user friendly any neural network training and pruning. arXiv:2409.09085.

[84] Tang, M., Liu, N., Yang, T., Fang, H., Lin, Q., Tan, Y., Chen, X., Liu, D., Zhong, K., & Ren, A. (2024). FreePrune: An automatic pruning framework across various granularities based on training-free evaluation. *IEEE TCAD*, 43(11), 4033–4044. https://doi.org/10.1109/tcad.2024.3443694

[85] Hu, Y., Chen, Y., Zou, X., & Liu, Y. (2025). Automatic channel pruning by neural network based on improved poplar optimisation. *Knowledge-Based Systems*, 310, 113002. https://doi.org/10.1016/j.knosys.2025.113002

[86] Palakonda, V., Tursunboev, J., Kang, J. M., & Moon, S. (2025). Metaheuristics for pruning convolutional neural networks: A comparative study. *Expert Systems with Applications*, 268, 126326. https://doi.org/10.1016/j.eswa.2024.126326

[87] Chen, S., & Zhao, Y. (2025). MLPruner: Pruning convolutional neural networks with automatic mask learning. *PeerJ Computer Science*, 11, e3132. https://doi.org/10.7717/peerj-cs.3132

[88] Chahbouni, A., El Manaa, K., Abouch, Y., El Manaa, I., Bossoufi, B., El Ghzaoui, M., & El Alami, R. (2025). Attention-guided differentiable channel pruning for efficient deep networks. *Machine Learning and Knowledge Extraction*, 7(4), 110. https://doi.org/10.3390/make7040110

[89] Pham, V. T., Zniyed, Y., & Nguyen, T. P. (2025). Singular values-driven automated filter pruning. *Neural Networks*, 192, 107857. https://doi.org/10.1016/j.neunet.2025.107857

[90] Samarin, A., Nazarenko, A., Kotenko, E., Toropov, A., Savelev, A., Motyko, A., & Malykh, V. (2026). Flow-guided neural pruning: Signal-flow framework for multi-architecture model compression. *Machine Learning and Knowledge Extraction*, 8(8), 236. https://doi.org/10.3390/make8080236

[91] Zhou, G., & Zhang, D. (2026). A dependency-aware global spectral-entropy framework for structured neural network pruning. *Applied Soft Computing*, 116186. https://doi.org/10.1016/j.asoc.2026.116186

[92] Fang, Y.-C., Li, W.-Z., Zeng, Y., Lu, Q.-N., & Lu, S.-L. (2025). Pushing to the limit: An attention-based dual-prune approach for highly-compacted CNN filter pruning. *Journal of Computer Science and Technology*, 40(3), 805–820. https://doi.org/10.1007/s11390-024-3536-3

[93] Ghimire, D., Kil, D., Jeong, S., Park, J., & Kim, S.-h. (2026). One-cycle structured pruning via stability-driven subnetwork search. *WACV*. arXiv:2501.13439. https://github.com/ghimiredhikura/OCSPruner

[94] Liu, Z., Cao, Y., Yu, Y., Qi, H., & Gui, J. (2025). Structure-aware automatic channel pruning by searching with graph embedding. arXiv:2506.11469.

[95] Li, X., & Xiu, X. (2025). GoPrune: Accelerated structured pruning with ℓ2,p-norm optimization. arXiv:2511.22120. https://github.com/xianchaoxiu/GoPrune

[96] Liu, X., Li, M., Li, X., Qu, L., Wang, G., Peng, Z., Song, Y., Liu, Z., Jiang, L., & Li, J. (2025). Enhanced structured lasso pruning with class-wise information. arXiv:2502.09125.

[97] Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 REINFORCE samples, get a baseline for free! *ICLR 2019 Workshop: Deep Reinforcement Learning Meets Structured Prediction*. https://openreview.net/forum?id=r1lgTGL5DE

[98] Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *NeurIPS*, 33, 21188–21198. arXiv:2010.16011.

[99] Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J., Pietquin, O., Üstün, A., & Hooker, S. (2024). Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. *ACL (Volume 1: Long Papers)*, 12248–12267. https://doi.org/10.18653/v1/2024.acl-long.662

[100] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.

[101] Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. *Journal of Machine Learning Research*, 5, 1471–1530.

[102] Langford, J., & Zhang, T. (2007). The epoch-greedy algorithm for multi-armed bandits with side information. *NeurIPS*, 20, 817–824.

[103] Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW*, 661–670. https://doi.org/10.1145/1772690.1772758

[104] Yuan, X., Li, S., Wei, J., Zhu, C., Wu, Y., Li, Q., Lv, J., Lan, X., Chen, J., & Liu, Y. (2025). FastForward Pruning: Efficient LLM pruning via single-step reinforcement learning. arXiv:2511.18977.

[105] Bai, L., Tabia, H., & Santos-Rodriguez, R. (2026). HiPP-Prune: Hierarchical preference-conditioned structured pruning for vision-language models. arXiv:2603.06270.

[106] Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., & Zhang, C. (2017). Learning efficient convolutional networks through network slimming. *ICCV*, 2755–2763. https://doi.org/10.1109/ICCV.2017.298

[107] Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). Rigging the lottery: Making all tickets winners. *ICML*, 2943–2952. arXiv:1911.11134.

[108] Li, Y., Adamczewski, K., Li, W., Gu, S., Timofte, R., & Van Gool, L. (2022). Revisiting random channel pruning for neural network compression. *CVPR*, 191–201. https://doi.org/10.1109/CVPR52688.2022.00029

[109] Le, D. H., & Hua, B.-S. (2021). Network pruning that matters: A case study on retraining variants. *ICLR*. arXiv:2105.03193.

[110] Wang, Y., Zhang, X., Xie, L., Zhou, J., Su, H., Zhang, B., & Hu, X. (2020). Pruning from scratch. *AAAI*, 34(07), 12273–12280. https://doi.org/10.1609/aaai.v34i07.6910

[111] Lee, S., & Song, B. C. (2022). Ensemble knowledge guided sub-network search and fine-tuning for filter pruning. *ECCV*, 569–585. https://doi.org/10.1007/978-3-031-20083-0_34

[112] Yu, J., & Huang, T. (2019). AutoSlim: Towards one-shot architecture search for channel numbers. arXiv:1903.11728.

[113] Tiwari, R., Bamba, U., Chavan, A., & Gupta, D. K. (2021). ChipNet: Budget-aware pruning with Heaviside continuous approximations. *ICLR*. arXiv:2102.07156.

[114] Shen, M., Yin, H., Molchanov, P., Mao, L., Liu, J., & Alvarez, J. M. (2022). Structural pruning via latency-saliency knapsack. *NeurIPS*, 35, 12894–12908. arXiv:2210.06659.

[115] Wu, Y.-C., Liu, C.-T., Chen, B.-Y., & Chien, S.-Y. (2020). Constraint-aware importance estimation for global filter pruning under multiple resource constraints. *CVPR Workshops*, 2935–2943. https://doi.org/10.1109/CVPRW50498.2020.00351

[116] Yang, T.-J., Howard, A., Chen, B., Zhang, X., Go, A., Sandler, M., Sze, V., & Adam, H. (2018). NetAdapt: Platform-aware neural network adaptation for mobile applications. *ECCV*, 289–304. https://doi.org/10.1007/978-3-030-01249-6_18

[117] Li, B., Wu, B., Su, J., & Wang, G. (2020). EagleEye: Fast sub-net evaluation for efficient neural network pruning. *ECCV*, 639–654. https://doi.org/10.1007/978-3-030-58536-5_38. arXiv:2007.02491.

[118] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456.

[119] Yu, J., & Huang, T. (2019). Universally slimmable networks and improved training techniques. *ICCV*, 1803–1811. https://doi.org/10.1109/ICCV.2019.00189

[120] Chin, T.-W., Ding, R., Zhang, C., & Marculescu, D. (2020). Towards efficient model compression via learned global ranking. *CVPR*, 1515–1525. https://doi.org/10.1109/CVPR42600.2020.00159

[121] Abdelfattah, M. S., Mehrotra, A., Dudziak, Ł., & Lane, N. D. (2021). Zero-cost proxies for lightweight NAS. *ICLR*. arXiv:2101.08134.

[122] Laurent, C., Ballas, C., George, T., Ballas, N., & Vincent, P. (2020). Revisiting loss modelling for unstructured pruning. arXiv:2006.12279.

[123] Zhou, D., Zhou, X., Zhang, W., Loy, C. C., Yi, S., Zhang, X., & Ouyang, W. (2020). EcoNAS: Finding proxies for economical neural architecture search. *CVPR*, 11393–11401. https://doi.org/10.1109/CVPR42600.2020.01141

[124] Liu, Y., Wang, X., & Zhang, M. (2025). Meta pruning via graph metanetworks: A universal meta learning framework for network pruning. arXiv:2506.12041.

[125] Yu, S., Mazaheri, A., & Jannesari, A. (2022). Topology-aware network pruning using multi-stage graph embedding and reinforcement learning. *ICML*, 25656–25667. arXiv:2102.03214.

[126] Ashok, A., Rhinehart, N., Beainy, F., & Kitani, K. M. (2018). N2N learning: Network to network compression via policy gradient reinforcement learning. *ICLR*. arXiv:1709.06030.

[127] Mu, J., Wang, M., Zhu, F., Yang, J., Lin, W., & Zhang, W. (2024). Boosting the convergence of reinforcement learning-based auto-pruning using historical data. *IEEE TCAD*, 43(2), 548–561. https://doi.org/10.1109/TCAD.2023.3319594

[128] Venkatesan, R., Swaminathan, G., Zhou, X., & Luo, A. (2020). Out-of-the-box channel pruned networks. arXiv:2004.14584.

[129] Gonzalez-Sahagun, G., Conant-Pablos, S. E., Ortiz-Bayliss, J. C., & Cruz-Duarte, J. M. (2024). A generalist reinforcement learning agent for compressing convolutional neural networks. *IEEE Access*, 12, 51100–51114. https://doi.org/10.1109/ACCESS.2024.3385857

[130] Gonzalez-Sahagun, G., Conant-Pablos, S. E., Ortiz-Bayliss, J. C., & Cruz-Duarte, J. M. (2024). A generalist reinforcement learning agent for compressing multiple convolutional networks using singular value decomposition. *IEEE Access*, 12, 136131–136147. https://doi.org/10.1109/ACCESS.2024.3457863

[131] Chin, T.-W., Marculescu, D., & Morcos, A. S. (2021). Width transfer: On the (in)variance of width optimization. *CVPR Workshops*, 2984–2993. https://doi.org/10.1109/CVPRW53098.2021.00334

[132] Li, L., Dong, P., Tang, Z., Liu, X., Wang, Q., Luo, W., Xue, W., Liu, Q., Chu, X., & Guo, Y. (2024). Discovering sparsity allocation for layer-wise pruning of large language models. *NeurIPS*, 37, 141292–141317. https://doi.org/10.52202/079017-4487

[133] Gautam, A. S., & Jha, S. (2026). RAMP: Reinforcement adaptive mixed precision quantization for efficient on device LLM inference. arXiv:2603.17891.

[134] Guo, B., Chang, X., Chao, F., Zheng, X., Lin, C.-M., Chen, Y., Shang, C., & Shen, Q. (2024). ARLP: Automatic multi-agent transformer reinforcement learning pruner for one-shot neural network pruning. *Knowledge-Based Systems*, 300, 112122. https://doi.org/10.1016/j.knosys.2024.112122

[135] Balemans, D., Huybrechts, T., Steckel, J., & Mercelis, S. (2025). Resource-aware neural network pruning using graph-based reinforcement learning. arXiv:2509.10526.

[136] Esmat, S., Banisharif, M., & Jannesari, A. (2026). AgenticPruner: MAC-constrained neural network compression via LLM-driven strategy search. arXiv:2601.12272.

[137] Liu, Y., Guo, Y., Guo, J., Jiang, L., & Chen, J. (2021). Conditional automated channel pruning for deep neural networks. *IEEE Signal Processing Letters*, 28, 1275–1279. https://doi.org/10.1109/LSP.2021.3088323

[138] Sukthanker, R. S., Zela, A., Staffler, B., Dooley, S., Grabocka, J., & Hutter, F. (2025). Multi-objective differentiable neural architecture search. *ICLR*. arXiv:2402.18213.

[139] Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. *ICLR*. arXiv:1908.09791.

[140] Lemaire, C., Achkar, A., & Jodoin, P.-M. (2019). Structured pruning of neural networks with budget-aware regularization. *CVPR*, 9100–9108. https://doi.org/10.1109/CVPR.2019.00932

[141] Cai, R., Muralidharan, S., Heinrich, G., Yin, H., Wang, Z., Kautz, J., & Molchanov, P. (2024). Flextron: Many-in-one flexible large language model. *ICML*, 5298–5311. arXiv:2406.10260.

[142] Sengupta, A., Chaudhary, S., & Chakraborty, T. (2025). You only prune once: Designing calibration-free model compression with policy learning. *ICLR*. arXiv:2501.15296.

[143] Ma, Z., Cui, Y., & Qin, Y. (2025). Beyond manually designed pruning policies with second-level performance prediction: A pruning framework for LLMs. arXiv:2508.02381.

[144] Chen, Y., Duan, J., Zhang, R., Liang, M., Wei, X., Pu, Y., … & Liu, X. (2026). End-to-end dynamic sparsity for resource-adaptive LLM inference. arXiv:2606.27743.

[145] He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019). Filter pruning via geometric median for deep convolutional neural networks acceleration. *CVPR*, 4335–4344. https://doi.org/10.1109/CVPR.2019.00447

[146] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186. https://doi.org/10.18653/v1/N19-1423

[147] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474

[148] Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical guidelines for efficient CNN architecture design. *ECCV*, 122–138. https://doi.org/10.1007/978-3-030-01264-9_8

[149] Ding, X., Zhang, X., Ma, N., Han, J., Ding, G., & Sun, J. (2021). RepVGG: Making VGG-style ConvNets great again. *CVPR*, 13728–13737. https://doi.org/10.1109/CVPR46437.2021.01352

[150] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. https://doi.org/10.1109/5.726791

[151] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *ICML*, 1928–1937.

[152] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
