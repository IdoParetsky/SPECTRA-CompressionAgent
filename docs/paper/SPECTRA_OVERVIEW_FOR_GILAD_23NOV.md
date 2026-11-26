# SPECTRA: a frozen deep-RL allocator for structured CNN pruning. Method, evidence and open problems

Ido Paretsky (MSc candidate), advisor Dr. Gilad Katz, Ben-Gurion University of the Negev.
Overview written on 23 November 2026 for the advisor; updated on 26 November with the results of 24–26 November (Section 7.5b, Section 8.8 and evidence tags E54–E62). **Every result below is preliminary**: no table is locked yet.

**How to read this document.** It is written as a self-contained paper for a reader who knows network pruning and deep reinforcement learning (RL) but not this project. Terms are defined where they first appear, and Appendix B collects them. Every number carries an evidence tag such as [E14]. Appendix A maps each tag to the numbered section of the project's results ledger (`docs/paper/RESULTS_LEDGER.md`) or to a named table file, so every figure can be checked. Accuracy results are quoted on a held-out TEST split (Section 3.6), with the validation, full-test-set and "honest" readings beside them where space allows. Every comparison is made at equal FLOPs or at equal parameters; where the two sides differ by more than 10 % on the other axis, we say so. Statements marked [H] are our interpretation, not a measurement. Published methods' numbers come from their own papers and recipes; they are set beside ours for orientation and are never a contest.

---

## Abstract

Structured pruning removes whole channels from a convolutional network (CNN), so the pruned network is genuinely smaller and cheaper. Once channels are ranked inside each layer, the decision that remains is the *allocation*: how much of each layer to keep under a global budget of parameters or FLOPs. Today allocation is solved one network at a time, by a search, by a reinforcement-learning controller trained for that network, or by a hand rule such as layer sensitivity. NEON (Hirsch & Katz, 2022) showed that a single deep-RL agent, trained offline on many dense networks and then frozen, can prune networks and datasets it has never seen. SPECTRA asks the same question for structured CNN pruning: can one frozen policy, trained once on a catalog of pretrained CNNs, allocate a budget on unseen CNNs at least as well as the best training-free rule run in the same pipeline, with no search and no training on the target?

SPECTRA treats allocation as a contextual bandit. The context is a network and a budget; the action is a whole plan, one keep fraction per coupled channel group; the reward is the change in validation accuracy of the cut network, read as cut or after a cheap proxy of recovery. A Transformer reads the network as a sequence of per-layer tokens and scores every channel group. A decoder turns the scores into integer widths that meet the budget exactly. The policy is trained with REINFORCE, using the mean reward of eight plans sampled for the same network as the baseline. Every plan, the agent's and every rule's, is cut once and recovered by the same 100-epoch fine-tune.

The findings, stated candidly:
1. **The agent learns.** The earlier step-by-step agents collapsed to "keep 90 % everywhere". The plan-level agent beats that heuristic by more than two points on a held-out network, at equal parameters, while keeping 1.5–1.8× its FLOPs [E3, E5].
2. **It transfers at the level of the best rule fixed in advance.** Frozen, it is level with the reference rules on nine held-out networks at every budget inside its training range, at five seeds and under both budget types: the parameter agent at equal parameters (where it often keeps more FLOPs than the rule, up to 1.20–1.77× on two networks at 25 %), the FLOPs agent at equal FLOPs [E13, E14]. At ten CIFAR-10 literature points, five of which carry a published result, it is level at nine and falls just below the cost-aware rule at 8× fewer parameters on ResNet-56 (−0.30) [E21]. On CIFAR-100 and Tiny-ImageNet, which it never trained on, the FLOPs agent is level at four of six points, above at one (where the even cut beats both) and below at one [E25]. On ImageNet ResNet-50 it lands within 0.2 points of the rule at both points (one seed, so no verdict) [E31]. It needs no per-network search. On nine held-out CIFAR-100 networks it is above AMC's per-network RL search run in the same pipeline (+0.97 at 45 % of FLOPs, +6.60 at 20 %), but the threshold rule is about as far above AMC [E61]. At 45 % of parameters kept it also beats a 100-plan per-network random search that uses its own training reward (+0.51; three seeds), at equal parameters while keeping 1.1–1.5× the search's FLOPs [E16].
3. **It does not beat the strongest rule.** No registered "beats" outcome holds in the final tables. The current agent, which learns to blend four rules, ties the current threshold rule, a simple training-free rule, at equal FLOPs [E35]. It puts 67–84 % of its weight on that rule, so the tie is mostly rule selection: it learned which rule suits these networks, which is real learning but not a new way to allocate [E35]. On VGG-19 at 8.9× fewer FLOPs, far below its training range, it is 1.7–2.2 points behind that rule, keeping about 0.55× the rule's parameters [E43]. It is also behind a recalibrated variant of the rule, though that variant's lead comes from keeping more parameters at the same FLOPs [E39, E40].
4. **We can say why** [H on measured evidence]. A long recovery erases most allocation differences at moderate budgets [E10, E21]. The training reward is read before recovery and misranks plans at deep budgets [E11, E12]. Among every plan we have tried, little headroom lies above the best rule at equal size: a 2-epoch screen gains nothing there, and a 100-epoch validation pick, an upper reference that shares its run's luck, reads +0.26 to +0.39 above the threshold rule [E10, E36, E42]. The action parameterizations keep the agent near the rules by design. Changes to the representation tie [E4, E28]. Since 23 November, three learned choices (which rule, which plan under a short recovery, and which size) have each failed to beat a simple rule on families the agent did not train on [E58, E59, E60].
5. **It is cheap.** On ResNet-56 CIFAR-10 a recovered network takes 7.7–8.5 minutes on one GPU with a GPU data pipeline, and 20.9–21.4 minutes with a stock data loader, the like-for-like figure. Eight published pipelines, timed on the same GPU with their own loaders, take 44 minutes to 5.7 hours; a ninth (GNN-RL) was still searching after 17 hours [E26].

We close with open problems: reward fidelity, a richer action space, richer state, preference-aware objectives, sequential decisions with backtracking, and short-recovery regimes.

---

## 1. Introduction

CNNs remain the workhorse of real-time and on-device vision, where compute and memory decide deployment. *Unstructured* pruning zeroes individual weights. It reaches high sparsity on paper but does not speed up dense GPU kernels. *Structured* pruning removes whole filters and channels, so the matrices the hardware multiplies actually shrink. Two decisions remain once the unit is fixed:
- **which** channels to remove inside a layer (the importance criterion, e.g. the L1 norm of each filter; Li et al., 2017);
- **how many** to remove from each layer (the allocation).

Residual additions and concatenations couple layers, so the allocation is made per *coupled channel group*: a set of layers that must keep the same channels (Fang et al., 2023).

**Allocation matters, but not everywhere.** At 25 % of parameters kept, a cost-aware sensitivity allocation beats an even cut by +0.88 points of TEST accuracy, averaged over nine held-out networks at three seeds (validation +1.03, full test set +0.95, honest +0.86), and by 2.1–3.4 points on three of them; at this equal-parameter budget it keeps 1.1–2.4× the even cut's FLOPs on eight of the nine [E12]. At five seeds the pooled gain is +0.96 [−0.03, +1.96] [E14]. On ResNet-56 at 5.5× to 8× compression the sensitivity rules beat the even cut by 1.9–3.3 points: at equal parameters, where the rule keeps 1.7–1.9× the even cut's FLOPs, and at equal FLOPs, where it keeps 14 % fewer parameters [E21, E43]. At the field's standard operating points (about 2–3× on CIFAR-10), however, the same rules beat the even cut by only +0.01 to +0.60 points after a 100-epoch recovery [E21]. Five of these six points are at equal FLOPs, where the rules keep 0.2–1.3× the even cut's parameters (about a fifth on VGG-16), and one is at equal parameters, where the rule keeps 1.15× the even cut's FLOPs [E43]. Controlled studies in the literature find the same: after a full recovery, sensible allocations end within a few tenths of a point of each other (Liu et al., 2019b; Wang et al., 2020; Lee & Song, 2022; Le & Hua, 2021; Li et al., 2022).

**Allocation is solved per network.** Search and RL methods (AMC, He et al., 2018; LeGR, Chin et al., 2020; GNN-RL, Yu et al., 2022) and meta-learned generators (MetaPruning, Liu et al., 2019a) search or train for each target. Global criteria (DepGraph, Fang et al., 2023) re-solve each model. Hand rules (Li et al., 2017) need a human to choose their form. Each new architecture or dataset means another search, or another human decision.

**NEON's idea.** NEON (Hirsch & Katz, 2022) trained one preference-aware deep-RL agent offline on many dense networks and applied it, frozen, to datasets it had never seen. SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) asks whether that idea survives the move to structured CNN pruning, where layers are coupled, where pruning must remove whole channels, and where a long recovery fine-tune follows the cut. The claim we set out to test is narrow. It is not "beat DepGraph on ResNet-56". It is: **a single frozen policy can allocate a parameter or FLOPs budget on unseen CNNs, with no per-target search, at least as well as the best training-free rule run in the same pipeline.**

**What this document reports.**
- A formulation of transferable allocation as a contextual bandit over whole plans, with a Transformer over per-layer tokens and a decoder that meets the budget exactly (Section 5).
- A same-pipeline evaluation protocol that isolates allocation (Section 3). It uses one cut and one fixed recovery for every arm, disjoint validation and test halves, an "honest" change net of the unpruned network's own gain under the same recovery, paired seeds, and decisions registered before any GPU time.
- Results: learning, transfer, literature operating points, harder datasets, a head-to-head table of accuracy, FLOPs and parameter changes beside the published methods, and runtime (Section 6).
- An analysis of why the agent reaches the best rule but does not pass it (Section 7), and a research agenda (Section 8).

---

## 2. Background and related work

**Structured pruning and coupling.** In a residual block the last convolution's output is added to the shortcut, so both must keep the same channel indices. In a DenseNet every layer's output is concatenated onto a growing feature bank. DepGraph (Fang et al., 2023) models these dependencies as a graph and prunes coupled groups. SPA (Wang et al., 2024) makes the same grouping framework-portable. SPECTRA uses a lightweight version of the same bookkeeping as infrastructure and claims no novelty for it (Section 3.2). Its claim is at the level of the *policy*, not the grouping tool.

**Allocation rules.** Li et al. (2017) measure each layer's sensitivity (the accuracy drop when that layer alone is pruned) and prune sensitive layers less. Tools such as NNI and PaddleSlim automate this as one global threshold on per-layer sensitivity curves. LAMP (Lee et al., 2021) rescales magnitudes per layer so that one global threshold allocates sparsity. Cost-aware allocators divide the loss a cut causes by the resource it saves, per layer as in the FLOPs utilization ratio of Chen et al. (2020), or as a knapsack over per-layer damage curves (SPDY; Frantar & Alistarh, 2022). Section 4 gives our versions of these rules.

**Search-based and RL allocators.** AMC (He et al., 2018) trains a DDPG agent per network, layer by layer, against the accuracy of the cut network. LeGR (Chin et al., 2020) evolves per-layer affine transforms of a global ranking. GNN-RL (Yu et al., 2022) embeds the network graph and runs RL per target. MetaPruning (Liu et al., 2019a) trains a weight-generating network per architecture family. Li et al. (2022) showed that the best of many random allocations, picked by a cheap selector, is a strong baseline. Each of these methods searches, or trains, per target. The closest frozen transfer we know for CNNs applies a graph metanetwork trained on ResNet-56 to ResNet-110 and back, within one family (Liu et al., 2025).

**Cheap proxies.** A search needs a fast estimate of post-recovery accuracy. EagleEye (Li et al., 2020) showed that the accuracy of a freshly cut network is a poor guide because the batch-normalisation (BN) statistics are stale, and that re-estimating them raises the rank correlation with post-fine-tune accuracy. Short fine-tunes are the other common proxy. Section 5.7 measures this fidelity for SPECTRA's own plans.

**NEON.** NEON (Hirsch & Katz, 2022) prunes dense (fully connected) networks layer by layer. At each layer an actor-critic agent reads layer and network feature maps (weight and activation statistics, architecture descriptors) and picks a new layer size from {1, 0.9, 0.8, 0.7, 0.6} of the original. The layer is replaced by a new, retrained one. The reward combines the compression achieved, c (the percentage of parameters removed), with the accuracy change relative to a user tolerance C: +c³ when accuracy improves, c when the drop stays within C, and −c³ when it exceeds C. NEON trained on 28 tabular datasets in a five-fold rotation and was tested on held-out datasets. In its Table 3 the configuration NEON 5 averages 24.59× compression at +0.49 % accuracy, against 13.26× at −0.16 % for the strongest compression baseline (this summary is taken from the project's reading notes on the NEON preprint, not re-read here). SPECTRA keeps NEON's central idea: a generic policy, trained offline and frozen. Section 7.1 lists what did not carry over and why.

---

## 3. The pruning pipeline

### 3.1 Networks and datasets

| Set | Networks | Role |
|---|---|---|
| Training catalog of the agents of record (the parameter and FLOPs agents whose results Tables 1–3 report) | 10 pretrained networks from four families. CIFAR-10: thin ResNet-20 at widths 8 and 10, thin ResNet-56 at width 6, ResNet-32, VGG-11-BN, VGG-13-BN, MobileNetV2 ×0.5 and ×1, DenseNet-40. SVHN: VGG-11-BN | Training only |
| Nine held-out networks | CIFAR-10: thin ResNet-56 at widths 4 and 8; ResNet-56 (a public checkpoint, chenyaofo); DepGraph's own ResNet-56; MobileNetV2 ×1.4; DenseNet-100; VGG-16-BN (chenyaofo). CIFAR-100: thin ResNet-20 at width 8; DepGraph's own VGG-19 | Same-pipeline comparisons. No checkpoint appears in any catalog |
| New architectures | Wide-ResNet-16-4, Wide-ResNet-28-2, PreAct-ResNet-20 and PreAct-ResNet-56, each on CIFAR-10 and CIFAR-100 (8 networks) | Transfer test, named before any agent read them |
| Literature networks | ResNet-56 (DepGraph's checkpoint; also OCSPruner's), ResNet-110, VGG-16-BN on CIFAR-10; VGG-19 and ResNet-56 on CIFAR-100; VGG-16-BN and ResNet-18 on Tiny-ImageNet (64 px); ResNet-50 on ImageNet (torchvision weights: 76.12 % top-1 by our own read, 76.13 % as torchvision publishes it) | Published operating points |
| Broad catalog (transfer program) | 45 networks from ten families on CIFAR-10 (26) and CIFAR-100 (19). The FLOPs agents train on 39 of them, because ShuffleNetV2 cannot be cut cleanly | Training of the current agent |
| 13 held-out CIFAR-100 networks | Chosen from two screens of 30 CIFAR-100 networks as networks where some plan beats the reference rule ("room", Section 6.6) | Test of the current agent |

A *thin* ResNet has first-stage width w (w, 2w and 4w channels per stage, against the standard 16, 32 and 64); the "width 4" network has 4/8/16 channels. These narrow networks are where allocation matters most and where the step-by-step agents failed. SVHN and Fashion-MNIST were dropped from catalogs and benchmarks on 5 November. On them allocation barely matters: at 25 % parameters kept, the cost-aware rule beats the even cut on 1 of 20 SVHN and Fashion-MNIST networks, against 6 of 8 CIFAR residual networks (validation, one seed) [E47].

### 3.2 Coupled channel groups

A `torch.fx` trace of the model describes the channel dimension of every tensor as an ordered list of segments:
- a convolution or linear layer opens a new segment;
- normalisation, activation and pooling layers pass segments through;
- a concatenation joins its inputs' segment lists, so each consumer sees each group at a known offset;
- an element-wise addition or multiplication merges its operands' segments into one group;
- a depthwise convolution ties its input and output.

Anything not understood (an unknown module, a tensor escaping to the output, a concatenation along a non-channel axis) marks the affected groups unprunable.

A **coupled channel group** is the resulting set of producers, depthwise convolutions, normalisations and consumers that share one channel dimension. The planned groups are the prunable groups at least two channels wide, ordered by depth. The classifier's output is never planned. Cutting a group rebuilds every producer, every normalisation over that dimension and every consumer's input slice in one edit, and a dummy forward pass verifies the result. If a cut cannot be expressed structurally, the code masks the channels instead and logs it. Any row with a masked edit is excluded from every reported result.

### 3.3 Budgets and cost models

A budget is a type b ∈ {parameters, FLOPs} and a kept fraction κ ∈ (0, 1) of the original. Parameters kept are counted exactly from the shapes of the rebuilt network. FLOPs are multiply-accumulates (MACs) per input sample at the dataset's resolution. Convolution and linear MACs are computed exactly from the widths. Normalisation and activation costs are modelled as linear in each group's width, with one coefficient measured by a real half-width cut of that group; unit tests check exact agreement with the cost of real cuts. Pooling and additions are not counted. Counters that count only convolutions and linear layers, as DepGraph, HRank and CHIP do, give kept fractions about 0.2–2 % (relative) lower for the same cut.

### 3.4 The cut

Inside a group, channels are ranked by L1 norm. Each producer's per-filter L1 norms are divided by their own maximum, so that a 3×3 convolution cannot outvote a 1×1 shortcut, and summed across producers. The top w_g channels survive. A plan, from any source, is realized in one shot through four guards:
1. it is decoded 0.02 below the target;
2. no group goes below a width floor of two channels;
3. the widths are rounded onto the widths the cutting routine can reach exactly, from its five keep rates {1.0, 0.9, 0.8, 0.7, 0.6}, stepping low-priority groups down and high-priority groups up until the plan sits within 0.01 below the target;
4. the cutting routine applies the per-group cuts and narrows the last cut so the network lands at the target.

Every reported arm lands within 0.01 of its target, with no masked edit and no forced extra cut to reach the target. By default the cutting routine never cuts the first convolution of a plain (VGG-style) network; in a ResNet the stem's channels are still cut through the stage-1 group. Published methods usually do cut the first convolution. At the deep VGG points where we set our rows beside theirs, we also ran with the lock lifted and report what the lock costs (Section 6.6).

### 3.5 Recovery

Every arm and the unpruned network (the *origin control*) get the same final fine-tune, the recipe of record:
- SGD, momentum 0.9, weight decay 5·10⁻⁴;
- learning rate 0.1, cosine-annealed to 0 over 100 epochs;
- batch 256, random crop and flip on the GPU, fp32, CUDA graphs;
- gradient norm clipped at 1.0;
- no early stopping, and the **last epoch is kept**.

An earlier default restored the epoch with the lowest training loss. That was epoch 1 on most pruned networks, so those rows measured "cut plus one epoch"; they are never mixed with keep-last rows. One recipe serves every family. It was adopted once it matched the slower per-family recipe on the family where the two had differed: mean change on the full test set −0.08 over 16 pairs [E45]. On ImageNet every arm gets the counterparts' 90-epoch flow (SGD at lr 0.01, divided by 10 at epochs 30 and 60, weight decay 10⁻⁴, batch 256).

### 3.6 Splits and reads

The 10,000-image test split of each CIFAR dataset is divided once, by a fixed permutation (seed 0), into a **validation half** (5,000 images) and a **TEST half** (5,000). Training reads only the validation half, and only on catalog networks. Nothing reads the TEST half before the final read: no selection, no stopping, no reward. For a pruned arm we report four changes, in percentage points:
- **TEST**: TEST-half accuracy after recovery minus the origin's TEST-half accuracy before recovery. This is the headline.
- **val**: the same on the validation half.
- **10k**: the same on all 10,000 images, comparable to published numbers.
- **honest**: TEST accuracy after recovery minus the *origin control's* TEST accuracy after the same recovery. A 100-epoch recovery also lifts the unpruned network (by +0.4 to +1.4 points on CIFAR-10 networks, and the 90-epoch ImageNet flow by +0.78 on ImageNet [E31, E50]), and "honest" removes that lift. For a difference between two pruned arms, honest is the difference of their honest values.

On ImageNet, TEST is the official 50,000-image validation set and val is 10,000 images held out of the training split. Seeds are 42–46 in the final tables and 42–44 elsewhere; the room screens of Section 6.6 used seeds 47–48, and ImageNet seed 42 only. An agent's evaluation seed is its training seed, and every rule runs at the same seed, so contrasts are paired by seed (same halves, same base checkpoint). A network's value is the mean over seeds, and a pooled value is the equal-weight mean over networks. Intervals are a bootstrap over networks × seeds. They are wide when one network dominates, and we always name that network.

### 3.7 Pre-registration and decision rules

Every experiment was registered in a version-controlled queue (`docs/SITTING_GPU_QUEUE.md`) before any GPU time: the question, the comparators, the decision rule with its thresholds, and what may not be claimed. Each registered comparison ends in a *verdict* (above, level or below; in the transfer program gain, level or loss). The main rules are:
- **One network:** agent minus reference rule ≥ +0.3 points is *above*; within ±0.3 is *level*; ≤ −0.3 is *below*. With five seeds, above and below also require the same sign on at least 4 of 5 seeds.
- **Pooled over the nine held-out networks:** *above* needs ≥ +0.3 and a positive difference on at least 6 of 9 networks; ≤ −0.3 is *below*; anything else is *level*.
- **Transfer program (from 15 November):** an exact two-sided Wilcoxon signed-rank test over networks. *Gain* needs a mean ≥ +0.2 and p < 0.05 after a Holm adjustment across the registered comparisons; *loss* is symmetric; *level* otherwise.

A rule fixed in advance serves as the **reference rule** for each budget type (Section 4.3).

### 3.8 Determinism and run-to-run noise

Without deterministic GPU algorithms, three runs of one identical cut spread over 0.42 points (ResNet-56) and 0.76 points (thin ResNet-20 on CIFAR-100) on TEST. Over 54 repeated settings the run-to-run SD is about 0.3 points per run [E19]. With deterministic algorithms on, two runs of the same setting agree to four decimals on every number, on five networks in four families, at 22–50 % more wall time [E19]. Every final table uses that setting.

### 3.9 The sealed final evaluation (registered, not yet run)

The final agent will train only on families that no published CIFAR pruning table tests: MobileNetV1, SqueezeNet, EfficientNet, MobileNetV3, HarDNet, DLA and ConvNeXt, four sizes each on CIFAR-10 and CIFAR-100, all trained by us. Of these 56 networks, 50 passed the accuracy and loading screens, and 49 entered the catalog after a cut-and-recover check [E38]. Fresh, *sealed* checkpoints of the literature's families will then be read once, after the agent is frozen: ResNet-20 to -110, VGG-11 to -19-BN, MobileNetV2 ×0.5 to ×1.4, DenseNet-40, WRN-16-4, WRN-28-2, PreAct-ResNet-20 and GoogLeNet, on CIFAR-10 and CIFAR-100. The reads use a fixed FLOPs grid (0.60, 0.45, 0.30, 0.20), with the registered verdict at 0.45. Published points exist for 18 of the 38 sealed networks [E38].

---

## 4. Training-free allocation rules

### 4.1 The sensitivity measurement

For each planned group g, s_g is the rise in cross-entropy over 4 training batches (batch 256) when that group alone is cut to half width by L1, with no fine-tune. This is a one-point (50 % cut) version of Li et al.'s (2017) per-layer sensitivity analysis, measured per coupled group. Small values are floored at 5 % of the median. The same cut also gives c_g, the parameters (or MACs) that the half cut saves. The measurement costs one forward pass over 4 batches per group. The agent's state carries the same numbers (Section 5.3), so the agent saves no measurement compared with the rules.

### 4.2 The rules

Each rule outputs a relative weight w_g per group. A common decoder then sets keep_g = clip(c · w_g, k_min, 1) and bisects the scalar c so the plan meets the budget (k_min = 0.1, plus the width floor). Every rule runs in the agent's own pipeline, with the same groups, L1 ranking, guards, cut, recovery and seeds.

| Rule | Weight per group | Notes |
|---|---|---|
| Even cut | w_g = 1 | The allocation-free baseline |
| Sensitivity rule | w_g = (s_g / median s)^0.5 | Per-group form of Li et al. (2017); under a FLOPs budget, the *FLOPs rule* |
| Cost-aware rule | w_g = (v_g / median v)^0.5, with v_g = s_g / c_g | Loss per unit of cost saved, as in Chen et al. (2020) and related work, in closed form |
| Current threshold rule | Each group's damage curve is measured at keeps 0.75, 0.5, 0.25 and 0.1 (loss rise and saving). One global loss threshold picks, per group, the smallest keep whose loss stays under it, with the threshold set so the estimated saving meets the budget | Li et al.'s single-threshold rule as NNI and PaddleSlim automate it |
| Keep-streams rule | Even cut, with every residual stream held at full width | A structural heuristic |
| Mild heuristic | Not a plan: keep 90 % of each group the cutting routine may cut, with a fine-tune after each cut, then the same final recovery | The policy every step-by-step agent collapsed to (Section 5.1) |
| Recalibrated threshold rule | The threshold rule, with BN statistics re-estimated (8 batches) on every probe before its loss is read | Section 6.6 |
| LAMP; global magnitude ranking | One-shot structured adaptations, run on our cut and recovery | Screened only (Section 6.6) |

We call the threshold rule as defined above the *current* threshold rule, to set it apart from its recalibrated variant; "the threshold rule" below always means the current one unless "recalibrated" is said.

### 4.3 Reference rules fixed in advance

Each budget type has one reference rule, chosen from pooled evidence before the agents' main reads. It was never chosen per network and never on TEST.
- **Equal parameters: the cost-aware rule.** At equal parameters it beat the plain sensitivity rule on all three networks measured (+0.47, +0.39, +0.44), keeping 1.26–1.47× its FLOPs on the two ResNets [E7].
- **Equal FLOPs: the FLOPs rule with the width floor.** At equal FLOPs the cost weighting hurt (−0.33 on the thin ResNet-56) [E7]. [H] In a CIFAR ResNet every 3×3 convolution costs about the same FLOPs, because each stage halves the feature map and doubles the channels, so a plain sensitivity rule is already cost-aware at equal FLOPs.

The **threshold rule** was added as a comparator later. Over two screens of 30 CIFAR-100 networks it sits above the FLOPs reference rule at equal FLOPs (TEST +0.56, val +0.60) and modestly above the parameter reference rule at equal parameters (TEST +0.24, val +0.21) [E33]. On fresh seeds of 13 held-out networks it beats the FLOPs rule at equal FLOPs (0.45) by +0.64 [+0.26, +1.06] (val +0.90, 10k +0.77; 11 of 13 networks) [E34]. The reference rules stay fixed for registered studies. Since 17 November every agent row is read beside all four rules: the reference rule, the even cut, the cost-aware or plain sensitivity rule, and the threshold rule. "The strongest rule" in this document means the strongest of these same-pipeline rules on the networks read, never a rule chosen on TEST.

---

## 5. The deep-RL allocator

### 5.1 History: step-by-step agents and why they collapsed

The project's first agent carried NEON's design over to CNNs. An episode visited a pretrained network's prunable layers in order, twice. At each layer the agent chose a keep rate from a small set (in the last version: identity, 0.9, 0.8, 0.7, 0.6), and in some versions a filter-ranking criterion. The environment cut the layer's coupled group, fine-tuned the whole network (12 epochs) and returned a reward: first NEON's tolerance-based reward, later the per-step change in validation accuracy toward a sampled target size. The state was the per-layer token sequence of Section 5.3, read by the Transformer of Section 5.4. The learner was A2C, then PPO with a critic (γ = 1, GAE λ = 0.95).

Every version, through the tenth, converged on the mild heuristic. Frozen, the last ones chose the 0.9 keep rate almost everywhere and tied the mild heuristic [E2]. A read-only audit of the tenth version's own training telemetry (207 episodes over 4 days, 14,703 steps, 10 networks) found the cause in the training signal, not in the features [E1]:
- **The per-step reward could not tell the actions apart.** Neighbouring actions differed by about 0.08 points in mean reward, against a per-step SD of about 1 point. Consecutive cut rewards in an episode correlated −0.33, so at least half of the reward variance was transient read noise. The action itself explained about 1 % of the variance.
- **Allocation is a deferred trade.** Sparing a sensitive layer earns exactly 0 now. Its cost appears tens of steps later, as deeper cuts elsewhere, and a critic with an effective horizon of about 20 steps did not carry that comparison. The immediate ordering was carried instead: smaller cuts lose less.
- **The policy followed that ordering.** The share of 0.9 cuts rose from 0.38 to 0.73 over training, and policy entropy fell from 1.47 to about 0.45. On the same return, at 60 % of parameters, the sensitivity rule beat mild by about four points, and its input was already in the agent's state.
- **Samples were scarce.** Only about 20 episodes per network were possible, because every step paid for a fine-tune (3.5–79 minutes per episode).

A supervised probe then showed that the representation was not the problem. Given the sensitivity columns, the default encoder imitates the sensitivity rule's plan on held-out networks with Spearman ρ +0.87 [E4]. A randomly initialised copy of the trained agent's encoder, read by a linear probe, did as well as the trained one (ρ +0.84 against +0.80). So RL training of the step-by-step agent had added nothing beyond random features [E4]. **The fix had to be in the decision unit and the reward.**

### 5.2 Formulation: a contextual bandit over whole plans

A pretrained CNN has G planned channel groups with widths w⁰_g. Given a budget type b and kept fraction κ, an allocation chooses integer widths 1 ≤ w_g ≤ w⁰_g, so that the network's cost under b is κ of the original. SPECTRA treats this as a **contextual bandit**:
- **context**: the network and the budget, (network, b, κ);
- **action**: a whole plan, one real-valued score per group, decoded to widths that meet the budget (Section 5.6);
- **reward**: one number per plan, the validation accuracy change of the cut network, read as cut or after a cheap proxy of recovery (Section 5.7).

There is one decision per network and budget, so credit assignment is exact. Comparing several plans for the same context removes the network's and the budget's difficulty from the gradient, with no critic. No full recovery runs inside the training loop: the reward reads each plan as cut or after a short proxy (a BN re-estimate, 32 SGD steps, or one epoch in one ablation line), and a training run takes hours instead of days. The budget is met by construction, so a plan cannot gain by under-spending.

### 5.3 State: one token per layer

The network is read as one token per leaf module: every convolution, linear layer, normalisation, activation and pooling layer. Each token has 63 columns. The 38 statistics columns (the first four rows below) are z-scored with a standardizer fitted once on a training catalog. The other columns are already bounded and are appended as they are. (`src/BERTInputModeler.py:_build_layer_tokens`, `NetworkFeatureExtraction/src/FeatureExtractors/`, `src/state_dump.py:layout`)

| Columns | Width | Content |
|---|---|---|
| Topology | 7 | Layer type (8 kinds), input and output channels, kernel, stride, padding, groups (for a linear layer: in and out features) |
| Activation statistics | 12 | On a fixed probe batch: mean, SD, skewness, kurtosis, min, quartiles, median, max, L1 and L2 norms, 10th percentile of absolute values |
| Weight statistics | 12 | The same twelve moments, of the layer's weights |
| Filter-norm distribution | 7 | Mean, SD, min, quartiles, median and max of the per-filter L1 norms |
| Position and structure | 4 | Relative depth, stem flag, coupled flag, normalised width |
| Group cost | 4 | The owning group's share of the network's parameters and of its MACs; its producer count relative to the largest group; cuts so far (0 at the origin) |
| Group sensitivity | 2 | log(s_g / median s) / 3 clipped to [−1, 1], and s_g's percentile rank among the groups |
| Budget | 2 | κ, and a constant 1 (the share of the cut still to do, inherited from the step-by-step agent) |
| Budget type | 1 | 0 for parameters, 1 for FLOPs |
| Inherited slots | 12 | Kept fraction, accuracy slack and pass progress, fixed at the origin's values; nine per-action cost slots, zeroed (the tenth carries the budget type above) |

Each token also carries a layer-type embedding and a sinusoidal position. The design keeps the tenth step-by-step agent's state contract unchanged, because the imitation probe showed it was sufficient once the sensitivity columns were present. Without them every encoder variant reached only ρ +0.45 to +0.59, and one fold fell to about 0 [E4]. **A new network therefore needs its sensitivity measurement before the agent can allocate.** That is the per-network cost of transfer, about one forward pass over 4 batches per group (about five for the rule-blending agent, which also measures the threshold rule's damage curves).

### 5.4 Encoder: a Transformer over the tokens

| Component | Setting (`src/Model/StateEncoder.py:SpectraStateEncoder`) |
|---|---|
| Input | Linear(63 → 256) + LayerNorm, plus the layer-type embedding, sinusoidal positions and a learned marker on one token (a remnant of the step-by-step agent, which marked the layer being cut) |
| Body | 3 pre-norm Transformer encoder layers: width 256, 8 heads, feed-forward 1024, GELU, no dropout in the plan agent |
| Coupling bias | One learned scalar, shared by all layers and heads, added to the attention logit of every pair of tokens assigned to the same coupled group. A group's tokens are its producers, depthwise convolutions and normalisations, and its consumers only when they produce no group of their own; activations and pooling layers stand alone. A structural attention bias as used in graph Transformers |
| Output | LayerNorm, then a zero-initialised linear head giving one score per token |
| Group score | μ_g is the mean token score over the layers that produce group g's channels (producers and depthwise convolutions), centred across groups |
| Size | About 2.4 M parameters |

Zero initialisation means the untrained agent proposes a sensible default: the even cut for the direct agent, and the reference rule's plan for the residual and rule-blending agents (Section 5.5). Learning is a departure from that default.

*Code detail.* The state built by the environment also holds five "action-cost" vectors: each keep rate, with the parameter and MAC fractions that cutting the step-by-step agent's current target layer at that rate would remove. `SpectraTokenFront.embed_tokens` appends them as five extra tokens that take part in attention. The plan policy reads only the layer tokens' outputs (`PlanPolicy.__init__` overrides the pooling), so these tokens are inherited context, not outputs.

### 5.5 Three action parameterizations

All three share the encoder and the per-group scores μ. They differ in how a score vector z becomes per-group weights before the decoder. During training z ~ N(μ, σ²I) (Section 5.8); at evaluation z = μ.

**(a) Direct agent** (the parameter-budget agent of record): keep_g = clip(sigmoid(z_g + β), k_min, 1), with β bisected to meet the budget (`plan_agent.decode`).

**(b) Residual agent** (the FLOPs-budget agent of record): a correction on the FLOPs rule's weights w_g (Section 4.2):

keep_g = clip(c · w_g · exp(0.3 · tanh(z_g / 0.3)), k_min, 1), with c bisected (`plan_agent.residual_decode`).

The tanh term is a **trust region**: every group's weight stays within e^{±0.3} of the rule's. With z = 0 the agent reproduces the rule's plan exactly. Rationale, from the record (all at equal FLOPs unless stated): a direct FLOPs agent learned (+0.51 over mild on one network) but stayed below the FLOPs rule (−0.57) [E8]. A residual without the bound was −1.89 below the rule on a held-out network, and −1.54 once each instance's advantages were divided by their spread. The bound then added +1.07 there, to −0.47 [E8]. One agent asked to handle both budget types failed: −0.52 below the catalog agent at equal parameters, and −0.66 below mild at equal FLOPs [E9].

**(c) Rule-blending agent** (the current agent, FLOPs budgets): the agent first blends four rules (the FLOPs rule as anchor, the even cut, the cost-aware rule and the threshold rule), then applies the residual correction (`plan_agent.menu_decode`):

log w_g = Σ_k π_k · log ŵ_{k,g} + 0.3 · tanh(z_g / 0.3),   π = softmax(u).

Here ŵ_k is rule k's weight vector divided by its geometric mean over groups. The four logits u come from a second, network-level head (`PlanPolicy.menu_logits`): a linear map from the mean of all token outputs, the log ratios of parameters kept by each rule's own plan to the anchor's (at the same budget), and κ. That head starts at zero weights with bias (6, 0, 0, 0). In every mean plan (at evaluation and in the trainer's monitors) shares under 0.005 are dropped, so the untrained agent's mean plan is the anchor's plan exactly; sampled plans use the plain softmax, in which the anchor starts at 0.9926. During training u ~ N(ψ, σ_u² I) is sampled with σ_u = 2. Rationale: on held-out networks where better plans exist, an agent trained on a broad catalog but bounded near the FLOPs rule could not reach them. At equal FLOPs it was level with the FLOPs rule (−0.17) and −0.89 below the threshold rule [E34]. Blending lets the agent move to another rule's plan when that plan is better.

### 5.6 Decoder: from scores to an exact budget

All decoders share the bisection to the budget (48 iterations on an analytic cost model) and the integer rounding. A polish step then moves single channels, on the groups rounded furthest the wrong way, while that brings the cost closer to κ. The plan's cost therefore matches the budget to within single-channel rounding. There is no size term in the reward: the decoder fixes the size, and every comparison is at equal size. Similar mechanisms exist elsewhere: bisection to an exact budget (ChipNet; Tiwari et al., 2021), and a single-step policy mapping per-layer scores to an exact budget for LLM pruning (Yuan et al., 2025).

### 5.7 Reward and its proxies

For a plan sampled in training, r = 100 · (acc_val(cut network after the proxy) − acc_val(origin)), in points, on the **validation half** of the catalog network's dataset. The 100-epoch recovery is far too slow to run inside training, so a proxy stands in for it.

| Proxy | What happens to the cut network before the read | Used by |
|---|---|---|
| Raw cut | Nothing: the cut network is read as is | The residual and rule-blending agents |
| BN-recalibrated cut | BN statistics re-estimated on 8 training batches, without gradient (EagleEye's adaptive BN) | An ablation line |
| 32-step fine-tune | 32 SGD steps (lr 0.01, momentum 0.9, weight decay 5·10⁻⁴) on fixed training batches, with the cut channels zeroed again after every step | The direct agent of record |
| One-epoch fine-tune | One epoch of fine-tuning, on the real structural cut | Fidelity reference; one warm-started ablation |

**Implementation.** The training cut is a *masked* cut: the channels the structural cut would remove are zeroed on a working copy. This computes the same function, and its kept size comes from the analytic model. At the start of every training run, a uniform plan is cut both ways on each catalog network, and the trainer checks that kept parameters agree within 10⁻⁶ and validation accuracy within 0.05 points; it logs a warning otherwise. All eight plans of an instance share the same proxy batches (common random numbers), so their rewards differ only through the plans.

**Choosing the proxy by measurement.** Each candidate proxy ranks a fixed set of real plans, and that ranking is compared with the ranking after the 100-epoch recovery.
- At 60 % parameters, near the rules, the raw cut ranked plans validly on the two voting networks (ρ +0.90 and +0.66), though only at +0.32 on a third, reported network, the thin ResNet-20 at width 2 [E11]. At 60 % FLOPs, on the thin ResNet-56, every proxy did (raw cut ρ +0.87) [E11].
- At 35 % parameters across eight of the nine networks (the thin ResNet-56 at width 4 had too few runs left after a registered identity check), the picture changed [E11]. The one-epoch fine-tune reached the validity threshold (ρ +0.60, 55 of 71 plan pairs ordered correctly, against a recovery-to-recovery ceiling of +0.73). The raw cut (+0.41) and the BN-recalibrated cut (+0.45) were weak. On the network where the agent failed (thin ResNet-20 at width 8 on CIFAR-100) they ranked plans *backwards* (−0.84, −0.85), while the one-epoch fine-tune did not (+0.94). The 32-step fine-tune at lr 0.01 came closest to valid (+0.58) at about a quarter of the one-epoch cost.

### 5.8 Training algorithm, data and compute

```
for instance i = 1 .. N:                                   # N = 12,000 (agents of record) or 24,000 (rule-blending)
    draw a catalog network n uniformly and a budget κ ~ U[κ_lo, κ_hi]
    μ = policy(tokens(n, κ, b))                             # one score per planned group
    z_k = μ + σ_i ε_k,  ε_k ~ N(0, I),  k = 1..8           # rule-blending: also u_k = ψ + 2 ε'_k
    widths_k = decode(z_k [, softmax(u_k)]; n, κ, b)
    r_k = 100 · (acc_val(proxy(masked_cut(n, widths_k))) − acc_val(n))
    A_k = r_k − mean_j r_j                                  # FLOPs lines: also divided by std_j r_j + 1e-6
    loss += −(1/(8·4)) Σ_k A_k · log N(z_k; μ, σ_i² I)      # rule-blending: + log N(u_k; ψ, 4 I)
    every 4 instances: clip gradient norm to 1.0, Adam step (lr 3·10⁻⁴)   # the step's gradient is the mean over its 4 instances
σ_i falls linearly from σ_0 to 0.4 σ_0 over the first 60 % of instances (σ_0 = 0.5 direct, 0.25 residual and rule-blending)
the frozen policy is the last iterate: nothing is selected on validation, and nothing reads TEST
```

The update is REINFORCE (Williams, 1992). Its baseline is the mean reward of the eight sibling plans, as in POMO (Kwon et al., 2020). The resulting advantage is 7/8 of the leave-one-out advantage (Kool et al., 2019), and the scheme is close in spirit to GRPO's group baseline (Shao et al., 2024). Under the trust region and the blending softmax, the map from the sampled action (z, u) to a plan is deterministic, so the log-density of the sampled action is exact and no Jacobian term is needed (`plan_trainer.run`). With the plain mean baseline the estimate is then unbiased up to the 7/8 factor; the FLOPs lines' division by the K-plan SD gives up exact unbiasedness for a common advantage scale across instances. There is no entropy bonus, no imitation term and no critic.

| Agent | Budget, training range | Action | Reward | Catalog, instances | Training cost (one RTX 4090) |
|---|---|---|---|---|---|
| Direct agent (parameter agent of record) | Parameters, κ ∈ [0.25, 0.85] | (a) | 32-step fine-tune | 10 networks; 12,000 (3,000 updates) | 20.0 GPU-hours [E26] |
| Residual agent (FLOPs agent of record) | FLOPs, κ ∈ [0.35, 0.85] | (b) | Raw cut; advantage divided by the K-plan SD | 10 networks; 12,000 | 2.9 GPU-hours [E26] |
| Rule-blending agent (current) | FLOPs, κ ∈ [0.35, 0.85] | (c) | Raw cut; advantage as for the residual agent | 39 networks; 24,000 | 7.9–8.2 hours per seed [E35] |

The direct agent trains with a width floor of one channel; the other two with two. Each line trains at seeds 42–44 (42–46 for the agents of record), one policy per seed.

### 5.9 Applying the frozen agent to a new network

Given a pretrained network and a budget, the frozen agent (`plan_agent.plan_for_env`):
1. traces the coupled groups;
2. measures each group's sensitivity and cost (and, for the rule-blending agent, the threshold rule's damage curves);
3. builds the token matrix, with κ and the budget type written in;
4. runs one forward pass for the mean scores μ (and, for the rule-blending agent, a second pass for the blending shares);
5. decodes μ at κ − 0.02;
6. applies the guards of Section 3.4;
7. cuts once and recovers.

There is no sampling, search or update. On CIFAR ResNet-56 the forward pass and decode take under 21 seconds, and the whole decision, mostly the cutting routine, takes 3.3–3.7 minutes [E26]. On ImageNet ResNet-50 the decision takes 44–107 GPU-minutes [E31].

### 5.10 Where training and evaluation differ

- The direct agent trains with a one-channel floor and is evaluated with two.
- Training rewards plans decoded at κ. Evaluation decodes at κ − 0.02 and rounds to the cutting routine's grid, while the state still reads κ.
- Training uses the masked cut; evaluation uses the structural cut, checked equivalent at the start of each training run.
- The rule-blending agent's evaluation runs used optional settings that speed up the cutting routine. These were checked to give bit-identical outputs on seven runs over six networks [E26].

### 5.11 Design rationale in brief

| Choice | Why |
|---|---|
| One plan per network, not a sequence of layer decisions | The step-by-step reward could not separate actions, and the allocation trade was deferred beyond the critic's horizon [E1] |
| Group baseline over 8 plans, no critic | It removes each network's and budget's difficulty from the gradient, with exact credit for a one-step decision |
| Exact budget in the decoder; reward = accuracy only | Every comparison is at equal size, and no plan can win by under-spending |
| Zero-initialised heads | The agent starts at the even cut or at a rule, and learning is a measurable departure from it |
| Sensitivity and cost in the state | Structure and statistics alone do not carry the allocation; the measurement does [E4] |
| Transformer with a coupling bias | It reads any architecture as a sequence and encodes which layers are cut together; simpler and larger encoders tie or lose (Section 6.8) |
| Proxies chosen by measured rank fidelity | The 100-epoch recovery cannot run inside training |
| One recovery for every arm, keep-last, honest change | It isolates allocation from the recovery recipe |

**Figure 1. Data flow.**

```
 pretrained CNN  +  budget (type b in {params, FLOPs}, kept fraction kappa)
        |
        +--> torch.fx dependency trace --> coupled channel groups g = 1..G
        +--> per-group sensitivity s_g and cost c_g   (half-width L1 cut, 4 batches)
        +--> per-layer tokens x_l in R^63   (structure, weight/activation statistics,
        |                                    group cost, sensitivity, kappa, b)
        v
 Transformer encoder (3 x d256, 8 heads, coupling bias) --> per-token scores
        |
        +--> group scores mu_g (mean over the group's producer tokens, centred)
        +--> [rule-blending only] network head --> shares pi over 4 rules
        v
 decoder: keep_g = clip(f(mu_g, rule weights) * c, k_min, 1);  c bisected so cost = kappa
        |   integer widths, polish, width floor, grid rounding (evaluation)
        v
 one structural cut (L1 survivors per group) --> 100-epoch recovery --> TEST / val / 10k / honest

 training only: 8 sampled plans per (network, kappa) --> masked cut + cheap proxy on val
                --> rewards r_k --> advantages r_k - mean(r) --> REINFORCE update
```

---

## 6. Experimental protocol and results

### 6.1 What is compared, and how to read the tables

- **Arms.** Every arm runs in the same pipeline: the frozen agents, the reference rule, the even cut, the cost-aware or plain sensitivity rule, and the threshold rule.
- **Published methods** appear beside our rows with their own base checkpoints, recipes and test sets. They are orientation, not same-pipeline comparators, and nothing here claims to beat them.
- **Units.** All differences are in points of top-1 accuracy. "Agent − rule" is paired by seed.
- **Axes.** Every same-pipeline comparison is at equal FLOPs (a FLOPs point) or at equal parameters (a parameter point). The other axis is free, and we state it where a pair differs on it by more than 10 %. Section 6.7 sets each literature point's accuracy, FLOPs and parameter changes side by side.
- **Status.** Every number is preliminary.

### 6.2 Does the agent learn?

| Agent, network (role), budget | Contrast | TEST | val | 10k | honest |
|---|---|---|---|---|---|
| Raw-cut direct agent trained on one network, on that network (thin ResNet-56, width 4), 60 % parameters | − mild heuristic | **+2.35** | +2.17 | +2.26 | +2.69 |
| Raw-cut direct agent trained on the 10-network catalog, on held-out thin ResNet-56 (width 4), 60 % parameters | − mild heuristic | **+2.20** | +2.20 | +2.20 | +2.58 |
| The same agent, on DepGraph's ResNet-56 (held out), 47 % parameters | − even cut | **+0.96** (+1.10 when re-read with every comparator run with the two-channel width floor) | +0.54 | +0.75 | +0.98 |
| The same agent, on the same two networks | − plain sensitivity rule | +0.12, +0.28 (first reads); +0.61 on DepGraph's ResNet-56 in the re-read | | | |

These are earlier direct agents, trained with the raw-cut reward (κ ∈ [0.35, 0.85]; 3,000 and 12,000 instances), not the parameter agent of record of Section 5.8. Three seeds for the one-network agent, five for the catalog agent's first reads and three for the re-read [E3, E5, E6]. All rows are at equal parameters. **The plan-level agent learns**, where every step-by-step agent had been a clone of the mild heuristic. On these first reads it reaches the level of the sensitivity rule, not above it. In the re-read it is above the plain sensitivity rule on DepGraph's ResNet-56 (+0.61) and level with the cost-aware rule (+0.11) [E6]; over the nine held-out networks at 60 % parameters it is level with the cost-aware rule (−0.01; three seeds) [E51]. At equal parameters it keeps 1.25–1.78× the FLOPs of mild or the even cut and 1.2–1.6× those of the sensitivity rule, so these are parameter-budget results, not compute claims [E3, E5, E6]. Longer training does not add to the gain. For a direct agent trained with the BN-recalibrated reward, the edge over the rule on the eight networks other than the fragile one is flat on validation from 3,000 to 12,000 instances [E44].

### 6.3 Same-pipeline comparison on the nine held-out networks

**Table 1. Agents of record against the reference rule, five deterministic seeds (42–46), nine held-out networks.** "Fragile network" is the thin ResNet-20 at width 8 on CIFAR-100 (72,000 parameters, few and narrow groups), which decides both tails.

| Budget kept | Agent − reference rule: TEST [95 % CI] | val | 10k | honest | Networks positive | Registered outcome | The other eight | Rule − even cut |
|---|---|---|---|---|---|---|---|---|
| Parameters 60 % | +0.05 [−0.17, +0.30] | +0.16 | +0.10 | −0.01 | 4 / 9 | level | −0.03 | +0.45 |
| Parameters 45 % | +0.29 [−0.08, +0.77] | +0.23 | +0.26 | +0.29 | 6 / 9 | level (0.01 under the bar) | +0.09 | +0.39 |
| Parameters 35 % | +0.24 [+0.02, +0.53] | +0.36 | +0.30 | +0.31 | 7 / 9 | level | +0.16 | +0.81 |
| Parameters 25 % | −0.22 [−1.52, +0.41] | −0.22 | −0.22 | −0.22 | 6 / 9 | level | +0.23 | +0.96 |
| FLOPs 60 % | −0.16 [−0.49, +0.07] | −0.20 | −0.18 | −0.22 | 3 / 9 | level | CIFAR-10 seven: −0.03 | |
| FLOPs 45 % | −0.16 [−0.71, +0.17] | −0.08 | −0.12 | −0.13 | 5 / 9 | level | CIFAR-10 seven: +0.04 | |

Parameter rows: the direct agent against the cost-aware rule, at equal parameters [E14]. FLOPs rows: the residual agent against the FLOPs rule, at equal FLOPs [E13]. At equal parameters the agent often keeps more FLOPs than the rule, up to 1.20–1.77× on two networks at 25 % [E14], and the cost-aware rule in turn keeps more FLOPs than the even cut (1.1–2.4× on eight of the nine networks at 25 %, three seeds) [E12].

**Reading.**
- **The frozen agents are at the level of their reference rules at every budget inside their training ranges.** The direct agent is slightly above its rule from 45 % to 35 %, and at 35 % the interval excludes zero, but TEST stays under the registered +0.3 bar. A three-seed "above" at 45 % (+0.42) did not survive five seeds [E14].
- **The fragile network decides both tails.** At 45 % the agent is +1.92 there, with every seed positive. At 25 % one seed's policy collapses on it and loses 15.40 points to the rule [E14]. The same failure is recorded for earlier agent lines: an all-or-nothing plan that keeps most groups whole and cuts a few to about a tenth of their width [E12].
- **Below the training range** the direct agent is below its rule: −0.86 [−2.65, +0.17] at 15 % and −0.43 at 10 % (val −1.01 and −0.41). This is again the fragile network (−6.78, −4.58); the other eight sit at the rule (−0.12, +0.09) [E15]. The residual agent is level with its rule at 25 % and 15 % FLOPs (−0.09, −0.23) and just below it at 10 % (−0.30, over 8 networks, three seeds) [E15].

### 6.4 Literature operating points on CIFAR-10

**Table 2. Five deterministic seeds; TEST change (mean ± SD over seeds) from the unpruned network's TEST accuracy, with the parameter change in parentheses.** "Agent" is the residual agent at FLOPs points and the direct agent at parameter points. FLOPs points are at equal FLOPs, where the agent and the rule keep parameters within 6 % of each other (the even cut differs more). Parameter points are at equal parameters, where the agent keeps 1.11–1.23× the rule's FLOPs at 0.38, 1.05–1.39× at 0.18 and 0.93–1.45× at 0.125 (per seed) [E21]. Published rows show each paper's change from its own base, on the full test set, with its own recipe, and with its FLOPs kept in parentheses. DepGraph publishes speed-ups but no parameter counts [E38]. Section 6.7 gives every arm's FLOPs and parameter changes side by side.

| Network (unpruned TEST) | Point | Agent | Reference rule | Even cut | Agent − rule: TEST (val / 10k / honest) | Published (own recipe; FLOPs kept) |
|---|---|---|---|---|---|---|
| ResNet-56, DepGraph's checkpoint (93.36) | FLOPs 0.39 (2.57×) | +0.02 ± 0.23 (−53 %) | −0.12 ± 0.24 (−54 %) | −0.66 ± 0.21 (−63 %) | +0.15 (+0.11 / +0.13 / +0.21) | DepGraph +0.11 (0.389); OCSPruner −0.51 (0.388, −58 % params, its own checkpoint) |
| | FLOPs 0.25 (4×) | −0.44 ± 0.15 (−70 %) | −0.58 ± 0.18 (−70 %) | −1.46 ± 0.32 (−76 %) | +0.14 (−0.02 / +0.06 / +0.19) | |
| | FLOPs 0.125 (8×) | −2.72 ± 0.38 (−89 %) | −2.64 ± 0.29 (−89 %) | −4.52 ± 0.39 (−87 %) | −0.07 (+0.32 / +0.12 / −0.26) | |
| | Params 0.38 (2.6×) | +0.36 ± 0.16 | +0.28 ± 0.25 | −0.32 ± 0.17 | +0.08 (+0.06 / +0.07 / −0.04) | |
| | Params 0.18 (5.5×) | −1.35 ± 0.21 | −1.08 ± 0.28 | −3.18 ± 0.32 | −0.27 (+0.23 / −0.02 / −0.30) | |
| | Params 0.125 (8×) | −1.98 ± 0.33 | −1.68 ± 0.18 | −4.96 ± 0.44 | −0.30 (−0.38 / −0.34 / −0.47), **below** | |
| ResNet-110 (93.84) | FLOPs 0.48 | +0.07 ± 0.39 (−43 %) | +0.10 ± 0.24 (−43 %) | +0.09 ± 0.21 (−50 %) | −0.04 (+0.22 / +0.09 / +0.04) | CHIP +0.94 (0.479, −48 %); FPGM +0.06 (0.477); OCSPruner −0.06 (0.464) |
| | FLOPs 0.40 | −0.04 ± 0.21 (−51 %) | −0.05 ± 0.18 (−51 %) | −0.19 ± 0.17 (−62 %) | +0.01 (−0.01 / +0.00 / −0.10) | DECORE +0.00 (0.382, −65 %); HRank −0.14 (0.418); LFPC −0.61 (0.397) |
| VGG-16-BN (93.74) | FLOPs 0.465 | +0.18 ± 0.17 (−88 %) | +0.14 ± 0.40 (−89 %) | −0.18 ± 0.19 (−47 %) | +0.04 (−0.07 / −0.01 / −0.12) | HRank −0.53 (0.465, −83 %); CHIP −0.10 (0.419, −82 %) |
| | FLOPs 0.34 | −0.51 ± 0.21 (−93 %) | −0.28 ± 0.23 (−93 %) | −0.32 ± 0.24 (−68 %) | −0.24 (−0.04 / −0.14 / −0.44) | CHIP −0.24 (0.334); HRank −1.62 (0.347) |

Sources: [E21] for our rows, [E43] for the parameter changes (the project's results table, Appendix A). Our 100-epoch recovery also lifts the unpruned network: by +0.64 to +1.38 on DepGraph's ResNet-56, about +0.4 to +0.7 on ResNet-110 and about +0.56 on VGG-16 [E21, E50]. Every honest change is therefore lower than the raw one, and most honest changes at these points are negative. The base checkpoints differ too (for example 93.74 against the 93.96 of the HRank lineage on VGG-16).

**Reading.**
- **At the field's standard points both agents are level with their rules.** There the rules themselves beat the even cut by only +0.01 to +0.60.
- **Toward 8× the allocation matters** (the rules beat the even cut by 1.9–3.3 points). The FLOPs agent tracks its rule. The parameter agent falls slightly below its rule, reaching "below" at 8× by the bar (−0.30), although it keeps up to 1.45× the rule's FLOPs there [E21].
- **Beside the literature,** our changes fall inside the range of the published changes at the same FLOPs points (−1.62 to +0.94). At DepGraph's two published ResNet-56 points, at equal FLOPs, the FLOPs rule's raw changes on the full test set (10k +0.56 at 2.11× and +0.24 at 2.57×; three seeds) sit beside DepGraph's published +0.24 and +0.11. Net of our recovery's lift of the unpruned network they are lower: at 2.57× the honest change is −0.70 [E50]. At 2.11× the rule keeps 0.53–0.60 of the parameters, against 0.508 for DepGraph's released model (DepGraph publishes no parameter count) [E50, E38]. These are different recipes and different evaluation code, and none of it is a win.

### 6.5 Harder datasets: CIFAR-100, Tiny-ImageNet and ImageNet

The residual (FLOPs) agent never trained on CIFAR-100, Tiny-ImageNet or ImageNet: its catalog is CIFAR-10 and SVHN.

**Table 3. Five seeds (ImageNet: one seed); every point at equal FLOPs.** Columns as in Table 2; Section 6.7 gives the parameter changes. On ImageNet the honest change is in parentheses: the unpruned ResNet-50 gains +0.78 from the same 90 epochs. ImageNet changes are from our own read of the original (76.12; torchvision publishes 76.13).

| Network (unpruned TEST) | Point | Agent | Reference rule | Even cut | Agent − rule: TEST (val / 10k / honest) | Published (own recipe; FLOPs kept) |
|---|---|---|---|---|---|---|
| VGG-19, DepGraph's, CIFAR-100 (73.96) | FLOPs 0.50 (2×) | −0.23 ± 0.40 | +0.02 ± 0.54 | −0.23 ± 0.34 | −0.25 (−0.24 / −0.25 / −0.08) | SOSP-H −0.51 (0.484); TPP −0.83 (0.508); GReg-1 −0.94 (0.521); L1 one-shot −1.01 (0.508) |
| | FLOPs 0.112 (8.9×), first convolution locked | −7.34 ± 1.45 | −6.53 ± 0.77 | −7.52 ± 0.40 | −0.82 (−0.25 / −0.53 / −0.78) | DepGraph −3.11; OCSPruner −3.60; TPP −6.24; GReg-2 −6.27; EigenDamage −8.16 (all at FLOPs 0.112–0.114) |
| ResNet-56, CIFAR-100 (73.18) | FLOPs 0.48 | −1.32 ± 0.39 | −1.40 ± 0.15 | −2.66 ± 0.26 | +0.08 (−0.15 / −0.04 / +0.38) | LFPC −0.58 (0.484); Jiang et al. −0.64 (0.478); TAS −0.93 (0.487, with distillation); FPGM −1.75 and SFP −2.61 (0.474, as run by others) |
| VGG-16-BN, Tiny-ImageNet 64 px (61.52) | FLOPs 0.62 | −1.19 ± 0.62 | −0.68 ± 0.30 | +0.15 ± 0.43 | −0.51 (−0.06 / −0.29 / −0.37) | Jiang et al. −1.04, on a different VGG-16 |
| | FLOPs 0.48 | −1.52 ± 0.85 | −2.78 ± 0.82 | −0.54 ± 0.40 | +1.26 (+0.32 / +0.79 / +1.39), **above** | Jiang et al. −1.55, on a different VGG-16 |
| ResNet-18, Tiny-ImageNet 64 px (65.44) | FLOPs 0.48 | −1.35 ± 0.40 | −0.62 ± 0.32 | −1.39 ± 0.36 | −0.74 (−0.39 / −0.56 / −0.54), **below** | none at 64 px |
| ResNet-50, ImageNet (76.12) | FLOPs 0.48 (2.08×) | −0.88 (−1.66) | −0.79 (−1.57) | −0.70 (−1.48) | −0.08 [−0.34, +0.17] (val +0.17) | CHIP +0.00 (0.513); DepGraph −0.32 (0.482); OCSPruner −0.34 (0.517); LeGR −0.80 (0.47); FPGM −1.32 (0.465); GNN-RL −1.82 (0.47) |
| | FLOPs 0.56 (1.79×) | −0.40 (−1.17) | −0.24 (−1.01) | −0.46 (−1.24) | −0.16 [−0.41, +0.09] (val −0.34) | CHIP +0.15 (0.552); OCSPruner −0.29 (0.570); LeGR −0.40 (0.58); FPGM −0.56 (0.578); HRank −1.17 (0.562) |

Sources: [E25, E31]. The ImageNet intervals are 95 % paired intervals over the 50,000 TEST images. They cover the TEST set's sampling only, not seeds, so no verdict is drawn from them.

**Reading.**
- **On datasets it never trained on, the frozen FLOPs agent is level with its rule at four of six points**, above it at one and below it at one [E25]. Its parameters stay within 10 % of the rule's except on VGG-16-BN Tiny-ImageNet at 0.62, where it keeps 14 % fewer [E25].
- **On VGG-16-BN Tiny-ImageNet the even cut beats both** the rule and the agent: the rule is 0.83 and 2.25 points below the even cut there, at equal FLOPs, while the even cut keeps 1.8–2.2× their parameters [E25]. The FLOPs rule itself can mislead on a new dataset.
- **At 8.9× on VGG-19 every arm loses about 7 points with the first convolution locked,** far behind DepGraph's −3.11. Unlocking it recovers 1.4–1.8 points for the threshold rule (Table 5). The unlocked threshold rule then comes within 0.3–0.7 points of DepGraph, and most of that gap is DepGraph's sparsity training (Section 6.6) [E41, E43]. Unlocked, the FLOPs rule, the even cut and the current agent stay 1.5–3.3 points behind DepGraph [E43]. This point is also far below the agent's training range.
- **On ImageNet the agent, the rule and the even cut lie within about 0.2 points of each other at both points** (one seed, so no verdict), and every same-pipeline pair's interval contains 0. All three are at equal FLOPs; the rule keeps 1.2–1.3× the even cut's parameters, and the agent 0.90–0.92× the rule's [E31]. The three sit with LeGR's published rows: 0.4–0.9 points of top-1 below DepGraph, OCSPruner and CHIP at 2.08×, and 0.1–0.6 below OCSPruner and CHIP at 1.79×, whose FLOPs are within 8 % of ours. Measured as a change from each paper's own base, the rule at 1.79× (−0.24) sits above OCSPruner (−0.29). Since the even cut lands with the agent, [H] the gap is more likely recovery (a sparse-learning stage, or a longer fine-tune) than allocation. This is not tested [E31].

### 6.6 The current agent against the strongest rule

**The transfer program.** It targets the networks where better plans exist. Two screens ran four agent-free plans (the reference rule, the even cut, the threshold rule, and the cost-aware rule at equal FLOPs or the plain sensitivity rule at equal parameters) on 30 CIFAR-100 networks at seeds 47–48, picking per network on validation from the other seed. They found *room* (the validation-picked plan beats the reference rule by ≥ 0.3) on 20 of 30 networks at equal FLOPs (TEST +0.81, val +0.93) and on 8 of 30 at equal parameters (TEST +0.81, val +0.76) [E33]. The equal-FLOPs room on VGG-BN and DenseNet-40 comes from plans that keep 1.5–4.4× the rule's parameters; at equal parameters it disappears [E33]. Thirteen held-out networks were drawn from these screens, and nine of them form the registered test set at FLOPs 0.45.

**Table 4. Nine held-out CIFAR-100 networks, FLOPs 0.45, at equal FLOPs, three seeds unless marked** [E34, E35, E37, E40]. The two registered verdicts are the first two rows. The threshold rule − FLOPs rule row covers all 13 held-out networks; in the last row the recalibrated rule is one seed per group, at FLOPs 0.20 and 0.45, against the agent's three. "Best of four" is, per network, the rule with the best validation result among the four agent-free plans. It is an upper envelope built from each network's own validation runs, not a rule a user can apply blind.

| Contrast | TEST [95 % CI] | val | 10k | Networks positive | Outcome |
|---|---|---|---|---|---|
| Rule-blending agent − FLOPs rule | **+0.73** [+0.22, +1.23] | +0.85 | +0.79 | 7 / 9 | level by the registered test (p = 0.039; Holm-adjusted 0.078) |
| Rule-blending agent − threshold rule | **+0.00** [−0.32, +0.44] | −0.06 | −0.03 | 3 / 9 | level |
| Rule-blending agent − best of four | −0.15 [−0.45, +0.15] | −0.22 | −0.19 | 4 / 9 | reported |
| Rule-blending agent − broad-catalog residual agent | +0.90 [+0.58, +1.26] | +1.00 | +0.95 | 9 / 9 | reported |
| Broad-catalog residual agent − FLOPs rule | −0.17 [−0.73, +0.39] | −0.16 | −0.16 | 4 / 9 | level |
| Broad-catalog residual agent − threshold rule | −0.89 [−1.32, −0.48] | −1.06 | −0.98 | 1 / 9 | reported |
| Threshold rule − FLOPs rule (all 13 networks) | +0.64 [+0.26, +1.06] | +0.90 | +0.77 | 11 / 13 | reported |
| Residual agent with its family withheld − residual agent trained with it | −0.35 [−0.55, −0.13] | −0.16 | −0.26 | 2 / 9 | not non-inferior at the −0.3 margin |
| Rule-blending agent − recalibrated threshold rule (18 groups, FLOPs 0.20 and 0.45) | −0.88 [−1.47, −0.27] | −1.22 | −1.05 | | behind |

Honest equals raw in these rows: every arm of a network and seed shares one origin control.

**Reading.**
- **The rule-blending agent is the first agent above the FLOPs reference rule in the mean on these CIFAR-100 networks,** on 7 of 9 networks and with an interval above zero. The registered test, after the Holm adjustment, still rates it level.
- **It ties the threshold rule** at equal FLOPs, keeping parameters within 5 % of it on average (1.33× on DenseNet-40). It learned mostly that rule (a mean share of 0.67–0.84 on it), and its shares vary by family, not by network [E35].
- **It nearly exhausts the four-rule envelope** (−0.15 below best of four), so a higher ceiling needs plans outside these four rules [E35].
- **Part of its gain over the FLOPs rule is parameters bought.** It keeps 0.436 of the parameters, against the rule's 0.387 and the threshold rule's 0.418. On the two VGG-BN networks it keeps 1.6–2× the rule's parameters [E35].
- **Where there is no equal-FLOPs room it costs.** On the two held-out networks that have room only at equal parameters, it is below the FLOPs rule (−0.51, −0.43) [E35].
- **Withholding a whole family from training costs about a third of a point** (shown for the residual agent) [E37].

**Short screens of further ideas,** each with its own validation-only rule: two were dropped, and the third passed its rule on a count alone [E36].
- *Two more rules in the blend* (passed on the count). LAMP or the global magnitude ranking is the validation-best plan on 4 of 9 networks at 0.45; the rule asked for at least 3 networks, or a mean gain of at least +0.20. These are narrow wins that do not follow families, worth +0.18 over the four-rule envelope on validation. Neither is validation-best on any network at 0.20, and LAMP collapses on the residual networks. The screen measures the plans, not an agent that blends them.
- *A per-network pick* (dropped). A one-epoch validation check choosing among seven plans adds +0.25 at 0.45 over the rule-blending agent, but loses −1.95 at 0.20, mostly through one picked plan whose recovery never trained.
- *Distillation during the recovery* (dropped). It does not help on average: it lowers the thin ResNets and raises MobileNetV2.

A perfect chooser, picking on validation after full recovery, would add only +0.05 at 0.45 beyond the one-epoch pick, so the choice among these seven plans is nearly used up there [E36].

**Table 5. Published methods' own widths, run through our cut and recovery, beside the current agent and the rules.** One seed unless marked; TEST change from the unpruned network. These runs (21 November) are tabulated in the project's results table but not yet written into the ledger [E43]. At the VGG points the first convolution is unlocked for every arm. "Locked" gives the threshold rule with our default lock, to show what the lock costs. All arms of a row are at equal FLOPs; their parameters differ, and Section 6.7 gives them for the agent, the threshold rule and the published method.

| Network, FLOPs kept | Rule-blending agent | Threshold rule | Threshold rule, locked | FLOPs rule | Even cut | Published widths, our pipeline | Published (own recipe) |
|---|---|---|---|---|---|---|---|
| ResNet-56, OCSPruner's checkpoint (93.00), 0.385 | +0.00 | +0.10 | | −0.06 | −0.04 | +0.06 | OCSPruner −0.51 (0.388) |
| ResNet-56, OCSPruner's checkpoint, 0.468 | +0.10 | +0.16 | | −0.08 | −0.32 | +0.04 | OCSPruner −0.17 (0.469) |
| ResNet-56, DepGraph's checkpoint (93.36), 0.474 (2.11×) | +0.50 | +0.18 ± 0.23 (2 seeds) | | +0.04 | −0.36 | +0.36 (DepGraph's widths) | DepGraph +0.24 |
| ResNet-110 (93.84), 0.464 | +0.22 | −0.24 | | +0.08 | +0.26 | +0.08 | OCSPruner −0.06 |
| ResNet-110, 0.331 | −0.32 | −0.32 | | −0.50 | −0.86 | −0.62 | |
| VGG-16-BN (93.74), 0.324 | −0.38 | −0.16 | −0.28 | −0.10 | −0.04 | −0.28 | |
| VGG-16-BN, 0.251 | −0.18 | +0.08 | −0.74 | −0.56 | −0.54 | −0.90 | |
| VGG-19, DepGraph's, CIFAR-100 (73.96), 0.112 (8.9×), 3 seeds | −5.50 ± 1.02 | −3.81 ± 0.32 | −5.23 ± 0.37 | −4.91 ± 0.42 | −5.76 ± 0.05 | −4.48 ± 0.26 | DepGraph −3.11; OCSPruner −3.60 |
| VGG-19, 0.110 (8.84×), 3 seeds | −5.63 ± 0.58 | −3.40 ± 0.75 | −5.22 ± 0.21 | −4.60 ± 0.42 | −6.41 ± 1.01 | −4.15 ± 0.85 | DepGraph −3.11 (published, 8.92×); one run of its released code on our GPU read −2.97 [E52] |

**Reading.**
- **At 2–4× on the CIFAR-10 networks, every same-pipeline plan, including the published methods' own widths, lands within about 1 point of the others, and usually within a few tenths** (single seeds except where marked).
- **At 8.9× on VGG-19 the picture splits.** The first-convolution lock costs the threshold rule 1.4–1.8 points. Unlocked, the threshold rule (−3.40 to −3.81) comes within 0.3–0.7 points of DepGraph's published −3.11, under a different recipe. The rule-blending agent, operating far below its training range (0.35–0.85), is 1.7–2.2 points below the threshold rule at equal FLOPs, keeping about 0.55× its parameters (3.7–4.0 % of the original, against 6.7–7.0 %) [E43].

**Inside DepGraph's own code** (DepGraph's released pipeline and recipe, three seeds, TEST on our split, every arm at equal FLOPs; exploratory, with bars set in advance but no verdict) [E41]. These runs compare the *threshold rule's* widths with DepGraph's learned allocation, not the agent.

| VGG-19 CIFAR-100 at 8.84× | Params kept | TEST [honest] | val | 10k |
|---|---|---|---|---|
| DepGraph as published (sparsity training, then its global cut) | 0.063 | −3.36 [−3.62] | −2.78 | −3.07 |
| Threshold rule re-measured after DepGraph's sparsity training | 0.067 | −3.37 [−3.63] | −2.84 | −3.10 |
| Sparsity training aimed at the threshold rule's channels | 0.067 | −3.38 [−3.64] | −3.03 | −3.20 |
| Threshold rule's widths measured on the dense network, after DepGraph's sparsity training | 0.067 | −3.55 [−3.81] | −3.31 | −3.43 |
| Threshold rule's widths, no sparsity training | 0.067 | −3.79 [−4.05] | −3.58 | −3.68 |
| DepGraph's widths, no sparsity training | 0.061 | −4.51 [−4.77] | −3.81 | −4.16 |

- **Without sparsity training, the rule's allocation is ahead of DepGraph's own widths on every seed** (10k +0.47).
- **Sparsity training is the rest of DepGraph's lead.** It buys DepGraph's widths about +1.1, but the rule's widths only +0.25.
- **Measured on the network that is actually cut,** after the sparsity training, or with the training aimed at the rule's channels, the training-free rule is level with DepGraph's full method (10k −0.03 and −0.13).
- **On ResNet-56 CIFAR-10 at 2.11×** it is level again: 10k +0.21, all three seeds positive, while keeping 13 % more parameters. DepGraph's own arm read +0.00 there, against its published +0.24.
- At 8.84× DepGraph's own arm reproduces its paper (10k −3.07, against the published −3.11 at 8.92×).
- No route tested so far gives the rule a lead at these points.

### 6.7 Head-to-head at the literature points: accuracy, FLOPs and parameters

The field reports a pruning result as an accuracy change at a FLOPs reduction, with the parameter reduction beside it. Table 6 puts every literature point of Tables 2, 3 and 5 in that form: per network and point, one DRL agent, one same-pipeline rule and the strongest published method [E53].

**Table 6. Accuracy change (points), FLOPs change (%) and parameter change (%), in that order, for the DRL agent, the rule and the strongest published method at each point.** Our rows give the TEST change and the FLOPs and parameter changes from the unpruned network, as mean ± SD over seeds; the number of seeds follows each label in parentheses, and one-seed rows have no SD. Each row's agent and rule are fixed by a stated rule, never by a test score: the current agent and the current threshold rule where both were run at that point (the 21 November runs); otherwise the earlier agent trained on the point's budget type and the rule fixed in advance for that type. The published method is the strongest published result within about 10 % of the point's FLOPs. It comes from its own paper, base checkpoint, recipe and epochs, read on its own full test set; it is set beside ours for orientation and is never a contest. *Earlier agent*: an agent of record of Tables 1–3, trained on parameter or on FLOPs budgets. *Current agent*: the rule-blending agent. *Floored*: run with the two-channel width floor. n/p: not published.

| Network (original TEST) | Point | DRL agent (seeds) | Acc. | FLOPs % | Params % | Rule (seeds) | Acc. | FLOPs % | Params % | Strongest published | Acc. | FLOPs % | Params % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-56, DepGraph's checkpoint, CIFAR-10 (93.36) | params 0.38 (2.6×) | Earlier agent, parameter budgets (5) | +0.36 ± 0.16 | −43.7 ± 3.5 | −62.0 ± 0.0 | Fixed in advance: cost-aware rule (5) | +0.28 ± 0.25 | −51.5 ± 1.3 | −62.1 ± 0.0 | none at this point | | | |
| | FLOPs 0.39 (2.57×) | Earlier agent, FLOPs budgets (5) | +0.02 ± 0.23 | −61.0 ± 0.0 | −52.6 ± 2.6 | Fixed in advance: floored FLOPs rule (5) | −0.12 ± 0.24 | −61.0 ± 0.0 | −53.9 ± 1.9 | DepGraph (CVPR 2023) | +0.11 | −61.1 | n/p |
| | FLOPs 0.474 (2.11×), our pipeline | Current agent (1) | +0.50 | −52.6 | −52.5 | Current threshold rule (2) | +0.18 ± 0.23 | −52.7 ± 0.1 | −44.6 ± 3.0 | DepGraph (CVPR 2023) | +0.24 | −52.6 | n/p |
| | FLOPs 0.474 (2.11×), in DepGraph's code and recipe (original 93.68) | not run here | | | | Current threshold rule after DepGraph's sparsity training (3) | +0.11 ± 0.22 | −52.7 ± 0.0 | −43.6 ± 1.8 | DepGraph, its own code, run by us (3) | −0.05 ± 0.26 | −52.9 ± 0.3 | −49.9 ± 1.6 |
| | FLOPs 0.25 (4×) | Earlier agent, FLOPs budgets (5) | −0.44 ± 0.15 | −75.0 ± 0.0 | −69.6 ± 1.0 | Fixed in advance: floored FLOPs rule (5) | −0.58 ± 0.18 | −75.0 ± 0.0 | −70.0 ± 0.7 | none at this point | | | |
| | FLOPs 0.125 (8×) | Earlier agent, FLOPs budgets (5) | −2.72 ± 0.38 | −87.5 ± 0.0 | −88.6 ± 0.6 | Fixed in advance: floored FLOPs rule (5) | −2.64 ± 0.29 | −87.6 ± 0.0 | −89.0 ± 0.5 | none at this point | | | |
| | params 0.18 (5.5×) | Earlier agent, parameter budgets (5) | −1.35 ± 0.21 | −64.7 ± 4.7 | −82.0 ± 0.1 | Fixed in advance: cost-aware rule (5) | −1.08 ± 0.28 | −71.7 ± 1.1 | −82.0 ± 0.0 | none at this point | | | |
| | params 0.125 (8×) | Earlier agent, parameter budgets (5) | −1.98 ± 0.33 | −73.3 ± 5.1 | −87.5 ± 0.0 | Fixed in advance: cost-aware rule (5) | −1.68 ± 0.18 | −78.7 ± 1.0 | −87.5 ± 0.0 | none at this point | | | |
| ResNet-56, OCSPruner's checkpoint, CIFAR-10 (93.00) | FLOPs 0.385 | Current agent (1) | +0.00 | −61.6 | −54.4 | Current threshold rule (1) | +0.10 | −61.6 | −50.1 | OCSPruner (WACV 2026) | −0.51 | −61.2 | −57.7 |
| | FLOPs 0.468 | Current agent (1) | +0.10 | −53.2 | −47.2 | Current threshold rule (1) | +0.16 | −53.3 | −41.5 | OCSPruner (WACV 2026) | −0.17 | −53.1 | n/p |
| ResNet-110, CIFAR-10 (93.84) | FLOPs 0.48 | Earlier agent, FLOPs budgets (5) | +0.07 ± 0.39 | −52.0 ± 0.0 | −42.5 ± 4.0 | Fixed in advance: floored FLOPs rule (5) | +0.10 ± 0.24 | −52.0 ± 0.1 | −42.9 ± 3.8 | CHIP (NeurIPS 2021) | +0.94 | −52.1 | −48.3 |
| | FLOPs 0.464 | Current agent (1) | +0.22 | −53.6 | −48.7 | Current threshold rule (1) | −0.24 | −53.7 | −42.8 | OCSPruner (WACV 2026) | −0.06 | −53.6 | n/p |
| | FLOPs 0.40 | Earlier agent, FLOPs budgets (5) | −0.04 ± 0.21 | −60.0 ± 0.0 | −51.1 ± 3.3 | Fixed in advance: floored FLOPs rule (5) | −0.05 ± 0.18 | −60.0 ± 0.0 | −51.2 ± 3.0 | DECORE (CVPR 2022) | +0.00 | −61.8 | −64.8 |
| | FLOPs 0.331 | Current agent (1) | −0.32 | −66.9 | −62.2 | Current threshold rule (1) | −0.32 | −66.9 | −57.0 | none at this point | | | |
| VGG-16-BN, CIFAR-10 (93.74) | FLOPs 0.465 | Earlier agent, FLOPs budgets (5) | +0.18 ± 0.17 | −53.5 ± 0.0 | −88.0 ± 0.6 | Fixed in advance: floored FLOPs rule (5) | +0.14 ± 0.40 | −53.5 ± 0.0 | −88.7 ± 0.8 | HRank (CVPR 2020) | −0.53 | −53.5 | −82.9 |
| | FLOPs 0.34 | Earlier agent, FLOPs budgets (5) | −0.51 ± 0.21 | −66.0 ± 0.0 | −92.7 ± 1.1 | Fixed in advance: floored FLOPs rule (5) | −0.28 ± 0.23 | −66.0 ± 0.0 | −93.0 ± 1.0 | CHIP (NeurIPS 2021) | −0.24 | −66.6 | n/p |
| | FLOPs 0.324 | Current agent (1) | −0.38 | −67.6 | −90.3 | Current threshold rule (1) | −0.16 | −67.6 | −85.9 | none at this point | | | |
| | FLOPs 0.251 | Current agent (1) | −0.18 | −74.9 | −91.8 | Current threshold rule (1) | +0.08 | −74.9 | −86.8 | none at this point | | | |
| VGG-19, DepGraph's checkpoint, CIFAR-100 (73.96) | FLOPs 0.50 (2×) | Earlier agent, FLOPs budgets (5) | −0.23 ± 0.40 | −50.0 ± 0.0 | −76.8 ± 0.6 | Fixed in advance: floored FLOPs rule (5) | +0.02 ± 0.54 | −50.0 ± 0.0 | −75.8 ± 0.0 | SOSP-H (ICLR 2022) | −0.51 | −51.6 | n/p |
| | FLOPs 0.112 (8.9×), first convolution locked | Earlier agent, FLOPs budgets (5) | −7.34 ± 1.45 | −88.8 ± 0.0 | −97.2 ± 0.5 | Fixed in advance: floored FLOPs rule (5) | −6.53 ± 0.77 | −88.8 ± 0.0 | −97.0 ± 0.3 | DepGraph (CVPR 2023) | −3.11 | −88.8 | n/p |
| | FLOPs 0.112 (8.9×) | Current agent (3) | −5.50 ± 1.02 | −88.8 ± 0.0 | −96.0 ± 0.2 | Current threshold rule (3) | −3.81 ± 0.32 | −88.8 ± 0.0 | −93.0 ± 0.1 | DepGraph (CVPR 2023) | −3.11 | −88.8 | n/p |
| | FLOPs 0.110 (8.84×), our pipeline | Current agent (3) | −5.63 ± 0.58 | −89.0 ± 0.1 | −96.2 ± 0.5 | Current threshold rule (3) | −3.40 ± 0.75 | −89.0 ± 0.0 | −93.3 ± 0.1 | DepGraph (CVPR 2023), at 8.92× | −3.11 | −88.8 | n/p |
| | FLOPs 0.110 (8.84×), in DepGraph's code and recipe | not run here | | | | Current threshold rule after DepGraph's sparsity training (3) | −3.37 ± 0.83 | −89.0 ± 0.0 | −93.3 ± 0.2 | DepGraph, its own code, run by us (3) | −3.36 ± 0.35 | −88.9 ± 0.1 | −93.7 ± 0.2 |
| ResNet-56, chenyaofo's checkpoint, CIFAR-100 (73.18) | FLOPs 0.48 | Earlier agent, FLOPs budgets (5) | −1.32 ± 0.39 | −52.1 ± 0.1 | −25.8 ± 2.5 | Fixed in advance: floored FLOPs rule (5) | −1.40 ± 0.15 | −52.1 ± 0.0 | −26.2 ± 1.7 | Jiang et al. | +0.00 | −48.8 | n/p |
| VGG-16-BN, Tiny-ImageNet 64 px (61.52) | FLOPs 0.62 | Earlier agent, FLOPs budgets (5) | −1.19 ± 0.62 | −38.0 ± 0.0 | −67.1 ± 3.9 | Fixed in advance: floored FLOPs rule (5) | −0.68 ± 0.30 | −38.0 ± 0.0 | −61.8 ± 0.0 | Jiang et al. | −1.04 | −37.9 | n/p |
| | FLOPs 0.48 | Earlier agent, FLOPs budgets (5) | −1.52 ± 0.85 | −52.0 ± 0.0 | −75.4 ± 2.1 | Fixed in advance: floored FLOPs rule (5) | −2.78 ± 0.82 | −52.0 ± 0.0 | −76.0 ± 1.1 | Jiang et al. | −1.55 | −52.1 | n/p |
| ResNet-18, Tiny-ImageNet 64 px (65.44) | FLOPs 0.48 | Earlier agent, FLOPs budgets (5) | −1.35 ± 0.40 | −52.1 ± 0.1 | −40.1 ± 4.3 | Fixed in advance: floored FLOPs rule (5) | −0.62 ± 0.32 | −52.1 ± 0.1 | −34.1 ± 0.1 | none at this point | | | |
| ResNet-50, ImageNet (76.12) | FLOPs 0.48 (2.08×) | Earlier agent, FLOPs budgets (1) | −0.88 | −52.0 | −44.5 | Fixed in advance: floored FLOPs rule (1) | −0.79 | −52.0 | −39.6 | CHIP (NeurIPS 2021) | +0.00 | −48.7 | n/p |
| | FLOPs 0.56 (1.79×) | Earlier agent, FLOPs budgets (1) | −0.40 | −44.0 | −34.5 | Fixed in advance: floored FLOPs rule (1) | −0.24 | −44.0 | −27.2 | CHIP (NeurIPS 2021) | +0.15 | −44.8 | n/p |

Notes to Table 6.
- Published bases differ from ours: CHIP and DECORE report ResNet-110 from 93.50 (ours 93.84), HRank and CHIP report VGG-16 from 93.96 (ours 93.74), and Jiang et al.'s Tiny-ImageNet VGG-16 is a different network, with a 4096-wide classifier [E53].
- Published FLOPs come from each paper's own counter, which counts only convolutions and linear layers; for the same cut ours reads 0.2–2 % (relative) more FLOPs kept (Section 3.3) [E38].
- The two rows "in DepGraph's code" run DepGraph's released pipeline and recipe on both sides (Section 6.6); there the ResNet-56 original reads 93.68 [E41].
- At VGG-19 0.110, one run of DepGraph's released code on our GPU read −2.97; DepGraph's published row is −3.11 at 8.92× (FLOPs kept 0.112) [E52, E38].
- ImageNet: one seed; TEST is the official 50,000-image validation set, and changes are from our own read of the original (76.12; torchvision publishes 76.13) [E31].
- Further published rows at these points are in Tables 2, 3 and 5.

**Reading.**
- **At equal FLOPs the arms keep different parameters.** Wherever both current arms ran, the current threshold rule keeps more parameters than the current agent: 1.1–1.6× on the CIFAR-10 networks and about 1.75× at 8.9× on VGG-19. At the three parameter points the earlier agent keeps 1.16–1.25× the cost-aware rule's FLOPs on average [E53].
- **Most published rows give no parameter count** (4 of the 18 published entries here do). Where one does, at about our FLOPs, the published method removes more parameters than our same-pipeline arms on the two ResNets (OCSPruner, CHIP, DECORE) and fewer on VGG-16 (HRank) [E53].

### 6.8 Transfer map and model ablations

**Table 7. Where the frozen policy holds and where it does not.**

| Test | Result | Evidence |
|---|---|---|
| Held-out networks of the training families, on CIFAR-10 and CIFAR-100 | Level with the reference rules at every budget inside the training range (Table 1) | E13, E14 |
| New residual architectures (WRN, PreAct-ResNet), all eight networks on CIFAR-10 and CIFAR-100 | Below the cost-aware rule at 45 % by the registered verdict (−0.32 [−0.88, +0.14], 2 of 8 positive), carried by Wide-ResNet-28-2 CIFAR-100 (−1.88); level at 25 % (−0.26); five seeds | E20 |
| The same, by dataset (reported, no verdict) | The four on CIFAR-10: −0.06 at 45 %, −0.11 at 25 %. The four on CIFAR-100, where architecture and dataset are both new: −0.57 and −0.40 | E20 |
| The same eight, with a validation fallback to the rule | Level (+0.09, +0.08) | E22 |
| Trained without any ResNet, read on DepGraph's ResNet-56 | Holds: +0.95 over the even cut (keeping 1.2–1.4× its FLOPs), −0.04 against the cost-aware rule. On the thin ResNet-56 it sits at the even cut's level, about a point below the rules | E6, E49 |
| Trained without any VGG, read on VGG-19 CIFAR-100 and VGG-16 CIFAR-10 | Fails on VGG-16 (−0.33 against the even cut, keeping about 1.3× its FLOPs), a point where the sensitivity rule is itself below the even cut and the agent that saw VGGs is below it too | E6, E49 |
| Families no catalog holds (RepVGG, ShuffleNetV2) | Uninformative: the rule is level with the even cut on RepVGG, and ShuffleNetV2 cannot be cut cleanly | E48 |
| FLOPs agent trained with no CIFAR-10, no plain ResNet and no VGG | The same as the agent of record at the 12 literature points (−0.04; val −0.13, 10k −0.09) and level with the rule (−0.07) | E27 |
| Parameter agent trained the same way | Worse than the agent of record (−0.51 at 45 %, −0.70 at 25 %) and below the rule at 25 % (−0.59); retrained on CIFAR-100 networks only, worse still (−0.91, −1.55) | E27 |
| CIFAR-100, Tiny-ImageNet, ImageNet (FLOPs agent) | Level at 4 of 6 points on CIFAR-100 and Tiny-ImageNet (above at one, below at one); on ImageNet within 0.2 points of the rule at both points (one seed, no verdict) | E25, E31 |

Every row is read at the agent's own budget type: parameter agents at equal parameters, FLOPs agents at equal FLOPs. A caveat for the rows that withhold families: the agents trained without ResNets or without VGGs, and the held-out parameter agent, kept the state standardizer fitted once on the 10-network CIFAR-10 and SVHN catalog, which holds ResNets and VGGs. Their token z-scoring therefore includes the withheld families' feature statistics [E49, E27]. The held-out FLOPs agent changed only its catalog, so [H] the same probably holds for it.

**Table 8. What we changed in the agent, and what happened.** TEST, pooled over the nine held-out networks against the reference rule unless stated, at the agent's own budget type (equal parameters or equal FLOPs); three seeds unless stated.

| Change | Result | Evidence |
|---|---|---|
| Reward: raw cut → BN-recalibrated cut (training range 25–85 %) | At 25 % parameters, −1.60 → −0.17 against the rule; on the fragile network −16.84 → −2.41 | E12 |
| Reward: 32-step fine-tune (the direct agent of record) | +0.16 at 25 % (three seeds); fragile network −0.93; at five seeds −0.22, with one collapsing seed | E12, E14 |
| One-epoch reward, warm-started on budgets 10–50 % | Level with the rule at every depth at five seeds; +1.52 over the agent of record at 15 %, mostly through the fragile network; below the rule at 15 % on new architectures (−0.42) | E44 |
| Training floor 0.25 instead of 0.1 | The all-or-nothing plan moves to the floor (fragile network −3.79) | E12 |
| Training range 10–85 % from scratch | Below the agent of record at four of five depths (−0.33 to −0.55) | E44 |
| Training length 3,000 → 12,000 instances | Flat on the other eight networks; the collapse on the fragile network appears from 6,000 instances | E44 |
| Adding 8 CIFAR-100 networks to the catalog | Neutral with the raw-cut reward (+0.02); helps the fragile network at 25 % with the BN reward (+2.09 against the rule); hurts at 25 % with the 32-step reward (−0.57 against the agent of record) | E44 |
| Catalog size 1 / 3 / 5 / 10 networks | One is not enough; three or five CIFAR-10 networks match ten at 45 % | E24 |
| Averaging three seeds' scores (one frozen ensemble) | Does not remove the collapse (fragile network −3.61) | E46 |
| Encoder: per-group MLP, no attention (half training) | Level at 45 % (+0.02), −0.34 at 25 % (the fragile network) | E28 |
| Encoder: Transformer 8× larger (d 512, 6 layers, 19 M parameters) | Level (−0.19, +0.57) | E28 |
| Inputs: only the rule's inputs (sensitivity, cost, budget, type, position) | Level with the Transformer on the nine networks except the fragile one; −0.40 on new architectures at 15 % | E28 |
| No sensitivity columns (imitation probe) | Spearman ρ 0.45–0.59, against 0.87 with them | E4 |
| Advantage divided by the K-plan SD (FLOPs residual) | No measurable balance effect (+0.44 over the unnormalised residual pooled over two networks, bars ±0.5) | E8 |
| Trust region on the residual | +1.07 on the held-out thin ResNet-56 at FLOPs 0.6 | E8 |
| Two-stage re-planning (plan to 50 %, recover 30 epochs, re-measure, plan to 25 %) | Level with one-shot (+0.30 pooled over nine, carried by the fragile network; the other eight −0.06) | E23 |
| Validation fallback to the rule (one-epoch validation check) | Removes the collapse (fragile network +0.58 against the rule at 25 %); level elsewhere; 28 other validation checks were no better | E22 |
| Blending four rules (current agent) | On the nine held-out CIFAR-100 networks at FLOPs 0.45 (Table 4): +0.73 over the FLOPs rule (level by test), +0.00 against the threshold rule | E35 |

The pattern is consistent. **Rewards closer to recovery help at deep budgets. Larger models do not help, and smaller ones lose only on the fragile network. Training length and catalog changes do not help reliably. The fragile network drives most pooled differences at 25 % and below.**

### 6.9 Cost

**Table 9. Time to a pruned, recovered ResNet-56 on CIFAR-10, one RTX 4090, minutes** [E26]. Each published method runs its official code, recipe and data loader. SPECTRA's rows assume the trained base exists.

| Pipeline | Decision (before the final fine-tune) | Final fine-tune (epochs) | Total |
|---|---|---|---|
| SPECTRA, FLOPs agent, fast GPU data pipeline | 3.3 | 4.4–4.5 (100) | 7.7–7.8 |
| SPECTRA, parameter agent, fast GPU data pipeline | 3.7 | 4.8 (100) | 8.5 |
| SPECTRA, stock data loader | 3.4–3.6 | 17.6–17.8 (100) | 20.9–21.4 |
| FPGM | (whole run) | soft pruning (200) | 44 |
| OCSPruner (includes training its base) | 28 | 43 | 71 |
| DepGraph | 65 | 20 (100) | 85 |
| CHIP | 20 | 81 (300) | 101 |
| AMC (LeGR's implementation) | 85 | 24 (60) | 109 |
| LeGR | 10 | 157 (400) | 167 |
| RL-Pruner | (whole run) | distillation interleaved | 271 |
| HRank | 338 | 6 (last of 55 stages) | 344 |
| GNN-RL | over 1,020 (search unfinished at its 17 h wall) | 13.6 (100) | over 1,034 |

- **Like for like** (stock loaders), the fastest published pipeline takes about twice SPECTRA's time.
- **On VGG-16** SPECTRA takes 3.8–4.0 minutes (10.0–10.3 with a stock loader), against 46–118 for OCSPruner, HRank and CHIP.
- **The agents' one-time training pays back** within 4.8 networks (FLOPs agent) and 33.7 networks (parameter agent) against the published pipelines timed (7.5 and 52.9 with a stock loader).
- **Optional settings made the cutting routine 3.3–4.4× faster** with bit-identical outputs [E26].
- **Fewer FLOPs are not faster on a small CIFAR ResNet.** 37–61 % fewer FLOPs run at 0.93–1.07× on this GPU at batch 256, while VGG-16 gains 1.32–1.65× [E18]. The saving SPECTRA claims is the time to produce a pruned network, not a faster pruned network.

---

## 7. Analysis: what works and what does not

### 7.1 What works

1. **The plan-level formulation learns where the step-by-step one could not.** More than two points over the mild heuristic at equal parameters, on one network and then on a held-out one [E3, E5].
2. **At equal parameters the agent reaches the cost-aware rule's level.** [H] From its reward and the cost columns of its state, it learns an allocation at that rule's level. Against the cost-blind sensitivity rule, the first catalog agent gained most (+0.49 to +0.80; three seeds) on the three networks where that rule is at or below the even cut, while keeping 1.4–1.85× its FLOPs [E51]. [H] At equal parameters the cost structure is the whole game: in a CIFAR ResNet a last-stage convolution holds about 16× the parameters of a first-stage one, at about the same FLOPs.
3. **A frozen policy transfers at the level of the best rule fixed in advance.** This holds on held-out networks, on new residual architectures on CIFAR-10, on new datasets for the FLOPs agent (level at four of six points, and within 0.2 points on ImageNet at one seed), and with CIFAR-10 and the benchmarked families withheld from the FLOPs agent's training catalog, though [H] probably not from its state standardizer (Section 6.8) [E13, E14, E20, E25, E27, E31].
4. **It beats a per-network search that uses its own reward**, at equal parameters. Against the best of 100 random plans in its own action space, scored by its own 32-step reward: +0.51 [+0.16, +0.91] at 45 % parameters (8 of 9 networks; three seeds), and +0.45 [−0.61, +1.31] at 25 % (6 of 9; the interval includes zero, and on the fragile network the search wins by 2.67). The agent keeps 1.1–1.5× the search's FLOPs at 45 % and up to 2.10× at 25 %, so [H] part of its edge may be FLOPs that a parameter budget does not count. The search itself is level with the cost-aware rule [E16].
5. **What matters after recovery is the allocation**, not which channels or which weights survive. Random channels at the agent's widths (the random baseline of Blalock et al., 2020) end level with L1 channels (+0.13, +0.23). The agent's architecture trained from scratch with matched compute ends level with the inherited weights, as Liu et al. (2019b) found for structured pruning [E17].
6. **The protocol is sound and reproducible.** It uses pre-registered verdicts, a TEST half that no decision reads, honest changes, deterministic recovery, and paired seeds [E19].
7. **It is fast** [E26].

What carried over from NEON is its idea (a generic, frozen policy) and its feature view of a layer. What did not carry over:
- the step-by-step decision process, and NEON's per-step fine-tune [E1];
- actor-critic training, replaced by a group baseline;
- the mixed accuracy-compression reward. The budget became an exact constraint, and NEON's preference returned as an accuracy-limit mode: state the accuracy you can lose, and SPECTRA bisects the size [E29];
- NEON's layer replacement, replaced by coupled channel cuts.

### 7.2 What does not work

1. **No registered "beats" holds against the strongest rule.** At five seeds every pooled verdict is level (Table 1). The current agent ties the threshold rule at equal FLOPs (Table 4).
2. **Deep budgets break the agent on fragile networks.** Its worst failures (an all-or-nothing plan on the thin ResNet-20 on CIFAR-100) appear at 25 % parameters and below, and at FLOPs far below the training range on VGG-19 [E12, E14, E15, E43].
3. **Transfer across families has a cost.** Withholding a family from training costs about a third of a point [E37]. On the eight new architectures the parameter agent falls below the rule at 45 % (−0.32, a registered verdict), carried by Wide-ResNet-28-2 on CIFAR-100; the four CIFAR-100 networks, where architecture and dataset are both new, read −0.57 (reported) [E20].
4. **The parameter agent did not survive a catalog without CIFAR-10** [E27].
5. **Several fixes did not help:** two-stage re-planning, seed ensembles, deeper training ranges, larger encoders, more training, a per-network pick and distillation [E23, E46, E44, E28, E36]. Two extra rules in the blend passed their screen only on a count, by a narrow margin (Section 6.6).

### 7.3 Why the agent stops at the rule [H, each grounded in measurements]

**(i) The long recovery saturates allocation at moderate budgets.**
- At the field's standard points the rules beat the even cut by only +0.01 to +0.60 [E21].
- After 100 epochs, the choice of channels within the agent's widths barely matters [E17].
- On ImageNet the agent, the rule and the even cut lie within about 0.2 points of each other (one seed, no verdict) [E31].
- Screens of tilted, perturbed and alternative plans found no allocation above either reference rule on the five networks where allocation matters (best alternative − rule: −0.04 at 60 % FLOPs, +0.04 at 45 % FLOPs, +0.22 at 60 % parameters) [E10].

On the nine-network pool (mostly CIFAR-10), every sensible plan lands within noise of the rule, and there no allocator, learned or designed, can show much. This matches the literature (Liu et al., 2019b; Le & Hua, 2021). On CIFAR-100 the screens found more room, much of it at equal FLOPs bought with parameters (Section 6.6).

**(ii) The training reward is read before recovery, and at deep budgets it ranks plans wrongly.** The raw-cut and BN-recalibrated rewards rank plans weakly at 35 % parameters and backwards on the fragile network, while a one-epoch fine-tune ranks them validly [E11]. The agent's failures sit exactly there, and every reward closer to recovery reduced them: raw cut −16.84, BN-recalibrated −2.41, 32-step −0.93 on the fragile network at 25 %, all at three seeds; at five seeds the 32-step agent reads −3.86 there, with one seed collapsing [E12, E14]. A 2-epoch gentle recovery ranks plans nearly as well as a repeat of the 100-epoch run (Kendall τ +0.52 against a noise ceiling of +0.55; the cut alone +0.16) [E42]. But it costs minutes per plan, too much for the reward of 12,000 × 8 training plans at current compute.

**(iii) There is little headroom above the strongest rule at equal size.** On 36 CIFAR-100 network-budget groups, a 2-epoch screen over a portfolio of plans (the four rules plus agent plans) picks plans +0.42 [+0.06, +0.87] above the threshold rule at equal FLOPs (val +0.49, 10k +0.45). The picked plans keep 22 % more parameters, though; held to the rule's size on both axes the gain is +0.00 [−0.18, +0.16] [E42]. A pick by 100-epoch validation accuracy, held to the same sizes, reads +0.26 to +0.39; it is an upper reference, not a selector, because it is read on the same trained network as its TEST outcome and so shares that run's luck [E42]. In hindsight, the best of the tried plans per group beats the threshold rule by 0.80 points on average (at equal FLOPs, parameters free), and the agent's plans are the weakest kind there (mean regret 2.87, against the threshold rule's 0.80) [E42]. A validation-based perfect chooser among seven plans adds only +0.05 at FLOPs 0.45 beyond the one-epoch pick [E36]. The room the screens found is real, but much of it at equal FLOPs is parameters bought (Section 6.6).

**(iv) The action parameterizations keep the agent near the rules by design.**
- The residual agent cannot leave e^{±0.3} of its rule's weights, so it cannot reach the even cut's or the threshold rule's plan where those are best [E34].
- The rule-blending agent can reach the four rules, but its gains stop at their envelope (−0.15) [E35].
- The direct agent is free, but it is trained on a reward that misleads it exactly where freedom matters [E11, E12].

**(v) Representation is not the bottleneck.**
- Imitation needs the sensitivity columns, but not a particular encoder: wide Transformer, set encoder, frozen BERT and group tokens all tie [E4].
- A larger Transformer ties, and a per-group model with only the rule's inputs ties except on one network [E28].

[H] This points to a better proposer of plans, or a better reward, rather than a better reader of the network.

### 7.4 Where it falls behind

- **Deep budgets, below or near the edge of the training range**: parameters 25 % and below, and FLOPs about 0.11 on VGG-19 (−5.5 against the threshold rule's −3.4 to −3.8, at equal FLOPs, with about 0.55× the rule's parameters) [E14, E15, E43].
- **Fragile networks**: few, narrow groups (the thin ResNet-20 on CIFAR-100), where one seed's all-or-nothing plan costs 15 points [E14].
- **New architecture on a new dataset**: Wide-ResNet-28-2 on CIFAR-100 (−1.88 at 45 %) [E20].
- **Where the reference rule misleads**: VGG-16-BN on Tiny-ImageNet, where the even cut beats both [E25].

The validation fallback removes the collapse on the fragile network and the gap on the new architectures, and turns the loss on ResNet-18 Tiny-ImageNet from −0.74 into a level −0.47, at the price of one extra epoch per candidate [E22]. It is model selection, not learning, and it is reported apart from the agent.

### 7.5 Recent negative and neutral results (22–23 November)

- **Recalibrating the threshold rule's measurement is a trade, not an improvement** [E39]. With BN statistics re-estimated on each probe, the threshold rule gains +0.46 [+0.02, +0.90] at equal FLOPs (+1.12 at FLOPs 0.20, −0.13 at 0.45; 36 groups on 21 networks; one seed per group), but keeps 17–32 % more parameters. At equal parameters it loses −0.93 [−1.33, −0.54] (val −0.87, 10k −0.90) and keeps 15–17 % fewer FLOPs. Recalibration moves plans toward layers heavy in parameters and light in FLOPs. A size-adjusted diagnostic (reported only) finds no gain on either axis.
- **The current agent is behind the recalibrated rule** at equal FLOPs (−0.88 [−1.47, −0.27]; −1.58 at 0.20, level at 0.45) [E40]. The rule keeps more parameters in 16 of 18 pairs, so the gap carries the same trade. Against the current threshold rule the agent stays level (+0.02 at 0.45, +0.31 at 0.20). The agent was never given the recalibrated signal.
- **Inside DepGraph's own code the training-free rule matches DepGraph but does not lead** [E41] (Section 6.6).
- **Short-recovery selection gains only by picking larger plans** [E42] (Section 7.3, iii).

### 7.5b Results of 24–26 November (added 26 November)

- **The tie with the threshold rule is mostly rule selection.** The rule-blending agent puts 67–84 % of its weight on the threshold rule [E35]. Given the same recalibrated measurement as the rule, it is level with the recalibrated rule: +0.02 [−0.32, +0.40] (val −0.24, 10k −0.11) [E58]. The final agent, which plays that rule in only 7 % of its plans, is behind the recalibrated rule at the deep cut: −1.14 [−2.03, −0.09] at FLOPs 0.20 and +0.11 at 0.45; over both points −0.52 [−1.08, +0.18] (val −0.85, 10k −0.68) [E58].
- **Three learned choices do not beat a simple rule on unseen families.**
  - *Which plan, when recovery is short.* With a 2-epoch recovery as the product, picking the plan with the best 2-epoch validation accuracy beats the threshold rule by +3.97 [+2.04, +6.48] (val +4.18, 10k +4.08). A plan kind chosen leave-family-out gains +0.00, and plan and network features fitted leave-family-out pick −5.63 [−8.61, −3.03] below the rule [E59].
  - *Which size.* Choosing each network's own size has room: +1.85 [+1.10, +3.09] utility points over one common size on nine networks, and +3.52 against the common size chosen on the training networks. A ridge head fitted on 46 other networks is −0.59 [−1.16, −0.03] below a simple rule that copies the size chosen by the most similar training network [E60].
  - *Which plan after the full recovery, read early.* At equal size in the deep FLOPs groups, a 2-epoch probe's pick gains +0.06 [−0.29, +0.40], while the 100-epoch validation pick, which shares its run's luck, reads +0.78. A predictor built on the probe's trajectory (losses, gradient norm, weight movement) ranks plans worse than the probe's own final validation read (τ −0.120), and a predictor on cut-time information ranks them at τ 0.081, against 0.518 for the probe and 0.545 for a replicate run [E54, E57].
- **AMC's per-network search, run in our pipeline** (800 episodes; it reproduces AMC's ResNet-56 CIFAR-10 result within 0.03), on the nine held-out CIFAR-100 networks: the frozen final agent is above it by +0.97 [−0.02, +2.03] at FLOPs 0.45 (val +0.73, 10k +0.85) and +6.60 [+3.11, +10.74] at 0.20 (val +6.45, 10k +6.52). The threshold rule is about as far above it (+0.69 and +5.84), so per-network RL search is the weaker allocator here; this is not evidence that the agent learned more than the rule [E61].
- **CIFAR-10 rule screen** (26 networks, four rules, no agent): the threshold rule is above the FLOPs reference rule in the mean but not by the registered test (val +0.24, p = 0.085; TEST +0.22, 10k +0.23), so the reference rule stays. 9 of 26 networks have room above it, against 20 of 30 on CIFAR-100 [E62].
- **The equilibrium reward must be read after recovery.** A size chosen from the cut's accuracy loses 2.3–7.8 utility points on TEST under every form tried [E55].
- **ImageNet short recovery.** 3- and 9-epoch recoveries land 4.06 and 2.41 points below the 90-epoch results; 27 epochs land 1.05 below at about a third of the GPU time (four of six cells) [E56].

### 7.6 Threats to validity

- Pooled results rest on nine (or eight, or nine-of-thirteen) networks. One network often carries a pooled number, and we name it each time.
- Three seeds in development tables; five in the final tables.
- One recovery recipe, and allocation rankings may differ under short recoveries (Section 8.6).
- CIFAR-scale training only. Tiny-ImageNet and ImageNet are frozen-transfer evaluation.
- The agent reads the same measured sensitivity as the rules, so it saves no measurement.
- The state standardizer was fitted once on the 10-network CIFAR-10 and SVHN catalog, so the parameter agents that withhold families are not fully blind to them, and [H] probably neither is the held-out FLOPs agent (Section 6.8) [E49, E27].
- The thirteen CIFAR-100 networks of Table 4 have now been read twice and are development networks. The sealed set (Section 3.9) is the clean test.
- Some rows of Table 5 are single runs not yet written into the results ledger.

---

## 8. Open problems and research directions

The following are open questions, not plans already decided.

**8.1 Reward fidelity: predicting post-recovery accuracy cheaply.** Every reward closer to recovery improved the agent at deep budgets (Section 7.3, ii). Open directions:
- a learned surrogate that predicts post-recovery accuracy from cheap reads (the cut, BN-recalibrated and few-step accuracies, plus plan features), trained on the more than 11,000 evaluation records already logged [E42] and validated by rank fidelity on held-out networks;
- multi-fidelity training: cheap rewards for most plans, a 2-epoch recovery for a subset, as in bandits with costly arms;
- rank-based rewards (each plan's rank among its K siblings), which need only order, not calibration.

**8.2 Action-space granularity and reach.**
- The residual agent's trust region and the blending agent's four-rule envelope bound what can be learned. A mixture over rules *and* free per-group moves, with the bound learned or annealed, would let the agent leave the envelope where its reward is trustworthy.
- Finer cut grids at evaluation (the cutting routine reaches five keep rates per step) and per-group importance criteria are further axes.
- A proposer that generates diverse candidates, combined with a short-recovery screen, is the natural next experiment: the 2-epoch screen is accurate, and the proposer is what is missing.

**8.3 Richer state.**
- The agent sees one sensitivity number per group, measured at a 50 % cut on unrecalibrated probes.
- Damage curves, recalibrated probes, second-order or Fisher information, and post-sparsity measurements are candidates. The last is the one measurement change with a clean gain so far (Section 6.6).
- Whether an agent can learn from such signals what the rules cannot use is untested.

**8.4 Preference-aware, multi-objective formulation instead of a hard budget.**
- SPECTRA fixes one budget exactly. Its accuracy-limit mode bisects the size per network and lands at the same size as the rule [E29].
- The equal-FLOPs results show that what often decides the outcome is how many *parameters* to spend at a FLOPs budget (Sections 6.6 and 7.5).
- A preference-conditioned policy is a candidate: conditioned on a target trade-off between accuracy, parameters and FLOPs, as NEON conditioned on a tolerance and as recent preference-conditioned pruners do (Bai et al., 2026), and rewarded on the joint frontier. It would turn the parameters-for-FLOPs trade from a confound into the object of learning.

**8.5 Sequential decisions with backtracking.**
- One-shot plans cannot react to what the network becomes, and simple two-stage re-planning did not help [E23].
- An agent that proposes, observes a short recovery, and revises (or backtracks) is a middle ground between the failed step-by-step design and the one-shot bandit. It keeps whole-plan credit while allowing correction.
- The step-by-step failure (Section 5.1) defines what such a design must avoid: per-step rewards dominated by noise, and deferred credit.

**8.6 Short-recovery regimes.**
- All allocation evidence here follows a 100-epoch recovery, which saturates allocation at moderate budgets.
- If recovery is short, as when the budget for recovery is itself constrained, allocation should matter more, and the gap between rules and learned plans may widen.
- On ImageNet ResNet-50, 3- and 9-epoch recoveries land 4.06 and 2.41 points below the 90-epoch ones, and the plans spread more when recovery is short [E56]. On CIFAR-100, with a 2-epoch recovery as the product, a 2-epoch validation screen beats the rule by +3.97; it is a system result for that regime, not a better frozen policy [E59].

**8.7 Where to test.**
- A transfer proof can show learning only where better plans exist. The room screens (Section 6.6) show where that is on CIFAR-100: mostly at equal FLOPs (on 10 of the 20 networks with room the better plans keep about the rule's parameters; on VGG-BN and DenseNet-40 they keep 1.5–4.4× more), and at equal parameters mainly on the thin ResNets [E33].
- The sealed evaluation (Section 3.9) is the clean, once-only test of the final agent.

**8.8 Early-training signals (NAP2-v2): the test now being built (26 November).**
- After the full recovery, the place left where a learned component could beat the rule is the choice among plans of the same size. There the 100-epoch validation pick reads +0.73 above the threshold rule at equal FLOPs and +0.78 at equal size in the deep groups, upper references that share their run's luck, while the 2-epoch probe's final accuracy finds +0.06 [E57, E59].
- NAP2-v2's measurements, taken during a 2-epoch recovery of each candidate plan, are the inputs not yet tested: loss-curve estimators, confidence and churn on a fixed probe set, per-layer weight and gradient statistics, gradient signal-to-noise, movement away from the cut, and a kernel forecast at the cut.
- The test fits a judge on networks with known 100-epoch outcomes and scores it on held-out families: by rank correlation within groups of equal size, against the probe's own reading (τ 0.518) and a replicate run (0.545), and by its pick against the threshold rule.
- Two precursors are negative: a cut-time predictor (τ 0.081) and a small trajectory predictor that ranks worse than the probe's own reading [E54, E57].
- If the judge passes, it becomes the agent's training reward, aligned with the recovered result, or its decision-time picker among proposed plans.

---

## 9. Code map and reproduction

The repository is public: `https://github.com/IdoParetsky/SPECTRA-CompressionAgent` (branch `master`).

| File | What it does |
|---|---|
| [`a2c_agent_reinforce_runner.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/a2c_agent_reinforce_runner.py) | Entry point. In test mode it runs the plan trainer (`SPECTRA_PLAN_TRAIN=1`) or the plan-following evaluation (`SPECTRA_EVAL_POLICY=alloc`), then the final recovery and the TEST, val and 10k reads. It also holds the step-by-step actor-critic, kept as history |
| [`src/plan_agent.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/plan_agent.py) | The agent. Cost models (`ParamModel`, `FlopModel`); decoders (`decode`, `scale_decode`, `residual_decode`, `menu_decode`); the policy (`PlanPolicy`: encoder, per-token head, rule-blending head); `MaskedCut`; per-network rewards (`NetInstance.reward`); frozen inference (`plan_for_env`) |
| [`src/plan_trainer.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/plan_trainer.py) | The contextual-bandit REINFORCE trainer: configuration, sampling, group baseline, Adam, checkpoints, resumable state (`run`, `config`, `sigma_at`) |
| [`src/Model/StateEncoder.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/Model/StateEncoder.py) | The Transformer with the coupling bias (`SpectraStateEncoder`), the per-token MLP variant (`SpectraSetEncoder`) and the wide variant |
| [`src/BERTInputModeler.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/BERTInputModeler.py) | Builds the token matrix (the state) |
| [`NetworkFeatureExtraction/src/FeatureExtractors/ModelFeatureExtractor.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/NetworkFeatureExtraction/src/FeatureExtractors/ModelFeatureExtractor.py) | Topology, weight and activation statistics; group-cost and action-cost features |
| [`src/state_dump.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/state_dump.py) | The token layout (`layout`) and the token-to-group map (`token_rows`) |
| [`src/channel_groups.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/channel_groups.py) | `torch.fx` dependency tracing into coupled channel groups (`build_channel_groups`); coupling ids for the attention bias |
| [`src/group_sensitivity.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/group_sensitivity.py) | The planned groups (`group_plan`); the per-group sensitivity and cost measurement (`group_sensitivity`), with optional BN-recalibrated probes; the state's sensitivity columns (`normalise`) |
| [`src/pruning.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/pruning.py) | Group importance by L1 (`group_importance`), survivor selection, structural rebuild of producers, normalisations and consumers (`prune_group_structurally`) |
| [`src/alloc_walk.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/alloc_walk.py) | The training-free rules (`weights`, `curve_weights`, `plan_targets`); random best-of-N; the validation fallback; LAMP and global magnitude; grid rounding (`grid_round_widths`); the plan-following cutting routine (the name is historical) |
| [`src/amc_search.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/amc_search.py) | A port of AMC's per-network DDPG search, as a comparator (default off) |
| [`src/plan_proxies.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/plan_proxies.py), [`src/recovery_edits.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/recovery_edits.py) | Proxy scorers for the fidelity studies and the fallback's validation check; BN recalibration |
| [`src/fortify.py`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/src/fortify.py) | Feature flags and the evaluation contract: width floor, first-layer lock, final fine-tune settings, determinism |
| [`configs/database_offline_v6_p5b2.json`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/configs/database_offline_v6_p5b2.json) | The 10-network training catalog of the agents of record |
| [`scripts/spectra.sbatch`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/scripts/spectra.sbatch) | The Slurm launcher and its profiles |
| [`docs/paper/RESULTS_LEDGER.md`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/docs/paper/RESULTS_LEDGER.md), [`docs/SITTING_GPU_QUEUE.md`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/docs/SITTING_GPU_QUEUE.md) | The results record (numbered sections) and the pre-registration of every experiment, with its exact environment |
| [`docs/paper/tables/SPECTRA_results_summary_23NOV.csv`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/docs/paper/tables/SPECTRA_results_summary_23NOV.csv), [`…_long_23NOV.csv`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/docs/paper/tables/SPECTRA_results_summary_long_23NOV.csv) and [`SPECTRA_head_to_head_23NOV.csv`](https://github.com/IdoParetsky/SPECTRA-CompressionAgent/blob/master/docs/paper/tables/SPECTRA_head_to_head_23NOV.csv) | The side-by-side results table: one row per network and point. The long file adds val, 10k, honest, parameters and FLOPs kept, and each row's ledger section. The head-to-head file condenses each row to one agent, one rule and one published method |

**Reproducing one run.** Every run is the runner, configured by environment variables, on one GPU. The exact variable set of each registered experiment is in its queue row and its job log.

1. **Train an agent.** Set `SPECTRA_PLAN_TRAIN=1` and the trainer's knobs:
   - `SPECTRA_PLAN_K=8`, `SPECTRA_PLAN_INSTANCES=12000`;
   - `SPECTRA_PLAN_KAPPA=0.25,0.85` (direct) or `0.35,0.85` (FLOPs lines);
   - `SPECTRA_PLAN_PROXY=fs32` (direct) or `cut` (FLOPs lines);
   - `SPECTRA_PLAN_BUDGET=flops`, `SPECTRA_PLAN_RESIDUAL=sens`, `SPECTRA_PLAN_RESIDUAL_ZMAX=0.3` and `SPECTRA_PLAN_MIN_WIDTH=walk` for the residual agent;
   - additionally `SPECTRA_PLAN_MENU=sens,uniform,sens_cost,sens_thresh` with `SPECTRA_PLAN_MENU_SIGMA=2` for the rule-blending agent;
   - `SPECTRA_PLAN_NORM_ADV=0` for the direct agent of record (the code's default is 1, which the FLOPs lines use).

   Point the runner at the catalog and its datasets. The policy is written to `plan_agent/policy_latest.pt` in the run directory.
2. **Evaluate a frozen agent on a network at a budget.** Set:
   - `SPECTRA_EVAL_POLICY=alloc`, `SPECTRA_ALLOC_KIND=agent`, `SPECTRA_PLAN_AGENT=<policy file>`;
   - `SPECTRA_ALLOC_BUDGET=flops` for a FLOPs budget;
   - `SPECTRA_ALLOC_GRID_ROUND=1`, `SPECTRA_PLAN_MIN_WIDTH=walk`, `SPECTRA_VAL_FROM_TEST=1` (the halves);
   - the recovery of record: `SPECTRA_EVAL_FINAL_FT_EPOCHS=100`, `SPECTRA_EVAL_FINAL_FT_LR=0.1`, `SPECTRA_EVAL_FINAL_FT_BATCH=256`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`, `SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH=1`, `SPECTRA_FT_AUG_GPU=1`;
   - `SPECTRA_EVAL_FINAL_FT_ORIGIN=1` for the origin control (the honest change), and `SPECTRA_EVAL_FT_DETERMINISTIC=1`.

   The code's own defaults for the final fine-tune (lr 0.01, batch 128, keep the lowest-loss epoch) are *not* the recipe of record, so these must be set.
3. **Evaluate a rule.** As step 2, with `SPECTRA_ALLOC_KIND` set to `uniform`, `sens`, `sens_cost` or `sens_thresh`.

---

## Appendix A. Evidence index

Each tag maps to sections of `docs/paper/RESULTS_LEDGER.md`, the project's results record, or to the named table file. Every result there is marked PRELIM.

| Tag | Ledger sections | What it supports |
|---|---|---|
| E1 | §331, §259 | Why the step-by-step agent collapsed: reward variance decomposition, the 0.9-cut share rising 0.38 → 0.73, entropy 1.47 → about 0.45, 207 episodes in 4 days; the sensitivity rule +4.05 over mild on the same return (§259) |
| E2 | §248, §339 | The tenth agent, frozen, ties mild (−0.18) and its census shows only the 0.9 cut (§248); an earlier agent picks 0.9 at every free decision (§339) |
| E3 | §341 | One-network plan agent: +2.35 over mild (val +2.17, 10k +2.26); +0.11 over plain sensitivity; 1.24–1.37× sensitivity's FLOPs |
| E4 | §344 | Imitation probe: ρ +0.872 with sensitivity columns; +0.446 to +0.586 without; random-init encoder +0.841 against trained +0.801 |
| E5 | §346 | Catalog direct agent: +2.20 over mild on the held-out thin ResNet-56; +0.96 over the even cut on DepGraph's ResNet-56; +0.12 and +0.28 over plain sensitivity |
| E6 | §363 | Re-read with realizable arms: +1.10 over the even cut; no-ResNet fold holds (+0.95; −0.04 against the cost-aware rule); no-VGG fold fails (−0.33 on VGG-16) |
| E7 | §355 | Cost-aware rule against plain sensitivity: +0.47 / +0.39 / +0.44 at equal parameters, keeping 1.26–1.47× its FLOPs on the two ResNets; −0.33 at FLOPs 0.6 |
| E8 | §357, §361, §364, §365 | Direct FLOPs agent +0.51 over mild, −0.57 against the FLOPs rule; unbounded residual −1.89; with normalisation −1.54 (+0.44 against unnormalised, pooled over two networks); trust region +1.07 over that |
| E9 | §360 | One agent for both budgets: −0.52 against the catalog agent at equal parameters; −0.66 against mild at equal FLOPs |
| E10 | §367, §368, §372 | No headroom at moderate budgets: best alternative − rule −0.04 (FLOPs 0.6), +0.04 (FLOPs 0.45), +0.22 (parameters 0.6) |
| E11 | §340, §351, §375, §379 | Proxy fidelity: raw cut valid at 60 % (ρ +0.90 / +0.66) and at FLOPs 0.6 (+0.87); at 35 % parameters one epoch +0.60 (ceiling +0.73), cut +0.41, BN +0.45, on the fragile network −0.84 / −0.85 against +0.94; 32 steps +0.58 |
| E12 | §374, §376, §377, §378, §380, §383, §387 | Deep-budget collapse and the reward fixes: −5.23 / −16.84 on the fragile network with the raw cut; floor 0.25 −3.79; BN reward −0.17 pooled at 25 % (fragile −2.41); 32-step +0.1615 (fragile −0.93). Also §380: the cost-aware rule over the even cut at 25 % parameters, three seeds, +0.88 (val +1.03, 10k +0.95, honest +0.86), +2.1 to +3.4 on three networks, keeping 1.1–2.4× the even cut's FLOPs on eight of nine |
| E13 | §405 | Final FLOPs table, five seeds: −0.16 at 60 % and 45 %, with val / 10k / honest |
| E14 | §406 | Final parameter table, five seeds: +0.05 / +0.29 / +0.24 / −0.22; the fragile network +1.92 and −3.86 (one seed −15.40); rule − even cut +0.45 / +0.39 / +0.81 / +0.96 |
| E15 | §413, §416 | Below the training range: parameters −0.86 and −0.43; FLOPs −0.09, −0.23, −0.30 (8 networks) |
| E16 | §396 | Agent against best-of-100 random plans with its own reward: +0.51 and +0.45; search − rule −0.09 and −0.29 |
| E17 | §394, §398 | Scratch training equals inherited weights at matched compute; random channels level with L1 channels (+0.13, +0.23) |
| E18 | §395 | Measured speed: 0.93–1.07× on ResNet-56, 1.32–1.65× on VGG-16 |
| E19 | §393, §397, §398, §401 | Run-to-run noise 0.42 / 0.76 points; SD about 0.3 per run; deterministic recovery exact on five networks, 22–50 % slower |
| E20 | §407 | New architectures: −0.32 at 45 % (WRN-28-2 CIFAR-100 −1.88), −0.26 at 25 %; CIFAR-10 four −0.06 / −0.11 |
| E21 | §408 | Literature points, five seeds (Table 2), agent − rule with val / 10k / honest; level at nine points, below at parameters 0.125 (−0.30, 4 of 5 seeds negative); rule − even cut +0.01 to +0.60 at standard points and 1.9–3.3 toward 8×; at the parameter points the agent keeps 1.11–1.23×, 1.05–1.39× and 0.93–1.45× the rule's FLOPs |
| E22 | §409, §410, §431, §432 | Validation fallback: removes the collapse (+0.58 on the fragile network at 25 %); +0.09 / +0.08 on new architectures; ResNet-18 Tiny-ImageNet becomes level (−0.74 → −0.47); 28 other validation checks no better |
| E23 | §411 | Two-stage plans: +0.30 against one-shot (other eight −0.06) |
| E24 | §415 | Catalog size: one network not enough; three or five match ten at 45 % |
| E25 | §418 | CIFAR-100 and Tiny-ImageNet points (Table 3), five seeds: level at four, above at VGG-16-BN Tiny-ImageNet 0.48, below at ResNet-18 Tiny-ImageNet 0.48 (−0.74); parameters kept per arm |
| E26 | §419, §425, §433, §435, §438 | Runtime (Table 9): 7.7–8.5 min with the GPU data pipeline, 20.9–21.4 with a stock loader; training cost 2.9 and 20.0 GPU-hours, pay-back 4.8 / 33.7 networks; faster cutting routine 3.3–4.4× with identical outputs (seven runs over six networks) |
| E27 | §420, §429 | Agents trained without CIFAR-10, plain ResNets and VGG: FLOPs agent −0.04 against the agent of record; parameter agent −0.51 / −0.70; CIFAR-100-only retrain −0.91 / −1.55 |
| E28 | §422, §426 | Encoder ablations (per-group MLP, 8× Transformer, rule-inputs only) |
| E29 | §423 | Accuracy-limit mode: the agent's chosen size − the rule's −0.014 / −0.024 / −0.014 (level) |
| E31 | §427, §428, §430 | ImageNet ResNet-50 at 2.08× and 1.79× (Table 3), one seed, no verdict; full-precision re-read, paired intervals; the original 76.12 by our read (torchvision 76.13); raw changes −0.88 / −0.79 / −0.70 and −0.40 / −0.24 / −0.46; parameters kept; origin control +0.78 |
| E33 | §436, §437, §439 | Room screens on 30 CIFAR-100 networks; threshold rule above the reference rules (+0.56 FLOPs, +0.24 parameters) |
| E34 | §440 | Broad-catalog residual agent: −0.17 against the FLOPs rule, −0.89 against the threshold rule; threshold rule − FLOPs rule +0.64 on 13 networks |
| E35 | §441 | Rule-blending agent: +0.73 against the FLOPs rule, +0.00 against the threshold rule, −0.15 against best of four; shares; parameters kept 0.436; training 7.9–8.2 h, σ_u 2, initial anchor logit 6 |
| E36 | §442, §443, §444 | LAMP and global magnitude screen (passed its rule on the count: validation-best on 4 of 9 networks at 0.45, +0.18 on validation; none at 0.20); per-network pick (DROP); perfect chooser +0.05 at 0.45 beyond the pick; distillation (DROP) |
| E37 | §445 | Families withheld: −0.35 against the agent trained with them; −0.52 against the FLOPs rule |
| E38 | §446, §447 | Sealed-set preparation (49 of 50 networks enter); DepGraph publishes no parameter counts; 18 of 38 sealed networks have published points |
| E39 | §448, §451 | Recalibrated threshold rule: +0.46 at equal FLOPs with 17–32 % more parameters; −0.93 at equal parameters |
| E40 | §449 | Rule-blending agent against the recalibrated rule: −0.88 (−1.58 at 0.20, −0.17 at 0.45); against the current threshold rule +0.17 over both points |
| E41 | §450, §453 | Inside DepGraph's code: rule widths without sparsity training +0.47 (10k); re-measured −0.03; aimed sparsity −0.13; ResNet-56 +0.21 |
| E42 | §452 | Short-recovery selection: +0.42 with 22 % more parameters; +0.00 at equal size; the 100-epoch validation pick held to equal size +0.39 (FLOPs groups) and +0.26 (parameter groups), an upper reference; τ +0.52 against ceiling +0.55; regret by kind (threshold rule 0.80, agent 2.87) |
| E43 | `docs/paper/tables/SPECTRA_results_summary_long_23NOV.csv`, rows whose ledger column reads "transplants 21 Nov" (the 21 November runs of the queue row for the AMC comparison, part B, with its amendment 4 on the first-convolution lock); not yet ledgered. The same file gives the parameter and FLOPs columns of Tables 2–3 | Table 5; parameter changes in Tables 2–3; the parameter and FLOPs ratios between arms quoted at the literature points |
| E44 | §388, §389, §400, §414, §417, §421 | Training length flat; CIFAR-100 in the catalog (+2.09 with BN, −0.57 with 32-step); one-epoch warm-start at five seeds; range 10–85 % from scratch −0.33 to −0.55 |
| E45 | §330, §334, §342, §348 | Recovery recipe: keep-last; 75–100 epochs suffice; one recipe matches per-family recipes (−0.08 over 16 pairs) |
| E46 | §386 | Three-seed score ensemble: fragile network −3.61 |
| E47 | §402, §403 | SVHN and Fashion-MNIST: the cost-aware rule beats the even cut on 1 of 20 networks at 25 % |
| E48 | §353, §359 | Families no catalog holds: the rule is level with the even cut on RepVGG; ShuffleNetV2 not cleanly cuttable |
| E49 | §354, §358 | Leave-one-family-out training lines |
| E50 | §362, §382, §385, §391, §392, §399 | Three-seed single-network verdicts at DepGraph's, VGG-16's and ResNet-110's points; recovery lifts the unpruned ResNet-56 by +0.64 to +1.38; the FLOPs rule's 10k +0.56 / +0.24 at DepGraph's two points, honest −0.70 at 2.57× (§382); at 2.11× it keeps 0.53–0.60 of the parameters, against 0.508 for DepGraph's released model (§362) |
| E51 | §370 | The first catalog agent at equal parameters on the nine held-out networks, three seeds: level with the cost-aware rule (−0.01); against the plain sensitivity rule +0.52, +0.49 and +0.80 on the three networks where that rule is at or below the even cut, keeping 1.4–1.85× its FLOPs |
| E52 | §234 | One run of DepGraph's released pipeline on our GPU at VGG-19 CIFAR-100: −2.97 (landed at 9.02× by its own log; FLOPs kept 0.110 on our counter) |
| E53 | `docs/paper/tables/SPECTRA_head_to_head_23NOV.csv` (built from the ledger sections and the 21 November runs behind E21, E25, E31, E41 and E43) | Table 6: per point, the agent, the rule and the strongest published method with accuracy, FLOPs and parameter changes; published bases and counters in its notes |
| E54 | §454 | Cut-time predictor of the recovered outcome: τ +0.081 [+0.034, +0.132] on held-out families (269 groups, 52 networks); size alone +0.225; the cut +0.156; the 2-epoch probe +0.518; a replicate run +0.545 |
| E55 | §455, §457, §459 | Equilibrium reward: a size chosen from the cut loses 2.3–7.8 utility points on TEST under every form; forms compared on calibrated grids |
| E56 | §458 | ImageNet ResNet-50 short recovery: 3 and 9 epochs 4.06 and 2.41 below 90 epochs; 27 epochs 1.05 below (four of six cells) |
| E57 | §460, §461, §462 | 2-epoch probe at equal size, deep FLOPs groups: +0.06 [−0.29, +0.40] against +0.78 for the in-run 100-epoch val pick; the trajectory predictor −0.120 in τ against the probe's val read; the probe's size choice NO-GO |
| E58 | §463, §464 | With the recalibrated measurement: the rule-blending agent level with the recalibrated rule (+0.02); the final agent −0.52 (−1.14 at 0.20, +0.11 at 0.45); the threshold rule in 72 % and 7 % of their plans |
| E59 | §465, §467 | Short-recovery regime: the 2-epoch val pick +3.97 [+2.04, +6.48] above the threshold rule; a leave-family-out kind +0.00; learned features −5.63 [−8.61, −3.03]; the in-run 100-epoch val pick +0.73 at equal FLOPs |
| E60 | §466, §468 | Size per network: room +1.85 [+1.10, +3.09] (held-out prices) and +3.52 (against the training networks' common size); the learned head −0.59 [−1.16, −0.03] against the nearest-network rule |
| E61 | §469 | AMC's per-network search in our pipeline on 9 held-out CIFAR-100 networks: the final agent +0.97 [−0.02, +2.03] and +6.60 [+3.11, +10.74] above it at FLOPs 0.45 and 0.20; the threshold rule +0.69 and +5.84 |
| E62 | §470 | CIFAR-10 rule screen (26 networks): threshold rule − FLOPs reference rule val +0.24, p 0.085 (LEVEL); room on 9 of 26 networks |

(Tags E30 and E32 are not used.)

---

## Appendix B. Glossary

- **Allocation**: the keep fraction of each coupled channel group, given a global budget.
- **Coupled channel group**: the layers that must keep the same channels: producers joined by an addition, segments of a concatenation, depthwise ties, and their normalisations and consumers.
- **Budget, κ**: the kept fraction of parameters or FLOPs (MACs).
- **Reference rule (fixed in advance)**: the cost-aware sensitivity rule at equal parameters, and the FLOPs sensitivity rule with the width floor at equal FLOPs.
- **Current threshold rule** (or simply the threshold rule): one global threshold on per-group damage curves (Li et al., 2017, as NNI and PaddleSlim run it). Its *recalibrated* variant re-estimates BN statistics on every probe first.
- **Agents of record**: the parameter (direct) and FLOPs (residual) agents whose results Tables 1–3 report; Table 6 calls them the earlier agents. The **current agent** is the later rule-blending agent.
- **Verdict**: the registered outcome of a comparison (above, level or below; gain, level or loss), fixed by a rule written before the run.
- **Equal FLOPs, equal parameters**: the axis a comparison holds fixed; the other axis is free.
- **Even cut**: the same keep fraction for every group.
- **Mild heuristic**: keep 90 % of each layer, repeatedly; the policy every step-by-step agent collapsed to.
- **Direct, residual and rule-blending agents**: the three action parameterizations of Section 5.5.
- **Instance**: one training draw of a network and a budget; eight plans are sampled per instance.
- **Raw cut, BN-recalibrated cut, 32-step fine-tune, one-epoch fine-tune**: the reward proxies of Section 5.7.
- **TEST, val, 10k, honest**: the four reads of Section 3.6.
- **Room**: some plan's validation result beats the reference rule's by ≥ 0.3 points on a network.
- **Best of four**: per network, the rule with the best validation result among four agent-free plans; an envelope, not a rule.
- **Fragile network**: the thin ResNet-20 at width 8 on CIFAR-100.
- **Validation fallback**: plan with both the agent and the rule, score both by one epoch of fine-tuning on the validation half, and keep the better. A labelled product variant, not the agent.
- **Sealed set**: fresh checkpoints of the literature's families, read once after the final agent is frozen.

---

## References

- Bai, L., Tabia, H., & Santos-Rodriguez, R. (2026). HiPP-Prune: Hierarchical preference-conditioned structured pruning for vision-language models. arXiv:2603.06270.
- Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? *MLSys*.
- Chen, Z., Niu, J., Xie, L., Liu, X., Wei, L., & Tian, Q. (2020). Network adjustment: Channel search guided by FLOPs utilization ratio. *CVPR*.
- Chin, T.-W., Ding, R., Zhang, C., & Marculescu, D. (2020). Towards efficient model compression via learned global ranking (LeGR). *CVPR*.
- Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: Towards any structural pruning. *CVPR*.
- Frantar, E., & Alistarh, D. (2022). SPDY: Accurate pruning with speedup guarantees. *ICML*.
- Ghimire, D., Kil, D., Jeong, S., Park, J., & Kim, S.-h. (2026). One-cycle structured pruning via stability-driven subnetwork search (OCSPruner). *WACV*.
- He, Y., Lin, J., Liu, Z., Wang, H., Li, L.-J., & Han, S. (2018). AMC: AutoML for model compression and acceleration on mobile devices. *ECCV*.
- He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019). Filter pruning via geometric median (FPGM). *CVPR*.
- Hirsch, L., & Katz, G. (2022). Multi-objective pruning of dense neural networks using deep reinforcement learning (NEON). *Information Sciences*, 610, 381–400.
- Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 REINFORCE samples, get a baseline for free! *ICLR Workshop*.
- Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *NeurIPS*.
- Le, D. H., & Hua, B.-S. (2021). Network pruning that matters: A case study on retraining variants. *ICLR*.
- Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2021). Layer-adaptive sparsity for the magnitude-based pruning (LAMP). *ICLR*.
- Lee, S., & Song, B. C. (2022). Ensemble knowledge guided sub-network search and fine-tuning for filter pruning. *ECCV*.
- Li, B., Wu, B., Su, J., & Wang, G. (2020). EagleEye: Fast sub-net evaluation for efficient neural network pruning. *ECCV*.
- Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient ConvNets. *ICLR*.
- Li, Y., Adamczewski, K., Li, W., Gu, S., Timofte, R., & Van Gool, L. (2022). Revisiting random channel pruning for neural network compression. *CVPR*.
- Lin, M., Ji, R., Wang, Y., Zhang, Y., Zhang, B., Tian, Y., & Shao, L. (2020). HRank: Filter pruning using high-rank feature map. *CVPR*.
- Liu, Y., Wang, X., & Zhang, M. (2025). Meta pruning via graph metanetworks. arXiv:2506.12041.
- Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K.-T., & Sun, J. (2019a). MetaPruning: Meta learning for automatic neural network channel pruning. *ICCV*.
- Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019b). Rethinking the value of network pruning. *ICLR*.
- Shao, Z., et al. (2024). DeepSeekMath (introduces GRPO). arXiv:2402.03300.
- Sui, Y., Yin, M., Xie, Y., Phan, H., Zonouz, S., & Yuan, B. (2021). CHIP: CHannel independence-based pruning for compact neural networks. *NeurIPS*.
- Tiwari, R., Bamba, U., Chavan, A., & Gupta, D. K. (2021). ChipNet: Budget-aware pruning with Heaviside continuous approximations. *ICLR*.
- Wang, X., Rachwan, J., Günnemann, S., & Charpentier, B. (2024). Structurally Prune Anything (SPA). arXiv:2403.18955.
- Wang, Y., Zhang, X., Xie, L., Zhou, J., Su, H., Zhang, B., & Hu, X. (2020). Pruning from scratch. *AAAI*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning (REINFORCE). *Machine Learning*, 8, 229–256.
- Yu, S., Mazaheri, A., & Jannesari, A. (2022). Topology-aware network pruning using multi-stage graph embedding and reinforcement learning (GNN-RL). *ICML*.
- Yuan, X., et al. (2025). FastForward Pruning: Efficient LLM pruning via single-step reinforcement learning. arXiv:2511.18977.

Other published rows in Tables 2, 3 and 6 (DECORE, LFPC, SFP, TAS, TPP, GReg, EigenDamage, SOSP-H, Jiang et al., RL-Pruner) are cited with full entries in the project draft's reference list (`docs/paper/SPECTRA_draft.md`, References).
