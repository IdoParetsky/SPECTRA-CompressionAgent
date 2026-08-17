# SPECTRA — paper draft (NEON format)

Working title: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Ido Paretsky · advisor Dr. Gilad Katz · Ben-Gurion University  
Predecessor: Hirsch & Katz, *Information Sciences* 2022 (NEON) [1].  
**Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md).** Do not paste a job summary mean here.

Status of this file: **DRAFT**. Experiment freeze 15 Sep 2026; paper 30 Sep.  
Section order matches NEON [1]: Introduction → Related Work → Approach → Evaluation → Results → Discussion → Conclusions.  
**Last restamp: 18 Aug 2026 ~02:30 IDT.**

### Durable snapshot — 18 Aug 2026 ~02:30 IDT (do not rely on chat)

Write new TEST here and in the ledger in the same turn. This block is the memory.

- **CIFAR-10 architecture transfer (C1–C5, C10 FLOP-floor):** LOCKED. Similar + unlike inside τ except skinny-deep ResNet-56 at 70% params (−15.9/−16.2/−17.2). FLOP floor 0.70 puts that net inside τ at ~91% params / 70% FLOPs (−8.9/−9.2/−9.6).
- **Claim C9 (frozen 10-net → CIFAR-100):** LOCKED mixed. VGG-16 BN −7.5/−7.8/−7.3 inside τ. Thin ResNets and RepVGG-A0 miss. ShuffleNet-v2×1 no TEST (grouped-conv crash — code, not Slurm). Ledger §21.
- **Digit-MNIST LeNet (held-out dataset):** three-seed TEST **gain** +2.8/+2.7/+2.9 pp. Toy net. Ledger §22.
- **SVHN r20-w8 (held-out width, dataset was in the 10-net train mix):** −2.0/−1.5/−2.0 inside τ. Not in-catalog r20-w16. Ledger §23.
- **C8 / 24-net:** still running. s42 in-catalog eval_test; s44 ~ep 383/720. Held-out afterok not started. Do not quote in-catalog eval.
- **Queued 18 Aug (not TEST):** ImageNet MobileNet-v2 frozen 3-ep s42; C10-thin param floor 0.80 (no FLOP floor); C100 residual SGD-recipe re-eval; C100 ShuffleNet rerun after dummy-forward rollback. Ledger §8.
- **C100 odds:** miss is dataset × residual/RepVGG family × Adam-40 FT, not a missing-C100-in-train bug and not “ResNets cannot prune” (same families work on CIFAR-10). Next lever is the C6 SGD recipe on the frozen actor, not a second C100 DRL.

Citation numbers **[1]–[78]** are those of the August 2024 thesis proposal, kept wherever the claim still holds. **[79]–[92]** are papers from the August 2026 literature survey (DepGraph was already [12] in the proposal). Tags: **LOCKED** / **DRAFT** / **TBD** / **CLAIM** — see [README.md](README.md).

**What this draft does *not* inherit from the proposal.** ImageNet [71] and Places365 [75] are not in the DRL fine-tune loop. Default keep-rates are `{1.0, 0.9, 0.8}`, not `{0.7, 0.6}`. Fine-tune is full-net, not layer-only. The reward is NEON’s preference function with *realized* param/FLOP credit, not the proposal’s skip-connection / filter-count cube. Frozen BERT is an ablation, not the default encoder. Channel grouping is environment bookkeeping, not a novelty claim (that layer is occupied by DepGraph [12] and SPA [81]).

---

## Abstract — DRAFT

Convolutional networks remain the workhorse of real-time vision, but their computational cost limits deployment where connectivity and compute are scarce [2, 5]. Pruning reduces that cost [6, 10]; *structured* pruning (channels, filters, feature maps) actually shrinks GPU work, unlike unstructured weight zeros [7, 11, 13]. Most structured methods, including recent architecture-agnostic grouping engines [12, 81], still *solve one target network at a time*.

SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) extends NEON [1] from dense DNNs to CNNs: one offline actor-critic, trained on a catalog of architectures and datasets, then applied to unseen checkpoints without retraining the agent. A user-set accuracy budget τ (NEON’s preference-aware reward) trades size against accuracy. Unlike NEON, the environment *rebuilds* Conv2d/BatchNorm groups (residuals [3], DenseNet concat [4], depthwise ties) so reported parameter and FLOP ratios are real shape changes.

On CIFAR-10 [73], the 10-net agent keeps similar-family and unlike-family held-out networks inside a 10 pp budget, except skinny-deep ResNets. Learned schedules tie greedy filter ranking [18] on easy nets and beat size-matched greedy on the hard similar ResNet-56. Encoder capacity (BERT / wider / set) did not move that hard net; catalog diversity did. CIFAR-100 is a recoverability problem: VGG-11 BN [76] admits a ~10% structured cut with a test-set *gain* under a long SGD recipe; residual, DenseNet, and MobileNet families have not yet shown a comparable cut. The same frozen 10-net agent, applied to CIFAR-100 with no extra agent training, keeps VGG-16 BN inside τ=10 (−7.5 / −7.8 / −7.3 pp) and misses on thin ResNets and RepVGG-A0; ShuffleNet-v2×1 produced no TEST row (grouped-conv crash). Digit-MNIST LeNet, a held-out dataset, is a three-seed TEST gain (+2.8 / +2.7 / +2.9 pp). SVHN ResNet-20 width 8 (new width; SVHN was already in the train mix) stays inside τ (−2.0 / −1.5 / −2.0 pp). **[C1–C7 LOCKED; C8 TBD; C9 LOCKED mixed; MNIST + SVHN-w8 three-seed LOCKED]**

---

## 1. Introduction — DRAFT

Convolutional Neural Networks (CNNs) have transformed computer vision — classification, detection, segmentation [2] — by stacking learned filters. Residual Networks [3] and DenseNets [4] made very deep stacks trainable via skip / dense connectivity. The cost of that accuracy is compute and memory. For edge and real-time settings the relevant question is not a leaderboard delta but a *user-set* accuracy budget: how much size can we drop without crossing τ.

CNN pruning [6, 10] removes redundant components. Magnitude, channel, and filter pruning [7, 11, 12, 18] work well on the network they were tuned for, and usually need another search when the architecture or dataset changes. Unstructured zeros [13–16] look strong on paper and do not reduce commodity-GPU inference cost. Structured cuts do, but residual adds and grouped convolutions make “keep 80% of filters” a different question from “keep 80% of parameters.”

NEON [1] showed that one preference-aware DRL agent, trained offline on many *dense* networks, can prune unseen dense nets without per-network retraining, and that a user can state the prune–accuracy trade-off in the reward. Applying that dense agent to *flattened* image data was the proposal’s negative control: modest size cuts and poor accuracy, worst on CIFAR-100 [73] (proposal §4). SPECTRA is the CNN-native answer to that gap — not a claim that NEON failed, a claim that image CNNs need a structured environment, a CNN-aware state, and a recoverability check before a dataset joins the train catalog.

**Contributions (LOCKED as intent; numbers in §5 / ledger):**

1. A structured CNN environment whose compression credit matches rebuilt layer shapes (residuals [3], DenseNet concat [4], grouped / depthwise ties). Grouping is infrastructure in the sense of DepGraph [12]; the thesis claim is the *policy*, not a new dependency algorithm.
2. A small Transformer state encoder [35] trained with A2C (one token per layer, coupling-aware attention). Frozen BERT remains an ablation; encoder capacity did not fix skinny-deep ResNets (ledger §16).
3. NEON’s preference-aware reward [1], with τ = 10 pp, scored on *realized* param/FLOP ratios. Same-loop greedy / mild / random rate-pickers (filter ranking after Li et al. [18]) so “beats greedy” is a schedule comparison, not a different importance criterion.
4. An offline train / similar / unlike protocol. On CIFAR-10, family and width transfer hold except skinny-deep ResNets. CIFAR-100 is limited by fine-tuning recoverability, not by “C100 missing from the 10-net train set.”

NEON’s 28 tabular datasets are not re-run. ImageNet overnight fine-tune is out of the loop (**CLAIM**, freeze 15 Sep). ViT / DeiT are out of scope [42].

---

## 2. Related Work — DRAFT

This section follows NEON’s related-work order [1]: reinforcement learning, neural-network pruning, then neural architecture search. CNN background and structured-CNN methods sit inside §2.2. The arena is generic *offline* DRL for structured CNNs with a user τ, versus same-loop rate-pickers. It is not ImageNet ResNet-50 versus 2024 grouping engines.

Two kinds of generality are both real and sit on different layers. **Mechanism-level generality** — one *tool* that can prune many architectures — is occupied well by DepGraph [12] and SPA [81]. **Policy-level generality** — one *agent* that transfers across nets and datasets under τ — is NEON’s idea [1], moved to CNNs. No 2023–2026 paper in the survey closes that lane.

### 2.1 Reinforcement learning

“Reinforcement Learning (RL) is the problem faced by an agent that learns behavior through trial-and-error interactions with a dynamic environment” [46]. Applications range from robotics [47] and games [48] to routing [49] and dialogue [50, 51]. At step *t* the agent observes state *s_t*, picks *a_t* ∈ *A*, the environment returns *s_{t+1}* and a scalar reward [1].

Deep RL combines RL with deep networks [52]. SPECTRA, like NEON, uses an on-policy policy-gradient family. NEON’s derivation is REINFORCE [53]; the implementation is A2C. The policy update (proposal Eq. 1) is

*h_{t+1} = h_t + α · G_t · ∇ log p(a_t | s_t; h_t)*,

with *G_t* the return from *t*. AMC [79] is the canonical *CNN* RL compressor: a DDPG controller searches layer-wise compression for a *given* MobileNet / VGG. SPECTRA’s contrast is the same as NEON vs per-task RL: the actor is trained once on a catalog and frozen at eval. Lookahead-search RL for channel pruning [57] is likewise a per-target search, not an offline multi-net agent.

### 2.2 Neural network pruning

#### 2.2.1 From dense nets to CNNs

Neural-network pruning dates to the early 1990s [6]. Optimal Brain Damage [8] and Optimal Brain Surgeon [9] used second-order criteria at high Hessian cost [10]. Modern CNNs spend most inference time in convolutions, so removing whole feature maps / filters is the practical lever [7, 10, 18].

ResNets [3] add identity skips; DenseNets [4] concatenate all preceding feature maps. Those links couple channels: pruning a filter in one layer forces aligned cuts in add / concat partners. That is why a CNN pruner cannot treat layers as independent dense maps the way NEON could.

#### 2.2.2 Structured vs unstructured

Structured pruning operates at channels, filters, or feature maps and reduces the matrices the GPU actually multiplies [7, 11, 12]. Unstructured pruning zeros individual weights [13–15] and typically wins compression *ratio* [16] without winning inference on dense kernels. SPECTRA is structured only.

#### 2.2.3 Global vs local; generalizability

Benchmarking, from the proposal, still has two axes. **Global optimality:** does the method see the whole net [17–26], or only one or two successive layers [8, 27]? **Generalizability:** can it prune a previously unseen architecture and dataset without another controller training? NEON [1] is the dense-DNN existence proof of the second axis, with a user trade-off later echoed in OCNNA [28] and in interactive plans such as CNNPruner [29]. Filter pruning for efficient ConvNets [18] is the importance ranking SPECTRA *uses inside each layer*; the DRL policy only chooses the *rate*. Auto-balanced filter pruning [19], GDP [20], lottery tickets [21], importance estimation [22], Gate Decorator [23], layer-adaptive magnitude [24], manifold-regularized pruning [25], and ThiNet-style algorithms [26] are global or near-global *per-model* methods. Lookahead magnitude pruning [27] is a far-sighted alternative of the same ranking family — conceptually related to SPECTRA’s look-ahead *greedy* control, not to the learned actor.

Proposal Table 1 still has the right axes. CONVNETS [18], AFP [19], GDP [20], DeepPruningES [30], FPAC [37], multi-layer residual compression [34], and DepGraph [12] are automatic and often global; none combine offline multi-architecture / multi-dataset training with an explicit user τ the way NEON / SPECTRA do. DepGraph’s and SPA’s [81] “adaptability” is *grouping* adaptability, not policy transfer. AMC [79] has a resource budget on a *given* net, not a transferred τ.

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
| SPECTRA | Y | Y | Y (policy) | Y | Y (user τ) |

#### 2.2.4 Mechanism-level generality: DepGraph and SPA

DepGraph [12] (CVPR 2023) models layer I/O dependencies and prunes coupled groups with a sparse-training + norm criterion, on CNNs, RNNs, GNNs, and Transformers. SPECTRA should thank it, then differentiate: skip/concat coupling and structured cuts are table stakes; they are not the thesis contribution. DepGraph *solves a given model*. It does not train an offline agent, does not encode τ as a reward, and does not transfer a schedule to an unseen net without re-solving.

SPA [81] (2024) attacks three practical barriers: coupled channels that differ by architecture, pruning at different training stages, and tools locked to one framework. It uses ONNX graphs and group-level importance (OBSPA is a post-training, calibration-light variant) and positions itself beyond DepGraph / OTO-v2. SPA’s “any architecture / any framework / any time” is about the *tool*. SPECTRA’s “any architecture / any dataset” is about a *transferred policy*. Using the same slogan would invite a fair referee objection. “Any time” in SPA means training stage, not “any previously unseen dataset without re-solving.”

HESSO [83] (OTO-lineage sparse optimizer) and Auto-Train-Once [82] (controller-guided prune-from-scratch, CVPR 2024) make train+prune less manual for a *given* DNN. FreePrune [84] automates across pruning granularities with training-free scores. All three are pipeline / criterion generality, not a cross-dataset DRL policy.

#### 2.2.5 2025–2026: still per-model

CNN structured pruning did not cool off. Metaheuristic channel search [85], comparative encodings for search-based CNN prune [86], learnable per-filter masks [87], differentiable attention-guided channel pruning [88], SVD-driven filter importance [89], flow-guided multi-architecture scores [90], and DepGraph-style coupling plus spectral entropy [91] all improve a criterion, a search, or a deployment pipeline for a given network. That raises the bar for “yet another pruning score.” It does not close the NEON→CNN offline-agent lane. Adjacent 2025–26 work — DualPrune [92], knowledge-distillation plus structured lightweight CNNs, agentic prune/quant pipelines — sits in the same per-model bucket. RL papers with “pruning” in the title in this window are mostly domain RL, not a generic CNN offline agent; the closest DRL neighbor remains AMC [79].

Other proposal-era CNN-pruning mechanisms still worth a clause: evolution strategy [30, 31], clustering / swarm [32], auto graph encoder-decoder [33], Transformer-related pruning under pretrain–finetune [36], FPAC [37], DETR pruning [38], global channel attention [39]. ConvNeXt [40], LLM pruning [41], ViT [42–44], and VAN [45] are out of SPECTRA’s structured-CNN loop.

MetaPruning [80] learns a PruningNet for channel configs of a target family — automatic, not an offline cross-dataset agent.

### 2.3 Neural architecture search

NAS explores automatic architecture design: search space, optimization method, candidate evaluation [1]. RL was applied to NAS in [54] and followed by [55–57]; evolution [58, 59], SMBO [60], and gradient-based search [61, 62] followed. Some works use CNN pruning to help NAS [63–65]; others use NAS to help CNN pruning [66–68]; a few prune or search at initialization without data [69, 70]. SPECTRA is **not** NAS. The agent does not invent a new topology; it walks an existing CNN and picks keep-rates. Keep this subsection short, as NEON did.

---

## 3. Approach — DRAFT (method LOCKED; equations TBD)

NEON §3 order: Overview, State, Action, Reward, Architecture, Training, Complexity.

### 3.1 Overview — DRAFT

SPECTRA uses a DRL agent to compress previously unseen convolutional architectures on datasets the *agent* need not have trained on, without additional *agent* training [1]. Each step: pick a keep-rate for the current channel group, structurally rebuild coupled layers, full-net fine-tune, observe accuracy and size, receive a NEON-style reward. An episode walks the network (optional extra passes). Training mixes many checkpoints; evaluation loads a frozen actor.

NEON’s dense working assumptions do not all transfer [proposal §3.1]. Skewness / kurtosis of fully-connected weights are a poor CNN prior; layer width is not a single integer when stride, padding, and concatenations set feature-map size; ImageNet-scale 224×224×3 [71] is a different regime from CIFAR 32×32×3 [73]. SPECTRA therefore replaces NEON’s fixed maps with a variable-length sequence of per-layer tokens and replaces “mask this dense layer” with “rebuild this Conv2d group.”

**Procedure (what we actually run):**

1. The agent sees the CNN plus the index of the group under compression, including skip / concat partners so identity paths stay aligned [3, 4].
2. A CNN-aware state encoder produces one token per layer (meta-features: type, width, depth, filter-L1 shape, action cost, coupling id).
3. The policy samples a keep-rate from `{1.0, 0.9, 0.8}` (C10). Stem and width-1 layers cannot take fake prunes (Fortify: a requested cut must drop at least one channel).
4. Coupled layers are resized together. Masked (non-structural) edits get no size credit. Classifier outputs are never shrunk.
5. The whole net is fine-tuned (40 epochs, patience 10 on C10). Layer-only freeze is **not** used: it produced 0/32 recoveries (ledger §12).
6. Reward uses realized param/FLOP ratios and Δacc vs τ.
7. The agent continues through the architecture. Skip topology is handled by grouping, not by a separate non-sequential “NEON on FC only” pass. Fully-connected tails are ordinary Linear groups, not a nested NEON black box.

### 3.2 State space — DRAFT

The state is a generic, architecture-agnostic *encoder*, with architecture-specific *contents* (this net’s depth, types, statistics, coupling graph) — the same distinction as NEON’s maps [1].

Default: `SpectraStateEncoder` (~2–3M), trained with A2C, one token per layer, learned target marker, attention bias on channel-coupling ids, filter-L1 shape statistics (moments and quantiles; not per-filter tokens, which do not scale [proposal optional A.2]), action-cost features, database-wide z-score. The Transformer inductive bias [35] is used as a *sequence encoder over layers*, not as a Vision Transformer [42].

Frozen `bert-base-uncased` remains `SPECTRA_STATE_ENCODER=bert`. **CLAIM:** encoder A/B (jobs 20140552/54/55/56) did not separate on r56-w4 (TEST −23.5 to −24.5 pp); BERT was slightly kinder only on easy r20-w2 (−2.1 vs −5.2). Do not reopen BERT as the default. See ledger §16.

### 3.3 Action space — DRAFT

Discrete keep-rates: the ratio of pruned to original *group width*. On C10 the menu is `{1.0, 0.9, 0.8}`. The proposal listed `{1.0, 0.9, 0.8, 0.7, 0.6}` [proposal §3.3]. Rates ≤0.7 stay out of the default menu: unconstrained eval hit ~17% params at about −40 pp on thin ResNet-56, and greedy 0.8 is already −24 pp on that net (ledger §13). C100 DRL used `{1.0, 0.98, 0.95}` because even 0.8 is not a recoverable 20% *parameter* cut on residuals.

After the rate is chosen, SPECTRA *replaces* the group (rebuilds Conv2d / BN / coupled consumers), it does not leave masked zeros. Eval identity-pads once parameters kept hit 0.70 so thin nets are not scored at 17% size. Training still uses τ, not that floor. We do not prune pooling operators as a first-class action (proposal §3.2 listed pooling window / stride as future knobs).

### 3.4 Reward function — DRAFT

SPECTRA uses NEON’s preference-aware reward [1]: linear credit for size cut while Δacc stays inside the user bound *C* (we write τ), a steep penalty when the bound is crossed, and a bonus when accuracy rises. Compression credit is **realized** param/FLOP ratio after rebuild, not the menu rate, and not the proposal’s *CF_ratio* / *SC_ratio* cube (proposal Eq. 2). Masked edits get no size credit. That is the CNN-specific change that *stood*: residual groups lie if one scores the nominal 0.8.

Skip-connection *preservation as an extra reward term* was not needed once grouping forbids breaking an add; keeping a skip is a hard constraint, not a bonus.

### 3.5 SPECTRA architecture — DRAFT

A2C agent + `NetworkEnv` + `torch.fx` channel groups (residual adds [3], DenseNet concat [4], ShuffleNet / depthwise). This is DepGraph-class bookkeeping [12] inside the env. The actor is small relative to frozen BERT.

### 3.6 Training process — DRAFT

CNNs cannot be sampled as freely as NEON’s random dense nets (proposal §3.5). The pool is variants of established families: VGG [76], ResNet [3], DenseNet [4], MobileNet [78], plus held-out unlike families (ShuffleNet, RepVGG) at eval. All train nets are pretrained to competence on their dataset, then the agent learns a pruning policy.

**Offline catalog (LOCKED 10-net):** CIFAR-10, SVHN [74], Fashion-MNIST [72] (32×32 RGB). No C100 until a family recovers a real 2–5% cut. No ImageNet [71] overnight FT. Seeds 42/43/44. 24-net catalog running (**PRELIM**). Manifest: `configs/offline_pools_manifest.json` (287 files mapped; do not train on every file).

### 3.7 Complexity — TBD

Wall-clock: DenseNet eval dominates; 7-day Slurm wall; training stops on a timer, eval still runs; `afterok` successors. Image datasets and deeper CNNs are substantially heavier than NEON’s tabular DNNs (proposal limitation; still true).

---

## 4. Evaluation — DRAFT

NEON §4: algorithms, setup, results, discussion. SPECTRA splits setup here and numbers in §5.

### 4.1 Compared methods — DRAFT

| Method | What it chooses | Same FT loop? |
|---|---|---|
| SPECTRA (DRL) | Layer and rate via A2C | yes |
| Greedy (“L1”) | Always strongest legal cut (keep 0.8); ranking L1/L2/SVD is ablation [18] | yes |
| Mild | Prefer keep 0.9 | yes |
| Random | Uniform legal rate | yes |

Heuristics pick **rate**, not a different ranking of filters, unless `SPECTRA_FILTER_IMPORTANCE` is set. Look-ahead greedy (`SPECTRA_EVAL_LOOKAHEAD=1`, job **20213131**) is the param-floor greedy control on r56-w4: TEST **−24.9 @ 0.722/0.477**. DepGraph [12] / SPA [81] are *not* same-loop baselines in this paper: they are related-work mechanism, not the arena.

### 4.2 Setup — LOCKED protocol / PRELIM catalogs

- Datasets in DRL train: CIFAR-10 [73], SVHN [74], Fashion-MNIST [72] (32×32 RGB). CIFAR-100 [73] is recoverability-first.
- Held-out C10: similar, unlike, thin (ledger §3–5).
- Metrics: TEST Δacc (pp), param_ratio, flops_ratio (fraction **kept**).
- Skip akamaster ResNet-32.
- Log `eval_train` is the CNN **train-image** loader on the same checkpoint, not “this architecture was in the DRL catalog.” Quote `eval_test` only.
- Architectures in the pool follow the proposal’s families [3, 4, 76, 78] at CIFAR scale; GoogLeNet [77] is not in the current 10-net catalog.

---

## 5. Evaluation results — fill from the ledger

Proposal §4 (NEON on flattened Fashion-MNIST / CIFAR / SVHN) is **motivation**, not a SPECTRA table. Do not mix those DNN-on-pixels numbers into §5.

### 5.1 Similar-family transfer (C10) — LOCKED

See ledger §3. Easy nets (VGG-19 BN [76], DenseNet-100 [4], ResNet-44 [3], MobileNet-v2×0.75 [78], thin ResNet-20 w16) stay inside τ=10 on seeds 42/43/44. Thin ResNet-56 w10 is the similar-family miss and is seed-sensitive (−9.2 / −12.4 / −13.0).

**Figure TBD:** bar or table of Δacc vs params kept, three seeds.

### 5.2 Unlike-family transfer (C10) — LOCKED (n=3 seeds)

See ledger §4. ShuffleNet-v2 and RepVGG, never in train, all inside τ=10 on seeds 42/43/44. RepVGG-A0 is −4.8 / −4.6 / −4.8 pp at 0.681/0.565 (same size). RepVGG-A1 is −4.7 / −4.4 / −4.3 at 0.650/0.521 (same size). ShuffleNet logs a mask fallback on some layers; quote TEST Δacc, do not claim every ShuffleNet layer was structurally resized.

### 5.3 Held-out thin ResNets and learned vs greedy — LOCKED

See ledger §5–6 and catalog ladder §17. Easy r20-w2: DRL ≈ greedy [18] at 60% params (s43 **−4.9 @ 0.600/0.760**; look-ahead greedy r20 **−4.0 @ 0.600/0.753**; FLOP-floor s42 **−2.1 @ 0.600/0.773**, s43 **−3.7 @ 0.600/0.763**; s44 FLOP-floor r20 **−4.0 @ 0.800/0.799**, a larger net). Hard r56-w4: C10-thin-only and encoder A/Bs stay near −24 pp; 10-net DRL **−15.9 / −16.2 / −17.2** at **identical** 0.704/0.550 (three seeds, **misses τ**). Unmatched greedy ~−24 at 0.667. Param-floor look-ahead greedy **−24.9 @ 0.722/0.477** (job 20213131). Eval-only FLOP floor 0.70 on the **same frozen 10-net actors**: s42 **−8.9 @ 0.907/0.702**, s43 **−9.2 @ 0.926/0.703**, s44 **−9.6 @ 0.907/0.703** — all **inside τ=10**, at ~91–93% params / 70% FLOPs (**LOCKED** three seeds). Similar r56-w10: DRL −13.0 vs greedy −23.1 at ~0.60 params. AMP, skinny-in-train, and budget-in-state did not move r56-w4 (ledger §18).

### 5.4 CIFAR-100 recoverability — LOCKED VGG / LOCKED tiny-cut others

See ledger §7. **CLAIM:** C100 is not a second genericity table until a 5–10% structured cut recovers on more than VGG [76]. Probe 20204214 is complete. VGG-11 BN recipe is two-run (TEST gains at 90–95% params). Residual keep-rate 0.8 still leaves ~95–96% params. DenseNet-40 keep 0.8 left 99.0% params (TEST −0.43). MobileNet-v2×1 keep 0.8–0.9: TEST gain at ~98–99% params but **val DROP** (~−12.5 pp). Do not describe early C100 failure as “the C10 agent was not trained on C100.” That failure mode was already visible in the proposal’s NEON-on-images C100 table (very low absolute accuracy); SPECTRA’s CNN probes show the *structured* version of the same hardness.

### 5.5 24-net catalog — TBD

Jobs 20201235 / 20202693 / 20204215. Question: does r56-w4 move past −15.9/−16.2/−17.2? As of 18 Aug 01:15, s42 is still in-catalog `eval_test` (not held-out). Afterok 20201260/63/65 have not started. Do not quote in-catalog or `eval_train`.

### 5.6 Frozen 10-net → CIFAR-100 (claim C9) — LOCKED mixed

See ledger §21. Same actors as §5.1–5.3, no extra agent training. VGG-16 BN [76] stays inside τ=10 (−7.5 / −7.8 / −7.3 pp at ~80–83% params). Thin ResNet-20 w16 and ResNet-56 w15 miss (≈ −14 to −19 pp). RepVGG-A0 misses (≈ −11 to −13 pp). ShuffleNet-v2×1 produced **no TEST** (grouped-conv crash). Read with §5.4: the families that recover from a structured cut are the families this frozen agent can transfer to.

### 5.7 Digit-MNIST LeNet (held-out dataset) — LOCKED three seeds

See ledger §22. Never in the 10-net train catalog (Fashion-MNIST was). TEST **+2.8 / +2.7 / +2.9 pp**. Toy 1-channel net, modest param cut. NEON-style cheap dataset cell, not ImageNet.

### 5.8 SVHN ResNet-20 width 8 (held-out width) — LOCKED three seeds

See ledger §23. SVHN [74] is in the 10-net train mix; this is a **new width**, not a held-out dataset. TEST **−2.0 / −1.5 / −2.0 pp**, all inside τ=10. Do not quote in-catalog r20-w16 or VGG-11 SVHN as transfer.

---

## 6. Discussion — DRAFT

**NEON’s pruning strategy, CNN version.** The agent learns *where* to cut. On easy C10 nets the strongest legal cut [18] is already good, so DRL ties greedy. On skinny-deep ResNets the schedule matters, but the environment is still harsh: residual groups decide params/FLOPs, and a 0.8 rate is not a 20% size cut [3]. Catalog diversity moved the hard thin ResNet-56; encoder width did not.

**Mechanism vs policy.** DepGraph [12] and SPA [81] are the right papers to thank. SPECTRA should not claim “first grouping for any CNN.” It should claim “first (in this lineage) offline preference-aware *policy* for structured CNNs,” with honest held-out splits.

**Why C100 is a different chapter.** Recoverability probes have no agent. If fine-tune cannot undo a mid-layer prune, A2C cannot learn a useful C100 policy. VGG-11 BN under SGD+aug 160 ep *can*; thin C100 ResNets at keep-rate 0.8 still keep ~95% of parameters (ledger §7.2). Mixing those nets into a C10 agent is how train returns go to −100, not how dataset transfer is demonstrated. Frozen 10-net → C100 TEST (ledger §21) matches that split: VGG-16 BN stays inside τ=10 on three seeds; thin ResNets and RepVGG-A0 miss; ShuffleNet-v2×1 crashed (no TEST). The proposal already flagged C100 as the hardest image set for a generic pruner; the CNN experiments refined that to *family-wise FT*, not “needs BERT.”

**Limitations (write them):** no ImageNet FT [71]; ShuffleNet grouping incomplete (mask fallback; C100 ShuffleNet eval crashed, no TEST); r56-w4 misses τ=10 at the 0.70-param operating point (DRL −15.9 vs look-ahead greedy −24.9); FLOP-floor 0.70 enters τ at ~91–93% params / 70% FLOPs (s42 −8.9, s43 −9.2, s44 −9.6); C100 residual recoverability open (C9 miss on thin ResNets); 24-net held-out eval not yet in; Places365 [75] and GoogLeNet [77] unused. Cheap-dataset transfer that *is* in: digit-MNIST LeNet TEST gain; SVHN r20-w8 inside τ (width held out, dataset in train). Do not quote overnight-matrix −1.2 pp or eval_train −1.7 / +6.0 / +7.4 as TEST (ledger §10).

---

## 7. Conclusions and future work — DRAFT

SPECTRA shows that NEON’s offline, preference-aware DRL protocol [1] extends to structured CNNs on CIFAR-10 for similar and unlike families, with skinny-deep ResNets as the documented failure mode. The same frozen agent transfers to digit-MNIST (TEST gain) and to a new SVHN ResNet width (inside τ), and to CIFAR-100 only on VGG — matching recoverability, not a blanket dataset-transfer win. The grouping problem that used to look like part of the thesis is professionally handled by DepGraph [12] and SPA [81]; that is a gift. The remaining question is still the one the proposal posed: whether a single preference-aware agent can carry a pruning policy across CNN families and datasets.

Future work: 24-net held-out eval, optional τ=5 vs τ=10 on the frozen actor, optional FLOP-floor A/B on C9 residuals, ImageNet as frozen/short eval at most.

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

## References

Proposal numbering **[1]–[78]** unchanged. Survey additions **[79]–[91]**.

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

[11] Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2018). Rethinking the value of network pruning. arXiv:1810.05270.

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

[24] Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2020). Layer-adaptive sparsity for the magnitude-based pruning. arXiv:2010.07611.

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

[43] Kuznedelev, D., Kurtic, E., Frantar, E., & Alistarh, D. (2024). CAP: Correlation-aware pruning for highly-accurate sparse vision models. *NeurIPS*, 36.

[44] He, H., Cai, J., Liu, J., Pan, Z., Zhang, J., Tao, D., & Zhuang, B. (2024). Pruning self-attentions into convolutional layers in single path. *IEEE TPAMI*.

[45] Guo, M. H., Lu, C. Z., Liu, Z. N., Cheng, M. M., & Hu, S. M. (2023). Visual attention network. *Computational Visual Media*, 9(4), 733–752.

[46] Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). Reinforcement learning: A survey. *JAIR*, 4, 237–285.

[47] Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *IJRR*, 32(11), 1238–1274.

[48] Kaiser, Ł., Babaeizadeh, M., Miłoś, P., Osiński, B., Campbell, R. H., Czechowski, K., … & Michalewski, H. (2019). Model-based reinforcement learning for Atari. *ICLR*.

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

[80] Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K. T., & Sun, J. (2019). MetaPruning: Meta learning for automatic channel pruning. *ICCV*.

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
