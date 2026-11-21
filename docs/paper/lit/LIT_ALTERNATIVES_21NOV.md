# Where is the headroom? Research directions after "agent = threshold rule" (literature synthesis, 21 Nov 2026)

Compiled 21 Nov 2026 by the hard-development subagent for the science sitting, ahead of the meeting with Gilad on Mon 23 Nov. Research only: nothing in the repo was changed except this file; no cluster, no git, no training.

**How sources were checked.** arXiv records (title, authors, venue comment, abstract) were read through the arXiv API; key papers were read in full text (ar5iv / arXiv HTML); two official-repo files and three venue pages were opened. A reference number in the text points to §14, where each source carries its own label.

**Labels.** [V] checked at the source this session (full text unless marked "abs" = abstract-level only). [R] verified in an earlier scan in this repo (file named in §14), not re-opened. [P] project evidence, by ledger section; project numbers are TEST 5k under the project's protocol, with val / 10k beside where the ledger line gave them. [H] my interpretation. [U] not verified.

**Plain names used here.** "The agent" = the frozen policy trained on many networks. "The threshold rule" = one global loss-rise threshold over per-group sensitivity curves, bisected to the budget. "The old rule" = the sensitivity rule registered earlier as the comparator. "Recovery" = the fine-tune after the cut (100 epochs in the record). C1–C8 are the eight candidates of the brief; N1 is a ninth that the search produced.

---

## 0. Executive summary

**The diagnosis (details in §1).** Three facts from the literature, each matched by the project's own evidence, explain why eleven agent generations ended at the best rule:

1. **After a long recovery, allocation is nearly saturated.** A pruned network's final accuracy is set mostly by its widths, and sensible width plans at one budget end within about a seed's spread [2, 3]. In-house: weights trained from scratch match inherited ones [P §394]; random channels match L1 channels at fixed widths [P §398]; no judge separates near-equal plans [P §432]; agent − threshold rule = +0.00.
2. **Allocation matters where recovery is short or absent.** That is why layer allocation is a crowded topic for LLMs, which are pruned without retraining [43–47], and why learned allocators there show large gains.
3. **At deep compression the published gains come from the training procedure, not the allocation**: sparse-regularised training before the cut, staged cuts, longer schedules [11, 26, 27]. In-house, the same widths lose 2.57 points when cut in one shot instead of in stages at 8.84× [P §310, one seed].

So the ranking favours directions that **change the regime or the lever** over directions that fight for the last +0.3 of allocation under a 100-epoch recovery.

| Rank | Direction | Novelty claim we could make | Closest prior art | Feasible in 12 weeks? | Risk | Headroom evidence |
|---|---|---|---|---|---|---|
| 1 | **C6. Evaluation paper**: a pre-registered, sealed-family test of learned against rule allocation | First pre-registered pruning study with architecture families sealed before the read; first same-recovery comparison of a cross-architecture learned allocator with rules, random search and published allocations; a one-threshold rule as the baseline to beat; measured test-selection inflation and "honest" deltas | Blalock [4]; PruningBench [5] (5 models, no learned allocator, no hold-out); Li 2022 [3]; Wang 2023 [6]; the NAS reality checks [14–16] | **High.** Most data exists; about 300 GPU-h more | **Low**: any outcome of the sealed read is a result. Venue risk only | Not needed |
| 2 | **N1 (new). The recovery-budget axis**: allocation with little or no retraining | First frozen cross-architecture allocation policy for calibration-only and few-epoch structured CNN pruning, trained on the very metric it is judged on; first map of how the allocation lever shrinks as recovery grows | RL-Pruner [38] (per network, hours); EagleEye [31]; AMC [41]; OBSPA [32]; the LLM allocation line [43–47] | **High.** Agents with these rewards exist; a cell costs 1/100 to 1/4 of a standard one | **Medium**: the threshold rule is built on calibration loss and may tie again | Literature: strong. In-house: unknown, but a zero-GPU read of logged one-epoch scores answers it in days |
| 3 | **C2. Learn the recovery procedure** (one shot or staged, sparse-regularised pre-training, learning rate, distillation) per network and budget | First generic frozen controller of the prune–recover procedure across architectures; closes the deep-compression gap with no per-network tuning | ALLR [7] (a hand rule); per-network learned schedules [71–73]; one-shot vs iterative [11]; AutoMC [75]; zero-shot fine-tuning selection [76, 77]; APQF [78] | **Medium.** Screen about 300 GPU-h; learned selector up to 900 more | **Medium-high**: may collapse to "always stage at deep cuts", a rule (which still closes the gap) | **Strongest**: +2.8 from sparse training at 8.9× [26]; staged beats one-shot at high ratios [11]; [P §310] |
| 4 | **C1. A transferable predictor, of the level** (the accuracy–budget curve), not of the ranking | "Pruning laws for CNNs": first cross-architecture predictor of post-recovery accuracy for structured pruning; an accuracy-target mode with no probes | Pruning Laws for LLMs [61]; GENNAPE [82]; per-model predictors [55, 84]; PruneShift's warning [13] | **High** for the level (existing rows, no GPU). Low for ranking near-equal plans | Low (level) / high (ranking) | Level: drops span 0 to 25 points. Ranking: in-house judges failed [P §432, §443] |
| 5 | **C5. Heavy-tail spectral allocation** as a rule and as agent inputs | First test of AlphaPruning-style allocation for structured CNN pruning, with and without recovery | AlphaPruning [44]; TempBalance [67] | **High**: days; an arm inside ranks 1–2 | Cheap; probably null after a long recovery | Published gains are all without retraining |
| 6 | **C7. Latency-aware generic pruning** | A frozen policy conditioned on a device latency table, zero-shot across devices | HALP [93]; SMCP [94]; MDP [81]; NetAdapt [92]; HELP [96]; MODNAS [97]; MCMC [98] | Low-medium: devices and latency tables needed | High: strong knapsack rules; no sign of an agent edge | The problem is real [P §395]; the agent's edge is unproven |
| 7 | **C8. Pruning pretrained backbones for downstream tasks** | A task-conditioned generic allocator | TransTailor [101]; Iofinova [102]; Caccia [104]; Iurada [105]; MULTIFLOW [106] | Low: new data pipeline | High | Against: task-agnostic pruning already transfers [105–107] |
| 8 | **C4. LLM / ViT layer allocation** | None left that fits a CNN thesis | OWL, AlphaPruning, DSA, BESA, EvoPress, OptiShear and at least eight more [43–60] | Compute is fine; the identity is not | High: discover-once-and-transfer exists (DSA 2024) | Large lever, already claimed; use it as motivation for rank 2 |
| 9 | **C3. Learned per-layer criterion selection** | None after a long recovery | LFPC [109]; evolved transferable criteria [110]; Pruner-Zero [49] | Easy, and pointless in this regime | High (null) | Against: [P §398]; [3]; [111] |

**Recommended package for Monday.** Rank 1 as the paper's spine, whatever the sealed read shows. Rank 2 as the constructive result, started with a zero-GPU read this week. Rank 3 as the lever that closes the gap to published results at deep cuts. Ranks 4 and 5 as cheap add-ons that reuse the same runs. Total new compute about 2,000–2,400 GPU-h, inside the stated budget (§12).

**Thesis identity.** Ranks 1–3 keep "a learned, generic, frozen component applied to unseen networks". Rank 2 keeps the DRL agent as it is. Rank 3 keeps the contextual-bandit framing, though with about a thousand labelled outcomes a small supervised selector is the honest learner. Rank 4 is supervised regression, not DRL.

**Threats to the core claim (§11).** No 2024–2026 paper found trains one allocation policy on many CNN architectures and applies it frozen to unseen families. The closest is unchanged since the 9 Oct scan [113]; its December 2025 revision still transfers only between two ResNets. Two things did change: (a) LLM-agent pruners [78, 121–123] are "generic with no training" by construction, though they loop per target; (b) the candid weakness is not priority but value: a referee will ask what the frozen agent buys over a one-line rule. The answer has to be cost against search-based methods, or a regime where it wins (ranks 2–3).

---

## 1. The diagnosis the ranking rests on

**1.1 Long recovery flattens allocation.**
- Liu et al. [2] and Li et al. [3]: the pruned architecture matters more than the inherited weights; the best of N random allocations comes within 0.5 % of search methods on ImageNet [R].
- Wang et al. [6] [V]: a re-implemented L1-norm baseline reaches 75.24 % at 2.31× on ResNet-50 ImageNet, "match or beat many filter pruning papers", once the fine-tune learning rate is set properly. The fine-tune learning rate alone moves a 95 %-pruned ResNet-34 on ImageNet-100 from 70.53 to 77.81 (their Tab. 4).
- In-house: scratch = inherited in three of four cells, and inherited weights help by +0.39 in the fourth [P §394]; random channels = L1 channels at the agent's widths (+0.13, +0.23) with a run-to-run SD of 0.40–0.45 per cell difference [P §398]; 28 candidate judges cannot pick between close plans (regret about 0.18 points, same as one epoch) [P §432]; a seven-way per-network pick on a one-epoch check gains TEST 5k +0.25 [+0.07, +0.43] (val +0.15, 10k +0.20) at FLOPs 0.45 and loses −1.95 (val −1.94) at 0.20 through one network whose picked plan never trained [P §443].
- Reading [H]: the per-network "oracle" room (about +0.8 over the old rule) is real against the old rule and was harvested by the threshold rule (+0.64). What remains over the threshold rule at moderate budgets is of the size of the noise, and part of any best-of-four figure is selection on noise.

**1.2 Allocation matters when recovery is short or absent.**
- EvoPress [47] [V], 70 % unstructured sparsity with SparseGPT [66], no retraining, Wiki2 perplexity for uniform / OWL / searched allocation: Llama-3-8B 85.84 / 48.07 / 28.76; Mistral-7B 23.08 / 17.22 / 14.42. The same paper shows that per-layer errors do not add up ("error monotonicity does not hold for LLMs"), which is the weak point of any rule that sums per-group losses. The draft records the same for SPDY's programme over per-layer losses [42] [R: draft §6].
- AlphaPruning [44] [V]: its evaluation "focuses on post-training pruning methods without fine-tuning" (its App. B).
- CNNs without fine-tuning lose whole points at mild cuts: OBSPA [32] [V] loses 0.80–3.73 points at 1.5–1.7× on CIFAR ResNet-50 / VGG-19; a 2026 fine-tune-free method [36] [V abs] keeps 65.4 % top-1 on ImageNet ResNet-50 at 30 % deletion (the unpruned accuracy is not in its abstract; a loss above 10 points is my reading [H]). In that range an allocation choice is worth points, not tenths.
- Zimmer et al. [7] [V]: the size of a pruning lever depends on the regime. With a 2-epoch retrain, the learning-rate schedule alone moves a one-shot-pruned ResNet-50 from 56.75 to 69.56 at 90 % sparsity, and from 73.51 to 73.69 at 70 %.

**1.3 At deep compression, the procedure is the lever.**
- DepGraph's own Table 1 [26] [R]: VGG-19 CIFAR-100 at about 8.9×, with sparse training 70.39, without 67.60: +2.8 from the training procedure on one allocation method.
- GReg [27] [R]: uniform L1 one-shot against growing regularisation at 8.84× / 17.3× / 44×: 66.05 vs 67.35, 61.59 vs 63.55, 51.36 vs 57.09. The regulariser's gain grows with depth.
- Janusz et al. [11] [V abs]: "one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios", for structured and unstructured pruning.
- PruningBench [5] [V]: sparsity regularisers help in "only 57.30 %" of its cases (BN-scale 77.78 %, group norm 42.31 %). Whether to regularise is network-dependent, which is a selection problem.
- In-house: on DepGraph's VGG-19 widths at 8.84×, one cut loses 2.57 points against the staged cut (10k −5.29 vs −2.72; DepGraph's own −2.97; one seed, pre-protocol row, no call) [P §310]. Two stages do not help at 25 % of parameters (+0.30 [−0.26, +1.37], val +0.29; and +0.12) [P §411]. Untuned distillation adds +0.00 val [P §444].

**1.4 Evaluation practice is itself an open problem.**
- Blalock et al. [4]: 81 papers, results not comparable [R]. Wang et al. [6] [V]: "Unfortunately, S2 is the most prevailing comparison setup" (only the base model is controlled). PruningBench [5] [V] fixes the fine-tune but has 5 models, no learned allocator, no held-out architecture, and does not say how the reported checkpoint is chosen.
- The official DepGraph benchmark script [25] [V] evaluates on `test_loader` every epoch, keeps the checkpoint with `if best_acc < acc:` and logs "Best Acc"; its CIFAR `val_dst` is built with `train=False`. So the reported number is the best epoch on the test set.
- ThinResNet [10] [V abs]: networks trained with a modern recipe and simple width scaling "clearly and utterly outperform all the literature we compared to".
- PruneShift [13] [V abs] (Aug 2026): a surrogate's rank correlation "can be near one while selection regret stays maximal"; predictive fit, decision reliability and method quality need separate evidence. This is the project's reward-fidelity finding, formalised by others.

---

## 2. Rank 1 (C6): the evaluation paper

**Closest prior art.**
- Blalock et al., MLSys 2020 [4]: meta-analysis of 81 papers and a library; unstructured focus; recommends practices, pre-registers nothing.
- PruningBench [5] [V]: 16 methods, 5 models (ResNet-18/50, VGG-19, ViT-small, YOLOv8), 3 datasets, 645 pruned models, fixed fine-tune. Allocation appears only as local / global / "protected global" (a 10 % floor per group). No learned or searched allocator is benchmarked; there is no architecture hold-out; checkpoint selection is unstated; seeds are partly averaged; the tables show no variance.
- Li et al., CVPR 2022 [3]: random allocation search against criteria, ResNet/VGG/MobileNet. Liu et al. [2]: scratch controls. Wang et al. [6]: fairness principle, learning-rate confound. Janusz et al. [11]: one-shot against iterative.
- The NAS precedent, the model for this paper [V abs]: Li and Talwalkar [14] (random search is a competitive NAS baseline), Yu et al. [15] ("state-of-the-art NAS algorithms perform similarly to the random policy"), Yang et al. [16] (a benchmark of 8 methods on 5 datasets scored against the average random architecture). Reality checks in other fields: metric learning [17] ("actual improvements over time have been marginal at best"), domain generalisation [18], deep RL [20, 21], variance in benchmarks [19], test-set reuse [22].
- Tabular width benchmarks exist for training from scratch (NATS-Bench's size space [23]) and for populations of sparsified small models [24]. None was found for prune-and-recover of trained CNNs [H].

**What is still open.** No pruning study found (i) registers its calls before the runs, (ii) seals architecture families until a single read, (iii) puts a cross-architecture learned allocator in the same recovery loop as rules and searches, (iv) corrects each row for what the unpruned network gains under the same recovery (the nearest ideas: scratch controls [2], and the finding that part of pruning's apparent gain is extra training and regularisation [12]), or (v) measures how much test-set checkpoint selection inflates published rows.

**The claim.** "Under one fixed recovery, across about 50 networks in 10+ families, including families sealed before the read, a single-threshold sensitivity rule is not beaten by any learned or searched allocator we could build or transplant; the allocation lever is 0.0–0.6 points at 2–2.6× and 2–3 points near 8×; published margins at standard points are of the size of test-selection inflation and of the unpruned network's own gain under the recovery."

**Experiments that prove it** (beyond the sealed read, due about 27 Nov).
1. **Allocation transplants.** Take the per-layer widths that published methods report or release (DepGraph is done [P §310]; add AMC [41], HRank, CHIP, EagleEye [31], OCSPruner [28] where widths can be rebuilt) and recover them with the fixed recipe on ResNet-56, VGG-16/19 and ResNet-50. Expected: within noise at 2×. About 100 GPU-h.
2. **Searched and learned comparators in the same loop**: per-network RL (AMC-style), random best-of-N with a recalibrated-BN score (EagleEye-style; one exists [P §396]), global ranking, LAMP, the heavy-tail rule of §6. About 150 GPU-h.
3. **Selection audit.** For each published pipeline already timed on the project's GPU, read best-on-test, last-epoch and validation-picked accuracy from the per-epoch logs (no GPU if the logs hold them; otherwise about 50 GPU-h).
4. **The lever map**: rule − even cut for every network × budget, with the recovery-length axis of §3 as its second dimension.

**Feasibility.** High. The protocol, the catalog, five-seed exact reproducibility, the honest delta and the cost table exist. Writing is the main cost.

**Venue.** TMLR fits best: its two criteria are whether "the claims made in the submission [are] supported by accurate and convincing evidence" and audience interest; "novelty of the studied method is not a necessary criterion" [29] [V]. The NeurIPS Evaluations & Datasets track (the renamed Datasets & Benchmarks track) lists "rigorous reproduction, auditing, and stress-testing of prior evaluations" and "negative results, critical analyses" in scope, and "submissions need not introduce a new model or outperform prior work"; its 2026 deadlines were 4 and 6 May [30] [V]. The 2027 call is not out [U]; a May 2027 deadline would fall after the thesis.

**Risk.** Low on results. Referee risks: "CIFAR-scale" (answer: the ImageNet ResNet-50 and Tiny-ImageNet rows), "one criterion, one recipe" (answer: §3's axis and §4's recipes), fewer methods than PruningBench (answer: more networks, held-out families, registration).

**Headroom evidence.** Not applicable: the contribution is the measurement. This is the direction that turns "no meat" into the result.

---

## 3. Rank 2 (N1, new): the recovery-budget axis

**The idea.** Keep the pipeline and the agent; vary how much recovery follows the cut: recalibrated batch-norm statistics only, 1, 5, 20, 100 epochs. Ask two questions. How does the spread between allocations change with recovery length? Does a frozen agent trained on the short-recovery score beat the rules where that score is the deployment metric? In that regime the reward-fidelity gap, the documented cause of the agent's deep-budget failures, does not exist: the reward is the objective.

**Closest prior art.**
- RL-Pruner [38] [V]: reinforcement learning of the layer-wise pruning distribution for CNNs, **per network**, over many small pruning steps; the reward is the cut network's accuracy measured before recovery (apparently on the test set), with optional distillation between steps; "the entire pruning process takes several hours" on one RTX 4090D; five CIFAR-100 networks. This is the direct per-network counterpart.
- AMC [41] rewards the cut network's accuracy; EagleEye [31] searches random plans by accuracy after batch-norm recalibration (rank correlation with post-fine-tune accuracy 0.52–0.85 [R]). Both per network.
- Post-training structured pruning of CNNs: OBSPA [32] [V] (apparently one global ranking, 2,048 calibration images on CIFAR, no fine-tuning), DFPC [33], UDFC [34], few-shot PRACTISE [35] [V abs] ("dropping blocks is a fundamentally superior approach" with tiny sets), response-geometry selection [36], layer-wise calibration [40]; for Transformers, mask search without retraining [39]. FCPTS [37] [V abs] learns a per-layer sparsity allocation in minutes for post-training sparsity, per network ("over 30 % improvement" on ResNet-50 at 80 % sparsity).
- The LLM line shows the size of the lever without retraining [43–47] and one allocation function that transfers across models (DSA [45] [R]).
- Zimmer et al. [7] and Li et al. [3] (25 → 100 epochs: +0.75 [R]) vary the retraining budget, not the allocation rule across it.

**What is still open.** No frozen cross-architecture allocator for retraining-free or few-epoch structured CNN pruning was found; every learned allocator in this regime is per network [37, 38, 41]. No study maps the allocation lever against recovery length across many networks [H: not found in the searches of §15].

**The claims.** (a) "The allocation lever falls from X points with no recovery to Y after 100 epochs, on 40+ networks": a measured account of why learned allocation is prized for LLMs and flat for long-recovery CNNs. (b) "One frozen policy, in one forward pass, matches or beats a per-network search (hundreds of evaluations; hours for RL-Pruner) and the best rule when recovery is at most one epoch."

**Experiments.**
0. **Zero GPU, this week.** Every one-epoch pick already logs each plan's one-epoch and recalibrated scores [P §443: "every subset of the seven plans can be scored from runs already made"]. Tabulate agent − threshold rule on those scores, per network and budget.
1. **Lever-against-recovery grid.** About 40 networks × 4 budgets × 8 arms (even cut, old rule, threshold, cost-aware, LAMP, global ranking, agent, random best-of-N) × 3 seeds, at recalibration-only, 1, 5 and 20 epochs; the 100-epoch column exists. The four short lengths together cost about a quarter of a standard cell (0.23 GPU-h per standard cell [P §443]): about 250 GPU-h.
2. **Matched-reward agents.** Agents trained on the recalibration and one-epoch scores exist; retrain on the current catalog with the rule-blending action, three seeds per regime (about 20 GPU-h per seed [R: draft §6]): about 180 GPU-h.
3. **Per-network search at equal regime**: random best-of-100 and an RL search, with their evaluation counts and wall time.
4. **A registered sealed read** at recovery ≤ 1 epoch on the sealed families, calls fixed in advance.

**Feasibility.** High: no new code path beyond a recovery-length flag and readers; cells are cheap; three weeks to a registered read.

**Risk.** Medium. The threshold rule is defined on calibration loss, so with no recovery it may be near-optimal at mild cuts and the agent may tie again. The agent's chance is at deep cuts, where per-group losses stop adding up [47; P: draft §6]. If it ties, claim (a) still stands and feeds rank 1.

**Headroom evidence.** Literature: strong (§1.2). In-house: the agent beats a 100-plan random search scored by its own reward (+0.51 and +0.45 after recovery) [P §396]; the direct comparison on short-recovery scores has not been read. That read is step 0.

**Optional extensions.** The choice of channels, irrelevant after long recovery, matters with none [36]; removing whole blocks may beat thin channels at deep cuts [35, 81]. Both widen the agent's action space and should wait for step 1.

---

## 4. Rank 3 (C2): learn the recovery procedure, not the allocation

**Closest prior art.**
- **Hand rules for retraining.** Learning-rate rewinding [9]; cyclic and large learning rates [8]; ALLR, which sets the starting rate from the pruning-induced weight change and the retraining budget [7] [V]; budgeted training [79].
- **Learned schedules, per network.** Just-in-Time Sparsity [71] (ICML 2022 workshop; RL learns per-layer sparsity schedules during training of small CNNs and ResNet-18) [U: abstract and snippets only]. Ganjdanesh et al., CVPR 2024 [72] [V abs]: an RL agent decides pruning while the weights train, for one network. Auto-Train-Once [73]: a controller network guides prune-from-scratch, per network. ICE-Pruning [74] [V abs, first half]: an iterative pipeline built to cut the cost of repeated fine-tuning.
- **Benchmarks.** One-shot against iterative [11]; PruningBench's regulariser study [5].
- **Selecting a recipe from prior experience.** AutoMC [75] [V abs] searches compression schemes using knowledge of how each method behaves; zero-shot selection of fine-tuning pipelines across datasets [76] and a meta-learned multi-fidelity predictor over 20k configurations [77] [V abs] do the cross-task version for fine-tuning, not for pruning.
- **LLM planners.** APQF [78] [V abs] (Aug 2026): LLM planners propose per-layer ratios, bit-widths "and the recovery strategy" for CNNs and ViTs, per target.

**What is still open.** A frozen controller, trained across many networks, that picks the prune–recover procedure for an unseen network and budget (one shot or k stages, regularised pre-training or not, learning rate, distillation) was not found. Per-network learned schedules and LLM planners exist; a cross-architecture learned one does not [H].

**The claim.** "A generic controller of the prune–recover procedure, trained offline and applied frozen, recovers the deep-compression gap to methods that tune their training per network." If one procedure wins everywhere, the claim shrinks to a recipe finding, still valuable: it lifts every same-loop arm at 8× and beyond and removes the 3–4 point gap that referees will see first.

**Experiments.**
1. **Screen** (about 300 GPU-h): 12 networks at FLOPs 0.20 and about 0.11, arms = {one shot, 2 and 4 stages at equal total epochs} × {regularised pre-training on / off} × {two learning rates} × {distillation tuned, off}, two seeds. Read (i) the best single procedure, (ii) whether the best differs by network (the PruningBench 57 % result predicts it does).
2. **If it differs:** build the outcome table on 30–40 catalog networks (6–8 procedures × 2 budgets × 2 seeds, about 1,000–1,300 cells at 1.5× cost: 400–500 GPU-h), fit the selector, and read it on sealed families against the best single procedure and the per-network oracle.
3. **ImageNet ResNet-50** at one deep point with the chosen procedure, for the literature table.

**Feasibility.** Medium. Staged cuts existed in the earlier step-by-step pipeline [P §310]; regularised pre-training needs a new default-off path. Six to eight weeks to a registered read.

**Risk.** Medium-high. The table is small, so a small supervised selector is more honest than a policy-gradient agent here. At moderate budgets the procedures will tie [P §411]; the study must live at deep cuts, where effects (2–3 points) exceed the noise (SD 0.4).

**Headroom evidence.** The strongest of all nine: §1.3.

---

## 5. Rank 4 (C1): a transferable accuracy predictor

**Closest prior art.**
- **Pruning.** SeerNet [84] [V abs]: a performance predictor trained per network on actively chosen compression policies, then gradient search on it. PPF [55] [R] and Ji et al. [90]: predictors of pruned-LLM performance, per model. EagleEye [31] is a proxy, not a learned predictor. HiReLC [124] [V abs] uses an MLP surrogate inside a per-network RL loop.
- **Laws.** Pruning Laws for LLMs [61] [V abs] (EMNLP 2026): scaling relations linking post-pruning performance to unpruned performance and pruning ratio; "average extrapolation error less than 7 %"; the functional form transfers to "unseen models in zero-shot and one-shot setups".
- **NAS.** Cross-family predictors: GENNAPE [82] [V abs] is pretrained on one benchmark and transfers to others and to MobileNet / ResNet families with little or no fine-tuning; predictor surveys [83]; encodings [89]. Rank fidelity drops among the best candidates [85] [R].
- **Weight-space models** that read diverse architectures, including residual and normalisation layers [87, 88]; and the reverse relation, robustness to pruning as a predictor of generalisation [86].
- **A caution.** PruneShift [13]: high rank correlation does not imply a good selected plan.

**What is still open.** A cross-architecture predictor of **post-recovery** accuracy for structured CNN pruning was not found. Two different targets hide under "predictor":
- **(a) Ranking near-equal plans on one network.** Low headroom: in-house judges up to a 5-epoch compressed recovery do not beat a one-epoch check [P §432]; the differences sit inside the noise [P §398]; the literature says fidelity breaks near the top [13, 85].
- **(b) Predicting the level**: how much accuracy an unseen network loses at each budget. At the same FLOPs 0.45 on CIFAR-100, the agent's plan loses TEST 5k 0.72 on VGG-19 (val 0.27) and 7.80 on ResNet-20 ×8 (val 6.93) [P §443]. Large signal, and useful: the accuracy-target mode currently needs five 20-epoch probes that under-read the final recovery [P: draft §6].

**The claim (for b).** "Pruning laws for CNNs: from a network's descriptors and its sensitivity curves, one model predicts the post-recovery accuracy–budget curve of an unseen architecture family within E points, and returns the smallest budget that meets an accuracy target with no probe."

**Experiment.** Zero GPU first: assemble every recovered row on the record (network, budget, plan, final change), fit a gradient-boosted model and a functional-form law, test leave-one-family-out; then score the accuracy-target mode against the probe search on the sealed families. A hundred GPU-hours at most for confirmation rows.

**Feasibility.** High for (b). **Risk.** Low for (b), with moderate novelty: a solid thesis chapter and a section of the rank-1 paper rather than a paper of its own. High for (a): do not fund it beyond the zero-GPU check that the same table allows.

---

## 6. Rank 5 (C5): heavy-tail spectral allocation

**Closest prior art.** AlphaPruning [44] [V]: the per-layer power-law exponent of the weight spectrum (PL_Alpha_Hill) is mapped linearly to layer sparsity, with one range parameter τ, normalised to the global target. Its results: LLMs with unstructured pruning; ConvNeXt and ViT/DeiT with Wanda [65], unstructured, "without fine-tuning" (Sec. 5.3); structured LLM pruning (LLM-Pruner [63], OSSCAR) in an appendix. It reports no channel pruning and no recovered CNN. TempBalance [67] [V abs] uses the same layer metrics to set per-layer learning rates; the metrics predict model quality across hundreds of pretrained vision and language models [68]. Training-free layer rules already in the project's menu: LAMP [69], ERK [70].

**What is still open.** Heavy-tail allocation for structured CNN pruning, with or without recovery, was not found in the searches (§15) [H].

**The claim.** "First test of heavy-tail allocation for structured CNN pruning": as a rule, and as two or three extra per-group inputs to the agent. Honest expectation: null after 100 epochs (§1.1), possibly useful with no recovery (§3), where all of its published gains lie.

**Experiment.** Compute the exponent per convolution (kernel reshaped to a matrix), average per channel group, map with τ ∈ {0.1, 0.2, 0.3}, decode with the existing budget decoder. Run it as one more arm in the grids of §2 and §3. Two to four days of code, 100–150 GPU-h.

**Risk.** Low cost, small standalone novelty. Worth doing because it is free inside ranks 1–2 and pre-empts a referee's "did you try AlphaPruning?".

---

## 7. Rank 6 (C7): hardware-latency-aware generic pruning

**Closest prior art.** NetAdapt [92] and AMC's latency mode [41] (per network, per budget, measured latency). HALP [93] [R]: a latency–saliency knapsack on one target device, with a latency lookup table. SMCP [94]. Multi-dimensional pruning [81] [V abs]: channels, layers and blocks jointly under a latency constraint, as a mixed-integer programme solved in one pass, "particularly at large pruning ratios" (its CVPR 2025 version is arXiv:2504.02168). CPrune [95] (compiler-informed). MCMC [98] [V abs]: one multi-objective RL pruner for latency, FLOPs and memory, per network. Device transfer exists in NAS, not in pruning: HELP [96] meta-learns latency prediction for unseen devices; MODNAS [97] [R] conditions one hypernetwork on a device embedding. HDAP [99] handles variation among copies of one device.

**What is still open.** A pruning policy conditioned on a device and applied frozen to unseen devices and networks was not found [H].

**The claim.** "One frozen allocator, given a device's latency table, meets a latency budget on unseen devices and networks."

**Experiment.** Latency tables per layer shape on at least three devices (a server GPU, a CPU, one edge board); a latency-budget decoder; comparators: a HALP-style knapsack and the FLOPs agent.

**Feasibility.** Low-medium: no edge devices are known to be at hand; real-latency measurement is fiddly; twelve weeks is tight.

**Risk.** High. The hand-rule family here (knapsack over latency tables) is strong and exact, and nothing in the record suggests the agent beats it.

**Headroom evidence.** The problem is real: a ResNet-56 with 37–61 % fewer FLOPs runs at 0.93–1.07× on an RTX 4090, VGG-16 gains 1.32–1.42×, and batch-1 latency is unchanged everywhere [P §395]. That argues for reporting measured speed in the rank-1 paper. It does not argue that a learned agent would win.

---

## 8. Rank 7 (C8): pruning pretrained backbones for downstream transfer

**Closest prior art.** Molchanov et al. [100]: channel pruning in the transfer setting (fine-grained datasets). TransTailor [101] [V abs]: prune the pretrained model with target-aware importance, per target task ("2.7 % accuracy improvement ... with 20 % fewer FLOPs" on Stanford Dogs). Iofinova et al. [102] [V abs]: unstructured sparse ImageNet models on twelve transfer tasks; sparse models "can match or even outperform" dense ones. Chen et al. [103]. Caccia et al. [104] [V abs]: zero-shot structured pruning of pretrained filters for compute-efficient transfer. Structured pruning adapters [108].

**Evidence against the premise.** A generic agent matters here only if the best allocation depends on the task. Recent work says pruning decisions transfer across tasks: Iurada et al. [105] [V abs] find that pruning on one task "retains the model's zero-shot performance also on unseen tasks"; MULTIFLOW [106] [V abs] builds one task-agnostic pruned vision-language model; winning tickets generalise across datasets [107]. In-house, allocation barely matters on easy datasets [P: draft §7, §402–§403].

**What is still open, and the claim.** A learned allocator conditioned on the downstream task and reused across tasks and backbones was not found [H]. The claim would be "one frozen policy allocates per task with no per-task search".

**Experiment.** An ImageNet backbone pruned once per task by the agent, by a task-blind rule and by TransTailor-style per-task importance, then fine-tuned on about ten downstream datasets.

**Feasibility and risk.** Low and high: a new data pipeline, 224-pixel fine-tuning costs, and a premise the literature weakens.

**Verdict.** Not for these twelve weeks.

---

## 9. Rank 8 (C4): layer allocation for LLMs and ViTs

**How crowded it is.** Non-uniform layer sparsity for LLMs, 2023–2026, each checked by its arXiv or proceedings record (labels in §14): OWL [43], AlphaPruning [44], DSA [45], BESA [46], EvoPress [47], OptiShear [48], maximum-redundancy allocation [50], a theory-derived layer schedule [51], DLP [52], LayerIF [53], single-step RL allocation [54], predictor-plus-RL allocation [55], preference-conditioned plan-level policies [56], LLM-designed pruning algorithms with skew-aware allocation [59], adaptive structured pruning [64] and structured searches [60]. For ViTs, isomorphic grouping [62].

**Has a frozen learned allocator been done there?** In substance, yes. DSA [45] [R] searches an allocation function once on one model and reuses it on other LLM families "without conducting additional searches". OptiShear [48] [V abs] claims "strong cross-task and cross-model generalizability" for its discovered metrics and ratios. An RL quantization policy transfers zero-shot between LLMs [57] [R]. A neural policy trained on many LLMs and applied frozen was not found, but the gap is narrow and the field moves monthly.

**Would CNN-trained insight transfer?** Not the agent: its inputs are channel-group features and its catalog is CIFAR-scale CNNs. What does transfer is the finding of §3: the lever is large exactly because nothing is retrained.

**The only claim left, and its test.** "A neural allocation policy trained on several LLMs and applied frozen to unseen ones beats OWL, AlphaPruning and DSA's function at equal sparsity", read as perplexity and zero-shot accuracy at 50–70 % sparsity. It would mean a new agent, new features and a new catalog.

**Feasibility and risk.** Compute is feasible (calibration-only methods fit one 24 GB GPU for 7B models [H]); the risk is high, because the field moves monthly and its obvious claim is taken.

**Verdict.** It would be a different thesis. Use the LLM literature as motivation and related work for rank 2.

---

## 10. Rank 9 (C3): learned per-layer criterion selection

**Closest prior art.** LFPC [109] [R] (CVPR 2020) learns which criterion each layer uses, by a differentiable sampler, per network. Liu, Kung and Wentzlaff [110] [V abs] (GECCO 2022) evolve closed-form channel-importance functions on small datasets that transfer to ImageNet: a learned, transferable criterion already exists. Pruner-Zero [49] and OptiShear [48] evolve pruning metrics for LLMs. A 2024 paper evolves filter criteria for pruning at initialisation [112] [U: title only].

**Evidence against headroom.** Common criteria rank filters almost identically [111] (NeurIPS 2021). Under random allocations, criteria matter little [3]. In-house, random channels equal L1 channels at the agent's widths [P §398].

**What is open, and its test.** Only the cross-architecture part: a frozen policy that picks a criterion per layer on unseen networks. The test is cheap (three or four criteria at fixed widths, 30 networks, about 60 GPU-h), and the in-house result predicts a null.

**Verdict.** Neither novel nor promising under a long recovery. It becomes relevant only inside rank 2 (no recovery), as a later extension.

**Also considered, not ranked.**
- **Discovering an allocation rule by program search** (as DSA [45] and Pruner-Zero [49] do for LLMs): open for recovered CNNs, but the room over the threshold rule is noise-sized at moderate budgets (§1.1). Worth a try only inside rank 2, where an evaluation costs seconds.
- **A per-network pick with a better judge**: closed by [P §432, §443].
- **An LLM-agent baseline** [78, 121]: not a direction, but a comparator the rank-1 paper should consider, because referees will ask.

---

## 11. Prior art for the core claim: a frozen, generic, learned pruning policy across architectures

Searches: three arXiv batches (2023–2026, 30 queries), five web searches, the list of papers citing NEON [1] (22 found), and the 9 Oct scan in this repo (§15).

| Work | What is learned and reused | Scope of the transfer shown | Threat | Positioning |
|---|---|---|---|---|
| Meta Pruning via Graph Metanetworks [113] [V, v3 of 15 Dec 2025; no venue listed] | A graph network that rewrites a network's weights so that a fixed group-norm criterion prunes it better; "once we have a metanetwork, we can prune as many networks as we want" | ResNet-56 ↔ ResNet-110 on CIFAR-10 (92.81 vs 92.08 without; 93.04 vs 92.40), and across CIFAR-10 / CIFAR-100 / SVHN on ResNet-56 (5 of 6 cells positive). Single numbers, one family, speed-ups set by hand [R: S9]. Main rows: ResNet-56 CIFAR-10, VGG-19 CIFAR-100, ResNet-50 ImageNet | **Medium.** Closest; it claims generality and priority | It edits weights and does not allocate; one family. SPECTRA: an allocation policy, families sealed, same-loop comparison |
| Generalist RL compression agents [114] [R] | An agent picks a technique or rank per layer | LeNet to LeNet | Low | Same architecture only |
| GNN-RL, AGMC, N2N, warm-start RL [115–118] [R] | Encoders or policies | Within one family, with part of the agent retrained or a warm start | Low | Not frozen |
| Out-of-the-box pruned networks [119] [R] | Pruning profiles | Same architecture, across datasets | Low | — |
| Balemans et al. [120] [R] (Sep 2025) | Per-model graph RL | None; names per-model policies "a fundamental limitation" | None; useful quote | Cite as the open problem |
| RL-Pruner [38], HiReLC [124], MCMC [98], Ganjdanesh [72] | Per-network RL | None | None | Per-network comparators |
| DSA [45], OptiShear [48], RAMP [57], HiPP-Prune [56], PruneNet [58], single-step RL [54] | Functions or policies for LLM sparsity or bit-width | Across LLMs (DSA, OptiShear, RAMP) or across ratios | Medium for wording | Scope every "first" to structured CNN pruning |
| Pruning Laws [61] (EMNLP 2026) | A functional form predicting pruned performance | Unseen LLMs, zero-shot | Low; relevant to §5 | Cite |
| **LLM-agent pruners**: AgenticPruner [121] [R], ProfilingAgent [122] [V abs], APQF [78] [V abs], agents pruning LLMs [123], LLM-designed algorithms [59] | Nothing is trained: a general LLM proposes per-layer ratios (and, in APQF, the recovery strategy) | Any architecture, by construction; each target gets its own propose–evaluate loop | **Medium, for positioning** | They are generic without being frozen policies: several evaluated iterations per target, through an external general-purpose LLM [H]. Compare cost and reproducibility; consider one as a baseline |
| Evolved transferable pruning functions [110] | A closed-form channel criterion | Small datasets to ImageNet | Low | A criterion, not an allocation |
| EfficientLLaVA [126] [V abs] | A pruning-policy search with 64 samples | "Generalizable" means to unseen data, on one model | None | — |
| RL-for-compression survey [125] [V abs] (IEEE Access 2026) | — | — | None | Cite as the taxonomy |

**Conclusion.** No 2024–2026 paper found already does "one allocation policy, trained offline on many CNN architectures, applied frozen to unseen families, compared under one recovery with rules". The wording of the 9 Oct scan still holds, with its scopes. New since then: the metanetwork paper's third version (same two-ResNet transfer), a growing LLM-agent family, and prediction laws for LLMs.

**The candid part.** Priority is not the weak point; value is. With agent − threshold rule = +0.00, the claim "first frozen generic allocator for CNNs" reads as "first to learn what a one-line rule already does". Three honest framings survive: (1) non-inferiority at zero per-network search cost, against search-based methods that take 10 minutes to 17+ hours; (2) the rank-1 result itself: the first rigorous evidence that learned allocation does not beat a threshold rule under long recovery; (3) a regime where the frozen agent does win (ranks 2–3), if the data deliver it.

---

## 12. What to bring to Gilad on Monday

**One sentence.** "When does a learned, generic pruning policy matter? Under a long recovery it does not beat a threshold rule, and we show that with a pre-registered, sealed test; it should matter where recovery is short and where the recovery procedure itself must be chosen, and we test both in twelve weeks."

| Weeks (from 23 Nov) | Work | New GPU-h (about) |
|---|---|---|
| 0–1 | Sealed read lands (27 Nov). Zero-GPU reads: agent against threshold rule on logged one-epoch scores (§3); level predictor on existing rows (§5); best-on-test against last-epoch from existing logs (§2) | 0 |
| 1–3 | Recovery-length grid with the heavy-tail arm (§3, §6). Deep-cut procedure screen (§4) | 250 + 300 |
| 3–6 | Matched-reward agents and a registered short-recovery read on sealed families (§3). Decide §4: one procedure, or a learned selector. Allocation transplants and same-loop comparators (§2) | 180 + 250 |
| 6–9 | Learned selector if warranted (§4). ImageNet rows with the chosen procedure | up to 900 |
| 9–12 | Thesis chapters; TMLR submission of the evaluation paper | 100 |

**Decisions to ask for.**
1. Is the evaluation paper (rank 1) acceptable as the spine of the thesis, with the frozen agent as its subject and its efficiency result?
2. May the short-recovery regime (rank 2) be registered as a new study on the sealed families?
3. May the learned component also be a controller of the recovery procedure (rank 3), not only of the allocation?
4. Venue: TMLR now, the NeurIPS evaluation track later, or both in sequence?

---

## 13. Notes on reading the evidence

- Every outside accuracy is that paper's own number under its own protocol; several select the checkpoint on the test set (§1.4).
- In-house numbers are quoted from single ledger lines read for this file, not from a full reading of the ledger. [P §310] is a pre-protocol, one-seed, 10k row and is a pointer only.
- "Not found" always means "not found by the searches in §15", never "does not exist".

---

## 14. References

Label after each entry: how it was checked. "API" = arXiv API record (title, authors, venue comment). Earlier repo scans: S9 = `docs/LIT_SCAN_9OCT_TRANSFER_BUDGET.md`; S13 = `docs/paper/LIT_ALLOCATION_PROXIES_13OCT.md`; S20 = `docs/paper/lit/LIT_SEALED_WEB_20NOV.md`; E20 = `docs/paper/LIT_EVAL_PRACTICES_20OCT.md`.

**Anchors and evaluation**
1. Hirsch, Katz. Multi-objective pruning of dense neural networks using deep reinforcement learning (NEON). Information Sciences 610:381–400, 2022. doi:10.1016/j.ins.2022.07.134. [R: S9]
2. Liu, Sun, Zhou, Huang, Darrell. Rethinking the Value of Network Pruning. ICLR 2019. arXiv:1810.05270. [V API; facts R: S13]
3. Li, Adamczewski, Li, Gu, Timofte, Van Gool. Revisiting Random Channel Pruning for Neural Network Compression. CVPR 2022. arXiv:2205.05676. [V API; facts R: S13]
4. Blalock, Gonzalez Ortiz, Frankle, Guttag. What is the State of Neural Network Pruning? MLSys 2020. arXiv:2003.03033. [V API; facts R: E20]
5. Li, Li, Xue, Fang, Zhou, et al., Song. A Comprehensive Study of Structural Pruning for Vision Models (PruningBench). arXiv:2406.12315 (v7). [V full text, ar5iv]
6. Wang, Qin, Bai, Fu. Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning. arXiv:2301.05219. [V full text]
7. Zimmer, Spiegel, Pokutta. How I Learned to Stop Worrying and Love Retraining. ICLR 2023. arXiv:2111.00843. [V main text; appendices B.1 and C.2 not read]
8. Le, Hua. Network Pruning That Matters: A Case Study on Retraining Variants. ICLR 2021. arXiv:2105.03193. [V API; facts R: S13]
9. Renda, Frankle, Carbin. Comparing Rewinding and Fine-tuning in Neural Network Pruning. ICLR 2020. arXiv:2003.02389. [V API]
10. Tessier, Boukli Hacene, Gripon. ThinResNet: A New Baseline for Structured Convolutional Networks Pruning. arXiv:2309.12854. [V abs]
11. Janusz, Wojnar, Li, Benini, Adamczewski. One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression. doi:10.3233/FAIA251100; arXiv:2508.13836. [V abs]
12. Jin, Carbin, Roy, Frankle, Dziugaite. Pruning's Effect on Generalization Through the Lens of Training and Regularization. NeurIPS 2022. arXiv:2210.13738. [V API and first half of the abstract]
13. Ye, Zhang. PruneShift: A Framework for Evaluating Decision Reliability in Structured Pruning. arXiv:2608.29765 (30 Aug 2026). [V abs]
14. Li, Talwalkar. Random Search and Reproducibility for Neural Architecture Search. UAI 2019. arXiv:1902.07638. [V abs]
15. Yu, Sciuto, Jaggi, Musat, Salzmann. Evaluating the Search Phase of Neural Architecture Search. arXiv:1902.08142. [V abs]
16. Yang, Esperança, Carlucci. NAS evaluation is frustratingly hard. ICLR 2020. arXiv:1912.12522. [V abs]
17. Musgrave, Belongie, Lim. A Metric Learning Reality Check. arXiv:2003.08505. [V abs]
18. Gulrajani, Lopez-Paz. In Search of Lost Domain Generalization. arXiv:2007.01434. [V API]
19. Bouthillier et al. Accounting for Variance in Machine Learning Benchmarks. arXiv:2103.03098. [V API]
20. Henderson, Islam, Bachman, Pineau, Precup, Meger. Deep Reinforcement Learning that Matters. AAAI 2018. arXiv:1709.06560. [V API]
21. Agarwal, Schwarzer, Castro, Courville, Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. arXiv:2108.13264. [V API]
22. Recht, Roelofs, Schmidt, Shankar. Do ImageNet Classifiers Generalize to ImageNet? arXiv:1902.10811. [V API]
23. Dong, Liu, Musial, Gabrys. NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size. IEEE TPAMI 2021. doi:10.1109/TPAMI.2021.3054824; arXiv:2009.00437. [V API]
24. Honegger, Schürholt, Borth. Sparsified Model Zoo Twins: Investigating Populations of Sparsified Neural Network Models. ICLR 2023 workshop. arXiv:2304.13718. [V API]
25. Torch-Pruning (official DepGraph repository), `reproduce/main.py` and `reproduce/registry.py`, master branch, read 21 Nov 2026: github.com/VainF/Torch-Pruning. [V]
26. Fang, Ma, Song, Mi, Wang. DepGraph: Towards Any Structural Pruning. CVPR 2023. arXiv:2301.12900. [V API; table values R: S20]
27. Wang, Qin, Zhang, Fu. Neural Pruning via Growing Regularization (GReg). ICLR 2021. arXiv:2012.09243. [V API; table values R: S20]
28. Ghimire, Kil, Jeong, Park, Kim. One-Cycle Structured Pruning via Stability-Driven Subnetwork Search (OCSPruner). arXiv:2501.13439. [V API; values R: S20]
29. TMLR acceptance criteria. jmlr.org/tmlr/acceptance-criteria.html. [V]
30. NeurIPS 2026 Evaluations & Datasets Track, call for papers. neurips.cc/Conferences/2026/CallForEvaluationsDatasets. [V]

**Short or no recovery**
31. Li, Wu, Su, Wang, Lin. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV 2020. arXiv:2007.02491. [R: S13]
32. Wang, Rachwan, Günnemann, Charpentier. Structurally Prune Anything: Any Architecture, Any Framework, Any Time (SPA, OBSPA). arXiv:2403.18955. [V full text]
33. Narshana, Murti, Bhattacharyya. DFPC: Data flow driven pruning of coupled channels without data. ICLR 2023. openreview.net/forum?id=mhnHqRqcjYU. [U: taken from the reference list of 32; not opened]
34. Bai, Chen, Shen, Qian, Liu. Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning. ICCV 2023. arXiv:2308.07209. [V API]
35. Wang, Wu. Practical Network Acceleration with Tiny Sets (PRACTISE). CVPR 2023. arXiv:2202.07861. [V abs]
36. Shu. Unified Response Geometry for Structured Pruning. arXiv:2609.18239 (Sep 2026). [V abs]
37. Gong, Yong, Wang, Guo, Wei, Ma, Liu. Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes (FCPTS). arXiv:2405.05808. [V abs; whether the sparsity is structured is not stated there]
38. Wang, Kindratenko. RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration. arXiv:2411.06463. [V full text]
39. Kwon, Kim, Mahoney, Hassoun, Keutzer, Gholami. A Fast Post-Training Pruning Framework for Transformers. NeurIPS 2022. arXiv:2204.09656. [V API]
40. Lazarevich, Kozlov, Malinin. Post-training deep neural network pruning via layer-wise calibration. arXiv:2104.15023. [V API]
41. He, Lin, Liu, Wang, Li, Han. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018. arXiv:1802.03494. [R: S9, S13]
42. Frantar, Alistarh. SPDY: Accurate Pruning with Speedup Guarantees. ICML 2022. arXiv:2201.13096. [V API; the claim used is R: draft §6]

**Layer allocation for LLMs and ViTs**
43. Yin et al. Outlier Weighed Layerwise Sparsity (OWL). ICML 2024. arXiv:2310.05175. [V API]
44. Lu, Zhou, Liu, Wang, Mahoney, Yang. AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved Layer-wise Pruning of Large Language Models. NeurIPS 2024. arXiv:2410.10912. [V full text to App. G; App. H not read]
45. Li, Dong, Tang, Liu, Wang, Luo, Xue, Liu, Chu, Guo. Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models (DSA). NeurIPS 2024. doi:10.52202/079017-4487. [R: S9]
46. Xu et al. BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation. arXiv:2402.16880. [V API]
47. Sieberling, Kuznedelev, Kurtic, Alistarh. EvoPress: Accurate Dynamic Model Compression via Evolutionary Search. ICML (camera-ready per the arXiv record). arXiv:2410.14649. [V full text]
48. Liu, He, Wu, Song. Beyond One-Size-Fits-All Pruning via Evolutionary Metric Search for Large Language Models (OptiShear). arXiv:2502.10735. [V abs]
49. Dong et al. Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models. ICML 2024. arXiv:2406.02924. [V API]
50. Gao, Zhao, Wang, Chen, Jing. Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs. arXiv:2503.18377. [V API]
51. Huang, Zhang, Zheng, Chao, Ji. Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective. arXiv:2502.14770. [V API]
52. Chen et al. DLP: Dynamic Layerwise Pruning in Large Language Models. ICML 2025. arXiv:2505.23807. [V API]
53. Askari, Gupta, Wang, Chhabra, Chen. LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions. NeurIPS 2025. arXiv:2505.23811. [V API]
54. Yuan et al. FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning. arXiv:2511.18977. [R: S9]
55. Ma, Cui, Qin. Beyond Manually Designed Pruning Policies with Second-Level Performance Prediction (PPF). arXiv:2508.02381. [R: S9]
56. Bai, Tabia, Santos-Rodriguez. HiPP-Prune: Hierarchical Preference-Conditioned Structured Pruning for Vision-Language Models. arXiv:2603.06270. [R: S9]
57. Gautam, Jha. RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference. arXiv:2603.17891. [R: S9]
58. Sengupta, Chaudhary, Chakraborty. You Only Prune Once (PruneNet). ICLR 2025. arXiv:2501.15296. [R: S9]
59. Kang, Lin, Yang, Dai, Wang. Automatic Pruning Discovery for Large Language Models. arXiv:2511.15390. [V abs]
60. Tang, Sieberling, Kurtic, Shen, Alistarh. DarwinLM: Evolutionary Structured Pruning of Large Language Models. COLM 2026. arXiv:2502.07780. [V API] Also: Li et al. Týr-the-Pruner. arXiv:2503.09657. [V API]
61. Sengupta, Chaudhary, Chakraborty. Pruning Laws for Large Language Models. EMNLP 2026. arXiv:2504.04342. [V abs]
62. Fang, Ma, Mi, Wang. Isomorphic Pruning for Vision Models. arXiv:2407.04616. [V API]
63. Ma, Fang, Wang. LLM-Pruner. NeurIPS 2023. arXiv:2305.11627. [V API]
64. An, Zhao, Yu, Tang, Wang. Fluctuation-based Adaptive Structured Pruning for Large Language Models. AAAI 2024. arXiv:2312.11983. [V API]
65. Sun, Liu, Bair, Kolter. A Simple and Effective Pruning Approach for Large Language Models (Wanda). ICLR 2024. arXiv:2306.11695. [V API]
66. Frantar, Alistarh. SparseGPT. arXiv:2301.00774. [V API]

**Heavy-tail metrics and layer rules**
67. Zhou, Pang, Liu, Martin, Mahoney, Yang. Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training (TempBalance). NeurIPS 2023. arXiv:2312.00359. [V abs]
68. Martin, Peng, Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications 12, 2021. doi:10.1038/s41467-021-24025-8; arXiv:2002.06716. [V API]
69. Lee, Park, Mo, Ahn, Shin. Layer-adaptive sparsity for the Magnitude-based Pruning (LAMP). ICLR 2021. arXiv:2010.07611. [V API]
70. Evci, Gale, Menick, Castro, Elsen. Rigging the Lottery: Making All Tickets Winners. ICML 2020. arXiv:1911.11134. [V API]

**Recovery procedure and recipe selection**
71. Tessera, Matowe, Pretorius, Rosman, Hooker. Just-in-Time Sparsity: Learning Dynamic Sparsity Schedules. ICML 2022 workshop on dynamic neural networks. icml.cc/virtual/2022/19284. [U: listing and snippets only]
72. Ganjdanesh, Gao, Huang. Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment. CVPR 2024. arXiv:2403.19490. [V abs]
73. Wu et al. Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch. arXiv:2403.14729. [V API]
74. Hu, Henderson, Cano. ICE-Pruning: An Iterative Cost-Efficient Pruning Pipeline for Deep Neural Networks. IJCNN 2025. arXiv:2505.07411. [V abs]
75. Wang, Wang, Shi. AutoMC: Automated Model Compression based on Domain Knowledge and Progressive search strategy. doi:10.1109/icde60146.2024.00147; arXiv:2201.09884. [V abs]
76. Öztürk, Ferreira, Jomaa, Schmidt-Thieme, Grabocka, Hutter. Zero-Shot AutoML with Pretrained Models. ICML 2022. arXiv:2206.08476. [V abs]
77. Pineda Arango, Ferreira, Kadra, Hutter, Grabocka. Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How. arXiv:2306.03828. [V abs]
78. Jafari, Bilwal, Zhou, Gelder, Jannesari. APQF: Agentic Profiling-Guided Structured Pruning and Mixed-Precision Quantization with Adaptive Fine-Tuning. arXiv:2608.05499 (Aug 2026). [V abs]
79. Li, Yumer, Ramanan. Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints. ICLR 2020. arXiv:1905.04753. [V API]
81. Sun, Lakshmanan, Shen, Lan, Chen, Alvarez. Multi-Dimensional Pruning: Joint Channel, Layer and Block Pruning with Latency Constraint. arXiv:2406.12079; CVPR 2025 version arXiv:2504.02168. [V abs]

**Predictors**
82. Mills et al. GENNAPE: Towards Generalized Neural Architecture Performance Estimators. AAAI 2023. arXiv:2211.17226. [V abs]
83. White, Zela, Ru, Liu, Hutter. How Powerful are Performance Predictors in Neural Architecture Search? NeurIPS 2021. arXiv:2104.01177. [V API]
84. Wang, Lu, Xiao, Liu, Zhou. Learning Accurate Performance Predictors for Ultrafast Automated Model Compression (SeerNet). IJCV. arXiv:2304.06393. [V abs]
85. Abdelfattah, Mehrotra, Dudziak, Lane. Zero-Cost Proxies for Lightweight NAS. ICLR 2021. arXiv:2101.08134. [R: S13]
86. Kuhn, Lyle, Gomez, Rothfuss, Gal. Robustness to Pruning Predicts Generalization in Deep Neural Networks. arXiv:2103.06002. [V API, title only]
87. Lim, Maron, Law, Lorraine, Lucas. Graph Metanetworks for Processing Diverse Neural Architectures. arXiv:2312.04501. [V abs]
88. Unterthiner, Keysers, Gelly, Bousquet, Tolstikhin. Predicting Neural Network Accuracy from Weights. arXiv:2002.11448. [V API]
89. Akhauri, Abdelfattah. Encodings for Prediction-based Neural Architecture Search. arXiv:2403.02484. [V API]
90. Ji, Cao, Liu. Pruning Large Language Models via Accuracy Predictor. arXiv:2309.09507. [V API]

**Latency**
92. Yang et al. NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications. ECCV 2018. arXiv:1804.03230. [V API; facts R: S9]
93. Shen, Yin, Molchanov, Mao, Liu, Alvarez. Structural Pruning via Latency-Saliency Knapsack (HALP). NeurIPS 2022. arXiv:2210.06659. [R: S9]
94. Humble, Shen, Albericio Latorre, Darve, Alvarez. Soft Masking for Cost-Constrained Channel Pruning. ECCV 2022. arXiv:2211.02206. [V API]
95. Kim, Kwon, Lee, Kim, Ha. CPrune: Compiler-Informed Model Pruning for Efficient Target-Aware DNN Execution. ECCV 2022. arXiv:2207.01260. [V API]
96. Lee, Lee, Chong, Hwang. HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning. NeurIPS 2021. arXiv:2106.08630. [V API]
97. Sukthanker, Zela, Staffler, Dooley, Grabocka, Hutter. Multi-objective Differentiable Neural Architecture Search (MODNAS). ICLR 2025. arXiv:2402.18213. [R: S9]
98. Li, Chen, Liu, Zhu, Tian, Liu. MCMC: Multi-Constrained Model Compression via One-Stage Envelope Reinforcement Learning. IEEE TNNLS 2024. doi:10.1109/TNNLS.2024.3353763. [V abs, Semantic Scholar record]
99. Zhang, Li, Lu, Yang, Tang. Hardware-Aware DNN Compression for Homogeneous Edge Devices. doi:10.1109/DOCS67533.2025.11200827; arXiv:2501.15240. [V API]

**Downstream transfer**
100. Molchanov, Tyree, Karras, Aila, Kautz. Pruning Convolutional Neural Networks for Resource Efficient Inference. ICLR 2017. arXiv:1611.06440. [V API]
101. Liu, Cai, Guo, Chen. TransTailor: Pruning the Pre-trained Model for Improved Transfer Learning. AAAI 2021. arXiv:2103.01542. [V abs]
102. Iofinova, Peste, Kurtz, Alistarh. How Well Do Sparse ImageNet Models Transfer? CVPR 2022. arXiv:2111.13445. [V abs]
103. Chen et al. The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models. CVPR 2021. arXiv:2012.06908. [V API]
104. Caccia et al. Towards Compute-Optimal Transfer Learning. arXiv:2304.13164. [V abs]
105. Iurada, Occhiena, Tommasi. A Second-Order Perspective on Pruning at Initialization and Knowledge Transfer. ICIAP 2025. arXiv:2509.24066. [V abs]
106. Farina, Mancini, Cunegatti, Liu, Iacca, Ricci. MULTIFLOW: Shifting Towards Task-Agnostic Vision-Language Pruning. CVPR 2024. arXiv:2404.05621. [V abs]
107. Morcos, Yu, Paganini, Tian. One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. NeurIPS 2019. arXiv:1906.02773. [V API]
108. Hedegaard, Alok, Jose, Iosifidis. Structured Pruning Adapters. arXiv:2211.10155. [V API]

**Criteria**
109. He, Ding, Liu, Zhu, Zhang, Yang. Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration (LFPC). CVPR 2020, CVF open access. [R: S20; no arXiv record found by title]
110. Liu, Kung, Wentzlaff. Evolving Transferable Neural Pruning Functions. GECCO 2022. doi:10.1145/3512290.3528694; arXiv:2110.10876. [V abs]
111. Huang, Shao, Wang, Lin, Luo. Convolution-Weight-Distribution Assumption: Rethinking the Criteria of Channel Pruning. NeurIPS 2021. arXiv:2004.11627. [V API; the finding as quoted in a search summary]
112. Chen et al. Evolving filter criteria for randomly initialized network pruning in image classification. Neurocomputing 2024. doi:10.1016/j.neucom.2024.127872. [U: title and venue only]

**Core claim**
113. Liu, Wang, Zhang. Meta Pruning via Graph Metanetworks: A Universal Meta Learning Framework for Network Pruning. arXiv:2506.12041 (v3, 15 Dec 2025). [V: v3 HTML, main text; appendices F and G not read]
114. Gonzalez-Sahagun, Conant-Pablos, Ortiz-Bayliss, Cruz-Duarte. Two generalist RL compression agents. IEEE Access 12, 2024. doi:10.1109/ACCESS.2024.3385857; doi:10.1109/ACCESS.2024.3457863. [R: S9, abstracts]
115. Yu, Mazaheri, Jannesari. Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning (GNN-RL). ICML 2022. arXiv:2102.03214. [R: S9]
116. Yu, Mazaheri, Jannesari. Auto Graph Encoder-Decoder for Neural Network Pruning (AGMC). ICCV 2021. arXiv:2011.12641. [R: S9]
117. Ashok, Rhinehart, Beainy, Kitani. N2N Learning. ICLR 2018. arXiv:1709.06030. [R: S9]
118. Mu et al. Boosting the Convergence of Reinforcement Learning-Based Auto-Pruning Using Historical Data. IEEE TCAD 2024. doi:10.1109/TCAD.2023.3319594. [R: S9]
119. Venkatesan, Swaminathan, Zhou, Luo. Out-of-the-box channel pruned networks. arXiv:2004.14584. [R: S9]
120. Balemans, Huybrechts, Steckel, Mercelis. Resource-Aware Neural Network Pruning Using Graph-based Reinforcement Learning. arXiv:2509.10526. [R: S9]
121. Esmat, Banisharif, Jannesari. AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search. arXiv:2601.12272. [R: S9]
122. Jafari, Sarkar, Bilwal, Jannesari. ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization. arXiv:2509.05584. [V abs]
123. Kodathala, Vunnam. LLMs can Compress LLMs: Adaptive Pruning by Agents. arXiv:2601.09694. [V abs, first half]
124. Baghdadi, Belhamidi, Belhadj, Boulmerka, Farhi. Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization. arXiv:2606.26002. [V abs]
125. Baradaran, Raji. Reinforcement Learning for Neural Network Compression: A Taxonomy of MDP Design, Action Spaces, Reward Engineering, and Learning Algorithms. IEEE Access, 2026. doi:10.1109/ACCESS.2026.3749513. [V abs, Semantic Scholar record]
126. Liang, Wang, Xu, Zhou, Lu. EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models. CVPR 2025. arXiv:2503.15369. [V abs]

(Numbers 80 and 91 are unused.)

---

## 15. Search log and limits

**Searches run (21 Nov 2026).**
- arXiv API, abstract-field queries, 2022/2023 onward, newest first, titles screened and promising records opened: 12 queries on transferable or generalisable pruning policies (a first attempt failed on URL quoting and was rerun); 14 on predictors, schedules, heavy tails, benchmarks, latency, downstream transfer, post-training pruning and rule discovery; 12 on offline or architecture-agnostic agents and adaptive schedules; 6 on "pruning agent", "universal pruning" and cross-architecture pruning; 14 title look-ups.
- Web searches: 19, on each direction and on the core claim.
- Semantic Scholar: the 22 listed citations of NEON, screened by title; three abstracts opened.
- Full texts read: PruningBench, Wang et al. 2023, Zimmer et al. (main text), AlphaPruning (to App. G), EvoPress, SPA, RL-Pruner, the metanetwork paper (v3, main text).

**Limits.**
- Abstract-keyword search under-covers IEEE, Elsevier and Springer venues with no arXiv copy; the NEON citation list is the only check there.
- Page summaries came through a small reading model; table values quoted from them were cross-checked against arXiv records or earlier repo scans where one existed. The values of EvoPress's table, PruningBench's observations, OBSPA's Table 4, RL-Pruner's Table 1 and the metanetwork paper's Tables 4–5 rest on that single reading.
- Not opened: DFPC [33]; the body of Just-in-Time Sparsity [71]; Zimmer et al.'s appendices on structured pruning and on allocation; AlphaPruning's App. H (structured LLM pruning; fine-tuning); the appendices of [113]; IEEE full texts of [98, 114, 125].
- A claim seen in a search summary, that the NeurIPS 2026 main track offers a "negative results" contribution type, was not confirmed on the call page and is not used.
- The standing session-start checks that poll the cluster were not run: the brief forbids cluster access.
