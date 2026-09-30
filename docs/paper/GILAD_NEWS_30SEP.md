# SPECTRA — weekly meeting with Gilad, 1 Oct 2026

**From:** Ido Paretsky. Prepared 30 Sep; numbers as of 19:45 IDT.
**This note replaces the 29 Sep status note.** Several of its bad-news items are now obsolete (§0).
**Your two standing topics** are §2 (the benchmarking methodology and roadmap) and §3 ("throw away the pruned layer and train a new one": NEON's layer replacement, carried to CNNs).
**Record:** every number is in `docs/paper/RESULTS_LEDGER.md`; the § in each row points there. All rows are preliminary.

**How to read the numbers**

- **Δ** is the change in **test** accuracy, in percentage points; negative means the pruned network is worse. "−2.5 @ 66 %" means −2.5 points with 66 % of the parameters kept (1.51× fewer).
- **Two halves of the test set.** Since this week the 10k CIFAR test set is split once, with a fixed seed, into a 5k *validation* half that makes every choice and a 5k *test* half that we report (§1.1). Numbers are on the test half unless marked **10k**. The 10k figure (both halves) is quoted only at points chosen by size, where nothing read the validation half.
- **The heuristic walk.** Every row this week is a no-agent walk: a fixed rule that keeps 90 % of each prunable layer group at every step (ops name: *mild*), two passes over the network unless stated. It makes the same cut at the same step in every run, so two recovery recipes are compared on **identical architectures**. After each cut the network is briefly fine-tuned (the *walk fine-tune*: Adam 1e-3, 40 epochs with patience 10 at test time; 12 epochs with patience 4 inside agent training).
- **The quoted point** is the most compressed point whose **validation** drop is still within the 10-point budget (τ = 10; ops name: *val_best*), or a point chosen by size (a *size point*: 80 % or 70 % of parameters kept, or a paper's FLOPs ratio). Test accuracy never chooses a point.
- **Benchmark networks** are our three benchmark cells: ResNet-56 on CIFAR-10, VGG-16 on CIFAR-10 and VGG-19 on CIFAR-100 (§2.2). **Diagnostic networks** are two deliberately narrow ResNets (ResNet-20 at width ×2, ResNet-56 at width ×4): cheap to walk, hard to prune.

## Agenda (about 60 minutes)

| | Item | Minutes |
|---|---|---|
| 0 | Executive summary: wins, results, what is crossed off or obsolete, the 29 Sep note's questions | 10 |
| 1 | The three advances, one paragraph each | 10 |
| 2 | Benchmarking methodology and roadmap | 10 |
| 3 | Layer replacement on CNNs: recommendation and questions | 10 |
| 4 | The agent: the first run under the corrected protocol; the multi-dataset train | 5 |
| 5–9 | Details as needed; what we do not claim; action items; questions for you; discussion | 15 |

---

## 0. Executive summary

SPECTRA's claim is unchanged. It is one frozen, generic pruning agent, trained once offline on a catalog of pretrained CNNs and applied to networks it never saw, with no per-target search or training. Every method we compare with trains, searches or regularizes on the target.

This week we found that the validation images that scored every pruning decision had been memorized by the pretrained networks. Scoring on held-out images instead, and giving the per-step recovery the standard CIFAR augmentation, removes most of the accuracy cost we had attributed to pruning. A full-width ResNet-56 on CIFAR-10 now loses 0.06 points at 1.51× parameter compression, and CIFAR-100 goes from 0 to 8 of 8 recoverable networks. The agent's first training run under the corrected protocol started on 30 Sep at 03:14, and its first test is due around 2–3 Oct. **Nothing below is an agent result yet.** Every row is the no-agent heuristic, which is now the bar the agent has to clear.

### Top five wins

1. **A validation leak, found and fixed** *(infrastructure and diagnosis; §141–§143).*
   - *The leak.* Every public checkpoint we prune was trained on all 50k CIFAR training images, and our validation set was a slice of those images.
   - *Its size.* Unpruned validation accuracy read 1.000 / 1.000 / 0.999 on the three benchmark networks. Their real test accuracy is 0.943 / 0.936 / 0.739.
   - *The fix.* Validation now comes from held-out test images (clean validation; ops name: *protocol P*). It tracks test within ~1.5 points.
   - *What it explains.* One flaw accounts for three failures: the agent's collapse into the heuristic, "unrecoverable" CIFAR-100, and no cut inside the budget on VGG-19 CIFAR-100.
2. **One recovery recipe now works on CIFAR-100, and the 16-network training catalog is built** *(result and infrastructure; §148).*
   - Last week the standard recipe admitted 0 of 8 CIFAR-100 networks. Four alternative recipes admitted 0–4 of 8, and the one at 4 broke the CIFAR-10 control. Now all 8 are admitted, with the standard recipe.
   - The catalog of 8 CIFAR-10 + 8 CIFAR-100 networks is emitted. A unit test checks that no training network is also a test network.
   - This answers the 29 Sep note's fourth question (one recipe for every network: yes) and unblocks NEON's multi-dataset training, on CNNs.
3. **Crop+flip in the per-step recovery makes the same cuts 2–6 points kinder** *(result; §148, §150, §152).*
   - At identical architectures and 66 % of parameters kept: ResNet-56 C10 −2.84 → **−0.06**; VGG-16 C10 −2.8 → **−0.5**; VGG-19 C100 −6.7 → **−2.5**.
   - It passed its pre-registered rule on all three benchmark networks and on the diagnostic networks. It is now the recovery for every method, including inside the agent's training.
4. **First rows at DepGraph's own sizes, on DepGraph's own checkpoints, with an honest accounting** *(result and infrastructure; §149, §153–§155).*
   - *The protocol.* A 100-epoch SGD final fine-tune, which is the literature's protocol. A control gives the same fine-tune to the unpruned network, so only recovery from pruning is credited (the *honest gain*).
   - *ResNet-56 C10.* At DepGraph's two FLOPs ratios (2.11× and 2.57×) the heuristic walk is **−1.52 / −2.11** (10k), against their +0.24 / +0.11. That is about 2 points behind, with no agent and no training on the target.
   - *VGG-19 C100.* **−1.62** (10k) at 68 % of parameters kept.
5. **A training run built to finish, and reproducible walks** *(infrastructure; §151, §153).*
   - *The run.* The agent's run resumes itself past the cluster's 6-day limit. It is protected from the cluster's automatic requeue, which would have deleted its resume file. Its test is pre-registered.
   - *The walks.* They are deterministic: re-walks of the same cells agree to 0.04 points on average. The walk's networks are saved, so every later fine-tune recipe starts from the same weights.
   - *The readout.* It refuses to credit a recipe that hurts the unpruned network.

### Results at a glance

No agent. Each before → after pair is the same architecture; only the protocol or the recipe differs.

| Network (dataset) | Kept | Before | After | What changed | § |
|---|---|---|---|---|---|
| Three benchmark nets, unpruned: validation vs test | 100 % | val 1.000 / 1.000 / 0.999 vs test 0.943 / 0.936 / 0.739 | val within ~1.5 of test | clean validation | §141–§142 |
| CIFAR-100 networks recoverable, of 8 | — | 0 | **8** | clean validation + crop+flip | §109 → §148 |
| ResNet-56 (C10) | 66 % | −2.84 | **−0.06** | crop+flip | §152 |
| VGG-16 (C10) | 66 % | −2.8 | **−0.5** | crop+flip | §152 |
| VGG-19 (C100) | 66 % | −6.7 | **−2.5** | crop+flip | §152 |
| Diagnostic ResNet-56 ×4 (C10) | 79.5 % | −7.6 | **−2.6** | crop+flip | §152 |
| Diagnostic ResNet-56 ×4 (C10) | 79.5 % | −8.3 | **−2.7** (honest gain +5.3) | 100-epoch final fine-tune | §154 |
| DepGraph's ResNet-56 (C10), 10k | FLOPs 46 % / 38 % | −3.81 / −4.60 | **−1.52 / −2.11** | 100-epoch final fine-tune | §153 · DepGraph +0.24 / +0.11 |
| DepGraph's VGG-19 (C100), 10k | 68 % | −6.09 | **−1.62** | crop+flip + final fine-tune | §147, §155 · DepGraph −3.11 at 8.92× FLOPs |

### The 29 Sep note's top losses, today

| 29 Sep loss | Today |
|---|---|
| Layer replacement does not carry from dense nets to residual CNNs | **Confirmed and closed.** It ran under clean validation with NEON's own stopping rule and was 12–54 points worse at the same steps (§3). |
| No learned policy beats the same-loop heuristics yet | **Still true, but every one of those agents was trained on memorized validation.** The first agent under the corrected protocol is training; its first test is ~2–3 Oct (§4). The reward-shape comparisons are re-run only if that train calls for it. |
| One fine-tune recipe does not yet recover CIFAR-100 (at most 4 of 8 under five recipes, and that one broke the CIFAR-10 control) | **Obsolete:** 8 of 8 with the standard recipe (§5.1). |
| VGG-19 CIFAR-100 selects the unpruned network | **Obsolete:** −2.5 at 66 % kept; −1.62 (10k) at 68 % with the final fine-tune. |
| Informed reset variants and optimizer schedules failed their rules | **Still true, and moot.** The least-squares refit, the principal-component rebuild and batch-norm re-estimation (§3), and AdamW with warm-up and cosine, RAdam and a 40-epoch budget at two learning rates (§129–§133), all failed. Clean validation and crop+flip did what none of them did, with the original optimizer. |

Two further lessons this week:
- **The long final fine-tune is a protocol for literature rows, not a free gain.**
  - *Where the walk's recovery is weak* it adds +4 to +5.5 points.
  - *Once the walk already uses crop+flip* it adds nothing: −0.4 to 0.0 points, while the unpruned control gains +0.5 (VGG-19 C100, §155).
  - *On the smallest network* (diagnostic ResNet-20, 5k parameters) it lifts only the unpruned control, by +3.5 (§154).
- **Three agent designs are parked, not disproven.** The two-decision head, the budget-plus-stop action and the group-as-token state were all trained on memorized validation. We re-run one agent under the corrected protocol before reopening any design question (§4.2).

### The 29 Sep note's questions to you: where they stand

| Question in the 29 Sep note | Answer this week |
|---|---|
| 1. Retry layer replacement under NEON's training-loss stopping rule before closing it? | Done, under clean validation: 12–54 points worse (§156). We propose to close it (§3). |
| 2. The linear in-budget reward vs NEON's cubic, when no cut had ever raised accuracy? | The "no increase" came from the recovery recipe, not the reward. With crop+flip the best cuts come within 0.1–0.3 points of the unpruned network, and 2 of 46 on VGG-19 C100 are slightly above it on validation (+0.28 at most; none on test) (§152, §155). The cubic's bonus branch is now at the edge of reach, but not yet worth a train. |
| 3. Does the benchmarking plan hold? | Yes, with this week's protocol changes and one correction: the "OCSPruner, 42 % of parameters" figure in our 21 and 27 Sep notes was its ResNet-56 point (§2). |
| 4. One fine-tune recipe for every network? | Yes. The same recipe admits 8 of 8 CIFAR-100 networks (§148), so CIFAR-100 joins the training pool. |

### How big is this?

- **Clean validation is a diagnostic breakthrough.** It is the most consequential finding since August, because one flaw explains three failures. It is not yet a result: nothing shows the agent beating the heuristic under it so far.
- **Crop+flip is a large correction.** It brings our recovery up to the literature's standard. Together with clean validation, it removes both reasons the agent had to prefer the mildest cut. Cuts looked more expensive than they were (memorized validation), and they were more expensive than necessary (un-augmented recovery).
- **The final fine-tune is protocol alignment.** It makes the literature rows comparable and does not change what the agent learns.
- **The results breakthrough would be a frozen agent at or above the heuristic at equal size, on networks it never trained on, under the corrected protocol.** The first read is around 2–3 Oct. Then comes the multi-dataset agent, transferred frozen to ImageNet.

---

## 1. The three advances

### 1.1 Validation on held-out test images (clean validation; ops name: protocol P)

**Main idea.** Every decision in the loop is now scored on images that neither the pretrained network nor our fine-tune has seen: the reward, the 10-point budget check and the choice of the quoted point. Every reported number comes from different images again. The 10k CIFAR test set is split once, with a fixed seed, into a 5k validation half and a 5k test half. Fine-tuning uses all 50k training images at batch 256. **Literature source.** Selecting on held-out data and reporting on separate data is the standard model-selection protocol. Halving the official test set into validation and test is how NAS-Bench-201 (Dong & Yang, ICLR 2020) builds its CIFAR-100 split. Zhang et al. (ICLR 2017) show that CIFAR networks fit their training images essentially perfectly, so a validation slice of the training split measures memorization. **Justification.** Every public checkpoint we prune (the model zoo, DepGraph's releases) was trained on all 50k training images, and our validation set was 5k of them. A cut therefore looked like a fall from memorized accuracy (≈ 1.000) to real accuracy: 6 to 26 points on the benchmark networks before any real damage. The reward, the budget check and the quoted point all measured forgetting, so the mildest policy always looked best. **Effect.** Unpruned validation now agrees with test within ~1.5 points. VGG-19 C100 has a cut inside the budget for the first time (−6.7 at 66 % kept, before crop+flip), and CIFAR-100 is admitted to training. The cost is precision. One standard error on a 5k half is ~0.4 points at 93 % accuracy and ~0.6 at 74 %, and a re-walk on a different GPU model can move a single point by up to 0.8 (§149), so we caption differences under ~1 point as noise. The walk fine-tune selects its epoch on training loss and never reads either half.

### 1.2 Crop+flip in the per-step recovery

**Main idea.** The short fine-tune after each cut now sees the augmentation the networks were trained with: pad each image by 4 pixels, take a random 32×32 crop, and flip it horizontally at random. It applies to CIFAR only; SVHN digits are not flipped. The optimizer, epochs and cuts are unchanged. **Literature source.** This is the standard CIFAR training augmentation of He et al. (CVPR 2016). Every network in our catalog was trained with it, and the CIFAR pruning papers we compare with fine-tune with it. **Justification.** Without augmentation the recovery fine-tune overfits the training images within a few epochs. Clean validation made that overfitting visible as lost accuracy, most of all on CIFAR-100, which has 500 images per class. A recovery that does not match the training conditions charges the pruning for the recovery's own weakness. **Effect.** At identical architectures crop+flip is kinder at 12 of 12 size points on CIFAR-100 (+0.9 to +6.2, mean +3.3). It is +2.2 to +4.9 points kinder on the three benchmark networks and +5.0 on the diagnostic ResNet-56. It met its pre-registered rule everywhere, so it is the recovery for every method from now on, and it is inside the agent's training. It costs 18 % more time per fine-tune epoch. It hurts one network: the 5k-parameter diagnostic ResNet-20 (64.8 % accuracy) loses 1–3 points. That is expected for a network that underfits (NetAug, Cai et al. ICLR 2022), and that network is a diagnostic, not a training network.

### 1.3 The 100-epoch SGD final fine-tune, with an unpruned control

**Main idea.** Once the walk has fixed the architecture, the pruned network gets one long recovery from the inherited weights: SGD, learning rate 0.01, momentum 0.9, weight decay 5e-4, cosine schedule, crop+flip, batch 128, 100 epochs. The unpruned network gets the same recipe (the *origin control*). We report the *honest gain*: the pruned network's improvement minus the unpruned network's improvement. A recipe that simply trains better than the original checkpoint earns no credit. **Literature source.** Long SGD fine-tuning after pruning is how CIFAR pruning results are published. PruningBench (2024) standardizes a 100-epoch SGD fine-tune, and DepGraph (Fang et al., CVPR 2023) fine-tunes after pruning. Le & Hua (ICLR 2021) show that the retraining schedule alone can reorder pruning methods. **Justification.** Setting our 40-epoch per-step recovery beside their long fine-tune would understate the pruning decisions. The origin control guards against the opposite error. On the diagnostic ResNet-20, for example, the recipe lifts the unpruned network by 3.5 points. The final fine-tune is used only in the rows set beside published numbers. Comparisons between the agent and the heuristics stay on the per-step recovery, and the reward never sees it. **Effect.** The honest gain depends on how good the walk's recovery already was:
- +4.1 to +5.5 on DepGraph's VGG-19 C100 and +5.3 to +5.5 on the diagnostic ResNet-56, both after the un-augmented walk;
- +1.2 to +1.8 on DepGraph's ResNet-56 C10;
- about zero once the walk already uses crop+flip (VGG-19 C100);
- negative on the diagnostic ResNet-20.

So crop+flip in the walk and the long final fine-tune recover much of the same accuracy: on the diagnostic ResNet-56 at 79.5 % kept, either one alone reaches about −2.6. We keep the long fine-tune for literature rows because it is the published protocol.

---

## 2. Benchmarking methodology and roadmap

### 2.1 The claim and the three bars

**The sentence we want to be able to write.** On DepGraph's CIFAR benchmark cells, plus the VGG-16 cell of OCSPruner and HRank, a single frozen SPECTRA agent prunes to size-matched operating points with no per-network search. It is trained once on a catalog that contains none of these architectures and is never adapted to a target. Its accuracy is reported next to the same-loop heuristics and next to the published, differently fine-tuned results.

The three bars come in this order, and one never substitutes for another:

| Bar | What it means | Status, 30 Sep |
|---|---|---|
| **1. Budget** | One frozen agent, zero per-network search or training | Met by design; written first |
| **2. Same loop, matched size** | On networks it never saw, the agent is at least as accurate as the same-loop heuristics at equal size, or reaches a size inside the budget that they cannot | Not met yet. First test on the diagnostic networks ~2–3 Oct (§4.1), then the benchmark cells |
| **3. Beside the published number, size-matched** | Our row at their size, next to their row; expected below, and said so in print | First heuristic rows exist: ~2 points behind DepGraph on ResNet-56 at equal FLOPs (§2.5) |

### 2.2 What we train on and what we test on

- **Training: one offline run, then frozen.**
  - *The catalog.* Pretrained networks from several families (thin and standard ResNets, VGG-BN, MobileNetV2, DenseNet). Today it is nine CIFAR-10 networks plus one SVHN VGG. Next is 8 CIFAR-10 + 8 CIFAR-100 (§4.3).
  - *Exclusion.* Every benchmark architecture is excluded by architecture, not only by weights: no standard-width ResNet-56, no VGG-16 on CIFAR-10, no VGG-19 on either dataset.
  - *Enforcement.* An automatic test fails if a training file and a test file ever share a network.
- **Test set 1: the benchmark cells** (how SPECTRA looks on the field's own benchmark).

  | Cell | Checkpoint we prune | Whose published results sit on this cell |
  |---|---|---|
  | ResNet-56 · CIFAR-10 | DepGraph's released weights (93.53 %; 93.2 % in our loader), plus a model-zoo copy | DepGraph, OCSPruner, AMC, FPGM, HRank, ResRep, GReg, C-SGD, SFP, Polar |
  | VGG-16 · CIFAR-10 | Model-zoo checkpoint (93.6–93.7 % in our loader) | OCSPruner, HRank, Network Slimming (VGG-19 variant), Li et al. |
  | VGG-19 · CIFAR-100 | DepGraph's released weights (73.50 %), plus a model-zoo copy (73.87 %, the same base PruningBench uses) | DepGraph, OCSPruner, GReg, EigenDamage, PruningBench |

- **Test set 2: transfer coverage** (did the frozen agent transfer?). Networks and datasets the agent never trained on:
  - *similar:* the same families at other widths and depths (ResNet-20 ×16, ResNet-56 ×10, ResNet-44, VGG-19 on CIFAR-10, MobileNetV2 ×0.75, DenseNet-100);
  - *unlike:* families never trained on (ShuffleNetV2, RepVGG);
  - *thin:* the two diagnostic networks, narrower than anything in training;
  - *datasets:* Fashion-MNIST, ImageNet and, for the multi-dataset agent, SVHN.
- **ImageNet is tested frozen, never trained on** (your directive).
  - *Why.* One CIFAR fine-tune epoch takes under a minute, and one ImageNet epoch takes about an hour. An agent episode would take days, and a training run about a year of one GPU.
  - *The test itself.* The frozen agent walks ResNet-50, MobileNetV2, DenseNet-121, VGG-16 and ShuffleNet with the same loop, at tens of GPU-hours per network. ResNet-50's bottleneck blocks appear in no training network.
- **Both test sets feed your two artifacts.** The *coverage matrix* is a family × dataset transfer map. The *Pareto plot* shows test Δ against parameters and FLOPs kept, with the heuristics and the published points. Neither replaces the other.

### 2.3 One protocol for every method (this week's changes in bold)

- **The same loop.**
  - *Shared parts.* The learned agent and the same-loop heuristics share the walk, the recovery and the selection rule. The heuristics are the 90 % rule and the L1-magnitude walk; your 18 Aug list adds greedy, random and look-ahead.
  - *Published numbers* keep their own fine-tuning protocol in the caption; nothing is converted.
- **Recovery at each cut.** Keep the surviving filters and fine-tune the whole network: Adam 1e-3, 40 epochs with patience 10 at test time. **Now with crop+flip** (§1.2).
- **Validation.** **Now a fixed half of the test set** (§1.1); results are on the other half.
- **Two operating points per method, on every cell.**
  - *(1) Our rule:* the most compressed point whose validation drop is within 10 points.
  - *(2) Their size:* the walk continues to the published compression. It is reported even if validation has left the budget, and labelled size-matched.

  Only (2) is printed beside a published number. Sizes are matched on FLOPs, the ratio the papers report, and parameters are printed beside them; published VGG points keep far fewer parameters than FLOPs (HRank: 17.1 % against 46.5 %). The published sizes:
  - *ResNet-56:* DepGraph's 2.11× and 2.57× (46 % and 38 % of FLOPs kept).
  - *VGG-16 (corrected):* HRank's 46.5 % and OCSPruner's 21.2 % of FLOPs kept. The earlier "OCSPruner, 42 % of parameters" was its ResNet-56 point.
  - *VGG-19 C100:* DepGraph's 8.92× (11 % of FLOPs kept).
- **The final fine-tune** (§1.3) is applied only to rows set beside published numbers, with the origin control. It is never used inside agent training, because 100 epochs per cut would multiply training by an order of magnitude and mix the solver's budget into the agent.
- **The discipline for every new idea.**
  - Each idea is a switch that is off by default.
  - It is tested first without an agent, on the fixed walk, against the live recipe.
  - Its adopt or kill rule is written down before the run.
  - A loser is crossed off in writing.

  This week's three advances and the layer-replacement verdict were all decided this way. Every evaluation of an agent replays its pinned policy contract, the exact settings it was trained with.

### 2.4 Metrics and budgets

**One row per method and network:**
- original accuracy, pruned accuracy, and Δ in points;
- parameters kept (fraction and millions) and FLOPs kept (fraction and millions);
- speed-up = 1 / FLOPs kept;
- the recovery recipe and its budget.

The original accuracy is always printed, because checkpoints differ between our loader and the papers'. Where a heuristic cannot reach the agent's size inside the budget, that is printed explicitly: the gap is the learned schedule's result.

| Method | Work on each new target network | Recovery |
|---|---|---|
| **SPECTRA** | **None.** One offline training on the catalog (days of one GPU), amortized over every later network | Per accepted cut, 40 epochs / patience 10 with crop+flip; a 2-pass walk of a full-width ResNet-56 or VGG-16 is 2–6 GPU-hours. Literature rows add one 100-epoch fine-tune |
| Same-loop heuristics | None | Same as SPECTRA |
| DepGraph | Dependency grouping, then group-sparsity learning on the target | Fine-tune |
| OCSPruner | One full training cycle from scratch on the target (300 SGD epochs on CIFAR, mean of 3 runs) | Inside the cycle |
| AMC | A reinforcement-learning search per target (DDPG, hundreds of episodes) | Fine-tune |
| Network Slimming, GReg, ResRep, C-SGD, Polar | Sparsity or regularized training on the target | Fine-tune, or iterate |
| PruningBench (a protocol) | Iterative pruning to a FLOPs target | Fixed 100-epoch SGD fine-tune |

Two totals will be reported:
- *Cost per additional network:* SPECTRA is lower by construction, at one walk.
- *Cost of the first network, including our offline training:* SPECTRA is higher.

Our costs come from Slurm job times. Theirs come from the epoch counts in their papers and scripts, multiplied by a CIFAR epoch measured on our GPU.

### 2.5 The counterparts, and where our rows stand

"Ours" rows are the no-agent heuristic walk under this week's protocol.

| Cell | Published result (before → after, at size) | Ours now | Pending |
|---|---|---|---|
| **ResNet-56 · C10** | **DepGraph** (CVPR 2023): 93.53 → 93.77 (+0.24) at 2.11×, 93.64 (+0.11) at 2.57×; −0.07 at 2.11× without its sparsity training. **OCSPruner** (WACV 2026): −0.32 at 38.9 % FLOPs / 41.4 % params. **HRank** (CVPR 2020): −0.09 at 50 % FLOPs. **GReg** (ICLR 2021): −0.18 / 0.00 at 2.55×. **ResRep** (ICCV 2021): 0.00 at 2.12×. **FPGM** (CVPR 2019): −0.33 at 1.70×. **AMC** (ECCV 2018): −0.9 at 2.0×. **C-SGD** +0.05 at 2.55×; **Polar** +0.03 at 1.88×; **SFP** −0.23 at 2.11× | DepGraph's checkpoint, walk + final fine-tune (10k): **−1.52 at 2.11×, −2.11 at 2.57×** (§153). Zoo checkpoint, crop+flip walk only: **−0.06 at 66 % params** (§152) | DepGraph's checkpoint with the crop+flip walk + final fine-tune (overnight); distillation and AutoAugment in the final fine-tune; the frozen agent |
| **VGG-16 · C10** | **HRank**: 93.96 → 93.43 (−0.53) at 46.5 % FLOPs / 17.1 % params. **OCSPruner**: 93.88 at 26.0 % FLOPs, 93.76 at 21.2 %; from a pretrained start 94.07 → 93.63 (−0.44) at 21.2 % FLOPs / 13.7 % params. **Network Slimming** (VGG-19 on C10): +0.14 at 49 % FLOPs / 11.5 % params | Crop+flip walk only: **−0.5 at 66 % params / 68 % FLOPs** (§152) | A 10-pass walk to 46.5 % and 21.2 % FLOPs, with the final fine-tune (running); the final fine-tune at 66 % (running); the frozen agent |
| **VGG-19 · C100** | **DepGraph**: 73.50 → 70.39 (−3.11) at 8.92×; −5.90 without its sparsity training. **OCSPruner**: 70.47 at ≈11 % FLOPs; from a pretrained start 73.58 → 69.98 at 11.2 % FLOPs / 10.1 % params. **GReg**: 74.02 → 67.55 / 67.75 at 8.84×. **PruningBench** (base 73.87 = our zoo copy): +0.01 at 2× (L2 magnitude); −1.45 at 4× (OBD-C); −3.96 at 8× (LAMP) | DepGraph's checkpoint, crop+flip walk + final fine-tune (10k): **−1.62 at 68 % params** (§155). Zoo copy, crop+flip walk only: **−2.5 at 66 %** (§152) | DepGraph's 8.92× size needs a ~11-pass walk: **proposed, not queued**; the frozen agent |

Published values are the papers' own, or as reprinted in DepGraph's Table 1. OCSPruner trains its own bases inside its cycle, so only its published rows are quoted. Network Slimming's ResNet-164 and DenseNet-40 points become rows once those networks are imported (action item 8); HRank's GoogLeNet is not planned. Liu et al. (ICLR 2019) is not a pruner: it shows that training the pruned architecture from scratch for a full budget matches fine-tuning. It is the basis of our scratch controls and of our reading of layer replacement (§3).

### 2.6 Roadmap and milestones

The dates are estimates at the measured pace, not deadlines.

1. **Now to 1 Oct: literature rows under the new protocol.**
   - DepGraph's ResNet-56 with the crop+flip walk and the final fine-tune (overnight).
   - The zoo ResNet-56 and VGG-16 likewise, and the VGG-16 walk to HRank's and OCSPruner's sizes.
   - The agent run's health check (~1 Oct 11:00).
2. **~2 Oct: the next code version.** When the no-agent queue drains:
   - build it (multi-dataset profile, a CIFAR-100 probe network, requeue safety, full provenance, the distillation teacher, crop+flip on the GPU to recover its 18 %) and smoke-test it;
   - train the new SVHN and Fashion-MNIST hold-out networks.
3. **~2 Oct night to 3 Oct: the first frozen-agent test**, on the diagnostic networks. This is bar 2 in its smallest form.
4. **If it passes:**
   - that agent runs on the three benchmark cells at both operating points, and on the coverage set;
   - the multi-dataset train launches (~8 days, to ~11–13 Oct), followed by ~3 days of hold-out tests (CIFAR-100 hold-outs, ImageNet, SVHN, Fashion-MNIST, coverage).

   **If it fails:** diagnose before any new train.
   - Replay the finished walks through the linear, cubic and NEON-exact rewards (no GPU).
   - Run a cubic-reward train, if cuts can gain accuracy.
   - Revisit the action menu.
5. **~7–12 Oct: the current run stops** by its own rule. Its final snapshots are tested, and its coverage set is walked.
6. **The thesis tables, all under the same protocol:**
   - the benchmark table (three cells × two operating points × methods);
   - the coverage matrix;
   - the Pareto plot.

| Milestone (ops name) | Fires when | Status | Then |
|---|---|---|---|
| Health check (M2) | Policy update 10: the critic's fit is positive on the last 3 updates and the policy is away from uniform | ~1 Oct 11:00 | A note; a flag is not a kill |
| The agent leaves the heuristic (M1) | A frozen snapshot is at or above the heuristic on both diagnostic networks at equal size, ≥ 1 point kinder somewhere or deeper inside the budget, and not a copy | First test ~2 Oct 22:00 to 3 Oct 23:30 | Benchmark cells and coverage set with that agent; the multi-dataset train on its launch condition (§4.3) |
| The agent copies the heuristic (M1-neg) | Two frozen snapshots are copies, or both are > 0.5 points worse on both networks | — | Diagnose first (step 4 above) |
| Crop+flip for every test walk (M3) | ≥ 1 point kinder on ≥ 2 of 3 benchmark networks, and the diagnostic guard holds | **Fired 30 Sep** (3 of 3) | Done |
| CIFAR-100 in the catalog (G0) | 8 of 8 admitted; catalog emitted and tested | **Done 30 Sep** | The multi-dataset train |
| A competitive-enough C10 literature row (M4) | DepGraph's ResNet-56, crop+flip walk + final fine-tune, within 1 point of DepGraph at 2.11× or 2.57× (10k) | Lands overnight | A row for you; never "beats" |
| A better final recipe (M5) | Distillation or AutoAugment ≥ +0.5 over the plain final fine-tune, with a healthy unpruned control | Queued | Adopted for every literature row |
| The cubic reward's bonus becomes reachable (M6) | A crop+flip walk on a full-width network has cuts with validation above the unpruned network | **Weakly fired 30 Sep:** 2 of 46 cuts on VGG-19 C100, +0.28 at most | Design the cubic-reward train (not a launch) |
| The next code version (G2) | The first agent test is submitted, or the no-agent queue drains | ~2 Oct | Build + smoke; new hold-out networks |
| The multi-dataset launch (G5) | The agent leaves the heuristic, the smoke is clean, catalog design A agreed, a free GPU | Proposed (§4.3) | ~8-day train, then the hold-out tests |
| The run stops (M7) | ≥ 250 episodes and 150 without a better probe, or the resume's time limit | ~7–12 Oct | Final snapshot tests, coverage set, the multi-dataset decision |

---

## 3. NEON's layer replacement on CNNs (recipes C-G and C-G+): worth more research?

**Main idea and source.** NEON (Hirsch & Katz 2022, §3 and Algorithm 1) does not remove neurons from the existing layer after a pruning action. It replaces the layer with a new one of the target width, initialized at random, freezes every other layer, and trains the new layer to convergence. The public source also rebuilds the next layer's input weights and a fresh batch-norm, and it stops on the training loss with patience 10.

SPECTRA's recovery does the opposite: it keeps the surviving filters and fine-tunes the whole network (recipe A). We carried NEON's construction to CNN layer groups in two forms:
- **C-G:** redraw the producing convolutions, their batch-norm and the consumer's input slice; freeze the rest; train the group.
- **C-G+:** C-G, then a short whole-network polish at one tenth of the learning rate.

**Why we tried it.** It is the recovery of the thesis's direct predecessor, and a frozen generic agent would inherit it naturally. Fresh weights might also avoid a bias that the surviving filters carry.

**Rigor.** In every redraw arm the new weights are Kaiming-normal, biases zero, and batch-norm reset to scale 1 / shift 0. Unit tests check that the redrawn weights change and, in the pruned-layer-only scope, that the consumer's weights do not. The empty result is the experiment, not a missed redraw.

**What we measured.** Same heuristic walk, same budget, no agent; the comparison is keep-the-survivors (recipe A) at the same steps.

| Recovery at each cut | Validation | Against keeping the survivors | § |
|---|---|---|---|
| C-G: random new group, trained to a validation plateau | memorized | no cut inside the budget on 3 of 3 networks (quoted point ≥ 98.8 % kept) | §100, §104 |
| C-G+: C-G, then a whole-network polish at 0.1× learning rate | memorized | no cut inside the budget on the two ResNet-56s; ResNet-20 −10.3 at 88 % kept vs −3.4 at 54 % | §101, §106 |
| Redraw the pruned layer only | memorized | same as C-G | §108 |
| C-PCA: a new layer built from the principal directions of the old activations | memorized | 1.1–2.6 points worse, at equal or shallower size | §127 |
| **C-G, NEON-literal** (training-loss stop, patience 10, up to 100 epochs) | **clean** | **12 to 54 points worse on validation** at the same steps; better on 0–4 % of cuts; stopped early by its pre-registered kill rule | §156 |
| *Keeps the survivors:* A + least-squares refit of every reader of the cut channels (He, Zhang & Sun 2017) | memorized | kinder on 1 of 3 networks (−6.2 vs −6.6), worse on 2; kept as an off switch | §126 |
| *Keeps the survivors:* A + batch-norm re-estimation alone | memorized | no gain | §128 |

The NEON-literal row, per network. Mean validation gap at the same steps:
- ResNet-56 C10: −30.7 (60 cuts);
- VGG-16 C10: −11.6 (25 cuts);
- diagnostic ResNet-20: −27.8 (16 cuts);
- diagnostic ResNet-56: −54.0 (28 cuts).

At the last step read, the ResNet-56 was at −40.4 against −3.2 for keep-the-survivors. This run also removes the 29 Sep note's caveat, that our group training stopped on validation patience 6, which is harsher than NEON's rule. With NEON's own rule it lost by more.

**Does this week's progress change the verdict?**
- **Clean validation: tested directly; it does not.** The memorized validation was the one confound able to hide a C-G win, because it charged every cut the memorization gap. Under clean validation the NEON-literal C-G lost by more, not less.
- **Crop+flip: not run with C-G, and it cannot close the gap.** It helps whichever recipe it is added to. But its gains for keep-the-survivors are 2–6 points, against a 12–54-point deficit. C-G's failure is not overfitting: a frozen network receives a random group whose channels no longer mean what the downstream layers expect.
- **The final fine-tune: it does not apply.** It acts after the walk, and under C-G the walk leaves the budget within the first few cuts, so there is no compressed architecture to finish. The fair network-level form of "fresh weights" is a separate, published experiment. It retrains the whole pruned architecture from scratch with a full budget (Liu et al., ICLR 2019, "scratch-B"). That control is queued at the architectures the heuristic found, on the diagnostic networks and on DepGraph's ResNet-56.

**Why a dense network tolerates replacement and a CNN does not (our hypothesis).**
- *Dense networks.* In NEON's dense networks the replaced layer owns its consumer and has hundreds of redundant units. Any good basis the new layer learns can be used downstream.
- *CNNs.* One channel dimension is shared by a whole group; in a ResNet, every block of a stage reads and writes the same residual stream. Every frozen downstream layer expects the surviving filters' specific features. A random group has to rediscover exactly those features, while keeping the survivors starts from them.
- *Not only skip connections.* VGG-16 has no residual connections and still loses 11.6 points.

**Recommendation: do not pursue C-G / C-G+ further as the per-step recovery.**
- Five constructions on four networks under both validation protocols, and replacement never helped. More GPU time has little expected value.
- Keep it in the thesis as a documented negative result of the NEON lineage: the table above plus the mechanism paragraph.
- Carry the "fresh weights" idea at the network level, through the scratch-B controls.
- If you want the table closed under the new protocol, there is one optional completeness cell: C-G+ with crop+flip on the three benchmark networks. It needs under one GPU-day, and the same kill rule stops it early if it loses as badly.

**Questions for you:**
1. Did NEON measure neuron removal with the survivors kept, on dense networks, before choosing replacement? If so, by how much did it lose? That decides whether the dense-versus-CNN contrast is a finding of the thesis or an assumption.
2. Is there a construction you have in mind that we have not tried? For example:
   - redraw, then the full whole-network fine-tune instead of training the group alone;
   - redraw plus distillation from the old layer's outputs;
   - replacement only on layers outside residual streams.
3. Is network-level retraining from scratch (Liu et al. 2019) the right place to carry the idea, or is it a different question?

---

## 4. The agent

### 4.1 The first training run under the corrected protocol (ops name: the Stage-4 train)

- **What it is.** The same agent design as our last training run (§136), with clean validation and crop+flip added and nothing else changed:
  - in-budget linear reward;
  - five actions: skip, or keep 90 % / 80 % of the group ranked by L1 or FPGM;
  - the CIFAR-10 + SVHN catalog;
  - the 12-epoch training fine-tune.

  It started on 30 Sep at 03:14. One train with both changes is the decisive single experiment: if it cannot leave the heuristic, a weaker recipe will not.
- **Where it is (30 Sep, 19:46).**
  - 22 episodes and 5 policy updates, with no errors.
  - A snapshot is frozen whenever the score on two probe networks improves. The only one so far is from episode 11, before any real learning. By rule it is not tested.
- **When.** The first snapshot frozen after policy update 20 (~2 Oct evening) is tested; if none freezes, a fallback test runs at episode 120 (~3 Oct). A test walk takes about 5 hours. Freeze tests are pre-authorized, at most one a day.
- **The pre-registered test.** The frozen snapshot walks the two diagnostic networks under the same protocol as the heuristic. It passes when all three hold:
  - no size point is more than 0.5 points worse at equal size;
  - at least one point is ≥ 1 point kinder, or reaches a deeper point inside the budget;
  - it is not a copy of the heuristic: it does not choose 90 % on ≥ 95 % of the layers.

  That is the thesis claim in its smallest form: a SPECTRA agent that beats its own heuristic under an honest protocol. The timeline and what follows are in §2.6.

### 4.2 Why earlier agents copied the heuristic, and what is fixed

| Cause | Remedy | Status |
|---|---|---|
| The cube-root on the in-budget reward made the mildest legal cut risk-optimal | A linear in-budget reward | Fixed. Its first agent went deeper than the heuristic but not kinder, on memorized validation. In the running train |
| The snapshot-selection score measured only depth and saturated at the heuristic's depth | Depth weighted by the remaining accuracy slack (the area score) | Fixed; in the running train |
| Memorized validation: every cut looked 6–26 points costlier | Clean validation | **Fixed this week**; in the running train |
| Un-augmented recovery: cuts were costlier than necessary | Crop+flip | **Fixed this week**; in the running train |
| The action is a per-layer keep rate, so "keep 90 %" means different things on a 16-channel stem and a 256-channel stage | Budget-plus-stop: remove 1, 2 or 4 % of the whole network through this group, or stop | Trained on memorized validation, never froze a snapshot; parked |
| The state is a sequence of layers, and channel coupling enters only as one attention scalar | One token per coupled group, with learned feeds / fed-by relations | Stopped (memorized validation); parked |
| Rate and ranking criterion share one decision | A two-decision head (keep-rate × criterion) | Froze into the heuristic on memorized validation; parked |

The running train tests the four fixed causes together. The three parked designs, and a shared actor-critic trunk that is designed but not run, reopen one at a time on top of the corrected recipe, and only after the agent leaves the heuristic. One finding carries over: the agent reads the network. Zeroing or shuffling its layer tokens changes 38 % (ResNet-20) and 53 % (ResNet-56) of a frozen agent's decisions on identical walks.

### 4.3 The multi-dataset train (ops name: N8)

This is NEON's offline multi-dataset training, on CNNs, with the 16-network catalog. The train and its pre-registered tests are written; it has not been launched.

- **The catalog.** 8 CIFAR-10 + 8 CIFAR-100 networks from a family × dataset grid (thin ResNets, ResNet-32, VGG-11/13, MobileNetV2 ×0.5/×1, DenseNet-40), at most two per cell. Every benchmark architecture is held out. The grid answers the August failure, where half of a 24-network pool was near-duplicate thin ResNets and the agent became a width specialist.
- **The risk.** The state includes activation statistics on the dataset's own images, so every held-out dataset is off-distribution for the agent.

| Design | Train on | Held out | For | Against |
|---|---|---|---|---|
| **A (recommended)** | CIFAR-10 + CIFAR-100 | SVHN, Fashion-MNIST, ImageNet | Three held-out datasets with three kinds of shift: domain (digits), modality (grayscale clothing) and scale (224 px, 1000 classes). CIFAR-100's accuracy regime (70–75 %) matches the ImageNet zoo's | Both training datasets are CIFAR, against NEON's ~22 per agent. SVHN and Fashion-MNIST have only 2 test networks each today |
| B | + SVHN | Fashion-MNIST, ImageNet | A second image domain in training | 2 SVHN networks against 16 CIFAR ones; loses the cleanest hold-out; SVHN needs its own augmentation rule |
| C | + SVHN + Fashion-MNIST | ImageNet | The most diverse pool | One held-out dataset, the most expensive to test |
| D (NEON's rotation) | Each dataset left out in turn | Each dataset once, plus ImageNet | NEON's own protocol | Four ~8-day trains; not affordable before one diverse train has worked |

**Three additions to A.**
1. Grow the SVHN and Fashion-MNIST test sets to about 6 networks each, with ShuffleNetV2 ×1, RepVGG-A0, MobileNetV2 ×0.5 and DenseNet-40. That is ~1–3 GPU-hours each, and it puts unseen families on unseen datasets.
2. Caption the comparison with the current run as a change of pool, not as "adding CIFAR-100".
3. Pre-register a follow-up that adds SVHN to training. It runs only if CIFAR passes and the dataset hold-outs fail, and it would measure NEON's lesson on CNNs: dataset diversity in training drives dataset transfer.

**What it must show (pre-registered):**

| Test | Pass line |
|---|---|
| It learns on a mixed pool | Healthy by update 10; a frozen snapshot by episode 120 |
| The new pool does not hurt CIFAR-10 | No point more than 0.5 worse than the current run's snapshot, same walk |
| Held-out CIFAR-100 networks, including two families never trained on | ≥ the heuristic at equal size on ≥ 4 of 6; not a copy |
| A dataset-conditioned policy (a possible new finding) | It cuts the same architecture consistently differently on CIFAR-10 and CIFAR-100 |
| ImageNet, frozen | ≥ the heuristic at equal size |
| Architecture transfer (the coverage matrix) | ≥ the heuristic on at least half the networks |
| SVHN and Fashion-MNIST | ≥ the heuristic on at least half the networks of each |

**Proposed launch condition** (fixed now, before the numbers are seen).
- *Launch without another meeting* when three things hold: the first agent test passes on both diagnostic networks and is not a copy of the heuristic; a 2-episode smoke run passes without code changes; and a GPU is free.
- *Come back to you* if the pass is marginal (under 1.5 points everywhere it passes) or anything else needs a decision.
- *The alternative:* start it in parallel after the health check. The risk is two runs copying the heuristic for a week, on two of our four GPUs.

---

## 5. Details

### 5.1 CIFAR-100 under clean validation (the admission gate)

- *The walk.* The 2-pass heuristic walk with the training fine-tune (Adam 1e-3, 12 epochs, patience 4), on the 8 CIFAR-100 candidates.
- *The admission rule.* The quoted point keeps ≤ 98 % of the parameters and loses ≤ 10 points of validation accuracy.
- *Before* (memorized validation, §109): **0 of 8** admitted. Most walks could not keep a single cut inside the budget.

*Now* (TEST Δ at the quoted point @ fraction of parameters kept, §148):

| Net (unpruned test accuracy) | Clean validation, no augmentation | Clean validation + crop+flip |
|---|---|---|
| thin ResNet-20 ×13 (0.700) | −9.2 @ 0.926 | −8.4 @ 0.662 |
| thin ResNet-56 ×9 (0.733) | −9.0 @ 0.941 | −9.5 @ 0.647 |
| ResNet-32 (0.706) | −9.5 @ 0.663 | **−5.3** @ 0.663 |
| VGG-11 (0.714) | −9.4 @ 0.659 | **−4.3** @ 0.659 |
| VGG-13 (0.751) | −9.3 @ 0.661 | **−4.8** @ 0.658 |
| MobileNetV2 ×0.5 (0.711) | −2.9 @ 0.692 | **−2.0** @ 0.692 |
| MobileNetV2 ×1 (0.747) | not finished (time limit) | **−0.4** @ 0.671 |
| DenseNet-40 (0.703) | not started (time limit) | −6.0 @ 0.696 |
| **Admitted** | **6 of 6 finished** | **8 of 8** |

CIFAR-100 was never unrecoverable; the measurement was wrong. The two thin residual networks admit with little margin (validation −9.7 and −8.8), so they are the first to watch in the multi-dataset train.

### 5.2 Crop+flip, all at identical architectures

| Setting | Without crop+flip | With crop+flip | § |
|---|---|---|---|
| CIFAR-100 gate, 12 size points on 6 networks | — | kinder at **12 of 12**, +0.9 to +6.2, mean **+3.3** | §148 |
| Diagnostic ResNet-56 ×4, training fine-tune, 79.5 % kept | −8.2 | **−5.9** (+2.3) | §150 |
| Diagnostic ResNet-56 ×4, training fine-tune, deepest point inside the budget | −10.6 @ 0.741 | **−5.1 @ 0.622** | §150 |
| Diagnostic ResNet-56 ×4, test fine-tune, 79.5 % kept | −7.6 | **−2.6** (+5.0) | §152 |
| ResNet-56 (C10), test fine-tune, quoted point | −2.84 @ 0.661 | **−0.06 @ 0.661** | §152 |
| VGG-16 (C10), test fine-tune, quoted point | −2.8 @ 0.657 | **−0.5 @ 0.657** | §152 |
| VGG-19 (C100), test fine-tune, quoted point | −6.7 @ 0.657 | **−2.5 @ 0.657** | §152 |
| VGG-19 (C100), test fine-tune, 68.8 % kept | −6.8 | **−1.9** (+4.9) | §152 |

The effect is largest where the un-augmented recovery overfits most: VGG-19 on CIFAR-100 gains +3.8 to +4.9 at equal size. The one network it hurts is the diagnostic ResNet-20 ×2 (§1.2). Crop+flip also passed the training-recipe rule (§150), so the agent trains with it.

### 5.3 The final fine-tune, cell by cell

TEST on the 5k half unless marked 10k. The honest gain is the pruned network's gain minus the unpruned network's gain (the origin change, in parentheses). The published counterparts are in §2.5.

| Cell | Walk recovery | Honest gain (origin change) | Final, at equal size |
|---|---|---|---|
| DepGraph's VGG-19, C100 (§149) | no augmentation | **+4.1 to +5.5** (+0.10) | −2.52 @ 68.4 % (10k −2.39); −3.28 @ 59.9 % (10k −3.04) |
| DepGraph's VGG-19, C100 (§155) | crop+flip | −0.5 to −0.9 (+0.50): the walk already recovered | −2.24 @ 68.4 % (10k **−1.62**); −2.94 @ 59.9 % (10k −2.97) |
| DepGraph's ResNet-56, C10 (§153) | no augmentation | **+1.2 to +1.8** (+0.42) | 10k **−1.52 at 46.3 % FLOPs**; **−2.11 at 38.0 % FLOPs** |
| Diagnostic ResNet-56 ×4, C10 (§154) | no augmentation | **+5.3 to +5.5** (+0.30) | −2.72 @ 79.5 %; −2.64 @ 75.6 % |
| Diagnostic ResNet-20 ×2, C10 (§154) | no augmentation | −3.8 to −4.5 (+3.46): crossed off | — |

In flight: DepGraph's ResNet-56 with the crop+flip walk. Over 130 cuts its walk tracks +2.6 points kinder on validation than the un-augmented walk; that is a validation read, not a test result. Its test rows land overnight.

---

## 6. What we do not claim

- No beat or match of DepGraph, or of any focused method on its home cell. We are ~2 points behind at equal FLOPs, with a heuristic walk and no training on the target.
- CIFAR-100 is not "solved"; it is admitted to training.
- No agent result under the corrected protocol yet.
- Test numbers are on a 5k half; 10k numbers appear only at points chosen by size.

## 7. Action items (ours)

| | Action | When | What it decides |
|---|---|---|---|
| 1 | DepGraph's ResNet-56 with the crop+flip walk + final fine-tune | Overnight; ledgered by morning | The literature ResNet-56 rows use the crop+flip walk if it is ≥ 1 point kinder at equal size; a competitive-enough row (M4) if within 1 point of DepGraph |
| 2 | Zoo ResNet-56 and VGG-16 with the crop+flip walk + final fine-tune; the VGG-16 walk to HRank's and OCSPruner's sizes | Running; 1–2 Oct | The CIFAR-10 literature rows at the published sizes |
| 3 | Agent health check at update 10; first frozen-agent test after update 20 | ~1 Oct 11:00; ~2 Oct night, result ~3 Oct | Whether the agent leaves the heuristic (§4.1) |
| 4 | Scratch-B controls (Liu et al. 2019): the walk's architectures retrained from scratch (200 epochs of SGD), beside the inherited weights | Queued | Whether inherited weights matter at these sizes; the network-level form of "fresh weights" |
| 5 | Distillation (Hinton et al. 2015) and AutoAugment (Cubuk et al., CVPR 2019) inside the final fine-tune, on DepGraph's ResNet-56 | Queued | A better final recipe (M5): adopted at ≥ +0.5 with a healthy unpruned control |
| 6 | A 3-pass walk that never cuts the residual streams | Queued, low priority | Whether the residual streams are what limits depth |
| 7 | The next code version and its smoke test; the new SVHN and Fashion-MNIST hold-out networks | ~2 Oct | Readiness for the multi-dataset train |
| 8 | Import the literature hold-out networks: ResNet-164 on CIFAR-10 and CIFAR-100 and DenseNet-40 on CIFAR-10 (Network Slimming's), PruningBench's ResNet-18/50 CIFAR-100 bases, AMC's Plain-20 | Not started; with item 7 | Published points on transfer networks too |
| 9 | **Proposed, not queued:** walk DepGraph's VGG-19 C100 to its 8.92× size (~11 passes) with the final fine-tune; run the other same-loop heuristics (L1-magnitude walk; greedy, random, look-ahead) on the three cells under the new protocol | Your view (§8) | A complete bar-3 row on the third cell; a complete bar-2 comparison set |
| 10 | Zero-GPU readouts: replay the finished walks through the linear, cubic and NEON-exact rewards; a census of how much each training network memorized its training set | With item 7 | Where each reward would stop; how widespread the memorization is |
| 11 | Write the layer-replacement negative result into the thesis (table + mechanism) | After your answer on §3 | Closes the question |
| 12 | Launch the multi-dataset train | On the condition in §4.3, if you agree | The transfer result (~8 days of one GPU) |

## 8. Questions for you

1. **Layer replacement.** The three questions at the end of §3.
2. **The benchmark plan** (the 29 Sep note's question 3, continued). Does it hold with this week's changes: validation from a test half, crop+flip in every method's recovery, the final fine-tune only beside published numbers, and the corrected VGG-16 sizes? Is there a cell or a method to add or drop?
3. **Validation from the test set.** Is a fixed half of the official test set acceptable as the validation set for the thesis and the paper?
   - *The alternative:* retrain each catalog network on 45k images, so that a never-seen 5k slice of the training split can serve as validation. That is ~1–3 GPU-hours per network, about 16 networks.
   - *The catch:* DepGraph's benchmark checkpoints cannot be retrained without changing the benchmark, so the benchmark rows would keep the test-half protocol either way.
4. **Which recovery for the literature rows?** The pre-registered protocol is the plain final fine-tune after an un-augmented walk. The crop+flip walk plus the final fine-tune is at least as good: 0.8 points better at 68 % kept on VGG-19 C100, and equal at 60 %. Should the literature rows always use the best recovery we have, stated as such?
5. **The same-loop heuristics.** Under the new protocol only the 90 % rule has run. For bar 2, do you want the full set from your 18 Aug list (L1-magnitude, greedy, random, look-ahead) on the three cells, a few GPU-days, or the 90 % rule and the L1 walk only?
6. **The agent's first bar.** Is "at or above the heuristic at equal size on the two diagnostic networks, and not a copy of it" the right first bar before the benchmark networks and hold-outs? Or would you rather see a benchmark network first?
7. **The multi-dataset train.** Do you agree with design A and its three additions (§4.3), rather than NEON's leave-one-dataset-out rotation? May it launch on the pre-registered condition without another meeting?
8. **The reward.** The best cuts now come within a few tenths of a point of the unpruned network. Is an A/B of NEON's cubic reward against the linear one worth running once the agent passes, or should it wait for the multi-dataset agent?

## 9. Discussion

1. **How strong the heuristic now is, and what the agent adds.** With clean validation and a proper recovery, a fixed 90 % rule is ~2 points behind DepGraph on its home cell. The agent's value has to show as a better size–accuracy trade-off than that rule, on networks it never trained on, at no per-target cost. Is that the right way to present the contribution?
2. **The memorized-validation finding as a methods point.** Any pruning or architecture-search pipeline that starts from public checkpoints and validates on a slice of the training split has the same flaw. Is it worth a thesis section, or a short standalone note?
3. **NEON's lineage in the thesis.** With layer replacement closed, SPECTRA inherits NEON's preference-aware reward and its offline multi-dataset training. How should the thesis frame the part that did not carry over to CNNs?
4. **What you want to see before the results chapter.** The decision points are in §2.6: the first agent test (~3 Oct) and, if it passes, the multi-dataset train and its hold-out tests.
5. **Parked ideas**, for your view:
   - *An attribution train* (clean validation alone, without crop+flip), only after a success, to separate the two fixes.
   - *Capacity-conditioned augmentation:* crop+flip off for networks that underfit (NetAug), only if such networks enter the catalog.
   - *Later recovery tweaks:* weight averaging (SWA or EMA), mixup or label smoothing, per-stage rates, batch 64.
   - *A training fine-tune closer to the test one.* At 12 epochs it is ~1 point harsher than the 40-epoch test fine-tune, so the agent trains in a harsher world than it is tested in.
   - *A cheap ImageNet diagnostic:* re-run one ImageNet test with the class count spoofed to 100. If the cuts move, the policy uses the class count as its dataset cue.
   - *Fine-tune schedule variants* (group-first training, cosine at 12 epochs, SGD with crop+flip): not queued, revisited only if the crop+flip train learns badly. LAMB, a 0.95 keep rate and rollback are crossed off.
