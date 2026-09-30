# SPECTRA — weekly meeting with Gilad, 1 Oct 2026

**From:** Ido Paretsky. Prepared 30 Sep; numbers as of 19:45 IDT.
**Record:** every number is in `docs/paper/RESULTS_LEDGER.md`; the § in each row points there. All rows are preliminary.

**How to read the numbers**

- **Δ** is the change in **test** accuracy, in percentage points; negative means the pruned network is worse. "−2.5 @ 66 %" means −2.5 points with 66 % of the parameters kept (1.51× fewer).
- **Two halves of the test set.** Since this week the 10k CIFAR test set is split once, with a fixed seed, into a 5k *validation* half that makes every choice and a 5k *test* half that we report (§1.1). Numbers are on the test half unless marked **10k**. The 10k figure (both halves) is quoted only at points chosen by size, where nothing read the validation half.
- **The heuristic walk.** Every row this week is a no-agent walk: a fixed rule that keeps 90 % of each prunable layer group at every step (ops name: *mild*), two passes over the network unless stated. It makes the same cut at the same step in every run, so two recovery recipes are compared on **identical architectures**. After each cut the network is briefly fine-tuned (the *walk fine-tune*: Adam 1e-3, 40 epochs with patience 10 at test time; 12 epochs with patience 4 inside agent training).
- **The quoted point** is the most compressed point whose **validation** drop is still within the 10-point budget (τ = 10; ops name: *val_best*), or a point chosen by size (a *size point*: 80 % or 70 % of parameters kept, or a paper's FLOPs ratio). Test accuracy never chooses a point.
- **Benchmark networks** are our three benchmark cells: ResNet-56 on CIFAR-10, VGG-16 on CIFAR-10 and VGG-19 on CIFAR-100 (model-zoo checkpoints, or DepGraph's released ones where stated). **Diagnostic networks** are two deliberately narrow ResNets (ResNet-20 at width ×2, ResNet-56 at width ×4): cheap to walk, hard to prune.

## Agenda (about 60 minutes)

| | Item | Minutes |
|---|---|---|
| 0 | Executive summary: five wins, the results table, what we crossed off, last week's questions | 10 |
| 1 | The three advances, one paragraph each | 10 |
| 2 | NEON's layer replacement on CNNs: recommendation and questions | 10 |
| 3 | Details, as needed: CIFAR-100, crop+flip, the final fine-tune, the literature sizes | — |
| 4 | The agent: the first training run under the corrected protocol; the multi-dataset train | 10 |
| 5–8 | What we do not claim; action items; questions for you; discussion | 20 |

---

## 0. Executive summary

We found that the validation images that scored every pruning decision had been memorized by the pretrained networks. Scoring on held-out images instead, and giving the per-step recovery the standard CIFAR augmentation, removes most of the accuracy cost we had attributed to pruning. A full-width ResNet-56 on CIFAR-10 now loses 0.06 points at 1.51× parameter compression, and CIFAR-100 goes from 0 to 8 of 8 recoverable networks. The agent's first training run under the corrected protocol started on 30 Sep at 03:14, and its first test is due around 2–3 Oct. **Nothing below is an agent result yet.** Every row is the no-agent heuristic, which is now the bar the agent has to clear.

### Top five wins

1. **A validation leak, found and fixed** *(infrastructure and diagnosis; §141–§143).*
   - *The leak.* Every public checkpoint we prune was trained on all 50k CIFAR training images, and our validation set was a slice of those images.
   - *Its size.* Unpruned validation accuracy read 1.000 / 1.000 / 0.999 on the three benchmark networks. Their real test accuracy is 0.943 / 0.936 / 0.739.
   - *The fix.* Validation now comes from held-out test images (clean validation; ops name: *protocol P*). It tracks test within ~1.5 points.
   - *What it explains.* One flaw accounts for three failures: the agent's collapse into the heuristic, "unrecoverable" CIFAR-100, and no cut inside the budget on VGG-19 CIFAR-100.
2. **One recovery recipe now works on CIFAR-100, and the 16-network training catalog is built** *(result and infrastructure; §148).*
   - Last week 0 of 8 CIFAR-100 networks were admitted, under every optimizer and schedule we tried. Now all 8 are.
   - The catalog of 8 CIFAR-10 + 8 CIFAR-100 networks is emitted. A unit test checks that no training network is also a test network.
   - This answers your fourth question from last week (one recipe for every network: yes) and unblocks NEON's multi-dataset training, on CNNs.
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
| DepGraph's VGG-19 (C100), 10k | 68 % | −6.09 | **−1.62** | crop+flip + final fine-tune | §147, §155 · DepGraph −3.11 at 11 % kept |

### Crossed off this week (proven negatives)

1. **NEON's layer replacement as the CNN recovery.**
   - Last week you asked whether it needed one retry under NEON's own stopping rule. The retry ran under clean validation and lost by 12–54 points at the same steps (§156).
   - Together with last week's four variants, that is five constructions on four networks under both validation protocols, and replacement never helped.
   - Recommendation and questions are in §2.
2. **Last month's verdicts that were scored on the memorized validation are void, not confirmed.**
   - "CIFAR-100 cannot be recovered" and "VGG-19 CIFAR-100 has no cut inside the budget" are already overturned (wins 2 and 3).
   - "The agent is a copy of the heuristic" is being re-measured by the running train. The reward-shape comparisons will be re-run only if that train calls for it.
3. **The optimizer, schedule and epoch budget were never the lever.** AdamW with warm-up and cosine decay, RAdam, and a 40-epoch budget at two learning rates all failed their pre-registered rules (§129–§133). Clean validation plus the standard augmentation did what none of them did, with the original optimizer.
4. **The long final fine-tune is a protocol for literature rows, not a free gain.**
   - *Where recovery is weak* it adds +4 to +5.5 points.
   - *Once the walk already uses crop+flip* it adds nothing: −0.4 to 0.0 points, while the unpruned control gains +0.5 (VGG-19 C100, §155).
   - *On the smallest network* (the diagnostic ResNet-20, 5k parameters) it lifts the unpruned network by +3.5 and its pruned versions not at all (§154).
5. **Three agent designs parked, not disproven.**
   - The two-decision head (keep-rate × ranking criterion) and the budget-plus-stop action both froze into the heuristic's behaviour (§135, §137).
   - The group-as-token train was stopped.
   - All three were scored on the memorized validation. We re-run one agent under the corrected protocol before reopening any design question.

### Last week's open questions: where they stand

| Your question, 27 Sep | Answer this week |
|---|---|
| 1. Retry layer replacement under NEON's training-loss stopping rule before closing it? | Done, under clean validation: 12–54 points worse (§156). We propose to close it (§2). |
| 2. The linear in-budget reward vs NEON's cubic, when no cut had ever raised accuracy? | The "no increase" came from the recovery recipe, not the reward. With crop+flip the best cuts come within 0.1–0.3 points of the unpruned network, and 2 of 46 on VGG-19 C100 are slightly above it on validation (+0.28 at most; none on test) (§152, §155). The cubic's bonus branch is now at the edge of reach, but not yet worth a train. |
| 3. Does the benchmarking plan hold? | Yes, with one correction. The "OCSPruner, 42 % of parameters" figure in our 21 and 27 Sep notes was its ResNet-56 point. The VGG-16 sizes are in §3.4, and a walk to them is running. |
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

## 2. NEON's layer replacement on CNNs (recipes C-G and C-G+): worth more research?

**Main idea and source.** NEON (Hirsch & Katz 2022, §3 and Algorithm 1) does not remove neurons from the existing layer after a pruning action. It replaces the layer with a new one of the target width, initialized at random, freezes every other layer, and trains the new layer to convergence. The public source also rebuilds the next layer's input weights and a fresh batch-norm, and it stops on the training loss with patience 10.

SPECTRA's recovery does the opposite: it keeps the surviving filters and fine-tunes the whole network (recipe A). We carried NEON's construction to CNN layer groups in two forms:
- **C-G:** redraw the producing convolutions, their batch-norm and the consumer's input slice; freeze the rest; train the group.
- **C-G+:** C-G, then a short whole-network polish at one tenth of the learning rate.

**Why we tried it.** It is the recovery of the thesis's direct predecessor, and a frozen generic agent would inherit it naturally. Fresh weights might also avoid a bias that the surviving filters carry.

**What we measured.** Same heuristic walk, same budget, no agent; the comparison is keep-the-survivors (recipe A) at the same steps.

| Recovery at each cut | Validation | Against keeping the survivors | § |
|---|---|---|---|
| C-G: random new group, trained to a validation plateau | memorized | no cut inside the budget on 3 of 3 networks (quoted point ≥ 98.8 % kept) | §100, §104 |
| C-G+: C-G, then a whole-network polish at 0.1× learning rate | memorized | no cut inside the budget on the two ResNet-56s; ResNet-20 −10.3 at 88 % kept vs −3.4 at 54 % | §101, §106 |
| Redraw the pruned layer only | memorized | same as C-G | §108 |
| C-PCA: a new layer built from the principal directions of the old activations | memorized | 1.1–2.6 points worse, at equal or shallower size | §127 |
| **C-G, NEON-literal** (training-loss stop, patience 10, up to 100 epochs) | **clean** | **12 to 54 points worse on validation** at the same steps; better on 0–4 % of cuts; stopped early by its pre-registered kill rule | §156 |

The last row, per network. Mean validation gap at the same steps:
- ResNet-56 C10: −30.7 (60 cuts);
- VGG-16 C10: −11.6 (25 cuts);
- diagnostic ResNet-20: −27.8 (16 cuts);
- diagnostic ResNet-56: −54.0 (28 cuts).

At the last step read, the ResNet-56 was at −40.4 against −3.2 for keep-the-survivors.

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

## 3. Details

### 3.1 CIFAR-100 under clean validation (the admission gate)

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

**Reading.** CIFAR-100 was never unrecoverable; the measurement was wrong. The two thin residual networks admit with little margin (validation −9.7 and −8.8), so they are the first to watch in the multi-dataset train.

### 3.2 Crop+flip, all at identical architectures

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

- *Largest where recovery overfits most.* VGG-19 on CIFAR-100 gains +3.8 to +4.9 at equal size.
- *The one network it hurts.* The diagnostic ResNet-20 ×2 loses 1–3 points (§1.2).
- *In training.* Crop+flip passed the training-recipe rule (§150), so the agent trains with it.

### 3.3 The final fine-tune, cell by cell

TEST on the 5k half unless marked 10k. "Honest gain" = the pruned network's gain minus the unpruned network's gain (the origin control).

| Cell | Walk recovery | Honest gain (origin change) | Final, at equal size | Published |
|---|---|---|---|---|
| DepGraph's VGG-19, C100 (§149) | no augmentation | **+4.1 to +5.5** (+0.10) | −2.52 @ 68.4 % (10k −2.39); −3.28 @ 59.9 % (10k −3.04) | DepGraph −3.11 at 11 % kept: far more compression; not comparable |
| DepGraph's VGG-19, C100 (§155) | crop+flip | −0.5 to −0.9 (+0.50): the walk already recovered | −2.24 @ 68.4 % (10k **−1.62**); −2.94 @ 59.9 % (10k −2.97) | as above |
| DepGraph's ResNet-56, C10 (§153) | no augmentation | **+1.2 to +1.8** (+0.42) | 10k **−1.52 at 46.3 % FLOPs**; **−2.11 at 38.0 % FLOPs** | DepGraph **+0.24 at 2.11×**, **+0.11 at 2.57×**, same checkpoint |
| Diagnostic ResNet-56 ×4, C10 (§154) | no augmentation | **+5.3 to +5.5** (+0.30) | −2.72 @ 79.5 %; −2.64 @ 75.6 % | — |
| Diagnostic ResNet-20 ×2, C10 (§154) | no augmentation | −3.8 to −4.5 (+3.46): crossed off | — | — |

Pending: DepGraph's ResNet-56 with the crop+flip walk lands overnight (§3.4). The model-zoo ResNet-56 and VGG-16 with the crop+flip walk are running.

### 3.4 The literature sizes, and one correction

- **DepGraph's ResNet-56, C10** (their checkpoint, 93.53 %).
  - *Published:* +0.24 at 2.11× FLOPs and +0.11 at 2.57×, after sparsity training on the target.
  - *Ours:* −1.52 / −2.11 (10k): the heuristic walk plus the final fine-tune (§153).
  - *In flight:* a crop+flip walk with the same final fine-tune lands overnight. Over 130 cuts its walk tracks +2.6 points kinder on validation than the un-augmented walk. That is an in-flight validation read, not a test result.
- **DepGraph's VGG-19, C100** (73.50 %). *Published:* −3.11 at 8.92× (11 % of parameters kept). Our 3-pass walk does not reach that size; our rows are at 53–68 % kept.
- **VGG-16, C10: a correction to the 21 and 27 Sep notes.**
  - "OCSPruner, 42 % of parameters" was OCSPruner's ResNet-56 point.
  - The published VGG-16 sizes are OCSPruner 21.2 % FLOPs / 13.7 % parameters, and HRank 46.5 % FLOPs / 17.1 % parameters.
  - A 10-pass heuristic walk to both FLOPs sizes, with the final fine-tune, is running.
- **The frame.** Competitive enough while transferring. Every published row trains, searches or regularizes on the target; a frozen agent never does.

---

## 4. The agent

### 4.1 The first training run under the corrected protocol (ops name: the Stage-4 train)

- **What it is.** The same agent design as our last training run (§136), with clean validation and crop+flip added and nothing else changed:
  - in-budget linear reward;
  - five actions: skip, or keep 90 % / 80 % of the group ranked by L1 or FPGM;
  - a CIFAR-10 + SVHN catalog;
  - the 12-epoch training fine-tune.

  It started on 30 Sep at 03:14. One train with both changes is the decisive single experiment: if it cannot leave the heuristic, a weaker recipe will not.
- **Where it is (30 Sep, 19:46).**
  - 22 episodes and 5 policy updates, with no errors.
  - A snapshot is frozen whenever the score on two probe networks improves. The only one so far is from episode 11, before any real learning. By rule it is not tested.
- **Timeline.**
  - A health check at update 10, around 1 Oct noon: the critic's fit must be positive and the policy away from uniform.
  - Update 20 around 2 Oct evening. The first snapshot frozen after it is tested automatically.
  - A fallback test at episode 120, around 3 Oct.
  - The run continues past the cluster's 6-day limit (around 6 Oct) to its stopping rule: at least 250 episodes, and 150 episodes without a better probe score.
- **The pre-registered test.** The frozen snapshot walks the two diagnostic networks under the same protocol as the heuristic. It passes when all three hold:
  - no size point is more than 0.5 points worse at equal size;
  - at least one point is ≥ 1 point kinder, or reaches a deeper point inside the budget;
  - it is not a copy of the heuristic: it does not choose 90 % on ≥ 95 % of the layers.

  That is the thesis claim in its smallest form: a SPECTRA agent that beats its own heuristic under an honest protocol. The benchmark networks and the hold-out sets follow.

### 4.2 The multi-dataset train (ops name: N8) and its launch condition

- **The catalog (design A).**
  - *Training:* 8 CIFAR-10 + 8 CIFAR-100 networks from a family × dataset grid: thin ResNets, ResNet-32, VGG-11/13, MobileNetV2 ×0.5/×1 and DenseNet-40, at most two per cell.
  - *Held out:* SVHN, Fashion-MNIST, ImageNet, and every benchmark architecture.
  - *Why a grid:* it answers the August failure, where half of a 24-network pool was near-duplicate thin ResNets and the agent became a width specialist.
- **Why design A.** It holds out three datasets with three kinds of shift:
  - domain: house-number digits (SVHN);
  - modality: 28-pixel grayscale clothing (Fashion-MNIST);
  - scale: 224 pixels and 1000 classes (ImageNet).

  CIFAR-100's accuracy regime (70–75 %) also matches the ImageNet model zoo's.
- **Its risk.** Both training datasets are CIFAR, against NEON's ~22 datasets per agent. The state includes activation statistics on the dataset's own images, so every held-out dataset is off-distribution.
- **Three additions to reduce that risk.**
  1. Grow the SVHN and Fashion-MNIST hold-outs from 2 networks to about 6 each, with ShuffleNetV2, RepVGG-A0, MobileNetV2 ×0.5 and DenseNet-40, at about 1–3 GPU-hours each.
  2. Caption the comparison with the current run as a change of pool, not as "adding CIFAR-100".
  3. Pre-register a follow-up that adds SVHN to training. It runs only if the CIFAR results pass and the dataset hold-outs fail. It would measure NEON's lesson on CNNs: dataset diversity in training drives dataset transfer.
- **Proposed launch condition** (fixed now, before the numbers are seen).
  - *Launch without another meeting* when three things hold: the first agent test passes on both diagnostic networks and is not a copy of the heuristic; a 2-episode smoke run passes without code changes; and a GPU is free.
  - *Come back to you* if the pass is marginal (under 1.5 points everywhere it passes) or anything else needs a decision.

---

## 5. What we do not claim

- No beat or match of DepGraph, or of any focused method on its home cell. We are ~2 points behind at equal FLOPs, with a heuristic walk and no training on the target.
- CIFAR-100 is not "solved"; it is admitted to training.
- No agent result under the corrected protocol yet.
- Test numbers are on a 5k half; 10k numbers appear only at points chosen by size.

## 6. Action items (ours)

| | Action | When | What it decides |
|---|---|---|---|
| 1 | Read DepGraph's ResNet-56 with the crop+flip walk + final fine-tune | lands overnight; ledgered by morning | Whether the literature ResNet-56 rows use the crop+flip walk (adopted if ≥ 1 point kinder at equal size); a first "within 1 point of DepGraph" row if it lands there |
| 2 | Read the model-zoo ResNet-56 and VGG-16 with the crop+flip walk + final fine-tune, and the VGG-16 walk to HRank's and OCSPruner's FLOPs sizes | running; 1–2 Oct | The CIFAR-10 literature rows at the published sizes |
| 3 | Agent health check at update 10; first frozen-agent test after update 20 | ~1 Oct noon; ~2 Oct evening, result ~3 Oct | Whether the agent leaves the heuristic (§4.1) |
| 4 | Scratch-B controls (Liu et al. 2019): the walk's architectures retrained from scratch (200 epochs of SGD), beside the inherited weights | queued | Whether inherited weights matter at these sizes; the network-level form of NEON's "fresh weights" |
| 5 | Knowledge distillation (Hinton et al. 2015) and AutoAugment (Cubuk et al., CVPR 2019) inside the final fine-tune, on DepGraph's ResNet-56 | queued | A better final recipe for every literature row: adopted at ≥ +0.5 over the plain fine-tune with a healthy unpruned control |
| 6 | Build the next code version; train the new SVHN and Fashion-MNIST hold-out networks | when the queue drains, ~2 Oct | Readiness for the multi-dataset train |
| 7 | Write the NEON-recovery negative result into the thesis (table + mechanism) | after your answer on §2 | Closes the question |
| 8 | Launch the multi-dataset train | on the condition in §4.2, if you agree | The transfer result (~8 days of one GPU) |

## 7. Questions for you

1. **NEON's recovery.** The three questions at the end of §2.
2. **Validation from the test set.** Is a fixed half of the official test set acceptable as the validation set for the thesis and the paper?
   - *The alternative:* retrain each catalog network on 45k images, so that a never-seen 5k slice of the training split can serve as validation. That is ~1–3 GPU-hours per network, about 16 networks.
   - *The catch:* DepGraph's benchmark checkpoints cannot be retrained without changing the benchmark, so the benchmark rows would keep the test-half protocol either way.
3. **Which recovery for the literature rows?** The pre-registered protocol is the plain final fine-tune after an un-augmented walk. The crop+flip walk plus the final fine-tune is at least as good: 0.8 points better at 68 % kept on VGG-19 C100, and equal at 60 %. Should the literature rows always use the best recovery we have, stated as such?
4. **The agent's first bar.** Is "at or above the heuristic at equal size on the two diagnostic networks, and not a copy of it" the right first bar before the benchmark networks and hold-outs? Or would you rather see a benchmark network first?
5. **The multi-dataset train.** Do you agree with design A and its three additions (§4.2)? May it launch on the pre-registered condition without another meeting?
6. **The reward.** The best cuts now come within a few tenths of a point of the unpruned network. Is an A/B of NEON's cubic reward against the linear one worth running once the agent passes, or should it wait for the multi-dataset agent?

## 8. Discussion

1. **How strong the heuristic now is, and what the agent adds.** With clean validation and a proper recovery, a fixed 90 % rule is ~2 points behind DepGraph on its home cell. The agent's value has to show as a better size–accuracy trade-off than that rule, on networks it never trained on, at no per-target cost. Is that the right way to present the contribution?
2. **The memorized-validation finding as a methods point.** Any pruning or architecture-search pipeline that starts from public checkpoints and validates on a slice of the training split has the same flaw. Is it worth a thesis section, or a short standalone note?
3. **NEON's lineage in the thesis.** With layer replacement closed, SPECTRA inherits NEON's preference-aware reward and its offline multi-dataset training. How should the thesis frame the part that did not carry over to CNNs?
4. **What you want to see before the results chapter.** The next decision points are the first agent test (~3 Oct) and, if it passes, the multi-dataset train (~8 days), followed by transfer tests on SVHN, Fashion-MNIST and ImageNet.
