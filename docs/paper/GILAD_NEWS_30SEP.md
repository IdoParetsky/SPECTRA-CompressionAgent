# SPECTRA — news for Gilad, 30 Sep 2026

Written for Ido to deliver, and for the ops chat to read once. Ledger sections are in `docs/paper/RESULTS_LEDGER.md`. All TEST numbers below are on a 5k half of the test set unless marked "10k".

**In one line each:**
1. **Clean validation.** The validation set that scored every cut was made of images the pretrained nets had memorized. With validation taken from held-out test images instead, CIFAR-100 goes from 0 of 8 recoverable nets to 8 of 8, and several negative verdicts reopen.
2. **Crop+flip in the fine-tune after each cut.** Recovering with the standard CIFAR augmentation, the one the nets were trained with, makes the same cuts 2–6 pp kinder. A full-width ResNet-56 on CIFAR-10 now loses 0.06 pp at 1.51× parameter compression.
3. **A 100-epoch SGD final fine-tune**, the literature's protocol, recovers 1.2–5.5 pp beyond what it gives the unpruned net. On DepGraph's own ResNet-56 we are ~2 pp behind their published points at equal FLOPs, with a heuristic walk.

## 1. Clean validation (protocol P) and the CIFAR-100 gate

**The flaw.**
- *How a cut is scored.* The environment scores a cut by the drop in validation accuracy. Since August, that validation set was a slice of the CIFAR **training** split.
- *Memorized images.* Every zoo checkpoint (chenyaofo, the thin ResNets, DepGraph's) was trained on those images and had memorized them.
- *The size of the gap.* Unpruned val accuracy read 1.000 / 1.000 / 0.999 on ResNet-56 C10, VGG-16 C10 and VGG-19 C100. Their real test accuracy is 0.943 / 0.936 / 0.739 (§141).
- *What the reward measured.* A cut then looked like a fall from memorized accuracy to real accuracy. The reward and every validation rule measured forgetting, not the loss of accuracy on new images.

**What it had decided, silently.**
- *Agent ≡ mild.* The mildest policy always scored best, so the agent converged to the mild heuristic (§136).
- *C100 unrecoverable.* The memorization gap is largest on CIFAR-100 (~26 pp on VGG-19). The 18–20 Sep gate admitted 0 of 8 C100 nets (§109), so C100 stayed out of the training pool.
- *Other dead ends.* The NEON-style layer replacement (C-G) and some reward shapes also read as dead on this val.

**The fix.**
- *The split.* The CIFAR test set (10k images the nets never saw) is split once into two fixed 5k halves.
- *Two roles.* One half is validation: every reward, band check and val-selected point. The other half is TEST: every number we quote. No image is used both to select and to report.
- *Training data and batch.* Fine-tuning uses the whole 50k training split, at batch 256.
- *10k numbers.* Both halves together are quoted only at points chosen by size, never by validation.

**The CIFAR-100 gate under P.**
- *The walk.* A no-agent 2-pass mild walk (the agent's heuristic baseline) with the training fine-tune (Adam 1e-3, 12 epochs, patience 4), on the 8 C100 candidates.
- *The admit rule.* The val-selected point keeps ≤ 98 % of the parameters and loses ≤ 10 pp of val accuracy.

*Before (legacy val, §109):* **0 of 8** admitted. Most walks could not keep a single cut in band: VGG-11 and MobileNetV2 ×1 stayed unpruned, and DenseNet-40 was −2.6 pp at 0.989 kept.

*Now* (TEST Δ at the val-selected point @ fraction of parameters kept):

| Net (unpruned TEST) | P, no aug | P + crop+flip |
|---|---|---|
| thin ResNet-20 ×13 (0.700) | −9.2 @ 0.926 | −8.4 @ 0.662 |
| thin ResNet-56 ×9 (0.733) | −9.0 @ 0.941 | −9.5 @ 0.647 |
| ResNet-32 (0.706) | −9.5 @ 0.663 | **−5.3** @ 0.663 |
| VGG-11 (0.714) | −9.4 @ 0.659 | **−4.3** @ 0.659 |
| VGG-13 (0.751) | −9.3 @ 0.661 | **−4.8** @ 0.658 |
| MobileNetV2 ×0.5 (0.711) | −2.9 @ 0.692 | **−2.0** @ 0.692 |
| MobileNetV2 ×1 (0.747) | not finished (wall) | **−0.4** @ 0.671 |
| DenseNet-40 (0.703) | not started (wall) | −6.0 @ 0.696 |
| **Admitted** | **6 / 6 finished** | **8 / 8** |

**What it means.** CIFAR-100 was never unrecoverable; the measurement was wrong. The 16-net catalog (8 C10 + 8 C100) is now emitted, so C100 can join the training pool. That is NEON's multi-dataset training, and it is what the ImageNet hold-out needs: an agent trained on C10 and C100, frozen, then tested on ImageNet with no ImageNet training.

## 2. Crop+flip in the walk fine-tune

**What changed.** After each pruning step the environment briefly fine-tunes the pruned net: Adam 1e-3, 12 epochs with patience 4 during training, 40 / 10 at TEST. Until now that fine-tune saw un-augmented images.
- *The augmentation.* A random 32×32 crop from the image padded by 4 pixels, plus a random horizontal flip.
- *Its origin.* It is the standard CIFAR augmentation (He et al. 2016), and every zoo net was trained with it.
- *Scope.* CIFAR only; SVHN digits are not flipped.

**Results, all at equal architectures.** Mild takes the same cut at the same step in both arms, so each pair compares identical networks. Only the recovery differs.

| Setting | Without crop+flip | With crop+flip | Ledger |
|---|---|---|---|
| C100 gate, 12 size points on 6 nets | — | kinder at **12 / 12**, +0.9 to +6.2 pp, mean **+3.3** | §148 |
| Thin ResNet-56 ×4 (C10), training fine-tune, at 0.795 kept | −8.2 | **−5.9** (+2.3) | §150 |
| ResNet-56 ×4, deepest point still in band | −10.6 @ 0.741 | **−5.1 @ 0.622** | §150 |
| ResNet-56 (C10), TEST fine-tune, val-selected | −2.84 @ 0.661 | **−0.06 @ 0.661** | §152 |
| VGG-16 (C10), TEST fine-tune, val-selected | −2.8 @ 0.657 | **−0.5 @ 0.657** | §152 |

- *The one net it hurts.* The 5k-parameter thin ResNet-20 ×2 (64.8 % accuracy) loses 1–3 pp. Augmentation hurts nets that underfit (NetAug, Cai et al. ICLR 2022). That net is a hold-out diagnostic, not a training net.
- *The third twin.* VGG-19 C100 has no TEST rows yet. On validation it is kinder at all 7 paired steps so far (+3.2 pp mean).
- *Cost.* +18 % time per fine-tune epoch.

**What it means.** Most of the measured cost of a cut came from poor recovery, not from the pruning. Crop+flip is in the Stage-4 train, and it is about to become the TEST-walk recipe for every method (decision (d): 2 of 3 twins meet the rule; the thin guard is pending).

## 3. The 100-epoch SGD final fine-tune

**What it is.** After the walk the architecture is fixed. It then gets a long recovery: SGD, learning rate 0.01, momentum 0.9, weight decay 5e-4, cosine schedule, crop+flip, batch 128, 100 epochs, from the inherited weights.
- *The origin control.* The same recipe is also applied to the unpruned net. That separates what the recipe gives any net from what it recovers from the pruning.
- *Honest gain* = (final − walk) − (origin change).
- *Why.* This is how DepGraph and PruningBench report. The SOTA-facing rows (bar 3) use it; the same-loop comparisons between the agent and the heuristics (bar 2) stay on the walk recipe.

**Results.**

| Cell | Honest gain | Final, at equal size | Published point |
|---|---|---|---|
| DepGraph's VGG-19, C100 (§149) | **+4.1 to +5.5 pp**; origin +0.10 | −2.52 @ 0.684 kept (10k −2.39); −3.28 @ 0.599 (10k −3.04) | DepGraph −3.11 at 8.92× (keep ≈ 0.11): far more compression; not comparable |
| DepGraph's ResNet-56, C10 (§153) | **+1.2 to +1.8 pp**; origin +0.42 | 10k **−1.52 at FLOPs 0.463**; **−2.11 at FLOPs 0.380** | DepGraph **+0.24 at 2.11×** and **+0.11 at 2.57×** on the same checkpoint |
| Thin ResNet-20 ×2, C10 (partial) | cross-off: the recipe lifts the unpruned net +3.5 pp, but not its pruned versions | — | — |

**What it means.**
- *On ResNet-56.* At DepGraph's two FLOPs points we are ~1.8 and ~2.2 pp behind. That is with a no-agent mild walk that used the old, un-augmented walk recipe. The crop+flip walk plus the same final fine-tune is running now (N3), together with knowledge distillation (N1) and AutoAugment (N2) in the final fine-tune.
- *The frame.* Competitive-enough while transferring, never "beats".

## 4. Is this a breakthrough?

- **Clean validation: yes, for the measurement.** It is the most consequential finding since August. One flaw explains the agent's collapse to mild, the C100 failure and several negative verdicts at once. It is a diagnostic breakthrough, not yet a result: nothing yet shows the agent beating mild under P.
- **Crop+flip: a large correction** that brings the recovery recipe up to the literature's standard. With P, it removes both reasons the agent had to prefer mild: cuts looked more expensive than they were (memorized val), and they were more expensive than necessary (un-augmented recovery).
- **The final fine-tune: protocol alignment,** not a breakthrough. It makes the SOTA rows comparable. It does not change what the agent learns.
- **The results breakthrough** would be a frozen Stage-4 agent at or above mild at equal keep, under P + crop+flip, on held-out nets (first read ~2–3 Oct). After that comes the diverse C10 + C100 agent, transferred frozen to ImageNet (`docs/N8_DIVERSE_TRAIN_ROADMAP.md`).

## 5. What we do not claim

- No beat or match of DepGraph: we are ~2 pp behind at equal FLOPs, with a heuristic walk.
- C100 is not "solved": it is admitted to training.
- P numbers are on 5k test halves. 10k numbers only at points chosen by size.

## 6. Running now

- **The Stage-4 train** (21737123) is the agent under P + crop+flip, since 30 Sep 03:14. A resume is chained, so it continues past its 6-day runtime fuse (~6 Oct) to its stopping rule: at least 250 episodes, and 150 episodes without a better probe.
- **No-agent cells:**
  - the crop+flip + final-fine-tune cells N3 (ResNet-56) and N4 (VGG-19);
  - knowledge distillation (N1) and AutoAugment (N2) in the final fine-tune;
  - train-from-scratch controls at the walk's architectures (Liu et al. ICLR 2019);
  - NEON's layer-replacement rule under clean val (C-G).
