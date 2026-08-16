# SPECTRA — paper draft (NEON format)

Working title: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Ido Paretsky · advisor Dr. Gilad Katz · Ben-Gurion University  
Predecessor: Hirsch & Katz, *Information Sciences* 2022 (NEON).  
**Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md).** Do not paste a job summary mean here.

Status of this file: **DRAFT**. Experiment freeze 15 Sep 2026; paper 30 Sep.  
Section order matches NEON (Hirsch & Katz 2022): Introduction → Related Work → Approach → Evaluation → Results → Discussion → Conclusions.

Tags: **LOCKED** / **DRAFT** / **TBD** — see [README.md](README.md).

---

## Abstract — DRAFT

SPECTRA is a single offline deep-reinforcement-learning agent that performs *structured* channel pruning on convolutional networks. Like NEON, it is trained once on a catalog of architectures and datasets and then applied to unseen checkpoints without retraining the agent. Unlike NEON, the environment resizes Conv2d/BatchNorm groups (residuals, DenseNet concat, depthwise ties) so reported parameter and FLOP ratios are real shape changes, not masked zeros.

On CIFAR-10, the 10-net agent keeps similar-family and unlike-family held-out networks inside a 10 pp accuracy budget, except skinny-deep ResNets. Learned schedules tie greedy rate-picking on easy nets and beat size-matched greedy on the hard similar ResNet-56. Encoder capacity (BERT / wider / set) did not move the hard thin ResNet-56; catalog diversity did (3-net ~−24 pp → 10-net −15.9 / −17.2). CIFAR-100 is treated as a recoverability problem: VGG-11 BN admits a ~10% structured cut with a test-set gain under a long SGD recipe; residual families have not yet shown a comparable cut. **[C1–C7 LOCKED; C8–C9 TBD]**

---

## 1. Introduction — DRAFT

Convolutional networks remain the default backbone for many embedded vision settings where a 10 pp accuracy drop is the user-set budget, not a leaderboard delta. Unstructured magnitude pruning does not reduce inference cost on commodity hardware; structured channel pruning does, but residual and grouped convolutions make “keep 80% of filters” a different question from “keep 80% of parameters.”

NEON showed that one preference-aware DRL agent, trained offline on many *dense* networks, can prune unseen dense nets without per-network retraining. SPECTRA asks whether that protocol survives CNNs: variable depth, skip connections, and channel coupling.

**Contributions (intended):**

1. A structured CNN environment whose compression credit matches rebuilt layer shapes.
2. A small Transformer state encoder trained with A2C (frozen BERT is an ablation, not the default).
3. An offline protocol with an explicit train / similar / unlike split, and same-loop greedy/mild/random controls.
4. Evidence that family and width transfer work on CIFAR-10 except skinny-deep ResNets, and that CIFAR-100 is limited by fine-tuning recoverability rather than encoder size.

NEON’s 28 tabular datasets are not re-run here. ImageNet overnight fine-tune is out of the loop (**CLAIM**, freeze 15 Sep).

---

## 2. Related Work — TBD (prose)

Follow NEON §2, then add CNN structured pruning.

| NEON subsection | SPECTRA counterpart | Status |
|---|---|---|
| 2.1 Reinforcement learning | A2C / REINFORCE lineage; AMC as *per-network* RL contrast | TBD |
| 2.2 Neural network pruning | Structured vs unstructured; DepGraph / FPAC / GDP as non-DRL SOTA (not the arena) | TBD — literature canvas exists, not in git |
| 2.3 Neural architecture search | Keep short: SPECTRA is not NAS | TBD |

**CLAIM:** The thesis arena is generic *offline* DRL for structured CNNs with a user τ, vs same-loop rate-pickers. It is not ImageNet ResNet-50 vs 2024 pruners.

---

## 3. Approach — DRAFT (method is largely LOCKED; equations TBD)

NEON §3 order: Overview, State, Action, Reward, Architecture, Training, Complexity.

### 3.1 Overview — DRAFT

One actor-critic. Each step: pick a compression rate for the current layer group, structurally rebuild coupled layers, full-net fine-tune, observe accuracy and size, receive NEON-style reward. Episode walks the network (and optional extra passes). Training mixes many checkpoints; evaluation loads a frozen actor on held-out checkpoints.

### 3.2 State space — DRAFT

Default: `SpectraStateEncoder` (~2–3M), one token per layer, learned target marker, attention bias on channel-coupling ids, filter-L1 shape statistics, action-cost features, database-wide z-score. Frozen `bert-base-uncased` remains `SPECTRA_STATE_ENCODER=bert`. **CLAIM:** encoder A/B (jobs 20140552/54/55/56) did not separate on r56-w4 (TEST −23.5 to −24.5 pp); BERT was slightly kinder only on easy r20-w2 (−2.1 vs −5.2). Do not reopen BERT as the default. See ledger §16.

### 3.3 Action space — DRAFT

Discrete keep-rates `{1.0, 0.9, 0.8}` on C10 (C100 DRL used `{1.0, 0.98, 0.95}`). Stem / width-1 layers cannot take fake prunes. Eval identity-pads once parameters kept hit 0.70 so thin nets are not scored at 17% size. Fortify: a requested cut must drop at least one channel.

### 3.4 Reward function — DRAFT

NEON preference-aware reward; `allowed_acc_reduction` τ = 10 pp. Compression credit uses **realized** param/FLOP ratios after rebuild, not the menu rate. Masked (non-structural) edits get no size credit.

### 3.5 SPECTRA architecture — DRAFT

A2C agent + `NetworkEnv` + `torch.fx` channel groups (residual adds, DenseNet concat, ShuffleNet/depthwise). Classifier outputs are never shrunk.

### 3.6 Training process — DRAFT

Offline catalog (10 nets: C10 ResNet/VGG/MobileNet/DenseNet + SVHN/Fashion VGG and thin ResNet). Seeds 42/43/44. 24-net catalog running (**PRELIM**). C100 joins train only after recoverability (**manifest**).

### 3.7 Complexity — TBD

Wall-clock: DenseNet eval dominates; 7-day Slurm wall; training stops on timer, eval still runs; `afterok` successors.

---

## 4. Evaluation — DRAFT

NEON §4: algorithms, setup, results, discussion. SPECTRA splits setup here and numbers in §5.

### 4.1 Compared methods — DRAFT

| Method | What it chooses | Same FT loop? |
|---|---|---|
| SPECTRA (DRL) | Layer and rate via A2C | yes |
| Greedy (“L1”) | Always strongest legal cut (keep 0.8); ranking L1/L2/SVD is ablation | yes |
| Mild | Prefer keep 0.9 | yes |
| Random | Uniform legal rate | yes |

Heuristics pick **rate**, not a different ranking of filters, unless `SPECTRA_FILTER_IMPORTANCE` is set. Look-ahead greedy (`SPECTRA_EVAL_LOOKAHEAD=1`) is the remaining size-matched control on r56-w4 (**TBD**, job 20213131).

### 4.2 Setup — LOCKED protocol / PRELIM catalogs

- Datasets in DRL train: CIFAR-10, SVHN, Fashion-MNIST (32×32 RGB).
- Held-out C10: similar, unlike, thin (see ledger §3–5).
- Metrics: TEST Δacc (pp), param_ratio, flops_ratio.
- Skip akamaster ResNet-32.
- Log `eval_train` is the CNN **train-image** loader on the same checkpoint, not “this architecture was in the DRL catalog.” Quote `eval_test` only (ledger quoting rules).
- CIFAR-100: recoverability probes first; optional in-domain DRL after. Early C100 failure was FT/env, not “C100 missing from the 10-net train set.”

---

## 5. Evaluation results — fill from the ledger

### 5.1 Similar-family transfer (C10) — LOCKED

See ledger §3. Easy nets (VGG-19 BN, DenseNet-100, ResNet-44, MobileNet-v2×0.75, thin ResNet-20 w16) stay inside τ=10 on seeds 42/43/44. Thin ResNet-56 w10 is the similar-family miss and is seed-sensitive (−9.2 / −12.4 / −13.0).

**Figure TBD:** bar or table of Δacc vs params kept, three seeds.

### 5.2 Unlike-family transfer (C10) — LOCKED (n=2 seeds)

See ledger §4. ShuffleNet-v2 and RepVGG, never in train, all inside τ=10. RepVGG-A0 is −4.8 pp at 0.681/0.565 on both seeds 42 and 44. Seed 43 **TBD** (20189049).

### 5.3 Held-out thin ResNets and learned vs greedy — LOCKED / TBD look-ahead

See ledger §5–6 and catalog ladder §17. Easy r20-w2: DRL ≈ greedy at 60% params. Hard r56-w4: C10-thin-only and encoder A/Bs stay near −24 pp; 10-net DRL −15.9 / −17.2 at **identical** 0.704/0.550; greedy ~−24 at 0.667 (unmatched). Similar r56-w10: DRL −13.0 vs greedy −23.1 at ~0.60 params. AMP, skinny-in-train, and budget-in-state did not move r56-w4 (ledger §18).

### 5.4 CIFAR-100 recoverability — LOCKED VGG / PRELIM residuals

See ledger §7. **CLAIM:** C100 is not a second genericity table until a 5–10% structured cut recovers on more than VGG. Probe 20204214 is complete. VGG-11 BN recipe is two-run (TEST gains at 90–95% params). Residual keep-rate 0.8 still leaves ~95–96% params. DenseNet-40 keep 0.8 left 99.0% params (TEST −0.43). MobileNet-v2×1 keep 0.8–0.9: TEST gain at ~98–99% params but **val DROP** (~−12.5 pp). Do not describe early C100 failure as “the C10 agent was not trained on C100.”

### 5.5 24-net catalog — TBD

Jobs 20201235 / 20202693 / 20204215. Question: does r56-w4 move past −15.9/−17.2?

---

## 6. Discussion — DRAFT

**NEON’s pruning strategy, CNN version.** The agent learns *where* to cut. On easy C10 nets the strongest legal cut is already good, so DRL ties greedy. On skinny-deep ResNets the schedule matters, but the environment is still harsh: residual groups decide params/FLOPs, and a 0.8 rate is not a 20% size cut.

**Why C100 is a different chapter.** Recoverability probes have no agent. If fine-tune cannot undo a mid-layer prune, A2C cannot learn a useful C100 policy. VGG-11 BN under SGD+aug 160 ep *can*; thin C100 ResNets at keep-rate 0.8 still keep ~95% of parameters (ledger §7.2). Mixing those nets into a C10 agent is how train returns go to −100, not how dataset transfer is demonstrated.

**Limitations (write them):** no ImageNet FT; ShuffleNet grouping incomplete (mask fallback); r56-w4 still misses τ=10; C100 residual recoverability open; look-ahead greedy not yet in; 24-net eval not yet in. Do not quote overnight-matrix −1.2 pp or eval_train −1.7 pp as TEST (ledger §10).

---

## 7. Conclusions and future work — DRAFT

SPECTRA shows that NEON’s offline, preference-aware DRL protocol extends to structured CNNs on CIFAR-10 for similar and unlike families, with skinny-deep ResNets as the documented failure mode. Future work: 24-net eval, size-matched look-ahead greedy, optional τ=5 vs τ=10 on the frozen actor, C100 only if more families recover, ImageNet as frozen/short eval at most.

---

## Appendix A — Protocol archaeology (8–16 Aug) — DRAFT

Numbers in [RESULTS_LEDGER.md](RESULTS_LEDGER.md) §§10–20. Do not promote these to §5 tables except where already LOCKED (C7, floor, Fortify).

**A.1 Inherited bugs.** Until the overhaul, `torch.nn` prune left shapes unchanged, FT was `range(0)`, DDP nested every step, and episode return was the last reward. Later compression numbers are real rebuilds (ledger §11).

**A.2 Fine-tune mode.** Layer-only FT: 0/32 recoveries. Full-net 40 ep + rates 1.0/0.9/0.8 made C10 recoverable (ledger §12). Group-aware freeze did not.

**A.3 Eval floor.** Without 0.70, held-out r20-w2 is −13 to −15 pp at 40% params; unconstrained r56-w4 is −42 pp at 17% params (20140546). With the floor, 2-net structural TEST is −3.1 @ 0.60/0.79 (20066579). Overnight **−1.2 pp is not TEST**.

**A.4 Encoder ablation.** Ledger §16. All four encoders ~−24 pp on r56-w4. Catalog diversity is the lever (§17).

**A.5 Mixed C10+C100.** 45–48% train-within −10 was C100, not the DRL recipe (ledger §14). Split datasets.

**A.6 C100 Adam probes.** 20158277: 2/34 val-OK, both ≥98.5% params. 20168590 crop+flip: 1/26 at 0.995 params. Not a 2–5% menu. VGG SGD 160-ep is the first real cut (ledger §7.1).

**A.7 Code SHA / catalogs.** Night git `e985d5e`. Train: `database_offline_train.json` (10) then `database_offline_wide.json` (24, running). Pool: `offline_pools_manifest.json` (287 mapped, not all trained).
