# Gilad's two points of 1 Oct: what was done, what we learned, what comes next

**Report for Ido, 4 Oct 2026** (Opus 5.5 science sitting, opened 11:41 on ops' hand-over). It answers three of Ido's asks:
- what was done on Gilad's first point (side metrics) and whether a metrics dev phase is worth it;
- what was done on his second point (how SOTA chooses filters, a second DRL agent, robustness vs verification, NAP2);
- a detailed account of the second-agent effort: expectations, results, way ahead.

Part III adds today's diagnosis of why no trained agent beat mild (M1-neg).

**How to read the numbers.**
- *TEST* rows are the ledger's (`docs/paper/RESULTS_LEDGER.md`), on protocol P's held-out 5k half unless marked 10k.
- *Probe* numbers (S0, S1, S2, the replays, A0) measure a lever. They are never a method's TEST row.
- Cost numbers come from run manifests and logs, with the GPU named.

**Documents of record** (this report summarizes them; they hold the detail):
- *Point 1:* `docs/paper/EFFICIENCY_AND_TRANSFER.md`.
- *Point 2:* `docs/paper/FILTER_SELECTION_NAP_DESIGN.md`.
- *Board:* `docs/paper/GILAD_OCT8_TRACKER.md`.
- *Cells:* `docs/SITTING_GPU_QUEUE.md`.

---

## 0. Summary

### Point 1: side metrics

- **Adopted and built (1–4 Oct).**
  - Per-step cost accounting for every run.
  - A 1 s GPU power sampler in every new job.
  - An explicit timer for the agent's decision.
  - A deployment bench: latency at batch 1, 64 and 256, throughput, peak memory, energy per image.
  - DepGraph re-run on our own RTX 4090.
  - Literature cost tables and a break-even model over K targets.
  - A GPU-side augmentation that makes the CIFAR walk 1.4–2.9× faster per epoch with the same accuracy.
- **What we learned.**
  - Fine-tuning is 98–99.7 % of a CIFAR walk.
  - The agent's decision costs 3–8 ms (about 0.01 % of a step).
  - The CIFAR walk was bound by the CPU augmentation loader, not the GPU.
  - At one target we are about 5× slower than DepGraph on its own ResNet-56 (405.6 + 15.8 min vs 85.1 min, same GPU model).
  - From about the third target on, we are cheaper: each extra target costs one final fine-tune.
  - Pruning CIFAR ResNet-56 barely changes GPU latency for anyone, DepGraph included. VGG throughput gains are real.
- **Where beating SOTA looks promising:**
  - per-target search cost (zero);
  - the marginal cost of the K-th target;
  - no per-target tuning;
  - possibly wall-clock at one target, if the fast walk holds accuracy. That is being measured now (FW).

  The catch: every one of these is shared with the no-agent mild walk. They are claims about SPECTRA's pipeline. They become claims about the agent only when the agent at least matches mild on accuracy, and today it does not (Part III).
- **Metrics we would miss without a dev phase:**
  - the fast walk's K = 1 time-to-target;
  - a time-to-each-size-point readout;
  - the one-time train's energy (no running train has the power sampler);
  - deployment rows for agent-chosen networks;
  - precise energy (NVML) and CO2e (BGU's PUE).
- **My view.** I agree: moderately high priority, below fixing the agent's reward (Part III). I took your GO for a narrow, cheap dev phase:
  - the FW cell is running (22127216);
  - this report and the efficiency file are updated;
  - a time-and-Wh-to-each-size-point readout is built (`cost_readout.py`).

  I did not install NVML, run a val-selected DepGraph, or bench agent networks; §I.6 says why.

### Point 2: filter selection, a second agent, NAPv2, robustness

- **The "how SOTA chooses filters" column** is done for 49 methods. Most methods learn or hand-set *how many* filters per layer, and pick *which* by magnitude, often after a sparsity-training phase.
- **The second agent** was designed (a hierarchical selection policy over channel tokens, with a safety shield). It was gated by three experiments and closed by the data on 3 Oct:
  - *S0:* which filters survive matters with little or no fine-tuning, much less after 40 epochs.
  - *S1:* a learned score built on NAPv2's statistics ranks filters like an oracle on held-out nets (τ 0.57–0.66 vs ≤ 0.42 for hand criteria).
  - *S2:* that better ranking does not buy recovered accuracy after 40 epochs on two unseen nets (+0.21 / −0.87 pp). The registered HARM call fired.

  Keep L1. The agent stays an allocation agent ("how many per group").
- **NAPv2.**
  - *Absorbed:* the pipeline (snapshot collector, the 12 statistics, log-normalization, map layout, AE / BiGRU trainers). We rebuilt it per filter as NAP-F, checked against NAPv2's own code to 1e-9.
  - *Not absorbed:* the pretrained predictor. It is a network-level NAS-cell predictor with no per-filter signal as shipped.
  - *What worked:* NAP-F's gradient statistics carry a ranking signal that transfers within the ResNet family. It fails across families (MobileNet-V2).
- **Robustness vs verification** is mapped: one table, four cheap hooks, and one question for Gilad (did he mean Guy Katz's verification line?).
- **Leap forward?** No accuracy leap. Two real results:
  - a clean matched negative: "allocation, not selection, is what a pruning agent must learn at our fine-tune budget";
  - the first evidence that NAP-style gradient descriptors carry transferable information about CNN filters.

### New today (zero GPU): M1-neg is a reward result, not an RL failure

- **Every TESTed actor plays one action at every decision.** Stage-4 ep0095 (two seeds), Stage-4 ep0131 and C2 ep0083 pick "keep 0.8, L1" at all 16 / 60 cuts of R20-w2 / R56-w4. The Budget arm always asks for its largest budget (4 % of the origin per cut) and never STOPs. M1-neg therefore compares two fixed schedules: uniform 0.8 vs uniform 0.9 (mild).
- **Why.** Every train paid NEON's three-way reward with τ = 10 pp against the origin. Inside the band, a cut earns its size, whatever it costs in accuracy, and the thin walks never leave the band.
  - Over a walk's fixed number of decisions, "always the largest cut" maximizes the return.
  - Replaying the walks through the live reward: R56-w4's 0.8 walk earns 270.5 against mild's 126.6. At equal depth (keep 0.63) they earn 124.9 vs 126.0, although their val accuracies differ by up to 3 pp.
  - The agent learned exactly what it was paid for.
- **Consequences.**
  - FR43's "the actor's widths survive a new seed" is trivially true of a constant policy; do not present it as robustness evidence.
  - A new action menu alone cannot help: the agent will pick its largest entry.
  - N10 (the cubic reward) is already covered by C1 / C2, which collapsed the same way.
- **Way ahead.**
  - *A0, submitted:* does any allocation beat uniform at equal size under our fine-tune? Without that headroom, a better reward can at best relearn mild.
  - *Then a reward that prices accuracy at equal size.* The existing slack-tapered mode with a 5 pp training band (`structural_unified`, `SPECTRA_TRAIN_TAU=5`) is the first shape in the replay that pays mild's gentler walks more than the 0.8 schedule on both nets. It needs your GO for a train.

### Jobs started in this sitting

| Job | What | State (4 Oct ~12:15) | Read |
|---|---|---|---|
| **22127216** `v9d-fw-dg-r56` | FW: the fast walk (12/4 + GPU crop+flip) to DepGraph's sizes on its ResNet-56, RTX 4090 | R since 11:56, `ise-4090-03`, 30 steps in 14 min | K = 1 cost vs DepGraph 85.1 min; calls K1-PARITY / K1-TRADE / SLOWER (queue file "FW") |
| 22127526 `alloc-smoke` | A0 plumbing, thin R56-w4 | **COMPLETED** in 2.6 min | plumbing only; it exposed a size-matching flaw, fixed before the cells started |
| **22127527 / 28 / 29** `alloc-thin-r56w4` / `alloc-dg-r56` / `alloc-cy-vgg16` | A0: allocation headroom at keep 0.6 and 0.35, 40-epoch recovery | PD (QOS) | calls HEADROOM / FLAT / HARM per net (queue file "A0") |

---

## Part I. Side metrics (Gilad's point 1)

### I.1 What Gilad asked

SOTA accuracy at their compression is out of reach on their home networks, so generalization and transfer are paramount. Highlight the side metrics we can beat: memory, runtime, wall-clock and the GPUs used. Write everything down for the paper. Analyze SPECTRA's TEST-time advantage from offline training. Measure as we go. Scan the SOTA for more such metrics.

### I.2 What we adopted and built

| Metric | Tool (in `tree_v9d`) | Since | First numbers |
|---|---|---|---|
| Per-step cost split: fine-tune, pruning surgery, validation pass, state features, gap between steps | `scripts/cost_readout.py` (login node, zero GPU) | 1 Oct | 8 jobs, 13 walks (EFFICIENCY §3.1) |
| GPU energy (Wh), power, utilization, memory, clocks | 1 s `nvidia-smi` sampler in `scripts/spectra.sbatch` → `gpu_samples.csv` | jobs submitted from 1 Oct ~11:00 | D5 pair 225.8 vs 181.5 Wh |
| The agent's decision time | `SPECTRA_TIME_DECIDE=1` → a `step.decide` stage, read by `cost_readout.py` | 2 Oct; first runs 4 Oct | 3.0–8.2 ms per decision (EFFICIENCY §3.4) |
| Deployment: latency at batch 1 / 64 / 256, throughput, peak memory, energy per image | `scripts/bench_deploy.py` + `.sbatch` (FP32, TF32 off, 3 repeats in fresh processes) | job 21942378, 2 Oct | EFFICIENCY §5.3 |
| A comparator measured on our own GPU | DepGraph (Torch-Pruning v1.6.1) re-run, `scripts/h2h_depgraph.sbatch` + `h2h_readout.py` | job 21943448, 2 Oct | ResNet-56 85.1 min, VGG-19 44.7 min, their pruned nets benched |
| Literature costs | tables: per-target search cost of learned pruners, ImageNet search costs, schedules, cost of one pruning step per criterion, deployment protocols, transfer prior work | 1 Oct, every number checked against its source | EFFICIENCY §4, §5.1, §8 |
| Break-even over K targets | K* = W / (C − F) | 1 Oct | EFFICIENCY §7 |
| Faster walk, same science | `SPECTRA_FT_AUG_GPU=1` (crop+flip on the GPU) | D5 2–3 Oct; adopted for new cells 4 Oct (§196) | 1.47 vs 4.29 s/epoch (R20-w2), 3.73 vs 5.24 (R56-w4) |

### I.3 What we learned

1. **Fine-tuning is the cost.** It is 97.9–99.7 % of every CIFAR walk measured and 84–85 % of the ImageNet MobileNet-V2 walk. Everything else (surgery, validation, state features, bookkeeping and the agent) is 0.4–2.0 s per step.
2. **The agent costs nothing that matters.**
   - The first explicit timer (4 Oct, RTX 4090) puts a decision at 7.4 / 3.4 ms (Budget actor) and 8.2 / 3.0 ms (C2 actor) on R20-w2 / R56-w4. That is about 0.01 % of an 80 s step.
   - The actor has 2.73 M parameters (10.9 MB), and peak GPU memory equals a heuristic walk's.
3. **The CIFAR walk was bound by the data loader.** Every network ran at 4.2–5.6 s per fine-tune epoch, whatever its size or GPU. So cross-GPU wall-clock comparisons mostly measured the CPU augmentation pipeline. GPU-side crop+flip removed that: 2.9× per epoch on R20-w2 and 1.4× on R56-w4, with the same TEST accuracy (D5-bis, §196).
4. **One target is slower than one-shot SOTA at the TEST recipe.** On DepGraph's own ResNet-56 and our RTX 4090 (N3, 21767189):
   - the 40/10 walk reaches DepGraph's 2.11× FLOPs after **405.6 min**, plus **15.8 min** for that point's final fine-tune;
   - DepGraph's whole pipeline takes **85.1 min** on the same GPU model (21943448);
   - OCSPruner reports 26 min on a 4090, training from scratch.
5. **Several targets amortize.** One walk passes every size point, so each extra target costs one final fine-tune: 15.7–16.6 min for ResNet-56, 8.6–8.8 min for VGG. Against DepGraph, break-even is about 3 targets for a walk to 0.70 kept and 8 for a walk to 0.36.
6. **FLOPs do not buy latency on CIFAR ResNets, for anyone.**
   - ResNet-56 on a 4090 is launch-bound at batch 1: our 0.36-kept net runs 4.86 ms against the origin's 5.00.
   - DepGraph's own 2.11× net is ×1.02 at batch 1 and ×0.96 (slower) at batch 256.
   - VGG gains are real: VGG-16 at batch 256 goes from 60k img/s (origin) to 96k (HRank's FLOPs point) and 162k (0.12 kept). DepGraph's ~9× VGG-19 is ×2.85.
   - Energy per image falls with size: DepGraph ResNet-56 5.83 → 4.65 mJ at 0.36 kept; VGG-16 6.81 → 2.18 mJ.
7. **ImageNet is not a cost win.** Our MobileNet-V2 walk costs 105–122 GPU-h on a 4090. That is inside the band of per-target learned searches (25–864 GPU-h).
8. **The one-time train is not free.** The Stage-4 chain has run about 105 h on one RTX 6000 Ada (since 30 Sep 03:14), and the four arm trains about 80 h each since 1 Oct. None of these trains has power samples (§I.5 item 3).
9. **NEON had the same shape.** Its frozen agent (NEON 5) beat AMC 4 on time but was slower than AMC 1 at one target (Table 7). Its time, like ours, went to retraining the pruned layers.

### I.4 Metrics where SPECTRA can plausibly beat SOTA

| Metric | Where we stand | Against | Condition |
|---|---|---|---|
| **Per-target search and agent-training cost** | **Zero**, by construction | AMC ≤ 1 h, AGMC 320 s, GNN-RL 0.5 GPU-h, TAS 3.83 GPU-h on CIFAR; 25–864 GPU-h on ImageNet; RL-Pruner "several hours"; AgenticPruner 7.5 epochs plus LLM calls | Certain. Shared with every heuristic, so it argues for SPECTRA over learned pruners only if the agent at least matches mild |
| **Marginal cost of the K-th target** | One final fine-tune: ~16 min ResNet-56, ~9 min VGG (4090) | DepGraph 85 min per target (our 4090); graph metanetworks 43–67 min; ResRep / CHIP / OTO 180–480 epochs | Measured. A win from K ≈ 3 (shallow walk) or 8 (deep walk). Same caveat |
| **No per-target hyperparameters** (layer ratios, regularizer strengths) | None to tune | CHIP takes per-layer counts as input; GReg: "We do not have strong rules to set them"; DepGraph needs a global ratio and sparse-learning settings | Qualitative, but reviewers recognize it |
| **Wall-clock at one target** | 405.6 + 15.8 min at 40/10 (a loss) | DepGraph 85.1 min | **Being measured (FW, 22127216):** the train recipe 12/4 plus GPU augmentation. The one metric where a dev phase could turn a stated loss into parity |
| **Energy per target (Wh)** | Sampler in place; FW gives the first walk Wh to a DepGraph size | DepGraph's re-run was sampled too | Comes with FW's read |
| **Decision overhead** | 3–8 ms per decision | RL searchers train per target; LLM agents pay inference per target | A footnote, not a headline |

Not promising: deployment latency on CIFAR ResNets (flat for every method), ImageNet walk cost, and K = 1 at the 40/10 recipe.

### I.5 Metrics we will miss without a dev phase

| # | Metric | Why it is missable | Cost to capture | Priority |
|---|---|---|---|---|
| 1 | K = 1 time-to-target with the fast walk | Only a dedicated walk measures it; the paper otherwise has to say "5× slower" | One GPU, ~2–3 h | **Done today: FW running** |
| 2 | Time and Wh to each size point | `cost_readout.py` gave whole-walk totals; per-point numbers were mined by hand | CPU, ~1 h of code | **Built today**: per-point lines in `cost_readout.py`, which reproduce N3's hand-mined minutes exactly |
| 3 | The one-time train's energy | All five running trains were submitted before the sampler existed (1 Oct ~11:00). Their energy can only be estimated (GPU-h × the SKU's mean power) | Free for the next train (sampler on by default). For the current chain, one measured mean power per SKU | Medium: say "estimate" in the paper |
| 4 | Deployment rows for agent-chosen networks | Freeze TESTs do not save trajectory models, and today's agent = a uniform 0.8 schedule, whose networks are uniform cuts | One bench job once an agent differs from uniform | Low until then |
| 5 | Precise energy (NVML) | The 1 s `nvidia-smi` average is adequate for Wh per walk; NVML is better for short windows | Installing `nvidia-ml-py` (your call) | Low |
| 6 | CO2e | Needs BGU's PUE and grid carbon intensity, otherwise conventional defaults | One email | Low |
| 7 | DepGraph selected on our val half | Their protocol picks epochs on the test set. This matters only if DepGraph and SPECTRA share an accuracy table | A patched copy of their `main.py`, one GPU-hour | Only if needed |

### I.6 My opinion, and what I did with the GO

**I agree it matters, at moderately high priority, below the reward fix.**

The case for it: the paper's argument is transfer and cost, not accuracy at home benchmarks. Reviewers will ask for counted search and train costs (Renda et al.; Lindauer & Hutter). Some numbers must be captured when the runs happen; they cannot be reconstructed later. Items 1–3 above are the ones that would be lost.

The limit: every cost advantage over learned pruners is shared with mild, the no-agent walk. The cost story becomes an agent story only when the agent at least matches mild. Part III shows the agent is currently a constant schedule, so metrics alone cannot carry the paper. That is why the dev phase is narrow.

**Done under the GO today:**
- **FW (22127216).** N3's exact line plus GPU augmentation and 12/4, on a 4090, with registered calls. It reports K = 1 minutes, Wh and accuracy at DepGraph's 1.67× / 2.11× / 2.57× FLOPs points.
- **This report.**
- **Efficiency file updated** (§11 open items).

**Also built:** the time-to-each-size-point readout (item 2) in `cost_readout.py`, so ops reads FW without hand-mining. On N3 it reproduces the hand-mined minutes exactly.

**Left for you:** NVML (a package install), BGU's PUE (an email), and a val-selected DepGraph (only if we need a shared accuracy table).

---

## Part II. Which filters, a second agent, NAPv2, robustness (Gilad's point 2)

### II.1 What Gilad asked

- How does the SOTA we benchmark against choose which filters to prune? Add that column to each paper's rows.
- Consider a second DRL agent that decides which filters to prune.
- Read the robustness vs verification literature in DRL.
- Above all, consider NAP2 as a CNN representation for supported decision-making: scan weights, activations and filters, and open the way to smarter choices than L1, FPGM or the two-decision head. Build on Michael Bohadana's NAPv2 repository.

### II.2 The column: how SOTA chooses filters

Done for **49 published methods** (design §2), and added to every living SOTA table and the literature canvas. A structured cut makes two decisions per coupled channel group: *how many* channels survive (allocation) and *which* ones (selection).
- **Selection is mostly magnitude.** DepGraph, GReg, ResRep, OCSPruner, Network Slimming and Polarization first train the doomed channels toward zero, then cut by norm.
- **Allocation is often not learned.** It is uniform or hand-set in HRank, GReg, CHIP, Li et al., FPGM, SFP and C-SGD.
- **The few learned pruners learn allocation only** (AMC, AGMC, GNN-RL, RL-Pruner), and select by magnitude or Taylor.
- **SPECTRA is in the same position:** its agent learns allocation, and selection is its L1 group vote.
- **Our two-decision head (keep-rate × criterion) has precedent.** LFPC (CVPR 2020), MFP, and closest, Balaskas et al. (IEEE TETC 2024): DDPG per-layer ratio plus a DQN choosing the pruning algorithm. What is new is the setting (frozen, offline, across families and datasets), not the head.

### II.3 The second agent: expectations, design, results, way ahead

**Expectations (written 1 Oct, before any result).**
- The literature predicts that which filters survive matters with short or no fine-tuning and at high sparsity, and fades by about 40 epochs. MFP's matched comparison: 84.80 vs 77.45 with no fine-tune, 93.26 vs 93.22 after 40 epochs.
- SPECTRA scores every step after a short in-loop fine-tune (12/4 in training, 40/10 at TEST). So a better selector might matter for the agent's training signal and for cost, even if final accuracy converges.
- **Breakthrough** would be a frozen, transferable selector that improves short-fine-tune recovery at matched allocation on held-out networks. No published method has one.
- **Also publishable** would be a clean negative: at our budget, which filters survive does not matter once allocation is fixed.

**Design (never trained; design §6.4).**
- *Hierarchy.* Level 1 is SPECTRA's existing actor, unchanged: it picks the group and the keep rate. Level 2 picks the survivors: every channel of the chosen group is a token (its NAP-F descriptor plus the level-1 state).
- *Scoring.* A small permutation-equivariant set transformer (2 layers, width 64, ~50K parameters) scores the tokens. The scores define a Plackett–Luce distribution, so masks are sampled by Gumbel-top-k with exact log-probabilities.
- *Pretraining.* The level-2 policy starts from the S1 scorer.
- *Reward, paired against L1.* Recovered val accuracy of the policy's mask minus L1's, at the same allocation, the same fine-tune seed and a short budget. Pairing removes the allocation's variance and the fine-tune seed's, so the reward is the selection effect itself.
- *Safety.* A "selection shield" from the verification literature: if the policy's mask is worse than L1's on calibration data by more than δ, use L1's. The second agent is then no worse than L1 by construction.
- *Cost.* About 2,000 group-level episodes, roughly one GPU-day.

**The gated ladder and its results** (probes, never TEST rows):

| Step | Question | What ran | Result | Gate |
|---|---|---|---|---|
| **S0** (1 Oct) | Does *which* matter at fixed allocation, under our fine-tune? | 3 literature cells: DepGraph R56 C10, VGG-16 C10, DepGraph VGG-19 C100. Every group cut to the same keep (0.8, 0.6). 10 criteria + random + anti-L1. Budgets 0 / BN-only / 1 / 3 / 10 / 40 epochs; L1 × 3 fine-tune seeds | Large effects at no or one-epoch fine-tune (e.g. VGG-16 activation +17.1 pp at budget 0). After 40 epochs only one criterion cleared the noise bar: L2 on VGG-19 C100, +1.03 vs a 0.95 bar, itself a norm. The single-channel ablation oracle's own masks ended **below** L1 at 40 epochs (−0.55 / −0.30 / −0.71 pp) | **M8 fired 3/3** (mostly at budgets ≤ 1) ⇒ S1 |
| **S1** (2 Oct, zero GPU) | Can a learned score from NAPv2-style features rank channels like the oracle on a network it never saw? | Learning-to-rank on S0's per-channel features, leave one network out | **G1 PASS 3/3.** Within-group Kendall τ vs the oracle 0.57 / 0.66 / 0.64, against 0.24 / 0.42 / 0.17 for the best hand criterion picked on that net. Nearly all the signal is in the **gradient** statistics | ⇒ S2 |
| **S2** (2–3 Oct) | Does that ranking buy recovered accuracy on two unseen nets? | MobileNet-V2 ×0.5 C10 and ResNet-56 C100; nap_f vs L1 masks, paired seeds | At 40 epochs: **+0.21 pp** on MobileNet-V2 (L1 seed SD 0.72) and **−0.87 pp** on ResNet-56 C100 (SE 0.44), 0.01 past the registered −0.86 HARM bar. No budget of 3 epochs or less passes on both nets. On ResNet-56 C100 at 40 epochs every non-L1 mask ends below L1 (oracle −1.03, random −1.44, anti-L1 −1.72) | **G2 HARM** ⇒ S3 closed |
| **S3** | Train the second agent | — | **Not trained** | closed 3 Oct |

**What we learned.**
1. **Beyond magnitude, selection is not a lever at our fine-tune budget.** Even the oracle's masks do not beat L1 after 40 epochs.
2. **Magnitude is necessary.** Random masks lose 1.4–1.5 pp to L1 after 40 epochs. Anti-L1 is catastrophic on MobileNet-V2 (−79 pp: it keeps dead channels).
3. **The learned ranking transfers within a family, not across families.** τ vs the oracle is 0.423 against L1's 0.292 on ResNet-56 C100 (the ResNet family was in S1's fit), but 0.254 against 0.286 on MobileNet-V2. Lesson for any learned score: hold out a family, not only a network.
4. **The one budget where masks differ a lot is BN recalibration with no fine-tune.** There, MobileNet-V2's oracle is +17.0 pp, of which nap_f captures +2.8. That would matter only if a BN-only proxy became the in-loop reward. The proxy-fidelity cells (§189, §195) found no valid cheap proxy, so it does not.

**Way ahead for selection.** S3 stays closed, and SPECTRA keeps the L1 group vote. S1b (a selector for a BN-only in-loop proxy) reopens only if a no-fine-tune proxy proves valid. Paper wording: "beyond magnitude, which channels survive did not change recovered accuracy at our fine-tune budget; allocation is what the agent must learn." Not "selection never matters": random and anti-L1 masks lose.

### II.4 NAPv2: what was absorbed, and what worked

**What NAPv2 is** (read from the code, design §5):
- *Lineage.* It continues Amsel & Katz's NAP2 and is published as Bohadana, Schneider & Katz (TMLR 2026).
- *What it predicts.* It is a **network-level** performance predictor. It snapshots every layer's weights and gradients during training and computes 12 statistics per tensor. It lays them out as a [65 layers × 100 values × 12 statistics] map per snapshot, compresses each map with an autoencoder, and scores the sequence with a BiGRU, at any point in training.
- *What it is trained on.* NAS-Bench-201 cells trained from scratch.

| Absorbed | Not absorbed, and why |
|---|---|
| The snapshot collector, the 12 statistic definitions, log-normalization, the map layout, the AE / BiGRU trainers with anytime truncation and Kendall evaluation | **The pretrained weights.** Trained on from-scratch NAS cells under SGD lr 0.1; our fine-tunes start from a converged, freshly cut network under Adam. Its accuracy error is roughly 12–29 points RMSE (its Fig. 6), so it supports ranking, not calibrated accuracy |
| **NAP-F:** the 12 statistics (16 numbers) computed **per filter**, on its weights and on its calibration-loss gradient. Checked against NAPv2's own `extract_layer_stats` to 1e-9 | **The per-unit block as a filter signal.** As shipped, its per-unit statistics run along the kernel-width axis, and its 100-value rows hold only the first one or two filters of each layer: no per-filter signal |
| NAPv2's network-level maps, recorded over every ResNet-56 fine-tune in S0 | **`NAP2Predictor.score()` inside the prune loop.** It would add a second partial training per step and predicts a different quantity (the 20 Aug verdict still holds for that use) |

**Did it work?** As a *descriptor*, yes: NAP-F's gradient statistics are what let S1 rank channels like the oracle on held-out networks (τ 0.57–0.66). As a *lever for recovered accuracy*, no (S2). The NAPv2 statistics are the part of Gilad's idea that survived.

**Where NAP goes next** (design §6.1):
- **R2: an anytime recovery predictor** from early fine-tune snapshots, used to rank candidate cuts or stop fine-tunes early. A cost lever, never the reward. It must beat the free proxies: BN-recalibrated accuracy and accuracy after one epoch.
- **R3: NAP-F as agent state.** Newly relevant after Part III: if the agent must learn allocation, per-group sensitivity features (the gradient statistics) are what its state lacks.

Both follow A0. If allocation has no headroom, neither has much to act on.

**Questions for Gilad / Michael:** trained AE / BiGRU weights and NAS-Bench-201 snapshots; the expected acknowledgment or co-authorship if NAP-F builds on NAPv2.

### II.5 Robustness vs verification in DRL

Mapped in design §7:
- **Robustness** is empirical: behavior under chosen attacks or noise, which can only over-estimate safety.
- **Verification** is a proof over a whole input set: sound, but complete only for small networks.
- **The DRL-verification line** is Guy Katz's group at HUJI (Reluplex, Marabou, whiRL, verification-driven agent selection). We found no DRL-verification paper by Gilad.

SPECTRA's hooks, ranked:
1. An action-stability certificate for the frozen actor (randomized smoothing over its state features).
2. Choosing among frozen seeds by their agreement on unseen networks (after Amir et al., CAV 2023).
3. The selection shield (II.3).
4. Provable Filter Pruning sensitivities as a feature.

**4 Oct caveat.** FR43 re-walked the Stage-4 actor with a new fine-tune seed: its widths matched at 17 / 17 and 61 / 61 points. For a constant policy (Part III) that is trivially true. Hook 1 means something only for a state-dependent actor.

### II.6 Was there a leap forward?

Not in accuracy. Two results move SPECTRA's science:
- **A matched negative that no paper has measured on these cells.** At SPECTRA's recovery budget, *which* filters survive is not a lever beyond magnitude; allocation is.
- **NAP-style gradient descriptors carry transferable information about CNN filters**, within a family. It is the first concrete use of Gilad's NAP idea at filter level.

Part III then shows the allocation agent is not learning allocation at all, and why. That makes A0 and the reward the next science, not the second agent.

---

## Part III. M1-neg diagnosed: the agent plays a constant schedule (4 Oct, zero GPU)

### III.1 The census

Every `step` event of the TEST walks, compared action by action with mild 21729557 (scripts `_tmp_s4oct_census*.sh`):

| Actor (TEST job) | Menu | R20-w2 free decisions | R56-w4 free decisions | Forced identity steps |
|---|---|---|---|---|
| Stage-4 ep0095, seed 42 (21990060) | keep 1.0, or 0.9 / 0.8 with L1 or FPGM | 16 / 16 at 0.8 (L1) | 60 / 60 at 0.8 (L1) | 26 / 54, the same steps as mild |
| Stage-4 ep0095, seed 43 (FR43, 22059502) | same | 16 / 16 at 0.8 | 60 / 60 at 0.8 | same |
| C2 NEON-raw ep0083 (22056144) | same | 16 / 16 at 0.8 | 60 / 60 at 0.8 | same |
| Stage-4 ep0131 (22124693, running) | same | 16 / 16 at 0.8 | 12 / 12 so far at 0.8 | same so far |
| Budget+STOP ep0131 (22059501) | budgets of 1 %, 2 % or 4 % of the origin's params per cut, or STOP | 4 % at every cut | 4 % at every cut | never STOP |

Mild plays 0.9 at the same decisions. Where 0.8 rounds to the same width on a thin group, a decision realizes no cut: 13 / 51 decisions cut for the 0.8 schedule, 12 / 52 for mild. **M1-neg is therefore a comparison of two fixed schedules, uniform 0.8 against uniform 0.9.** On TEST (§193, §197–§199):
- The 0.8 schedule is more than 0.5 pp worse at its first cut on both nets.
- It is **kinder** than mild over keep 0.73–0.62 on R56-w4. This replicates across two walks (14 of 17 keeps with a gap of at least +0.5 pp).
- R20-w2's first-point deficit does not replicate (−1.42 with seed 42, −0.42 with seed 43).
- One walk's noise is 0.5–1.2 pp (RW43, FR43).

In training, PPO samples from the policy, so train walks are mixed. The deterministic TEST takes the most likely action, which is the same at every decision.

### III.2 Why: the reward pays size inside a 10 pp band

All five trains used `SPECTRA_REWARD_MODE=structural` with τ = `--allowed_acc_reduction` = 10 pp and no train-only band. They differ only in scale: `cbrt_cubes` (Stage-4, Budget, factored), `cbrt_miss` (C1), `raw` (C2). Per step, with Δ = the pruned net's val accuracy minus the **origin's** (cumulative) and ρ = the step's realized parameter cut in %:

| Arm | Condition | Reward (`cbrt_cubes` / `cbrt_miss` / `raw`) |
|---|---|---|
| gain | Δ > 0 | +ρ / +ρ³ / +ρ³ |
| in band | −10 ≤ Δ ≤ 0 | **+ρ, whatever Δ is** |
| miss | Δ < −10 | −ρ / −ρ / −ρ³ |

The Budget arm also pays 100 × the slack-weighted in-band area when it STOPs. That area grows with every legal cut, so stopping early never pays.

The walks have a fixed number of decisions (2 passes over the groups). The thin walks end 4–5 pp below origin on val, so the band never binds. Under this reward, "the largest cut at every decision" is the return-maximizing policy. Replaying the TEST walks through the live `compute_reward` (val only, `scripts/reward_replay.py`):

| Net | Walk | Return over the walk (live / C1 = C2) | Return to equal depth | Val Δ at that depth (pp) |
|---|---|---|---|---|
| R56-w4 | mild × 3 (21729557, RW43, D5-bis) | 126.6 / 126.6 (to keep 0.622) | **126.0** at keep 0.626 | −4.94 / −4.76 / −5.12 |
| R56-w4 | 0.8 schedule × 3 (21990060, FR43, C2) | **270.5** / 270.5 (to keep 0.389) | **124.9** at keep 0.628 | −7.98 / −4.64 / −5.70 |
| R20-w2 | mild × 3 | 102.9 / 1566–2267 (to keep 0.522) | 91.5 at keep 0.587 | −4.36 / −4.82 / −3.52 |
| R20-w2 | 0.8 schedule × 3 | **144.2** / **8255–8544** (to keep 0.413) | 70.5 at keep 0.587 | −2.94 / −3.22 / −3.34 |

- At equal depth the reward is blind to accuracy: on R56-w4 it pays 126.0 vs 124.9 while val differs by up to 3 pp.
- Over the walk it pays the 0.8 schedule twice as much, because 0.8 goes deeper in the same number of decisions without leaving the band.
- C1 and C2's cubic gain arm makes it starker on R20-w2: a big cut while the net is still above the origin pays ρ³.

The agent learned exactly what it was paid for. Our TEST metric is accuracy at equal size; the training reward is size within a wide band. They are misaligned.

### III.3 What it means

- **M1-neg is a reward-design result, not an RL-capacity result.** PPO found the reward's optimum.
- **FR43's stability is trivial** for a constant policy. Do not present it as robustness evidence (II.5).
- **A new action menu alone cannot help:** the agent will pick its largest entry.
- **N10** (the cubic reward under P, triggered by gain-arm cuts) is already covered by C1 / C2. Both had the cubic gain arm, and C2 collapsed the same way.
- **N8** (the diverse train) would inherit the same collapse. It waits.
- **For the paper:** "under a band reward the agent learns a constant schedule" is itself a finding about NEON's reward transferred to CNNs. NEON's dense-network walks were short and its band was hit more often.

### III.4 Way ahead

1. **A0 (submitted; queue file "A0").** Does any allocation beat uniform at equal size under our 40-epoch recovery?
   - *Allocations:* uniform (3 fine-tune seeds), a sensitivity rule (Li et al. 2017, continuous), its reverse, and 4 random draws.
   - *Cells:* thin R56-w4 (an M1 net), DepGraph ResNet-56, VGG-16, each at keep 0.6 and 0.35.
   - *Prior:* Liu et al. (ICLR 2019) found learned allocations help VGG more than ResNets, so a split is plausible.
   - *If FLAT everywhere,* a reward fix can at best relearn mild. The agent's case then rests on schedule, stopping and transfer cost, and the 8 Oct slides say so.
   - *If HEADROOM on some family,* that family belongs in the agent's TEST suite and the reward fix is worth a train.
   - *Already seen in the smoke* (unmatched, 1 epoch, not a call): the sensitivity rule +8.7 pp over uniform on val, its reverse −18. Allocation clearly matters at short fine-tune. The cells say whether it survives 40 epochs.
2. **A reward that prices accuracy at equal size, pre-checked by replay before any train.** Replaying the six thin walks:
   - *τ = 10.* Every shape the trains used, and even the slack-tapered `structural_unified` (F1: in band ρ · (τ + Δ) / τ), pays the 0.8 schedule more over the walk.
   - *F1 with a 5 pp training band.* It is the first shape that pays the gentler mild walks more on both nets: 78 vs 69 on R20-w2, 43 vs −16 on R56-w4. But at equal depth on R56-w4 it still pays the 0.8 walks more (50.5 vs 43.0), although their val accuracy there is lower on average (−6.11 vs −4.94 pp). The slack-weighted sum pays early cuts, not the accuracy reached, so F1 alone does not price accuracy at equal size.
   - *Config only.* Both switches exist (`SPECTRA_REWARD_MODE=structural_unified`, `SPECTRA_TRAIN_TAU=5`).
   - *Alternatives for a sitting:* fixed-budget episodes (AMC's resource-constrained mode: the target size is sampled per episode and given to the actor, and the reward is val accuracy at the target); or NEON's preference weight made explicit and sampled per episode.
   - *A train needs your GO.* Pre-registered success: the frozen actor plays at least two different actions on a TEST net, and M1 at equal keep.
3. **R3: per-group sensitivity in the state** (NAP-F gradient statistics or A0's sensitivity), only if A0 finds headroom. Without state that distinguishes groups, a state-dependent allocation cannot be learned.
4. **N8 waits** for a reward that passes the replay pre-check and an A0 that shows headroom on at least one family.

---

## Appendix: where things are

- *Cost and side metrics:* `docs/paper/EFFICIENCY_AND_TRANSFER.md`. Tools: `scripts/cost_readout.py`, `scripts/bench_deploy.py`, `scripts/h2h_readout.py`, `scripts/spectra.sbatch` (sampler).
- *Selection, NAPv2, robustness:* `docs/paper/FILTER_SELECTION_NAP_DESIGN.md`. Tools: `scripts/selection_probe.py`, `selection_scorer_s1.py`, `selection_probe_s2.py`.
- *A0:* `scripts/allocation_probe.py` + `.sbatch`, `tests/test_allocation_probe.py`. Outputs in `tree_v9d/runs/allocation_probe/alloc_<net>_<job>/results/`.
- *Reward replay:* `scripts/reward_replay.py` (O38, 1 Oct). Today's equal-depth replays are sitting scripts, summarized in III.2 and ledger §200.
- *Ledger:* §188 (S0), §191 (S1), §192 (S2), §193 / §197 / §198 / §199 (freeze TESTs, M1-neg), §196 (D5 equivalence, re-walk noise), **§200** (today's diagnosis).
