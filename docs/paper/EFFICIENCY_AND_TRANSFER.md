# SPECTRA cost, deployment and transfer evidence (paper file)

Started 1 Oct 2026 on Ido's 09:29 request. It holds the side metrics the paper argues from: per-target search cost, the agent's overhead at TEST, wall-clock and GPU-hours per target and per K targets, memory, energy, deployment latency, and the transfer protocol. SPECTRA numbers come from our own logs, with job IDs and the GPU named in each run's manifest. Literature numbers carry their source, their GPU and what they include. Third-party estimates are marked. Every citation below was checked against the primary source on 1 Oct: two web fact-checks, plus NEON's Table 7 read from the paper PDF.

Tools: `scripts/cost_readout.py` (login node, zero GPU), `scripts/bench_deploy.py` + `scripts/bench_deploy.sbatch` (one GPU), the 1 s `nvidia-smi` sampler in `scripts/spectra.sbatch` that writes `gpu_samples.csv`, and the DepGraph re-run `scripts/h2h_depgraph.sbatch` with its readout `scripts/h2h_readout.py`. Their use is in §4.5 and §9.

## 1. The claim this file supports

On accuracy at a given compression, focused SOTA on its home benchmark is ahead. On DepGraph's own ResNet-56 checkpoint, our no-agent walk plus final fine-tune lands at −0.46 pp at 2.11× FLOPs on the full 10k test set, against DepGraph's +0.24 pp (ledger §157). Gilad's directive stands: never claim a beat there. SPECTRA's contribution is a **frozen generic agent**. It is trained once offline, then applied to unseen CNN families and datasets with no per-target search, no agent training and no per-layer ratio tuning. The paper should argue from the costs that design removes, and from the breadth of transfer, not from the accuracy column. It should also say plainly where the design does not pay yet (§6.2).

## 2. Bottom line (1 Oct)

- **The agent's own cost is negligible.** The Stage-4 actor has 2.73 M parameters (10.9 MB in FP32). Compared with a heuristic walk on the same GPU type, an actor adds at most 0.33 s per step on R56-w4 and 0.004 s on R20-w2. That is about 0.4% of a step. Peak GPU memory is the same, 0.36 GB, for actor and heuristic walks on an RTX 3090. Both figures are upper bounds until the runner has an explicit agent timer (§3.4).
- **A TEST's cost is recovery fine-tuning.** Fine-tuning is 97.9–99.7% of every CIFAR walk measured (8 jobs, 13 network walks). The rest (pruning surgery, validation pass, state features, bookkeeping and the agent) is 0.4–2.0 s per step. On ImageNet MobileNet-V2, fine-tuning is 84–85% of the walk, and the per-step validation pass is the other 15%.
- **Per-target search cost is zero.** Per-target learned searches cost 320 s to 3.8 GPU-hours on CIFAR. On ImageNet they cost 25 GPU-hours (EagleEye) up to 864 (NetAdapt, by EagleEye's estimate) (§4.1–4.2).
- **One target on CIFAR is not a win at the current TEST recipe.** On an RTX 4090, our 40/10 walk on ResNet-56 takes 3.6 h (down to 0.70 of params kept) or 9.1 h (down to 0.36), plus 16 min per final fine-tune. DepGraph takes **85 min** per target on our 4090 (job 21943448). OCSPruner takes 26 min on a 4090, including training the network from scratch.
- **DepGraph re-run on our GPU is in.** Job **21943448** COMPLETED 2 Oct 03:11 (2.3 h, TB=0): Torch-Pruning v1.6.1 official pipeline on an RTX 4090 from the same released checkpoints we prune. Numbers in §4.5. Never a ledger row. Never “beats.”
- **Several targets amortize.** One walk passes through every size point, so each extra target costs one final fine-tune: 15.7–16.6 min for R56, 8.6–8.8 min for VGG on a 4090. Against DepGraph on a 4090, break-even is about 3 targets for the 0.70-deep walk and about 8 for the 0.36-deep walk (§7).
- **Three cost levers are measured or measurable.**
  - The 12/4 recovery budget runs 3.3× fewer fine-tune epochs than 40/10 (measured).
  - A no-fine-tune proxy, such as BN recalibration, would make the walk take minutes. The proxy-fidelity cell (21941343–48, **COMPLETED**, ledger §189) is **uninformative** at these cut sizes (ceiling ρ +0.41). Widen cuts before reading 12/4 vs 40/10.
  - CIFAR fine-tuning is input-pipeline-bound, so GPU-side augmentation is a free speedup if an equivalence A/B passes (§3.3).
- **No CNN work transfers a frozen agent across families and unseen datasets.** The closest works either transfer within one architecture or warm-start a new search (§8).
- **Deployment metrics were measured on our GPUs.** Job **21942378** (RTX 4090, COMPLETED 01:08) records latency at batch 1, 64 and 256, throughput, peak memory and energy per image. Most pruning papers report FLOPs only (§5).

## 3. SPECTRA measured costs

From `scripts/cost_readout.py` over each run's manifest, `events/*.jsonl` and log. "Non-FT" is everything in a step that isn't fine-tuning: prune, validation pass, state features, and the gap between steps (bookkeeping plus the agent's decision). The walk recipe is Adam 1e-3, with up to 40 epochs and patience 10 (40/10) at TEST, or 12 epochs and patience 4 (12/4) in training. The final fine-tune is SGD 0.01, cosine schedule, 100 epochs. P means the 5k val / 5k TEST split of the CIFAR test set at batch 256. Aug means crop+flip in the walk's fine-tune.

### 3.1 TEST walks

| Job | Walk | GPU (node) | Net | Steps (cuts) | Walk | FT share | FT s/step | Non-FT s/step | FT epochs | Final FT | Peak alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21729557 | mild, P+aug, 40/10 | RTX 4090 (ise-4090-20) | R20-w2 | 42 (16) | 43.9 min | 99.3% | 62.3 | 0.42 | 610 | — | 0.36 GB |
| 21729557 | same | same | R56-w4 | 114 (60) | 209.3 min | 99.5% | 109.6 | 0.85 | 2386 | — | 0.36 GB |
| 21729556 | mild, P+aug, 12/4 | RTX 3090 (cs-pheno-06) | R20-w2 | 42 (16) | 16.4 min | 98.1% | 23.0 | 0.46 | 192 | — | 0.36 GB |
| 21729556 | same | same | R56-w4 | 114 (60) | 67.8 min | 98.2% | 35.1 | 0.92 | 714 | — | 0.36 GB |
| 21767189 | mild, P+aug, 40/10 + final | RTX 4090 (ise-4090-19) | DepGraph R56 | 285 (150) | 545.5 min | 99.5% | 114.2 | 0.89 | 5884 | 5 × 15.7 min | 1.32 GB |
| 21809595 | same | RTX 4090 (cs-4090-07) | chenyaofo R56 | 114 (60) | 213.5 min | 99.5% | 111.7 | 0.89 | 2334 | 4 × 16.6 min | 3.17 GB (job) |
| 21809595 | same | same | VGG-16-BN | 30 (28) | 77.4 min | 99.7% | 154.2 | 0.83 | 1099 | 4 × 8.8 min | (job) |
| 21814029 | same, L2 ranking | RTX 4090 (ise-4090-18) | VGG-16-BN | 150 (140) | 371.7 min | 99.7% | 148.2 | 0.76 | 5479 | 4 × 8.6 min | 3.17 GB |
| 21737105 | same | RTX 4090 (cs-4090-07) | DepGraph VGG-19, C100 | 48 (45) | 126.9 min | 99.7% | 158.1 | 0.81 | 1800 | 4 × 8.6 min | 3.39 GB |
| 21512868 | frozen actor, legacy split, 40/10 | RTX 3090 (cs-pheno-08) | R20-w2 | 42 (16) | 27.7 min | 97.9% | 38.8 | 0.82 | 640 | — | 0.36 GB |
| 21512868 | same | same | R56-w4 | 114 (60) | 151.6 min | 98.7% | 78.7 | 1.66 | 2376 | — | 0.36 GB |
| 21725472 | frozen actor, legacy split, 40/10 | RTX 2080 Ti (ise-pheno-05) | R20-w2 | 42 (16) | 38.6 min | 98.3% | 54.2 | 0.95 | 640 | — | 0.20 GB |
| 21725472 | same | same | R56-w4 | 114 (60) | 307.2 min | 99.2% | 160.4 | 1.98 | 2384 | — | 0.20 GB |
| 20360208 | frozen actor (pre-audit), 3-epoch budget | RTX 4090 (ise-4090-14) | MobileNet-V2, ImageNet | 104 (62) | 105.0 h | 84.3% | 3062 | 572 | 126 | — | 11.05 GB |
| 20382192 | same | RTX 4090 (ise-4090-14) | MobileNet-V2, ImageNet | 104 (67) | 122.3 h | 84.9% | 3594 | 639 | 132 | — | 10.98 GB |

These rows are cost only. Accuracy for the walks that have it is in the ledger: §152 for 21729557, §157 for 21767189, §164 for 21809595. The actor rows (21512868, 21725472) used the legacy 10k evaluation split without augmentation, so they don't share a column with the P rows on accuracy. The ImageNet runs predate the 4 Sep audit: their policy was sampled. Their cost is valid; their accuracy is not quoted.

### 3.2 Training (one-time)

| Job | What | GPU (node) | Elapsed (1 Oct 11:00) | Episodes | Median episode | Peak alloc |
|---|---|---|---|---|---|---|
| 21737123 | Stage-4 train, 10-net catalog, P+aug, 12/4 | RTX 6000 Ada (ise-cpu256-32) | 31.8 h, running; resume 21767188 chained | 39 | 28.4 min | 6.11 GB |
| 21938807 | G2 cubic-gain reward arm | RTX 4090 (cs-4090-07) | 8.0 h, running | 12 | 26.6 min | 6.14 GB |
| 21938810 | G2 raw-NEON reward arm | RTX 4090 (ise-4090-15) | 7.6 h, running | — | — | — |

Stage-4 has no warm-start or resume keys in its manifest, so the one-time cost of the agent we TEST is this chain's GPU-hours. Report the chain total when it ends. Fine-tuning is 97.6–98.9% of training time per network as well, so the train has the same cost structure as a TEST walk, repeated over episodes.

### 3.3 GPUs, and what bounds the walk

The GPU types we use, all named from run manifests:
- RTX 4090: ise-4090-14/15/18/19/20/21, cs-4090-07/10.
- RTX 3090: cs-pheno-06/08.
- RTX 2080 Ti: ise-pheno-05.
- RTX 6000 Ada: ise-cpu256-32. Slurm's `rtx_6000` label on this cluster is the Ada card.

With P+aug, every CIFAR walk runs at **4.2–5.6 s per fine-tune epoch**, whatever the network and GPU:

| Network | GPU | s/epoch |
|---|---|---|
| R20-w2 (thin) | 4090 | 4.3 |
| R20-w2 (thin) | 3090 | 5.0 |
| R56-w4 | 4090 | 5.2 |
| R56-w4 | 3090 | 5.6 |
| DepGraph R56 | 4090 | 5.5 |
| VGG-16-BN | 4090 | 4.2 |
| VGG-19, C100 | 4090 | 4.2 |

So the CIFAR walk is bound by the CPU augmentation pipeline, not by the GPU. Under the legacy protocol (no augmentation, default batch size, 10k split), the same 3090 type runs R56-w4 at 3.8 s/epoch. Two consequences:
- Cross-GPU wall-clock comparisons on CIFAR mostly measure the data loader. Quote GPU-hours with the SKU, and compare against literature numbers measured on the same SKU where they exist: OCSPruner and the graph-metanetwork paper both used an RTX 4090.
- GPU-resident CIFAR with on-GPU crop+flip is a speedup that leaves the science unchanged, provided an equivalence A/B shows the same augmentation distribution and the same TEST numbers. It is a recipe change, so it goes in a new cell only, never mid-train.

### 3.4 The agent's overhead, and its caveats

The cleanest pair available today is on the same GPU type (RTX 3090): the actor walk 21512868 against the mild walk 21729556, on the same two thin nets. The gap between steps is 0.666 s against 0.34 s on R56-w4, and 0.044 s against 0.040 s on R20-w2. State features cost the same (0.21 vs 0.19 s). So the actor adds at most about 0.33 s per step, about 0.4% of the actor walk's 80 s step.

That is an upper bound: the two runs differ in protocol (legacy vs P), and the gap between steps also holds bookkeeping. Their validation pass differs too (0.7 s vs 0.33 s), which reflects the 10k vs 5k evaluation split, not the agent. Peak allocation is 0.36 GB for both. The explicit timer is built (2 Oct, `tree_v9d`, default off). Under `SPECTRA_TIME_DECIDE=1` each eval-walk decision becomes a `step.decide` stage: the actor's forward and pick, or the heuristic's pick, timed the same way for both. `scripts/cost_readout.py` prints `decide … ms` per net. The first frozen-actor TEST that sets the flag replaces this bound with a measurement.

## 4. Literature: what pruning costs other methods

### 4.1 Per-target search cost of learned pruners (CIFAR)

| Method | Per-target cost as reported | GPU | Includes | How filters are chosen (how many · which) | Source |
|---|---|---|---|---|---|
| AMC (He et al., ECCV 2018) | "within 1 hour" (CIFAR-10) | 1× TITAN Xp | RL search; fine-tune extra | DDPG per-layer ratio · largest-magnitude channels + least-squares refit | [arXiv:1802.03494](https://arxiv.org/abs/1802.03494) §4.1 |
| AGMC (Yu et al., ICCV 2021) | (320 ± 30) s, ResNet-56, 300 episodes | RTX 8000 | search | DDPG on a GCN embedding · smallest L2 (code) | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Auto_Graph_Encoder-Decoder_for_Neural_Network_Pruning_ICCV_2021_paper.html) §4.2 (not in arXiv v1) |
| GNN-RL (Yu et al., ICML 2022) | "within half a GPU hour" | V100 | search | PPO on a GNN · smallest L2 (code) | [arXiv:2102.03214](https://arxiv.org/abs/2102.03214) |
| TAS (Dong & Yang, NeurIPS 2019) | 3.83 GPU-h, ResNet-32 | 1× V100 | search | searched widths · none: first *k* channels, re-initialised and distilled | [arXiv:1905.09717](https://arxiv.org/abs/1905.09717) Table 2 |
| DSA (Ning et al., ECCV 2020) | DSA 5 GPU-h; AMC ≈ 3 GPU-h; traditional pretrain → allocate → fine-tune "up to 10 GPU hours" (ResNet-56) | — | DSA's estimates | differentiable keep ratios · top BN-scale magnitude | [arXiv:2004.02164](https://arxiv.org/abs/2004.02164) §5.3 |
| RL-Pruner (Wang & Kindratenko, 2024) | "several hours", then a 100-epoch post-training (every method in its Table 2) | — | search + post-train | learned per-layer sparsity · smallest Taylor score | [arXiv:2411.06463](https://arxiv.org/abs/2411.06463) §4.3 |
| AgenticPruner (Esmat et al., 2026) | 7.5 effective full-dataset epochs of search, plus LLM calls (Claude 3.5 Sonnet) | — | search | LLM-proposed ratios · Taylor, ranked inside dependency groups | [arXiv:2601.12272](https://arxiv.org/abs/2601.12272) |
| Graph metanetworks (Liu, Wang, Zhang, 2025) | meta-train 357 min once; then prune + fine-tune 67 min (100+100 epochs) or 43 min (60+60), vs DepGraph 84 min (100 SL + 100 FT) | 1× RTX 4090 | per target, after the one-time meta-train | global ranking · smallest group L2 after the metanetwork rewrites the weights | [arXiv:2506.12041](https://arxiv.org/abs/2506.12041) App. A.4, Table 7 |
| **SPECTRA** | **0** per-target search; one-time train 31.8+ GPU-h | RTX 6000 Ada | the walk is the TEST (§3.1) | frozen agent sets each coupled group's keep-rate · L1 group vote | this file §3 |

The "how filters are chosen" column, here and in §4.2, §4.3 and §8.1, is the one-line form of `FILTER_SELECTION_NAP_DESIGN.md` §2 (49 published methods, checked against each paper's text or code on 1 Oct).

### 4.2 ImageNet search and pruning cost (ResNet-50 unless noted)

| Method | Cost | Basis | How filters are chosen (how many · which) | Source |
|---|---|---|---|---|
| NetAdapt | ~195 h | HALP's estimate, V100-normalized, fine-tune excluded | greedy per-layer proposals under a latency table · largest L2 kept | [HALP, arXiv:2210.06659](https://arxiv.org/abs/2210.06659) Table 3 |
| NetAdapt | 864 GPU-h | EagleEye's estimate (10^4 training iterations); network not named | as above | [EagleEye, arXiv:2007.02491](https://arxiv.org/abs/2007.02491) Table 2 |
| ThiNet | ~210 h; ≫1750 GPU-h incl. training the selected net; 244 epochs (196 + 48) | HALP T3; PaS T3; ABCPruner T3 | hand-set per-layer rate · greedy least next-layer reconstruction error | HALP; [PaS, arXiv:2206.01198](https://arxiv.org/abs/2206.01198); [ABCPruner, arXiv:2001.08565](https://arxiv.org/abs/2001.08565) |
| EagleEye | 25 GPU-h self-reported (1000 candidates, adaptive BN, 10–20 s each on a 2080 Ti); 30 h (HALP T3); 75 GPU-h incl. training (PaS T3) | mixed | random ratio candidates, picked by adaptive-BN accuracy · smallest L1 | EagleEye Table 2, §4.3 |
| HALP | 6.5 h GPU + 0.5 h CPU | V100-normalized, fine-tune excluded | global latency knapsack · highest-Taylor prefix per layer | HALP Table 3 |
| DMCP | 40-epoch search on 16× GTX 1080 Ti (batch 1024), then training from scratch on 32 GPUs; 120 GPU-h in PaS T3 | — | learned Markov widths · none: first *k* channels | [arXiv:2005.03354](https://arxiv.org/abs/2005.03354) §4.1 |
| PaS | 60 GPU-h incl. training the selected net | — | emerges from learned masks · learned binary channel mask | PaS Table 3 |
| MetaPruning / ABCPruner | 160 epochs (32 + 128) / 102 epochs (12 + 90) | epochs | evolutionary / bee-colony width search · first *k* (MetaPruning) / random filters (ABCPruner) | ABCPruner Table 3 |
| TAS (ResNet-18) | about 59 h on 4× V100 (≈ 236 GPU-h) | search | searched widths · first *k*, re-initialised | TAS §4.3 |
| TPP | 41 h on 4× V100 (≈ 164 GPU-h), incl. 90-epoch retraining | total | preset ratios · lowest L1, regularised away | [arXiv:2207.12534](https://arxiv.org/abs/2207.12534) App. A |
| OCSPruner | 27 h 6 min on 2× RTX 4090 (≈ 54 GPU-h), from scratch; baseline training is 37 h 55 min | total | global threshold · lowest group L2 under a growing penalty | [arXiv:2501.13439](https://arxiv.org/abs/2501.13439) Supp. Table 9 (WACV 2026) |
| Once-for-All | 1,200 V100 GPU-h once; for N = 40 deployments 1.2k GPU-h total, 0.34k lbs CO2e, $3.7k on AWS | amortized | evolutionary search with an accuracy predictor · largest-L1 prefix | [arXiv:1908.09791](https://arxiv.org/abs/1908.09791) Table 1 |
| APQ | 2400 + 0.5N GPU-h | amortized | joint architecture / pruning / quantisation search · UNVERIFIED | [arXiv:2006.08509](https://arxiv.org/abs/2006.08509) Table 2 |
| DepGraph | 30 sparse-learning + 90 fine-tune epochs, 8 GPUs, AMP | recipe | learned global sparsity · lowest group L2 after sparse training | [Torch-Pruning](https://github.com/VainF/Torch-Pruning) `scripts/prune/imagenet/resnet50_group_sl.sh` |
| **SPECTRA** (MobileNet-V2) | 105–122 GPU-h on 1× RTX 4090, 3-epoch recovery budget, no final fine-tune | pre-audit walk | frozen agent per coupled group · L1 group vote | §3.1 |

Wall-clock figures on several GPUs are converted to GPU-hours above (TAS, TPP, OCSPruner). Don't place them beside single-GPU figures unconverted.

### 4.3 One-shot and regularization pipelines: schedules and wall-clock

| Method | Schedule / time | How filters are chosen (how many · which) | Source |
|---|---|---|---|
| DepGraph, ResNet-56 CIFAR-10 | Official logs, read with `scripts/h2h_readout.py`; the GPU is not named. At 2.11×: sparse learning 76.9 min + pruning 14 s + fine-tune 30.3 min = 107.4 min, best epoch 93.89, last epoch 93.83. At 2.55×: 113.2 min. VGG-19 CIFAR-100 at 8.84× (it reaches 8.97×): 36.5 + 0.2 + 11.7 = 48.3 min, best 70.60, last 70.31. 100 SL + 100 FT epochs are argparse defaults, after 200-epoch pretraining. The reproduction picks its best epoch **on the CIFAR-10 test set** (`reproduce/registry.py` L128–129, `main.py` L158–179, L285–287) | learned global sparsity to a speed-up target · lowest group L2 after sparse training | [arXiv:2301.12900](https://arxiv.org/abs/2301.12900); Torch-Pruning `reproduce/` |
| OCSPruner, ResNet-56 CIFAR-10 | 26 min total vs 25 min baseline training, 1× RTX 4090, from scratch | global binary-searched threshold · lowest group L2 under a growing penalty, once stable | OCSPruner Supp. Table 9 |
| CHIP | fine-tune 300 epochs (CIFAR-10) / 180 (ImageNet); per-layer filter counts are inputs | hand-set per-layer counts · lowest channel independence (nuclear-norm drop) | [arXiv:2110.13981](https://arxiv.org/abs/2110.13981) §4.1, Alg. 1 |
| ResRep | 480 epochs (CIFAR-10 ResNet-56/110), 180 (ImageNet) | global compactor ranking to a FLOPs target · smallest compactor norm | [arXiv:2007.03260](https://arxiv.org/abs/2007.03260) §4.1 |
| OTO / ATO | OTO: 300 epochs (CIFAR-10), 120 (ImageNet). ATO: 300 (CIFAR), 240 (ImageNet ResNets) | emerges from group sparsity · groups projected to zero (ATO: a controller's mask) | [arXiv:2107.07467](https://arxiv.org/abs/2107.07467); [arXiv:2403.14729](https://arxiv.org/abs/2403.14729) |
| HRank | "For each layer, we retrain the network for 30 epochs after pruning" | hand-set per-layer rates · lowest mean feature-map rank | [arXiv:2002.10179](https://arxiv.org/abs/2002.10179) §4.1 |
| ThiNet | "we fine-tune one or two epochs after the pruning of one layer" | hand-set per-layer rate · greedy least reconstruction error | [arXiv:1707.06342](https://arxiv.org/abs/1707.06342) §3.1 |
| GReg | per-layer ratios: "We do not have strong rules to set them" | preset per-stage ratios · lowest L1, then a growing L2 penalty | [arXiv:2012.09243](https://arxiv.org/abs/2012.09243) App. A.1 |
| ICE-Pruning, ResNet-152 CIFAR-10, 60% | 1943 s vs 3793 s for naive iterative pruning (1 FT epoch per step), RTX 3090 | uniform ratio · smallest L1 (pluggable) | [arXiv:2505.07411](https://arxiv.org/abs/2505.07411) Table III(a) |

### 4.4 Cost of one pruning step

PruningBench times one pruning step (ResNet-50, CIFAR-100; [arXiv:2406.12315](https://arxiv.org/abs/2406.12315) Table 2):

| Criterion | Time per step |
|---|---|
| Random | 0.104 s |
| Magnitude L2 | 0.136 s |
| Magnitude L1 | 0.137 s |
| BN-scale | 0.141 s |
| LAMP | 0.150 s |
| FPGM | 0.163 s |
| Taylor | 3.74 s |
| OBD-C | 7.56 s |
| ThiNet | 33.6 s |
| CP | 2 min 51 s |
| OBD-Hessian | 5 min 5 s |
| HRank | 34 min 32 s |

SPECTRA's pruning surgery takes 0.01–0.10 s per step (the "prune" stage behind §3.1), which is in the magnitude-criterion range. With the validation pass, state features and the agent (at most 0.33 s), a step's non-fine-tune cost is 0.4–2.0 s. The data-driven criteria, from Taylor to HRank, cost 3.7 s to 34 min per step.

### 4.5 DepGraph re-run on our RTX 4090 (job 21943448)

The literature costs above come from other GPUs or from third parties. This re-run gives one comparator measured on our own card.

- **What runs.** Torch-Pruning v1.6.1 (commit e80127d), `reproduce/main.py --mode prune --method group_sl --global-pruning --reg 5e-4 --finetune`. It runs at DepGraph's published settings: ResNet-56 CIFAR-10 at `--speed-up 2.11`, and VGG-19 CIFAR-100 at `--speed-up 8.84`.
- **Same checkpoints.** The checkpoints are DepGraph's released ones, which are the ones SPECTRA prunes. Re-evaluated through their loaders on 1 Oct: ResNet-56 93.53, VGG-19 73.49 (their README says 73.50).
- **What it measures.**
  - Wall-clock and energy per stage: sparse learning, pruning and fine-tune, from their log timestamps and the job's `gpu_samples.csv`.
  - The accuracy of the best epoch and of the last epoch.
  - Their pruned networks, timed by `scripts/bench_deploy.py --model` in the same job on the same card: batch 1 / 64 / 256, 3 repeats, with speedup against their own origin export.
- **Readout.** `python scripts/h2h_readout.py runs/h2h_depgraph/job_21943448` in `tree_v9d`. Job **21943448 COMPLETED** 2 Oct 03:11 (`ise-4090-04`, 2.3 h, exit 0, TB=0). Their protocol selects the best epoch on the **10k test** set. Quote beside SPECTRA; never a ledger TEST; never “beats.”

| Cell | Wall-clock | Sparse-learn best acc | After prune+FT best / last | FLOPs | Params (bench) |
|---|---|---|---|---|---|
| ResNet-56 C10, `--speed-up 2.11` | **5104 s (85.1 min)** | 93.44 | **93.80 / 93.77** | 127.12 → 60.21 M (2.11×) | 0.856 → 0.432 M |
| VGG-19 C100, `--speed-up 8.84` | **2682 s (44.7 min)** | 72.46 | **70.78 / 70.53** | 512.73 → 56.83 M (9.02×) | 20.087 → 1.220 M |

Official published logs (other hardware) were 93.89 / 93.83 and 70.60 / 70.31. Our 4090 is **0.09 pp** under their R56 best and **0.18 pp** over their VGG-19 best. SPECTRA’s crop+flip walk on their R56 (ledger §157, **10k**) is **−0.46 at 2.11×** vs their published **+0.24**. Do not mix that 10k row with the 5k P half.

Deployment of *their* pruned nets, median of 3, same card (§5.3): R56 at 2.11× is still throughput-bound at batch 1 (origin 5.02 ms → 4.90 ms, ×1.02) and slightly *slower* at batch 256 (49172 → 47071 img/s). VGG-19 at ~9× is almost unchanged at batch 1 (1.76 → 1.75 ms) and **×2.85** at batch 256 (50212 → 143349 img/s).

- **Pitfalls found in the dry run.**
  - The v1.6.1 README's prune commands omit `--finetune`. Without it the script stops at the cut, so we pass it.
  - The registry reads `<dataroot>/torchdata/` and downloads whatever it cannot verify there. The job's data root is `scratch_audit/third_party/data`, with symlinks to our CIFAR copies.
  - `--mode test` crashes in v1.6.1 (`logger_name` is unbound). It is not used.
- **Selection.** Their sparse-learning and fine-tune stages both keep the best epoch by **test** accuracy.
  - "Best epoch" is their published protocol.
  - "Last epoch" removes the fine-tune selection, but the pruned model still comes from a test-selected sparse-learning epoch.
  - Their official logs show how small the fine-tune selection effect is: 0.06 pp on ResNet-56 at 2.11× (93.89 vs 93.83) and 0.29 pp on VGG-19 (70.60 vs 70.31).
- **How to quote it.**
  - Compare DepGraph's 10k-test numbers only with our 10k legacy TEST rows (ledger §157), not with the 5k P TEST half.
  - Put it **beside** SPECTRA. Never call it a beat.
  - Keep the ledger for SPECTRA TESTs; this re-run goes here.

## 5. Deployment metrics

### 5.1 What papers report, and why we measure

Most pruning papers report FLOPs and parameters only. Blalock et al. ask for compression ratio and *theoretical* speedup (the MAC ratio), an appropriate control, and at least five operating points ([arXiv:2003.03033](https://arxiv.org/abs/2003.03033) §6). Fewer papers report measured latency:

| Paper | GPU | Batch | Protocol |
|---|---|---|---|
| FPGM ([arXiv:1811.00250](https://arxiv.org/abs/1811.00250)) | GTX 1080 | 64 | forward time |
| SFP ([arXiv:1808.06866](https://arxiv.org/abs/1808.06866)) | GTX 1080 | 64 | forward time |
| ThiNet | M40 | 32 | — |
| Isomorphic Pruning ([arXiv:2407.04616](https://arxiv.org/abs/2407.04616)) | RTX A5000 | 256 (GPU), 8 (CPU) | 100 repeats |
| HALP | TITAN V | 256; batch 1 in App. G | — |
| LayerMerge ([arXiv:2406.12837](https://arxiv.org/abs/2406.12837)) | 2080 Ti, TensorRT | 128 | 300 warm-up + 200 timed passes |

Measuring matters because FLOPs don't predict latency or energy:
- **NetAdapt** ([arXiv:1804.03230](https://arxiv.org/abs/1804.03230)): "a network of 19% less MACs incurs 29% longer latency in practice" (Pixel 1 CPU).
- **Torch-Pruning README:** ResNet-50 at batch 64 runs 45.22 ms at 4.12 GMACs but 46.53 ms at 3.68 GMACs.
- **HALP:** EagleEye-1G is 2.38× faster at batch 256 but only 1.06× at batch 1 (Table 1 and App. Table 7).
- **PruneEnergyAnalyzer** ([BDCC 2025](https://www.mdpi.com/2504-2289/9/8/200)): VGG-11/16 pruned below 20% consume more energy than the unpruned model (RTX 3080).

### 5.2 Our protocol (`scripts/bench_deploy.py`)

Each candidate is set up as follows:
- **Mode and input:** eval mode, synthetic input on the device at the dataset's resolution.
- **Arithmetic:** FP32, with TF32 off on both the matmul and cuDNN flags. PyTorch's cuDNN TF32 flag defaults to on ([PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html)). Turning it off makes Turing and Ampere-or-newer cards run the same arithmetic.
- **Algorithm choice:** `cudnn.benchmark` on.

Then, per batch size (1, 64 and 256):
- 50 warm-up iterations.
- At least 300 timed iterations and at least 10 s, each iteration timed with CUDA events. The numbers follow Torch-Pruning's `torch_pruning/utils/benchmark.py`.
- Reported: median, p90, mean and std of the latency; throughput; peak allocated memory; board power sampled every 100 ms over the timed loop. Idle power is recorded first.
- Three repeats, each in a fresh process.

Fine-tuned copies share their walk twin's architecture and are timed once. The paper table uses one SKU (RTX 4090), plus a second SKU for a cross-GPU row.

Energy caveats ([Yang et al., arXiv:2312.02741](https://arxiv.org/abs/2312.02741); SC24 version):
- On Ampere and newer, `power.draw` is a 1 s average.
- On A100 and H100, `nvidia-smi` captures 25 ms out of every 100 ms.

The ≥ 10 s window gives at least 10 averaging windows. Energy per image is therefore approximate; latency and throughput are exact. For CO2e, use Patterson et al.'s formula ([arXiv:2104.10350](https://arxiv.org/abs/2104.10350) §2.5): kWh = hours × processors × average power × PUE / 1000. Then multiply by the grid's carbon intensity. Strubell et al.'s defaults, PUE 1.58 and 0.954 lbs CO2e/kWh ([ACL 2019](https://aclanthology.org/P19-1355/)), apply only if BGU's own figures are unavailable.

### 5.3 Results

Job **21942378 COMPLETED** 2 Oct 01:08 (1.25 h, `ise-4090-18`, RTX 4090, exit 0, TB=0, 0 `template failed`). Three repeats in fresh processes; 270 jsonl rows in `tree_v9d/runs/bench_deploy/bench_21942378_r{1,2,3}.jsonl`. Never a ledger TEST row. Δacc stays in the walk's ledger section. Numbers below are the **median of 3 repeats**.

Literature cells, origin vs the walk's `val_best` (or DepGraph 2.11× = `size_flop0.47`):

| Net (walk) | Point | Params / MACs | bs 1 latency (ms) | bs 256 img/s | Peak MB @256 | mJ/img @256 |
|---|---|---|---|---|---|---|
| DG R56 (21767189) | origin | 1.00 / 1.00 | 5.00 | 48022 | 193 | 5.83 |
| DG R56 (21767189) | 2.11× | 0.47 / 0.46 | 5.01 | 48497 | 133 | 4.89 |
| DG R56 (21767189) | val_best | 0.36 / 0.37 | 4.86 | 47956 | 66 | 4.65 |
| zoo R56 (21809595) | origin | 1.00 / 1.00 | 5.01 | 48420 | 98 | 5.80 |
| zoo R56 (21809595) | val_best | 0.66 / 0.66 | 4.97 | 48198 | 87 | 5.55 |
| VGG-16 (21809595) | origin | 1.00 / 1.00 | 1.55 | 59639 | 2321 | 6.75 |
| VGG-16 (21809595) | val_best | 0.66 / 0.68 | 1.53 | 73813 | 238 | 5.51 |
| VGG-16 10-pass (21814029) | origin | 1.00 / 1.00 | 1.56 | 59545 | 257 | 6.81 |
| VGG-16 10-pass (21814029) | HRank FLOPs | 0.44 / 0.46 | 1.53 | 96339 | 1378 | 4.22 |
| VGG-16 10-pass (21814029) | val_best | 0.12 / 0.15 | 1.52 | 161920 | 740 | 2.18 |
| DG VGG-19 (21737105) | origin | 1.00 / 1.00 | 1.79 | 49192 | 295 | 8.16 |
| DG VGG-19 (21737105) | val_best | 0.53 / 0.55 | 1.75 | 74694 | 261 | 5.56 |

On a 4090, CIFAR ResNet-56 is **throughput-bound at batch 1**: cutting to 0.36 kept barely moves 5.00 → 4.86 ms. VGG-16 at batch 256 does move: origin 60k img/s → val_best 74k → HRank-FLOPs 96k → deep val_best 162k. Peak memory on VGG-16 origin is noisy across repeats (2321 vs 257 MB on the 10-pass origin); quote throughput and energy, not that peak, until a sitting re-reads the jsonl. The agent's own architectures get the same bench when Stage-4 freeze TESTs land (§9).

DepGraph’s own pruned nets from job **21943448** (same card, median of 3, never ledger):

| Net | Point | Params / MACs | bs 1 latency (ms) | bs 256 img/s | vs origin @256 |
|---|---|---|---|---|---|
| DepGraph R56 | origin | 0.856 M / 126.57 M | 5.02 | 49172 | ×1.00 |
| DepGraph R56 | group_sl 2.11× | 0.432 M / 59.83 M | 4.90 | 47071 | ×0.96 |
| DepGraph VGG-19 | origin | 20.087 M / 511.95 M | 1.76 | 50212 | ×1.00 |
| DepGraph VGG-19 | group_sl ~9× | 1.220 M / 56.53 M | 1.75 | 143349 | **×2.85** |

Table to fill (one row per architecture; Δacc from the ledger section of that run):

| Net | Point | Params kept | MACs kept | bs 1 latency (speedup) | bs 256 throughput (speedup) | Peak MB at bs 256 | mJ/img at bs 256 | TEST Δacc |
|---|---|---|---|---|---|---|---|---|

## 6. Where SPECTRA beats recent papers, and where it doesn't

### 6.1 Claims the evidence supports

1. **Per-target search and agent-training cost: zero.** Compare AMC (≤ 1 h, TITAN Xp), AGMC (320 s, RTX 8000), GNN-RL (0.5 GPU-h, V100) and TAS (3.83 GPU-h, V100) on CIFAR, and 25 to 864 GPU-h for ImageNet searches. Compare RL-Pruner ("several hours") and AgenticPruner (7.5 effective epochs plus LLM calls per target). The one-time cost (§3.2) is amortized over every target, like Once-for-All's 1,200 V100 GPU-h.
2. **Each extra size target costs one final fine-tune:** about 16 min for R56 and about 9 min for VGG on a 4090. One-shot and regularization pipelines re-run per target: DepGraph 84 min and the graph-metanetwork method 43–67 min on a 4090, ResRep/CHIP/OTO 180–480 epochs.
3. **No per-layer ratio or regularizer tuning.** CHIP takes per-layer filter counts as inputs. GReg: "We do not have strong rules to set them". DepGraph needs a global ratio and sparse-learning hyperparameters.
4. **Breadth of transfer.** No CNN work claims a frozen agent across families and unseen datasets (§8). Others state the opposite:
   - GNN-RL: "a pruning strategy for a given DNN is not transferable to a different DNN".
   - Balemans et al.: policies "specific to the model architecture, dataset, and target compression rate".
   - AutoSlim: an ImageNet-searched configuration "cannot generalize to CIFAR10".
5. **The agent at TEST costs ≤ 0.4% of a step and no memory.** RL searchers, by contrast, train an agent per target, and LLM agents pay per-target inference.
6. **Selection hygiene.** We select on a val split and quote the held-out TEST half. Torch-Pruning's CIFAR reproduction keeps its best epoch by test accuracy. State this when DepGraph's CIFAR numbers sit next to ours.
7. **Reporting.** Measured latency, throughput, memory and energy on named GPUs at three batch sizes, plus walk and train energy. That goes beyond FLOPs-only reporting and beyond Blalock's theoretical-speedup minimum.

### 6.2 Limits to state in the paper

1. **One target on CIFAR is slower than one-shot SOTA at the current 40/10 recipe:**
   - our ResNet-56 walk is 3.6–9.1 h, plus 16 min per final, on a 4090;
   - DepGraph takes 84 min and OCSPruner 26 min on a 4090.

   OCSPruner trains from scratch, so it needs the full training pipeline and data per target; SPECTRA starts from a pretrained checkpoint. Say so, but don't use it as an excuse.
2. **The ImageNet walk sits inside the search-method band.** Our MobileNet-V2 walk costs 105–122 GPU-h on a 4090, without a final fine-tune. That is not a cost win on ImageNet.
3. **The one-time train is not free.** It is 31.8+ GPU-h so far, and the chain total gets reported. It pays back only if the agent is reused across targets.
4. **Accuracy at equal compression on home benchmarks** (ledger §157).
5. **The NEON precedent has the same shape.** NEON's frozen agent (NEON 5) beats AMC 4 on time, but it is slower than AMC 1 on all three dataset sizes: 0.67 vs 0.62, 5.81 vs 3.25 and 13.35 vs 4.03 min (Table 7, §8). NEON's own time is "dedicated to the re-training of the pruned layers", just as SPECTRA's is the recovery fine-tune.

## 7. Break-even over K size targets

SPECTRA(K) = W + K·F, where W is the walk to the deepest target and F is one final fine-tune. A per-target pipeline costs K·C. Break-even is K* = W / (C − F), when C > F. All numbers below are for ResNet-56 on CIFAR-10 on an RTX 4090. F is each walk's own final: 16.6 min for the 0.70-deep walk (21809595), 15.7 min for the 0.36-deep walk (21767189).

One-time costs are excluded on both sides: SPECTRA's agent train (§3.2) and the graph-metanetwork meta-train (357 min). Counting them would favour neither side cleanly, since our train serves every family and dataset while theirs serves one setting.

| Comparator | C per target | K*, walk to 0.70 kept (W = 213.5 min) | K*, walk to 0.36 kept (W = 545.5 min) | K*, 12/4 recovery (W ≈ 165 min, deep walk ÷ 3.3) |
|---|---|---|---|---|
| DepGraph, 100 SL + 100 FT | **85 min** | 3.1 | 8.0 | 2.4 |
| Graph metanetworks, 100 + 100 | 67 min | 4.2 | 10.6 | 3.2 |
| Graph metanetworks, 60 + 60 | 43 min | 8.1 | 20 | 6.0 |
| OCSPruner, from scratch | 26 min | 23 | 53 | 16 |

A walk without per-step fine-tuning (a proxy, if the proxy-fidelity cell allows it) brings W to minutes. K* then drops to about 1 against everything except OCSPruner, where the final fine-tune alone (16 min) is most of its 26 min. DepGraph’s **C = 85 min** is now **our** 4090 (job 21943448, ResNet-56 5104 s) and replaces the third-party 84 min; K* is unchanged at ~3 / ~8. The graph-metanetwork rows stay third-party.

## 8. Transfer: closest prior work and the evaluation protocol

### 8.1 Closest prior work

| Work | What transfers | Per-target cost after transfer | How filters are chosen (how many · which) | Source |
|---|---|---|---|---|
| NEON (Hirsch & Katz, Inf. Sci. 2022), "Multi-objective pruning of dense neural networks using deep reinforcement learning" | A DRL agent trained offline on many datasets, applied "without additional training"; **dense nets only** | Table 7 (minutes; small / medium / large data): NEON 5 = 0.67 / 5.81 / 13.35; AMC 4 = 2.89 / 13.37 / 17.53, of which 2.86 / 13.2 / 17.3 is AMC's agent training; AMC 1 = 0.62 / 3.25 / 4.03 | DRL agent per layer · none: the layer is re-initialised at the new width and retrained | [doi:10.1016/j.ins.2022.07.134](https://doi.org/10.1016/j.ins.2022.07.134) |
| Out-of-the-box channel pruned networks (Venkatesan et al., 2020) | Layer-wise profiles from one RL policy over 8 ResNet-20s (CIFAR-10/100), reused on TinyImageNet and ImageNet; **same architecture only** | profile reuse + fine-tune | PPO keep fraction per layer · random channels | [arXiv:2004.14584](https://arxiv.org/abs/2004.14584) |
| Meta Pruning via Graph Metanetworks (Liu, Wang, Zhang, 2025) | "a feedforward through the metanetwork and some standard finetuning"; transfer shown between similar datasets and ResNet-56 ↔ 110 | 43–67 min on a 4090 | global ranking · smallest group L2 after the metanetwork edit | [arXiv:2506.12041](https://arxiv.org/abs/2506.12041) |
| GNN-RL / AGMC (Yu et al.) | The encoder is reused, then a new search runs: GNN-RL ResNet-56 → 44 "only updated the MLP component"; AGMC ResNet-56 → 20 "only updated the decoder parameters", 100 episodes | a per-network search | RL per-layer ratio · smallest L2 (code) | [arXiv:2102.03214](https://arxiv.org/abs/2102.03214); [arXiv:2011.12641](https://arxiv.org/abs/2011.12641) |
| N2N (Ashok et al., ICLR 2018) | A policy pre-trained on smaller teacher networks warm-starts training on larger ones | a continued search | REINFORCE layer removal and shrinkage · UNVERIFIED | [arXiv:1709.06030](https://arxiv.org/abs/1709.06030) |
| Mu et al. (IEEE TCAD 2024) | The agent is warm-started across pruning ratios, models and datasets: 1.5–2.5× faster RL pruning | a continued search | DDPG per-layer ratio · LASSO reconstruction | [arXiv:2107.08815](https://arxiv.org/abs/2107.08815) |
| Evolved pruning functions (Liu, Kung, Wentzlaff, GECCO 2022) | A scoring *criterion* evolved in 98 GPU-days transfers to unseen datasets; not an agent | the criterion's pipeline | fixed ratios · the evolved score itself | [arXiv:2110.10876](https://arxiv.org/abs/2110.10876) |
| AgenticPruner (2026) | An LLM prior with in-context learning | 7.5 effective epochs of search | LLM-proposed ratios · Taylor inside dependency groups | [arXiv:2601.12272](https://arxiv.org/abs/2601.12272) |

The paper's novelty sentence, supported by both fact-checks: no published CNN pruning work applies one frozen learned agent across CNN families and unseen datasets without per-target search or agent fine-tuning. NEON did this for dense networks.

### 8.2 Protocol the paper should follow

1. **Source × target table.** Rows are the training families and datasets; columns are held-out families and held-out datasets, using the hold-out lists in the ledger and runbook. Include a no-transfer column: the same-loop heuristics (mild, L1, random, greedy, look-ahead) at the same walk and the same fine-tune.
2. **Equal retraining for every method.** Le & Hua: "Pruning algorithms should be compared in the same retraining configurations" ([ICLR 2021](https://openreview.net/forum?id=Cb54AMqHQFP)). Random and uniform allocation at the same fine-tune are the neutral baseline (Li et al., CVPR 2022, [arXiv:2205.05676](https://arxiv.org/abs/2205.05676)).
3. **Count the search as part of the method.** Renda et al. note that reported costs omit "training a reinforcement learning agent to predict pruning rates" ([arXiv:2003.02389](https://arxiv.org/abs/2003.02389) §6). Lindauer & Hutter ask that selection runtime be counted ([JMLR 2020](https://jmlr.org/papers/v21/20-056.html)). SPECTRA reports its one-time train beside per-target costs (§3.2, §7).
4. **Leave-one-task-out as the rotating hold-out design** (TransNAS-Bench-101, [arXiv:2105.11871](https://arxiv.org/abs/2105.11871)). Compare frozen transfer against few-step adaptation and against an agent trained from scratch on the target, with cost and Δacc for each.
5. **Seeds and paired tests per target.**

## 9. Measuring as we go

- **Walk and train cost, every run.** On the login node: `python scripts/cost_readout.py <job ids or run dirs> --jsonl runs/cost/<tag>.jsonl`. It reads the manifest, events, log and `gpu_samples.csv`, and prints per network: walk minutes, fine-tune share, per-step stage split, fine-tune epochs by budget, final fine-tunes, GPU-hours and Wh.
- **Energy.** `scripts/spectra.sbatch` in `tree_v9d` writes `gpu_samples.csv` every second for submissions from 1 Oct, about 11:00 (power, utilization, memory, SM clock, temperature). `SPECTRA_GPU_SAMPLES=0` turns it off. Jobs from older trees get an estimate: GPU-hours × the mean board power measured on the same SKU in a `tree_v9d` job. Label it as an estimate.
- **Deployment.** For every TEST run that saved `traj_models`: `sbatch --gpus=rtx_4090:1 --nice=35 --exclude=<runbook list> scripts/bench_deploy.sbatch <run dirs>`, from the tree the run used. Rows go to `runs/bench_deploy/bench_<job>_r{1,2,3}.jsonl`.
- **Comparator re-runs.** `python scripts/h2h_readout.py <h2h out dir | Torch-Pruning log>` prints stage minutes, Wh, best and last epoch, params and FLOPs kept, and the bench medians with speedups (§4.5).
- **Where results go.** Paste into §3 (cost), §4.5 (comparators) and §5.3 (deployment) here. TEST accuracy stays in the ledger.

## 10. Paper table templates

1. **Per-target cost:** method | one-time cost | per-target search | per-target fine-tune | total per target | GPU | source.
2. **K-target amortization:** method | K = 1 / 3 / 5 / 10 | GPU (§7).
3. **Deployment:** net | point | params kept | MACs kept | bs 1 latency | bs 256 throughput | peak memory | energy per image | TEST Δacc (§5.3).
4. **Transfer:** source → target | frozen SPECTRA | mild / L1 / random at equal fine-tune | Δ vs best heuristic | paired p (§8.2).

## 11. Open items for Ido

1. **Done (1 Oct): `torch_pruning` is in the `spectra` env.**
   - torch-pruning 1.6.1, plus einops 0.8.1 because it imports einops. Both went in with `--no-deps`, so torch 2.4.1 and numpy 1.24.4 are unchanged.
   - The Torch-Pruning repo (v1.6.1) is cloned at `scratch_audit/third_party/Torch-Pruning`.
   - The head-to-head re-run is job 21943448 (§4.5).
2. **Install `nvidia-ml-py`.** In-process NVML energy readings are more precise than sampling `power.draw`.
3. **Add an explicit `agent.decide` stage timer and a per-walk cost event to the runner,** in the next tree only, never under live jobs. It replaces the §3.4 upper bound with a measurement. *The timer is built (2 Oct):*
   - `SPECTRA_TIME_DECIDE=1` in `tree_v9d`, default off; `submit.sh` exports it.
   - Tests: `tests/test_decide_timer.py` 8/8.
   - No job sets it yet. Set it on the next frozen-actor TEST.
   - The per-walk cost event is not built: `cost_readout.py` already derives per-net cost from the stage events.
4. **Get BGU's PUE and grid carbon intensity for CO2e,** or quote the conventional defaults with a caveat.
5. **Run a GPU-side CIFAR augmentation equivalence A/B as its own cell** (§3.3), since the walk is input-bound. *Submitted 2 Oct* as D5. The flag is `SPECTRA_FT_AUG_GPU=1`. Off-arm 21982372 is R; on-arm 21982373 is PD. The calls are in `docs/SITTING_GPU_QUEUE.md` "D5". Paste the s/epoch and TEST readout into §3.3 when both complete.
6. **Optional: a val-selected DepGraph variant.** Pick DepGraph's sparse-learning and fine-tune epochs on our 5k val half, and quote our 5k P TEST half. That puts DepGraph on SPECTRA's own protocol. It needs a patched copy of their `main.py`, so it is no longer their exact pipeline. Run it only if the paper puts DepGraph and SPECTRA in the same accuracy table.
