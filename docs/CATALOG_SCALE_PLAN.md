# Catalog scale-up plan (design, 15 Oct 2026)

Ido's GO (15 Oct, ~11:50, after the group meeting):
1. Start the inventory and the split design now. No GPU is needed.
2. Run the first big-catalog train once T7-C and LC-T6BN have read.
3. Pretrain networks for new families later.

Nothing in this file is registered. Each train and each evaluation wave gets its own queue row, with calls and comparators, before it runs (`docs/SITTING_GPU_QUEUE.md`). This plan scales up the LADDER and the fold rows of `docs/TRANSFER_COVERAGE_PROTOCOL.md` §4.

## 0. Why

- **The agent makes one decision per network,** so a training instance costs the same with 10 networks or 300. Only the number of instances grows.
- **Our catalog is small.** NEON trained on 30 random networks per dataset over 28 datasets, several hundred per fold [1]; SPECTRA trains on 10. In RL generalisation studies, the gap to unseen environments closes only with hundreds to thousands of training environments (Cobbe et al. 2019, 2020).
- **Our own evidence: coverage helps where it covers.** Going from 10 to 18 networks (adding CIFAR-100) was neutral pooled (+0.02), and it helped by +0.61 on the held-out CIFAR-100 VGG (§356). The hard network's failure is systematic in what the line learned (§386).
- **A big test catalog adds power.** With 9 test networks one network decides a call, and the pooled 95 % ranges are about ±0.5 (§383, §387). Four times as many networks should roughly halve them [H].
- **Expectations [H]:** gains in the worst case, at tight budgets and on unseen datasets and families. No large pooled jump at 45–60 %, where the hindsight-best plan is only about +0.2 above the rule (§367, §368, §372).

## 1. Inventory (cluster `/home/paretsky/spectra_pretrained_networks`, listed 15 Oct 11:55)

Each file name gives the network's top-1, its parameters (M) and its FLOPs (M). Of 296 files, 139 are CIFAR-scale CNNs:

| Status | Count |
|---|---|
| Training catalog T1 (10) and T1-C's CIFAR-100 additions (8) | 18 |
| The 9-network test pool, plus the guard r20-w2 | 10 |
| **New usable candidates** | **64** |
| Literature anchors, test only (DFPC MobileNetV2 C10 / C100 and VGG-19 C100; ResNet-110 C10; PruningBench ResNet-18 and VGG-19 C100) | 7 |
| Excluded families (ShuffleNetV2: not realizable; RepVGG: no lever; §353, §359) | 18 |
| Duplicate sources (akamaster / gnn_rl copies of the standard ResNets; ResNet-1202) | 10 |
| Below the accuracy floor (C10 < 80 %, C100 < 50 %: thin ResNets at widths 1–3, as NEON replaced networks under 60 %) | 12 |

New usable candidates by family and dataset:

| Family | CIFAR-10 | CIFAR-100 | SVHN | Fashion-MNIST | Total |
|---|---|---|---|---|---|
| ResNet (thin r20 / r56 at widths 3–16; chenyaofo R20 / R44) | 25 | 24 | 2 | 1 | 52 |
| VGG-BN | 2 | 2 | 0 | 1 | 5 |
| MobileNetV2 | 1 | 2 | 1 | 1 | 5 |
| DenseNet | 0 | 0 | 1 | 1 | 2 |
| **Total** | 28 | 28 | 4 | 4 | **64** |

**The pool is ResNet-heavy.** Two steps follow:
- the trainer must sample a family first, then a network;
- the other families need pretrained variants (§3).

## 2. Splits (fixed before any big-catalog read)

Every test network is out of every training catalog. The 9-network pool and the literature anchors stay held out in every fold.

| Split | What is new to the agent | Proposed members (from §1) |
|---|---|---|
| S1 similar architectures | a size it has not seen, in a family and dataset it has | thin r20 / r56 at held-out widths (5, 9, 11, 13, 15); MobileNetV2 ×0.75; VGG-19 C10; the pool's r56-w4, r56-w8, MobileNetV2 ×1.4, DenseNet-100, VGG-16 C10, chenyaofo and DepGraph R56 |
| S2 unseen dataset | the dataset | every Fashion-MNIST network (VGG-11, MobileNetV2 ×0.5, DenseNet-40, r20-w16), with Fashion-MNIST out of training. CIFAR-100 is the second rotation: the "excluded" branch of D4 |
| S3 unseen family | the family | one family out per fold, rotating DenseNet → MobileNetV2 → VGG (§4 of the transfer protocol) |
| S4 unseen both | the family and the dataset | the S3 family's networks on the S2 dataset |
| Anchors | literature cells | DepGraph R56 C10, DepGraph VGG-19 C100, ResNet-110 C10, VGG-16 C10, PruningBench ResNet-18 / VGG-19 C100, DFPC networks |

The training catalog is every remaining recoverable network, sampled family first.
- **The "product" agent** trains on all folds and is tested on new families (§3) and on the anchors.
- **The paper** headlines the held-out splits and reports the product agent as the deployment result (meeting page §14).

## 3. Pretraining to balance the families (GO for later; register before submit)

Each network is trained with the catalog's standard CIFAR recipe, then put through the §148 recoverability gate.

| Group | Networks | Approx. count |
|---|---|---|
| VGG-BN variants | VGG-11 / 13 / 16 at width ×0.5 and ×0.75; C10, C100, SVHN | 12–18 |
| MobileNetV2 | ×0.35, ×0.75 and ×1.4 on SVHN / Fashion-MNIST; ×0.35 on C10 / C100 | 6–8 |
| DenseNet | DenseNet-40 at growth 6 / 24; DenseNet-100 on C100 / SVHN | 6–8 |
| New families, for S3 / S4 and the product test | WRN-16-4, PreAct-ResNet-18, ResNeXt-29 (small), RegNet-CIFAR, GoogLeNet-CIFAR; factories only, no CIFAR checkpoints on disk (transfer protocol §3) | 8–12 |

Cost [H]: about 1 GPU-hour per network on average. A 100-epoch G2 run takes 13–20 minutes on most CIFAR networks and up to 1.5 hours on DenseNet-100. That puts the full table at about 35–50 GPU-hours.

**Wave 1 (built 17 Oct 01:06–04:04; queue row PRE-W1).** 28 networks, trained by `train_pretrained_checkpoint.py --select last` (the last epoch, never the best-on-test one), source tag `spectra-w1`, about 17 GPU-h in all. Every network is above its floor (CIFAR-10 ≥ 80, CIFAR-100 ≥ 50, SVHN ≥ 90, Fashion-MNIST ≥ 88). All 28 passed the load screen: SPECTRA's loader reproduces each saved accuracy within 0.01 points. **Screened at params 0.25 (REC-W1, §402):** all 28 are realizable and recoverable, so all may enter a catalog. Seven carry a lever (sens_cost − uniform ≥ +0.5 on val): six of the eight CIFAR WRN / PreAct-ResNet networks, and VGG-19-BN on Fashion-MNIST. The other SVHN and Fashion-MNIST networks are lever-limited at that budget: allocation hardly moves them. Final top-1 (%):

| Network | CIFAR-10 | CIFAR-100 | SVHN | Fashion-MNIST |
|---|---|---|---|---|
| WRN-16-4 | 93.88 | 73.11 | 96.72 | 91.49 [unstable last epoch: best 94.93] |
| WRN-28-2 | 93.41 | 71.20 | 97.10 | 94.68 |
| PreAct-ResNet-20 | 91.20 | 67.52 | 96.62 | 94.65 |
| PreAct-ResNet-56 | 93.23 | 69.49 | 96.91 | 94.96 |
| VGG-13-BN | — | — | 96.49 | 95.27 |
| VGG-16-BN | — | — | 96.65 | 94.86 |
| VGG-19-BN | — | — | 96.66 | 94.96 |
| MobileNetV2 ×0.75 | — | — | 96.65 | 94.97 |
| MobileNetV2 ×1.0 | — | — | 96.83 | 95.09 |
| MobileNetV2 ×1.4 | — | — | 96.90 | 95.06 |

The best-on-test epoch would have read 0.0 to 0.8 points higher (3.4 on WRN-16-4 Fashion-MNIST). That is the selection bias the old default carried. **Wave 2 still to build:** VGG width variants, and ResNeXt-29, RegNet and GoogLeNet. They need new factories with tests.

## 4. The ladder

The same splits are read at three catalog sizes: 10 (T1's), about 30, and about 80+. Sizes are counted after family balancing. Three choices fix the line:
- **Instances.** LC-T6BN (§388) found that 12,000 instances are not the bottleneck on the ten-network catalog. T8-fs (§400) suggests that a larger catalog needs more instances: 12,000 over 18 networks fell below the ten-network agent at 25 %. At least 1,000 instances per catalog network is the floor [H], with snapshots saved and the length chosen on validation networks, never on TEST.
- **Reward.** The rung train uses the reward of record. bn8 is the cheap first rung, about 1.6 s per instance; fs32 is the stronger one, about 7 s per instance (§387).
- **Change per rung.** One change only: the catalog.

## 5. Calls to register (per rung, per split; TEST 5k, paired by seed)

- **P_split** = agent − the budget's prior, pooled over the split's networks. BEATS-PRIOR-P-POOL if ≥ +0.3 with at least two thirds of the networks positive; BELOW-PRIOR-POOL if ≤ −0.3; AT-PRIOR-POOL between.
- **SCALE** = the rung's agent − the 10-network agent on the same split. CATALOG-HELPS if ≥ +0.3; SAME within ±0.3; CATALOG-HURTS if ≤ −0.3.
- **Coverage tier** (reported, not a call): one seed per network on the full split, agent against the rule. The report gives the win rate, the mean, the worst network, and a bootstrap over networks.

## 6. Gates before the first big train

1. T7-C and LC-T6BN have read (Ido's condition).
2. Every training network passes the §148 recoverability gate at the training budgets. For the plan agent this is read at the deep end of κ: floored sens_cost at params 0.25, recovered with G2, must keep val within 10 points of the origin, and both it and uniform must land with no fallback or masked edit (REC-W1, §402).
3. Every S-split network that carries a call passes a lever screen: sens − uniform ≥ +0.5 at the read budget. A network below it is reported as LEVER-LIMITED (§353).
4. The standardizer is refitted on the new catalog and recorded.
5. Disk space: the checkpoint folder is 65 GB, mostly ImageNet models. Probe the quota and free space before a wave (`.claude/rules/spectra-tooling.md`).
6. Every network passes a load screen before any use. Its unpruned accuracy on val must come within 1 point of its source's reported accuracy, through SPECTRA's own loader and normalisation. A structural check alone is not enough. From 14 Aug to 16 Oct the akamaster model file loaded every akamaster net at chance, because its init overwrote the shortcut identity. ResNet-110 still passed every landing and masking check (LIT-B, 16 Oct). The fixed copy is `spectra_models_instantiation/resnet_akamaster_a.py`.

## 7. Never

- A test network, or an anchor, in any training catalog.
- A split, a catalog or a rung chosen after reading TEST.
- A network dropped from a split after its read. The last-resort replacement Ido described is registered before the read, with its reason.
- "Transfers to unseen families" beyond the folds read.

[1] Hirsch & Katz, *Information Sciences* 2022 (NEON), §4.2.
